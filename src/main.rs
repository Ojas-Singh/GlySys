use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use glysys::{
    BuildOptions, ComponentLibrary, FixOptions, MissingResidues, Naming, ProtonationOverrides,
    StructureFixer, SystemBuilder,
};

#[derive(Debug, Parser)]
#[command(
    name = "glysysbuilder",
    version,
    about = "Prepare solvated Amber/GLYCAM systems in pure Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Analyze models, chains, glycans, attachment sites, and residue support.
    Inspect(InputArgs),
    /// Parameterize, solvate, ionize, and write Amber/GROMACS files.
    Prepare(PrepareArgs),
    /// Repair a structure (PDBFixer-style) and write the fixed PDB plus a JSON report.
    Fix(FixArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MissingResiduesArg {
    None,
    Internal,
    All,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NamingArg {
    Pdb,
    Amber,
}

#[derive(Debug, Args)]
struct FixArgs {
    #[command(flatten)]
    input: InputArgs,
    /// Fixed PDB output path.
    #[arg(short, long)]
    output: PathBuf,
    /// Optional JSON report output path.
    #[arg(long)]
    report: Option<PathBuf>,
    /// pH for side-chain protonation states.
    #[arg(long, default_value_t = 7.0)]
    ph: f64,
    /// Model residues listed in SEQRES but missing from the coordinates.
    #[arg(long, value_enum, default_value = "all")]
    missing_residues: MissingResiduesArg,
    /// Keep modified residues (MSE, SEP, ...) instead of replacing them.
    #[arg(long)]
    keep_nonstandard: bool,
    /// Do not rebuild missing heavy atoms.
    #[arg(long)]
    no_missing_atoms: bool,
    /// Do not add hydrogens.
    #[arg(long)]
    no_hydrogens: bool,
    /// Remove crystallographic waters.
    #[arg(long)]
    remove_water: bool,
    /// Remove ligands, ions and other heterogens (glycans are kept).
    #[arg(long)]
    remove_heterogens: bool,
    /// Remove carbohydrate residues.
    #[arg(long)]
    remove_glycans: bool,
    /// Skip clash relief of rebuilt atoms.
    #[arg(long)]
    no_relax: bool,
    /// Residue naming in the output PDB.
    #[arg(long, value_enum, default_value = "pdb")]
    naming: NamingArg,
    #[command(flatten)]
    components: ComponentArgs,
}

#[derive(Debug, Args)]
struct InputArgs {
    /// Input PDB file.
    input: PathBuf,
    /// Load defaults from a TOML or JSON file.
    #[arg(long)]
    config: Option<PathBuf>,
    /// PDB MODEL number.
    #[arg(long)]
    model: Option<u32>,
    /// Preferred alternate-location identifier.
    #[arg(long)]
    altloc: Option<char>,
    /// Residue state override, for example A:42=HID.
    #[arg(long = "protonation", value_parser = parse_override)]
    protonation: Vec<(String, String)>,
}

#[derive(Debug, Args)]
struct PrepareArgs {
    #[command(flatten)]
    input: InputArgs,
    /// Output directory.
    #[arg(short, long)]
    output: PathBuf,
    /// Solute-to-box-face padding in Å.
    #[arg(long)]
    padding: Option<f64>,
    /// Added NaCl concentration in mol/L.
    #[arg(long)]
    salt: Option<f64>,
    /// Write an unsolvated, non-periodic system without water or ions.
    #[arg(long)]
    no_water: bool,
    /// Solvate with water but do not add neutralizing ions or salt.
    #[arg(long)]
    no_ions: bool,
    /// Deterministic ion-placement tie-breaking seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Replace an existing generated bundle.
    #[arg(long)]
    overwrite: bool,
    /// Repair the structure first (missing atoms, hydrogens, modified residues).
    #[arg(long)]
    fix: bool,
    #[command(flatten)]
    components: ComponentArgs,
}

/// Where Chemical Component Dictionary definitions come from.
#[derive(Debug, Args)]
struct ComponentArgs {
    /// Local Chemical Component Dictionary mmCIF files or directories (repeatable).
    #[arg(long = "ccd")]
    ccd: Vec<PathBuf>,
    /// Never download component definitions from the RCSB.
    #[arg(long)]
    offline: bool,
    /// Directory caching downloaded component definitions.
    #[arg(long)]
    ccd_cache: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(arguments) => {
            let options = options(&arguments, None)?;
            let builder = SystemBuilder::new(options)?;
            let report = builder.inspect_pdb(&arguments.input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Prepare(arguments) => {
            let mut options = options(&arguments.input, Some(&arguments))?;
            options.repair = options.repair || arguments.fix;
            let contents = std::fs::read_to_string(&arguments.input.input)?;
            let mut builder = SystemBuilder::new(options)?;
            let requests = builder.component_requests(&contents)?;
            *builder.components_mut() = component_library(requests, &arguments.components)?;
            let prepared = builder.prepare_pdb_str(&contents)?;
            prepared.write_bundle(&arguments.output)?;
            for warning in &prepared.report().warnings {
                match warning {
                    glysys::BuildWarning::SmallMoleculeParameterized(message) => {
                        eprintln!("small molecule: {message}")
                    }
                    glysys::BuildWarning::StructureRepaired(message) => eprintln!("{message}"),
                    _ => {}
                }
            }
            println!(
                "Prepared {} atoms ({} waters, {} Na+, {} Cl-) in {}",
                prepared.report().total_atoms,
                prepared.report().waters,
                prepared.report().sodium_ions,
                prepared.report().chloride_ions,
                arguments.output.display()
            );
        }
        Command::Fix(arguments) => {
            let build = options(&arguments.input, None)?;
            let fix_options = FixOptions {
                model: build.model,
                altloc: build.altloc,
                protonation: build.protonation,
                ph: arguments.ph,
                missing_residues: match arguments.missing_residues {
                    MissingResiduesArg::None => MissingResidues::None,
                    MissingResiduesArg::Internal => MissingResidues::Internal,
                    MissingResiduesArg::All => MissingResidues::All,
                },
                replace_nonstandard: !arguments.keep_nonstandard,
                add_missing_atoms: !arguments.no_missing_atoms,
                add_hydrogens: !arguments.no_hydrogens,
                keep_water: !arguments.remove_water,
                keep_heterogens: !arguments.remove_heterogens,
                keep_glycans: !arguments.remove_glycans,
                relax: !arguments.no_relax,
                naming: match arguments.naming {
                    NamingArg::Pdb => Naming::Pdb,
                    NamingArg::Amber => Naming::Amber,
                },
            };
            let contents = std::fs::read_to_string(&arguments.input.input)?;
            let mut fixer = StructureFixer::new(fix_options)?;
            let requests = fixer.component_requests(&contents)?;
            let library = component_library(requests, &arguments.components)?;
            *fixer.components_mut() = library;
            let fixed = fixer.fix_pdb_str(&contents)?;
            std::fs::write(&arguments.output, &fixed.pdb)?;
            if let Some(report) = &arguments.report {
                std::fs::write(report, serde_json::to_string_pretty(&fixed.report)?)?;
            }
            let report = &fixed.report;
            println!(
                "Fixed {} -> {} atoms ({} heavy atoms rebuilt, {} residues modelled, {} hydrogens added) in {}",
                report.atoms_in,
                report.atoms_out,
                report.heavy_atoms_added,
                report.residues_added,
                report.hydrogens_added,
                arguments.output.display()
            );
            for warning in &report.warnings {
                eprintln!("warning: {warning}");
            }
        }
    }
    Ok(())
}

fn options(input: &InputArgs, prepare: Option<&PrepareArgs>) -> anyhow::Result<BuildOptions> {
    let mut options = if let Some(path) = &input.config {
        BuildOptions::from_config_file(path)?
    } else {
        BuildOptions::default()
    };
    if let Some(model) = input.model {
        options.model = model;
    }
    if let Some(altloc) = input.altloc {
        options.altloc = Some(altloc);
    }
    let mut overrides: BTreeMap<String, String> = options.protonation.residues;
    overrides.extend(input.protonation.iter().cloned());
    options.protonation = ProtonationOverrides {
        residues: overrides,
    };
    if let Some(prepare) = prepare {
        if let Some(padding) = prepare.padding {
            options.padding_angstrom = padding;
        }
        if let Some(salt) = prepare.salt {
            options.salt_molar = salt;
        }
        if let Some(seed) = prepare.seed {
            options.seed = seed;
        }
        if prepare.no_water {
            options.add_water = false;
            options.add_ions = false;
        } else if prepare.no_ions {
            options.add_ions = false;
        }
        options.overwrite = prepare.overwrite || options.overwrite;
    }
    Ok(options)
}

/// Chemical component definitions for ligands and modified residues:
/// local files first, then the cache, then the RCSB (unless offline).
fn component_library(
    requests: Vec<String>,
    arguments: &ComponentArgs,
) -> anyhow::Result<ComponentLibrary> {
    let mut library = ComponentLibrary::new();
    for path in &arguments.ccd {
        let files = if path.is_dir() {
            std::fs::read_dir(path)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("cif"))
                })
                .collect()
        } else {
            vec![path.clone()]
        };
        for file in files {
            library.add_cif(&std::fs::read_to_string(&file)?)?;
        }
    }
    let requests = requests
        .into_iter()
        .filter(|id| library.get(id).is_none())
        .collect::<Vec<_>>();
    let cache = arguments.ccd_cache.clone().or_else(|| {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
            .map(|base| base.join("glysys").join("ccd"))
    });
    let mut downloaded = 0;
    for id in requests {
        let cached = cache.as_ref().map(|dir| dir.join(format!("{id}.cif")));
        if let Some(path) = &cached
            && let Ok(text) = std::fs::read_to_string(path)
            && library.add_cif(&text).is_ok()
        {
            continue;
        }
        if arguments.offline {
            continue;
        }
        let url = format!("https://files.rcsb.org/ligands/download/{id}.cif");
        let text = match ureq::get(&url).call() {
            Ok(mut response) => response.body_mut().read_to_string()?,
            Err(error) => {
                eprintln!("warning: could not download component {id}: {error}");
                continue;
            }
        };
        if library.add_cif(&text).is_ok() {
            downloaded += 1;
            if let Some(path) = cached {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(path, &text).ok();
            }
        }
    }
    if downloaded > 0 {
        eprintln!("Downloaded {downloaded} chemical component definitions from the RCSB");
    }
    Ok(library)
}

fn parse_override(value: &str) -> Result<(String, String), String> {
    let (selector, state) = value
        .split_once('=')
        .ok_or_else(|| "expected SELECTOR=STATE, for example A:42=HID".to_string())?;
    if selector.trim().is_empty() || state.trim().is_empty() {
        return Err("selector and state must both be non-empty".into());
    }
    Ok((
        selector.trim().to_string(),
        state.trim().to_ascii_uppercase(),
    ))
}

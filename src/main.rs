use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use glysysbuilder::{BuildOptions, ProtonationOverrides, SystemBuilder};

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
            let options = options(&arguments.input, Some(&arguments))?;
            let builder = SystemBuilder::new(options)?;
            let prepared = builder.prepare_pdb(&arguments.input.input)?;
            prepared.write_bundle(&arguments.output)?;
            println!(
                "Prepared {} atoms ({} waters, {} Na+, {} Cl-) in {}",
                prepared.report().total_atoms,
                prepared.report().waters,
                prepared.report().sodium_ions,
                prepared.report().chloride_ions,
                arguments.output.display()
            );
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

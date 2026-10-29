//! Electronic continuum correction (ECC) with the conventions of prosECCo75
//! (Nencini et al., J. Chem. Theory Comput. 2024, 20, 7546; parameters from
//! https://gitlab.com/sparkly/prosecco/prosECCo75, ions v20221028).
//!
//! Ions take the prosECCo75 scaled-charge models (charges of 0.75 and 1.5 e
//! with their own Lennard-Jones parameters), and every integer-charged
//! carbohydrate block carries 0.75 of its charge: the terminal oxygens of a
//! carboxylate, sulfate or phosphate (or the hydrogens of an ammonium) are
//! scaled by 0.75 and the central atom takes the rest, as prosECCo75 does for
//! saccharide carboxylates (oxygens scaled, carbon adjusted to the block
//! total). GLYCAM has no charge groups, so the block is the residue: a GLYCAM
//! uronic acid or SO3 residue goes from -1 to -0.75 (a quarter of the integer
//! charge is removed, so GLYCAM's sulfation correction, which leaves the sugar
//! and its SO3 slightly off integer, still cancels). Protein residues and water
//! are not scaled (prosECCo75 scales CHARMM36 proteins, which is not done here
//! for ff14SB; such a system gets a warning).

use std::collections::BTreeSet;

use crate::BuildWarning;
use crate::model::System;

/// The prosECCo75 scaled-charge ion models: (element, atom type, charge,
/// sigma in nm, epsilon in kJ/mol).  The recommended ones of the release:
/// Na_s (Kohagen 2016), K_s (Mason 2012), Cl_s and Ca_s (Kohagen/Jungwirth
/// 2014), Mg_ss (Duboué-Dijon 2017, the "small" version).
const ECC_IONS: &[(u8, &str, f64, f64, f64)] = &[
    (11, "Na_s", 0.75, 0.211_500, 0.544_284),
    (19, "K_s", 0.75, 0.315_400, 0.418_680),
    (17, "Cl_s", -0.75, 0.378_240, 0.418_400),
    (20, "Ca_s", 1.5, 0.253_800, 0.507_250),
    (12, "Mg_ss", 1.5, 0.136_000, 3.661_000),
];

/// The scale factor of ECC.
pub(crate) const SCALE: f64 = 0.75;

/// What [`apply`] changed, for the report.
#[derive(Debug, Default)]
pub(crate) struct Scaled {
    pub ions: usize,
    pub groups: usize,
}

/// Apply ECC to `system` in place.  `is_protein(residue name)` tells which
/// residues are left alone; water (`WAT`) is never scaled.
pub(crate) fn apply(
    system: &mut System,
    is_protein: impl Fn(&str) -> bool,
    warnings: &mut Vec<BuildWarning>,
) -> Scaled {
    let mut scaled = Scaled::default();
    let mut neighbours = vec![Vec::new(); system.atoms.len()];
    for bond in &system.bonds {
        let [a, b] = bond.atoms;
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    let mut charged_protein = BTreeSet::new();
    for residue in system.residues.clone().iter() {
        let atoms = residue.first_atom..residue.first_atom + residue.atom_count;
        if residue.name == "WAT" || residue.name == "HOH" {
            continue;
        }
        let net: f64 = system.atoms[atoms.clone()].iter().map(|a| a.charge).sum();
        let charge = net.round();
        // a single ion
        if residue.atom_count == 1 {
            let atom = &mut system.atoms[residue.first_atom];
            if let Some(&(_, kind, q, sigma, epsilon)) =
                ECC_IONS.iter().find(|(element, _, q, ..)| {
                    *element == atom.element && (atom.charge - q / SCALE).abs() < 1.0e-3
                })
            {
                atom.atom_type = kind.to_string();
                atom.charge = q;
                // Amber stores Rmin/2 in angstrom and epsilon in kcal/mol.
                atom.radius = sigma * 10.0 * 2f64.powf(1.0 / 6.0) / 2.0;
                atom.epsilon = epsilon / 4.184;
                scaled.ions += 1;
            }
            continue;
        }
        if charge.abs() < 0.5 {
            continue;
        }
        if is_protein(&residue.name) {
            charged_protein.insert(format!("{}{}", residue.name, residue.number));
            continue;
        }
        // Terminal atoms of the charged groups: oxygens with one heavy
        // neighbour and no hydrogen around a centre that has at least two of
        // them (carboxylate, sulfate, phosphate; not an amide carbonyl), or
        // the hydrogens of an ammonium.
        let in_residue = |atom: usize| atoms.contains(&atom);
        let heavy = |atom: usize| system.atoms[atom].element != 1;
        let terminal_oxygen = |atom: usize| {
            system.atoms[atom].element == 8
                && neighbours[atom].iter().filter(|&&n| heavy(n)).count() == 1
                && !neighbours[atom].iter().any(|&n| !heavy(n))
        };
        let mut centres = Vec::new();
        let mut terminals = Vec::new();
        for centre in atoms.clone() {
            let element = system.atoms[centre].element;
            let ends: Vec<usize> = if charge < 0.0 && matches!(element, 6 | 15 | 16) {
                neighbours[centre]
                    .iter()
                    .copied()
                    .filter(|&n| in_residue(n) && terminal_oxygen(n))
                    .collect()
            } else if charge > 0.0 && element == 7 {
                neighbours[centre]
                    .iter()
                    .copied()
                    .filter(|&n| !heavy(n))
                    .collect()
            } else {
                Vec::new()
            };
            let enough = if element == 7 {
                ends.len() >= 3
            } else {
                ends.len() >= 2
            };
            if enough {
                centres.push(centre);
                terminals.extend(ends);
            }
        }
        if centres.is_empty() {
            // no recognisable group: scale the residue as a whole
            for atom in atoms.clone() {
                system.atoms[atom].charge *= SCALE;
            }
            warnings.push(BuildWarning::ChargeScaling(format!(
                "{}{}: no carboxylate, sulfate, phosphate or ammonium group found; all its charges were scaled by {SCALE}",
                residue.name, residue.number
            )));
        } else {
            for &atom in &terminals {
                system.atoms[atom].charge *= SCALE;
            }
            let now: f64 = system.atoms[atoms.clone()].iter().map(|a| a.charge).sum();
            // GLYCAM's sulfation correction leaves a sugar and its SO3 residue a
            // little off integer (it cancels in the pair): remove a quarter of the
            // integer charge only, so the pair, and the system, keep 0.75 of theirs.
            let target = net - (1.0 - SCALE) * charge;
            let correction = (target - now) / centres.len() as f64;
            for &centre in &centres {
                system.atoms[centre].charge += correction;
            }
        }
        scaled.groups += 1;
    }
    if !charged_protein.is_empty() {
        warnings.push(BuildWarning::ChargeScaling(format!(
            "ECC scales ions and carbohydrates only; {} charged protein residue(s) keep their ff14SB charges ({}{})",
            charged_protein.len(),
            charged_protein.iter().take(6).cloned().collect::<Vec<_>>().join(", "),
            if charged_protein.len() > 6 { ", ..." } else { "" }
        )));
    }
    scaled
}

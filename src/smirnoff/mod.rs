//! Generic small-molecule force field: OpenFF Sage 2.2.1 (SMIRNOFF) with
//! AM1-BCC partial charges.
//!
//! Parameters come from `openff_unconstrained-2.2.1.offxml` (Open Force Field
//! Initiative, CC-BY-4.0), converted to JSON with units in kcal/mol, Å and
//! degrees.  AM1-BCC bond charge corrections are the original AM1-BCC values
//! as SMIRKS from openff-recharge (MIT licence).  Partial charges are AM1
//! Mulliken charges at the supplied geometry plus those corrections.

mod am1;
mod molecule;
mod smarts;

use std::collections::HashMap;

pub use molecule::{MolAtom, MolBond, Molecule};
pub use smarts::Pattern;

const SAGE_JSON: &str = include_str!("../../data/openff/openff_unconstrained-2.2.1.json");
const BCC_JSON: &str = include_str!("../../data/openff/am1bcc-with-phosphorus.json");

#[derive(Debug, serde::Deserialize)]
struct SageData {
    bonds: Vec<BondParameter>,
    angles: Vec<AngleParameter>,
    propers: Vec<TorsionParameter>,
    impropers: Vec<TorsionParameter>,
    vdw: Vec<VdwParameter>,
    vdw_scale14: f64,
    electrostatics_scale14: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct BondParameter {
    smirks: String,
    id: String,
    k: f64,
    length: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AngleParameter {
    smirks: String,
    id: String,
    k: f64,
    angle: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TorsionTerm {
    periodicity: i32,
    phase: f64,
    k: f64,
    idivf: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TorsionParameter {
    smirks: String,
    id: String,
    terms: Vec<TorsionTerm>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VdwParameter {
    smirks: String,
    id: String,
    epsilon: f64,
    rmin_half: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct BccParameter {
    smirks: String,
    value: f64,
}

/// Compiled force field: parsed SMIRKS for every parameter.
pub struct ForceField {
    bonds: Vec<(Pattern, BondParameter)>,
    angles: Vec<(Pattern, AngleParameter)>,
    propers: Vec<(Pattern, TorsionParameter)>,
    impropers: Vec<(Pattern, TorsionParameter)>,
    vdw: Vec<(Pattern, VdwParameter)>,
    bcc: Vec<(Pattern, f64)>,
    aromaticity_cases: Vec<Pattern>,
    pub vdw_scale14: f64,
    pub electrostatics_scale14: f64,
}

/// (atoms, Amber force constant, equilibrium value, parameter id).
pub type AssignedBond = ([usize; 2], f64, f64, String);
pub type AssignedAngle = ([usize; 3], f64, f64, String);
/// (atoms, k, periodicity, phase in radians, improper, parameter id).
pub type AssignedTorsion = ([usize; 4], f64, i32, f64, bool, String);

/// Parameters assigned to one molecule (energies kcal/mol, lengths Å).
#[derive(Debug, Clone)]
pub struct Assignment {
    /// E = k (r - r0)^2 (Amber convention), r0 in Å.
    pub bonds: Vec<AssignedBond>,
    /// E = k (theta - theta0)^2, theta0 in radians.
    pub angles: Vec<AssignedAngle>,
    pub torsions: Vec<AssignedTorsion>,
    /// (epsilon, rmin_half, parameter id) per atom.
    pub vdw: Vec<(f64, f64, String)>,
    pub charges: Vec<f64>,
}

/// Why a molecule could not be parameterized.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentError(pub String);

impl std::fmt::Display for AssignmentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl ForceField {
    /// Load and compile the embedded Sage 2.2.1 force field.
    pub fn sage() -> Result<Self, AssignmentError> {
        let data: SageData = serde_json::from_str(SAGE_JSON)
            .map_err(|error| AssignmentError(format!("Sage data: {error}")))?;
        let bcc: Vec<BccParameter> = serde_json::from_str(BCC_JSON)
            .map_err(|error| AssignmentError(format!("AM1-BCC data: {error}")))?;
        let compile = |smirks: &str| {
            Pattern::parse(smirks).map_err(|error| AssignmentError(format!("{smirks}: {error}")))
        };
        let x = "[#6X3,#7X2,#15X2,#7X3+1,#15X3+1,#8X2+1,#16X2+1:N]";
        let y = "[#6X2-1,#7X2-1,#8X2,#16X2,#7X3,#15X3:N]";
        let tag = |pattern: &str, index: usize| pattern.replace('N', &index.to_string());
        // The AM1-BCC aromaticity model (Jakalian et al. 2002), as encoded
        // in openff-recharge: cases 1-5 evaluated on the Kekulé structure.
        let cases = [
            format!(
                "{}1=@{}-@{}=@{}-@{}=@{}-@1",
                tag(x, 1),
                tag(x, 2),
                tag(x, 3),
                tag(x, 4),
                tag(x, 5),
                tag(x, 6)
            ),
            format!(
                "{}1=@{}-@{}=@{}-@{}-,:@{}-@1",
                tag(x, 1),
                tag(x, 2),
                tag(x, 3),
                tag(x, 4),
                tag(x, 5),
                tag(x, 6)
            ),
            format!(
                "{}1=@{}-@{}-,:@{}~@{}-,:@{}-@1",
                tag(x, 1),
                tag(x, 2),
                tag(x, 3),
                tag(x, 4),
                tag(x, 5),
                tag(x, 6)
            ),
            format!(
                "[#6+1:1]1-@{}=@{}-@{}=@{}-@{}=@{}-@1",
                tag(x, 2),
                tag(x, 3),
                tag(x, 4),
                tag(x, 5),
                tag(x, 6),
                tag(x, 7)
            ),
            format!(
                "{}1-@{}=@{}-@{}=@{}-@1",
                tag(y, 1),
                tag(x, 2),
                tag(x, 3),
                tag(x, 4),
                tag(x, 5)
            ),
        ];
        Ok(Self {
            bonds: data
                .bonds
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p)))
                .collect::<Result<_, AssignmentError>>()?,
            angles: data
                .angles
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p)))
                .collect::<Result<_, AssignmentError>>()?,
            propers: data
                .propers
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p)))
                .collect::<Result<_, AssignmentError>>()?,
            impropers: data
                .impropers
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p)))
                .collect::<Result<_, AssignmentError>>()?,
            vdw: data
                .vdw
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p)))
                .collect::<Result<_, AssignmentError>>()?,
            bcc: bcc
                .into_iter()
                .map(|p| Ok((compile(&p.smirks)?, p.value)))
                .collect::<Result<_, AssignmentError>>()?,
            aromaticity_cases: cases.iter().map(|c| compile(c)).collect::<Result<_, _>>()?,
            vdw_scale14: data.vdw_scale14,
            electrostatics_scale14: data.electrostatics_scale14,
        })
    }

    /// Assign valence, van der Waals and AM1-BCC parameters.
    pub fn assign(&self, molecule: &Molecule) -> Result<Assignment, AssignmentError> {
        if let Some(atom) = molecule
            .atoms
            .iter()
            .find(|atom| !am1::supported(atom.element))
        {
            return Err(AssignmentError(format!(
                "the generic force field (Sage + AM1-BCC) covers H, C, N, O, F, P, S, Cl, Br and I; {} is element {}",
                atom.name, atom.element
            )));
        }
        let bonds = self.assign_bonds(molecule)?;
        let angles = self.assign_angles(molecule)?;
        let torsions = self.assign_torsions(molecule)?;
        let vdw = self.assign_vdw(molecule)?;
        let charges = self.am1bcc_charges(molecule)?;
        Ok(Assignment {
            bonds,
            angles,
            torsions,
            vdw,
            charges,
        })
    }

    /// Last matching parameter wins, keyed by the canonical tagged tuple.
    fn label<T: Clone>(
        molecule: &Molecule,
        parameters: &[(Pattern, T)],
        canonical: fn(&[usize]) -> Vec<usize>,
    ) -> HashMap<Vec<usize>, (Vec<usize>, T)> {
        let mut assigned = HashMap::new();
        for (pattern, parameter) in parameters {
            for tuple in pattern.match_tagged(molecule) {
                assigned.insert(canonical(&tuple), (tuple, parameter.clone()));
            }
        }
        assigned
    }

    fn assign_bonds(&self, molecule: &Molecule) -> Result<Vec<AssignedBond>, AssignmentError> {
        let labels = Self::label(molecule, &self.bonds, canonical_linear);
        molecule
            .bonds
            .iter()
            .map(|bond| {
                let key = canonical_linear(&bond.atoms);
                let (_, parameter) = labels.get(&key).ok_or_else(|| {
                    AssignmentError(format!(
                        "no Sage bond parameter for {}-{}",
                        molecule.atoms[bond.atoms[0]].name, molecule.atoms[bond.atoms[1]].name
                    ))
                })?;
                // SMIRNOFF E = k/2 (r - r0)^2; Amber E = k (r - r0)^2.
                Ok((
                    bond.atoms,
                    parameter.k / 2.0,
                    parameter.length,
                    parameter.id.clone(),
                ))
            })
            .collect()
    }

    fn assign_angles(&self, molecule: &Molecule) -> Result<Vec<AssignedAngle>, AssignmentError> {
        let labels = Self::label(molecule, &self.angles, canonical_linear);
        let mut result = Vec::new();
        for center in 0..molecule.len() {
            let neighbors = molecule.neighbors[center]
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>();
            for (i, &a) in neighbors.iter().enumerate() {
                for &b in &neighbors[i + 1..] {
                    let key = canonical_linear(&[a, center, b]);
                    let (tuple, parameter) = labels.get(&key).ok_or_else(|| {
                        AssignmentError(format!(
                            "no Sage angle parameter for {}-{}-{}",
                            molecule.atoms[a].name,
                            molecule.atoms[center].name,
                            molecule.atoms[b].name
                        ))
                    })?;
                    result.push((
                        [tuple[0], tuple[1], tuple[2]],
                        parameter.k / 2.0,
                        parameter.angle.to_radians(),
                        parameter.id.clone(),
                    ));
                }
            }
        }
        Ok(result)
    }

    fn assign_torsions(
        &self,
        molecule: &Molecule,
    ) -> Result<Vec<AssignedTorsion>, AssignmentError> {
        let mut result = Vec::new();
        let labels = Self::label(molecule, &self.propers, canonical_linear);
        // Every proper torsion i-j-k-l of the molecule must be parameterized.
        for bond in &molecule.bonds {
            let [j, k] = bond.atoms;
            for &(i, _) in &molecule.neighbors[j] {
                if i == k {
                    continue;
                }
                for &(l, _) in &molecule.neighbors[k] {
                    if l == j || l == i {
                        continue;
                    }
                    let key = canonical_linear(&[i, j, k, l]);
                    let (tuple, parameter) = labels.get(&key).ok_or_else(|| {
                        AssignmentError(format!(
                            "no Sage torsion parameter for {}-{}-{}-{}",
                            molecule.atoms[i].name,
                            molecule.atoms[j].name,
                            molecule.atoms[k].name,
                            molecule.atoms[l].name
                        ))
                    })?;
                    for term in &parameter.terms {
                        let idivf = term.idivf.unwrap_or(1.0);
                        result.push((
                            [tuple[0], tuple[1], tuple[2], tuple[3]],
                            term.k / idivf,
                            term.periodicity,
                            term.phase.to_radians(),
                            false,
                            parameter.id.clone(),
                        ));
                    }
                }
            }
        }
        // Impropers: trefoil over the three peripheral permutations, k/3 each.
        let impropers = Self::label(molecule, &self.impropers, canonical_improper);
        let mut keys = impropers.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            let (tuple, parameter) = &impropers[&key];
            let (center, others) = (tuple[1], [tuple[0], tuple[2], tuple[3]]);
            for (p, q, r) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
                for term in &parameter.terms {
                    let idivf = term.idivf.unwrap_or(3.0);
                    result.push((
                        [others[p], center, others[q], others[r]],
                        term.k / idivf,
                        term.periodicity,
                        term.phase.to_radians(),
                        true,
                        parameter.id.clone(),
                    ));
                }
            }
        }
        Ok(result)
    }

    fn assign_vdw(&self, molecule: &Molecule) -> Result<Vec<(f64, f64, String)>, AssignmentError> {
        let labels = Self::label(molecule, &self.vdw, |tuple| tuple.to_vec());
        (0..molecule.len())
            .map(|atom| {
                labels
                    .get(&vec![atom])
                    .map(|(_, p)| (p.epsilon, p.rmin_half, p.id.clone()))
                    .ok_or_else(|| {
                        AssignmentError(format!(
                            "no Sage vdW parameter for {}",
                            molecule.atoms[atom].name
                        ))
                    })
            })
            .collect()
    }

    /// AM1 Mulliken charges plus the original AM1-BCC bond charge corrections.
    pub fn am1bcc_charges(&self, molecule: &Molecule) -> Result<Vec<f64>, AssignmentError> {
        let elements = molecule
            .atoms
            .iter()
            .map(|atom| atom.element)
            .collect::<Vec<_>>();
        if let Some(atom) = molecule
            .atoms
            .iter()
            .find(|atom| !am1::supported(atom.element))
        {
            return Err(AssignmentError(format!(
                "AM1-BCC charges are not available for element {} ({})",
                atom.element, atom.name
            )));
        }
        let positions = molecule
            .atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let total_charge = molecule
            .atoms
            .iter()
            .map(|atom| atom.formal_charge)
            .sum::<i32>();
        let result = am1::am1(&elements, &positions, total_charge).map_err(AssignmentError)?;
        let mut charges = result.charges;
        let bcc_molecule = self.am1bcc_aromaticity(molecule);
        let mut done = std::collections::HashSet::new();
        for (pattern, value) in &self.bcc {
            for tuple in pattern.match_tagged(&bcc_molecule) {
                let (a, b) = (tuple[0], tuple[1]);
                if done.contains(&(a, b)) || done.contains(&(b, a)) {
                    continue;
                }
                charges[a] += value;
                charges[b] -= value;
                done.insert((a, b));
            }
        }
        if done.len() != molecule.bonds.len() {
            let missing = molecule
                .bonds
                .iter()
                .find(|bond| {
                    !done.contains(&(bond.atoms[0], bond.atoms[1]))
                        && !done.contains(&(bond.atoms[1], bond.atoms[0]))
                })
                .map(|bond| {
                    format!(
                        "{}-{}",
                        molecule.atoms[bond.atoms[0]].name, molecule.atoms[bond.atoms[1]].name
                    )
                })
                .unwrap_or_default();
            return Err(AssignmentError(format!(
                "no AM1-BCC bond correction for {missing}"
            )));
        }
        Ok(charges)
    }

    /// Copy of `molecule` with AM1-BCC-model aromaticity flags.
    fn am1bcc_aromaticity(&self, molecule: &Molecule) -> Molecule {
        let mut result = molecule.clone();
        result
            .aromatic_atom
            .iter_mut()
            .for_each(|flag| *flag = false);
        result
            .aromatic_bond
            .iter_mut()
            .for_each(|flag| *flag = false);
        // The model is evaluated on the Kekulé structure: bond primitives
        // test Kekulé orders even after a ring has been flagged aromatic.
        result.kekule_bonds = true;
        let mut ar6 = std::collections::HashSet::new();
        let set = |result: &mut Molecule, ring: &[usize]| {
            for &atom in ring {
                result.aromatic_atom[atom] = true;
            }
            for (position, &atom) in ring.iter().enumerate() {
                let next = ring[(position + 1) % ring.len()];
                if let Some(bond) = result.bond_between(atom, next)
                    && result.ring_bond[bond]
                {
                    result.aromatic_bond[bond] = true;
                }
            }
        };
        // Case 1: fully alternating six-ring.
        let case1 = self.aromaticity_cases[0].match_tagged(&result);
        for ring in &case1 {
            set(&mut result, ring);
            ar6.extend(ring.iter().copied());
        }
        // Cases 2 and 3: rings fused to already aromatic six-rings.
        // openff-recharge requires (0-based) match positions 4-5 (case 2) and
        // 2-5 (case 3) to be in aromatic six-rings already.
        for (case, required) in [(1usize, &[4usize, 5][..]), (2, &[2, 3, 4, 5][..])] {
            loop {
                let before = ar6.len();
                let matches = self.aromaticity_cases[case]
                    .match_tagged(&result)
                    .into_iter()
                    .filter(|ring| required.iter().all(|&k| ar6.contains(&ring[k])))
                    .collect::<Vec<_>>();
                for ring in &matches {
                    set(&mut result, ring);
                    ar6.extend(ring.iter().copied());
                }
                if ar6.len() == before {
                    break;
                }
            }
        }
        // Case 4: tropylium; case 5: five-membered heteroaromatics.
        let case4 = self.aromaticity_cases[3].match_tagged(&result);
        for ring in &case4 {
            set(&mut result, ring);
            ar6.extend(ring.iter().copied());
        }
        let case5 = self.aromaticity_cases[4]
            .match_tagged(&result)
            .into_iter()
            .filter(|ring| !ar6.contains(&ring[1]) && !ar6.contains(&ring[2]))
            .collect::<Vec<_>>();
        for ring in &case5 {
            if ring.iter().any(|atom| ar6.contains(atom)) {
                // A five-membered ring fused to an aromatic six-ring (purine,
                // indole): AmberTools types only its lone-pair atom as
                // aromatic and keeps the ring's Kekulé bonds.
                result.aromatic_atom[ring[0]] = true;
            } else {
                set(&mut result, ring);
            }
        }
        result.kekule_bonds = false;
        result
    }
}

/// AM1 Mulliken charges and heat of formation (kcal/mol) at a geometry (Å).
pub fn am1_mulliken(
    elements: &[u8],
    positions: &[[f64; 3]],
    charge: i32,
) -> Result<(Vec<f64>, f64), AssignmentError> {
    let result = am1::am1(elements, positions, charge).map_err(AssignmentError)?;
    Ok((result.charges, result.heat_of_formation))
}

/// Atomic number of an element symbol (case-insensitive).
pub fn atomic_number(symbol: &str) -> Option<u8> {
    const SYMBOLS: &[&str] = &[
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S",
        "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA",
        "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD",
        "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA",
    ];
    let upper = symbol.trim().to_ascii_uppercase();
    let upper = if upper == "D" { "H".to_string() } else { upper };
    SYMBOLS
        .iter()
        .position(|s| *s == upper)
        .map(|i| i as u8 + 1)
}

/// Molecule from a Chemical Component Dictionary definition (ideal geometry).
pub fn molecule_from_component(component: &crate::Component) -> Result<Molecule, AssignmentError> {
    let atoms = component
        .atoms
        .iter()
        .map(|atom| {
            Ok(MolAtom {
                name: atom.name.clone(),
                element: atomic_number(&atom.element)
                    .ok_or_else(|| AssignmentError(format!("unknown element {}", atom.element)))?,
                formal_charge: atom.formal_charge,
                position: atom.ideal.map(|p| [p.x, p.y, p.z]).ok_or_else(|| {
                    AssignmentError(format!("{} has no ideal coordinates", atom.name))
                })?,
            })
        })
        .collect::<Result<Vec<_>, AssignmentError>>()?;
    let index = component
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| (atom.name.as_str(), i))
        .collect::<HashMap<_, _>>();
    let bonds = component
        .bonds
        .iter()
        .map(|bond| {
            Ok(MolBond {
                atoms: [
                    *index
                        .get(bond.first.as_str())
                        .ok_or_else(|| AssignmentError(bond.first.clone()))?,
                    *index
                        .get(bond.second.as_str())
                        .ok_or_else(|| AssignmentError(bond.second.clone()))?,
                ],
                order: bond.order,
            })
        })
        .collect::<Result<Vec<_>, AssignmentError>>()?;
    Ok(Molecule::new(atoms, bonds))
}

/// Bonds/angles/propers: the tuple and its reverse are the same term.
fn canonical_linear(tuple: &[usize]) -> Vec<usize> {
    let forward = tuple.to_vec();
    let reverse = tuple.iter().rev().copied().collect::<Vec<_>>();
    forward.min(reverse)
}

/// Impropers: same central atom (second) and the same set of other atoms.
fn canonical_improper(tuple: &[usize]) -> Vec<usize> {
    let mut others = [tuple[0], tuple[2], tuple[3]];
    others.sort_unstable();
    vec![tuple[1], others[0], others[1], others[2]]
}

#[cfg(test)]
mod tests {
    use super::molecule::tests::molecule;
    use super::*;

    fn ethanol() -> Molecule {
        let mut m = molecule(
            &[6, 6, 8, 1, 1, 1, 1, 1, 1],
            &[
                (0, 1, 1),
                (1, 2, 1),
                (0, 3, 1),
                (0, 4, 1),
                (0, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 8, 1),
            ],
        );
        let positions = [
            [-1.2130, -0.2030, 0.0],
            [0.1580, 0.4870, 0.0],
            [1.1550, -0.5200, 0.0],
            [-1.9690, 0.5850, 0.0],
            [-1.3230, -0.8330, 0.8850],
            [-1.3230, -0.8330, -0.8850],
            [0.2720, 1.1250, 0.8850],
            [0.2720, 1.1250, -0.8850],
            [2.0240, -0.1000, 0.0],
        ];
        for (atom, position) in m.atoms.iter_mut().zip(positions) {
            atom.position = position;
        }
        m
    }

    #[test]
    fn sage_parameterizes_ethanol_completely() {
        let ff = ForceField::sage().unwrap();
        let molecule = ethanol();
        let assignment = ff.assign(&molecule).unwrap();
        assert_eq!(assignment.bonds.len(), 8);
        assert_eq!(assignment.angles.len(), 13);
        // C-C bond is b1 in Sage 2.2.1; the hydroxyl hydrogen is n12.
        assert_eq!(assignment.bonds[0].3, "b1");
        assert!(assignment.vdw[8].2.starts_with('n'));
        let total: f64 = assignment.charges.iter().sum();
        assert!(total.abs() < 1e-6, "{total}");
        // AM1-BCC ethanol oxygen is about -0.6 e.
        assert!(
            assignment.charges[2] < -0.5 && assignment.charges[2] > -0.75,
            "{:?}",
            assignment.charges
        );
    }
}

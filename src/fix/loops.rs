//! Missing polymer residues: SEQRES alignment, loop closure and tails.

use std::collections::BTreeMap;

use super::chemistry;
use super::geometry::{
    Grid, V, add, cross, distance, distance2, dot, normalize, place, rotate_about, scale, sub,
};
use super::work::{Origin, ResidueKind, WAtom, WResidue, Work};

const N_CA: f64 = 1.458;
const CA_C: f64 = 1.525;
const C_N: f64 = 1.329;
const C_O: f64 = 1.231;
const ANGLE_N_CA_C: f64 = 111.2;
const ANGLE_CA_C_N: f64 = 116.2;
const ANGLE_C_N_CA: f64 = 121.7;
const ANGLE_CA_C_O: f64 = 120.5;

/// Which absent SEQRES residues to model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingResidues {
    /// Do not model absent residues.
    None,
    /// Only gaps between observed residues.
    Internal,
    /// Internal gaps and absent terminal residues (PDBFixer behaviour).
    #[default]
    All,
}

/// A run of absent residues located by SEQRES alignment.
#[derive(Debug, Clone)]
pub(crate) struct Gap {
    pub chain: String,
    /// Residue index (in `work.residues`) preceding the gap.
    pub after: Option<usize>,
    /// Residue index following the gap.
    pub before: Option<usize>,
    pub names: Vec<String>,
    /// Exact residue numbers when known (REMARK 465).
    pub numbers: Option<Vec<(i32, Option<char>)>>,
}

fn standard_name(name: &str) -> String {
    let (alias, _) = chemistry::residue_alias(name);
    chemistry::substitution(&alias)
        .map(str::to_string)
        .unwrap_or(alias)
}

fn polymer_like(residue: &WResidue) -> bool {
    residue.is_polymer() || (residue.kind == ResidueKind::Ligand && residue.parent.is_some())
}

/// Whether an observed residue matches a SEQRES / REMARK 465 residue name.
fn same_residue(residue: &WResidue, expected: &str) -> bool {
    let expected = standard_name(expected);
    standard_name(&residue.name) == expected
        || standard_name(&residue.input_name) == expected
        || residue.parent.as_deref() == Some(expected.as_str())
}

/// Locate unobserved polymer residues.
///
/// REMARK 465 lists them with their exact residue numbers and insertion
/// codes, so it is used whenever present.  Otherwise SEQRES is aligned to the
/// observed numbering, as PDBFixer does.
pub(crate) fn find_gaps(work: &Work) -> Vec<Gap> {
    let mut chains = BTreeMap::<String, Vec<usize>>::new();
    for (index, residue) in work.residues.iter().enumerate() {
        if polymer_like(residue) {
            chains.entry(residue.chain.clone()).or_default().push(index);
        }
    }
    let mut unobserved = BTreeMap::<String, Vec<(i32, Option<char>, String)>>::new();
    for (key, name) in &work.unobserved {
        unobserved.entry(key.chain.clone()).or_default().push((
            key.number,
            key.insertion_code,
            standard_name(name),
        ));
    }
    let mut gaps = Vec::new();
    for (chain, indices) in chains {
        if let Some(missing) = unobserved.get(&chain) {
            gaps.extend(gaps_from_remark(work, &chain, &indices, missing));
        } else {
            gaps.extend(gaps_from_seqres(work, &chain, &indices));
        }
    }
    gaps
}

fn gaps_from_remark(
    work: &Work,
    chain: &str,
    indices: &[usize],
    missing: &[(i32, Option<char>, String)],
) -> Vec<Gap> {
    // Merge observed and unobserved residues by (number, insertion code).
    let order = |number: i32, icode: Option<char>| (number, icode.map_or(0, |c| c as u32 + 1));
    let mut entries = indices
        .iter()
        .map(|&index| {
            let residue = &work.residues[index];
            (
                order(residue.number, residue.icode),
                Some(index),
                String::new(),
                (residue.number, residue.icode),
            )
        })
        .chain(missing.iter().map(|(number, icode, name)| {
            (
                order(*number, *icode),
                None,
                name.clone(),
                (*number, *icode),
            )
        }))
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.0);
    let mut gaps = Vec::new();
    let mut previous = None;
    let mut pending: Vec<(String, (i32, Option<char>))> = Vec::new();
    let flush = |after: Option<usize>,
                 before: Option<usize>,
                 pending: &mut Vec<(String, (i32, Option<char>))>,
                 gaps: &mut Vec<Gap>| {
        if !pending.is_empty() {
            let (names, numbers) = std::mem::take(pending).into_iter().unzip();
            gaps.push(Gap {
                chain: chain.to_string(),
                after,
                before,
                names,
                numbers: Some(numbers),
            });
        }
    };
    for (_, observed, name, number) in entries {
        match observed {
            Some(index) => {
                flush(previous, Some(index), &mut pending, &mut gaps);
                previous = Some(index);
            }
            None => pending.push((name, number)),
        }
    }
    flush(previous, None, &mut pending, &mut gaps);
    gaps
}

fn gaps_from_seqres(work: &Work, chain: &str, indices: &[usize]) -> Vec<Gap> {
    let Some(sequence) = work.seqres.get(chain) else {
        return Vec::new();
    };
    // Insertion codes shift all later residue numbers by one.
    let mut ids = Vec::with_capacity(indices.len());
    let mut shift = 0;
    for &index in indices {
        if work.residues[index].icode.is_some() {
            shift += 1;
        }
        ids.push(work.residues[index].number + shift);
    }
    let (Some(&min), Some(&max)) = (ids.iter().min(), ids.iter().max()) else {
        return Vec::new();
    };
    let span = (max - min + 1) as usize;
    if span > sequence.len() || span > 100_000 {
        return Vec::new();
    }
    let mut gapped = vec![None; span];
    for (&index, &id) in indices.iter().zip(&ids) {
        gapped[(id - min) as usize] = Some(index);
    }
    let offset = (0..=sequence.len() - span).find(|&offset| {
        gapped
            .iter()
            .zip(&sequence[offset..])
            .all(|(observed, expected)| {
                observed.is_none_or(|index| same_residue(&work.residues[index], expected))
            })
    });
    let Some(offset) = offset else {
        return Vec::new();
    };
    let mut gaps = Vec::new();
    let mut pending = Vec::new();
    let mut previous = None;
    for (position, expected) in sequence.iter().enumerate() {
        let observed = position
            .checked_sub(offset)
            .and_then(|relative| gapped.get(relative))
            .and_then(|entry| *entry);
        match observed {
            Some(index) => {
                if !pending.is_empty() {
                    gaps.push(Gap {
                        chain: chain.to_string(),
                        after: previous,
                        before: Some(index),
                        names: std::mem::take(&mut pending),
                        numbers: None,
                    });
                }
                previous = Some(index);
            }
            None => pending.push(standard_name(expected)),
        }
    }
    if !pending.is_empty() {
        gaps.push(Gap {
            chain: chain.to_string(),
            after: previous,
            before: None,
            names: pending,
            numbers: None,
        });
    }
    gaps
}

/// Deterministic generator for loop sampling.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Backbone (phi, psi) basins sampled when modelling residues.
fn sample_basin(name: &str, random: &mut SplitMix) -> (f64, f64) {
    let pick = random.next();
    let jitter = |random: &mut SplitMix| (random.next() - 0.5) * 30.0;
    let (phi, psi) = if name == "PRO" {
        (-65.0, if pick < 0.6 { 145.0 } else { -35.0 })
    } else if name == "GLY" && pick < 0.25 {
        (80.0, 10.0)
    } else if pick < 0.45 {
        (-120.0, 135.0)
    } else if pick < 0.75 {
        (-70.0, 145.0)
    } else {
        (-63.0, -42.0)
    };
    let phi = if name == "PRO" {
        phi
    } else {
        phi + jitter(random)
    };
    (phi.to_radians(), (psi + jitter(random)).to_radians())
}

/// Backbone atoms (N, CA, C, O) of modelled residues.
type Backbone = [V; 4];

fn clash_score(points: &[V], environment: &Grid, positions: &[V], exclude: &[V]) -> f64 {
    let mut score = 0.0;
    for point in points {
        for index in environment.near(*point, 3.2) {
            let other = positions[index];
            if exclude
                .iter()
                .any(|excluded| distance2(*excluded, other) < 1.0e-6)
            {
                continue;
            }
            let d2 = distance2(*point, other);
            if d2 < 3.2 * 3.2 {
                let overlap = 3.2 - d2.sqrt();
                score += overlap * overlap;
            }
        }
    }
    score
}

/// Close an internal gap between `previous` and `next` by cyclic coordinate descent.
pub(crate) fn build_internal(
    previous: &WResidue,
    next: &WResidue,
    names: &[String],
    environment: &Grid,
    positions: &[V],
    seed: u64,
) -> Option<(Vec<Backbone>, f64)> {
    let (ca0, c0, o0) = (
        previous.position("CA")?,
        previous.position("C")?,
        previous.position("O")?,
    );
    let target = [
        next.position("N")?,
        next.position("CA")?,
        next.position("C")?,
    ];
    let mut random = SplitMix(seed ^ 0x5EED_1004);
    let mut best: Option<(f64, Vec<Backbone>, f64)> = None;
    let attempts = if names.len() <= 2 { 12 } else { 24 };
    for _ in 0..attempts {
        // Chain atoms: for each residue N, CA, C, O; then ghost N, CA, C.
        let mut atoms = Vec::with_capacity(names.len() * 4 + 3);
        let n1 = place(
            o0,
            ca0,
            c0,
            C_N,
            ANGLE_CA_C_N.to_radians(),
            std::f64::consts::PI,
        );
        let mut previous_n = n1;
        let mut previous_ca_c = (ca0, c0);
        let mut torsions = Vec::new();
        for name in names {
            let (phi, psi) = sample_basin(name, &mut random);
            torsions.push((phi, psi));
            let n = previous_n;
            let ca = place(
                previous_ca_c.0,
                previous_ca_c.1,
                n,
                N_CA,
                ANGLE_C_N_CA.to_radians(),
                std::f64::consts::PI,
            );
            let c = place(previous_ca_c.1, n, ca, CA_C, ANGLE_N_CA_C.to_radians(), phi);
            let next_n = place(n, ca, c, C_N, ANGLE_CA_C_N.to_radians(), psi);
            let o = place(
                next_n,
                ca,
                c,
                C_O,
                ANGLE_CA_C_O.to_radians(),
                std::f64::consts::PI,
            );
            atoms.extend([n, ca, c, o]);
            previous_ca_c = (ca, c);
            previous_n = next_n;
        }
        let ghost_n = previous_n;
        let ghost_ca = place(
            previous_ca_c.0,
            previous_ca_c.1,
            ghost_n,
            N_CA,
            ANGLE_C_N_CA.to_radians(),
            std::f64::consts::PI,
        );
        let ghost_c = place(
            previous_ca_c.1,
            ghost_n,
            ghost_ca,
            CA_C,
            ANGLE_N_CA_C.to_radians(),
            -1.2,
        );
        atoms.extend([ghost_n, ghost_ca, ghost_c]);
        let ghost = atoms.len() - 3;

        // Rotatable bonds: (axis start, axis end, first moving atom index).
        let mut axes = Vec::new();
        for (residue, name) in names.iter().enumerate() {
            let base = residue * 4;
            if name != "PRO" {
                axes.push((base, base + 1, base + 2));
            }
            axes.push((base + 1, base + 2, base + 3));
        }
        axes.push((ghost, ghost + 1, ghost + 2));
        let mut rmsd = f64::INFINITY;
        for _ in 0..300 {
            for &(start, end, first_moving) in &axes {
                let origin = atoms[start];
                let Some(axis) = normalize(sub(atoms[end], origin)) else {
                    continue;
                };
                let (mut numerator, mut denominator) = (0.0, 0.0);
                for (offset, goal) in target.iter().enumerate() {
                    let moving = atoms[ghost + offset];
                    let r = perpendicular(sub(moving, origin), axis);
                    let f = perpendicular(sub(*goal, origin), axis);
                    numerator += dot(axis, cross(r, f));
                    denominator += dot(r, f);
                }
                let theta = numerator.atan2(denominator);
                if theta.abs() < 1.0e-6 {
                    continue;
                }
                // Moving atoms: everything after the axis end, except the O of
                // the residue whose phi is rotated (it follows C) is included.
                for atom in atoms.iter_mut().skip(first_moving) {
                    *atom = rotate_about(*atom, origin, axis, theta);
                }
            }
            rmsd = ((0..3)
                .map(|offset| distance2(atoms[ghost + offset], target[offset]))
                .sum::<f64>()
                / 3.0)
                .sqrt();
            if rmsd < 0.05 {
                break;
            }
        }
        // Rebuild carbonyl oxygens in the final peptide planes.
        for residue in 0..names.len() {
            let base = residue * 4;
            let next_n = atoms[base + 4];
            atoms[base + 3] = place(
                next_n,
                atoms[base + 1],
                atoms[base + 2],
                C_O,
                ANGLE_CA_C_O.to_radians(),
                std::f64::consts::PI,
            );
        }
        let backbone = atoms[..names.len() * 4]
            .chunks(4)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect::<Vec<_>>();
        let flat = backbone.iter().flatten().copied().collect::<Vec<_>>();
        let exclude = [ca0, c0, o0, target[0], target[1], target[2]];
        let score = rmsd * 20.0 + clash_score(&flat, environment, positions, &exclude);
        if best
            .as_ref()
            .is_none_or(|(best_score, _, _)| score < *best_score)
        {
            best = Some((score, backbone, rmsd));
        }
    }
    best.map(|(_, backbone, rmsd)| (backbone, rmsd))
}

fn perpendicular(vector: V, axis: V) -> V {
    sub(vector, scale(axis, dot(vector, axis)))
}

/// Pseudo C-beta from backbone N, CA, C (ideal L-amino-acid geometry).
fn pseudo_cb(n: V, ca: V, c: V) -> V {
    let b = sub(ca, n);
    let c_vec = sub(c, ca);
    let a = cross(b, c_vec);
    add(
        ca,
        add(
            add(scale(a, -0.582_734_31), scale(b, 0.568_028_27)),
            scale(c_vec, -0.540_674_66),
        ),
    )
}

/// Partial tail during beam search: built residues and the growth frame.
#[derive(Clone)]
struct TailState {
    built: Vec<Backbone>,
    /// C-terminal growth: (CA, C, next N); N-terminal: (N, CA, C) of the
    /// residue the next one attaches to.
    frame: [V; 3],
    score: f64,
}

const BEAM_WIDTH: usize = 8;

fn candidate_torsions(name: &str) -> Vec<(f64, f64)> {
    let phis: &[f64] = match name {
        "PRO" => &[-65.0],
        "GLY" => &[-150.0, -120.0, -90.0, -65.0, 70.0, 90.0],
        _ => &[-160.0, -140.0, -120.0, -100.0, -80.0, -65.0],
    };
    let psis: &[f64] = &[170.0, 150.0, 130.0, 110.0, -20.0, -40.0, -60.0];
    phis.iter()
        .flat_map(|&phi| {
            psis.iter()
                .map(move |&psi| (phi.to_radians(), psi.to_radians()))
        })
        .collect()
}

fn tail_step_score(
    atoms: &[V],
    built: &[Backbone],
    environment: &Grid,
    positions: &[V],
    center: V,
    exclude: &[V],
) -> f64 {
    let mut score = clash_score(atoms, environment, positions, exclude) * 10.0;
    // Earlier modelled residues, except the directly bonded neighbour.
    for backbone in built.iter().rev().skip(1) {
        for atom in backbone {
            for new in atoms {
                let d = distance(*atom, *new);
                if d < 3.2 {
                    score += 10.0 * (3.2 - d) * (3.2 - d);
                }
            }
        }
    }
    // Prefer pointing away from the molecule to avoid burying the tail.
    score - 0.1 * distance(atoms[1], center)
}

/// Grow absent C-terminal residues outward from `last` (beam search).
pub(crate) fn build_c_tail(
    last: &WResidue,
    names: &[String],
    environment: &Grid,
    positions: &[V],
    center: V,
) -> Option<Vec<Backbone>> {
    let (ca, c, o) = (
        last.position("CA")?,
        last.position("C")?,
        last.position("O")?,
    );
    let n = place(
        o,
        ca,
        c,
        C_N,
        ANGLE_CA_C_N.to_radians(),
        std::f64::consts::PI,
    );
    let mut beam = vec![TailState {
        built: Vec::new(),
        frame: [ca, c, n],
        score: 0.0,
    }];
    for name in names {
        let mut expanded = Vec::new();
        for state in &beam {
            let [ca, c, n] = state.frame;
            for (phi, psi) in candidate_torsions(name) {
                let new_ca = place(
                    ca,
                    c,
                    n,
                    N_CA,
                    ANGLE_C_N_CA.to_radians(),
                    std::f64::consts::PI,
                );
                let new_c = place(c, n, new_ca, CA_C, ANGLE_N_CA_C.to_radians(), phi);
                let next_n = place(n, new_ca, new_c, C_N, ANGLE_CA_C_N.to_radians(), psi);
                let new_o = place(
                    next_n,
                    new_ca,
                    new_c,
                    C_O,
                    ANGLE_CA_C_O.to_radians(),
                    std::f64::consts::PI,
                );
                let mut scored = vec![n, new_ca, new_c, new_o];
                if name != "GLY" {
                    scored.push(pseudo_cb(n, new_ca, new_c));
                }
                let score = state.score
                    + tail_step_score(
                        &scored,
                        &state.built,
                        environment,
                        positions,
                        center,
                        &[c, ca],
                    );
                let mut built = state.built.clone();
                built.push([n, new_ca, new_c, new_o]);
                expanded.push(TailState {
                    built,
                    frame: [new_ca, new_c, next_n],
                    score,
                });
            }
        }
        expanded.sort_by(|a, b| a.score.total_cmp(&b.score));
        expanded.truncate(BEAM_WIDTH);
        beam = expanded;
    }
    beam.into_iter().next().map(|state| state.built)
}

/// Grow absent N-terminal residues backward from `first` (beam search);
/// returned in chain order.
pub(crate) fn build_n_tail(
    first: &WResidue,
    names: &[String],
    environment: &Grid,
    positions: &[V],
    center: V,
) -> Option<Vec<Backbone>> {
    let (n, ca, c) = (
        first.position("N")?,
        first.position("CA")?,
        first.position("C")?,
    );
    let mut beam = vec![TailState {
        built: Vec::new(),
        frame: [n, ca, c],
        score: 0.0,
    }];
    for name in names.iter().rev() {
        let mut expanded = Vec::new();
        for state in &beam {
            let [n, ca, c] = state.frame;
            for (phi_next, psi) in candidate_torsions(name) {
                // phi of the residue after the new one fixes the new carbonyl C.
                let new_c = place(c, ca, n, C_N, ANGLE_C_N_CA.to_radians(), phi_next);
                let new_o = place(ca, n, new_c, C_O, 122.7f64.to_radians(), 0.0);
                let new_ca = place(
                    ca,
                    n,
                    new_c,
                    CA_C,
                    ANGLE_CA_C_N.to_radians(),
                    std::f64::consts::PI,
                );
                let new_n = place(n, new_c, new_ca, N_CA, ANGLE_N_CA_C.to_radians(), psi);
                let mut scored = vec![new_n, new_ca, new_c, new_o];
                if name != "GLY" {
                    scored.push(pseudo_cb(new_n, new_ca, new_c));
                }
                let score = state.score
                    + tail_step_score(
                        &scored,
                        &state.built,
                        environment,
                        positions,
                        center,
                        &[n, ca],
                    );
                let mut built = state.built.clone();
                built.push([new_n, new_ca, new_c, new_o]);
                expanded.push(TailState {
                    built,
                    frame: [new_n, new_ca, new_c],
                    score,
                });
            }
        }
        expanded.sort_by(|a, b| a.score.total_cmp(&b.score));
        expanded.truncate(BEAM_WIDTH);
        beam = expanded;
    }
    beam.into_iter().next().map(|mut state| {
        state.built.reverse();
        state.built
    })
}

/// Create a modelled residue from backbone coordinates.
pub(crate) fn modelled_residue(
    uid: usize,
    chain: &str,
    number: i32,
    name: &str,
    backbone: &Backbone,
) -> WResidue {
    let atoms = ["N", "CA", "C", "O"]
        .iter()
        .zip(backbone)
        .map(|(atom, position)| WAtom {
            name: (*atom).into(),
            element: atom[..1].into(),
            position: *position,
            occupancy: 1.0,
            b_factor: 0.0,
            serial: None,
            origin: Origin::Modelled,
        })
        .collect();
    WResidue {
        uid,
        chain: chain.into(),
        number,
        icode: None,
        input_name: name.into(),
        name: name.into(),
        variant: None,
        kind: ResidueKind::Protein,
        atoms,
        ter_after: false,
        prev: None,
        next: None,
        open_before: false,
        open_after: false,
        modelled: true,
        parent: None,
        template: None,
    }
}

pub(crate) fn centroid(points: &[V]) -> V {
    let sum = points.iter().fold([0.0; 3], |sum, point| add(sum, *point));
    scale(sum, 1.0 / points.len().max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fix::geometry::angle;

    fn residue_from(backbone: &Backbone, number: i32) -> WResidue {
        modelled_residue(number as usize, "A", number, "ALA", backbone)
    }

    #[test]
    fn closes_a_gap_between_two_residues() {
        // Build an extended 6-residue strand and delete the middle four.
        let mut chain = Vec::new();
        let mut n = [0.0, 0.0, 0.0];
        let mut ca = [N_CA, 0.0, 0.0];
        let mut c = place([0.0, 1.0, 0.0], n, ca, CA_C, ANGLE_N_CA_C.to_radians(), 0.0);
        for index in 0..6 {
            let next_n = place(
                n,
                ca,
                c,
                C_N,
                ANGLE_CA_C_N.to_radians(),
                135f64.to_radians(),
            );
            let o = place(
                next_n,
                ca,
                c,
                C_O,
                ANGLE_CA_C_O.to_radians(),
                std::f64::consts::PI,
            );
            chain.push([n, ca, c, o]);
            let next_ca = place(
                ca,
                c,
                next_n,
                N_CA,
                ANGLE_C_N_CA.to_radians(),
                std::f64::consts::PI,
            );
            let next_c = place(
                c,
                next_n,
                next_ca,
                CA_C,
                ANGLE_N_CA_C.to_radians(),
                (-120f64).to_radians(),
            );
            n = next_n;
            ca = next_ca;
            c = next_c;
            let _ = index;
        }
        let previous = residue_from(&chain[0], 1);
        let next = residue_from(&chain[5], 6);
        let names = vec!["ALA".to_string(); 4];
        let grid = Grid::new(4.0);
        let (loop_backbone, rmsd) =
            build_internal(&previous, &next, &names, &grid, &[], 7).unwrap();
        assert_eq!(loop_backbone.len(), 4);
        assert!(rmsd < 0.3, "closure rmsd {rmsd}");
        let last = loop_backbone[3];
        assert!((distance(last[2], chain[5][0]) - C_N).abs() < 0.4);
        assert!((angle(last[0], last[1], last[2]).to_degrees() - ANGLE_N_CA_C).abs() < 1.0);
    }
}

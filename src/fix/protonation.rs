//! Protonation and chemical variants of protein residues.

use std::collections::{BTreeMap, HashMap};

use super::chemistry::{binds_thiolate, is_metal, protonated_at};
use super::geometry::{Grid, V, add, angle, distance, normalize, scale, sub};
use super::work::{ResidueKind, Work};

pub(crate) struct Decision {
    pub residue: usize,
    pub variant: String,
    pub reason: String,
}

/// Assign `variant` for protein residues; returns human-readable decisions.
pub(crate) fn assign(
    work: &mut Work,
    ph: f64,
    overrides: &BTreeMap<String, String>,
) -> Vec<Decision> {
    let index_of = work.index_of();
    let partners = work.explicit_partners();
    let uid_kind = work
        .residues
        .iter()
        .map(|residue| (residue.uid, (residue.kind, residue.name.clone())))
        .collect::<HashMap<_, _>>();

    // Heavy atoms for proximity queries.
    let mut points = Vec::new();
    let mut info = Vec::new();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        for atom in &residue.atoms {
            if atom.is_hydrogen() {
                continue;
            }
            points.push(atom.position);
            info.push((residue_index, atom.name.clone(), atom.element.clone()));
        }
    }
    let mut grid = Grid::new(4.0);
    for (index, point) in points.iter().enumerate() {
        grid.insert(index, *point);
    }
    let nearby = |point: V, radius: f64| {
        grid.near(point, radius)
            .filter(|&index| distance(points[index], point) <= radius)
            .collect::<Vec<_>>()
    };

    // Disulfides: close SG pairs and declared SSBONDs.
    let mut disulfide = HashMap::<usize, usize>::new();
    for (residue_index, residue) in work.residues.iter().enumerate() {
        if residue.kind != ResidueKind::Protein || residue.name != "CYS" {
            continue;
        }
        let Some(sg) = residue.position("SG") else {
            continue;
        };
        for index in nearby(sg, 2.5) {
            let (other, name, _) = &info[index];
            if *other != residue_index
                && name == "SG"
                && work.residues[*other].name == "CYS"
                && work.residues[*other].kind == ResidueKind::Protein
            {
                disulfide.insert(residue_index, *other);
            }
        }
        if let Some(bonded) = partners.get(&(residue.uid, "SG".to_string())) {
            for (uid, atom) in bonded {
                if atom == "SG"
                    && let Some(&other) = index_of.get(uid)
                    && work.residues[other].name == "CYS"
                    && work.residues[other]
                        .position("SG")
                        .is_some_and(|other_sg| distance(other_sg, sg) < 3.0)
                {
                    disulfide.insert(residue_index, other);
                }
            }
        }
    }

    let mut decisions = Vec::new();
    for residue_index in 0..work.residues.len() {
        let residue = &work.residues[residue_index];
        if residue.kind != ResidueKind::Protein {
            continue;
        }
        let selector = format!(
            "{}:{}{}",
            residue.chain,
            residue.number,
            residue.icode.map(String::from).unwrap_or_default()
        );
        let bonded_to_glycan = |atom: &str| {
            partners
                .get(&(residue.uid, atom.to_string()))
                .is_some_and(|bonded| {
                    bonded.iter().any(|(uid, _)| {
                        uid_kind
                            .get(uid)
                            .is_some_and(|(kind, _)| *kind == ResidueKind::Glycan)
                    })
                })
                || residue.position(atom).is_some_and(|position| {
                    nearby(position, 1.75).into_iter().any(|index| {
                        let (other, name, _) = &info[index];
                        work.residues[*other].kind == ResidueKind::Glycan && name == "C1"
                    })
                })
        };
        let metal_near = |atom: &str, radius: f64, thiolate: bool| {
            residue.position(atom).is_some_and(|position| {
                nearby(position, radius).into_iter().any(|index| {
                    let element = &info[index].2;
                    info[index].0 != residue_index
                        && if thiolate {
                            binds_thiolate(element)
                        } else {
                            is_metal(element)
                        }
                })
            })
        };
        let externally_bonded = |atom: &str| {
            partners
                .get(&(residue.uid, atom.to_string()))
                .is_some_and(|bonded| !bonded.is_empty())
        };
        let (variant, reason): (Option<String>, String) = if let Some(state) =
            overrides.get(&selector)
        {
            (Some(state.clone()), "user override".into())
        } else if let Some(variant) = residue.variant.clone().filter(|_| residue.name != "CYS") {
            (Some(variant), "named in the input".into())
        } else {
            match residue.name.as_str() {
                "CYS" => {
                    if let Some(&partner) = disulfide.get(&residue_index) {
                        let other = &work.residues[partner];
                        (
                            Some("CYX".into()),
                            format!("disulfide with {}:{}", other.chain, other.number),
                        )
                    } else if metal_near("SG", 2.8, true) {
                        (Some("CYM".into()), "thiolate coordinating a metal".into())
                    } else if externally_bonded("SG") {
                        (Some("CYX".into()), "SG covalently bonded".into())
                    } else if protonated_at("CYS", ph) == Some(false) {
                        (Some("CYM".into()), format!("deprotonated at pH {ph}"))
                    } else {
                        (None, String::new())
                    }
                }
                "HIS" => {
                    if protonated_at("HIS", ph) == Some(true) {
                        (Some("HIP".into()), format!("protonated at pH {ph}"))
                    } else if metal_near("ND1", 2.8, false) && !metal_near("NE2", 2.8, false) {
                        (Some("HIE".into()), "ND1 coordinates a metal".into())
                    } else if metal_near("NE2", 2.8, false) && !metal_near("ND1", 2.8, false) {
                        (Some("HID".into()), "NE2 coordinates a metal".into())
                    } else {
                        let score = |donor: &str| {
                            histidine_hbond(work, residue_index, donor, &points, &info, &nearby)
                        };
                        let (nd1, ne2) = (score("ND1"), score("NE2"));
                        if nd1 > ne2 + 1.0e-6 {
                            (Some("HID".into()), "ND1-H hydrogen bond".into())
                        } else if ne2 > 0.0 {
                            (Some("HIE".into()), "NE2-H hydrogen bond".into())
                        } else {
                            (Some("HIE".into()), "default tautomer".into())
                        }
                    }
                }
                "ASP" if protonated_at("ASP", ph) == Some(true) => {
                    (Some("ASH".into()), format!("protonated at pH {ph}"))
                }
                "GLU" if protonated_at("GLU", ph) == Some(true) => {
                    (Some("GLH".into()), format!("protonated at pH {ph}"))
                }
                "LYS" if protonated_at("LYS", ph) == Some(false) => {
                    (Some("LYN".into()), format!("deprotonated at pH {ph}"))
                }
                "ASN" if bonded_to_glycan("ND2") => (Some("NLN".into()), "N-glycosylated".into()),
                "SER" if bonded_to_glycan("OG") => (Some("OLS".into()), "O-glycosylated".into()),
                "THR" if bonded_to_glycan("OG1") => (Some("OLT".into()), "O-glycosylated".into()),
                "HYP" if bonded_to_glycan("OD1") => (Some("OLP".into()), "O-glycosylated".into()),
                _ => (None, String::new()),
            }
        };
        if let Some(variant) = variant {
            decisions.push(Decision {
                residue: residue_index,
                variant: variant.clone(),
                reason,
            });
            work.residues[residue_index].variant = Some(variant);
        } else if residue.name == "CYS" {
            work.residues[residue_index].variant = None;
        }
    }
    // Record disulfide bonds explicitly for the output.
    let pairs = disulfide
        .iter()
        .filter(|(a, b)| a < b)
        .map(|(&a, &b)| (work.residues[a].uid, work.residues[b].uid))
        .collect::<Vec<_>>();
    for (a, b) in pairs {
        work.add_bond((a, "SG".into()), (b, "SG".into()));
    }
    decisions
}

/// Hydrogen-bond strength of a hypothetical N-H on a histidine ring nitrogen.
fn histidine_hbond(
    work: &Work,
    residue_index: usize,
    donor: &str,
    points: &[V],
    info: &[(usize, String, String)],
    nearby: &dyn Fn(V, f64) -> Vec<usize>,
) -> f64 {
    let residue = &work.residues[residue_index];
    let Some(nitrogen) = residue.position(donor) else {
        return 0.0;
    };
    let ring_neighbors: &[&str] = if donor == "ND1" {
        &["CG", "CE1"]
    } else {
        &["CD2", "CE1"]
    };
    let Some(direction) = ring_neighbors
        .iter()
        .filter_map(|name| residue.position(name))
        .map(|position| normalize(sub(nitrogen, position)))
        .collect::<Option<Vec<_>>>()
        .and_then(|units| units.into_iter().reduce(add))
        .and_then(normalize)
    else {
        return 0.0;
    };
    let hydrogen = add(nitrogen, scale(direction, 1.01));
    let mut best: f64 = 0.0;
    for index in nearby(nitrogen, 3.5) {
        let (other, name, element) = &info[index];
        if *other == residue_index || !matches!(element.as_str(), "O" | "N") {
            continue;
        }
        // Backbone amide N and amine/guanidinium N are donors, not acceptors.
        let other_residue = &work.residues[*other];
        if element == "N"
            && other_residue.kind == ResidueKind::Protein
            && !(other_residue.name == "HIS" && matches!(name.as_str(), "ND1" | "NE2"))
        {
            continue;
        }
        let acceptor = points[index];
        let theta = angle(nitrogen, hydrogen, acceptor).to_degrees();
        if theta > 130.0 {
            let d = distance(nitrogen, acceptor);
            best = best.max(1.0 - ((d - 2.9).abs() / 0.6).min(1.0) + (theta - 130.0) / 100.0);
        }
    }
    best
}

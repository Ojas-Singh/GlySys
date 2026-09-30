//! PDB output of a repaired structure.

use std::collections::HashMap;

use super::chemistry::standard_atom_name;
use super::geometry::distance;
use super::work::{ResidueKind, Work};
use crate::pdb::encode_hybrid36;

/// Residue and atom naming convention of the written PDB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Naming {
    /// wwPDB names (HIS, CYS, ASN, NAG); protonation is carried by hydrogens.
    #[default]
    Pdb,
    /// Amber/GLYCAM names (HID/HIE/HIP, CYX, ASH, NLN, GLYCAM sugar codes).
    Amber,
}

pub(crate) fn residue_name(
    work: &Work,
    index: usize,
    naming: Naming,
    glycam: &HashMap<usize, String>,
) -> String {
    let residue = &work.residues[index];
    match (naming, residue.kind) {
        (Naming::Pdb, ResidueKind::Protein) => residue.name.clone(),
        (Naming::Amber, ResidueKind::Protein) => match residue.name.as_str() {
            "NH2" => "NHE".into(),
            "HIS" => residue.variant.clone().unwrap_or_else(|| "HIE".into()),
            _ => residue
                .variant
                .clone()
                .unwrap_or_else(|| residue.name.clone()),
        },
        (Naming::Amber, ResidueKind::Glycan) => glycam
            .get(&index)
            .cloned()
            .unwrap_or_else(|| residue.input_name.clone()),
        (_, ResidueKind::Nucleic) | (_, ResidueKind::Water) => residue.name.clone(),
        _ => residue.input_name.clone(),
    }
}

fn format_atom_name(name: &str, element: &str) -> String {
    if name.len() >= 4 || element.len() == 2 {
        format!("{name:<4}")
    } else {
        format!(" {name:<3}")
    }
}

/// Serialize the working structure as a PDB file.
pub(crate) fn write(
    work: &Work,
    naming: Naming,
    glycam: &HashMap<usize, String>,
    remarks: &[String],
) -> String {
    let mut out = String::new();
    for remark in remarks {
        out.push_str(&format!("REMARK 250 {remark}\n"));
    }
    let index_of = work.index_of();
    let names = (0..work.residues.len())
        .map(|index| residue_name(work, index, naming, glycam))
        .collect::<Vec<_>>();
    // SSBOND and LINK records from explicit inter-residue bonds.
    let mut ssbond = 0;
    for ((uid_a, atom_a), (uid_b, atom_b)) in &work.bonds {
        let (Some(&a), Some(&b)) = (index_of.get(uid_a), index_of.get(uid_b)) else {
            continue;
        };
        let (ra, rb) = (&work.residues[a], &work.residues[b]);
        let (Some(pa), Some(pb)) = (ra.position(atom_a), rb.position(atom_b)) else {
            continue;
        };
        if atom_a == "SG" && atom_b == "SG" && ra.name == "CYS" && rb.name == "CYS" {
            ssbond += 1;
            out.push_str(&format!(
                "SSBOND {:>3} CYS {:1} {:>4}{:1}   CYS {:1} {:>4}{:1}{:23}{:>6} {:>6} {:>5.2}\n",
                ssbond,
                ra.chain,
                ra.number,
                ra.icode.unwrap_or(' '),
                rb.chain,
                rb.number,
                rb.icode.unwrap_or(' '),
                "",
                "1555",
                "1555",
                distance(pa, pb)
            ));
        } else {
            out.push_str(&format!(
                "LINK        {:<4} {:>3} {:1}{:>4}{:1}               {:<4} {:>3} {:1}{:>4}{:1}  {:>6} {:>6} {:>5.2}\n",
                format_atom_name(atom_a, &ra.atom(atom_a).map(|x| x.element.clone()).unwrap_or_default()),
                names[a],
                ra.chain,
                ra.number,
                ra.icode.unwrap_or(' '),
                format_atom_name(atom_b, &rb.atom(atom_b).map(|x| x.element.clone()).unwrap_or_default()),
                names[b],
                rb.chain,
                rb.number,
                rb.icode.unwrap_or(' '),
                "1555",
                "1555",
                distance(pa, pb)
            ));
        }
    }

    let mut serial = 0u32;
    let mut serials = HashMap::<(usize, usize), u32>::new();
    for (index, residue) in work.residues.iter().enumerate() {
        let record = if residue.is_polymer() && !residue.is_cap() {
            "ATOM  "
        } else {
            "HETATM"
        };
        let amber_name = residue
            .template
            .clone()
            .unwrap_or_else(|| residue.name.clone());
        for (atom_index, atom) in residue.atoms.iter().enumerate() {
            serial += 1;
            serials.insert((index, atom_index), serial);
            let name = if naming == Naming::Pdb {
                standard_atom_name(&amber_name, &atom.name)
            } else {
                atom.name.clone()
            };
            let element = if atom.element.len() == 2 {
                format!(
                    "{}{}",
                    &atom.element[..1],
                    atom.element[1..].to_ascii_lowercase()
                )
            } else {
                atom.element.clone()
            };
            let number = if (-999..=9999).contains(&residue.number) {
                format!("{:>4}", residue.number)
            } else {
                encode_hybrid36(residue.number.max(0) as u32, 4)
            };
            out.push_str(&format!(
                "{record}{:>5} {} {:>3} {:1}{}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                encode_hybrid36(serial, 5),
                format_atom_name(&name, &atom.element),
                names[index],
                residue.chain.chars().next().unwrap_or(' '),
                number,
                residue.icode.unwrap_or(' '),
                atom.position[0],
                atom.position[1],
                atom.position[2],
                atom.occupancy,
                atom.b_factor,
                element.to_ascii_uppercase(),
            ));
        }
        // TER closes every polymer segment: chain ends and unbridged breaks.
        let in_polymer = residue.is_polymer() || residue.prev.is_some() || residue.next.is_some();
        if in_polymer && residue.next != Some(index + 1) {
            serial += 1;
            out.push_str(&format!(
                "TER   {:>5}      {:>3} {:1}{:>4}{:1}\n",
                encode_hybrid36(serial, 5),
                names[index],
                residue.chain.chars().next().unwrap_or(' '),
                residue.number,
                residue.icode.unwrap_or(' ')
            ));
        }
    }

    // CONECT: heterogen covalent bonds and explicit inter-residue bonds.
    let mut conect = Vec::<(u32, u32)>::new();
    for (index, residue) in work.residues.iter().enumerate() {
        if residue.is_polymer() || matches!(residue.kind, ResidueKind::Water | ResidueKind::Ion) {
            continue;
        }
        if residue.atoms.len() > 400 {
            continue;
        }
        for (i, first) in residue.atoms.iter().enumerate() {
            for (j, second) in residue.atoms.iter().enumerate().skip(i + 1) {
                let limit = if first.is_hydrogen() || second.is_hydrogen() {
                    if first.is_hydrogen() && second.is_hydrogen() {
                        continue;
                    }
                    1.35
                } else {
                    1.95
                };
                if distance(first.position, second.position) < limit {
                    conect.push((serials[&(index, i)], serials[&(index, j)]));
                }
            }
        }
    }
    for ((uid_a, atom_a), (uid_b, atom_b)) in &work.bonds {
        let (Some(&a), Some(&b)) = (index_of.get(uid_a), index_of.get(uid_b)) else {
            continue;
        };
        let find = |residue: usize, name: &str| {
            work.residues[residue]
                .atoms
                .iter()
                .position(|atom| atom.name == name)
                .map(|atom| serials[&(residue, atom)])
        };
        if let (Some(first), Some(second)) = (find(a, atom_a), find(b, atom_b)) {
            conect.push((first, second));
        }
    }
    let mut partners = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for (a, b) in conect {
        partners.entry(a).or_default().push(b);
        partners.entry(b).or_default().push(a);
    }
    for (atom, mut bonded) in partners {
        bonded.sort_unstable();
        bonded.dedup();
        for chunk in bonded.chunks(4) {
            out.push_str(&format!("CONECT{}", encode_hybrid36(atom, 5)));
            for other in chunk {
                out.push_str(&encode_hybrid36(*other, 5));
            }
            out.push('\n');
        }
    }
    out.push_str("END\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atom_names_follow_pdb_column_alignment() {
        assert_eq!(format_atom_name("CA", "C"), " CA ");
        assert_eq!(format_atom_name("ZN", "ZN"), "ZN  ");
        assert_eq!(format_atom_name("HD21", "H"), "HD21");
    }
}

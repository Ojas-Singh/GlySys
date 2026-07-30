use std::collections::HashMap;

use crate::amber_data;
use crate::model::Vec3;
use crate::{BuildError, Result};

#[derive(Debug, Clone)]
pub(crate) struct TemplateAtom {
    pub name: String,
    pub atom_type: String,
    pub element: u8,
    pub charge: f64,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub(crate) struct Template {
    pub name: String,
    pub atoms: Vec<TemplateAtom>,
    pub bonds: Vec<[usize; 2]>,
}

impl Template {
    pub(crate) fn atom(&self, name: &str) -> Option<(usize, &TemplateAtom)> {
        self.atoms
            .iter()
            .enumerate()
            .find(|(_, atom)| atom.name == name)
    }
}

#[derive(Debug)]
pub(crate) struct TemplateSet {
    central: HashMap<String, Template>,
    n_terminal: HashMap<String, Template>,
    c_terminal: HashMap<String, Template>,
    glycans: HashMap<String, Template>,
    ions: HashMap<String, Template>,
    tip3p_box: Template,
}

impl TemplateSet {
    pub(crate) fn load() -> Result<Self> {
        let mut central = parse_off(amber_data::AMINO)?;
        central.extend(parse_off(amber_data::GLYCAM_AMINO)?);
        let mut n_terminal = parse_off(amber_data::AMINO_N)?;
        n_terminal.extend(parse_off(amber_data::GLYCAM_AMINO_N)?);
        let mut c_terminal = parse_off(amber_data::AMINO_C)?;
        c_terminal.extend(parse_off(amber_data::GLYCAM_AMINO_C)?);
        let glycans = parse_prep(amber_data::GLYCAM_PREP)?;
        let ions = parse_off(amber_data::ATOMIC_IONS)?;
        let box_templates = parse_off(amber_data::TIP3P_BOX)?;
        let tip3p_box = box_templates
            .get("TIP3PBOX")
            .or_else(|| box_templates.values().next())
            .cloned()
            .ok_or_else(|| BuildError::ForceField("TIP3P box template is empty".into()))?;
        Ok(Self {
            central,
            n_terminal,
            c_terminal,
            glycans,
            ions,
            tip3p_box,
        })
    }

    pub(crate) fn protein(
        &self,
        name: &str,
        n_terminal: bool,
        c_terminal: bool,
    ) -> Option<&Template> {
        let mapped = match name {
            "HIS" => "HIE",
            other => other,
        };
        if n_terminal {
            self.n_terminal
                .get(mapped)
                .or_else(|| self.n_terminal.get(&format!("N{mapped}")))
        } else if c_terminal {
            self.c_terminal
                .get(mapped)
                .or_else(|| self.c_terminal.get(&format!("C{mapped}")))
        } else {
            self.central.get(mapped)
        }
    }

    pub(crate) fn glycan(&self, name: &str) -> Option<&Template> {
        self.glycans.get(name)
    }

    pub(crate) fn glycan_names(&self) -> impl Iterator<Item = &str> {
        self.glycans.keys().map(String::as_str)
    }

    pub(crate) fn ion(&self, name: &str) -> Option<&Template> {
        self.ions.get(name)
    }

    pub(crate) fn tip3p_box(&self) -> &Template {
        &self.tip3p_box
    }
}

fn parse_off(contents: &str) -> Result<HashMap<String, Template>> {
    #[derive(Default)]
    struct Partial {
        atoms: Vec<TemplateAtom>,
        positions: Vec<Vec3>,
        bonds: Vec<[usize; 2]>,
    }
    #[derive(Clone, Copy)]
    enum Section {
        None,
        Atoms,
        Positions,
        Connectivity,
    }
    let mut partials: HashMap<String, Partial> = HashMap::new();
    let mut current_name = String::new();
    let mut section = Section::None;
    for line in contents.lines() {
        if let Some(header) = line.strip_prefix("!entry.") {
            let Some((name, tail)) = header.split_once(".unit.") else {
                section = Section::None;
                continue;
            };
            current_name = name.to_string();
            partials.entry(current_name.clone()).or_default();
            section = if tail.starts_with("atoms ") {
                Section::Atoms
            } else if tail.starts_with("positions ") {
                Section::Positions
            } else if tail.starts_with("connectivity ") {
                Section::Connectivity
            } else {
                Section::None
            };
            continue;
        }
        if line.starts_with('!') || line.trim().is_empty() {
            continue;
        }
        let Some(partial) = partials.get_mut(&current_name) else {
            continue;
        };
        match section {
            Section::Atoms => {
                let fields = split_amber_fields(line);
                if fields.len() >= 8 {
                    partial.atoms.push(TemplateAtom {
                        name: fields[0].clone(),
                        atom_type: fields[1].clone(),
                        element: fields[6].parse().unwrap_or(0),
                        charge: fields[7].parse().map_err(|_| {
                            BuildError::ForceField(format!(
                                "invalid charge in OFF entry {current_name}"
                            ))
                        })?,
                        position: Vec3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                    });
                }
            }
            Section::Positions => {
                let values = line
                    .split_whitespace()
                    .filter_map(|value| value.parse::<f64>().ok())
                    .collect::<Vec<_>>();
                if values.len() == 3 {
                    partial.positions.push(Vec3 {
                        x: values[0],
                        y: values[1],
                        z: values[2],
                    });
                }
            }
            Section::Connectivity => {
                let values = line
                    .split_whitespace()
                    .filter_map(|value| value.parse::<usize>().ok())
                    .collect::<Vec<_>>();
                if values.len() >= 2 && values[0] != 0 && values[1] != 0 {
                    partial.bonds.push([values[0] - 1, values[1] - 1]);
                }
            }
            Section::None => {}
        }
    }
    partials
        .into_iter()
        .filter(|(_, partial)| !partial.atoms.is_empty())
        .map(|(name, mut partial)| {
            if partial.positions.len() == partial.atoms.len() {
                for (atom, position) in partial.atoms.iter_mut().zip(partial.positions) {
                    atom.position = position;
                }
            }
            Ok((
                name.clone(),
                Template {
                    name,
                    atoms: partial.atoms,
                    bonds: partial.bonds,
                },
            ))
        })
        .collect()
}

fn parse_prep(contents: &str) -> Result<HashMap<String, Template>> {
    #[derive(Debug)]
    struct InternalAtom {
        atom: TemplateAtom,
        bond_to: isize,
        angle_to: isize,
        dihedral_to: isize,
        distance: f64,
        angle: f64,
        dihedral: f64,
    }
    let lines = contents.lines().collect::<Vec<_>>();
    let mut templates = HashMap::new();
    let mut index = 0usize;
    while index < lines.len() {
        let header = lines[index].split_whitespace().collect::<Vec<_>>();
        if header.len() < 2 || header[0].len() != 3 || header[1] != "INT" {
            index += 1;
            continue;
        }
        let name = header[0];
        index += 3;
        let mut internal = Vec::new();
        while index < lines.len() {
            let line = lines[index].trim();
            if line == "LOOP" || line == "DONE" || line.is_empty() {
                break;
            }
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() >= 11 && fields[0].parse::<usize>().is_ok() {
                internal.push(InternalAtom {
                    atom: TemplateAtom {
                        name: fields[1].to_string(),
                        atom_type: fields[2].to_string(),
                        element: element_from_type(fields[2]),
                        charge: fields[10].parse().unwrap_or(0.0),
                        position: Vec3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                    },
                    bond_to: fields[4].parse().unwrap_or(0),
                    angle_to: fields[5].parse().unwrap_or(0),
                    dihedral_to: fields[6].parse().unwrap_or(0),
                    distance: fields[7].parse().unwrap_or(0.0),
                    angle: fields[8].parse().unwrap_or(0.0),
                    dihedral: fields[9].parse().unwrap_or(0.0),
                });
            }
            index += 1;
        }
        // Amber PREP files conventionally separate the atom table from the
        // LOOP section with a blank line.  Do not mistake that separator for
        // the end of the residue, or ring-closing bonds will be lost.
        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }
        let mut loop_pairs = Vec::new();
        if index < lines.len() && lines[index].trim() == "LOOP" {
            index += 1;
            while index < lines.len() {
                let line = lines[index].trim();
                if line == "DONE" || line.is_empty() {
                    break;
                }
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if fields.len() == 2 {
                    loop_pairs.push((fields[0].to_string(), fields[1].to_string()));
                }
                index += 1;
            }
        }
        let mut all_positions = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 0.0,
            },
        ];
        for item in internal.iter().skip(3) {
            let a = reference_position(&all_positions, item.bond_to);
            let b = reference_position(&all_positions, item.angle_to);
            let c = reference_position(&all_positions, item.dihedral_to);
            all_positions.push(place_internal(
                a,
                b,
                c,
                item.distance,
                item.angle,
                item.dihedral,
            ));
        }
        let real_start = internal
            .iter()
            .position(|atom| atom.atom.atom_type != "DU")
            .unwrap_or(3);
        let mut atoms = internal
            .into_iter()
            .enumerate()
            .skip(real_start)
            .map(|(atom_index, mut item)| {
                item.atom.position = all_positions.get(atom_index).copied().unwrap_or(Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });
                (item, atom_index)
            })
            .collect::<Vec<_>>();
        let original_to_real = atoms
            .iter()
            .enumerate()
            .map(|(real, (_, original))| (*original + 1, real))
            .collect::<HashMap<_, _>>();
        let mut bonds = Vec::new();
        for (real, (item, _)) in atoms.iter().enumerate() {
            if let Some(&parent) = original_to_real.get(&(item.bond_to as usize)) {
                bonds.push([real, parent]);
            }
        }
        let name_to_index = atoms
            .iter()
            .enumerate()
            .map(|(atom_index, (item, _))| (item.atom.name.clone(), atom_index))
            .collect::<HashMap<_, _>>();
        for (first, second) in loop_pairs {
            if let (Some(&first), Some(&second)) =
                (name_to_index.get(&first), name_to_index.get(&second))
            {
                bonds.push([first, second]);
            }
        }
        let atoms = atoms.drain(..).map(|(item, _)| item.atom).collect();
        templates.insert(
            name.to_string(),
            Template {
                name: name.to_string(),
                atoms,
                bonds,
            },
        );
        index += 1;
    }
    Ok(templates)
}

fn reference_position(positions: &[Vec3], one_based: isize) -> Vec3 {
    if one_based <= 0 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    } else {
        positions
            .get(one_based as usize - 1)
            .copied()
            .unwrap_or(Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            })
    }
}

fn place_internal(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    distance: f64,
    angle_deg: f64,
    torsion_deg: f64,
) -> Vec3 {
    let unit = |v: [f64; 3]| {
        let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / length, v[1] / length, v[2] / length]
    };
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let ba = unit([a.x - b.x, a.y - b.y, a.z - b.z]);
    let cb = unit([b.x - c.x, b.y - c.y, b.z - c.z]);
    let normal = unit(cross(ba, cb));
    let binormal = cross(normal, ba);
    let theta = angle_deg.to_radians();
    let phi = torsion_deg.to_radians();
    let direction = [
        -theta.cos() * ba[0] + theta.sin() * (phi.cos() * binormal[0] + phi.sin() * normal[0]),
        -theta.cos() * ba[1] + theta.sin() * (phi.cos() * binormal[1] + phi.sin() * normal[1]),
        -theta.cos() * ba[2] + theta.sin() * (phi.cos() * binormal[2] + phi.sin() * normal[2]),
    ];
    Vec3 {
        x: a.x + distance * direction[0],
        y: a.y + distance * direction[1],
        z: a.z + distance * direction[2],
    }
}

fn split_amber_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for character in line.chars() {
        match character {
            '"' => quoted = !quoted,
            character if character.is_ascii_whitespace() && !quoted => {
                if !current.is_empty() {
                    fields.push(std::mem::take(&mut current));
                }
            }
            other => current.push(other),
        }
    }
    if !current.is_empty() {
        fields.push(current);
    }
    fields
}

fn element_from_type(atom_type: &str) -> u8 {
    match atom_type.chars().next().unwrap_or('X').to_ascii_uppercase() {
        'H' => 1,
        'C' => 6,
        'N' => 7,
        'O' => 8,
        'P' => 15,
        'S' => 16,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_amber_residue_libraries() {
        let templates = TemplateSet::load().unwrap();
        let ala = templates.protein("ALA", false, false).unwrap();
        assert_eq!(ala.atoms.len(), 10);
        assert!(ala.atom("CA").is_some());
        let glycan = templates.glycan("0YB").unwrap();
        let c1 = glycan.atom("C1").unwrap().0;
        let c2 = glycan.atom("C2").unwrap().0;
        assert!(
            glycan
                .bonds
                .iter()
                .any(|bond| bond.contains(&c1) && bond.contains(&c2)),
            "GLYCAM PREP LOOP bond C1-C2 was not parsed"
        );
        assert!(templates.ion("Na+").is_some());
        assert!(!templates.tip3p_box().atoms.is_empty());
    }
}

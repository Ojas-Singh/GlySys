use std::collections::HashMap;

use crate::amber_data;
use crate::{BuildError, Result};

#[derive(Debug, Clone, Copy)]
pub(crate) struct BondParameter {
    pub force: f64,
    pub length: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AngleParameter {
    pub force: f64,
    pub degrees: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TorsionParameter {
    pub force: f64,
    pub periodicity: i32,
    pub phase_degrees: f64,
}

#[derive(Debug, Default)]
pub(crate) struct ParameterSet {
    masses: HashMap<String, f64>,
    bonds: HashMap<[String; 2], BondParameter>,
    angles: HashMap<[String; 3], AngleParameter>,
    dihedrals: Vec<([String; 4], TorsionParameter)>,
    impropers: Vec<([String; 4], TorsionParameter)>,
    nonbonded: HashMap<String, (f64, f64)>,
    nonbonded_aliases: HashMap<String, String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Mass,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Nonbond,
}

impl ParameterSet {
    pub(crate) fn load() -> Result<Self> {
        let mut parameters = Self::default();
        parameters.parse_legacy(amber_data::PARM10)?;
        parameters.parse_frcmod(amber_data::FF14SB)?;
        parameters.parse_legacy(amber_data::GLYCAM)?;
        parameters.parse_frcmod(amber_data::TIP3P)?;
        parameters.parse_frcmod(amber_data::IONS_JC)?;
        for _ in 0..4 {
            let aliases = parameters.nonbonded_aliases.clone();
            for (alias, source) in aliases {
                if let Some(value) = parameters.nonbonded.get(&source).copied() {
                    parameters.nonbonded.entry(alias).or_insert(value);
                }
            }
        }
        Ok(parameters)
    }

    pub(crate) fn mass(&self, atom_type: &str, element: u8) -> f64 {
        self.masses
            .get(atom_type)
            .copied()
            .unwrap_or_else(|| element_mass(element))
    }

    pub(crate) fn nonbonded(&self, atom_type: &str) -> Result<(f64, f64)> {
        self.nonbonded
            .get(atom_type)
            .copied()
            .ok_or_else(|| BuildError::MissingParameter {
                kind: "nonbonded",
                types: atom_type.to_string(),
            })
    }

    pub(crate) fn bond(&self, first: &str, second: &str) -> Result<BondParameter> {
        self.bonds
            .get(&canonical_pair(first, second))
            .copied()
            .ok_or_else(|| BuildError::MissingParameter {
                kind: "bond",
                types: format!("{first}-{second}"),
            })
    }

    pub(crate) fn angle(&self, first: &str, second: &str, third: &str) -> Result<AngleParameter> {
        let direct = [first.to_string(), second.to_string(), third.to_string()];
        let reverse = [third.to_string(), second.to_string(), first.to_string()];
        self.angles
            .get(&direct)
            .or_else(|| self.angles.get(&reverse))
            .copied()
            .ok_or_else(|| BuildError::MissingParameter {
                kind: "angle",
                types: format!("{first}-{second}-{third}"),
            })
    }

    pub(crate) fn dihedrals(&self, types: [&str; 4]) -> Result<Vec<TorsionParameter>> {
        best_torsions(&self.dihedrals, types, false).ok_or_else(|| BuildError::MissingParameter {
            kind: "dihedral",
            types: types.join("-"),
        })
    }

    pub(crate) fn improper(&self, types: [&str; 4]) -> Option<Vec<TorsionParameter>> {
        best_torsions(&self.impropers, types, true)
    }

    fn parse_legacy(&mut self, contents: &str) -> Result<()> {
        let blocks = split_blocks(contents);
        if blocks.len() < 5 {
            return Err(BuildError::ForceField(
                "legacy Amber parameter file has too few sections".into(),
            ));
        }
        self.parse_lines(Section::Mass, blocks[0].iter().skip(1).copied())?;
        self.parse_lines(Section::Bond, blocks[1].iter().copied())?;
        self.parse_lines(Section::Angle, blocks[2].iter().copied())?;
        self.parse_lines(Section::Dihedral, blocks[3].iter().copied())?;
        self.parse_lines(Section::Improper, blocks[4].iter().copied())?;
        if let Some(equivalences) = blocks.get(5) {
            for line in equivalences {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if let Some(source) = fields.first() {
                    for alias in &fields[1..] {
                        self.nonbonded_aliases
                            .insert((*alias).to_string(), (*source).to_string());
                    }
                }
            }
        }
        for block in blocks.iter().skip(5) {
            if block
                .iter()
                .any(|line| line.trim_start().starts_with("MOD4"))
                || block.iter().any(|line| parse_nonbond_line(line).is_some())
            {
                self.parse_lines(Section::Nonbond, block.iter().copied())?;
            }
        }
        Ok(())
    }

    fn parse_frcmod(&mut self, contents: &str) -> Result<()> {
        let mut section = Section::None;
        for line in contents.lines().skip(1) {
            section = match line.trim() {
                "MASS" => Section::Mass,
                "BOND" => Section::Bond,
                "ANGL" | "ANGLE" => Section::Angle,
                "DIHE" => Section::Dihedral,
                "IMPR" | "IMPROPER" => Section::Improper,
                "NONB" | "NONBON" => Section::Nonbond,
                _ => {
                    if !line.trim().is_empty() {
                        self.parse_lines(section, std::iter::once(line))?;
                    }
                    section
                }
            };
        }
        Ok(())
    }

    fn parse_lines<'a>(
        &mut self,
        section: Section,
        lines: impl Iterator<Item = &'a str>,
    ) -> Result<()> {
        for raw in lines {
            let line = raw.split('!').next().unwrap_or("").trim_end();
            if line.trim().is_empty()
                || line.trim_start().starts_with("MOD4")
                || line.trim() == "END"
            {
                continue;
            }
            match section {
                Section::Mass => {
                    let fields = line.split_whitespace().collect::<Vec<_>>();
                    if fields.len() >= 2
                        && let Ok(mass) = fields[1].parse()
                    {
                        self.masses.insert(fields[0].to_string(), mass);
                    }
                }
                Section::Bond => {
                    if let Some((types, values)) = parse_typed_values(line, 2)
                        && values.len() >= 2
                    {
                        self.bonds.insert(
                            canonical_pair(&types[0], &types[1]),
                            BondParameter {
                                force: values[0],
                                length: values[1],
                            },
                        );
                    }
                }
                Section::Angle => {
                    if let Some((types, values)) = parse_typed_values(line, 3)
                        && values.len() >= 2
                    {
                        self.angles.insert(
                            [types[0].clone(), types[1].clone(), types[2].clone()],
                            AngleParameter {
                                force: values[0],
                                degrees: values[1],
                            },
                        );
                    }
                }
                Section::Dihedral => {
                    if let Some((types, values)) = parse_typed_values(line, 4)
                        && values.len() >= 4
                    {
                        let divider = values[0].abs().max(1.0);
                        self.dihedrals.push((
                            [
                                types[0].clone(),
                                types[1].clone(),
                                types[2].clone(),
                                types[3].clone(),
                            ],
                            TorsionParameter {
                                force: values[1] / divider,
                                phase_degrees: values[2],
                                periodicity: values[3].abs().round() as i32,
                            },
                        ));
                    }
                }
                Section::Improper => {
                    if let Some((types, values)) = parse_typed_values(line, 4)
                        && values.len() >= 3
                    {
                        self.impropers.push((
                            [
                                types[0].clone(),
                                types[1].clone(),
                                types[2].clone(),
                                types[3].clone(),
                            ],
                            TorsionParameter {
                                force: values[0],
                                phase_degrees: values[1],
                                periodicity: values[2].abs().round() as i32,
                            },
                        ));
                    }
                }
                Section::Nonbond => {
                    if let Some((atom_type, radius, epsilon)) = parse_nonbond_line(line) {
                        self.nonbonded.insert(atom_type, (radius, epsilon));
                    }
                }
                Section::None => {}
            }
        }
        Ok(())
    }
}

fn split_blocks(contents: &str) -> Vec<Vec<&str>> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    for line in contents.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        blocks.push(current);
    }
    blocks
}

fn parse_typed_values(line: &str, type_count: usize) -> Option<(Vec<String>, Vec<f64>)> {
    let prefix_width = match type_count {
        2 => 5,
        3 => 8,
        4 => 11,
        _ => return None,
    };
    if line.len() < prefix_width {
        return None;
    }
    let prefix = &line[..prefix_width];
    let types = prefix
        .split('-')
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();
    if types.len() != type_count || types.iter().any(String::is_empty) {
        return None;
    }
    let values = line[prefix_width..]
        .split_whitespace()
        .take_while(|value| value.parse::<f64>().is_ok())
        .filter_map(|value| value.parse().ok())
        .collect::<Vec<_>>();
    Some((types, values))
}

fn parse_nonbond_line(line: &str) -> Option<(String, f64, f64)> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 || matches!(fields[0], "MOD4" | "RE" | "END") {
        return None;
    }
    Some((
        fields[0].to_string(),
        fields[1].parse().ok()?,
        fields[2].parse().ok()?,
    ))
}

fn canonical_pair(first: &str, second: &str) -> [String; 2] {
    if first <= second {
        [first.to_string(), second.to_string()]
    } else {
        [second.to_string(), first.to_string()]
    }
}

fn best_torsions(
    parameters: &[([String; 4], TorsionParameter)],
    query: [&str; 4],
    improper: bool,
) -> Option<Vec<TorsionParameter>> {
    let permutations = if improper {
        vec![
            query,
            [query[1], query[0], query[2], query[3]],
            [query[0], query[3], query[2], query[1]],
            [query[3], query[1], query[2], query[0]],
            [query[1], query[3], query[2], query[0]],
            [query[3], query[0], query[2], query[1]],
        ]
    } else {
        vec![query, [query[3], query[2], query[1], query[0]]]
    };
    let mut best_score = None;
    let mut found = Vec::new();
    for (pattern, parameter) in parameters {
        let score = permutations
            .iter()
            .filter(|candidate| {
                pattern
                    .iter()
                    .zip(candidate.iter())
                    .all(|(expected, actual)| expected == "X" || expected == *actual)
            })
            .map(|_| pattern.iter().filter(|value| value.as_str() != "X").count())
            .max();
        let Some(score) = score else {
            continue;
        };
        match best_score {
            Some(best) if score < best => {}
            Some(best) if score == best => found.push(*parameter),
            _ => {
                best_score = Some(score);
                found.clear();
                found.push(*parameter);
            }
        }
    }
    (!found.is_empty()).then_some(found)
}

pub(crate) fn element_mass(element: u8) -> f64 {
    match element {
        1 => 1.008,
        6 => 12.01,
        7 => 14.01,
        8 => 16.00,
        11 => 22.989_77,
        15 => 30.97,
        16 => 32.06,
        17 => 35.45,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_combined_parameter_precedence() {
        let parameters = ParameterSet::load().unwrap();
        assert!((parameters.mass("CX", 6) - 12.01).abs() < 1e-6);
        assert!((parameters.bond("C", "N").unwrap().length - 1.335).abs() < 1e-6);
        assert!(parameters.angle("N", "CX", "C").is_ok());
        assert!(parameters.dihedrals(["X", "C", "N", "X"]).is_ok());
        assert!(parameters.nonbonded("OW").is_ok());
        assert!(parameters.nonbonded("Na+").is_ok());
    }
}

//! SMARTS / SMIRKS parsing and substructure matching.
//!
//! Supports the primitives used by SMIRNOFF force fields and AM1-BCC bond
//! charge corrections: `*`, `#n`, element symbols, `a`/`A`, `X`, `x`, `H`,
//! `D`, `R`, `r`, `v`, formal charges, recursive `$(...)`, logical `!`, `&`,
//! `,`, `;`, bond primitives `- = # : ~ @`, branches, ring closures and atom
//! map indices.

use super::molecule::Molecule;

#[derive(Debug, Clone, PartialEq)]
enum AtomPrimitive {
    Any,
    AtomicNumber(u8),
    Aromatic,
    Aliphatic,
    TotalConnectivity(usize),
    RingConnectivity(usize),
    HydrogenCount(usize),
    Degree(usize),
    /// `R` alone: in any ring; `Rn`: in n SSSR rings.
    RingMembership(Option<usize>),
    /// `r` alone: in any ring; `rn`: in a smallest ring of size n.
    RingSize(Option<usize>),
    Valence(usize),
    Charge(i32),
    Recursive(Box<Pattern>),
}

#[derive(Debug, Clone, PartialEq)]
enum Expr<P> {
    Primitive(P),
    Not(Box<Expr<P>>),
    And(Vec<Expr<P>>),
    Or(Vec<Expr<P>>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BondPrimitive {
    Single,
    Double,
    Triple,
    Aromatic,
    Any,
    Ring,
}

#[derive(Debug, Clone, PartialEq)]
struct PatternAtom {
    expr: Expr<AtomPrimitive>,
    map: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct PatternBond {
    atoms: [usize; 2],
    /// `None` is the implicit SMARTS bond: single or aromatic.
    expr: Option<Expr<BondPrimitive>>,
}

/// A parsed SMARTS pattern; `map` indices define SMIRKS tagged atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    atoms: Vec<PatternAtom>,
    bonds: Vec<PatternBond>,
    adjacency: Vec<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmartsError(pub String);

impl std::fmt::Display for SmartsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid SMARTS: {}", self.0)
    }
}

impl Pattern {
    pub fn parse(text: &str) -> Result<Self, SmartsError> {
        let mut parser = Parser {
            chars: text.chars().collect(),
            position: 0,
            atoms: Vec::new(),
            bonds: Vec::new(),
            ring_open: Default::default(),
        };
        parser.chain(None)?;
        if parser.position != parser.chars.len() {
            return Err(SmartsError(format!(
                "unexpected '{}' in {text}",
                parser.chars[parser.position]
            )));
        }
        if !parser.ring_open.is_empty() {
            return Err(SmartsError(format!("unclosed ring in {text}")));
        }
        let mut adjacency = vec![Vec::new(); parser.atoms.len()];
        for (index, bond) in parser.bonds.iter().enumerate() {
            adjacency[bond.atoms[0]].push((bond.atoms[1], index));
            adjacency[bond.atoms[1]].push((bond.atoms[0], index));
        }
        Ok(Self {
            atoms: parser.atoms,
            bonds: parser.bonds,
            adjacency,
        })
    }

    /// Pattern atom indices ordered by their map number (`:1`, `:2`, ...).
    pub fn tagged(&self) -> Vec<usize> {
        let mut tagged = self
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(index, atom)| atom.map.map(|map| (map, index)))
            .collect::<Vec<_>>();
        tagged.sort();
        tagged.into_iter().map(|(_, index)| index).collect()
    }

    /// Every distinct assignment of the tagged atoms (in map order).
    pub fn match_tagged(&self, molecule: &Molecule) -> Vec<Vec<usize>> {
        let tagged = self.tagged();
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for start in 0..molecule.len() {
            let mut mapping = vec![usize::MAX; self.atoms.len()];
            self.search(molecule, 0, start, &mut mapping, &mut |mapping| {
                let key = tagged.iter().map(|&t| mapping[t]).collect::<Vec<_>>();
                if seen.insert(key.clone()) {
                    results.push(key);
                }
                false
            });
        }
        results
    }

    /// Whether the pattern matches with its first atom on `atom`.
    pub fn matches_at(&self, molecule: &Molecule, atom: usize) -> bool {
        if self.atoms.is_empty() {
            return false;
        }
        let mut mapping = vec![usize::MAX; self.atoms.len()];
        let mut found = false;
        self.search(molecule, 0, atom, &mut mapping, &mut |_| {
            found = true;
            true
        });
        found
    }

    /// Depth-first extension of a partial mapping. The callback returns
    /// `true` to stop the search.
    fn search(
        &self,
        molecule: &Molecule,
        pattern_atom: usize,
        molecule_atom: usize,
        mapping: &mut Vec<usize>,
        on_match: &mut dyn FnMut(&[usize]) -> bool,
    ) -> bool {
        if mapping.contains(&molecule_atom)
            || !atom_matches(&self.atoms[pattern_atom].expr, molecule, molecule_atom)
        {
            return false;
        }
        // Bonds to already-mapped pattern atoms (ring closures, back edges).
        for &(other, bond) in &self.adjacency[pattern_atom] {
            if mapping[other] != usize::MAX {
                let Some(molecule_bond) = molecule.bond_between(molecule_atom, mapping[other])
                else {
                    return false;
                };
                if !bond_matches(&self.bonds[bond].expr, molecule, molecule_bond) {
                    return false;
                }
            }
        }
        mapping[pattern_atom] = molecule_atom;
        let stop = match self.next_unmapped(mapping) {
            None => on_match(mapping),
            Some((next, parent, bond)) => {
                let mut stop = false;
                for &(candidate, molecule_bond) in &molecule.neighbors[mapping[parent]] {
                    if !bond_matches(&self.bonds[bond].expr, molecule, molecule_bond) {
                        continue;
                    }
                    if self.search(molecule, next, candidate, mapping, on_match) {
                        stop = true;
                        break;
                    }
                }
                stop
            }
        };
        mapping[pattern_atom] = usize::MAX;
        stop
    }

    /// Next pattern atom bonded to a mapped one (patterns are connected).
    fn next_unmapped(&self, mapping: &[usize]) -> Option<(usize, usize, usize)> {
        for (atom, &mapped) in mapping.iter().enumerate() {
            if mapped == usize::MAX {
                continue;
            }
            for &(other, bond) in &self.adjacency[atom] {
                if mapping[other] == usize::MAX {
                    return Some((other, atom, bond));
                }
            }
        }
        None
    }
}

fn atom_matches(expr: &Expr<AtomPrimitive>, molecule: &Molecule, atom: usize) -> bool {
    match expr {
        Expr::Primitive(primitive) => primitive_matches(primitive, molecule, atom),
        Expr::Not(inner) => !atom_matches(inner, molecule, atom),
        Expr::And(parts) => parts.iter().all(|part| atom_matches(part, molecule, atom)),
        Expr::Or(parts) => parts.iter().any(|part| atom_matches(part, molecule, atom)),
    }
}

fn primitive_matches(primitive: &AtomPrimitive, molecule: &Molecule, atom: usize) -> bool {
    match primitive {
        AtomPrimitive::Any => true,
        AtomPrimitive::AtomicNumber(number) => molecule.atoms[atom].element == *number,
        AtomPrimitive::Aromatic => molecule.aromatic_atom[atom],
        AtomPrimitive::Aliphatic => !molecule.aromatic_atom[atom],
        AtomPrimitive::TotalConnectivity(value) | AtomPrimitive::Degree(value) => {
            molecule.degree(atom) == *value
        }
        AtomPrimitive::RingConnectivity(value) => molecule.ring_connectivity(atom) == *value,
        AtomPrimitive::HydrogenCount(value) => molecule.hydrogen_count(atom) == *value,
        AtomPrimitive::RingMembership(None) | AtomPrimitive::RingSize(None) => {
            molecule.in_ring(atom)
        }
        AtomPrimitive::RingMembership(Some(value)) => molecule.ring_sizes[atom].len() == *value,
        AtomPrimitive::RingSize(Some(value)) => molecule.ring_sizes[atom].contains(value),
        AtomPrimitive::Valence(value) => molecule.valence(atom) == *value,
        AtomPrimitive::Charge(value) => molecule.atoms[atom].formal_charge == *value,
        AtomPrimitive::Recursive(pattern) => pattern.matches_at(molecule, atom),
    }
}

fn bond_matches(expr: &Option<Expr<BondPrimitive>>, molecule: &Molecule, bond: usize) -> bool {
    match expr {
        None => {
            (molecule.aromatic_bond[bond] && !molecule.kekule_bonds)
                || molecule.bonds[bond].order == 1
        }
        Some(expr) => bond_expr_matches(expr, molecule, bond),
    }
}

fn bond_expr_matches(expr: &Expr<BondPrimitive>, molecule: &Molecule, bond: usize) -> bool {
    match expr {
        Expr::Primitive(primitive) => {
            let aromatic = molecule.aromatic_bond[bond] && !molecule.kekule_bonds;
            let order = molecule.bonds[bond].order;
            if molecule.kekule_bonds && *primitive == BondPrimitive::Aromatic {
                return false;
            }
            match primitive {
                BondPrimitive::Single => !aromatic && order == 1,
                BondPrimitive::Double => !aromatic && order == 2,
                BondPrimitive::Triple => order == 3,
                BondPrimitive::Aromatic => aromatic,
                BondPrimitive::Any => true,
                BondPrimitive::Ring => molecule.ring_bond[bond],
            }
        }
        Expr::Not(inner) => !bond_expr_matches(inner, molecule, bond),
        Expr::And(parts) => parts
            .iter()
            .all(|part| bond_expr_matches(part, molecule, bond)),
        Expr::Or(parts) => parts
            .iter()
            .any(|part| bond_expr_matches(part, molecule, bond)),
    }
}

struct Parser {
    chars: Vec<char>,
    position: usize,
    atoms: Vec<PatternAtom>,
    bonds: Vec<PatternBond>,
    ring_open: std::collections::BTreeMap<usize, (usize, Option<Expr<BondPrimitive>>)>,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    fn error(&self, message: &str) -> SmartsError {
        SmartsError(format!(
            "{message} at position {} of {}",
            self.position,
            self.chars.iter().collect::<String>()
        ))
    }

    /// A chain of atoms, branches and ring closures after `previous`.
    fn chain(&mut self, mut previous: Option<usize>) -> Result<(), SmartsError> {
        loop {
            let Some(character) = self.peek() else {
                return Ok(());
            };
            if character == ')' {
                return Ok(());
            }
            if character == '(' {
                let anchor = previous.ok_or_else(|| self.error("branch without atom"))?;
                self.position += 1;
                self.chain(Some(anchor))?;
                if self.peek() != Some(')') {
                    return Err(self.error("unclosed branch"));
                }
                self.position += 1;
                continue;
            }
            let bond = self.bond_expr()?;
            match self.peek() {
                Some(digit) if digit.is_ascii_digit() || digit == '%' => {
                    let atom = previous.ok_or_else(|| self.error("ring closure without atom"))?;
                    let number = self.ring_number()?;
                    if let Some((other, open_bond)) = self.ring_open.remove(&number) {
                        let expr = bond.or(open_bond);
                        self.bonds.push(PatternBond {
                            atoms: [other, atom],
                            expr,
                        });
                    } else {
                        self.ring_open.insert(number, (atom, bond));
                    }
                }
                Some(_) => {
                    let atom = self.atom()?;
                    if let Some(previous) = previous {
                        self.bonds.push(PatternBond {
                            atoms: [previous, atom],
                            expr: bond,
                        });
                    } else if bond.is_some() {
                        return Err(self.error("bond before the first atom"));
                    }
                    previous = Some(atom);
                }
                None => return Err(self.error("dangling bond")),
            }
        }
    }

    fn ring_number(&mut self) -> Result<usize, SmartsError> {
        if self.peek() == Some('%') {
            self.position += 1;
            let digits = self
                .chars
                .get(self.position..self.position + 2)
                .ok_or_else(|| self.error("bad %nn"))?;
            self.position += 2;
            return digits
                .iter()
                .collect::<String>()
                .parse()
                .map_err(|_| self.error("bad ring number"));
        }
        let digit = self
            .peek()
            .and_then(|c| c.to_digit(10))
            .ok_or_else(|| self.error("ring number"))?;
        self.position += 1;
        Ok(digit as usize)
    }

    fn atom(&mut self) -> Result<usize, SmartsError> {
        let character = self.peek().ok_or_else(|| self.error("expected atom"))?;
        let (expr, map) = if character == '[' {
            self.position += 1;
            let expr = self.atom_low()?;
            let mut map = None;
            if self.peek() == Some(':') {
                self.position += 1;
                map = Some(self.number().ok_or_else(|| self.error("map index"))?);
            }
            if self.peek() != Some(']') {
                return Err(self.error("expected ']'"));
            }
            self.position += 1;
            (expr, map)
        } else if character == '*' {
            self.position += 1;
            (Expr::Primitive(AtomPrimitive::Any), None)
        } else {
            (self.organic_atom()?, None)
        };
        self.atoms.push(PatternAtom { expr, map });
        Ok(self.atoms.len() - 1)
    }

    fn organic_atom(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        for (symbol, number, aromatic) in [
            ("Cl", 17, false),
            ("Br", 35, false),
            ("B", 5, false),
            ("C", 6, false),
            ("N", 7, false),
            ("O", 8, false),
            ("S", 16, false),
            ("P", 15, false),
            ("F", 9, false),
            ("I", 53, false),
            ("b", 5, true),
            ("c", 6, true),
            ("n", 7, true),
            ("o", 8, true),
            ("s", 16, true),
            ("p", 15, true),
        ] {
            if self.starts_with(symbol) {
                self.position += symbol.len();
                return Ok(element_expr(number, Some(aromatic)));
            }
        }
        Err(self.error("unknown atom"))
    }

    fn starts_with(&self, text: &str) -> bool {
        text.chars()
            .enumerate()
            .all(|(offset, c)| self.chars.get(self.position + offset) == Some(&c))
    }

    fn number(&mut self) -> Option<usize> {
        let start = self.position;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.position += 1;
        }
        (self.position > start).then(|| {
            self.chars[start..self.position]
                .iter()
                .collect::<String>()
                .parse()
                .unwrap()
        })
    }

    fn atom_low(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        let mut parts = vec![self.atom_or()?];
        while self.peek() == Some(';') {
            self.position += 1;
            parts.push(self.atom_or()?);
        }
        Ok(collapse(parts, Expr::And))
    }

    fn atom_or(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        let mut parts = vec![self.atom_and()?];
        while self.peek() == Some(',') {
            self.position += 1;
            parts.push(self.atom_and()?);
        }
        Ok(collapse(parts, Expr::Or))
    }

    fn atom_and(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        let mut parts = vec![self.atom_unary()?];
        loop {
            match self.peek() {
                Some('&') => {
                    self.position += 1;
                    parts.push(self.atom_unary()?);
                }
                Some(c) if !matches!(c, ';' | ',' | ']' | ':' | ')') => {
                    parts.push(self.atom_unary()?)
                }
                _ => break,
            }
        }
        Ok(collapse(parts, Expr::And))
    }

    fn atom_unary(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        if self.peek() == Some('!') {
            self.position += 1;
            return Ok(Expr::Not(Box::new(self.atom_unary()?)));
        }
        self.atom_primitive()
    }

    fn atom_primitive(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        let character = self
            .peek()
            .ok_or_else(|| self.error("expected primitive"))?;
        let primitive = match character {
            '*' => {
                self.position += 1;
                AtomPrimitive::Any
            }
            '#' => {
                self.position += 1;
                AtomPrimitive::AtomicNumber(
                    self.number().ok_or_else(|| self.error("atomic number"))? as u8,
                )
            }
            '$' => {
                self.position += 1;
                if self.peek() != Some('(') {
                    return Err(self.error("expected '(' after '$'"));
                }
                let start = self.position + 1;
                let mut depth = 0;
                let mut end = start;
                for (offset, c) in self.chars[self.position..].iter().enumerate() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                end = self.position + offset;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if depth != 0 {
                    return Err(self.error("unclosed recursive SMARTS"));
                }
                let inner = self.chars[start..end].iter().collect::<String>();
                self.position = end + 1;
                AtomPrimitive::Recursive(Box::new(Pattern::parse(&inner)?))
            }
            '+' | '-' => {
                self.position += 1;
                let sign = if character == '+' { 1 } else { -1 };
                let mut magnitude = self.number().map(|n| n as i32);
                if magnitude.is_none() {
                    let mut count = 1;
                    while self.peek() == Some(character) {
                        self.position += 1;
                        count += 1;
                    }
                    magnitude = Some(count);
                }
                AtomPrimitive::Charge(sign * magnitude.unwrap())
            }
            'a' if !self.starts_with("as") && !self.starts_with("al") => {
                self.position += 1;
                AtomPrimitive::Aromatic
            }
            'A' if !self.starts_with("Al")
                && !self.starts_with("Ag")
                && !self.starts_with("As")
                && !self.starts_with("Au") =>
            {
                self.position += 1;
                AtomPrimitive::Aliphatic
            }
            'X' | 'x' | 'D' | 'v' => {
                self.position += 1;
                let value = self.number().unwrap_or(1);
                match character {
                    'X' => AtomPrimitive::TotalConnectivity(value),
                    'x' => AtomPrimitive::RingConnectivity(value),
                    'D' => AtomPrimitive::Degree(value),
                    _ => AtomPrimitive::Valence(value),
                }
            }
            'H' if self.position > 0 && self.chars[self.position - 1] != '['
                || self
                    .chars
                    .get(self.position + 1)
                    .is_some_and(|c| c.is_ascii_digit()) =>
            {
                self.position += 1;
                AtomPrimitive::HydrogenCount(self.number().unwrap_or(1))
            }
            'R' if !self.starts_with("Rb")
                && !self.starts_with("Ru")
                && !self.starts_with("Rh") =>
            {
                self.position += 1;
                AtomPrimitive::RingMembership(self.number())
            }
            'r' => {
                self.position += 1;
                AtomPrimitive::RingSize(self.number())
            }
            _ => return self.bracket_element(),
        };
        Ok(Expr::Primitive(primitive))
    }

    fn bracket_element(&mut self) -> Result<Expr<AtomPrimitive>, SmartsError> {
        const ELEMENTS: &[(&str, u8)] = &[
            ("Cl", 17),
            ("Br", 35),
            ("Na", 11),
            ("Mg", 12),
            ("Li", 3),
            ("Zn", 30),
            ("Ca", 20),
            ("Fe", 26),
            ("Cu", 29),
            ("Mn", 25),
            ("Co", 27),
            ("Ni", 28),
            ("Se", 34),
            ("Si", 14),
            ("Al", 13),
            ("As", 33),
            ("Xe", 54),
            ("Rb", 37),
            ("Cs", 55),
            ("H", 1),
            ("B", 5),
            ("C", 6),
            ("N", 7),
            ("O", 8),
            ("F", 9),
            ("P", 15),
            ("S", 16),
            ("K", 19),
            ("I", 53),
        ];
        for &(symbol, number) in ELEMENTS {
            if self.starts_with(symbol) {
                self.position += symbol.len();
                return Ok(element_expr(number, Some(false)));
            }
        }
        for (symbol, number) in [
            ("se", 34u8),
            ("as", 33),
            ("b", 5),
            ("c", 6),
            ("n", 7),
            ("o", 8),
            ("p", 15),
            ("s", 16),
        ] {
            if self.starts_with(symbol) {
                self.position += symbol.len();
                return Ok(element_expr(number, Some(true)));
            }
        }
        Err(self.error("unknown primitive"))
    }

    fn bond_expr(&mut self) -> Result<Option<Expr<BondPrimitive>>, SmartsError> {
        if !self.peek().is_some_and(is_bond_character) {
            return Ok(None);
        }
        Ok(Some(self.bond_low()?))
    }

    fn bond_low(&mut self) -> Result<Expr<BondPrimitive>, SmartsError> {
        let mut parts = vec![self.bond_or()?];
        while self.peek() == Some(';') {
            self.position += 1;
            parts.push(self.bond_or()?);
        }
        Ok(collapse(parts, Expr::And))
    }

    fn bond_or(&mut self) -> Result<Expr<BondPrimitive>, SmartsError> {
        let mut parts = vec![self.bond_and()?];
        while self.peek() == Some(',') {
            self.position += 1;
            parts.push(self.bond_and()?);
        }
        Ok(collapse(parts, Expr::Or))
    }

    fn bond_and(&mut self) -> Result<Expr<BondPrimitive>, SmartsError> {
        let mut parts = vec![self.bond_unary()?];
        loop {
            match self.peek() {
                Some('&') => {
                    self.position += 1;
                    parts.push(self.bond_unary()?);
                }
                Some(c) if is_bond_character(c) && c != ';' && c != ',' => {
                    parts.push(self.bond_unary()?)
                }
                _ => break,
            }
        }
        Ok(collapse(parts, Expr::And))
    }

    fn bond_unary(&mut self) -> Result<Expr<BondPrimitive>, SmartsError> {
        let character = self.peek().ok_or_else(|| self.error("expected bond"))?;
        self.position += 1;
        Ok(match character {
            '!' => Expr::Not(Box::new(self.bond_unary()?)),
            '-' => Expr::Primitive(BondPrimitive::Single),
            '=' => Expr::Primitive(BondPrimitive::Double),
            '#' => Expr::Primitive(BondPrimitive::Triple),
            ':' => Expr::Primitive(BondPrimitive::Aromatic),
            '~' => Expr::Primitive(BondPrimitive::Any),
            '@' => Expr::Primitive(BondPrimitive::Ring),
            _ => return Err(self.error("unknown bond")),
        })
    }
}

fn is_bond_character(character: char) -> bool {
    matches!(
        character,
        '-' | '=' | '#' | ':' | '~' | '@' | '!' | ';' | ',' | '&'
    )
}

fn element_expr(number: u8, aromatic: Option<bool>) -> Expr<AtomPrimitive> {
    let element = Expr::Primitive(AtomPrimitive::AtomicNumber(number));
    match aromatic {
        Some(true) => Expr::And(vec![element, Expr::Primitive(AtomPrimitive::Aromatic)]),
        Some(false) => Expr::And(vec![element, Expr::Primitive(AtomPrimitive::Aliphatic)]),
        None => element,
    }
}

fn collapse<P>(mut parts: Vec<Expr<P>>, combine: fn(Vec<Expr<P>>) -> Expr<P>) -> Expr<P> {
    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        combine(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smirnoff::molecule::tests::{benzene, molecule};

    fn ethanol() -> Molecule {
        // C0 C1 O2, H3-5 on C0, H6-7 on C1, H8 on O2
        molecule(
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
        )
    }

    #[test]
    fn parses_and_matches_sage_style_patterns() {
        let ethanol = ethanol();
        let bond = Pattern::parse("[#6X4:1]-[#8X2H1+0:2]").unwrap();
        assert_eq!(bond.match_tagged(&ethanol), vec![vec![1, 2]]);
        let ch = Pattern::parse("[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]").unwrap();
        assert_eq!(ch.match_tagged(&ethanol), vec![vec![6], vec![7]]);
        let torsion = Pattern::parse("[*:1]-[#6X4:2]-[#6X4:3]-[*:4]").unwrap();
        assert_eq!(torsion.match_tagged(&ethanol).len(), 3 * 3 * 2);
        let recursive = Pattern::parse("[#6X4$(*-[#8]):1]").unwrap();
        assert_eq!(recursive.match_tagged(&ethanol), vec![vec![1]]);
        let negated = Pattern::parse("[#6;!$(*-[#8]):1]").unwrap();
        assert_eq!(negated.match_tagged(&ethanol), vec![vec![0]]);
    }

    #[test]
    fn distinguishes_aromatic_ring_bonds() {
        let benzene = benzene();
        let aromatic = Pattern::parse("[#6a:1]:[#6a:2]").unwrap();
        assert_eq!(aromatic.match_tagged(&benzene).len(), 12);
        let single = Pattern::parse("[#6:1]-[#6:2]").unwrap();
        assert!(single.match_tagged(&benzene).is_empty());
        let implicit = Pattern::parse("[#6:1][#6:2]").unwrap();
        assert_eq!(implicit.match_tagged(&benzene).len(), 12);
        let ring = Pattern::parse("[*;r6:1]~;@[*;r6:2]").unwrap();
        assert_eq!(ring.match_tagged(&benzene).len(), 12);
        let closure = Pattern::parse("[#6:1]1:[#6]:[#6]:[#6]:[#6]:[#6]:1").unwrap();
        assert_eq!(closure.match_tagged(&benzene).len(), 6);
        let x = Pattern::parse("[#6X3x2:1]").unwrap();
        assert_eq!(x.match_tagged(&benzene).len(), 6);
    }

    #[test]
    fn rejects_malformed_patterns() {
        assert!(Pattern::parse("[#6").is_err());
        assert!(Pattern::parse("[#6](").is_err());
        assert!(Pattern::parse("[#6]1-[#6]").is_err());
    }
}

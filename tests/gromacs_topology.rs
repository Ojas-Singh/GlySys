//! The GROMACS topology has to carry what the Amber files carry: the 1-4
//! scale factors of each force field and the exclusions of rigid water.
use glysys::{BuildOptions, SystemBuilder};

fn prepare(pdb: &str, add_water: bool) -> glysys::ParameterizedSystem {
    let options = BuildOptions {
        add_water,
        add_ions: false,
        padding_angstrom: 6.0,
        ..Default::default()
    };
    SystemBuilder::new(options)
        .unwrap()
        .prepare_pdb_str(pdb)
        .unwrap()
}

fn topology(pdb: &str, add_water: bool) -> String {
    prepare(pdb, add_water).bundle_strings().unwrap()["system.top"].clone()
}

fn section<'a>(topology: &'a str, name: &str) -> Vec<Vec<&'a str>> {
    let header = format!("[ {name} ]");
    topology
        .lines()
        .skip_while(|line| line.trim() != header)
        .skip(1)
        .take_while(|line| !line.trim_start().starts_with('['))
        .map(|line| line.split(';').next().unwrap_or_default())
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
        .collect()
}

/// Charge scale factors of the 1-4 pairs, with how many pairs carry each.
fn charge_scales(topology: &str) -> Vec<(String, usize)> {
    let mut counts = std::collections::BTreeMap::new();
    for row in section(topology, "pairs") {
        assert_eq!(row.len(), 8, "pair row {row:?}");
        assert_eq!(row[2], "2", "pairs carry their own scale factors");
        *counts.entry(format!("{:.4}", row[3].parse::<f64>().unwrap())).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

#[test]
fn glycan_one_four_pairs_carry_the_glycam_scale_factors() {
    // GLYCAM06 sets SCEE = SCNB = 1 for carbohydrate torsions, so their 1-4
    // electrostatics and Lennard-Jones terms count in full; the few amide
    // torsions of an N-acetyl group keep the Amber factors. Each row has to
    // state the factors of its own pair.
    let system = prepare(include_str!("../example/gotw-a9e1ab91.pdb"), false);
    let top = system.bundle_strings().unwrap()["system.top"].clone();
    let expected: std::collections::BTreeMap<(usize, usize), (f64, f64)> = system
        .one_four_pairs()
        .into_iter()
        .map(|(pair, scee, scnb)| ((pair[0] + 1, pair[1] + 1), (scee, scnb)))
        .collect();
    let rows = section(&top, "pairs");
    assert_eq!(rows.len(), expected.len());
    let atoms = system.atoms();
    for row in &rows {
        assert_eq!(row[2], "2", "pairs carry their own scale factors");
        let (first, second): (usize, usize) = (row[0].parse().unwrap(), row[1].parse().unwrap());
        let (scee, scnb) = expected[&(first, second)];
        let charge_scale: f64 = row[3].parse().unwrap();
        assert!((charge_scale - 1. / scee).abs() < 1e-9, "{row:?}");
        let well = (atoms[first - 1].lennard_jones_epsilon()
            * atoms[second - 1].lennard_jones_epsilon())
        .sqrt()
            * 4.184
            / scnb;
        let written: f64 = row[7].parse().unwrap();
        assert!((written - well).abs() <= 1e-6 * well.max(1e-9), "{row:?}");
        let charges: (f64, f64) = (row[4].parse().unwrap(), row[5].parse().unwrap());
        assert!((charges.0 - atoms[first - 1].charge()).abs() < 1e-7);
        assert!((charges.1 - atoms[second - 1].charge()).abs() < 1e-7);
    }
    let scales = charge_scales(&top);
    let unscaled = scales.iter().find(|(scale, _)| scale == "1.0000").unwrap().1;
    assert!(unscaled > 20 * (rows.len() - unscaled), "{scales:?}");
}

#[test]
fn protein_one_four_pairs_keep_the_amber_scale_factors() {
    let top = topology(include_str!("fixtures/dipeptide.pdb"), false);
    let scales = charge_scales(&top);
    assert_eq!(scales.len(), 1, "{scales:?}");
    assert_eq!(scales[0].0, "0.8333");
}

#[test]
fn every_pair_of_atoms_in_a_water_is_excluded() {
    let top = topology(include_str!("fixtures/dipeptide.pdb"), true);
    let waters = section(&top, "settles").len();
    assert!(waters > 10);
    let mut excluded = std::collections::BTreeSet::new();
    for row in section(&top, "exclusions") {
        let atoms: Vec<usize> = row.iter().map(|value| value.parse().unwrap()).collect();
        for &other in &atoms[1..] {
            excluded.insert((atoms[0].min(other), atoms[0].max(other)));
        }
    }
    assert_eq!(excluded.len(), 3 * waters);
    for row in section(&top, "settles") {
        let oxygen: usize = row[0].parse().unwrap();
        for pair in [(oxygen, oxygen + 1), (oxygen, oxygen + 2), (oxygen + 1, oxygen + 2)] {
            assert!(excluded.contains(&pair), "water at {oxygen} lacks {pair:?}");
        }
    }
}

//! Immutable packing, preserving the CPU evaluator's exclusions and first 1–4 scale.
use crate::device::{Atom, Error, Special, Term, Topology};
use glysys::{ParameterizedSystem, Vec3};
use glysys_energy::{EnergyEvaluator, EnergyOptions};
use std::collections::BTreeMap;

pub struct PreparedTopology {
    atoms: Vec<Atom>,
    terms: Vec<Term>,
    incidence: Vec<[u32; 2]>,
    specials: Vec<Special>,
    dense_special_lookup_base: Option<u32>,
    pub origin: Vec3,
}
impl PreparedTopology {
    pub fn new(
        system: &ParameterizedSystem,
        options: &EnergyOptions,
        groups: &[u8],
    ) -> Result<Self, Error> {
        EnergyEvaluator::new(system, options.clone())
            .map_err(|_| Error::Input("energy options"))?;
        let n = system.atom_count();
        if n == 0 || n > u32::MAX as usize || groups.len() != n || groups.iter().any(|g| *g > 2) {
            return Err(Error::Input("atom groups"));
        }
        let origin = system.atoms()[0].position();
        let mut atoms: Vec<_> = system
            .atoms()
            .iter()
            .zip(groups)
            .map(|(a, g)| Atom {
                ff: [
                    a.charge() as f32,
                    a.lennard_jones_radius() as f32,
                    a.lennard_jones_epsilon() as f32,
                    a.gb_radius() as f32,
                ],
                more: [a.gb_screen() as f32, *g as f32, 0., 0.],
                ranges: [0; 4],
            })
            .collect();
        let mut terms = Vec::new();
        let mut lists = vec![Vec::new(); n];
        let mut push =
            |ids: &[usize], parameters: [f32; 4], reference: [f32; 4]| -> Result<(), Error> {
                if ids.iter().any(|i| *i >= n) {
                    return Err(Error::Input("term atom index"));
                }
                let index = u32::try_from(terms.len()).map_err(|_| Error::Capacity)?;
                let mut packed = [0; 4];
                for (local, &id) in ids.iter().enumerate() {
                    packed[local] = id as u32;
                    lists[id].push([index, local as u32]);
                }
                terms.push(Term {
                    ids: packed,
                    parameters,
                    reference,
                });
                Ok(())
            };
        for t in system.bonds() {
            push(
                &t.atoms(),
                [t.force() as f32, t.length() as f32, 0., 0.],
                [0.; 4],
            )?;
        }
        for t in system.angles() {
            push(
                &t.atoms(),
                [t.force() as f32, t.radians() as f32, 0., 1.],
                [0.; 4],
            )?;
        }
        for t in system.dihedrals() {
            push(
                &t.atoms(),
                [
                    t.force() as f32,
                    t.periodicity() as f32,
                    t.phase() as f32,
                    if t.is_improper() { 3. } else { 2. },
                ],
                [0.; 4],
            )?;
        }
        for r in &options.restraints {
            push(
                &[r.atom],
                [r.force as f32, 0., 0., 4.],
                [
                    (r.reference.x - origin.x) as f32,
                    (r.reference.y - origin.y) as f32,
                    (r.reference.z - origin.z) as f32,
                    0.,
                ],
            )?;
        }
        let mut incidence = Vec::new();
        for (a, list) in atoms.iter_mut().zip(lists) {
            a.ranges[0] = u32::try_from(incidence.len()).map_err(|_| Error::Capacity)?;
            incidence.extend(list);
            a.ranges[1] = u32::try_from(incidence.len()).map_err(|_| Error::Capacity)?;
        }
        let mut exceptions: Vec<BTreeMap<usize, (f32, f32)>> = system
            .exclusions()
            .iter()
            .map(|set| set.iter().map(|i| (*i, (0., 0.))).collect())
            .collect();
        let mut scales = BTreeMap::new();
        for t in system.dihedrals().iter().filter(|t| !t.is_improper()) {
            let ids = t.atoms();
            let key = (ids[0].min(ids[3]), ids[0].max(ids[3]));
            scales.entry(key).or_insert((
                t.electrostatic_14_scale() as f32,
                t.lennard_jones_14_scale() as f32,
            ));
        }
        for ((a, b), scale) in scales {
            exceptions[a].insert(b, scale);
            exceptions[b].insert(a, scale);
        }
        let mut specials = Vec::new();
        for (a, list) in atoms.iter_mut().zip(exceptions) {
            a.ranges[2] = u32::try_from(specials.len()).map_err(|_| Error::Capacity)?;
            specials.extend(list.into_iter().map(|(other, (scee, scnb))| Special {
                other: other as u32,
                scee,
                scnb,
                spare: 0,
            }));
            a.ranges[3] = u32::try_from(specials.len()).map_err(|_| Error::Capacity)?;
        }
        Ok(Self {
            atoms,
            terms,
            incidence,
            specials,
            dense_special_lookup_base: None,
            origin,
        })
    }

    /// Append a dense symmetric pair-scale table when it fits the caller's
    /// explicit allocation allowance. Rows are indexed by `(atom_i, atom_j)`.
    /// `None` leaves the sorted sparse exception lists available as fallback.
    pub fn append_dense_special_lookup(&mut self, max_bytes: u64) -> Result<Option<u32>, Error> {
        if let Some(base) = self.dense_special_lookup_base {
            return Ok(Some(base));
        }
        let n = self.atoms.len();
        let entries = n.checked_mul(n).ok_or(Error::Capacity)?;
        let bytes = (entries as u64)
            .checked_mul(std::mem::size_of::<Special>() as u64)
            .ok_or(Error::Capacity)?;
        if bytes > max_bytes {
            return Ok(None);
        }
        let base = u32::try_from(self.specials.len()).map_err(|_| Error::Capacity)?;
        let end = usize::try_from(base)
            .ok()
            .and_then(|base| base.checked_add(entries))
            .ok_or(Error::Capacity)?;
        if end > u32::MAX as usize {
            return Err(Error::Capacity);
        }
        let mut table = Vec::with_capacity(entries);
        for _i in 0..n {
            for j in 0..n {
                table.push(Special {
                    other: j as u32,
                    scee: 1.0,
                    scnb: 1.0,
                    spare: 0,
                });
            }
        }
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            for entry in &self.specials[atom.ranges[2] as usize..atom.ranges[3] as usize] {
                let other = entry.other as usize;
                if other >= n {
                    return Err(Error::Input("special pair atom index"));
                }
                table[atom_index * n + other].scee = entry.scee;
                table[atom_index * n + other].scnb = entry.scnb;
            }
        }
        self.specials.extend(table);
        self.dense_special_lookup_base = Some(base);
        Ok(Some(base))
    }

    pub fn dense_special_lookup_base(&self) -> Option<u32> {
        self.dense_special_lookup_base
    }

    pub fn view(&self) -> Topology<'_> {
        Topology {
            atoms: &self.atoms,
            terms: &self.terms,
            incidence: &self.incidence,
            specials: &self.specials,
        }
    }
    pub fn coordinates(
        &self,
        coordinates: &[Vec3],
        movable: &[bool],
    ) -> Result<Vec<[f32; 4]>, Error> {
        if coordinates.len() != self.atoms.len() || movable.len() != coordinates.len() {
            return Err(Error::Input("coordinate or movable mask dimensions"));
        }
        Ok(coordinates
            .iter()
            .zip(movable)
            .map(|(p, m)| {
                [
                    (p.x - self.origin.x) as f32,
                    (p.y - self.origin.y) as f32,
                    (p.z - self.origin.z) as f32,
                    if *m { 1. } else { 0. },
                ]
            })
            .collect())
    }
}

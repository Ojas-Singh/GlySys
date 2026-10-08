//! Nose–Hoover temperature coupling and Parrinello–Rahman pressure coupling
//! for the leap-frog integrator.
//!
//! The equations and their discretization are those of the GROMACS `md`
//! integrator, so a GROMACS recipe (`tcoupl = Nose-Hoover`,
//! `pcoupl = Parrinello-Rahman`, `pcoupltype = isotropic`) keeps its meaning:
//!
//! - each temperature group `g` has a friction `ξ_g` with
//!   `dξ_g/dt = (T_g − T0_g) / Q_g`, `Q_g = (τ_g / 2π)² T0_g`, where `T_g` is
//!   the kinetic temperature of the group's half-step velocities;
//! - the box lengths have velocities `ḃ` with
//!   `b̈_d = W⁻¹ V a b_d`, `W⁻¹ = 4π² β / (3 τ_p² max(b))` and
//!   `a = (P − P0) · (1/3) Σ_d 1/b_d²`, with compressibility `β`;
//! - a velocity is advanced as
//!   `v' = [v (1 − f) + a Δt − Δt_p M v] / (1 + f)`, `f = ½ Δt_T ξ_g`,
//!   `M_d = ḃ_d / b_d`, and after the position update the box and all
//!   coordinates are scaled by `μ_d = 1 + Δt_p ḃ_d / b_d`.
//!
//! Both couplings act every `n` steps with the time step `n Δt`, on the step
//! after the one whose kinetic energy and pressure they use.
//!
//! The pressure is the scalar atomic pressure including constraint forces,
//! where GROMACS weights the three diagonal components of the pressure tensor
//! by `1/b_d²`. For an isotropic liquid the two have the same average.
use super::{Result, SimulationProtocol, invalid};
use serde::{Deserialize, Serialize};

/// GROMACS' default `nsttcouple` and `nstpcouple`.
pub const DEFAULT_COUPLING_INTERVAL: usize = 10;
/// GROMACS' default `nstcomm`.
pub const DEFAULT_COM_REMOVAL_INTERVAL: usize = 100;
pub const DEFAULT_PRESSURE_TAU_PS: f64 = 5.0;
/// Isothermal compressibility of water, bar⁻¹.
pub const WATER_COMPRESSIBILITY_PER_BAR: f64 = 4.5e-5;
/// A coupling must act at least this many times per coupling period.
const STEPS_PER_PERIOD: f64 = 20.0;

/// Thermostat and barostat variables carried from step to step.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CouplingState {
    /// Nose–Hoover friction of each temperature group, ps⁻¹.
    pub thermostat_velocity: Vec<f64>,
    /// Its time integral, for the conserved energy.
    pub thermostat_position: Vec<f64>,
    /// Velocities of the three box lengths, Å/ps.
    pub box_velocity: [f64; 3],
    /// Pressure the barostat acts on next, bar.
    pub pressure_bar: f64,
    /// Step whose forces and velocities gave `pressure_bar`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_step: Option<usize>,
}

impl CouplingState {
    pub fn new(groups: usize) -> Self {
        Self {
            thermostat_velocity: vec![0.; groups],
            thermostat_position: vec![0.; groups],
            ..Self::default()
        }
    }
}

/// Atoms a group name selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    All,
    Water,
    NotWater,
}

const WATER_NAMES: [&str; 6] = ["wat", "water", "sol", "hoh", "tip3", "tip3p"];

/// Group names as GROMACS index files and the GOTW recipe write them:
/// `System`, `WAT` (or `Water`, `SOL`), and the rest of the system as
/// `System_&_!WAT` (or `non-Water`).
fn selection(name: &str) -> Result<Selection> {
    let lower = name.trim().to_ascii_lowercase();
    if lower == "system" {
        return Ok(Selection::All);
    }
    if WATER_NAMES.contains(&lower.as_str()) {
        return Ok(Selection::Water);
    }
    let complement = lower
        .strip_prefix("system_&_!")
        .or_else(|| lower.strip_prefix("non-"))
        .or_else(|| lower.strip_prefix('!'));
    if complement.is_some_and(|rest| WATER_NAMES.contains(&rest)) {
        return Ok(Selection::NotWater);
    }
    Err(invalid(format!(
        "coupling group '{name}' is not understood; use System, WAT or System_&_!WAT"
    )))
}

fn members(selection: Selection, is_water: &[bool]) -> Vec<usize> {
    (0..is_water.len())
        .filter(|&atom| match selection {
            Selection::All => true,
            Selection::Water => is_water[atom],
            Selection::NotWater => !is_water[atom],
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct TemperatureGroup {
    pub name: String,
    pub reference_temperature_k: f64,
    pub tau_ps: f64,
    /// Kinetic degrees of freedom: three per atom, less one per constraint
    /// and the share of removed center-of-mass motion.
    pub degrees_of_freedom: f64,
}

impl TemperatureGroup {
    /// `1/Q` of the thermostat, K⁻¹ ps⁻².
    pub fn inverse_mass(&self) -> f64 {
        let period = self.tau_ps / std::f64::consts::TAU;
        1. / (period * period * self.reference_temperature_k)
    }
}

/// The coupling set-up of one prepared system and protocol.
#[derive(Clone, Debug)]
pub struct CouplingPlan {
    pub groups: Vec<TemperatureGroup>,
    pub group_of_atom: Vec<u32>,
    /// Atoms whose common center-of-mass velocity is removed together.
    pub com_groups: Vec<Vec<usize>>,
    pub temperature_interval: usize,
    pub pressure_interval: usize,
    /// Zero when center-of-mass motion is left alone.
    pub com_interval: usize,
    pub pressure_tau_ps: f64,
    pub compressibility_per_bar: f64,
    pub reference_pressure_bar: f64,
    pub has_pressure_coupling: bool,
}

impl CouplingPlan {
    /// `constraints_of_atom` counts the distance constraints each atom takes
    /// part in; every constraint removes half a degree of freedom from each
    /// of its two atoms.
    pub fn new(
        protocol: &SimulationProtocol,
        is_water: &[bool],
        constraints_of_atom: &[u32],
    ) -> Result<Self> {
        let atoms = is_water.len();
        if constraints_of_atom.len() != atoms || atoms == 0 {
            return Err(invalid("coupling plan needs one entry per atom"));
        }
        let dt_ps = protocol.timestep_fs * 0.001;

        let requested: Vec<(String, f64, f64)> = if protocol.thermostat_groups.is_empty() {
            if protocol.friction_per_ps <= 0. {
                return Err(invalid(
                    "Nose-Hoover needs thermostat groups or a positive friction (1/tau)",
                ));
            }
            vec![(
                "System".into(),
                protocol.temperature_k,
                1. / protocol.friction_per_ps,
            )]
        } else {
            protocol
                .thermostat_groups
                .iter()
                .map(|g| (g.name.clone(), g.reference_temperature_k, g.tau_ps))
                .collect()
        };
        let mut group_of_atom = vec![u32::MAX; atoms];
        for (index, (name, _, _)) in requested.iter().enumerate() {
            for atom in members(selection(name)?, is_water) {
                if group_of_atom[atom] != u32::MAX {
                    return Err(invalid("temperature groups overlap"));
                }
                group_of_atom[atom] = index as u32;
            }
        }
        if group_of_atom.contains(&u32::MAX) {
            return Err(invalid("temperature groups do not cover every atom"));
        }

        let atom_dof = |atom: usize| 3. - 0.5 * constraints_of_atom[atom] as f64;
        let mut dof = vec![0.; requested.len()];
        for atom in 0..atoms {
            dof[group_of_atom[atom] as usize] += atom_dof(atom);
        }

        let com_mode = protocol
            .com_mode
            .as_deref()
            .map(|mode| mode.trim().to_ascii_lowercase());
        let com_groups: Vec<Vec<usize>> = match com_mode.as_deref() {
            Some("none") => Vec::new(),
            Some("linear") | None => {
                if protocol.com_groups.is_empty() {
                    vec![(0..atoms).collect()]
                } else {
                    let mut seen = vec![false; atoms];
                    let mut groups = Vec::new();
                    for name in &protocol.com_groups {
                        let group = members(selection(name)?, is_water);
                        if group.iter().any(|&atom| std::mem::replace(&mut seen[atom], true)) {
                            return Err(invalid("center-of-mass groups overlap"));
                        }
                        groups.push(group);
                    }
                    groups
                }
            }
            Some(other) => {
                return Err(invalid(format!(
                    "center-of-mass mode '{other}' is not supported; use linear or none"
                )));
            }
        };
        let com_groups: Vec<Vec<usize>> = com_groups.into_iter().filter(|g| !g.is_empty()).collect();
        let com_interval = if com_groups.is_empty() {
            0
        } else {
            protocol
                .com_removal_interval
                .unwrap_or(DEFAULT_COM_REMOVAL_INTERVAL)
        };
        // Removing a group's center-of-mass motion takes three degrees of
        // freedom, shared among the temperature groups its atoms belong to.
        if com_interval > 0 {
            let unremoved = dof.clone();
            for group in &com_groups {
                let total: f64 = group.iter().map(|&atom| atom_dof(atom)).sum();
                if total <= 3. {
                    continue;
                }
                let mut share = vec![0.; requested.len()];
                for &atom in group {
                    share[group_of_atom[atom] as usize] += atom_dof(atom);
                }
                for (index, part) in share.iter().enumerate() {
                    dof[index] -= 3. * part / total;
                }
            }
            if dof.iter().zip(&unremoved).any(|(d, u)| *d <= 0. && *u > 0.) {
                return Err(invalid("a temperature group has no degrees of freedom left"));
            }
        }

        let mut groups = Vec::with_capacity(requested.len());
        for ((name, reference_temperature_k, tau_ps), degrees_of_freedom) in
            requested.into_iter().zip(dof)
        {
            if degrees_of_freedom <= 0. {
                return Err(invalid(format!("temperature group '{name}' is empty")));
            }
            if !(reference_temperature_k > 0. && tau_ps > 0.) {
                return Err(invalid(
                    "temperature groups need a positive temperature and tau",
                ));
            }
            groups.push(TemperatureGroup {
                name,
                reference_temperature_k,
                tau_ps,
                degrees_of_freedom,
            });
        }

        let shortest_tau = groups.iter().map(|g| g.tau_ps).fold(f64::INFINITY, f64::min);
        let temperature_interval = interval(
            protocol.temperature_coupling_interval,
            shortest_tau,
            dt_ps,
            "temperature",
        )?;
        let has_pressure_coupling = protocol.has_npt();
        let pressure_tau_ps = protocol.pressure_tau_ps.unwrap_or(DEFAULT_PRESSURE_TAU_PS);
        let pressure_interval = if has_pressure_coupling {
            interval(
                protocol.pressure_coupling_interval,
                pressure_tau_ps,
                dt_ps,
                "pressure",
            )?
        } else {
            protocol
                .pressure_coupling_interval
                .unwrap_or(DEFAULT_COUPLING_INTERVAL)
        };
        let compressibility_per_bar = protocol
            .pressure_compressibility_bar_inverse
            .first()
            .copied()
            .unwrap_or(WATER_COMPRESSIBILITY_PER_BAR);
        Ok(Self {
            groups,
            group_of_atom,
            com_groups,
            temperature_interval,
            pressure_interval,
            com_interval,
            pressure_tau_ps,
            compressibility_per_bar,
            reference_pressure_bar: protocol.pressure_bar,
            has_pressure_coupling,
        })
    }

    pub fn degrees_of_freedom(&self) -> f64 {
        self.groups.iter().map(|g| g.degrees_of_freedom).sum()
    }

    /// True on the steps where a coupling with this interval acts: the step
    /// after each multiple of the interval.
    pub fn acts_on(step: usize, interval: usize) -> bool {
        interval <= 1 || step % interval == 1
    }

    /// True when the pressure of `step` is needed by the barostat.
    pub fn needs_pressure(&self, step: usize) -> bool {
        self.has_pressure_coupling && step.is_multiple_of(self.pressure_interval.max(1))
    }

    /// Advance the thermostat frictions over `dt_ps` from the half-step
    /// temperatures of the groups.
    pub fn advance_thermostats(&self, state: &mut CouplingState, temperatures: &[f64], dt_ps: f64) {
        for (index, group) in self.groups.iter().enumerate() {
            let old = state.thermostat_velocity[index];
            let new = old
                + dt_ps * group.inverse_mass() * (temperatures[index] - group.reference_temperature_k);
            state.thermostat_velocity[index] = new;
            state.thermostat_position[index] += 0.5 * dt_ps * (old + new);
        }
    }

    /// Energy stored in the thermostats, kcal/mol: with the kinetic and
    /// potential energy it is conserved in an NVT run.
    pub fn thermostat_energy(&self, state: &CouplingState) -> f64 {
        self.groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let kt = super::KB * group.reference_temperature_k;
                let velocity = state.thermostat_velocity[index];
                group.degrees_of_freedom
                    * kt
                    * (0.5 * velocity * velocity / (group.inverse_mass() * group.reference_temperature_k)
                        + state.thermostat_position[index])
            })
            .sum()
    }

    /// Accelerate the box velocities over `dt_ps` from a pressure difference.
    pub fn accelerate_box(
        &self,
        box_velocity: &mut [f64; 3],
        box_angstrom: [f64; 3],
        pressure_bar: f64,
        dt_ps: f64,
    ) {
        let longest = box_angstrom.iter().copied().fold(0., f64::max);
        let inverse_mass = 4. * std::f64::consts::PI.powi(2) * self.compressibility_per_bar
            / (3. * self.pressure_tau_ps * self.pressure_tau_ps * longest);
        let volume: f64 = box_angstrom.iter().product();
        let relative = (pressure_bar - self.reference_pressure_bar)
            * box_angstrom.iter().map(|b| 1. / (b * b)).sum::<f64>()
            / 3.;
        for (velocity, length) in box_velocity.iter_mut().zip(box_angstrom) {
            *velocity += dt_ps * inverse_mass * volume * relative * length;
        }
    }
}

/// Coupling interval: the requested one if it resolves the coupling period,
/// otherwise the default shortened to do so.
fn interval(requested: Option<usize>, tau_ps: f64, dt_ps: f64, what: &str) -> Result<usize> {
    let longest = ((tau_ps / (STEPS_PER_PERIOD * dt_ps)).floor() as usize).max(1);
    match requested {
        Some(steps) if steps > longest => Err(invalid(format!(
            "{what} coupling every {steps} steps is too coarse for tau = {tau_ps} ps; use at most {longest}"
        ))),
        Some(steps) => Ok(steps),
        None => Ok(DEFAULT_COUPLING_INTERVAL.min(longest)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ensemble, PressureCoupling, SimulationStage, SolventModel, Thermostat, ThermostatGroup};

    fn protocol() -> SimulationProtocol {
        SimulationProtocol {
            timestep_fs: 2.0,
            solvent: SolventModel::Explicit,
            thermostat: Thermostat::NoseHoover,
            constraints: crate::ConstraintModel::Settle,
            thermostat_groups: vec![
                ThermostatGroup {
                    name: "WAT".into(),
                    tau_ps: 1.0,
                    reference_temperature_k: 300.0,
                },
                ThermostatGroup {
                    name: "System_&_!WAT".into(),
                    tau_ps: 1.0,
                    reference_temperature_k: 300.0,
                },
            ],
            com_mode: Some("linear".into()),
            com_groups: vec!["WAT".into(), "System_&_!WAT".into()],
            ..SimulationProtocol::default()
        }
    }

    /// Two rigid waters and a five-atom solute with one constrained bond.
    fn atoms() -> (Vec<bool>, Vec<u32>) {
        let mut is_water = vec![true; 6];
        is_water.extend([false; 5]);
        let mut constraints = vec![2; 6];
        constraints.extend([1, 1, 0, 0, 0]);
        (is_water, constraints)
    }

    #[test]
    fn group_names_of_the_gotw_recipe_are_understood() {
        assert_eq!(selection("WAT").unwrap(), Selection::Water);
        assert_eq!(selection("Water").unwrap(), Selection::Water);
        assert_eq!(selection("System_&_!WAT").unwrap(), Selection::NotWater);
        assert_eq!(selection("non-Water").unwrap(), Selection::NotWater);
        assert_eq!(selection("System").unwrap(), Selection::All);
        assert!(selection("Protein").is_err());
    }

    #[test]
    fn degrees_of_freedom_follow_constraints_and_removed_motion() {
        let (is_water, constraints) = atoms();
        let plan = CouplingPlan::new(&protocol(), &is_water, &constraints).unwrap();
        // waters: 2 x (9 - 3) - 3; solute: 15 - 1 - 3
        assert!((plan.groups[0].degrees_of_freedom - 9.).abs() < 1e-12);
        assert!((plan.groups[1].degrees_of_freedom - 11.).abs() < 1e-12);
        assert_eq!(plan.temperature_interval, 10);
        assert_eq!(plan.com_interval, 100);

        let mut whole = protocol();
        whole.com_groups.clear();
        let plan = CouplingPlan::new(&whole, &is_water, &constraints).unwrap();
        // 26 in all, three removed in proportion 12 : 14
        assert!((plan.degrees_of_freedom() - 23.).abs() < 1e-12);
        assert!((plan.groups[0].degrees_of_freedom - (12. - 3. * 12. / 26.)).abs() < 1e-12);

        let mut none = protocol();
        none.com_mode = Some("none".into());
        let plan = CouplingPlan::new(&none, &is_water, &constraints).unwrap();
        assert!((plan.degrees_of_freedom() - 26.).abs() < 1e-12);
        assert_eq!(plan.com_interval, 0);
    }

    #[test]
    fn a_single_group_takes_its_period_from_the_friction() {
        let (is_water, constraints) = atoms();
        let mut single = protocol();
        single.thermostat_groups.clear();
        single.friction_per_ps = 2.0;
        let plan = CouplingPlan::new(&single, &is_water, &constraints).unwrap();
        assert_eq!(plan.groups.len(), 1);
        assert!((plan.groups[0].tau_ps - 0.5).abs() < 1e-12);
    }

    #[test]
    fn intervals_shorten_for_fast_coupling_and_reject_coarse_requests() {
        let (is_water, constraints) = atoms();
        let mut fast = protocol();
        fast.thermostat_groups[0].tau_ps = 0.2; // 100 steps per period: every 5
        let plan = CouplingPlan::new(&fast, &is_water, &constraints).unwrap();
        assert_eq!(plan.temperature_interval, 5);
        fast.temperature_coupling_interval = Some(10);
        assert!(CouplingPlan::new(&fast, &is_water, &constraints).is_err());
        assert!(CouplingPlan::acts_on(1, 10) && CouplingPlan::acts_on(11, 10));
        assert!(!CouplingPlan::acts_on(0, 10) && !CouplingPlan::acts_on(10, 10));
        assert!(CouplingPlan::acts_on(0, 1));
    }

    #[test]
    fn thermostat_friction_grows_when_the_group_is_hot() {
        let (is_water, constraints) = atoms();
        let plan = CouplingPlan::new(&protocol(), &is_water, &constraints).unwrap();
        let mut state = CouplingState::new(2);
        plan.advance_thermostats(&mut state, &[330., 300.], 0.02);
        // (2 pi / tau)^2 (T/T0 - 1) dt
        let expected = (std::f64::consts::TAU).powi(2) * 0.1 * 0.02;
        assert!((state.thermostat_velocity[0] - expected).abs() < 1e-12);
        assert_eq!(state.thermostat_velocity[1], 0.);
        assert!((state.thermostat_position[0] - 0.5 * 0.02 * expected).abs() < 1e-15);
    }

    #[test]
    fn box_acceleration_matches_the_isotropic_parrinello_rahman_equation() {
        let (is_water, constraints) = atoms();
        let mut npt = protocol();
        npt.pressure_coupling = PressureCoupling::ParrinelloRahman;
        npt.pressure_tau_ps = Some(5.0);
        npt.pressure_compressibility_bar_inverse = vec![4.5e-5];
        npt.stages = Some(vec![SimulationStage {
            id: "production".into(),
            ensemble: Ensemble::Npt,
            steps: 100,
            barostat_adaptation: false,
        }]);
        let plan = CouplingPlan::new(&npt, &is_water, &constraints).unwrap();
        assert!(plan.needs_pressure(20) && !plan.needs_pressure(21));
        // cubic box: b'' = 4 pi^2 beta (P - P0) b / (3 tau^2)
        let mut velocity = [0.; 3];
        plan.accelerate_box(&mut velocity, [40.; 3], 101., 0.02);
        let expected = 4. * std::f64::consts::PI.powi(2) * 4.5e-5 * 100. * 40. / (3. * 25.) * 0.02;
        for v in velocity {
            assert!((v - expected).abs() < 1e-12 * expected);
        }
        // at the reference pressure the box is not accelerated
        let mut velocity = [0.; 3];
        plan.accelerate_box(&mut velocity, [40., 50., 44.], 1., 0.02);
        assert_eq!(velocity, [0.; 3]);
    }
}

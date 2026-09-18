//! Backend-neutral simulation session ownership.
//!
//! A session keeps the prepared chemistry, physical protocol, integrator
//! state, and output schedule together. Native and browser adapters can use
//! this contract without making the core library depend on a UI or a GPU
//! API. The CPU driver is the reference implementation; GPU adapters attach
//! to the same state contract in their owning crates.

use crate::explicit::ExplicitSimulation;
use crate::{
    CpuSimulation, Result, SimulationProtocol, SimulationState, SolventModel, TrajectoryChunk,
};
use glysys::ParameterizedSystem;

enum Driver<'a> {
    Explicit(ExplicitSimulation<'a>),
    Implicit(CpuSimulation<'a>),
}

/// CPU-only compatibility session. High-level backend selection belongs to
/// `glysys-runtime`; this type remains private to callers that explicitly
/// need the reference implementation.
pub struct CpuSimulationSession<'a> {
    driver: Driver<'a>,
    requested_backend: String,
    actual_backend: String,
    fallback_reason: Option<String>,
}

impl<'a> CpuSimulationSession<'a> {
    /// Construct a session using the CPU reference path. `auto` is accepted
    /// as a compatibility selector; native GPU ownership is deliberately
    /// supplied by the platform adapter rather than hidden inside Rayon.
    pub fn new(
        system: &'a ParameterizedSystem,
        protocol: SimulationProtocol,
        requested_backend: impl Into<String>,
    ) -> Result<Self> {
        let requested_backend = requested_backend.into().to_ascii_lowercase();
        if !matches!(
            requested_backend.as_str(),
            "auto" | "cpu" | "webgpu" | "vulkan"
        ) {
            return Err(crate::Error::Invalid(format!(
                "unsupported simulation backend '{}', expected auto, cpu, webgpu, or vulkan",
                requested_backend
            )));
        }
        let fallback_reason = (requested_backend != "cpu").then(|| {
            "CPU reference session selected; a platform GPU adapter may replace this driver".into()
        });
        let driver = if protocol.solvent == SolventModel::Explicit {
            Driver::Explicit(ExplicitSimulation::new(system, protocol)?)
        } else {
            Driver::Implicit(CpuSimulation::new(system, protocol)?)
        };
        Ok(Self {
            driver,
            actual_backend: "cpu".into(),
            requested_backend,
            fallback_reason,
        })
    }

    pub fn requested_backend(&self) -> &str {
        &self.requested_backend
    }

    pub fn actual_backend(&self) -> &str {
        &self.actual_backend
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    pub fn state(&self) -> &SimulationState {
        match &self.driver {
            Driver::Explicit(simulation) => &simulation.state,
            Driver::Implicit(simulation) => &simulation.state,
        }
    }

    pub fn state_mut(&mut self) -> &mut SimulationState {
        match &mut self.driver {
            Driver::Explicit(simulation) => &mut simulation.state,
            Driver::Implicit(simulation) => &mut simulation.state,
        }
    }

    pub fn advance(&mut self, steps: usize) -> Result<TrajectoryChunk> {
        match &mut self.driver {
            Driver::Explicit(simulation) => simulation.advance(steps),
            Driver::Implicit(simulation) => simulation.advance(steps),
        }
    }
}

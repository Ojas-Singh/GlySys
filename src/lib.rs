//! Native Rust preparation of solvated Amber/GLYCAM molecular systems.
//!
//! The crate never invokes AmberTools, acpype, GROMACS, OpenMM, Python, or any
//! other external program.  Amber and GROMACS files are written directly.

mod amber_data;
mod error;
mod forcefield;
mod model;
mod options;
mod pdb;
mod prepare;
mod report;
mod solvate;
mod writers;

pub use error::{BuildError, BuildWarning, Result};
pub use model::PreparedSystem;
pub use options::{BuildOptions, ForceFieldProfile, ProtonationOverrides};
pub use prepare::SystemBuilder;
pub use report::{BuildReport, GlycanReport, ResidueRef};

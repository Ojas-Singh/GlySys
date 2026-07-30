//! Native Rust preparation of solvated Amber/GLYCAM molecular systems.
//!
//! The crate never invokes AmberTools, acpype, GROMACS, OpenMM, Python, or any
//! other external program.  Amber and GROMACS files are written directly.

mod amber_data;
mod error;
mod forcefield;
pub mod io;
mod model;
mod options;
mod pdb;
mod prepare;
mod report;
mod solvate;
mod structure;
mod writers;

pub use error::{BuildError, BuildWarning, Result};
pub use model::{Angle, Atom, Bond, Dihedral, ParameterizedSystem, PreparedSystem, Residue, Vec3};
pub use options::{BuildOptions, ForceFieldProfile, ProtonationOverrides};
pub use prepare::SystemBuilder;
pub use report::{BuildReport, GlycanReport, ResidueRef};
pub use structure::{
    AppendMap, AtomId, GlycanTree, GlycosylationSite, ResidueAnnotation, ResidueId, Structure,
    StructureAtom, StructureResidue, SystemMetadata, read_pdb, read_pdb_str, write_pdb,
    write_pdb_string,
};

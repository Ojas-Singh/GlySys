//! Molecular structure and parameterized-system input/output.

pub mod pdb {
    pub use crate::structure::{
        read_pdb as read, read_pdb_str as read_str, write_pdb as write,
        write_pdb_string as write_string,
    };
}

pub mod amber {
    use std::path::Path;

    use crate::{ParameterizedSystem, Result};

    pub fn to_prmtop(system: &ParameterizedSystem) -> Result<String> {
        crate::writers::amber::write_prmtop(&system.system)
    }

    pub fn to_inpcrd(system: &ParameterizedSystem) -> String {
        crate::writers::amber::write_inpcrd(&system.system)
    }

    pub fn write(
        system: &ParameterizedSystem,
        prmtop: impl AsRef<Path>,
        inpcrd: impl AsRef<Path>,
    ) -> Result<()> {
        let prmtop = prmtop.as_ref();
        let inpcrd = inpcrd.as_ref();
        std::fs::write(prmtop, to_prmtop(system)?)
            .map_err(crate::error::write_error(prmtop.to_path_buf()))?;
        std::fs::write(inpcrd, to_inpcrd(system))
            .map_err(crate::error::write_error(inpcrd.to_path_buf()))
    }
}

pub mod gromacs {
    use std::path::Path;

    use crate::{ParameterizedSystem, Result};

    pub fn to_topology(system: &ParameterizedSystem) -> String {
        crate::writers::gromacs::write_topology(&system.system)
    }

    pub fn to_gro(system: &ParameterizedSystem) -> String {
        crate::writers::gromacs::write_gro(&system.system)
    }

    pub fn write(
        system: &ParameterizedSystem,
        topology: impl AsRef<Path>,
        gro: impl AsRef<Path>,
    ) -> Result<()> {
        let topology = topology.as_ref();
        let gro = gro.as_ref();
        std::fs::write(topology, to_topology(system))
            .map_err(crate::error::write_error(topology.to_path_buf()))?;
        std::fs::write(gro, to_gro(system)).map_err(crate::error::write_error(gro.to_path_buf()))
    }
}

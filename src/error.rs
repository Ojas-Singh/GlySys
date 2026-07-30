use std::path::PathBuf;

/// Errors that prevent construction of a chemically complete system.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid PDB: {0}")]
    InvalidPdb(String),
    #[error("requested MODEL {0} does not exist")]
    ModelNotFound(u32),
    #[error("unsupported residue {residue}: {reason}")]
    UnsupportedResidue { residue: String, reason: String },
    #[error("residue {residue} is missing heavy atoms: {atoms}")]
    MissingHeavyAtoms { residue: String, atoms: String },
    #[error("ambiguous connectivity: {0}")]
    AmbiguousConnectivity(String),
    #[error("force-field data error: {0}")]
    ForceField(String),
    #[error("no parameter for {kind} atom types {types}")]
    MissingParameter { kind: &'static str, types: String },
    #[error("solute charge {0:.6} is not within tolerance of an integer")]
    NonIntegralCharge(f64),
    #[error("not enough solvent molecules to place {requested} ions (only {available} eligible)")]
    InsufficientSolvent { requested: usize, available: usize },
    #[error("output directory already contains generated files: {0}")]
    OutputExists(PathBuf),
    #[error("invalid option: {0}")]
    InvalidOption(String),
    #[error("glycan analysis failed: {0}")]
    Glycan(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
}

/// Non-fatal decisions recorded in the build manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(tag = "code", content = "message", rename_all = "snake_case")]
pub enum BuildWarning {
    AlternateLocationSelected(String),
    HistidineStateInferred(String),
    ExistingSolventRemoved(String),
    GlycanNameNormalized(String),
    InputHydrogensRebuilt(String),
    InputGlycanHydrogensPreserved(String),
}

pub type Result<T> = std::result::Result<T, BuildError>;

pub(crate) fn read_error(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> BuildError {
    let path = path.into();
    move |source| BuildError::Read { path, source }
}

pub(crate) fn write_error(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> BuildError {
    let path = path.into();
    move |source| BuildError::Write { path, source }
}

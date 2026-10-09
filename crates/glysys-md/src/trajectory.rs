//! Trajectory output of the native runner.
//!
//! `jsonl` keeps one self-describing JSON frame per line with every
//! coordinate as text: convenient for short validation runs, far too large
//! for production (about 60 bytes per atom per frame).
//!
//! `dcd` writes a CHARMM/NAMD DCD file (single-precision coordinates and the
//! box of every frame), which MDTraj, cpptraj, MDAnalysis and VMD read, next
//! to `trajectory.pdb` (the atoms of the file, for a topology) and
//! `frames.jsonl` (step, time, stage, energies, temperature, pressure,
//! density and box of every frame). With the solute alone a long glycan run
//! is a few megabytes per hundred nanoseconds, and the stripping and
//! conversion steps of a GROMACS workflow are not needed.
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use glysys::{ParameterizedSystem, Vec3};
use glysys_dynamics::TrajectoryFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryFormat {
    Jsonl,
    Dcd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryAtoms {
    /// Every atom, molecules made whole in the box.
    All,
    /// The solute only: the atoms before the added water and ions.
    Solute,
}

/// AKMA time unit of CHARMM in picoseconds.
const AKMA_PS: f64 = 0.048_888_21;
const HEADER_FRAMES: u64 = 8;
const HEADER_STEPS: u64 = 20;

pub struct DcdWriter {
    file: File,
    atoms: usize,
    frames: u32,
    interval: u32,
    scratch: Vec<u8>,
}

impl DcdWriter {
    pub fn create(path: &Path, atoms: usize, interval_steps: usize, timestep_ps: f64) -> Result<Self> {
        let atoms_i32 = i32::try_from(atoms).context("too many atoms for a DCD file")?;
        let interval = u32::try_from(interval_steps).context("frame interval too large for a DCD file")?;
        let mut header = Vec::with_capacity(276);
        header.extend_from_slice(&84i32.to_le_bytes());
        header.extend_from_slice(b"CORD");
        let mut control = [0i32; 20];
        control[1] = interval as i32; // first saved step
        control[2] = interval as i32;
        control[9] = i32::from_le_bytes(((timestep_ps / AKMA_PS) as f32).to_le_bytes());
        control[10] = 1; // a unit cell precedes every frame
        control[19] = 24; // CHARMM version, which selects this layout
        for value in control {
            header.extend_from_slice(&value.to_le_bytes());
        }
        header.extend_from_slice(&84i32.to_le_bytes());
        let titles = [
            format!("{:<80}", "GlySys trajectory"),
            format!("{:<80}", "coordinates in angstrom; molecules whole"),
        ];
        header.extend_from_slice(&164i32.to_le_bytes());
        header.extend_from_slice(&2i32.to_le_bytes());
        for title in &titles {
            header.extend_from_slice(&title.as_bytes()[..80]);
        }
        header.extend_from_slice(&164i32.to_le_bytes());
        header.extend_from_slice(&4i32.to_le_bytes());
        header.extend_from_slice(&atoms_i32.to_le_bytes());
        header.extend_from_slice(&4i32.to_le_bytes());
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("creating {}", path.display()))?;
        file.write_all(&header)?;
        Ok(Self {
            file,
            atoms,
            frames: 0,
            interval,
            scratch: Vec::new(),
        })
    }

    /// Reopen a file this writer made, to add frames after a restart.
    pub fn append(path: &Path, atoms: usize) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut header = [0u8; 276];
        file.read_exact(&mut header)
            .with_context(|| format!("{} is not a DCD file", path.display()))?;
        let int = |offset: usize| i32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        if int(0) != 84 || &header[4..8] != b"CORD" || int(48) != 1 || int(268) as usize != atoms {
            bail!(
                "{} does not hold a GlySys DCD trajectory of {atoms} atoms",
                path.display()
            );
        }
        let frames = int(8) as u32;
        let interval = int(16) as u32;
        // Drop a frame that a stopped run left half written.
        let frame_bytes = (56 + 3 * (8 + 4 * atoms)) as u64;
        file.set_len(276 + frame_bytes * u64::from(frames))?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            atoms,
            frames,
            interval,
            scratch: Vec::new(),
        })
    }

    pub fn frames(&self) -> u32 {
        self.frames
    }

    pub fn write_frame(&mut self, box_angstrom: [f64; 3], coordinates: &[Vec3]) -> Result<()> {
        if coordinates.len() < self.atoms {
            bail!(
                "frame has {} atoms; the trajectory holds {}",
                coordinates.len(),
                self.atoms
            );
        }
        let coordinates = &coordinates[..self.atoms];
        let record = (4 * self.atoms) as i32;
        self.scratch.clear();
        self.scratch.extend_from_slice(&48i32.to_le_bytes());
        // a, gamma, b, beta, alpha, c with the angles in degrees
        for value in [box_angstrom[0], 90., box_angstrom[1], 90., 90., box_angstrom[2]] {
            self.scratch.extend_from_slice(&value.to_le_bytes());
        }
        self.scratch.extend_from_slice(&48i32.to_le_bytes());
        for axis in 0..3 {
            self.scratch.extend_from_slice(&record.to_le_bytes());
            for position in coordinates {
                let value = [position.x, position.y, position.z][axis] as f32;
                self.scratch.extend_from_slice(&value.to_le_bytes());
            }
            self.scratch.extend_from_slice(&record.to_le_bytes());
        }
        self.file.write_all(&self.scratch)?;
        self.frames += 1;
        // Keep the header's frame count current, so the file can be read
        // while the run is going and after a run that was stopped.
        self.file.seek(SeekFrom::Start(HEADER_FRAMES))?;
        self.file.write_all(&(self.frames as i32).to_le_bytes())?;
        self.file.seek(SeekFrom::Start(HEADER_STEPS))?;
        self.file
            .write_all(&((self.frames.saturating_mul(self.interval)) as i32).to_le_bytes())?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.flush()?;
        Ok(())
    }
}

enum Sink {
    Jsonl(BufWriter<File>),
    Dcd {
        coordinates: DcdWriter,
        scalars: BufWriter<File>,
    },
}

/// Writes the frames of one run directory.
pub struct TrajectoryWriter {
    sink: Sink,
    atoms: usize,
}

/// Files a run directory may already hold from an earlier run.
pub const TRAJECTORY_FILES: [&str; 4] = [
    "trajectory.jsonl",
    "trajectory.dcd",
    "trajectory.pdb",
    "frames.jsonl",
];

impl TrajectoryWriter {
    /// Open the trajectory of `output`, continuing the files of an earlier
    /// run when they exist.
    pub fn open(
        output: &Path,
        system: &ParameterizedSystem,
        format: TrajectoryFormat,
        selection: TrajectoryAtoms,
        save_every: usize,
        timestep_ps: f64,
    ) -> Result<Self> {
        let atoms = match selection {
            TrajectoryAtoms::All => system.atom_count(),
            TrajectoryAtoms::Solute => system.solute_atom_count(),
        };
        if atoms == 0 {
            bail!("the trajectory selection holds no atoms");
        }
        let append = |name: &str| -> Result<BufWriter<File>> {
            Ok(BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(output.join(name))
                    .with_context(|| format!("opening {name}"))?,
            ))
        };
        let sink = match format {
            TrajectoryFormat::Jsonl => Sink::Jsonl(append("trajectory.jsonl")?),
            TrajectoryFormat::Dcd => {
                let path = output.join("trajectory.dcd");
                let coordinates = if path.exists() {
                    DcdWriter::append(&path, atoms)?
                } else {
                    let topology = match selection {
                        TrajectoryAtoms::All => system.pdb_string(),
                        TrajectoryAtoms::Solute => system.solute().pdb_string(),
                    };
                    fs::write(output.join("trajectory.pdb"), topology)?;
                    DcdWriter::create(&path, atoms, save_every, timestep_ps)?
                };
                Sink::Dcd {
                    coordinates,
                    scalars: append("frames.jsonl")?,
                }
            }
        };
        Ok(Self { sink, atoms })
    }

    pub fn write(&mut self, frame: &TrajectoryFrame) -> Result<()> {
        match &mut self.sink {
            Sink::Jsonl(writer) => {
                if self.atoms == frame.coordinates.len() {
                    serde_json::to_writer(&mut *writer, frame)?;
                } else {
                    let mut selected = frame.clone();
                    selected.coordinates.truncate(self.atoms);
                    serde_json::to_writer(&mut *writer, &selected)?;
                }
                writer.write_all(b"\n")?;
            }
            Sink::Dcd {
                coordinates,
                scalars,
            } => {
                coordinates.write_frame(frame.box_angstrom, &frame.coordinates)?;
                let record = serde_json::json!({
                    "frame": coordinates.frames() - 1,
                    "step": frame.step,
                    "timePs": frame.time_ps,
                    "segment": frame.segment,
                    "potentialEnergy": frame.potential_energy,
                    "kineticEnergy": frame.kinetic_energy,
                    "temperatureK": frame.temperature_k,
                    "boxAngstrom": frame.box_angstrom,
                    "pressureBar": frame.pressure_bar,
                    "pressureEstimator": frame.pressure_estimator,
                    "densityGMl": frame.density_g_ml,
                });
                serde_json::to_writer(&mut *scalars, &record)?;
                scalars.write_all(b"\n")?;
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        match &mut self.sink {
            Sink::Jsonl(writer) => writer.flush()?,
            Sink::Dcd { scalars, .. } => scalars.flush()?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(atoms: usize, shift: f64) -> Vec<Vec3> {
        (0..atoms)
            .map(|i| Vec3 {
                x: i as f64 + shift,
                y: 2. * i as f64 - shift,
                z: 0.5 * shift,
            })
            .collect()
    }

    /// Reads a DCD file back the way its format is documented, independently
    /// of the writer's own bookkeeping.
    fn read(path: &Path) -> (usize, Vec<([f64; 3], Vec<[f32; 3]>)>) {
        let bytes = fs::read(path).unwrap();
        let int = |o: usize| i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!(&bytes[4..8], b"CORD");
        let (frames, atoms) = (int(8) as usize, int(268) as usize);
        let mut offset = 276;
        let mut out = Vec::new();
        for _ in 0..frames {
            assert_eq!(int(offset), 48);
            let cell: Vec<f64> = (0..6)
                .map(|k| f64::from_le_bytes(bytes[offset + 4 + 8 * k..offset + 12 + 8 * k].try_into().unwrap()))
                .collect();
            offset += 56;
            let mut xyz = vec![[0f32; 3]; atoms];
            for axis in 0..3 {
                assert_eq!(int(offset) as usize, 4 * atoms);
                for (atom, slot) in xyz.iter_mut().enumerate() {
                    let at = offset + 4 + 4 * atom;
                    slot[axis] = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                }
                offset += 8 + 4 * atoms;
            }
            out.push(([cell[0], cell[2], cell[5]], xyz));
        }
        assert_eq!(offset, bytes.len());
        (atoms, out)
    }

    #[test]
    fn dcd_frames_round_trip_and_a_restart_appends() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("t.dcd");
        let mut writer = DcdWriter::create(&path, 5, 5000, 0.002).unwrap();
        writer.write_frame([40., 50., 45.], &frame(7, 0.25)).unwrap();
        writer.write_frame([40.1, 50.1, 45.1], &frame(7, 1.5)).unwrap();
        drop(writer);
        // a half-written third frame, as a killed run leaves it
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0u8; 37]).unwrap();
        drop(file);

        let mut writer = DcdWriter::append(&path, 5).unwrap();
        assert_eq!(writer.frames(), 2);
        writer.write_frame([40.2, 50.2, 45.2], &frame(5, -3.)).unwrap();
        drop(writer);

        let (atoms, frames) = read(&path);
        assert_eq!(atoms, 5);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1].0, [40.1, 50.1, 45.1]);
        assert_eq!(frames[0].1[3], [3.25, 5.75, 0.125]);
        assert_eq!(frames[2].1[4], [1., 11., -1.5]);
        assert!(DcdWriter::append(&path, 6).is_err());
    }
}

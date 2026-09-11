//! On-the-fly statistics accumulated during simulation without saving frames.
//!
//! This is the extension point for hydration-site occupancy, water
//! orientation and H-bond statistics, and residence tracking: implementors see
//! every saved production frame (wrapped oxygen positions plus the box) and
//! keep compact state that survives checkpoints. The first real consumer,
//! [`WaterOccupancy`], bins oxygen density on a 1 A grid for hydration maps.
use glysys::Vec3;
use serde::{Deserialize, Serialize};

/// Observer over saved production frames. Implementations must be
/// deterministic given the same frame sequence.
pub trait Accumulator: Send + Sync {
    fn name(&self) -> &'static str;
    fn observe(&mut self, oxygen: &[Vec3], box_angstrom: [f64; 3]);
    fn frames_observed(&self) -> u64;
}

/// 1 A oxygen-density grid over the periodic box, stored row-major with x
/// varying fastest. Counts are saturating; divide by frames for occupancy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterOccupancy {
    pub spacing_angstrom: f64,
    pub dimensions: [usize; 3],
    pub counts: Vec<u32>,
    pub frames: u64,
}

impl WaterOccupancy {
    pub fn new(box_angstrom: [f64; 3], spacing_angstrom: f64) -> Self {
        let spacing = spacing_angstrom.max(0.25);
        let dimensions = [
            ((box_angstrom[0] / spacing).ceil() as usize).max(1),
            ((box_angstrom[1] / spacing).ceil() as usize).max(1),
            ((box_angstrom[2] / spacing).ceil() as usize).max(1),
        ];
        let total = dimensions[0] * dimensions[1] * dimensions[2];
        Self {
            spacing_angstrom: spacing,
            dimensions,
            counts: vec![0; total],
            frames: 0,
        }
    }

    /// Fractional occupancy per voxel (counts divided by observed frames).
    pub fn occupancy(&self) -> Vec<f64> {
        if self.frames == 0 {
            return vec![0.; self.counts.len()];
        }
        self.counts
            .iter()
            .map(|c| *c as f64 / self.frames as f64)
            .collect()
    }
}

impl Accumulator for WaterOccupancy {
    fn name(&self) -> &'static str {
        "water-occupancy-v1"
    }

    fn observe(&mut self, oxygen: &[Vec3], box_angstrom: [f64; 3]) {
        let [nx, ny, nz] = self.dimensions;
        for o in oxygen {
            let mut ix = (o.x / box_angstrom[0].max(1e-9) * nx as f64).floor() as i64;
            let mut iy = (o.y / box_angstrom[1].max(1e-9) * ny as f64).floor() as i64;
            let mut iz = (o.z / box_angstrom[2].max(1e-9) * nz as f64).floor() as i64;
            ix = ix.rem_euclid(nx as i64);
            iy = iy.rem_euclid(ny as i64);
            iz = iz.rem_euclid(nz as i64);
            let index = (ix as usize * ny + iy as usize) * nz + iz as usize;
            if let Some(slot) = self.counts.get_mut(index) {
                *slot = slot.saturating_add(1);
            }
        }
        self.frames += 1;
    }

    fn frames_observed(&self) -> u64 {
        self.frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_bins_wrapped_positions() {
        let mut acc = WaterOccupancy::new([10., 10., 10.], 1.);
        assert_eq!(acc.dimensions, [10, 10, 10]);
        acc.observe(
            &[Vec3 { x: 1.2, y: 1.2, z: 1.2 }, Vec3 { x: 11.2, y: -0.5, z: 25. } ],
            [10., 10., 10.],
        );
        // Second oxygen wraps to voxel (1, 9, 5).
        let occ = acc.occupancy();
        assert_eq!(occ[(1 * 10 + 1) * 10 + 1], 1.);
        assert_eq!(occ[(1 * 10 + 9) * 10 + 5], 1.);
        assert_eq!(acc.frames_observed(), 1);
    }
}

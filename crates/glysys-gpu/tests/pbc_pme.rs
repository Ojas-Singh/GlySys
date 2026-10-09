//! Native validation of the reciprocal-space PME kernels (`pbc_pme.wgsl`)
//! against the f64 CPU engine `glysys_energy::pme::PmeEngine` on the same
//! mesh, with the same alpha and interpolation order 4.
//!
//! The device sees f32 charges, positions and box lengths; the CPU engine is
//! given exactly those numbers widened to f64, so the difference measured is
//! the arithmetic of the kernels (f32 transforms, the fixed-point mesh) and
//! nothing else. Tests return early, with a note, when no adapter exists.
use glysys::Vec3;
use glysys_energy::pbc::BoxVectors;
use glysys_energy::pme::{PmeEngine, PmeParameters};
use glysys_gpu::{
    GpuContext, GpuContextOptions, PME_BOX_OFFSET_BYTES, PmeKernel, PmeMesh, PmeShared, PmeSizing,
};
use std::sync::{Mutex, OnceLock};

const ALPHA: f64 = 0.3470;
/// Energy accumulators of the tile engine after the per-atom gradient words.
const ENERGY_SLOTS: usize = 9;

// One device at a time: the adapter is shared with other users of the machine
// and concurrent device teardown is fragile on software Vulkan.
static GPU_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn context(gpu_timestamps: bool) -> Option<GpuContext> {
    let options = GpuContextOptions {
        gpu_timestamps,
        label: "GlySys PME validation".into(),
        ..GpuContextOptions::default()
    };
    match pollster::block_on(GpuContext::new(options)) {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping: no GPU adapter ({error})");
            None
        }
    }
}

/// Deterministic uniform numbers in [0, 1).
struct Uniform(u64);

impl Uniform {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Point charges with a net charge of exactly zero, at positions given as box
/// fractions plus whole box images, so the same system exists for any box.
struct Charges {
    q: Vec<f32>,
    fraction: Vec<[f64; 3]>,
    image: Vec<[f64; 3]>,
}

impl Charges {
    /// Charges are multiples of 1/1024 up to 1 e in magnitude, so their sum is
    /// exact in f32 and f64. Every 19th atom is uncharged, and about a quarter
    /// of the atoms sit one or two box lengths outside the box.
    fn random(n: usize, seed: u64) -> Self {
        let mut random = Uniform(seed);
        let mut units: Vec<i32> = (0..n)
            .map(|atom| {
                let unit = (random.next() * 2049.0).floor() as i32 - 1024;
                if atom % 19 == 7 { 0 } else { unit }
            })
            .collect();
        // Take the excess off the charged atoms, a little from each.
        let mut excess: i32 = units.iter().sum();
        while excess != 0 {
            for unit in units.iter_mut().filter(|unit| **unit != 0) {
                let moved = (*unit - excess.signum() * excess.abs().min(32)).clamp(-1024, 1024);
                if moved != 0 {
                    excess -= *unit - moved;
                    *unit = moved;
                }
            }
        }
        assert_eq!(units.iter().sum::<i32>(), 0, "net charge");
        let fraction = (0..n)
            .map(|_| [random.next(), random.next(), random.next()])
            .collect();
        let image = (0..n)
            .map(|atom| {
                let shift = |period: usize, by: f64| if atom % period == 1 { by } else { 0.0 };
                [
                    shift(7, 1.0),
                    shift(11, -2.0),
                    shift(13, 2.0) + shift(5, -1.0),
                ]
            })
            .collect();
        Self {
            q: units.iter().map(|unit| *unit as f32 / 1024.0).collect(),
            fraction,
            image,
        }
    }

    fn len(&self) -> usize {
        self.q.len()
    }

    /// Positions in the frame of `sys`: shifted by minus half the box.
    fn positions(&self, box_xyz: [f32; 3]) -> Vec<[f32; 3]> {
        self.fraction
            .iter()
            .zip(&self.image)
            .map(|(fraction, image)| {
                std::array::from_fn(|axis| {
                    ((fraction[axis] + image[axis] - 0.5) * f64::from(box_xyz[axis])) as f32
                })
            })
            .collect()
    }
}

struct Reference {
    gradients: Vec<[f64; 3]>,
    energy: f64,
    virial: f64,
}

/// The CPU engine on the numbers the device sees.
fn reference(
    charges: &[f32],
    positions: &[[f32; 3]],
    box_xyz: [f32; 3],
    grid: [u32; 3],
) -> Reference {
    let parameters = PmeParameters::new(ALPHA, grid.map(|points| points as usize), 4).unwrap();
    let q: Vec<f64> = charges.iter().map(|q| f64::from(*q)).collect();
    let mut engine = PmeEngine::from_charges(&q, &[], parameters).unwrap();
    let coords: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3 {
            x: f64::from(p[0]),
            y: f64::from(p[1]),
            z: f64::from(p[2]),
        })
        .collect();
    let box_vec = BoxVectors::new(
        f64::from(box_xyz[0]),
        f64::from(box_xyz[1]),
        f64::from(box_xyz[2]),
    )
    .unwrap();
    let mut gradients = vec![
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        charges.len()
    ];
    let long_range = engine
        .evaluate_into(&coords, &box_vec, &mut gradients, true)
        .unwrap();
    let terms = engine.last_terms();
    // No excluded pairs and no net charge: the virial is the reciprocal one.
    assert_eq!(terms.excluded_pairs, 0.0);
    assert_eq!(terms.background, 0.0);
    Reference {
        gradients: gradients.iter().map(|g| [g.x, g.y, g.z]).collect(),
        energy: terms.reciprocal,
        virial: long_range.virial,
    }
}

struct Output {
    /// `pair_grad`, one `vec4` per atom.
    gradients: Vec<[f32; 4]>,
    /// The accumulator words.
    acc: Vec<u32>,
}

impl Output {
    /// 64-bit fixed point with scale 2^32 in two words.
    fn fixed(&self, word: usize) -> i64 {
        ((u64::from(self.acc[word + 1]) << 32) | u64::from(self.acc[word])) as i64
    }

    fn slot(&self, n: usize, slot: usize) -> f64 {
        self.fixed(6 * n + 2 * slot) as f64 / 4_294_967_296.0
    }

    fn energy(&self, n: usize) -> f64 {
        self.slot(n, 1)
    }

    fn virial(&self, n: usize) -> f64 {
        self.slot(n, 3)
    }
}

/// The buffers `ResidentPbc` shares with the mesh, and the mesh.
struct Harness {
    context: GpuContext,
    n: usize,
    sys: wgpu::Buffer,
    pair_grad: wgpu::Buffer,
    acc: wgpu::Buffer,
    mesh: PmeMesh,
}

impl Harness {
    fn new(context: &GpuContext, n: usize, grid: [u32; 3]) -> Self {
        let device = context.device();
        let storage = wgpu::BufferUsages::STORAGE;
        let copy = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: storage | copy,
                mapped_at_creation: false,
            })
        };
        let sys = buffer("test sys", 32 * n as u64);
        let pair_grad = buffer("test pair gradients", 16 * n as u64);
        let acc = buffer("test accumulators", 4 * (6 * n + 2 * ENERGY_SLOTS) as u64);
        let sizing = PmeSizing::with_grid(n as u32, grid).unwrap();
        assert!(
            sizing.largest_binding_bytes()
                <= context.limits().max_storage_buffer_binding_size as u64
        );
        let mesh = PmeMesh::new(
            device,
            context.queue(),
            sizing,
            PmeShared {
                sys: &sys,
                pair_grad: &pair_grad,
                acc: &acc,
            },
            ALPHA as f32,
        );
        Self {
            context: context.clone(),
            n,
            sys,
            pair_grad,
            acc,
            mesh,
        }
    }

    fn set_system(&self, charges: &[f32], positions: &[[f32; 3]]) {
        let mut data = Vec::with_capacity(2 * self.n);
        for (q, p) in charges.iter().zip(positions) {
            data.push([*q, 3.2, 0.1, 12.0]);
            data.push([p[0], p[1], p[2], 0.0]);
        }
        self.context
            .queue()
            .write_buffer(&self.sys, 0, bytemuck::cast_slice(&data));
    }

    fn encode(&self, pass: &mut wgpu::ComputePass<'_>, energy: bool) {
        for &kernel in PmeMesh::chain(energy) {
            self.dispatch(pass, kernel);
        }
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, kernel: PmeKernel) {
        let (pipeline, bind_group) = self.mesh.kernel(kernel);
        let (x, y) = self.mesh.groups(kernel);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn clear_acc(&self) {
        let zeros = vec![0u32; 6 * self.n + 2 * ENERGY_SLOTS];
        self.context
            .queue()
            .write_buffer(&self.acc, 0, bytemuck::cast_slice(&zeros));
    }

    /// One evaluation on a pair gradient preset to `preset` for every atom.
    /// The accumulators are not cleared here.
    fn run(&self, energy: bool, preset: [f32; 3]) -> Output {
        let device = self.context.device();
        let queue = self.context.queue();
        let initial = vec![[preset[0], preset[1], preset[2], 0.0f32]; self.n];
        queue.write_buffer(&self.pair_grad, 0, bytemuck::cast_slice(&initial));
        let (grad_bytes, acc_bytes) = (self.pair_grad.size(), self.acc.size());
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: grad_bytes + acc_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            self.encode(&mut pass, energy);
        }
        encoder.copy_buffer_to_buffer(&self.pair_grad, 0, &staging, 0, grad_bytes);
        encoder.copy_buffer_to_buffer(&self.acc, 0, &staging, grad_bytes, acc_bytes);
        queue.submit(Some(encoder.finish()));
        let bytes = read(device, &staging);
        Output {
            gradients: bytemuck::pod_collect_to_vec(&bytes[..grad_bytes as usize]),
            acc: bytemuck::pod_collect_to_vec(&bytes[grad_bytes as usize..]),
        }
    }
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait).unwrap();
    receiver.recv().unwrap().unwrap();
    let bytes = slice.get_mapped_range().to_vec();
    buffer.unmap();
    bytes
}

struct Errors {
    gradient: f64,
    gradient_max: f64,
    energy: f64,
    virial: f64,
}

fn compare(tag: &str, n: usize, output: &Output, reference: &Reference) -> Errors {
    let (mut error2, mut norm2, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (got, want) in output.gradients.iter().zip(&reference.gradients) {
        assert_eq!(
            got[3], 0.0,
            "{tag}: the fourth gradient component stays zero"
        );
        for axis in 0..3 {
            let delta = f64::from(got[axis]) - want[axis];
            error2 += delta * delta;
            norm2 += want[axis] * want[axis];
            worst = worst.max(delta.abs());
        }
    }
    let errors = Errors {
        gradient: (error2 / norm2).sqrt(),
        gradient_max: worst,
        energy: ((output.energy(n) - reference.energy) / reference.energy).abs(),
        virial: ((output.virial(n) - reference.virial) / reference.virial).abs(),
    };
    eprintln!(
        "{tag}: gradient rms error / rms {:.2e} (rms {:.3}, worst {:.2e} kcal/mol/A), \
         energy {:.4} vs {:.4} kcal/mol (rel {:.2e}), virial {:.4} vs {:.4} (rel {:.2e})",
        errors.gradient,
        (norm2 / (3 * n) as f64).sqrt(),
        errors.gradient_max,
        output.energy(n),
        reference.energy,
        errors.energy,
        output.virial(n),
        reference.virial,
        errors.virial,
    );
    errors
}

fn assert_within(tag: &str, errors: &Errors) {
    assert!(
        errors.gradient < 2e-4,
        "{tag}: gradient {:.3e}",
        errors.gradient
    );
    assert!(errors.energy < 2e-5, "{tag}: energy {:.3e}", errors.energy);
    assert!(errors.virial < 1e-4, "{tag}: virial {:.3e}", errors.virial);
}

fn bits(gradients: &[[f32; 4]]) -> Vec<[u32; 4]> {
    gradients.iter().map(|g| g.map(f32::to_bits)).collect()
}

/// Everything the kernels promise, for one mesh.
fn validate(context: &GpuContext, tag: &str, box_xyz: [f32; 3], grid: [u32; 3], seed: u64) {
    let charges = Charges::random(3_000, seed);
    let n = charges.len();
    let harness = Harness::new(context, n, grid);
    let positions = charges.positions(box_xyz);
    harness.set_system(&charges.q, &positions);
    harness.mesh.upload(context.queue(), box_xyz);
    harness.clear_acc();

    // Energy chain against the CPU engine.
    let first = harness.run(true, [0.0; 3]);
    let expected = reference(&charges.q, &positions, box_xyz, grid);
    assert_within(tag, &compare(tag, n, &first, &expected));
    // Only the electrostatic energy and the virial were touched.
    for (word, value) in first.acc.iter().enumerate() {
        let energy_words = [6 * n + 2, 6 * n + 3, 6 * n + 6, 6 * n + 7];
        assert!(
            *value == 0 || energy_words.contains(&word),
            "{tag}: accumulator word {word} written"
        );
    }

    // The same input again: identical bits, which also shows the charge mesh
    // was left empty. The accumulators are not cleared by the kernels, so
    // the fixed-point totals double exactly.
    let second = harness.run(true, [0.0; 3]);
    assert_eq!(
        bits(&first.gradients),
        bits(&second.gradients),
        "{tag}: determinism"
    );
    for slot in [1, 3] {
        let word = 6 * n + 2 * slot;
        assert_eq!(
            second.fixed(word),
            2 * first.fixed(word),
            "{tag}: slot {slot} accumulates"
        );
    }

    // Force-only chain: the same gradients, and no accumulator is touched.
    harness.clear_acc();
    let forces = harness.run(false, [0.0; 3]);
    assert_eq!(
        bits(&first.gradients),
        bits(&forces.gradients),
        "{tag}: force-only chain"
    );
    assert!(
        forces.acc.iter().all(|word| *word == 0),
        "{tag}: force-only chain wrote acc"
    );

    // The gradient is added to what the pair kernel stored. The device may
    // fuse the last multiplication of the gradient with this addition, so the
    // sum is the rounded one only to a few units in the last place.
    let preset = [3.0f32, -5.0, 7.0];
    let added = harness.run(false, preset);
    for (atom, (sum, alone)) in added.gradients.iter().zip(&forces.gradients).enumerate() {
        for axis in 0..3 {
            let expected = preset[axis] + alone[axis];
            let tolerance = 4.0 * f32::EPSILON * expected.abs().max(alone[axis].abs()).max(1.0);
            assert!(
                (sum[axis] - expected).abs() <= tolerance,
                "{tag}: atom {atom} axis {axis}: {} is not {} + {}",
                sum[axis],
                preset[axis],
                alone[axis]
            );
        }
        assert_eq!(sum[3], 0.0);
        if charges.q[atom] == 0.0 {
            assert_eq!(alone[..3], [0.0; 3], "{tag}: uncharged atom {atom}");
        }
    }
}

#[test]
fn reciprocal_pme_matches_the_cpu_engine() {
    let _guard = gpu_test_guard();
    let Some(context) = context(false) else {
        return;
    };
    eprintln!("adapter: {}", context.adapter_info().name);
    let box_xyz = [40.0f32, 44.0, 52.0];
    let lengths = box_xyz.map(f64::from);
    // A loose spacing gives two different sizes; the default one a cube.
    let loose = PmeSizing::new(3_000, lengths, 1.5).unwrap().grid;
    assert_eq!(loose, [32, 32, 64]);
    validate(
        &context,
        "40x44x52 A, mesh 32x32x64",
        box_xyz,
        loose,
        0x9E37_79B9_7F4A_7C15,
    );
    let fine = PmeSizing::new(3_000, lengths, 1.2).unwrap().grid;
    assert_eq!(fine, [64, 64, 64]);
    validate(
        &context,
        "40x44x52 A, mesh 64x64x64",
        box_xyz,
        fine,
        0x2545_F491_4F6C_DD1D,
    );
}

/// Lines longer than a workgroup (128 and 256 points, two and four elements
/// per lane) and the shortest ones, on every axis.
#[test]
fn reciprocal_pme_handles_every_line_length_on_every_axis() {
    let _guard = gpu_test_guard();
    let Some(context) = context(false) else {
        return;
    };
    for (box_xyz, grid) in [
        ([14.0f32, 120.0, 250.0], [16u32, 128, 256]),
        ([250.0, 14.0, 120.0], [256, 16, 128]),
        ([120.0, 250.0, 14.0], [128, 256, 16]),
        ([60.0, 28.0, 14.0], [64, 32, 16]),
    ] {
        assert_eq!(
            PmeSizing::new(3_000, box_xyz.map(f64::from), 1.0)
                .unwrap()
                .grid,
            grid
        );
        let tag = format!(
            "{}x{}x{} A, mesh {}x{}x{}",
            box_xyz[0], box_xyz[1], box_xyz[2], grid[0], grid[1], grid[2]
        );
        validate(
            &context,
            &tag,
            box_xyz,
            grid,
            0xD1B5_4A32_D192_ED03 ^ u64::from(grid[0]),
        );
    }
}

/// Nothing that depends on the box is cached: a new box through `upload`, and
/// one written straight into the uniform the way a device barostat does it.
#[test]
fn reciprocal_pme_follows_a_box_change() {
    let _guard = gpu_test_guard();
    let Some(context) = context(false) else {
        return;
    };
    let grid = [64u32, 64, 64];
    let charges = Charges::random(3_000, 0xA076_1D64_78BD_642F);
    let n = charges.len();
    let harness = Harness::new(&context, n, grid);
    let evaluate = |tag: &str, box_xyz: [f32; 3]| {
        let positions = charges.positions(box_xyz);
        harness.set_system(&charges.q, &positions);
        harness.clear_acc();
        let output = harness.run(true, [0.0; 3]);
        let expected = reference(&charges.q, &positions, box_xyz, grid);
        let errors = compare(tag, n, &output, &expected);
        assert_within(tag, &errors);
        output
    };

    let start = [40.0f32, 44.0, 52.0];
    harness.mesh.upload(context.queue(), start);
    let before = evaluate("box 40x44x52 A", start);

    let scaled = start.map(|length| length * 1.01);
    harness.mesh.upload(context.queue(), scaled);
    let after = evaluate("box scaled by 1.01 (upload)", scaled);
    // The energy of the scaled system is a different number, so agreement
    // with the CPU engine is not agreement with a stale box.
    let change = (after.energy(n) - before.energy(n)).abs() / before.energy(n).abs();
    assert!(change > 1e-3, "energy changed by {change:.2e} only");

    let barostat = [start[0] * 0.98, start[1] * 1.005, start[2] * 1.03];
    context.queue().write_buffer(
        harness.mesh.uniform(),
        PME_BOX_OFFSET_BYTES,
        bytemuck::cast_slice(&barostat),
    );
    evaluate("box rewritten in the uniform (anisotropic)", barostat);
}

/// Wall time and device time of the force-only chain. Printed, not asserted:
/// the adapter is shared.
#[test]
fn reciprocal_pme_timing() {
    let _guard = gpu_test_guard();
    let Some(context) = context(true) else {
        return;
    };
    eprintln!("adapter: {}", context.adapter_info().name);
    for (atoms, box_xyz, grid) in [
        (9_000usize, [45.0f32, 45.0, 45.0], [64u32, 64, 64]),
        (100_000, [100.0, 100.0, 100.0], [128, 128, 128]),
    ] {
        time_chain(&context, atoms, box_xyz, grid);
    }
}

fn time_chain(context: &GpuContext, atoms: usize, box_xyz: [f32; 3], grid: [u32; 3]) {
    const CHAINS: usize = 200;
    const SUBMISSIONS: usize = 4;
    let charges = Charges::random(atoms, 0x5851_F42D_4C95_7F2D);
    let harness = Harness::new(context, atoms, grid);
    harness.set_system(&charges.q, &charges.positions(box_xyz));
    harness.mesh.upload(context.queue(), box_xyz);
    harness.clear_acc();
    let device = context.device();
    let queue = context.queue();
    let submit = |chains: usize, energy: bool| {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for _ in 0..chains {
                harness.encode(&mut pass, energy);
            }
        }
        queue.submit(Some(encoder.finish()));
    };
    // Warm up: pipelines, first use of the buffers.
    submit(8, false);
    submit(8, true);
    device.poll(wgpu::PollType::Wait).unwrap();
    let mut wall = [0.0f64; 2];
    for (energy, slot) in [(false, 0), (true, 1)] {
        let started = std::time::Instant::now();
        for _ in 0..SUBMISSIONS {
            submit(CHAINS, energy);
        }
        device.poll(wgpu::PollType::Wait).unwrap();
        wall[slot] = started.elapsed().as_secs_f64() * 1e6 / (CHAINS * SUBMISSIONS) as f64;
    }
    eprintln!(
        "{atoms} atoms, mesh {}x{}x{}: {:.1} us per force-only chain, {:.1} us per energy chain \
         (wall, {SUBMISSIONS} submissions of {CHAINS} chains)",
        grid[0], grid[1], grid[2], wall[0], wall[1],
    );

    // Device time per kernel from timestamps written between the dispatches.
    let Some(period_ns) = context.gpu_timestamp_period_ns() else {
        eprintln!("  timestamp queries are not available on this adapter");
        return;
    };
    const PROFILED: usize = 100;
    for energy in [false, true] {
        let chain = PmeMesh::chain(energy);
        let queries = (PROFILED * (chain.len() + 1)) as u32;
        let bytes = 8 * u64::from(queries);
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pme kernel timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pme timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pme timestamp readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let mut query = 0;
            for _ in 0..PROFILED {
                pass.write_timestamp(&query_set, query);
                query += 1;
                for &kernel in chain {
                    harness.dispatch(&mut pass, kernel);
                    pass.write_timestamp(&query_set, query);
                    query += 1;
                }
            }
        }
        encoder.resolve_query_set(&query_set, 0..queries, &resolved, 0);
        encoder.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let data = read(device, &staging);
        let stamps: Vec<u64> = bytemuck::pod_collect_to_vec(&data);
        let mut per_kernel = vec![0.0f64; chain.len()];
        for sample in stamps.chunks(chain.len() + 1) {
            for (kernel, pair) in sample.windows(2).enumerate() {
                per_kernel[kernel] += pair[1].wrapping_sub(pair[0]) as f64 * period_ns * 1e-3;
            }
        }
        let report: Vec<String> = chain
            .iter()
            .zip(&per_kernel)
            .map(|(kernel, total)| {
                format!("{} {:.1}", kernel.stage_name(), total / PROFILED as f64)
            })
            .collect();
        eprintln!(
            "  device us per kernel ({}): {} | total {:.1}",
            if energy { "energy" } else { "forces" },
            report.join(", "),
            per_kernel.iter().sum::<f64>() / PROFILED as f64,
        );
    }
}

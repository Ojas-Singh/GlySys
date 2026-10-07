//! Device self-test: do integers survive in float lanes?
//!
//! The kernels of this crate carry integers (atom indices, sentinels, the
//! thermostat's random-number words) as bit patterns in the `w` lane of
//! `vec4<f32>` values. Small integers are denormal floats and `0xffffffff` is
//! a NaN; WGSL does not promise that such patterns survive being copied as
//! floats. [`bit_patterns`] copies a set of patterns the way the kernels do
//! and reports what came back different, so a device that does not preserve
//! them can be told from one that fails for another reason.
use crate::context::GpuContext;
use crate::device::Error;
use wgpu::util::DeviceExt;

/// A pattern that did not come back as it went in.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AlteredPattern {
    pub pattern: String,
    pub returned: String,
}

/// One way of copying the patterns, and the patterns it altered.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BitPatternPath {
    pub name: &'static str,
    pub altered: Vec<AlteredPattern>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct BitPatternReport {
    pub patterns: usize,
    pub paths: Vec<BitPatternPath>,
    /// Patterns for which the sentinel comparison gave the wrong answer.
    pub sentinel_mistakes: Vec<String>,
}
impl BitPatternReport {
    /// Whether every path returned every pattern unchanged.
    pub fn preserved(&self) -> bool {
        self.paths.iter().all(|path| path.altered.is_empty()) && self.sentinel_mistakes.is_empty()
    }
}

const PATHS: [&str; 7] = [
    "bufferLoad",
    "local",
    "workgroup",
    "select",
    "cast",
    "localFromCast",
    "storeRewriteLoad",
];

/// Zero, small integers (denormal floats), the edges of the normal range,
/// infinities, signalling and quiet NaNs of both signs, and a run of
/// xorshift words like the thermostat's.
pub fn patterns() -> Vec<u32> {
    let mut patterns = vec![
        0,
        1,
        2,
        3,
        7,
        31,
        255,
        4096,
        65_535,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fa0_0000,
        0x7fc0_0000,
        0x7fc0_0001,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0x807f_ffff,
        0xff80_0000,
        0xff80_0001,
        0xffc0_0000,
        0xffff_fffe,
        0xffff_ffff,
    ];
    let mut word = 0x9e37_79b9u32;
    while patterns.len() < 64 {
        word ^= word << 13;
        word ^= word >> 17;
        word ^= word << 5;
        patterns.push(word);
    }
    patterns
}

/// Run the self-test on the device of `context`.
pub async fn bit_patterns(context: &GpuContext) -> Result<BitPatternReport, Error> {
    let device = context.device();
    let queue = context.queue();
    let patterns = patterns();
    let count = patterns.len();
    let floats: Vec<[u32; 4]> = patterns
        .iter()
        .enumerate()
        .map(|(index, &pattern)| {
            [
                (index as f32 + 0.25).to_bits(),
                1.5f32.to_bits(),
                (-2.75f32).to_bits(),
                pattern,
            ]
        })
        .collect();
    let storage = wgpu::BufferUsages::STORAGE;
    let pattern_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selftest.patterns"),
        contents: bytemuck::cast_slice(&patterns),
        usage: storage,
    });
    let float_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selftest.floats"),
        contents: bytemuck::cast_slice(&floats),
        usage: storage,
    });
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selftest.scratch"),
        size: count as u64 * 16,
        usage: storage,
        mapped_at_creation: false,
    });
    let result_bytes = count as u64 * 32;
    let results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selftest.results"),
        size: result_bytes,
        usage: storage | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selftest.staging"),
        size: result_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("selftest"),
        entries: &[
            entry(0, true),
            entry(1, true),
            entry(2, false),
            entry(3, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("selftest"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("selftest"),
        source: wgpu::ShaderSource::Wgsl(include_str!("selftest.wgsl").into()),
    });
    let pipeline = |entry_point: &'static str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let write = pipeline("write_scratch");
    let rewrite = pipeline("rewrite_scratch");
    let collect = pipeline("collect");
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("selftest"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pattern_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: float_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scratch.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: results.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("selftest"),
    });
    // one pass per kernel, as the dynamics kernels run
    for kernel in [&write, &rewrite, &rewrite, &rewrite, &collect] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("selftest"),
            timestamp_writes: None,
        });
        pass.set_pipeline(kernel);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results, 0, &staging, 0, result_bytes);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..result_bytes);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::Wait)
        .map_err(|error| Error::Execution(error.to_string()))?;
    rx.await
        .map_err(|error| Error::Execution(error.to_string()))?
        .map_err(|error| Error::Execution(error.to_string()))?;
    let mapped = slice.get_mapped_range();
    let returned = bytemuck::cast_slice::<u8, [u32; 8]>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    let hex = |value: u32| format!("{value:#010x}");
    let mut paths: Vec<BitPatternPath> = PATHS
        .iter()
        .map(|&name| BitPatternPath {
            name,
            altered: Vec::new(),
        })
        .collect();
    let mut sentinel_mistakes = Vec::new();
    for (&pattern, got) in patterns.iter().zip(&returned) {
        for (path, &value) in paths.iter_mut().zip(got.iter()) {
            if value != pattern {
                path.altered.push(AlteredPattern {
                    pattern: hex(pattern),
                    returned: hex(value),
                });
            }
        }
        if (got[7] == 1) != (pattern == u32::MAX) {
            sentinel_mistakes.push(hex(pattern));
        }
    }
    Ok(BitPatternReport {
        patterns: count,
        paths,
        sentinel_mistakes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuContextOptions;

    #[test]
    fn the_patterns_cover_denormals_nans_and_ordinary_words() {
        let patterns = patterns();
        assert_eq!(patterns.len(), 64);
        let exponent = |p: u32| (p >> 23) & 0xff;
        assert!(patterns.iter().any(|&p| p != 0 && exponent(p) == 0));
        assert!(
            patterns
                .iter()
                .any(|&p| exponent(p) == 0xff && p & 0x007f_ffff != 0)
        );
        // selftest.wgsl reads its sentinel from this slot
        assert_eq!(patterns[26], u32::MAX);
    }

    #[test]
    fn this_device_keeps_integers_in_float_lanes() {
        pollster::block_on(async {
            let Ok(context) = GpuContext::new(GpuContextOptions::default()).await else {
                eprintln!("no GPU: skipped");
                return;
            };
            let report = bit_patterns(&context).await.unwrap();
            assert_eq!(report.patterns, 64);
            assert_eq!(report.paths.len(), 7);
            assert!(report.preserved(), "{report:#?}");
        });
    }
}

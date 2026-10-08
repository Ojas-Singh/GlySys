//! Rules the shaders must keep that no device on the test bench will enforce.
//!
//! WGSL lets an implementation store one component of a vector by loading the
//! vector, replacing the component and storing the vector. Two invocations
//! that each store a *different* component of the same vector therefore race,
//! and the loser's value is gone. Vulkan drivers store the component alone, so
//! the race never shows there; Metal rewrites the vector. The tile kernels once
//! packed the sorted slots of four atoms into one `vec4` this way: in Safari
//! half the slots were lost, the far exclusions built on them were not applied,
//! and bonded sulfurs repelled each other with 4,000 kcal/mol/A.
//!
//! So: no shader stores a single component of an element of a buffer or of
//! workgroup memory, unless it is listed below with the reason it is safe.

const SHADERS: &[(&str, &str)] = &[
    ("dynamics.wgsl", include_str!("../src/dynamics.wgsl")),
    (
        "dynamics_observables.wgsl",
        include_str!("../src/dynamics_observables.wgsl"),
    ),
    ("energy.wgsl", include_str!("../src/energy.wgsl")),
    ("hydration.wgsl", include_str!("../src/hydration.wgsl")),
    (
        "implicit_tiles.wgsl",
        include_str!("../src/implicit_tiles.wgsl"),
    ),
    ("pbc.wgsl", include_str!("../src/pbc.wgsl")),
    ("pbc_tiles.wgsl", include_str!("../src/pbc_tiles.wgsl")),
    (
        "resident_rng.wgsl",
        include_str!("../src/resident_rng.wgsl"),
    ),
    ("selftest.wgsl", include_str!("../src/selftest.wgsl")),
    ("steric.wgsl", include_str!("../src/steric.wgsl")),
];

/// Component stores that are safe, with the reason.
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "energy.wgsl",
        "born[index].z=",
        "each element of `born` is written by the one invocation that owns `index`; the others only read `.x`, which that store leaves as it was",
    ),
    (
        "energy.wgsl",
        "born[idx].z=",
        "the same store in the dynamics kernel: lane 0 of the workgroup that owns `idx` is the only writer of the element",
    ),
];

/// Names declared at module scope in the storage or workgroup address space.
fn shared_names(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line
                .split_once("var<storage")
                .or_else(|| line.split_once("var<workgroup"))?
                .1;
            let name = rest.split_once('>')?.1.trim_start();
            let end = name.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
            Some(&name[..end])
        })
        .collect()
}

/// The index of the bracket that closes the one at `open`.
fn matching_bracket(text: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, &byte) in text[open..].iter().enumerate() {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `text` at `at` begins an assignment operator (`=`, `+=`, ...) and
/// not a comparison.
fn assigns(text: &[u8], mut at: usize) -> bool {
    while text.get(at) == Some(&b' ') {
        at += 1;
    }
    if matches!(
        text.get(at),
        Some(b'+' | b'-' | b'*' | b'/' | b'|' | b'&' | b'^' | b'%')
    ) {
        at += 1;
    }
    text.get(at) == Some(&b'=') && text.get(at + 1) != Some(&b'=')
}

/// Statements of `source` that store one component of an element of `name`.
fn component_stores(source: &str, name: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in source.lines() {
        let code = line.split("//").next().unwrap_or_default();
        for statement in code.split([';', '{', '}']) {
            let statement = statement.trim();
            let bytes = statement.as_bytes();
            let Some(rest) = statement.strip_prefix(name) else {
                continue;
            };
            if !rest.starts_with('[') {
                continue;
            }
            let Some(close) = matching_bracket(bytes, name.len()) else {
                continue;
            };
            let after = close + 1;
            let component = match bytes.get(after) {
                // a swizzle or a member: `.x`, `.xyz`, `.w`
                Some(b'.') => {
                    let end = after
                        + 1
                        + bytes[after + 1..]
                            .iter()
                            .take_while(|byte| byte.is_ascii_alphanumeric() || **byte == b'_')
                            .count();
                    let member = &statement[after + 1..end];
                    let swizzle = !member.is_empty()
                        && member.len() <= 4
                        && member.bytes().all(|b| b"xyzwrgba".contains(&b));
                    (swizzle && assigns(bytes, end)).then_some(())
                }
                // a second index: `v[i][j]`
                Some(b'[') => matching_bracket(bytes, after)
                    .filter(|&end| assigns(bytes, end + 1))
                    .map(|_| ()),
                _ => None,
            };
            if component.is_some() {
                found.push(statement.to_string());
            }
        }
    }
    found
}

#[test]
fn no_shader_stores_one_component_of_a_shared_vector() {
    let mut violations = Vec::new();
    let mut allowed_seen = vec![false; ALLOWED.len()];
    for (file, source) in SHADERS {
        for name in shared_names(source) {
            for statement in component_stores(source, name) {
                let compact: String = statement.split_whitespace().collect();
                let allowed = ALLOWED
                    .iter()
                    .position(|(f, prefix, _)| f == file && compact.starts_with(prefix));
                match allowed {
                    Some(index) => allowed_seen[index] = true,
                    None => violations.push(format!("{file}: {statement}")),
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "a store to one component of a shared vector may rewrite the whole vector and race with \
         stores to its other components; give each writer its own element, or list the store in \
         ALLOWED with the reason it is safe:\n{}",
        violations.join("\n")
    );
    for ((file, prefix, _), seen) in ALLOWED.iter().zip(allowed_seen) {
        assert!(
            seen,
            "ALLOWED lists `{prefix}` in {file}, which is no longer there"
        );
    }
}

#[test]
fn the_rule_finds_what_it_is_for() {
    let source = "@group(0) @binding(6) var<storage, read_write> blocks: array<vec4<f32>>;\n\
        var<workgroup> shared: array<vec4<f32>, 64>;\n\
        fn f() {\n\
          blocks[34u * n() + atom / 4u][atom % 4u] = bitcast<f32>(rank);\n\
          shared[lid].w += 1.0; if (a) { blocks[i].xy = vec2<f32>(0.0); }\n\
          blocks[rank_idx(atom)] = vec4<f32>(0.0);\n\
          let same = blocks[i].x == blocks[j][k];\n\
        }\n";
    assert_eq!(shared_names(source), ["blocks", "shared"]);
    assert_eq!(
        component_stores(source, "blocks"),
        [
            "blocks[34u * n() + atom / 4u][atom % 4u] = bitcast<f32>(rank)",
            "blocks[i].xy = vec2<f32>(0.0)"
        ]
    );
    assert_eq!(component_stores(source, "shared"), ["shared[lid].w += 1.0"]);
}

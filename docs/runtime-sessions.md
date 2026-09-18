# Unified execution sessions

The workspace has one backend-neutral lifecycle above the physics crates.

```text
glysys (prepared chemistry and snapshots)
    -> glysys-energy (terms, scoring, hydration)
    -> glysys-dynamics (synchronous CPU/reference state)
    -> glysys-gpu (resident wgpu resources and canonical WGSL)
    -> glysys-runtime (session selection, fallback, diagnostics, checkpoints)
    -> glysys-md / glysys-wasm / ReGlyco adapters
```

`glysys-dynamics` owns the scientific CPU algorithms. Its
`CpuSimulationSession` is intentionally CPU-only and does not select a device.
`glysys-gpu` owns only low-level workload resources. A coordinator creates one
`GpuContext` per job or process; it owns the adapter, device, queue, legal
limits, pipeline registry, and aggregate allocation ledger. Resident evaluators
receive that context and never request another adapter or device.

`glysys-runtime` owns the policy boundary. `ExecutionOptions` selects
`Auto`, `Cpu`, or `Gpu`, the memory policy, validation mode, thread limit, and
submission limits. `ExecutionSession` is the common asynchronous simulation
adapter; `PreparationSession`, `HydrationSession`, `ScoringSession`, and
`StericSession` keep their domain-specific results while using the same context
and diagnostic contract. The CPU path runs synchronously inside `advance` so
the physics crates remain independent of executors and browser types.

Normal GPU mode does not run CPU reference scoring. If an execution, capacity,
device-loss, or nonfinite error is recoverable and the request was `Auto`, the
session restores its last host-committed state, records the reason, and retries
the bounded work on CPU. Explicit `Gpu` returns the structured error. A final
report separates `actualBackend` from `validationMode`; a GPU run with a later
CPU fallback is reported as mixed by the adapter that observed both stages.

Runtime checkpoints use schema **v3** and contain the canonical
`SimulationState` plus runtime diagnostics. The worker message contract is
version **2** and every message carries a job identity and sequence. Older
checkpoint envelopes, raw pre-runtime state records, and worker messages are
rejected with a migration error; there is one writer and no compatibility
decoder in the runtime.

Native `glysys-md` is a command adapter around `ExecutionSession`. The browser
worker owns one context for its active job, passes that handle through
preparation/hydration/dynamics, and releases it when the job ends. The worker
does message sequencing, cancellation, transferable buffers, and storage; it
does not implement physics, device creation, fallback policy, or allocation
math. ReGlyco owns search/proposal order and Cookbook boundary policy, while
runtime scoring and steric sessions own reusable GPU resources.

The current browser dynamics adapter retains its richer staged/NPT progress
surface for protocols that are outside the runtime GPU capability gate. It still
uses the shared `GpuContext` and reports its own diagnostics; moving those
remaining protocol-specific stages into the runtime is a later physics-neutral
follow-up once their result contract is narrowed.

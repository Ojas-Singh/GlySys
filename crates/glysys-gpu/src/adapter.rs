//! Adapter identity for error attribution across GPU backends.

/// One-line adapter identity (`name (backend/vendor/device)`) so driver
/// rejections in user reports name the exact GPU and backend involved.
pub(crate) fn describe(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{} ({:?}/0x{:x}/0x{:x})",
        info.name, info.backend, info.vendor, info.device
    )
}

//! CPU topology helpers.
//!
//! Hybrid x86 CPUs (Intel 12th generation and newer) mix fast performance cores
//! with slower efficiency cores. Windows is free to migrate a process's threads
//! between them, and ONNX Runtime's worker pool does exactly that: the same
//! inference that takes ~31 ms on the performance cores takes ~37 ms when some
//! workers land on efficiency cores. Pinning the process to the performance
//! cores removes that variance and also leaves the efficiency cores free for
//! games and other background work.
//!
//! The policy only applies when the CPU actually reports both kinds of cores,
//! so homogeneous machines are untouched.

use std::sync::atomic::{AtomicUsize, Ordering};

static PERFORMANCE_CPUS: AtomicUsize = AtomicUsize::new(0);

/// Number of logical CPUs in the performance-core set, or zero when the process
/// was not restricted (homogeneous CPU, unsupported platform, or failure).
pub fn performance_cpus() -> usize {
    PERFORMANCE_CPUS.load(Ordering::Relaxed)
}

/// Restrict this process to the performance cores of a hybrid CPU.
///
/// Call once at startup, before any ONNX Runtime session is created, so the
/// inference threads inherit the affinity. Returns the number of logical CPUs
/// the process was restricted to.
pub fn pin_to_performance_cores() -> Option<usize> {
    let selected = imp::pin();
    if let Some(count) = selected {
        PERFORMANCE_CPUS.store(count, Ordering::Relaxed);
    }
    selected
}

#[cfg(windows)]
mod imp {
    use std::{ffi::c_void, io::Error};

    const RELATION_PROCESSOR_CORE: u32 = 0;
    const ERROR_INSUFFICIENT_BUFFER: i32 = 122;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            length: *mut u32,
        ) -> i32;
        fn SetProcessAffinityMask(process: *mut c_void, mask: usize) -> i32;
    }

    pub fn pin() -> Option<usize> {
        let mut length = 0u32;
        // SAFETY: a null buffer with a zero length is the documented way to ask
        // how much room the processor information needs.
        let first = unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                std::ptr::null_mut(),
                &mut length,
            )
        };
        if first != 0 || Error::last_os_error().raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER) {
            return None;
        }
        if length == 0 {
            return None;
        }

        let mut buffer = vec![0u8; length as usize];
        // SAFETY: the buffer is writable and `length` describes its size.
        let second = unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                buffer.as_mut_ptr(),
                &mut length,
            )
        };
        if second == 0 {
            return None;
        }
        buffer.truncate(length as usize);

        let mut cores: Vec<(bool, u16, usize)> = Vec::new();
        let mut offset = 0usize;
        while offset + 8 <= buffer.len() {
            let relationship = read_u32(&buffer[offset..offset + 4])?;
            let size = read_u32(&buffer[offset + 4..offset + 8])? as usize;
            if size < 8 || offset + size > buffer.len() {
                break;
            }
            if relationship == RELATION_PROCESSOR_CORE && size >= 24 {
                let body = &buffer[offset + 8..offset + size];
                let smt = body[0] & 1 != 0;
                let group_count = read_u16(&body[22..24])? as usize;
                let mut mask = 0usize;
                let mut group = 0u16;
                for index in 0..group_count {
                    let base = 24 + index * 16;
                    if base + 10 > body.len() {
                        break;
                    }
                    mask |= usize::from_ne_bytes(body[base..base + 8].try_into().ok()?);
                    group = read_u16(&body[base + 8..base + 10])?;
                }
                if mask != 0 {
                    cores.push((smt, group, mask));
                }
            }
            offset += size;
        }

        let has_smt = cores.iter().any(|core| core.0);
        let has_plain = cores.iter().any(|core| !core.0);
        // Homogeneous CPUs (all cores with SMT) do not need a restriction.
        if !(has_smt && has_plain) {
            return None;
        }
        // SetProcessAffinityMask only accepts one processor group.
        if cores.iter().any(|core| core.1 != 0) {
            return None;
        }
        let mask = cores
            .iter()
            .filter(|core| core.0)
            .fold(0usize, |mask, core| mask | core.2);
        if mask == 0 {
            return None;
        }
        // SAFETY: GetCurrentProcess returns a valid pseudo handle.
        let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) };
        if ok == 0 {
            return None;
        }
        Some(mask.count_ones() as usize)
    }

    fn read_u16(bytes: &[u8]) -> Option<u16> {
        Some(u16::from_ne_bytes(bytes.try_into().ok()?))
    }

    fn read_u32(bytes: &[u8]) -> Option<u32> {
        Some(u32::from_ne_bytes(bytes.try_into().ok()?))
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn pin() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn selection_is_idempotent() {
        // Running twice must not change the recorded CPU count.
        let first = super::pin_to_performance_cores();
        let second = super::pin_to_performance_cores();
        assert_eq!(first, second);
    }
}

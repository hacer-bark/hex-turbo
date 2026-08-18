//! Emits two convenience `cfg` aliases so the source never has to repeat the
//! SIMD feature matrix at every gate:
//!
//! * `unsafe_simd` — at least one kernel that uses `unsafe` is compiled in.
//!   When it is absent the crate is pure safe scalar Rust and carries
//!   `#![forbid(unsafe_code)]`.
//! * `x86_simd` — an x86 AVX kernel is compiled in, i.e. runtime CPU detection
//!   is needed.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(unsafe_simd)");
    println!("cargo::rustc-check-cfg=cfg(x86_simd)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let x86 = matches!(arch.as_str(), "x86" | "x86_64");
    let x86_simd = x86 && std::env::var_os("CARGO_FEATURE_SIMD").is_some();

    if x86_simd {
        println!("cargo::rustc-cfg=x86_simd");
        // Every SIMD kernel this crate has is an x86 one for now; when a NEON
        // kernel lands, this becomes `x86_simd || aarch64+neon`.
        println!("cargo::rustc-cfg=unsafe_simd");
    }
}

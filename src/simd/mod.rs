// mod sse4;
mod avx2;
#[cfg(feature = "avx512")]
mod avx512;

// pub use sse4::*;
pub use avx2::*;
#[cfg(feature = "avx512")]
pub use avx512::*;

// `x86_simd` (from build.rs) is already "x86 with some AVX kernel", so each arm
// only needs to add its own feature.
#[cfg(all(x86_simd, feature = "avx2"))]
mod avx2;
#[cfg(all(x86_simd, feature = "avx512-vbmi"))]
mod avx512_vbmi;

#[cfg(all(x86_simd, feature = "avx2"))]
pub(crate) use avx2::{decode_slice_avx2, encode_slice_avx2};
#[cfg(all(x86_simd, feature = "avx512-vbmi"))]
pub(crate) use avx512_vbmi::{decode_slice_avx512_vbmi, encode_slice_avx512_vbmi};

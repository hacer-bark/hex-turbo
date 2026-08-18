mod avx2;
mod avx512;

pub(crate) use avx2::{decode_slice_avx2, encode_slice_avx2};
pub(crate) use avx512::{decode_slice_avx512, encode_slice_avx512};

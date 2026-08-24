mod avx2;
mod avx512_vbmi;

pub(crate) use avx2::{decode_slice_avx2, encode_slice_avx2};
pub(crate) use avx512_vbmi::{decode_slice_avx512_vbmi, encode_slice_avx512_vbmi};

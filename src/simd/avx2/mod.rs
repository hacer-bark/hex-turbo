use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- CONSTANTS ---

// Duplicated 16-byte tables for AVX2 pshufb (Encoding)
const HEX_TABLE_UPPER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
];

const HEX_TABLE_LOWER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

// Duplicated 16-byte LUTs and weights for AVX2 (Decoding)
const LUT_HI: [u8; 32] = [
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];
const LUT_LO: [u8; 32] = [
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0, 128, 192, 192, 192, 192,
    192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
];
const WEIGHTS: [u8; 32] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16,
    1, 16, 1, 16, 1,
];

// --- ENCODING ---

/// Input length at or above which the wide encoder is used.
///
/// The two encode loops have different cost curves, measured on this machine
/// (Coffee Lake, core cycles/byte, min estimator, interleaved A/B under
/// SCHED_FIFO on a pinned core):
///
/// | bytes | narrow | wide  | wide vs narrow |
/// |-------|--------|-------|----------------|
/// |   256 | 0.2226 | 0.2482|         +11.5% |
/// |   512 | 0.1837 | 0.2009|          +9.4% |
/// |  1024 | 0.1775 | 0.1620|          -8.7% |
/// |  2048 | 0.1731 | 0.1512|         -12.7% |
/// |  4096 | 0.1809 | 0.1381|         -23.7% |
///
/// The narrow loop is bound on a single port and so runs at its floor from the
/// first iteration; the wide loop spreads the same work over three ports and
/// only reaches its (much lower) floor once enough iterations are in flight.
/// Crossover sits between 768 and 1024 bytes.
const ENC_WIDE_MIN: usize = 1024;

/// Encodes `input` as hex into `dst_slice`.
///
/// # Safety
/// `dst_slice` must hold at least `input.len() * 2` bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn encode_slice_avx2(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let table_ptr = if config.uppercase {
        HEX_TABLE_UPPER.as_ptr()
    } else {
        HEX_TABLE_LOWER.as_ptr()
    };
    let table = unsafe { _mm256_loadu_si256(table_ptr as *const __m256i) };

    // Wide path: 16 input bytes -> 32 output bytes with a single `vpshufb`.
    //
    // `vpmovzxbw` widens each input byte to a 16-bit lane, which is exactly the
    // span of its two output characters. Shifting the lane both ways puts the
    // high nibble in the low half and the low nibble in the high half, so one
    // table lookup emits both characters already in output order and the store
    // needs no lane fixup at all. Two port-5 uops per 16 bytes, against the six
    // per 32 bytes the narrow path spends on `vpunpck` + `vperm2i128`.
    macro_rules! encode16 {
        ($src:expr, $dst:expr, $mask:expr) => {{
            // Lane i = 0x00_bb for input byte i.
            let w = _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128($src as *const __m128i) });
            // `vpshufb` ignores bits 4-6 of an index but zeroes the output byte
            // on bit 7, so both nibbles are masked clean.
            let idx = _mm256_and_si256(
                _mm256_or_si256(_mm256_slli_epi16(w, 8), _mm256_srli_epi16(w, 4)),
                $mask,
            );
            let chars = _mm256_shuffle_epi8(table, idx);
            unsafe { _mm256_storeu_si256($dst as *mut __m256i, chars) };
        }};
    }

    // Narrow path: 32 input bytes -> 64 output bytes.
    //
    // Both nibble planes are looked up across a full vector and then woven back
    // together. That costs three extra port-5 uops per 32 bytes, but the loop is
    // limited by that one port rather than by how much work is in flight, so it
    // hits its floor immediately -- which is what short inputs need.
    let mask_0f = _mm256_set1_epi8(0x0F);
    macro_rules! encode32 {
        ($src:expr, $dst:expr) => {{
            let v = unsafe { _mm256_loadu_si256($src as *const __m256i) };

            let low_idx = _mm256_and_si256(v, mask_0f);
            let high_idx = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);

            let low_chars = _mm256_shuffle_epi8(table, low_idx);
            let high_chars = _mm256_shuffle_epi8(table, high_idx);

            // Interleave (high, low) per byte, then repair the 128-bit lane
            // split that `vpunpck` leaves behind.
            let inter_lo = _mm256_unpacklo_epi8(high_chars, low_chars);
            let inter_hi = _mm256_unpackhi_epi8(high_chars, low_chars);

            let out0 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
            let out1 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);

            unsafe { _mm256_storeu_si256($dst as *mut __m256i, out0) };
            unsafe { _mm256_storeu_si256($dst.add(32) as *mut __m256i, out1) };
        }};
    }

    let mut src = input;

    if src.len() >= ENC_WIDE_MIN {
        // Kept inside the branch: hoisting it would make every short input pay
        // for a constant it never uses.
        let nibble_mask = _mm256_set1_epi16(0x0F0F);

        while src.len() >= 128 {
            let p = src.as_ptr();
            encode16!(p, dst, nibble_mask);
            encode16!(p.add(16), dst.add(32), nibble_mask);
            encode16!(p.add(32), dst.add(64), nibble_mask);
            encode16!(p.add(48), dst.add(96), nibble_mask);
            encode16!(p.add(64), dst.add(128), nibble_mask);
            encode16!(p.add(80), dst.add(160), nibble_mask);
            encode16!(p.add(96), dst.add(192), nibble_mask);
            encode16!(p.add(112), dst.add(224), nibble_mask);

            src = &src[128..];
            dst = unsafe { dst.add(256) };
        }
    }

    while src.len() >= 32 {
        encode32!(src.as_ptr(), dst);

        src = &src[32..];
        dst = unsafe { dst.add(64) };
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) } as usize;
        scalar::encode_slice(config, src, &mut dst_slice[dst_off..]);
    }
}

// --- DECODING ---

#[target_feature(enable = "avx2")]
pub unsafe fn decode_slice_avx2(input: &[u8], dst_slice: &mut [u8]) -> Result<(), Error> {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let lut_hi = unsafe { _mm256_loadu_si256(LUT_HI.as_ptr() as *const __m256i) };
    let lut_lo = unsafe { _mm256_loadu_si256(LUT_LO.as_ptr() as *const __m256i) };
    let weights = unsafe { _mm256_loadu_si256(WEIGHTS.as_ptr() as *const __m256i) };

    let mask_0f = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();

    // Validity is folded across the whole input and inspected once, after the
    // loops. `vpminub` keeps the weakest byte seen, so a single zero anywhere
    // survives to the end -- which removes a `vpcmpeqb`/`vpmovmskb`/branch
    // triple from the middle of every iteration.
    let mut valid_acc = _mm256_set1_epi8(-1);

    macro_rules! decode_chunk {
        ($input:expr) => {{
            let lo = _mm256_and_si256($input, mask_0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_0f);

            let hi_props = _mm256_shuffle_epi8(lut_hi, hi);
            let lo_props = _mm256_shuffle_epi8(lut_lo, lo);

            let valid = _mm256_and_si256(hi_props, lo_props);

            let offset = _mm256_and_si256(hi_props, mask_0f);
            let nibbles = _mm256_add_epi8(lo, offset);

            let pairs = _mm256_maddubs_epi16(nibbles, weights);

            (pairs, valid)
        }};
    }

    let mut src = input;

    while src.len() >= 128 {
        let v0 = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.as_ptr().add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.as_ptr().add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.as_ptr().add(96) as *const __m256i) };

        let (r0, v0_val) = decode_chunk!(v0);
        let (r1, v1_val) = decode_chunk!(v1);
        let (r2, v2_val) = decode_chunk!(v2);
        let (r3, v3_val) = decode_chunk!(v3);

        let v01 = _mm256_min_epu8(v0_val, v1_val);
        let v23 = _mm256_min_epu8(v2_val, v3_val);
        valid_acc = _mm256_min_epu8(valid_acc, _mm256_min_epu8(v01, v23));

        let packed01 = _mm256_packus_epi16(r0, r1);
        let ordered01 = _mm256_permute4x64_epi64(packed01, 0xD8);
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, ordered01) };

        let packed23 = _mm256_packus_epi16(r2, r3);
        let ordered23 = _mm256_permute4x64_epi64(packed23, 0xD8);
        unsafe { _mm256_storeu_si256(dst.add(32) as *mut __m256i, ordered23) };

        src = &src[128..];
        dst = unsafe { dst.add(64) };
    }

    while src.len() >= 32 {
        let v = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        let (pairs, valid) = decode_chunk!(v);
        valid_acc = _mm256_min_epu8(valid_acc, valid);

        let low = _mm256_castsi256_si128(pairs);
        let high = _mm256_extracti128_si256(pairs, 1);
        let res = _mm_packus_epi16(low, high);

        unsafe { _mm_storeu_si128(dst as *mut __m128i, res) };

        src = &src[32..];
        dst = unsafe { dst.add(16) };
    }

    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(valid_acc, zero)) != 0 {
        return Err(Error::InvalidCharacter);
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) } as usize;
        scalar::decode_slice(src, &mut dst_slice[dst_off..])?;
    }

    Ok(())
}

// Verification: Kani proofs, intrinsic stubs, and the Miri coverage suite.
#[cfg(any(kani, test))]
mod verify;

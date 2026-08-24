//! AVX-512 VBMI verification: an Intel-pseudocode software model of the one
//! instruction Miri can't execute, plus the Miri coverage suite. Split out of
//! the production module purely to keep it lean; nothing here is compiled
//! into a normal build.

// Re-exported to the `miri` submodule below; not compiled in a plain
// `cargo test`, so the glob looks unused there.
#[allow(unused_imports)]
use super::*;

/// A Rust model of `vpmultishiftqb`, the one AVX-512 VBMI instruction Miri's
/// x86 intrinsic interpreter doesn't implement. `zmm_multishift_epi64_epi8`
/// in the parent module routes to this under `cfg(miri)` instead of calling
/// `_mm512_multishift_epi64_epi8` directly.
///
/// Transcribed line for line from the `_mm512_multishift_epi64_epi8`
/// pseudocode in the Intel Intrinsics Guide (data version 3.6.9), quoted in
/// the comments directly above the Rust statement it became: nothing here is
/// paraphrased or simplified, so a reader can diff the two by eye. The one
/// systematic departure is that Intel addresses bits (`i := j*64`, then
/// `a[i+j*8+7:i+j*8]`) while Rust indexes bytes, so every bit offset from the
/// guide is kept verbatim and divided by 8 at the point of access.
///
/// Checked against the real instruction, on hardware that has it, by
/// [`avx512_vbmi_multishift_model_matches_hardware`] below.
// Every index below is `(q + j*8) / 8` or `n / 8` with `q < 512`, `j < 8`,
// `n < 512` -- always < 64, the length of every array here -- so the
// indexing this transcription does letter-for-letter cannot panic.
#[allow(clippy::indexing_slicing)]
#[allow(dead_code)]
pub(in crate::simd::avx512_vbmi) mod intrinsic_models {
    use super::*;
    use std::mem::transmute;

    /// Reads bit `n` of a little-endian byte vector, for the pseudocode's
    /// single-bit indexing. Scaffolding, not from Intel.
    const fn bit(v: &[u8; 64], n: usize) -> u8 {
        (v[n / 8] >> (n % 8)) & 1
    }

    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_multishift_epi64_epi8
    #[allow(clippy::needless_range_loop)]
    pub(in crate::simd::avx512_vbmi) unsafe fn multishift_epi64_epi8_model(
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR i := 0 to 7
        for i in 0..8 {
            // 	q := i * 64
            let q = i * 64;
            // 	FOR j := 0 to 7
            for j in 0..8 {
                // 		tmp8 := 0
                let mut tmp8: u8 = 0;
                // 		ctrl := a[q+j*8+7:q+j*8] & 63
                let ctrl = usize::from(a[(q + j * 8) / 8]) & 63;
                // 		FOR l := 0 to 7
                for l in 0..8 {
                    // 			tmp8[l] := b[q+((ctrl+l) & 63)]
                    tmp8 |= bit(&b, q + ((ctrl + l) & 63)) << l;
                }
                // 		ENDFOR
                // 		dst[q+j*8+7:q+j*8] := tmp8[7:0]
                dst[(q + j * 8) / 8] = tmp8;
            }
            // 	ENDFOR
        }
        // ENDFOR
        // dst[MAX:512] := 0
        // NOTE: `__m512i` is exactly 512 bits; there is nothing above to zero.

        unsafe { transmute(dst) }
    }
}

/// Checks [`intrinsic_models::multishift_epi64_epi8_model`] against the real
/// `vpmultishiftqb` on AVX-512-VBMI hardware, under plain `cargo test`.
/// Skips, loudly, on a host without the ISA rather than failing, so it is
/// free to run everywhere and does the real work wherever the silicon turns
/// up.
#[cfg(all(test, not(miri)))]
mod avx512_vbmi_stub_equivalence {
    use super::intrinsic_models::multishift_epi64_epi8_model;
    use super::*;
    use std::mem::transmute;

    /// Bit patterns worth checking: zero, all-ones, alternating, a per-byte
    /// ramp (so every possible `ctrl` value 0..=63 appears in `a`), and some
    /// deterministic noise.
    fn probes() -> Vec<[u8; 64]> {
        let byte = |i: usize| u8::try_from(i).expect("index below the 64-byte vector width");

        let mut out = vec![[0x00; 64], [0xFF; 64], [0xAA; 64], [0x55; 64]];
        out.push(core::array::from_fn(byte));
        out.push(core::array::from_fn(|i| byte(i % 64)));
        out.push(core::array::from_fn(|i| 0xFF - byte(i % 64)));

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..8 {
            out.push(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::try_from(state >> 56).expect("shifted down to 8 bits")
            }));
        }
        out
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn compare_all() {
        let probes = probes();
        // SAFETY: `__m512i` has no invalid bit patterns, so it and `[u8; 64]`
        // are freely transmutable both ways.
        let bytes = |v: __m512i| -> [u8; 64] { unsafe { transmute::<__m512i, [u8; 64]>(v) } };
        let zmm = |b: [u8; 64]| -> __m512i { unsafe { transmute::<[u8; 64], __m512i>(b) } };

        for x in &probes {
            for y in &probes {
                let (a, b) = (zmm(*x), zmm(*y));
                assert_eq!(
                    bytes(_mm512_multishift_epi64_epi8(a, b)),
                    bytes(unsafe { multishift_epi64_epi8_model(a, b) }),
                    "_mm512_multishift_epi64_epi8: a={x:02x?} b={y:02x?}"
                );
            }
        }
    }

    #[test]
    fn avx512_vbmi_multishift_model_matches_hardware() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi"))
        {
            eprintln!("skipping: host CPU lacks AVX-512-VBMI");
            return;
        }
        unsafe { compare_all() };
    }
}

// --- MIRI (UB DETECTION) ---

#[cfg(all(test, miri))]
mod avx512_vbmi_miri_tests {
    use super::{decode_slice_avx512_vbmi, encode_slice_avx512_vbmi};
    use crate::{Config, Error};

    // Reference crate
    use hex::encode as ref_encode_lower;

    // --- Deterministic Generator ---
    fn get_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    // --- Helpers ---
    fn verify_roundtrip(config: Config, input: &[u8]) {
        let len = input.len();

        // --- Encoding ---
        let expected_lower = ref_encode_lower(input);
        let expected = if config.uppercase {
            expected_lower.to_ascii_uppercase()
        } else {
            expected_lower
        };

        let mut enc_buf = vec![0u8; len * 2];
        unsafe {
            encode_slice_avx512_vbmi(config, input, &mut enc_buf);
        }

        assert_eq!(
            &enc_buf[..],
            expected.as_bytes(),
            "AVX-512 VBMI encoding mismatch (len={len})"
        );

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe {
            decode_slice_avx512_vbmi(&enc_buf, &mut dec_buf)
                .expect("AVX-512 VBMI decoder failed on valid own output")
        };

        assert_eq!(
            &dec_buf[..],
            input,
            "AVX-512 VBMI round-trip failed (len={len})"
        );
    }

    fn run_avx512_vbmi_tests(uppercase: bool) {
        let config = Config { uppercase };

        // Miri is slow, so this is boundary coverage rather than random
        // lengths: every step size the kernels have (32 and 64 bytes of
        // input), the 128-byte encode block, the 256-character decode block,
        // one byte either side of each, and the (Miri-only, much smaller)
        // `PREFETCH_MIN`/`NONTEMPORAL_MIN` crossings at 200/300 -- see the
        // `#[cfg(miri)]` constants in `mod.rs`. Those two branches use real
        // hardware instructions (`asm!`-based prefetch and non-temporal
        // stores) that Miri cannot execute at all at the production
        // thresholds (512 KiB / 1 MiB), so without the shrunk constants they
        // would never run under Miri.
        let boundaries = [
            0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 199, 200, 201, 255, 256, 257,
            299, 300, 301, 512,
        ];

        for &len in &boundaries {
            let input = get_data(len);
            verify_roundtrip(config, &input);
        }
    }

    // --- Tests ---

    #[test]
    fn miri_avx512_vbmi_lower_roundtrip() {
        run_avx512_vbmi_tests(false);
    }

    #[test]
    fn miri_avx512_vbmi_upper_roundtrip() {
        run_avx512_vbmi_tests(true);
    }

    #[test]
    fn miri_avx512_vbmi_decode_mixed_case() {
        // 128 bytes = 256 characters, one full unrolled decode block.
        let input = get_data(128);
        let hex_lower = ref_encode_lower(&input).into_bytes();

        // Deterministically mix case
        let mixed_hex: Vec<u8> = hex_lower
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                if i % 2 == 0 {
                    b.to_ascii_uppercase()
                } else {
                    b
                }
            })
            .collect();

        let mut dec_buf = vec![0u8; 128];
        unsafe {
            decode_slice_avx512_vbmi(&mixed_hex, &mut dec_buf)
                .expect("AVX-512 VBMI decoder failed on valid mixed-case input")
        };

        assert_eq!(&dec_buf[..], input);
    }

    #[test]
    fn miri_avx512_vbmi_decode_streaming_store_every_dst_alignment() {
        // Exercises the non-temporal-store path's `dst`-alignment handling:
        // the head-fixup that walks to a 32-byte boundary before the first
        // streamed store. Unlike the AVX2 encoder's streaming path, there is
        // no odd-`dst.addr()` case to skip here -- the decoder streams
        // unconditionally past `NONTEMPORAL_MIN`, which is 300 characters
        // under Miri, so 320 characters of input crosses it regardless of how
        // much of the front the head-fixup eats. Sweeping every offset from 0
        // to 33 covers every residue mod 32 at least once.
        let input = get_data(160);
        let hex = ref_encode_lower(&input);
        assert_eq!(hex.len(), 320);

        for off in 0..34usize {
            let mut buf = vec![0u8; input.len() + off];
            unsafe {
                decode_slice_avx512_vbmi(hex.as_bytes(), &mut buf[off..])
                    .expect("AVX-512 VBMI decoder failed on valid input")
            };
            assert_eq!(&buf[off..], &input[..], "dst offset {off}");
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_errors() {
        let mut out = [0u8; 128];

        // 1. Invalid character inside the unrolled 256-character block.
        let mut invalid_block = vec![b'0'; 256];
        invalid_block[200] = b'g';
        let res = unsafe { decode_slice_avx512_vbmi(&invalid_block, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 2. Invalid character in the 64-character step.
        let mut invalid_simd = vec![b'0'; 128];
        invalid_simd[65] = b'g';
        let res = unsafe { decode_slice_avx512_vbmi(&invalid_simd, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 3. Invalid character in the scalar tail.
        let mut invalid_tail = vec![b'0'; 66];
        invalid_tail[65] = b'g';
        let res = unsafe { decode_slice_avx512_vbmi(&invalid_tail, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 4. A byte >= 0x80: the permute ignores bit 7, so this one aliases
        //    onto a valid low-half entry and is only caught by the input test.
        let mut high_bit = vec![b'0'; 128];
        high_bit[70] = 0xB0; // b'0' | 0x80
        let res = unsafe { decode_slice_avx512_vbmi(&high_bit, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));
    }
}

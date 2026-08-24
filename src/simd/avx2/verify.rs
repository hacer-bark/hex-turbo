//! AVX2 verification: the Miri coverage suite. Split out of the production
//! module purely to keep it lean; nothing here is compiled into a normal
//! build.

// Re-exported to the `miri` submodule below; not compiled in a plain
// `cargo test`, so the glob looks unused there.
#[allow(unused_imports)]
use super::*;

// --- MIRI (FORMAL VERIFICATION) ---

#[cfg(all(test, miri))]
mod avx2_miri_tests {
    use super::{decode_slice_avx2, encode_slice_avx2};
    use crate::{Config, Error};

    // Reference crate
    use hex::encode as ref_encode_lower;

    // --- Deterministic Generator ---
    // Random generation is slow under Miri; sequential bytes exercise the
    // same bitwise logic just as effectively.
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
            encode_slice_avx2(config, input, &mut enc_buf);
        }

        assert_eq!(
            &enc_buf[..],
            expected.as_bytes(),
            "AVX2 encoding mismatch (len={len})"
        );

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe {
            decode_slice_avx2(&enc_buf, &mut dec_buf)
                .expect("AVX2 decoder failed on valid own output")
        };

        assert_eq!(&dec_buf[..], input, "AVX2 round-trip failed (len={len})");
    }

    fn run_avx2_tests(uppercase: bool) {
        let config = Config { uppercase };

        // Miri is slow, so this is boundary coverage rather than random
        // lengths: the scalar tail (0..16), the 32-byte AVX2 step, the
        // 128-byte encode block, one byte either side of each, and the
        // (Miri-only, much smaller) `PREFETCH_MIN`/`NONTEMPORAL_MIN`
        // crossings at 200/300 -- see the `#[cfg(miri)]` constants in
        // `mod.rs`. Those two branches use real hardware instructions
        // (`asm!`-based prefetch and non-temporal stores) that Miri cannot
        // execute at all at the production thresholds (32 KiB / 2 MiB), so
        // without the shrunk constants they would never run under Miri.
        let boundaries = [
            0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 199, 200, 201, 299, 300, 301, 512,
        ];

        for &len in &boundaries {
            let input = get_data(len);
            verify_roundtrip(config, &input);
        }
    }

    // --- Tests ---

    #[test]
    fn miri_avx2_lower_roundtrip() {
        run_avx2_tests(false);
    }

    #[test]
    fn miri_avx2_upper_roundtrip() {
        run_avx2_tests(true);
    }

    #[test]
    fn miri_avx2_decode_mixed_case() {
        // 64 length ensures we trigger exactly 2 full AVX2 loops (64 bytes = 128 hex chars)
        let input = get_data(64);
        let hex_lower = ref_encode_lower(&input).into_bytes();

        // Deterministically mix case (avoids heavy `rand` in Miri)
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

        let mut dec_buf = vec![0u8; 64];
        unsafe {
            decode_slice_avx2(&mixed_hex, &mut dec_buf)
                .expect("AVX2 decoder failed on valid mixed-case input")
        };

        assert_eq!(&dec_buf[..], input);
    }

    #[test]
    fn miri_avx2_encode_streaming_store_every_dst_alignment() {
        // Exercises the non-temporal-store path's `dst`-alignment handling:
        // the head-fixup that walks to a 32-byte boundary before the first
        // streamed store, and the odd-`dst.addr()` case that skips streaming
        // entirely (`vmovntdq` requires an aligned address, and an odd
        // address can never become 32-byte aligned by advancing an even
        // distance). `NONTEMPORAL_MIN` is 300 bytes under Miri, so 320 bytes
        // of input crosses it regardless of how much of the front the
        // head-fixup eats. Sweeping every offset from 0 to 33 covers every
        // parity and every residue mod 32 at least once.
        let input = get_data(320);
        let expected = ref_encode_lower(&input);

        for off in 0..34usize {
            let mut buf = vec![0u8; input.len() * 2 + off];
            unsafe {
                encode_slice_avx2(Config { uppercase: false }, &input, &mut buf[off..]);
            }
            assert_eq!(&buf[off..], expected.as_bytes(), "dst offset {off}");
        }
    }

    #[test]
    fn miri_avx2_decode_errors() {
        let mut out = [0u8; 128];

        // 1. Invalid character in SIMD region (64 hex chars = 32 bytes = 1 AVX2 chunk)
        let mut invalid_simd = vec![b'0'; 64];
        invalid_simd[33] = b'g'; // 'g' is strictly invalid in hex
        let res = unsafe { decode_slice_avx2(&invalid_simd, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 2. Invalid character in Scalar/Tail region (66 chars = 64 SIMD + 2 scalar)
        let mut invalid_tail = vec![b'0'; 66];
        invalid_tail[65] = b'g';
        let res = unsafe { decode_slice_avx2(&invalid_tail, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));
    }
}

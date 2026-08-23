//! Differential tests: every kernel that the current feature set dispatches to,
//! against the reference `hex` crate, across every length that crosses a block
//! boundary.
//!
//! Run twice to cover both halves of the dispatch matrix:
//!
//! ```text
//! cargo test --test differential                                  # SIMD (if the host has it)
//! cargo test --test differential --no-default-features --features std
//! ```

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use hex_turbo::{Error, LOWER_CASE, UPPER_CASE};

const fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| (xorshift(&mut s) >> 24).to_le_bytes()[0])
        .collect()
}

/// Lengths that straddle every stride in the crate: scalar (4/8), AVX2 (32/128),
/// AVX-512 (64/256), plus their neighbours.
fn lengths() -> Vec<usize> {
    let mut v: Vec<usize> = (0..=140).collect();
    for base in [255usize, 256, 257, 511, 512, 513, 1023, 1024, 4096, 10_000] {
        v.push(base);
    }
    v
}

#[test]
fn encode_matches_reference() {
    for len in lengths() {
        let input = random_bytes(len, 0x9E37_79B9_7F4A_7C15 ^ len as u64);

        let lower = LOWER_CASE.encode(&input);
        let upper = UPPER_CASE.encode(&input);

        assert_eq!(
            lower,
            hex::encode(&input),
            "lowercase mismatch at len {len}"
        );
        assert_eq!(
            upper,
            hex::encode_upper(&input),
            "uppercase mismatch at len {len}"
        );

        // Zero-allocation API must agree with the allocating one.
        let mut buf = vec![0u8; len * 2];
        let n = LOWER_CASE.encode_into(&input, &mut buf).unwrap();
        assert_eq!(n, len * 2);
        assert_eq!(buf, lower.as_bytes(), "encode_into mismatch at len {len}");
    }
}

#[test]
fn decode_roundtrips_and_accepts_mixed_case() {
    for len in lengths() {
        let input = random_bytes(len, 0xD1B5_4A32_D192_ED03 ^ len as u64);
        let lower = LOWER_CASE.encode(&input);
        let upper = UPPER_CASE.encode(&input);

        assert_eq!(LOWER_CASE.decode(&lower).unwrap(), input, "lower at {len}");
        assert_eq!(LOWER_CASE.decode(&upper).unwrap(), input, "upper at {len}");
        assert_eq!(
            UPPER_CASE.decode(&lower).unwrap(),
            input,
            "engine-swap at {len}"
        );

        // Alternating case within the same input.
        let mixed: Vec<u8> = lower
            .bytes()
            .zip(upper.bytes())
            .enumerate()
            .map(|(i, (l, u))| if i % 2 == 0 { l } else { u })
            .collect();
        assert_eq!(LOWER_CASE.decode(&mixed).unwrap(), input, "mixed at {len}");

        let mut buf = vec![0u8; len];
        let n = LOWER_CASE.decode_into(&lower, &mut buf).unwrap();
        assert_eq!(n, len);
        assert_eq!(buf, input, "decode_into mismatch at len {len}");
    }
}

#[test]
fn rejects_invalid_characters_at_every_position() {
    // One byte of garbage, walked across the whole payload, must be caught
    // wherever it lands — including inside a wide unrolled block.
    for len in [2usize, 8, 16, 32, 34, 64, 66, 128, 130, 256, 258] {
        let input = random_bytes(len / 2, 0x2545_F491_4F6C_DD1D ^ len as u64);
        let good = LOWER_CASE.encode(&input);

        for pos in 0..good.len() {
            for bad in [b'g', b'G', b'/', b':', b'@', 0x00, 0xFF] {
                let mut corrupt = good.clone().into_bytes();
                corrupt[pos] = bad;
                assert_eq!(
                    LOWER_CASE.decode(&corrupt),
                    Err(Error::InvalidCharacter),
                    "len {len}, pos {pos}, byte {bad:#04x} was accepted"
                );
            }
        }
    }
}

#[test]
fn rejects_odd_lengths_and_short_buffers() {
    for len in [1usize, 3, 31, 33, 63, 65, 129] {
        let odd = "a".repeat(len);
        assert_eq!(
            LOWER_CASE.decode(&odd),
            Err(Error::InvalidLength),
            "len {len}"
        );

        let mut buf = vec![0u8; len / 2];
        assert_eq!(
            LOWER_CASE.decode_into(&odd, &mut buf),
            Err(Error::InvalidLength),
            "decode_into len {len}"
        );
    }

    let data = b"hello world, hello world, hello world";
    let mut small = vec![0u8; data.len() * 2 - 1];
    assert_eq!(
        LOWER_CASE.encode_into(data, &mut small),
        Err(Error::BufferTooSmall)
    );

    let encoded = LOWER_CASE.encode(data);
    let mut small = vec![0u8; data.len() - 1];
    assert_eq!(
        LOWER_CASE.decode_into(&encoded, &mut small),
        Err(Error::BufferTooSmall)
    );
}

#[test]
fn oversized_output_buffer_is_untouched_past_the_written_prefix() {
    let input = random_bytes(100, 7);
    let mut buf = vec![0xAAu8; 400];
    let n = LOWER_CASE.encode_into(&input, &mut buf).unwrap();
    assert_eq!(n, 200);
    assert!(
        buf[200..].iter().all(|&b| b == 0xAA),
        "wrote past encoded_len"
    );

    let encoded = LOWER_CASE.encode(&input);
    let mut buf = vec![0xAAu8; 400];
    let n = LOWER_CASE.decode_into(&encoded, &mut buf).unwrap();
    assert_eq!(n, 100);
    assert!(
        buf[100..].iter().all(|&b| b == 0xAA),
        "wrote past decoded_len"
    );
}

#[test]
fn validity_matches_reference_for_every_byte_value() {
    // The scalar decoder validates arithmetically rather than by table, so walk
    // all 256 byte values through every position of a block that is wide enough
    // to reach each kernel's main loop, and compare accept/reject with `hex`.
    for len in [2usize, 8, 16, 32, 64, 72] {
        let filler = "a".repeat(len - 2);

        for pos in [0usize, 1, len / 2, len - 2, len - 1] {
            for byte in 0u8..=255 {
                let mut s = format!("{filler}aa").into_bytes();
                s[pos] = byte;

                let ours = LOWER_CASE.decode(&s);
                let theirs = hex::decode(&s);

                assert_eq!(
                    ours.is_ok(),
                    theirs.is_ok(),
                    "len {len}, pos {pos}, byte {byte:#04x}: ours={ours:?} hex={theirs:?}"
                );
                if let (Ok(a), Ok(b)) = (&ours, &theirs) {
                    assert_eq!(a, b, "len {len}, pos {pos}, byte {byte:#04x}");
                }
            }
        }
    }
}

/// The AVX2 kernels take different paths past 32 KiB (software prefetching)
/// and past 2 MiB (non-temporal stores). The streaming path also has an
/// alignment precondition -- `vmovntdq` faults on an unaligned address -- so
/// every destination misalignment has to round-trip, including the odd ones
/// the kernel is expected to decline to stream for.
#[test]
fn large_inputs_match_reference_at_every_destination_alignment() {
    for &len in &[32 * 1024 + 1, 2 * 1024 * 1024 + 1, 2 * 1024 * 1024 + 129] {
        let input = random_bytes(len, 0xA5A5_0F0F_1234_5678 ^ len as u64);
        let expected = hex::encode(&input);

        for off in [0usize, 1, 2, 8, 15, 16, 17, 30, 31] {
            let mut enc = vec![0u8; len * 2 + off];
            let n = LOWER_CASE.encode_into(&input, &mut enc[off..]).unwrap();
            assert_eq!(n, len * 2);
            assert_eq!(
                &enc[off..],
                expected.as_bytes(),
                "encode mismatch at len {len}, dst offset {off}"
            );

            let mut dec = vec![0u8; len + off];
            let n = LOWER_CASE
                .decode_into(expected.as_bytes(), &mut dec[off..])
                .unwrap();
            assert_eq!(n, len);
            assert_eq!(
                &dec[off..],
                &input[..],
                "decode mismatch at len {len}, dst offset {off}"
            );
        }
    }
}

/// A bad character anywhere in a large input must still be rejected -- the
/// prefetching and streaming loops accumulate validity the same way the plain
/// one does, but they are separate loops and are not otherwise exercised.
#[test]
fn large_inputs_reject_invalid_characters() {
    for &len in &[32 * 1024 + 1, 2 * 1024 * 1024 + 1] {
        let input = random_bytes(len, 0x1357_9BDF ^ len as u64);
        let good = hex::encode(&input);

        for pos in [0usize, 1, 1000, len, len * 2 - 1] {
            for bad in [b'g', b'/', b':', b'@', b'G', 0x00, 0xFF, b' '] {
                let mut s = good.clone().into_bytes();
                s[pos] = bad;
                assert!(
                    LOWER_CASE.decode(&s).is_err(),
                    "accepted {bad:#04x} at position {pos} of a {len}-byte input"
                );
            }
        }
    }
}

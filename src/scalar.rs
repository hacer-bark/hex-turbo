//! Scalar hex kernels.
//!
//! This module is the portable fallback used when no SIMD kernel applies —
//! non-x86 targets, hosts without AVX2/AVX-512, inputs too short to amortize a
//! vector setup, and `--no-default-features` builds.
//!
//! It contains **no `unsafe` code at all** (`#![forbid(unsafe_code)]`): bounds
//! are expressed through slice splitting and `chunks_exact`, which the
//! optimizer turns back into the same unchecked pointer walk. In a build with
//! every SIMD kernel disabled, the whole crate inherits that and carries
//! `forbid(unsafe_code)` crate-wide.
//!
//! # Why these two shapes
//!
//! The two directions are limited by opposite halves of the core, which is why
//! they use different techniques rather than one shared one. Measured on
//! Skylake-family hardware (`llvm-mca -mcpu=skylake` on the emitted inner
//! loops, cross-checked against TSC-derived core cycles):
//!
//! * **Encode is load-limited.** A 512-byte table maps one input byte straight
//!   to the two characters it encodes, so a group is one load per byte instead
//!   of two nibble lookups. That saturates the two load ports (16 load uops per
//!   8-byte group) at ~1.0 cycles/byte, and the ALU ports idle. Doing the
//!   arithmetic in registers instead (`nibbles_to_ascii`, still used by the
//!   decoder) is ALU-limited at ~4.9 ALU uops/byte and measured ~55% slower.
//! * **Decode is ALU-limited.** Arithmetic beats a table here for the same
//!   reason in reverse: a table decode needs two loads per *character*, which
//!   costs more load traffic than the encoder's, while the arithmetic form
//!   needs one load per 8 characters. It runs ~9 ALU uops/byte against idle
//!   load ports, so the group count per iteration — not the op count — is what
//!   moves it: four independent 8-character groups per iteration overlap their
//!   dependency chains (~17 cycles each, ~5 cycles of port pressure each).
//!
//! Two things that look right on paper and are *not* done here, both rejected
//! on measurement: splitting each iteration between the table and arithmetic
//! paths to balance the ports (register pressure costs more than the balance
//! gains — decode regressed ~10%), and unrolling decode to eight groups
//! (~29% regression).

#![forbid(unsafe_code)]

use crate::{Config, Error};

// --- CONSTANTS ---

const LOWER_ALPHABET: [u8; 16] = *b"0123456789abcdef";
const UPPER_ALPHABET: [u8; 16] = *b"0123456789ABCDEF";

/// Maps one input byte straight to the two characters it encodes, packed
/// little-endian so the first character lands in the low byte.
// Every index below is `i >> 4` or `i & 0x0F`, i.e. `0..16` into a 16-entry
// array, or `i` itself bounded by the `while i < 256` loop into the 256-entry
// `table` -- both provably in range, just not to clippy's static analysis.
#[allow(clippy::indexing_slicing)]
const fn pair_table(alphabet: [u8; 16]) -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = (alphabet[i >> 4] as u16) | ((alphabet[i & 0x0F] as u16) << 8);
        i += 1;
    }
    table
}

static LOWER_PAIRS: [u16; 256] = pair_table(LOWER_ALPHABET);
static UPPER_PAIRS: [u16; 256] = pair_table(UPPER_ALPHABET);

/// One set bit per byte lane.
const ONES: u64 = 0x0101_0101_0101_0101;

/// Distance from `'0' + n` to the character for a nibble `>= 10`:
/// `b'a' - b'0' - 10`. Only the lowercase bias is needed — the decoder folds
/// uppercase input onto lowercase before its round-trip check, and the encoder
/// uses tables rather than arithmetic.
const LOWER_BIAS: u64 = 39;

// `digit`/`lower`/`upper` are all ASCII bytes built from `b'0'`/`b'a'`/`b'A'`
// plus a `0..16` offset, so every index below is in range by construction.
#[allow(clippy::indexing_slicing)]
const HEX_DECODE_TABLE: [u8; 256] = {
    let mut table = [0xFFu8; 256];

    let mut i = 0u8;
    while i < 10 {
        let digit = b'0' + i;
        table[digit as usize] = i;
        i += 1;
    }

    i = 10;
    while i < 16 {
        let lower = b'a' + (i - 10);
        let upper = b'A' + (i - 10);
        table[lower as usize] = i;
        table[upper as usize] = i;
        i += 1;
    }

    table
};

// --- ENCODING ---

/// Maps eight 4-bit nibbles (one per byte, all `<= 0x0F`) to their ASCII hex
/// characters, branchlessly and without a lookup table.
///
/// `letter_bias` is `b'a' - b'0' - 10 == 39` for lowercase, `7` for uppercase.
#[inline]
const fn nibbles_to_ascii(nib: u64, letter_bias: u64) -> u64 {
    // `n + 6` carries into bit 4 exactly when `n >= 10`, so this is a
    // per-byte "is a letter" flag without a single comparison.
    let letters = (nib.wrapping_add(0x0606_0606_0606_0606) >> 4) & ONES;

    // Each byte of `letters` is 0 or 1 and the bias fits in a byte, so the
    // multiply cannot carry between lanes.
    nib.wrapping_add(0x3030_3030_3030_3030)
        .wrapping_add(letters.wrapping_mul(letter_bias))
}

/// Encodes `input` as hex into `dst`.
///
/// `dst` must be at least `input.len() * 2` bytes long; anything past that is
/// left untouched. A short `dst` cannot cause UB — it panics.
// Every index/slice below is either `[0..7]` into a `chunks_exact(8)`/
// `chunks_exact_mut(16)` window (exactly that length by definition) or a
// `usize::from(u8)` index into a 256-entry table (always in range) or a
// prefix `dst[..len * 2]`, in bounds by the `dst.len() >= input.len() * 2`
// contract documented above.
#[allow(clippy::indexing_slicing)]
#[inline]
pub(crate) fn encode_slice(config: Config, input: &[u8], dst: &mut [u8]) {
    let len = input.len();

    if len == 0 {
        return;
    }

    // The only thing the alphabet choice changes is how far past '9' a letter
    // sits: 'a' - '0' - 10 == 39, 'A' - '0' - 10 == 7.
    let pairs: &[u16; 256] = if config.uppercase {
        &UPPER_PAIRS
    } else {
        &LOWER_PAIRS
    };

    let len_aligned = len & !7;
    let (body_in, tail_in) = input.split_at(len_aligned);
    let (body_out, tail_out) = dst[..len * 2].split_at_mut(len_aligned * 2);

    // Main loop: 8 input bytes -> 16 output chars, one table load per byte.
    for (src, out) in body_in.chunks_exact(8).zip(body_out.chunks_exact_mut(16)) {
        let lo = u64::from(pairs[usize::from(src[0])])
            | u64::from(pairs[usize::from(src[1])]) << 16
            | u64::from(pairs[usize::from(src[2])]) << 32
            | u64::from(pairs[usize::from(src[3])]) << 48;
        let hi = u64::from(pairs[usize::from(src[4])])
            | u64::from(pairs[usize::from(src[5])]) << 16
            | u64::from(pairs[usize::from(src[6])]) << 32
            | u64::from(pairs[usize::from(src[7])]) << 48;

        out[..8].copy_from_slice(&lo.to_le_bytes());
        out[8..].copy_from_slice(&hi.to_le_bytes());
    }

    // Tail handling (0-7 remaining bytes)
    for (i, &b) in tail_in.iter().enumerate() {
        let pair = pairs[usize::from(b)];
        let [lo, hi] = pair.to_le_bytes();
        tail_out[2 * i] = lo;
        tail_out[2 * i + 1] = hi;
    }
}

// --- DECODING ---

/// Decodes 8 hex characters (little-endian packed) into the 4 bytes they
/// represent, or `None` if any of the 8 is not a hex digit.
///
/// Validation is a re-encode: map each character to the nibble it *would*
/// decode to, encode that nibble back, and compare against the (case-folded)
/// input. Hex encoding is injective, so anything that isn't a hex digit fails
/// to round-trip — which replaces three SWAR range checks with one comparison.
#[inline]
const fn decode_u64(chars: u64) -> (u32, u64) {
    // Bit 6 is set for `A-Z`/`a-z` (and for `@`-ish neighbours, which the
    // round-trip check below rejects), clear for `0-9`.
    let letters = (chars >> 6) & ONES;

    // `'a' & 0x0F == 1`, so letters need +9 to land on 10..=15.
    let nibbles = (chars & 0x0F0F_0F0F_0F0F_0F0F).wrapping_add(letters.wrapping_mul(9));

    // Fold `A-F` onto `a-f` so one lowercase re-encode covers both cases.
    let folded = chars | letters.wrapping_mul(0x20);

    // Two ways to fail, tested as one branch:
    //   * a nibble overflowed its 4 bits (`'g'..='o'` land on 16..=24, and
    //     16 re-encodes to `'g'` again — so the round-trip alone would accept
    //     them);
    //   * the round-trip disagrees with the input, which catches everything
    //     else, including bytes with the high bit set.
    let bad = (nibbles & 0xF0F0_F0F0_F0F0_F0F0) | (nibbles_to_ascii(nibbles, LOWER_BIAS) ^ folded);

    // Fuse each 16-bit lane's two nibble bytes into one output byte:
    // `(c0 << 4) | c1`, then gather the four lanes into a `u32`.
    let fused = (nibbles << 4) | (nibbles >> 8);
    let mut packed = fused & 0x00FF_00FF_00FF_00FF;
    packed = (packed | (packed >> 8)) & 0x0000_FFFF_0000_FFFF;
    packed = (packed | (packed >> 16)) & 0x0000_0000_FFFF_FFFF;

    // `packed` is masked to the low 32 bits above, so this truncation is lossless.
    let [b0, b1, b2, b3, ..] = packed.to_le_bytes();

    (u32::from_le_bytes([b0, b1, b2, b3]), bad)
}

/// Reads 8 characters starting at `src[off]`.
///
/// # Panics
/// Panics if `src` has fewer than `off + 8` bytes; every caller passes an
/// 8-byte-aligned window from a `chunks_exact(8)`/`chunks_exact(32)` split.
#[allow(clippy::indexing_slicing)]
#[inline]
fn load_u64(src: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        src[off],
        src[off + 1],
        src[off + 2],
        src[off + 3],
        src[off + 4],
        src[off + 5],
        src[off + 6],
        src[off + 7],
    ])
}

/// Decodes hex `input` into `dst`.
///
/// `dst` must be at least `input.len() / 2` bytes long. On `Err` the contents
/// of `dst` are unspecified (the kernel may have written whole groups before
/// reaching the invalid character), but always in-bounds.
// As in `encode_slice`: every index/slice is a `chunks_exact` window sized to
// exactly what it's indexed with, a `usize::from(u8)` index into a 256-entry
// table, or a prefix in bounds by the `dst.len() >= input.len() / 2` contract
// documented above.
#[allow(clippy::indexing_slicing)]
#[inline]
pub(crate) fn decode_slice(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    let len = input.len();

    if len == 0 {
        return Ok(());
    }
    if !len.is_multiple_of(2) {
        return Err(Error::InvalidLength);
    }

    let len_aligned = len & !31;
    let (body_in, rest_in) = input.split_at(len_aligned);
    let (body_out, rest_out) = dst[..len / 2].split_at_mut(len_aligned / 2);

    // Main loop: 32 input chars -> 16 output bytes as four independent chains.
    // Each chain is long (~17 cycles of dependent arithmetic) but only ~5
    // cycles of port pressure, so the win here is overlap, not fewer uops.
    for (src, out) in body_in.chunks_exact(32).zip(body_out.chunks_exact_mut(16)) {
        let (g0, bad0) = decode_u64(load_u64(src, 0));
        let (g1, bad1) = decode_u64(load_u64(src, 8));
        let (g2, bad2) = decode_u64(load_u64(src, 16));
        let (g3, bad3) = decode_u64(load_u64(src, 24));

        if bad0 | bad1 | bad2 | bad3 != 0 {
            return Err(Error::InvalidCharacter);
        }

        out[0..4].copy_from_slice(&g0.to_le_bytes());
        out[4..8].copy_from_slice(&g1.to_le_bytes());
        out[8..12].copy_from_slice(&g2.to_le_bytes());
        out[12..16].copy_from_slice(&g3.to_le_bytes());
    }

    // Up to three 8-character groups may be left over.
    let mid_len = rest_in.len() & !7;
    let (mid_in, tail_in) = rest_in.split_at(mid_len);
    let (mid_out, tail_out) = rest_out.split_at_mut(mid_len / 2);

    for (src, out) in mid_in.chunks_exact(8).zip(mid_out.chunks_exact_mut(4)) {
        let (word, bad) = decode_u64(load_u64(src, 0));

        if bad != 0 {
            return Err(Error::InvalidCharacter);
        }

        out.copy_from_slice(&word.to_le_bytes());
    }

    // Tail handling (remaining even number of chars < 8)
    for (src, out) in tail_in.chunks_exact(2).zip(tail_out.iter_mut()) {
        let high = HEX_DECODE_TABLE[usize::from(src[0])];
        let low = HEX_DECODE_TABLE[usize::from(src[1])];

        if (high | low) & 0xF0 != 0 {
            return Err(Error::InvalidCharacter);
        }

        *out = (high << 4) | low;
    }

    Ok(())
}

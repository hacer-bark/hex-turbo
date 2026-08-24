use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _MM_HINT_T0, _mm_loadu_si128, _mm_packus_epi16, _mm_prefetch,
    _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16,
    _mm256_min_epu8, _mm256_movemask_epi8, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _MM_HINT_T0, _mm_loadu_si128, _mm_packus_epi16, _mm_prefetch,
    _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16,
    _mm256_min_epu8, _mm256_movemask_epi8, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

// `_mm256_stream_si256`/`_mm_sfence` lower to inline `asm!`, which Miri never
// executes -- imported only for the real-hardware path; the Miri build routes
// around them entirely (see `stream_store`/`sfence` below).
#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{_mm_sfence, _mm256_stream_si256};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{_mm_sfence, _mm256_stream_si256};

// --- CONSTANTS ---

// Duplicated 16-byte tables for AVX2 pshufb (Encoding)
const HEX_TABLE_UPPER: [u8; 32] = *b"0123456789ABCDEF0123456789ABCDEF";

const HEX_TABLE_LOWER: [u8; 32] = *b"0123456789abcdef0123456789abcdef";

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

// --- MEMORY-HIERARCHY THRESHOLDS ---
//
// Both kernels run at their compute floor only while their working set is in
// L1. Past that the loops are waiting on memory, and the two constants below
// decide when it is worth spending instructions to do something about it.
// Measured on Coffee Lake (i7-8750H, 32 KiB L1d, 256 KiB L2, 9 MiB L3) against
// a cold cache, which is the state a freshly allocated output buffer is in.

/// Input length at which the loops start issuing prefetches.
///
/// The hardware prefetcher handles a single sequential stream well but is
/// slower to cover two streams that advance at different rates, which is what
/// both kernels are. Prefetching the load stream is worth ~12% from 32 KiB
/// through 1 MiB; below that the two extra uops per block are pure loss (~3%
/// at 4 KiB), so the whole apparatus sits behind one length test.
#[cfg(not(miri))]
const PREFETCH_MIN: usize = 32 * 1024;
#[cfg(miri)]
const PREFETCH_MIN: usize = 200;

/// How far ahead of the load stream to prefetch, in bytes.
///
/// 512 measured better than 256 everywhere it mattered, and beyond that the
/// distance starts running past the page the loop is on.
#[cfg(not(miri))]
const PREFETCH_DIST: usize = 512;
#[cfg(miri)]
const PREFETCH_DIST: usize = 8;

/// Input length at which the encoder switches to non-temporal stores.
///
/// An ordinary store reads the line it is about to completely overwrite
/// (read-for-ownership). The encoder writes twice what it reads, so once the
/// output no longer fits in cache that read is roughly half of all the memory
/// traffic in the kernel; `vmovntdq` skips it, and is worth ~43% at 4 MiB and
/// above. Below ~2 MiB the output still lives in L3 and streaming it out to
/// DRAM instead is a large *loss*, so the threshold is deliberately
/// conservative.
///
/// The decoder writes only half what it reads and gets no measurable benefit
/// from streaming at any size, so it does not do this.
#[cfg(not(miri))]
const NONTEMPORAL_MIN: usize = 2 * 1024 * 1024;
#[cfg(miri)]
const NONTEMPORAL_MIN: usize = 300;

// --- SCALAR TAILS ---
//
// Out of line, and marked cold, for a reason that has nothing to do with the
// tail itself: inlined, the scalar kernel's register demand is what forces
// every call -- including the ones that never reach a tail -- to push and pop
// five callee-saved registers. That is ten uops a 32-byte payload cannot
// amortize, and hoisting it out is worth ~10% on short inputs.

#[cold]
#[inline(never)]
fn encode_tail(config: Config, src: &[u8], dst: &mut [u8]) {
    scalar::encode_slice(config, src, dst);
}

#[cold]
#[inline(never)]
fn decode_tail(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    scalar::decode_slice(src, dst)
}

// --- STREAMING STORE ---
//
// `_mm256_stream_si256` lowers to inline `asm!`, which Miri categorically
// refuses to execute (not a missing-intrinsic gap, an unconditional "unsupported
// operation" abort) -- so under Miri it is routed through an ordinary store
// instead, the same way `zmm_multishift_epi64_epi8` routes around the one VBMI
// instruction Miri can't run. This is exact, not an approximation: non-temporal
// is a caching hint that changes write-combining behaviour, not the bytes
// written or their visibility order to a single-threaded reader, so the two
// are interchangeable for anything but a benchmark. `_mm_sfence` orders
// non-temporal stores against later loads; with no non-temporal stores under
// Miri, there is nothing for it to order, so it becomes a no-op there too.

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn stream_store(dst: *mut __m256i, val: __m256i) {
    #[cfg(miri)]
    unsafe {
        _mm256_storeu_si256(dst, val);
    }
    #[cfg(not(miri))]
    unsafe {
        _mm256_stream_si256(dst, val);
    }
}

#[inline]
fn sfence() {
    #[cfg(not(miri))]
    unsafe {
        _mm_sfence();
    }
}

// --- ENCODING ---

/// 16 input bytes -> 32 output bytes with a single `vpshufb`.
///
/// `vpmovzxbw` widens each input byte to a 16-bit lane, which is exactly the
/// span of its two output characters. Shifting the lane both ways puts the
/// high nibble in the low half and the low nibble in the high half, so one
/// table lookup emits both characters already in output order and the store
/// needs no lane fixup at all: two port-5 uops per 16 bytes, against the six
/// per 32 bytes that a lookup-then-interleave costs.
macro_rules! encode16 {
    ($table:expr, $mask:expr, $src:expr, $dst:expr, $store:ident) => {{
        // Lane i = 0x00_bb for input byte i.
        let w = _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128($src.cast::<__m128i>()) });
        // `vpshufb` ignores bits 4-6 of an index but zeroes the output byte on
        // bit 7, so both nibbles are masked clean.
        let idx = _mm256_and_si256(
            _mm256_or_si256(_mm256_slli_epi16(w, 8), _mm256_srli_epi16(w, 4)),
            $mask,
        );
        unsafe { $store($dst.cast::<__m256i>(), _mm256_shuffle_epi8($table, idx)) };
    }};
}

/// One 128-byte block: eight `encode16` steps, optionally preceded by a
/// prefetch of the two lines the loop will reach in `PREFETCH_DIST` bytes.
macro_rules! encode_block {
    ($table:expr, $mask:expr, $p:expr, $dst:expr, $store:ident, $prefetch:expr) => {{
        let p = $p;
        if $prefetch {
            unsafe {
                _mm_prefetch(p.add(PREFETCH_DIST).cast::<i8>(), _MM_HINT_T0);
                _mm_prefetch(p.add(PREFETCH_DIST + 64).cast::<i8>(), _MM_HINT_T0);
            }
        }
        encode16!($table, $mask, p, $dst, $store);
        encode16!($table, $mask, p.add(16), $dst.add(32), $store);
        encode16!($table, $mask, p.add(32), $dst.add(64), $store);
        encode16!($table, $mask, p.add(48), $dst.add(96), $store);
        encode16!($table, $mask, p.add(64), $dst.add(128), $store);
        encode16!($table, $mask, p.add(80), $dst.add(160), $store);
        encode16!($table, $mask, p.add(96), $dst.add(192), $store);
        encode16!($table, $mask, p.add(112), $dst.add(224), $store);
    }};
}

/// 32 input bytes -> 64 output bytes.
///
/// Both nibble planes are looked up across a full vector and then woven back
/// together. That costs three more port-5 uops per 32 bytes than `encode16`
/// does, but half as many uops overall, which is the better trade for the
/// dozens-of-bytes remainder this step exists to serve.
macro_rules! encode32 {
    ($table:expr, $mask_0f:expr, $src:expr, $dst:expr) => {{
        let v = unsafe { _mm256_loadu_si256($src.cast::<__m256i>()) };

        let low_chars = _mm256_shuffle_epi8($table, _mm256_and_si256(v, $mask_0f));
        let high_chars =
            _mm256_shuffle_epi8($table, _mm256_and_si256(_mm256_srli_epi16(v, 4), $mask_0f));

        // Interleave (high, low) per byte, then repair the 128-bit lane split
        // that `vpunpck` leaves behind.
        let inter_lo = _mm256_unpacklo_epi8(high_chars, low_chars);
        let inter_hi = _mm256_unpackhi_epi8(high_chars, low_chars);

        unsafe {
            _mm256_storeu_si256(
                $dst.cast::<__m256i>(),
                _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20),
            );
            _mm256_storeu_si256(
                $dst.add(32).cast::<__m256i>(),
                _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31),
            );
        }
    }};
}

/// Encodes `input` as hex into `dst_slice`.
///
/// # Safety
/// `dst_slice` must hold at least `input.len() * 2` bytes.
// Every `src[N..]` here re-slices past a `while src.len() >= N` guard that
// just ran, and `dst_slice[dst_off..]` uses `dst_off`, which the SAFETY
// comment at its call site proves stays within `dst_slice`.
#[allow(clippy::indexing_slicing)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_slice_avx2(config: Config, input: &[u8], dst_slice: &mut [u8]) {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let table_ptr = if config.uppercase {
        HEX_TABLE_UPPER.as_ptr()
    } else {
        HEX_TABLE_LOWER.as_ptr()
    };
    let table = unsafe { _mm256_loadu_si256(table_ptr.cast::<__m256i>()) };
    let nibble_mask = _mm256_set1_epi16(0x0F0F);
    let mask_0f = _mm256_set1_epi8(0x0F);

    let mut src = input;

    // Everything a small input has no use for sits behind the one test it has
    // to make anyway. An input with no whole block in it reaches the 32-byte
    // loop after a single compare, exactly as it did before any of the
    // large-input machinery existed.
    if src.len() >= 128 {
        if src.len() >= PREFETCH_MIN {
            // `vmovntdq` faults on an unaligned address, and every 16-byte step
            // advances `dst` by exactly 32 -- so the alignment of `dst` never
            // changes once the loop is running and has to be fixed up front, by
            // encoding however many bytes it takes to reach a boundary. An odd
            // `dst` can never reach one (every step moves an even distance), so
            // that case simply keeps the ordinary stores.
            if src.len() >= NONTEMPORAL_MIN && dst.addr() & 1 == 0 {
                let head = ((32 - (dst.addr() & 31)) & 31) / 2;
                if head > 0 {
                    // Through the cold tail, not `scalar::encode_slice`
                    // directly: inlining the scalar kernel here would put its
                    // register demand back on the hot path and bring the five
                    // callee-saved pushes back with it.
                    //
                    // SAFETY: `dst` has `dst_slice.len() - (dst.offset_from(dst_start))`
                    // bytes left, which is `>= head * 2` since `head <= 15` and
                    // the `src.len() >= 128` guard above guarantees at least
                    // 256 bytes of destination remain.
                    //
                    // Built from the raw pointer rather than `&mut dst_slice[..head * 2]`:
                    // slice indexing reborrows through `&mut *dst_slice` first,
                    // which (on the Stacked Borrows model Miri checks) would
                    // invalidate `dst` for the whole buffer -- including the
                    // disjoint remainder this function keeps writing through
                    // `dst` after this call returns -- not just the `head * 2`
                    // bytes actually handed to `encode_tail`.
                    let head_dst = unsafe { core::slice::from_raw_parts_mut(dst, head * 2) };
                    encode_tail(config, &src[..head], head_dst);
                    src = &src[head..];
                    dst = unsafe { dst.add(head * 2) };
                }

                while src.len() >= 128 + PREFETCH_DIST + 64 {
                    encode_block!(table, nibble_mask, src.as_ptr(), dst, stream_store, true);
                    src = &src[128..];
                    dst = unsafe { dst.add(256) };
                }
                while src.len() >= 128 {
                    encode_block!(table, nibble_mask, src.as_ptr(), dst, stream_store, false);
                    src = &src[128..];
                    dst = unsafe { dst.add(256) };
                }
                // Streaming stores are only ordered against a fence.
                sfence();
            } else {
                while src.len() >= 128 + PREFETCH_DIST + 64 {
                    encode_block!(
                        table,
                        nibble_mask,
                        src.as_ptr(),
                        dst,
                        _mm256_storeu_si256,
                        true
                    );
                    src = &src[128..];
                    dst = unsafe { dst.add(256) };
                }
            }
        }

        while src.len() >= 128 {
            encode_block!(
                table,
                nibble_mask,
                src.as_ptr(),
                dst,
                _mm256_storeu_si256,
                false
            );
            src = &src[128..];
            dst = unsafe { dst.add(256) };
        }
    }

    while src.len() >= 32 {
        encode32!(table, mask_0f, src.as_ptr(), dst);
        src = &src[32..];
        dst = unsafe { dst.add(64) };
    }

    // Without this step a remainder of 16..31 bytes would go to the scalar
    // kernel at roughly eight times the cost per byte -- which is half of all
    // possible lengths, and was worth 2x at 48 bytes.
    if src.len() >= 16 {
        encode16!(table, nibble_mask, src.as_ptr(), dst, _mm256_storeu_si256);
        src = &src[16..];
        dst = unsafe { dst.add(32) };
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        encode_tail(config, src, &mut dst_slice[dst_off..]);
    }
}

// --- DECODING ---

/// 32 hex characters -> 16 packed bytes, plus a per-character validity mask.
///
/// `LUT_HI` holds 128 for the digit row and 73 for the two letter rows, and
/// `LUT_LO` holds a mask per column; a character is valid exactly when the two
/// share a set bit. The same `LUT_HI` byte doubles as the nibble bias, because
/// 73 contributes 9 below bit 4 and 128 contributes nothing.
///
/// Masking the *sum* rather than the table output is what makes that second
/// role free -- and it is also why the shape matters: `(lo + hi_props) & 0x0F`
/// is provably at most 15, so LLVM can see that `vpmaddubsw` cannot exceed 255
/// and stops emitting a `vpminsw` clamp ahead of every `vpackuswb`. Those
/// clamps were four wasted uops per 128 characters, about 9% of the loop.
macro_rules! decode32 {
    ($lut_hi:expr, $lut_lo:expr, $weights:expr, $mask_0f:expr, $p:expr, $off:expr) => {{
        let v = unsafe { _mm256_loadu_si256($p.add($off).cast::<__m256i>()) };

        let lo = _mm256_and_si256(v, $mask_0f);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), $mask_0f);

        let hi_props = _mm256_shuffle_epi8($lut_hi, hi);
        let lo_props = _mm256_shuffle_epi8($lut_lo, lo);

        let nibbles = _mm256_and_si256(_mm256_add_epi8(lo, hi_props), $mask_0f);

        (
            _mm256_maddubs_epi16(nibbles, $weights),
            _mm256_and_si256(hi_props, lo_props),
        )
    }};
}

/// Decodes hex `input` into `dst_slice`.
///
/// # Safety
/// `dst_slice` must hold at least `input.len() / 2` bytes.
// Same reasoning as `encode_slice_avx2`: every `src[N..]` follows a
// `while src.len() >= N` guard, and `dst_slice[dst_off..]` is proven in
// bounds by the SAFETY comment at its call site.
#[allow(clippy::indexing_slicing)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn decode_slice_avx2(input: &[u8], dst_slice: &mut [u8]) -> Result<(), Error> {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let lut_hi = unsafe { _mm256_loadu_si256(LUT_HI.as_ptr().cast::<__m256i>()) };
    let lut_lo = unsafe { _mm256_loadu_si256(LUT_LO.as_ptr().cast::<__m256i>()) };
    let weights = unsafe { _mm256_loadu_si256(WEIGHTS.as_ptr().cast::<__m256i>()) };

    let mask_0f = _mm256_set1_epi8(0x0F);

    // Validity is folded across the whole input and inspected once, after the
    // loops. `vpminub` keeps the weakest byte seen, so a single zero anywhere
    // survives to the end -- which removes a `vpcmpeqb`/`vpmovmskb`/branch
    // triple from the middle of every iteration.
    let mut valid_acc = _mm256_set1_epi8(-1);

    /// 64 characters -> 32 bytes. `vpackuswb` works within 128-bit lanes, so
    /// the halves come out interleaved and one `vpermq` puts them back.
    macro_rules! pair {
        ($p:expr, $off:expr, $dst_off:expr) => {{
            let (r0, a0) = decode32!(lut_hi, lut_lo, weights, mask_0f, $p, $off);
            let (r1, a1) = decode32!(lut_hi, lut_lo, weights, mask_0f, $p, $off + 32);
            valid_acc = _mm256_min_epu8(valid_acc, _mm256_min_epu8(a0, a1));
            unsafe {
                _mm256_storeu_si256(
                    dst.add($dst_off).cast::<__m256i>(),
                    _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8),
                );
            }
        }};
    }

    let mut src = input;

    // Nested, for the same reason as in the encoder: a short input should not
    // have to test a threshold it can never meet.
    if src.len() >= 128 {
        if src.len() >= PREFETCH_MIN {
            while src.len() >= 128 + PREFETCH_DIST + 64 {
                let p = src.as_ptr();
                unsafe {
                    _mm_prefetch(p.add(PREFETCH_DIST).cast::<i8>(), _MM_HINT_T0);
                    _mm_prefetch(p.add(PREFETCH_DIST + 64).cast::<i8>(), _MM_HINT_T0);
                }
                pair!(p, 0, 0);
                pair!(p, 64, 32);
                src = &src[128..];
                dst = unsafe { dst.add(64) };
            }
        }

        while src.len() >= 128 {
            let p = src.as_ptr();
            pair!(p, 0, 0);
            pair!(p, 64, 32);
            src = &src[128..];
            dst = unsafe { dst.add(64) };
        }
    }

    while src.len() >= 32 {
        let (pairs, valid) = decode32!(lut_hi, lut_lo, weights, mask_0f, src.as_ptr(), 0);
        valid_acc = _mm256_min_epu8(valid_acc, valid);

        let res = _mm_packus_epi16(
            _mm256_castsi256_si128(pairs),
            _mm256_extracti128_si256(pairs, 1),
        );
        unsafe { _mm_storeu_si128(dst.cast::<__m128i>(), res) };

        src = &src[32..];
        dst = unsafe { dst.add(16) };
    }

    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(valid_acc, _mm256_setzero_si256())) != 0 {
        return Err(Error::InvalidCharacter);
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        decode_tail(src, &mut dst_slice[dst_off..])?;
    }

    Ok(())
}

// Verification: the Miri coverage suite.
#[cfg(test)]
mod verify;

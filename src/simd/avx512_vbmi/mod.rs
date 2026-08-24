use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256i, __m512i, _MM_HINT_T0, _mm_prefetch, _mm256_loadu_si256, _mm256_storeu_si256,
    _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16, _mm512_loadu_si512, _mm512_maddubs_epi16,
    _mm512_movepi8_mask, _mm512_or_si512, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_storeu_si512,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, __m512i, _MM_HINT_T0, _mm_prefetch, _mm256_loadu_si256, _mm256_storeu_si256,
    _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16, _mm512_loadu_si512, _mm512_maddubs_epi16,
    _mm512_movepi8_mask, _mm512_or_si512, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_storeu_si512,
};

// `_mm256_stream_si256`/`_mm_sfence` lower to inline `asm!`, which Miri never
// executes -- imported only for the real-hardware path; the Miri build routes
// around them entirely (see `stream_store`/`sfence` below).
#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{_mm_sfence, _mm256_stream_si256};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{_mm_sfence, _mm256_stream_si256};

// `vpmultishiftqb` is the one VBMI instruction Miri's x86 intrinsic
// interpreter doesn't implement yet; `zmm_multishift_epi64_epi8` below routes
// around it under `cfg(miri)` with a software model instead. The other two
// VBMI permutes Miri does support natively.
#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::_mm512_multishift_epi64_epi8;
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::_mm512_multishift_epi64_epi8;

// --- CONSTANTS ---

// The hex alphabet, repeated four times. `vpermb` reads only bits 5:0 of an
// index, so a table that repeats every 16 bytes turns any byte whose low
// nibble is the value into a correct lookup -- bits 5:4 are free to hold
// garbage. That is what lets the encoder skip masking its nibbles entirely.
const HEX_TABLE_UPPER: [u8; 64] =
    *b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";

const HEX_TABLE_LOWER: [u8; 64] =
    *b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// `vpmultishiftqb` control for the encoder.
///
/// Each source qword is the `vpmovzxbw` image of four input bytes,
/// `[b0, 0, b1, 0, b2, 0, b3, 0]`. `vpmultishiftqb` emits eight bytes per
/// qword, each an arbitrary 8-bit window into it, so one instruction produces
/// all eight characters of those four bytes: the window at bit `16*i + 4`
/// carries the high nibble of `b_i` (the zero byte above it supplies the zero
/// padding), and the window at bit `16*i` carries `b_i` itself, whose high
/// nibble the replicated table above then ignores.
const MULTISHIFT_CTL: [u8; 64] = [
    4, 0, 20, 16, 36, 32, 52, 48, 4, 0, 20, 16, 36, 32, 52, 48, 4, 0, 20, 16, 36, 32, 52, 48, 4, 0,
    20, 16, 36, 32, 52, 48, 4, 0, 20, 16, 36, 32, 52, 48, 4, 0, 20, 16, 36, 32, 52, 48, 4, 0, 20,
    16, 36, 32, 52, 48, 4, 0, 20, 16, 36, 32, 52, 48,
];

/// ASCII -> nibble value for the decoder, with `0xFF` for everything that is
/// not a hex digit.
///
/// `vpermi2b` indexes a 128-byte table with bits 6:0 of each byte, which
/// covers all of ASCII in a single instruction -- so the whole two-LUT
/// AND-combine that the AVX2 kernel needs to validate a character collapses
/// into the same lookup that produces its value. Bit 7 is *ignored* by the
/// permute, so a byte >= 0x80 aliases onto a low-half entry and can come back
/// looking valid; the kernel catches those separately, from the input itself.
const DECODE_TABLE: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 255, 255, 255,
    255, 255, 255, 255, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 10, 11, 12, 13,
    14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
];

/// `vpmaddubsw` weights: each character pair folds to `hi * 16 + lo * 1`.
const WEIGHTS: [u8; 64] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16,
    1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
];

// --- MEMORY-HIERARCHY THRESHOLDS ---
//
// Measured on Zen 5 (EPYC 9R45, 48 KiB L1d, 1 MiB L2, 8 MiB L3) against a cold
// cache. The two kernels want *opposite* treatment here, and the AVX2 kernel
// on Coffee Lake wanted the opposite of both, so neither of these numbers
// should be assumed to travel.

/// Input length at which the encoder starts issuing prefetches.
///
/// Worth ~9% at 1 MiB and above, where the loop is waiting on DRAM. Below it
/// the two extra uops per block are loss, and unlike the AVX2 kernel the
/// crossover here is high: the encoder's own store stream is dense enough that
/// the hardware prefetcher keeps up on its own until the working set leaves
/// L2. The decoder measured no benefit from prefetching at any size.
#[cfg(not(miri))]
const PREFETCH_MIN: usize = 512 * 1024;
#[cfg(miri)]
const PREFETCH_MIN: usize = 200;

/// How far ahead of the load stream to prefetch, in bytes.
#[cfg(not(miri))]
const PREFETCH_DIST: usize = 512;
#[cfg(miri)]
const PREFETCH_DIST: usize = 8;

/// Input length at which the decoder switches to non-temporal stores.
///
/// Worth ~15% at 1 MiB and ~16% at 4 MiB. Below roughly this size the output
/// still lives in L2/L3 and streaming it to DRAM instead is a clear loss
/// (measured +1.5% at a 256 KiB output), so the threshold is set where the
/// output stops fitting.
///
/// The *encoder* does not do this, which is the reverse of the AVX2 kernel:
/// on Zen 5 non-temporal stores made encoding 33-50% slower at every size
/// tried. Writing two bytes per byte read saturates the write-combining
/// buffers, and once they are thrashing every partial line goes out
/// separately, which costs far more than the read-for-ownership it saves.
#[cfg(not(miri))]
const NONTEMPORAL_MIN: usize = 1024 * 1024;
#[cfg(miri)]
const NONTEMPORAL_MIN: usize = 300;

// Out of line for the same reason as in the AVX2 kernel: inlined, the scalar
// tail's register demand forces every call -- including the ones that never
// reach a tail -- to push and pop callee-saved registers.
//
// They carry the feature set themselves, which matters more than it looks: a
// plain `fn` is compiled for the baseline target, so the scalar loop inside it
// gets SSE2 and nothing else, while the same loop inlined into the kernel used
// to be auto-vectorized with the full AVX-512 register file. Dropping it to
// baseline cost 35-100% on every input whose length is not a whole number of
// vector steps -- which is most lengths.
//
// They are also deliberately not `#[cold]`: that would additionally ask LLVM
// to optimize them for size.

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn encode_tail(config: Config, src: &[u8], dst: &mut [u8]) {
    scalar::encode_slice(config, src, dst);
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn decode_tail(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    scalar::decode_slice(src, dst)
}

#[cfg(miri)]
use self::verify::intrinsic_models as m;

/// `vpmultishiftqb`, routed through a software model under Miri, which cannot
/// execute the real instruction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_multishift_epi64_epi8(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(miri)]
    {
        unsafe { m::multishift_epi64_epi8_model(a, b) }
    }
    #[cfg(not(miri))]
    {
        _mm512_multishift_epi64_epi8(a, b)
    }
}

/// `_mm256_stream_si256`, routed through an ordinary store under Miri, whose
/// interpreter refuses to run the underlying `vmovntdq` (it lowers to inline
/// `asm!`, which Miri never executes) -- same reasoning as
/// `zmm_multishift_epi64_epi8` above. Exact, not an approximation: streaming is
/// a caching hint, not a value or ordering difference for a single-threaded
/// reader. `_mm_sfence` orders non-temporal stores against later loads; with
/// none of those under Miri, it becomes a no-op there too.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
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

/// 32 input bytes -> 64 output bytes in three vector uops.
///
/// `vpmovzxbw` widens each byte to the 16-bit lane its two characters will
/// occupy, `vpmultishiftqb` cuts both nibbles out of that lane and lands them
/// already in output order, and `vpermb` turns them into ASCII. No masking, no
/// lane fixup, and nothing to reassemble before the store -- against the
/// twelve uops per 64 bytes that lookup-then-interleave-then-permute costs.
macro_rules! encode32 {
    ($table:expr, $ctl:expr, $src:expr, $dst:expr, $store:ident) => {{
        let widened = _mm512_cvtepu8_epi16(unsafe { _mm256_loadu_si256($src.cast::<__m256i>()) });
        let nibbles = unsafe { zmm_multishift_epi64_epi8($ctl, widened) };
        unsafe { $store($dst.cast::<_>(), _mm512_permutexvar_epi8(nibbles, $table)) };
    }};
}

/// One 128-byte block, optionally preceded by a prefetch of the two lines the
/// loop will reach in `PREFETCH_DIST` bytes.
macro_rules! encode_block {
    ($table:expr, $ctl:expr, $p:expr, $dst:expr, $prefetch:expr) => {{
        let p = $p;
        if $prefetch {
            unsafe {
                _mm_prefetch(p.add(PREFETCH_DIST).cast::<i8>(), _MM_HINT_T0);
                _mm_prefetch(p.add(PREFETCH_DIST + 64).cast::<i8>(), _MM_HINT_T0);
            }
        }
        encode32!($table, $ctl, p, $dst, _mm512_storeu_si512);
        encode32!($table, $ctl, p.add(32), $dst.add(64), _mm512_storeu_si512);
        encode32!($table, $ctl, p.add(64), $dst.add(128), _mm512_storeu_si512);
        encode32!($table, $ctl, p.add(96), $dst.add(192), _mm512_storeu_si512);
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
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn encode_slice_avx512_vbmi(config: Config, input: &[u8], dst_slice: &mut [u8]) {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let table_ptr = if config.uppercase {
        HEX_TABLE_UPPER.as_ptr()
    } else {
        HEX_TABLE_LOWER.as_ptr()
    };
    let table = unsafe { _mm512_loadu_si512(table_ptr.cast::<__m512i>()) };
    let ctl = unsafe { _mm512_loadu_si512(MULTISHIFT_CTL.as_ptr().cast::<__m512i>()) };

    let mut src = input;

    // Everything a small input has no use for sits behind the one test it has
    // to make anyway, so an input with no whole block in it reaches the
    // 32-byte loop after a single compare.
    if src.len() >= 128 {
        if src.len() >= PREFETCH_MIN {
            while src.len() >= 128 + PREFETCH_DIST + 64 {
                encode_block!(table, ctl, src.as_ptr(), dst, true);
                src = &src[128..];
                dst = unsafe { dst.add(256) };
            }
        }

        while src.len() >= 128 {
            encode_block!(table, ctl, src.as_ptr(), dst, false);
            src = &src[128..];
            dst = unsafe { dst.add(256) };
        }
    }

    while src.len() >= 32 {
        encode32!(table, ctl, src.as_ptr(), dst, _mm512_storeu_si512);
        src = &src[32..];
        dst = unsafe { dst.add(64) };
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        encode_tail(config, src, &mut dst_slice[dst_off..]);
    }
}

// --- DECODING ---

/// 64 hex characters -> 32 bytes, plus the evidence needed to reject them.
///
/// One `vpermi2b` over the 128-entry table yields each character's value, or
/// `0xFF` if it is not a hex digit. OR-ing that against the raw input folds
/// both failure modes into one sign bit per byte: `0xFF` from the table, and
/// any input byte >= 0x80, which the permute's 7-bit index would otherwise
/// have aliased onto a valid-looking entry.
macro_rules! decode64 {
    ($tbl_lo:expr, $tbl_hi:expr, $weights:expr, $src:expr) => {{
        let chars = unsafe { _mm512_loadu_si512($src.cast::<__m512i>()) };
        let values = _mm512_permutex2var_epi8($tbl_lo, chars, $tbl_hi);
        (
            _mm512_maddubs_epi16(values, $weights),
            _mm512_or_si512(values, chars),
        )
    }};
}

/// Decodes hex `input` into `dst_slice`.
///
/// # Safety
/// `dst_slice` must hold at least `input.len() / 2` bytes.
// Same reasoning as `encode_slice_avx512_vbmi`.
#[allow(clippy::indexing_slicing)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn decode_slice_avx512_vbmi(
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<(), Error> {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let tbl_lo = unsafe { _mm512_loadu_si512(DECODE_TABLE.as_ptr().cast::<__m512i>()) };
    let tbl_hi = unsafe { _mm512_loadu_si512(DECODE_TABLE.as_ptr().add(64).cast::<__m512i>()) };
    let weights = unsafe { _mm512_loadu_si512(WEIGHTS.as_ptr().cast::<__m512i>()) };

    let mut src = input;

    macro_rules! block256 {
        ($store:ident) => {{
            let (p0, s0) = decode64!(tbl_lo, tbl_hi, weights, src.as_ptr());
            let (p1, s1) = decode64!(tbl_lo, tbl_hi, weights, src.as_ptr().add(64));
            let (p2, s2) = decode64!(tbl_lo, tbl_hi, weights, src.as_ptr().add(128));
            let (p3, s3) = decode64!(tbl_lo, tbl_hi, weights, src.as_ptr().add(192));

            // One mask extraction for the whole block: the four evidence
            // vectors are folded with plain ORs, which LLVM contracts into
            // `vpternlog`.
            let bad = _mm512_or_si512(_mm512_or_si512(s0, s1), _mm512_or_si512(s2, s3));
            if _mm512_movepi8_mask(bad) != 0 {
                return Err(Error::InvalidCharacter);
            }

            unsafe {
                $store(dst.cast::<__m256i>(), _mm512_cvtepi16_epi8(p0));
                $store(dst.add(32).cast::<__m256i>(), _mm512_cvtepi16_epi8(p1));
                $store(dst.add(64).cast::<__m256i>(), _mm512_cvtepi16_epi8(p2));
                $store(dst.add(96).cast::<__m256i>(), _mm512_cvtepi16_epi8(p3));
            }
            src = &src[256..];
            dst = unsafe { dst.add(128) };
        }};
    }

    if src.len() >= 256 {
        if src.len() >= NONTEMPORAL_MIN {
            // `vmovntdq` faults on an unaligned address, and every step
            // advances `dst` by a multiple of 32 -- so its alignment never
            // changes once the loop is running and has to be fixed up front,
            // by decoding however many characters it takes to reach a
            // boundary. Two characters per output byte, hence the doubling.
            let head = ((32 - (dst.addr() & 31)) & 31) * 2;
            if head > 0 {
                // Through the cold tail rather than `scalar::decode_slice`
                // directly, so the scalar kernel's register demand stays off
                // the hot path.
                //
                // SAFETY: `dst` has `dst_slice.len() - (dst.offset_from(dst_start))`
                // bytes left, which is `>= head / 2` since `head <= 62` and the
                // `src.len() >= 256` guard above guarantees at least 128 bytes
                // of destination remain.
                //
                // Built from the raw pointer rather than `&mut dst_slice[..head / 2]`:
                // slice indexing reborrows through `&mut *dst_slice` first,
                // which (on the Stacked Borrows model Miri checks) would
                // invalidate `dst` for the whole buffer -- including the
                // disjoint remainder this function keeps writing through `dst`
                // after this call returns -- not just the `head / 2` bytes
                // actually handed to `decode_tail`.
                let head_dst = unsafe { core::slice::from_raw_parts_mut(dst, head / 2) };
                decode_tail(&src[..head], head_dst)?;
                src = &src[head..];
                dst = unsafe { dst.add(head / 2) };
            }

            while src.len() >= 256 {
                block256!(stream_store);
            }
            // Streaming stores are only ordered against a fence.
            sfence();
        }

        while src.len() >= 256 {
            block256!(_mm256_storeu_si256);
        }
    }

    while src.len() >= 64 {
        let (pairs, evidence) = decode64!(tbl_lo, tbl_hi, weights, src.as_ptr());
        if _mm512_movepi8_mask(evidence) != 0 {
            return Err(Error::InvalidCharacter);
        }
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), _mm512_cvtepi16_epi8(pairs)) };
        src = &src[64..];
        dst = unsafe { dst.add(32) };
    }

    if !src.is_empty() {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        decode_tail(src, &mut dst_slice[dst_off..])?;
    }

    Ok(())
}

// Verification: the Miri coverage suite. Needed under plain `cfg(miri)` too,
// not just `cfg(test)`: the `--lib` target Miri builds alongside `--tests`
// has no `test` cfg of its own, but still needs `intrinsic_models` for the
// `vpmultishiftqb` shim above.
#[cfg(any(test, miri))]
mod verify;

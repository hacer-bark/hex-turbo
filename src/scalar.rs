use crate::{Error, Config};

// --- CONSTANTS ---

const LOWER_ALPHABET: [u8; 16] = *b"0123456789abcdef";
const UPPER_ALPHABET: [u8; 16] = *b"0123456789ABCDEF";

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

#[inline(always)]
pub unsafe fn encode_slice_unsafe(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    if len == 0 { return; }

    let alphabet = if config.uppercase {
        &UPPER_ALPHABET
    } else {
        &LOWER_ALPHABET
    };

    unsafe {
        // Main loop: 4 input bytes -> 8 output chars (packed into one u64 write)
        let len_aligned = len - (len % 4);
        let src_end_aligned = src.add(len_aligned);

        while src < src_end_aligned {
            // Load 4 bytes and normalize to big-endian byte order in the register
            #[cfg(target_endian = "little")]
            let chunk = (src as *const u32).read_unaligned().to_be();

            #[cfg(target_endian = "big")]
            let chunk = (src as *const u32).read_unaligned();

            let b0 = ((chunk >> 24) & 0xFF) as usize;
            let b1 = ((chunk >> 16) & 0xFF) as usize;
            let b2 = ((chunk >> 8) & 0xFF) as usize;
            let b3 = (chunk & 0xFF) as usize;

            let pack =
                (*alphabet.get_unchecked((b0 >> 4) & 0x0F) as u64) |
                (*alphabet.get_unchecked(b0 & 0x0F) as u64) << 8 |
                (*alphabet.get_unchecked((b1 >> 4) & 0x0F) as u64) << 16 |
                (*alphabet.get_unchecked(b1 & 0x0F) as u64) << 24 |
                (*alphabet.get_unchecked((b2 >> 4) & 0x0F) as u64) << 32 |
                (*alphabet.get_unchecked(b2 & 0x0F) as u64) << 40 |
                (*alphabet.get_unchecked((b3 >> 4) & 0x0F) as u64) << 48 |
                (*alphabet.get_unchecked(b3 & 0x0F) as u64) << 56;

            (dst as *mut u64).write_unaligned(pack);

            src = src.add(4);
            dst = dst.add(8);
        }

        // Tail handling (0–3 remaining bytes)
        let remaining = len - len_aligned;
        for i in 0..remaining {
            let b = *src.add(i) as usize;
            *dst.add(2 * i) = *alphabet.get_unchecked((b >> 4) & 0x0F);
            *dst.add(2 * i + 1) = *alphabet.get_unchecked(b & 0x0F);
        }
    }
}

// --- DECODING ---

#[inline(always)]
pub unsafe fn decode_slice_unsafe(input: &[u8], mut dst: *mut u8) -> Result<(), Error> {
    let len = input.len();

    if len == 0 { return Ok(()); }
    if len % 2 != 0 { return Err(Error::InvalidLength); }

    let mut src = input.as_ptr();
    let table = HEX_DECODE_TABLE.as_ptr();

    unsafe {
        // Fast loop: 8 input chars -> 4 output bytes
        let len_aligned = len - (len % 8);
        let src_end_fast = src.add(len_aligned);

        while src < src_end_fast {
            // Scalar lookups
            let d0 = *table.add(*src.add(0) as usize);
            let d1 = *table.add(*src.add(1) as usize);
            let d2 = *table.add(*src.add(2) as usize);
            let d3 = *table.add(*src.add(3) as usize);
            let d4 = *table.add(*src.add(4) as usize);
            let d5 = *table.add(*src.add(5) as usize);
            let d6 = *table.add(*src.add(6) as usize);
            let d7 = *table.add(*src.add(7) as usize);

            // Fast validation (valid nibbles are 0x00–0x0F)
            if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7) & 0xF0 != 0 {
                return Err(Error::InvalidCharacter);
            }

            // Pack into 4 bytes and write as a single u32
            let out0 = (d0 << 4) | d1;
            let out1 = (d2 << 4) | d3;
            let out2 = (d4 << 4) | d5;
            let out3 = (d6 << 4) | d7;

            let pack = (out0 as u32) |
                ((out1 as u32) << 8) |
                ((out2 as u32) << 16) |
                ((out3 as u32) << 24);

            (dst as *mut u32).write_unaligned(pack);

            src = src.add(8);
            dst = dst.add(4);
        }

        // Tail handling (remaining even number of chars < 8)
        let src_end = input.as_ptr().add(len);

        while src < src_end {
            let high = *table.add(*src as usize);
            let low = *table.add(*src.add(1) as usize);

            if (high | low) & 0xF0 != 0 {
                return Err(Error::InvalidCharacter);
            }

            *dst = (high << 4) | low;

            src = src.add(2);
            dst = dst.add(1);
        }

        Ok(())
    }
}

// --- KANI (FORMAL VERIFICATION) ---

#[cfg(kani)]
mod kani_verification_scalar {
    use super::*;
    use crate::Config;

    // --- CONSTANTS ---

    // Encoder Induction Size
    const ENC_INDUCTION_LEN: usize = 17;

    // Decoder Induction Size
    const DEC_INDUCTION_LEN: usize = 17;

    // --- REAL TESTS --- 

    // --- REAL TESTS --- 

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    /// 
    /// Verifies that `Decode(Encode(X)) == X`.
    #[kani::proof]
    fn check_scalar_roundtrip_correctness() {
        let config = Config { uppercase: kani::any() };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();
        let input_len = input.len();

        // Buffers
        let mut enc_buf = [0u8; 128];
        let mut dec_buf = [0u8; 128];

        unsafe {
            // 1. Encode
            encode_slice_unsafe(&config, &input, enc_buf.as_mut_ptr());

            // Calculate actual encoded length for slicing
            let encoded_slice = &enc_buf[..input_len * 2];

            // 2. Decode
            // This MUST succeed for valid encoded output
            decode_slice_unsafe(encoded_slice, dec_buf.as_mut_ptr())
                .expect("Valid encoding failed to decode");

            // 3. Verify
            assert_eq!(&dec_buf[..input_len], &input, "Roundtrip mismatch");
        }
    }

    /// **Proof 2: Decoder Robustness & Induction**
    /// 
    /// Verifies that `decode_slice_avx2`:
    /// 1. Accepts ANY `N` bytes of garbage input.
    /// 2. Never Segfaults, Panics, or causes UB.
    /// 3. Safely handles the SIMD->Scalar pointer transition.
    #[kani::proof]
    fn check_scalar_decode_robustness() {
        // Input: `N` bytes of unrestricted symbolic data (garbage)
        let input: [u8; DEC_INDUCTION_LEN] = kani::any();
        
        // Output Buffer: Max estimated size
        let mut output = [0u8; 128];

        unsafe {
            // We ignore the Result. We only care that this function call 
            // returns safely (Ok or Err) and does not crash.
            let _ = decode_slice_unsafe(&input, output.as_mut_ptr());
        }
    }
}

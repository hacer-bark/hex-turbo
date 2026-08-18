//! # Serde Integration for `hex-turbo`
//!
//! **Transparent, zero-overhead, high-performance** conversion between hexadecimal strings
//! and binary data types (`Vec<u8>`, `[u8; N]`, `&[u8]`, `bytes::Bytes`, `Cow<[u8]>`, etc.).
//!
//! This module is only compiled when the **`serde`** feature is enabled.
//!
//! ## Why this exists
//!
//! When working with cryptographic data, blockchain payloads, API responses, etc.,
//! you almost always want to **store `Vec<u8>` internally** (for speed & memory efficiency)
//! but **serialize/deserialize as clean hex strings** in JSON, YAML, TOML, Bincode, etc.
//!
//! This module gives you exactly that with **one line of attribute** and zero boilerplate.
//!
//! ## Two variants
//!
//! | Module          | Output case | Recommended for                              |
//! |-----------------|-------------|----------------------------------------------|
//! | `hex`           | lowercase   | **Default** – Ethereum, Bitcoin, most APIs   |
//! | `hex_upper`     | UPPERCASE   | Legacy systems, some hardware, or spec reqs  |
//!
//! ## Support for Fixed-Size Arrays
//!
//! For fixed-length binary data like hashes or keys (e.g., `[u8; 32]` for SHA-256),
//! use the specialized modules `hex32` or `hex_upper32`.
//!
//! These enforce **exact length** during deserialization, failing with a descriptive error
//! if the decoded data isn't precisely 32 bytes.
//!
//! Other sizes (e.g., 20, 64) can be added on request or by copying the code.
//!
//! ## Full Example
//!
//! ```rust
//! use serde::{Serialize, Deserialize};
//! use hex_turbo::serde::hex;           // lowercase (most common)
//! use hex_turbo::serde::hex_upper;     // uppercase when needed
//! use hex_turbo::serde::hex32;         // lowercase for [u8; 32]
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct BlockHeader {
//!     /// Automatically becomes a lowercase hex string in JSON
//!     #[serde(with = "hex")]
//!     block_hash: Vec<u8>,
//!
//!     /// Fixed-size array with length enforcement
//!     #[serde(with = "hex32")]
//!     merkle_root: [u8; 32],
//!
//!     /// Uppercase signature (e.g. for some hardware wallets)
//!     #[serde(with = "hex_upper")]
//!     signature: Vec<u8>,
//! }
//! ```
//!
//! Works with **any** Serde format (JSON, YAML, TOML, MessagePack, Bincode, Postcard…).

use serde::{de, Deserializer, Serializer};

use crate::{LOWER_CASE, UPPER_CASE};

/// **Lowercase** hexadecimal (recommended default).
///
/// This submodule provides Serde helper functions that convert binary data
/// to/from **lowercase** hexadecimal strings using `hex-turbo::LOWER_CASE`.
pub mod hex {
    use super::*;

    /// Serializes any byte container as a **lowercase** hexadecimal string.
    ///
    /// ## Supported input types (serialization)
    ///
    /// - `Vec<u8>`
    /// - `&[u8]`
    /// - `[u8; N]` for any constant `N`
    /// - `&[u8; N]`
    /// - `Cow<[u8]>`
    /// - `bytes::Bytes`, `bytes::BytesMut`
    /// - Any type that implements `AsRef<[u8]>`
    ///
    /// ## Performance
    ///
    /// Uses the same AVX512 → AVX2 → scalar path as the rest of `hex-turbo`.
    /// No extra allocations beyond the final string.
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u8]>,
        S: Serializer,
    {
        let encoded = LOWER_CASE.encode(value.as_ref());
        serializer.serialize_str(&encoded)
    }

    /// Deserializes a hexadecimal string (or byte slice) into `Vec<u8>`.
    ///
    /// ## Accepted input formats
    ///
    /// - JSON string: `"48656c6c6f"` (lowercase or uppercase accepted)
    /// - `&str`
    /// - Raw bytes provided by some serializers (`visit_bytes`)
    ///
    /// ## Error handling
    ///
    /// Returns a clear Serde error with a helpful message when:
    /// - Length is odd
    /// - Invalid hex character is found
    ///
    /// **Note**: Accepts **both** uppercase and lowercase characters on input.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HexVisitor;

        impl<'de> de::Visitor<'de> for HexVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(
                    "a valid hexadecimal string (even length, characters 0-9a-fA-F)"
                )
            }

            fn visit_str<E>(self, value: &str) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                LOWER_CASE.decode(value).map_err(de::Error::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                LOWER_CASE.decode(value).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(HexVisitor)
    }
}

/// **Uppercase** hexadecimal serialization.
///
/// Use this submodule when the external format or specification requires
/// **UPPERCASE** hexadecimal strings (e.g. some legacy protocols or hardware).
pub mod hex_upper {
    use super::*;

    /// Serializes any byte container as an **uppercase** hexadecimal string.
    ///
    /// See [`hex::serialize`](super::hex::serialize) for full list of supported types
    /// and performance characteristics.
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u8]>,
        S: Serializer,
    {
        let encoded = UPPER_CASE.encode(value.as_ref());
        serializer.serialize_str(&encoded)
    }

    /// Deserializes a hexadecimal string into `Vec<u8>`, producing uppercase on output.
    ///
    /// Accepts **both** uppercase and lowercase input (same as `hex::deserialize`).
    ///
    /// See [`hex::deserialize`](super::hex::deserialize) for full error documentation.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HexVisitor;

        impl<'de> de::Visitor<'de> for HexVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(
                    "a valid hexadecimal string (even length, characters 0-9a-fA-F)"
                )
            }

            fn visit_str<E>(self, value: &str) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                UPPER_CASE.decode(value).map_err(de::Error::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<Vec<u8>, E>
            where
                E: de::Error,
            {
                UPPER_CASE.decode(value).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(HexVisitor)
    }
}

/// **Lowercase** hexadecimal for fixed-size `[u8; 32]`.
///
/// This submodule is identical to `hex` but deserializes strictly to `[u8; 32]`,
/// enforcing exact length.
pub mod hex32 {
    use super::*;

    /// Serializes a 32-byte container as a **lowercase** hexadecimal string.
    ///
    /// ## Supported input types
    ///
    /// - `[u8; 32]`
    /// - `&[u8; 32]`
    /// - Any `AsRef<[u8]>` where the underlying slice is exactly 32 bytes (checked at runtime if needed)
    ///
    /// See [`hex::serialize`](super::hex::serialize) for performance details.
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u8]>,
        S: Serializer,
    {
        let bytes = value.as_ref();
        if bytes.len() != 32 {
            return Err(serde::ser::Error::custom(format!("expected 32 bytes, got {}", bytes.len())));
        }
        let encoded = LOWER_CASE.encode(bytes);
        serializer.serialize_str(&encoded)
    }

    /// Deserializes a hexadecimal string into exactly `[u8; 32]`.
    ///
    /// Fails if the decoded data is not precisely 32 bytes.
    ///
    /// ## Accepted input formats
    ///
    /// Same as [`hex::deserialize`](super::hex::deserialize).
    ///
    /// ## Error handling
    ///
    /// - All errors from [`hex::deserialize`](super::hex::deserialize)
    /// - Additional length mismatch error if != 32 bytes
    ///
    /// **Note**: Input hex can be uppercase or lowercase.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HexVisitor;

        impl<'de> de::Visitor<'de> for HexVisitor {
            type Value = [u8; 32];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(
                    "a valid hexadecimal string representing exactly 32 bytes (64 hex chars)"
                )
            }

            fn visit_str<E>(self, value: &str) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                let vec = LOWER_CASE.decode(value).map_err(de::Error::custom)?;
                if vec.len() != 32 {
                    return Err(de::Error::custom(format!("expected 32 bytes, got {}", vec.len())));
                }
                Ok(vec.try_into().expect("length already checked"))
            }

            fn visit_string<E>(self, value: String) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                let vec = LOWER_CASE.decode(value).map_err(de::Error::custom)?;
                if vec.len() != 32 {
                    return Err(de::Error::custom(format!("expected 32 bytes, got {}", vec.len())));
                }
                Ok(vec.try_into().expect("length already checked"))
            }
        }

        deserializer.deserialize_str(HexVisitor)
    }
}

/// **Uppercase** hexadecimal for fixed-size `[u8; 32]`.
///
/// This submodule is identical to `hex_upper` but deserializes strictly to `[u8; 32]`,
/// enforcing exact length.
pub mod hex_upper32 {
    use super::*;

    /// Serializes a 32-byte container as an **uppercase** hexadecimal string.
    ///
    /// See [`hex32::serialize`](super::hex32::serialize) for details.
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u8]>,
        S: Serializer,
    {
        let bytes = value.as_ref();
        if bytes.len() != 32 {
            return Err(serde::ser::Error::custom(format!("expected 32 bytes, got {}", bytes.len())));
        }
        let encoded = UPPER_CASE.encode(bytes);
        serializer.serialize_str(&encoded)
    }

    /// Deserializes a hexadecimal string into exactly `[u8; 32]`.
    ///
    /// Fails if the decoded data is not precisely 32 bytes.
    ///
    /// See [`hex32::deserialize`](super::hex32::deserialize) for details.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HexVisitor;

        impl<'de> de::Visitor<'de> for HexVisitor {
            type Value = [u8; 32];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(
                    "a valid hexadecimal string representing exactly 32 bytes (64 hex chars)"
                )
            }

            fn visit_str<E>(self, value: &str) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                let vec = UPPER_CASE.decode(value).map_err(de::Error::custom)?;
                if vec.len() != 32 {
                    return Err(de::Error::custom(format!("expected 32 bytes, got {}", vec.len())));
                }
                Ok(vec.try_into().expect("length already checked"))
            }

            fn visit_string<E>(self, value: String) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<[u8; 32], E>
            where
                E: de::Error,
            {
                let vec = UPPER_CASE.decode(value).map_err(de::Error::custom)?;
                if vec.len() != 32 {
                    return Err(de::Error::custom(format!("expected 32 bytes, got {}", vec.len())));
                }
                Ok(vec.try_into().expect("length already checked"))
            }
        }

        deserializer.deserialize_str(HexVisitor)
    }
}

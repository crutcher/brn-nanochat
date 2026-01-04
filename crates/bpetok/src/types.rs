//! # Common Types and Traits
use core::hash::Hash;
use num_traits::{FromPrimitive, Num, ToPrimitive, Unsigned};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

/// A type that can be used as a token in a BPE-based tokenizer.
pub trait TokenType:
    'static
    + Debug
    + Clone
    + Copy
    + Hash
    + Send
    + Sync
    + Unsigned
    + FromPrimitive
    + ToPrimitive
    + Ord
    + serde::Serialize
    + for<'de> serde::Deserialize<'de>
{
}

impl<T> TokenType for T where
    T: 'static
        + Debug
        + Clone
        + Copy
        + Hash
        + Send
        + Sync
        + Unsigned
        + FromPrimitive
        + ToPrimitive
        + Ord
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
{
}

/// Returns true if the token is a byte token.
pub fn is_byte_token<T: TokenType>(token: T) -> bool {
    token < T::from_usize(crate::validators::U8_SIZE).unwrap()
}

/// A pair of tokens.
pub type Pair<T> = (T, T);

/// A type that can be used as a word count.
pub trait CountType:
    Num
    + AddAssign
    + SubAssign
    + Default
    + Copy
    + Debug
    + Display
    + Send
    + Sync
    + Hash
    + Ord
    + FromPrimitive
{
}

impl<T> CountType for T where
    T: Num
        + AddAssign
        + SubAssign
        + Default
        + Copy
        + Debug
        + Display
        + Send
        + Sync
        + Hash
        + Ord
        + FromPrimitive
{
}

/// A type that can be used as a string key.
pub trait StringChunkType:
    for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

impl<T> StringChunkType for T where
    T: for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_byte_token() {
        assert!(is_byte_token(0_u16));
        assert!(is_byte_token(0_u32));
        assert!(is_byte_token(0_u64));
        assert!(is_byte_token(0_usize));

        assert!(is_byte_token(255_u16));
        assert!(is_byte_token(255_u32));
        assert!(is_byte_token(255_u64));
        assert!(is_byte_token(255_usize));

        assert!(!is_byte_token(256_u16));
        assert!(!is_byte_token(256_u32));
    }
}

/// [`Pair<T>`] to T map.
pub type MergeMap<T> = ahash::AHashMap<Pair<T>, T>;

/// T to [`Pair<T>`] map.
pub type ExpansionMap<T> = ahash::AHashMap<T, Pair<T>>;

/// Check if a type is `Send`.
#[cfg(test)]
pub(crate) fn check_is_send<S: Send>(_: S) {}

#[cfg(test)]
/// Check if a type is `Sync`.
pub(crate) fn check_is_sync<S: Sync>(_: S) {}

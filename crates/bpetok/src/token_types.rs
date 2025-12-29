use core::hash::Hash;
use num_traits::{FromPrimitive, Num, PrimInt};
use std::fmt::Debug;
use std::ops::AddAssign;

/// A type that can be used as a token in a BPE-based tokenizer.
pub trait TokenType: Debug + Clone + Copy + Hash + Send + Sync + FromPrimitive + PrimInt {}

impl<T> TokenType for T where T: Debug + Clone + Copy + Hash + Send + Sync + FromPrimitive + PrimInt {}

/// A pair of tokens.
pub type Pair<T> = (T, T);

/// A type that can be used as a word count.
pub trait CountType:
    Num + AddAssign + Default + Copy + Debug + Send + Sync + Hash + Ord + FromPrimitive
{
}

impl<T> CountType for T where
    T: Num + AddAssign + Default + Copy + Debug + Send + Sync + Hash + Ord + FromPrimitive
{
}

/// A type that can be used as a string key.
pub trait ChunkType:
    for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

impl<T> ChunkType for T where
    T: for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

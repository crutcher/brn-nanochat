use core::hash::Hash;
use std::fmt::Debug;

/// A type that can be used as a token in a BPE-based tokenizer.
pub trait Token:
    Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sync
{
}

impl<T> Token for T where
    T: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sized + Sync
{
}

/// A pair of tokens.
pub type Pair<T> = (T, T);

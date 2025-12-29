use core::hash::Hash;

/// A type that can be used as a token in a BPE-based tokenizer.
pub trait Token: Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sync {}

impl<T> Token for T where
    T: Clone + Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Send + Sized + Sync
{
}

/// A pair of tokens.
pub type Pair<T> = (T, T);

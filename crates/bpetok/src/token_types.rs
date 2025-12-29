use core::hash::Hash;
use num_traits::{FromPrimitive, PrimInt};
use std::fmt::Debug;

/// A type that can be used as a token in a BPE-based tokenizer.
pub trait Token: Debug + Clone + Copy + Hash + Send + Sync + FromPrimitive + PrimInt {}

impl<T> Token for T where T: Debug + Clone + Copy + Hash + Send + Sync + FromPrimitive + PrimInt {}

/// A pair of tokens.
pub type Pair<T> = (T, T);

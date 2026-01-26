//! # `WordChuck` LLM Tokenizer
#![warn(missing_docs, unused)]

extern crate alloc;

#[cfg(feature = "rayon")]
pub mod rayon;

#[cfg(feature = "training")]
pub mod training;

pub mod decoders;
pub mod encoders;
pub mod regex;
pub mod segmentation;
pub mod types;
pub mod util;
pub mod vocab;

//! # `wordchuck` LLM Tokenizer
//!
//! This is a high-performance LLM tokenizer suite.
//!
//! `wordchuck` is compatible with `nanochat/rustbpe` and `tiktoken` tokenizers.
//!
//! See:
//! * [`encoders`] to encode text into tokens.
//! * [`decoders`] to decode tokens into text.
//! * [`training`] to train a [`vocab::UnifiedTokenVocab`].
//! * [`vocab`] to manage token vocabularies, vocab io, and pre-trained tokenizers.
//!
//! A number of pretrained public tokenizers are available through:
//! * [`vocab::public`]
#![warn(missing_docs, unused)]
#![cfg_attr(not(feature = "std"), no_std)]

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
pub mod vocab;

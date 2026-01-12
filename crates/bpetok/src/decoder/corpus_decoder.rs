//! # Corpus Decoder
//! Experimental.

use crate::decoder::TokenDecoder;
use crate::types::{BinaryPairMap, ExpansionMap, TokenType, is_byte_token};
use crate::vocab::data::BPEMapTokenVocab;
use ahash::AHashMap;
use std::collections::hash_map;
use std::ops::Range;

/// A token dictionary [`TokenDecoder<T>`], with a shared corpus buffer.
#[derive(Clone)]
pub struct CorpusDecoder<T: TokenType> {
    /// Token to byte slice mapping.
    /// Does not include byte-tokens.
    pub slices: AHashMap<T, Range<usize>>,

    /// The corpus buffer.
    pub corpus: Vec<u8>,
}

impl<T: TokenType> CorpusDecoder<T> {
    /// Creates a new corpus decoder.
    pub fn new(
        mut slices: AHashMap<T, Range<usize>>,
        mut corpus: Vec<u8>,
    ) -> Self {
        slices.shrink_to_fit();
        corpus.shrink_to_fit();

        Self { slices, corpus }
    }

    /// Creates a new corpus decoder.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(data)))]
    pub fn from_bpe(data: &BPEMapTokenVocab<T>) -> Self {
        Self::from_pairs(&data.pairs)
    }

    /// Creates a new corpus decoder.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(merge_map)))]
    pub fn from_pairs(merge_map: &BinaryPairMap<T>) -> Self {
        let expansion_map: ExpansionMap<T> = merge_map
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();

        let mut tokens = expansion_map.keys().copied().collect::<Vec<_>>();
        tokens.sort();
        tokens.reverse();

        // Materialize tokens starting with the highest ranked tokens first.

        let mut mmaps: Vec<MaterializationMap<T>> = Default::default();
        let mut mmap_index: AHashMap<T, usize> = Default::default();
        let mut total_size: usize = 0;
        for &token in &tokens {
            if mmap_index.contains_key(&token) {
                continue;
            }
            let mmap = MaterializationMap::materialize(token, &expansion_map, &|t| {
                if let Some(&i) = mmap_index.get(&t) {
                    let mmap = &mmaps[i];
                    return mmap.try_get(t);
                }
                None
            });
            total_size += mmap.buf.len();
            let idx = mmaps.len();
            mmap.tokens().for_each(|t| {
                mmap_index.insert(t, idx);
            });
            mmaps.push(mmap);
        }

        let mut slices: AHashMap<T, Range<usize>> = AHashMap::with_capacity(merge_map.len());
        let mut corpus: Vec<u8> = Vec::with_capacity(total_size);

        for mmap in mmaps {
            if slices.contains_key(&mmap.root) {
                continue;
            }

            let offset = corpus.len();
            corpus.extend_from_slice(&mmap.buf);

            for (t, slice) in mmap.slices {
                // Every slice is in the highest ranked token it is a child of.
                slices
                    .entry(t)
                    .or_insert_with(|| slice.start + offset..slice.end + offset);

                // Alternatively, we could override; and push them to the lowest ranked:
                // slices.insert(t, slice.start + offset..slice.end + offset);

                // Timing seems to not care.
            }
        }

        Self::new(slices, corpus)
    }
}

impl<T: TokenType> TokenDecoder<T> for CorpusDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.slices.keys().copied()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    ) {
        for t in tokens {
            if let Some(b) = t.to_u8() {
                buf.push(b);
            } else {
                let range = self.slices.get(t).expect("Token not found in slice map");
                let slice = &self.corpus[range.clone()];
                buf.extend_from_slice(slice);
            }
        }
    }

    fn size_estimate(&self) -> usize {
        size_of::<hash_map::Entry<T, Range<usize>>>() * self.slices.len() + self.corpus.len()
    }
}

/// Represents a materialized sequence of tokens and their byte slices.
struct MaterializationMap<T: TokenType> {
    root: T,
    buf: Vec<u8>,
    slices: AHashMap<T, Range<usize>>,
}

impl<T: TokenType> MaterializationMap<T> {
    /// Creates a new materialization map for the given token.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(token, expansion_map, maybe_slice))
    )]
    fn materialize<'a, F>(
        token: T,
        expansion_map: &ExpansionMap<T>,
        maybe_slice: &F,
    ) -> Self
    where
        F: Fn(T) -> Option<&'a [u8]>,
    {
        assert!(
            !is_byte_token(token),
            "MaterializationMap should only be used for non-byte tokens"
        );

        let mut mmap = Self {
            root: token,
            buf: Default::default(),
            slices: Default::default(),
        };

        mmap.expand(token, expansion_map, maybe_slice);

        mmap
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(self, token, expansion_map, maybe_slice))
    )]
    fn expand<'a, F>(
        &mut self,
        token: T,
        expansion_map: &ExpansionMap<T>,
        maybe_slice: &F,
    ) where
        F: Fn(T) -> Option<&'a [u8]>,
    {
        if is_byte_token(token) {
            // byte tokens are not inserted into the slice map.
            let b = token.to_u8().unwrap();
            self.buf.push(b);
            return;
        }

        if let Some(slice) = self.try_get(token) {
            // The current buffer is def in the cache lines;
            // so copy from self first.
            let slice = slice.to_owned();

            // we're already in the slices map, so don't update it.
            self.buf.extend_from_slice(&slice);
            return;
        }
        if let Some(slice) = maybe_slice(token) {
            let start = self.buf.len();
            self.slices.insert(token, start..start + slice.len());

            self.buf.extend_from_slice(slice);
            return;
        }

        let pair = expansion_map
            .get(&token)
            .expect("token not found in expansion_map");
        let (a, b) = pair.to_owned();
        let start = self.buf.len();
        self.expand(a, expansion_map, maybe_slice);
        self.expand(b, expansion_map, maybe_slice);
        let end = self.buf.len();
        self.slices.insert(token, start..end);
    }

    /// Returns an iterator over the non-byte tokens in this map.
    fn tokens(&self) -> impl Iterator<Item = T> {
        self.slices.keys().copied()
    }

    /// Returns the byte slice for the given token, if it exists.
    fn try_get(
        &self,
        token: T,
    ) -> Option<&[u8]> {
        self.slices
            .get(&token)
            .map(|range| &self.buf[range.clone()])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::scanning_encoder::ScanningEncoder;
    use crate::tokenizer::{EncoderData, TokenEncoder};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::data::WordMapTokenVocab;
    use crate::vocab::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_corpus_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BPETokenVocabTrainer::new_with_vocab_size(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let TrainResults {
            word_pattern,
            bpe_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let word_vocab = WordMapTokenVocab::from_bpe(&bpe_vocab);

        let encoder_data = Arc::new(EncoderData {
            word_pattern: word_pattern.into(),
            word_vocab,
            bpe_vocab: bpe_vocab.clone(),
        });

        let encoder = ScanningEncoder::<T>::new(encoder_data.clone(), Default::default());

        let decoder = CorpusDecoder::from_bpe(&encoder_data.bpe_vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}

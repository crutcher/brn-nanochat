//! # Corpus Decoder
//! Experimental.

use crate::{Pair, TokenDecoder, TokenType, is_byte_token};
use ahash::AHashMap;
use std::collections::hash_map;
use std::ops::Range;

/// Represents a materialized sequence of tokens and their byte slices.
pub struct MaterializationMap<T: TokenType> {
    root: T,
    buf: Vec<u8>,
    slices: AHashMap<T, Range<usize>>,
}

impl<T: TokenType> MaterializationMap<T> {
    /// Creates a new materialization map for the given token.
    pub fn materialize<'a, F>(
        token: T,
        token_to_pair: &AHashMap<T, Pair<T>>,
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

        mmap.expand(token, token_to_pair, maybe_slice);

        mmap
    }

    fn expand<'a, F>(
        &mut self,
        token: T,
        token_to_pair: &AHashMap<T, Pair<T>>,
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

        let pair = token_to_pair
            .get(&token)
            .expect("token not found in token_to_pair");
        let (a, b) = pair.to_owned();
        let start = self.buf.len();
        self.expand(a, token_to_pair, maybe_slice);
        self.expand(b, token_to_pair, maybe_slice);
        let end = self.buf.len();
        self.slices.insert(token, start..end);
    }

    /// Returns the root token of this map.
    pub fn root(&self) -> T {
        self.root
    }

    /// Returns the underlying buffer.
    pub fn buf(&self) -> &[u8] {
        &self.buf
    }

    /// Returns an iterator over the non-byte tokens in this map.
    pub fn tokens(&self) -> impl Iterator<Item = T> {
        self.slices.keys().copied()
    }

    /// Returns the slice map.
    pub fn slices(&self) -> &AHashMap<T, Range<usize>> {
        &self.slices
    }

    /// Returns the byte slice for the given token, if it exists.
    pub fn try_get(
        &self,
        token: T,
    ) -> Option<&[u8]> {
        self.slices
            .get(&token)
            .map(|range| &self.buf[range.clone()])
    }
}

/// A token decoder.
#[derive(Clone)]
pub struct CorpusDecoder<T: TokenType> {
    /// Token to byte slice mapping.
    /// Does not include byte-tokens.
    slices: AHashMap<T, Range<usize>>,

    corpus: Vec<u8>,
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
    pub fn from_merge_map(merges: &AHashMap<Pair<T>, T>) -> Self {
        let token_to_pair: AHashMap<T, Pair<T>> =
            merges.iter().map(|(&pair, &token)| (token, pair)).collect();

        let mut tokens = token_to_pair.keys().copied().collect::<Vec<_>>();
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
            let mmap = MaterializationMap::materialize(token, &token_to_pair, &|t| {
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

        let mut slices: AHashMap<T, Range<usize>> = AHashMap::with_capacity(merges.len());
        let mut corpus: Vec<u8> = Vec::with_capacity(total_size);

        for mmap in mmaps {
            if slices.contains_key(&mmap.root()) {
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

    /// Gets the corpus buffer.
    pub fn corpus(&self) -> &[u8] {
        &self.corpus
    }

    /// Gets the slice map.
    pub fn slices(&self) -> &AHashMap<T, Range<usize>> {
        &self.slices
    }
}

impl<T: TokenType> TokenDecoder<T> for CorpusDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.slices.keys().copied()
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{check_is_send, check_is_sync};
    use crate::{Tokenizer, TokenizerOptions};
    use compact_str::CompactString;

    #[test]
    fn test_corpus_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = TokenizerOptions::with_capacity(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let tokenizer: Tokenizer<T> =
            options.train_from_sample_iterator::<T, K, C, _>(samples.iter());

        let decoder = CorpusDecoder::from_merge_map(&tokenizer.merge_map);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = tokenizer.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}

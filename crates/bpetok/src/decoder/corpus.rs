//! # Corpus Decoder
//! Experimental.

use crate::{Pair, TokenDecoder, TokenType, is_byte_token};
use ahash::AHashMap;
use std::collections::{HashMap, hash_map};
use std::ops::Range;

/// Represents a materialized sequence of tokens and their byte slices.
pub struct MaterializationMap<T: TokenType> {
    root: T,
    buf: Vec<u8>,
    slices: HashMap<T, Range<usize>>,
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
            buf: Vec::new(),
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

    /// Returns the slice map.
    pub fn slices(&self) -> &HashMap<T, Range<usize>> {
        &self.slices
    }

    /// Returns an iterator over the tokens in this map.
    pub fn token_iter(&self) -> impl Iterator<Item = T> {
        self.slices.keys().copied()
    }

    /// Returns a list of all tokens in this map.
    pub fn tokens(&self) -> Vec<T> {
        self.token_iter().collect()
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
    slices: HashMap<T, Range<usize>>,
    corpus: Vec<u8>,
}

impl<T: TokenType> CorpusDecoder<T> {
    /// Creates a new corpus decoder.
    pub fn new(
        slices: HashMap<T, Range<usize>>,
        corpus: Vec<u8>,
    ) -> Self {
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
            mmap.token_iter().for_each(|t| {
                mmap_index.insert(t, idx);
            });
            mmaps.push(mmap);
        }

        let mut slices: HashMap<T, Range<usize>> = HashMap::with_capacity(merges.len());
        let mut corpus: Vec<u8> = Vec::with_capacity(total_size);

        for mmap in mmaps {
            if slices.contains_key(&mmap.root()) {
                continue;
            }

            let offset = corpus.len();
            corpus.extend_from_slice(&mmap.buf);

            for (t, slice) in mmap.slices {
                slices
                    .entry(t)
                    .or_insert_with(|| slice.start + offset..slice.end + offset);
            }
        }

        slices.shrink_to_fit();
        corpus.shrink_to_fit();

        Self::new(slices, corpus)
    }

    /// Gets the corpus buffer.
    pub fn corpus(&self) -> &[u8] {
        &self.corpus
    }

    /// Gets the slice map.
    pub fn slices(&self) -> &HashMap<T, Range<usize>> {
        &self.slices
    }
}

impl<'a, T: TokenType> TokenDecoder<T> for CorpusDecoder<T> {
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let tokens = tokens.as_ref();

        let mut buf = Vec::with_capacity(tokens.len() * 2);
        for t in tokens {
            if let Some(b) = t.to_u8() {
                buf.push(b);
            } else {
                let range = self.slices.get(t).expect("Token not found in slice map");
                let slice = &self.corpus[range.clone()];
                buf.extend_from_slice(slice);
            }
        }

        buf
    }

    fn size_estimate(&self) -> (usize, usize) {
        (
            size_of::<hash_map::Entry<T, Range<usize>>>() * self.slices.len(),
            self.corpus.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let decoder = CorpusDecoder::from_merge_map(&tokenizer.merges);

        for sample in samples {
            let tokens = tokenizer.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}

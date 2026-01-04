//! # Chunk Pair Scan Tokenizer

use crate::decoder::TokenDecoder;
use crate::decoder::corpus_decoder::CorpusDecoder;
use crate::tokenizer::TokenEncoder;
use crate::types::{Pair, TokenType};
use crate::validators::expect_regex;
use crate::vocab::data::TokenVocabData;
use crate::{DEFAULT_PARALLEL, validators};
use fancy_regex::Regex;
use std::sync::Arc;

/// A training for [`Tokenizer`]s.
#[derive(Debug, Clone)]
pub struct ChunkPairScanTokenizerOptions {
    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl ChunkPairScanTokenizerOptions {
    /// Sets whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self {
            parallel: validators::expect_parallel(parallel),
        }
    }
}

impl Default for ChunkPairScanTokenizerOptions {
    fn default() -> Self {
        Self {
            parallel: DEFAULT_PARALLEL,
        }
    }
}

/// A Byte Pair Encoding / Decoding Tokenizer.
#[derive(Debug)]
pub struct ChunkPairScanTokenizer<T: TokenType> {
    /// Core data describing a BPE Tokenizer.
    pub data: Arc<TokenVocabData<T>>,

    /// Tokenizer options.
    pub options: ChunkPairScanTokenizerOptions,

    /// The compiled regex pattern.
    pub compiled_pattern: Regex,
}

impl<T: TokenType> ChunkPairScanTokenizer<T> {
    /// Construct a new Tokenizer.
    pub fn new<D>(
        data: D,
        options: ChunkPairScanTokenizerOptions,
    ) -> Self
    where
        D: Into<Arc<TokenVocabData<T>>>,
    {
        #[cfg(not(feature = "rayon"))]
        if options.parallel {
            panic!("Parallel processing requires the `rayon` feature to be enabled.");
        }

        let data = data.into();
        let compiled_pattern = expect_regex(&data.pattern);
        Self {
            data,
            options,
            compiled_pattern,
        }
    }

    /// Vocab Size.
    pub fn max_token(&self) -> T {
        self.data.max_token()
    }

    /// Memory usage estimate in bytes.
    pub fn size_estimate(&self) -> usize {
        self.data.size_estimate()
    }

    /// Encode a chunk of text into token IDs.
    #[tracing::instrument(skip(self, chunk))]
    pub fn encode_chunk<S: AsRef<str>>(
        &self,
        chunk: S,
    ) -> Vec<T> {
        let chunk = chunk.as_ref();

        // Convert chunk to bytes then to tokens.
        let mut chunk_tokens: Vec<T> = chunk.bytes().map(|b| T::from_u8(b).unwrap()).collect();

        // Apply merges iteratively
        while chunk_tokens.len() >= 2 {
            // Find the best pair to merge
            let mut best_pair: Option<(usize, Pair<T>, T)> = None;

            for i in 0..chunk_tokens.len() - 1 {
                let pair: Pair<T> = (chunk_tokens[i], chunk_tokens[i + 1]);
                if let Some(&new_id) = self.data.merge_map.get(&pair)
                    && (best_pair.is_none() || new_id < best_pair.unwrap().2)
                {
                    best_pair = Some((i, pair, new_id));
                }
            }

            // If we found a pair to merge, apply it
            if let Some((idx, _pair, new_id)) = best_pair {
                chunk_tokens[idx] = new_id;
                chunk_tokens.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }

        chunk_tokens
    }

    #[tracing::instrument(skip(self, text))]
    fn split_groups<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.compiled_pattern
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
    }

    /// Encode a string into token IDs in parallel.
    ///
    /// Uses parallel processing, ignoring the `parallel` flag.
    #[cfg(feature = "rayon")]
    pub fn encode_rayon<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        use rayon::prelude::*;

        // This is significantly worse?

        self.split_groups(text.as_ref())
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Encode a string into token IDs serially.
    ///
    /// Uses serial processing, ignoring the `parallel` flag.
    pub fn encode_serial<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        self.split_groups(text.as_ref())
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Build a [`TokenDecoder`] from this [`ChunkPairScanTokenizer`].
    pub fn to_decoder(&self) -> impl TokenDecoder<T> {
        CorpusDecoder::from_data(&self.data)
    }
}

impl<T: TokenType> TokenEncoder<T> for ChunkPairScanTokenizer<T> {
    fn data(&self) -> &Arc<TokenVocabData<T>> {
        &self.data
    }

    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.data.pair_tokens()
    }

    #[tracing::instrument(skip(self, text))]
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        if self.options.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            // We just fall back to serial because rayon is currently slower.
        }

        self.encode_serial(text)
    }
}

#[cfg(test)]
mod tests {
    use crate::decoder::TokenDecoder;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::cps_tokenizer::ChunkPairScanTokenizer;
    use crate::types;
    use crate::vocab::training::trainer::VocabTrainer;
    use compact_str::CompactString;

    #[test]
    #[cfg(feature = "rayon")]
    fn test_tokenizer_parallel() {
        test_tokenizer(true);
    }

    #[test]
    fn test_tokenizer_serial() {
        test_tokenizer(false);
    }

    fn test_tokenizer(parallel: bool) {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = VocabTrainer::with_capacity(1000).with_parallel(parallel);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let data = options.train_vocab_from_sample_iter::<T, K, C, _>(samples.iter());
        let tokenizer = ChunkPairScanTokenizer::new(data, Default::default());

        // compile time checks.
        types::check_is_send(&tokenizer);
        types::check_is_sync(&tokenizer);

        assert_eq!(tokenizer.max_token(), 292);

        let decoder = tokenizer.to_decoder();

        // compile time checks.
        types::check_is_send(&decoder);
        types::check_is_sync(&decoder);

        for sample in samples {
            assert_eq!(decoder.decode_to_string(tokenizer.encode(sample)), sample);
        }
    }
}

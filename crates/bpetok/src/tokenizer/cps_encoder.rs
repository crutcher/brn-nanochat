//! # Chunk Pair Scan Tokenizer

use crate::decoder::TokenDecoder;
use crate::decoder::corpus_decoder::CorpusDecoder;
use crate::decoder::dictionary_decoder::DictionaryDecoder;
use crate::regex_pool::RegexPool;
use crate::tokenizer::TokenEncoder;
use crate::types::{Pair, TokenType, VocabMap};
use crate::validators::expect_regex;
use crate::vocab::data::TokenVocabData;
use crate::vocab::training::tiktoken_io::save_tiktoken_vocab;
use crate::{DEFAULT_PARALLEL, validators};
use ahash::AHashMap;
use std::collections::hash_map;
use std::ops::Range;
use std::sync::Arc;

/// Config options for the [`CPSEncoder`].
#[derive(Debug, Clone)]
pub struct CPSEncoderOptions {
    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl CPSEncoderOptions {
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

impl Default for CPSEncoderOptions {
    fn default() -> Self {
        Self {
            parallel: DEFAULT_PARALLEL,
        }
    }
}

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct CPSEncoder<T: TokenType> {
    /// Core data describing a BPE Tokenizer.
    pub data: Arc<TokenVocabData<T>>,

    /// Tokenizer options.
    pub options: CPSEncoderOptions,

    regex_pool: RegexPool,

    vocab_map: VocabMap<T>,
}

impl<T: TokenType> CPSEncoder<T> {
    /// Construct a new Tokenizer.
    pub fn new<D>(
        data: D,
        options: CPSEncoderOptions,
    ) -> Self
    where
        D: Into<Arc<TokenVocabData<T>>>,
    {
        #[cfg(not(feature = "rayon"))]
        if options.parallel {
            panic!("Parallel processing requires the `rayon` feature to be enabled.");
        }

        let data = data.into();
        let regex_pool = RegexPool::new(expect_regex(&data.pattern), 10);

        let decoder = DictionaryDecoder::from_data(&data);
        let chunk_map = decoder
            .dictionary
            .into_iter()
            .map(|(k, v)| (v, k))
            .collect();
        Self {
            data,
            options,
            regex_pool,
            vocab_map: chunk_map,
        }
    }

    /// Save the chunk map to a tiktoken vocab file.
    pub fn save_tiktoken_vocab(
        &self,
        path: &str,
    ) -> anyhow::Result<()> {
        save_tiktoken_vocab(&self.vocab_map, path)
    }

    /// Memory usage estimate in bytes.
    pub fn size_estimate(&self) -> usize {
        let data_size = self.data.size_estimate();

        let chunk_meta = size_of::<hash_map::Entry<Vec<u8>, T>>() * self.vocab_map.len();
        let chunk_sum = self.vocab_map.keys().map(|b| b.len()).sum::<usize>();

        data_size + chunk_meta + chunk_sum
    }

    /// Append a chunk of text into token IDs.
    #[tracing::instrument(skip(self, buf, chunk))]
    pub fn append_encode_chunk<'a>(
        &self,
        buf: &mut Vec<T>,
        chunk: &'a [u8],
        cache: Option<&mut AHashMap<&'a [u8], Range<usize>>>,
    ) {
        if chunk.len() == 1 {
            buf.push(T::from_u8(chunk[0]).unwrap());
            return;
        }

        if let Some(token) = self.vocab_map.get(chunk) {
            buf.push(*token);
            return;
        }

        if let Some(cache) = &cache
            && let Some(r) = cache.get(chunk)
        {
            buf.extend_from_within(r.clone());
            return;
        }

        // Reuse the output buffer for the merges.
        let start = buf.len();
        chunk.iter().for_each(|&b| buf.push(T::from_u8(b).unwrap()));

        while (buf.len() - start) >= 2 {
            // Find the best pair to merge
            let mut best_pair: Option<(usize, Pair<T>, T)> = None;

            for i in start..(buf.len() - 1) {
                let pair: Pair<T> = (buf[i], buf[i + 1]);
                if let Some(&new_id) = self.data.merge_map.get(&pair)
                    && (best_pair.is_none() || new_id < best_pair.unwrap().2)
                {
                    best_pair = Some((i, pair, new_id));
                }
            }

            // If we found a pair to merge, apply it
            if let Some((idx, _pair, new_id)) = best_pair {
                buf[idx] = new_id;
                buf.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }

        if let Some(cache) = cache {
            cache.insert(chunk, start..buf.len());
        }
    }

    /// Encode a chunk of text into token IDs.
    #[tracing::instrument(skip(self, chunk))]
    pub fn encode_chunk<S: AsRef<str>>(
        &self,
        chunk: S,
    ) -> Vec<T> {
        let chunk_bytes: &[u8] = chunk.as_ref().as_bytes();
        let mut tokens = Vec::with_capacity(chunk_bytes.len());
        self.append_encode_chunk(&mut tokens, chunk_bytes, None);
        tokens
    }

    /// Encode a string into token IDs serially.
    ///
    /// Uses serial processing, ignoring the `parallel` flag.
    pub fn encode_serial<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());
        let mut cache = AHashMap::with_capacity(tokens.len());
        for chunk in self
            .regex_pool
            .get()
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
        {
            self.append_encode_chunk(&mut tokens, chunk.as_bytes(), Some(&mut cache));
        }
        tokens
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

        let text1 = text.as_ref();
        self.regex_pool
            .get()
            .find_iter(text1)
            .map(|m| m.unwrap().as_str())
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Build a [`TokenDecoder`] from this [`CPSEncoder`].
    pub fn to_decoder(&self) -> impl TokenDecoder<T> {
        CorpusDecoder::from_data(&self.data)
    }
}

impl<T: TokenType> TokenEncoder<T> for CPSEncoder<T> {
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

    /// Encode a batch of text into tokens.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        if self.options.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                batch
                    .par_iter()
                    .map(|text| self.encode_serial(text))
                    .collect()
            }
        } else {
            batch.iter().map(|text| self.encode_serial(text)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decoder::TokenDecoder;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::cps_encoder::CPSEncoder;
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

        let options = VocabTrainer::new_with_vocab_size(1000).with_parallel(parallel);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let data = options.train_vocab_from_sample_iter::<T, K, C, _>(samples.iter());
        let tokenizer = CPSEncoder::new(data, Default::default());

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

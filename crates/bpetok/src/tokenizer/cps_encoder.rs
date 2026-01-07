//! # Chunk Pair Scan Tokenizer

use crate::DEFAULT_PARALLEL;
use crate::decoder::TokenDecoder;
use crate::decoder::corpus_decoder::CorpusDecoder;
use crate::decoder::dictionary_decoder::DictionaryDecoder;
use crate::tokenizer::TokenEncoder;
use crate::types::{TokenType, VocabMap};
use crate::util::regex::regex_pool::RegexWrapperPool;
use crate::util::regex::regex_wrapper::{RegexPatternLabel, RegexWrapper};
use crate::util::validators;
use crate::vocab::data::TokenVocabData;
use crate::vocab::tiktoken_io::save_tiktoken_vocab;
use std::collections::hash_map;
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

    regex_pool: RegexWrapperPool,

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

        let regex: Arc<RegexWrapper> = RegexPatternLabel::Adaptive(data.pattern.clone())
            .compile()
            .unwrap()
            .into();
        let regex_pool = RegexWrapperPool::from(regex);

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
    pub fn append_encode_chunk(
        &self,
        buf: &mut Vec<T>,
        chunk: &[u8],
    ) {
        if chunk.len() == 1 {
            buf.push(T::from_u8(chunk[0]).unwrap());
            return;
        }

        if let Some(token) = self.vocab_map.get(chunk) {
            buf.push(*token);
            return;
        }

        // Reuse the output buffer as a stack.
        // Append the byte-tokens to the buffer.
        let start = buf.len();
        buf.extend(chunk.into_iter().map(|&b| T::from_u8(b).unwrap()));

        // Incrementally shrink the "stack" (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while buf.len() >= stop {
            // Find the pair which merges to the lowest ranked token.
            let mut best_match: Option<(usize, T)> = None;

            for idx in start..buf.len() - 1 {
                let pair = (buf[idx], buf[idx + 1]);

                if let Some(&merge_token) = self.data.merge_map.get(&pair)
                    && (best_match.is_none() || (merge_token < best_match.unwrap().1))
                {
                    best_match = Some((idx, merge_token));
                }
            }

            if let Some((idx, merge_token)) = best_match {
                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                buf[idx] = merge_token;
                buf.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }
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

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());

        for chunk in self.regex_pool.get().find_iter(text).map(|m| m.as_str()) {
            self.append_encode_chunk(&mut tokens, chunk.as_bytes());
        }
        tokens
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
                batch.par_iter().map(|text| self.encode(text)).collect()
            }
        } else {
            batch.iter().map(|text| self.encode(text)).collect()
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

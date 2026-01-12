//! # Chunk Pair Scan Tokenizer

use crate::DEFAULT_PARALLEL;
use crate::decoder::TokenDecoder;
use crate::decoder::corpus_decoder::CorpusDecoder;
use crate::tokenizer::TokenEncoder;
use crate::types::TokenType;
use crate::util::regex::regex_wrapper::{RegexPatternLabel, RegexWrapper};
use crate::util::validators;
use crate::vocab::data::{BPEMapTokenVocab, TokenVocab, WordMapTokenVocab};
use std::sync::Arc;

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct EncoderData<T: TokenType> {
    /// Regex pattern for word splitting.
    pub word_pattern: RegexPatternLabel,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub word_vocab: WordMapTokenVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub bpe_vocab: BPEMapTokenVocab<T>,
}

impl<T: TokenType> EncoderData<T> {
    /// Maximum token ID in the vocabulary.
    pub fn max_token(&self) -> T {
        self.bpe_vocab.max_token().max(self.word_vocab.max_token())
    }
}

/// Config options for the [`ScanningEncoder`].
#[derive(Debug, Clone)]
pub struct ScanningEncoderOptions {
    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl ScanningEncoderOptions {
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

impl Default for ScanningEncoderOptions {
    fn default() -> Self {
        Self {
            parallel: DEFAULT_PARALLEL,
        }
    }
}

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct ScanningEncoder<T: TokenType> {
    /// Data for the encoder.
    pub data: Arc<EncoderData<T>>,

    /// Tokenizer options.
    pub options: ScanningEncoderOptions,

    /// Regex for word splitting.
    pub word_regex: RegexWrapper,
}

impl<T: TokenType> ScanningEncoder<T> {
    /// Construct a new encoder..
    pub fn new(
        data: Arc<EncoderData<T>>,
        options: ScanningEncoderOptions,
    ) -> Self {
        #[cfg(not(feature = "rayon"))]
        if options.parallel {
            panic!("Parallel processing requires the `rayon` feature to be enabled.");
        }

        let word_regex: RegexWrapper = data.word_pattern.compile().unwrap();
        Self {
            data,
            options,
            word_regex,
        }
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

        if let Some(token) = self.data.word_vocab.get(chunk) {
            buf.push(token);
            return;
        }

        // Reuse the output buffer as a stack.
        // Append the byte-tokens to the buffer.
        let start = buf.len();
        buf.extend(chunk.iter().map(|&b| T::from_u8(b).unwrap()));

        // Incrementally shrink the "stack" (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while buf.len() >= stop {
            // Find the pair which merges to the lowest ranked token.
            let mut best_match: Option<(usize, T)> = None;

            for idx in start..buf.len() - 1 {
                let pair = (buf[idx], buf[idx + 1]);

                if let Some(&merge_token) = self.data.bpe_vocab.pairs.get(&pair)
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

    /// Build a [`TokenDecoder`] from this [`ScanningEncoder`].
    pub fn to_decoder(&self) -> impl TokenDecoder<T> {
        CorpusDecoder::from_bpe(&self.data.bpe_vocab)
    }
}

impl<T: TokenType> TokenEncoder<T> for ScanningEncoder<T> {
    fn max_token(&self) -> T {
        self.data.max_token()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());

        for chunk in self.word_regex.find_iter(text).map(|m| m.as_str()) {
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
    use crate::tokenizer::scanning_encoder::ScanningEncoder;
    use crate::tokenizer::{EncoderData, TokenEncoder};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::data::WordMapTokenVocab;
    use crate::vocab::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use compact_str::CompactString;
    use std::sync::Arc;

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

        let options = BPETokenVocabTrainer::new_with_vocab_size(1000).with_parallel(parallel);

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

        let bpe_vocab = bpe_vocab;
        let word_vocab = WordMapTokenVocab::from_bpe(&bpe_vocab);

        let encoder_data = Arc::new(EncoderData {
            word_pattern: word_pattern.into(),
            word_vocab,
            bpe_vocab,
        });

        let encoder = ScanningEncoder::<T>::new(encoder_data.clone(), Default::default());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = encoder.to_decoder();
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            assert_eq!(decoder.decode_to_string(encoder.encode(sample)), sample);
        }
    }
}

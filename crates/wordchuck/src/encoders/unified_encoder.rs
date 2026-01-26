//! # Encoder for [`UnifiedTokenVocab`].

use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::text_segmentor::{TextSegmentor, WordRef};
use crate::types::TokenType;
use crate::vocab::ByteTokenTable;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use alloc::sync::Arc;

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct UnifiedVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    byte_table: Arc<ByteTokenTable<T>>,
    segmentor: TextSegmentor,
}

impl<T: TokenType> UnifiedVocabEncoder<T> {
    /// Construct an encoder from data.
    pub fn new(data: Arc<UnifiedTokenVocab<T>>) -> Self {
        let segmentor = data.segmentation.clone().into();

        let byte_table = data.pair_vocab.byte_table().clone();

        Self {
            data,
            segmentor,
            byte_table,
        }
    }
}

impl<T: TokenType> TokenVocabIndex<T> for UnifiedVocabEncoder<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.data.unordered_tokens_iter()
    }
    fn max_token(&self) -> T {
        self.data.max_token()
    }
}

impl<T: TokenType> TokenEncoder<T> for UnifiedVocabEncoder<T> {
    fn pattern(&self) -> String {
        self.data.segmentation.pattern()
    }

    fn special_vocab(&self) -> Option<&SpecialWordsTokenVocab<T>> {
        self.data.segmentation.special_vocab()
    }

    fn split_words<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        self.segmentor.split_words(text)
    }

    /// Encode a word chunk into token IDs.
    fn encode_append_span(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // NOTE: You may think that a bypass for single-byte words
        // would speed things up here, before the hash lookup.
        // On real sample data, it appears to incur a small *penalty*.

        // Correctness-wise - Some words may not exist in the pair mappings.
        //
        // Speed-wise - This is a wash; the hash is slow enough that the
        // cache hits don't speed us up.
        if let Some(token) = self.data.word_vocab.lookup_token(span) {
            tokens.push(token);
            return;
        }

        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        tokens.extend(span.iter().map(|&b| self.byte_table.get_token(b)));

        // Incrementally shrink the working memory (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| {
                    self.data
                        .pair_vocab
                        .pairs()
                        .get(&(w[0], w[1]))
                        .map(|&token| (token, idx))
                })
                .min()
            {
                // Adjust the window index.
                let idx = start + idx;

                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                tokens[idx] = token;
                tokens.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::DictionaryDecoder;
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::bpe_trainer::BinaryPairVocabTrainerOptions;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::TokenVocabIndex;
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use alloc::sync::Arc;
    use compact_str::CompactString;

    #[test]
    fn test_encoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BinaryPairVocabTrainerOptions::new(OA_GPT3_CL100K_WORD_PATTERN, 1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let mut trainer = options.init::<K, C>();
        trainer.update_from_samples(samples.iter());

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());

        let mut vocab = trainer.train(byte_table.clone()).unwrap();

        vocab.specials_vocab_mut().add_str_word("<|HI|>", 3000);

        let special_sample = "hello <|HI|> world";

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.clone().into());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
        check_is_send(&decoder);
        check_is_sync(&decoder);

        // Special handling.
        let tokens = encoder.encode(special_sample);
        assert_eq!(
            decoder.try_decode_to_string(tokens).unwrap(),
            special_sample
        );

        for sample in samples {
            let tokens = encoder.encode(sample);
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
        }
    }
}

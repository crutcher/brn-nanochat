//! # Encoder for [`UnifiedTokenVocab`].

use crate::encoders::token_encoder::TokenEncoder;
use crate::segmentation::text_segmentor::{SpanRef, TextSegmentor};
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

        // FIXME: this should be a property of the data.
        assert_eq!(&data.pair_vocab.byte_table(), &data.word_vocab.byte_table());

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
    ) -> Vec<SpanRef<'a>> {
        self.segmentor.split_spans(text)
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

        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..end]`.
        // - CURRENT[i] := tokens[start + i]
        self.byte_table.append_tokens(span, tokens);
        let mut end = tokens.len();

        // Define PAIR_RANKS as `tokens[end..]`
        // - there are `(end - start) - 1` items in PAIR_RANKS.
        // - PAIR_RANKS[i] := tokens[end + i]
        // - PAIR_RANKS[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))

        let get_pair_rank = {
            |tok: &mut [T], i: usize| match self
                .data
                .pair_vocab
                .pairs()
                .get(&(tok[start + i], tok[start + i + 1]))
            {
                Some(&token) => token,
                None => T::max_value(),
            }
        };

        for i in 1..(end - start) {
            let rank = get_pair_rank(tokens, i - 1);
            tokens.push(rank);
        }

        while let Some((t, i)) = tokens[end..]
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| {
                if t == T::max_value() {
                    None
                } else {
                    Some((t, i))
                }
            })
            .min()
        {
            // At this point, i selects CURRENT[i], PAIR_RANKS[i] such that:
            // - PAIR_RANKS[i] != max_value
            // - PAIR_RANKS[i] is smallest

            // We need to merge CURRENT[i..=i+1] and PAIR_RANKS[i..=i+1]

            // Set CURRENT[i] to the new target rank.
            tokens[start + i] = t;

            if i > 0 {
                // If there is a preceding token, recompute PAIR_RANKS[i-1].
                tokens[end + i - 1] = get_pair_rank(tokens, i - 1);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            // Order matters here for the indices.
            tokens.remove(end + i);
            tokens.remove(start + i + 1);

            end -= 1;

            if end + i < tokens.len() {
                // If there is a following token, recompute PAIR_RANKS[i].
                tokens[end + i] = get_pair_rank(tokens, i);
            }
        }

        // Drop the PAIR_RANKS buffer.
        tokens.truncate(end);
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

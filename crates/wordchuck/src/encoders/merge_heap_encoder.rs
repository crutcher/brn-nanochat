//! # Encoder for [`UnifiedTokenVocab`].

use crate::encoders::token_encoder::TokenEncoder;
use crate::regex::{RegexSupplierHandle, RegexWrapperHandle};
use crate::segmentation::text_segmentor::{SpanRef, TextSegmentor};
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use alloc::sync::Arc;

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct MergeHeapVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    /// Text Segmentor.
    pub segmentor: TextSegmentor,
}

impl<T: TokenType> MergeHeapVocabEncoder<T> {
    /// Construct an encoder from data.
    pub fn init<F>(
        data: Arc<UnifiedTokenVocab<T>>,
        re_factory: F,
    ) -> Self
    where
        F: Fn(RegexWrapperHandle) -> RegexSupplierHandle,
    {
        let segmentor = TextSegmentor::from_config(data.segmentation.clone(), re_factory);

        Self { data, segmentor }
    }

    /// Compiler Hint.
    fn lookup_token(
        &self,
        span: &[u8],
    ) -> Option<T> {
        self.data.lookup_token(span)
    }

    /// Compiler Hint.
    fn lookup_pair(
        &self,
        pair: &(T, T),
    ) -> Option<&T> {
        self.data.lookup_pair(pair)
    }

    /// Compiler Hint.
    fn append_tokens(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        self.data.byte_table().append_tokens(span, tokens);
    }
}

impl<T: TokenType> TokenVocabIndex<T> for MergeHeapVocabEncoder<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.data.unordered_tokens_iter()
    }

    fn max_token(&self) -> T {
        self.data.max_token()
    }
}

impl<T: TokenType> TokenEncoder<T> for MergeHeapVocabEncoder<T> {
    fn pattern(&self) -> String {
        self.data.segmentation.pattern()
    }

    fn special_vocab(&self) -> &SpecialWordsTokenVocab<T> {
        self.data.segmentation.special_vocab()
    }

    fn split_spans<'a>(
        &self,
        text: &'a str,
    ) -> Vec<SpanRef<'a>> {
        self.segmentor.split_spans(text)
    }

    fn encode_append_span(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // Correctness-wise - Some words may not exist in the pair mappings.
        if let Some(token) = self.lookup_token(span) {
            tokens.push(token);
            return;
        }

        // We reuse the output buffer as our working memory.
        // - `start` is the first index of the working memory buffer.
        let start = tokens.len();

        // Define CURRENT as `tokens[start..end]`.
        // - CURRENT[i] := tokens[start + i]
        self.append_tokens(span, tokens);
        let mut end = tokens.len();

        // Define PAIR_RANKS as `tokens[end..]`
        // - there are `(end - start) - 1` items in PAIR_RANKS.
        // - PAIR_RANKS[i] := tokens[end + i]
        // - PAIR_RANKS[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))

        let get_pair_rank = {
            |tok: &mut [T], i: usize| {
                let pair = &(tok[start + i], tok[start + i + 1]);
                match self.lookup_pair(pair) {
                    Some(&token) => token,
                    None => T::max_value(),
                }
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
    use crate::encoders::merge_heap_encoder::MergeHeapVocabEncoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::regex::default_regex_supplier;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::tooling::testing::build_test_vocab;
    use crate::vocab::{TokenVocabIndex, UnifiedTokenVocab};
    use alloc::sync::Arc;

    #[test]
    fn test_encoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());
        let segmentation = SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN);
        let vocab = build_test_vocab(byte_table.clone(), segmentation);

        let mut seg = vocab.segmentation.clone();
        seg.add_str_word("<|HI|>", 3000);

        let vocab: Arc<UnifiedTokenVocab<T>> =
            UnifiedTokenVocab::init(seg, vocab.span_vocab, vocab.pair_vocab).into();

        let special_sample = "hello <|HI|> world";

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone(), default_regex_supplier);
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 3000);

        let decoder = DictionaryDecoder::from_unified_vocab(vocab);
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

//! # Encoder for [`UnifiedTokenVocab`].

use crate::decoders::dictionary_decoder::DictionaryDecoder;
use crate::encoders::text_segmentor::{TextSegmentor, WordRef};
use crate::encoders::token_encoder::TokenEncoder;
use crate::types::TokenType;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use crate::vocab::word_vocab::WordMapTokenVocab;
use std::sync::Arc;

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct UnifiedVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    segmentor: TextSegmentor,
}

impl<T: TokenType> UnifiedVocabEncoder<T> {
    /// Construct an encoder from data.
    pub fn new(data: Arc<UnifiedTokenVocab<T>>) -> Self {
        let specials = match &data.specials {
            Some(specials) => specials
                .words
                .keys()
                .map(|word| String::from_utf8(word.clone()).unwrap())
                .collect::<Vec<String>>()
                .into(),
            None => None,
        };

        let segmentor = TextSegmentor::create(data.word_pattern.clone(), specials.as_deref());

        Self { data, segmentor }
    }

    /// Build a [`DictionaryDecoder`] from this [`UnifiedVocabEncoder`].
    pub fn to_decoder(&self) -> DictionaryDecoder<T> {
        self.data.to_decoder()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for UnifiedVocabEncoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.data.compound_tokens_iter()
    }
    fn max_token(&self) -> T {
        self.data.max_token()
    }
}

impl<T: TokenType> TokenEncoder<T> for UnifiedVocabEncoder<T> {
    fn pattern(&self) -> String {
        self.data.word_pattern.as_str().to_string()
    }

    fn special_vocab(&self) -> Option<&WordMapTokenVocab<T>> {
        self.data.specials.as_ref()
    }

    fn split_words<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        self.segmentor.split_words(text)
    }

    /// Encode a word chunk into token IDs.
    fn encode_append_word(
        &self,
        word: &str,
        tokens: &mut Vec<T>,
    ) {
        let chunk = word.as_bytes();
        if chunk.len() == 1 {
            tokens.push(T::from_u8(chunk[0]).unwrap());
            return;
        }

        if let Some(token) = self.data.word_vocab.lookup_token(chunk) {
            tokens.push(token);
            return;
        }

        // Reuse the output buffer as a stack.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        tokens.extend(chunk.iter().map(|&b| T::from_u8(b).unwrap()));

        // Incrementally shrink the "stack" (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the pair which merges to the lowest ranked token.
            let mut best_match: Option<(usize, T)> = None;

            for idx in start..tokens.len() - 1 {
                let pair = (tokens[idx], tokens[idx + 1]);

                if let Some(&merge_token) = self.data.pair_vocab.pairs.get(&pair)
                    && (best_match.is_none() || (merge_token < best_match.unwrap().1))
                {
                    best_match = Some((idx, merge_token));
                }
            }

            if let Some((idx, merge_token)) = best_match {
                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                tokens[idx] = merge_token;
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
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::{BinaryPairVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::TokenVocabIndex;
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_encoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BinaryPairVocabTrainer::new_with_vocab_size(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let TrainResults {
            word_pattern,
            pair_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let mut vocab: UnifiedTokenVocab<T> =
            UnifiedTokenVocab::new(word_pattern.into()).with_pair_vocab(pair_vocab);

        vocab.specials_vocab_mut().add_str_word("<|HI|>", 3000);

        let special_sample = "hello <|HI|> world";

        let encoder = UnifiedVocabEncoder::<T>::new(Arc::new(vocab));
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = encoder.to_decoder();
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

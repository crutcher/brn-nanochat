//! # Encoder for [`UnifiedTokenVocab`].

use crate::decoders::dictionary_decoder::DictionaryDecoder;
use crate::encoders::text_segmentor::{TextSegmentor, WordRef};
use crate::encoders::token_encoder::TokenEncoder;
use crate::types::TokenType;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use crate::vocab::word_vocab::WordMapTokenVocab;
use alloc::sync::Arc;

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct UnifiedVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    byte_table: [T; 256],
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

        let mut byte_table: [T; 256] = (0..=255)
            .map(|b| T::from_u8(b).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // If there are byte table overrides, apply them.
        data.word_vocab.words.iter().for_each(|(bs, &token)| {
            if bs.len() == 1 {
                byte_table[bs[0] as usize] = token;
            }
        });

        Self {
            data,
            segmentor,
            byte_table,
        }
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

        // NOTE: You may think that a bypass for single-byte words
        // would speed things up here, before the hash lookup.
        // On real sample data, it appears to incur a small *penalty*.

        // Correctness-wise - Some words may not exist in the pair mappings.
        //
        // Speed-wise - This is a wash; the hash is slow enough that the
        // cache hits don't speed us up.
        if let Some(token) = self.data.word_vocab.lookup_token(chunk) {
            tokens.push(token);
            return;
        }

        // Reuse the output buffer as our working memory.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        tokens.extend(chunk.iter().map(|&b| self.byte_table[b as usize]));

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
                        .pairs
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
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::BinaryPairVocabTrainerOptions;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::TokenVocabIndex;
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

        let mut vocab = trainer.train::<T>().unwrap();

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

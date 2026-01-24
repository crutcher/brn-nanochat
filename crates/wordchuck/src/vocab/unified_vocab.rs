//! # Unified Token Vocabulary

use crate::decoders::dictionary_decoder::DictionaryDecoder;
use crate::regex::RegexWrapperPattern;
use crate::types::TokenType;
use crate::vocab::byte_span_vocab::ByteSpanTokenMapVocab;
use crate::vocab::pair_vocab::PairTokenMapVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::AHashMap;

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Regex pattern for word splitting.
    pub word_pattern: RegexWrapperPattern,

    /// Special tokens vocabulary.
    pub specials: Option<ByteSpanTokenMapVocab<T>>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub word_vocab: ByteSpanTokenMapVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub pair_vocab: PairTokenMapVocab<T>,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Create a new default token vocabulary.
    ///
    /// # Arguments
    /// * `word_pattern`: Regex pattern for word splitting.
    pub fn new(word_pattern: RegexWrapperPattern) -> Self {
        Self {
            word_pattern,
            specials: None,
            word_vocab: Default::default(),
            pair_vocab: Default::default(),
        }
    }

    /// Mutable reference to the special tokens vocabulary.
    ///
    /// Will create the vocabulary if it doesn't exist.
    pub fn specials_vocab_mut(&mut self) -> &mut ByteSpanTokenMapVocab<T> {
        if self.specials.is_none() {
            self.specials = Some(Default::default());
        }
        self.specials.as_mut().unwrap()
    }

    /// Replace the word-split regex pattern.
    pub fn with_word_pattern(
        self,
        word_pattern: RegexWrapperPattern,
    ) -> Self {
        Self {
            word_pattern,
            ..self
        }
    }

    /// Shrinks the capacity of the underlying data structures to fit its current size.
    pub fn shrink_to_fit(&mut self) {
        self.word_vocab.shrink_to_fit();
    }

    /// Replace special tokens vocabulary.
    pub fn with_specials(
        self,
        specials: Option<ByteSpanTokenMapVocab<T>>,
    ) -> Self {
        Self { specials, ..self }
    }

    /// Replace the binary-pair vocabulary.
    pub fn with_pair_vocab(
        self,
        pair_vocab: PairTokenMapVocab<T>,
    ) -> Self {
        Self { pair_vocab, ..self }
    }

    /// Replace the word vocabulary.
    pub fn with_word_vocab(
        self,
        word_vocab: ByteSpanTokenMapVocab<T>,
    ) -> Self {
        Self { word_vocab, ..self }
    }

    /// Compiled expansion dictionary.
    pub fn compiled_dictionary(&self) -> AHashMap<T, Vec<u8>> {
        let mut export_vocab = self.word_vocab.clone();

        export_vocab.extend_from_pair_vocab(&self.pair_vocab, false);

        if let Some(specials) = &self.specials {
            for (chunk, &t) in specials.span_map().iter() {
                export_vocab.add_bytes_word(chunk.clone(), t);
            }
        }

        export_vocab
            .iter()
            .map(|(chunk, &token)| (token, chunk.to_vec()))
            .collect()
    }

    /// Compile the unified vocabulary into a dictionary decoders.
    pub fn to_decoder(&self) -> DictionaryDecoder<T> {
        DictionaryDecoder::new(self.compiled_dictionary())
    }
}

impl<T: TokenType> TokenVocabIndex<T> for UnifiedTokenVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        let mut tokens = self.pair_vocab.unordered_tokens_iter().collect::<Vec<_>>();

        tokens.extend(self.word_vocab.unordered_tokens_iter());

        tokens.into_iter()
    }

    fn max_token(&self) -> T {
        self.pair_vocab.max_token().max(self.word_vocab.max_token())
    }
}

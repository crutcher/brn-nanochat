//! # Unified Token Vocabulary

use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::pair_vocab::PairTokenMapVocab;
use crate::vocab::span_vocab::SpanTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::{AHashMap, AHashSet};

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Text Segmentation Configuration
    pub segmentation: SegmentationConfig<T>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub word_vocab: SpanTokenVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub pair_vocab: PairTokenMapVocab<T>,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Build a new [`UnifiedTokenVocab`] from a [`SpanTokenVocab`].
    pub fn from_span_vocab(
        segmentation: SegmentationConfig<T>,
        span_vocab: SpanTokenVocab<T>,
    ) -> Self {
        let pair_vocab = span_vocab.to_pair_vocab();
        Self::init(segmentation, span_vocab, pair_vocab)
    }

    /// Build a new [`UnifiedTokenVocab`] from a [`PairTokenMapVocab`].
    pub fn from_pair_vocab(
        segmentation: SegmentationConfig<T>,
        pair_vocab: PairTokenMapVocab<T>,
    ) -> Self {
        let word_vocab = pair_vocab
            .to_span_pairs()
            .collect::<SpanTokenMap<T>>()
            .into();
        Self::from_span_vocab(segmentation, word_vocab)
    }

    /// Initialize a [`UnifiedTokenVocab`].
    pub fn init(
        segmentation: SegmentationConfig<T>,
        word_vocab: SpanTokenVocab<T>,
        pair_vocab: PairTokenMapVocab<T>,
    ) -> Self {
        assert_eq!(word_vocab.byte_table(), pair_vocab.byte_table());

        let tokens = word_vocab.unordered_tokens_iter().collect::<AHashSet<_>>();
        for ((a, b), c) in pair_vocab.pairs() {
            for t in [a, b, c].iter() {
                assert!(
                    tokens.contains(t),
                    "pair token {t:?} not found in word vocab"
                );
            }
        }
        for t in segmentation.specials.unordered_tokens_iter() {
            assert!(
                !tokens.contains(&t),
                "special token {t:?} found in word vocab"
            );
        }

        Self {
            segmentation,
            word_vocab,
            pair_vocab,
        }
    }

    /// Extend the vocabulary with the given special words.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        Self {
            segmentation: self.segmentation.with_special_words(special_words),
            ..self
        }
    }

    /// Compiled expansion dictionary.
    pub fn unified_dictionary(&self) -> AHashMap<T, Vec<u8>> {
        let mut tmp = AHashMap::default();

        self.word_vocab.iter().for_each(|(chunk, &token)| {
            tmp.insert(chunk.clone(), token);
        });

        for (span, token) in self.pair_vocab.to_span_pairs() {
            if tmp.contains_key(&span) {
                continue;
            }
            tmp.insert(span, token);
        }

        if let Some(specials) = &self.segmentation.special_vocab() {
            for (chunk, &t) in specials.span_map().iter() {
                tmp.insert(chunk.clone(), t);
            }
        }

        tmp.into_iter()
            .map(|(chunk, token)| (token, chunk))
            .collect()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for UnifiedTokenVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.word_vocab
            .unordered_tokens_iter()
            .chain(self.segmentation.specials.unordered_tokens_iter())
    }
}

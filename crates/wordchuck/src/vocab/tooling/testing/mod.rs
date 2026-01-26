//! # Vocab Testing Tools

use crate::segmentation::SegmentationConfig;
use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::{ByteTokenTable, SpanTokenVocab, UnifiedTokenVocab};
use std::sync::Arc;

/// Create a new test vocabulary.
pub fn new_test_vocab<T: TokenType, C>(
    byte_table: Arc<ByteTokenTable<T>>,
    segmentation: C,
) -> UnifiedTokenVocab<T>
where
    C: Into<SegmentationConfig<T>>,
{
    let span_map: SpanTokenMap<T> = Default::default();
    let span_vocab = SpanTokenVocab::init(byte_table, span_map).unwrap();

    UnifiedTokenVocab::from_span_vocab(segmentation.into(), span_vocab)
}

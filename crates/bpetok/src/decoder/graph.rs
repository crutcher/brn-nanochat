//! # Graph Decoder

use crate::data::TokenizerData;
use crate::{ExpansionMap, MergeMap, TokenDecoder, TokenType};
use ahash::AHashMap;
use std::collections::hash_map;
use std::ops::Range;

/// A decoder for [`Tokenizer`] decoder with a production graph.
#[derive(Clone)]
pub struct GraphDecoder<T: TokenType> {
    /// Token to pair mapping.
    ///
    /// Does not include byte-tokens.
    pub graph: ExpansionMap<T>,
}

impl<T: TokenType> GraphDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut graph: ExpansionMap<T>) -> Self {
        graph.shrink_to_fit();
        Self { graph }
    }

    /// Build a [`GraphDecoder`] from this [`TokenizerData`].
    #[tracing::instrument(skip(data))]
    pub fn from_data(data: &TokenizerData<T>) -> Self {
        Self::from_merge_map(&data.merge_map)
    }

    /// Build a [`GraphDecoder`] from this [`Tokenizer`].
    #[tracing::instrument(skip(merge_map))]
    pub fn from_merge_map(merge_map: &MergeMap<T>) -> Self {
        let expansion_map =
            AHashMap::from_iter(merge_map.iter().map(|(&pair, &token)| (token, pair)));
        Self::new(expansion_map)
    }
}

impl<T: TokenType> TokenDecoder<T> for GraphDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.graph.keys().copied()
    }

    #[tracing::instrument(skip(self, buf, tokens))]
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    ) {
        let mut tokens = tokens.iter();
        let mut stack: Vec<T> = Vec::with_capacity(16);
        while let Some(t) = stack.pop().or_else(|| tokens.next().copied()) {
            if let Some(b) = t.to_u8() {
                buf.push(b);
                continue;
            }
            let (a, b) = self.graph.get(&t).expect("Token not found in slice map");
            stack.push(*b);
            stack.push(*a);
        }
    }

    fn size_estimate(&self) -> usize {
        size_of::<hash_map::Entry<T, Range<usize>>>() * self.graph.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;
    use crate::builder::TokenizerBuilder;
    use crate::types::{check_is_send, check_is_sync};
    use compact_str::CompactString;

    #[test]
    fn test_graph_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = TokenizerBuilder::with_capacity(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let data = options.train_from_sample_iterator::<T, K, C, _>(samples.iter());
        let tokenizer = Tokenizer::new(data.clone());

        let decoder = GraphDecoder::from_data(&data);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = tokenizer.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}

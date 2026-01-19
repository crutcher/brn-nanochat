//! Byte Pair Merge

use crate::types::{TokenType, WordToTokenMap};

/// The start index of a token.
pub struct TokenSpanIndex<T: TokenType> {
    rank: T,
    index: usize,
}

impl<T: TokenType> TokenSpanIndex<T> {
    fn new(
        index: usize,
        rank: T,
    ) -> Self {
        Self { index, rank }
    }
}

fn _byte_pair_merge<T: TokenType>(
    word_map: &WordToTokenMap<T>,
    piece: &[u8],
    spans: &mut Vec<TokenSpanIndex<T>>,
    tokens: &mut Vec<T>,
) {
    let max_token = T::max_value();
    let mut min_span = TokenSpanIndex::new(usize::MAX, max_token);

    spans.clear();

    // This assumes that each byte has a unique token in `word_map`.

    for i in 0..piece.len() - 1 {
        let rank = *word_map.get(&piece[i..i + 2]).unwrap_or(&max_token);
        if rank < min_span.rank {
            min_span = TokenSpanIndex::new(i, rank);
        }
        spans.push(TokenSpanIndex::new(i, rank));
    }
    spans.push(TokenSpanIndex::new(piece.len() - 1, max_token));
    spans.push(TokenSpanIndex::new(piece.len(), max_token));

    // `spans` indexes `piece` into `spans.len() - 1` spans.
    //
    // `spans[-i]` is an end marker and does not correspond to a span.
    //
    // - `piece[spans[i].index..spans[i + 1].index]` is the byte content of the i-th span.
    // - each span exists in `word_map` and has a rank.
    //
    // - `spans[i].rank` is the rank of the token that would result
    //   from merging the i-th and i+1-th span.

    let get_rank = {
        |spans: &Vec<TokenSpanIndex<T>>, i: usize| {
            if (i + 3) < spans.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // spans[i + 1], see comment in the main loop.
                *word_map
                    .get(&piece[spans[i].index..spans[i + 3].index])
                    .unwrap_or(&max_token)
            } else {
                max_token
            }
        }
    };

    // While there are spans which can be merged,
    // merge the pair with the lowest rank.

    while min_span.rank != max_token {
        let i = min_span.index;
        // Update spans[i] and spans[i - 1] before removing spans[i + 1], since
        // `spans.remove(i + 1)` will thrash the cache.
        if i > 0 {
            spans[i - 1].rank = get_rank(spans, i - 1);
        }
        spans[i].rank = get_rank(spans, i);
        spans.remove(i + 1);

        min_span = TokenSpanIndex::new(usize::MAX, max_token);
        for (i, &TokenSpanIndex { rank, .. }) in spans[..spans.len() - 1].iter().enumerate() {
            if rank < min_span.rank {
                min_span = TokenSpanIndex::new(i, rank);
            }
        }
    }

    // Each remaining span is a single token.

    spans
        .windows(2)
        .for_each(|span| tokens.push(word_map[&piece[span[0].index..span[1].index]]));
}

/// Byte Pair Merge encoding, appending to a buffer.
pub fn byte_pair_encode_append<T: TokenType>(
    piece: &[u8],
    ranks: &WordToTokenMap<T>,
    tokens: &mut Vec<T>,
) {
    if piece.len() == 1 {
        tokens.push(ranks[piece]);
        return;
    }

    let mut parts = Vec::with_capacity(piece.len());

    _byte_pair_merge(ranks, piece, &mut parts, tokens);
}

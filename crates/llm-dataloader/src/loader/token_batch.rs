use wordchipper::TokenType;

pub struct TokenBatch<T: TokenType> {
    pub batch: Vec<Vec<T>>,
}

impl<T: TokenType> TokenBatch<T> {
    pub fn total_tokens(&self) -> usize {
        self.batch.iter().map(|x| x.len()).sum()
    }
}

//! # Tiktoken IO

use crate::types::{TokenType, VocabMap};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Save the chunk map to a tiktoken vocab file.
pub fn save_tiktoken_vocab<T: TokenType>(
    vocab_map: &VocabMap<T>,
    path: &str,
) -> anyhow::Result<()> {
    let mut vocab: Vec<_> = vocab_map.iter().collect();
    vocab.sort_by_key(|(_, t)| **t);

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for (chunk, token) in vocab {
        writeln!(
            writer,
            "{} {}",
            BASE64_STANDARD.encode(chunk),
            token.to_u64().unwrap()
        )?;
    }

    Ok(())
}

/// Load a tiktoken vocab file into a [`VocabMap`].
pub fn load_tiktoken_vocab<T: TokenType>(path: &str) -> anyhow::Result<VocabMap<T>> {
    let mut vocab = VocabMap::new();
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line: String = line?;
        let s = line.as_str();

        let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
        assert_eq!(parts.len(), 2);

        let chunk = parts[0];
        let chunk = BASE64_STANDARD.decode(chunk).unwrap();

        let token: u64 = parts[1].parse()?;
        let token = T::from_u64(token).unwrap();

        vocab.insert(chunk, token);
    }
    Ok(vocab)
}

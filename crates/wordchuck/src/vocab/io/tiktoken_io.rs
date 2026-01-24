//! # Tiktoken Vocabulary IO

use crate::types::{ByteSpanTokenMap, TokenType};
use ahash::AHashMap;
use anyhow::Context;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Load a [`ByteSpanTokenMap`] from a tiktoken vocab file.
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_span_map_from_tiktoken_path<T, P>(path: P) -> anyhow::Result<ByteSpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    load_span_map_from_tiktoken_reader(reader)
}

/// Update a [`ByteSpanTokenMap`] from a tiktoken vocab [`BufRead`] stream.
///
/// # Arguments
/// * `span_map` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn load_span_map_from_tiktoken_reader<T, R>(reader: R) -> anyhow::Result<ByteSpanTokenMap<T>>
where
    T: TokenType,
    R: BufRead,
{
    let mut vocab: AHashMap<Vec<u8>, T> = Default::default();

    let stream = reader.lines();
    for line in stream {
        let line = line?;
        let s: &str = line.as_ref();

        let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
        assert_eq!(parts.len(), 2);

        let chunk = parts[0];
        let chunk = BASE64_STANDARD.decode(chunk)?;

        let token: u64 = parts[1].parse()?;
        let token = T::from_u64(token).context("token out of range")?;

        vocab.insert(chunk, token);
    }

    Ok(vocab)
}

/// Save a [`ByteSpanTokenMap`] to a tiktoken vocab file.
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_span_map_to_tiktoken_path<T: TokenType, P: AsRef<Path>>(
    span_map: &ByteSpanTokenMap<T>,
    path: P,
) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    save_span_map_to_tiktoken_writer(span_map, &mut writer)
}

/// Save a [`ByteSpanTokenMap`] to a [`Write`] writer.
pub fn save_span_map_to_tiktoken_writer<T, W>(
    span_map: &ByteSpanTokenMap<T>,
    writer: &mut W,
) -> anyhow::Result<()>
where
    T: TokenType,
    W: Write,
{
    let mut items: Vec<(T, &Vec<u8>)> = span_map
        .iter()
        .map(|(chunk, &token)| (token, chunk))
        .collect();
    items.sort_by_key(|(t, _)| *t);

    for (token, chunk) in items {
        writeln!(
            writer,
            "{} {}",
            BASE64_STANDARD.encode(chunk),
            token.to_u64().unwrap()
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::ByteSpanTokenMapVocab;

    #[test]
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut vocab = ByteSpanTokenMapVocab::<T>::default();
        vocab.add_str_word("apple", 300);
        vocab.add_str_word("banana", 301);
        vocab.add_str_word("pear", 302);

        let span_map = vocab.span_map();

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_span_map_to_tiktoken_path(span_map, &path).expect("Failed to save vocab");

                let loaded_vocab =
                    load_span_map_from_tiktoken_path(&path).expect("Failed to load vocab");

                assert_eq!(&loaded_vocab, span_map);

                Ok(())
            })
            .unwrap();
    }
}

//! # Tiktoken Vocabulary IO

use crate::types::TokenType;
use crate::vocab::WordMapTokenVocab;
use anyhow::Context;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Load a [`WordMapTokenVocab`] from a tiktoken vocab file.
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_word_map_from_tiktoken_path<T, P>(path: P) -> anyhow::Result<WordMapTokenVocab<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut vocab = WordMapTokenVocab::default();
    update_word_map_from_tiktoken_reader(&mut vocab, reader)?;

    Ok(vocab)
}

/// Update a [`WordMapTokenVocab`] from a tiktoken vocab [`BufRead`] stream.
///
/// # Arguments
/// * `vocab` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn update_word_map_from_tiktoken_reader<T, R>(
    vocab: &mut WordMapTokenVocab<T>,
    reader: R,
) -> anyhow::Result<()>
where
    T: TokenType,
    R: BufRead,
{
    update_word_map_from_tiktoken_iter(vocab, reader.lines())
}

/// Update a [`WordMapTokenVocab`] from a tiktoken vocab file.
///
/// # Arguments
/// * `vocab` - the vocabulary to extend.
/// * `stream` - the line iterator.
pub fn update_word_map_from_tiktoken_iter<T>(
    vocab: &mut WordMapTokenVocab<T>,
    stream: impl Iterator<Item = std::io::Result<String>>,
) -> anyhow::Result<()>
where
    T: TokenType,
{
    for line in stream {
        let line = line?;
        let s: &str = line.as_ref();

        let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
        assert_eq!(parts.len(), 2);

        let chunk = parts[0];
        let chunk = BASE64_STANDARD.decode(chunk)?;

        let token: u64 = parts[1].parse()?;
        let token = T::from_u64(token).context("token out of range")?;

        vocab.add_bytes_word(chunk, token);
    }
    Ok(())
}

/// Save a [`WordMapTokenVocab`] to a tiktoken vocab file.
///
/// # Arguments
/// * `vocab` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_word_map_to_tiktoken_path<T: TokenType, P: AsRef<Path>>(
    vocab: &WordMapTokenVocab<T>,
    path: P,
) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    save_word_map_to_tiktoken_writer(vocab, &mut writer)
}

/// Save a [`WordMapTokenVocab`] to a [`Write`] writer.
pub fn save_word_map_to_tiktoken_writer<T, W>(
    vocab: &WordMapTokenVocab<T>,
    writer: &mut W,
) -> anyhow::Result<()>
where
    T: TokenType,
    W: Write,
{
    let mut items: Vec<(T, &Vec<u8>)> =
        vocab.iter().map(|(chunk, &token)| (token, chunk)).collect();
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

    #[test]
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut vocab = WordMapTokenVocab::<T>::default();
        vocab.add_str_word("apple", 300);
        vocab.add_str_word("banana", 301);
        vocab.add_str_word("pear", 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_word_map_to_tiktoken_path(&vocab, &path).expect("Failed to save vocab");

                let loaded_vocab =
                    load_word_map_from_tiktoken_path(&path).expect("Failed to load vocab");

                assert_eq!(&vocab, &loaded_vocab);

                Ok(())
            })
            .unwrap();
    }
}

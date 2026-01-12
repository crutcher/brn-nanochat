use arrow::array::StringArray;
use bpetok::decoder::TokenDecoder;
use bpetok::decoder::corpus_decoder::CorpusDecoder;
use bpetok::decoder::dictionary_decoder::DictionaryDecoder;
use bpetok::decoder::expansion_decoder::ExpansionDecoder;
use bpetok::tokenizer::{EncoderData, ScanningEncoder, TokenEncoder};
use bpetok::types::TokenType;
use bpetok::vocab::data::{TokenVocab, WordMapTokenVocab};
use bpetok::vocab::training::trainer::{BPETokenVocabTrainer, TrainResults};
use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use compact_str::CompactString;
use nanochat_data::dataset::DatasetCacheConfig;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Example tokenizer trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "..8")]
    pub shards: Vec<Slice>,

    /// Vocab size.
    #[arg(long, default_value = "65536")]
    pub vocab_size: usize,

    /// Time the avg encode/decode.
    #[arg(long, default_value = "false")]
    pub time_encode_decode: bool,

    /// Encode/Decode Batch size.
    #[arg(long, default_value = "512")]
    pub batch_size: usize,

    /// Optional Tiktoken save path.
    #[arg(long)]
    pub tiktoken_save_path: Option<String>,

    /// Number of timing batches to use.
    #[arg(long, default_value = "20")]
    pub num_timing_batches: usize,

    /// Enable verbose output.
    #[arg(long, default_value = "false")]
    pub verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.verbose {
        println!("{:#?}", args);
    }

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir);
    if args.verbose {
        println!("{:#?}", cache_config);
    }

    let shards: Vec<usize> = {
        let max_shard = cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for slice in &args.shards {
            for idx in slice.into_iter() {
                let shard = idx.expect_elem_index(max_shard);
                collected.insert(shard);
            }
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    let mut cache = cache_config.init()?;

    // TODO: `indicatif` for optional progress bar for users waiting on this.
    println!("Loading Shards: {shards:?}");
    println!("...");
    cache.load_shards(&shards)?;

    type T = u32;
    type C = u32;
    type K = CompactString;

    println!();
    println!("Training Tokenizer on shards: {:?}", shards);
    let t0 = std::time::Instant::now();

    let cache_ref = &cache;

    // Note: rather than repeating this, this should be a func to generate the Iterator.
    // But I can't work out the lifetimes.
    let samples = shards.clone().into_iter().flat_map(move |shard| {
        cache_ref
            .read_cached_batches(shard)
            .expect("failed to read batch")
            .flat_map(|batch| {
                batch
                    .iter()
                    .flat_map(|sample| {
                        let text = sample
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        text.iter()
                            .map(|s| s.unwrap().to_string())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
    });

    // TODO: `indicatif` for optional progress bar for users waiting on this.
    let TrainResults::<T> {
        word_pattern,
        bpe_vocab,
    } = BPETokenVocabTrainer::new_with_vocab_size(args.vocab_size)
        .train_vocab_from_sample_iter::<T, K, C, _>(samples)
        .expect("training failed");

    let training_duration = std::time::Instant::now().duration_since(t0);
    println!("- training_duration: {:#?}", training_duration);
    println!("- vocab_size: {:#?}", bpe_vocab.max_token());

    let word_vocab = WordMapTokenVocab::from_bpe(&bpe_vocab);

    let encoder_data = Arc::new(EncoderData {
        word_pattern: word_pattern.into(),
        word_vocab,
        bpe_vocab,
    });

    let encoder: ScanningEncoder<T> =
        ScanningEncoder::<T>::new(encoder_data.clone(), Default::default());

    if let Some(path) = args.tiktoken_save_path {
        encoder_data.word_vocab.save_to_tiktoken_vocab(&path)?;
        println!("- tiktoken vocab: {path:?}");
    }

    if args.time_encode_decode {
        let mut samples = Vec::new();
        {
            for batch in cache
                .read_cached_batches(shards[0])?
                .take(args.num_timing_batches)
            {
                let batch = batch?;
                let column = batch
                    .column_by_name("text")
                    .expect("failed to find 'text' column in batch")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();

                for val in column {
                    let val = val.unwrap().to_string();
                    samples.push(val);
                }
            }
        }

        println!();
        println!("Samples Summary:");
        let sample_count = samples.len();
        println!("- count: {}", sample_count);
        let avg_size = samples.iter().map(|s| s.len()).sum::<usize>() / sample_count;
        println!("- avg size: {avg_size}");

        let sample_batches: Vec<&[String]> = samples.chunks(args.batch_size).collect::<Vec<_>>();
        let num_batches = sample_batches.len();

        println!();
        println!("Timing Config:");
        println!("- batch size: {}", args.batch_size);

        println!();
        println!("Timing CPSEncoder Encode:");
        let mut token_batches: Vec<Vec<Vec<T>>> = Vec::with_capacity(sample_batches.len());
        {
            let batch_times_ns = sample_batches.iter().map(|batch| {
                let t0 = std::time::Instant::now();
                let token_batch: Vec<Vec<T>> = encoder.encode_batch(batch);
                let t1 = std::time::Instant::now();

                token_batches.push(token_batch);

                t1.duration_since(t0).as_nanos() as u64
            });

            let avg_batch_time_ns = batch_times_ns.sum::<u64>() / num_batches as u64;
            println!(
                "- batch avg: {:#?}",
                Duration::from_nanos(avg_batch_time_ns)
            );

            let avg_sample_time_ns = avg_batch_time_ns / args.batch_size as u64;
            println!(
                "- sample avg: {:#?}",
                Duration::from_nanos(avg_sample_time_ns)
            );

            let b_p_ns = avg_size as f64 / avg_sample_time_ns as f64;
            let b_p_s = b_p_ns * 1e9;
            let mb_p_s = b_p_s / 1e6;
            println!("- avg bps: {:.2} MB/s", mb_p_s);
        }

        println!();
        let expansion_decoder = ExpansionDecoder::from_bpe(&encoder_data.bpe_vocab);
        time_decoder(
            "ExpansionDecoder",
            &expansion_decoder,
            &sample_batches,
            &token_batches,
            args.batch_size,
        );

        println!();
        let dict_decoder = DictionaryDecoder::from_bpe(&encoder_data.bpe_vocab);
        time_decoder(
            "DictionaryDecoder",
            &dict_decoder,
            &sample_batches,
            &token_batches,
            args.batch_size,
        );

        println!();
        let corpus_decoder = CorpusDecoder::from_bpe(&encoder_data.bpe_vocab);
        time_decoder(
            "CorpusDecoder",
            &corpus_decoder,
            &sample_batches,
            &token_batches,
            args.batch_size,
        );
    }

    Ok(())
}

fn time_decoder<T: TokenType, D: TokenDecoder<T>>(
    name: &str,
    decoder: &D,
    sample_batches: &[&[String]],
    token_batches: &[Vec<Vec<T>>],
    batch_size: usize,
) {
    let num_batches = token_batches.len();
    println!("Timing Decode: {name}");
    println!("- decoder est bytes: {}", decoder.size_estimate());

    let batch_times_ns = sample_batches
        .iter()
        .zip(token_batches.iter())
        .map(|(sample, batch)| {
            let t0 = std::time::Instant::now();
            let decoded_sample = decoder.decode_batch_to_strings(batch);
            let t1 = std::time::Instant::now();
            let delta = t1.duration_since(t0);

            assert_eq!(sample, &decoded_sample);

            delta.as_nanos() as u64
        });

    let avg_batch_time_ns = batch_times_ns.sum::<u64>() / num_batches as u64;
    println!("- batch avg: {:?}", Duration::from_nanos(avg_batch_time_ns));

    let avg_sample_time_ns = avg_batch_time_ns / batch_size as u64;
    println!(
        "- sample avg: {:?}",
        Duration::from_nanos(avg_sample_time_ns)
    );
}

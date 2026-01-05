use arrow::array::StringArray;
use bpetok::decoder::TokenDecoder;
use bpetok::decoder::corpus_decoder::CorpusDecoder;
use bpetok::decoder::dictionary_decoder::DictionaryDecoder;
use bpetok::decoder::expansion_decoder::ExpansionDecoder;
use bpetok::tokenizer::{CPSEncoder, TokenEncoder};
use bpetok::types::TokenType;
use bpetok::vocab::data::TokenVocabData;
use bpetok::vocab::training::trainer::VocabTrainer;
use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use compact_str::CompactString;
use nanochat_data::dataset::DatasetCacheConfig;
use std::collections::HashSet;
use std::time::Duration;

/// Example tokenizer trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Vocab size.
    #[arg(long, default_value = "100_000")]
    pub vocab_size: usize,

    /// Time the avg encode/decode.
    #[arg(long, default_value = "false")]
    pub time_encode_decode: bool,

    /// Encode/Decode Batch size.
    #[arg(long, default_value = "32")]
    pub batch_size: usize,

    /// Optional Tiktoken save path.
    #[arg(long)]
    pub tiktoken_save_path: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("{:#?}", args);

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir);
    println!("{:#?}", cache_config);

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

    println!("Loading Shards ...: {shards:?}");
    cache.load_shards(&shards)?;

    type T = u32;
    type C = u32;
    type K = CompactString;

    let trainer = VocabTrainer::new_with_vocab_size(args.vocab_size);

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

    let data: TokenVocabData<T> = trainer.train_vocab_from_sample_iter::<T, K, C, _>(samples);
    let tokenizer = CPSEncoder::new(data.clone(), Default::default());

    let training_duration = std::time::Instant::now().duration_since(t0);
    println!("- training_duration: {:#?}", training_duration);
    println!("- vocab_size: {:#?}", tokenizer.max_token());
    println!("- size_estimate: {:#?}", tokenizer.size_estimate());

    if let Some(path) = args.tiktoken_save_path {
        tokenizer.save_tiktoken_vocab(&path)?;
        println!("- tiktoken vocab: {path:?}");
    }

    if args.time_encode_decode {
        // TODO: `indicatif` for optional progress bar for users waiting on this.

        let mut samples = Vec::new();
        let num_batches = 8;
        for batch in cache.read_cached_batches(shards[0])?.take(num_batches) {
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
        println!();
        println!("Samples Summary:");
        let count = samples.len();
        println!("- count: {}", count);
        let avg_size = samples.iter().map(|s| s.len()).sum::<usize>() / count;
        println!("- avg size: {avg_size}");

        let sample_batches: Vec<&[String]> = samples.chunks(args.batch_size).collect::<Vec<_>>();

        println!();
        println!("Timing Config:");
        println!("- batch size: {}", args.batch_size);

        println!();
        println!("Timing CPSEncoder Encode:");
        let mut token_batches: Vec<Vec<Vec<T>>> = Vec::with_capacity(sample_batches.len());
        {
            let times_ns = sample_batches.iter().map(|batch| {
                let t0 = std::time::Instant::now();
                let token_batch: Vec<Vec<T>> = tokenizer.encode_batch(batch);
                let t1 = std::time::Instant::now();

                token_batches.push(token_batch);

                t1.duration_since(t0).as_nanos() as u64
            });
            let avg_ns = times_ns.sum::<u64>() / count as u64;
            println!("- batch avg: {:#?}", Duration::from_nanos(avg_ns));
            println!(
                "- sample avg: {:#?}",
                Duration::from_nanos(avg_ns / args.batch_size as u64)
            );
        }

        println!();
        let expansion_decoder = ExpansionDecoder::from_data(&data);
        time_decoder(
            "ExpansionDecoder",
            &expansion_decoder,
            &sample_batches,
            &token_batches,
            args.batch_size,
        );

        println!();
        let dict_decoder = DictionaryDecoder::from_tokenizer(&expansion_decoder);
        time_decoder(
            "DictionaryDecoder",
            &dict_decoder,
            &sample_batches,
            &token_batches,
            args.batch_size,
        );

        println!();
        let corpus_decoder = CorpusDecoder::from_data(&data);
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
    let count = token_batches.len();
    println!("Timing Decode: {name}");
    println!("- decoder est bytes: {}", decoder.size_estimate());

    let times_ns = sample_batches
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

    let avg_ns = times_ns.sum::<u64>() / count as u64;
    println!("- batch avg: {:?}", Duration::from_nanos(avg_ns));
    println!(
        "- sample avg: {:?}",
        Duration::from_nanos(avg_ns / batch_size as u64)
    );
}

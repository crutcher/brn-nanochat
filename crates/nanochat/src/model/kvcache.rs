//! # KV Cache

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::module::Module;
use burn::prelude::{Backend, s};
use burn::tensor::DType;

/// KV Cache
#[derive(Module, Debug)]
pub struct KVCache<B: Backend> {
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,

    pos: usize,
    cache: Option<Tensor<B, 6>>,
}

impl<B: Backend> KVCache<B> {
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Insert and extend a (k, v) pair.
    ///
    /// # Arguments
    /// - `layer_idx`: the block layer index.
    /// - `k`: the ``[B, H_kv, T, D]`` key tensor.
    /// - `v`: the ``[B, H_kv, T, D]`` value tensor.
    ///
    /// # Returns
    /// - the extended (k, v) ``[B, H_kv, T, D]`` pair.
    pub fn insert_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [t_add] = unpack_shape_contract!(
            ["B", "H_kv", "T_add", "D"],
            &k.dims(),
            &["T_add"],
            &[
                ("B", self.batch_size),
                ("H_kv", self.num_heads),
                ("D", self.head_dim)
            ]
        );
        assert_shape_contract_periodically!(
            ["B", "H_kv", "T_add", "D"],
            &v.dims(),
            &[
                ("B", self.batch_size),
                ("H_kv", self.num_heads),
                ("T_add", t_add),
                ("D", self.head_dim)
            ]
        );

        let dtype = k.dtype();
        let device = k.device();

        let mut cache = if let Some(cache) = self.cache.take() {
            cache
        } else {
            self.allocate(self.seq_len, dtype, &device)
        };

        let t0 = self.pos;
        let t1 = t0 + t_add;

        // Grow the cache if needed.
        let cached_size = cache.dims()[4];
        if t1 > cached_size {
            let needed_t = self.growth_target(t1);

            cache = self
                .allocate(needed_t, dtype, &device)
                .slice_assign(s![.., .., .., .., ..cached_size, ..], cache);
        }

        // Insert k, v into the cache.
        cache = cache
            .slice_assign(s![layer_idx, 0, .., .., t0..t1], k.unsqueeze())
            .slice_assign(s![layer_idx, 1, .., .., t0..t1], v.unsqueeze());

        // Return the full cached keys/values up to the current position.
        let k = cache
            .clone()
            .slice(s![layer_idx, 0, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);
        let v = cache
            .clone()
            .slice(s![layer_idx, 1, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);

        self.cache = cache.into();

        // Increment pos after the last layer.
        if layer_idx == self.num_layers - 1 {
            self.pos = t1;
        }

        (k, v)
    }

    /// Prefill given another `KVCache`.
    ///
    /// - This cache must be `None`.
    /// - The other cache must be `Some`.
    /// - The `num_layers`, `num_heads`, and `head_dim` must match.
    /// - The `batch_size` must match, or `other.batch_size` must be 1.
    pub fn prefill(
        &mut self,
        other: &KVCache<B>,
    ) {
        assert!(self.cache.is_none(), "Cannot prefill a non-empty KV cache.");
        assert!(
            other.cache.is_some(),
            "Cannot prefill from a None KV cache."
        );

        assert_eq!(self.num_layers, other.num_layers);
        assert_eq!(self.num_heads, other.num_heads);
        assert_eq!(self.head_dim, other.head_dim);

        if self.batch_size != other.batch_size && other.batch_size != 1 {
            panic!(
                "Incompatible pre-fill batch size: {} vs {}",
                self.batch_size, other.batch_size
            );
        }
        assert!(self.seq_len >= other.seq_len);

        let other_cache = other.cache.as_ref().unwrap();

        let cache = self.allocate(other.seq_len, other_cache.dtype(), &other_cache.device());

        let source = other_cache.clone();
        let mut source_shape = source.dims();
        source_shape[2] = self.batch_size;
        let other_cache = source.expand(source_shape);

        self.cache = cache
            .slice_assign(s![.., .., .., .., ..other.pos, ..], other_cache)
            .into();
        self.pos = other.pos;
    }

    fn allocate(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Tensor<B, 6> {
        Tensor::<B, 6>::empty(
            [
                self.num_layers,
                2,
                self.batch_size,
                self.num_heads,
                seq_len,
                self.head_dim,
            ],
            device,
        )
        .cast(dtype)
    }

    pub fn growth_target(
        &self,
        size: usize,
    ) -> usize {
        const CHUNK_SIZE: usize = 1024;
        (size.div_ceil(CHUNK_SIZE) + 1) * CHUNK_SIZE
    }
}

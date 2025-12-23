//! GPT Block MLP

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};

/// Common meta for [`MLP`] and [`MLPConfig`].
pub trait MLPMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`MLP`].
#[derive(Config, Debug)]
pub struct MLPConfig {
    /// Embedding Size.
    pub n_embed: usize,

    /// Internal Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl MLPMeta for MLPConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }
}

impl MLPConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> MLP<B> {
        MLP {
            c_fc: LinearConfig::new(self.n_embed(), self.hidden_size())
                .with_bias(false)
                .init(device),
            act: self.activation.init(device),
            c_proj: LinearConfig::new(self.hidden_size(), self.n_embed())
                .with_bias(false)
                .init(device),
        }
    }

    /// Return the size of the hidden layer.
    pub fn hidden_size(&self) -> usize {
        self.n_embed * self.expansion_factor
    }
}

/// GPT Block MLP Module
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub c_fc: Linear<B>,
    pub act: Activation<B>,
    pub c_proj: Linear<B>,
}

impl<B: Backend> MLPMeta for MLP<B> {
    fn n_embed(&self) -> usize {
        self.c_fc.weight.dims()[0]
    }
}

impl<B: Backend> MLP<B> {
    /// MLP Forward Pass.
    ///
    /// # Arguments
    /// - `x`: a ``[batch, time, embed]`` input.
    ///
    /// # Returns
    /// a ``[batch, time, embed]`` result.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, time] = unpack_shape_contract!(
            ["batch", "time", "embed"],
            &x.dims(),
            &["batch", "time"],
            &[("embed", self.n_embed())]
        );

        let x = self.c_fc.forward(x);
        let x = self.act.forward(x);
        let x = x.square();
        let x = self.c_proj.forward(x);

        assert_shape_contract_periodically!(
            ["batch", "time", "embed"],
            &x.dims(),
            &[("batch", batch), ("time", time), ("embed", self.n_embed())]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::Wgpu;
    use burn::tensor::Distribution;

    #[test]
    fn test_mlp_config() {
        let cfg = MLPConfig::new(3);

        assert_eq!(cfg.n_embed, 3);
        assert_eq!(cfg.expansion_factor, 4);

        assert_eq!(cfg.n_embed(), 3);
        assert_eq!(cfg.hidden_size(), 3 * 4);
    }

    #[test]
    fn test_mlp() {
        type B = Wgpu;
        let device = Default::default();

        for activation in [ActivationConfig::Relu, ActivationConfig::Gelu] {
            for ef in [4, 3] {
                let b = 2;
                let t = 3;
                let n_embed = 10;

                let cfg = MLPConfig::new(n_embed)
                    .with_expansion_factor(ef)
                    .with_activation(activation.clone());

                let mlp: MLP<B> = cfg.init(&device);

                assert_eq!(mlp.n_embed(), n_embed);

                let input = Tensor::random([b, t, n_embed], Distribution::Default, &device);
                let output = mlp.forward(input.clone());

                let x = input;
                assert_shape_contract!(
                    ["batch", "time", "embed"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("embed", n_embed)]
                );

                let x = mlp.c_fc.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "hidden"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("hidden", ef * n_embed)]
                );

                let x = mlp.act.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "hidden"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("hidden", ef * n_embed)]
                );

                let x = x.clone() * x;
                let x = mlp.c_proj.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "embed"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("embed", n_embed)]
                );

                output.to_data().assert_eq(&x.to_data(), true);
            }
        }
    }
}

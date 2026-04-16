use std::marker::PhantomData;

use burn::{
    Tensor,
    grad_clipping::GradientClipping,
    module::{
        AutodiffModule,
        ModuleMapper,
        Param,
        ParamId,
    },
    optim::{
        GradientsParams,
        LearningRate,
        MultiGradientsParams,
        Optimizer,
        SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        record::AdaptorRecord,
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use hashbrown::{
    HashMap,
    HashSet,
};

use crate::optimizers::{
    clone_simple_optimizer,
    compat::GradAdaptor,
};

/// A group of [`ParamId`] assigned to a single optimizer instance.
#[derive(Clone)]
pub struct OptimizerGroup<B, O>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    /// The Parameters assigned to this group.
    pub params: HashSet<ParamId>,

    /// The optimizer instance assigned to this group.
    pub optim: O,

    phantom: PhantomData<B>,
}

impl<B, O> OptimizerGroup<B, O>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    /// Create a new `GroupOptimizer` with the given parameters and optimizer.
    pub fn new(
        params: HashSet<ParamId>,
        optim: O,
    ) -> Self {
        Self {
            params,
            optim,
            phantom: PhantomData,
        }
    }

    /// Build a [`OptimizerGroup`] from a [`OptimizerAdaptor`].
    pub fn from_adaptor<M>(
        params: HashSet<ParamId>,
        adaptor: &OptimizerAdaptor<O, M, B>,
    ) -> Self
    where
        M: AutodiffModule<B>,
    {
        Self::new(params, clone_simple_optimizer(adaptor))
    }
}

/// Error during `GroupOptimizerAdaptor2` construction.
#[derive(Debug)]
pub enum GroupOptimizerError {
    /// A `ParamId` was assigned to more than one optimizer group.
    DuplicateParamId {
        param_id: ParamId,
        /// (`type_tag`, index) of the first assignment
        first: (usize, usize),
        /// (`type_tag`, index) of the conflicting assignment
        second: (usize, usize),
    },
}

/// Parameter group optimizer adaptor for N=2 `SimpleOptimizer` types.
#[derive(Clone)]
pub struct GroupOptimizerAdaptor2<O1, O2, M, B>
where
    O1: SimpleOptimizer<B::InnerBackend>,
    O2: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    groups1: Vec<OptimizerGroup<B, O1>>,
    groups2: Vec<OptimizerGroup<B, O2>>,

    /// `ParamId` → (`type_tag`, `group_index`)
    /// `type_tag`: 0 → O1, 1 → O2
    dispatch: HashMap<ParamId, (usize, usize)>,

    #[allow(clippy::type_complexity)]
    records: (
        Vec<HashMap<ParamId, AdaptorRecord<O1, B>>>,
        Vec<HashMap<ParamId, AdaptorRecord<O2, B>>>,
    ),

    grad_clipping: Option<GradientClipping>,
    _module: PhantomData<M>,
}

impl<O1, O2, M, B> GroupOptimizerAdaptor2<O1, O2, M, B>
where
    O1: SimpleOptimizer<B::InnerBackend>,
    O2: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    /// Construct and validate.
    ///
    /// Returns an error if any `ParamId` appears in more than one group.
    pub fn new(
        groups1: Vec<OptimizerGroup<B, O1>>,
        groups2: Vec<OptimizerGroup<B, O2>>,
    ) -> Result<Self, GroupOptimizerError> {
        let mut dispatch = HashMap::new();

        for (idx, group) in groups1.iter().enumerate() {
            for &param_id in &group.params {
                if let Some(&first) = dispatch.get(&param_id) {
                    return Err(GroupOptimizerError::DuplicateParamId {
                        param_id,
                        first,
                        second: (0, idx),
                    });
                }
                dispatch.insert(param_id, (0, idx));
            }
        }

        for (idx, group) in groups2.iter().enumerate() {
            for &param_id in &group.params {
                if let Some(&first) = dispatch.get(&param_id) {
                    return Err(GroupOptimizerError::DuplicateParamId {
                        param_id,
                        first,
                        second: (1, idx),
                    });
                }
                dispatch.insert(param_id, (1, idx));
            }
        }

        let records = (
            vec![HashMap::new(); groups1.len()],
            vec![HashMap::new(); groups2.len()],
        );

        Ok(Self {
            groups1,
            groups2,
            dispatch,
            records,
            grad_clipping: None,
            _module: PhantomData,
        })
    }

    /// Accces group1.
    pub fn groups1(&self) -> &[OptimizerGroup<B, O1>] {
        &self.groups1
    }

    /// Accces group2.
    pub fn groups2(&self) -> &[OptimizerGroup<B, O2>] {
        &self.groups2
    }

    /// Mutate group1.
    pub fn groups1_mut(&mut self) -> &mut [OptimizerGroup<B, O1>] {
        &mut self.groups1
    }

    /// Mutate group2.
    pub fn groups2_mut(&mut self) -> &mut [OptimizerGroup<B, O2>] {
        &mut self.groups2
    }

    /// Sets the gradient clipping.
    ///
    /// # Arguments
    ///
    /// * `gradient_clipping` - The gradient clipping.
    ///
    /// # Returns
    ///
    /// The optimizer.
    pub fn with_grad_clipping(
        mut self,
        grad_clipping: GradientClipping,
    ) -> Self {
        self.grad_clipping = Some(grad_clipping);
        self
    }

    fn step_common(
        &mut self,
        lr: LearningRate,
        module: M,
        mut grads: GradAdaptor,
    ) -> M {
        module.map(&mut GroupOptimizerMapper2 {
            groups1: &self.groups1,
            groups2: &self.groups2,
            dispatch: &self.dispatch,
            records1: &mut self.records.0,
            records2: &mut self.records.1,
            grads: &mut grads,
            lr,
            grad_clipping: self.grad_clipping.as_ref(),
            _phantom: PhantomData::<M>,
        })
    }
}

type GroupRecord2<O1, O2, B> = (
    Vec<HashMap<ParamId, AdaptorRecord<O1, B>>>,
    Vec<HashMap<ParamId, AdaptorRecord<O2, B>>>,
);

impl<O1, O2, M, B> Optimizer<M, B> for GroupOptimizerAdaptor2<O1, O2, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O1: SimpleOptimizer<B::InnerBackend>,
    O2: SimpleOptimizer<B::InnerBackend>,
{
    type Record = GroupRecord2<O1, O2, B>;

    fn step(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: GradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    fn step_multi(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: MultiGradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    fn to_record(&self) -> Self::Record {
        self.records.clone()
    }

    fn load_record(
        mut self,
        record: Self::Record,
    ) -> Self {
        self.records = record;
        self
    }
}

struct GroupOptimizerMapper2<'a, M, B, O1, O2>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
    O1: SimpleOptimizer<B::InnerBackend>,
    O2: SimpleOptimizer<B::InnerBackend>,
{
    groups1: &'a Vec<OptimizerGroup<B, O1>>,
    groups2: &'a Vec<OptimizerGroup<B, O2>>,

    dispatch: &'a HashMap<ParamId, (usize, usize)>,

    records1: &'a mut Vec<HashMap<ParamId, AdaptorRecord<O1, B>>>,
    records2: &'a mut Vec<HashMap<ParamId, AdaptorRecord<O2, B>>>,

    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    grad_clipping: Option<&'a GradientClipping>,

    _phantom: PhantomData<M>,
}

impl<M, B, O1, O2> ModuleMapper<B> for GroupOptimizerMapper2<'_, M, B, O1, O2>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
    O1: SimpleOptimizer<B::InnerBackend>,
    O2: SimpleOptimizer<B::InnerBackend>,
{
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();

        let Some((grad, device)) = self.grads.remove::<B::InnerBackend, D>(id) else {
            return Param::from_mapped_value(id, tensor, mapper);
        };

        // ParamIds not in dispatch are left untouched (no gradient applied).
        // This handles params not assigned to any group gracefully.
        let Some(&(type_tag, idx)) = self.dispatch.get(&id) else {
            return Param::from_mapped_value(id, tensor, mapper);
        };

        let is_require_grad = tensor.is_require_grad();

        let tensor = if tensor.device() != device {
            tensor.to_device(&device)
        } else {
            tensor
        };

        let grad = if let Some(clipping) = self.grad_clipping {
            clipping.clip_gradient(grad)
        } else {
            grad
        };

        let tensor = match type_tag {
            0 => step_group::<B, O1, D>(
                &self.groups1[idx].optim,
                &mut self.records1[idx],
                id,
                tensor.inner(),
                grad,
                &device,
                self.lr,
            ),
            1 => step_group::<B, O2, D>(
                &self.groups2[idx].optim,
                &mut self.records2[idx],
                id,
                tensor.inner(),
                grad,
                &device,
                self.lr,
            ),
            _ => unreachable!("GroupOptimizerAdaptor2 only has type tags 0 and 1"),
        };

        let mut tensor = Tensor::from_inner(tensor);
        if is_require_grad {
            tensor = tensor.require_grad();
        }

        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Execute a single optimizer step for one parameter, managing record
/// load/store.
///
/// Factored out to avoid duplicating the record-management logic per type arm.
fn step_group<B, O, const D: usize>(
    optim: &O,
    records: &mut HashMap<ParamId, AdaptorRecord<O, B>>,
    id: ParamId,
    tensor: Tensor<B::InnerBackend, D>,
    grad: Tensor<B::InnerBackend, D>,
    device: &<B::InnerBackend as Backend>::Device,
    lr: LearningRate,
) -> Tensor<B::InnerBackend, D>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    let (key, record) = records.remove_entry(&id).unzip();
    let state = record.map(|r| O::to_device(r.into_state(), device));

    let (tensor, state) = optim.step(lr, tensor, grad, state);

    if let Some(state) = state {
        records.insert(key.unwrap_or(id), AdaptorRecord::from_state(state));
    }

    tensor
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_nothing() {}
}

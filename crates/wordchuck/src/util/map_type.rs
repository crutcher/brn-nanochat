//! # Map Type Trait
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{Hash};
use std::mem::transmute;
use std::ops::{Index };
use ahash::AHashMap;

/// A type that can be used as a ``{ key -> value }`` hash lookup map.
pub trait MapType<K, V>:
  for<'a> Index<&'a K, Output = V> + IntoIterator<Item = (K, V)> + Default
where
    K: Eq + Hash,
{
    /// The key type.
    type Key;

    /// The value type.
    type Value;

    /// See: [`std::collections::HashMap::contains_key`].
    fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq;

    /// See: [`std::collections::HashMap::get`].
    fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq;
}

/// A mutable extension of [`MapType`].
pub trait MutMapType<K, V>: MapType<K, V> + Extend<(K, V)>
where
    K: Eq + Hash,
{
    /// See: [`std::collections::HashMap::insert`].
    fn insert(&mut self, k: K, v: V) -> Option<V>;
}

impl<K, V> MapType<K, V> for AHashMap<K, V>
where
    K: Eq + Hash,
{
    type Key = K;
    type Value = V;

    fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq {
        let base: &HashMap<K, V> = unsafe { transmute(self) };
        base.contains_key(k)
    }

    fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq { self.get(k) }
}

impl<K, V> MutMapType<K, V> for AHashMap<K, V>
where
    K: Eq + Hash,
{
    fn insert(&mut self, k: K, v: V) -> Option<V> { self.insert(k, v) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    fn common_test<M>()
    where
        M: MutMapType<u32, f64>,
    {
        let mut map = M::default();

        assert_eq!(map.get(&1), None);

        map.insert(1, 2.0);
        assert_eq!(map.get(&1), Some(&2.0));
    }

    #[test]
    fn test_ahash_map_type() {
        common_test::<AHashMap<u32, f64>>();

        let m: AHashMap<u32, f64> = Default::default();
        m.contains_key(&1);

    }
}
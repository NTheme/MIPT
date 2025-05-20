#![forbid(unsafe_code)]

use std::mem::replace;
use std::vec::IntoIter;
use std::{borrow::Borrow, iter::FromIterator, ops::Index};

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Debug, PartialEq, Eq)]
pub struct FlatMap<K, V>(Vec<(K, V)>);

impl<K: Ord, V> FlatMap<K, V> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn as_slice(&self) -> &[(K, V)] {
        self.0.as_slice()
    }

    fn lower_bound<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.0.is_empty() || *self.0[0].0.borrow() >= *key {
            return 0;
        }

        let mut l = 0;
        let mut r = self.0.len();
        while r - l > 1 {
            let m = (l + r) / 2;
            if *self.0[m].0.borrow() < *key {
                l = m;
            } else {
                r = m;
            }
        }
        r
    }

    fn index_by_key<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let index = self.lower_bound(key);
        if index < self.0.len() && *self.0[index].0.borrow() == *key {
            Some(index)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let index = self.lower_bound(&key);
        if index < self.0.len() && *self.0[index].0.borrow() == key {
            Some(replace(&mut self.0[index].1, value))
        } else {
            self.0.insert(index, (key, value));
            None
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Some(&self.0[self.index_by_key(key)?].1)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Some(self.0.remove(self.index_by_key(key)?).1)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Some(self.0.remove(self.index_by_key(key)?))
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<K, Q, V> Index<&Q> for FlatMap<K, V>
where
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        &self.0[self.index_by_key(key).unwrap()].1
    }
}

impl<K: Ord, V> Extend<(K, V)> for FlatMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Ord, V> From<Vec<(K, V)>> for FlatMap<K, V> {
    fn from(value: Vec<(K, V)>) -> Self {
        let mut res = Self::new();
        res.extend(value);
        res
    }
}

impl<K: Ord, V> From<FlatMap<K, V>> for Vec<(K, V)> {
    fn from(value: FlatMap<K, V>) -> Self {
        let mut res = Vec::new();
        res.extend(value);
        res
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for FlatMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut res = FlatMap::new();
        res.extend(iter);
        res
    }
}

impl<K: Ord, V> IntoIterator for FlatMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

mod set_ops;

use crate::bitvector::*;
use crate::ffi::TieredVec32 as TieredVec; // default tiered vector
use crate::ffi::{UniquePtr, WithinUniquePtr};
use crate::sliced_int::SlicedInt;
use crate::trievec::*;
use num_traits::cast::AsPrimitive;
use num_traits::sign::Unsigned;
use num_traits::PrimInt;
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::BTreeMap;

pub struct WordSet<const PREFIX_BITS: usize, const SUFFIX_BITS: usize>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    pub(crate) prefixes: Bitvector,
    pub(crate) tiered: UniquePtr<TieredVec>,
    pub(crate) suffix_containers: Vec<TrieVec<{ SUFFIX_BITS.div_ceil(8) }>>,
    pub(crate) empty_containers: Vec<usize>,
}

impl<const PREFIX_BITS: usize, const SUFFIX_BITS: usize> WordSet<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    const PREFIX_BITS: usize = PREFIX_BITS;
    const SUFFIX_BITS: usize = SUFFIX_BITS;
    const THRESHOLD: usize = 1024;

    pub fn new() -> Self {
        assert!(
            PREFIX_BITS <= 32,
            "PREFIX_BITS={PREFIX_BITS} but it should be ≤ 32"
        );
        assert!(SUFFIX_BITS > 0, "SUFFIX_BITS should be ≠ 0");
        Self {
            prefixes: Bitvector::new_with_bitlength(Self::PREFIX_BITS),
            tiered: TieredVec::new().within_unique_ptr(),
            suffix_containers: Vec::new(),
            empty_containers: Vec::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.suffix_containers
            .iter()
            .map(|container| container.len())
            .sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prefixes.count() == 0
    }

    #[inline]
    pub fn split_prefix_suffix<T: PrimInt + Unsigned + AsPrimitive<usize>>(
        word: T,
    ) -> (usize, SlicedInt<{ SUFFIX_BITS.div_ceil(8) }>) {
        let suffix_mask: T = (T::one() << Self::SUFFIX_BITS) - T::one();
        (
            (word >> Self::SUFFIX_BITS).as_(),
            SlicedInt::<{ SUFFIX_BITS.div_ceil(8) }>::from_int(word & suffix_mask),
        )
    }

    #[inline]
    pub fn merge_prefix_suffix<T: PrimInt + Unsigned + AsPrimitive<usize>>(
        prefix: usize,
        suffix: SlicedInt<{ SUFFIX_BITS.div_ceil(8) }>,
    ) -> T
    where
        usize: AsPrimitive<T>,
    {
        let prefix: T = prefix.as_();
        let suffix: T = suffix.get();
        (prefix << SUFFIX_BITS) | suffix
    }

    #[inline]
    pub fn contains<T: PrimInt + Unsigned + AsPrimitive<usize>>(&self, word: T) -> bool {
        let (prefix, suffix) = Self::split_prefix_suffix(word);
        if !self.prefixes.contains(prefix) {
            return false;
        }
        let rank = self.prefixes.rank(prefix);
        let id = self.tiered.get(rank) as usize;
        self.suffix_containers[id].contains(&suffix)
    }

    pub fn insert<T: PrimInt + Unsigned + AsPrimitive<usize>>(&mut self, word: T) -> bool {
        let (prefix, suffix) = Self::split_prefix_suffix(word);
        let mut absent = self.prefixes.insert(prefix);
        let rank = self.prefixes.rank(prefix);
        if absent {
            match self.empty_containers.pop() {
                Some(id) => {
                    self.suffix_containers[id].insert(suffix);
                    self.tiered.insert(rank, id as u32);
                }
                None => {
                    let id = self.suffix_containers.len();
                    self.suffix_containers
                        .push(TrieVec::<{ SUFFIX_BITS.div_ceil(8) }>::new_with_one(suffix));
                    self.tiered.insert(rank, id as u32);
                }
            };
        } else {
            let id = self.tiered.get(rank) as usize;
            absent = self.suffix_containers[id].insert(suffix);
            self.adapt_container_grow(id);
        }
        absent
    }

    pub fn remove<T: PrimInt + Unsigned + AsPrimitive<usize>>(&mut self, word: T) -> bool {
        let (prefix, suffix) = Self::split_prefix_suffix(word);
        let mut present = self.prefixes.contains(prefix);
        if present {
            let rank = self.prefixes.rank(prefix);
            let id = self.tiered.get(rank) as usize;
            present = self.suffix_containers[id].remove(&suffix);
            self.adapt_container_shrink(id);
            if self.suffix_containers[id].is_empty() {
                self.empty_containers.push(id);
                self.tiered.remove(rank);
                self.prefixes.remove(prefix);
            }
        }
        present
    }

    pub fn contains_all<T: PrimInt + Unsigned + AsPrimitive<usize>>(
        &mut self,
        words: &[T],
    ) -> bool {
        let prefixes_suffixes: Vec<_> = words
            .iter()
            .map(|&word| Self::split_prefix_suffix(word))
            .collect();
        for group in prefixes_suffixes.chunk_by(|(p1, _), (p2, _)| p1 == p2) {
            let prefix = group[0].0;
            if !self.prefixes.contains(prefix) {
                return false;
            }
            let rank = self.prefixes.rank(prefix);
            let id = self.tiered.get(rank) as usize;
            for (_, suffix) in group.iter() {
                if !self.suffix_containers[id].contains(suffix) {
                    return false;
                }
            }
        }
        true
    }

    pub fn contains_batch<T: PrimInt + Unsigned + AsPrimitive<usize>>(
        &mut self,
        words: &[T],
    ) -> Vec<bool> {
        let mut res = Vec::with_capacity(words.len());
        let prefixes_suffixes: Vec<_> = words
            .iter()
            .map(|&word| Self::split_prefix_suffix(word))
            .collect();
        for group in prefixes_suffixes.chunk_by(|(p1, _), (p2, _)| p1 == p2) {
            let prefix = group[0].0;
            if !self.prefixes.contains(prefix) {
                res.resize(res.len() + group.len(), false);
                continue;
            }
            let rank = self.prefixes.rank(prefix);
            let id = self.tiered.get(rank) as usize;
            for (_, suffix) in group.iter() {
                res.push(self.suffix_containers[id].contains(suffix));
            }
        }
        res
    }

    pub fn insert_batch<T: PrimInt + Unsigned + AsPrimitive<usize>>(&mut self, words: &[T]) {
        let prefixes_suffixes: Vec<_> = words
            .iter()
            .map(|&word| Self::split_prefix_suffix(word))
            .collect();
        for group in prefixes_suffixes.chunk_by(|(p1, _), (p2, _)| p1 == p2) {
            let prefix = group[0].0;
            let absent = self.prefixes.insert(prefix);
            let rank = self.prefixes.rank(prefix);
            let id = if absent {
                match self.empty_containers.pop() {
                    Some(id) => {
                        self.tiered.insert(rank, id as u32);
                        id
                    }
                    None => {
                        let id = self.suffix_containers.len();
                        self.suffix_containers
                            .push(TrieVec::<{ SUFFIX_BITS.div_ceil(8) }>::new());
                        self.tiered.insert(rank, id as u32);
                        id
                    }
                }
            } else {
                self.tiered.get(rank) as usize
            };
            self.suffix_containers[id].insert_iter(group.iter().map(|&(_, suffix)| suffix));
            self.adapt_container_grow(id);
        }
    }

    pub fn remove_batch<T: PrimInt + Unsigned + AsPrimitive<usize>>(&mut self, words: &[T]) {
        let prefixes_suffixes: Vec<_> = words
            .iter()
            .map(|&word| Self::split_prefix_suffix(word))
            .collect();
        for group in prefixes_suffixes.chunk_by(|(p1, _), (p2, _)| p1 == p2) {
            let prefix = group[0].0;
            if self.prefixes.contains(prefix) {
                let rank = self.prefixes.rank(prefix);
                let id = self.tiered.get(rank) as usize;
                self.suffix_containers[id].remove_iter(group.iter().map(|&(_, suffix)| suffix));
                if self.suffix_containers[id].is_empty() {
                    self.empty_containers.push(id);
                    self.tiered.remove(rank);
                    self.prefixes.remove(prefix);
                }
                self.adapt_container_shrink(id);
            }
        }
    }

    #[inline]
    fn adapt_container_grow(&mut self, id: usize) {
        if self.suffix_containers[id].len() > Self::THRESHOLD {
            self.suffix_containers[id].as_trie();
        }
    }

    #[inline]
    fn adapt_container_shrink(&mut self, id: usize) {
        if self.suffix_containers[id].len() <= Self::THRESHOLD {
            self.suffix_containers[id].as_vec();
        }
    }

    #[inline]
    pub fn prefix_load(&self) -> f64 {
        self.tiered.len() as f64 / (1 << Self::PREFIX_BITS) as f64
    }

    pub fn buckets_sizes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.prefixes.iter().enumerate().map(|(rank, prefix)| {
            let id = self.tiered.get(rank) as usize;
            (prefix, self.suffix_containers[id].len())
        })
    }

    pub fn buckets_size_count(&self) -> BTreeMap<usize, usize> {
        let mut size_count = BTreeMap::new();
        for (_, size) in self.buckets_sizes() {
            size_count.entry(size).and_modify(|e| *e += 1).or_insert(1);
        }
        size_count
    }

    pub fn buckets_load_repartition(&self) -> BTreeMap<usize, f64> {
        let size_count = self.buckets_size_count();
        let total_size: usize = size_count.iter().map(|(&s, &c)| s * c).sum();
        size_count
            .iter()
            .map(|(&s, &c)| (s, (s * c) as f64 / total_size as f64))
            .collect()
    }

    pub fn buckets_nodes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.prefixes.iter().enumerate().map(|(rank, prefix)| {
            let id = self.tiered.get(rank) as usize;
            (prefix, self.suffix_containers[id].count_nodes())
        })
    }

    pub fn buckets_node_count(&self) -> BTreeMap<usize, usize> {
        let mut node_count = BTreeMap::new();
        for (_, size) in self.buckets_nodes() {
            node_count.entry(size).and_modify(|e| *e += 1).or_insert(1);
        }
        node_count
    }

    #[inline]
    pub fn iter<T: PrimInt + Unsigned + AsPrimitive<usize>>(&self) -> impl Iterator<Item = T> + '_
    where
        usize: AsPrimitive<T>,
    {
        WordSetIterator {
            wordset: self,
            prefix_iter: self.prefixes.iter(),
            prefix: None,
            suffix_iter: None,
            suffix: None,
        }
    }

    /// Libere la capacite excedentaire du tableau de conteneurs.
    ///
    /// `suffix_containers` croit par doublement : a la fin d'une construction
    /// il peut etre rempli a ~55 % seulement (mesure : 18,67 M seaux utilises
    /// pour 2^25 = 33,55 M emplacements alloues). Cette methode rend la
    /// difference au systeme.
    ///
    /// Attention : `shrink_to_fit` REALLOUE et COPIE. Pendant l'appel, l'ancien
    /// et le nouveau tampon coexistent, donc le PIC de memoire augmente meme si
    /// l'occupation finale baisse. A n'appeler que lorsque les insertions sont
    /// terminees, typiquement juste avant la serialisation.
    pub fn shrink_to_fit(&mut self) {
        self.suffix_containers.shrink_to_fit();
        self.empty_containers.shrink_to_fit();
    }

    /// Occupation du tableau de conteneurs : (utilises, capacite alloues).
    /// Sert a mesurer ce que `shrink_to_fit` peut rendre.
    pub fn containers_load(&self) -> (usize, usize) {
        (
            self.suffix_containers.len(),
            self.suffix_containers.capacity(),
        )
    }

    /// Instrumentation A1 : ou part la memoire de l'index ?
    ///
    /// Renvoie un rapport texte : repartition des k-mers entre seaux stockes
    /// en `Vec` et seaux stockes en `Trie`, nombre de noeuds de trie, octets
    /// par couche, et histogramme des tailles de seau.
    ///
    /// Cout : parcourt tous les tries (`count_nodes` est recursif).
    /// A n'appeler qu'a la demande, jamais dans un chemin critique.
    pub fn memory_report(&self) -> String {
        // Couts unitaires lus dans le code de CBL :
        //   TrieNode = TinyBitvector([u64; 4]) = 32 o  +  Vec<Trie> = 24 o  => 56 o
        //   chaque enfant coute 8 o (slot du Vec parent) + 8 o (Box)
        const NODE_BYTES: usize = 56;
        const CHILD_PTR: usize = 16;
        const VEC_HEADER: usize = 24;

        let sb = Self::SUFFIX_BITS.div_ceil(8);

        let mut n_vec = 0usize;
        let mut n_trie = 0usize;
        let mut k_vec = 0usize;
        let mut k_trie = 0usize;
        let mut b_vec = 0usize;
        let mut b_trie = 0usize;
        let mut nodes_total = 0usize;
        let mut max_bucket = 0usize;
        let mut hist = [0usize; 32]; // k-mers par classe log2 de taille de seau

        for c in self.suffix_containers.iter() {
            let len = c.len();
            if len == 0 {
                continue;
            }
            if len > max_bucket {
                max_bucket = len;
            }
            let class = ((usize::BITS - len.leading_zeros()) as usize).min(31);
            hist[class] += len;
            if len > Self::THRESHOLD {
                n_trie += 1;
                k_trie += len;
                let nodes = c.count_nodes();
                nodes_total += nodes;
                b_trie += VEC_HEADER + nodes * (NODE_BYTES + CHILD_PTR);
            } else {
                n_vec += 1;
                k_vec += len;
                b_vec += VEC_HEADER + len * sb;
            }
        }

        let m = (k_vec + k_trie).max(1);
        let b_prefix = (1usize << Self::PREFIX_BITS) / 8;
        let b_tiered = 8 * self.prefixes.count(); // 1 pointeur par prefixe occupe (MINORANT)
        let b_total = b_prefix + b_tiered + b_vec + b_trie;

        let pct = |x: usize| 100.0 * (x as f64) / (m as f64);
        let bits = |b: usize| 8.0 * (b as f64) / (m as f64);

        let mut s = String::new();
        s.push_str(&format!(
            "PREFIX_BITS={}  SUFFIX_BITS={} ({} o/suffixe)  THRESHOLD={}\n",
            Self::PREFIX_BITS,
            Self::SUFFIX_BITS,
            sb,
            Self::THRESHOLD
        ));
        s.push_str(&format!("k-mers                 : {}\n", m));
        s.push_str(&format!(
            "seaux occupes          : {} (vec {}, trie {})\n",
            n_vec + n_trie,
            n_vec,
            n_trie
        ));
        s.push_str(&format!("plus gros seau         : {}\n", max_bucket));
        s.push_str(&format!(
            "k-mers en mode vec     : {} ({:.2} %)\n",
            k_vec,
            pct(k_vec)
        ));
        s.push_str(&format!(
            "k-mers en mode trie    : {} ({:.2} %)   <== PLAFOND DE LA PISTE\n",
            k_trie,
            pct(k_trie)
        ));
        s.push_str(&format!(
            "noeuds de trie         : {} ({:.4} noeud / k-mer en trie)\n",
            nodes_total,
            (nodes_total as f64) / (k_trie.max(1) as f64)
        ));
        s.push_str("--- memoire : octets, puis bits/k-mer rapportes a TOUT l'index ---\n");
        s.push_str(&format!(
            "bitvector prefixes     : {:>16} o  {:>9.4} b/kmer\n",
            b_prefix,
            bits(b_prefix)
        ));
        s.push_str(&format!(
            "tiered vector (minorant): {:>15} o  {:>9.4} b/kmer\n",
            b_tiered,
            bits(b_tiered)
        ));
        s.push_str(&format!(
            "suffixes en vec        : {:>16} o  {:>9.4} b/kmer\n",
            b_vec,
            bits(b_vec)
        ));
        s.push_str(&format!(
            "tries                  : {:>16} o  {:>9.4} b/kmer   ({:.2} % du total)\n",
            b_trie,
            bits(b_trie),
            100.0 * (b_trie as f64) / (b_total.max(1) as f64)
        ));
        s.push_str(&format!(
            "TOTAL                  : {:>16} o  {:>9.4} b/kmer\n",
            b_total,
            bits(b_total)
        ));
        s.push_str("--- repartition des k-mers par taille de seau ---\n");
        for i in 0..32 {
            if hist[i] > 0 {
                s.push_str(&format!(
                    "  taille < 2^{:<2} : {:>16} ({:.2} %)\n",
                    i,
                    hist[i],
                    pct(hist[i])
                ));
            }
        }
        s
    }
}

impl<const PREFIX_BITS: usize, const SUFFIX_BITS: usize> Default
    for WordSet<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    fn default() -> Self {
        Self::new()
    }
}

struct WordSetIterator<
    'a,
    T: PrimInt + Unsigned + AsPrimitive<usize>,
    const PREFIX_BITS: usize,
    const SUFFIX_BITS: usize,
> where
    [(); SUFFIX_BITS.div_ceil(8)]:,
    usize: AsPrimitive<T>,
{
    wordset: &'a WordSet<PREFIX_BITS, SUFFIX_BITS>,
    prefix_iter: BitvectorIterator<'a>,
    prefix: Option<usize>,
    suffix_iter: Option<TrieVecIterator<'a, { SUFFIX_BITS.div_ceil(8) }>>,
    suffix: Option<SlicedInt<{ SUFFIX_BITS.div_ceil(8) }>>,
}

impl<
        'a,
        T: PrimInt + Unsigned + AsPrimitive<usize>,
        const PREFIX_BITS: usize,
        const SUFFIX_BITS: usize,
    > Iterator for WordSetIterator<'a, T, PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
    usize: AsPrimitive<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        while self.suffix.is_none() {
            self.prefix = self.prefix_iter.next();
            let rank = self.wordset.prefixes.rank(self.prefix?);
            let id = self.wordset.tiered.get(rank) as usize;
            self.suffix_iter = Some(self.wordset.suffix_containers[id].iter());
            self.suffix = self.suffix_iter.as_mut().unwrap().next();
        }
        let word =
            WordSet::<PREFIX_BITS, SUFFIX_BITS>::merge_prefix_suffix(self.prefix?, self.suffix?);
        self.suffix = self.suffix_iter.as_mut().unwrap().next();
        Some(word)
    }
}

impl<const PREFIX_BITS: usize, const SUFFIX_BITS: usize> Clone for WordSet<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    fn clone(&self) -> Self {
        let tiered = TieredVec::new().within_unique_ptr();
        for i in 0..self.tiered.len() {
            tiered.insert(i, self.tiered.get(i));
        }
        Self {
            prefixes: self.prefixes.clone(),
            tiered,
            suffix_containers: self.suffix_containers.clone(),
            empty_containers: self.empty_containers.clone(),
        }
    }
}

impl<const PREFIX_BITS: usize, const SUFFIX_BITS: usize> Serialize
    for WordSet<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.tiered.len()))?;
        for (rank, prefix) in self.prefixes.iter().enumerate() {
            let prefix = prefix as u32;
            let id = self.tiered.get(rank) as usize;
            map.serialize_entry(&prefix, &self.suffix_containers[id])?;
        }
        map.end()
    }
}

struct WordSetVisitor<const PREFIX_BITS: usize, const SUFFIX_BITS: usize> {}

impl<'de, const PREFIX_BITS: usize, const SUFFIX_BITS: usize> Visitor<'de>
    for WordSetVisitor<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    type Value = WordSet<PREFIX_BITS, SUFFIX_BITS>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
        formatter.write_str("a wordset")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
        let mut wordset = WordSet {
            prefixes: Bitvector::new_with_bitlength(PREFIX_BITS),
            tiered: TieredVec::new().within_unique_ptr(),
            suffix_containers: Vec::with_capacity(access.size_hint().unwrap_or(0)),
            empty_containers: Vec::new(),
        };
        while let Some((prefix, suffix_container)) = access.next_entry::<u32, _>()? {
            let prefix = prefix as usize;
            let rank = wordset.suffix_containers.len();
            wordset.prefixes.insert(prefix);
            wordset.tiered.insert(rank, rank as u32);
            wordset.suffix_containers.push(suffix_container);
        }
        Ok(wordset)
    }
}

impl<'de, const PREFIX_BITS: usize, const SUFFIX_BITS: usize> Deserialize<'de>
    for WordSet<PREFIX_BITS, SUFFIX_BITS>
where
    [(); SUFFIX_BITS.div_ceil(8)]:,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(WordSetVisitor {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{thread_rng, SeedableRng};

    const N: usize = 1_000_000;
    const PREFIX_BITS: usize = 24;
    const SUFFIX_BITS: usize = 8;

    #[test]
    fn test_insert_contains_remove() {
        let mut v0 = (0..(2 * N)).step_by(2).collect_vec();
        let mut v1 = (0..(2 * N)).skip(1).step_by(2).collect_vec();
        let mut rng = thread_rng();
        v0.shuffle(&mut rng);
        v1.shuffle(&mut rng);
        let mut set = WordSet::<PREFIX_BITS, SUFFIX_BITS>::new();
        for &i in v0.iter() {
            assert!(set.insert(i));
        }
        for &i in v0.iter() {
            assert!(set.contains(i));
        }
        for &i in v1.iter() {
            assert!(!set.contains(i));
        }
        for &i in v0.iter() {
            assert!(set.remove(i));
        }
        for &i in v0.iter() {
            assert!(!set.contains(i));
        }
        assert!(set.is_empty());
    }

    #[test]
    fn test_random_insert_contains_remove() {
        let mut v0 = (0..(2 * N)).step_by(2).collect_vec();
        let mut rng = StdRng::seed_from_u64(42);
        v0.shuffle(&mut rng);

        let mut set = WordSet::<PREFIX_BITS, SUFFIX_BITS>::new();
        for &i in v0.iter() {
            assert!(set.insert(i));
        }
        assert_eq!(set.count(), N);
        v0.shuffle(&mut rng);

        for &i in v0.iter() {
            assert!(set.contains(i));
        }
        v0.shuffle(&mut rng);

        for &i in v0.iter() {
            assert!(set.remove(i));
        }
        assert!(set.is_empty());
    }

    #[test]
    fn test_batch_operations() {
        let mut set = WordSet::<PREFIX_BITS, SUFFIX_BITS>::new();
        let v0 = (0..(2 * N)).step_by(2).collect_vec();
        let v1 = (0..(2 * N)).skip(1).step_by(2).collect_vec();
        set.insert_batch(&v0);
        assert!(set.contains_batch(&v0).iter().all(|&b| b));
        for &i in v1.iter() {
            assert!(!set.contains(i));
        }
        set.remove_batch(&v0);
        for &i in v0.iter() {
            assert!(!set.contains(i));
        }
        assert!(set.is_empty());
    }

    #[test]
    fn test_wordset_iter() {
        let mut set = WordSet::<PREFIX_BITS, SUFFIX_BITS>::new();
        set.insert(1u64);
        set.insert(42u64);
        set.insert((1 << SUFFIX_BITS) - 1u64);
        set.insert((1 << SUFFIX_BITS) + 10u64);
        set.insert(10 * (1 << SUFFIX_BITS) + 10u64);
        let mut iter = set.iter::<u64>();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(42));
        assert_eq!(iter.next(), Some((1 << SUFFIX_BITS) - 1u64));
        assert_eq!(iter.next(), Some((1 << SUFFIX_BITS) + 10));
        assert_eq!(iter.next(), Some(10 * (1 << SUFFIX_BITS) + 10));
        assert_eq!(iter.next(), None);
    }
}

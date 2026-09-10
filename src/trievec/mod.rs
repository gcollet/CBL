mod set_ops;

use crate::sliced_int::SlicedInt;
use crate::trie::{Trie, TrieIterator};
use core::slice::Iter;
use serde::ser::SerializeTupleVariant;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Taille de la charge utile inline, en octets.
///
/// La variante `Vec` occupe deja 24 octets (ptr + len + cap), donc l'enum
/// mesure 32 octets (24 + discriminant aligne sur 8). En bornant la variante
/// `Small` a 1 + 23 = 24 octets, **la taille de l'enum reste inchangee** :
/// `suffix_containers` ne grossit pas d'un seul octet.
const INLINE_BYTES: usize = 23;

/// Reinterprete la charge utile inline comme une tranche de `SlicedInt<BYTES>`.
///
/// # Securite
/// `SlicedInt<BYTES>` est `#[repr(transparent)]` sur `[u8; BYTES]` : meme taille,
/// alignement 1, aucun bit invalide. Le tableau `[u8; INLINE_BYTES]` est toujours
/// initialise, et `len * BYTES <= INLINE_BYTES` est garanti par `INLINE_CAP`.
#[inline(always)]
unsafe fn inline_slice<const BYTES: usize>(
    len: u8,
    data: &[u8; INLINE_BYTES],
) -> &[SlicedInt<BYTES>] {
    core::slice::from_raw_parts(data.as_ptr().cast::<SlicedInt<BYTES>>(), len as usize)
}

#[inline(always)]
unsafe fn inline_slice_mut<const BYTES: usize>(
    len: u8,
    data: &mut [u8; INLINE_BYTES],
) -> &mut [SlicedInt<BYTES>] {
    core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<SlicedInt<BYTES>>(), len as usize)
}

#[derive(Debug, Clone)]
enum TrieOrVec<const BYTES: usize> {
    Vec(Vec<SlicedInt<BYTES>>),
    Trie(Trie<BYTES>, usize),
    /// Petits seaux stockes sur place : aucune allocation tas.
    /// `u8` = nombre d'elements, `[u8; INLINE_BYTES]` = leur concatenation.
    Small(u8, [u8; INLINE_BYTES]),
}

// ---------------------------------------------------------------------------
// Serialisation : `Small` s'ecrit EXACTEMENT comme `Vec` (variante 0, sequence
// de SlicedInt). Le format de fichier est donc identique a celui de la version
// d'origine, ce qui permet de comparer les index octet pour octet.
// ---------------------------------------------------------------------------

impl<const BYTES: usize> Serialize for TrieOrVec<BYTES> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            TrieOrVec::Vec(vec) => {
                serializer.serialize_newtype_variant("TrieOrVec", 0, "Vec", &vec[..])
            }
            TrieOrVec::Small(len, data) => {
                let slice = unsafe { inline_slice::<BYTES>(*len, data) };
                serializer.serialize_newtype_variant("TrieOrVec", 0, "Vec", &slice[..])
            }
            TrieOrVec::Trie(trie, len) => {
                let mut sv = serializer.serialize_tuple_variant("TrieOrVec", 1, "Trie", 2)?;
                sv.serialize_field(trie)?;
                sv.serialize_field(len)?;
                sv.end()
            }
        }
    }
}

/// Miroir exact de l'enum d'origine, uniquement pour la deserialisation.
/// Les variantes et leur ordre sont identiques : meme format sur le fil.
#[derive(Deserialize)]
enum TrieOrVecDe<const BYTES: usize> {
    Vec(Vec<SlicedInt<BYTES>>),
    Trie(Trie<BYTES>, usize),
}

impl<'de, const BYTES: usize> Deserialize<'de> for TrieOrVec<BYTES> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match TrieOrVecDe::<BYTES>::deserialize(deserializer)? {
            TrieOrVecDe::Vec(vec) => TrieVec::<BYTES>::repr_from_vec(vec),
            TrieOrVecDe::Trie(trie, len) => TrieOrVec::Trie(trie, len),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieVec<const BYTES: usize>(TrieOrVec<BYTES>);

impl<const BYTES: usize> TrieVec<BYTES> {
    /// Nombre d'elements tenant dans la charge utile inline.
    /// Vaut 4 pour BYTES=5 (p=28..32), 3 pour BYTES=6 (p=20..24), 0 si BYTES>23.
    const INLINE_CAP: usize = if BYTES == 0 || BYTES > INLINE_BYTES {
        0
    } else {
        INLINE_BYTES / BYTES
    };

    #[inline(always)]
    fn empty_repr() -> TrieOrVec<BYTES> {
        if Self::INLINE_CAP > 0 {
            TrieOrVec::Small(0, [0u8; INLINE_BYTES])
        } else {
            TrieOrVec::Vec(Vec::new())
        }
    }

    /// Choisit la representation la plus compacte pour un vecteur donne.
    #[inline]
    fn repr_from_vec(vec: Vec<SlicedInt<BYTES>>) -> TrieOrVec<BYTES> {
        if vec.len() <= Self::INLINE_CAP {
            let mut data = [0u8; INLINE_BYTES];
            let len = vec.len() as u8;
            {
                let dst = unsafe { inline_slice_mut::<BYTES>(len, &mut data) };
                dst.copy_from_slice(&vec[..]);
            }
            TrieOrVec::Small(len, data)
        } else {
            TrieOrVec::Vec(vec)
        }
    }

    /// Repasse en representation inline si le contenu y tient de nouveau.
    /// Utilise apres un aller-retour par `Vec` dans les operations ensemblistes.
    #[inline]
    fn compact(&mut self) {
        if Self::INLINE_CAP == 0 {
            return;
        }
        let fits = matches!(&self.0, TrieOrVec::Vec(vec) if vec.len() <= Self::INLINE_CAP);
        if fits {
            if let TrieOrVec::Vec(vec) =
                core::mem::replace(&mut self.0, TrieOrVec::Vec(Vec::new()))
            {
                self.0 = Self::repr_from_vec(vec);
            }
        }
    }

    /// Algorithme d'insertion triee d'origine, extrait pour etre applique a
    /// l'identique quelle que soit la representation de depart.
    fn insert_sorted_into<I: Iterator<Item = SlicedInt<BYTES>>>(
        vec: &mut Vec<SlicedInt<BYTES>>,
        it: I,
    ) {
        let stop = vec.len();
        let mut i = 0;
        for x in it {
            while i < stop && x > vec[i] {
                i += 1;
            }
            if i == stop || x < vec[i] {
                vec.push(x);
            }
        }
    }

    /// Algorithme de suppression triee d'origine (indices collectes puis
    /// `swap_remove` en ordre inverse), extrait pour la meme raison.
    fn remove_sorted_from<I: Iterator<Item = SlicedInt<BYTES>>>(
        vec: &mut Vec<SlicedInt<BYTES>>,
        it: I,
    ) {
        let stop = vec.len();
        let mut i = 0;
        let mut deletions = Vec::new();
        for x in it {
            while i < stop && x > vec[i] {
                i += 1;
            }
            if i < stop && x == vec[i] {
                deletions.push(i);
            }
        }
        for &i in deletions.iter().rev() {
            vec.swap_remove(i);
        }
    }

    /// Convertit une representation inline en `Vec` (debordement de capacite).
    #[inline]
    fn spill(&mut self) {
        if let TrieOrVec::Small(len, data) = &self.0 {
            let vec = unsafe { inline_slice::<BYTES>(*len, data) }.to_vec();
            self.0 = TrieOrVec::Vec(vec);
        }
    }

    #[inline]
    pub fn new() -> Self {
        Self(Self::empty_repr())
    }

    #[inline]
    pub fn new_with_one(x: SlicedInt<BYTES>) -> Self {
        if Self::INLINE_CAP > 0 {
            let mut data = [0u8; INLINE_BYTES];
            {
                let dst = unsafe { inline_slice_mut::<BYTES>(1, &mut data) };
                dst[0] = x;
            }
            Self(TrieOrVec::Small(1, data))
        } else {
            Self(TrieOrVec::Vec(vec![x]))
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TrieOrVec::Vec(vec) => vec.len(),
            TrieOrVec::Trie(_, len) => *len,
            TrieOrVec::Small(len, _) => *len as usize,
        }
    }

    #[inline]
    pub fn count_nodes(&self) -> usize {
        match &self.0 {
            TrieOrVec::Vec(vec) => vec.len(),
            TrieOrVec::Trie(trie, _) => trie.count_nodes(),
            TrieOrVec::Small(len, _) => *len as usize,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            TrieOrVec::Vec(vec) => vec.is_empty(),
            TrieOrVec::Trie(_, len) => *len == 0,
            TrieOrVec::Small(len, _) => *len == 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        match &mut self.0 {
            TrieOrVec::Vec(vec) => {
                vec.clear();
            }
            TrieOrVec::Small(len, _) => {
                *len = 0;
            }
            TrieOrVec::Trie(_, _) => {
                self.0 = Self::empty_repr();
            }
        }
    }

    #[inline]
    pub fn contains(&self, x: &SlicedInt<BYTES>) -> bool {
        match &self.0 {
            TrieOrVec::Vec(vec) => vec.contains(x),
            TrieOrVec::Trie(trie, _) => trie.contains(&x.to_be_bytes()),
            TrieOrVec::Small(len, data) => unsafe { inline_slice::<BYTES>(*len, data) }.contains(x),
        }
    }

    pub fn insert(&mut self, x: SlicedInt<BYTES>) -> bool {
        // Cas inline traite d'abord ; on ne deborde vers `Vec` qu'a saturation.
        let mut overflow = false;
        if let TrieOrVec::Small(len, data) = &mut self.0 {
            let n = *len as usize;
            if unsafe { inline_slice::<BYTES>(*len, data) }.contains(&x) {
                return false;
            }
            if n < Self::INLINE_CAP {
                let dst = unsafe { inline_slice_mut::<BYTES>(*len + 1, data) };
                dst[n] = x;
                *len += 1;
                return true;
            }
            overflow = true;
        }
        if overflow {
            self.spill();
        }
        match &mut self.0 {
            TrieOrVec::Trie(trie, len) => {
                let absent = trie.insert(&x.to_be_bytes());
                if absent {
                    *len += 1;
                }
                absent
            }
            TrieOrVec::Vec(vec) => {
                if !vec.contains(&x) {
                    vec.push(x);
                    return true;
                }
                false
            }
            TrieOrVec::Small(_, _) => unreachable!("deborde juste avant"),
        }
    }

    pub fn remove(&mut self, x: &SlicedInt<BYTES>) -> bool {
        match &mut self.0 {
            TrieOrVec::Trie(trie, len) => {
                let present = trie.remove(&x.to_be_bytes());
                if present {
                    *len -= 1;
                }
                present
            }
            TrieOrVec::Vec(vec) => {
                if let Some(i) = vec.iter().position(|y| y == x) {
                    vec.swap_remove(i);
                    return true;
                }
                false
            }
            TrieOrVec::Small(len, data) => {
                let n = *len as usize;
                let slice = unsafe { inline_slice_mut::<BYTES>(*len, data) };
                if let Some(i) = slice.iter().position(|y| y == x) {
                    // meme semantique que `swap_remove` : le dernier prend la place
                    slice[i] = slice[n - 1];
                    *len -= 1;
                    return true;
                }
                false
            }
        }
    }

    #[inline]
    pub fn insert_iter<I: Iterator<Item = SlicedInt<BYTES>>>(&mut self, it: I) {
        for x in it {
            self.insert(x);
        }
    }

    #[inline]
    pub fn insert_sorted_iter<I: Iterator<Item = SlicedInt<BYTES>>>(&mut self, it: I) {
        match &self.0 {
            TrieOrVec::Trie(_, _) => {
                self.insert_iter(it);
            }
            // On repasse par `Vec` pour appliquer EXACTEMENT l'algorithme
            // d'origine : l'ordre final des elements dans le conteneur est alors
            // identique, donc les fichiers serialises le sont aussi.
            TrieOrVec::Small(_, _) => {
                self.spill();
                if let TrieOrVec::Vec(vec) = &mut self.0 {
                    Self::insert_sorted_into(vec, it);
                }
                self.compact();
            }
            TrieOrVec::Vec(_) => {
                if let TrieOrVec::Vec(vec) = &mut self.0 {
                    Self::insert_sorted_into(vec, it);
                }
            }
        }
    }

    #[inline]
    pub fn remove_iter<I: Iterator<Item = SlicedInt<BYTES>>>(&mut self, it: I) {
        for x in it {
            self.remove(&x);
        }
    }

    #[inline]
    pub fn remove_sorted_iter<I: Iterator<Item = SlicedInt<BYTES>>>(&mut self, it: I) {
        match &self.0 {
            TrieOrVec::Trie(_, _) => {
                self.remove_iter(it);
            }
            // Idem : l'algorithme d'origine collecte les indices puis applique
            // `swap_remove` en ordre INVERSE. Une suppression element par element
            // donnerait le meme ensemble mais un ordre different, donc des octets
            // differents a la serialisation. C'est ce qui faisait diverger
            // `inter` et `sym-diff`.
            TrieOrVec::Small(_, _) => {
                self.spill();
                if let TrieOrVec::Vec(vec) = &mut self.0 {
                    Self::remove_sorted_from(vec, it);
                }
                self.compact();
            }
            TrieOrVec::Vec(_) => {
                if let TrieOrVec::Vec(vec) = &mut self.0 {
                    Self::remove_sorted_from(vec, it);
                }
            }
        }
    }

    pub fn as_trie(&mut self) {
        match &self.0 {
            TrieOrVec::Vec(vec) => {
                let mut trie = Trie::new();
                for x in vec.iter() {
                    trie.insert(&x.to_be_bytes());
                }
                self.0 = TrieOrVec::Trie(trie, vec.len());
            }
            TrieOrVec::Small(len, data) => {
                // En pratique jamais atteint : `as_trie` n'est appele qu'au-dela
                // de THRESHOLD (1024), tres au-dessus de INLINE_CAP.
                let slice = unsafe { inline_slice::<BYTES>(*len, data) };
                let mut trie = Trie::new();
                for x in slice.iter() {
                    trie.insert(&x.to_be_bytes());
                }
                self.0 = TrieOrVec::Trie(trie, *len as usize);
            }
            TrieOrVec::Trie(_, _) => {}
        }
    }

    pub fn as_vec(&mut self) {
        if let TrieOrVec::Trie(trie, _) = &self.0 {
            let vec: Vec<SlicedInt<BYTES>> = trie
                .iter()
                .map(|bytes: [u8; BYTES]| SlicedInt::from_be_bytes(&bytes))
                .collect();
            self.0 = Self::repr_from_vec(vec);
        }
    }

    #[inline]
    pub fn sort(&mut self) {
        match &mut self.0 {
            TrieOrVec::Vec(vec) => vec.sort_unstable(),
            TrieOrVec::Small(len, data) => {
                unsafe { inline_slice_mut::<BYTES>(*len, data) }.sort_unstable()
            }
            TrieOrVec::Trie(_, _) => {}
        }
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> TrieVecIterator<'a, BYTES>
    where
        SlicedInt<BYTES>: 'a,
    {
        match &self.0 {
            TrieOrVec::Vec(vec) => TrieVecIterator::Vec(vec.iter()),
            TrieOrVec::Trie(trie, _) => TrieVecIterator::Trie(trie.iter()),
            // La tranche inline est un `&[SlicedInt]` : on reutilise la variante Vec.
            TrieOrVec::Small(len, data) => {
                TrieVecIterator::Vec(unsafe { inline_slice::<BYTES>(*len, data) }.iter())
            }
        }
    }

    #[inline]
    pub fn iter_sorted<'a>(&'a mut self) -> TrieVecIterator<'a, BYTES>
    where
        SlicedInt<BYTES>: 'a,
    {
        match &mut self.0 {
            TrieOrVec::Vec(vec) => {
                vec.sort_unstable();
                TrieVecIterator::Vec(vec.iter())
            }
            TrieOrVec::Trie(trie, _) => TrieVecIterator::Trie(trie.iter()),
            TrieOrVec::Small(len, data) => {
                let slice = unsafe { inline_slice_mut::<BYTES>(*len, data) };
                slice.sort_unstable();
                TrieVecIterator::Vec(slice.iter())
            }
        }
    }
}

impl<const BYTES: usize> Default for TrieVec<BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

pub enum TrieVecIterator<'a, const BYTES: usize> {
    Vec(Iter<'a, SlicedInt<BYTES>>),
    Trie(TrieIterator<'a, BYTES>),
}

impl<'a, const BYTES: usize> Iterator for TrieVecIterator<'a, BYTES> {
    type Item = SlicedInt<BYTES>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Vec(iter) => iter.next().copied(),
            Self::Trie(iter) => iter.next().map(|bytes| SlicedInt::from_be_bytes(&bytes)),
        }
    }
}

#[cfg(test)]
mod inline_tests {
    use super::*;

    const B: usize = 5; // BYTES pour p=28..32

    #[test]
    fn taille_de_lenum_inchangee() {
        assert_eq!(core::mem::size_of::<TrieVec<B>>(), 32);
        assert_eq!(TrieVec::<B>::INLINE_CAP, 4);
        assert_eq!(TrieVec::<6>::INLINE_CAP, 3);
    }

    #[test]
    fn insertion_inline_puis_debordement() {
        let mut tv = TrieVec::<B>::new();
        assert!(tv.is_empty());
        for i in 0..4u64 {
            assert!(tv.insert(SlicedInt::<B>::from_int(i + 1)));
        }
        assert_eq!(tv.len(), 4);
        assert!(matches!(tv.0, TrieOrVec::Small(4, _)));
        // doublon refuse, sans debordement
        assert!(!tv.insert(SlicedInt::<B>::from_int(3u64)));
        assert!(matches!(tv.0, TrieOrVec::Small(4, _)));
        // cinquieme element : bascule vers Vec
        assert!(tv.insert(SlicedInt::<B>::from_int(5u64)));
        assert_eq!(tv.len(), 5);
        assert!(matches!(tv.0, TrieOrVec::Vec(_)));
        for i in 0..5u64 {
            assert!(tv.contains(&SlicedInt::<B>::from_int(i + 1)));
        }
    }

    #[test]
    fn suppression_inline() {
        let mut tv = TrieVec::<B>::new();
        for i in 0..4u64 {
            tv.insert(SlicedInt::<B>::from_int(i + 1));
        }
        assert!(tv.remove(&SlicedInt::<B>::from_int(2u64)));
        assert_eq!(tv.len(), 3);
        assert!(!tv.contains(&SlicedInt::<B>::from_int(2u64)));
        assert!(!tv.remove(&SlicedInt::<B>::from_int(2u64)));
        for i in [1u64, 3, 4] {
            assert!(tv.contains(&SlicedInt::<B>::from_int(i)));
        }
    }

    #[test]
    fn iteration_et_tri() {
        let mut tv = TrieVec::<B>::new();
        for i in [4u64, 1, 3, 2] {
            tv.insert(SlicedInt::<B>::from_int(i));
        }
        let mut v: Vec<u64> = tv.iter_sorted().map(|x| x.get::<u64>()).collect();
        assert_eq!(v, vec![1, 2, 3, 4]);
        v = tv.iter().map(|x| x.get::<u64>()).collect();
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn ordre_identique_apres_suppression_triee() {
        // Le conteneur inline et le conteneur Vec doivent finir dans le MEME
        // ordre : c'est ce qui garantit des fichiers serialises identiques.
        let vals: Vec<SlicedInt<B>> = (1..=4u64).map(SlicedInt::<B>::from_int).collect();
        let mut small = TrieVec::<B>::new();
        for v in &vals {
            small.insert(*v);
        }
        let mut as_vec = TrieVec::<B>(TrieOrVec::Vec(vals.clone()));
        let dels: Vec<SlicedInt<B>> = [2u64, 3].iter().map(|&i| SlicedInt::<B>::from_int(i)).collect();
        small.remove_sorted_iter(dels.iter().copied());
        as_vec.remove_sorted_iter(dels.iter().copied());
        assert_eq!(
            bincode::serialize(&small).unwrap(),
            bincode::serialize(&as_vec).unwrap()
        );
    }

    #[test]
    fn ordre_identique_apres_insertion_triee() {
        let vals: Vec<SlicedInt<B>> = (1..=3u64).map(SlicedInt::<B>::from_int).collect();
        let mut small = TrieVec::<B>::new();
        for v in &vals {
            small.insert(*v);
        }
        let mut as_vec = TrieVec::<B>(TrieOrVec::Vec(vals.clone()));
        let ins: Vec<SlicedInt<B>> = [2u64, 5, 7].iter().map(|&i| SlicedInt::<B>::from_int(i)).collect();
        small.insert_sorted_iter(ins.iter().copied());
        as_vec.insert_sorted_iter(ins.iter().copied());
        assert_eq!(
            bincode::serialize(&small).unwrap(),
            bincode::serialize(&as_vec).unwrap()
        );
    }

    #[test]
    fn aller_retour_de_representation() {
        let vec: Vec<SlicedInt<B>> = (1..=3u64).map(SlicedInt::<B>::from_int).collect();
        let repr = TrieVec::<B>::repr_from_vec(vec.clone());
        assert!(matches!(repr, TrieOrVec::Small(3, _)));
        let big: Vec<SlicedInt<B>> = (1..=10u64).map(SlicedInt::<B>::from_int).collect();
        assert!(matches!(
            TrieVec::<B>::repr_from_vec(big),
            TrieOrVec::Vec(_)
        ));
    }

    #[test]
    fn serialisation_identique_a_un_vec() {
        // Un conteneur Small doit produire exactement les memes octets
        // qu'un conteneur Vec de meme contenu : c'est ce qui garantit
        // l'identite des fichiers d'index.
        let mut small = TrieVec::<B>::new();
        for i in 1..=3u64 {
            small.insert(SlicedInt::<B>::from_int(i));
        }
        let vec_repr = TrieVec::<B>(TrieOrVec::Vec(
            (1..=3u64).map(SlicedInt::<B>::from_int).collect(),
        ));
        let a = bincode::serialize(&small).unwrap();
        let b = bincode::serialize(&vec_repr).unwrap();
        assert_eq!(a, b);
        let back: TrieVec<B> = bincode::deserialize(&a).unwrap();
        assert_eq!(back.len(), 3);
        assert!(matches!(back.0, TrieOrVec::Small(3, _)));
    }
}

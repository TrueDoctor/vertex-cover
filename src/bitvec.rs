use core::arch::x86_64::*;
use std::ops::{BitAnd, BitOr, BitOrAssign, Not};

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct BitVec256 {
    // Store data directly as AVX2 register to avoid loads
    simd: __m256i,
}

// Iterator over set bits
pub struct BitIter {
    // Current chunk being processed
    chunk: u64,
    // Index of current chunk (0-3)
    chunk_idx: usize,
    // Base offset for current chunk
    base: usize,
    // Remaining chunks
    remaining: [u64; 3],
}

impl Iterator for BitIter {
    type Item = usize;

    // #[inline]
    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path for empty chunk
        while self.chunk == 0 {
            if self.chunk_idx >= 3 {
                return None;
            }
            self.chunk = self.remaining[self.chunk_idx];
            self.base = (self.chunk_idx + 1) * 64;
            self.chunk_idx += 1;
        }

        // Use leading zeros to find next set bit
        unsafe {
            // LZCNT returns number of leading zeros
            let zeros = _lzcnt_u64(self.chunk) as usize;
            // Clear the highest set bit we found
            self.chunk &= !(1u64 << (63 - zeros));
            // Return the bit position
            Some(self.base + (63 - zeros))
        }
    }

    // #[inline]
    #[inline(never)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_ones = unsafe {
            _popcnt64(self.chunk as i64) as usize
                + self.remaining[self.chunk_idx..]
                    .iter()
                    .map(|&x| _popcnt64(x as i64) as usize)
                    .sum::<usize>()
        };
        (remaining_ones, Some(remaining_ones))
    }
}

impl std::fmt::Debug for BitVec256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let mut arr = [0u64; 4];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);
            f.debug_struct("BitVec256").field("data", &arr).finish()
        }
    }
}

impl Default for BitVec256 {
    #[inline]
    fn default() -> Self {
        unsafe {
            Self {
                simd: _mm256_setzero_si256(),
            }
        }
    }
}

impl BitVec256 {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    // Create from u64 array
    #[inline]
    pub unsafe fn from_u64s(arr: [u64; 4]) -> Self {
        Self {
            simd: _mm256_loadu_si256(arr.as_ptr() as *const __m256i),
        }
    }

    #[inline]
    pub unsafe fn as_u64s(&self) -> [u64; 4] {
        let mut arr = [0u64; 4];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);
        arr
    }

    // Set a specific bit
    #[inline(never)]
    pub fn set(&mut self, index: usize) {
        unsafe {
            // Convert to array, modify, convert back to avoid multiple loads/stores
            let mut arr = [0u64; 4];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);

            let array_idx = index / 64;
            let bit_idx = index % 64;
            arr[array_idx] |= 1u64 << bit_idx;

            self.simd = _mm256_loadu_si256(arr.as_ptr() as *const __m256i);
        }
    }
    #[inline(never)]
    pub fn set_mask(&mut self, index: Self) {
        self.bitor_assign(index);
    }

    // Set a specific bit
    #[inline(never)]
    pub fn clear(&mut self, index: usize) {
        unsafe {
            // Convert to array, modify, convert back to avoid multiple loads/stores
            let mut arr = [0u64; 4];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);

            let array_idx = index / 64;
            let bit_idx = index % 64;
            arr[array_idx] &= !(1u64 << bit_idx);

            self.simd = _mm256_loadu_si256(arr.as_ptr() as *const __m256i);
        }
    }

    // Get a specific bit
    #[inline(never)]
    pub fn get(&self, index: usize) -> bool {
        unsafe {
            let mut arr = [0u64; 4];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);

            let array_idx = index / 64;
            let bit_idx = index % 64;
            (arr[array_idx] & (1u64 << bit_idx)) != 0
        }
    }
    #[inline(never)]
    pub fn get_mask(self, index: Self) -> bool {
        !(self & index).is_zero()
    }

    // Returns an iterator over set bits
    #[inline(never)]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        unsafe {
            let [first, a, b, c] = self.as_u64s();
            BitIter {
                chunk: first,
                chunk_idx: 0,
                base: 0,
                remaining: [a, b, c],
            }
        }
    }

    // Count the number of set bits using POPCNT
    #[inline(never)]
    pub fn count_ones(&self) -> u32 {
        unsafe {
            // Convert __m256i to __m256i counts using VPOPCNT
            let popcnt = _mm256_popcnt_epi16(self.simd);

            // Horizontal sum of 64-bit integers using AVX-512
            let sum = _mm256_reduce_add_epi16(popcnt);

            sum as u32
        }
    }

    // pub fn count_ones(&self) -> u32 {
    //     unsafe {
    //         let mut arr = [0u64; 4];
    //         _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.simd);

    //         arr.iter().map(|&x| _popcnt64(x as i64) as u32).sum()
    //     }
    // }

    // Check if all bits are zero
    #[inline]
    // #[inline(never)]
    pub fn is_zero(&self) -> bool {
        unsafe {
            // Compare with zero
            let mask = _mm256_cmpeq_epi64(self.simd, _mm256_setzero_si256());
            // Check if all lanes are equal to zero
            _mm256_testc_si256(mask, _mm256_set1_epi64x(-1)) == 1
        }
    }

    // #[inline]
    #[inline(never)]
    pub fn and_not(&self, other: &Self) -> Self {
        unsafe {
            Self {
                // Use ANDNOT instruction directly - this computes (!b & a) in one operation
                simd: _mm256_andnot_si256(other.simd, self.simd),
            }
        }
    }

    pub(crate) fn from_bit(n: u32) -> Self {
        let mut arr = [0; 4];
        let array_idx = n / 64;
        let bit_idx = n % 64;
        arr[array_idx as usize] |= 1u64 << bit_idx;

        Self {
            simd: unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) },
        }
    }
}

// Implement bitwise AND using AVX2
impl BitAnd for BitVec256 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                simd: _mm256_and_si256(self.simd, rhs.simd),
            }
        }
    }
}

// Implement bitwise OR using AVX2
impl BitOr for BitVec256 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                simd: _mm256_or_si256(self.simd, rhs.simd),
            }
        }
    }
}

// Implement bitwise NOT using AVX2
impl Not for BitVec256 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        unsafe {
            Self {
                simd: _mm256_xor_si256(self.simd, _mm256_set1_epi64x(-1)),
            }
        }
    }
}

// Implement BitAndAssign
impl std::ops::BitAndAssign for BitVec256 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        unsafe {
            self.simd = _mm256_and_si256(self.simd, rhs.simd);
        }
    }
}

// Implement BitOrAssign
impl std::ops::BitOrAssign for BitVec256 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        unsafe {
            self.simd = _mm256_or_si256(self.simd, rhs.simd);
        }
    }
}

// Implement AndNot assign operation
impl BitVec256 {
    #[inline]
    pub fn and_not_assign(&mut self, other: &Self) {
        unsafe {
            self.simd = _mm256_andnot_si256(other.simd, self.simd);
        }
    }
}

// Implement index operator for convenient bit access
impl std::ops::Index<usize> for BitVec256 {
    type Output = bool;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if self.get(index) {
            &true
        } else {
            &false
        }
    }
}

// Additional helper traits
impl PartialEq for BitVec256 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mask = _mm256_cmpeq_epi64(self.simd, other.simd);
            _mm256_testc_si256(mask, _mm256_set1_epi64x(-1)) == 1
        }
    }
}

impl Eq for BitVec256 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_not() {
        let mut a = BitVec256::new();
        let mut b = BitVec256::new();

        // Set test pattern
        a.set(0);
        a.set(1);
        a.set(64);

        b.set(1);
        b.set(2);
        b.set(64);

        let result = a.and_not(&b);
        assert!(result.get(0)); // 1 & !0 = 1
        assert!(!result.get(1)); // 1 & !1 = 0
        assert!(!result.get(2)); // 0 & !1 = 0
        assert!(!result.get(64)); // 1 & !1 = 0
    }

    #[test]
    fn test_assign_ops() {
        let mut a = BitVec256::new();
        let mut b = BitVec256::new();

        a.set(0);
        a.set(128);
        b.set(0);
        b.set(255);

        // Test &=
        let mut c = a;
        c &= b;
        assert!(c.get(0));
        assert!(!c.get(128));
        assert!(!c.get(255));

        // Test |=
        let mut c = a;
        c |= b;
        assert!(c.get(0));
        assert!(c.get(128));
        assert!(c.get(255));

        // Test and_not_assign
        let mut c = a;
        c.and_not_assign(&b);
        assert!(!c.get(0));
        assert!(c.get(128));
        assert!(!c.get(255));
    }

    #[test]
    fn test_set_get() {
        let mut bv = BitVec256::new();
        bv.set(0);
        bv.set(255);
        assert!(bv.get(0));
        assert!(bv.get(255));
        assert!(!bv.get(1));
    }

    #[test]
    fn test_bitwise_ops() {
        let mut a = BitVec256::new();
        let mut b = BitVec256::new();

        a.set(0);
        a.set(128);
        b.set(0);
        b.set(255);

        let and = a & b;
        assert!(and.get(0));
        assert!(!and.get(128));
        assert!(!and.get(255));

        let or = a | b;
        assert!(or.get(0));
        assert!(or.get(128));
        assert!(or.get(255));

        let not = !a;
        assert!(!not.get(0));
        assert!(!not.get(128));
        assert!(not.get(1));
    }

    #[test]
    fn test_count_ones() {
        let mut bv = BitVec256::new();
        assert_eq!(bv.count_ones(), 0);

        bv.set(0);
        bv.set(64);
        bv.set(128);
        bv.set(255);
        assert_eq!(bv.count_ones(), 4);
    }

    #[test]
    fn test_equal() {
        let mut a = BitVec256::new();
        let mut b = BitVec256::new();

        assert_eq!(a, b);

        a.set(0);
        assert_ne!(a, b);

        b.set(0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_iter_ones() {
        let mut bv = BitVec256::new();
        bv.set(0);
        bv.set(64);
        bv.set(128);
        bv.set(255);

        let ones: Vec<usize> = bv.iter_ones().collect();
        assert_eq!(ones, vec![0, 64, 128, 255]);
    }
}

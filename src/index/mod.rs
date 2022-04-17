use crate::traits::Index;

pub struct FullSearch {
    len: usize,
}

impl FullSearch {
    pub fn new<T>(data: &[T]) -> Self {
        let len = data.len();
        Self { len }
    }
}

impl<T> Index<T> for FullSearch {
    type It = std::ops::Range<usize>;

    fn construct(data: &[T]) -> Self {
        Self::new(data)
    }

    fn neighbors(&self, _center: &T) -> Self::It {
        0..self.len
    }
}

pub struct Octree<const NBIT: usize> {
    origin: [f64; 3],
    size: f64,
    sample_index: Vec<Vec<usize>>,
}

impl<const NBIT: usize> Octree<NBIT> {
    pub fn new(data: &[[f64; 3]]) -> Self {
        // compute size
        let (min, max) = data.iter().fold(
            ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
            |(mut max, mut min), vs| {
                min.iter_mut().zip(vs).for_each(|(min, &v)| {
                    if v < *min {
                        *min = v
                    }
                });
                max.iter_mut().zip(vs).for_each(|(max, &v)| {
                    if v > *max {
                        *max = v
                    }
                });
                (min, max)
            },
        );
        let origin = min;
        let size = min
            .iter()
            .zip(&max)
            .fold(0_f64, |max_size, (x0, x1)| max_size.max(x1 - x0))
            / f64::powi(2., i32::try_from(NBIT).unwrap());

        let n = usize::pow(2, u32::try_from(3 * NBIT).unwrap());
        let mut sample_index = Vec::with_capacity(n);
        for _i in 0..n {
            sample_index.push(Vec::new());
        }

        for (k, x) in data.iter().enumerate() {
            let n = Self::index(&size, &origin, x);
            sample_index[n].push(k);
        }

        for indices in sample_index.iter_mut() {
            indices.sort();
            indices.resize(indices.len(), 0);
        }

        Self {
            origin,
            size,
            sample_index,
        }
    }

    #[inline]
    fn index(size: &f64, origin: &[f64; 3], center: &[f64; 3]) -> usize {
        let shifts = [0, NBIT, 2 * NBIT];
        let mut n = 0;
        for ((shift, x), x0) in shifts.into_iter().zip(center).zip(origin) {
            let i = ((x - x0) / size) as usize;
            assert!(i < 1 << NBIT);
            n |= i << shift;
        }
        n
    }
}

impl<const NBIT: usize> Index<[f64; 3]> for Octree<NBIT> {
    fn construct(data: &[[f64; 3]]) -> Self {
        Self::new(data)
    }

    fn neighbors(&self, center: &[f64; 3]) -> Self::It {
        let n = Self::index(&self.size, &self.origin, center);
        Vec::from(&self.sample_index[n][..]).into_iter()
    }

    type It = <Vec<usize> as IntoIterator>::IntoIter;
}

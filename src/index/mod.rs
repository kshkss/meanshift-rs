use crate::traits::Index;

#[derive(Debug, Clone)]
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

    fn refresh(&mut self, data: &[T]) {
        assert_eq!(self.len, data.len());
    }

    fn neighbors(&self, _center: &T) -> Self::It {
        0..self.len
    }
}

#[derive(Debug, Clone)]
pub struct Octree<const NBIT: usize> {
    origin: [f64; 3],
    size: f64,
    sample_index: Vec<Vec<usize>>,
    where_: Vec<Where>,
}

#[derive(Debug, Clone)]
struct Where {
    node_index: usize,
    obj_index: usize,
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

        let mut where_ = Vec::with_capacity(data.len());
        for (k, x) in data.iter().enumerate() {
            let n = Self::index(&size, &origin, x);
            sample_index[n].push(k);
            where_.push(Where {
                node_index: n,
                obj_index: sample_index[n].len(),
            });
        }

        Self {
            origin,
            size,
            sample_index,
            where_,
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
    fn refresh(&mut self, data: &[[f64; 3]]) {
        assert_eq!(self.where_.len(), data.len());

        for (k, x) in data.iter().enumerate() {
            let n = Self::index(&self.size, &self.origin, x);
            let old = self.where_[k].clone();
            assert_eq!(k, self.sample_index[old.node_index][old.obj_index]);
            self.sample_index[old.node_index].swap_remove(old.obj_index);
            if old.obj_index < self.sample_index[old.node_index].len() {
                self.where_[self.sample_index[old.node_index][old.obj_index]] = Where {
                    node_index: old.node_index,
                    obj_index: old.obj_index,
                };
            }
            self.sample_index[n].push(k);
            self.where_[k] = Where {
                node_index: n,
                obj_index: self.sample_index[n].len(),
            };
        }
    }

    fn neighbors(&self, center: &[f64; 3]) -> Self::It {
        let n = Self::index(&self.size, &self.origin, center);
        Vec::from(&self.sample_index[n][..]).into_iter()
    }

    type It = <Vec<usize> as IntoIterator>::IntoIter;
}

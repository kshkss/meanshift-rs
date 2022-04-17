pub mod index;
pub mod traits;

use crate::traits::Index;

pub struct Meanshift<const N: usize> {
    max_iter: usize,
    atol: [f64; N],
    rtol: [f64; N],
    threshold: f64,
}

impl<const N: usize> Meanshift<N> {
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            atol: [1e-8; N],
            rtol: [1e-8; N],
            threshold: 0.5,
        }
    }

    pub fn with_max_iter(self, max_iter: usize) -> Self {
        Self { max_iter, ..self }
    }

    pub fn with_atol(self, atol: [f64; N]) -> Self {
        Self { atol, ..self }
    }

    pub fn with_rtol(self, rtol: [f64; N]) -> Self {
        Self { rtol, ..self }
    }

    pub fn with_threshold(self, threshold: f64) -> Self {
        Self { threshold, ..self }
    }

    pub fn mode(
        &self,
        f: impl Fn(&[f64], &[f64]) -> f64,
        samples: &[[f64; N]],
        init: &[f64; N],
    ) -> ([f64; N], f64) {
        let index = index::FullSearch::new(samples);
        self.mode_with_index(f, samples, &index, init)
    }

    pub fn mode_with_index(
        &self,
        f: impl Fn(&[f64], &[f64]) -> f64,
        samples: &[[f64; N]],
        index: &impl Index<[f64; N]>,
        init: &[f64; N],
    ) -> ([f64; N], f64) {
        let mut result = init.clone();
        let mut weight = 0.;
        for _i in 0..self.max_iter {
            let mut mean = [0.; N];
            weight = std::f64::MIN_POSITIVE;
            for k in index.neighbors(&result) {
                let row = &samples[k];
                let w = f(&result, row);
                weight += w;
                for (m, x) in mean.iter_mut().zip(row) {
                    *m += (*x - *m) * w / weight;
                }
            }
            if result
                .iter()
                .zip(&mean)
                .zip(&self.atol)
                .zip(&self.rtol)
                .all(|(((m0, m1), atol), rtol)| {
                    (m0 - m1).abs() < atol + rtol * m0.abs().max(m1.abs())
                })
            {
                return (mean, weight);
            }
            result = mean;
        }
        (result, weight)
    }

    pub fn clustering(
        &self,
        f: impl Fn(&[f64], &[f64]) -> f64,
        samples: &[[f64; N]],
        seeds: &[[f64; N]],
    ) -> (Vec<usize>, Vec<[f64; N]>) {
        let index = index::FullSearch::new(samples);
        self.clustering_with_index(f, samples, &index, seeds)
    }

    pub fn clustering_with_index<T>(
        &self,
        f: impl Fn(&[f64], &[f64]) -> f64,
        samples: &[[f64; N]],
        index: &T,
        seeds: &[[f64; N]],
    ) -> (Vec<usize>, Vec<[f64; N]>)
    where
        T: Index<[f64; N]>,
    {
        let mut centers = Vec::new();
        let mut weights = Vec::new();
        for (k, row) in seeds.iter().enumerate() {
            let (m, w) = self.mode_with_index(&f, samples, index, row);
            centers.push(m);
            weights.push((w, k));
        }

        // Postporocess
        weights.sort_unstable_by(|&(v1, _), (v2, _)| {
            v1.partial_cmp(v2).expect("A seeds converged to nan")
        });
        let index = T::construct(&centers);

        let mut unique_centers = Vec::<[f64; N]>::new();
        let mut labels = Vec::with_capacity(seeds.len());
        labels.resize(seeds.len(), 0);
        let mut candidate = Vec::with_capacity(seeds.len());
        candidate.resize(seeds.len(), true);
        for (_, i) in weights.into_iter() {
            if candidate[i] {
                candidate[i] = false;
                let center = &centers[i];
                let label = unique_centers.len();
                for k in index.neighbors(&center) {
                    let row = &centers[k];
                    if candidate[k] && f(center, row) > self.threshold * f(center, center) {
                        labels[k] = label;
                        candidate[k] = false;
                    }
                }
                unique_centers.push(center.clone());
            }
        }
        (labels, unique_centers)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

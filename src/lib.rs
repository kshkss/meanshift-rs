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
        init: &[f64; N],
        data: &[[f64; N]],
    ) -> ([f64; N], f64) {
        let mut result = init.clone();
        let mut weight = 0.;
        for _i in 0..self.max_iter {
            let mut mean = [0.; N];
            weight = std::f64::MIN_POSITIVE;
            for row in data.iter() {
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
                .all(|(((m0, m1), atol), rtol)| (m0 - m1).abs() < atol + rtol * m0.abs().max(m1.abs()))
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
        seeds: &[[f64; N]],
        data: &[[f64; N]],
    ) -> (Vec<usize>, Vec<[f64; N]>) {
        use std::collections::BinaryHeap;
        let mut heap = BinaryHeap::new();
        for (k, row) in seeds.iter().enumerate() {
            let (m, w) = self.mode(&f, row, data);
            heap.push(Center(k, w, m));
        }

        let mut unique_centers = Vec::<[f64; N]>::with_capacity(seeds.len());
        let mut labels = Vec::with_capacity(seeds.len());
        labels.resize(seeds.len(), 0);
        while !heap.is_empty() {
            let Center(k, _, row) = heap.pop().expect("somthing wrong");
            let mut unique = true;
            for (i, center) in unique_centers.iter().enumerate() {
                if f(center, &row) > self.threshold * f(center, center) {
                    labels[k] = i;
                    unique = false;
                    break;
                }
            }
            if unique {
                unique_centers.push(row);
                labels[k] = unique_centers.len() - 1;
            }
        }
        (labels, unique_centers)
    }
}

use std::cmp::Ordering;

#[derive(Debug, Clone)]
struct Center<const N: usize>(usize, f64, [f64; N]);

impl<const N: usize> PartialEq for Center<N> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<const N: usize> Eq for Center<N> {}

impl<const N: usize> PartialOrd for Center<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl<const N: usize> Ord for Center<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(ordering) = self.partial_cmp(&other) {
            ordering
        } else {
            self.0.cmp(&other.0)
        }
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

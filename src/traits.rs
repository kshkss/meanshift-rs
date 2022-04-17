pub trait Index<T> {
    type It: Iterator<Item = usize>;
    fn refresh(&mut self, samples: &[T]);
    fn neighbors(&self, center: &T) -> Self::It;
}

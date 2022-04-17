pub trait Index<T> {
    type It: Iterator<Item = usize>;
    fn construct(samples: &[T]) -> Self;
    fn neighbors(&self, center: &T) -> Self::It;
}

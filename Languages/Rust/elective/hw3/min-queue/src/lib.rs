#![forbid(unsafe_code)]

use std::collections::VecDeque;

#[derive(Default)]
pub struct MinQueue<T> {
    data: VecDeque<(T, usize)>,
    min_list: VecDeque<(T, usize)>,
}

impl<T: Clone + Ord> MinQueue<T> {
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
            min_list: VecDeque::new(),
        }
    }

    pub fn push(&mut self, val: T) {
        loop {
            match self.min_list.back() {
                Some((last, _)) if *last > val => _ = self.min_list.pop_back(),
                _ => break,
            }
        }

        let index = match self.data.back() {
            Some((_, index)) => index + 1,
            None => 0usize,
        };
        self.data.push_back((val.clone(), index));
        self.min_list.push_back((val, index));
    }

    pub fn pop(&mut self) -> Option<T> {
        match self.data.pop_front() {
            None => None,
            Some((val, index)) => {
                if self.min_list.front()?.1 == index {
                    _ = self.min_list.pop_front();
                }
                Some(val)
            }
        }
    }

    pub fn front(&self) -> Option<&T> {
        match self.data.front() {
            None => None,
            Some((val, _)) => Some(val),
        }
    }

    pub fn min(&self) -> Option<&T> {
        match self.min_list.front() {
            None => None,
            Some((val, _)) => Some(val),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::iter::Peekable;
use std::rc::Rc;

pub struct LazyCycle<I>
where
    I: Iterator,
{
    orig: I,
    buffer: Vec<I::Item>,
    pos: usize,
    done: bool,
}

impl<I> LazyCycle<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn new(iter: I) -> Self {
        LazyCycle {
            orig: iter,
            buffer: Vec::new(),
            pos: 0,
            done: false,
        }
    }
}

impl<I> Iterator for LazyCycle<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.done {
            match self.orig.next() {
                Some(v) => {
                    self.buffer.push(v.clone());
                    return Some(v);
                }
                None => {
                    self.done = true;
                    self.pos = 0;
                }
            }
        }
        if self.buffer.is_empty() {
            None
        } else {
            let item = self.buffer[self.pos].clone();
            self.pos = (self.pos + 1) % self.buffer.len();
            Some(item)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct Extract<I: Iterator> {
    buffer: VecDeque<I::Item>,
    iter: I,
}

impl<I: Iterator> Extract<I> {
    fn new(buffer: VecDeque<I::Item>, iter: I) -> Self {
        Extract { buffer, iter }
    }
}

impl<I: Iterator> Iterator for Extract<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.buffer.pop_front() {
            Some(v)
        } else {
            self.iter.next()
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    inner: Rc<RefCell<Inner<I>>>,
    index: usize,
}

struct Inner<I>
where
    I: Iterator,
    I::Item: Clone,
{
    iter: I,
    buf: Vec<Option<I::Item>>,
    finished: bool,
}

impl<I> Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn new(iter: I) -> (Self, Self) {
        let inner = Rc::new(RefCell::new(Inner {
            iter,
            buf: Vec::new(),
            finished: false,
        }));
        (
            Tee {
                inner: inner.clone(),
                index: 0,
            },
            Tee { inner, index: 0 },
        )
    }
}

impl<I> Iterator for Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inner = self.inner.borrow_mut();
        if self.index < inner.buf.len() {
            return if let Some(item) = inner.buf[self.index].take() {
                self.index += 1;
                Some(item)
            } else {
                self.index += 1;
                None
            };
        }

        if !inner.finished {
            return match inner.iter.next() {
                Some(v) => {
                    let ret = v.clone();
                    inner.buf.push(Some(v));
                    self.index += 1;
                    Some(ret)
                }
                None => {
                    inner.finished = true;
                    None
                }
            };
        }
        None
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct GroupBy<I, F, V>
where
    I: Iterator,
    F: FnMut(&I::Item) -> V,
    V: Eq,
{
    iter: Peekable<I>,
    f: F,
}

impl<I, F, V> GroupBy<I, F, V>
where
    I: Iterator,
    F: FnMut(&I::Item) -> V,
    V: Eq,
{
    fn new(iter: I, f: F) -> Self {
        GroupBy {
            iter: iter.peekable(),
            f,
        }
    }
}

impl<I, F, V> Iterator for GroupBy<I, F, V>
where
    I: Iterator,
    F: FnMut(&I::Item) -> V,
    V: Eq,
{
    type Item = (V, Vec<I::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter.next()?;
        let key = (self.f)(&first);
        let mut group = vec![first];
        while let Some(peek) = self.iter.peek() {
            if (self.f)(peek) == key {
                group.push(self.iter.next().unwrap());
            } else {
                break;
            }
        }
        Some((key, group))
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait ExtendedIterator: Iterator {
    fn lazy_cycle(self) -> LazyCycle<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        LazyCycle::new(self)
    }

    fn extract(mut self, index: usize) -> (Option<Self::Item>, Extract<Self>)
    where
        Self: Sized,
    {
        let mut buf = VecDeque::new();
        let mut i = 0;
        while i < index {
            if let Some(v) = self.next() {
                buf.push_back(v);
            } else {
                break;
            }
            i += 1;
        }
        let extracted = self.next();
        (extracted, Extract::new(buf, self))
    }

    fn tee(self) -> (Tee<Self>, Tee<Self>)
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Tee::new(self)
    }

    fn group_by<F, V>(self, func: F) -> GroupBy<Self, F, V>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> V,
        V: Eq,
    {
        GroupBy::new(self, func)
    }
}

impl<I: Iterator> ExtendedIterator for I {}

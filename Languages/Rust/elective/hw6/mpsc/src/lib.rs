#![forbid(unsafe_code)]

use std::{cell::RefCell, collections::VecDeque, fmt::Debug, rc::Rc};
use thiserror::Error;

////////////////////////////////////////////////////////////////////////////////

struct Stream<T> {
    elements: VecDeque<T>,
    is_closed: bool,
}

impl<T> Stream<T> {
    fn new() -> Self {
        Self {
            elements: VecDeque::new(),
            is_closed: false,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Error, Debug)]
#[error("channel is closed")]
pub struct SendError<T> {
    pub value: T,
}

pub struct Sender<T> {
    stream: Rc<RefCell<Stream<T>>>,
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut stream_ref = self.stream.borrow_mut();
        if stream_ref.is_closed {
            return Err(SendError { value });
        }
        stream_ref.elements.push_back(value);
        Ok(())
    }

    pub fn is_closed(&self) -> bool {
        self.stream.borrow().is_closed
    }

    pub fn same_channel(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.stream, &other.stream)
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            stream: self.stream.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if Rc::strong_count(&self.stream) == 2 {
            self.stream.borrow_mut().is_closed = true;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Error, Debug)]
pub enum ReceiveError {
    #[error("channel is empty")]
    Empty,
    #[error("channel is closed")]
    Closed,
}

pub struct Receiver<T> {
    stream: Rc<RefCell<Stream<T>>>,
}

impl<T> Receiver<T> {
    pub fn recv(&mut self) -> Result<T, ReceiveError> {
        let mut stream_ref = self.stream.borrow_mut();
        match stream_ref.elements.pop_front() {
            None => Err(if stream_ref.is_closed {
                ReceiveError::Closed
            } else {
                ReceiveError::Empty
            }),
            Some(val) => Ok(val),
        }
    }

    pub fn close(&mut self) {
        self.stream.borrow_mut().is_closed = true
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.close()
    }
}

////////////////////////////////////////////////////////////////////////////////

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let storage = Rc::new(RefCell::new(Stream::new()));
    (
        Sender::<T> {
            stream: storage.clone(),
        },
        Receiver::<T> { stream: storage },
    )
}

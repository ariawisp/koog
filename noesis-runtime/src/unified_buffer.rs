// Unified buffer views for zero-copy channel separation

use std::ops::Range;

/// A view into a unified buffer without copying
#[derive(Clone, Debug)]
pub struct BufferView {
    offset: usize,
    length: usize,
}

impl BufferView {
    pub fn new(offset: usize, length: usize) -> Self {
        BufferView { offset, length }
    }
    
    pub fn range(&self) -> Range<usize> {
        self.offset..(self.offset + self.length)
    }
    
    pub fn offset(&self) -> usize {
        self.offset
    }
    
    pub fn length(&self) -> usize {
        self.length
    }
    
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}
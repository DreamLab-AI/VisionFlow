// Adaptive Heap for WASM - Implements Pull/Insert/BatchPrepend operations
// This is the sophisticated data structure required by the paper's algorithm

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/
/
pub struct AdaptiveHeap {
    
    block_size: usize,

    
    primary_heap: BinaryHeap<HeapEntry>,

    
    blocks: Vec<Block>,

    
    active_block: usize,

    
    vertex_distances: HashMap<u32, f32>,

    
    capacity: usize,
}

/
#[derive(Clone, Debug)]
struct HeapEntry {
    vertex: u32,
    distance: f32,
}

/
#[derive(Clone, Debug)]
struct Block {
    entries: Vec<HeapEntry>,
    min_distance: f32,
    max_distance: f32,
}

impl AdaptiveHeap {
    
    pub fn new(capacity: usize) -> Self {
        let block_size = (capacity as f32).sqrt().ceil() as usize;

        Self {
            block_size,
            primary_heap: BinaryHeap::with_capacity(capacity),
            blocks: Vec::new(),
            active_block: 0,
            vertex_distances: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    
    pub fn insert(&mut self, vertex: u32, distance: f32) {
        
        if let Some(&existing_dist) = self.vertex_distances.get(&vertex) {
            if existing_dist <= distance {
                return; 
            }
        }

        self.vertex_distances.insert(vertex, distance);
        self.primary_heap.push(HeapEntry { vertex, distance });
    }

    
    
    pub fn batch_prepend(&mut self, vertices: &[u32], distances: &[f32]) {
        if vertices.is_empty() {
            return;
        }

        
        let mut block = Block {
            entries: Vec::with_capacity(vertices.len()),
            min_distance: f32::INFINITY,
            max_distance: f32::NEG_INFINITY,
        };

        
        for (&vertex, &distance) in vertices.iter().zip(distances.iter()) {
            
            if let Some(&existing_dist) = self.vertex_distances.get(&vertex) {
                if existing_dist <= distance {
                    continue;
                }
            }

            self.vertex_distances.insert(vertex, distance);
            block.entries.push(HeapEntry { vertex, distance });
            block.min_distance = block.min_distance.min(distance);
            block.max_distance = block.max_distance.max(distance);
        }

        if !block.entries.is_empty() {
            
            self.blocks.push(block);

            
            if self.blocks.len() > self.block_size {
                self.merge_blocks();
            }
        }
    }

    
    pub fn pull(&mut self, m: usize) -> Vec<(u32, f32)> {
        let mut result = Vec::with_capacity(m);

        
        self.merge_blocks();

        
        for _ in 0..m {
            if let Some(entry) = self.extract_min() {
                result.push((entry.vertex, entry.distance));
            } else {
                break;
            }
        }

        result
    }

    
    fn extract_min(&mut self) -> Option<HeapEntry> {
        while let Some(entry) = self.primary_heap.pop() {
            
            if let Some(&current_dist) = self.vertex_distances.get(&entry.vertex) {
                if (current_dist - entry.distance).abs() < 1e-6 {
                    
                    self.vertex_distances.remove(&entry.vertex);
                    return Some(entry);
                }
                
            }
        }
        None
    }

    
    fn merge_blocks(&mut self) {
        if self.blocks.is_empty() {
            return;
        }

        
        let mut all_entries = Vec::new();
        for block in self.blocks.drain(..) {
            all_entries.extend(block.entries);
        }

        
        all_entries.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        
        for entry in all_entries {
            
            if let Some(&current_dist) = self.vertex_distances.get(&entry.vertex) {
                if (current_dist - entry.distance).abs() < 1e-6 {
                    self.primary_heap.push(entry);
                }
            }
        }
    }

    
    pub fn size(&self) -> usize {
        self.vertex_distances.len()
    }

    
    pub fn is_empty(&self) -> bool {
        self.vertex_distances.is_empty()
    }

    
    pub fn clear(&mut self) {
        self.primary_heap.clear();
        self.blocks.clear();
        self.vertex_distances.clear();
    }

    
    pub fn peek_min(&self) -> Option<f32> {
        
        let heap_min = self.primary_heap.peek().map(|e| e.distance);
        let block_min = self
            .blocks
            .iter()
            .map(|b| b.min_distance)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        match (heap_min, block_min) {
            (Some(h), Some(b)) => Some(h.min(b)),
            (Some(h), None) => Some(h),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

// Implement ordering for heap entries (min-heap based on distance)
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex && self.distance == other.distance
    }
}

impl Eq for HeapEntry {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_pull() {
        let mut heap = AdaptiveHeap::new(100);

        
        heap.insert(1, 5.0);
        heap.insert(2, 3.0);
        heap.insert(3, 7.0);
        heap.insert(4, 1.0);

        
        let result = heap.pull(2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (4, 1.0));
        assert_eq!(result[1], (2, 3.0));
    }

    #[test]
    fn test_batch_prepend() {
        let mut heap = AdaptiveHeap::new(100);

        
        heap.insert(1, 5.0);
        heap.insert(2, 3.0);

        
        let vertices = vec![3, 4, 5];
        let distances = vec![2.0, 6.0, 1.0];
        heap.batch_prepend(&vertices, &distances);

        
        let result = heap.pull(5);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], (5, 1.0)); 
        assert_eq!(result[1], (3, 2.0));
        assert_eq!(result[2], (2, 3.0));
    }

    #[test]
    fn test_duplicate_handling() {
        let mut heap = AdaptiveHeap::new(100);

        
        heap.insert(1, 5.0);
        heap.insert(1, 3.0); 
        heap.insert(1, 7.0); 

        let result = heap.pull(1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (1, 3.0)); 
    }

    #[test]
    fn test_block_merging() {
        let mut heap = AdaptiveHeap::new(100);

        
        for i in 0..5 {
            let vertices = vec![i * 10, i * 10 + 1];
            let distances = vec![i as f32, i as f32 + 0.5];
            heap.batch_prepend(&vertices, &distances);
        }

        
        let result = heap.pull(10);
        assert_eq!(result.len(), 10);

        
        for i in 1..result.len() {
            assert!(result[i].1 >= result[i - 1].1);
        }
    }
}

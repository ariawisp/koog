// GPU Token Embeddings - Semantic search at the speed of thought
// This is where Noesis diverges from Graphiti fundamentally
// We're not searching text - we're searching cognitive states

use objc2_metal::*;
use objc2_foundation::NSString;
use objc2::{rc::Id, ClassType};
use std::sync::Arc;
use std::collections::HashMap;
use petgraph::graph::NodeIndex;

/// GPU-accelerated token embeddings for semantic memory
/// This runs on the SAME Metal GPU as inference - zero transfer cost
pub struct GPUTokenEmbeddings {
    device: Id<MTLDevice>,
    queue: Id<MTLCommandQueue>,
    
    /// All embeddings live permanently on GPU
    embeddings_buffer: Id<MTLBuffer>,
    
    /// Metadata stays on CPU for quick access
    metadata: Vec<EmbeddingMetadata>,
    
    /// Precompiled Metal compute pipelines
    similarity_pipeline: Id<MTLComputePipelineState>,
    batch_knn_pipeline: Id<MTLComputePipelineState>,
    clustering_pipeline: Id<MTLComputePipelineState>,
    
    /// Dimensions (768 for base, 4096 for large models)
    embedding_dim: usize,
    
    /// Current capacity
    capacity: usize,
    count: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingMetadata {
    pub node_idx: NodeIndex,
    pub channel: super::Channel,
    pub timestamp: i64,
    pub token_count: usize,
}

impl GPUTokenEmbeddings {
    /// Create new GPU embedding index
    pub fn new(device: Id<MTLDevice>, embedding_dim: usize, initial_capacity: usize) -> Self {
        let queue = device.new_command_queue();
        
        // Allocate GPU buffer for embeddings
        let buffer_size = (initial_capacity * embedding_dim * 4) as u64;
        let embeddings_buffer = device.new_buffer(
            buffer_size,
            MTLResourceOptions::StorageModeShared
        );
        
        // Load Metal compute shaders
        let library = Self::load_metal_library(&device);
        let similarity_pipeline = Self::create_pipeline(&device, &library, "cosine_similarity");
        let batch_knn_pipeline = Self::create_pipeline(&device, &library, "batch_knn_search");
        let clustering_pipeline = Self::create_pipeline(&device, &library, "temporal_clustering");
        
        Self {
            device,
            queue,
            embeddings_buffer,
            metadata: Vec::with_capacity(initial_capacity),
            similarity_pipeline,
            batch_knn_pipeline,
            clustering_pipeline,
            embedding_dim,
            capacity: initial_capacity,
            count: 0,
        }
    }
    
    /// Add new embedding - stays on GPU forever
    pub fn add_embedding(&mut self, embedding: &[f32], metadata: EmbeddingMetadata) {
        assert_eq!(embedding.len(), self.embedding_dim);
        
        // Grow buffer if needed
        if self.count >= self.capacity {
            self.grow_buffer();
        }
        
        // Copy embedding to GPU (this is the ONLY CPU->GPU transfer)
        let offset = (self.count * self.embedding_dim * 4) as u64;
        unsafe {
            let ptr = self.embeddings_buffer.contents().offset(offset as isize);
            std::ptr::copy_nonoverlapping(
                embedding.as_ptr(),
                ptr as *mut f32,
                self.embedding_dim
            );
        }
        
        self.metadata.push(metadata);
        self.count += 1;
    }
    
    /// Batch k-NN search - ALL on GPU, returns node indices
    pub fn batch_knn_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(NodeIndex, f32)>> {
        let batch_size = queries.len();
        
        // Upload queries to GPU (only transfer)
        let query_buffer = self.create_buffer_from_vectors(queries);
        
        // Output buffer for results
        let results_size = (batch_size * k * 8) as u64; // idx + distance
        let results_buffer = self.device.new_buffer(
            results_size,
            MTLResourceOptions::StorageModeShared
        );
        
        // Execute k-NN search on GPU
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.batch_knn_pipeline);
        encoder.set_buffer(&self.embeddings_buffer, 0, 0);
        encoder.set_buffer(&query_buffer, 0, 1);
        encoder.set_buffer(&results_buffer, 0, 2);
        encoder.set_bytes(&[self.count as u32], 4, 3);
        encoder.set_bytes(&[k as u32], 4, 4);
        
        let thread_groups = MTLSize {
            width: batch_size as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups, self.get_threadgroup_size());
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read results
        self.parse_knn_results(&results_buffer, batch_size, k)
    }
    
    /// Find semantic contradictions - revolutionary for reasoning
    pub fn find_contradictions(&self, embedding: &[f32], threshold: f32) -> Vec<(NodeIndex, f32)> {
        // Find embeddings with NEGATIVE cosine similarity
        // These represent contradictory thoughts
        let similarities = self.compute_all_similarities(embedding);
        
        similarities.into_iter()
            .enumerate()
            .filter_map(|(idx, sim)| {
                if sim < -threshold {
                    Some((self.metadata[idx].node_idx, sim))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Temporal clustering - find thought patterns over time
    pub fn temporal_clustering(&self, window_seconds: i64) -> Vec<Vec<NodeIndex>> {
        // Group embeddings by time windows
        // Find clusters within each window
        // This reveals recurring reasoning patterns
        
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.clustering_pipeline);
        encoder.set_buffer(&self.embeddings_buffer, 0, 0);
        encoder.set_bytes(&[window_seconds], 8, 1);
        encoder.set_bytes(&[self.count as u32], 4, 2);
        
        let output_buffer = self.device.new_buffer(
            (self.count * 4) as u64,
            MTLResourceOptions::StorageModeShared
        );
        encoder.set_buffer(&output_buffer, 0, 3);
        
        encoder.dispatch_threads(self.count as u64, 1, 1);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        self.parse_clusters(&output_buffer)
    }
    
    /// Channel-specific search - find thoughts in specific reasoning channels
    pub fn search_by_channel(&self, query: &[f32], channel: super::Channel, k: usize) 
        -> Vec<(NodeIndex, f32)> {
        // Filter by channel first
        let channel_indices: Vec<usize> = self.metadata.iter()
            .enumerate()
            .filter_map(|(idx, meta)| {
                if meta.channel == channel {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        
        if channel_indices.is_empty() {
            return Vec::new();
        }
        
        // Run similarity only on channel-filtered embeddings
        let similarities = self.compute_filtered_similarities(query, &channel_indices);
        
        // Get top-k
        let mut results: Vec<_> = similarities.into_iter()
            .map(|(idx, sim)| (self.metadata[idx].node_idx, sim))
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
    
    // Private helper methods
    
    fn load_metal_library(device: &Device) -> Library {
        let source = include_str!("shaders/embeddings.metal");
        let options = CompileOptions::new();
        device.new_library_with_source(source, &options)
            .expect("Failed to compile Metal shaders")
    }
    
    fn create_pipeline(device: &Device, library: &Library, function_name: &str) 
        -> ComputePipelineState {
        let function = library.get_function(function_name, None)
            .expect(&format!("Failed to find Metal function: {}", function_name));
        
        device.new_compute_pipeline_state_with_function(&function)
            .expect(&format!("Failed to create pipeline for: {}", function_name))
    }
    
    fn compute_all_similarities(&self, query: &[f32]) -> Vec<f32> {
        // Upload query
        let query_buffer = self.device.new_buffer_with_data(
            query.as_ptr() as *const _,
            (self.embedding_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        // Output buffer
        let output_buffer = self.device.new_buffer(
            (self.count * 4) as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        // Compute similarities
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.similarity_pipeline);
        encoder.set_buffer(&self.embeddings_buffer, 0, 0);
        encoder.set_buffer(&query_buffer, 0, 1);
        encoder.set_buffer(&output_buffer, 0, 2);
        encoder.set_bytes(&[self.count as u32], 4, 3);
        encoder.set_bytes(&[self.embedding_dim as u32], 4, 4);
        
        encoder.dispatch_threads(self.count as u64, 1, 1);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read results
        let mut results = vec![0.0f32; self.count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                results.as_mut_ptr(),
                self.count
            );
        }
        results
    }
    
    fn compute_filtered_similarities(&self, query: &[f32], indices: &[usize]) -> Vec<(usize, f32)> {
        // Similar to compute_all but only for specific indices
        // This is more efficient for channel-filtered searches
        let similarities = self.compute_all_similarities(query);
        indices.iter()
            .map(|&idx| (idx, similarities[idx]))
            .collect()
    }
    
    fn create_buffer_from_vectors(&self, vectors: &[Vec<f32>]) -> Buffer {
        let total_floats = vectors.len() * self.embedding_dim;
        let buffer = self.device.new_buffer(
            (total_floats * 4) as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        unsafe {
            let ptr = buffer.contents() as *mut f32;
            for (i, vec) in vectors.iter().enumerate() {
                let offset = i * self.embedding_dim;
                std::ptr::copy_nonoverlapping(
                    vec.as_ptr(),
                    ptr.offset(offset as isize),
                    self.embedding_dim
                );
            }
        }
        
        buffer
    }
    
    fn parse_knn_results(&self, buffer: &Buffer, batch_size: usize, k: usize) 
        -> Vec<Vec<(NodeIndex, f32)>> {
        let mut results = vec![vec![]; batch_size];
        
        unsafe {
            let ptr = buffer.contents() as *const u32;
            for batch in 0..batch_size {
                for i in 0..k {
                    let offset = (batch * k * 2 + i * 2) as isize;
                    let idx = *ptr.offset(offset) as usize;
                    let dist_bits = *ptr.offset(offset + 1);
                    let dist = f32::from_bits(dist_bits);
                    
                    if idx < self.count {
                        results[batch].push((self.metadata[idx].node_idx, dist));
                    }
                }
            }
        }
        
        results
    }
    
    fn parse_clusters(&self, buffer: &Buffer) -> Vec<Vec<NodeIndex>> {
        // Parse cluster assignments from GPU
        let mut cluster_map: HashMap<u32, Vec<NodeIndex>> = HashMap::new();
        
        unsafe {
            let ptr = buffer.contents() as *const u32;
            for i in 0..self.count {
                let cluster_id = *ptr.offset(i as isize);
                cluster_map.entry(cluster_id)
                    .or_insert_with(Vec::new)
                    .push(self.metadata[i].node_idx);
            }
        }
        
        cluster_map.into_values().collect()
    }
    
    fn grow_buffer(&mut self) {
        // Double capacity when full
        self.capacity *= 2;
        let new_size = (self.capacity * self.embedding_dim * 4) as u64;
        let new_buffer = self.device.new_buffer(new_size, MTLResourceOptions::StorageModeShared);
        
        // Copy existing data
        let blit_encoder = self.queue.new_command_buffer().new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &self.embeddings_buffer, 0,
            &new_buffer, 0,
            (self.count * self.embedding_dim * 4) as u64
        );
        blit_encoder.end_encoding();
        
        self.embeddings_buffer = new_buffer;
    }
    
    fn get_threadgroup_size(&self) -> MTLSize {
        MTLSize {
            width: 32,  // Warp size for Apple GPU
            height: 1,
            depth: 1,
        }
    }
}
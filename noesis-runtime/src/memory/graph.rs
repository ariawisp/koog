// Token Graph - The core data structure for token-native memory
// This replaces Graphiti's EpisodicNode/EntityNode/CommunityNode with pure token operations

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::Channel;
use anyhow::Result;

/// A chunk of tokens with temporal and channel metadata
/// This is our atomic unit of memory - no text, just tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenChunk {
    pub tokens: Vec<u8>,
    pub channel: Channel,
    pub timestamp: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>, // Optional token-level embeddings
    pub metadata: HashMap<String, String>, // Additional metadata
}

/// The temporal graph structure
/// Unlike Graphiti's multiple node types, we have one unified token-based structure
pub struct TokenGraph {
    /// The underlying directed graph
    graph: DiGraph<TokenChunk, EdgeType>,
    
    /// Index for fast node lookup by timestamp
    time_index: HashMap<i64, Vec<NodeIndex>>,
    
    /// Index for fast lookup by channel
    channel_index: HashMap<Channel, Vec<NodeIndex>>,
    
    /// Statistics for monitoring
    stats: GraphStats,
}

/// Edge types in our temporal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    /// Temporal edge - connects chunks in time sequence
    Temporal { weight: f32 },
    
    /// Semantic edge - connects related chunks by meaning
    Semantic { similarity: f32 },
    
    /// Causal edge - represents cause-effect relationships
    Causal { confidence: f32 },
    
    /// Fork edge - represents branching in parallel exploration
    Fork { branch_id: String },
}

// Channel is imported from crate root

/// Query types for memory retrieval
#[derive(Debug, Clone)]
pub enum MemoryQuery {
    /// Query by channel
    ByChannel(Channel),
    
    /// Query by time range
    ByTimeRange(DateTime<Utc>, DateTime<Utc>),
    
    /// Query from a specific checkpoint
    ByCheckpoint(String),
    
    /// Semantic query with embedding
    Semantic(Vec<f32>),
}

/// Statistics for monitoring graph performance
#[derive(Debug, Default, Serialize, Deserialize)]
struct GraphStats {
    node_count: usize,
    edge_count: usize,
    total_tokens: usize,
    channels: HashMap<Channel, usize>,
}

/// Snapshot of TokenGraph state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenGraphSnapshot {
    pub node_count: usize,
    pub edge_count: usize,
    pub total_tokens: usize,
    pub channels: HashMap<Channel, usize>,
    pub timestamp: DateTime<Utc>,
}

impl TokenGraph {
    /// Create a new empty token graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            time_index: HashMap::new(),
            channel_index: HashMap::new(),
            stats: GraphStats::default(),
        }
    }
    
    /// Capture a snapshot of the current state for checkpointing
    pub fn capture_state_snapshot(&self) -> Result<TokenGraphSnapshot> {
        Ok(TokenGraphSnapshot {
            node_count: self.stats.node_count,
            edge_count: self.stats.edge_count,
            total_tokens: self.stats.total_tokens,
            channels: self.stats.channels.clone(),
            timestamp: Utc::now(),
        })
    }
    
    /// Add a new token chunk to the graph
    pub fn add_chunk(&mut self, mut chunk: TokenChunk) -> NodeIndex {
        // Ensure metadata is initialized
        if chunk.metadata.is_empty() {
            chunk.metadata = HashMap::new();
        }
        
        // Update statistics
        self.stats.node_count += 1;
        self.stats.total_tokens += chunk.tokens.len();
        *self.stats.channels.entry(chunk.channel).or_insert(0) += 1;
        
        // Add to time index
        let timestamp_key = chunk.timestamp.timestamp();
        self.time_index
            .entry(timestamp_key)
            .or_insert_with(Vec::new)
            .push(NodeIndex::new(self.stats.node_count - 1));
        
        // Add to channel index
        self.channel_index
            .entry(chunk.channel)
            .or_insert_with(Vec::new)
            .push(NodeIndex::new(self.stats.node_count - 1));
        
        // Add node to graph
        self.graph.add_node(chunk)
    }
    
    /// Connect two chunks with a temporal edge
    pub fn connect_temporal(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, EdgeType::Temporal { weight: 1.0 });
        self.stats.edge_count += 1;
    }
    
    /// Connect two chunks with a semantic edge
    pub fn connect_semantic(&mut self, from: NodeIndex, to: NodeIndex, similarity: f32) {
        self.graph.add_edge(from, to, EdgeType::Semantic { similarity });
        self.stats.edge_count += 1;
    }
    
    /// Connect two chunks with a causal edge
    pub fn connect_causal(&mut self, cause: NodeIndex, effect: NodeIndex, confidence: f32) {
        self.graph.add_edge(cause, effect, EdgeType::Causal { confidence });
        self.stats.edge_count += 1;
    }
    
    /// Create a fork edge for parallel exploration
    pub fn create_fork(&mut self, from: NodeIndex, to: NodeIndex, branch_id: String) {
        self.graph.add_edge(from, to, EdgeType::Fork { branch_id });
        self.stats.edge_count += 1;
    }
    
    /// Get a chunk by node index
    pub fn get_chunk(&self, idx: NodeIndex) -> Option<&TokenChunk> {
        self.graph.node_weight(idx)
    }
    
    /// Find all chunks within a time range
    pub fn chunks_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<NodeIndex> {
        let start_ts = start.timestamp();
        let end_ts = end.timestamp();
        
        let mut results = Vec::new();
        for (ts, nodes) in &self.time_index {
            if *ts >= start_ts && *ts <= end_ts {
                results.extend(nodes);
            }
        }
        
        results
    }
    
    /// Find all chunks in a specific channel
    pub fn chunks_by_channel(&self, channel: Channel) -> Vec<NodeIndex> {
        self.channel_index
            .get(&channel)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Traverse the graph from a starting node
    pub fn traverse_from(&self, start: NodeIndex, max_depth: usize) -> Vec<NodeIndex> {
        let mut visited = Vec::new();
        let mut queue = vec![(start, 0)];
        let mut seen = HashMap::new();
        
        while let Some((node, depth)) = queue.pop() {
            if depth > max_depth {
                continue;
            }
            
            if seen.contains_key(&node) {
                continue;
            }
            
            seen.insert(node, depth);
            visited.push(node);
            
            // Add neighbors to queue
            for neighbor in self.graph.neighbors(node) {
                if !seen.contains_key(&neighbor) {
                    queue.push((neighbor, depth + 1));
                }
            }
        }
        
        visited
    }
    
    /// Find the shortest path between two nodes
    pub fn shortest_path(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<NodeIndex>> {
        petgraph::algo::astar(
            &self.graph,
            from,
            |n| n == to,
            |_| 1,  // Edge weight (uniform for now)
            |_| 0,  // Heuristic (none for now)
        ).map(|(_, path)| path)
    }
    
    /// Compute PageRank-style importance scores
    /// This helps identify the most important memory chunks
    pub fn compute_importance(&self) -> HashMap<NodeIndex, f32> {
        let mut scores = HashMap::new();
        let damping = 0.85;
        let iterations = 20;
        
        // Initialize scores
        let n = self.graph.node_count() as f32;
        for node in self.graph.node_indices() {
            scores.insert(node, 1.0 / n);
        }
        
        // Power iteration
        for _ in 0..iterations {
            let mut new_scores = HashMap::new();
            
            for node in self.graph.node_indices() {
                let mut score = (1.0 - damping) / n;
                
                // Sum contributions from incoming edges
                for neighbor in self.graph.neighbors_directed(node, Direction::Incoming) {
                    let neighbor_score = scores.get(&neighbor).copied().unwrap_or(0.0);
                    let out_degree = self.graph.neighbors(neighbor).count() as f32;
                    if out_degree > 0.0 {
                        score += damping * neighbor_score / out_degree;
                    }
                }
                
                new_scores.insert(node, score);
            }
            
            scores = new_scores;
        }
        
        scores
    }
    
    /// Add a node to the graph with specific data
    pub fn add_node(&mut self, buffer: Vec<u8>, channel: crate::Channel, timestamp: i64) -> Result<NodeIndex> {
        let chunk = TokenChunk {
            tokens: buffer,
            channel,
            timestamp: DateTime::from_timestamp_nanos(timestamp),
            embedding: None,
            metadata: HashMap::new(),
        };
        Ok(self.add_chunk(chunk))
    }
    
    /// Apply resolution to conflicting nodes
    pub fn apply_resolution(&mut self, node_a: NodeIndex, node_b: NodeIndex, resolution: crate::gpu_optimized::OptimizedBuffer) -> Result<()> {
        // Mark the resolution in the graph
        // Convert buffer to bytes - need proper implementation
        let resolution_bytes = vec![]; // TODO: Add proper buffer reading
        let resolution_chunk = TokenChunk {
            tokens: resolution_bytes,
            channel: crate::Channel::Commentary,
            timestamp: Utc::now(),
            embedding: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "resolution".to_string());
                meta.insert("node_a".to_string(), format!("{:?}", node_a));
                meta.insert("node_b".to_string(), format!("{:?}", node_b));
                meta
            },
        };
        
        let resolution_node = self.add_chunk(resolution_chunk);
        
        // Connect resolution to both conflicting nodes
        self.connect_causal(node_a, resolution_node, 0.9);
        self.connect_causal(node_b, resolution_node, 0.9);
        
        Ok(())
    }
    
    /// Get tokens from a node
    pub fn get_node_tokens(&self, idx: NodeIndex) -> Option<Vec<u32>> {
        self.get_chunk(idx).map(|chunk| {
            // Convert bytes back to u32 tokens
            chunk.tokens
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect()
        })
    }
    
    /// Search k-nearest neighbors
    pub fn search_knn(&self, embedding: &[f32], k: usize) -> Result<Vec<(NodeIndex, f32)>> {
        // Simplified k-NN search - in production would use FAISS or similar
        let mut results = Vec::new();
        
        for node_idx in self.graph.node_indices() {
            // Compute similarity (simplified - would use actual embeddings)
            let score = 0.5; // Placeholder
            results.push((node_idx, score));
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Compute embedding for the entire graph
    pub fn compute_embedding(&self) -> Result<Vec<f32>> {
        // Simplified - would aggregate node embeddings
        Ok(vec![0.0; 4096])
    }
    
    /// Serialize the graph
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // Simplified serialization - just serialize the stats
        // In production, would serialize the entire graph structure
        bincode::serialize(&self.stats).map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))
    }
    
    /// Deserialize a graph
    pub fn deserialize(_data: &[u8]) -> Result<Self> {
        // Simplified deserialization - create a new empty graph
        // In production, would deserialize the entire graph structure
        Ok(TokenGraph::new())
    }
    
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.stats.node_count
    }
    
    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.stats.edge_count
    }
    
    /// Merge two graphs (for parallel branch merging)
    pub fn merge(&mut self, other: &TokenGraph) -> Result<()> {
        // This is simplified - in production we'd need conflict resolution
        let mut node_mapping: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        
        // Add all nodes from other graph
        for node in other.graph.node_indices() {
            if let Some(chunk) = other.graph.node_weight(node) {
                let new_idx = self.add_chunk(chunk.clone());
                node_mapping.insert(node, new_idx);
            }
        }
        
        // Add edges with proper mapping
        for edge in other.graph.edge_references() {
            if let (Some(&new_source), Some(&new_target)) = 
                (node_mapping.get(&edge.source()), node_mapping.get(&edge.target())) {
                self.graph.add_edge(new_source, new_target, edge.weight().clone());
                self.stats.edge_count += 1;
            }
        }
        
        Ok(())
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> String {
        format!(
            "Nodes: {}, Edges: {}, Tokens: {}, Channels: {:?}",
            self.stats.node_count,
            self.stats.edge_count,
            self.stats.total_tokens,
            self.stats.channels
        )
    }
    
    /// Prune old chunks to manage memory
    pub fn prune_before(&mut self, cutoff: DateTime<Utc>) {
        let cutoff_ts = cutoff.timestamp();
        let mut nodes_to_remove = Vec::new();
        
        for (ts, nodes) in &self.time_index {
            if *ts < cutoff_ts {
                nodes_to_remove.extend(nodes);
            }
        }
        
        for node in nodes_to_remove {
            self.graph.remove_node(node);
            self.stats.node_count -= 1;
        }
        
        // Clean up indexes
        self.time_index.retain(|ts, _| *ts >= cutoff_ts);
    }
}
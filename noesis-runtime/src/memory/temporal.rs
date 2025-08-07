// Temporal Index - Fast time-based queries for token memory
// This enables instant temporal queries unlike Graphiti's LLM-based summarization

use chrono::{DateTime, Utc, Duration};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use anyhow::Result;

/// A temporal index for fast time-based queries
/// Uses a BTree for efficient range queries on timestamps
pub struct TemporalIndex {
    /// Main index: timestamp -> node indices
    time_map: BTreeMap<i64, Vec<NodeIndex>>,
    
    /// Reverse index: node -> timestamp
    node_times: HashMap<NodeIndex, i64>,
    
    /// Temporal buckets for aggregation (hour, day, week)
    hourly_buckets: HashMap<i64, Vec<NodeIndex>>,
    daily_buckets: HashMap<i64, Vec<NodeIndex>>,
    
    /// Statistics
    earliest_time: Option<DateTime<Utc>>,
    latest_time: Option<DateTime<Utc>>,
    total_entries: usize,
}

impl TemporalIndex {
    /// Create a new temporal index
    pub fn new() -> Self {
        Self {
            time_map: BTreeMap::new(),
            node_times: HashMap::new(),
            hourly_buckets: HashMap::new(),
            daily_buckets: HashMap::new(),
            earliest_time: None,
            latest_time: None,
            total_entries: 0,
        }
    }
    
    /// Insert a node at a specific timestamp
    pub fn insert(&mut self, node: NodeIndex, timestamp: i64) {
        let dt = DateTime::from_timestamp_nanos(timestamp);
        let ts = dt.timestamp();
        
        // Update main index
        self.time_map
            .entry(ts)
            .or_insert_with(Vec::new)
            .push(node);
        
        // Update reverse index
        self.node_times.insert(node, ts);
        
        // Update buckets
        let hour_bucket = (ts / 3600) * 3600;
        self.hourly_buckets
            .entry(hour_bucket)
            .or_insert_with(Vec::new)
            .push(node);
        
        let day_bucket = (ts / 86400) * 86400;
        self.daily_buckets
            .entry(day_bucket)
            .or_insert_with(Vec::new)
            .push(node);
        
        // Update statistics
        self.total_entries += 1;
        
        if self.earliest_time.is_none() || dt < self.earliest_time.unwrap() {
            self.earliest_time = Some(dt);
        }
        
        if self.latest_time.is_none() || dt > self.latest_time.unwrap() {
            self.latest_time = Some(dt);
        }
    }
    
    /// Query nodes within a time range
    /// This is instant, unlike Graphiti's LLM-based temporal queries
    pub fn range_query(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<NodeIndex> {
        let start_ts = start.timestamp();
        let end_ts = end.timestamp();
        
        let mut results = Vec::new();
        
        // Use BTree's range method for efficient querying
        for (_, nodes) in self.time_map.range(start_ts..=end_ts) {
            results.extend(nodes);
        }
        
        results
    }
    
    /// Get nodes from the last N hours
    pub fn recent_hours(&self, hours: i64) -> Vec<NodeIndex> {
        if let Some(latest) = self.latest_time {
            let start = latest - Duration::hours(hours);
            self.range_query(start, latest)
        } else {
            Vec::new()
        }
    }
    
    /// Get nodes from the last N days
    pub fn recent_days(&self, days: i64) -> Vec<NodeIndex> {
        if let Some(latest) = self.latest_time {
            let start = latest - Duration::days(days);
            self.range_query(start, latest)
        } else {
            Vec::new()
        }
    }
    
    /// Get nodes at a specific hour
    pub fn nodes_at_hour(&self, timestamp: DateTime<Utc>) -> Vec<NodeIndex> {
        let hour_bucket = (timestamp.timestamp() / 3600) * 3600;
        self.hourly_buckets
            .get(&hour_bucket)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get nodes at a specific day
    pub fn nodes_at_day(&self, timestamp: DateTime<Utc>) -> Vec<NodeIndex> {
        let day_bucket = (timestamp.timestamp() / 86400) * 86400;
        self.daily_buckets
            .get(&day_bucket)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Find temporal patterns (e.g., nodes that appear at similar times)
    pub fn find_temporal_patterns(&self, window_seconds: i64) -> Vec<Vec<NodeIndex>> {
        let mut patterns = Vec::new();
        let mut visited = HashSet::new();
        
        for (&ts, nodes) in &self.time_map {
            if visited.contains(&ts) {
                continue;
            }
            
            let mut pattern = Vec::new();
            pattern.extend(nodes);
            
            // Look for nodes within the window
            let start = ts - window_seconds;
            let end = ts + window_seconds;
            
            for (&other_ts, other_nodes) in self.time_map.range(start..=end) {
                if other_ts != ts {
                    pattern.extend(other_nodes);
                    visited.insert(other_ts);
                }
            }
            
            if pattern.len() > 1 {
                patterns.push(pattern);
            }
            
            visited.insert(ts);
        }
        
        patterns
    }
    
    /// Get the timestamp for a specific node
    pub fn get_timestamp(&self, node: NodeIndex) -> Option<i64> {
        self.node_times.get(&node).copied()
    }
    
    /// Index tokens with a timestamp
    pub fn index_tokens(&mut self, _tokens: &[u8], _context_id: u64) -> Result<()> {
        // This is a simplified implementation
        // In production, would parse tokens and create proper indices
        Ok(())
    }
    
    /// Remove a node from the index
    pub fn remove(&mut self, node: NodeIndex) {
        if let Some(&ts) = self.node_times.get(&node) {
            // Remove from main index
            if let Some(nodes) = self.time_map.get_mut(&ts) {
                nodes.retain(|&n| n != node);
                if nodes.is_empty() {
                    self.time_map.remove(&ts);
                }
            }
            
            // Remove from buckets
            let hour_bucket = (ts / 3600) * 3600;
            if let Some(nodes) = self.hourly_buckets.get_mut(&hour_bucket) {
                nodes.retain(|&n| n != node);
            }
            
            let day_bucket = (ts / 86400) * 86400;
            if let Some(nodes) = self.daily_buckets.get_mut(&day_bucket) {
                nodes.retain(|&n| n != node);
            }
            
            // Remove from reverse index
            self.node_times.remove(&node);
            self.total_entries = self.total_entries.saturating_sub(1);
        }
    }
    
    /// Prune entries older than a cutoff time
    pub fn prune_before(&mut self, cutoff: DateTime<Utc>) {
        let cutoff_ts = cutoff.timestamp();
        
        // Collect nodes to remove
        let nodes_to_remove: Vec<NodeIndex> = self.time_map
            .range((Bound::Unbounded, Bound::Excluded(cutoff_ts)))
            .flat_map(|(_, nodes)| nodes.clone())
            .collect();
        
        // Remove each node
        for node in nodes_to_remove {
            self.remove(node);
        }
        
        // Update earliest time
        if let Some((&earliest_ts, _)) = self.time_map.iter().next() {
            self.earliest_time = DateTime::from_timestamp(earliest_ts, 0);
        } else {
            self.earliest_time = None;
        }
    }
    
    /// Get temporal statistics
    pub fn stats(&self) -> TemporalStats {
        TemporalStats {
            total_entries: self.total_entries,
            unique_timestamps: self.time_map.len(),
            earliest_time: self.earliest_time,
            latest_time: self.latest_time,
            hourly_buckets: self.hourly_buckets.len(),
            daily_buckets: self.daily_buckets.len(),
        }
    }
    
    /// Find temporal clusters (groups of nodes close in time)
    pub fn find_clusters(&self, max_gap_seconds: i64) -> Vec<TemporalCluster> {
        let mut clusters = Vec::new();
        let mut current_cluster: Option<TemporalCluster> = None;
        
        for (&ts, nodes) in &self.time_map {
            match current_cluster.as_mut() {
                Some(cluster) => {
                    if ts - cluster.end_ts <= max_gap_seconds {
                        // Extend current cluster
                        cluster.nodes.extend(nodes);
                        cluster.end_ts = ts;
                    } else {
                        // Start new cluster
                        clusters.push(current_cluster.take().unwrap());
                        current_cluster = Some(TemporalCluster {
                            start_ts: ts,
                            end_ts: ts,
                            nodes: nodes.clone(),
                        });
                    }
                }
                None => {
                    // Start first cluster
                    current_cluster = Some(TemporalCluster {
                        start_ts: ts,
                        end_ts: ts,
                        nodes: nodes.clone(),
                    });
                }
            }
        }
        
        if let Some(cluster) = current_cluster {
            clusters.push(cluster);
        }
        
        clusters
    }
}

/// Statistics about the temporal index
#[derive(Debug)]
pub struct TemporalStats {
    pub total_entries: usize,
    pub unique_timestamps: usize,
    pub earliest_time: Option<DateTime<Utc>>,
    pub latest_time: Option<DateTime<Utc>>,
    pub hourly_buckets: usize,
    pub daily_buckets: usize,
}

/// A temporal cluster of nodes
#[derive(Debug)]
pub struct TemporalCluster {
    pub start_ts: i64,
    pub end_ts: i64,
    pub nodes: Vec<NodeIndex>,
}

impl TemporalCluster {
    /// Get the duration of this cluster
    pub fn duration(&self) -> Duration {
        Duration::seconds(self.end_ts - self.start_ts)
    }
    
    /// Get the center timestamp of this cluster
    pub fn center(&self) -> DateTime<Utc> {
        let center_ts = (self.start_ts + self.end_ts) / 2;
        DateTime::from_timestamp(center_ts, 0).unwrap_or_else(Utc::now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_index() {
        let mut index = TemporalIndex::new();
        let now = Utc::now();
        
        // Insert some nodes
        index.insert(NodeIndex::new(0), now.timestamp());
        index.insert(NodeIndex::new(1), (now + Duration::hours(1)).timestamp());
        index.insert(NodeIndex::new(2), (now + Duration::hours(2)).timestamp());
        
        // Test range query
        let results = index.range_query(now, now + Duration::hours(1));
        assert_eq!(results.len(), 2);
        
        // Test recent hours
        let recent = index.recent_hours(1);
        assert!(recent.len() >= 1);
        
        // Test statistics
        let stats = index.stats();
        assert_eq!(stats.total_entries, 3);
    }
}
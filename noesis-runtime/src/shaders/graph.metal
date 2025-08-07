// Metal Shaders for Graph Operations
// Cognitive topology computed at GPU speed

#include <metal_stdlib>
using namespace metal;

// PageRank iteration - parallel importance computation
kernel void pagerank_iteration(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices  
    device const float* edge_weights [[buffer(2)]],    // Edge weights
    device float* scores [[buffer(3)]],                // PageRank scores (in/out)
    constant float& damping [[buffer(4)]],             // Damping factor
    constant uint& node_count [[buffer(5)]],           // Number of nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;
    
    // Calculate contribution from incoming edges
    float incoming_rank = 0.0;
    uint row_start = row_offsets[tid];
    uint row_end = row_offsets[tid + 1];
    
    for (uint i = row_start; i < row_end; i++) {
        uint neighbor = col_indices[i];
        float weight = edge_weights[i];
        
        // Get neighbor's out-degree
        uint neighbor_out_degree = row_offsets[neighbor + 1] - row_offsets[neighbor];
        if (neighbor_out_degree > 0) {
            incoming_rank += (scores[neighbor] * weight) / float(neighbor_out_degree);
        }
    }
    
    // Update score with damping
    scores[tid] = (1.0 - damping) / float(node_count) + damping * incoming_rank;
}

// Parallel BFS for shortest paths
kernel void parallel_bfs(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices
    device atomic_uint* distances [[buffer(2)]],       // Distance matrix
    constant uint& batch_size [[buffer(3)]],           // Number of source nodes
    constant uint& node_count [[buffer(4)]],           // Number of nodes
    uint2 tid [[thread_position_in_grid]]              // (batch_idx, node_idx)
) {
    uint batch_idx = tid.x;
    uint node_idx = tid.y;
    
    if (batch_idx >= batch_size || node_idx >= node_count) return;
    
    uint base_offset = batch_idx * node_count;
    uint current_dist = atomic_load_explicit(&distances[base_offset + node_idx], 
                                            memory_order_relaxed);
    
    // If this node has been reached
    if (current_dist < UINT_MAX) {
        uint row_start = row_offsets[node_idx];
        uint row_end = row_offsets[node_idx + 1];
        
        // Update all neighbors
        for (uint i = row_start; i < row_end; i++) {
            uint neighbor = col_indices[i];
            uint neighbor_offset = base_offset + neighbor;
            
            // Try to update neighbor's distance
            uint old_dist = atomic_load_explicit(&distances[neighbor_offset], 
                                                memory_order_relaxed);
            uint new_dist = current_dist + 1;
            
            while (new_dist < old_dist) {
                if (atomic_compare_exchange_weak_explicit(
                    &distances[neighbor_offset],
                    &old_dist,
                    new_dist,
                    memory_order_relaxed,
                    memory_order_relaxed
                )) {
                    break;
                }
            }
        }
    }
}

// Louvain modularity optimization for community detection
kernel void louvain_modularity(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices
    device const float* edge_weights [[buffer(2)]],    // Edge weights
    device atomic_uint* communities [[buffer(3)]],     // Community assignments
    constant uint& node_count [[buffer(4)]],           // Number of nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;
    
    // Calculate modularity gain for each neighbor community
    uint current_community = atomic_load_explicit(&communities[tid], 
                                                 memory_order_relaxed);
    float best_gain = 0.0;
    uint best_community = current_community;
    
    uint row_start = row_offsets[tid];
    uint row_end = row_offsets[tid + 1];
    
    // Check each neighbor's community
    for (uint i = row_start; i < row_end; i++) {
        uint neighbor = col_indices[i];
        uint neighbor_community = atomic_load_explicit(&communities[neighbor], 
                                                      memory_order_relaxed);
        
        if (neighbor_community != current_community) {
            // Simplified modularity gain calculation
            float weight = edge_weights[i];
            float gain = weight;  // Simplified - production would calculate actual modularity
            
            if (gain > best_gain) {
                best_gain = gain;
                best_community = neighbor_community;
            }
        }
    }
    
    // Move to best community if gain is positive
    if (best_gain > 0.0) {
        atomic_store_explicit(&communities[tid], best_community, 
                            memory_order_relaxed);
    }
}

// Temporal flow propagation
kernel void temporal_flow(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices
    device const float* edge_weights [[buffer(2)]],    // Edge weights
    device float* flow [[buffer(3)]],                  // Flow values (in/out)
    constant uint& node_count [[buffer(4)]],           // Number of nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;
    
    // Calculate incoming flow
    float incoming_flow = 0.0;
    uint row_start = row_offsets[tid];
    uint row_end = row_offsets[tid + 1];
    
    for (uint i = row_start; i < row_end; i++) {
        uint neighbor = col_indices[i];
        float weight = edge_weights[i];
        incoming_flow += flow[neighbor] * weight * 0.1;  // Decay factor
    }
    
    // Update flow with decay
    flow[tid] = flow[tid] * 0.9 + incoming_flow;
}

// Betweenness centrality - finds bridge nodes
kernel void betweenness_centrality(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices
    device atomic_float* centrality [[buffer(2)]],     // Centrality scores
    device const uint* shortest_paths [[buffer(3)]],   // Precomputed shortest paths
    constant uint& node_count [[buffer(4)]],           // Number of nodes
    uint2 tid [[thread_position_in_grid]]              // (source, target)
) {
    uint source = tid.x;
    uint target = tid.y;
    
    if (source >= node_count || target >= node_count || source == target) return;
    
    // Find shortest path from source to target
    uint path_length = shortest_paths[source * node_count + target];
    
    if (path_length > 0 && path_length < UINT_MAX) {
        // For each intermediate node on path
        // Simplified - production would trace actual paths
        for (uint intermediate = 0; intermediate < node_count; intermediate++) {
            if (intermediate != source && intermediate != target) {
                uint dist_si = shortest_paths[source * node_count + intermediate];
                uint dist_it = shortest_paths[intermediate * node_count + target];
                
                // Check if intermediate is on shortest path
                if (dist_si + dist_it == path_length) {
                    atomic_fetch_add_explicit(&centrality[intermediate], 
                                             1.0 / float(path_length),
                                             memory_order_relaxed);
                }
            }
        }
    }
}

// Cycle detection - finds loops in reasoning
kernel void detect_cycles(
    device const uint* row_offsets [[buffer(0)]],      // CSR row offsets
    device const uint* col_indices [[buffer(1)]],      // CSR column indices
    device uint* visited [[buffer(2)]],                // Visited flags
    device uint* in_stack [[buffer(3)]],               // Stack flags for DFS
    device atomic_uint* cycle_count [[buffer(4)]],     // Number of cycles found
    constant uint& node_count [[buffer(5)]],           // Number of nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) return;
    
    // Simplified cycle detection
    // Mark as visited
    visited[tid] = 1;
    in_stack[tid] = 1;
    
    uint row_start = row_offsets[tid];
    uint row_end = row_offsets[tid + 1];
    
    for (uint i = row_start; i < row_end; i++) {
        uint neighbor = col_indices[i];
        
        if (in_stack[neighbor] == 1) {
            // Found a cycle
            atomic_fetch_add_explicit(cycle_count, 1, memory_order_relaxed);
        }
    }
    
    in_stack[tid] = 0;
}

// Graph diameter computation
kernel void graph_diameter(
    device const uint* distances [[buffer(0)]],        // All-pairs shortest paths
    device atomic_uint* diameter [[buffer(1)]],        // Maximum distance
    constant uint& node_count [[buffer(2)]],           // Number of nodes
    uint2 tid [[thread_position_in_grid]]              // (i, j) pair
) {
    uint i = tid.x;
    uint j = tid.y;
    
    if (i >= node_count || j >= node_count) return;
    
    uint dist = distances[i * node_count + j];
    
    if (dist < UINT_MAX) {
        // Update diameter if this distance is larger
        uint old_diameter = atomic_load_explicit(diameter, memory_order_relaxed);
        
        while (dist > old_diameter) {
            if (atomic_compare_exchange_weak_explicit(
                diameter,
                &old_diameter,
                dist,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
                break;
            }
        }
    }
}
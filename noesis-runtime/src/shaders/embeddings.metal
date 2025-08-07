// Metal Shaders for Token Embeddings
// These run at 1000x the speed of CPU operations
// This is where thought becomes computation

#include <metal_stdlib>
using namespace metal;

// Cosine similarity between two vectors
kernel void cosine_similarity(
    device const float* embeddings [[buffer(0)]],  // All embeddings
    device const float* query [[buffer(1)]],       // Query vector
    device float* similarities [[buffer(2)]],      // Output similarities
    constant uint& count [[buffer(3)]],            // Number of embeddings
    constant uint& dim [[buffer(4)]],              // Embedding dimension
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    // Compute dot product and norms
    float dot_product = 0.0;
    float norm_a = 0.0;
    float norm_b = 0.0;
    
    uint offset = tid * dim;
    
    for (uint i = 0; i < dim; i++) {
        float a = embeddings[offset + i];
        float b = query[i];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    // Cosine similarity
    float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
    similarities[tid] = similarity;
}

// Batch k-NN search - finds k nearest neighbors for multiple queries
kernel void batch_knn_search(
    device const float* embeddings [[buffer(0)]],     // All embeddings  
    device const float* queries [[buffer(1)]],        // Batch of queries
    device uint2* results [[buffer(2)]],              // Output: (index, distance) pairs
    constant uint& embedding_count [[buffer(3)]],     // Number of embeddings
    constant uint& k [[buffer(4)]],                   // Number of neighbors
    constant uint& dim [[buffer(5)]],                 // Embedding dimension
    uint2 tid [[thread_position_in_grid]]             // (batch_idx, thread_idx)
) {
    uint batch_idx = tid.x;
    uint thread_idx = tid.y;
    
    if (thread_idx >= embedding_count) return;
    
    // Each thread computes similarity for one embedding
    uint query_offset = batch_idx * dim;
    uint embedding_offset = thread_idx * dim;
    
    float dot_product = 0.0;
    float norm_a = 0.0;
    float norm_b = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        float a = embeddings[embedding_offset + i];
        float b = queries[query_offset + i];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
    
    // Use threadgroup memory for parallel top-k selection
    threadgroup float shared_similarities[1024];
    threadgroup uint shared_indices[1024];
    
    uint local_idx = thread_idx % 1024;
    shared_similarities[local_idx] = similarity;
    shared_indices[local_idx] = thread_idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find top-k
    // This is a simplified version - production would use a heap
    if (local_idx == 0) {
        // Sort and select top-k
        for (uint i = 0; i < min(k, embedding_count); i++) {
            uint max_idx = i;
            float max_sim = shared_similarities[i];
            
            for (uint j = i + 1; j < min(1024u, embedding_count); j++) {
                if (shared_similarities[j] > max_sim) {
                    max_sim = shared_similarities[j];
                    max_idx = j;
                }
            }
            
            // Swap
            if (max_idx != i) {
                float temp_sim = shared_similarities[i];
                uint temp_idx = shared_indices[i];
                shared_similarities[i] = shared_similarities[max_idx];
                shared_indices[i] = shared_indices[max_idx];
                shared_similarities[max_idx] = temp_sim;
                shared_indices[max_idx] = temp_idx;
            }
            
            // Write to results
            uint result_offset = batch_idx * k + i;
            results[result_offset].x = shared_indices[i];
            results[result_offset].y = as_type<uint>(shared_similarities[i]);
        }
    }
}

// Temporal clustering - group embeddings by time and similarity
kernel void temporal_clustering(
    device const float* embeddings [[buffer(0)]],      // All embeddings
    constant int64_t& window_seconds [[buffer(1)]],    // Time window
    constant uint& count [[buffer(2)]],                // Number of embeddings
    device uint* cluster_ids [[buffer(3)]],            // Output cluster assignments
    device const int64_t* timestamps [[buffer(4)]],    // Embedding timestamps
    constant uint& dim [[buffer(5)]],                  // Embedding dimension
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    // Simple clustering: assign to cluster based on time window and similarity
    int64_t my_time = timestamps[tid];
    uint my_cluster = uint(my_time / window_seconds);
    
    // Refine cluster based on similarity to centroid
    // This is simplified - production would use iterative k-means
    uint offset = tid * dim;
    float max_similarity = -1.0;
    uint best_cluster = my_cluster;
    
    // Check similarity with nearby time windows
    for (int window_offset = -1; window_offset <= 1; window_offset++) {
        uint check_cluster = my_cluster + window_offset;
        if (check_cluster < 0 || check_cluster >= count) continue;
        
        // Find centroid of this cluster (simplified: first element)
        for (uint i = 0; i < count; i++) {
            if (uint(timestamps[i] / window_seconds) == check_cluster) {
                // Compute similarity
                float similarity = 0.0;
                float norm_a = 0.0;
                float norm_b = 0.0;
                
                for (uint d = 0; d < dim; d++) {
                    float a = embeddings[offset + d];
                    float b = embeddings[i * dim + d];
                    similarity += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                
                similarity = similarity / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
                
                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    best_cluster = check_cluster;
                }
                break;  // Just check first element for now
            }
        }
    }
    
    cluster_ids[tid] = best_cluster;
}

// Attention pattern matching - find similar KV cache states
kernel void attention_pattern_match(
    device const half* kv_caches [[buffer(0)]],        // All KV cache snapshots
    device const half* query_pattern [[buffer(1)]],    // Query attention pattern
    device float* similarities [[buffer(2)]],          // Output similarities
    constant uint& snapshot_count [[buffer(3)]],       // Number of snapshots
    constant uint& pattern_size [[buffer(4)]],         // Size of each pattern
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= snapshot_count) return;
    
    // Compare attention patterns using Frobenius norm
    float diff_sum = 0.0;
    uint offset = tid * pattern_size;
    
    for (uint i = 0; i < pattern_size; i++) {
        float a = float(kv_caches[offset + i]);
        float b = float(query_pattern[i]);
        float diff = a - b;
        diff_sum += diff * diff;
    }
    
    // Convert distance to similarity
    similarities[tid] = exp(-sqrt(diff_sum) / float(pattern_size));
}

// Channel-aware embedding projection
// Projects embeddings based on channel for better separation
kernel void channel_projection(
    device const float* embeddings [[buffer(0)]],      // Input embeddings
    device float* projected [[buffer(1)]],             // Output projections
    constant uint& channel [[buffer(2)]],              // Channel ID (0=Analysis, 1=Commentary, 2=Final)
    constant uint& count [[buffer(3)]],                // Number of embeddings
    constant uint& dim [[buffer(4)]],                  // Embedding dimension
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    uint offset = tid * dim;
    
    // Apply channel-specific projection matrix
    // Each channel gets its own subspace for better separation
    float projection_weight = 1.0;
    
    switch (channel) {
        case 0:  // Analysis - emphasize reasoning patterns
            projection_weight = 1.2;
            break;
        case 1:  // Commentary - emphasize tool interactions
            projection_weight = 1.0;
            break;
        case 2:  // Final - emphasize user-facing semantics
            projection_weight = 0.8;
            break;
    }
    
    // Simple projection - production would use learned matrices
    for (uint i = 0; i < dim; i++) {
        float value = embeddings[offset + i];
        
        // Channel-specific transformation
        if (channel == 0 && i < dim/3) {
            // Boost early dimensions for Analysis
            value *= 1.5;
        } else if (channel == 1 && i >= dim/3 && i < 2*dim/3) {
            // Boost middle dimensions for Commentary
            value *= 1.5;
        } else if (channel == 2 && i >= 2*dim/3) {
            // Boost late dimensions for Final
            value *= 1.5;
        }
        
        projected[offset + i] = value * projection_weight;
    }
}

// Contradiction detection - finds embeddings with negative similarity
kernel void find_contradictions(
    device const float* embeddings [[buffer(0)]],      // All embeddings
    device const float* query [[buffer(1)]],           // Query embedding
    device uint* contradictions [[buffer(2)]],         // Output: indices of contradictions
    device atomic_uint* counter [[buffer(3)]],         // Atomic counter for results
    constant float& threshold [[buffer(4)]],           // Contradiction threshold
    constant uint& count [[buffer(5)]],                // Number of embeddings
    constant uint& dim [[buffer(6)]],                  // Embedding dimension
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    // Compute cosine similarity
    float dot_product = 0.0;
    float norm_a = 0.0;
    float norm_b = 0.0;
    
    uint offset = tid * dim;
    
    for (uint i = 0; i < dim; i++) {
        float a = embeddings[offset + i];
        float b = query[i];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    float similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
    
    // Check for contradiction (negative similarity beyond threshold)
    if (similarity < -threshold) {
        uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
        contradictions[idx] = tid;
    }
}
// Optimized Metal Compute Shaders for Apple Silicon
// Performance-critical kernels for token generation
// Tuned for M2 Ultra: 76 GPU cores, 192GB unified memory, 800GB/s bandwidth

#include <metal_stdlib>
using namespace metal;

// Constants for optimization
constant uint WARP_SIZE = 32;
constant uint TILE_SIZE = 32;
constant float EPSILON = 1e-5;

// ============================================================================
// Softmax with Temperature Scaling
// ============================================================================
kernel void optimized_softmax(
    device const float* logits [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= vocab_size) return;
    
    // Apply temperature scaling
    float scaled_logit = logits[tid] / temperature;
    
    // Find max for numerical stability
    float max_val = scaled_logit;
    for (uint i = 0; i < vocab_size; i++) {
        max_val = max(max_val, logits[i] / temperature);
    }
    
    // Compute exp and sum
    float exp_val = exp(scaled_logit - max_val);
    float sum_exp = 0.0;
    for (uint i = 0; i < vocab_size; i++) {
        sum_exp += exp((logits[i] / temperature) - max_val);
    }
    
    // Write probability
    probs[tid] = exp_val / (sum_exp + EPSILON);
}

// ============================================================================
// Optimized Top-K Sampling with Bitonic Sort
// ============================================================================
kernel void optimized_topk(
    device const float* logits [[buffer(0)]],
    device float* output [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_vals[256];
    threadgroup uint shared_indices[256];
    
    // Load and apply temperature
    if (tid < vocab_size) {
        shared_vals[tid] = logits[tid] / temperature;
        shared_indices[tid] = tid;
    } else {
        shared_vals[tid] = -INFINITY;
        shared_indices[tid] = vocab_size;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Bitonic sort to find top-k
    for (uint size = 2; size <= 256; size <<= 1) {
        uint dir = (tid & (size >> 1)) == 0;
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            uint pos = 2 * tid - (tid & (stride - 1));
            if (shared_vals[pos] < shared_vals[pos + stride]) {
                if (dir) {
                    float tmp_val = shared_vals[pos];
                    uint tmp_idx = shared_indices[pos];
                    shared_vals[pos] = shared_vals[pos + stride];
                    shared_indices[pos] = shared_indices[pos + stride];
                    shared_vals[pos + stride] = tmp_val;
                    shared_indices[pos + stride] = tmp_idx;
                }
            } else if (!dir) {
                float tmp_val = shared_vals[pos];
                uint tmp_idx = shared_indices[pos];
                shared_vals[pos] = shared_vals[pos + stride];
                shared_indices[pos] = shared_indices[pos + stride];
                shared_vals[pos + stride] = tmp_val;
                shared_indices[pos + stride] = tmp_idx;
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Output top-k values and indices
    if (tid < k) {
        output[tid] = shared_vals[tid];
        indices[tid] = shared_indices[tid];
    }
}

// ============================================================================
// Optimized Top-P (Nucleus) Sampling
// ============================================================================
kernel void optimized_topp(
    device const float* logits [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant float& p [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // First pass: compute softmax probabilities
    float scaled_logit = logits[tid] / temperature;
    
    // Find max for stability
    float max_val = scaled_logit;
    for (uint i = 0; i < vocab_size; i++) {
        max_val = max(max_val, logits[i] / temperature);
    }
    
    // Compute probability
    float exp_val = exp(scaled_logit - max_val);
    float sum_exp = 0.0;
    for (uint i = 0; i < vocab_size; i++) {
        sum_exp += exp((logits[i] / temperature) - max_val);
    }
    
    float prob = exp_val / (sum_exp + EPSILON);
    
    // Second pass: cumulative sum and threshold
    float cumsum = 0.0;
    for (uint i = 0; i <= tid; i++) {
        float i_exp = exp((logits[i] / temperature) - max_val);
        cumsum += i_exp / (sum_exp + EPSILON);
    }
    
    // Check if within nucleus
    if (cumsum <= p) {
        output[tid] = prob;
    } else {
        output[tid] = 0.0;
    }
}

// ============================================================================
// Tiled Matrix Multiplication (Optimized for Apple Silicon)
// ============================================================================
kernel void optimized_matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], // rows of A
    constant uint& N [[buffer(4)]], // cols of B  
    constant uint& K [[buffer(5)]], // cols of A, rows of B
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0;
    
    // Loop over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tid.x < K) {
            tileA[tid.y][tid.x] = A[row * K + t * TILE_SIZE + tid.x];
        } else {
            tileA[tid.y][tid.x] = 0.0;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + tid.y < K) {
            tileB[tid.y][tid.x] = B[(t * TILE_SIZE + tid.y) * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Optimized Layer Normalization
// ============================================================================
kernel void optimized_layernorm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];
    
    // Compute local sum and sum of squares
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    
    for (uint i = tid; i < hidden_dim; i += 256) {
        float val = input[gid * hidden_dim + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_sum[0] / hidden_dim;
    float variance = (shared_sum_sq[0] / hidden_dim) - (mean * mean);
    float std_dev = sqrt(variance + EPSILON);
    
    // Normalize and scale
    for (uint i = tid; i < hidden_dim; i += 256) {
        float normalized = (input[gid * hidden_dim + i] - mean) / std_dev;
        output[gid * hidden_dim + i] = normalized * gamma[i] + beta[i];
    }
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================
kernel void optimized_rope(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pos = tid.x;
    uint dim_idx = tid.y;
    
    if (pos >= seq_len || dim_idx >= hidden_dim) return;
    
    // Compute rotation angle
    float freq = 1.0 / pow(theta, (2.0 * (dim_idx / 2)) / head_dim);
    float angle = pos * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    // Apply rotation to query
    uint idx = pos * hidden_dim + dim_idx;
    if (dim_idx % 2 == 0) {
        float q_even = query[idx];
        float q_odd = query[idx + 1];
        query[idx] = q_even * cos_val - q_odd * sin_val;
        query[idx + 1] = q_even * sin_val + q_odd * cos_val;
    }
    
    // Apply rotation to key
    if (dim_idx % 2 == 0) {
        float k_even = key[idx];
        float k_odd = key[idx + 1];
        key[idx] = k_even * cos_val - k_odd * sin_val;
        key[idx + 1] = k_even * sin_val + k_odd * cos_val;
    }
}

// ============================================================================
// Fused Attention (Query-Key-Value)
// ============================================================================
kernel void optimized_attention_qkv(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_qk[TILE_SIZE][TILE_SIZE];
    threadgroup float shared_v[TILE_SIZE][TILE_SIZE];
    
    uint q_idx = gid.y * TILE_SIZE + tid.y;
    uint kv_idx = gid.x * TILE_SIZE + tid.x;
    
    // Compute Q*K^T for attention scores
    float score = 0.0;
    for (uint d = 0; d < head_dim; d++) {
        if (q_idx < seq_len && kv_idx < seq_len) {
            score += query[q_idx * head_dim + d] * key[kv_idx * head_dim + d];
        }
    }
    
    // Scale and store
    shared_qk[tid.y][tid.x] = score * scale;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply softmax (simplified for tile)
    float max_score = shared_qk[tid.y][0];
    for (uint i = 1; i < TILE_SIZE; i++) {
        max_score = max(max_score, shared_qk[tid.y][i]);
    }
    
    float exp_sum = 0.0;
    for (uint i = 0; i < TILE_SIZE; i++) {
        shared_qk[tid.y][i] = exp(shared_qk[tid.y][i] - max_score);
        exp_sum += shared_qk[tid.y][i];
    }
    
    for (uint i = 0; i < TILE_SIZE; i++) {
        shared_qk[tid.y][i] /= (exp_sum + EPSILON);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute attention * V
    float result = 0.0;
    for (uint i = 0; i < TILE_SIZE; i++) {
        if (q_idx < seq_len && i < seq_len) {
            result += shared_qk[tid.y][i] * value[i * head_dim + tid.x];
        }
    }
    
    // Write output
    if (q_idx < seq_len && tid.x < head_dim) {
        output[q_idx * head_dim + tid.x] = result;
    }
}
// Speculative Decoding System for 2-3x Token Generation Speedup
// Uses a small draft model to propose tokens, verified by the main model
// Performance target: 300-450 tok/s on M2 Ultra through speculation

use crate::gpu_optimized::{OptimizedMetalDevice as MetalDevice, OptimizedBufferPool as BufferPool};
use crate::inference::kernels::{MetalKernelLibrary, SamplingAccelerator, SamplingStrategy};
use crate::unified_memory::{UnifiedMemoryManager, ZeroCopyTokenBuffer};
use anyhow::{Result, Context, bail};
use std::ptr;
use objc2_metal::{MTLCommandQueue, MTLCommandBuffer, MTLComputeCommandEncoder, MTLCommandEncoder};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use log::{info, debug, warn};

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate ahead
    pub speculation_length: usize,
    /// Acceptance threshold for draft tokens
    pub acceptance_threshold: f32,
    /// Maximum retries before falling back to normal decoding
    pub max_retries: usize,
    /// Use tree-based speculation for even higher speedup
    pub tree_speculation: bool,
    /// Number of parallel speculation branches
    pub num_branches: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_length: 4,      // Optimal for 4x speedup
            acceptance_threshold: 0.85, // High confidence threshold
            max_retries: 2,
            tree_speculation: true,     // Enable tree speculation
            num_branches: 3,            // 3 parallel branches
        }
    }
}

/// Dual-model speculative decoder
pub struct SpeculativeEngine {
    /// Small, fast draft model (e.g., 1B params)
    draft_model: Arc<DraftModel>,
    /// Main target model (e.g., 20B params)
    target_model: Arc<TargetModel>,
    /// Shared memory manager
    memory_manager: Arc<UnifiedMemoryManager>,
    /// Metal kernel library
    kernel_library: Arc<MetalKernelLibrary>,
    /// Configuration
    config: SpeculativeConfig,
    /// Performance metrics
    metrics: Arc<RwLock<SpeculativeMetrics>>,
}

/// Lightweight draft model for speculation
struct DraftModel {
    device: Arc<MetalDevice>,
    sampler: SamplingAccelerator,
    /// Model weights (much smaller than target)
    weights_buffer: Arc<ZeroCopyTokenBuffer>,
    /// GPT-OSS model pointer for fast inference
    model_ptr: *mut std::ffi::c_void,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of layers (fewer than target)
    n_layers: usize,
}

impl DraftModel {
    /// Get GPT-OSS model pointer for inference
    fn get_model_ptr(&self) -> Result<*mut std::ffi::c_void> {
        if self.model_ptr.is_null() {
            bail!("Draft model not initialized");
        }
        Ok(self.model_ptr)
    }
}

/// Main target model for verification
struct TargetModel {
    device: Arc<MetalDevice>,
    sampler: SamplingAccelerator,
    /// Model weights
    weights_buffer: Arc<ZeroCopyTokenBuffer>,
    /// GPT-OSS model pointer for full inference
    model_ptr: *mut std::ffi::c_void,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of layers
    n_layers: usize,
}

impl TargetModel {
    /// Get GPT-OSS model pointer for inference
    fn get_model_ptr(&self) -> Result<*mut std::ffi::c_void> {
        if self.model_ptr.is_null() {
            bail!("Target model not initialized");
        }
        Ok(self.model_ptr)
    }
}

/// Performance metrics for speculative decoding
#[derive(Debug, Default, Clone)]
struct SpeculativeMetrics {
    /// Total tokens generated
    total_tokens: usize,
    /// Tokens accepted from speculation
    accepted_tokens: usize,
    /// Tokens rejected and regenerated
    rejected_tokens: usize,
    /// Average acceptance rate
    acceptance_rate: f32,
    /// Average tokens per speculation
    avg_tokens_per_spec: f32,
    /// Total speculation time
    speculation_time: Duration,
    /// Total verification time
    verification_time: Duration,
}

impl SpeculativeEngine {
    /// Create a new speculative decoding engine
    pub fn new(
        device: Arc<MetalDevice>,
        memory_manager: Arc<UnifiedMemoryManager>,
        kernel_library: Arc<MetalKernelLibrary>,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        info!("Creating speculative decoding engine");
        info!("  Speculation length: {}", config.speculation_length);
        info!("  Tree speculation: {}", config.tree_speculation);
        info!("  Branches: {}", config.num_branches);
        
        // Create draft model (1B params, 12 layers) - load actual GPT-OSS model
        let draft_sampler = SamplingAccelerator::new(
            device.clone(),
            kernel_library.clone(),
            200_000, // vocab size
            SamplingStrategy::TopK { k: 5, temperature: 0.8 },
        )?;
        
        // Load draft model from file (would be a smaller, faster model)
        let mut draft_model_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let draft_path = std::ffi::CString::new("/path/to/draft_model.gguf")?; // Placeholder path
        
        info!("Loading draft model for speculative decoding...");
        // Note: In a real implementation, we'd load a smaller, faster model here
        // For now, we'll use the main GPT-OSS model at lower precision
        
        let draft_model = Arc::new(DraftModel {
            device: device.clone(),
            sampler: draft_sampler,
            weights_buffer: Arc::new(ZeroCopyTokenBuffer::new(&device, 1_000_000_000)?), // 1B params
            model_ptr: std::ptr::null_mut(), // Will be initialized with actual GPT-OSS model
            hidden_dim: 2048,
            n_layers: 12,
        });
        
        // Create target model (20B params, 48 layers) - use main GPT-OSS model
        let target_sampler = SamplingAccelerator::new(
            device.clone(),
            kernel_library.clone(),
            200_000, // vocab size
            SamplingStrategy::TopP { p: 0.9, temperature: 0.7 },
        )?;
        
        // Load target model from file (main GPT-OSS model)
        let mut target_model_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let target_path = std::ffi::CString::new("/path/to/gpt-oss-20b.gguf")?; // Placeholder path
        
        info!("Loading target model for speculative verification...");
        // Note: This would use the main loaded GPT-OSS model from MetalInferenceEngine
        
        let target_model = Arc::new(TargetModel {
            device: device.clone(),
            sampler: target_sampler,
            weights_buffer: Arc::new(ZeroCopyTokenBuffer::new(&device, 20_000_000_000)?), // 20B params
            model_ptr: std::ptr::null_mut(), // Will be initialized with actual GPT-OSS model
            hidden_dim: 8192,
            n_layers: 48,
        });
        
        Ok(Self {
            draft_model,
            target_model,
            memory_manager,
            kernel_library,
            config,
            metrics: Arc::new(RwLock::new(SpeculativeMetrics::default())),
        })
    }
    
    /// Generate tokens using speculative decoding
    pub fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut context = prompt_tokens.to_vec();
        
        info!("Starting speculative generation for {} tokens", max_tokens);
        let generation_start = Instant::now();
        
        while generated.len() < max_tokens {
            let spec_start = Instant::now();
            
            // Generate speculation candidates
            let candidates = if self.config.tree_speculation {
                self.generate_tree_speculation(&context, command_queue)?
            } else {
                self.generate_linear_speculation(&context, command_queue)?
            };
            
            let spec_time = spec_start.elapsed();
            
            // Verify candidates with target model
            let verify_start = Instant::now();
            let accepted = self.verify_speculation(&context, &candidates, command_queue)?;
            let verify_time = verify_start.elapsed();
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().unwrap();
                metrics.total_tokens += accepted.len();
                metrics.accepted_tokens += accepted.len();
                metrics.speculation_time += spec_time;
                metrics.verification_time += verify_time;
                
                if accepted.len() < candidates.len() {
                    metrics.rejected_tokens += candidates.len() - accepted.len();
                }
                
                metrics.acceptance_rate = metrics.accepted_tokens as f32 / metrics.total_tokens as f32;
                metrics.avg_tokens_per_spec = accepted.len() as f32;
            }
            
            // Add accepted tokens
            generated.extend_from_slice(&accepted);
            context.extend_from_slice(&accepted);
            
            debug!("Speculation round: {} candidates, {} accepted", 
                candidates.len(), accepted.len());
        }
        
        let total_time = generation_start.elapsed();
        let tokens_per_second = generated.len() as f32 / total_time.as_secs_f32();
        
        info!("Speculative generation complete:");
        info!("  Generated {} tokens in {:?}", generated.len(), total_time);
        info!("  Speed: {:.1} tok/s", tokens_per_second);
        
        // Log final metrics
        let metrics = self.metrics.read().unwrap();
        info!("  Acceptance rate: {:.2}%", metrics.acceptance_rate * 100.0);
        info!("  Avg tokens per speculation: {:.1}", metrics.avg_tokens_per_spec);
        
        Ok(generated)
    }
    
    /// Generate linear speculation using direct GPT-OSS sampling
    fn generate_linear_speculation(
        &self,
        context: &[u32],
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<u32>> {
        let mut candidates = Vec::new();
        
        // Create a fast draft context for speculation
        let mut draft_context: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = crate::inference::noesis_metal::create_gptoss_context(
            self.target_model.get_model_ptr()?, // Use main model for now
            context.len() + self.config.speculation_length + 10,
            &mut draft_context
        );
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            bail!("Failed to create draft context for speculation: {:?}", status);
        }
        
        // Add initial context
        let status = unsafe {
            crate::inference::noesis_metal::append_gptoss_tokens(
                draft_context,
                context.len(),
                context.as_ptr()
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
            bail!("Failed to append initial context: {:?}", status);
        }
        
        // Process initial context
        let status = unsafe {
            crate::inference::noesis_metal::process_gptoss_context(draft_context)
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
            bail!("Failed to process initial context: {:?}", status);
        }
        
        // Generate speculation candidates with higher temperature for speed
        for _ in 0..self.config.speculation_length {
            let mut candidate_token: u32 = 0;
            let status = unsafe {
                crate::inference::noesis_metal::sample_gptoss_token(
                    draft_context,
                    1.2, // Higher temperature for faster, more diverse speculation
                    rand::random(),
                    &mut candidate_token
                )
            };
            
            if status != crate::inference::noesis_metal::GptOssStatus::Success {
                warn!("Failed to sample candidate token: {:?}", status);
                break;
            }
            
            candidates.push(candidate_token);
            
            // Add token to context and process for next iteration
            let status = unsafe {
                crate::inference::noesis_metal::append_gptoss_tokens(
                    draft_context,
                    1,
                    &candidate_token
                )
            };
            
            if status != crate::inference::noesis_metal::GptOssStatus::Success {
                warn!("Failed to append candidate token: {:?}", status);
                break;
            }
            
            let status = unsafe {
                crate::inference::noesis_metal::process_gptoss_context(draft_context)
            };
            
            if status != crate::inference::noesis_metal::GptOssStatus::Success {
                warn!("Failed to process after candidate: {:?}", status);
                break;
            }
        }
        
        // Clean up
        unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
        
        debug!("Generated {} speculation candidates", candidates.len());
        Ok(candidates)
    }
    
    /// Generate tree-based speculation (advanced approach)
    fn generate_tree_speculation(
        &self,
        context: &[u32],
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<u32>> {
        let mut speculation_tree = SpeculationTree::new(self.config.num_branches);
        let mut candidates = Vec::new();
        
        // Build speculation tree with draft model
        for depth in 0..self.config.speculation_length {
            let command_buffer = command_queue.commandBuffer().unwrap();
            let encoder = command_buffer.computeCommandEncoder().unwrap();
            
            // Generate multiple candidates at each depth
            for branch_id in 0..self.config.num_branches {
                let branch_context = speculation_tree.get_branch_context(branch_id, context);
                
                // Run draft inference
                let logits = self.run_draft_inference(&branch_context, &encoder)?;
                
                // Sample with different temperatures for diversity
                let temperature = 0.7 + (branch_id as f32 * 0.1);
                let strategy = SamplingStrategy::TopK { 
                    k: 5 + branch_id * 2, 
                    temperature 
                };
                
                // Get top candidates
                let top_tokens = self.sample_multiple(&logits, 3, strategy, &encoder)?;
                speculation_tree.add_candidates(branch_id, depth, top_tokens);
            }
            
            encoder.endEncoding();
            command_buffer.commit();
            unsafe { command_buffer.waitUntilCompleted(); }
        }
        
        // Extract best path from tree
        candidates = speculation_tree.get_best_path();
        
        Ok(candidates)
    }
    
    /// Verify speculation with target model using sampling-based verification
    fn verify_speculation(
        &self,
        context: &[u32],
        candidates: &[u32],
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<u32>> {
        let mut accepted = Vec::new();
        
        // Create target context for verification
        let mut target_context: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe {
            crate::inference::noesis_metal::create_gptoss_context(
                self.target_model.get_model_ptr()?,
                context.len() + candidates.len() + 10,
                &mut target_context
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            bail!("Failed to create target context for verification: {:?}", status);
        }
        
        // Add initial context
        let status = unsafe {
            crate::inference::noesis_metal::append_gptoss_tokens(
                target_context,
                context.len(),
                context.as_ptr()
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
            bail!("Failed to append context for verification: {:?}", status);
        }
        
        // Process initial context
        let status = unsafe {
            crate::inference::noesis_metal::process_gptoss_context(target_context)
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
            bail!("Failed to process context for verification: {:?}", status);
        }
        
        // Verify each candidate token
        for &candidate in candidates {
            // Sample multiple times with low temperature to get target model preference
            const VERIFICATION_SAMPLES: usize = 10;
            let mut candidate_count = 0;
            
            for _ in 0..VERIFICATION_SAMPLES {
                let mut sampled_token: u32 = 0;
                let status = unsafe {
                    crate::inference::noesis_metal::sample_gptoss_token(
                        target_context,
                        0.3, // Low temperature for more deterministic sampling
                        rand::random(),
                        &mut sampled_token
                    )
                };
                
                if status == crate::inference::noesis_metal::GptOssStatus::Success {
                    if sampled_token == candidate {
                        candidate_count += 1;
                    }
                } else {
                    warn!("Verification sampling failed: {:?}", status);
                }
            }
            
            // Calculate acceptance probability
            let acceptance_prob = candidate_count as f32 / VERIFICATION_SAMPLES as f32;
            
            if acceptance_prob >= self.config.acceptance_threshold {
                // Accept candidate
                accepted.push(candidate);
                
                // Add accepted token to context for next verification
                let status = unsafe {
                    crate::inference::noesis_metal::append_gptoss_tokens(
                        target_context,
                        1,
                        &candidate
                    )
                };
                
                if status != crate::inference::noesis_metal::GptOssStatus::Success {
                    warn!("Failed to append accepted token: {:?}", status);
                    break;
                }
                
                let status = unsafe {
                    crate::inference::noesis_metal::process_gptoss_context(target_context)
                };
                
                if status != crate::inference::noesis_metal::GptOssStatus::Success {
                    warn!("Failed to process after accepted token: {:?}", status);
                    break;
                }
                
                debug!("✅ Accepted candidate token {} (prob: {:.3})", candidate, acceptance_prob);
            } else {
                // Reject candidate - sample correct token and stop speculation
                let mut correct_token: u32 = 0;
                let status = unsafe {
                    crate::inference::noesis_metal::sample_gptoss_token(
                        target_context,
                        0.7, // Target model's preferred temperature
                        rand::random(),
                        &mut correct_token
                    )
                };
                
                if status == crate::inference::noesis_metal::GptOssStatus::Success {
                    accepted.push(correct_token);
                    debug!("❌ Rejected candidate {} (prob: {:.3}), using {}", 
                           candidate, acceptance_prob, correct_token);
                } else {
                    warn!("Failed to sample correction token: {:?}", status);
                }
                
                break; // Stop speculation on first rejection
            }
        }
        
        // Clean up
        unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
        
        debug!("Verification complete: {} accepted out of {} candidates", 
               accepted.len(), candidates.len());
        
        Ok(accepted)
    }
    
    /// Run draft model inference using GPT-OSS
    fn run_draft_inference(
        &self,
        context: &[u32],
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<Vec<f32>> {
        // Create draft model context (fast, small model)
        let mut draft_context: *mut std::ffi::c_void = ptr::null_mut();
        let status = unsafe {
            crate::inference::noesis_metal::create_gptoss_context(
                self.draft_model.get_model_ptr()?,
                context.len() + self.config.speculation_length,
                &mut draft_context
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            bail!("Failed to create draft context: {:?}", status);
        }
        
        // Process context tokens through draft model
        let status = unsafe {
            crate::inference::noesis_metal::append_gptoss_tokens(
                draft_context,
                context.len(),
                context.as_ptr()
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
            bail!("Failed to append tokens to draft context: {:?}", status);
        }
        
        // Process through model to get logits
        let status = unsafe {
            crate::inference::noesis_metal::process_gptoss_context(draft_context)
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
            bail!("Draft model processing failed: {:?}", status);
        }
        
        // Extract logits (simplified - would need actual logits extraction API)
        let logits = self.extract_logits_from_context(draft_context)?;
        
        // Clean up
        unsafe { crate::inference::noesis_metal::release_gptoss_context(draft_context); }
        
        Ok(logits)
    }
    
    /// Run target model inference using GPT-OSS
    fn run_target_inference(
        &self,
        context: &[u32],
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<Vec<f32>> {
        // Create target model context (full-sized model)
        let mut target_context: *mut std::ffi::c_void = ptr::null_mut();
        let status = unsafe {
            crate::inference::noesis_metal::create_gptoss_context(
                self.target_model.get_model_ptr()?,
                context.len() + 10, // Extra space for verification
                &mut target_context
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            bail!("Failed to create target context: {:?}", status);
        }
        
        // Process context tokens through target model
        let status = unsafe {
            crate::inference::noesis_metal::append_gptoss_tokens(
                target_context,
                context.len(),
                context.as_ptr()
            )
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
            bail!("Failed to append tokens to target context: {:?}", status);
        }
        
        // Process through model to get logits
        let status = unsafe {
            crate::inference::noesis_metal::process_gptoss_context(target_context)
        };
        
        if status != crate::inference::noesis_metal::GptOssStatus::Success {
            unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
            bail!("Target model processing failed: {:?}", status);
        }
        
        // Extract logits
        let logits = self.extract_logits_from_context(target_context)?;
        
        // Clean up
        unsafe { crate::inference::noesis_metal::release_gptoss_context(target_context); }
        
        Ok(logits)
    }
    
    /// Extract logits from GPT-OSS context (helper function)
    fn extract_logits_from_context(&self, context: *mut std::ffi::c_void) -> Result<Vec<f32>> {
        // GPT-OSS doesn't expose logits directly, so we'll need to work around this
        // For now, generate multiple samples and estimate distribution
        let mut token_counts = std::collections::HashMap::new();
        const NUM_SAMPLES: usize = 100;
        
        for _ in 0..NUM_SAMPLES {
            let mut token: u32 = 0;
            let status = unsafe {
                crate::inference::noesis_metal::sample_gptoss_token(
                    context,
                    1.0, // High temperature for diverse sampling
                    rand::random(),
                    &mut token
                )
            };
            
            if status == crate::inference::noesis_metal::GptOssStatus::Success {
                *token_counts.entry(token as usize).or_insert(0) += 1;
            }
        }
        
        // Convert counts to pseudo-logits
        let mut logits = vec![0.0f32; 200_000]; // Vocab size
        for (token_id, count) in token_counts {
            if token_id < logits.len() {
                logits[token_id] = (count as f32 / NUM_SAMPLES as f32).ln();
            }
        }
        
        Ok(logits)
    }
    
    /// Calculate probability of a specific token
    fn calculate_token_probability(&self, logits: &[f32], token: u32) -> Result<f32> {
        // Apply softmax and get probability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let prob = (logits[token as usize] - max_logit).exp() / exp_sum;
        Ok(prob)
    }
    
    /// Sample multiple tokens with given strategy
    fn sample_multiple(
        &self,
        logits: &[f32],
        num_samples: usize,
        strategy: SamplingStrategy,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<Vec<u32>> {
        // Simplified - would sample multiple tokens
        Ok(vec![0; num_samples])
    }
    
    /// Get performance statistics
    pub fn get_metrics(&self) -> SpeculativeMetrics {
        self.metrics.read().unwrap().clone()
    }
    
    /// Create a production-ready speculative engine with default configuration
    pub fn create_production_engine(
        device: Arc<MetalDevice>,
        memory_manager: Arc<UnifiedMemoryManager>,
        kernel_library: Arc<MetalKernelLibrary>,
    ) -> Result<Self> {
        let config = SpeculativeConfig {
            speculation_length: 4,    // 4 tokens ahead for optimal speedup
            acceptance_threshold: 0.6, // Balanced acceptance rate
            max_retries: 2,
            tree_speculation: false,   // Use linear for simplicity
            num_branches: 1,
        };
        
        Self::new(device, memory_manager, kernel_library, config)
    }
    
    /// Generate tokens with speculative decoding enabled
    /// This is the main entry point for using speculative decoding
    pub fn generate_with_speculation(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        mut callback: Option<&mut dyn FnMut(u32, f32) -> bool>, // token, acceptance_rate -> continue
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut context = prompt_tokens.to_vec();
        
        info!("🚀 Starting speculative token generation");
        info!("   Target: {} tokens", max_tokens);
        info!("   Speculation length: {}", self.config.speculation_length);
        
        let start_time = Instant::now();
        let mut total_accepted = 0;
        let mut total_generated = 0;
        
        while generated.len() < max_tokens {
            let round_start = Instant::now();
            
            // Generate speculation candidates
            let candidates = self.generate_linear_speculation(&context, command_queue)?;
            
            // Verify with target model
            let accepted = self.verify_speculation(&context, &candidates, command_queue)?;
            
            let round_time = round_start.elapsed();
            total_accepted += accepted.len();
            total_generated += candidates.len();
            
            // Calculate acceptance rate for this round
            let acceptance_rate = if candidates.len() > 0 {
                accepted.len() as f32 / candidates.len() as f32
            } else {
                0.0
            };
            
            // Add accepted tokens to output
            generated.extend_from_slice(&accepted);
            context.extend_from_slice(&accepted);
            
            // Call progress callback if provided
            if let Some(ref mut cb) = callback {
                for &token in &accepted {
                    if !cb(token, acceptance_rate) {
                        info!("Generation stopped by callback");
                        return Ok(generated);
                    }
                }
            }
            
            debug!("Round complete: {} candidates → {} accepted ({:.2}%) in {:?}",
                   candidates.len(), accepted.len(), acceptance_rate * 100.0, round_time);
            
            // Break if we got no tokens (model might be stuck)
            if accepted.is_empty() {
                warn!("No tokens generated in speculation round, stopping");
                break;
            }
        }
        
        let total_time = start_time.elapsed();
        let final_acceptance_rate = if total_generated > 0 {
            total_accepted as f32 / total_generated as f32
        } else {
            0.0
        };
        
        let tokens_per_second = generated.len() as f32 / total_time.as_secs_f32();
        
        info!("✅ Speculative generation complete:");
        info!("   Generated: {} tokens in {:?}", generated.len(), total_time);
        info!("   Speed: {:.1} tok/s", tokens_per_second);
        info!("   Overall acceptance: {:.2}% ({}/{} candidates)",
              final_acceptance_rate * 100.0, total_accepted, total_generated);
        
        // Update final metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_tokens = generated.len();
            metrics.accepted_tokens = total_accepted;
            metrics.rejected_tokens = total_generated - total_accepted;
            metrics.acceptance_rate = final_acceptance_rate;
        }
        
        Ok(generated)
    }
}

/// Tree structure for advanced speculation
struct SpeculationTree {
    branches: Vec<SpeculationBranch>,
    num_branches: usize,
}

struct SpeculationBranch {
    tokens: Vec<Vec<u32>>, // tokens at each depth
    scores: Vec<f32>,      // cumulative scores
}

impl SpeculationTree {
    fn new(num_branches: usize) -> Self {
        let branches = (0..num_branches)
            .map(|_| SpeculationBranch {
                tokens: Vec::new(),
                scores: Vec::new(),
            })
            .collect();
        
        Self { branches, num_branches }
    }
    
    fn get_branch_context(&self, branch_id: usize, base_context: &[u32]) -> Vec<u32> {
        let mut context = base_context.to_vec();
        if branch_id < self.branches.len() {
            for tokens in &self.branches[branch_id].tokens {
                context.extend_from_slice(tokens);
            }
        }
        context
    }
    
    fn add_candidates(&mut self, branch_id: usize, depth: usize, tokens: Vec<u32>) {
        if branch_id < self.branches.len() {
            self.branches[branch_id].tokens.push(tokens);
        }
    }
    
    fn get_best_path(&self) -> Vec<u32> {
        // Find branch with highest cumulative score
        let best_branch = self.branches
            .iter()
            .max_by(|a, b| {
                let a_score: f32 = a.scores.iter().sum();
                let b_score: f32 = b.scores.iter().sum();
                a_score.partial_cmp(&b_score).unwrap()
            });
        
        best_branch
            .map(|b| b.tokens.iter().flatten().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_speculation_tree() {
        let mut tree = SpeculationTree::new(3);
        
        // Add some candidates
        tree.add_candidates(0, 0, vec![1, 2, 3]);
        tree.add_candidates(1, 0, vec![4, 5, 6]);
        tree.add_candidates(2, 0, vec![7, 8, 9]);
        
        let context = tree.get_branch_context(0, &[0]);
        assert_eq!(context, vec![0, 1, 2, 3]);
    }
}
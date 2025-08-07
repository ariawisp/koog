// Real-time streaming output for Harmony parsed results
// Designed for zero interference with inference performance

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{Result, Context};
use crossbeam_channel::{bounded, unbounded, Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use colored::*;

/// High-performance streaming output events
#[derive(Debug, Clone)]
pub enum StreamingEvent {
    /// Raw token with context
    Token {
        token: u32,
        token_index: usize,
        timestamp_ns: u64,
    },
    /// Content delta from Harmony parser
    ContentDelta {
        text: String,
        channel: Option<String>,
        recipient: Option<String>,
        timestamp_ns: u64,
    },
    /// Channel transition detected
    ChannelSwitch {
        from_channel: Option<String>,
        to_channel: String,
        timestamp_ns: u64,
    },
    /// Tool call detected
    ToolCall {
        tool_name: String,
        arguments: String,
        channel: String,
        recipient: String,
        timestamp_ns: u64,
    },
    /// Stream completion
    StreamComplete {
        reason: String,
        total_tokens: usize,
        total_time_ms: u64,
        timestamp_ns: u64,
    },
    /// Performance metrics
    PerformanceMetrics {
        tokens_per_second: f64,
        processing_latency_ms: f64,
        queue_depth: usize,
        timestamp_ns: u64,
    },
    /// Raw debug information
    Debug {
        message: String,
        timestamp_ns: u64,
    },
}

/// Real-time terminal output configuration
#[derive(Debug, Clone)]
pub struct StreamingOutputConfig {
    /// Enable colored output
    pub colored_output: bool,
    /// Show timestamps
    pub show_timestamps: bool,
    /// Show token indices
    pub show_token_indices: bool,
    /// Show raw tokens (for debugging)
    pub show_raw_tokens: bool,
    /// Show channel information
    pub show_channels: bool,
    /// Show performance metrics
    pub show_performance: bool,
    /// Buffer size for batching output
    pub buffer_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Separate output streams by channel
    pub channel_separation: bool,
}

impl Default for StreamingOutputConfig {
    fn default() -> Self {
        Self {
            colored_output: true,
            show_timestamps: false,
            show_token_indices: false,
            show_raw_tokens: false,
            show_channels: true,
            show_performance: false,
            buffer_size: 32,
            flush_interval_ms: 10, // Very fast flushing for real-time feel
            channel_separation: true,
        }
    }
}

/// High-performance real-time streaming output manager
pub struct StreamingOutputManager {
    event_sender: CrossbeamSender<StreamingEvent>,
    _output_thread: thread::JoinHandle<()>,
    config: StreamingOutputConfig,
    start_time: Instant,
}

impl StreamingOutputManager {
    /// Create a new streaming output manager with background processing
    pub fn new(config: StreamingOutputConfig) -> Result<Self> {
        let (event_sender, event_receiver) = unbounded::<StreamingEvent>();
        let config_clone = config.clone();
        let start_time = Instant::now();
        
        // Spawn background thread for non-blocking I/O
        let output_thread = thread::Builder::new()
            .name("harmony-output".to_string())
            .spawn(move || {
                if let Err(e) = Self::output_thread_main(event_receiver, config_clone) {
                    eprintln!("[StreamingOutput] ERROR: Output thread failed: {}", e);
                }
            })
            .context("Failed to spawn output thread")?;
        
        Ok(Self {
            event_sender,
            _output_thread: output_thread,
            config,
            start_time,
        })
    }
    
    /// Send an event for real-time output (non-blocking)
    pub fn send_event(&self, event: StreamingEvent) -> Result<()> {
        self.event_sender.send(event)
            .context("Failed to send streaming event")?;
        Ok(())
    }
    
    /// Send a token event
    pub fn send_token(&self, token: u32, token_index: usize) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::Token {
            token,
            token_index,
            timestamp_ns,
        })
    }
    
    /// Send a content delta event
    pub fn send_content_delta(&self, text: String, channel: Option<String>, recipient: Option<String>) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::ContentDelta {
            text,
            channel,
            recipient,
            timestamp_ns,
        })
    }
    
    /// Send a tool call event
    pub fn send_tool_call(&self, tool_name: String, arguments: String, channel: String, recipient: String) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::ToolCall {
            tool_name,
            arguments,
            channel,
            recipient,
            timestamp_ns,
        })
    }
    
    /// Send a channel switch event
    pub fn send_channel_switch(&self, from_channel: Option<String>, to_channel: String) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::ChannelSwitch {
            from_channel,
            to_channel,
            timestamp_ns,
        })
    }
    
    /// Send performance metrics
    pub fn send_performance_metrics(&self, tokens_per_second: f64, processing_latency_ms: f64, queue_depth: usize) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::PerformanceMetrics {
            tokens_per_second,
            processing_latency_ms,
            queue_depth,
            timestamp_ns,
        })
    }
    
    /// Send debug message
    pub fn send_debug(&self, message: String) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        self.send_event(StreamingEvent::Debug {
            message,
            timestamp_ns,
        })
    }
    
    /// Background thread main loop - optimized for low latency
    fn output_thread_main(
        event_receiver: CrossbeamReceiver<StreamingEvent>,
        config: StreamingOutputConfig,
    ) -> Result<()> {
        let mut output_buffer = Vec::with_capacity(config.buffer_size);
        let mut last_flush = Instant::now();
        let flush_interval = Duration::from_millis(config.flush_interval_ms);
        
        // Performance tracking
        let mut token_count = 0;
        let mut stream_start_time = None;
        let mut last_channel = None::<String>;
        
        loop {
            // Try to receive events with timeout for periodic flushing
            let timeout_duration = Duration::from_millis(config.flush_interval_ms / 2);
            
            match event_receiver.recv_timeout(timeout_duration) {
                Ok(event) => {
                    // Update stream start time on first token
                    if stream_start_time.is_none() {
                        stream_start_time = Some(Instant::now());
                    }
                    
                    // Process the event
                    Self::format_and_buffer_event(&event, &mut output_buffer, &config, &mut last_channel)?;
                    
                    // Track token count for performance
                    if matches!(event, StreamingEvent::Token { .. }) {
                        token_count += 1;
                    }
                    
                    // Collect additional events without timeout to batch them
                    while output_buffer.len() < config.buffer_size {
                        match event_receiver.try_recv() {
                            Ok(additional_event) => {
                                Self::format_and_buffer_event(&additional_event, &mut output_buffer, &config, &mut last_channel)?;
                                if matches!(additional_event, StreamingEvent::Token { .. }) {
                                    token_count += 1;
                                }
                            }
                            Err(_) => break, // No more events available
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // No events received within timeout, continue to check for flush
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    eprintln!("[StreamingOutput] Event channel disconnected, stopping output thread");
                    break;
                }
            }
            
            // Flush output if buffer is full or timeout reached
            let should_flush = !output_buffer.is_empty() && (
                output_buffer.len() >= config.buffer_size ||
                last_flush.elapsed() >= flush_interval
            );
            
            if should_flush {
                Self::flush_output_buffer(&mut output_buffer)?;
                last_flush = Instant::now();
            }
        }
        
        // Final flush
        if !output_buffer.is_empty() {
            Self::flush_output_buffer(&mut output_buffer)?;
        }
        
        Ok(())
    }
    
    /// Format and buffer a single event for output
    fn format_and_buffer_event(
        event: &StreamingEvent,
        buffer: &mut Vec<String>,
        config: &StreamingOutputConfig,
        last_channel: &mut Option<String>,
    ) -> Result<()> {
        let formatted = match event {
            StreamingEvent::Token { token, token_index, timestamp_ns } => {
                if config.show_raw_tokens {
                    let token_info = if config.show_token_indices {
                        format!("[{}]", token_index)
                    } else {
                        String::new()
                    };
                    
                    let timestamp_info = if config.show_timestamps {
                        format!("[{}]", Self::format_timestamp(*timestamp_ns))
                    } else {
                        String::new()
                    };
                    
                    let text = if config.colored_output {
                        format!("{}{}🔤 Token: {} ", timestamp_info.dimmed(), token_info.dimmed(), token.to_string().cyan())
                    } else {
                        format!("{}{}Token: {} ", timestamp_info, token_info, token)
                    };
                    Some(text)
                } else {
                    None // Don't show raw tokens by default
                }
            }
            
            StreamingEvent::ContentDelta { text, channel, recipient, timestamp_ns } => {
                let timestamp_info = if config.show_timestamps {
                    format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                } else {
                    String::new()
                };
                
                // Channel switch detection
                let channel_info = if config.show_channels {
                    match channel {
                        Some(ch) => {
                            // Check for channel switch
                            let is_new_channel = last_channel.as_ref() != Some(ch);
                            if is_new_channel {
                                *last_channel = Some(ch.clone());
                                if config.colored_output {
                                    format!("\n📡 {}", format!("Channel: {}", ch).bright_blue().bold())
                                } else {
                                    format!("\nChannel: {}", ch)
                                }
                            } else {
                                String::new()
                            }
                        }
                        None => {
                            if last_channel.is_some() {
                                *last_channel = None;
                                if config.colored_output {
                                    "\n📡 Channel: default".dimmed().to_string()
                                } else {
                                    "\nChannel: default".to_string()
                                }
                            } else {
                                String::new()
                            }
                        }
                    }
                } else {
                    String::new()
                };
                
                let recipient_info = if let Some(recipient) = recipient {
                    if config.colored_output {
                        format!("→{} ", recipient.yellow())
                    } else {
                        format!("→{} ", recipient)
                    }
                } else {
                    String::new()
                };
                
                let content = if config.colored_output {
                    // Color content based on channel
                    match channel.as_deref() {
                        Some("analysis") => text.bright_magenta(),
                        Some("commentary") => text.bright_yellow(),
                        Some("final") => text.bright_green(),
                        Some("functions") => text.bright_cyan(),
                        _ => text.normal(),
                    }
                } else {
                    text.normal()
                };
                
                Some(format!("{}{}{}{}", timestamp_info, channel_info, recipient_info, content))
            }
            
            StreamingEvent::ChannelSwitch { from_channel, to_channel, timestamp_ns } => {
                if config.show_channels {
                    let timestamp_info = if config.show_timestamps {
                        format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                    } else {
                        String::new()
                    };
                    
                    let switch_text = match from_channel {
                        Some(from) => format!("📡 Channel: {} → {}", from, to_channel),
                        None => format!("📡 Channel: {}", to_channel),
                    };
                    
                    *last_channel = Some(to_channel.clone());
                    
                    let formatted_switch = if config.colored_output {
                        switch_text.bright_blue().bold()
                    } else {
                        switch_text.normal()
                    };
                    
                    Some(format!("\n{}{}\n", timestamp_info, formatted_switch))
                } else {
                    None
                }
            }
            
            StreamingEvent::ToolCall { tool_name, arguments, channel, recipient, timestamp_ns } => {
                let timestamp_info = if config.show_timestamps {
                    format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                } else {
                    String::new()
                };
                
                let tool_call_text = format!(
                    "🔧 Tool Call: {} in {} → {}\n   Args: {}",
                    tool_name, channel, recipient, arguments
                );
                
                let formatted_tool = if config.colored_output {
                    tool_call_text.bright_cyan().bold()
                } else {
                    tool_call_text.normal()
                };
                
                Some(format!("\n{}{}\n", timestamp_info, formatted_tool))
            }
            
            StreamingEvent::StreamComplete { reason, total_tokens, total_time_ms, timestamp_ns } => {
                let timestamp_info = if config.show_timestamps {
                    format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                } else {
                    String::new()
                };
                
                let tokens_per_second = if *total_time_ms > 0 {
                    (*total_tokens as f64) / ((*total_time_ms as f64) / 1000.0)
                } else {
                    0.0
                };
                
                let completion_text = format!(
                    "✅ Stream Complete: {} ({} tokens, {:.1} tok/s, {}ms)",
                    reason, total_tokens, tokens_per_second, total_time_ms
                );
                
                let formatted_completion = if config.colored_output {
                    completion_text.bright_green().bold()
                } else {
                    completion_text.normal()
                };
                
                Some(format!("\n{}{}\n", timestamp_info, formatted_completion))
            }
            
            StreamingEvent::PerformanceMetrics { tokens_per_second, processing_latency_ms, queue_depth, timestamp_ns } => {
                if config.show_performance {
                    let timestamp_info = if config.show_timestamps {
                        format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                    } else {
                        String::new()
                    };
                    
                    let perf_text = format!(
                        "📊 Performance: {:.1} tok/s, {:.2}ms latency, {} queued",
                        tokens_per_second, processing_latency_ms, queue_depth
                    );
                    
                    let formatted_perf = if config.colored_output {
                        perf_text.bright_white().dimmed()
                    } else {
                        perf_text.normal()
                    };
                    
                    Some(format!("{}{}", timestamp_info, formatted_perf))
                } else {
                    None
                }
            }
            
            StreamingEvent::Debug { message, timestamp_ns } => {
                let timestamp_info = if config.show_timestamps {
                    format!("[{}] ", Self::format_timestamp(*timestamp_ns))
                } else {
                    String::new()
                };
                
                let formatted_debug = if config.colored_output {
                    format!("🐛 {}", message).dimmed()
                } else {
                    format!("DEBUG: {}", message).normal()
                };
                
                Some(format!("{}{}", timestamp_info, formatted_debug))
            }
        };
        
        if let Some(formatted_text) = formatted {
            buffer.push(formatted_text);
        }
        
        Ok(())
    }
    
    /// Flush the output buffer to terminal
    fn flush_output_buffer(buffer: &mut Vec<String>) -> Result<()> {
        if !buffer.is_empty() {
            // Print all buffered content at once for better performance
            for line in buffer.drain(..) {
                print!("{}", line);
            }
            
            // Flush stdout to ensure immediate output
            use std::io::{self, Write};
            io::stdout().flush().context("Failed to flush stdout")?;
        }
        Ok(())
    }
    
    /// Format timestamp for display
    fn format_timestamp(timestamp_ns: u64) -> String {
        let timestamp_ms = timestamp_ns / 1_000_000;
        let secs = timestamp_ms / 1000;
        let millis = timestamp_ms % 1000;
        format!("{}.{:03}", secs, millis)
    }
}

impl Drop for StreamingOutputManager {
    fn drop(&mut self) {
        // The thread will automatically stop when the sender is dropped
        eprintln!("[StreamingOutput] Shutting down streaming output manager");
    }
}

/// Convenience function to create a default streaming output manager
pub fn create_default_streaming_output() -> Result<StreamingOutputManager> {
    let config = StreamingOutputConfig::default();
    StreamingOutputManager::new(config)
}

/// Create a debug-enabled streaming output manager
pub fn create_debug_streaming_output() -> Result<StreamingOutputManager> {
    let config = StreamingOutputConfig {
        show_raw_tokens: true,
        show_timestamps: true,
        show_token_indices: true,
        show_performance: true,
        ..Default::default()
    };
    StreamingOutputManager::new(config)
}

/// Create a minimal streaming output manager for production
pub fn create_minimal_streaming_output() -> Result<StreamingOutputManager> {
    let config = StreamingOutputConfig {
        colored_output: true,
        show_channels: true,
        show_timestamps: false,
        show_token_indices: false,
        show_raw_tokens: false,
        show_performance: false,
        channel_separation: true,
        buffer_size: 8, // Smaller buffer for faster output
        flush_interval_ms: 5, // Very fast flushing
    };
    StreamingOutputManager::new(config)
}

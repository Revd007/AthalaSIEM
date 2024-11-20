import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc

class ModelOptimizer:
    def __init__(self, 
                 model: nn.Module,
                 device: str = None,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 gradient_accumulation_steps: int = 4,
                 max_gradient_norm: float = 1.0):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_gradient_norm = max_gradient_norm
        
        # Initialize optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=1,
            steps_per_epoch=1000,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Initialize memory management
        self.memory_manager = MemoryManager(device=self.device)
        
        # Initialize thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize queues for continuous learning
        self.training_queue = Queue(maxsize=1000)
        self.inference_queue = Queue(maxsize=1000)
        
        # Start background training thread
        self.training_thread = threading.Thread(target=self._continuous_training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

    async def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with automatic mixed precision"""
        self.model.train()
        
        try:
            # Move batch to device
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            with autocast():
                outputs = self.model(**batch_data)
                loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.max_gradient_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_gradient_norm
                )
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Memory optimization
            await self.memory_manager.optimize()
            
            return {
                'loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return {'loss': float('inf')}

    async def inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference with optimized memory usage"""
        self.model.eval()
        
        try:
            # Process input asynchronously
            processed_input = await self._preprocess_input(input_data)
            
            with torch.no_grad(), autocast():
                outputs = self.model(**processed_input)
            
            # Process output asynchronously
            result = await self._postprocess_output(outputs)
            
            return result
            
        except Exception as e:
            print(f"Error in inference: {e}")
            return {'error': str(e)}

    def _continuous_training_loop(self):
        """Background thread for continuous model updates"""
        while True:
            try:
                # Get batch from queue
                batch_data = self.training_queue.get()
                if batch_data is None:
                    continue
                
                # Create event loop for async training
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Perform training step
                loop.run_until_complete(self.train_step(batch_data))
                
                # Clean up
                loop.close()
                
            except Exception as e:
                print(f"Error in continuous training loop: {e}")
                continue

    async def _preprocess_input(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess input data asynchronously"""
        def _process():
            processed = {}
            for key, value in input_data.items():
                if isinstance(value, np.ndarray):
                    processed[key] = torch.from_numpy(value).to(self.device)
                elif isinstance(value, torch.Tensor):
                    processed[key] = value.to(self.device)
                else:
                    processed[key] = value
            return processed
        
        # Run preprocessing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _process)

    async def _postprocess_output(self, outputs: Any) -> Dict[str, Any]:
        """Postprocess model outputs asynchronously"""
        def _process():
            if isinstance(outputs, torch.Tensor):
                return outputs.cpu().numpy()
            elif isinstance(outputs, dict):
                return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                       for k, v in outputs.items()}
            return outputs
        
        # Run postprocessing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _process)

class MemoryManager:
    def __init__(self, device: str):
        self.device = device
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    async def optimize(self):
        """Optimize memory usage"""
        if self.device == 'cuda':
            # Clear GPU cache if memory usage is high
            if self._get_gpu_memory_usage() > self.memory_threshold:
                torch.cuda.empty_cache()
                gc.collect()
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage"""
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated + reserved) / total
        return 0.0
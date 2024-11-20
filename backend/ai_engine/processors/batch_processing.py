from typing import List, Dict, Any, Generator
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from collections import deque
import time

class BatchProcessor:
    def __init__(self, 
                 batch_size: int = 32,
                 max_queue_size: int = 1000,
                 processing_interval: float = 0.1,
                 num_workers: int = 4):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.processing_interval = processing_interval
        self.num_workers = num_workers
        
        self.event_queue = deque(maxlen=max_queue_size)
        self.thread_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=num_workers)
        
        # Initialize processing flags
        self.is_processing = False
        self.last_process_time = time.time()
        
    async def add_event(self, event: Dict[str, Any]):
        """Add event to processing queue"""
        if len(self.event_queue) < self.max_queue_size:
            self.event_queue.append(event)
            
            # Trigger processing if conditions are met
            await self._check_processing_conditions()
        else:
            logging.warning("Event queue is full, dropping event")
    
    async def _check_processing_conditions(self):
        """Check if batch processing should be triggered"""
        current_time = time.time()
        time_condition = (current_time - self.last_process_time) >= self.processing_interval
        size_condition = len(self.event_queue) >= self.batch_size
        
        if (time_condition or size_condition) and not self.is_processing:
            await self.process_batch()
    
    async def process_batch(self):
        """Process a batch of events"""
        try:
            self.is_processing = True
            
            # Get batch of events
            batch = []
            while len(batch) < self.batch_size and self.event_queue:
                batch.append(self.event_queue.popleft())
            
            if not batch:
                return
            
            # Process batch in parallel
            results = await self._parallel_process(batch)
            
            # Update processing timestamp
            self.last_process_time = time.time()
            
            return results
            
        finally:
            self.is_processing = False
    
    async def _parallel_process(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch items in parallel"""
        loop = asyncio.get_event_loop()
        
        # Split batch into chunks for parallel processing
        chunk_size = max(1, len(batch) // self.num_workers)
        chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(
                self.process_executor,
                self._process_chunk,
                chunk
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for sublist in results for item in sublist]
    
    def _process_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of events"""
        processed_chunk = []
        for event in chunk:
            try:
                processed_event = self._process_single_event(event)
                processed_chunk.append(processed_event)
            except Exception as e:
                logging.error(f"Error processing event: {e}")
        return processed_chunk
    
    def _process_single_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event"""
        # Implement your event processing logic here
        return event

class BatchDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.feature_extractor(self.data[idx])

class OptimizedBatchProcessor(BatchProcessor):
    def __init__(self, 
                 model,
                 feature_extractor,
                 batch_size: int = 32,
                 max_queue_size: int = 1000,
                 processing_interval: float = 0.1,
                 num_workers: int = 4,
                 device: str = None):
        super().__init__(batch_size, max_queue_size, processing_interval, num_workers)
        
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    async def _parallel_process(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch using optimized ML model"""
        # Create dataset and dataloader
        dataset = BatchDataset(batch, self.feature_extractor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        results = []
        with torch.no_grad():
            for batch_features in dataloader:
                # Move batch to device
                if isinstance(batch_features, torch.Tensor):
                    batch_features = batch_features.to(self.device)
                elif isinstance(batch_features, dict):
                    batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
                
                # Get model predictions
                predictions = self.model(batch_features)
                
                # Process predictions
                processed_results = self._process_predictions(predictions, batch)
                results.extend(processed_results)
        
        return results
    
    def _process_predictions(self, predictions: torch.Tensor, original_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process model predictions"""
        predictions = predictions.cpu().numpy()
        processed_results = []
        
        for pred, original_event in zip(predictions, original_batch):
            processed_event = original_event.copy()
            processed_event.update({
                'prediction': pred,
                'processed_timestamp': time.time()
            })
            processed_results.append(processed_event)
        
        return processed_results

class BatchOptimizer:
    def __init__(self):
        self.batch_stats = {
            'processing_times': [],
            'batch_sizes': [],
            'memory_usage': []
        }
    
    def optimize_batch_size(self, 
                          processor: BatchProcessor,
                          test_sizes: List[int],
                          test_data: List[Dict[str, Any]]) -> int:
        """Find optimal batch size"""
        optimal_batch_size = test_sizes[0]
        best_efficiency = float('inf')
        
        for batch_size in test_sizes:
            processor.batch_size = batch_size
            efficiency = self._measure_efficiency(processor, test_data)
            
            if efficiency < best_efficiency:
                best_efficiency = efficiency
                optimal_batch_size = batch_size
        
        return optimal_batch_size
    
    def _measure_efficiency(self, 
                          processor: BatchProcessor,
                          test_data: List[Dict[str, Any]]) -> float:
        """Measure processing efficiency"""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        # Process test data
        asyncio.run(self._process_test_data(processor, test_data))
        
        processing_time = time.time() - start_time
        memory_used = self._get_memory_usage() - memory_start
        
        # Calculate efficiency metric
        efficiency = processing_time * memory_used / len(test_data)
        
        # Update stats
        self.batch_stats['processing_times'].append(processing_time)
        self.batch_stats['batch_sizes'].append(processor.batch_size)
        self.batch_stats['memory_usage'].append(memory_used)
        
        return efficiency
    
    async def _process_test_data(self, 
                               processor: BatchProcessor,
                               test_data: List[Dict[str, Any]]):
        """Process test data through batch processor"""
        for event in test_data:
            await processor.add_event(event)
        
        # Ensure all events are processed
        while len(processor.event_queue) > 0:
            await processor.process_batch()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
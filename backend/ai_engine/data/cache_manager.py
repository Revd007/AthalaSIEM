from typing import Dict, Any, Optional
import torch
from collections import OrderedDict
import threading
import time

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                item, timestamp = self.cache[key]
                if time.time() - timestamp <= self.ttl:
                    self.cache.move_to_end(key)
                    return item
                else:
                    del self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())
    
    def _cleanup_loop(self):
        """Background thread for cache cleanup"""
        while True:
            time.sleep(60)  # Check every minute
            current_time = time.time()
            with self.lock:
                keys_to_remove = [
                    key for key, (_, timestamp) in self.cache.items()
                    if current_time - timestamp > self.ttl
                ]
                for key in keys_to_remove:
                    del self.cache[key]
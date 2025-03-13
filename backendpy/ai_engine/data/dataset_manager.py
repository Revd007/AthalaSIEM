import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path

class AIDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data = []
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        # Load different types of data
        self.load_data(data_path)
        
    def load_data(self, data_path: str):
        """Load data from multiple sources"""
        try:
            # Load structured data (CSV/JSON)
            self._load_structured_data(data_path)
            
            # Load text data
            self._load_text_data(data_path)
            
            # Load system interaction data
            self._load_system_data(data_path)
            
            # Load user interaction data
            self._load_user_data(data_path)
            
            self.logger.info(f"Loaded {len(self.data)} total samples")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _load_structured_data(self, base_path: str):
        """Load structured data from CSV/JSON files"""
        path = Path(base_path) / 'structured'
        
        # Load event data
        events_df = pd.read_csv(path / 'events.csv')
        for _, row in events_df.iterrows():
            self.data.append({
                'type': 'event',
                'input': row['event_data'],
                'label': row['event_type'],
                'metadata': json.loads(row['metadata'])
            })
        
        # Load alert data
        alerts_df = pd.read_csv(path / 'alerts.csv')
        for _, row in alerts_df.iterrows():
            self.data.append({
                'type': 'alert',
                'input': row['alert_data'],
                'label': row['severity'],
                'metadata': json.loads(row['metadata'])
            })

    def _load_text_data(self, base_path: str):
        """Load text data for NLP training"""
        path = Path(base_path) / 'text'
        
        # Load conversations
        with open(path / 'conversations.json', 'r') as f:
            conversations = json.load(f)
            for conv in conversations:
                self.data.append({
                    'type': 'conversation',
                    'input': conv['user_input'],
                    'response': conv['ai_response'],
                    'context': conv['context'],
                    'metadata': conv['metadata']
                })
        
        # Load commands and responses
        with open(path / 'commands.json', 'r') as f:
            commands = json.load(f)
            for cmd in commands:
                self.data.append({
                    'type': 'command',
                    'input': cmd['command'],
                    'output': cmd['output'],
                    'success': cmd['success'],
                    'metadata': cmd['metadata']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
            
        return item

class DatasetManager:
    def __init__(self, config: Dict):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation and test datasets"""
        # Load full dataset
        full_dataset = AIDataset(self.config['data_path'])
        
        # Split dataset
        train_size = self.config.get('train_size', 0.7)
        val_size = self.config.get('val_size', 0.15)
        test_size = self.config.get('test_size', 0.15)
        
        train_data, temp_data = train_test_split(
            full_dataset.data,
            train_size=train_size,
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_size/(test_size + val_size),
            random_state=42
        )
        
        # Create datasets
        self.train_dataset = AIDataset(train_data)
        self.val_dataset = AIDataset(val_data)
        self.test_dataset = AIDataset(test_data)
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        return train_loader, val_loader, test_loader
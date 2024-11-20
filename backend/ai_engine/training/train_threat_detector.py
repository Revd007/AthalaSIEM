import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .. import ThreatDetector
import logging
import json
from typing import List, Dict
import numpy as np

class ThreatDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ThreatDetectorTrainer:
    def __init__(self, model_config: Dict):
        self.model = ThreatDetector(num_classes=model_config['num_classes'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_data: List[Dict], val_data: List[Dict], epochs: int = 10):
        # Prepare datasets
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        val_texts = [item['text'] for item in val_data]
        val_labels = [item['label'] for item in val_data]
        
        train_dataset = ThreatDataset(train_texts, train_labels, self.model.tokenizer)
        val_dataset = ThreatDataset(val_texts, val_labels, self.model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            logging.info(f'Epoch {epoch+1}/{epochs}:')
            logging.info(f'Average training loss: {avg_train_loss:.4f}')
            logging.info(f'Average validation loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('best_model.pt')
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
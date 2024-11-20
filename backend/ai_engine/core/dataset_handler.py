import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import json

class UniversalDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 features: list,
                 target: str = None,
                 transform=None):
        self.data = data
        self.features = features
        self.target = target
        self.transform = transform
        
        # Prepare data
        self.X = self._prepare_features()
        self.y = self._prepare_labels() if target else None
        
    def _prepare_features(self) -> np.ndarray:
        return self.data[self.features].values
        
    def _prepare_labels(self) -> np.ndarray:
        return self.data[self.target].values
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features = self.X[idx]
        if self.transform:
            features = self.transform(features)
            
        item = {'features': torch.FloatTensor(features)}
        
        if self.y is not None:
            item['label'] = torch.LongTensor([self.y[idx]])[0]
            
        return item

class CyberSecurityDataHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    async def process_datasets(self, 
                             dataset_configs: List[Dict[str, Any]]) -> Tuple[DataLoader, DataLoader]:
        """Process multiple datasets and combine them"""
        all_data = []
        
        for config in dataset_configs:
            try:
                # Load dataset
                df = self._load_dataset(config['path'], config['type'])
                
                # Process specific dataset
                df = self._process_specific_dataset(df, config['dataset_name'])
                
                all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {config['path']}: {str(e)}")
                continue
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Process combined data
        processed_df = self._process_data(combined_df)
        
        # Split data
        train_df, test_df = self._split_data(processed_df)
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_df, is_train=True)
        test_loader = self._create_dataloader(test_df, is_train=False)
        
        return train_loader, test_loader
    
    def _load_dataset(self, path: str, file_type: str) -> pd.DataFrame:
        """Load dataset from various sources"""
        if path.startswith('hf://'):
            # Handle Hugging Face datasets
            path = path.replace('hf://', '')
            if file_type == 'csv':
                return pd.read_csv(path)
            elif file_type == 'tsv':
                return pd.read_csv(path, sep='\t')
            elif file_type == 'json':
                return pd.read_json(path, lines=True)
            elif file_type == 'parquet':
                return pd.read_parquet(path)
        else:
            # Handle local files
            if file_type == 'csv':
                return pd.read_csv(path)
            elif file_type == 'tsv':
                return pd.read_csv(path, sep='\t')
            elif file_type == 'json':
                return pd.read_json(path, lines=True)
            elif file_type == 'parquet':
                return pd.read_parquet(path)
                
        raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_specific_dataset(self, 
                                df: pd.DataFrame, 
                                dataset_name: str) -> pd.DataFrame:
        """Process specific dataset based on its characteristics"""
        if dataset_name == 'cyber_threat_intelligence':
            # Process cyber threat intelligence dataset
            df = self._process_threat_intelligence(df)
        elif dataset_name == 'code_vulnerability':
            # Process code vulnerability dataset
            df = self._process_vulnerability(df)
        elif dataset_name == 'firewall_trivia':
            # Process firewall trivia dataset
            df = self._process_firewall(df)
            
        return df
    
    def _process_threat_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cyber threat intelligence dataset"""
        # Add specific processing for threat intelligence data
        df = df.copy()
        
        # Handle text columns
        text_columns = ['description', 'threat_type', 'source']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                df[col] = df[col].astype(str).str.lower()
        
        # Handle numeric columns
        numeric_columns = ['severity', 'confidence']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _process_vulnerability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process code vulnerability dataset"""
        df = df.copy()
        
        # Process code-related columns
        if 'code' in df.columns:
            df['code_length'] = df['code'].str.len()
            df['has_function'] = df['code'].str.contains('function|def', case=False)
        
        # Process vulnerability type
        if 'vulnerability_type' in df.columns:
            df['vulnerability_type'] = self.label_encoder.fit_transform(df['vulnerability_type'])
        
        return df
    
    def _process_firewall(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process firewall dataset"""
        df = df.copy()
        
        # Process IP addresses and ports
        if 'source_ip' in df.columns:
            df['is_internal'] = df['source_ip'].str.startswith(('10.', '192.168.', '172.'))
        
        # Process protocols
        if 'protocol' in df.columns:
            df['protocol'] = self.label_encoder.fit_transform(df['protocol'])
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """General data processing"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Fill categorical
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[
                (df[col] >= Q1 - 1.5 * IQR) & 
                (df[col] <= Q3 + 1.5 * IQR)
            ]
            
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for combined dataset"""
        # Add timestamp-based features if available
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Add text-based features if available
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[f'{col}_length'] = df[col].str.len()
        
        return df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test"""
        train_size = int(len(df) * 0.8)
        return df[:train_size], df[train_size:]
    
    def _create_dataloader(self, 
                          df: pd.DataFrame,
                          is_train: bool = True) -> DataLoader:
        """Create dataloader"""
        features = [col for col in df.columns if col != self.config['target_column']]
        
        dataset = UniversalDataset(
            df,
            features=features,
            target=self.config['target_column'],
            transform=self._transform_features if is_train else None
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=is_train,
            num_workers=self.config['num_workers']
        )
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using scaler"""
        return self.scaler.transform(features.reshape(1, -1))[0]
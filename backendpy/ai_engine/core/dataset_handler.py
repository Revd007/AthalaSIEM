from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import json
from docx import Document
import PyPDF2
import bs4
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chardet
import dask.dataframe as dd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import random

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
        """Prepare feature columns"""
        try:
            # If features list is empty, use all columns except target
            if not self.features and self.target:
                self.features = [col for col in self.data.columns if col != self.target]
            elif not self.features:
                self.features = self.data.columns.tolist()
                
            # Convert to numeric, replacing non-numeric with 0
            numeric_data = self.data[self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
            return numeric_data.values
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            # Return empty array with correct shape
            return np.zeros((len(self.data), len(self.features)))
        
    def _prepare_labels(self) -> np.ndarray:
        """Prepare target labels"""
        try:
            if self.target not in self.data.columns:
                raise KeyError(f"Target column '{self.target}' not found in data")
                
            # Convert to numeric labels
            label_encoder = LabelEncoder()
            return label_encoder.fit_transform(self.data[self.target].values)
            
        except Exception as e:
            logging.error(f"Error preparing labels: {e}")
            # Return dummy labels
            return np.zeros(len(self.data))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features = self.X[idx]
        if self.transform:
            features = self.transform(features)
            
        item = {'features': torch.FloatTensor(features)}
        
        if self.y is not None:
            item['labels'] = torch.LongTensor([self.y[idx]])[0]
            
        return item

    def _load_text_file(self, file_path: str) -> pd.DataFrame:
        try:
            # Detect the file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Read the file with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                
            # Clean and process the lines
            data = [line.strip() for line in lines if line.strip()]
            return pd.DataFrame(data, columns=['content'])
            
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {str(e)}")
            return pd.DataFrame()

    def _load_json_file(self, file_path: str) -> pd.DataFrame:
        """Load JSON file with special handling for configuration files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Check if this is a MITRE ATT&CK mapping file
            if isinstance(data, dict) and 'techniques' in data:
                # Extract only the relevant technique information
                techniques = []
                for technique in data['techniques']:
                    technique_data = {
                        'techniqueID': technique.get('techniqueID'),
                        'tactic': technique.get('tactic'),
                        'score': technique.get('score'),
                        'comment': technique.get('comment'),
                        # Convert lists to JSON strings to maintain data integrity
                        'metadata': json.dumps(technique.get('metadata', [])),
                        'links': json.dumps(technique.get('links', []))
                    }
                    techniques.append(technique_data)
                return pd.DataFrame(techniques)
                
            # For regular JSON files
            return pd.json_normalize(data)
                
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame()

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data while preserving configuration"""
        try:
            # Create a temporary copy for deduplication
            temp_df = df.copy()
            
            # Temporarily convert lists to strings only for deduplication
            list_cols = temp_df.applymap(lambda x: isinstance(x, list)).any()
            cols_with_lists = list_cols[list_cols].index
            
            for col in cols_with_lists:
                temp_df[col] = temp_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
            
            # Get unique indices using the temporary dataframe
            unique_indices = ~temp_df.duplicated()
            
            # Use the indices on the original dataframe
            return df[unique_indices].reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return df

class CyberSecurityDataHandler:
    def __init__(self, data_path: Path, device: torch.device):
        self.data_path = data_path
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.recent_events = []  # Cache untuk recent events
        self.max_events = 100    # Maksimum events yang disimpan
        
        # Initialize transformers
        self.robust_scaler = RobustScaler()
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Define column types
        self.numeric_features = [
            'severity_score', 'confidence_score', 'duration',
            'bytes_sent', 'bytes_received', 'packet_count'
        ]
        self.categorical_features = [
            'event_type', 'protocol', 'source_type', 'destination_type'
        ]
        
        # Initialize column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.robust_scaler, self.numeric_features),
                ('cat', self.onehot_encoder, self.categorical_features)
            ],
            remainder='passthrough'
        )

    async def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from dataset"""
        try:
            events_path = self.data_path / "events.json"
            if events_path.exists():
                with open(events_path) as f:
                    events = json.load(f)
                    return events[-limit:]
            return []
        except Exception as e:
            self.logger.error(f"Error getting recent events: {e}")
            return []

    def add_event(self, event: Dict[str, Any]):
        """Add new event to recent events"""
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_events:
            self.recent_events.pop(0)  # Remove oldest event

        # Save to file
        try:
            events_path = self.data_path / "events.json"
            with open(events_path, 'w') as f:
                json.dump(self.recent_events, f)
        except Exception as e:
            self.logger.error(f"Error saving events: {e}")

    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using multiple methods"""
        # Isolation Forest detection
        isolation_labels = self.isolation_forest.fit_predict(
            data[self.numeric_features]
        )
        
        # Z-score detection
        z_scores = np.abs(stats.zscore(data[self.numeric_features]))
        z_score_outliers = (z_scores > 3).any(axis=1)
        
        # Modified Box-Plot (IQR) detection with different thresholds per feature
        iqr_outliers = np.zeros(len(data), dtype=bool)
        for col in self.numeric_features:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.config.get('outlier_thresholds', {}).get(col, 1.5)
            iqr_outliers |= (
                (data[col] < (Q1 - threshold * IQR)) | 
                (data[col] > (Q3 + threshold * IQR))
            )
        
        # Combine detections (majority voting)
        combined_outliers = (
            (isolation_labels == -1).astype(int) + 
            z_score_outliers.astype(int) + 
            iqr_outliers.astype(int)
        ) >= 2
        
        return combined_outliers

    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers based on configuration"""
        outliers = self.detect_outliers(data)
        
        strategy = self.config.get('outlier_strategy', 'remove')
        if strategy == 'remove':
            return data[~outliers]
        elif strategy == 'clip':
            for col in self.numeric_features:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = self.config.get('outlier_thresholds', {}).get(col, 1.5)
                data.loc[outliers, col] = data.loc[outliers, col].clip(
                    lower=Q1 - threshold * IQR,
                    upper=Q3 + threshold * IQR
                )
            return data
        elif strategy == 'transform':
            # Apply Box-Cox transformation to reduce impact of outliers
            for col in self.numeric_features:
                if (data[col] > 0).all():
                    data.loc[outliers, col], _ = stats.boxcox(data.loc[outliers, col])
            return data
        
        return data

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data with outlier handling and feature transformation"""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Handle outliers
            data = self.handle_outliers(data)
            
            # Apply transformations
            transformed_data = self.preprocessor.fit_transform(data)
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numeric features
        for col in self.numeric_features:
            if data[col].isnull().any():
                # Use median for numeric features
                data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical features
        for col in self.categorical_features:
            if data[col].isnull().any():
                # Use mode for categorical features
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data

    def get_feature_names(self) -> List[str]:
        """Get list of transformed feature names"""
        numeric_features = self.numeric_features
        categorical_features = []
        
        # Get encoded feature names for categorical variables
        if hasattr(self.onehot_encoder, 'get_feature_names_out'):
            categorical_features = self.onehot_encoder.get_feature_names_out(
                self.categorical_features
            )
        
        return list(numeric_features) + list(categorical_features)

    def load_datasets(self, dataset_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load multiple datasets and combine them"""
        data_frames = []
        
        for config in dataset_configs:
            try:
                path = config['path']
                file_type = config.get('format') or config.get('type')
                
                if not path or not file_type:
                    self.logger.error(f"Invalid dataset configuration: {config}")
                    continue
                    
                if not os.path.exists(path):
                    self.logger.error(f"File not found: {path}")
                    continue

                df = self.loader.load_file(path, file_type)
                
                # Process specific dataset if needed
                if 'name' in config:
                    df = self._process_specific_dataset(df, config['name'])
                
                data_frames.append(df)
                
            except Exception as e:
                self.logger.error(f"Error loading dataset {config.get('name', 'unknown')}: {str(e)}")
                continue

        if not data_frames:
            raise ValueError("No datasets were successfully loaded")

        try:
            combined_df = pd.concat(data_frames, ignore_index=True)
            return self._validate_data(combined_df)
        except Exception as e:
            raise ValueError(f"Error combining datasets: {str(e)}")

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
        df = df.copy()

        # Handle text columns
        text_columns = ['description', 'threat_type', 'source']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str).str.lower()

        # Handle numeric columns
        numeric_columns = ['severity', 'confidence']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Cek jika semua nilai menjadi NaN setelah konversi
                if df[col].isnull().all():
                    self.logger.error(f"All values in column '{col}' are non-numeric after conversion.")
                    raise ValueError(f"Column '{col}' contains all non-numeric values.")

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
        """Process data in chunks"""
        try:
            processed_chunks = []
            for chunk_start in range(0, len(df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(df))
                chunk = df.iloc[chunk_start:chunk_end].copy()
                
                # Remove duplicates in chunk
                chunk.drop_duplicates(inplace=True)
                
                # Handle missing values in chunk
                missing_count = chunk.isnull().sum().sum()
                self.logger.info(f"Chunk {chunk_start}-{chunk_end}: {missing_count} missing values")
                
                processed_chunks.append(chunk)
                
                # Clear memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Concatenate all chunks
            return pd.concat(processed_chunks, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        self.logger.info(f"Before handling missing values, total missing values: {df.isnull().sum().sum()}")
        
        # Fill numeric missing values
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(df[numeric_cols].mean())
        
        # Fill categorical missing values
        df.loc[:, categorical_cols] = df.loc[:, categorical_cols].fillna('unknown')
        
        self.logger.info(f"After handling missing values, total missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.logger.info(f"Processing outliers for column '{col}'")
            
            # Check for sufficient non-NaN values
            if df[col].dropna().nunique() < 2:
                self.logger.warning(
                    f"Column '{col}' does not have enough unique, non-NaN values to compute outliers. Skipping."
                )
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0 or np.isnan(IQR):
                self.logger.warning(
                    f"IQR is zero or NaN for column '{col}'. Skipping outlier removal for this column."
                )
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            filtered_df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            if filtered_df.empty:
                self.logger.warning(
                    f"Outlier removal for column '{col}' would remove all data. Skipping this column."
                )
                continue
            else:
                df = filtered_df
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for combined dataset"""
        if df.empty:
            return df
        
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        # Prepare dictionary for new columns
        new_columns = {}
        
        # Add timestamp-based features if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            new_columns.update({
                'hour': df['timestamp'].dt.hour,
                'day_of_week': df['timestamp'].dt.dayofweek
            })
        
        # Add text-based features if available
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            new_columns[f'{col}_length'] = df[col].str.len()
        
        # Combine all new columns at once
        if new_columns:
            new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
            return new_df
        
        return df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")
            
        # Get validation split from config or use default
        val_split = self.config.get('validation_split', 0.2)
        
        # Ensure we have enough data to split
        min_samples = 2  # Minimum samples needed for train and test
        if len(df) < min_samples:
            raise ValueError(f"Not enough samples to split. Found {len(df)}, need at least {min_samples}")
            
        # Calculate split index
        split_idx = int(len(df) * (1 - val_split))
        
        # Ensure both splits have at least one sample
        if split_idx == 0 or split_idx == len(df):
            split_idx = len(df) // 2
            
        # Split the data
        train_df = df.iloc[:split_idx]
        
        return train_df, df[split_idx:]
    
    def _create_dataloader(self, df: pd.DataFrame, is_train: bool = True) -> DataLoader:
        """Create DataLoader with empty dataset handling"""

        if df.empty:
            raise ValueError("Cannot create DataLoader from empty DataFrame")

        features = df.columns.tolist()

        if self.config['target_column'] and self.config['target_column'] in df.columns:
            features.remove(self.config['target_column'])
            target = df[self.config['target_column']]
        else:
            self.logger.warning("Target column is not specified or not in DataFrame.")
            target = None

        if not features:
            raise ValueError("No features available in DataFrame")

        dataset = UniversalDataset(
            df,
            features=features,
            target=target,
            transform=self._transform_features if is_train else None
        )

        if len(dataset) == 0:
            raise ValueError("Dataset is empty after processing")

        batch_size = min(self.config.get('batch_size', 32), len(dataset))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.config.get('num_workers', 0)
        )
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using scaler"""
        return self.scaler.transform(features.reshape(1, -1))[0]

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data before processing with chunking"""
        if df.empty:
            raise ValueError("DataFrame is empty before validation.")

        try:
            # Process in chunks to avoid memory issues
            chunks = []
            for chunk_start in range(0, len(df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(df))
                chunk = df.iloc[chunk_start:chunk_end].copy()
                
                # Validate numeric columns in chunk
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                if chunk[numeric_cols].isnull().all().all():
                    self.logger.warning(f"Chunk {chunk_start}-{chunk_end} contains all null numeric values")
                    continue
                
                chunks.append(chunk)
            
            if not chunks:
                raise ValueError("No valid chunks after validation")
            
            return pd.concat(chunks, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {e}")
            raise

    def validate_dataset_directory(self, dataset_dir: str) -> Dict[str, Any]:
        """Validasi direktori dataset dan return statistiknya"""
        try:
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                raise ValueError(f"Dataset directory not found: {dataset_dir}")
            
            # Hitung statistik file
            stats = {
                'total_files': 0,
                'by_type': {},
                'total_size': 0
            }
            
            # Scan semua file
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    file_type = file_path.suffix.lower()
                    file_size = file_path.stat().st_size
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            self.logger.info(f"Dataset validation complete: {stats['total_files']} files found")
            return stats
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            raise

    def process_dataset_files(self, dataset_dir: str) -> pd.DataFrame:
        """Process all files dataset and combine them into one DataFrame"""
        try:
            # Define dtypes for problematic columns
            dtype_map = {
                'metadata_github_created_at': 'object',
                'metadata_github_updated_at': 'object'
            }
            
            # Use pandas instead of dask for more control over data types
            dfs = []
            for file_path in Path(dataset_dir).rglob('*.csv'):
                try:
                    # Read CSV with specified dtypes
                    df = pd.read_csv(
                        file_path, 
                        dtype=dtype_map,
                        parse_dates=['metadata_github_created_at', 'metadata_github_updated_at'],
                        date_parser=lambda x: pd.to_datetime(x, utc=True, errors='coerce')
                    )
                    dfs.append(df)
                    self.logger.debug(f"Successfully loaded {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
                    continue
            
            if not dfs:
                raise ValueError("No valid CSV files were loaded")
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Convert datetime columns to proper format
            datetime_cols = ['metadata_github_created_at', 'metadata_github_updated_at']
            for col in datetime_cols:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_datetime(combined_df[col], utc=True, errors='coerce')
            
            self.logger.info(f"Successfully processed {len(dfs)} files into DataFrame with shape {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            raise

    def _load_text_file(self, file_path: str) -> pd.DataFrame:
        """Load and process text file"""
        try:
            # Detect file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Read file with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                lines = file.readlines()
            
            # Clean and process lines
            processed_lines = []
            for line in lines:
                clean_line = line.strip()
                if clean_line:  # Skip empty lines
                    processed_lines.append({
                        'content': clean_line,
                        'type': Path(file_path).stem.split('.')[0],  # Extract type from filename
                        'timestamp': datetime.now().isoformat()
                    })
            
            return pd.DataFrame(processed_lines)
            
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {str(e)}")
            return pd.DataFrame()

    def _load_pdf_file(self, file_path: str) -> pd.DataFrame:
        """Load and process PDF file"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_data = []
                for page in reader.pages:
                    text = page.extract_text().strip()
                    if text:
                        text_data.append(text)
            return pd.DataFrame({'text': text_data})
        except Exception as e:
            self.logger.error(f"Error loading PDF file {file_path}: {str(e)}")
            return pd.DataFrame()

    def _prepare_data_for_training(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame to tensor with memory-efficient chunking"""
        try:
            tensors = []
            for chunk_start in range(0, len(df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(df))
                chunk = df.iloc[chunk_start:chunk_end]
                
                # Convert chunk to tensor
                chunk_tensor = torch.tensor(chunk.values, dtype=torch.float32)
                
                # Move chunk to device
                chunk_tensor = chunk_tensor.to(self.device)
                tensors.append(chunk_tensor)
                
                # Clear CUDA cache after each chunk
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Concatenate all chunks
            return torch.cat(tensors, dim=0)
            
        except Exception as e:
            self.logger.error(f"Error preparing data for training: {e}")
            raise

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        # Konversi float64 ke float32
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # Konversi int64 ke int32
        int_cols = df.select_dtypes(include=['int64']).columns
        df[int_cols] = df[int_cols].astype('int32')
        
        return df

    def prepare_data_loaders(
        self, 
        dataset_path: Path,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation and test data loaders"""
        try:
            # Load dataset
            df = pd.read_csv(dataset_path, low_memory=False)
            
            # Print available columns untuk debugging
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Untuk sementara, kita buat dummy label jika tidak ada kolom target
            if 'label' not in df.columns:
                self.logger.warning("Target column 'label' not found, creating dummy labels")
                df['label'] = 0  # Dummy labels
            
            # Get target column from config
            target_column = 'label'  # Hardcode untuk sementara
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Convert all features to numeric
            for col in feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Split dataset
            train_df, temp_df = train_test_split(df, test_size=(val_split + test_split), random_state=42)
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=test_split/(val_split + test_split),
                random_state=42
            )
            
            # Create datasets
            train_dataset = UniversalDataset(
                train_df, 
                features=feature_columns,
                target=target_column
            )
            val_dataset = UniversalDataset(
                val_df, 
                features=feature_columns,
                target=target_column
            )
            test_dataset = UniversalDataset(
                test_df, 
                features=feature_columns,
                target=target_column
            )
            
            # Create data loaders with minimal workers
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Reduce workers to avoid multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            self.logger.info(f"Created data loaders - Train: {len(train_loader.dataset)}, "
                            f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error preparing data loaders: {e}")
            raise

class UniversalDatasetLoader:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.supported_formats = {
            'csv': self._load_csv,
            'txt': self._load_text,
            'docx': self._load_docx,
            'pdf': self._load_pdf,
            'html': self._load_html,
            'json': self._load_json
        }
        self.logger = logging.getLogger(__name__)

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load CSV dataset with encoding detection"""
        try:
            # First try with encoding from config
            csv_encoding = self.config.get('data_settings', {}).get('csv_encoding')
            if csv_encoding:
                try:
                    return pd.read_csv(path, encoding=csv_encoding)
                except Exception as e:
                    self.logger.warning(f"Failed to read with configured encoding {csv_encoding}: {e}")

            # Try detecting encoding
            with open(path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding']
                
            try:
                return pd.read_csv(path, encoding=detected_encoding)
            except Exception as e:
                self.logger.warning(f"Failed to read with detected encoding {detected_encoding}: {e}")

            # Try common encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    return pd.read_csv(path, encoding=encoding)
                except Exception:
                    continue

            raise ValueError("Failed to read CSV with any encoding")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def _load_text(self, path: str) -> pd.DataFrame:
        """Load text file with encoding detection"""
        try:
            # Try detecting encoding first
            with open(path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    def _load_docx(self, path: str) -> pd.DataFrame:
        """Load DOCX file"""
        try:
            doc = Document(path)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return pd.DataFrame({'text': paragraphs})
        except Exception as e:
            raise ValueError(f"Error loading DOCX file: {str(e)}")

    def _load_pdf(self, path: str) -> pd.DataFrame:
        """Load PDF file"""
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_data = []
                for page in reader.pages:
                    text = page.extract_text().strip()
                    if text:
                        text_data.append(text)
            return pd.DataFrame({'text': text_data})
        except Exception as e:
            raise ValueError(f"Error loading PDF file: {str(e)}")

    def _load_html(self, path: str) -> pd.DataFrame:
        """Load HTML file and convert to DataFrame"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                soup = bs4.BeautifulSoup(file.read(), 'html.parser')
                
                # Extract text content
                text = soup.get_text(separator=' ', strip=True)
                
                # Extract links
                links = [a.get('href', '') for a in soup.find_all('a', href=True)]
                
                # Extract headers
                headers = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
                
                # Create DataFrame
                data = {
                    'content': [text],
                    'links': [links],
                    'headers': [headers],
                    'source_path': [path],
                    'timestamp': [datetime.now().isoformat()]
                }
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error loading HTML file {path}: {str(e)}")
            raise ValueError(f"Error loading HTML file: {str(e)}")

    def _load_json(self, path: str) -> pd.DataFrame:
        """Load JSON file with special handling for configuration files"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Check if this is a MITRE ATT&CK mapping file
            if isinstance(data, dict) and 'techniques' in data:
                # Extract only the relevant technique information
                techniques = []
                for technique in data['techniques']:
                    technique_data = {
                        'techniqueID': technique.get('techniqueID'),
                        'tactic': technique.get('tactic'),
                        'score': technique.get('score'),
                        'comment': technique.get('comment'),
                        # Convert lists to JSON strings to maintain data integrity
                        'metadata': json.dumps(technique.get('metadata', [])),
                        'links': json.dumps(technique.get('links', []))
                    }
                    techniques.append(technique_data)
                return pd.DataFrame(techniques)
                
            # For regular JSON files
            return pd.json_normalize(data)
                
        except Exception as e:
            self.logger.error(f"Error loading {path}: {str(e)}")
            return pd.DataFrame()

    def load_file(self, path: str, file_type: str) -> pd.DataFrame:
        """Load a file based on its type"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_type = file_type.lower()
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: {list(self.supported_formats.keys())}")
        
        try:
            df = self.supported_formats[file_type](path)
            if df.empty:
                raise ValueError(f"No data loaded from file: {path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_type} file {path}: {str(e)}")
            raise

class BrowserAutomation:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=chrome_options)
        
    async def search_and_collect(self, query: str) -> Dict[str, Any]:
        try:
            self.driver.get(f"https://www.google.com/search?q={query}")
            soup = bs4.BeautifulSoup(self.driver.page_source, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='g'):
                title = result.find('h3')
                link = result.find('a')
                snippet = result.find('div', class_='VwiC3b')
                
                if title and link and snippet:
                    results.append({
                        'title': title.text,
                        'url': link['href'],
                        'snippet': snippet.text
                    })
                    
            return {
                'query': query,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Browser automation error: {e}")
            return None

class CyberSecurityIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        return self.data_iter

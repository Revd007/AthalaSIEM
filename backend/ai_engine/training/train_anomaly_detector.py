import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ..models.anomaly_detector import VariationalAutoencoder
from sklearn.preprocessing import StandardScaler
import logging

class AnomalyDetectorTrainer:
    def __init__(self, input_dim: int, latent_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VariationalAutoencoder(input_dim, latent_dim).to(self.device)
        self.scaler = StandardScaler()
        
    def prepare_data(self, data: np.ndarray) -> DataLoader:
        scaled_data = self.scaler.fit_transform(data)
        tensor_data = torch.FloatTensor(scaled_data)
        dataset = TensorDataset(tensor_data)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self, train_loader: DataLoader, epochs: int = 100):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                # Forward pass
                recon_x, mu, log_var = self.model(x)
                
                # Compute loss
                recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader.dataset)
            logging.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
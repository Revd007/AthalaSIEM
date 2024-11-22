import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np

class ThreatDetector(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Multiple heads for different threat aspects
        self.threat_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5 severity levels
        )
        
        self.confidence_scorer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for interpretability
        self.attention = nn.MultiheadAttention(768, 8)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            pooled_output.unsqueeze(0),
            pooled_output.unsqueeze(0),
            pooled_output.unsqueeze(0)
        )
        
        # Get predictions from different heads
        threat_logits = self.threat_classifier(pooled_output)
        severity_logits = self.severity_classifier(pooled_output)
        confidence = self.confidence_scorer(pooled_output)
        
        return {
            'threat_logits': threat_logits,
            'severity_logits': severity_logits,
            'confidence': confidence,
            'attention_weights': attn_weights
        }
    
    def predict(self, text: str) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            outputs = self(inputs['input_ids'], inputs['attention_mask'])
            
            return {
                'threat_probs': F.softmax(outputs['threat_logits'], dim=1).numpy(),
                'severity_probs': F.softmax(outputs['severity_logits'], dim=1).numpy(),
                'confidence': outputs['confidence'].numpy(),
                'attention_weights': outputs['attention_weights'].numpy()
            }
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """Provide explanation for the prediction"""
        prediction = self.predict(text)
        tokens = self.tokenizer.tokenize(text)
        
        # Get attention weights for tokens
        attention = prediction['attention_weights'][0]  # First attention head
        token_importance = attention.mean(axis=0)
        
        # Get top contributing tokens
        top_k = 5
        top_indices = token_importance.argsort()[-top_k:][::-1]
        
        important_tokens = [(tokens[i], float(token_importance[i])) for i in top_indices]
        
        return {
            'prediction': prediction,
            'important_tokens': important_tokens,
            'explanation': self._generate_explanation(prediction, important_tokens)
        }
    
    def _generate_explanation(self, prediction: Dict[str, np.ndarray], important_tokens: List[tuple]) -> str:
        threat_level = prediction['threat_probs'].argmax()
        severity_level = prediction['severity_probs'].argmax()
        confidence = float(prediction['confidence'])
        
        explanation = (
            f"Threat Level: {threat_level} (Confidence: {confidence:.2f})\n"
            f"Severity Level: {severity_level}\n"
            f"Key Indicators:\n"
        )
        
        for token, importance in important_tokens:
            explanation += f"- {token}: {importance:.3f}\n"
            
        # Combine insights from multiple models for more robust detection
        pattern_insights = ""
        risk_insights = ""
        behavior_insights = ""
        
        # Get pattern recognition insights if available
        if hasattr(self, 'pattern_recognizer'):
            pattern_scores = self.pattern_recognizer(torch.tensor(prediction['attention_weights']))
            if pattern_scores['pattern_detected'].item() > 0.5:
                pattern_insights = "\nSuspicious patterns detected in text structure"
        
        # Get risk assessment if available 
        if hasattr(self, 'risk_assessor'):
            risk_factors = self.risk_assessor(torch.tensor(prediction['attention_weights']))
            if risk_factors['overall_risk'].item() > 0.7:
                risk_insights = f"\nHigh risk factors detected (Risk score: {risk_factors['overall_risk'].item():.2f})"
        
        # Get behavior analysis if available
        if hasattr(self, 'behavior_analyzer'):
            behavior = self.behavior_analyzer(torch.tensor(prediction['attention_weights']))
            if behavior['risk_score'].item() > 0.6:
                behavior_insights = f"\nConcerning behavior patterns identified (Score: {behavior['risk_score'].item():.2f})"
        
        # Combine all insights
        explanation += pattern_insights + risk_insights + behavior_insights
        
        return explanation
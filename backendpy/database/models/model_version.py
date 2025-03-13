from tortoise import fields
from tortoise.models import Model
from datetime import datetime
from typing import Dict, Any

class ModelVersion(Model):
    id = fields.IntField(pk=True)
    version = fields.CharField(max_length=50, unique=True)
    config = fields.JSONField()
    metrics = fields.JSONField()
    status = fields.CharField(max_length=20)  # 'active' or 'archived'
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "model_versions"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'version': self.version,
            'config': self.config,
            'metrics': self.metrics,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
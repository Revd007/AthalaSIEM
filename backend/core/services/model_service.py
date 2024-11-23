from typing import Dict, Any, List
import logging
from database.models import ModelVersion
from ai_engine.core.model_manager import ModelManager
from datetime import datetime

class ModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager()

    async def upgrade_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade model dengan konfigurasi baru"""
        try:
            self.logger.info("Starting model upgrade")
            
            # Simpan versi model saat ini
            current_version = await self._save_model_version()
            
            # Upgrade model
            upgrade_result = await self.model_manager.upgrade_model(config)
            
            # Simpan versi baru
            new_version = await ModelVersion.create(
                version=upgrade_result['version'],
                config=config,
                metrics=upgrade_result['metrics'],
                status='active',
                created_at=datetime.now()
            )
            
            return {
                'status': 'success',
                'previous_version': current_version.version,
                'new_version': new_version.version,
                'metrics': upgrade_result['metrics']
            }
            
        except Exception as e:
            self.logger.error(f"Model upgrade failed: {e}")
            raise

    async def rollback_model(self, version: str) -> Dict[str, Any]:
        """Rollback model ke versi sebelumnya"""
        try:
            self.logger.info(f"Rolling back to version {version}")
            
            # Dapatkan versi yang dituju
            target_version = await ModelVersion.get(version=version)
            if not target_version:
                raise ValueError(f"Version {version} not found")
            
            # Rollback model
            rollback_result = await self.model_manager.rollback_model(
                target_version.config
            )
            
            # Update status versi
            await self._update_version_status(version)
            
            return {
                'status': 'success',
                'current_version': version,
                'metrics': rollback_result['metrics']
            }
            
        except Exception as e:
            self.logger.error(f"Model rollback failed: {e}")
            raise

    async def get_all_versions(self) -> List[Dict[str, Any]]:
        """Dapatkan semua versi model"""
        try:
            versions = await ModelVersion.all()
            return [version.to_dict() for version in versions]
        except Exception as e:
            self.logger.error(f"Failed to get model versions: {e}")
            raise

    async def get_health_metrics(self) -> Dict[str, Any]:
        """Dapatkan metrics kesehatan model"""
        try:
            return await self.model_manager.get_health_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get health metrics: {e}")
            raise

    async def _save_model_version(self) -> ModelVersion:
        """Simpan versi model saat ini"""
        current_config = await self.model_manager.get_current_config()
        current_metrics = await self.model_manager.get_current_metrics()
        
        return await ModelVersion.create(
            version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=current_config,
            metrics=current_metrics,
            status='archived',
            created_at=datetime.now()
        )

    async def _update_version_status(self, active_version: str):
        """Update status semua versi"""
        await ModelVersion.filter(version=active_version).update(status='active')
        await ModelVersion.filter(version__not=active_version).update(status='archived')
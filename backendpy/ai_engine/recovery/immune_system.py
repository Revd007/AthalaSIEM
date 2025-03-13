from typing import Dict, Any, List
import logging
import asyncio
from datetime import datetime

class ImmuneSystem:
    """
    Simulates biological immune system behavior for cyber defense
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_cells = {}  # Store signatures of known threats
        self.active_responses = {}  # Track ongoing immune responses
        self.immunity_strength = {}  # Track immunity levels against threats
        
    async def isolate_infected_components(self, infected_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Isolate infected system components"""
        try:
            isolation_results = {
                'isolated_components': [],
                'failed_isolations': [],
                'timestamp': datetime.utcnow().isoformat()
            }

            for component in infected_components:
                try:
                    # Implement isolation logic
                    await self._isolate_component(component)
                    isolation_results['isolated_components'].append({
                        'component_id': component.get('id'),
                        'type': component.get('type'),
                        'status': 'isolated'
                    })
                except Exception as e:
                    self.logger.error(f"Failed to isolate component {component.get('id')}: {e}")
                    isolation_results['failed_isolations'].append({
                        'component_id': component.get('id'),
                        'error': str(e)
                    })

            return isolation_results

        except Exception as e:
            self.logger.error(f"Error in component isolation: {e}")
            raise

    async def strengthen_defenses(self, threat_type: Dict[str, Any]) -> Dict[str, Any]:
        """Strengthen system defenses against specific threats"""
        try:
            # Update immunity strength
            threat_id = threat_type.get('id', 'unknown')
            current_strength = self.immunity_strength.get(threat_id, 0)
            self.immunity_strength[threat_id] = min(current_strength + 0.2, 1.0)

            # Update memory cells with new threat signature
            self.memory_cells[threat_id] = {
                'signature': threat_type.get('characteristics', {}),
                'last_seen': datetime.utcnow().isoformat(),
                'immunity_level': self.immunity_strength[threat_id]
            }

            return {
                'threat_id': threat_id,
                'immunity_level': self.immunity_strength[threat_id],
                'defenses_updated': True,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error strengthening defenses: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get current immune system status"""
        try:
            return {
                'memory_cells': len(self.memory_cells),
                'active_responses': len(self.active_responses),
                'immunity_levels': self.immunity_strength,
                'system_health': await self._calculate_system_health(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting immune system status: {e}")
            raise

    async def _isolate_component(self, component: Dict[str, Any]) -> None:
        """Internal method to isolate a component"""
        try:
            component_id = component.get('id')
            if not component_id:
                raise ValueError("Component ID is required")

            # Add to active responses
            self.active_responses[component_id] = {
                'start_time': datetime.utcnow().isoformat(),
                'status': 'isolating',
                'type': component.get('type')
            }

            # Simulate isolation process
            await asyncio.sleep(0.1)  # Non-blocking delay

            self.active_responses[component_id]['status'] = 'isolated'

        except Exception as e:
            self.logger.error(f"Error in component isolation: {e}")
            raise

    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        try:
            if not self.immunity_strength:
                return 1.0

            # Average immunity strength
            avg_immunity = sum(self.immunity_strength.values()) / len(self.immunity_strength)
            
            # Factor in active threats
            active_threat_penalty = len(self.active_responses) * 0.1
            
            health_score = max(0.0, min(1.0, avg_immunity - active_threat_penalty))
            return round(health_score, 2)

        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
            return 0.0 
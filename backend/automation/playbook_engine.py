import yaml
import asyncio
from typing import Dict, Any, List
import logging
from datetime import datetime
from pathlib import Path
from automation.actions import SecurityActions
from database.models import PlaybookRun, Alert
from database.connection import get_db

class PlaybookEngine:
    def __init__(self):
        self.playbooks = self.load_playbooks()
        self.security_actions = SecurityActions(config={})  # Load from config
        self.running_playbooks: Dict[str, asyncio.Task] = {}

    def load_playbooks(self) -> Dict[str, Dict]:
        playbooks_dir = Path("backend/automation/playbooks")
        playbooks = {}
        
        for playbook_file in playbooks_dir.glob("*.yaml"):
            try:
                with open(playbook_file, 'r') as f:
                    playbook_data = yaml.safe_load(f)
                    playbooks[playbook_data['name']] = playbook_data
            except Exception as e:
                logging.error(f"Error loading playbook {playbook_file}: {e}")
        
        return playbooks

    async def execute_playbook(self, playbook_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a playbook with given context
        """
        playbook = self.playbooks.get(playbook_id)
        if not playbook:
            raise ValueError(f"Playbook {playbook_id} not found")

        # Create playbook run record
        run_record = PlaybookRun(
            alert_id=context.get('alert_id'),
            playbook_id=playbook_id,
            status='running',
            start_time=datetime.utcnow()
        )

        try:
            db = await anext(get_db())
            db.add(run_record)
            await db.commit()
            await db.refresh(run_record)

            # Execute playbook steps
            for step in playbook['steps']:
                result = await self.execute_step(step, context)
                context.update(result)

            # Update run record on success
            run_record.status = 'completed'
            run_record.result = context
            run_record.end_time = datetime.utcnow()

            return {
                'status': 'success',
                'playbook_id': playbook_id,
                'results': context
            }

        except Exception as e:
            logging.error(f"Error executing playbook {playbook_id}: {e}")
            run_record.status = 'failed'
            run_record.result = {'error': str(e)}
            run_record.end_time = datetime.utcnow()
            raise

        finally:
            db = await anext(get_db())
            db.add(run_record)
            await db.commit()

    async def execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step in the playbook
        """
        action = step['action']
        parameters = step.get('parameters', {})
        
        # Resolve any variables in parameters
        resolved_params = self.resolve_variables(parameters, context)
        
        if action == 'block_ip':
            return await self.security_actions.block_ip(resolved_params['ip'])
        elif action == 'quarantine_host':
            return await self.security_actions.quarantine_host(resolved_params['hostname'])
        elif action == 'send_alert':
            return await self.security_actions.send_alert(resolved_params)
        elif action == 'update_alert_status':
            return await self.update_alert_status(resolved_params['alert_id'], resolved_params['status'])
        else:
            raise ValueError(f"Unknown action: {action}")

    def resolve_variables(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve variables in parameters using context
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                var_name = value[2:-1]
                if var_name in context:
                    resolved[key] = context[var_name]
                else:
                    raise ValueError(f"Variable {var_name} not found in context")
            else:
                resolved[key] = value
        return resolved

    async def update_alert_status(self, alert_id: int, status: str) -> Dict[str, Any]:
        """
        Update alert status in database
        """
        db = await anext(get_db())
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.status = status
            await db.commit()
            return {'status': 'success', 'message': f'Alert status updated to {status}'}
        return {'status': 'error', 'message': 'Alert not found'}

    async def get_playbook_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get status of a running playbook
        """
        db = await anext(get_db())
        run = db.query(PlaybookRun).filter(PlaybookRun.id == run_id).first()
        if run:
            return {
                'status': run.status,
                'start_time': run.start_time,
                'end_time': run.end_time,
                'result': run.result
            }
        return {'status': 'not_found'}

    async def stop_playbook(self, run_id: str) -> Dict[str, Any]:
        """
        Stop a running playbook
        """
        if run_id in self.running_playbooks:
            self.running_playbooks[run_id].cancel()
            db = await anext(get_db())
            run = db.query(PlaybookRun).filter(PlaybookRun.id == run_id).first()
            if run:
                run.status = 'cancelled'
                run.end_time = datetime.utcnow()
                await db.commit()
            return {'status': 'cancelled'}
        return {'status': 'not_found'}
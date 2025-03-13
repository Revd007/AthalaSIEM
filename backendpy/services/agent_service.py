from asyncio.log import logger
from typing import List, Optional, Dict, Any
import uuid
import secrets
from datetime import datetime
from sqlalchemy import select, text, insert
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.agent import Agent, AgentStatus, AgentType, CollectorType
from schemas.agent import AgentCreate, AgentUpdate
import json
from sqlalchemy.dialects.postgresql import insert as pg_insert

class AgentService:
    async def create_agent(self, db: AsyncSession, agent_data: AgentCreate) -> Dict[str, Any]:
        """Create new agent"""
        try:
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            
            try:
                # Input: "windows" -> "windows_collector"
                agent_type_input = f"{agent_data.type.lower()}_collector"
                collector_type_input = agent_data.collector_type.lower()
                
                logger.info(f"Attempting to create agent with type: {agent_type_input}, collector: {collector_type_input}")
                
                # Create agent instance using raw SQL with named parameters
                stmt = text("""
                    INSERT INTO agents (
                        name, type, collector_type, status, ip_address, port,
                        use_ssl, api_key, collector_config, enabled_sources, filters,
                        created_at, updated_at
                    ) VALUES (
                        :name, :type::agent_type, :collector_type::collector_type, :status::agent_status,
                        :ip_address, :port, :use_ssl, :api_key, :config::jsonb, :sources::jsonb, :filters::jsonb,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    ) RETURNING *
                """)
                
                # Execute insert with parameters as dict
                params = {
                    "name": agent_data.name,
                    "type": agent_type_input,
                    "collector_type": collector_type_input,
                    "status": 'inactive',
                    "ip_address": agent_data.ip_address,
                    "port": int(agent_data.port),
                    "use_ssl": agent_data.use_ssl,
                    "api_key": api_key,
                    "config": json.dumps(agent_data.collector_config or {}),
                    "sources": json.dumps(agent_data.enabled_sources or []),
                    "filters": json.dumps({})
                }
                
                result = await db.execute(stmt, params)
                await db.commit()
                
                # Get the inserted row
                row = result.fetchone()
                if row is None:
                    raise ValueError("Failed to create agent")
                
                # Convert to dictionary and format response
                return {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.type.replace('_collector', ''),  # Remove _collector suffix for response
                    "collector_type": row.collector_type,
                    "status": row.status,
                    "ip_address": row.ip_address,
                    "port": str(row.port),  # Convert port to string for response
                    "use_ssl": row.use_ssl,
                    "api_key": row.api_key,
                    "collector_config": row.collector_config,
                    "enabled_sources": row.enabled_sources,
                    "filters": row.filters,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                    "last_heartbeat": row.last_heartbeat.isoformat() if row.last_heartbeat else None
                }
                
            except ValueError as e:
                raise ValueError(f"Invalid agent type or collector type: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            await db.rollback()
            raise

    async def get_agents(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all agents"""
        try:
            # Use raw SQL to get proper type casting
            query = text("""
                SELECT 
                    id, name, type, collector_type, status, ip_address, 
                    port, use_ssl, api_key, collector_config, enabled_sources, 
                    filters, last_heartbeat, created_at, updated_at
                FROM agents
                ORDER BY created_at DESC
                OFFSET :skip
                LIMIT :limit
            """)
            
            result = await db.execute(query, {"skip": skip, "limit": limit})
            rows = result.fetchall()
            
            return [
                {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.type.replace('_collector', ''),  # Remove _collector suffix for response
                    "collector_type": row.collector_type,
                    "status": row.status,
                    "ip_address": row.ip_address,
                    "port": str(row.port),  # Convert port to string for response
                    "use_ssl": row.use_ssl,
                    "api_key": row.api_key,
                    "collector_config": json.loads(row.collector_config) if row.collector_config else {},
                    "enabled_sources": json.loads(row.enabled_sources) if row.enabled_sources else [],
                    "filters": json.loads(row.filters) if row.filters else {},
                    "last_heartbeat": row.last_heartbeat.isoformat() if row.last_heartbeat else None,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat()
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error getting agents: {str(e)}")
            raise

    async def get_agent(self, db: AsyncSession, agent_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get agent by ID"""
        try:
            query = text("SELECT * FROM agents WHERE id = :agent_id")
            result = await db.execute(query, {"agent_id": agent_id})
            row = result.fetchone()
            
            if row is None:
                return None
                
            agent_dict = dict(row._mapping)
            # Parse JSON fields
            for field in ['collector_config', 'enabled_sources', 'filters']:
                if agent_dict.get(field):
                    agent_dict[field] = json.loads(agent_dict[field])
                    
            return agent_dict
            
        except Exception as e:
            logger.error(f"Error getting agent: {str(e)}")
            raise

    async def update_agent_status(self, db: AsyncSession, agent_id: uuid.UUID, status: str) -> Optional[Dict[str, Any]]:
        """Update agent status"""
        try:
            query = text("""
                UPDATE agents 
                SET status = :status::agent_status, 
                    last_heartbeat = CURRENT_TIMESTAMP
                WHERE id = :agent_id
                RETURNING *
            """)
            
            result = await db.execute(query, {
                "agent_id": agent_id,
                "status": status.lower()  # Convert to lowercase to match enum
            })
            
            await db.commit()
            
            row = result.fetchone()
            if row is None:
                return None
                
            agent_dict = dict(row._mapping)
            # Parse JSON fields
            for field in ['collector_config', 'enabled_sources', 'filters']:
                if agent_dict.get(field):
                    agent_dict[field] = json.loads(agent_dict[field])
                    
            return agent_dict
            
        except Exception as e:
            logger.error(f"Error updating agent status: {str(e)}")
            await db.rollback()
            raise
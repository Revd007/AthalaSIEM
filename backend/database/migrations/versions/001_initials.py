"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-03-20
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('severity', sa.Integer(), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('raw_data', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
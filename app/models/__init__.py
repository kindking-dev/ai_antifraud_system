"""
Model Registry.
Imports all models so Alembic can discover them for migrations.
"""

from app.models.db_entities import Base, TransactionLog

__all__ = ["Base", "TransactionLog"]

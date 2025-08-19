"""
Kumo RFM client for Dataset Director.
Uses Redis for persistence and KumoRFM LocalTable/LocalGraph for predictions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import redis
from dotenv import load_dotenv

# Import Kumo RFM components
try:
    from kumoai.experimental import rfm
    RFM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"KumoRFM not available: {e}")
    rfm = None
    RFM_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
KUMO_API_KEY = os.getenv("KUMO_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TEMP_DIR = Path("/tmp/vibe-data-director")
TEMP_DIR.mkdir(exist_ok=True)


class KumoClient:
    """Client for KumoRFM operations with Redis persistence."""

    def __init__(self):
        """Initialize KumoRFM client with Redis backend."""
        self.initialized = False

        # Initialize Redis
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✓ Redis connected")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory fallback: {e}")
            self.redis_client = None
            self._memory_store = {}  # Fallback to memory if Redis unavailable

        if not KUMO_API_KEY:
            logger.warning("KUMO_API_KEY not set, running in stub mode")
            return

        if RFM_AVAILABLE:
            try:
                # Initialize RFM with API key
                rfm.init(api_key=KUMO_API_KEY)
                self.initialized = True
                logger.info("✓ KumoRFM initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize KumoRFM: {e}")
                self.initialized = False
        else:
            logger.warning("KumoRFM not available, running in stub mode")
            self.initialized = False

    def _store_data(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """Store data in Redis or memory fallback."""
        try:
            if self.redis_client:
                json_data = json.dumps(data, default=str)
                self.redis_client.setex(key, ttl, json_data)
            else:
                self._memory_store[key] = data
            return True
        except Exception as e:
            logger.error(f"Failed to store data for {key}: {e}")
            return False

    def _get_data(self, key: str) -> Any:
        """Retrieve data from Redis or memory fallback."""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                return self._memory_store.get(key)
        except Exception as e:
            logger.error(f"Failed to get data for {key}: {e}")
            return None

    def _append_data(self, key: str, new_items: List[Dict], ttl: int = 3600) -> bool:
        """Append items to a list in Redis."""
        existing = self._get_data(key) or []
        existing.extend(new_items)
        return self._store_data(key, existing, ttl)

    def create_session_tables(
        self,
        session_id: str,
        specs: List[Dict[str, Any]],
        targets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store initial table data in Redis for a session.

        Args:
            session_id: Session ID for scoping
            specs: List of spec dictionaries
            targets: List of target dictionaries

        Returns:
            Dictionary with table metadata
        """
        tables = {}

        try:
            # Store specs in Redis
            specs_key = f"table:specs:{session_id}"
            if self._store_data(specs_key, specs, ttl=7200):
                tables[f"specs_{session_id}"] = {"count": len(specs), "stored": True}
                logger.info(f"Stored {len(specs)} specs in Redis")

            # Store targets in Redis
            targets_key = f"table:targets:{session_id}"
            if self._store_data(targets_key, targets, ttl=7200):
                tables[f"targets_{session_id}"] = {"count": len(targets), "stored": True}
                logger.info(f"Stored {len(targets)} targets in Redis")

            # Initialize empty samples list in Redis
            samples_key = f"table:samples:{session_id}"
            if self._store_data(samples_key, [], ttl=7200):
                tables[f"samples_{session_id}"] = {"count": 0, "stored": True}
                logger.info("Initialized samples table in Redis")

            return tables

        except Exception as e:
            logger.error(f"Failed to create session tables: {e}")
            return {}

    def add_samples(
        self,
        session_id: str,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add samples to Redis storage.

        Args:
            session_id: Session ID
            samples: List of sample dictionaries

        Returns:
            Metadata about the operation
        """
        if not samples:
            return {"count": 0, "stored": False}

        samples_key = f"table:samples:{session_id}"

        try:
            # Append to existing samples
            if self._append_data(samples_key, samples, ttl=7200):
                # Get updated count
                all_samples = self._get_data(samples_key) or []
                logger.info(f"Added {len(samples)} samples to Redis (total: {len(all_samples)})")
                return {"count": len(all_samples), "stored": True}
            else:
                return {"count": 0, "stored": False}

        except Exception as e:
            logger.error(f"Failed to add samples: {e}")
            return {"count": 0, "stored": False}

    def get_local_tables_for_prediction(
        self,
        session_id: str
    ) -> Optional[Dict[str, 'rfm.LocalTable']]:
        """
        Retrieve data from Redis and create LocalTables for RFM prediction.

        Args:
            session_id: Session ID

        Returns:
            Dictionary of LocalTables ready for RFM, or None if failed
        """
        if not self.initialized or not RFM_AVAILABLE:
            logger.warning("RFM not available for predictions")
            return None

        try:
            tables = {}

            # Get samples from Redis
            samples_key = f"table:samples:{session_id}"
            samples_data = self._get_data(samples_key)
            if samples_data:
                samples_df = pd.DataFrame(samples_data)
                if not samples_df.empty:
                    # Convert timestamp if needed
                    if 'ts' in samples_df.columns:
                        samples_df['ts'] = pd.to_datetime(samples_df['ts'])

                    samples_table = rfm.LocalTable(
                        samples_df,
                        name=f"samples_{session_id}",
                        primary_key="sample_id",
                        time_column="ts"
                    ).infer_metadata()
                    tables[f"samples_{session_id}"] = samples_table
                    logger.info(f"Created LocalTable for samples with {len(samples_df)} rows")

            # Get specs from Redis
            specs_key = f"table:specs:{session_id}"
            specs_data = self._get_data(specs_key)
            if specs_data:
                specs_df = pd.DataFrame(specs_data)
                if not specs_df.empty:
                    specs_table = rfm.LocalTable(
                        specs_df,
                        name=f"specs_{session_id}",
                        primary_key="spec_id"
                    ).infer_metadata()
                    tables[f"specs_{session_id}"] = specs_table
                    logger.info(f"Created LocalTable for specs with {len(specs_df)} rows")

            # Get targets from Redis
            targets_key = f"table:targets:{session_id}"
            targets_data = self._get_data(targets_key)
            if targets_data:
                targets_df = pd.DataFrame(targets_data)
                if not targets_df.empty:
                    # Targets table doesn't need a primary key - it's linked via spec_id
                    targets_table = rfm.LocalTable(
                        targets_df,
                        name=f"targets_{session_id}"
                    ).infer_metadata()
                    # Set spec_id as ID type for linking
                    if 'spec_id' in targets_table.columns:
                        targets_table['spec_id'].stype = "ID"
                    tables[f"targets_{session_id}"] = targets_table
                    logger.info(f"Created LocalTable for targets with {len(targets_df)} rows")

            return tables if tables else None

        except Exception as e:
            logger.error(f"Failed to create LocalTables from Redis data: {e}")
            return None

    def get_table_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about session tables from Redis.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with table statistics
        """
        stats = {
            "samples_count": 0,
            "specs_count": 0,
            "targets_count": 0,
            "tables": []
        }

        # Get samples count
        samples_data = self._get_data(f"table:samples:{session_id}")
        if samples_data:
            stats["samples_count"] = len(samples_data)
            stats["tables"].append({
                "name": f"samples_{session_id}",
                "rows": len(samples_data),
                "stored_in": "redis" if self.redis_client else "memory"
            })

        # Get specs count
        specs_data = self._get_data(f"table:specs:{session_id}")
        if specs_data:
            stats["specs_count"] = len(specs_data)
            stats["tables"].append({
                "name": f"specs_{session_id}",
                "rows": len(specs_data),
                "stored_in": "redis" if self.redis_client else "memory"
            })

        # Get targets count
        targets_data = self._get_data(f"table:targets:{session_id}")
        if targets_data:
            stats["targets_count"] = len(targets_data)
            stats["tables"].append({
                "name": f"targets_{session_id}",
                "rows": len(targets_data),
                "stored_in": "redis" if self.redis_client else "memory"
            })

        return stats

    def clear_session_data(self, session_id: str) -> bool:
        """
        Clear all data for a session from Redis.

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        try:
            if self.redis_client:
                keys = [
                    f"table:samples:{session_id}",
                    f"table:specs:{session_id}",
                    f"table:targets:{session_id}"
                ]
                for key in keys:
                    self.redis_client.delete(key)
            else:
                # Clear from memory store
                keys_to_delete = [k for k in self._memory_store.keys() if session_id in k]
                for key in keys_to_delete:
                    del self._memory_store[key]

            logger.info(f"Cleared data for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session data: {e}")
            return False


# Global client instance
_client = None

def get_kumo_client() -> KumoClient:
    """Get or create the global KumoClient instance."""
    global _client
    if _client is None:
        _client = KumoClient()
    return _client

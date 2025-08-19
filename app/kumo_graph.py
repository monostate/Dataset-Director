"""
Kumo RFM Graph builder for Dataset Director.
Builds LocalGraph from LocalTables for KumoRFM predictions.
"""

import logging
from typing import Dict, Optional

# Import Kumo RFM components
try:
    from kumoai.experimental import rfm
    RFM_AVAILABLE = True
except ImportError:
    logging.warning("KumoRFM not available, using stub mode")
    RFM_AVAILABLE = False
    rfm = None

logger = logging.getLogger(__name__)


def build_graph(
    local_tables: Dict[str, 'rfm.LocalTable']
) -> Optional['rfm.LocalGraph']:
    """
    Build a LocalGraph from LocalTables for RFM predictions.

    Args:
        local_tables: Dictionary of table names to LocalTable objects

    Returns:
        LocalGraph if successful, None otherwise
    """
    if not RFM_AVAILABLE or not local_tables:
        logger.warning("Cannot build graph: RFM not available or no tables")
        return None

    try:
        # Extract the tables
        table_list = list(local_tables.values())

        # Create LocalGraph
        graph = rfm.LocalGraph(tables=table_list)

        # Find samples, specs, and targets tables
        samples_table = None
        specs_table = None
        targets_table = None

        for name, table in local_tables.items():
            if "samples" in name:
                samples_table = table
            elif "specs" in name:
                specs_table = table
            elif "targets" in name:
                targets_table = table

        # Add links between tables
        if samples_table and specs_table:
            # Link samples to specs via spec_id
            try:
                graph.link(
                    src_table=samples_table.name,
                    fkey="spec_id",
                    dst_table=specs_table.name
                )
                logger.info(f"Linked {samples_table.name} to {specs_table.name}")
            except Exception as e:
                logger.warning(f"Could not link samples to specs: {e}")

        if targets_table and specs_table:
            # Link targets to specs via spec_id
            try:
                graph.link(
                    src_table=targets_table.name,
                    fkey="spec_id",
                    dst_table=specs_table.name
                )
                logger.info(f"Linked {targets_table.name} to {specs_table.name}")
            except Exception as e:
                logger.warning(f"Could not link targets to specs: {e}")

        logger.info(f"Built LocalGraph with {len(table_list)} tables")
        return graph

    except Exception as e:
        logger.error(f"Failed to build LocalGraph: {e}")
        return None


def validate_graph(graph: 'rfm.LocalGraph') -> bool:
    """
    Validate that the LocalGraph is properly configured.

    Args:
        graph: LocalGraph to validate

    Returns:
        True if valid, False otherwise
    """
    if not RFM_AVAILABLE or graph is None:
        return False

    try:
        # Check that graph has tables
        if not hasattr(graph, 'tables') or len(graph.tables) == 0:
            logger.error("Graph has no tables")
            return False

        # Check for required tables
        has_samples = any("samples" in t.name for t in graph.tables.values())
        has_specs = any("specs" in t.name for t in graph.tables.values())

        if not has_samples:
            logger.warning("Graph missing samples table")
        if not has_specs:
            logger.warning("Graph missing specs table")

        logger.info("Graph validation passed")
        return True

    except Exception as e:
        logger.error(f"Graph validation failed: {e}")
        return False


def generate_spec_id(
    class_name: str,
    style: str,
    negation: bool
) -> str:
    """
    Generate a unique spec ID.

    Args:
        class_name: Class name
        style: Style (formal/casual)
        negation: Whether negation is included

    Returns:
        Unique spec ID
    """
    negation_flag = "1" if negation else "0"
    return f"{class_name}|{style}|{negation_flag}"

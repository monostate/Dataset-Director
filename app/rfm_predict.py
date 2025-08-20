"""
RFM Prediction module for Dataset Director.
Builds PQL queries and runs predictions via KumoRFM.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv

# Import Kumo SDK with proper package
try:
    from kumoai import Graph, PredictiveQuery
    from kumoai.experimental.rfm import KumoRFM  # Correct import path!
    KUMO_AVAILABLE = True
    RFM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Kumo SDK/RFM import issue: {e}")
    KUMO_AVAILABLE = False
    RFM_AVAILABLE = False
    KumoRFM = None
    Graph = object
    PredictiveQuery = None

load_dotenv()
logger = logging.getLogger(__name__)


class PQLType(Enum):
    """Types of PQL queries."""
    COVERAGE = "coverage"
    SPECS = "specs"
    QUALITY = "quality"


# PQL Templates - adjusted for KumoRFM syntax
PQL_TEMPLATES = {
    # Use canonical table names and PQL syntax with FOR on a non-PK column
    PQLType.COVERAGE: """PREDICT COUNT(samples.*, 0, 10, minutes)
FOR specs.class = '{class_name}'""",

    # Non-temporal spec recommendation to avoid time-column requirement on specs
    PQLType.SPECS: """PREDICT specs.spec_id
FOR classes.class = '{class_name}'""",

    PQLType.QUALITY: """PREDICT AVG(runs.metric_f1, 0, 1, minutes)
FOR specs.spec_id = '{spec_id}'"""
}


def get_coverage_pql(class_name: str) -> str:
    """
    Get PQL for coverage prediction.

    Args:
        class_name: The class to predict coverage for

    Returns:
        PQL query string
    """
    # For KumoRFM, we format the query directly with the class name
    pql = PQL_TEMPLATES[PQLType.COVERAGE].format(class_name=class_name)
    return pql


def get_specs_pql(class_name: str) -> str:
    """
    Get PQL for next specs prediction.

    Args:
        class_name: The class to get specs for

    Returns:
        PQL query string
    """
    pql = PQL_TEMPLATES[PQLType.SPECS].format(class_name=class_name)
    return pql


def get_quality_pql(spec_id: str) -> str:
    """
    Get PQL for quality prediction (future enhancement).

    Args:
        spec_id: The spec to predict quality for

    Returns:
        PQL query string
    """
    pql = PQL_TEMPLATES[PQLType.QUALITY].format(spec_id=spec_id)
    return pql


def run_pql_with_rfm(
    graph: Graph,
    pql: str,
    evaluate: bool = False
) -> Any:
    """
    Run a PQL query using KumoRFM.

    Args:
        graph: Kumo Graph object
        pql: PQL query string
        evaluate: Whether to run in evaluation mode

    Returns:
        Prediction result or evaluation metrics
    """
    if not KUMO_AVAILABLE or not RFM_AVAILABLE or not KumoRFM:
        logger.warning("KumoRFM not available, returning stub result")
        return _get_stub_result(pql, {})

    try:
        # Initialize KumoRFM
        rfm = KumoRFM(graph=graph)

        if evaluate:
            # Run evaluation
            pql_eval = f"EVALUATE {pql}"
            metrics = rfm.evaluate(pql_eval)
            logger.info(f"RFM evaluation metrics: {metrics}")
            return metrics
        else:
            # Run prediction - returns DataFrame
            result_df = rfm.predict(pql)
            logger.info(f"RFM prediction result: {result_df}")

            # Process result based on query type
            return _process_rfm_result(pql, result_df)

    except Exception as e:
        logger.error(f"Failed to run RFM prediction: {e}")
        return _get_stub_result(pql, {})


def run_pql_with_predictive_query(
    graph: Graph,
    pql: str,
    non_blocking: bool = False
) -> Any:
    """
    Run a PQL query using PredictiveQuery (for training models).

    Args:
        graph: Kumo Graph object
        pql: PQL query string
        non_blocking: Whether to run asynchronously

    Returns:
        PredictiveQuery object or training results
    """
    if not KUMO_AVAILABLE or not PredictiveQuery:
        logger.warning("PredictiveQuery not available, returning stub")
        return None

    try:
        # Create PredictiveQuery
        pquery = PredictiveQuery(graph=graph, query=pql)

        # Validate the query
        pquery.validate(verbose=True)

        # For predictions, we would normally:
        # 1. Generate training table
        # 2. Create model plan
        # 3. Train with Trainer
        # 4. Generate predictions

        # For now, return the pquery object
        return pquery

    except Exception as e:
        logger.error(f"Failed to create PredictiveQuery: {e}")
        return None


def run_pql_prediction(
    graph: Graph,
    pql: str,
    parameters: Dict[str, Any],
    use_rfm: bool = True
) -> Any:
    """
    Run a PQL prediction using either KumoRFM or PredictiveQuery.

    Args:
        graph: Kumo Graph object
        pql: PQL query string
        parameters: Parameters to bind to the query (for compatibility)
        use_rfm: Whether to use KumoRFM (True) or PredictiveQuery (False)

    Returns:
        Prediction result (format depends on query type)
    """
    logger.info("Running PQL prediction")

    # Note: PQL already has literal values embedded (no placeholders)
    # Parameters are ignored for KumoRFM since we use literal values

    if use_rfm:
        return run_pql_with_rfm(graph, pql, evaluate=False)
    else:
        return run_pql_with_predictive_query(graph, pql)


def run_pql_with_predictive_query_and_anchor(
    graph: Graph,
    pql: str,
    anchor_time: Optional[datetime],
) -> Any:
    """
    Run PQL using PredictiveQuery and set a specific anchor_time when possible.
    Falls back to run_pql_with_rfm on failure.
    """
    if not KUMO_AVAILABLE or not PredictiveQuery:
        logger.warning("PredictiveQuery not available; falling back to RFM")
        return run_pql_with_rfm(graph, pql, evaluate=False)

    try:
        pquery = PredictiveQuery(graph=graph, query=pql)
        try:
            pquery.validate(verbose=True)
        except Exception as ve:
            logger.warning(f"PredictiveQuery validate() warning: {ve}")

        plan = None
        try:
            plan = pquery.suggest_prediction_table_plan()
        except Exception as pe:
            logger.warning(f"suggest_prediction_table_plan unavailable: {pe}")

        # Set anchor_time if the plan supports it
        if plan is not None and anchor_time is not None:
            try:
                iso_anchor = anchor_time.isoformat()
                setattr(plan, 'anchor_time', iso_anchor)
            except Exception as ae:
                logger.warning(f"Failed to set anchor_time on plan: {ae}")

        # Generate prediction table
        try:
            if plan is not None:
                pred_table = pquery.generate_prediction_table(plan)
            else:
                pred_table = pquery.generate_prediction_table()
        except Exception as ge:
            logger.warning(f"generate_prediction_table failed ({ge}), retry without plan")
            pred_table = pquery.generate_prediction_table()

        # Predict
        try:
            result_df = pquery.predict(pred_table)
        except TypeError:
            result_df = pquery.predict()

        return _process_rfm_result(pql, result_df)
    except Exception as e:
        logger.error(f"PredictiveQuery with anchor failed: {e}")
        return run_pql_with_rfm(graph, pql, evaluate=False)


def _process_rfm_result(pql: str, result_df: Any) -> Any:
    """
    Process RFM result DataFrame to match API contract.

    Args:
        pql: PQL query string
        result_df: Result DataFrame from RFM

    Returns:
        Processed result matching endpoint expectations
    """
    import pandas as pd

    if result_df is None or (isinstance(result_df, pd.DataFrame) and result_df.empty):
        return _get_stub_result(pql, {})

    # Determine query type and extract appropriate value
    if "COUNT(samples.*" in pql:
        # Coverage query - robustly extract a numeric value from the first row
        if isinstance(result_df, pd.DataFrame) and len(result_df) > 0:
            row0 = result_df.iloc[0]
            # 1) Prefer numeric columns
            for col in result_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        val = row0[col]
                        if not pd.isna(val):
                            return int(val)
                except Exception:
                    continue
            # 2) Try to coerce any cell to float
            for col in result_df.columns:
                try:
                    val = float(row0[col])
                    if not pd.isna(val):
                        return int(val)
                except Exception:
                    continue
            return 0
        return 0

    elif ("LIST_DISTINCT(specs.spec_id" in pql) or ("PREDICT specs.spec_id" in pql):
        # Specs query - extract list of spec IDs
        if isinstance(result_df, pd.DataFrame) and len(result_df) > 0:
            # Try typical columns first
            for candidate in ["spec_id", "specs.spec_id", "value"]:
                if candidate in result_df.columns:
                    try:
                        vals = result_df[candidate].dropna().tolist()
                        # Flatten if nested lists
                        out = []
                        for v in vals:
                            if isinstance(v, list):
                                out.extend(v)
                            else:
                                out.append(v)
                        return [str(x) for x in out]
                    except Exception:
                        pass
            # Fallback: use first column
            try:
                return [str(x) for x in result_df.iloc[:, 0].dropna().tolist()]
            except Exception:
                return []
        return []

    elif "AVG(runs.metric_f1" in pql:
        # Quality query - extract average as float
        if isinstance(result_df, pd.DataFrame) and len(result_df) > 0:
            value = result_df.iloc[0, 1] if result_df.shape[1] > 1 else result_df.iloc[0, 0]
            return float(value) if not pd.isna(value) else 0.0
        return 0.0

    else:
        # Unknown query type - return raw result
        return result_df


def _get_stub_result(pql: str, parameters: Dict[str, Any]) -> Any:
    """
    Get stub result for testing without actual RFM.

    Args:
        pql: PQL query string
        parameters: Query parameters

    Returns:
        Stub result based on query type
    """
    # Determine query type from PQL
    if "COUNT(samples.*" in pql:
        # Coverage query - return predicted count
        return 0  # Stub: no predicted samples yet

    elif "LIST_DISTINCT(specs.spec_id" in pql:
        # Specs query - return list of spec IDs
        return []  # Stub: empty list for now

    elif "AVG(runs.metric_f1" in pql:
        # Quality query - return predicted F1 score
        return 0.0  # Stub: no quality data yet

    else:
        # Unknown query type
        return None


def batch_run_predictions(
    graph: Graph,
    queries: List[Dict[str, Any]],
    use_rfm: bool = True
) -> List[Dict[str, Any]]:
    """
    Run multiple PQL predictions in batch.

    Args:
        graph: Kumo Graph object
        queries: List of query definitions with pql and parameters
        use_rfm: Whether to use KumoRFM or PredictiveQuery

    Returns:
        List of results corresponding to each query
    """
    results = []

    for query_def in queries:
        pql = query_def.get("pql", "")
        parameters = query_def.get("parameters", {})

        try:
            result = run_pql_prediction(graph, pql, parameters, use_rfm=use_rfm)
            results.append({
                "success": True,
                "result": result,
                "query": query_def
            })
        except Exception as e:
            logger.error(f"Failed to run query: {e}")
            results.append({
                "success": False,
                "error": str(e),
                "query": query_def
            })

    return results


def validate_pql(pql: str) -> bool:
    """
    Validate PQL syntax (basic validation).

    Args:
        pql: PQL query string

    Returns:
        True if valid, False otherwise
    """
    required_keywords = ["PREDICT", "FOR"]

    for keyword in required_keywords:
        if keyword not in pql:
            logger.error(f"PQL missing required keyword: {keyword}")
            return False

    # Additional validation for KumoRFM
    if "EVALUATE" in pql and "PREDICT" not in pql.replace("EVALUATE", ""):
        logger.error("EVALUATE must be followed by PREDICT")
        return False

    return True


def get_available_pql_templates() -> Dict[str, str]:
    """
    Get all available PQL templates.

    Returns:
        Dictionary of PQL type to template string
    """
    return {
        pql_type.value: template
        for pql_type, template in PQL_TEMPLATES.items()
    }


class PredictionCache:
    """Simple in-memory cache for prediction results."""

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache with TTL.

        Args:
            ttl_seconds: Time to live for cached results
        """
        self.cache: Dict[str, Any] = {}
        self.ttl_seconds = ttl_seconds
        self.timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        import time

        if key in self.cache:
            # Check if expired
            if time.time() - self.timestamps.get(key, 0) < self.ttl_seconds:
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.timestamps[key]

        return None

    def set(self, key: str, value: Any) -> None:
        """Cache a result."""
        import time
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.timestamps.clear()


# Global cache instance
prediction_cache = PredictionCache()

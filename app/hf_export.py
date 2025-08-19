"""
HuggingFace Dataset export module for Dataset Director.
Handles creating and uploading datasets to HuggingFace Hub.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

# Import HuggingFace Hub
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_AVAILABLE = True
except ImportError:
    logging.warning("HuggingFace Hub not installed, running in stub mode")
    HF_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")


def export_to_huggingface(
    session_id: str,
    repo_id: str,
    samples: List[Dict[str, Any]]
) -> str:
    """
    Export samples to a HuggingFace dataset repository.

    Args:
        session_id: Session ID for tracking
        repo_id: HuggingFace repo ID (format: username/dataset-name)
        samples: List of sample dictionaries

    Returns:
        URL to the created/updated repository
    """
    logger.info(f"Exporting {len(samples)} samples to HF repo {repo_id}")

    if not HF_AVAILABLE:
        logger.warning("HuggingFace Hub not available, returning stub URL")
        return f"https://huggingface.co/datasets/{repo_id}"

    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set, attempting anonymous upload")

    try:
        # Initialize HF API and get the actual HF username if needed
        api = HfApi(token=HF_TOKEN)

        # If repo_id doesn't contain a slash, prepend with username
        if "/" not in repo_id:
            try:
                whoami_info = api.whoami()
                username = whoami_info["name"]
                repo_id = f"{username}/{repo_id}"
                logger.info(f"Using full repo_id: {repo_id}")
            except Exception as e:
                logger.warning(f"Could not get HF username: {e}")
                # Fall back to original repo_id

        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=HF_TOKEN,
                repo_type="dataset",
                exist_ok=True,
                private=False  # Set to True for private datasets
            )
            logger.info(f"Created/verified HF repository: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            # Continue anyway - repo might already exist

        # Prepare dataset
        df = pd.DataFrame(samples)

        # Clean up columns for export
        export_columns = ["text", "class", "style", "negation", "source", "spec_id"]
        df = df[export_columns]

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save as CSV (primary format)
            csv_path = tmp_path / "dataset.csv"
            df.to_csv(csv_path, index=False)

            # Save as Parquet (alternative format)
            parquet_path = tmp_path / "dataset.parquet"
            df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

            # Save as JSONL (for easy streaming)
            jsonl_path = tmp_path / "dataset.jsonl"
            df.to_json(jsonl_path, orient="records", lines=True)

            # Create dataset card
            dataset_card = create_dataset_card(
                session_id=session_id,
                repo_id=repo_id,
                num_samples=len(samples),
                classes=df["class"].unique().tolist(),
                styles=df["style"].unique().tolist()
            )
            card_path = tmp_path / "README.md"
            card_path.write_text(dataset_card)

            # Upload files
            for file_path in [csv_path, parquet_path, jsonl_path, card_path]:
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"data/{file_path.name}" if file_path.name != "README.md" else "README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                logger.info(f"Uploaded {file_path.name} to HF")

        # Return repository URL
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Successfully exported to: {repo_url}")
        return repo_url

    except Exception as e:
        logger.error(f"Failed to export to HuggingFace: {e}")
        raise


def create_dataset_card(
    session_id: str,
    repo_id: str,
    num_samples: int,
    classes: List[str],
    styles: List[str]
) -> str:
    """
    Create a dataset card (README.md) for the HuggingFace repository.

    Args:
        session_id: Session ID for tracking
        repo_id: Repository ID
        num_samples: Number of samples in dataset
        classes: List of unique classes
        styles: List of unique styles

    Returns:
        Dataset card markdown content
    """
    timestamp = datetime.utcnow().isoformat()

    card_content = f"""---
license: mit
task_categories:
- text-classification
language:
- en
size_categories:
- n<1K
tags:
- synthetic
- vibe-data-director
---

# Dataset Card for {repo_id}

## Dataset Description

This dataset was generated using Vibe Data Director, a tool for creating and curating text classification datasets.

### Dataset Summary

- **Session ID**: `{session_id}`
- **Generated**: {timestamp}
- **Total Samples**: {num_samples}
- **Classes**: {', '.join(classes)}
- **Styles**: {', '.join(styles)}

## Dataset Structure

### Data Fields

- `text` (string): The text content of the sample
- `class` (string): The classification label
- `style` (string): The style variant applied
- `negation` (boolean): Whether negation is applied
- `source` (string): Source of the sample (seed/hf/synthetic)
- `spec_id` (string): Unique specification identifier

### Data Splits

This dataset contains a single split with all samples.

## Dataset Creation

### Curation Rationale

This dataset was created to provide targeted training data for text classification tasks.

### Source Data

The data includes:
- Seed samples uploaded by users
- Potentially augmented samples from HuggingFace
- Synthetic samples (if applicable)

## Considerations for Using the Data

### Social Impact of Dataset

This is a small, curated dataset intended for research and development purposes.

### Discussion of Biases

As with any curated dataset, there may be inherent biases in the selection and labeling of samples.

### Other Known Limitations

- Limited to {num_samples} samples
- May not represent full diversity of real-world data

## Additional Information

### Dataset Curators

Created using Vibe Data Director

### Licensing Information

MIT License

### Citation Information

```bibtex
@misc{{vibe_data_director_{session_id},
  title={{Dataset created with Vibe Data Director}},
  author={{Vibe Data Director}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

### Contributions

Generated automatically by Vibe Data Director (Session: {session_id})
"""

    return card_content


def validate_repo_id(repo_id: str) -> bool:
    """
    Validate HuggingFace repository ID format.

    Args:
        repo_id: Repository ID to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    # Pattern: username/repo-name
    pattern = r"^[a-zA-Z0-9-_]+/[a-zA-Z0-9-_]+$"

    if not re.match(pattern, repo_id):
        logger.error(f"Invalid repo_id format: {repo_id}")
        return False

    return True


def download_from_huggingface(
    repo_id: str,
    file_name: str = "dataset.csv"
) -> pd.DataFrame:
    """
    Download a dataset from HuggingFace (helper function).

    Args:
        repo_id: HuggingFace repository ID
        file_name: Name of the file to download

    Returns:
        DataFrame with the dataset
    """
    if not HF_AVAILABLE:
        logger.warning("HuggingFace Hub not available")
        return pd.DataFrame()

    try:
        from huggingface_hub import hf_hub_download

        # Download file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"data/{file_name}",
            repo_type="dataset",
            token=HF_TOKEN
        )

        # Read based on file type
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_name.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")

        logger.info(f"Downloaded {len(df)} samples from {repo_id}")
        return df

    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        return pd.DataFrame()


def list_dataset_files(repo_id: str) -> List[str]:
    """
    List files in a HuggingFace dataset repository.

    Args:
        repo_id: Repository ID

    Returns:
        List of file paths in the repository
    """
    if not HF_AVAILABLE:
        return []

    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN
        )

        return list(files)

    except Exception as e:
        logger.error(f"Failed to list repository files: {e}")
        return []

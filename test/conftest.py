import os
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Keep integration tests opt-in for the default `pytest test/` run."""
    if os.environ.get("RUN_INTEGRATION_TESTS") == "1":
        return False
    path = Path(str(collection_path))
    return "integration" in path.parts and path.name.startswith("test_")

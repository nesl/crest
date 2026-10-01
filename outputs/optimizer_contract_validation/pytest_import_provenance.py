"""Read pytest's imported module provenance without modifying the test suite."""
from pathlib import Path
import sys
import pytest

root = Path.cwd().resolve()
class CheckoutProvenance:
    def pytest_collection_finish(self, session):
        for module, expected in (("crest", root / "src/crest/__init__.py"),
                                 ("estimate", root / "analysis_scripts/llm_token_cost/estimate.py")):
            actual = Path(sys.modules[module].__file__).resolve()
            assert actual == expected.resolve(), (module, actual, expected)
            print(f"IMPORT PROVENANCE {module}: {actual}")

raise SystemExit(pytest.main([
    "test/test_optimizer_contract.py", "analysis_scripts/llm_token_cost/test_estimate.py",
    "--collect-only", "-q", "-p", "no:cacheprovider",
], plugins=[CheckoutProvenance()]))

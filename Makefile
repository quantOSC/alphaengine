PYTHON ?= python

.PHONY: lint typecheck test goldens golden-wheel version-sync engine-version-check no-engine-copies-check gateway-doc-check boundary-check docs-check check

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

typecheck:
	$(PYTHON) -m mypy src

version-sync:
	$(PYTHON) scripts/check_version_sync.py

engine-version-check:
	$(PYTHON) scripts/check_engine_version.py

no-engine-copies-check:
	$(PYTHON) scripts/check_no_engine_copies.py

gateway-doc-check:
	$(PYTHON) scripts/gen_gateway_doc.py --check

boundary-check:
	$(PYTHON) scripts/check_public_boundary.py

docs-check:
	$(PYTHON) scripts/gen_docs.py --check

test:
	$(PYTHON) -m pytest

goldens:
	$(PYTHON) -m pytest tests -m golden -q

golden-wheel:
	$(PYTHON) scripts/golden_wheel.py

check: version-sync engine-version-check no-engine-copies-check gateway-doc-check boundary-check docs-check lint typecheck test golden-wheel

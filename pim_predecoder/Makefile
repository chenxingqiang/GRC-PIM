.PHONY: test test-fast test-rtl install-dev

REPO_ROOT := $(abspath ..)
PY ?= python3
export PATH := /opt/homebrew/bin:/usr/local/bin:$(PATH)

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

test-fast:
	cd $(REPO_ROOT) && $(PY) -m pytest pim_predecoder/tests -m "not rtl" -v

test: test-fast
	cd $(REPO_ROOT) && $(PY) -m pytest pim_predecoder/tests -m rtl -v

test-rtl:
	cd $(REPO_ROOT) && $(PY) -m pytest pim_predecoder/tests -m rtl -v

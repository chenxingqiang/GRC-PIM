# AGENTS.md

## Cursor Cloud specific instructions

This repo hosts two related Python research/reproducibility subprojects. Neither is a
service — both are CLI + pytest workflows with **no web server, database, or long-running dev
process**. Standard commands live in the READMEs and `pim_predecoder/Makefile`; prefer those.

- **`pim_predecoder/` (GRC-PIM):** GF(3) NB-LDPC + multi-cycle PIM ECC simulation (base stack).
- **`cim_resldpc/` (CiM-ResLDPC):** residual-error-calibrated edge RAG / information-retrieval
  built on top of `pim_predecoder`. Only present on the `cim-resldpc` feature branch.

### Shared environment caveats

- **Run everything from the repository root.** Modules are invoked as
  `python3 -m pim_predecoder.<...>` or `python3 -m cim_resldpc.<...>` and expect CWD = repo root.
- **Python version:** `pyproject.toml` pins `>=3.14,<3.15`, but the VM has Python 3.12 and
  everything works on it. Do not try to install Python 3.14.
- **Dependencies install into the user site with `--break-system-packages`** (system Python is
  externally managed). The startup update script does this and is idempotent. Console scripts
  (`pytest`, `torchrun`, `transformers`, etc.) land in `~/.local/bin`, which is **not on PATH** —
  invoke tools as modules, e.g. `python3 -m pytest`, not bare `pytest`.
- **No linter/formatter is configured** (no ruff/flake8/black/mypy config anywhere).
- **Submodule `deps/NBLDPC_PIM`** supplies the GF(3) H/G parity-check matrices required by both
  subprojects; it is already checked out on the VM.

### pim_predecoder (GRC-PIM)

- Fast tests: `python3 -m pytest pim_predecoder/tests -m "not rtl"` (or `make -C pim_predecoder
  test-fast`). The `-m rtl` Verilog cosim tests need `iverilog` + `vvp` (Icarus Verilog), which
  are **not installed** and are deselected by the fast suite.
- Experiments (`pim_predecoder.experiments.gemv` / `.ablation`) train a small INT8 GEMV
  calibrator on CPU, write JSON to `pim_predecoder/results/`, and cache `.pt` checkpoints in
  `pim_predecoder/_cache/` (reused unless `--force_train`).

### cim_resldpc (CiM-ResLDPC)

- Depends on `pim_predecoder` plus IR extras in `cim_resldpc/requirements-ir.txt`
  (`scikit-learn`, `sentence-transformers`, `transformers`, `datasets`). The update script
  installs these when the file is present.
- `scikit-learn` is **required** for the default `tfidf_svd` encoder (v1.2 extended eval), not
  just the optional HF path — without it several `test_extended_v12` tests fail.
- Tests: `python3 -m pytest cim_resldpc/tests -v`. HF-model tests use `pytest.importorskip` and
  download small models (e.g. `google/flan-t5-small`) from the HF hub on first run, so they need
  network access; a couple more skip when an `emb_cache` `.npz` fixture is absent.
- End-to-end eval (hello-world): `python3 -m cim_resldpc.experiments.run_sigir_eval --out_json
  <path>`. `results/*.json` are tracked bundled artifacts — write experiment output to a temp
  path (or clean up) rather than overwriting them.

### Paper (`docs/paper/`)

- The SIGIR-AP paper draft is `docs/paper/latex/main.tex` (ACM `acmart`, `sigconf`). Build with
  `latexmk -pdf -interaction=nonstopmode main` from `docs/paper/latex/` (equivalent to the
  header's `pdflatex; bibtex; pdflatex; pdflatex`). It compiles to a clean 3-page PDF with all
  14 `references.bib` entries resolved.
- **The LaTeX toolchain is a system dependency and is NOT installed by the update script.** To
  build the paper you need TeX Live with the `acmart` class + `ACM-Reference-Format.bst`:
  `texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-publishers
  texlive-bibtex-extra latexmk`, plus the scalable Type1 fonts `cm-super` and `lmodern`.
  Without `cm-super`/`lmodern`, `acmart` + microtype font expansion fails with
  "pdfTeX error (font expansion): auto expansion is only possible with scalable fonts".
- Build outputs (`main.pdf`, `main.aux`, `main.bbl`, ...) are untracked and should not be
  committed; `.gitignore` already ignores `*.log`.

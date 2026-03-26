# Companion Files Alongside Jupyter Scoring/Evaluation Notebooks

Research into what files tools ACTUALLY create on disk alongside `.ipynb` files.

---

## 1. Papermill-Executed Scoring Notebook

### What papermill creates on disk

Papermill is minimal — it creates exactly **one file**: the output notebook. No logs, no temp files, no metadata directories.

```
project/
├── score.ipynb                  # input/template notebook (unchanged)
├── output.ipynb                 # executed notebook with outputs + papermill metadata
└── (nothing else from papermill itself)
```

### How the output notebook differs from the template

The output notebook is a standard `.ipynb` JSON file with these differences:

1. **Injected parameters cell** — A new code cell tagged `injected-parameters` inserted immediately after the cell tagged `parameters`. Contains only the overridden values.

2. **Cell-level `papermill` metadata** — Every cell gains a `metadata.papermill` object:

```json
{
  "metadata": {
    "papermill": {
      "duration": 1.234567,
      "start_time": "2025-01-15T10:30:00.000000",
      "end_time": "2025-01-15T10:30:01.234567",
      "status": "completed",
      "exception": false
    },
    "tags": []
  }
}
```

3. **Notebook-level `papermill` metadata** — Top-level `metadata.papermill` records:

```json
{
  "metadata": {
    "papermill": {
      "default_parameters": {},
      "parameters": {"param_name": "value"},
      "input_path": "score.ipynb",
      "output_path": "output.ipynb",
      "duration": 12.345,
      "start_time": "2025-01-15T10:30:00.000000",
      "end_time": "2025-01-15T10:30:12.345000",
      "exception": false,
      "version": "2.6.0"
    }
  }
}
```

4. **Cell outputs populated** — `execution_count` filled in, `outputs` arrays populated with results.

5. **The `parameters` cell is preserved unchanged** — default values stay; the injected cell overrides them.

### Typical papermill project directory

| Path | Content | Why it exists | Always? |
|------|---------|---------------|---------|
| `notebooks/score.ipynb` | Template notebook with `parameters` tag | Input template | Yes |
| `output/score_run1.ipynb` | Executed notebook with outputs | Papermill output | Yes |
| `requirements.txt` | `papermill`, `jupyter`, `pandas`, etc. | Dependency management | Yes |
| `config/params.yaml` | Parameter values for runs | Externalized config | Common |
| `Makefile` or `run.sh` | `papermill score.ipynb output.ipynb -p ...` | Automation | Common |
| `data/` | Input data files | Data for scoring | Common |
| `.gitignore` | Ignores `output/`, checkpoints | Clean VCS | Yes |
| `README.md` | Usage instructions | Documentation | Common |

---

## 2. Jupyter/JupyterLab Project Directory

### What Jupyter creates in the working directory

Jupyter creates exactly **one directory** in the working directory: `.ipynb_checkpoints/`. Nothing else.

The `~/.jupyter/` config directory is in the user's home, NOT the project directory. The runtime files (`connection-*.json`, PIDs) go to `$XDG_RUNTIME_DIR/jupyter/` (Linux) or `~/Library/Jupyter/runtime/` (macOS), NOT the working directory.

### .ipynb_checkpoints/ exact structure

```
project/
├── score.ipynb
├── analysis.ipynb
└── .ipynb_checkpoints/
    ├── score-checkpoint.ipynb
    └── analysis-checkpoint.ipynb
```

**Naming convention:** `{notebook_name}-checkpoint.ipynb`

**Content:** Each checkpoint file is a **complete copy** of the `.ipynb` JSON file as it was at the last manual save ("Save and Checkpoint"). It is structurally identical to the main notebook — same JSON schema, same cell structure. It is NOT updated during autosave.

**Key facts:**
- Created automatically the first time a notebook is saved
- One checkpoint file per notebook
- Recreated automatically if the directory is deleted
- Should always be in `.gitignore`

### What `jupyter nbconvert --execute` creates

```bash
jupyter nbconvert --to notebook --execute score.ipynb
# Creates: score.nbconvert.ipynb  (default output suffix)

jupyter nbconvert --to notebook --execute --inplace score.ipynb
# Overwrites: score.ipynb  (no new file)

jupyter nbconvert --to html --execute score.ipynb
# Creates: score.html

jupyter nbconvert --to notebook --execute score.ipynb --output executed_score.ipynb
# Creates: executed_score.ipynb
```

Default behavior with `--to notebook`: appends `.nbconvert.ipynb` suffix.

| Command variant | File created | Notes |
|----------------|-------------|-------|
| `--to notebook --execute nb.ipynb` | `nb.nbconvert.ipynb` | Default suffix |
| `--to notebook --execute --inplace nb.ipynb` | overwrites `nb.ipynb` | No new file |
| `--to html --execute nb.ipynb` | `nb.html` | HTML report |
| `--to html nb.ipynb` (no execute) | `nb.html` | Static conversion |
| `--to script nb.ipynb` | `nb.py` | Python script extraction |

### Typical data science project with evaluation notebooks

| Path | Content | Why it exists | Always? |
|------|---------|---------------|---------|
| `score.ipynb` | Evaluation notebook | Main notebook | Yes |
| `.ipynb_checkpoints/` | Auto-created by Jupyter | Checkpoint snapshots | Yes (if ever opened in Jupyter) |
| `.ipynb_checkpoints/score-checkpoint.ipynb` | Copy of score.ipynb at last manual save | Recovery | Yes |
| `requirements.txt` | Package versions | Reproducibility | Yes |
| `.gitignore` | `.ipynb_checkpoints` etc. | VCS hygiene | Yes |
| `data/` | CSV, JSON, parquet files | Input data | Common |
| `utils.py` or `helpers.py` | Shared functions imported by notebooks | Code reuse | Common |
| `__pycache__/` | Python bytecode cache | Auto-created by Python imports | If .py files exist |
| `README.md` | Project description | Documentation | Common |

---

## 3. Kaggle-Style Scoring

### Kaggle kernel directory structure

When you run `kaggle kernels pull -k username/kernel-slug -p ./kernel_dir -m`:

```
kernel_dir/
├── kernel-metadata.json
└── kernel-slug.ipynb
```

When you prepare a kernel for `kaggle kernels push`:

```
kernel_dir/
├── kernel-metadata.json
├── my-scoring-kernel.ipynb
└── (optional data files referenced by the notebook)
```

### kernel-metadata.json exact content

```json
{
  "id": "username/my-scoring-kernel",
  "id_no": 12345,
  "title": "My Scoring Kernel",
  "code_file": "my-scoring-kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "false",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": ["username/my-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Field details:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes* | URL slug: `"username/kernel-slug"` |
| `id_no` | int | Yes* | Numeric kernel ID (preferred over `id` if both present) |
| `title` | string | New kernels | Display title, linked to slug |
| `code_file` | string | Yes | Path to notebook/script file (relative to metadata location) |
| `language` | string | Yes | `"python"`, `"r"`, or `"rmarkdown"` |
| `kernel_type` | string | Yes | `"notebook"` or `"script"` |
| `is_private` | string | No | Default `"true"` |
| `enable_gpu` | string | No | Default `"false"` |
| `enable_internet` | string | No | Default `"false"` |
| `dataset_sources` | list | No | `["username/dataset-slug", ...]` |
| `competition_sources` | list | No | `["competition-slug", ...]` |
| `kernel_sources` | list | No | `["username/kernel-slug", ...]` |
| `model_sources` | list | No | `["username/model-slug/framework/variation/version", ...]` |

*One of `id` or `id_no` must be specified.

### Companion files alongside Kaggle scoring notebook

| Path | Content | Why it exists | Always? |
|------|---------|---------------|---------|
| `kernel-metadata.json` | Kernel config (see above) | Required by Kaggle API | Yes (for push/pull) |
| `my-kernel.ipynb` | The notebook itself | Main code file | Yes |
| `submission.csv` | Output predictions | Kaggle competition output format | If competition kernel |
| `__notebook_source__.ipynb` | Internal Kaggle name for pulled kernels | Kaggle's internal naming | Sometimes |

---

## 4. Generic Python ML Project with Notebooks

### requirements.txt

```
jupyter==1.1.1
notebook==7.3.2
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
```

Note: Exact versions are examples — the key is that versions are pinned, and the list reflects generic ML evaluation tooling (not task-specific).

### .gitignore (combined Python + Jupyter standard)

```gitignore
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
build/
dist/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*
.virtual_documents/

# IPython
profile_default/
ipython_config.py

# Environment variables
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data (too large for git)
data/*.csv
data/*.json
data/*.parquet
*.h5
*.pkl

# Outputs
output/
*.html
```

### pyproject.toml (minimal, for a project that uses notebooks)

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "evaluation"
version = "0.1.0"
description = "Model evaluation toolkit"
requires-python = ">=3.10"
dependencies = [
    "jupyter",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### README.md (alongside evaluation notebooks)

```markdown
# Evaluation

Notebooks for model evaluation and scoring.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the scoring notebook:

```bash
jupyter nbconvert --to notebook --execute score.ipynb
```

Or with papermill:

```bash
papermill score.ipynb output/score_result.ipynb -p data_path data/input.csv
```

## Project Structure

```
├── score.ipynb          # Main evaluation notebook
├── data/                # Input data
├── output/              # Execution results
├── requirements.txt     # Dependencies
└── README.md
```
```

### Common config files

| File | Content | Frequency |
|------|---------|-----------|
| `pyproject.toml` | Build system + project metadata + tool config | Very common (modern) |
| `setup.cfg` | Legacy package config (being replaced by pyproject.toml) | Declining |
| `requirements.txt` | Pinned dependencies | Very common |
| `requirements.in` | Unpinned direct dependencies (pip-tools workflow) | Common |
| `.env` | Environment variables (API keys, paths) | Common, never committed |
| `Makefile` | Automation shortcuts (`make run`, `make clean`) | Common |
| `tox.ini` / `nox.toml` | Test automation across Python versions | Occasional |
| `.pre-commit-config.yaml` | Git hook configuration | Occasional |
| `environment.yml` | Conda environment specification | Common in academic settings |

### Data files typically present

| Path | Format | Purpose |
|------|--------|---------|
| `data/input.csv` | CSV | Input data for evaluation |
| `data/predictions.csv` | CSV | Model predictions |
| `data/` (directory) | Mixed | Data storage |
| `data/raw/` | Mixed | Unprocessed data |
| `data/processed/` | Mixed | Cleaned data |
| `config.json` or `config.yaml` | JSON/YAML | Runtime parameters |
| `models/` | `.pkl`, `.joblib`, `.h5` | Saved model artifacts |

---

## 5. Complete Reference: All Scenarios Combined

### Files that are ALWAYS present (tool-created, unavoidable)

| File/Dir | Created by | Trigger |
|----------|-----------|---------|
| `.ipynb_checkpoints/` | Jupyter | Opening any notebook in Jupyter |
| `.ipynb_checkpoints/{name}-checkpoint.ipynb` | Jupyter | First manual save of notebook |
| `__pycache__/` | Python | Any `import` of local `.py` files |
| Output notebook (e.g., `output.ipynb`) | Papermill | Running `papermill input.ipynb output.ipynb` |
| `{name}.nbconvert.ipynb` | nbconvert | Running `nbconvert --to notebook --execute` |

### Files that are conventionally present (human-created, expected)

| File | Scenario | Notes |
|------|----------|-------|
| `requirements.txt` | All Python projects | Expected in any shareable project |
| `.gitignore` | All git repos | Expected if using version control |
| `README.md` | All shareable projects | Expected documentation |
| `kernel-metadata.json` | Kaggle only | Required for `kaggle kernels push` |
| `data/` directory | Most ML projects | Where input data lives |
| `utils.py` / `helpers.py` | Projects with shared code | Imported by notebooks |
| `Makefile` | Automation-heavy projects | Convenience commands |
| `config.yaml` / `params.yaml` | Parameterized runs | Externalized configuration |

### Minimal believable directory for each scenario

**Papermill scoring project:**
```
scoring/
├── score.ipynb
├── output/
│   └── score_executed.ipynb
├── data/
│   └── input.csv
├── requirements.txt
├── .gitignore
├── .ipynb_checkpoints/
│   └── score-checkpoint.ipynb
└── README.md
```

**Jupyter evaluation project:**
```
evaluation/
├── evaluate.ipynb
├── utils.py
├── data/
│   └── results.csv
├── requirements.txt
├── .gitignore
├── .ipynb_checkpoints/
│   └── evaluate-checkpoint.ipynb
├── __pycache__/
│   └── utils.cpython-310.pyc
└── README.md
```

**Kaggle kernel directory:**
```
kernel/
├── scoring-kernel.ipynb
└── kernel-metadata.json
```

**Generic ML notebook project:**
```
project/
├── score.ipynb
├── data/
│   ├── train.csv
│   └── predictions.csv
├── requirements.txt
├── .gitignore
├── .ipynb_checkpoints/
│   └── score-checkpoint.ipynb
├── README.md
└── pyproject.toml
```

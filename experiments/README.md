# Tutorial notebooks

Five notebooks covering the framework's public API. Each opens in Colab and
clones the repository, so none of them require a local installation.

| Notebook | Covers |
|---|---|
| [Experiment_1.ipynb](Experiment_1.ipynb) | End-to-end first run: closed-set identification on ECG-ID |
| [load_dataset_Module.ipynb](load_dataset_Module.ipynb) | **Dataset loading** — the unified loader API across all seven datasets |
| [run_Module.ipynb](run_Module.ipynb) | **Protocol switching** — driving the eight evaluation regimes directly |
| [Custom_Model.ipynb](Custom_Model.ipynb) | **Custom model integration** — the contract, and the one-line registration |
| [Custom_Dataset.ipynb](Custom_Dataset.ipynb) | Benchmarking your own dataset |

## Reproducing the published results

The notebooks teach the API. To reproduce the published result tables, use the
shipped configurations instead, which are the single source of truth for what
was run:

```bash
# Inspect the plan
python -m scripts.reproduce_tables --table 5 --dry-run

# Run a dataset
python -m scripts.reproduce_tables --dataset ecgid --run

# Assemble the tables
python -m scripts.reproduce_tables --collect --output-dir reproduced_tables
```

See [configs/paper_reproduction/README.md](../configs/paper_reproduction/README.md).

## A note on removed files

Earlier versions of this folder contained `<dataset>_experiments.ipynb` and
`<dataset>_gather_results.py` for each of the seven datasets. They were
removed rather than repaired, for two reasons:

- **They no longer ran.** They called an API that no longer exists
  (`run_verification`, `load_all_sessions`, `enrollment_mode=`,
  `load_session("Session_1")`, `preprocessing_params=` with `window_len` and
  `bandpass` keys), and would have failed on their first cell.
- **They were superseded.** They hard-coded 2 training epochs and 3-beat
  merging, which correspond to no published result. The configurations in
  `configs/paper_reproduction/` reproduce the tables exactly, and
  `scripts/reproduce_tables.py` runs and collects them.

## Keeping these current

`tests/test_tutorial_notebooks.py` checks that every framework symbol these
notebooks import still exists. If an API is renamed, that test fails and names
the notebook, so the tutorials cannot silently rot again.

```bash
python -m pytest tests/test_tutorial_notebooks.py -q
```

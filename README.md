# SardiniaFireAI

**Multi-sensor wildfire event reconstruction, benchmarking, and machine learning in Sardinia, Italy.**

SardiniaFireAI is a Python research pipeline that processes NASA FIRMS active-fire detections (VIIRS and MODIS), reconstructs wildfire events with ST-DBSCAN, evaluates candidate associations against official Sardinian CFVA wildfire perimeters, replays detections chronologically, and explores burned-area prioritization with machine learning.

The work is motivated in part by the evaluation questions of the Satellites on Fire technical challenge. **This is retrospective research software, not a validated real-time warning system.**

## Scope and current implementation

| Component | Status |
| --- | --- |
| Historical NASA FIRMS VIIRS + MODIS acquisition | Implemented |
| CFVA wildfire-perimeter ingestion and quality checks | Implemented |
| Sardinian territorial mask (geoBoundaries/ISTAT) | Used in historical preprocessing |
| ST-DBSCAN (dense/sparse) | Implemented and tested |
| Incremental ST-DBSCAN and event transitions | Implemented and tested |
| Spatial/temporal candidate matching and fire-size coverage | Evaluated retrospectively |
| Event thermal, temporal, multisensor and spatial features | Implemented |
| Preliminary alerts, FRP policies, lifecycle audit | Implemented experimentally |
| Ridge, Random Forest, HGB, tabular MLP | Evaluated on provisional labels |
| Spatial CNN, spatial MLP, hybrid CNN+MLP | Experimental; no demonstrated improvement |
| Sentinel-2/Landsat image-model integration | **Not implemented** |

## Repository layout

```text
configs/                  YAML source, clustering, feature and evaluation settings
src/sardinia_fire/        Reusable ingestion, preprocessing, clustering,
                          evaluation, feature, ranking and model modules
scripts/ingestion/        Source acquisition, schema checks and preprocessing
scripts/evaluation/       Reference matching and auditing
scripts/experiments/      Clustering, alerts, ML and deep learning experiments
scripts/visualization/    Characterization and GIS export tools
tests/                    Automated tests
data/raw/                 Source downloads (local, may be ignored by Git)
data/processed/           Processed Parquet data (local, may be ignored)
outputs/tables/           Generated reports (local, may be ignored)
```

## Data sources

- **NASA FIRMS:** `VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP`, `VIIRS_NOAA21_SP`, and `MODIS_SP`. Historical active-fire data require a FIRMS `MAP_KEY`: https://firms.modaps.eosdis.nasa.gov/
- **CFVA / Regione Autonoma della Sardegna:** official wildfire perimeter records (reference catalog assembled for 2019–2025). The reference fire date is a **calendar date, not a confirmed ignition timestamp**.
- **geoBoundaries / ISTAT:** Sardinian administrative boundaries used for territorial filtering: https://www.geoboundaries.org/ . Preserve source attribution and relevant dataset licenses.

Satellite detections are point observations, not automatically confirmed independent wildfires. A candidate match with a CFVA perimeter does not prove unique event attribution. Unmatched events are **unlabeled, not verified negatives**.

## Requirements and installation

Development environment: Python **3.11** (minimum declared requirement `>=3.11`). Run commands from the repository root.

**Windows PowerShell**

```powershell
git clone https://github.com/reimanrey93/SardiniaFireAI.git
cd SardiniaFireAI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install scipy torch
```

**Linux/macOS**

```bash
git clone https://github.com/reimanrey93/SardiniaFireAI.git
cd SardiniaFireAI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install scipy torch
```

**Dependency note:** at the time of this README update, `pyproject.toml` does not explicitly declare SciPy or PyTorch, both of which are required by parts of the pipeline. The local DL experiments ran with `torch 2.14.1+cpu`. Check https://pytorch.org/get-started/locally/ for platform-specific installation options. These installation commands describe the **updated local working tree**; some experimental files are not yet committed to the remote `main` branch.

## Configuration and data acquisition

Configuration files:

```text
configs/sources.yaml
configs/clustering/st_dbscan.yaml
configs/features/event_features.yaml
configs/evaluation/event_matching.yaml
configs/evaluation/event_matching_2024_06_09.yaml
```

The checked-in `configs/clustering/st_dbscan.yaml` is a **July 2024** configuration, with `eps_space_m=1500`, `eps_time_h=24`, `min_samples=3`, `metric_crs=EPSG:32632`, and `implementation=sparse`. Do not assume it automatically orchestrates the June–September experiment.

Set the NASA FIRMS key without writing it into committed files:

```powershell
$env:FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"   # Windows PowerShell
```

```bash
export FIRMS_MAP_KEY="YOUR_FIRMS_MAP_KEY" # Linux/macOS
```

Download one month at a time (repeat for June, July, August and September 2024):

```bash
python scripts/ingestion/14_download_firms_month.py --year 2024 --month 6
python scripts/ingestion/14_download_firms_month.py --year 2024 --month 7
python scripts/ingestion/14_download_firms_month.py --year 2024 --month 8
python scripts/ingestion/14_download_firms_month.py --year 2024 --month 9
```

The downloader supports an optional `--overwrite`. Check outputs under `data/raw/firms/` and `data/processed/firms/`. The monthly download command **does not by itself produce** the territorial, season-wide historical Parquet dataset.

CFVA ingestion and QC utilities are in `scripts/ingestion/` (notably `07_consolidate_cfva.py` and `09_build_cfva_reference.py`). Building `data/processed/cfva/cfva_benchmark_reference.parquet` requires the upstream consolidated CFVA dataset at the paths those scripts specify.

## Reproduction: July 2024 configured workflow

Once the required four July FIRMS product files exist:

```bash
python scripts/experiments/run_st_dbscan.py
python scripts/experiments/extract_event_features.py
python scripts/evaluation/evaluate_events.py
python scripts/evaluation/build_ranking_dataset.py
```

**Check prerequisites and configurations before each command.** `run_st_dbscan.py` uses `configs/clustering/st_dbscan.yaml`. The feature and evaluation scripts load separate YAML configurations; their paths must match the same July study period. The ranking script currently uses July-specific file paths. Generated outputs include:

```text
data/processed/clustering/st_dbscan_2024_07.parquet
data/processed/features/st_dbscan_events_2024_07.parquet
data/processed/evaluation/ranking_dataset_2024_07.parquet
```

## Reproduction: June–September 2024 research snapshot

The **seasonal experiments are a separate workflow**. Their principal prerequisite files are:

```text
data/processed/historical/firms_sardinia_2024_06_09.parquet
data/processed/clustering/st_dbscan_2024_06_09.parquet
data/processed/cfva/cfva_benchmark_reference.parquet
data/processed/evaluation/ranking_dataset_2024_06_09.parquet
```

The June–September seasonal dataset was prepared from normalized FIRMS detections with Sardinian territorial filtering (including a 500 m coastal tolerance), and seasonal clustering. **A single fully automated command to regenerate all of these seasonal inputs from a fresh clone has not yet been verified.** The July-only `evaluate_events.py` and `build_ranking_dataset.py` must not be presented as producing their June–September counterparts unchanged.

### Incremental replay

With the historical detections and seasonal cluster file present:

```bash
python scripts/experiments/run_incremental.py
```

Produces:

```text
data/processed/operational/operational_events_2024_06_09.parquet
data/processed/operational/operational_detections_2024_06_09.parquet
data/processed/operational/operational_transitions_2024_06_09.csv
```

The script verifies the final incremental partition against the retrospective clustering result. The simulation groups arrivals by FIRMS acquisition timestamp; its final catalog is not a complete archive of every intermediate snapshot.

### Preliminary-alert lifecycle

With the operational reference files in place:

```bash
python scripts/experiments/run_alert_lifecycle.py
```

Uses `src/sardinia_fire/ranking/preliminary_alerts.py` and `alert_lifecycle.py`, writing transition logs under `data/processed/operational/alerts/` for the `all`, `global_3mw` and `product_p50` policies. The product-P50 threshold is based on final-season noise, so that comparison is **retrospectively calibrated**, not a prospective evaluation. Alert transitions are detection-level; event-level lead times require avoiding duplicate event counting and separating alerts emitted after event creation.

### Machine-learning experiments

The season-wide ranking dataset is required:

```text
data/processed/evaluation/ranking_dataset_2024_06_09.parquet
```

Scripts for tabular models:

```bash
python scripts/experiments/run_ml_baselines.py
python scripts/experiments/run_ml_comparison_mlp.py
python scripts/experiments/run_ml_feature_ablation.py
```

The ablation evaluates non-FRP, FRP-only and combined features using Ridge, Random Forest, HistGradientBoosting and MLP. Predictions are made out of fold with **five GroupKFold splits by official reference fire ID**. Feature transformations and imputers are fitted on the training subset. CFVA matching/quality variables are excluded from predictors. Models estimate `log1p(burned_area_ha)` and invert that transformation for area errors; prediction clipping is derived from training labels only.

Key output files:

```text
outputs/tables/ml_feature_ablation_2024_06_09.csv
outputs/tables/ml_feature_ablation_oof_2024_06_09.csv
outputs/tables/ml_ranking_uncertainty_2024_06_09.csv
```

**Important:** the ranking uncertainty audit was previously run as an interactive Python block; `ml_ranking_uncertainty_2024_06_09.csv` is a saved analysis result, not currently guaranteed to be regenerated by a named repository script.

Deep-learning scripts (PyTorch required):

```bash
python scripts/experiments/audit_spatial_cnn_inputs.py
python scripts/experiments/run_spatial_dl.py
python scripts/experiments/run_hybrid_cnn.py
```

The custom spatial raster uses three 32×32 channels over a fixed 8×8 km window: detection count, transformed VIIRS FRP, and transformed MODIS FRP. The spatial MLP and CNN use these raster inputs; the hybrid CNN additionally takes tabular FIRMS features. They are **exploratory**, not recommended as the strongest current ranking models.

`run_torch_optimizer_comparison.py` exists locally as an optional optimization experiment; **no executed benchmark result is claimed for it here**.

Many experiment scripts refuse to overwrite existing outputs. Preserve current results, choose a new output destination, or explicitly archive old outputs before rerunning; do not delete research results unintentionally.

## Reference results (historical research snapshot)

Data: Sardinia, June–September 2024, four FIRMS products.

| Quantity | Reported value |
| --- | ---: |
| FIRMS point detections | 1,385 |
| ST-DBSCAN reconstructed events | 121 |
| Grouped detections | 1,098 |
| Final noise detections | 287 |
| Chronological acquisition updates | 480 |
| Official CFVA reference fires (core period) | 792 |
| CFVA fires with an individual FIRMS candidate | 197 / 792 (24.87%) |
| CFVA fires with a grouped ST-DBSCAN candidate | 102 / 792 (12.88%) |
| Provisionally one-to-one area-labeled events | 56 |
| Higher-quality A+B labeled events | 38 |

Coverage is **candidate association coverage** under the selected spatial and calendar-date tolerances; it is not a verified estimate of operational recall. The size distribution and tolerance sensitivity matter greatly, particularly for small fires.

Illustrative exploratory **global Spearman** results on 56 provisional labels:

| Ranking approach | Spearman |
| --- | ---: |
| Detection count + FRP rank baseline | 0.687 |
| Ridge, combined features | 0.716 |
| Tabular MLP, combined features | 0.725 |
| Spatial CNN | 0.171 |
| Hybrid CNN + tabular MLP | 0.269 |

The tabular MLP's observed advantage over the rank baseline is **not statistically conclusive** in the paired bootstrap audit; the rank baseline also produced stronger NDCG@10 than that MLP in the reported experiment. Reusing these 56 fires for architecture selection can produce optimistic estimates. The models have **not** been validated on an independent year or region.

## Testing and quality checks

```bash
python -m ruff check src tests scripts
python -m pytest -q
python -m pytest -v
```

Last reported **local** validation: **139 tests passed** (23 test modules). This count refers to the local working tree with new alert/ML/CNN modules, **not necessarily the remote GitHub `main` branch**. These tests establish software behavior, not complete scientific validation.

Useful final checks before committing:

```bash
git status --short
git diff --check
git ls-files scripts/experiments/run_ml_feature_ablation.py
```

A current reproduction record should include Git commit, Python/dependency versions, YAML configurations, input data versions and checksums, and produced tables. Verify that all locally added scripts and tests are actually tracked before claiming that remote `main` reproduces these results.

## Known limitations

1. The **seasonal data preparation and benchmark build are not yet verified as one clean-clone, end-to-end workflow**; separate July and season-wide configurations must not be confused.
2. External data must be accessed under provider terms, and API credentials must not be committed.
3. CFVA dates are not verified ignition times. Internal event-identification delay is **not ignition-to-detection latency**.
4. Candidate CFVA matches can be ambiguous; unmatched detections are not confirmed nonfires.
5. Only 56 provisional area labels were available for the reported ML comparisons, including 38 with A+B evidence.
6. Retrospective clustering can leak future event identity into apparent early predictions; early-time inference requires a causal snapshot evaluation.
7. CNN input rasters are sparse and do not encode Sentinel-2 or Landsat imagery.
8. Model exploration and repeated comparisons require an independent external validation set before claiming generalizable ML gains.

## Future work

Independent annual/geographic evaluation; expanded CFVA labels; documented season-wide build; calibrated prospective alerts; improved ranking metrics; and optional Sentinel-2/Landsat imagery integration.

## Author

**Reiman Alfonso Azcuy** — [GitHub](https://github.com/reimanrey93)

## Attribution

NASA FIRMS, Regione Autonoma della Sardegna / CFVA, and geoBoundaries / ISTAT. Respect each provider's attribution, access, licensing, and redistribution requirements.


# SardiniaFireAI

Multi-sensor wildfire event reconstruction and benchmarking in Sardinia, Italy.

## Overview

SardiniaFireAI is a reproducible Python research pipeline for detecting, reconstructing, and evaluating wildfire events using satellite observations and official fire records.

The project builds upon methodological developments from the Satellites on Fire Challenge and adapts them to the Sardinian context.

## Research objectives

- Integrate multi-sensor satellite fire detections.
- Reconstruct wildfire events using spatiotemporal clustering.
- Benchmark ST-DBSCAN, HDBSCAN, and event-tracking approaches.
- Evaluate reconstructed events against official fire records.
- Analyze detection latency, fragmentation, and event matching.
- Develop machine-learning approaches for event prioritization.

## Data sources

Planned sources include:

- NASA FIRMS: VIIRS and MODIS active-fire detections.
- Regione Autonoma della Sardegna: official wildfire records and burned-area perimeters.
- Sentinel-2: spectral analysis and burned-area assessment.

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest -v
python -m ruff check src tests
```

## Status

Early development — v0.1.0.

## Author

Reiman Alfonso Azcuy

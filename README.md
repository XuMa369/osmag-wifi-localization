# osmag-wifi-localization

Implementation of "WiFi-based Global Localization in Large-Scale Environments Leveraging Structural Priors from osmAG"

This project presents a WiFi-based localization framework for autonomous robotics in large-scale indoor environments where GPS is unavailable. By leveraging ubiquitous wireless infrastructure and OpenStreetMap Area Graph (osmAG) structural priors, the system supports:

- **AP Position Estimation** — Reverse-estimating physical Access Point locations from crowdsourced fingerprint data
- **Fingerprint-based KNN Localization** — Positioning via K-Nearest Neighbors on RSSI feature vectors
- **AP-based Trilateration Localization** — Iterative position optimization using estimated AP coordinates with wall-aware RSSI correction

---

## Table of Contents

- [osmag-wifi-localization](#osmag-wifi-localization)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
    - [1) AP Localization](#1-ap-localization)
    - [2) Robot Fingerprint KNN Localization](#2-robot-fingerprint-knn-localization)
    - [3) Robot AP-based Localization](#3-robot-ap-based-localization)
  - [Project Structure](#project-structure)
  - [Map Data Format](#map-data-format)

---

## Environment

- **Python**: 3.10+
- **OS**: Ubuntu 22.04 (tested)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/osmag-wifi-localization.git
cd osmag-wifi-localization

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.23.0 | Numerical computation |
| scipy | ≥ 1.8.0 | Optimization (minimize, least_squares, curve_fit) |
| scikit-learn | ≥ 1.7.0 | KNN regression |
| shapely | ≥ 2.1.1 | Geometric computation (polygon, line intersection) |
| pyproj | ≥ 3.7.1 | WGS84 geodesic distance calculation |
| PyYAML | ≥ 5.4.1 | YAML configuration parsing |

---

## Quick Start

The system provides three main pipelines. A typical workflow runs them in order: **AP Localization → Fingerprint KNN → AP-based Localization**.

### 1) AP Localization

Estimate physical positions of WiFi Access Points from fingerprint observations.

```bash
python ap_localization.py \
  --config ./config/ap_localization_config.yaml \
  --log-level INFO
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Path to YAML configuration file | `./config/ap_localization_config.yaml` |
| `--log-level` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

- **Input** (configured in YAML):
  - `file_paths.input_osm_file` → `./map/wifi_data.osm` (fingerprint database + building polygons)
  - `file_paths.template_osm_file` → `./map/base_map.osm` (base map template)
- **Output**:
  - `file_paths.output_osm_file` → `./map/AP_MAP.osm` (estimated AP positions in OSM format)
  - Console statistics: initial error, final error, improvement percentage

**Processing pipeline:**  
Data Loading → Signal Model Optimization → Data Preprocessing (grouping, floor filtering, trajectory sampling) → Building Constraint Extraction → Iterative AP Position Estimation (with wall detection + RSSI correction) → Result Export

### 2) Robot Fingerprint KNN Localization

Localize test points using K-Nearest Neighbors on the RSSI fingerprint database.

```bash
python robot_fingerprint_localization.py \
  --fingerprint ./map/wifi_data.osm \
  --test ./map/Non-FingerprintedAreas.osm \
  --polygon ./map/wifi_data.osm \
  -k 5 --log-level INFO
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fingerprint` | OSM file containing the fingerprint database | (required) |
| `--test` | OSM file containing test points | (required) |
| `--polygon` | OSM file containing building polygon constraints | (required) |
| `-k` | Number of nearest neighbors | `5` |
| `--no-boundary` | Disable polygon boundary constraint | `false` |

- **Output**: Localization metrics (mean error, std, RMSE, 95th percentile)

### 3) Robot AP-based Localization

Localize fingerprint points using estimated AP positions via iterative trilateration.

```bash
python robot_AP_localization.py \
  --ap-map ./map/AP_MAP.osm \
  --fingerprint ./map/Non-FingerprintedAreas.osm \
  --polygon ./map/wifi_data.osm \
  --iter 10 --log-level INFO
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ap-map` | OSM file with estimated AP positions (output of step 1) | (required) |
| `--fingerprint` | OSM file with test fingerprint points | (required) |
| `--polygon` | OSM file with building polygon constraints | (required) |
| `--rssi0` | RSSI at 1 meter reference distance (dBm) | `-28.79` |
| `--n` | Path loss exponent | `2.14` |
| `--wall` | Wall attenuation factor (dB) | `3.55` |
| `--iter` | Number of wall-detection iterations | `10` |

- **Output**: Localization results with error analysis

---

## Project Structure

```
osmag-wifi-localization/
├── ap_localization.py                 # Entry: AP position estimation
├── robot_fingerprint_localization.py  # Entry: KNN fingerprint localization
├── robot_AP_localization.py           # Entry: AP-based trilateration localization
├── requirements.txt                   # Python dependencies
│
├── config/
│   └── ap_localization_config.yaml    # Full system configuration
│
├── core/                              # Core business logic
│   ├── data_loader.py                 # OSM data loading and parsing
│   ├── models.py                      # Data models (APGroupData, ProcessingResult)
│   ├── signal_model.py                # RSSI propagation model optimization
│   ├── preprocessor.py                # Data preprocessing (grouping, filtering, sampling)
│   ├── position_estimator.py          # AP position estimation engine
│   ├── fingerprint_knn.py             # KNN-based fingerprint localizer
│   ├── fingerprint_point.py           # AP-based fingerprint point localizer
│   └── result_manager.py             # Result saving and statistics reporting
│
├── algorithms/                        # Low-level algorithms
│   ├── point_estimator.py             # Polygon-constrained trilateration optimizer
│   └── rssi_optimizer.py              # RSSI model parameter fitting (RSSI₀, n, wall)
│
├── io_layer/                          # I/O abstraction
│   ├── osm_parser.py                  # OSM XML parser (AP nodes, fingerprints, polygons)
│   └── osm_writer.py                  # OSM XML writer (AP positions → OSM format)
│
├── utils/                             # Utility modules
│   ├── building_constraints.py        # Building polygon extraction and clustering
│   ├── configuration.py               # YAML/JSON config loader with dataclasses
│   ├── data_processing.py             # RSSI deduplication at same locations
│   ├── geometry.py                    # Geodesic distance, 3D distance, angle computation
│   ├── signal.py                      # RSSI ↔ distance conversion (log-distance model)
│   ├── trajectory_filter.py           # Smart trajectory sampling with spatial constraints
│   └── wall_learning.py               # Wall/no-wall training data extraction
│
└── map/                               # Sample map data (OSM format)
    ├── wifi_data.osm                  # Fingerprint database + building polygons
    ├── base_map.osm                   # Base map template
    ├── AP_MAP.osm                     # Estimated AP positions (output)
    ├── FingerprintedAreas.osm         # Fingerprinted area definitions
    └── Non-FingerprintedAreas.osm     # Test points (non-fingerprinted areas)
```

---

## Map Data Format

This project uses **OSM XML** as the data interchange format with custom `osmAG:WiFi:*` tags:

| Tag Key | Description | Example |
|---------|-------------|---------|
| `osmAG:node:type` | Node type identifier | `AP`, `fingerprint` |
| `osmAG:WiFi:AP:level` | Floor level of AP | `1` |
| `osmAG:WiFi:AP:learn` | Learning status flag | `1` |
| `osmAG:WiFi:BSSID:<band>:<idx>` | AP MAC address by frequency band | `AA:BB:CC:DD:EE:FF` |
| `osmAG:WiFi:RSSI:<idx>` | RSSI measurement value | `-65` |
| `osmAG:WiFi:Freq:<idx>` | Frequency band indicator | `2.4G`, `5G` |
| `osmAG:WiFi:Fingerprint:Floor` | Fingerprint floor level | `1` |

Building polygons are represented as standard OSM `way` elements with node references defining boundary vertices.




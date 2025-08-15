# osmag-wifi-localization

**Official implementation of ["WiFi-based Global Localization in Large-Scale Environments Leveraging Structural Priors from osmAG"](https://arxiv.org/abs/2508.10144)**

This project presents a novel WiFi-based localization framework for autonomous robotics in large-scale indoor environments where GPS is unavailable. By leveraging ubiquitous wireless infrastructure and OpenStreetMap Area Graph (osmAG) structural priors.
## Environment
- Python: 3.10 
- Ubuntu 22.04



## Quick Start

### 1) AP Localization 
```bash
python ap_localization.py --config ./config/ap_localization_config.yaml --log-level INFO
```
- Input (defaults in config):
  - `file_paths.input_osm_file`: `./map/wifi_data.osm`
  - `file_paths.template_osm_file`: `./map/base_map.osm`
- Output:
  - `file_paths.output_osm_file`: `./map/AP_MAP.osm` (estimated AP positions are written)
  - Console statistics with final/initial errors and improvement

### 2) Robot fingerprint KNN Localization
```bash
python robot_fingerprint_localization.py \
  --fingerprint ./map/wifi_data.osm \
  --test ./map/Non-FingerprintedAreas.osm \
  --polygon ./map/wifi_data.osm \
  -k 5 --log-level INFO
```
- Add `--no-boundary` to disable polygon boundary constraint
- Prints localization metrics to console

### 3) Robot access point Localization 
```bash
python robot_AP_localization.py \
  --ap-map ./map/AP_MAP.osm \
  --fingerprint ./map/Non-FingerprintedAreas.osm \
  --polygon ./map/wifi_data.osm \
  --iter 10 --log-level INFO
```
- Optional model parameters (with defaults): `--rssi0`, `--n`, `--wall`
- Outputs analyzed results to console



## Project Structure
- `ap_localization.py`: AP localization entry
- `robot_fingerprint_localization.py`: KNN-based localization entry
- `robot_AP_localization.py`: Iterative robot localization entry
- `core/`: data loading, preprocessing, signal model optimization, position estimation, result saving, fingerprint algorithms
- `algorithms/`: lower-level point estimator and RSSI optimizer
- `io_layer/`: OSM parser and writer
- `utils/`: configuration, building constraints, geometry, filtering, wall learning
- `map/`, `config/`: sample paths referenced by defaults


## Citation

If you use this work in your research, please cite:

```bibtex
@misc{ma2025wifibasedgloballocalizationlargescale,
      title={WiFi-based Global Localization in Large-Scale Environments Leveraging Structural Priors from osmAG}, 
      author={Xu Ma and Jiajie Zhang and Fujing Xie and Sören Schwertfeger},
      year={2025},
      eprint={2508.10144},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.10144}, 
}
```



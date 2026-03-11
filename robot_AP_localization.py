#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Fingerprint Point Localization Script (thin entry)
"""

import os
import sys
import argparse
import logging
from io_layer.osm_parser import OsmDataParser, APMapParser, FingerprintParser
from utils.building_constraints import find_largest_polygon
from core.fingerprint_point import WiFiFingerprintLocalizer, analyze_fingerprint_results



def main():
    parser = argparse.ArgumentParser(description="WiFi fingerprint point indoor localization")
    parser.add_argument("--ap-map", default="./map/AP_MAP.osm", help="AP map file path")
    parser.add_argument("--fingerprint", default="./map/Non-FingerprintedAreas.osm", help="Fingerprint file path")
    parser.add_argument("--polygon", default="./map/wifi_data.osm", help="Polygon map file path")
    parser.add_argument("--iter", type=int, default=10, help="Total iteration number")
    parser.add_argument("--rssi0", type=float, default=-28.79169776352291, help="Signal model A (rssi@1m)")
    parser.add_argument("--n", type=float, default=2.542972169273509, help="Path loss exponent")
    parser.add_argument("--wall", type=float, default=-10.772449287430723, help="Wall attenuation value")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("WiFi Fingerprint Point Indoor Localization System")
    logger.info("%s", "="*60)

    ap_map_file = args.ap_map
    fingerprint_file = args.fingerprint
    polygon_file = args.polygon

    for file_path, file_name in [(ap_map_file, "AP map file"), (fingerprint_file, "Fingerprint file"), (polygon_file, "Polygon map file")]:
        if not os.path.exists(file_path):
            logger.error("%s does not exist: %s", file_name, file_path)
            sys.exit(2)

    logger.info("Step 1: Parsing AP position reference data")
    ap_parser = APMapParser(ap_map_file)
    ap_positions = ap_parser.parse_ap_map()
    if not ap_positions:
        logger.error("No valid AP position data found in %s", ap_map_file)
        sys.exit(3)

    logger.info("Step 2: Parsing fingerprint point data to be localized")
    fingerprint_parser = FingerprintParser(fingerprint_file)
    fingerprint_data = fingerprint_parser.parse_fingerprint_data()
    if not fingerprint_data:
        logger.error("No valid fingerprint point data found in %s", fingerprint_file)
        sys.exit(4)

    polygon_parser = OsmDataParser(polygon_file)
    polygon_parser.parse()
    polygon_parser.extract_polygon_edges()
    _, _, _, _, _, polygons, polygon_edges, _ = polygon_parser.get_data()

    building_clusters_result = find_largest_polygon(polygons)

    logger.info("Step 3: Execute iterative fingerprint point localization")
    iter_num_total = args.iter
    rssi_0 = args.rssi0
    n = args.n
    ave_val = args.wall
    localizer = WiFiFingerprintLocalizer(ap_positions, polygon_edges, rssi_0=rssi_0, n=n, ave_val=ave_val)
    results = localizer.localize_fingerprint_points(fingerprint_data, polygons, building_clusters_result, iter_num_total)

    if results:
        logger.info("Step 4: Processing localization results")
    else:
        logger.warning("No localization results available.")

    analyze_fingerprint_results(results)

    logger.info("Fingerprint point localization program completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error("Unhandled exception in fingerprint_point_localization: %s", e, exc_info=True)
        sys.exit(1)
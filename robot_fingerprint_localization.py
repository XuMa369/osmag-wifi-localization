#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Indoor Localization Script with Boundary Constraints (thin entry)
"""

import os
import sys
import argparse
import logging
from io_layer.osm_parser import WiFiFingerprintParser, WiFiTestPointParser
from core.fingerprint_knn import WiFiKNNLocalizer, calculate_statistical_metrics



def main():
    parser = argparse.ArgumentParser(description="WiFi KNN localization with optional boundary constraints")
    parser.add_argument("--fingerprint", default="./map/wifi_data.osm", help="Fingerprint database file path")
    parser.add_argument("--test", default="./map/Non-FingerprintedAreas.osm", help="Test point file path")
    parser.add_argument("--polygon", default="./map/wifi_data.osm", help="Building boundary file path (OSM)")
    parser.add_argument("--no-boundary", action="store_true", help="Disable boundary constraint")
    parser.add_argument("-k", type=int, default=5, help="K value for KNN")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    fingerprint_file = args.fingerprint
    test_file = args.test
    polygon_file = args.polygon

    use_boundary_constraint = not args.no_boundary

    required_files = [(fingerprint_file, "Fingerprint database file"), (test_file, "Test point file")]
    if use_boundary_constraint:
        required_files.append((polygon_file, "Building boundary file"))
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            logger.error("%s does not exist: %s", file_desc, file_path)
            sys.exit(2)

    logger.info("Step 1: Parsing fingerprint database")
    fp_parser = WiFiFingerprintParser(fingerprint_file)
    fingerprint_data = fp_parser.parse_fingerprint_data()
    if not fingerprint_data:
        logger.error("No fingerprint data parsed from %s", fingerprint_file)
        sys.exit(3)

    logger.info("Step 2: Parsing test point data")
    tp_parser = WiFiTestPointParser(test_file)
    test_points = tp_parser.parse_test_points()
    if not test_points:
        logger.error("No test points parsed from %s", test_file)
        sys.exit(4)

    logger.info("Step 3: Initializing KNN localizer")
    localizer = WiFiKNNLocalizer(
        fingerprint_data=fingerprint_data,
        all_macs=fp_parser.all_macs,
        k=args.k,
        polygon_file=polygon_file if use_boundary_constraint else None,
        use_boundary_constraint=use_boundary_constraint,
    )

    logger.info("Step 4: Performing localization")
    results = localizer.localize(test_points)
    calculate_statistical_metrics(results)

    logger.info("Localization completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error("Unhandled exception in fingerprint_knn_localization: %s", e, exc_info=True)
        sys.exit(1) 


    
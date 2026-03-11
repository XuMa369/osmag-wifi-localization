#!/usr/bin/env python3
"""
AP Localization System - Object-Oriented Refactored Version
Refactoring Principles: Single Responsibility, Open-Closed Principle, Dependency Inversion
"""

import argparse
from typing import Optional
import logging

from utils.configuration import ConfigurationManager
from utils.building_constraints import find_largest_polygon

from core.models import APGroupData, ProcessingResult
from core.data_loader import DataLoader
from core.preprocessor import DataPreprocessor
from core.signal_model import SignalModelOptimizer
from core.position_estimator import APPositionEstimator
from core.result_manager import ResultManager

logger = logging.getLogger(__name__)


class APLocalizationSystem:
    """AP localization system main controller - coordinates all components to complete the entire localization process"""

    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            config_file = './config/ap_localization_config.yaml'
        self.config_manager = ConfigurationManager(config_file)
        self.data_loader = DataLoader(self.config_manager)
        self.preprocessor = DataPreprocessor(self.config_manager)
        self.signal_optimizer = SignalModelOptimizer(self.config_manager)
        self.position_estimator = APPositionEstimator(self.config_manager)
        self.result_manager = ResultManager(self.config_manager)
        self._logger = logging.getLogger(__name__)
        
    def run(self) -> ProcessingResult:
        self._logger.info("Starting AP localization system")
        self._logger.info("Step 1: Data loading")
        (ap_to_position, ap_level, target_ap, _, all_mac,
         polygons, polygon_edges, ap_learn_flag) = self.data_loader.load_osm_data()
        self._logger.info("Step 2: Signal propagation model optimization")
        rssi_0_opt, n_opt, ave_val = self.signal_optimizer.optimize_propagation_model(
            target_ap, ap_to_position, polygon_edges, ap_level, ap_learn_flag
        )

        self._logger.info("Step 3: Data preprocessing")
        ap_groups = self.preprocessor.group_aps_by_prefix(target_ap, ap_learn_flag)
        ap_groups = self.preprocessor.filter_cross_floor_signals(ap_groups)
        ap_groups = self.preprocessor.apply_trajectory_filtering(ap_groups)
        self._logger.info("Step 4: Building analysis")
        largest_area = find_largest_polygon(polygons)
        self._logger.info("Step 5: AP position estimation may take several minutes; please wait...")
        result = self.position_estimator.estimate_ap_positions(
            ap_groups, largest_area, polygons, polygon_edges, rssi_0_opt, n_opt, ave_val, ap_to_position, ap_level
        )
        self._logger.info("Step 6: Result saving")
        self.result_manager.save_results(result.estimated_positions, ap_groups)
        self.result_manager.print_statistics(result.statistics)
        self._logger.info("AP localization system completed")
        return result


def main(config_file: Optional[str] = None):
    try:
        system = APLocalizationSystem(config_file)
        result = system.run()

        logger.info("Successfully localized %d APs", len(result.estimated_positions))
    except Exception as e:
        logger.error("System error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AP Localization System - Object-Oriented Refactored Version')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Configuration file path (supports .yaml and .json formats)')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    main(args.config)

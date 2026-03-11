from typing import Dict, List
import logging

from utils.configuration import ConfigurationManager
from io_layer.osm_parser import OsmDataParser  # for typing reference if needed
from io_layer.osm_writer import save_estimated_positions_to_osm
from core.models import APGroupData


class ResultManager:
    """Result manager - responsible for result saving and output"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager.get_config()
        self._logger = logging.getLogger(__name__)

    def save_results(self, estimated_positions: Dict[str, List[float]], ap_groups: Dict[str, APGroupData]) -> None:
        self._save_to_osm_format(estimated_positions, ap_groups)

    def _save_to_osm_format(self, estimated_positions: Dict[str, List[float]], ap_groups: Dict[str, APGroupData]) -> None:
        osm_output_file = self.config.file_paths.output_osm_file
        template_file = self.config.file_paths.template_osm_file
        ap_groups_dict = {}
        for key, group in ap_groups.items():
            if key in estimated_positions:
                ap_groups_dict[key] = {
                    'mac': group.mac,
                    'pos': group.pos,
                    'rssis': group.rssis,
                    'target_ap_level': group.target_ap_level,
                    'ap_level_est': group.ap_level,
                }
        save_estimated_positions_to_osm(
            estimated_positions=estimated_positions,
            ap_groups=ap_groups_dict,
            output_file=osm_output_file,
            template_file=template_file,
        )

    def print_statistics(self, statistics: Dict[str, float]) -> None:
        """Pretty-print localization accuracy statistics"""
        if not statistics:
            return
        self._logger.info("Localization accuracy results:")
        self._logger.info("   - Final localization average error: %.2f ± %.2f meters", statistics['final_mean'], statistics['final_std'])
        self._logger.info("   - Initial localization average error: %.2f ± %.2f meters", statistics['initial_mean'], statistics['initial_std'])
        self._logger.info("   - Accuracy improvement: %.2f meters", statistics['improvement'])
        self._logger.info("   - Improvement percentage: %.1f%%", statistics['improvement_percent'])
        self._logger.info("   - Final error median: %.2f meters", statistics['final_median'])
        self._logger.info("   - 95%% localization error ≤ %.2f meters", statistics['percentile_95']) 
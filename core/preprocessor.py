from collections import Counter
from typing import Dict
import numpy as np

from utils.configuration import ConfigurationManager
from utils.trajectory_filter import filter_trajectory_data
from utils.data_processing import deduplicate_positions_with_avg_rssi
from core.models import APGroupData


class DataPreprocessor:
    """Data preprocessor - responsible for data cleaning, filtering and preprocessing"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager.get_config()

    def group_aps_by_prefix(self, target_ap_list: Dict, ap_learn_flag: Dict) -> Dict[str, APGroupData]:
        ap_groups: Dict[str, APGroupData] = {}
        for tagap in target_ap_list:
            for mac in target_ap_list[tagap]['mac']:
                if mac not in ap_learn_flag or ap_learn_flag[mac] is False:
                    ap_group_key = mac[:-1]
                    if ap_group_key not in ap_groups:
                        ap_groups[ap_group_key] = APGroupData(
                            mac=[], pos=[], rssis=[], target_ap_level=[], ap_level=0
                        )
                    group = ap_groups[ap_group_key]
                    group.mac.append(mac)
                    floor_height = self.config.floor_height
                    group.pos.append([tagap[0], tagap[1], (target_ap_list[tagap]['level'] - 1) * floor_height])
                    group.rssis.append(target_ap_list[tagap]['mac'][mac])
                    group.target_ap_level.append(target_ap_list[tagap]['level'])
        return ap_groups

    def filter_cross_floor_signals(self, ap_groups: Dict[str, APGroupData]) -> Dict[str, APGroupData]:
        for _, group in ap_groups.items():
            counter = Counter(group.target_ap_level)
            ap_appear_level = max(counter, key=counter.get)
            group.ap_level = ap_appear_level
            pos_array = np.array(group.pos)
            sig_array = np.array(group.rssis)
            level_array = np.array(group.target_ap_level)
            mask = level_array == ap_appear_level
            group.pos = pos_array[mask].tolist()
            group.rssis = sig_array[mask].tolist()
        return ap_groups

    def apply_trajectory_filtering(self, ap_groups: Dict[str, APGroupData]) -> Dict[str, APGroupData]:
        filter_config = self.config.trajectory_filter
        for _, group in ap_groups.items():
            if filter_config.enable_filter:
                filtered_positions, filtered_rssis = filter_trajectory_data(
                    positions=group.pos,
                    rssis=group.rssis,
                    min_distance=filter_config.min_distance,
                    rssi_threshold=filter_config.rssi_threshold,
                    max_samples=filter_config.max_samples,
                )
            else:
                filtered_positions = group.pos
                filtered_rssis = group.rssis
            unique_pos_tuples, averages = deduplicate_positions_with_avg_rssi(filtered_positions, filtered_rssis)
            group.pos = unique_pos_tuples
            group.rssis = averages
        return ap_groups 
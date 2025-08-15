from typing import Dict, Tuple, Optional

from utils.configuration import ConfigurationManager
from utils.wall_learning import process_rssi_data, process_wall_line
from algorithms.rssi_optimizer import RSSIOptimizer


class SignalModelOptimizer:
    """Signal propagation model optimizer - responsible for RSSI propagation parameter optimization"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager.get_config()
        self.optimizer: Optional[RSSIOptimizer] = None

    def optimize_propagation_model(
        self,
        target_ap_list: Dict,
        ap_to_position: Dict,
        polygon_edges: Dict,
        ap_level: Dict,
        ap_learn_flag: Dict,
    ) -> Tuple[float, float, float]:
        learn_level = self.config.learn_level
        lines, learn_no_wall_dis_rss, _ = process_rssi_data(
            target_ap_list, ap_to_position, polygon_edges, ap_level, learn_level, ap_learn_flag
        )
        _, learn_parameter, _ = process_wall_line(lines, polygon_edges, learn_level)
        self.optimizer = RSSIOptimizer(learn_no_wall_dis_rss, learn_parameter)
        no_wall_result, _, _ = self.optimizer.optimize_no_wall()
        rssi_0_opt, n_opt = no_wall_result.x
        learn_parameter, ave_val = self.optimizer.optimize_wall_attenuation(rssi_0_opt, n_opt)
        return rssi_0_opt, n_opt, ave_val 
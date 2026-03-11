from typing import Dict, List
import numpy as np
from shapely.geometry import LineString
from shapely.ops import nearest_points

from utils.configuration import ConfigurationManager
from utils.building_constraints import extract_building_constraint_polygons
from utils.geometry import calculate_precise_distance
from utils.signal import rssi_to_distance
from algorithms.point_estimator import PointEstimator
from core.models import APGroupData, ProcessingResult


class APPositionEstimator:
    """AP position estimator - responsible for AP position estimation and optimization"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager.get_config()

    def _process_single_ap_group(
        self,
        ap_group_key: str,
        group: APGroupData,
        largest_area,
        polygons,
        polygon_edges,
        confidence_config,
        rssi_0_opt: float,
        n_opt: float,
        ave_val: float,
        iter_num_total: int,
    ):
        positions = group.pos
        rssis = group.rssis
        initial_lon = np.mean([pos[0] for pos in positions])
        initial_lat = np.mean([pos[1] for pos in positions])
        ap_level_est = group.ap_level
        floor_height = self.config.floor_height if self.config else 3.2
        initial_alt = ap_level_est * floor_height
        initial_guess = [initial_lon, initial_lat, initial_alt]
        distances = np.array([rssi_to_distance(rssi, A=rssi_0_opt, n=n_opt) for rssi in rssis])
        building_constraint_polygons = extract_building_constraint_polygons(largest_area, ap_level_est)
        constraint_polygons = building_constraint_polygons if building_constraint_polygons else polygons
        estimator = PointEstimator(positions, distances, constraint_polygons, rssis=rssis, confidence_config=confidence_config)
        result_one = estimator.estimate_point(initial_guess=initial_guess)
        initial_result = result_one
        if result_one is None or not hasattr(result_one, 'x'):
            return {
                'estimated_position': initial_guess,
                'initial_result': None,
                'initial_positioning_data_entry': {
                    'positions': positions,
                    'rssis': rssis,
                    'initial_pos': initial_guess,
                },
            }
        initial_positioning_data_entry = {
            'positions': positions,
            'rssis': rssis,
            'initial_pos': result_one.x,
        }
        for _ in range(1, iter_num_total):
            temp_rssis = list(rssis)
            flags = np.zeros(len(positions))
            for pos_num, pos_ap in enumerate(positions):
                if result_one is None or not hasattr(result_one, 'x'):
                    break
                ap_appear_level = group.target_ap_level[0]
                ap_line = LineString([(pos_ap[0], pos_ap[1]), (result_one.x[0], result_one.x[1])])
                closest_edge = None
                closest_distance = float('inf')
                ap_intersection = None
                for edge in polygon_edges.get(ap_appear_level, []):
                    if ap_line.intersects(edge):
                        ap_intersection = ap_line.intersection(edge)
                        ap_intersection_line_distance = calculate_precise_distance(pos_ap[0], pos_ap[1], ap_intersection.x, ap_intersection.y)
                        if ap_intersection_line_distance < closest_distance:
                            closest_distance = ap_intersection_line_distance
                            closest_edge = edge
                if closest_edge is not None:
                    if largest_area is not None and ap_intersection is not None:
                        current_level = ap_level_est
                        if current_level in largest_area and largest_area[current_level]:
                            boundary = largest_area[current_level][0]['polygon'].boundary
                            nearest_points(ap_intersection, boundary)[1]
                    distance_threshold = self.config.wall_detection.distance_threshold if self.config else 10
                    rssi_threshold = self.config.wall_detection.rssi_threshold if self.config else -30
                    if closest_distance < distance_threshold and temp_rssis[pos_num] < rssi_threshold:
                        flags[pos_num] = 1
                        temp_rssis[pos_num] = temp_rssis[pos_num] - ave_val
            filter_dis = []
            filter_pos = []
            err_rssi = []
            for rssi_val, pos in zip(temp_rssis, positions):
                filter_dis.append(rssi_to_distance(rssi_val, A=rssi_0_opt, n=n_opt))
                filter_pos.append(pos)
                err_rssi.append(rssi_val)
            if len(filter_dis) < 3:
                distances = np.array([rssi_to_distance(rv, A=rssi_0_opt, n=n_opt) for rv in temp_rssis])
                estimator = PointEstimator(positions, distances, constraint_polygons, rssis=temp_rssis, confidence_config=confidence_config)
                result = estimator.estimate_point(initial_guess=initial_guess)
            else:
                initial_lon = np.mean([pos[0] for pos in filter_pos])
                initial_lat = np.mean([pos[1] for pos in filter_pos])
                initial_alt = ap_level_est * floor_height
                initial_guess = [initial_lon, initial_lat, initial_alt]
                estimator = PointEstimator(filter_pos, filter_dis, constraint_polygons, rssis=err_rssi, confidence_config=confidence_config)
                result = estimator.estimate_point(initial_guess=initial_guess)
            result_one = result
        estimated_position = initial_guess if (result_one is None or not hasattr(result_one, 'x')) else [result_one.x[0], result_one.x[1], result_one.x[2]]
        return {
            'estimated_position': estimated_position,
            'initial_result': initial_result,
            'initial_positioning_data_entry': initial_positioning_data_entry,
        }

    def estimate_ap_positions(
        self,
        ap_groups: Dict[str, APGroupData],
        largest_area: Dict,
        polygons: List,
        polygon_edges: Dict,
        rssi_0_opt: float,
        n_opt: float,
        ave_val: float,
        ap_to_position: Dict,
        ap_level: Dict,
    ) -> ProcessingResult:
        estimated_positions: Dict[str, List[float]] = {}
        error_array: List[float] = []
        initial_errors: List[float] = []
        confidence_config = {
            'enable_weighting': self.config.confidence.enable_weighting,
            'strong_signal_threshold': self.config.confidence.strong_signal_threshold,
            'weak_signal_threshold': self.config.confidence.weak_signal_threshold,
            'max_weight': self.config.confidence.max_weight,
            'min_weight': self.config.confidence.min_weight,
            'weight_method': self.config.confidence.weight_method,
        }
        iter_num_total = self.config.optimization.max_iterations
        for ap_group_key, group in ap_groups.items():
            if len(group.pos) >= 3:
                res = self._process_single_ap_group(
                    ap_group_key,
                    group,
                    largest_area,
                    polygons,
                    polygon_edges,
                    confidence_config,
                    rssi_0_opt,
                    n_opt,
                    ave_val,
                    iter_num_total,
                )
                if group.mac and group.mac[0] in ap_to_position:
                    estimated_positions[ap_group_key] = res['estimated_position']
                    if group.mac[0] in ap_level:
                        mac = group.mac[0]
                        gt_position = ap_to_position[mac]
                        gt_level = ap_level[mac]
                        horizontal_error = calculate_precise_distance(
                            res['estimated_position'][0], res['estimated_position'][1], gt_position[0], gt_position[1]
                        )
                        vertical_error = abs(gt_level * self.config.floor_height - res['estimated_position'][2])
                        final_error = np.sqrt(horizontal_error ** 2 + vertical_error ** 2)
                        if res['initial_result'] is not None and hasattr(res['initial_result'], 'x'):
                            initial_horizontal_error = calculate_precise_distance(
                                res['initial_result'].x[0], res['initial_result'].x[1], gt_position[0], gt_position[1]
                            )
                            initial_vertical_error = abs(gt_level * self.config.floor_height - res['initial_result'].x[2])
                            initial_error = np.sqrt(initial_horizontal_error ** 2 + initial_vertical_error ** 2)
                            initial_errors.append(initial_error)
                        error_array.append(final_error)
        statistics = self._calculate_statistics(error_array, initial_errors)
        return ProcessingResult(
            estimated_positions=estimated_positions,
            error_array=error_array,
            initial_errors=initial_errors,
            statistics=statistics,
        )

    def _calculate_statistics(self, final_errors: List[float], initial_errors: List[float]) -> Dict[str, float]:
        if not final_errors:
            return {}
        final_errors_arr = np.array(final_errors)
        initial_errors_arr = np.array(initial_errors) if initial_errors else np.array([0.0])
        return {
            'final_mean': float(np.mean(final_errors_arr)),
            'final_std': float(np.std(final_errors_arr)),
            'final_median': float(np.median(final_errors_arr)),
            'initial_mean': float(np.mean(initial_errors_arr)),
            'initial_std': float(np.std(initial_errors_arr)),
            'improvement': float(np.mean(initial_errors_arr) - np.mean(final_errors_arr)),
            'improvement_percent': float(((np.mean(initial_errors_arr) - np.mean(final_errors_arr)) / max(np.mean(initial_errors_arr), 1e-9) * 100)),
            'percentile_95': float(np.percentile(final_errors_arr, 95)),
        } 
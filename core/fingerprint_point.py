from typing import Dict, List
import copy
import numpy as np
from shapely.geometry import Point, LineString
import logging

from algorithms.point_estimator import PointEstimator
from utils.signal import rssi_to_distance
from utils.geometry import calculate_precise_distance


class WiFiFingerprintLocalizer:
    """AP Position-based WiFi Fingerprint Point Localizer"""

    def __init__(self, ap_positions: Dict, polygon_edges: Dict, rssi_0: float = -30, n: float = 2.0, ave_val: float = -15):
        """
        Initialize Fingerprint Point Localizer

        Args:
            ap_positions: AP position dictionary
            polygon_edges: Polygon boundary data
            rssi_0: RSSI propagation model parameter
            n: Path loss exponent
            ave_val: Wall attenuation value
        """
        self.ap_positions = ap_positions
        self.polygon_edges = polygon_edges
        self.rssi_0 = rssi_0
        self.n = n
        self.ave_val = ave_val
        self._logger = logging.getLogger(__name__)

        # RSSI confidence weight configuration
        self.confidence_config = {
            'enable_weighting': True,
            'strong_signal_threshold': -10,
            'weak_signal_threshold': -80,
            'max_weight': 3.0,
            'min_weight': 0.3,
            'weight_method': 'exponential'
        }

    def _get_target_building_polygons(self, point_info: Dict, building_clusters_result: Dict = None) -> List:
        """
        Determine building cluster polygons where fingerprint point is located

        Args:
            point_info: Fingerprint point information dictionary
            building_clusters_result: Building cluster results

        Returns:
            List of building polygons containing the fingerprint point
        """
        if not building_clusters_result or point_info['level'] not in building_clusters_result:
            return []

        point = Point(point_info['position'][0], point_info['position'][1])
        target_polygons = []

        # Check which building cluster contains the fingerprint point
        for cluster in building_clusters_result[point_info['level']]:
            polygon = cluster['polygon']
            if polygon.contains(point) or polygon.touches(point):
                target_polygons.append(polygon)

        return target_polygons

    def _is_outer_wall(self, edge: LineString, level: int, building_clusters_result: Dict = None) -> bool:
        """
        Determine if an edge is an outer wall of a building

        Args:
            edge: Boundary line segment
            level: Floor level
            building_clusters_result: Building cluster results

        Returns:
            Whether it is an outer wall
        """
        if not building_clusters_result or level not in building_clusters_result:
            return True  # If no building cluster data, default to outer wall

        # Check if edge belongs to any building's outer boundary
        for cluster in building_clusters_result[level]:
            polygon = cluster['polygon']
            exterior_boundary = polygon.boundary

            # Check if edge intersects or overlaps with building outer boundary
            if isinstance(exterior_boundary, LineString):
                if edge.intersects(exterior_boundary) or edge.overlaps(exterior_boundary):
                    return True
            elif hasattr(exterior_boundary, 'geoms'):
                for boundary_part in exterior_boundary.geoms:
                    if edge.intersects(boundary_part) or edge.overlaps(boundary_part):
                        return True

        return False

    def localize_fingerprint_points(self, fingerprint_data: Dict, polygons: Dict, building_clusters_result: Dict = None, iter_num_total: int = 10) -> Dict:
        """Localize fingerprint point positions - using iterative optimization method with building clusters"""
        results = {}

        for point_name, point_info in fingerprint_data.items():
            if not point_info['rssi_data']:
                continue

            # Determine building clusters where fingerprint point is located
            target_building_polygons = self._get_target_building_polygons(point_info, building_clusters_result)
            if not target_building_polygons:
                # Fall back to using all polygons on this floor
                if building_clusters_result and point_info['level'] in building_clusters_result:
                    target_building_polygons = [cluster['polygon'] for cluster in building_clusters_result[point_info['level']]]
                else:
                    target_building_polygons = [poly[0] for poly in polygons if int(poly[1]) == point_info['level']]

            # Collect available AP positions and corresponding RSSI values
            ap_positions_list = []
            rssi_values = []

            for mac, rssi in point_info['rssi_data'].items():
                # Find corresponding AP position
                ap_found = False
                for _, ap_info in self.ap_positions.items():
                    if mac in ap_info['mac_list']:
                        # Prioritize floor matching, ignore floor constraint if no match found
                        if ap_info['level'] == point_info['level'] or not ap_found:
                            ap_positions_list.append(ap_info['position'])
                            rssi_values.append(rssi)
                            ap_found = True
                            if ap_info['level'] == point_info['level']:
                                break

            # Integrate APs at same positions and sort by signal strength
            if ap_positions_list:
                position_rssi_map = {}
                for i, position in enumerate(ap_positions_list):
                    pos_key = (round(position[0], 6), round(position[1], 6))
                    if pos_key not in position_rssi_map:
                        position_rssi_map[pos_key] = {
                            'position': position,
                            'rssi': rssi_values[i]
                        }
                    else:
                        if rssi_values[i] > position_rssi_map[pos_key]['rssi']:
                            position_rssi_map[pos_key] = {
                                'position': position,
                                'rssi': rssi_values[i]
                            }
                sorted_positions = sorted(position_rssi_map.values(), key=lambda x: x['rssi'], reverse=True)
                max_aps = min(3, len(sorted_positions))
                selected_positions = sorted_positions[:max_aps]
                ap_positions_list = [item['position'] for item in selected_positions]
                rssi_values = [item['rssi'] for item in selected_positions]

            if len(ap_positions_list) < 3:
                continue

            try:
                constraint_polygons = []
                for polygon in target_building_polygons:
                    constraint_polygons.append((polygon, point_info['level']))

                result = self._iterative_localization(
                    ap_positions_list,
                    rssi_values,
                    point_info['level'],
                    constraint_polygons,
                    building_clusters_result,
                    iter_num_total
                )

                if result and result.success:
                    predicted_position = result.x
                    true_position = point_info['position']
                    error_meters = calculate_precise_distance(
                        predicted_position[0], predicted_position[1],
                        true_position[0], true_position[1]
                    )
                    results[point_name] = {
                        'true_position': true_position,
                        'predicted_position': (predicted_position[0], predicted_position[1]),
                        'error_meters': error_meters,
                        'level': point_info['level'],
                        'num_aps': len(ap_positions_list),
                        'rssi_range': (min(rssi_values), max(rssi_values))
                    }
                else:
                    pass
            except Exception as exc:
                self._logger.exception("Failed to localize fingerprint point %s", point_name)

        return results

    def _iterative_localization(self, ap_positions_list: List, rssi_values: List,
                                point_level: int, constraint_polygons: List, building_clusters_result: Dict = None,
                                iter_num_total: int = 10):
        """
        Iterative localization method - based on iterative optimization logic from ap_loc.py
        Uses building clusters to constrain the localization result within building boundaries

        Args:
            ap_positions_list: List of AP positions
            rssi_values: List of RSSI values
            point_level: Fingerprint point floor level
            constraint_polygons: Constraint polygon data (specific building boundaries)
            building_clusters_result: Building clusters result for wall detection
            iter_num_total: Number of iterations

        Returns:
            Final localization result
        """
        initial_distances = np.array([rssi_to_distance(rssi, A=self.rssi_0, n=self.n) for rssi in rssi_values])

        ap_positions_array = np.array(ap_positions_list)
        initial_guess = [
            np.mean(ap_positions_array[:, 0]),
            np.mean(ap_positions_array[:, 1]),
            point_level * 3.2
        ]

        initial_estimator = PointEstimator(
            ap_positions_list,
            initial_distances,
            constraint_polygons,
            rssis=rssi_values,
            confidence_config=self.confidence_config
        )

        initial_result = initial_estimator.estimate_point(initial_guess=initial_guess)
        if not initial_result.success:
            return None

        current_result = initial_result
        initial_error = self._calculate_localization_error(current_result.x, ap_positions_list, rssi_values)

        for iter_num in range(1, iter_num_total):
            corrected_rssis, _ = self._detect_walls_and_correct_rssi(
                ap_positions_list, rssi_values, current_result.x, point_level, building_clusters_result
            )
            corrected_distances = np.array([rssi_to_distance(rssi, A=self.rssi_0, n=self.n) for rssi in corrected_rssis])
            valid_positions, valid_distances, valid_rssis = self._filter_valid_points(
                ap_positions_list, corrected_distances, corrected_rssis, min_points=3
            )
            if len(valid_positions) < 3:
                valid_positions = ap_positions_list
                valid_distances = corrected_distances
                valid_rssis = corrected_rssis

            updated_initial_guess = [
                np.mean([pos[0] for pos in valid_positions]),
                np.mean([pos[1] for pos in valid_positions]),
                point_level * 3.2
            ]

            estimator = PointEstimator(
                valid_positions,
                valid_distances,
                constraint_polygons,
                rssis=valid_rssis,
                confidence_config=self.confidence_config
            )

            result = estimator.estimate_point(initial_guess=updated_initial_guess)
            if result.success:
                current_error = self._calculate_localization_error(result.x, ap_positions_list, rssi_values)
                _ = initial_error - current_error
                current_result = result
            else:
                pass

        final_error = self._calculate_localization_error(current_result.x, ap_positions_list, rssi_values)
        _ = initial_error - final_error
        return current_result

    def _detect_walls_and_correct_rssi(self, ap_positions_list: List, rssi_values: List,
                                       point_position: List, point_level: int, building_clusters_result: Dict = None):
        """
        Detect wall obstruction and correct RSSI values using building clusters
        """
        corrected_rssis = copy.deepcopy(rssi_values)
        wall_corrections = 0
        correction_details = []

        if point_level not in self.polygon_edges:
            return corrected_rssis, {'wall_corrections': 0, 'details': []}

        for i, (ap_pos, original_rssi) in enumerate(zip(ap_positions_list, rssi_values)):
            point_ap_line = LineString([(point_position[0], point_position[1]), (ap_pos[0], ap_pos[1])])
            closest_edge = None
            closest_distance = float('inf')
            intersection_point = None

            for edge in self.polygon_edges.get(point_level, []):
                if point_ap_line.intersects(edge):
                    intersection = point_ap_line.intersection(edge)
                    if hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                        intersection_distance = calculate_precise_distance(
                            point_position[0], point_position[1], intersection.x, intersection.y
                        )
                        if intersection_distance < closest_distance:
                            closest_distance = intersection_distance
                            closest_edge = edge
                            intersection_point = intersection

            should_correct = False
            if closest_edge is not None and intersection_point is not None:
                edge_coords = list(closest_edge.coords)
                if len(edge_coords) >= 2:
                    from utils.geometry import calculate_precise_distance as _dist
                    edge_length = _dist(edge_coords[0][0], edge_coords[0][1], edge_coords[-1][0], edge_coords[-1][1])
                    is_outer_wall = self._is_outer_wall(closest_edge, point_level, building_clusters_result)
                    if closest_distance < 10 and original_rssi < -30 and edge_length > 1:
                        if is_outer_wall:
                            should_correct = True
                        elif closest_distance < 5:
                            should_correct = True

            if should_correct:
                corrected_rssis[i] = original_rssi - self.ave_val
                wall_corrections += 1
                correction_details.append({
                    'ap_index': i,
                    'original_rssi': original_rssi,
                    'corrected_rssi': corrected_rssis[i],
                    'wall_attenuation': self.ave_val,
                    'intersection_distance': closest_distance
                })

        correction_info = {'wall_corrections': wall_corrections, 'details': correction_details}
        return corrected_rssis, correction_info

    def _filter_valid_points(self, positions: List, distances: List, rssis: List,
                             min_points: int = 3, max_distance: float = 100.0):
        """Filter valid localization points"""
        valid_indices = []
        for i, (_, dist, rssi) in enumerate(zip(positions, distances, rssis)):
            if dist <= max_distance and -100 <= rssi <= -20:
                valid_indices.append(i)
        if len(valid_indices) < min_points:
            valid_indices = list(range(len(positions)))
        valid_positions = [positions[i] for i in valid_indices]
        valid_distances = [distances[i] for i in valid_indices]
        valid_rssis = [rssis[i] for i in valid_indices]
        return valid_positions, valid_distances, valid_rssis

    def _calculate_localization_error(self, estimated_position: List, ap_positions_list: List, rssi_values: List):
        """Calculate localization error metric (for iterative optimization evaluation)"""
        total_error = 0.0
        for ap_pos, rssi in zip(ap_positions_list, rssi_values):
            actual_distance = calculate_precise_distance(estimated_position[0], estimated_position[1], ap_pos[0], ap_pos[1])
            theoretical_distance = rssi_to_distance(rssi, A=self.rssi_0, n=self.n)
            distance_error = abs(actual_distance - theoretical_distance)
            total_error += distance_error
        return total_error / len(ap_positions_list) if ap_positions_list else float('inf')


def analyze_fingerprint_results(results: Dict):
    """Analyze fingerprint point localization results"""
    if not results:
        return
    errors = [r['error_meters'] for r in results.values()]
    logging.info("Total test points: %d", len(results))
    logging.info("Average localization error: %.2fm", np.mean(errors))
    logging.info("Maximum localization error: %.2fm", np.max(errors))
    logging.info("Minimum localization error: %.2fm", np.min(errors))
    logging.info("Standard deviation: %.2fm", np.std(errors))

from typing import Dict, Tuple, Optional
import os
import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KNeighborsRegressor
import logging

from utils.geometry import calculate_precise_distance
from utils.building_constraints import find_largest_polygon
from io_layer.osm_parser import OsmDataParser
from utils.configuration import ConfigurationManager


class WiFiKNNLocalizer:
    """KNN-based WiFi Localizer with Building Boundary Constraints"""

    def __init__(self, fingerprint_data: Dict, all_macs: set, k: int = 5, polygon_file: str = None, use_boundary_constraint: bool = True):
        """
        Initialize KNN localizer with boundary constraints

        Args:
            fingerprint_data: Fingerprint database data
            all_macs: Set of all MAC addresses
            k: K value for KNN algorithm
            polygon_file: OSM file containing building polygon data
            use_boundary_constraint: Whether to use boundary constraints
        """
        self.fingerprint_data = fingerprint_data
        self.all_macs = sorted(list(all_macs))
        self.k = k
        self.feature_matrix = None
        self.position_matrix = None
        self.knn_model = None
        self.use_boundary_constraint = use_boundary_constraint
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_config()
        self.polygons = []
        self.building_clusters = {}
        self.polygon_file = polygon_file
        self._logger = logging.getLogger(__name__)
        if self.use_boundary_constraint and polygon_file:
            self._load_boundary_data()
        self._build_feature_matrix()
        self._train_knn_model()

    def _load_boundary_data(self):
        """Load and process building boundary data"""
        if not self.polygon_file or not os.path.exists(self.polygon_file):
            self.use_boundary_constraint = False
            return
        try:
            polygon_parser = OsmDataParser(self.polygon_file)
            polygon_parser.parse()
            polygon_parser.extract_polygon_edges()
            _, _, _, _, _, polygons, _, _ = polygon_parser.get_data()
            self.polygons = polygons
            self.building_clusters = find_largest_polygon(polygons)
        except Exception as exc:
            self._logger.exception("Failed to load boundary data from %s; disabling boundary constraints", self.polygon_file)
            self.use_boundary_constraint = False

    def _find_valid_boundary_for_position(self, position: Tuple[float, float], level: int) -> Optional[Polygon]:
        if not self.use_boundary_constraint or level not in self.building_clusters:
            return None
        point = Point(position[0], position[1])
        for cluster in self.building_clusters[level]:
            if cluster['polygon'] and cluster['polygon'].contains(point):
                return cluster['polygon']
        min_distance = float('inf')
        closest_polygon = None
        for cluster in self.building_clusters[level]:
            if cluster['polygon']:
                distance = point.distance(cluster['polygon'])
                if distance < min_distance:
                    min_distance = distance
                    closest_polygon = cluster['polygon']
        return closest_polygon

    def _constrain_position_to_boundary(self, predicted_position: Tuple[float, float], level: int) -> Tuple[float, float]:
        if not self.use_boundary_constraint:
            return predicted_position
        boundary_polygon = self._find_valid_boundary_for_position(predicted_position, level)
        if boundary_polygon is None:
            return predicted_position
        point = Point(predicted_position[0], predicted_position[1])
        if boundary_polygon.contains(point):
            return predicted_position
        try:
            nearest_point = boundary_polygon.exterior.interpolate(boundary_polygon.exterior.project(point))
            constrained_position = (nearest_point.x, nearest_point.y)
            original_distance = calculate_precise_distance(
                predicted_position[0], predicted_position[1], constrained_position[0], constrained_position[1]
            )
            if original_distance > self.config.fingerprint_knn.boundary_adjust_log_threshold_m:
                pass
            return constrained_position
        except Exception as exc:
            self._logger.warning("Failed to constrain position to boundary at level %s; using unconstrained position", level)
            return predicted_position

    def _build_feature_matrix(self):
        positions = []
        features = []
        for position, data in self.fingerprint_data.items():
            feature_vector = []
            for mac in self.all_macs:
                if mac in data['mac'] and data['mac'][mac]:
                    avg_rssi = np.mean(data['mac'][mac])
                    feature_vector.append(avg_rssi)
                else:
                    feature_vector.append(self.config.fingerprint_knn.missing_rssi_fill)
            positions.append(position)
            features.append(feature_vector)
        self.position_matrix = np.array(positions)
        self.feature_matrix = np.array(features)

    def _train_knn_model(self):
        self.knn_model = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        self.knn_model.fit(self.feature_matrix, self.position_matrix)

    def _extract_wifi_features_from_fingerprint(self, wifi_data: Dict) -> Optional[np.ndarray]:
        try:
            feature_vector = []
            for mac in self.all_macs:
                if mac in wifi_data and wifi_data[mac]:
                    avg_rssi = np.mean(wifi_data[mac])
                    feature_vector.append(avg_rssi)
                else:
                    feature_vector.append(self.config.fingerprint_knn.missing_rssi_fill)
            return np.array(feature_vector).reshape(1, -1)
        except Exception as exc:
            self._logger.exception("Failed to extract WiFi features for fingerprint")
            return None

    def localize(self, test_points: Dict) -> Dict:
        results = {}
        for point_name, point_data in test_points.items():
            features = self._extract_wifi_features_from_fingerprint(point_data['wifi_data'])
            if features is None:
                continue
            predicted_position = self.knn_model.predict(features)[0]
            predicted_position_tuple = tuple(predicted_position)
            if self.use_boundary_constraint:
                constrained_position = self._constrain_position_to_boundary(predicted_position_tuple, point_data['level'])
            else:
                constrained_position = predicted_position_tuple
            true_position = point_data['position']
            unconstrained_error = calculate_precise_distance(
                lon1=predicted_position[0], lat1=predicted_position[1], lon2=true_position[0], lat2=true_position[1]
            )
            constrained_error = calculate_precise_distance(
                lon1=constrained_position[0], lat1=constrained_position[1], lon2=true_position[0], lat2=true_position[1]
            )
            results[point_name] = {
                'true_position': true_position,
                'predicted_position_unconstrained': predicted_position_tuple,
                'predicted_position_constrained': constrained_position,
                'error_unconstrained': unconstrained_error,
                'error_constrained': constrained_error,
                'level': point_data['level'],
                'boundary_applied': self.use_boundary_constraint and constrained_position != predicted_position_tuple
            }
            if self.use_boundary_constraint:
                improvement = unconstrained_error - constrained_error
                if improvement > 0:
                    pass
                elif improvement < 0:
                    pass
            else:
                pass
        return results


def calculate_statistical_metrics(results: Dict) -> Dict:
    """Calculate statistical metrics for localization results"""
    if not results:
        return {}
    has_boundary_constraints = any(r.get('boundary_applied', False) for r in results.values())
    if has_boundary_constraints:
        errors = [r['error_constrained'] for r in results.values()]
    else:
        errors = [r['error_unconstrained'] for r in results.values()]
    mean_error = np.mean(errors)
    std_dev = np.std(errors)
    median_error = np.median(errors)
    percentile_95 = np.percentile(errors, 95)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    logging.info("Total test points: %d", len(results))
    logging.info("Mean Error (m): %.2f", mean_error)
    logging.info("Std. Dev. (m): %.2f", std_dev)
    logging.info("95%% Percentile (m): %.2f", percentile_95)
    logging.info("RMSE (m): %.2f", rmse)
    min_error = np.min(errors)
    max_error = np.max(errors)
    metrics = {
        'total_points': len(results),
        'mean_error': mean_error,
        'std_dev': std_dev,
        'median_error': median_error,
        'percentile_95': percentile_95,
        'rmse': rmse,
        'min_error': min_error,
        'max_error': max_error,
        'errors': errors
    }
    return metrics 
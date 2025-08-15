  
import numpy as np
from scipy.optimize import minimize, least_squares

from pyproj import Geod
import math
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

class PointEstimator:
    def __init__(self, known_points, distances, polygons, rssis=None, confidence_config=None):
        """
        Initialize PointEstimator class with polygon boundary constraints
        
        :param known_points: List of 3D coordinates of known points, format [[lon1, lat1, alt1], [lon2, lat2, alt2], ...]
        :param distances: Distance from each known point to the point to be determined, format [distance1, distance2, ...]
        :param polygons: Polygon data - list of (polygon, level) tuples
        :param rssis: List of corresponding RSSI values (optional)
        :param confidence_config: Confidence configuration dictionary (optional)
        """
        self.known_points = np.array(known_points)
        self.distances = np.array(distances)
        self.polygons = polygons
        self.rssis = np.array(rssis) if rssis is not None else None
        self.confidence_config = confidence_config or {
            'enable_weighting': True,
            'strong_signal_threshold': -50,    # Strong signal threshold (dBm)
            'weak_signal_threshold': -80,      # Weak signal threshold (dBm)
            'max_weight': 3.0,                 # Maximum weight
            'min_weight': 0.3,                 # Minimum weight
            'weight_method': 'exponential'     # Weight calculation method: 'linear', 'exponential', 'sigmoid'
        }
        self.weights = self._calculate_rssi_weights() if self.rssis is not None else None
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.find_min_max_coordinates()
        self.optimization_history = []
        
        # Extract polygon boundaries for constraints
        self.constraint_polygons = self._extract_constraint_polygons()
        self.use_polygon_constraints = len(self.constraint_polygons) > 0


    def _calculate_rssi_weights(self):
        """
        Calculate confidence weights based on RSSI values
        Stronger signals have higher confidence and higher weights
        
        Returns:
            numpy.ndarray: Weights array
        """
        if self.rssis is None or not self.confidence_config.get('enable_weighting', False):
            return np.ones(len(self.known_points))
        
        config = self.confidence_config
        strong_threshold = config.get('strong_signal_threshold', -50)
        weak_threshold = config.get('weak_signal_threshold', -80)
        max_weight = config.get('max_weight', 3.0)
        min_weight = config.get('min_weight', 0.3)
        method = config.get('weight_method', 'exponential')
        
        # Normalize RSSI values to 0-1 range
        # Better signals (closer to 0) have higher normalized values
        rssi_normalized = np.clip(
            (self.rssis - weak_threshold) / (strong_threshold - weak_threshold),
            0, 1
        )
        
        if method == 'linear':
            # Linear weight mapping
            weights = min_weight + (max_weight - min_weight) * rssi_normalized
            
        elif method == 'exponential':
            # Exponential weight mapping, emphasizing strong signals
            weights = min_weight + (max_weight - min_weight) * np.power(rssi_normalized, 0.5)
            
        elif method == 'sigmoid':
            # S-shaped weight mapping, changes fastest at medium signal strength
            sigmoid_input = 10 * (rssi_normalized - 0.5)  # Adjust range to -5 to 5
            sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))
            weights = min_weight + (max_weight - min_weight) * sigmoid_output
            
        else:
            # Default use exponential method
            weights = min_weight + (max_weight - min_weight) * np.power(rssi_normalized, 0.5)
        
        # Ensure weights are within reasonable range
        weights = np.clip(weights, min_weight, max_weight)
        
        return weights

    def find_min_max_coordinates(self):
        """
        Find the boundary range of all polygons for use as initial bounds
        """
        min_lon, min_lat = float('inf'), float('inf')
        max_lon, max_lat = float('-inf'), float('-inf')

        for polygon_data in self.polygons:
            if isinstance(polygon_data, tuple) and len(polygon_data) >= 2:
                polygon = polygon_data[0]
            else:
                polygon = polygon_data
                
            if hasattr(polygon, 'exterior'):
                x, y = polygon.exterior.xy
                min_lon = min(min_lon, min(x))
                max_lon = max(max_lon, max(x))
                min_lat = min(min_lat, min(y))
                max_lat = max(max_lat, max(y))

        return min_lon, min_lat, max_lon, max_lat
    
    def _extract_constraint_polygons(self):
        """
        Extract polygon list for constraints

        Returns:
            List[Polygon]: Constraint polygon list
        """
        constraint_polygons = []
        if self.polygons:
            for polygon_data in self.polygons:
                if isinstance(polygon_data, tuple) and len(polygon_data) >= 2:
                    polygon, level = polygon_data[0], polygon_data[1]
                    if hasattr(polygon, 'exterior'):  # Shapely Polygon object
                        constraint_polygons.append(polygon)
                elif hasattr(polygon_data, 'exterior'):  # Direct Polygon object
                    constraint_polygons.append(polygon_data)
        return constraint_polygons

    def _is_point_in_constraint_area(self, lon, lat):
        """
        Check if point is within constraint area (within any polygon)

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            bool: Whether within constraint area
        """
        if not self.use_polygon_constraints:
            return True
            
        point = Point(lon, lat)
        
        # Check if point is within any constraint polygon
        for polygon in self.constraint_polygons:
            if polygon.contains(point) or polygon.touches(point):
                return True
        return False

    def _polygon_constraint_function(self, params):
        """
        Polygon constraint function for scipy.optimize.minimize

        Args:
            params: [lon, lat, alt] optimization parameters

        Returns:
            float: Constraint value, >= 0 means constraint satisfied, < 0 means constraint violated
        """
        lon, lat, alt = params
        
        if not self.use_polygon_constraints:
            return 1.0  # Return positive value when no constraints
            
        point = Point(lon, lat)
        
        # Find the nearest polygon distance
        min_distance_to_polygon = float('inf')

        for polygon in self.constraint_polygons:
            if polygon.contains(point) or polygon.touches(point):
                return 1.0  # Point is inside polygon, constraint satisfied
            else:
                # Calculate distance from point to polygon boundary
                distance_to_boundary = point.distance(polygon.boundary)
                min_distance_to_polygon = min(min_distance_to_polygon, distance_to_boundary)

        # Return negative distance, indicating degree of constraint violation
        return -min_distance_to_polygon

    def _penalized_objective_function(self, params):
        """
        Objective function with penalty term to ensure solution is within polygon

        Args:
            params: [lon, lat, alt] optimization parameters

        Returns:
            float: Penalized objective function value
        """
        lon, lat, alt = params

        # Base objective function (localization error)
        base_error = self.ap_residuals(params)

        # Polygon constraint penalty term
        if self.use_polygon_constraints:
            if not self._is_point_in_constraint_area(lon, lat):
                # Calculate distance to nearest polygon as penalty
                point = Point(lon, lat)
                min_distance = float('inf')

                for polygon in self.constraint_polygons:
                    distance_to_polygon = point.distance(polygon.boundary)
                    min_distance = min(min_distance, distance_to_polygon)

                # Add large penalty term (approximate conversion from degrees to meters)
                penalty = min_distance * 111000 * 1000  # Large penalty coefficient
                return base_error + penalty
        
        return base_error

    

   

    def ap_residuals(self, params):
        """
        Weighted residuals function
        Stronger measured points have higher weights
        """
        lon, lat, height = params
        res = 0
        geod = Geod(ellps='WGS84')
        
        for i, ((lon_i, lat_i, height_i), _distance) in enumerate(zip(self.known_points, self.distances)):
            _, _, distance2 = geod.inv(lon, lat, lon_i, lat_i)
            
            height_err = (height_i - height)**2
            
            d_estimated = math.sqrt(distance2**2 + height_err)
            error = abs(d_estimated - _distance)
            
            # Apply weights: Larger weights mean greater contribution to total error
            if self.weights is not None:
                weight = self.weights[i]
                weighted_error = error * weight
            else:
                weighted_error = error
                
            res += weighted_error
        return res

    def weighted_residuals_for_least_squares(self, params):
        """
        Weighted residuals function for least_squares
        Returns residuals vector instead of scalar sum
        """
        lon, lat, height = params
        residuals = []
        geod = Geod(ellps='WGS84')
        
        for i, ((lon_i, lat_i, height_i), _distance) in enumerate(zip(self.known_points, self.distances)):
            _, _, distance2 = geod.inv(lon, lat, lon_i, lat_i)
            
            height_err = (height_i - height)**2
            
            d_estimated = math.sqrt(distance2**2 + height_err)
            error = d_estimated - _distance
            
            # Apply weights: Larger weights mean greater influence of the measurement point
            if self.weights is not None:
                weight_factor = np.sqrt(self.weights[i])  # Square because least_squares squares residuals
                weighted_error = error * weight_factor
            else:
                weighted_error = error
                
            residuals.append(weighted_error)
        
        return np.array(residuals)

    def _find_valid_initial_guess(self, initial_guess):
        """
        Find a valid initial guess point within the constraint area

        Args:
            initial_guess: Original initial guess [lon, lat, alt]

        Returns:
            list: Valid initial guess point
        """
        lon, lat, alt = initial_guess
        
        if not self.use_polygon_constraints:
            return initial_guess

        # Check if original guess is already within constraint area
        if self._is_point_in_constraint_area(lon, lat):
            return initial_guess

        # Try to find near the centroid of constraint polygons
        for polygon in self.constraint_polygons:
            centroid = polygon.centroid
            candidate_lon, candidate_lat = centroid.x, centroid.y

            if self._is_point_in_constraint_area(candidate_lon, candidate_lat):
                return [candidate_lon, candidate_lat, alt]
        
        # If centroid doesn't work, try points on polygon boundary
        for polygon in self.constraint_polygons:
            # Get some points on polygon boundary
            boundary_coords = list(polygon.exterior.coords)
            for coord in boundary_coords[::len(boundary_coords)//5]:  # Take several representative points
                candidate_lon, candidate_lat = coord[0], coord[1]
                if self._is_point_in_constraint_area(candidate_lon, candidate_lat):
                    return [candidate_lon, candidate_lat, alt]
        
        # Finally try random search within the bounding box of constraint area
        min_lon, min_lat, max_lon, max_lat = self.find_min_max_coordinates()
        for _ in range(50):  # Try at most 50 times
            candidate_lon = np.random.uniform(min_lon, max_lon)
            candidate_lat = np.random.uniform(min_lat, max_lat)
            if self._is_point_in_constraint_area(candidate_lon, candidate_lat):
                return [candidate_lon, candidate_lat, alt]

        return initial_guess

    def estimate_point(self, initial_guess=[0, 0, 0], bounds=None, use_constrained_optimization=True):
        """
        Use optimization algorithm to estimate coordinates with polygon boundary constraints
        
        :param initial_guess: Initial guess for optimization, default is [0, 0, 0]
        :param bounds: Manual bounds (optional)
        :param use_constrained_optimization: Whether to use constrained optimization
        :return: Optimized 3D coordinate point, or None if optimization fails
        """
        self.optimization_history = []  # Clear history

        # Find valid initial guess point
        valid_initial_guess = self._find_valid_initial_guess(initial_guess)

        # Set boundary constraints
        if bounds is None:
            bounds_for_optimization = [
                [self.min_lon, self.max_lon],           # lon bounds
                [self.min_lat, self.max_lat],           # lat bounds
                [valid_initial_guess[2] - (self._get_floor_height()), valid_initial_guess[2] + (self._get_floor_height())]  # alt bounds
            ]
        else:
            bounds_for_optimization = bounds

        best_result = None
        best_error = float('inf')
        
        # Method 1: Use constrained SLSQP optimization algorithm
        if use_constrained_optimization and self.use_polygon_constraints:
            try:
                # Define constraints
                constraints = [
                    {
                        'type': 'ineq',
                        'fun': self._polygon_constraint_function
                    }
                ]
                
                result_slsqp = minimize(
                    self.ap_residuals,
                    valid_initial_guess,
                    method='SLSQP',
                    bounds=bounds_for_optimization,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False, 'iprint': 0}
                )
                
                if result_slsqp.success and self._is_point_in_constraint_area(result_slsqp.x[0], result_slsqp.x[1]):
                    error = self.ap_residuals(result_slsqp.x)
                    if error < best_error:
                        best_result = result_slsqp
                        best_error = error
            except Exception as e:
                logger.warning("SLSQP optimization failed: %s", e, exc_info=True)
        
        # Method 2: Use penalty function method
        try:
            result_penalty = minimize(
                self._penalized_objective_function,
                valid_initial_guess,
                method='L-BFGS-B',
                bounds=bounds_for_optimization,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )

            if result_penalty.success:
                # Check if result is within constraint area
                is_valid = not self.use_polygon_constraints or self._is_point_in_constraint_area(
                    result_penalty.x[0], result_penalty.x[1]
                )

                if is_valid:
                    error = self.ap_residuals(result_penalty.x)
                    if error < best_error:
                        best_result = result_penalty
                        best_error = error
        except Exception as e:
            logger.warning("Penalty optimization (L-BFGS-B) failed: %s", e, exc_info=True)
        
        # Method 3: Fallback to TRF algorithm (for comparison)
        if best_result is None:
            try:
                # Convert boundary format for least_squares
                bounds_lower = [b[0] for b in bounds_for_optimization]
                bounds_upper = [b[1] for b in bounds_for_optimization]

                result_trf = least_squares(
                    self.weighted_residuals_for_least_squares,
                    valid_initial_guess,
                    bounds=(bounds_lower, bounds_upper),
                    method='trf',
                    verbose=0
                )

                if result_trf.success:
                    # Check if result is within constraint area
                    is_valid = not self.use_polygon_constraints or self._is_point_in_constraint_area(
                        result_trf.x[0], result_trf.x[1]
                    )

                    if is_valid:
                        error = self.ap_residuals(result_trf.x)
                        best_result = result_trf
                        best_error = error
                    else:
                        # Save result as backup even if not in constraint area
                        if best_result is None:
                            best_result = result_trf
                            best_error = error
            except Exception as e:
                logger.warning("Least-squares optimization (TRF) failed: %s", e, exc_info=True)
        
        # Output result statistics
        if best_result is not None:
            final_lon, final_lat, final_alt = best_result.x
            is_in_constraint = not self.use_polygon_constraints or self._is_point_in_constraint_area(final_lon, final_lat)

            # Calculate weight statistics
            if self.weights is not None:
                original_weights = self.weights
                self.weights = None
                _ = self.ap_residuals(best_result.x)
                self.weights = original_weights

            return best_result
        else:
            return None

    def _objective(self, x):
        """
        Objective function: Calculate squared sum of distance error between the point to be determined and known points
        
        :param x: Current point to be optimized in 3D coordinates
        :return: Squared sum of distance error
        """
        total_error = 0
        for i, point in enumerate(self.known_points):
            distance_error = np.linalg.norm(x - point) - self.distances[i]
            total_error += distance_error ** 2
        self.optimization_history.append(x)  # Save coordinates of each optimization step
        return total_error

    def _get_floor_height(self):
        try:
            from utils.configuration import ConfigurationManager
            cfg = ConfigurationManager().get_config()
            return getattr(cfg, 'floor_height', 3.2)
        except Exception:
            return 3.2


    

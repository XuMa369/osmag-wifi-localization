from pyproj import Geod
import math
import numpy as np
from utils.configuration import ConfigurationManager

# Global Geod object to avoid repeated initialization
_geod_instance = None


def get_geod_instance():
    """Get or create a global Geod instance for distance calculations"""
    global _geod_instance
    if _geod_instance is None:
        _geod_instance = Geod(ellps='WGS84')
    return _geod_instance


def calculate_precise_distance(lon1, lat1, lon2, lat2):
    """
    Calculate distance between two points using PyProj's Geod class.

    Args:
        lon1: Longitude of first point
        lat1: Latitude of first point
        lon2: Longitude of second point
        lat2: Latitude of second point

    Returns:
        Distance in meters
    """
    geod = get_geod_instance()
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)  # lon, lat, lon, lat
    return distance


def calculate_distance(lon1, lat1, lon2, lat2, ap_level, target_ap_level):
    """Calculate 3D distance with vertical level difference using configured floor height.

    Args:
        lon1: Longitude of AP point
        lat1: Latitude of AP point
        lon2: Longitude of target point
        lat2: Latitude of target point
        ap_level: Floor level of AP
        target_ap_level: Floor level of target

    Returns:
        3D distance in meters considering horizontal geodesic distance and vertical floor offset.
    """
    geod = get_geod_instance()
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    cfg = ConfigurationManager().get_config()
    floor_height = getattr(cfg, 'floor_height', 3.2)
    c = math.sqrt(distance ** 2 + (abs(ap_level - target_ap_level + 1) * floor_height) ** 2)
    return c


def calculate_angle_and_normal(line, edge):
    """Calculate the angle between a line and the normal of an edge in degrees, and return the normalized edge normal vector."""
    line_vector = np.array(line.coords[1]) - np.array(line.coords[0])
    edge_vector = np.array(edge.coords[1]) - np.array(edge.coords[0])
    edge_normal = np.array([-edge_vector[1], edge_vector[0]])

    dot_product = np.dot(line_vector, edge_normal)
    magnitude_line = np.linalg.norm(line_vector)
    magnitude_normal = np.linalg.norm(edge_normal)

    cos_angle = dot_product / (magnitude_line * magnitude_normal)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    edge_normal = edge_normal / magnitude_normal  # Normalize the normal vector

    return angle_deg, edge_normal


def calculate_polygon_area(polygon):
    """Calculate actual geographic area of polygon (square meters)."""
    geod = Geod(ellps="WGS84")
    coords = list(polygon.exterior.coords)
    area, _ = geod.polygon_area_perimeter(
        lons=[coord[0] for coord in coords],
        lats=[coord[1] for coord in coords]
    )
    return abs(area) 
from typing import Dict, List, Tuple
from shapely.geometry import Polygon

from utils.geometry import calculate_polygon_area


class BuildingConstraints:
    """Class encapsulation for building constraint extraction."""

    @staticmethod
    def extract_building_constraint_polygons(largest_area: Dict, target_level: int):
        """
        Extract building constraint polygons for specified floor from largest_area
        Maintains exactly the same logic and return structure as original implementation in ap_loc.py

        Args:
            largest_area: Building cluster result dictionary
            target_level: Target floor level

        Returns:
            List[Tuple]: Constraint polygon list, format: [(polygon, level), ...]
        """
        constraint_polygons = []

        if largest_area and target_level in largest_area:
            building_clusters = largest_area[target_level]
            for cluster in building_clusters:
                polygon = cluster['polygon']
                level = cluster['level']
                constraint_polygons.append((polygon, level))
        return constraint_polygons


# Compatibility function: Keep original function name, delegate to class method internally, ensure external calls remain unchanged

def extract_building_constraint_polygons(largest_area: Dict, target_level: int):
    return BuildingConstraints.extract_building_constraint_polygons(largest_area, target_level)


def find_largest_polygon(polygons):
    """
    Calculate the largest polygon by floor and building cluster

    Args:
        polygons -- List of multiple polygons, each element in (polygon, poly_level) format

    Returns:
        Dictionary grouped by floor, each floor contains list of building clusters with their largest polygon
    """
    # Group polygons by floor
    level_polygons: Dict[int, List[Polygon]] = {}
    for polygon, poly_level in polygons:
        if poly_level not in level_polygons:
            level_polygons[poly_level] = []
        level_polygons[poly_level].append(polygon)

    result: Dict[int, List[Dict]] = {}

    for level, poly_list in level_polygons.items():
        building_clusters: List[List[Polygon]] = []
        processed = [False] * len(poly_list)

        for i, polygon in enumerate(poly_list):
            if processed[i]:
                continue

            current_cluster = [polygon]
            processed[i] = True

            stack = [i]
            while stack:
                current_idx = stack.pop()
                current_poly = poly_list[current_idx]

                for j, other_polygon in enumerate(poly_list):
                    if processed[j]:
                        continue

                    if (current_poly.intersects(other_polygon) or
                        current_poly.touches(other_polygon) or
                        current_poly.contains(other_polygon) or
                        other_polygon.contains(current_poly)):
                        current_cluster.append(other_polygon)
                        processed[j] = True
                        stack.append(j)

            building_clusters.append(current_cluster)

        floor_results = []
        for cluster_idx, cluster in enumerate(building_clusters):
            largest_area = 0
            largest_polygon = None

            for polygon in cluster:
                area = calculate_polygon_area(polygon)
                if area > largest_area:
                    largest_area = area
                    largest_polygon = polygon

            floor_results.append({
                'building_id': cluster_idx,
                'area': largest_area,
                'polygon': largest_polygon,
                'level': level,
                'cluster_size': len(cluster)
            })

        result[int(level)] = floor_results

    return result


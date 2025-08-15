from shapely.geometry import LineString
from typing import Dict, Tuple
import logging

from utils.geometry import calculate_distance, calculate_angle_and_normal

logger = logging.getLogger(__name__)


def process_rssi_data(target_ap_list, ap_to_position, polygon_edges, ap_level, learn_level, ap_learn_flag):
    lines = {}
    learn_no_wall_dis_rss = {'dis': [], 'rssi': []}
    learn_wall_dis_rss = {'dis': [], 'rssi': []}

    # Use int-level keys primarily; fall back to str for robustness
    edges = polygon_edges.get(learn_level, []) or polygon_edges.get(str(learn_level), [])

    learning_aps = [(tagap, ap_data) for tagap, ap_data in target_ap_list.items()
                    if ap_data.get('learn') is True]

    valid_macs = {mac: pos for mac, pos in ap_to_position.items()
                  if ap_learn_flag.get(mac) is True}

    distance_cache: Dict[Tuple[float, float, float, float, int, int], float] = {}
    wall_cache: Dict[Tuple[float, float, float, float], bool] = {}

    for tagap, ap_data in learning_aps:
        tagap_level = ap_data['level']
        tagap_macs = ap_data['mac']

        same_floor_macs = {mac: pos for mac, pos in valid_macs.items()
                           if ap_level.get(mac) == tagap_level}

        for mac, mac_pos in same_floor_macs.items():
            if mac not in tagap_macs:
                continue

            dist_key = (mac_pos[0], mac_pos[1], tagap[0], tagap[1], ap_level[mac], tagap_level)

            if dist_key in distance_cache:
                distance = distance_cache[dist_key]
            else:
                distance = calculate_distance(
                    mac_pos[0], mac_pos[1],
                    tagap[0], tagap[1],
                    ap_level[mac], tagap_level
                )
                distance_cache[dist_key] = distance

            wall_key = (tagap[0], tagap[1], mac_pos[0], mac_pos[1])

            if wall_key in wall_cache:
                has_wall = wall_cache[wall_key]
            else:
                has_wall = False
                if edges:
                    line = LineString([tagap, mac_pos])
                    for edge in edges:
                        if line.intersects(edge):
                            has_wall = True
                            break
                wall_cache[wall_key] = has_wall

            ap_rssi = tagap_macs[mac]

            if not has_wall:
                learn_no_wall_dis_rss['dis'].append(distance)
                learn_no_wall_dis_rss['rssi'].append(ap_rssi)
            else:
                learn_wall_dis_rss['dis'].append(distance)
                learn_wall_dis_rss['rssi'].append(ap_rssi)

                line_key = f"{tagap[0]},{tagap[1]}-{mac_pos[0]},{mac_pos[1]}"
                lines[line_key] = {'rssi': ap_rssi, 'dis': distance}

    return lines, learn_no_wall_dis_rss, learn_wall_dis_rss


def process_wall_line(lines, polygon_edges, learn_level):
    attenuation_edges = []
    nums = []
    learn_parameter = {}

    for line_key in lines:
        try:
            start_coords, end_coords = line_key.split('-')
            start_x, start_y = map(float, start_coords.split(','))
            end_x, end_y = map(float, end_coords.split(','))
            line = LineString([(start_x, start_y), (end_x, end_y)])

            line_intersections = []
            for edge in polygon_edges.get(learn_level, []):
                if line.intersects(edge):
                    line_intersections.append(edge)

            if len(line_intersections) == 2:
                for edge in line_intersections:
                    if edge not in attenuation_edges:
                        attenuation_edges.append(edge)
                        learn_parameter[edge] = {'dis': [], 'rssi': [], 'at_val': 0, 'angle': [], 'bias': 0}

                    learn_parameter[edge]['dis'].append(lines[line_key]['dis'])
                    learn_parameter[edge]['rssi'].append(lines[line_key]['rssi'])
                    angle, _ = calculate_angle_and_normal(line, edge)
                    learn_parameter[edge]['angle'].append(angle)

            nums.append(len(line_intersections))

        except (ValueError, IndexError) as e:
            logger.debug("Failed to parse line_key '%s': %s. Falling back to generic bucket.", line_key, e, exc_info=True)
            if line_key not in learn_parameter:
                learn_parameter[line_key] = {'dis': [], 'rssi': [], 'at_val': 0, 'angle': [], 'bias': 0}

            learn_parameter[line_key]['dis'].append(lines[line_key]['dis'])
            learn_parameter[line_key]['rssi'].append(lines[line_key]['rssi'])
            learn_parameter[line_key]['angle'].append(0.0)

    return attenuation_edges, learn_parameter, nums 
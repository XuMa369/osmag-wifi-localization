import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import logging

logger = logging.getLogger(__name__)


def save_estimated_positions_to_osm(estimated_positions, ap_groups, output_file, template_file=None):
    """
    Save estimated AP positions to OSM file format

    Args:
        estimated_positions: Estimated AP position dictionary {mac: [x, y, z]}
        ap_groups: AP group information containing MAC addresses and floor information
        output_file: Output OSM file path
        template_file: Template OSM file path (optional)
    """
 
    osm_root = ET.Element('osm')
    osm_root.set('version', '0.6')
    osm_root.set('generator', 'wifi_loc_ap_positioning')

    if template_file and os.path.exists(template_file):
        try:
            template_tree = ET.parse(template_file)
            template_root = template_tree.getroot()
            for child in template_root:
                osm_root.append(child)
        except Exception as e:
            logger.warning("Failed to parse template OSM file '%s': %s. Proceeding without template.", template_file, e, exc_info=True)

    base_node_id = -1000000
    current_id = base_node_id

    ap_nodes_to_add = []

    for ap_group_key in ap_groups:
        if ap_group_key not in estimated_positions:
            continue

        mac_list = ap_groups[ap_group_key]['mac']
        if not mac_list:
            continue

        position = estimated_positions[ap_group_key]
        level = ap_groups[ap_group_key].get('ap_level_est', ap_groups[ap_group_key].get('ap_level_gt'))

        ap_node_info = {
            'id': current_id,
            'lat': position[1],
            'lon': position[0],
            'level': level,
            'mac_list': mac_list,
            'ap_group_key': ap_group_key
        }

        ap_nodes_to_add.append(ap_node_info)
        current_id -= 1

    for ap_info in ap_nodes_to_add:
        node = ET.SubElement(osm_root, 'node')
        node.set('id', str(ap_info['id']))
        node.set('action', 'modify')
        node.set('visible', 'true')
        node.set('lat', f"{ap_info['lat']:.11f}")
        node.set('lon', f"{ap_info['lon']:.11f}")

        node_type_tag = ET.SubElement(node, 'tag')
        node_type_tag.set('k', 'osmAG:node:type')
        node_type_tag.set('v', 'AP')

        level_tag = ET.SubElement(node, 'tag')
        level_tag.set('k', 'osmAG:WiFi:AP:level')
        level_tag.set('v', str(ap_info['level']))

        freq_24g_tag = ET.SubElement(node, 'tag')
        freq_24g_tag.set('k', 'osmAG:WIFI:Frequency:2.4G')
        freq_24g_tag.set('v', '2.4G')

        freq_5g_tag = ET.SubElement(node, 'tag')
        freq_5g_tag.set('k', 'osmAG:WIFI:Frequency:5G')
        freq_5g_tag.set('v', '5G')

        freq_58g_tag = ET.SubElement(node, 'tag')
        freq_58g_tag.set('k', 'osmAG:WIFI:Frequency:5.8G')
        freq_58g_tag.set('v', '5.8G')

        mac_list = ap_info['mac_list']

        for idx, mac in enumerate(mac_list):
            if idx < 5:
                bssid_tag = ET.SubElement(node, 'tag')
                bssid_tag.set('k', f'osmAG:WiFi:BSSID:5G:{idx}')
                bssid_tag.set('v', mac.upper())

    rough_string = ET.tostring(osm_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent='  ')

    lines = pretty_xml.split('\n')
    pretty_xml = '\n'.join([line for line in lines if line.strip()])

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

    return output_file

import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString

class OsmDataParser:
    def __init__(self, osm_file):
        self.osm_file = osm_file
        self.nodes = {}
        self.target_nodes = {}
        self.rssi_value = {}
        self.rssi = []
        self.ap_to_position = {}
        self.ap_level = {}
        self.ap_frequency = {}  # New: Store AP frequency information
        self.target_ap = {}
        self.way_data = []
        self.tree = ET.parse(self.osm_file)
        self.root = self.tree.getroot()
        self.all_mac = []
        self.polygons = []
        self.polygon_edges = {}
        self.ap_learn_flag = {}
        self.freq_2_4G = [
            2412,  # Channel 1
            2417,  # Channel 2
            2422,  # Channel 3
            2427,  # Channel 4
            2432,  # Channel 5
            2437,  # Channel 6
            2442,  # Channel 7
            2447,  # Channel 8
            2452,  # Channel 9
            2457,  # Channel 10
            2462,  # Channel 11
            2467,  # Channel 12
            2472   # Channel 13
        ]

        # 5G band frequency list (5.1-5.35GHz) (MHz)
        self.freq_5G = [
            5180,  # Channel 36
            5200,  # Channel 40
            5220,  # Channel 44
            5240,  # Channel 48
            5260,  # Channel 52
            5280,  # Channel 56
            5300,  # Channel 60
            5320   # Channel 64
        ]

        # 5.8G band frequency list (5.725-5.825GHz) (MHz)
        self.freq_5_8G = [
            5745,  # Channel 149
            5765,  # Channel 153
            5785,  # Channel 157
            5805,  # Channel 161
            5825   # Channel 165
        ]

    def parse(self):
        for element in self.root:
            if element.tag == 'node':
                self._parse_node(element)
            elif element.tag == 'way':
                self._parse_way(element)

    def _parse_node(self, element):
        node_id = element.attrib['id']
        lat = float(element.attrib['lat'])
        lon = float(element.attrib['lon'])

        self.nodes[node_id] = (lon, lat)

        tags = list(element.iter('tag'))
        
        # Check osmAG:node:type tag
        node_type = None
        for tag in tags:
            if tag.attrib.get('k') == 'osmAG:node:type':
                node_type = tag.attrib.get('v')
                break
        
        # Process based on node type
        if node_type == 'AP':
            # Process AP node, store to ap_to_position
            self._parse_ap_node(tags, lon, lat)
        elif node_type == 'fingerprint':
            # Process fingerprint node, store to target_ap
            self._parse_fingerprint_node(tags, lon, lat)
       

    def _parse_ap_node(self, tags, lon, lat):
        """Process AP node, store to ap_to_position"""
        # Temporary storage for BSSID and floor information
        bssids = []
        level_value = None
        learn_value = False
        for tag in tags:
            if tag.attrib.get('k') == 'osmAG:WiFi:AP:learn':
                learn_value = True  
                break
        
        # Traverse all tags, extract WiFi related information
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            
            # Extract BSSID information (supports 2.4G, 5G, 5.8G bands)
            if tag_key.startswith('osmAG:WiFi:BSSID:'):
                bssid_lower = tag_value.lower()
                self.ap_to_position[bssid_lower] = (lon, lat)
                self.ap_learn_flag[bssid_lower] = learn_value
                if bssid_lower not in self.all_mac:
                    self.all_mac.append(bssid_lower)
                bssids.append(bssid_lower)
           
            # Extract floor information
            elif tag_key == 'osmAG:WiFi:AP:level':
                level_value = int(tag_value)
            
        
        # Set floor information for all found BSSIDs
        if level_value is not None:
            for bssid in bssids:
                self.ap_level[bssid] = level_value

    def _parse_fingerprint_node(self, tags, lon, lat):
        """Process fingerprint node, store to target_ap"""
        # Initialize position data structure
        self.target_ap[(lon, lat)] = self.target_ap.get((lon, lat), {})
        self.target_ap[(lon, lat)]['mac'] = {}
        self.target_ap[(lon, lat)]['level'] = 0
        self.target_ap[(lon, lat)]['learn'] = False
        
        # Temporary storage for BSSID, RSSI and frequency mapping
        bssid_map = {}
        rssi_map = {}
        freq_map = {}
        
        # Traverse all tags, extract WiFi related information
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            
            # Extract floor information
            if tag_key == 'osmAG:WiFi:Fingerprint:Floor':
                self.target_ap[(lon, lat)]['level'] = int(tag_value)
            
            if tag_key == 'osmAG:WiFi:Learn':
                self.target_ap[(lon, lat)]['learn'] = True
                
            # Extract BSSID information osmAG:WiFi:BSSID:X
            elif tag_key.startswith('osmAG:WiFi:BSSID:'):
                # Extract index, e.g., extract "1" from "osmAG:WiFi:BSSID:1"
                index = tag_key.split(':')[-1]
                bssid_map[index] = tag_value.lower()
            
            # Extract RSSI information osmAG:WiFi:RSSI:X
            elif tag_key.startswith('osmAG:WiFi:RSSI:'):
                # Extract index
                index = tag_key.split(':')[-1]
                rssi_map[index] = float(tag_value)
            
            # Extract frequency information osmAG:WiFi:Freq:X
            elif tag_key.startswith('osmAG:WiFi:Freq:'):
                # Extract index
                index = tag_key.split(':')[-1]
                freq_map[index] = int(tag_value)
        
        # Combine and store BSSID, RSSI and frequency information
        for index in bssid_map.keys():
            bssid = bssid_map[index]
            rssi = rssi_map.get(index)
            freq = freq_map.get(index)
            
            # Only store when BSSID, RSSI and frequency all exist
            if bssid and rssi is not None and freq is not None and freq in self.freq_5G:
                if bssid not in self.target_ap[(lon, lat)]['mac']:
                    self.target_ap[(lon, lat)]['mac'][bssid] = []
                
                # Store RSSI value to list
                self.target_ap[(lon, lat)]['mac'][bssid].append(rssi)

                # If this BSSID is not recorded yet, add to all_mac list
                if bssid not in self.all_mac:
                    self.all_mac.append(bssid)
                
              
    def _parse_wifi_node(self, tag, tags, lon, lat):
     
        # New format adaptation: Check if contains WiFi BSSID tags
        # New format uses osmAG:WiFi:BSSID:band:index format
        wifi_bssids = {}  # Changed to dictionary, store BSSID and corresponding band information
        frequencies = []
        level_tag = None
        
        # Traverse all tags, extract WiFi related information
        for tag_ in tags:
            tag_key = tag_.attrib.get('k', '')
            tag_value = tag_.attrib.get('v', '')
            
            # Extract BSSID information (supports 2.4G, 5G, 5.8G bands)
            if tag_key.startswith('osmAG:WiFi:BSSID:'):
                # Parse band information, e.g., osmAG:WiFi:BSSID:2.4G:0 -> 2.4G
                parts = tag_key.split(':')
                if len(parts) >= 4:
                    frequency_band = parts[3]  # Extract band (2.4G, 5G, 5.8G)
                    wifi_bssids[tag_value.lower()] = frequency_band
                else:
                    wifi_bssids[tag_value.lower()] = 'unknown'
                
            # Extract frequency information
            elif tag_key.startswith('osmAG:WIFI:Frequency:'):
                frequencies.append(tag_value)
                
            # Extract level information (supports both new and old formats)
            elif tag_key in ['osmAG:WiFi:level', 'osmAG:WiFi:AP:level']:
                level_tag = tag_value
        
        # If WiFi BSSID information is found, process this node
        if wifi_bssids:
            # Create position mapping for each BSSID
            for bssid, frequency_band in wifi_bssids.items():
                self.ap_to_position[bssid] = (lon, lat)
                if bssid not in self.all_mac:
                    self.all_mac.append(bssid)
                    
                # Set frequency information
                self.ap_frequency[bssid] = frequency_band

                # Set level information
                if level_tag:
                    self.ap_level[bssid] = int(level_tag)
                else:
                    # If no level information, default to 0
                    self.ap_level[bssid] = 0


    def _parse_way(self, element):
        way_nodes = [self.nodes[nd.attrib['ref']] for nd in element.iter('nd') if nd.attrib['ref'] in self.nodes]
        if way_nodes:
            way_level = None
            for tag in element.iter('tag'):
                if tag.attrib['k'] == 'level':
                    try:
                        way_level = int(tag.attrib['v'])
                    except Exception:
                        way_level = tag.attrib['v']

            way_tuple = tuple(way_nodes)
            if way_level is not None:
                self.way_data.append((way_tuple, way_level))

    def extract_polygon_edges(self):
        
        for way , way_level in self.way_data:
            if len(way) > 2 and Polygon(way).is_valid:
                poly = Polygon(way)
                self.polygons.append((poly, way_level))

        # Extract boundaries of each polygon and convert to line segments
        
        for polygon, poly_level in self.polygons:
            # normalize level to int if possible
            try:
                level_key = int(poly_level)
            except Exception:
                level_key = poly_level
            if level_key not in self.polygon_edges:
                self.polygon_edges[level_key] = []
            exterior_coords = list(polygon.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                edge = LineString([exterior_coords[i], exterior_coords[i + 1]])
                self.polygon_edges[level_key].append(edge)
                
    def get_data(self):
        return self.ap_to_position, self.ap_level, self.target_ap, self.way_data, self.all_mac, self.polygons, self.polygon_edges, self.ap_learn_flag

# Additional parser classes unified here for reuse by scripts

class APMapParser:
    """AP Map Parser - Read positioned AP locations from AP_map.osm file"""
    
    def __init__(self, ap_map_file: str, floor_height: float = None):
        self.ap_map_file = ap_map_file
        self.ap_positions = {}
        self.all_aps = []
        if floor_height is None:
            try:
                from utils.configuration import ConfigurationManager
                cfg = ConfigurationManager().get_config()
                self.floor_height = getattr(cfg, 'floor_height', 3.2)
            except Exception:
                self.floor_height = 3.2
        else:
            self.floor_height = floor_height
    
    def parse_ap_map(self):
        tree = ET.parse(self.ap_map_file)
        root = tree.getroot()
        ap_count = 0
        for element in root:
            if element.tag == 'node':
                if self._parse_ap_node(element):
                    ap_count += 1
        return self.ap_positions
    
    def _parse_ap_node(self, element):
        try:
            lat = float(element.attrib['lat'])
            lon = float(element.attrib['lon'])
            node_id = element.attrib['id']
        except Exception:
            return False
        tags = list(element.iter('tag'))
        is_ap = False
        level = 1
        mac_list = []
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            if tag_key == 'osmAG:node:type' and tag_value == 'AP':
                is_ap = True
            elif tag_key == 'osmAG:WiFi:AP:level':
                level = int(tag_value)
            elif tag_key.startswith('osmAG:WiFi:BSSID:'):
                mac_list.append(tag_value.lower())
        if is_ap:
            if mac_list:
                ap_group_key = mac_list[0][:-1] if len(mac_list[0]) > 0 else node_id
            else:
                ap_group_key = node_id
            self.ap_positions[ap_group_key] = {
                'position': (lon, lat, level * self.floor_height),
                'level': level,
                'mac_list': mac_list,
                'node_id': node_id
            }
            self.all_aps.append(ap_group_key)
            return True
        return False


class FingerprintParser:
    """Fingerprint Parser - Read fingerprint points to be localized from fingerprint file"""
    
    def __init__(self, fingerprint_file: str):
        self.fingerprint_file = fingerprint_file
        self.fingerprint_data = {}
    
    def parse_fingerprint_data(self):
        tree = ET.parse(self.fingerprint_file)
        root = tree.getroot()
        for element in root:
            if element.tag == 'node':
                if self._parse_fingerprint_node(element):
                    continue
        return self.fingerprint_data
    
    def _parse_fingerprint_node(self, element):
        try:
            lat = float(element.attrib['lat'])
            lon = float(element.attrib['lon'])
            node_id = element.attrib.get('id', '')
        except Exception:
            return False
        tags = list(element.iter('tag'))
        node_type = None
        for tag in tags:
            if tag.attrib.get('k') == 'osmAG:node:type':
                node_type = tag.attrib.get('v')
                break
        if node_type != 'fingerprint':
            return False
        bssid_map = {}
        rssi_map = {}
        freq_map = {}
        floor_level = 1
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            if tag_key == 'osmAG:WiFi:Fingerprint:Floor':
                floor_level = int(tag_value)
            elif tag_key.startswith('osmAG:WiFi:BSSID:'):
                index = tag_key.split(':')[-1]
                bssid_map[index] = tag_value.lower()
            elif tag_key.startswith('osmAG:WiFi:RSSI:'):
                index = tag_key.split(':')[-1]
                rssi_map[index] = float(tag_value)
            elif tag_key.startswith('osmAG:WiFi:Freq:'):
                index = tag_key.split(':')[-1]
                freq_map[index] = int(tag_value)
        if bssid_map and rssi_map:
            rssi_data = {}
            for index in bssid_map.keys():
                bssid = bssid_map[index]
                rssi = rssi_map.get(index)
                freq = freq_map.get(index)
                if rssi is not None and freq is not None and 5180 <= freq <= 5320:
                    rssi_data[bssid] = rssi
            if rssi_data:
                point_name = f"fingerprint_point_{node_id}"
                self.fingerprint_data[point_name] = {
                    'position': (lon, lat),
                    'level': floor_level,
                    'rssi_data': rssi_data
                }
                return True
        return False


class WiFiFingerprintParser:
    """WiFi Fingerprint Database Parser"""
    
    def __init__(self, fingerprint_osm_file: str):
        self.fingerprint_osm_file = fingerprint_osm_file
        self.fingerprint_data = {}
        self.all_macs = set()
    
    def parse_fingerprint_data(self):
        tree = ET.parse(self.fingerprint_osm_file)
        root = tree.getroot()
        for element in root:
            if element.tag == 'node':
                self._parse_fingerprint_node(element)
        return self.fingerprint_data
    
    def _parse_fingerprint_node(self, element):
        try:
            lat = float(element.attrib['lat'])
            lon = float(element.attrib['lon'])
        except Exception:
            return
        tags = list(element.iter('tag'))
        node_type = None
        for tag in tags:
            if tag.attrib.get('k') == 'osmAG:node:type':
                node_type = tag.attrib.get('v')
                break
        if node_type != 'fingerprint':
            return
        bssid_map = {}
        rssi_map = {}
        freq_map = {}
        floor_level = 0
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            if tag_key == 'osmAG:WiFi:Fingerprint:Floor':
                floor_level = int(tag_value)
            elif tag_key.startswith('osmAG:WiFi:BSSID:'):
                index = tag_key.split(':')[-1]
                bssid_map[index] = tag_value.lower()
            elif tag_key.startswith('osmAG:WiFi:RSSI:'):
                index = tag_key.split(':')[-1]
                rssi_map[index] = float(tag_value)
            elif tag_key.startswith('osmAG:WiFi:Freq:'):
                index = tag_key.split(':')[-1]
                freq_map[index] = int(tag_value)
        if bssid_map and rssi_map:
            position = (lon, lat)
            self.fingerprint_data[position] = {'mac': {}, 'level': floor_level}
            for index in bssid_map.keys():
                bssid = bssid_map[index]
                rssi = rssi_map.get(index)
                freq = freq_map.get(index)
                if rssi is not None and freq is not None and 5180 <= freq <= 5320:
                    if bssid not in self.fingerprint_data[position]['mac']:
                        self.fingerprint_data[position]['mac'][bssid] = []
                    self.fingerprint_data[position]['mac'][bssid].append(rssi)
                    self.all_macs.add(bssid)


class WiFiTestPointParser:
    """WiFi Test Point Parser - Parse fingerprint data as test points"""
    
    def __init__(self, test_osm_file: str):
        self.test_osm_file = test_osm_file
        self.test_points = {}
    
    def parse_test_points(self):
        tree = ET.parse(self.test_osm_file)
        root = tree.getroot()
        for element in root:
            if element.tag == 'node':
                self._parse_fingerprint_node_as_test_point(element)
        return self.test_points
    
    def _parse_fingerprint_node_as_test_point(self, element):
        try:
            lat = float(element.attrib['lat'])
            lon = float(element.attrib['lon'])
            node_id = element.attrib.get('id', '')
        except Exception:
            return
        tags = list(element.iter('tag'))
        node_type = None
        for tag in tags:
            if tag.attrib.get('k') == 'osmAG:node:type':
                node_type = tag.attrib.get('v')
                break
        if node_type != 'fingerprint':
            return
        bssid_map = {}
        rssi_map = {}
        freq_map = {}
        floor_level = 0
        for tag in tags:
            tag_key = tag.attrib.get('k', '')
            tag_value = tag.attrib.get('v', '')
            if tag_key == 'osmAG:WiFi:Fingerprint:Floor':
                floor_level = int(tag_value)
            elif tag_key.startswith('osmAG:WiFi:BSSID:'):
                index = tag_key.split(':')[-1]
                bssid_map[index] = tag_value.lower()
            elif tag_key.startswith('osmAG:WiFi:RSSI:'):
                index = tag_key.split(':')[-1]
                rssi_map[index] = float(tag_value)
            elif tag_key.startswith('osmAG:WiFi:Freq:'):
                index = tag_key.split(':')[-1]
                freq_map[index] = int(tag_value)
        if bssid_map and rssi_map:
            wifi_data = {}
            for index in bssid_map.keys():
                bssid = bssid_map[index]
                rssi = rssi_map.get(index)
                freq = freq_map.get(index)
                if rssi is not None and freq is not None and 5180 <= freq <= 5320:
                    if bssid not in wifi_data:
                        wifi_data[bssid] = []
                    wifi_data[bssid].append(rssi)
            if wifi_data:
                self.test_points[f"test_point_{node_id}"] = {
                    'position': (lon, lat),
                    'level': floor_level,
                    'wifi_data': wifi_data
                }
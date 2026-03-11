from typing import Dict, List, Tuple, Optional
import os

from utils.configuration import ConfigurationManager
from io_layer.osm_parser import OsmDataParser


class DataLoader:
    """Data loader - responsible for loading and parsing data from OSM files"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager.get_config()
        self.parser: Optional[OsmDataParser] = None

    def load_osm_data(self, osm_filename: str = None) -> Tuple[Dict, Dict, Dict, List, List, List, Dict, Dict]:
        """Load OSM data"""
        osm_file_path = osm_filename if osm_filename else self.config.file_paths.input_osm_file
        if not isinstance(osm_file_path, str) or not osm_file_path:
            raise ValueError("Invalid OSM file path provided to DataLoader.load_osm_data")
        if not os.path.exists(osm_file_path):
            raise FileNotFoundError(f"OSM file does not exist: {osm_file_path}")
        try:
            self.parser = OsmDataParser(osm_file_path)
            self.parser.parse()
            self.parser.extract_polygon_edges()
            return self.parser.get_data()
        except Exception as e:
            raise RuntimeError(f"Failed to load and parse OSM data from '{osm_file_path}'") from e 
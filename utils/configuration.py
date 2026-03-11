"""
Configuration System - Centralized configuration management for AP localization
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryFilterConfig:
    """Configuration for trajectory filtering"""
    min_distance: float = 2.0
    rssi_threshold: float = 2.0
    max_samples: int = 6
    enable_filter: bool = True


@dataclass
class ConfidenceConfig:
    """Configuration for RSSI confidence weighting"""
    enable_weighting: bool = True
    strong_signal_threshold: float = -50.0
    weak_signal_threshold: float = -80.0
    max_weight: float = 1.0
    min_weight: float = 1.0
    weight_method: str = 'exponential'  # 'linear', 'exponential', 'sigmoid'


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms"""
    max_iterations: int = 10
    convergence_threshold: float = 0.1
    use_building_constraints: bool = True
    wall_attenuation_default: float = 3.0
    max_function_evaluations: int = 10000
    optimizer_max_iterations: int = 1000
    function_tolerance: float = 1e-9


@dataclass
class WallDetectionConfig:
    """Configuration for wall detection"""
    distance_threshold: float = 10.0
    rssi_threshold: float = -30.0


@dataclass
class SignalModelConfig:
    """Configuration for signal propagation model"""
    default_rssi_0: float = -28.879257951315253
    default_path_loss_exponent: float = 2.6132845414003243
    max_reasonable_distance: float = 10000.0
    min_reasonable_distance: float = 0.1


@dataclass
class SystemConfig:
    """Configuration for system parameters"""
    base_node_id: int = -1000000
    penalty_coefficient: float = 111000000



@dataclass
class FilePathConfig:
    """Configuration for file paths"""
    input_osm_file: str = './map/wifi_data.osm'
    output_osm_file: str = './map/AP_MAP.osm'
    template_osm_file: str = './map/base_map.osm'
    output_base_dir: Optional[str] = None


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint point localization (defaults match existing logic)."""
    # Signal model defaults for fingerprint localization
    rssi_0_default: float = -28.79169776352291
    path_loss_exponent_default: float = 2.542972169273509
    wall_attenuation: float = -10.772449287430723

    # Confidence weights (specific to fingerprint script current defaults)
    enable_weighting: bool = True
    strong_signal_threshold: float = -10.0
    weak_signal_threshold: float = -80.0
    max_weight: float = 3.0
    min_weight: float = 0.3
    weight_method: str = 'exponential'

    # AP selection and validation
    ap_selection_top_k: int = 3
    valid_min_points: int = 3
    valid_max_distance: float = 100.0
    rssi_valid_min: float = -100.0
    rssi_valid_max: float = -20.0

    # Wall correction thresholds
    wall_distance_threshold_outer: float = 10.0
    wall_distance_threshold_inner: float = 5.0
    min_wall_length_m: float = 1.0


@dataclass
class FingerprintKNNConfig:
    """Configuration for KNN-based fingerprint localization (defaults match existing logic)."""
    missing_rssi_fill: float = -100.0
    boundary_adjust_log_threshold_m: float = 1.0


@dataclass
class RSSIOptimizerConfig:
    """Configuration for RSSI optimizer defaults (defaults match existing logic)."""
    reference_distance_d0: float = 1.0
    curve_fit_maxfev: int = 10000
    min_points_for_curve_fit: int = 3
    small_positive_distance: float = 1e-6
    default_sigma_rssi0: float = 5.0
    default_sigma_n: float = 0.5
    default_wall_attenuation_fallback: float = -15.0


@dataclass
class MainSystemConfig:
    """Main system configuration"""
    # Processing parameters
    rssi_num_threshold: int = 15
    learn_level: int = 1
    floor_height: float = 3.2  # Floor height (meters)

    # Component configurations
    file_paths: FilePathConfig = None
    trajectory_filter: TrajectoryFilterConfig = None
    confidence: ConfidenceConfig = None
    optimization: OptimizationConfig = None
    wall_detection: WallDetectionConfig = None
    signal_model: SignalModelConfig = None
    system: SystemConfig = None
    # Fingerprint-related configurations
    fingerprint: FingerprintConfig = None
    fingerprint_knn: FingerprintKNNConfig = None
    # RSSI optimizer configuration
    rssi_optimizer: RSSIOptimizerConfig = None

    def __post_init__(self):
        """Initialize sub-configurations if not provided"""
        if self.trajectory_filter is None:
            self.trajectory_filter = TrajectoryFilterConfig()
        if self.confidence is None:
            self.confidence = ConfidenceConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.wall_detection is None:
            self.wall_detection = WallDetectionConfig()
        if self.signal_model is None:
            self.signal_model = SignalModelConfig()
        if self.system is None:
            self.system = SystemConfig()
        if self.fingerprint is None:
            self.fingerprint = FingerprintConfig()
        if self.fingerprint_knn is None:
            self.fingerprint_knn = FingerprintKNNConfig()
        if self.rssi_optimizer is None:
            self.rssi_optimizer = RSSIOptimizerConfig()

class ConfigurationManager:
    """
    Centralized configuration management system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config = MainSystemConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            # Initialize default sub-configurations
            self._initialize_default_config()
    
    def load_from_file(self, config_file: str) -> MainSystemConfig:
        """
        Load configuration from YAML or JSON file

        Args:
            config_file: Path to configuration file

        Returns:
            Loaded MainSystemConfig object
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)

            self.config = self._dict_to_config(config_dict)
            self.logger.info(f"Configuration loaded from {config_file}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            self.logger.info("Using default configuration")
            self._initialize_default_config()

        return self.config

    def get_config(self) -> MainSystemConfig:
        """Get current configuration"""
        return self.config
    


    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MainSystemConfig:
        """Convert dictionary to MainSystemConfig object"""
        # Extract sub-configurations
        file_paths_dict = config_dict.get('file_paths', {})
        trajectory_filter_dict = config_dict.get('trajectory_filter', {})
        confidence_dict = config_dict.get('confidence', {})
        optimization_dict = config_dict.get('optimization', {})
        wall_detection_dict = config_dict.get('wall_detection', {})
        signal_model_dict = config_dict.get('signal_model', {})
        system_dict = config_dict.get('system', {})
        fingerprint_dict = config_dict.get('fingerprint', {})
        fingerprint_knn_dict = config_dict.get('fingerprint_knn', {})
        rssi_optimizer_dict = config_dict.get('rssi_optimizer', {})

        # Create sub-configuration objects
        file_paths = FilePathConfig(**file_paths_dict)
        trajectory_filter = TrajectoryFilterConfig(**trajectory_filter_dict)
        confidence = ConfidenceConfig(**confidence_dict)
        optimization = OptimizationConfig(**optimization_dict)
        wall_detection = WallDetectionConfig(**wall_detection_dict)
        signal_model = SignalModelConfig(**signal_model_dict)
        system = SystemConfig(**system_dict)
        fingerprint = FingerprintConfig(**fingerprint_dict)
        fingerprint_knn = FingerprintKNNConfig(**fingerprint_knn_dict)
        rssi_optimizer = RSSIOptimizerConfig(**rssi_optimizer_dict)

        # Create main configuration
        main_config_dict = {k: v for k, v in config_dict.items()
                           if k not in ['file_paths', 'trajectory_filter', 'confidence', 'optimization',
                                       'wall_detection', 'signal_model', 'system', 'fingerprint', 'fingerprint_knn', 'rssi_optimizer']}

        config = MainSystemConfig(
            file_paths=file_paths,
            trajectory_filter=trajectory_filter,
            confidence=confidence,
            optimization=optimization,
            wall_detection=wall_detection,
            signal_model=signal_model,
            system=system,
            fingerprint=fingerprint,
            fingerprint_knn=fingerprint_knn,
            rssi_optimizer=rssi_optimizer,
            **main_config_dict
        )

        return config
    

    
    def _initialize_default_config(self):
        """Initialize default sub-configurations"""
        if self.config.file_paths is None:
            self.config.file_paths = FilePathConfig()
        if self.config.trajectory_filter is None:
            self.config.trajectory_filter = TrajectoryFilterConfig()
        if self.config.confidence is None:
            self.config.confidence = ConfidenceConfig()
        if self.config.optimization is None:
            self.config.optimization = OptimizationConfig()
        if self.config.wall_detection is None:
            self.config.wall_detection = WallDetectionConfig()
        if self.config.signal_model is None:
            self.config.signal_model = SignalModelConfig()
        if self.config.system is None:
            self.config.system = SystemConfig()
        if self.config.fingerprint is None:
            self.config.fingerprint = FingerprintConfig()
        if self.config.fingerprint_knn is None:
            self.config.fingerprint_knn = FingerprintKNNConfig()
        if self.config.rssi_optimizer is None:
            self.config.rssi_optimizer = RSSIOptimizerConfig()

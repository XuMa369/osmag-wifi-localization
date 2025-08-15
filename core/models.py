from dataclasses import dataclass
from typing import Dict, List


@dataclass
class APGroupData:
    """AP group data structure"""
    mac: List[str]
    pos: List[List[float]]
    rssis: List[float]
    target_ap_level: List[int]
    ap_level: int


@dataclass
class ProcessingResult:
    """Processing result data structure"""
    estimated_positions: Dict[str, List[float]]
    error_array: List[float]
    initial_errors: List[float]
    statistics: Dict[str, float] 
from typing import List, Tuple


import numpy as np

class DataProcessor:
    """Data preprocessing class encapsulation."""

    @staticmethod
    def deduplicate_positions_with_avg_rssi(filtered_positions: List[Tuple[float, float, float]], filtered_rssis: List[float]):
        """
        Deduplicate positions of filtered sampling points and average RSSI for same positions.
        Maintains exactly the same logic as original ap_loc.py.

        Args:
            filtered_positions: Filtered position information list
            filtered_rssis: Filtered RSSI list

        Returns:
            (unique_pos_tuples, averages): Deduplicated position tuple list and corresponding average RSSI list
        """
        pos_tuples = [tuple(pos) for pos in filtered_positions]
        unique_pos_tuples = []
        averages = []

        seen_positions = set()
        for i, pos_tuple in enumerate(pos_tuples):
            if pos_tuple not in seen_positions:
                seen_positions.add(pos_tuple)
                indices = [j for j, p in enumerate(pos_tuples) if p == pos_tuple]
                avg_rssi = np.mean([filtered_rssis[j] for j in indices])
                unique_pos_tuples.append(pos_tuple)
                averages.append(avg_rssi)

        return unique_pos_tuples, averages


# Compatibility function: Keep original function name, delegate to class method internally, ensure external calls remain unchanged
def deduplicate_positions_with_avg_rssi(filtered_positions: List[Tuple[float, float, float]], filtered_rssis: List[float]):
    return DataProcessor.deduplicate_positions_with_avg_rssi(filtered_positions, filtered_rssis)


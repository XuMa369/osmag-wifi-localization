
from utils.geometry import calculate_precise_distance

import numpy as np

class TrajectoryFilter:
    """Trajectory filtering algorithm class encapsulation."""

    @staticmethod
    def filter_trajectory_data(positions, rssis, min_distance=2.0, rssi_threshold=3.0, max_samples=50):
        """
        Intelligent sampling point filtering algorithm based on signal strength priority and spatial dispersion
        Maintains exactly the same original logic
        """
        if len(positions) != len(rssis):
            raise ValueError("Position and signal strength list lengths are inconsistent")
        if len(positions) <= 3:
            return positions, rssis
        flat_rssis = []
        for rssi in rssis:
            if isinstance(rssi, (list, tuple)):
                if len(rssi) > 0:
                    if isinstance(rssi[0], (int, float)):
                        flat_rssis.append(float(rssi[0]))
                    else:
                        flat_rssis.append(-80.0)
                else:
                    flat_rssis.append(-80.0)
            elif isinstance(rssi, (int, float)):
                flat_rssis.append(float(rssi))
            else:
                flat_rssis.append(-80.0)
        rssis = flat_rssis
        indexed_data = [(i, rssi, pos) for i, (rssi, pos) in enumerate(zip(rssis, positions))]
        indexed_data.sort(key=lambda x: x[1], reverse=True)
        selected_indices = []
        selected_positions = []
        selected_rssis = []
        for idx, rssi, pos in indexed_data:
            if len(selected_indices) >= max_samples:
                break
            too_close = False
            for selected_pos in selected_positions:
                distance = calculate_precise_distance(pos[0], pos[1], selected_pos[0], selected_pos[1])
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                selected_indices.append(idx)
                selected_positions.append(pos)
                selected_rssis.append(rssi)
        if len(selected_positions) < max(3, max_samples // 4):
            adaptive_min_distance = min_distance * 0.7
            selected_indices = []
            selected_positions = []
            selected_rssis = []
            for idx, rssi, pos in indexed_data:
                if len(selected_indices) >= max_samples:
                    break
                too_close = False
                for selected_pos in selected_positions:
                    distance = calculate_precise_distance(pos[0], pos[1], selected_pos[0], selected_pos[1])
                    if distance < adaptive_min_distance:
                        too_close = True
                        break
                if not too_close:
                    selected_indices.append(idx)
                    selected_positions.append(pos)
                    selected_rssis.append(rssi)
        if len(selected_positions) < 3:
            selected_indices = []
            selected_positions = []
            selected_rssis = []
            for i, (idx, rssi, pos) in enumerate(indexed_data):
                if len(selected_indices) >= max(5, min(max_samples, len(indexed_data)//3)):
                    break
                selected_indices.append(idx)
                selected_positions.append(pos)
                selected_rssis.append(rssi)
        if selected_rssis and len(selected_positions) > 1:
            distances = []
            for i in range(len(selected_positions)):
                for j in range(i+1, len(selected_positions)):
                    dist = calculate_precise_distance(
                        selected_positions[i][0], selected_positions[i][1],
                        selected_positions[j][0], selected_positions[j][1]
                    )
                    distances.append(dist)
            _ = np.mean(distances)
            _ = np.min(distances)
            rssi_improvement = max(selected_rssis) - min(selected_rssis)
            _ = len(selected_positions) * np.mean(selected_rssis) / len(positions)
        filtered_positions = selected_positions
        filtered_rssis = selected_rssis
        return filtered_positions, filtered_rssis


# Compatibility function: Keep original function name, delegate to class method internally, ensure external calls remain unchanged

def filter_trajectory_data(positions, rssis, min_distance=2.0, rssi_threshold=3.0, max_samples=50):
    return TrajectoryFilter.filter_trajectory_data(positions, rssis, min_distance, rssi_threshold, max_samples)


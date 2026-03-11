import numpy as np
import logging

logger = logging.getLogger(__name__)


def rssi_to_distance(rssi, A=-28.879257951315253, n=2.6132845414003243):
    """
    Convert RSSI value to distance with numerical stability checks

    Args:
        rssi: RSSI value
        A: Signal strength parameter
        n: Path loss exponent

    Returns:
        Distance value (meters), returns a reasonable default value if input is invalid
    """
    if not np.isfinite([rssi, A, n]).all():
        return 1.0

    if n <= 0:
        return 1.0

    try:
        exponent = (A - rssi) / (10 * n)

        if exponent > 10:
            return 10000.0
        elif exponent < -2:
            return 0.1

        distance = 10 ** exponent

        if not np.isfinite(distance):
            return 1.0

        if distance > 10000:
            return 10000.0
        elif distance < 0.1:
            return 0.1

        return distance

    except Exception as e:
        logger.debug("rssi_to_distance encountered an exception: %s. Returning fallback 1.0.", e, exc_info=True)
        return 1.0 
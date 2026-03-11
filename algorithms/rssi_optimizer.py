import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.optimize import curve_fit
from utils.configuration import ConfigurationManager
import logging

logger = logging.getLogger(__name__)


class RSSIOptimizer:
    def __init__(self, learn_no_wall_dis_rss, eq):
        self.learn_no_wall_dis_rss = learn_no_wall_dis_rss
        self.eq = eq
        cfg = ConfigurationManager().get_config()
        self.cfg = cfg
        self.d0 = cfg.rssi_optimizer.reference_distance_d0  # Reference distance d0

    def signal_model(self, distance, rssi_0, n):
        """
        Signal propagation model
        
        Args:
            distance: Array of distances
            rssi_0: Reference RSSI value
            n: Path loss exponent
            
        Returns:
            numpy.ndarray: Predicted RSSI values array
        """
        # Ensure distance is a numpy array and greater than 0
        distance = np.asarray(distance, dtype=np.float64)
        
        # Avoid log(0) error by setting distances less than or equal to 0 to a small positive number
        distance = np.where(distance <= 0, self.cfg.rssi_optimizer.small_positive_distance, distance)
        
        # Calculate RSSI values
        result = rssi_0 - 10 * n * np.log10(distance)
        
        # Ensure the result is a float array
        return np.asarray(result, dtype=np.float64)

    def optimize_no_wall(self):
        initial_params = [-40, 2.0]  # Initial guess for rssi_0 and n

        distances = np.asarray(self.learn_no_wall_dis_rss['dis'], dtype=np.float64).flatten()
        rssi_measured = np.asarray(self.learn_no_wall_dis_rss['rssi'], dtype=np.float64).flatten()

        valid_mask = (
            np.isfinite(distances) & 
            np.isfinite(rssi_measured) & 
            (distances > 0) & 
            (rssi_measured != 0)
        )
        
        distances = distances[valid_mask]
        rssi_measured = rssi_measured[valid_mask]
            
        # Check if data is empty or insufficient
        if len(distances) == 0 or len(rssi_measured) == 0:
    
            # Create a mock result object
            class MockResult:
                def __init__(self, x):
                    self.x = x
                    self.success = True
            
            # Return default parameters and default uncertainty values
            mock_result = MockResult(initial_params)
            sigma_rssi_0 = self.cfg.rssi_optimizer.default_sigma_rssi0  # Default uncertainty
            sigma_n = self.cfg.rssi_optimizer.default_sigma_n      # Default uncertainty
            
            return mock_result, sigma_rssi_0, sigma_n
        
        if len(distances) < self.cfg.rssi_optimizer.min_points_for_curve_fit:
            # Use minimize method only
            no_wall_result = minimize(self.loss_function, initial_params, args=(distances, rssi_measured))
            # Use default uncertainty values
            sigma_rssi_0 = self.cfg.rssi_optimizer.default_sigma_rssi0
            sigma_n = self.cfg.rssi_optimizer.default_sigma_n
            return no_wall_result, sigma_rssi_0, sigma_n
        
        try:
            # Use minimize method for optimization
            no_wall_result = minimize(self.loss_function, initial_params, args=(distances, rssi_measured))
            
            # Use curve_fit for parameter fitting and uncertainty estimation
            popt, pcov = curve_fit(
                self.signal_model, 
                distances, 
                rssi_measured,
                p0=initial_params,
                maxfev=self.cfg.rssi_optimizer.curve_fit_maxfev  # Increase maximum iteration count
            )
            rssi_0, n = popt  # Fitted rssi_0 and n
            
            # Check if covariance matrix is valid
            if np.any(np.diag(pcov) < 0):
                sigma_rssi_0 = self.cfg.rssi_optimizer.default_sigma_rssi0
                sigma_n = self.cfg.rssi_optimizer.default_sigma_n
            else:
                sigma_rssi_0, sigma_n = np.sqrt(np.diag(pcov))  # Parameter uncertainty (standard deviation)
                
        except Exception as e:
            logger.warning("curve_fit failed; falling back to minimize-only. Reason: %s", e, exc_info=True)
            # Use minimize method only
            no_wall_result = minimize(self.loss_function, initial_params, args=(distances, rssi_measured))
            # Use default uncertainty values
            sigma_rssi_0 = self.cfg.rssi_optimizer.default_sigma_rssi0
            sigma_n = self.cfg.rssi_optimizer.default_sigma_n

        return no_wall_result, sigma_rssi_0, sigma_n
    
    def loss_function(self, params, distances, rssi_measured):
        rssi_0, n = params
        rssi_predicted = rssi_0 - 10 * n * np.log10(distances)
        return np.mean((rssi_predicted - rssi_measured) ** 2)

    def optimize_wall_attenuation(self, L_d0, n):
        ave_val = []
        for eq_key in self.eq:
           
            distances = self.eq[eq_key]['dis']
            signal_strengths = self.eq[eq_key]['rssi']
            initial_guess = self.eq[eq_key]['at_val']
    
            angle = self.eq[eq_key]['angle']
          
            result = least_squares(self.residuals, initial_guess, args=(distances, signal_strengths, L_d0, self.d0, n, angle))
            if len(distances) >= 7:
                ave_val.append(result.x[0])
            
            self.eq[eq_key]['at_val'] = result.x[0]
         
        
        # Check if ave_val is empty to avoid np.mean warning for empty slice
        if len(ave_val) == 0:
            return self.eq, self.cfg.rssi_optimizer.default_wall_attenuation_fallback
        
        return self.eq, np.mean(ave_val)

    def residuals(self, initial_guess, distances, signal_strengths, L_d0, d0, n, angle):
        """
        Define residual function
        """

        w = initial_guess
        res = []
        for i in range(len(distances)):

            predicted_signal = L_d0 - 10 * n * np.log10(distances[i] / d0) + w
            
            res.append(predicted_signal - signal_strengths[i])
        return np.array(res).flatten()

  
   

# Parameters file - centralizing all simulation parameters

import numpy as np

#  Add all missing parameters used in simulation.py

# Buffer and system parameters
num_arms = 100
N_PRIORITIES = 3  #  Define N_PRIORITIES constant used in simulation

# Network and processing parameters
processing_time_dnn = 30  # milliseconds
frame_interval = 33.3  # milliseconds (30 fps)

# Simulation parameters
c_tilde = 1.0  # Confidence radius parameter for UCB

# Threshold parameters (from mab_functions.py)
threshold_min = 0.0
threshold_max = 1.0  #  Changed to 1.0 for confidence thresholds

def network_delay(transmission_rate):
    """
    Calculate network delay based on transmission rate
     Added function that was referenced in simulation.py
    """
    # Simple model: higher transmission rate = lower delay
    # This is a placeholder - adjust based on your network model
    base_delay = 100  # base delay in ms
    rate_factor = 1000 / max(transmission_rate, 1)  # avoid division by zero
    delay = base_delay + rate_factor + np.random.exponential(10)  # add some randomness
    return max(delay, 10)  # minimum 10ms delay

def generate_poisson_process(arrival_rate, duration):
    """
    Generate Poisson arrival process
     Added function for generating arrivals
    
    Args:
        arrival_rate: Average number of arrivals per time unit
        duration: Total duration of simulation
    
    Returns:
        List of boolean values indicating arrivals at each time step
    """
    arrivals = []
    for _ in range(duration):
        # Probability of arrival in this time step
        prob = arrival_rate / duration if duration > 0 else 0
        arrival = np.random.random() < prob
        arrivals.append(arrival)
    return arrivals

# Default confidence distribution parameters
DEFAULT_CONF_BRANCH_1_PARAMS = (0.7, 0.15)  # mean=0.7, std=0.15
DEFAULT_CONF_BRANCH_2_PARAMS = (0.8, 0.12)  # mean=0.8, std=0.12
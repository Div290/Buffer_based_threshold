# simulation

import numpy as np
import pandas as pd
from buffers import FIFO_buffer, Strict_Priority_buffer, HybridBuffer  #  Import buffer classes
from mab_functions import * 
from parameters import *  

#  Create aliases for buffer classes to match simulation usage
SimpleFIFOBuffer = FIFO_buffer
ThreePriorityBuffers = Strict_Priority_buffer

def run_simulation_system(turns, samples, poisson_process, transmission_rate, buffer_capacity, arrival_rate, system_type='fifo'):
    #  Proper buffer initialization
    if system_type == 'fifo':
        buffer_system = SimpleFIFOBuffer(buffer_capacity)
    elif system_type == 'priority':
        buffer_system = ThreePriorityBuffers(buffer_capacity)
    elif system_type == 'hybrid':
        buffer_system = HybridBuffer(buffer_capacity)
    else:
        buffer_system = SimpleFIFOBuffer(buffer_capacity)  # Default fallback

    avg_reward = [0.0] * num_arms
    times_chosen = [0] * num_arms
    upper_bounds = [float('inf')] * num_arms  #  Initialize with high values for exploration
    delays = [network_delay(transmission_rate) for _ in range(turns)]

    # Metrics of interest
    drop_probability = []
    correct_classifications = []
    total_throughput = []
    total_arrivals = []
    classification_errors = []
    buffer_occupancy = []
    priority_buffer_occupancy = [[] for _ in range(N_PRIORITIES)]
    thresholds = []
    computing_costs = []
    edge_processed = []
    cloud_processed = []
    edge_count = 0
    cloud_count = 0
    
    #  Add latency tracking
    processing_latencies = []

    # Main counters
    drop_samples = processed_samples = correct_samples = total_arrivals_count = 0
    dnn_next_free_time = t = 0

    # Arm initialization -  Better initialization
    initial_samples = samples[:min(num_arms, len(samples))].copy()
    for arm in range(num_arms):
        if arm < len(initial_samples):
            s = initial_samples[arm]
            times_chosen[arm] = 1
            reward = reward_function(0, 1, s)
            avg_reward[arm] = reward
            upper_bounds[arm] = avg_reward[arm] + confidence_radius(1, times_chosen[arm], c_tilde)
        else:
            avg_reward[arm] = 0.0
            times_chosen[arm] = 0
            upper_bounds[arm] = float('inf')  # High value for unexplored arms

    remaining_samples = samples[len(initial_samples):].copy() if len(samples) > len(initial_samples) else []

    # Main simulation loop
    for current_turn in range(turns):
        # New sample arrivals
        new_sample = None
        if current_turn < len(poisson_process) and poisson_process[current_turn] and remaining_samples:
            new_sample = remaining_samples.pop(0)
            total_arrivals_count += 1

        total_arrivals.append(total_arrivals_count)

        # Processing when DNN is available
        if current_turn >= dnn_next_free_time:
            sample_to_process = None
            processing_start_time = current_turn  #  Track when processing starts
            
            if new_sample:
                sample_to_process = new_sample
                new_sample = None
            else:
                sample_to_process = buffer_system.get_next_sample()

            if sample_to_process:
                # Arm selection
                chosen_arm = choose_arm(current_turn + 1, upper_bounds)
                threshold = arm_to_threshold(chosen_arm)

                current_queue_length = buffer_system.get_total_length()
                reward = reward_function(current_queue_length, current_turn + 1, sample_to_process)
                
                #  Proper UCB update
                if times_chosen[chosen_arm] > 0:
                    avg_reward[chosen_arm] += (reward - avg_reward[chosen_arm]) / times_chosen[chosen_arm]
                else:
                    avg_reward[chosen_arm] = reward
                    
                times_chosen[chosen_arm] += 1
                upper_bounds[chosen_arm] = avg_reward[chosen_arm] + confidence_radius(current_turn + 1, times_chosen[chosen_arm], c_tilde)

                # Processing decision based on confidence threshold
                processing_latency = 0
                if sample_to_process['conf_branch_1'] >= threshold:
                    result = sample_to_process['correct_branch_1']
                    edge_count += 1
                    processing_location = 'edge'
                    processing_latency = processing_time_dnn  # Only DNN processing time
                else:
                    result = sample_to_process['correct_branch_2']
                    cloud_count += 1
                    processing_location = 'cloud'
                    #  Add network delay for cloud processing
                    if current_turn < len(delays):
                        network_latency = delays[current_turn]
                    else:
                        network_latency = network_delay(transmission_rate)
                    processing_latency = processing_time_dnn + network_latency

                processing_latencies.append(processing_latency)

                # Track correctly classified samples (throughput)
                if result == 1:
                    correct_samples += 1

                correct_classifications.append(result)
                computing_costs.append(offloading_cost(current_queue_length))
                thresholds.append(threshold)
                edge_processed.append(edge_count)
                cloud_processed.append(cloud_count)
                processed_samples += 1

                # Processing time calculation for DNN availability
                dnn_duration = processing_time_dnn / frame_interval
                if processing_location == 'cloud':
                    if current_turn < len(delays):
                        dnn_duration += delays[current_turn] / frame_interval
                    else:
                        dnn_duration += network_delay(transmission_rate) / frame_interval

                dnn_next_free_time = current_turn + dnn_duration

        # Buffer management
        if new_sample:
            #  Handle different buffer types properly
            if system_type == 'fifo':
                success = buffer_system.add_sample(new_sample)
            else:
                success = buffer_system.add_sample(new_sample)
                
            if not success:
                drop_samples += 1

        # Metrics update
        drop_probability.append(drop_samples / total_arrivals_count if total_arrivals_count else 0)
        total_throughput.append(correct_samples / total_arrivals_count if total_arrivals_count else 0)
        buffer_occupancy.append(buffer_system.get_total_length())

        # Track priority-specific buffer occupancy
        priority_lengths = buffer_system.get_buffer_lengths()
        for i in range(N_PRIORITIES):
            if i < len(priority_lengths):
                priority_buffer_occupancy[i].append(priority_lengths[i])
            else:
                priority_buffer_occupancy[i].append(0)

    classification_errors = [1 - acc for acc in correct_classifications]

    return {
        'drop_probability': drop_probability,
        'correct_classifications': correct_classifications,
        'total_throughput': total_throughput,
        'total_arrivals': total_arrivals,
        'classification_errors': classification_errors,
        'buffer_occupancy': buffer_occupancy,
        'priority_buffer_occupancy': priority_buffer_occupancy,
        'thresholds': thresholds,
        'computing_costs': computing_costs,
        'edge_processed': edge_processed,
        'cloud_processed': cloud_processed,
        'edge_count': edge_count,
        'cloud_count': cloud_count,
        'avg_rewards': avg_reward,
        'correct_samples': correct_samples,
        'processed_samples': processed_samples,
        'processing_latencies': processing_latencies,  #  Added latency tracking
        'drop_count': drop_samples,  #  Added drop count
        'total_arrivals_count': total_arrivals_count  #  Added total arrivals count
    }

#  Add experiment runner function as requested
def run_experiment(buffer_type, arrival_rates, turns, n_samples, transmission_rate, buffer_capacity, 
                   conf_branch_1_params=DEFAULT_CONF_BRANCH_1_PARAMS, 
                   conf_branch_2_params=DEFAULT_CONF_BRANCH_2_PARAMS):
    """
    Run experiments with different arrival rates and return results as DataFrame
    
    Args:
        buffer_type: 'fifo', 'priority', or 'hybrid'
        arrival_rates: List of arrival rates to test
        turns: Number of simulation turns
        n_samples: Number of samples to generate
        transmission_rate: Network transmission rate
        buffer_capacity: Buffer capacity
        conf_branch_1_params: Tuple (mean, std) for conf_branch_1 normal distribution
        conf_branch_2_params: Tuple (mean, std) for conf_branch_2 normal distribution
    
    Returns:
        pandas DataFrame with experiment results
    """
    results = []
    
    for arrival_rate in arrival_rates:
        # Generate samples for this experiment
        samples = generate_samples(n_samples, conf_branch_1_params, conf_branch_2_params)
        
        # Generate Poisson arrival process
        poisson_process = generate_poisson_process(arrival_rate, turns)
        
        # Run simulation
        result = run_simulation_system(
            turns=turns,
            samples=samples,
            poisson_process=poisson_process,
            transmission_rate=transmission_rate,
            buffer_capacity=buffer_capacity,
            arrival_rate=arrival_rate,
            system_type=buffer_type
        )
        
        # Calculate final metrics
        final_throughput = result['correct_samples'] / result['total_arrivals_count'] if result['total_arrivals_count'] > 0 else 0
        avg_latency = np.mean(result['processing_latencies']) if result['processing_latencies'] else 0
        final_drop_rate = result['drop_count'] / result['total_arrivals_count'] if result['total_arrivals_count'] > 0 else 0
        cloud_offloading_rate = result['cloud_count'] / result['processed_samples'] if result['processed_samples'] > 0 else 0
        
        # Additional useful metrics
        avg_buffer_occupancy = np.mean(result['buffer_occupancy']) if result['buffer_occupancy'] else 0
        avg_computing_cost = np.mean(result['computing_costs']) if result['computing_costs'] else 0
        
        experiment_result = {
            'buffer_type': buffer_type,
            'arrival_rate': arrival_rate,
            'total_goodput': final_throughput,  # Requested metric
            'latency_ms': avg_latency,          # Requested metric
            'drop_rate': final_drop_rate,       # Requested metric
            'cloud_offloading': cloud_offloading_rate,  # Requested metric
            'avg_buffer_occupancy': avg_buffer_occupancy,
            'avg_computing_cost': avg_computing_cost,
            'total_arrivals': result['total_arrivals_count'],
            'correct_samples': result['correct_samples'],
            'processed_samples': result['processed_samples'],
            'edge_processed': result['edge_count'],
            'cloud_processed': result['cloud_count']
        }
        
        results.append(experiment_result)
    
    return pd.DataFrame(results)

def save_experiment_results(results_df, filename):
    """
    Save experiment results to CSV file
    
    Args:
        results_df: DataFrame with experiment results
        filename: Name of file to save (should end with .csv)
    """
    results_df.to_csv(filename, index=False)
    print(f"Experiment results saved to {filename}")
    return filename
from collections import deque
import heapq
import math
import numpy as np
import pandas as pd

# parameters
NUM_ARMS = 100
N_PRIORITIES = 3
PROCESSING_TIME_DNN = 30.0        # ms
FRAME_INTERVAL = 33.3             # ms (approx 30 fps)
C_TILDE = 1.0
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

def network_delay(transmission_rate):
    base_delay = 100.0
    rate_factor = 1000.0 / max(transmission_rate, 1e-6)
    delay = base_delay + rate_factor + np.random.exponential(10.0)
    return max(delay, 10.0)

def generate_poisson_process(arrival_rate, duration, mode='per_turn_prob'):
    """Generate boolean arrival flags per time-step.

    Modes:
      - 'per_turn_prob': arrival_rate is probability of arrival each turn (0..1).
      - 'poisson_per_turn': arrival_rate is lambda per turn for Poisson; treat >0 as arrival.
      - 'total_expected': arrival_rate is expected total number over duration -> p = arrival_rate/duration.
    """
    if duration <= 0:
        return []
    if mode == 'per_turn_prob':
        p = float(arrival_rate)
        p = min(max(p, 0.0), 1.0)
        return [np.random.random() < p for _ in range(duration)]
    elif mode == 'poisson_per_turn':
        # returns True if at least one arrival in the turn
        return [np.random.poisson(lam=float(arrival_rate)) > 0 for _ in range(duration)]
    else:  # total_expected
        p = float(arrival_rate) / float(duration)
        p = min(max(p, 0.0), 1.0)
        return [np.random.random() < p for _ in range(duration)]

# --- Buffers (same API as your simulation) ---
class FIFOBuffer:
    def __init__(self, max_capacity):
        self.buffer = deque()
        self.max_capacity = int(max_capacity)

    def add_sample(self, sample):
        if len(self.buffer) < self.max_capacity:
            self.buffer.append(sample)
            return True
        else:
            # drop oldest to make room (preserve behaviour)
            self.buffer.popleft()
            self.buffer.append(sample)
            return True

    def get_next_sample(self):
        return self.buffer.popleft() if self.buffer else None

    def get_total_length(self):
        return len(self.buffer)

    def get_buffer_lengths(self):
        return [len(self.buffer), 0, 0]

class StrictPriorityBuffer:
    def __init__(self, max_capacity):
        self.buffers = [deque() for _ in range(N_PRIORITIES)]
        self.max_capacity = int(max_capacity)

    def _assign_priority(self, sample):
        conf = sample.get('conf_branch_1', 0.0)
        if conf >= 0.8:
            return 1
        elif conf >= 0.6:
            return 2
        else:
            return 3

    def add_sample(self, sample, priority=None):
        if priority is None:
            priority = self._assign_priority(sample)
        idx = max(0, min(priority - 1, N_PRIORITIES - 1))
        if len(self.buffers[idx]) < self.max_capacity:
            self.buffers[idx].append(sample)
            return True
        else:
            return False

    def get_next_sample(self):
        for b in self.buffers:
            if b:
                return b.popleft()
        return None

    def get_total_length(self):
        return sum(len(b) for b in self.buffers)

    def get_buffer_lengths(self):
        return [len(b) for b in self.buffers]

class HybridBuffer:
    def __init__(self, max_capacity):
        self.buffer = []  # min-heap by (priority, counter)
        self.max_capacity = int(max_capacity)
        self.counter = 0

    def _assign_priority(self, sample):
        conf = sample.get('conf_branch_1', 0.0)
        if conf >= 0.8:
            return 1
        elif conf >= 0.6:
            return 2
        else:
            return 3

    def add_sample(self, sample, priority=None):
        if priority is None:
            priority = self._assign_priority(sample)
        entry = (priority, self.counter, sample)
        self.counter += 1
        if len(self.buffer) < self.max_capacity:
            heapq.heappush(self.buffer, entry)
            return True
        else:
            # replace worst (largest priority number) if new is better
            worst = max(self.buffer, key=lambda x: (x[0], x[1]))
            if entry[0] < worst[0]:
                self.buffer.remove(worst)
                heapq.heapify(self.buffer)
                heapq.heappush(self.buffer, entry)
                return True
            else:
                return False

    def get_next_sample(self):
        if self.buffer:
            _, _, sample = heapq.heappop(self.buffer)
            return sample
        return None

    def get_total_length(self):
        return len(self.buffer)

    def get_buffer_lengths(self):
        counts = [0]*N_PRIORITIES
        for p, _, _ in self.buffer:
            if 1 <= p <= N_PRIORITIES:
                counts[p-1] += 1
        return counts

# --- Simple MAB helpers (UCB-like) ---
def confidence_radius(current_time, times_chosen, c_tilde=C_TILDE):
    if times_chosen <= 0:
        return float('inf')
    return math.sqrt(c_tilde * math.log(max(1.0, current_time)) / times_chosen)

def choose_arm(current_time, upper_bounds):
    ub = np.array(upper_bounds, dtype=float)
    candidates = np.where(ub == ub.max())[0]
    return int(np.random.choice(candidates))

def arm_to_threshold(arm_index, num_arms=NUM_ARMS):
    return THRESHOLD_MIN + (arm_index / max(1, num_arms-1)) * (THRESHOLD_MAX - THRESHOLD_MIN)

def offloading_cost(queue_length):
    return 0.01 * queue_length

def reward_function(current_queue_length, current_time, sample, penalty_offload=0.0):
    r = sample.get('correct_branch_1', 0)
    return float(r) - 0.001 * float(current_queue_length) - penalty_offload

# --- Dataset helper: convert DataFrame rows to simulation samples ---
def dataframe_to_samples(df, n_samples=None, shuffle=True):
    df2 = df.copy()
    if shuffle:
        df2 = df2.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if n_samples is not None:
        df2 = df2.iloc[:n_samples]
    samples = []
    for _, row in df2.iterrows():
        s = {
            'conf_branch_1': float(row.get('conf_branch_1', 0.0)),
            'correct_branch_1': int(row.get('correct_branch_1', 0)),
            'delta_inf_time_branch_1': float(row.get('delta_inf_time_branch_1', 0.0)),
            'cum_inf_time_branch_1': float(row.get('cum_inf_time_branch_1', 0.0)),
            'prediction_branch_1': int(row.get('prediction_branch_1', -1)),
            'conf_branch_2': float(row.get('conf_branch_2', 0.0)),
            'correct_branch_2': int(row.get('correct_branch_2', 0)),
            'delta_inf_time_branch_2': float(row.get('delta_inf_time_branch_2', 0.0)),
            'cum_inf_time_branch_2': float(row.get('cum_inf_time_branch_2', 0.0)),
            'prediction_branch_2': int(row.get('prediction_branch_2', -1))
        }
        samples.append(s)
    return samples

# --- Main simulation function ---
def run_simulation_system(turns, samples, poisson_process, transmission_rate, buffer_capacity,
                          arrival_rate=0.0, system_type='fifo'):
    # buffer selection
    if system_type == 'fifo':
        buffer_system = FIFOBuffer(buffer_capacity)
    elif system_type == 'priority':
        buffer_system = StrictPriorityBuffer(buffer_capacity)
    elif system_type == 'hybrid':
        buffer_system = HybridBuffer(buffer_capacity)
    else:
        buffer_system = FIFOBuffer(buffer_capacity)

    num_arms = NUM_ARMS
    avg_reward = [0.0] * num_arms
    times_chosen = [0] * num_arms
    upper_bounds = [float('inf')] * num_arms

    # initialize arms with up to num_arms initial samples (if available)
    initial_samples = samples[:min(num_arms, len(samples))]
    # --- IMPORTANT CHANGE: do NOT remove initial_samples from remaining_samples
    remaining_samples = samples.copy() if len(samples) > 0 else []

    for arm in range(num_arms):
        if arm < len(initial_samples):
            s = initial_samples[arm]
            times_chosen[arm] = 1
            r = reward_function(0, 1, s)
            avg_reward[arm] = r
            upper_bounds[arm] = avg_reward[arm] + confidence_radius(1, times_chosen[arm], C_TILDE)
        else:
            avg_reward[arm] = 0.0
            times_chosen[arm] = 0
            upper_bounds[arm] = float('inf')

    delays = [network_delay(transmission_rate) for _ in range(turns)]

    # Metrics and counters
    drop_samples = 0
    processed_samples = 0
    correct_samples = 0
    total_arrivals_count = 0
    dnn_next_free_time = 0.0

    drop_probability = []
    correct_classifications = []
    total_throughput = []
    total_arrivals = []
    classification_errors = []
    buffer_occupancy = []
    priority_buffer_occupancy = [[] for _ in range(N_PRIORITIES)]
    thresholds = []
    computing_costs = []
    edge_processed = 0
    cloud_processed = 0
    processing_latencies = []

    for current_turn in range(turns):
        # arrivals
        new_sample = None
        if current_turn < len(poisson_process) and poisson_process[current_turn] and remaining_samples:
            new_sample = remaining_samples.pop(0)
            total_arrivals_count += 1

        total_arrivals.append(total_arrivals_count)

        # DNN free?
        if current_turn >= dnn_next_free_time:
            sample_to_process = None
            if new_sample:
                sample_to_process = new_sample
                new_sample = None
            else:
                sample_to_process = buffer_system.get_next_sample()

            if sample_to_process is not None:
                chosen_arm = choose_arm(current_turn + 1, upper_bounds)
                threshold = arm_to_threshold(chosen_arm, num_arms)

                current_queue_length = buffer_system.get_total_length()
                reward = reward_function(current_queue_length, current_turn + 1, sample_to_process)

                # update UCB stats (increment then update avg)
                if times_chosen[chosen_arm] > 0:
                    avg_reward[chosen_arm] += (reward - avg_reward[chosen_arm]) / times_chosen[chosen_arm]
                else:
                    avg_reward[chosen_arm] = reward
                times_chosen[chosen_arm] += 1
                upper_bounds[chosen_arm] = avg_reward[chosen_arm] + confidence_radius(current_turn + 1, times_chosen[chosen_arm], C_TILDE)

                # decision: process at edge or cloud
                if sample_to_process.get('conf_branch_1', 0.0) >= threshold:
                    result = sample_to_process.get('correct_branch_1', 0)
                    edge_processed += 1
                    processing_latency = PROCESSING_TIME_DNN
                    processing_location = 'edge'
                else:
                    result = sample_to_process.get('correct_branch_2', 0)
                    cloud_processed += 1
                    network_latency = delays[current_turn] if current_turn < len(delays) else network_delay(transmission_rate)
                    processing_latency = PROCESSING_TIME_DNN + network_latency
                    processing_location = 'cloud'

                processing_latencies.append(processing_latency)

                if result == 1:
                    correct_samples += 1

                correct_classifications.append(result)
                computing_costs.append(offloading_cost(current_queue_length))
                thresholds.append(threshold)
                processed_samples += 1

                # compute next free time for DNN (in turns)
                dnn_duration = PROCESSING_TIME_DNN / FRAME_INTERVAL
                if processing_location == 'cloud':
                    dnn_duration += network_latency / FRAME_INTERVAL
                dnn_next_free_time = current_turn + dnn_duration

        # buffer insertion if new sample was not processed immediately
        if new_sample is not None:
            success = buffer_system.add_sample(new_sample)
            if not success:
                drop_samples += 1

        # metrics record
        drop_probability.append(drop_samples / total_arrivals_count if total_arrivals_count else 0.0)
        total_throughput.append(correct_samples / total_arrivals_count if total_arrivals_count else 0.0)
        buffer_occupancy.append(buffer_system.get_total_length())
        lengths = buffer_system.get_buffer_lengths()
        for i in range(N_PRIORITIES):
            priority_buffer_occupancy[i].append(lengths[i] if i < len(lengths) else 0)

    classification_errors = [1.0 - float(x) for x in correct_classifications]

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
        'edge_count': edge_processed,
        'cloud_count': cloud_processed,
        'avg_rewards': avg_reward,
        'correct_samples': correct_samples,
        'processed_samples': processed_samples,
        'processing_latencies': processing_latencies,
        'drop_count': drop_samples,
        'total_arrivals_count': total_arrivals_count
    }

# --- Experiment runner ---
def run_experiment_from_dataframe(df, buffer_type, arrival_rates, turns, n_samples, transmission_rate, buffer_capacity, arrival_mode='per_turn_prob'):
    results = []
    for arrival_rate in arrival_rates:
        samples = dataframe_to_samples(df, n_samples=n_samples, shuffle=True)
        poisson_process = generate_poisson_process(arrival_rate, turns, mode=arrival_mode)
        res = run_simulation_system(turns=turns, samples=samples, poisson_process=poisson_process,
                                    transmission_rate=transmission_rate, buffer_capacity=buffer_capacity,
                                    arrival_rate=arrival_rate, system_type=buffer_type)
        final_throughput = res['correct_samples'] / res['total_arrivals_count'] if res['total_arrivals_count'] > 0 else 0.0
        avg_latency = float(np.mean(res['processing_latencies'])) if res['processing_latencies'] else 0.0
        final_drop_rate = res['drop_count'] / res['total_arrivals_count'] if res['total_arrivals_count'] > 0 else 0.0
        cloud_offloading_rate = res['cloud_count'] / res['processed_samples'] if res['processed_samples'] > 0 else 0.0
        avg_buffer_occupancy = float(np.mean(res['buffer_occupancy'])) if res['buffer_occupancy'] else 0.0
        avg_computing_cost = float(np.mean(res['computing_costs'])) if res['computing_costs'] else 0.0

        results.append({
            'buffer_type': buffer_type,
            'arrival_rate': arrival_rate,
            'arrival_mode': arrival_mode,
            'total_goodput': final_throughput,
            'latency_ms': avg_latency,
            'drop_rate': final_drop_rate,
            'cloud_offloading': cloud_offloading_rate,
            'avg_buffer_occupancy': avg_buffer_occupancy,
            'avg_computing_cost': avg_computing_cost,
            'total_arrivals': res['total_arrivals_count'],
            'correct_samples': res['correct_samples'],
            'processed_samples': res['processed_samples'],
            'edge_processed': res['edge_count'],
            'cloud_processed': res['cloud_count']
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    import os
    path = os.getenv("SIM_CSV_PATH", "inf_data_ee_mobilenet_1_branches_crescent_id_3_laptop_cifar10.csv")
    df = pd.read_csv(path)
    print("Loaded", len(df), "rows. Columns:", df.columns.tolist())
    out = run_experiment_from_dataframe(df, buffer_type='fifo', arrival_rates=[2, 3], turns=10000, n_samples=1000, transmission_rate=100.0, buffer_capacity=7)
    print(out)

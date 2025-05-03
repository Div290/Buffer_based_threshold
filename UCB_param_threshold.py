import numpy as np
import pandas as pd
from scipy.stats import poisson, expon
import param
import os

# Load data
csv_path = param.dataframe_save_path
if not os.path.exists(csv_path):
    print("ERROR: CSV file not found!")
    exit(1)

df = pd.read_csv(csv_path)
df_grouped = df.groupby("Sample_ID").agg({
    "Confidence": list,
    "Prediction": list,
    "Label": "first"
}).reset_index()
df = df_grouped

exit_layer = param.exit_layer

class UCBParam:
    """UCB for parameter vectors theta = (theta1, theta2)."""
    def __init__(self, theta_candidates):
        self.theta_candidates = theta_candidates  # list of tuples (theta1, theta2)
        self.counts = np.zeros(len(theta_candidates))  # N(theta)
        self.values = np.zeros(len(theta_candidates))  # Q(theta) average reward

    def select_theta(self, t):
        """Select theta maximizing UCB score."""
        ucb_scores = self.values + np.sqrt(2 * np.log(t + 1) / (self.counts + 1))
        return np.argmax(ucb_scores)

    def update(self, index, reward):
        """Update counts and average reward for chosen theta."""
        self.counts[index] += 1
        n = self.counts[index]
        self.values[index] += (reward - self.values[index]) / n

class QueueSystem:
    def __init__(self, arrival_rate, theta_candidates, max_queue_size=10, p_f=1.0, p_s=5.0, mu=0.01, kappa=0.05):
        self.arrival_rate = arrival_rate
        self.max_queue_size = max_queue_size
        self.queue = []
        self.processed_samples = 0
        self.lost_samples = 0
        self.total_samples = 0
        self.current_time = 0
        self.server_busy_until = 0
        self.next_arrival_time = expon.rvs(scale=1/self.arrival_rate)
        self.acc = 0
        self.t = 0
        self.exit = 0

        self.ucb = UCBParam(theta_candidates)
        self.p_f = p_f  # fast processing time
        self.p_s = p_s  # slow processing time
        self.mu = mu    # penalty slope
        self.kappa = kappa  # penalty intercept

    def run(self, total_samples):
        while self.total_samples < total_samples or self.queue:
            if not self.queue or self.next_arrival_time < self.server_busy_until:
                self.current_time = self.next_arrival_time
                self.handle_arrivals()
                self.next_arrival_time = self.current_time + expon.rvs(scale=1/self.arrival_rate)
            else:
                self.current_time = self.server_busy_until
                self.process_sample()

            if self.total_samples > 1:
                print(f"Queue Size: {len(self.queue)}, Correctly classified: {self.acc/self.total_samples:.4f}, "
                      f"Processed: {self.processed_samples}, Lost: {self.lost_samples/self.total_samples:.4f}, Total: {self.total_samples}")

        while self.queue:
            self.process_sample()

        print("\nFinal UCB values:", self.ucb.values)
        print("Total Processed Samples:", self.processed_samples)
        print("Total Lost Samples:", self.lost_samples)
        print("Total Samples:", self.total_samples)
        exit_fraction = self.exit / self.processed_samples if self.processed_samples > 0 else 0.0
        print("Fraction of exited samples:", exit_fraction)
        print("Accuracy is", self.acc / self.processed_samples if self.processed_samples > 0 else 0.0)
        best_index = np.argmax(self.ucb.values)
        print(f"Best theta: {self.ucb.theta_candidates[best_index]}, with average reward {self.ucb.values[best_index]:.4f}")

    def handle_arrivals(self):
        arrivals = poisson.rvs(self.arrival_rate)
        available_space = self.max_queue_size - len(self.queue)
        num_accepted = min(arrivals, available_space)
        num_lost = arrivals - num_accepted

        new_samples = []
        for _ in range(num_accepted):
            if self.t < len(df):
                confidence = df["Confidence"][self.t][exit_layer]
                confidence_last = df["Confidence"][self.t][-1]
                prediction_exit = df["Prediction"][self.t][exit_layer]
                prediction_final = df["Prediction"][self.t][-1]
                true_label = df["Label"][self.t]
                self.t += 1
                new_samples.append({
                    'confidence': confidence,
                    'confidence_last': confidence_last,
                    'prediction_exit': prediction_exit,
                    'prediction_final': prediction_final,
                    'true_label': true_label,
                    'arrival_time': self.current_time
                })

        self.t += num_lost
        self.queue.extend(new_samples)
        self.lost_samples += num_lost
        self.total_samples += arrivals

    def process_sample(self):
        q = len(self.queue)
        if q == 0:
            return

        # Select theta_t using UCB
        self.t += 1
        theta_index = self.ucb.select_theta(self.t)
        theta = self.ucb.theta_candidates[theta_index]

        # Compute threshold alpha_t = theta_1 * q + theta_2
        alpha_t = theta[0] * q + theta[1]

        sample = self.queue.pop(0)
        C_I = sample['confidence']
        C_L = sample['confidence_last']

        if C_I >= alpha_t:
            processing_time = self.p_f
            reward = 0
            prediction = sample['prediction_exit']
            self.exit += 1
        else:
            processing_time = self.p_s
            reward = max(C_L - C_I, 0) - (self.mu * q - self.kappa)
            prediction = sample['prediction_final']

        if prediction == sample['true_label']:
            self.acc += 1

        self.server_busy_until = self.current_time + processing_time
        self.processed_samples += 1

        # Update UCB for chosen theta
        self.ucb.update(theta_index, reward)

# Example usage:
theta_candidates = [(t1, t2) for t1 in np.arange(0, 1.1, 0.1) for t2 in np.arange(0, 1.1, 0.1)]
queue_system = QueueSystem(param.arrival_rate, theta_candidates, param.max_queue_size, p_f=param.e_process_time, p_s=param.c_process_time, mu=0.01, kappa=0.05)
queue_system.run(df.shape[0])

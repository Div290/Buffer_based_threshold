import pandas as pd
import param
import os
import numpy as np
from scipy.stats import poisson, expon

# Load CSV
csv_path = param.dataframe_save_path
if not os.path.exists(csv_path):
    print("CSV file not found! Run:\npython main.py --pretrain --src dataset --batch_size 32 --pre_epochs 2")
    exit(1)

df = pd.read_csv(csv_path)
df_grouped = df.groupby("Sample_ID").agg({
    "Confidence": list,
    "Prediction": list,
    "Label": "first"
}).reset_index()
df = df_grouped
print("Loaded dataset with", df.shape[0], "samples")

exit_layer = param.exit_layer

class TsallisPolicyGradient:
    def __init__(self, thresholds, tsallis_lambda=1.5, lr=0.1, baseline_beta=0.9):
        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        self.preferences = np.zeros(self.n_arms)
        self.tsallis_lambda = tsallis_lambda
        self.lr = lr
        self.baseline_beta = baseline_beta
        self.baseline = 0.0  # Initialize baseline

    def exp_tsallis(self, u):
        lam = self.tsallis_lambda
        if lam == 1:
            return np.exp(u)
        return np.maximum(1 + (1 - lam) * u, 1e-8) ** (1 / (1 - lam))

    def softmax(self):
        tsallis_exp = self.exp_tsallis(self.preferences)
        probs = tsallis_exp / np.sum(tsallis_exp)
        return probs

    def select_arm(self):
        probs = self.softmax()
        return np.random.choice(self.n_arms, p=probs)

    def update(self, arm, reward):
        probs = self.softmax()

        # Update baseline using exponential moving average
        self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * reward
        adjusted_reward = reward - self.baseline

        for a in range(self.n_arms):
            indicator = 1 if a == arm else 0
            theta = self.preferences[a]
            lam = self.tsallis_lambda
            denom = max(1 + (1 - lam) * theta, 1e-8)  # Avoid division by zero
            grad = (indicator - probs[a]) * (1 / denom)
            self.preferences[a] += self.lr * adjusted_reward * grad


class QueueSystem:
    def __init__(self, arrival_rate, threshold_choices, max_queue_size=10):
        self.arrival_rate = arrival_rate
        self.max_queue_size = max_queue_size
        self.threshold_choices = threshold_choices
        self.pg_instances = {
            q: TsallisPolicyGradient(threshold_choices[q]) for q in range(1, max_queue_size + 1)
        }
        self.queue = []
        self.processed_samples = 0
        self.lost_samples = 0
        self.total_samples = 0
        self.current_time = 0
        self.server_busy_until = 0
        self.next_arrival_time = expon.rvs(scale=1/self.arrival_rate)
        self.acc = 0
        self.exit = 0
        self.t = 0

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
                acc = self.acc / self.total_samples
                inc = (self.processed_samples - self.acc) / self.total_samples
                loss = self.lost_samples / self.total_samples
                print(f"Queue: {len(self.queue)} | Acc: {acc:.4f} | Inc: {inc:.4f} | Proc: {self.processed_samples} | Lost: {loss:.4f}")

        while self.queue:
            self.process_sample()

        print("\nFinal Preferences:", {q: pg.preferences for q, pg in self.pg_instances.items()})
        print("Processed:", self.processed_samples)
        print("Lost:", self.lost_samples)
        print("Accuracy:", self.acc / self.processed_samples)
        print("\nBest thresholds:")
        for q, pg in self.pg_instances.items():
            best = pg.thresholds[np.argmax(pg.preferences)]
            print(f"Queue {q}: Threshold = {best:.2f}")

    def handle_arrivals(self):
        arrivals = poisson.rvs(self.arrival_rate)
        space = self.max_queue_size - len(self.queue)
        accepted = min(arrivals, space)
        lost = arrivals - accepted

        for _ in range(accepted):
            if self.t < len(df):
                row = df.iloc[self.t]
                self.queue.append({
                    'confidence': row["Confidence"][exit_layer],
                    'confidence_last': row["Confidence"][-1],
                    'prediction_exit': row["Prediction"][exit_layer],
                    'prediction_final': row["Prediction"][-1],
                    'true_label': row["Label"],
                    'arrival_time': self.current_time
                })
            self.t += 1

        self.t += lost
        self.total_samples += arrivals
        self.lost_samples += lost

    def process_sample(self):
        qsize = len(self.queue)
        if qsize == 0:
            return

        sample = self.queue.pop(0)
        pg = self.pg_instances[qsize]
        arm = pg.select_arm()
        threshold = pg.thresholds[arm]

        c_i = sample['confidence']
        c_l = sample['confidence_last']

        if c_i >= threshold:
            time = 0.1
            reward = -time
            prediction = sample['prediction_exit']
            self.exit += 1
        else:
            time = 0.8
            reward = max(c_l - c_i, 0) - (0.1 * qsize - 0.05) - (1/10000) * time
            prediction = sample['prediction_final']

        if prediction == sample['true_label']:
            self.acc += 1

        self.server_busy_until = self.current_time + time
        pg.update(arm, reward)
        self.processed_samples += 1

# Example configuration
threshold_choices = {
    q: [0.55, 0.65, 0.75, 0.85, 0.95] for q in range(1, 11)
}

total_samples = df.shape[0]
queue_system = QueueSystem(param.arrival_rate, threshold_choices, param.max_queue_size)
queue_system.run(total_samples)

import pandas as pd
import param
import os
import numpy as np
from scipy.stats import poisson, expon

# Path to the CSV file
csv_path = param.dataframe_save_path  # Modify as needed


# Check if the file exists
if not os.path.exists(csv_path):
    print("ERROR: CSV file not found!")
    print('Please first create the CSV file by running the following command:\n')
    print('python main.py --pretrain --src dataset --batch_size 32 --pre_epochs 2\n')
    exit(1)  # Exit the program with an error code

# Load the data
df = pd.read_csv(csv_path)
print("CSV file loaded successfully!")


# Load the DataFrame (assuming it's already read as `df`)

# Group by Sample_ID and aggregate Confidence & Prediction as lists
df_grouped = df.groupby("Sample_ID").agg({
    "Confidence": list,
    "Prediction": list,
    "Label": "first"  # Assuming the true label is the same across exits
}).reset_index()


df = df_grouped

print("The dataset size is", df.shape[0])


exit_layer = param.exit_layer
t = 0
exit = 0

class PolicyGradient:
    """Policy Gradient (REINFORCE) with running average baseline for threshold selection."""
    def __init__(self, thresholds, lr=0.1):
        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        self.preferences = np.zeros(self.n_arms)
        self.probs = np.ones(self.n_arms) / self.n_arms
        self.lr = lr
        self.baseline = 0
        self.beta = 0.9  # smoothing factor for running average

    def softmax(self):
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # stability
        self.probs = exp_prefs / np.sum(exp_prefs)
        return self.probs

    def select_arm(self):
        self.softmax()
        return np.random.choice(self.n_arms, p=self.probs)

    def update(self, arm, reward):
        # Update baseline as a running average
        self.baseline = self.beta * self.baseline + (1 - self.beta) * reward

        self.softmax()
        grad = -self.probs
        grad[arm] += 1
        self.preferences += self.lr * (reward - self.baseline) * grad


class QueueSystem:
    def __init__(self, arrival_rate, threshold_choices, max_queue_size=10):
        self.arrival_rate = arrival_rate
        self.max_queue_size = max_queue_size
        self.threshold_choices = threshold_choices
        self.pg_instances = {q: PolicyGradient(threshold_choices[q]) for q in range(1, max_queue_size + 1)}

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

    def run(self, total_samples):
        while self.total_samples < total_samples or self.queue:
            if not self.queue or self.next_arrival_time < self.server_busy_until:
                self.current_time = self.next_arrival_time
                self.handle_arrivals()
                self.next_arrival_time = self.current_time + expon.rvs(scale=1/self.arrival_rate)
            else:
                self.current_time = self.server_busy_until
                self.process_sample()

            current_accuracy = self.acc
            exit_fraction = self.exit / self.processed_samples if self.processed_samples > 0 else 0.0
            if self.total_samples > 1:
                print(f"Queue Size: {len(self.queue)}, Correctly classified: {current_accuracy/self.total_samples:.4f},  Incorrectly classified: {(self.processed_samples - current_accuracy)/self.total_samples:.4f}, Processed: {self.processed_samples}, Lost: {self.lost_samples/self.total_samples}, Total: {self.total_samples}")

        while self.queue:
            self.process_sample()

        print("\nFinal Policy Preferences:", {q: pg.preferences for q, pg in self.pg_instances.items()})
        print("Total Processed Samples:", self.processed_samples)
        print("Total Lost Samples:", self.lost_samples)
        print("Total Samples:", self.total_samples)
        print("Fraction of exited samples:", exit_fraction)
        print("Accuracy is", current_accuracy / self.processed_samples)
        print("\nThreshold that maximized policy preference for each buffer size:")
        for q, pg in self.pg_instances.items():
            max_index = np.argmax(pg.preferences)
            best_threshold = pg.thresholds[max_index]
            print(f"Buffer Size {q}: Best Threshold = {best_threshold:.3f}")

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
                new_samples.append({'confidence': confidence, 'confidence_last': confidence_last, 'prediction_exit': prediction_exit, 'prediction_final': prediction_final, 'true_label': true_label, 'arrival_time': self.current_time})

        self.t += num_lost
        self.queue.extend(new_samples)
        self.lost_samples += num_lost
        self.total_samples += arrivals

    def process_sample(self):
        current_queue_size = len(self.queue)
        if current_queue_size == 0:
            return

        sample = self.queue.pop(0)
        # reward = 0#[_ for _ in range(self.max_queue_size)]

        pg = self.pg_instances
        arm = pg[current_queue_size].select_arm()
        threshold = pg[current_queue_size].thresholds[arm]

        confidence = sample['confidence']
        confidence_last = sample['confidence_last']

        if confidence >= threshold:
            processing_time = 0.1
            reward = -processing_time
            prediction = sample['prediction_exit']
            self.exit += 1
        else:
            processing_time = 0.8
            # for j in range(self.max_queue_size):
            reward = confidence_last - confidence - (1/(10*self.max_queue_size)) * current_queue_size - (1/10000)*processing_time
            prediction = sample['prediction_final']

        if prediction == sample['true_label']:
            self.acc += 1

        self.server_busy_until = self.current_time + processing_time
        pg[current_queue_size].update(arm, reward)
        self.processed_samples += 1

# Thresholds
threshold_choices = {
    q: [0.55, 0.65, 0.75, 0.85, 0.95] for q in range(1, 11)
}

total_samples = df.shape[0]
queue_system = QueueSystem(param.arrival_rate, threshold_choices, param.max_queue_size)
queue_system.run(total_samples)

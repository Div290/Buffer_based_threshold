import pandas as pd
import param
import os

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


import numpy as np
from scipy.stats import poisson, expon

exit_layer = param.exit_layer
t = 0
exit = 0

class UCB:
    """Upper Confidence Bound (UCB) for dynamic decision making."""
    def __init__(self, thresholds):
        self.n_arms = len(thresholds)
        self.counts = np.zeros(self.n_arms)  # Number of times each arm is selected
        self.values = np.zeros(self.n_arms)  # Estimated reward per arm
        self.thresholds = thresholds  # Store thresholds for this queue size

    def select_arm(self):
        """ Selects an arm using UCB formula. """
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            return int(total_counts)  # Ensure each arm is tried at least once

        ucb_values = self.values + np.sqrt(2 * np.log(total_counts + 1e-6) / (self.counts + 1e-5))
        return int(np.argmax(ucb_values))  # Explicitly cast to int

    def update(self, arm, reward):
        """ Updates UCB estimates with a new reward. """
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class QueueSystem:
    """Queueing system with Poisson arrivals and processing times."""
    def __init__(self, arrival_rate, threshold_choices, max_queue_size=10):
        self.arrival_rate = arrival_rate  # Poisson arrival rate (λ)
        self.max_queue_size = max_queue_size
        self.threshold_choices = threshold_choices  # Different choices per queue size

        # Initialize UCB instances with different threshold sets for each queue size
        self.ucb_instances = {q: UCB(threshold_choices[q]) for q in range(1, max_queue_size + 1)}

        self.queue = []  # Queue of samples
        self.processed_samples = 0
        self.lost_samples = 0
        self.total_samples = 0
        self.current_time = 0  # Track simulation time
        self.server_busy_until = 0  # When server will be free
        self.next_arrival_time = expon.rvs(scale=1/self.arrival_rate)  # First arrival
        self.acc = 0
        self.t = 0
        self.exit = 0

    def run(self, total_samples):
        """Run the queue simulation until all samples are processed."""
        while self.total_samples < total_samples or self.queue:
            # Determine the next event (either arrival or end of processing)
            if not self.queue or self.next_arrival_time < self.server_busy_until:
                self.current_time = self.next_arrival_time
                self.handle_arrivals()
                self.next_arrival_time = self.current_time + expon.rvs(scale=1/self.arrival_rate)
            else:
                self.current_time = self.server_busy_until
                self.process_sample()

            # Compute current accuracy
            current_accuracy = self.acc
            exit_fraction = self.exit / self.processed_samples if self.processed_samples > 0 else 0.0
            if self.total_samples>1:
              print(f"Queue Size: {len(self.queue)}, Correctly classified: {current_accuracy/self.total_samples:.4f},  Incorrectly classified: {(self.processed_samples - current_accuracy)/self.total_samples:.4f}, Processed: {self.processed_samples}, Lost: {self.lost_samples/self.total_samples}, Total: {self.total_samples}")

        # Process remaining samples in the queue before stopping
        while self.queue:
            self.process_sample()

        # Print final results
        print("\nFinal UCB values:", {q: ucb.values for q, ucb in self.ucb_instances.items()})
        print("Total Processed Samples:", self.processed_samples)
        print("Total Lost Samples:", self.lost_samples)
        print("Total Samples:", self.total_samples)
        print("Fraction of exited samples:", exit_fraction)
        print("Accuracy is", current_accuracy / self.processed_samples)
        # Find the threshold that maximized the UCB value for each buffer size
        print("\nThreshold that maximized UCB value for each buffer size:")
        for q, ucb in self.ucb_instances.items():
            if len(ucb.values) > 0:
                max_index = np.argmax(ucb.values)  # Index of max UCB value
                best_threshold = ucb.thresholds[max_index]  # Best threshold
                print(f"Buffer Size {q}: Best Threshold = {best_threshold:.3f}")
    def handle_arrivals(self):
        """Handles new arrivals based on Poisson process with interarrival times."""

        arrivals = poisson.rvs(self.arrival_rate)  # Number of arrivals
        available_space = self.max_queue_size - len(self.queue)
        num_accepted = min(arrivals, available_space)
        num_lost = arrivals - num_accepted

        # Create sample objects with random confidence
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
        """Processes one sample if the server is available."""
        current_queue_size = len(self.queue)  # Get current queue size

        if current_queue_size == 0:
            return  # No UCB update if queue is empty

        sample = self.queue.pop(0)  # Get sample
        reward = [_ for _ in range(self.max_queue_size)]

        ucb = self.ucb_instances#[current_queue_size]  # Get the UCB instance for this queue size
        # print(ucb)
        arm = ucb[current_queue_size].select_arm()
        threshold = ucb[current_queue_size].thresholds[arm]  # Pick threshold based on the arm selection

        confidence = sample['confidence']
        confidence_last = sample['confidence_last']

        if confidence >= threshold:
            processing_time = param.e_process_time  # Fast processing
            for j in range(self.max_queue_size):
                reward[j] = -processing_time
            # reward = -processing_time
            prediction = sample['prediction_exit']
            self.exit += 1
        else:
            processing_time = param.c_process_time # Slow processing
            for j in range(self.max_queue_size):
                # print((1/(500*self.max_queue_size)) * j)
                # reward[j] = max(0, confidence_last - confidence - (1/10*self.max_queue_size) * j - (1/10000)*processing_time)
                reward[j] = confidence_last - confidence - (1/(10*self.max_queue_size)) * j - (1/10000)*processing_time
                # print(reward[j])
            # reward = confidence_last - confidence - (1/(10*self.max_queue_size)) * len(self.queue) - (1/10000)*processing_time
            prediction = sample['prediction_final']

        if prediction == sample['true_label']:
            self.acc += 1  # Count correct predictions

        self.server_busy_until = self.current_time + processing_time  # Set server busy time
        for j in range(self.max_queue_size):
            ucb[j+1].update(arm, reward[j])
        # ucb.update(arm, reward)

        self.processed_samples += 1  # Count as processed


# Define different threshold choices for each queue size
threshold_choices = {
    1: [0.55, 0.65, 0.75, 0.85, 0.95],   # Lower thresholds for small queues
    2: [0.55, 0.65, 0.75, 0.85, 0.95],
    3: [0.55, 0.65, 0.75, 0.85, 0.95],
    4: [0.55, 0.65, 0.75, 0.85, 0.95],
    5: [0.55, 0.65, 0.75, 0.85, 0.95],
    6: [0.55, 0.65, 0.75, 0.85, 0.95],
    7: [0.55, 0.65, 0.75, 0.85, 0.95],
    8: [0.55, 0.65, 0.75, 0.85, 0.95],
    9: [0.55, 0.65, 0.75, 0.85, 0.95],
    10: [0.55, 0.65, 0.75, 0.85, 0.95],   # Higher thresholds for full queues
}


total_samples = df.shape[0]  # Total number of samples to process
# Run the simulation
queue_system = QueueSystem(param.arrival_rate, threshold_choices, param.max_queue_size)
queue_system.run(total_samples)

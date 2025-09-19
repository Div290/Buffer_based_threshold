from copy import copy
from typing import Union, Dict, List
from multiprocessing import Array, Lock
import numpy as np
from math import ceil, floor

from .utils import AdaEEUpdate, BasePolicy, AdaEEBasic
from ..scheduler import ScheduleAction, time_wrapper


def generate_theta_pairs(step=0.01):
    """
    Enumerate all grid pairs (theta1, theta2) with spacing `step` (default 0.01)
    such that for all q in {0,...,10}, a = theta1*q + theta2 stays in [0.5, 1],
    and theta1 > 0 (strictly positive).
    """
    scale = int(round(1 / step))  # step=0.01 -> 100
    pairs = []

    # θ2 ∈ [0.5, 1]
    k2_min = int(round(0.5 * scale))
    k2_max = int(round(1.0 * scale))

    for k2 in range(k2_min, k2_max + 1):
        theta2 = k2 / scale

        # From endpoint constraints at q=0 and q=10:
        # θ1 ∈ [(0.5 - θ2)/10, (1 - θ2)/10]
        lower = (0.5 - theta2) / 10.0
        upper = (1.0 - theta2) / 10.0

        # Enforce strict positivity: θ1 >= step on the grid (i.e., k1 >= 1)
        k1_min = max(1, ceil(lower * scale - 1e-12))
        k1_max = floor(upper * scale + 1e-12)

        if k1_min > k1_max:
            continue  # no valid θ1 for this θ2

        for k1 in range(k1_min, k1_max + 1):
            theta1 = k1 / scale
            pairs.append((theta1, theta2))

    return pairs


class UCBParam(BasePolicy):
    def __init__(self, theta_candidates, lock: Lock = None):
        self.theta_candidates = theta_candidates
        self.n_arms = len(theta_candidates)
        self.selected = Array('i', self.n_arms)
        self.counts = Array('i', self.n_arms)
        self.values = Array('d', self.n_arms)
        self.lock = lock

    def _counts_view(self):
        return np.frombuffer(self.counts.get_obj(), dtype=np.int32)

    def _values_view(self):
        return np.frombuffer(self.values.get_obj())

    def _selected_view(self):
        return np.frombuffer(self.selected.get_obj(), dtype=np.int32)

    def get_values(self):
        return self._values_view().copy()

    def get_counts(self):
        return self._counts_view().copy()

    def get_selected(self):
        return self._selected_view().copy()

    def select_arm(self, t):
        selected = self._selected_view()
        zeros = np.where(selected == 0)[0]
        if len(zeros) > 0:
            arms = zeros[0]
        else:
            ucb_scores = self.get_values() + np.sqrt(2 * np.log(t + 1e-8) / (self.get_counts() + 1))
            arms = np.argmax(ucb_scores)

        with self.lock:
            self.selected[arms] = 1

        return arms

    def update(self, index, reward):
        with self.lock:
            self.counts[index] += 1
            n = self.counts[index]
            self.values[index] += (reward - self.values[index]) / n


class AdaEEParam(AdaEEBasic):
    def __init__(
            self,
            model_profile: Union[str, Dict],
            maxsize: int = None,
            arrival_estimate_window: int = None,
            log_path: str = None,
            active_gates: List[int] = None,
            theta_candidates: List[List[float]] = None,
            mu: float = 0.01,
            kappa: float = 0.05,
            **kwargs
    ):
        max_memory_size = None  # only support queue length
        super(AdaEEParam, self).__init__(
            model_profile, maxsize, max_memory_size, arrival_estimate_window, log_path, **kwargs
        )
        self.skip_exit_threshold = kwargs.get("skip_exit_threshold", 1.1)
        self.active_gates = active_gates if active_gates else [i for i in range(len(self._gates) - 1)]
        self.preset_threshold = [self.skip_exit_threshold + 0.1] * len(self._gates)
        self.preset_threshold[-1] = 0
        self.mu = mu
        self.kappa = kappa

        self._name = 'AdaEEPolicyGradient'

        if theta_candidates is None:
            theta_candidates = generate_theta_pairs(step=0.01)

        lock = Lock()
        self.policies = {
            0: UCBParam(theta_candidates, lock)
        }
        self.t = 0

    def scheduler_type(self):
        return "adaee_param"

    def get_info(self):
        return "AdaEEUCBParam scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        ucb = self.policies[0]
        self.t += 1
        arm = ucb.select_arm(self.t)
        theta = ucb.theta_candidates[arm]
        alpha_t = max(min(theta[0] * self.q_size + theta[1], self.skip_exit_threshold - 0.01), 0)
        action = copy(self.preset_threshold)
        for g in self.active_gates:
            action[g] = alpha_t
        transmit, partition_layers, select_destination = self._transmit_action()
        schedule_action = ScheduleAction(multi_gates=action,
                                         skip_exit_threshold=self.skip_exit_threshold,
                                         transmit=transmit,
                                         partition_layers=partition_layers,
                                         transmit_destination=select_destination,
                                         queue_idx=self.q_size,
                                         arm_idx=int(arm))
        self.logger.log("INFO", f"Exit Threshold: {action}")
        return schedule_action

    def update(self, adaee_update: AdaEEUpdate):
        queue_idx = adaee_update.queue_idx
        arm_idx = adaee_update.arm_idx
        if adaee_update.early_exit:
            reward = 0
        else:
            early_exit_confidence = adaee_update.early_exit_confidence
            final_confidence = adaee_update.final_confidence
            reward = np.max((final_confidence - early_exit_confidence), 0) - (self.mu * queue_idx - self.kappa)

        previous_value = self.policies[0].values[arm_idx]
        self.policies[0].update(arm_idx, reward)
        new_value = self.policies[0].values[arm_idx]
        self.log_reward_update(queue_idx, previous_value, new_value)

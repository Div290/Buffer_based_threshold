from typing import Union, Dict, List
from multiprocessing import Array, Value, Lock

import numpy as np

from .utils import AdaEEUpdate, AdaEEBasic, BasePolicy
from ..scheduler import ScheduleAction, time_wrapper


class UCB(BasePolicy):
    """Upper Confidence Bound (UCB) for dynamic decision making with multiprocessing support."""

    def __init__(self, thresholds, q_idx, lock: Lock = None):
        self.q_idx = q_idx

        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        self.lock = lock or Lock()

        self.counts = Array('i', [0] * self.n_arms)
        self.values = Array('d', [0.0] * self.n_arms)
        self.selected = Array('i', [0] * self.n_arms)
        self.select_ucb_count = Value('i', 0)

    def _counts_view(self):
        return np.frombuffer(self.counts.get_obj(), dtype=np.int32)

    def _values_view(self):
        return np.frombuffer(self.values.get_obj())

    def _selected_view(self):
        return np.frombuffer(self.selected.get_obj(), dtype=np.int32)

    def select_arm(self):
        with self.lock:
            counts = self._counts_view()
            values = self._values_view()
            selected = self._selected_view()

            total_counts = np.sum(counts)
            zeros = np.where(selected == 0)[0]
            if len(zeros) > 0:
                arm = zeros[0]
            else:
                ucb_values = values + np.sqrt(2 * np.log(total_counts + 1e-6) / (counts + 1e-5))
                arm = int(np.argmax(ucb_values))
                self.select_ucb_count.value += 1

            selected[arm] = 1
        return arm

    def update(self, arm, reward):
        with self.lock:
            self.counts[arm] += 1
            self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def get_values(self):
        return self._values_view().copy()

    def get_counts(self):
        return self._counts_view().copy()

    def get_selected(self):
        return self._selected_view().copy()


class AdaEEBuffer(AdaEEBasic):
    def __init__(
            self,
            model_profile: Union[str, Dict],
            maxsize: int = None,
            arrival_estimate_window: int = None,
            log_path: str = None,
            n_action_options: int = 3,
            action_set: List = None,
            active_gates: List[int] = None,
            mu: float = 0.1,
            kappa: float = 0.05,
            **kwargs
    ):
        super(AdaEEBuffer, self).__init__(model_profile, maxsize, arrival_estimate_window, log_path,
                                          n_action_options, action_set, active_gates, **kwargs)
        self._name = 'AdaEEBuffer'
        self.mu = mu
        self.kappa = kappa

        locks = [Lock() for _ in range(self.maxsize)]
        self.policies = {q: UCB(self.action_set[q], q_idx=q, lock=locks[q]) for q in range(self.maxsize)}

    @property
    def scheduler_type(self):
        return "adaee_buffer"

    def get_info(self):
        return "AdaEEBuffer scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        ubc = self.policies[self.q_size]
        arm = ubc.select_arm()
        action = ubc.thresholds[arm]
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

    def update(self, adaee_update: AdaEEUpdate, **kwargs):
        # update the rewards
        queue_idx = adaee_update.queue_idx
        arm_idx = adaee_update.arm_idx
        if adaee_update.early_exit:
            reward = 0

        else:
            early_exit_confidence = adaee_update.early_exit_confidence
            final_confidence = adaee_update.final_confidence

            reward = np.max((final_confidence - early_exit_confidence), 0) - (self.mu * queue_idx - self.kappa)

        previous_value = self.policies[queue_idx].values[arm_idx]
        self.policies[queue_idx].update(arm_idx, reward)
        new_value = self.policies[queue_idx].values[arm_idx]
        self.log_reward_update(queue_idx, previous_value, new_value)

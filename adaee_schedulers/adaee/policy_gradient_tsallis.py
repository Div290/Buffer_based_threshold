from typing import Union, Dict, List
from multiprocessing import Array, Value, Lock
import numpy as np

from .utils import AdaEEUpdate, AdaEEBasic, BasePolicy
from ..scheduler import ScheduleAction, time_wrapper


class TsallisPolicyGradient(BasePolicy):
    def __init__(self, thresholds, tsallis_lambda=1.5, lr=0.1, baseline_beta=0.9, prefs: Array = None,
                 baseline: Value = None, lock: Lock = None):
        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        if prefs is None:
            self.preferences = Array('d', self.n_arms)
        else:
            self.preferences = prefs

        if baseline is None:
            self.baseline = Value('d', 0)
        else:
            self.baseline = baseline
        self.lock = lock
        self.tsallis_lambda = tsallis_lambda
        self.lr = lr
        self.baseline_beta = baseline_beta

        self.selected = Array('i', self.n_arms)
        self.counts = Array('i', self.n_arms)  # Number of times each arm is selected

    def _counts_view(self):
        return np.frombuffer(self.counts.get_obj(), dtype=np.int32)

    def _preference_view(self):
        return np.frombuffer(self.preferences.get_obj())

    def _selected_view(self):
        return np.frombuffer(self.selected.get_obj(), dtype=np.int32)

    def exp_tsallis(self, u):
        lam = self.tsallis_lambda
        if lam == 1:
            return np.exp(u)
        return np.maximum(1 + (1 - lam) * u, 1e-8) ** (1 / (1 - lam))

    def softmax(self):
        prefs = np.frombuffer(self.preferences.get_obj())
        exp_prefs = self.exp_tsallis(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs

    def select_arm(self):
        probs = self.softmax()
        choose_arm = np.random.choice(self.n_arms, p=probs)
        with self.lock:
            self.selected[choose_arm] = 1
            self.counts[choose_arm] += 1
        return choose_arm

    def update(self, arm, reward):
        with self.lock:
            prefs = self._preference_view()
            probs = self.softmax()

            # Update baseline
            current_baseline = self.baseline.value
            self.baseline.value = self.baseline_beta * current_baseline + (1 - self.baseline_beta) * reward
            adjusted_reward = reward - self.baseline.value

            lam = self.tsallis_lambda
            for a in range(self.n_arms):
                indicator = 1 if a == arm else 0
                theta = prefs[a]
                denom = max(1 + (1 - lam) * theta, 1e-8)
                grad = (indicator - probs[a]) * (1 / denom)
                prefs[a] += self.lr * adjusted_reward * grad

    def get_values(self):
        return self._preference_view().copy()

    def get_counts(self):
        return self._counts_view().copy()

    def get_selected(self):
        return self._selected_view().copy()


class AdaEETsallisPolicyGradient(AdaEEBasic):
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
        super(AdaEETsallisPolicyGradient, self).__init__(model_profile, maxsize, arrival_estimate_window,
                                                         log_path, n_action_options, action_set, active_gates, **kwargs)

        self._name = 'AdaEETSAPolicyGradient'
        self.mu = mu
        self.kappa = kappa

        tsallis_lambda = kwargs.get('tsallis_lambda', 1.5)
        lr = kwargs.get('lr', 0.1)
        baseline_beta = kwargs.get('baseline_beta', 0.9)
        baseline = kwargs.get('baseline', 0.)

        shared_prefs = [Array('d', self.n_action_options) for _ in range(self.maxsize)]
        shared_baseline = [Value('d', baseline) for _ in range(self.maxsize)]
        locks = [Lock() for _ in range(self.maxsize)]
        self.policies = {
            q: TsallisPolicyGradient(self.action_set[q], tsallis_lambda=tsallis_lambda, lr=lr,
                                     baseline_beta=baseline_beta, prefs=shared_prefs[q], baseline=shared_baseline[q],
                                     lock=locks[q]) for
            q in range(self.maxsize)
        }

    @property
    def scheduler_type(self):
        return "adaee_tsa_pg"

    def get_info(self):
        return "AdaEETsallisPolicyGradient scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        pg = self.policies[self.q_size]
        arm = pg.select_arm()
        action = pg.thresholds[arm]
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
            reward = -adaee_update.final_process_time  # processing time of early exit
        else:
            early_exit_confidence = adaee_update.early_exit_confidence
            final_confidence = adaee_update.final_confidence
            reward = np.max((final_confidence - early_exit_confidence), 0) - (self.mu * queue_idx - self.kappa)

        previous_value = self.policies[queue_idx].preferences[arm_idx]
        self.policies[queue_idx].update(arm_idx, reward)
        new_value = self.policies[queue_idx].preferences[arm_idx]
        self.log_reward_update(queue_idx, previous_value, new_value)

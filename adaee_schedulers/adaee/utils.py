from typing import Union, Dict, List, Optional

import numpy as np
from smt.sampling_methods import LHS
from tqdm import tqdm
from dataclasses import dataclass

from ..scheduler import Scheduler


class BasePolicy:
    def select_arm(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        raise NotImplementedError


@dataclass
class AdaEEUpdate:
    queue_idx: int = None
    arm_idx: int = None
    early_exit: bool = False
    early_exit_layer: Optional[int] = None
    final_confidence: Optional[float] = None
    early_exit_confidence: float = None
    early_process_time: float = None
    final_process_time: Optional[float] = None


class AdaEEBasic(Scheduler):
    def __init__(
            self,
            model_profile: Union[str, Dict],
            maxsize: int = None,
            arrival_estimate_window: int = None,
            log_path: str = None,
            n_action_options: int = 3,
            action_set: List = None,
            active_gates: List[int] = None,
            **kwargs
    ):
        max_memory_size = None  # only support queue length
        super(AdaEEBasic, self).__init__(model_profile, maxsize, max_memory_size,
                                         arrival_estimate_window, log_path, **kwargs)

        assert n_action_options >= 2
        self.n_action_options = len(action_set) if action_set else n_action_options
        self.active_gates = active_gates if active_gates else [i for i in range(len(self._gates) - 1)]

        # initialize actions and rewards
        if action_set is None:
            actions = self._generate_actions()
            self.action_set = {i: a for i, a in enumerate(actions)}
        elif isinstance(action_set, list):
            action_set = self._generate_valid_action(action_set)
            self.action_set = {q: {i: a for i, a in enumerate(action_set)} for q in range(self.maxsize)}
        elif isinstance(action_set, dict):
            action_set = {q: self._generate_valid_action(action_set[q]) for q in range(self.maxsize)}
            self.action_set = action_set
        else:
            self.action_set = action_set

        self.skip_exit_threshold = kwargs.get("skip_exit_threshold", 1)

        self._name = 'AdaEEBasic'

    def _generate_valid_action(self, action_set):
        action_set = np.array(action_set) if not isinstance(action_set, np.ndarray) else action_set
        if action_set.ndim == 1:
            action_set = action_set.reshape(-1, 1)
        actions = np.ones((self.n_action_options, len(self._gates)))
        actions[:, self.active_gates] = action_set
        actions[:, -1] = 0
        return actions

    def _generate_actions(self) -> List[List[float]]:
        # each gate except the final gate has n options
        low_bound = 0.01
        upper_bound = 1.01
        gate_with_options = len(self.active_gates)

        # Use LHS to generate actions
        action_sampler = LHS(xlimits=np.array([[low_bound, upper_bound]] * gate_with_options), criterion='ese',
                             random_state=42)
        actions = list()
        count = 0
        while len(actions) < self.n_action_options and count < 1000:
            actions_sample = action_sampler(self.n_action_options)
            if len(self.active_gates) < len(self._gates) - 1:
                actions_sample = self._complete_actions(actions_sample)
            actions_sample = [np.concatenate([action, np.array([0])]) for action in
                              tqdm(actions_sample, desc="validate actions") if
                              self._is_valid(action)]
            actions.extend(actions_sample)
            count += 1
        return actions

    def _complete_actions(self, actions):
        """
        complete the generated actions if activated gates are provided
        :param actions:
        :return:
        """
        num_actions, action_dim = actions.shape
        action_matrix_size = len(self._gates) - 1
        completed_actions = np.ones((num_actions, action_matrix_size))
        for i, col_index in enumerate(tqdm(self.active_gates, desc="completing actions")):
            completed_actions[:, col_index] = actions[:, i]
        return completed_actions

    @staticmethod
    def _is_valid(combination):
        for i in range(len(combination) - 1):
            # if a gate is open then only keep the action which the remaining gates are all opened
            if combination[i] <= 0 and combination[i] != combination[i + 1]:
                return False
        return True

    def compute_reward(self, update: AdaEEUpdate) -> float:
        return compute_reward(update, self.maxsize)

    def log_reward_update(self, idx, prev_val, new_val):
        self.logger.log('info', f"Reward Update: {idx}: {prev_val} -> {new_val}")


def compute_reward(update: AdaEEUpdate, maxsize: int) -> float:
    if update.early_exit:
        return -update.final_process_time
    confidence_diff = np.max((update.final_confidence - update.early_exit_confidence), 0)
    penalty = (1 / (10 * maxsize)) * update.queue_idx + (1 / 10000) * update.final_process_time
    return confidence_diff - penalty

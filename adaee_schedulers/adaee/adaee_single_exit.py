from typing import Union, Dict, List
from multiprocessing import Lock

import numpy as np

from .utils import AdaEEUpdate, AdaEEBasic
from .adaee_buffer import UCB
from ..scheduler import ScheduleAction, time_wrapper


class AdaEEOrigin(AdaEEBasic):
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
        super(AdaEEOrigin, self).__init__(model_profile, maxsize, arrival_estimate_window, log_path,
                                          n_action_options, action_set, active_gates, **kwargs)
        self._name = 'AdaEEOrigin'
        self.action_set = self.action_set[0]
        self.policies = {0: UCB(self.action_set, q_idx=0, lock=Lock())}

    @property
    def scheduler_type(self):
        return "adaee_origin"

    def get_info(self):
        return "AdaEEOrigin Scheduler"

    @time_wrapper
    def schedule(self, **kwargs) -> ScheduleAction:
        ubc = self.policies[0]
        arm = ubc.select_arm()
        action = ubc.thresholds[arm]
        transmit, partition_layers, select_destination = self._transmit_action()
        schedule_action = ScheduleAction(multi_gates=action,
                                         skip_exit_threshold=self.skip_exit_threshold,
                                         transmit=transmit,
                                         partition_layers=partition_layers,
                                         transmit_destination=select_destination,
                                         queue_idx=0,
                                         arm_idx=int(arm))
        self.logger.log("INFO", f"Exit Threshold: {action}")
        return schedule_action

    def update(self, adaee_update: AdaEEUpdate, **kwargs):
        # update the rewards
        arm_idx = adaee_update.arm_idx
        if adaee_update.early_exit:
            reward = - adaee_update.final_process_time
        else:
            extra_processing_time = adaee_update.final_process_time - adaee_update.early_process_time  # processing time of final exit

            early_exit_confidence = adaee_update.early_exit_confidence
            final_confidence = adaee_update.final_confidence

            reward = np.max((final_confidence - early_exit_confidence), 0) - (
                    1 / 10000) * extra_processing_time

        previous_value = self.policies[0].values[arm_idx]
        self.policies[0].update(arm_idx, reward)
        new_value = self.policies[0].values[arm_idx]
        self.log_reward_update(0, previous_value, new_value)

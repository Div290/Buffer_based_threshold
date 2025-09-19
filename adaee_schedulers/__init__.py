from .scheduler import Scheduler, ScheduleAction, time_wrapper
from .adaee import AdaEEBuffer, AdaEEParam, AdaEEOrigin, AdaEEPolicyGradient, AdaEETsallisPolicyGradient

__all__ = [
    'ScheduleAction', 'time_wrapper', 'Scheduler',
    'AdaEEBuffer', 'AdaEEParam', 'AdaEEOrigin', 'AdaEEPolicyGradient', 'AdaEETsallisPolicyGradient',
    'init_scheduler'
]


def init_scheduler(model_profile,  scheduler_config, maxsize=None, log_path=None):
    scheduler_type = scheduler_config.pop('type')
    arrival_estimate_window = scheduler_config.pop('arrival_estimate_window', None)
    if scheduler_type == 'adaee_origin':
        return AdaEEOrigin(model_profile, maxsize=maxsize, log_path=log_path,
                           arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_buffer':
        return AdaEEBuffer(model_profile, maxsize=maxsize, log_path=log_path,
                           arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_param':
        return AdaEEParam(model_profile, maxsize=maxsize, log_path=log_path,
                          arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_pg':
        return AdaEEPolicyGradient(model_profile, maxsize=maxsize, log_path=log_path,
                                   arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    elif scheduler_type == 'adaee_tsa_pg':
        return AdaEETsallisPolicyGradient(model_profile, maxsize=maxsize, log_path=log_path,
                                          arrival_estimate_window=arrival_estimate_window, **scheduler_config)
    else:
        raise ValueError(f"invalid scheduler type {scheduler_type}")

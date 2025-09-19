# Instructions for Using AdaEE Schedulers in a real-world testbed

## Initialization
```python
from adaee_schedulers import init_scheduler

model_config = "adaee_schedulers/configs/example_model_config.json"
scheduler_config = "adaee_schedulers/configs/example_tsa_pg_config.json"

scheduler = init_scheduler(model_config, scheduler_config, maxsize=10)
```

## Usage
The finite capacity buffer is integraded into the scheduler. You can add jobs as follows:
```python
job = {
    "job_id": 1,
    # other job parameters
}

scheduler.pre_add(job)
scheduler.q_put(job)
```
Get the job from the scheduler
```python
job = scheduler.q_get()
```
Get the scheduling decision
```python
decision = scheduler.schedule()
```
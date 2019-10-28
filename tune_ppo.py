import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    #stop={"episode_reward_mean": 200},
    checkpoint_freq=20,
    local_dir="~/ray_results",    
    config={
        "env": "DualUR5HuskyPickAndPlace-v1",
        "num_gpus": 0,
        "num_workers": 25,
        "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
        "lambda": 0.95,
        "gamma": 0.998,
        "kl_coeff": 1.0,
        "clip_param": 0.2,
        "eager": False,
    },
)

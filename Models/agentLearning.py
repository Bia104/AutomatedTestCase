from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from Models.enviromentSimulation import TaxiGymEnv
import os

env = TaxiGymEnv()

log_path = "../ppo_elements/logs/"
model_path = "../ppo_elements/models/"

model = PPO("MultiInputPolicy", env, verbose=1)
#model = PPO.load(os.path.join(model_path, "best_model"), env=env)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)

eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    best_model_save_path=model_path,
    log_path=log_path,
    eval_freq=300_000,
    n_eval_episodes=1000,
    deterministic=True,
    verbose=1
)

model.learn(total_timesteps=1_500_000, callback=eval_callback)

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


class AgentLearning:
    def __init__(self, env):
        self.env = env
        self.model = None

    def train_agent(self, path:str = "./ppo_elements/"):
        print("Training new agent...")
        self.model = PPO("MultiInputPolicy", self.env, verbose=1)

        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)

        eval_callback = EvalCallback(
            self.env,
            callback_on_new_best=stop_callback,
            best_model_save_path=path,
            log_path=path,
            eval_freq=300_000,
            n_eval_episodes=1000,
            deterministic=True,
            verbose=1
        )

        self.model.learn(total_timesteps=1_500_000, callback=eval_callback)

        return self.model

    def load_or_train_model(self, path:str = "./ppo_elements/"):
        model_path = os.path.join(path, "best_model.zip")
        if os.path.exists(os.path.abspath(model_path)):
            PPO.load(model_path, env = self.env)
            print("Loaded existing agent model...")
        else:
            self.train_agent()

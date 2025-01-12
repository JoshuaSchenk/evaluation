import gymnasium as gym
import torch

from tianshou.policy import TD3Policy, DDPGPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import offpolicy_trainer
from tianshou.env import SubprocVectorEnv

from safe_rl_lib.action_masking.ray import RayMaskingEnvWrapper
from safe_rl_lib.action_masking.generator import GeneratorMaskingEnvWrapper
from safe_rl_lib.definitions import ActionConstraintsComputationFn, FailSafeActionFn

def create_env(env_name, action_constraints_fn=None, fail_safe_action_fn=None, masking_type=None):
    env = gym.make(env_name)
    if masking_type == 'ray':
        env = RayMaskingEnvWrapper(env, action_constraints_fn, fail_safe_action_fn)
    elif masking_type == 'generator':
        env = GeneratorMaskingEnvWrapper(env, action_constraints_fn, fail_safe_action_fn)
    return env

def run_experiment(env_name, policy_type, masking_type, num_runs=10, num_episodes=100):
    results = []
    for run in range(num_runs):
        env = create_env(env_name, action_constraints_fn, fail_safe_action_fn, masking_type)
        train_envs = SubprocVectorEnv([lambda: create_env(env_name, action_constraints_fn, fail_safe_action_fn, masking_type) for _ in range(8)])
        test_envs = SubprocVectorEnv([lambda: create_env(env_name, action_constraints_fn, fail_safe_action_fn, masking_type) for _ in range(8)])

        # Define the policy
        if policy_type == 'TD3':
            policy = TD3Policy(...)
        elif policy_type == 'DDPG':
            policy = DDPGPolicy(...)

        # Collector
        train_collector = Collector(policy, train_envs, ReplayBuffer(size=20000))
        test_collector = Collector(policy, test_envs)

        # Trainer
        result = offpolicy_trainer(
            policy, train_collector, test_collector,
            max_epoch=num_episodes, step_per_epoch=10000, step_per_collect=512,
            episode_per_test=100, batch_size=64, update_per_step=0.1,
        )
        results.append(result)

    return results

if __name__ == '__main__':
    env_name = 'Walker2d-v2'
    action_constraints_fn = ...  # Define your action constraints function
    fail_safe_action_fn = ...  # Define your fail-safe action function

    # Run experiments for TD3 with Ray masking
    td3_ray_results = run_experiment(env_name, 'TD3', 'ray')

    # Run experiments for TD3 with Generator masking
    td3_generator_results = run_experiment(env_name, 'TD3', 'generator')

    # Run experiments for DDPG with Ray masking
    ddpg_ray_results = run_experiment(env_name, 'DDPG', 'ray')

    # Run experiments for DDPG with Generator masking
    ddpg_generator_results = run_experiment(env_name, 'DDPG', 'generator')

    # Save results
    torch.save('td3_ray_results.npy', td3_ray_results)
    torch.save('td3_generator_results.npy', td3_generator_results)
    torch.save('ddpg_ray_results.npy', ddpg_ray_results)
    torch.save('ddpg_generator_results.npy', ddpg_generator_results)
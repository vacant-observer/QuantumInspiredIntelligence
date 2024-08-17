import numpy as np
from scipy.stats import norm
import gymnasium as gym
from typing import List, Callable
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

class QuantumMutationWave:
    def __init__(self, shape: tuple, num_samples: int = 10):
        self.shape = shape
        self.num_samples = num_samples
        self.wave = self.initialize_wave()

    def initialize_wave(self):
        return [norm(loc=0, scale=0.1) for _ in range(np.prod(self.shape))]

    def sample(self):
        samples = np.array([wave.rvs(self.num_samples) for wave in self.wave]).T
        return samples.reshape((self.num_samples,) + self.shape)

    def collapse(self, optimal_point):
        optimal_point = optimal_point.flatten()
        for i, wave in enumerate(self.wave):
            new_loc = (wave.mean() + optimal_point[i]) / 2
            new_scale = max(wave.std() * 0.95, 0.01)  # Prevent scale from becoming too small
            self.wave[i] = norm(loc=new_loc, scale=new_scale)

class QuantumInspiredElement:
    def __init__(self, n_dimensions: int, n_actions: int):
        self.weights = np.random.rand(n_dimensions, n_actions) * 2 - 1  # Initialize between -1 and 1
        self.mutation_wave = QuantumMutationWave((n_dimensions, n_actions))

    def get_action(self, observation):
        action_values = np.dot(observation, self.weights)
        return np.argmax(action_values)

    def mutate(self, fitness_function: Callable):
        mutations = self.mutation_wave.sample()
        fitnesses = [fitness_function(self.weights + mutation) for mutation in mutations]
        best_index = np.argmax(fitnesses)
        best_mutation = mutations[best_index]
        self.weights += best_mutation
        self.mutation_wave.collapse(best_mutation)

class QuantumInspiredSystem:
    def __init__(self, num_elements: int, n_dimensions: int, n_actions: int, env_name: str):
        self.elements = [QuantumInspiredElement(n_dimensions, n_actions) for _ in range(num_elements)]
        self.n_dimensions = n_dimensions
        self.n_actions = n_actions
        self.env_name = env_name
        self.best_fitness = float('-inf')
        self.best_element = None

    def get_action(self, observation):
        if self.best_element:
            return self.best_element.get_action(observation)
        return random.choice(range(self.n_actions))

    def evaluate_fitness(self, weights):
        total_reward = 0
        env = gym.make(self.env_name)
        observation, _ = env.reset()
        done = False
        while not done:
            action_values = np.dot(observation, weights)
            action = np.argmax(action_values)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        return total_reward

    def evolve(self, num_iterations: int):
        fitness_history = []
        for _ in tqdm(range(num_iterations)):
            for element in self.elements:
                element.mutate(self.evaluate_fitness)
            
            # Update best element
            current_best = max(self.elements, key=lambda e: self.evaluate_fitness(e.weights))
            current_best_fitness = self.evaluate_fitness(current_best.weights)
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_element = current_best
            
            fitness_history.append(self.best_fitness)
        
        return fitness_history

def visualize_training(fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title('Best Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Total Reward)')
    plt.show()

def run_best_policy(system, episodes=5):
    env = gym.make(system.env_name, render_mode='human')
    for episode in range(episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = system.get_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1} total reward: {total_reward}")
    env.close()

def train_and_evaluate_quantum(env_name, num_elements=50, num_iterations=100, eval_episodes=10):
    env = gym.make(env_name)
    n_dimensions = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    system = QuantumInspiredSystem(num_elements=num_elements, n_dimensions=n_dimensions, n_actions=n_actions, env_name=env_name)
    
    start_time = time.time()
    fitness_history = system.evolve(num_iterations=num_iterations)
    training_time = time.time() - start_time
    
    # Evaluate the best policy
    total_rewards = []
    for _ in range(eval_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = system.get_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return mean_reward, std_reward, training_time, fitness_history

def train_and_evaluate_sb3(env_name, algorithm, total_timesteps, eval_episodes=10):
    env = gym.make(env_name)
    
    if algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=0)
    elif algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=0)
    else:
        raise ValueError("Unsupported algorithm")
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    
    return mean_reward, std_reward, training_time

def run_comparison(env_names, num_elements=50, num_iterations=100, total_timesteps=10000, eval_episodes=10):
    results = {}
    
    for env_name in env_names:
        print(f"\nEvaluating environment: {env_name}")
        results[env_name] = {}
        
        # Quantum-Inspired System
        print("Training Quantum-Inspired System...")
        q_mean, q_std, q_time, q_history = train_and_evaluate_quantum(env_name, num_elements, num_iterations, eval_episodes)
        results[env_name]['Quantum'] = {'mean': q_mean, 'std': q_std, 'time': q_time, 'history': q_history}
        
        # DQN
        print("Training DQN...")
        dqn_mean, dqn_std, dqn_time = train_and_evaluate_sb3(env_name, 'DQN', total_timesteps, eval_episodes)
        results[env_name]['DQN'] = {'mean': dqn_mean, 'std': dqn_std, 'time': dqn_time}
        
        # PPO
        print("Training PPO...")
        ppo_mean, ppo_std, ppo_time = train_and_evaluate_sb3(env_name, 'PPO', total_timesteps, eval_episodes)
        results[env_name]['PPO'] = {'mean': ppo_mean, 'std': ppo_std, 'time': ppo_time}
    
    return results

def visualize_results(results):
    for env_name, env_results in results.items():
        plt.figure(figsize=(12, 6))
        
        algorithms = list(env_results.keys())
        means = [env_results[alg]['mean'] for alg in algorithms]
        stds = [env_results[alg]['std'] for alg in algorithms]
        times = [env_results[alg]['time'] for alg in algorithms]
        
        # Plot mean rewards
        plt.subplot(1, 2, 1)
        plt.bar(algorithms, means, yerr=stds, capsize=5)
        plt.title(f'{env_name}: Mean Reward')
        plt.ylabel('Mean Reward')
        
        # Plot training times
        plt.subplot(1, 2, 2)
        plt.bar(algorithms, times)
        plt.title(f'{env_name}: Training Time')
        plt.ylabel('Training Time (s)')
        
        plt.tight_layout()
        plt.show()
        
# Main execution
if __name__ == "__main__":
    env_names = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    results = run_comparison(env_names, num_elements=50, num_iterations=1, total_timesteps=25000, eval_episodes=10)
    visualize_results(results)
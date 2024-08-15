import numpy as np
from scipy.stats import norm
import gymnasium as gym
from typing import List, Callable
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Main execution
if __name__ == "__main__":
    env_name = 'LunarLander-v2'  # You can change this to any environment
    env = gym.make(env_name)
    n_dimensions = env.observation_space.shape[0]
    n_actions = env.action_space.n
    system = QuantumInspiredSystem(num_elements=50, n_dimensions=n_dimensions, n_actions=n_actions, env_name=env_name)
    
    fitness_history = system.evolve(num_iterations=100)  # Increased iterations for more complex environments
    
    visualize_training(fitness_history)
    
    print(f"Best fitness achieved: {system.best_fitness}")
    
    # Run and visualize best policy
    run_best_policy(system)
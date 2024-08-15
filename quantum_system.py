import numpy as np
from scipy.stats import norm
import gymnasium as gym
from typing import List, Callable
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class QuantumMutationWave:
    def __init__(self, num_dimensions: int, num_samples: int = 1): # increase num_samples for better results
        self.num_dimensions = num_dimensions
        self.num_samples = num_samples
        self.wave = self.initialize_wave()

    def initialize_wave(self):
        return [norm(loc=0, scale=0.1) for _ in range(self.num_dimensions)]

    def sample(self):
        return np.array([wave.rvs(self.num_samples) for wave in self.wave]).T

    def collapse(self, optimal_point):
        for i, wave in enumerate(self.wave):
            new_loc = (wave.mean() + optimal_point[i]) / 2
            new_scale = max(wave.std() * 0.95, 0.01)  # Prevent scale from becoming too small
            self.wave[i] = norm(loc=new_loc, scale=new_scale)

class QuantumInspiredElement:
    def __init__(self, n_dimensions: int):
        self.weights = np.random.rand(n_dimensions) * 2 - 1  # Initialize between -1 and 1
        self.mutation_wave = QuantumMutationWave(n_dimensions)

    def get_action(self, observation):
        return 1 if np.dot(observation, self.weights) > 0 else 0

    def mutate(self, fitness_function: Callable):
        mutations = self.mutation_wave.sample()
        fitnesses = [fitness_function(self.weights + mutation) for mutation in mutations]
        best_index = np.argmax(fitnesses)
        best_mutation = mutations[best_index]
        self.weights += best_mutation
        self.mutation_wave.collapse(best_mutation)

class QuantumInspiredSystem:
    def __init__(self, num_elements: int, n_dimensions: int):
        self.elements = [QuantumInspiredElement(n_dimensions) for _ in range(num_elements)]
        self.n_dimensions = n_dimensions
        self.best_fitness = float('-inf')
        self.best_element = None

    def get_action(self, observation):
        if self.best_element:
            return self.best_element.get_action(observation)
        return random.choice([0, 1])

    def evaluate_fitness(self, weights):
        total_reward = 0
        env = gym.make('CartPole-v1')
        observation, _ = env.reset()
        done = False
        while not done:
            action = 1 if np.dot(observation, weights) > 0 else 0
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

def run_best_policy(system, env_name='CartPole-v1', episodes=5):
    env = gym.make(env_name, render_mode='human')
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
    env = gym.make('CartPole-v1')
    n_dimensions = env.observation_space.shape[0]  # 4 for CartPole-v1
    system = QuantumInspiredSystem(num_elements=50, n_dimensions=n_dimensions)
    
    fitness_history = system.evolve(num_iterations=1) # increase num_iterations for better results
    
    visualize_training(fitness_history)
    
    print(f"Best fitness achieved: {system.best_fitness}")
    
    # Run and visualize best policy
    run_best_policy(system)
  

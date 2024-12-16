
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.distributions import Normal
from scipy.stats import norm
import sys
import tensorflow as tf

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')


# Disable progress bars globally
tf.keras.utils.disable_interactive_logging()


# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CUDA Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        ).to(device) 
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)

class TemporalNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], output_size)
        ).to(device)
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.hidden_sizes = [400, 400]
        
        self.mean_net = TemporalNetwork(
            input_size=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=state_dim
        ).to(device)
        
        self.logvar_net = TemporalNetwork(
            input_size=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=state_dim
        ).to(device)
        
        self.min_var = 1e-6
        self.max_var = 1e6
    
    def forward(self, states, actions):
      inputs = torch.cat([states.to(device), actions.to(device)], dim=-1)
      mean = self.mean_net(inputs)
      logvar = self.logvar_net(inputs)
      var = torch.clamp(torch.exp(logvar), self.min_var, self.max_var)
      return mean, var

class CardiacDeepPILCO:
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(device)
        
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(),
            lr=5e-4,
            eps=1e-8
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-4,
            eps=1e-8
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=1e-4,
            eps=1e-8
        )
        
        self.buffer_size = 5000
        self.batch_size = 32
        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.reward_buffer = []
        
        self.n_particles = 100
        self.horizon = 20
        self.gamma = 0.99
        
        self.base_exploration = 1.0
        self.min_exploration = 0.1
        self.exploration_decay = 0.995
        
        self.rewards_history = []
        self.episode_count = 0
        self.setpoint = None
    
    def set_target_state(self, target):
        self.setpoint = torch.tensor(target, dtype=torch.float32).to(device)
        print(f"Target state set to: {self.setpoint}")
    
    def compute_reward(self, states, actions):
        if self.setpoint is None:
          raise ValueError("Target state not set")

        # Ensure states is a torch tensor
        if not isinstance(states, torch.Tensor):
          states = torch.tensor(states, dtype=torch.float32)

        # Ensure states is 2D
        if states.dim() == 1:
          states = states.unsqueeze(0)

        # Convert setpoint to torch tensor if not already
        if not isinstance(self.setpoint, torch.Tensor):
          setpoint = torch.tensor(self.setpoint, dtype=torch.float32)
        else:
          setpoint = self.setpoint

        # Pad or truncate setpoint to match states dimensions
        if setpoint.numel() < states.shape[1]:
          padded_setpoint = torch.zeros(states.shape[1], dtype=states.dtype, device=states.device)
          padded_setpoint[:setpoint.numel()] = setpoint
          setpoint = padded_setpoint
        elif setpoint.numel() > states.shape[1]:
          setpoint = setpoint[:states.shape[1]]

        # Compute rewards with gradient tracking
        distance = torch.norm(states - setpoint, dim=1)
        reward = torch.exp(-distance) - 0.1 * torch.norm(actions, dim=1)

        # Compare reward with setpoint and return the difference
        return reward - self.setpoint.norm()
		
    def get_exploration_factor(self):
        return max(self.min_exploration,
                   self.base_exploration * (self.exploration_decay ** self.episode_count))
    
    def get_action(self, state):
        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.policy(state_tensor)
            action = action.squeeze().detach().cpu().numpy()

            exploration_noise = np.random.normal(0, self.get_exploration_factor(), size=self.action_dim)
            action = action + exploration_noise
            
            return np.clip(action, -1, 1)
    
    def train_value_network(self, states, rewards, next_states):
        states = states.to(device).to(torch.float32)
        rewards = rewards.to(device).to(torch.float32)
        next_states = next_states.to(device).to(torch.float32)
        
        current_values = self.value_net(states)
        next_values = self.value_net(next_states)
        target_values = rewards + self.gamma * next_values.detach()
        
        value_loss = F.mse_loss(current_values, target_values)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def train_dynamics(self, verbose=False):
        if len(self.state_buffer) < self.batch_size:
            return None
        
        losses = []
        for iteration in range(50):
            indices = np.random.choice(len(self.state_buffer), self.batch_size)
            
            states = torch.FloatTensor([self.state_buffer[i] for i in indices]).to(device)
            actions = torch.FloatTensor([self.action_buffer[i] for i in indices]).to(device)
            next_states = torch.FloatTensor([self.next_state_buffer[i] for i in indices]).to(device)
            
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            mean, var = self.dynamics_model(states, actions)
            loss = 0.5 * (torch.pow(next_states - mean, 2) / var + torch.log(var)).mean()
            
            self.dynamics_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
            self.dynamics_optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and iteration % 10 == 0:
                print(f"Dynamics Iteration {iteration}, Loss: {loss.item():.6f}")
        
        return np.mean(losses)
    
    def train_policy(self, initial_states, n_iterations=30):
        processed_states = [state[0] if isinstance(state, tuple) else state for state in initial_states]
        states = torch.FloatTensor(np.array(processed_states)).to(device).to(torch.float32)
        exploration_factor = self.get_exploration_factor()
        
        for iteration in range(n_iterations):
            total_reward = 0
            
            batch_size = 10
            for batch in range(0, self.n_particles, batch_size):
                batch_reward = 0
                current_states = states[batch:batch+batch_size].clone().to(device)
                
                for t in range(self.horizon):
                    actions = self.policy(current_states)
                    mean, var = self.dynamics_model(current_states, actions)
                    
                    eps = torch.randn_like(mean)
                    next_states = mean + torch.sqrt(var) * eps
                    
                    reward = self.compute_reward(next_states, actions)
                    
                    if t > 0:
                        state_differences = torch.cdist(next_states, current_states)
                        exploration_bonus = exploration_factor * state_differences.mean()
                        reward += exploration_bonus
                    
                    batch_reward += reward.mean()
                    current_states = next_states
                
                total_reward += batch_reward * (batch_size / self.n_particles)
            
            avg_reward = total_reward / self.n_particles
            self.policy_optimizer.zero_grad()
            (-avg_reward).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            if iteration % 5 == 0:
                print(f"Policy Iteration {iteration}, Reward: {avg_reward.item():.6f}")
    
    def add_experience(self, state, action, next_state, reward):
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        state = torch.tensor(state).to(torch.float32).tolist()
        action = torch.tensor(action).to(torch.float32).tolist()
        next_state = torch.tensor(next_state).to(torch.float32).tolist()
        
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.next_state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.next_state_buffer.pop(0)
            self.reward_buffer.pop(0)

class ModelPredictionCollector:
    def __init__(self):
        self.mean_predictions = []
        self.variances = []
    
    def collect(self, mean, var):
        self.mean_predictions.append(mean.detach().cpu().numpy())
        self.variances.append(var.detach().cpu().numpy())
    
    def get_predictions(self):
        return {
            'mean': np.array(self.mean_predictions),
            'variance': np.array(self.variances)
        }

def plot_learning_metrics(rewards_history, dynamics_losses, value_losses, exploration_factors, episode, setpoint_norm, rat_type):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Deep PILCO Learning Metrics', fontsize=16)
    
    # Rewards plot with both raw and smoothed curves
    ax1.plot(rewards_history, alpha=0.3, label='Raw Reward')
    ax1.plot(np.array(rewards_history) - setpoint_norm, alpha=0.3, label='Reward - Setpoint Norm')
    window = min(50, len(rewards_history)//10)
    if len(rewards_history) > window:
        smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, 'r', label='Smoothed Reward')
    ax1.set_title('Learning Progress')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curves
    if dynamics_losses:
        ax2.plot(dynamics_losses)
        ax2.set_yscale('log')
        ax2.set_title('Dynamics Model Loss')
        ax2.set_xlabel('Training Iteration')
        ax2.grid(True)
    
    # Value Network Loss
    if value_losses:
        ax3.plot(value_losses)
        ax3.set_yscale('log')
        ax3.set_title('Value Network Loss')
        ax3.set_xlabel('Training Iteration')
        ax3.grid(True)
    
    # Exploration decay
    ax4.plot(exploration_factors)
    ax4.set_title('Exploration Factor')
    ax4.set_xlabel('Episode')
    ax4.grid(True)
    
    # Reward Comparison
    setpoint_array = [setpoint_norm] * len(rewards_history)
    ax5.plot(np.arange(len(rewards_history)), setpoint_array, label='Setpoint Norm')
    ax5.plot(np.arange(len(rewards_history)), rewards_history, label='Reward')
    ax5.plot(np.arange(len(rewards_history)), np.array(rewards_history) - setpoint_norm, label='Reward - Setpoint Norm')
    ax5.set_title('Reward Comparison')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Value')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'outputs/learning_metrics_ep{episode}_{rat_type}.png')
    plt.close(fig)
    
    
def plot_model_predictions(mean_predictions, var_predictions, episode, rat_type):
    mean_predictions = np.array(mean_predictions).squeeze()
    var_predictions = np.array(var_predictions).squeeze()

    if mean_predictions.ndim > 1:
        mean_predictions = mean_predictions[:, 0]
        var_predictions = var_predictions[:, 0]

    plt.figure(figsize=(10, 6))
    timesteps = range(len(mean_predictions))
    plt.plot(timesteps, mean_predictions, 'b-', label='Mean prediction')
    plt.fill_between(
        timesteps,
        mean_predictions - 2 * np.sqrt(var_predictions),
        mean_predictions + 2 * np.sqrt(var_predictions),
        alpha=0.2, label='95% confidence'
    )
    plt.title('Model Prediction Uncertainty')
    plt.xlabel('Timestep')
    plt.ylabel('State Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/model_predictions_ep_{rat_type}.png')
    plt.close()

def plot_model_weights(dynamics_model, policy, value_net, episode, rat_type):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Weights', fontsize=16)

    # Plot dynamics model weights
    weights = dynamics_model.mean_net.network[0].weight.detach().cpu().numpy()
    ax1.imshow(weights, cmap='viridis')
    ax1.set_title('Dynamics Model Mean Net Weights')
    ax1.set_xlabel('Output Dimension')
    ax1.set_ylabel('Input Dimension')

    weights = dynamics_model.logvar_net.network[0].weight.detach().cpu().numpy()
    ax2.imshow(weights, cmap='viridis')
    ax2.set_title('Dynamics Model Log-Variance Net Weights')
    ax2.set_xlabel('Output Dimension')
    ax2.set_ylabel('Input Dimension')

    # Plot policy weights
    weights = policy[0].weight.detach().cpu().numpy()
    ax3.imshow(weights, cmap='viridis')
    ax3.set_title('Policy Net Weights')
    ax3.set_xlabel('Output Dimension')
    ax3.set_ylabel('Input Dimension')

    # Plot value network weights
    weights = value_net.network[0].weight.detach().cpu().numpy()
    ax4.imshow(weights, cmap='viridis')
    ax4.set_title('Value Network Weights')
    ax4.set_xlabel('Output Dimension')
    ax4.set_ylabel('Input Dimension')

    plt.tight_layout()
    plt.savefig(f'outputs/model_weights_ep_{rat_type}.png')
    plt.close(fig)
    
    
def train(env, agent, rat_type, n_episodes=20000, checkpoint_freq=1000, plot_freq=100):
    rewards_history = []
    dynamics_losses = []
    value_losses = []
    exploration_factors = []
    prediction_collector = ModelPredictionCollector()
    
    running_reward = None
    alpha = 0.1
    best_reward = float('-inf')
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        state_tensor = torch.FloatTensor(state[0] if isinstance(state, tuple) else state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, var = agent.dynamics_model(state_tensor, torch.zeros((1, agent.action_dim)).to(device))
            mean, var = mean.to(device), var.to(device)
            prediction_collector.collect(mean, var)
        
        done = False
        step_count = 0
        max_steps = 500
        
        while not done and step_count < max_steps:
            action = agent.get_action(state)
            next_state, reward, done1, done2, info = env.step(action)
            done = done1 or done2
            
            processed_state = next_state[0] if isinstance(next_state, tuple) else next_state
            
            # Collect model predictions
            state_tensor = torch.FloatTensor(processed_state).reshape(1, -1).to(device)
            action_tensor = torch.FloatTensor(action).reshape(1, -1).to(device)
            
            with torch.no_grad():
                mean, var = agent.dynamics_model(state_tensor, action_tensor)
                prediction_collector.collect(mean, var)
            
            # Shape reward if needed
            shaped_reward = reward * (1 + step_count / max_steps)
            
            agent.add_experience(state, action, next_state, shaped_reward)
            episode_states.append(processed_state)
            episode_actions.append(action)
            episode_rewards.append(shaped_reward)
            
            episode_reward += reward
            state = next_state
            step_count += 1
        
        # Update running reward
        if running_reward is None:
            running_reward = episode_reward
        else:
            running_reward = (1 - alpha) * running_reward + alpha * episode_reward
        
        rewards_history.append(episode_reward)
        agent.episode_count += 1
        
        # Periodic logging and training
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"Steps completed: {step_count}")
            print(f"Episode reward: {episode_reward:.4f}")
            print(f"Running reward: {running_reward:.4f}")
        
        # Model training
        if episode % 5 == 0 and len(agent.state_buffer) >= agent.batch_size:
            try:
                # Train dynamics model
                dynamics_loss = agent.train_dynamics(verbose=False)
                if dynamics_loss is not None:
                    dynamics_losses.append(dynamics_loss)
                
                # Train policy
                if len(episode_states) > 0:
                    agent.train_policy(episode_states)
                
                # Train value network
                if len(episode_states) > 1:
                    states_tensor = torch.stack([torch.tensor(s).float() for s in episode_states[:-1]])
                    rewards_tensor = torch.stack([torch.tensor(r).float() for r in episode_rewards[:-1]])
                    next_states_tensor = torch.stack([torch.tensor(s).float() for s in episode_states[1:]])
                    
                    # Ensure consistent dimensions
                    min_dim = min(states_tensor.shape[1], rewards_tensor.shape[0], next_states_tensor.shape[1])
                    states_tensor = states_tensor[:, :min_dim]
                    rewards_tensor = rewards_tensor[:min_dim]
                    next_states_tensor = next_states_tensor[:, :min_dim]
                    
                    value_loss = agent.train_value_network(states_tensor, rewards_tensor, next_states_tensor)
                    value_losses.append(value_loss)
            
            except Exception as episode_error:
                print(f"Error in episode {episode}: {episode_error}")
                continue
        
        # Periodic plotting and checkpointing
        if (episode + 1) % plot_freq == 0:
            # Plot learning metrics
            plot_learning_metrics(
				rewards_history, 
				dynamics_losses, 
				value_losses, 
				exploration_factors, 
				episode,
				agent.setpoint.norm().item(),
				rat_type
			)
            
            # Plot model predictions
            predictions = prediction_collector.get_predictions()
            if len(predictions['mean']) > 0:
                plot_model_predictions(
                    predictions['mean'][-100:], 
                    predictions['variance'][-100:], 
                    episode, rat_type
                )
                
            # Plot model weights
            plot_model_weights(
                agent.dynamics_model,
                agent.policy,
                agent.value_net,
                episode, rat_type
            )
        
        # Model checkpointing
        if (episode + 1) % checkpoint_freq == 0:
            torch.save({
                'episode': episode,
                'dynamics_model': agent.dynamics_model.state_dict(),
                'policy': agent.policy.state_dict(),
                'value_net': agent.value_net.state_dict(),
                'rewards_history': rewards_history,
                'best_reward': best_reward
            }, f'outputs/checkpoint_ep{episode+1}.pt')
        
        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'episode': episode,
                'dynamics_model': agent.dynamics_model.state_dict(),
                'policy': agent.policy.state_dict(),
                'value_net': agent.value_net.state_dict(),
                'best_reward': best_reward
            }, 'outputs/best_model.pt')
    
    return (rewards_history, dynamics_losses, value_losses, 
            exploration_factors, prediction_collector.get_predictions())

def main():
    import logging
    import traceback
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('outputs/training_log.txt'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        # Print Python and library versions
        print(f"Python Version: {sys.version}")
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"PyTorch Version: {torch.__version__}")

        # Import necessary modules
        from rl_cardiac.tcn_model import TCN_config
        from rl_cardiac.cardiac_model import CardiacModel_Env

        # List of rat types to try
        rat_types = ['healthy_stable', 'healthy_exercise', 'hypertension_stable', 'hypertension_exercise']

        for rat_type in rat_types:
            print(f"Training for rat type: {rat_type}")

            # Set up environment parameters
            tcn_model = TCN_config(rat_type)
            env = CardiacModel_Env(tcn_model, rat_type)

            # Initialize agent
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            agent = CardiacDeepPILCO(state_dim, action_dim)

            # Set target state
            try:
                setpoint = env.setpoints
            except AttributeError:
                initial_state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
                setpoint = initial_state[:state_dim//2]

            print(f"Setpoint: {setpoint}")
            logging.info(f"Using environment setpoint: {setpoint}")
            agent.set_target_state(setpoint)

            # Training parameters
            n_episodes = 2000
            checkpoint_freq = 100
            plot_freq = 100

            # Start training
            logging.info(f"Starting training for rat type: {rat_type}")
            rewards_history, dynamics_losses, value_losses, exploration_factors, model_predictions = train(
                env, agent, rat_type,
                n_episodes=n_episodes,
                checkpoint_freq=checkpoint_freq,
                plot_freq=plot_freq
            )
            logging.info(f"Training completed successfully for rat type: {rat_type}")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
            # Plot MAP
            ax1.plot(np.arange(len(rewards_history)), rewards_history)
            ax1.set_title(f"MAP vs Iteration for {rat_type}")
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('MAP')

            # Plot HR
            setpoint_array = [setpoint] * len(rewards_history)
            ax2.plot(np.arange(len(rewards_history)), setpoint_array)
            ax2.set_title(f"HR vs Iteration for {rat_type}")
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('HR')

            plt.savefig(f'outputs/MAP_HR_vs_Iteration_{rat_type}.png')
            plt.close()

    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        env.close()

if __name__ == "__main__":
    main()
        

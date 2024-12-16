
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from scipy.stats import norm
import os
import datetime
import gymnasium as gym
import rl_dbs.gym_oscillator
import rl_dbs.gym_oscillator.envs
import rl_dbs.oscillator_cpp
import logging
import sys
from pathlib import Path

env = rl_dbs.gym_oscillator.envs.oscillatorEnv()

# Setup logging
def setup_logging(log_dir='logs'):
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f'{log_dir}/deep_pilco_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        logger.info(f"Initializing ValueNetwork with state_dim: {state_dim}")
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)

        # Initialize weights
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.0)
        logger.info("ValueNetwork initialized successfully")

    def forward(self, state):
        state = state.to(device)
        return self.network(state)

class TemporalNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        logger.info(f"Initializing TemporalNetwork with input_size: {input_size}, "
                   f"hidden_sizes: {hidden_sizes}, output_size: {output_size}")
        
        layers = []
        current_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
            logger.debug(f"Added layer {i+1} with size {hidden_size}")

        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers).to(device)

        # Initialize weights
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        logger.info("TemporalNetwork initialized successfully")

    def forward(self, x):
        x = x.to(device)
        return self.network(x)

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=250, action_dim=1):
        super().__init__()
        logger.info(f"Initializing DynamicsModel with state_dim: {state_dim}, action_dim: {action_dim}")
      
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.hidden_sizes = [512, 512, 256, 256, 128]

        logger.info(f"Creating mean network with input_dim: {self.input_dim}")
        self.mean_net = TemporalNetwork(
            input_size=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=state_dim
        ).to(device)

        logger.info("Creating logvar network")
        self.logvar_net = TemporalNetwork(
            input_size=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=state_dim
        ).to(device)

        self.min_var = 1e-6
        self.max_var = 1e6
        
        logger.info("DynamicsModel initialized successfully")

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=-1)
        mean = self.mean_net(inputs)
        logvar = self.logvar_net(inputs)
        var = torch.clamp(torch.exp(logvar), self.min_var, self.max_var)
        return mean, var

class DeepPILCO:
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        logger.info(f"Initializing DeepPILCO with state_dim: {state_dim}, action_dim: {action_dim}, hidden_dim: {hidden_dim}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize networks
        logger.info("Initializing networks...")
        self.dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)

        # Initialize policy network
        logger.info("Creating policy network")
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Tanh()
        ).to(device)

        # Initialize optimizers
        logger.info("Setting up optimizers...")
        self.dynamics_optimizer = torch.optim.AdamW(
            self.dynamics_model.parameters(),
            lr=5e-4,
            weight_decay=1e-4,
            eps=1e-8
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=3e-4,
            weight_decay=1e-5,
            eps=1e-8
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-8
        )

        # Initialize learning rate schedulers
        logger.info("Setting up learning rate schedulers...")
        self.dynamics_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dynamics_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Initialize experience buffer
        logger.info("Initializing experience buffer...")
        self.buffer_size = 500
        self.batch_size = 64
        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.reward_buffer = []

        # Set training parameters
        self.n_particles = 100
        self.horizon = 20
        self.gamma = 0.99

        # Set exploration parameters
        self.epsilon = 0.05
        self.base_exploration = 1.0
        self.min_exploration = 0.1
        self.exploration_decay = 0.995

        self.rewards_history = []
        self.episode_count = 0
        
        logger.info("DeepPILCO initialization completed successfully")

    def get_exploration_factor(self):
        factor = max(self.min_exploration,
                    self.base_exploration * (self.exploration_decay ** self.episode_count))
        logger.debug(f"Current exploration factor: {factor:.4f}")
        return factor

    def get_action(self, state):
        with torch.no_grad():
            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-1, 1, size=(1,))
                logger.debug("Taking random exploration action")
                return action

            if isinstance(state, tuple):
                state = state[0]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.policy(state_tensor)
            action = action.cpu().squeeze().numpy()
            logger.debug(f"Policy action: {action}")
            return np.array([action])

    def compute_reward(self, states, actions):
          logger.debug("Computing reward...")
          X_t = states[:, -1]
          X_mean = torch.mean(states, dim=1)
          
          # Match environment's reward function
          desync_measure = -torch.pow(X_t - X_mean, 2)
          desync_reward = desync_measure.mean()
          action_penalty = 2 * torch.abs(actions).mean()
          reward = desync_reward - action_penalty + 0.1
          
          logger.debug(f"Reward components: desync={desync_reward:.4f}, action_penalty={action_penalty:.4f}")
          return reward

    def train_dynamics(self, verbose=True):
        if len(self.state_buffer) < self.batch_size:
            logger.debug("Not enough samples in buffer for dynamics training")
            return None
            
        logger.info("Training dynamics model...")
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
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 0.5)
            self.dynamics_optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and iteration % 10 == 0:
                logger.info(f"Dynamics Iteration {iteration}, Loss: {loss.item():.6f}")
                
        avg_loss = np.mean(losses)
        logger.info(f"Dynamics training completed. Average loss: {avg_loss:.6f}")
        return avg_loss

    def train_value_network(self, states, rewards, next_states):
        logger.info("Training value network...")
        states = states.to(device)
        rewards = rewards.to(device).reshape(-1, 1)
        next_states = next_states.to(device)

        current_values = self.value_net(states)
        next_values = self.value_net(next_states)
        target_values = rewards + self.gamma * next_values.detach()
        
        value_loss = F.mse_loss(current_values, target_values)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        logger.info(f"Value network training completed. Loss: {value_loss.item():.6f}")
        return value_loss.item()

    def train_policy(self, initial_states, n_iterations=30):
        logger.info("Training policy...")
        processed_states = [state[0] if isinstance(state, tuple) else state for state in initial_states]
        states = torch.FloatTensor(np.array(processed_states)).to(device)
        exploration_factor = self.get_exploration_factor()

        for iteration in range(n_iterations):
            total_reward = 0
            logger.debug(f"Policy iteration {iteration}/{n_iterations}")

            batch_size = 10
            for batch in range(0, self.n_particles, batch_size):
                batch_reward = 0
                current_states = states[batch:batch+batch_size].clone()
                
                for t in range(self.horizon):
                    actions = self.policy(current_states)
                    mean, var = self.dynamics_model(current_states, actions)

                    smoothed_next_states = []
                    for i in range(current_states.size(0)):
                        next_state_sample = mean[i] + torch.sqrt(var[i]) * torch.randn_like(mean[i]).to(device)
                        smoothed_next_states.append(next_state_sample)

                    smoothed_next_states = torch.stack(smoothed_next_states)

                    reward = self.compute_reward(current_states, actions)
                    if t > 0:
                        state_differences = torch.cdist(smoothed_next_states, current_states)
                        exploration_bonus = exploration_factor * state_differences.mean()
                        reward += exploration_bonus
                        logger.debug(f"Time step {t}, Exploration bonus: {exploration_bonus:.4f}")

                    batch_reward += reward
                    current_states = smoothed_next_states

                total_reward += batch_reward * (batch_size / self.n_particles)

            avg_reward = total_reward / self.n_particles
            self.policy_optimizer.zero_grad()
            (-avg_reward).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            if iteration % 5 == 0:
                logger.info(f"Policy Iteration {iteration}, Reward: {avg_reward.item():.6f}")

    def add_experience(self, state, action, next_state, reward):
        logger.debug("Adding experience to buffer...")
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        state = torch.tensor(state, device='cpu').float().tolist()
        action = torch.tensor(action, device='cpu').float().tolist()
        next_state = torch.tensor(next_state, device='cpu').float().tolist()

        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.next_state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.next_state_buffer.pop(0)
            self.reward_buffer.pop(0)
        
        logger.debug(f"Buffer size: {len(self.state_buffer)}")

def clear_gpu_memory():
   """Utility function to clear GPU memory"""
   if torch.cuda.is_available():
       logger.info("Clearing GPU memory...")
       torch.cuda.empty_cache()
       logger.info("GPU memory cleared")

def print_gpu_memory():
   """Utility function to print GPU memory usage"""
   if torch.cuda.is_available():
       allocated = torch.cuda.memory_allocated()/1e9
       cached = torch.cuda.memory_reserved()/1e9
       logger.info(f"GPU memory allocated: {allocated:.2f} GB")
       logger.info(f"GPU memory cached: {cached:.2f} GB")

def save_checkpoint(agent, episode, rewards_history, dynamics_losses, value_losses,
                  weight_tracker, filename, metadata=None):
   """Save comprehensive checkpoint with metadata"""
   logger.info(f"Saving checkpoint at episode {episode}...")
   
   checkpoint = {
       'episode': episode,
       'dynamics_model_state': agent.dynamics_model.state_dict(),
       'policy_state': agent.policy.state_dict(),
       'value_net_state': agent.value_net.state_dict(),
       'dynamics_optimizer_state': agent.dynamics_optimizer.state_dict(),
       'policy_optimizer_state': agent.policy_optimizer.state_dict(),
       'value_optimizer_state': agent.value_optimizer.state_dict(),
       'rewards_history': rewards_history,
       'dynamics_losses': dynamics_losses,
       'value_losses': value_losses,
       'state_buffer': agent.state_buffer,
       'action_buffer': agent.action_buffer,
       'next_state_buffer': agent.next_state_buffer,
       'reward_buffer': agent.reward_buffer,
       'weight_history': weight_tracker.weight_history,
       'random_state': {
           'numpy': np.random.get_state(),
           'torch': torch.get_rng_state(),
           'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
       },
       'metadata': metadata or {},
       'timestamp': datetime.datetime.now().isoformat()
   }

   logger.info("Saving checkpoint file...")
   torch.save(checkpoint, filename)
   logger.info(f"Checkpoint saved successfully at {filename}")

   # Save backup
   backup_filename = f"backup_{os.path.basename(filename)}"
   torch.save(checkpoint, backup_filename)
   logger.info(f"Backup checkpoint saved at {backup_filename}")

def load_checkpoint(agent, filename):
    """Load checkpoint and restore training state"""
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)

    # Load model states
    agent.dynamics_model.load_state_dict(checkpoint['dynamics_model_state'])
    agent.policy.load_state_dict(checkpoint['policy_state'])
    agent.value_net.load_state_dict(checkpoint['value_net_state'])

    # Load optimizer states
    agent.dynamics_optimizer.load_state_dict(checkpoint['dynamics_optimizer_state'])
    agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
    agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])

    # Load buffers
    agent.state_buffer = checkpoint['state_buffer']
    agent.action_buffer = checkpoint['action_buffer']
    agent.next_state_buffer = checkpoint['next_state_buffer']
    agent.reward_buffer = checkpoint['reward_buffer']

    # Restore random states
    np.random.set_state(checkpoint['random_state']['numpy'])
    if torch.cuda.is_available() and checkpoint['random_state']['cuda'] is not None:
        torch.cuda.set_rng_state(checkpoint['random_state']['cuda'])

    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    print(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'Not available')}")

    return (checkpoint['episode'], checkpoint['rewards_history'],
            checkpoint['dynamics_losses'], checkpoint['value_losses'],
            checkpoint['weight_history'])
    
def train(env, agent, n_episodes=20000, checkpoint_freq=1000, plot_freq=100,
         resume_from=None, checkpoint_dir='checkpoints'):
   """Main training loop with extensive logging"""
   logger.info(f"Starting training for {n_episodes} episodes")
   logger.info(f"Checkpoint frequency: {checkpoint_freq}, Plot frequency: {plot_freq}")
   
   os.makedirs(checkpoint_dir, exist_ok=True)
   weight_tracker = WeightTracker()
   exploration_factors = []

   # Initialize or load from checkpoint
   if resume_from:
       logger.info(f"Loading checkpoint from {resume_from}")
       start_episode, rewards_history, dynamics_losses, value_losses, weight_history = \
           load_checkpoint(agent, resume_from)
       weight_tracker.weight_history = weight_history
       logger.info(f"Resuming training from episode {start_episode + 1}")
   else:
       logger.info("Starting fresh training")
       start_episode = 0
       rewards_history = []
       dynamics_losses = []
       value_losses = []

   try:
       for episode in range(start_episode, n_episodes):
           logger.info(f"\nStarting Episode {episode + 1}/{n_episodes}")
           episode_start_time = datetime.datetime.now()
           
           state = env.reset()
           episode_reward = 0
           episode_states = []
           episode_actions = []
           episode_rewards = []

           for step in range(500):
               action = agent.get_action(state)
               next_state, reward, done1, done2, info = env.step(action)

               if isinstance(state, tuple):
                   processed_state = state[0]
               else:
                   processed_state = state

               shaped_reward = reward * (1 + step/500)
               logger.debug(f"Step {step}: Reward = {reward:.4f}, Shaped Reward = {shaped_reward:.4f}")

               agent.add_experience(state, action, next_state, shaped_reward)
               episode_states.append(processed_state)
               episode_actions.append(action)
               episode_rewards.append(shaped_reward)

               episode_reward += reward
               state = next_state

               if done1 or done2:
                   logger.info(f"Episode ended at step {step}")
                   break

           rewards_history.append(episode_reward)
           agent.episode_count += 1
           logger.info(f"Episode {episode + 1} completed. Total reward: {episode_reward:.4f}")

           # Training and tracking
           if episode % 5 == 0 and len(agent.state_buffer) >= agent.batch_size:
               logger.info("Starting training cycle...")
               weight_tracker.track_weights(agent)

               dynamics_loss = agent.train_dynamics(verbose=False)
               if dynamics_loss is not None:
                   dynamics_losses.append(dynamics_loss)
                   logger.info(f"Dynamics Loss: {dynamics_loss:.6f}")

               agent.train_policy(episode_states)

               try:
                   states_tensor = torch.stack([
                       torch.FloatTensor(s) for s in episode_states[:-1]
                   ]).to(device)
                   rewards_tensor = torch.FloatTensor(episode_rewards[:-1]).to(device)
                   next_states_tensor = torch.stack([
                       torch.FloatTensor(s) for s in episode_states[1:]
                   ]).to(device)

                   value_loss = agent.train_value_network(
                       states_tensor, rewards_tensor, next_states_tensor)
                   value_losses.append(value_loss)
                   logger.info(f"Value Network Loss: {value_loss:.6f}")
               except Exception as e:
                   logger.error(f"Value network training failed: {str(e)}")

           # Memory management
           if episode % 10 == 0:
               clear_gpu_memory()
               print_gpu_memory()

           exploration_factors.append(agent.get_exploration_factor())

           # Progress tracking and visualization
           if (episode + 1) % plot_freq == 0:
               plot_learning_metrics(rewards_history, dynamics_losses, value_losses, exploration_factors)
               weight_tracker.plot_weight_trajectories()

           # Save checkpoint
           if (episode + 1) % checkpoint_freq == 0:
               checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pt')
               save_checkpoint(
                   agent, episode, rewards_history, dynamics_losses, value_losses,
                   weight_tracker, checkpoint_path,
                   metadata={'epsilon': agent.epsilon}
               )

           episode_duration = datetime.datetime.now() - episode_start_time
           logger.info(f"Episode duration: {episode_duration}")

   except KeyboardInterrupt:
       logger.warning("\nTraining interrupted by user. Saving checkpoint...")
       save_checkpoint(
           agent, episode, rewards_history, dynamics_losses, value_losses,
           weight_tracker, os.path.join(checkpoint_dir, 'interrupted_checkpoint.pt')
       )
       raise

   logger.info("Training completed successfully")
   return rewards_history, dynamics_losses, value_losses, weight_tracker

def plot_learning_metrics(rewards_history, dynamics_losses, value_losses, exploration_factors):
    """Plot comprehensive learning metrics including rewards, losses and exploration"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deep PILCO Learning Metrics', fontsize=16)

    # Plot rewards
    ax1.plot(rewards_history, alpha=0.3, label='Raw')
    window = min(50, len(rewards_history)//10)
    if len(rewards_history) > window:
        smoothed = np.convolve(rewards_history,
                             np.ones(window)/window,
                             mode='valid')
        ax1.plot(smoothed, 'r', label='Smoothed')
    ax1.set_title('Learning Progress')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot dynamics loss
    if dynamics_losses:
        ax2.plot(dynamics_losses)
        ax2.set_yscale('log')
        ax2.set_title('Dynamics Model Loss')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Loss (log scale)')
        ax2.grid(True)

    # Plot value loss
    if value_losses:
        ax3.plot(value_losses)
        ax3.set_yscale('log')
        ax3.set_title('Value Network Loss')
        ax3.set_xlabel('Training Iteration')
        ax3.set_ylabel('Loss (log scale)')
        ax3.grid(True)

    # Plot exploration
    ax4.plot(exploration_factors)
    ax4.set_title('Exploration Factor')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'learning_metrics_{timestamp}.png')
    logger.info(f"Learning metrics plot saved")
    plt.close()

class WeightTracker:
    def __init__(self):
        self.weight_history = {
            'policy': [],
            'dynamics': [],
            'value': []
        }
    def track_weights(self, agent):
        """Track weights and gradients for all networks"""
        # Track policy weights
        policy_weights = []
        for name, param in agent.policy.named_parameters():
            if 'weight' in name:
                policy_weights.append({
                    'name': name,
                    'norm': torch.norm(param.data).item(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'grad_norm': torch.norm(param.grad).item() if param.grad is not None else 0
                })
        self.weight_history['policy'].append(policy_weights)

        # Track dynamics model weights
        dynamics_weights = []
        for name, param in agent.dynamics_model.named_parameters():
            if 'weight' in name:
                dynamics_weights.append({
                    'name': name,
                    'norm': torch.norm(param.data).item(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'grad_norm': torch.norm(param.grad).item() if param.grad is not None else 0
                })
        self.weight_history['dynamics'].append(dynamics_weights)

        # Track value network weights
        value_weights = []
        for name, param in agent.value_net.named_parameters():
            if 'weight' in name:
                value_weights.append({
                    'name': name,
                    'norm': torch.norm(param.data).item(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'grad_norm': torch.norm(param.grad).item() if param.grad is not None else 0
                })
        self.weight_history['value'].append(value_weights)
    
    def plot_weight_trajectories(self):
        """Plot weight analysis for all networks"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Network Weight Analysis', fontsize=16)

        for idx, network_name in enumerate(['policy', 'dynamics', 'value']):
            # Plot weight norms
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]

            weights = self.weight_history[network_name]
            if not weights:
                continue

            # Extract metrics for each layer
            layers = len(weights[0])
            for layer in range(layers):
                norms = [w[layer]['norm'] for w in weights]
                grad_norms = [w[layer]['grad_norm'] for w in weights]

                ax1.plot(norms, label=f'Layer {layer+1}')
                ax1.set_title(f'{network_name} Weight Norms')
                ax1.set_xlabel('Training Step')
                ax1.set_ylabel('L2 Norm')
                ax1.legend()
                ax1.grid(True)

                ax2.plot(grad_norms, label=f'Layer {layer+1}')
                ax2.set_title(f'{network_name} Gradient Norms')
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Gradient L2 Norm')
                ax2.set_yscale('log')
                ax2.legend()
                ax2.grid(True)

        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'weight_analysis_{timestamp}.png')
        logger.info(f"Weight analysis plot saved")
        plt.close()

def main():
    logger.info("Starting Deep PILCO DBS training script")
    
    # Create environment
    logger.info("Creating environment...")
    env = rl_dbs.gym_oscillator.envs.oscillatorEnv()
    logger.info("Environment created successfully")

    # Initialize agent
    logger.info("Initializing agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    agent = DeepPILCO(state_dim, action_dim)
    logger.info("Agent initialized successfully")

    # Setup checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory created at: {checkpoint_dir}")

    # Check for existing checkpoints
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_ep')],
            key=lambda x: int(x.split('ep')[1].split('.')[0])
        )
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")

    # Training configuration
    config = {
        'n_episodes': 10000,
        'checkpoint_freq': 1000,
        'plot_freq': 100,
        'resume_from': None
    }
    
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    try:
        # Train the agent
        rewards_history, dynamics_losses, value_losses, weight_tracker = train(
            env, agent, **config
        )

        # Save final results
        logger.info("Saving final results...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_results = {
            'rewards_history': rewards_history,
            'dynamics_losses': dynamics_losses,
            'value_losses': value_losses,
            'weight_history': weight_tracker.weight_history,
            'config': config,
            'timestamp': timestamp
        }
        
        results_path = f'results/final_results_{timestamp}.pt'
        os.makedirs('results', exist_ok=True)
        torch.save(final_results, results_path)
        logger.info(f"Final results saved to {results_path}")

        # Plot final learning curves
        logger.info("Generating final plots...")
        plot_learning_metrics(rewards_history, dynamics_losses, value_losses, 
                            [agent.get_exploration_factor() for _ in range(len(rewards_history))])
        weight_tracker.plot_weight_trajectories()
        
        # Print final statistics
        logger.info("\nTraining Complete! Final Statistics:")
        logger.info(f"Total episodes trained: {len(rewards_history)}")
        logger.info(f"Final average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.4f}")
        logger.info(f"Best reward achieved: {max(rewards_history):.4f}")
        logger.info(f"Final dynamics loss: {dynamics_losses[-1]:.6f}")
        logger.info(f"Final value loss: {value_losses[-1]:.6f}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
    finally:
        logger.info("Script finished")

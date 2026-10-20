"""
Deep Q-Learning (DQN) on Gymnasium FrozenLake — heavily commented study copy.

Flow:
state -> choose action -> environment returns reward/next state
      -> save experience -> sample old experiences -> calculate targets
      -> compare predictions with targets -> update policy network.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F



# 1. DQN MODEL: maps a state representation to one Q value per action

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()  # Initializes PyTorch's module machinery.

        # First fully connected layer: input features -> hidden neurons.
        # FrozenLake has 16 states, represented by 16 element one hot vectors.
        self.fc1 = nn.Linear(in_states, h1_nodes)

        # Output layer: one score (Q value) for each action.
        # Here out_actions=4: left, down, right, up.
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        # Linear transformation of the input state.
        x = self.fc1(x)

        # ReLU sets negative activations to zero and keeps positive ones.
        # This non-linearity lets the network learn more than a linear function.
        x = F.relu(x)

        # Produce action-value estimates. No softmax: Q-values are not probabilities.
        x = self.out(x)
        return x



# 2. REPLAY MEMORY: stores and randomly reuses past experiences

class ReplayMemory:
    def __init__(self, maxlen):
        # A bounded queue. When full, appending removes the oldest experience.
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # An experience is (state, action, next_state, reward, terminated).
        self.memory.append(transition)

    def sample(self, sample_size):
        # Choose sample_size experiences at random, without replacement.
        # The caller must ensure the memory contains enough entries.
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Allows len(memory_object) to return the number of stored experiences.
        return len(self.memory)


# 3. TRAINING CONTROLLER: coordinates environment and learning

class FrozenLakeDQL:
    # Hyperparameters: adjustable choices that influence learning.
    learning_rate_a = 0.001       # Adam's step-size setting (alpha).
    discount_factor_g = 0.9       # Gamma: how much future values are discounted.
    network_sync_rate = 10        # Environment-step threshold for target-network copy.
    replay_memory_size = 1000     # Maximum experiences retained.
    mini_batch_size = 32          # Experiences used for one optimization update.

    # MSE compares predicted Q-values against target Q-values.
    loss_fn = nn.MSELoss()
    optimizer = None              # Created later, once policy_dqn exists.
    ACTIONS = ['L', 'D', 'R', 'U']  # Action indices 0,1,2,3 for display.

    def train(self, episodes, render=False, is_slippery=False):
        """Train the agent for the requested number of episodes."""
        # Create a 4x4 FrozenLake environment.
        # render=True displays it; otherwise training runs without a window.
        env = gym.make(
            'FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
            render_mode='human' if render else None
        )

        # For the 4x4 map: 16 discrete states and 4 possible actions.
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1  # 100% random actions initially: maximum exploration.
        memory = ReplayMemory(self.replay_memory_size)

        # Policy network is the one whose parameters will be trained.
        # Input width=16 (one-hot state), hidden width=16, output width=4 actions.
        policy_dqn = DQN(num_states, num_states, num_actions)

        # Target network supplies reference Q-values for learning targets.
        target_dqn = DQN(num_states, num_states, num_actions)

        # Start them with identical weights and biases.
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Adam updates only policy_dqn. The target network is not optimized directly.
        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=self.learning_rate_a
        )

        # One slot per episode; 1 means that episode reached the goal (reward=1).
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []  # Epsilon values recorded when appended below.
        step_count = 0         # Counts environment steps for target synchronization.

        for i in range(episodes):
            # Gymnasium reset returns (observation, info); [0] takes the state.
            state = env.reset()[0]
            terminated = False  # True for a terminal outcome, e.g. goal or hole.
            truncated = False   # True when an external limit stops the episode.

            # Continue until the episode terminates or is truncated.
            while not terminated and not truncated:

                # Epsilon-greedy action selection balances exploration/exploitation.
                if random.random() < epsilon:
                    # Explore: choose a random valid action (0=L,1=D,2=R,3=U).
                    action = env.action_space.sample()
                else:
                    # Exploit: choose the action with the largest predicted Q-value.
                    # no_grad is appropriate because selecting an action isn't training.
                    with torch.no_grad():
                        action = policy_dqn(
                            self.state_to_dqn_input(state, num_states)
                        ).argmax().item()
                        # argmax -> index of largest score; item -> Python integer.

                # Execute action. Gymnasium returns next state, reward, two flags, info.
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Store the transition for later replay-based learning.
                # This original version stores terminated but not truncated.
                memory.append((state, action, new_state, reward, terminated))

                state = new_state  # The next decision starts from the new state.
                step_count += 1    # Count this environment step.

            # FrozenLake's goal reward is 1. This marks the episode as successful.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Start learning only if the buffer is large enough AND at least one
            # success has been recorded so far. The success condition is a design
            # choice in this example, not a universal DQN requirement.
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon toward 0, encouraging more greedy choices later.
                # Important: this runs whenever the if-condition runs, not exactly
                # once per episode. So decay speed depends on how often training runs.
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Hard-update target network by copying policy parameters.
                # Check uses > rather than >=. Counter resets after synchronization.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()  # Release the environment.

        # Save policy weights/biases; test() recreates the model architecture to load them.
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

        # Plot recent success count. The slice includes up to 101 episodes (x-100 to x).
        # Despite the original comment, this is a rolling sum, not an average.
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)
        plt.plot(sum_rewards)

        # Epsilon history is recorded only when optimization occurs, so its length
        # may not equal the number of episodes.
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.savefig('frozen_lake_dql.png')
        
    # 4. OPTIMIZATION: calculate targets, loss, and update policy network


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Read the network input width rather than hardcoding 16.
        num_states = policy_dqn.fc1.in_features
        current_q_list = []  # Policy predictions, one vector per experience.
        target_q_list = []   # Desired Q-value vectors, one per experience.

        for state, action, new_state, reward, terminated in mini_batch:
            # Q-learning target:
            # terminal:     target = reward
            # non-terminal: target = reward + gamma * max_a Q_target(next_state, a)
            if terminated:
                # Episode ended: no future return is added.
                # Goal target=1; hole target=0 in this FrozenLake setup.
                target = torch.FloatTensor([reward])
            else:
                # Estimate the best future value with the target network.
                # no_grad means gradients do not flow through this target calculation.
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g *
                        target_dqn(
                            self.state_to_dqn_input(new_state, num_states)
                        ).max()
                    )

            # Current policy prediction for ALL actions at the original state.
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Start with target-network values for all actions at the original state.
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))

            # Replace only the value for the action actually taken.
            # Example: action=2 means replace the "right" entry.
            target_q[action] = target
            target_q_list.append(target_q)

        # Stack 32 vectors of 4 Q-values into tensors of shape [32, 4].
        current_q_batch = torch.stack(current_q_list)
        target_q_batch = torch.stack(target_q_list)

        # Scalar loss measuring the mean squared difference between these batches.
        loss = self.loss_fn(current_q_batch, target_q_batch)

        self.optimizer.zero_grad()  # Clear gradients left from the previous update.
        loss.backward()             # Calculate gradients of loss w.r.t. policy parameters.
        self.optimizer.step()       # Adam uses those gradients to update policy parameters.


    # 5. STATE ENCODING: integer state -> one-hot input vector

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        # Example: state=1, num_states=4 -> [0., 1., 0., 0.].
        # For 16 states, the vector has 16 elements, exactly one set to 1.
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor


    # 6. TESTING: load saved policy and act greedily without learning

    def test(self, episodes, is_slippery=False):
        env = gym.make(
            'FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
            render_mode='human'
        )
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Recreate the same architecture and load trained weights.
        policy_dqn = DQN(num_states, num_states, num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()  # Evaluation mode; this model has no dropout/batchnorm.

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            while not terminated and not truncated:
                # Greedy testing: no epsilon exploration and no gradient tracking.
                with torch.no_grad():
                    action = policy_dqn(
                        self.state_to_dqn_input(state, num_states)
                    ).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()


    # 7. PRINT each state's Q-values and preferred action

    def print_dqn(self, dqn):
        num_states = dqn.fc1.in_features
        for s in range(num_states):
            q_values = ''

            # Run the network and convert its output tensor into a Python list.
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q) + ' '

            q_values = q_values.rstrip()  # Remove final space.
            # argmax returns the index of the largest Q-value; ACTIONS maps index to label.
            best_action = self.ACTIONS[
                dqn(self.state_to_dqn_input(s, num_states)).argmax()
            ]

            # Print state, preferred action, and all four Q-values.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()  # New line after each row of four states.



# 8. ENTRY POINT: run training and then test the saved policy

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    is_slippery = False  # Deterministic movement; set True to enable slipping.
    frozen_lake.train(1000, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)

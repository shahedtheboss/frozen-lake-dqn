# Deep Q-Learning on FrozenLake 

A hands-on implementation of **Deep Q-Learning (DQN)** using [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/), built to understand how a reinforcement learning agent learns to navigate an environment through trial and error.

The project uses Gymnasium's 4×4 FrozenLake environment, where an agent must navigate from its starting position to a goal while avoiding holes.

##  Project Overview

Instead of following a fixed path, the agent learns action values using a neural network. It interacts with the environment, collects experiences, and updates its predictions to improve its decision-making.

The implementation is heavily commented to make the reasoning behind each component easier to understand.

##  How It Works

The training process follows this loop:

1. **Observe the state** — The agent receives its current position on the FrozenLake grid.
2. **Choose an action** — An epsilon-greedy strategy balances random exploration with choosing the action that has the highest predicted Q-value.
3. **Interact with the environment** — The agent moves and receives a reward and the next state.
4. **Store the experience** — The transition is saved in replay memory.
5. **Sample past experiences** — A random mini-batch is used to train the policy network.
6. **Calculate target Q-values** — The target network helps estimate the expected return of an action.
7. **Update the network** — The predicted Q-values are compared with target values using a loss function, and the optimizer updates the model parameters.
8. **Synchronize the networks** — The target network periodically receives the policy network's weights.

Over time, these updates can help the agent learn which actions are more likely to lead to the goal.

##  Key Concepts

| Concept | Purpose |
|---|---|
| Deep Q-Network (DQN) | Estimates the value of each possible action in a state. |
| Q-values | Represent the estimated long-term return of taking an action. |
| Epsilon-greedy exploration | Balances exploration with exploiting the current policy. |
| Experience replay | Stores transitions and samples them randomly for training. |
| Target network | Provides more stable target estimates during learning. |
| Bellman target | Combines the immediate reward with the estimated future return. |
| Mean Squared Error (MSE) | Measures the difference between predicted and target Q-values. |
| Adam optimizer | Updates the policy network's parameters. |

##  Environment

- **Environment:** `FrozenLake-v1`
- **Map:** 4×4 grid
- **State representation:** One-hot encoded state vector
- **Action space:** Left, Down, Right, Up
- **Default setting:** `is_slippery=False`
- **Reward:** The agent receives a positive reward for reaching the goal; falling into a hole ends the episode without a positive reward.

With `is_slippery=False`, movement is deterministic. Enabling slippery mode introduces uncertainty into the environment's transitions.

##  Tech Stack

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shahedtheboss/frozen-lake-dqn.git
cd frozen-lake-dqn
```

Replace `shahedtheboss` with your GitHub username.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch gymnasium numpy matplotlib
```

### 4. Run the implementation

Save the Python implementation as `frozen_lake_dqn.py`, then run:

```bash
python frozen_lake_dqn.py
```

The script is configured to train for 1,000 episodes and then test the learned policy over 10 episodes.

During execution, it saves:

- `frozen_lake_dql.pt` — the trained policy network's weights.
- `frozen_lake_dql.png` — plots showing the rolling reward total and epsilon history.

The exact learning outcome depends on the implementation, environment settings, and training behavior.

##  Training Visualization

The training script generates a plot containing:

- **Reward history:** The number of successful episodes within a rolling window of up to 100 episodes.
- **Epsilon history:** How the exploration rate changes during training.

These plots help inspect the learning process, but a successful training run alone does not establish that the agent performs reliably. Repeated evaluation and further experimentation would be needed to assess performance.

##  Learning Goals

This project is part of my journey into reinforcement learning and PyTorch, with an emphasis on understanding the implementation rather than treating the algorithm as a black box.

The main goals are to:

- Understand the DQN training loop from environment interaction to gradient updates.
- Explore how experience replay and target networks work together.
- Understand how Bellman targets are constructed.
- Connect reinforcement learning theory with working Python code.
- Build a foundation for more complex reinforcement learning and robotics projects.

##  Future Improvements

- Compare learning behavior with slippery and non-slippery environments.
- Experiment with learning rates, discount factors, batch sizes, and target-network update intervals.
- Evaluate performance over multiple independent runs.
- Compare reward curves and success rates across different configurations.
- Explore improvements to the training procedure and alternative reinforcement learning algorithms.

##  License

This project is intended for educational and learning purposes. Add a license file if you want to specify the terms under which others may use or modify the code.

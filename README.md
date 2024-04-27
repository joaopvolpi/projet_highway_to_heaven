# Projet_highway_to_heaven
![13c18d3069ab9c6e5bf864a06cd7f096](https://github.com/joaopvolpi/projet_highway_to_heaven/assets/52925699/da9dac4a-ca83-46b0-91b4-a8114792835f)

This repository contains the final reports and implementation code for a project focusing on evaluating different reinforcement learning (RL) algorithms within complex simulated environments. The work is split among different members of the group, with this particular repository dedicated to the Racetrack, Merge, Highway, Roundabout environments from the `highway-env` Python library.

## Project Overview

The aim is to explore, implement, and assess the effectiveness of Deep Q-Network (DQN), Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) algorithms in guiding a vehicle through traffic on a predefined track, competing or coexisting with other cars. In every environment the agent is designed to:

- Stay within the track boundaries
- Maximize speed for higher rewards
- Avoid collisions with other vehicles

# Racetrack and Roundabout Environment Overview

### PPO Configuration: 
Employs an MlpPolicy with two hidden layers, a batch size of 64, and runs for 3,000 timesteps. Key hyperparameters include a learning rate of 0.0005 and a gamma of 0.8.

### DDPG Configuration: 
Uses an MlpPolicy with NormalActionNoise for exploration, running similarly for 3,000 timesteps with architecture tailored for continuous action spaces.

### Key Experiments and Results
#### Racetrack Environment: 
The PPO model outperformed DDPG, showing better stability and adherence to the track. DDPG, however, faced challenges like drifting which highlighted the need for further parameter tuning.

#### Roundabout Environment: 
Testing in a roundabout setting revealed that while the PPO model managed basic driving tasks, it struggled with more complex merging scenarios. This highlighted the need for more diverse scenario training.

# Highway Env
The DQN architecture employed consists of a feedforward neural network with two layers, using ReLU activation functions and an AdamW optimizer. The training involved initializing the DQN and a target DQN, using an epsilon-greedy policy for action selection, and employing a replay buffer for experience replay.

Key adaptations were made to optimize the architecture for complex environments like the highway scenario. Hyperparameter tuning was critical in enhancing the model’s performance, with adjustments in learning rates and sigma values. The learning process was visualized through loss graphs, highlighting the training challenges and the need for careful parameter tuning.

The project demonstrated that while the DQN could navigate the highway environment, challenges like avoiding crashes and handling complex maneuvers persisted. Future work will focus on enhancing model robustness and expanding testing to various scenarios to achieve more reliable performance.


## References

- [Highway-env GitHub Repository](https://github.com/eleurent/highway-env)
- [Deep Deterministic Policy Gradient (DDPG) Paper](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization (PPO) Paper](https://arxiv.org/abs/1707.06347)
- [Deep Q Network (DQN) Paper](https://arxiv.org/abs/1312.5602)

## Contributors
- Diego Ruiz Ponsoda
- João Pedro Monteiro Volpi
- Lucas Vitoriano de Queiroz Lira

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

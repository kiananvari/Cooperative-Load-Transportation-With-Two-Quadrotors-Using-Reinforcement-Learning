# Cooperative-Load-Transportation-With-Two-Quadrotors-Using-Reinforcement-Learning
The Synchronous Quadcopter Control project aims to synchronize two carrier quadcopters to move a rod from the origin to the destination. The project utilizes deep curriculum reinforcement learning techniques to train the quadcopters to perform this task in a synchronized manner while maintaining balance.

# Project Specifications 
- Agents exist in an environment with intersections, seas, and gates as targets.
- Each road in the intersection is 60 meters long, with a gate as the goal at the end.
- Quadcopters are connected to a two-meter-long rod in the center of the intersection.
- The agents' task is to move the rod from the starting point to the targets while maintaining balance and avoiding the ground or sea.
- The quadcopters must simultaneously lower the bar height to land next to the target after reaching each gate.

# Suggested Approach
The project suggests the following approach:

1- Using deep learning techniques to train the quadcopters.

2- Implementing the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.

3- Incorporating Curriculum Learning to divide the learning process into stages, starting from simple tasks and gradually progressing to more difficult ones.

## The stages of the gradual learning approach are as follows:

1- Establishing balance: Training the agents to maintain balance and stabilize themselves in the air.

2- Reducing the distance: Teaching the agents to move towards the target and minimize the distance between their current position and the desired location.

3- Moving with the right angle: Training the agents to move with the correct angle towards the target, ensuring precise movement along specific paths.

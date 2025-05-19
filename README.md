# Automated Reinforcement Learning-Based Test Case Generation and Prioritization for Games

This project implements an automated framework for generating and prioritizing test cases for games 
using reinforcement learning (RL), static analysis, and model-based testing techniques. 
The system is designed to train an RL agent in a game-like environment and use its learned behavior to create 
intelligent and diverse test cases, followed by a code-coverage-based prioritization system.

## 🧠 Project Overview

The project consists of the following major components:

- **Custom Game Environment**: A `TaxiGymEnv` simulating a grid-based taxi game using Gymnasium.
- **Reinforcement Learning Agent**: Trained with PPO (Proximal Policy Optimization) using Stable-Baselines3 and Shimmy for compatibility.
- **Finite State Machine (FSM)**: Encodes the game's logic and state transitions.
- **Test Case Generator**: Generates valid and partial behavior traces from the trained agent.
- **Coverage Mapper**: Uses `coverage.py` and `ast` to map test executions to the functions they touch.
- **Prioritizer**: Uses Git diff and coverage data to re-order test cases by relevance after code changes.


## 🧪 Requirements

- Python 3.10+
- Gymnasium
- Shimmy
- Stable-Baselines3
- Coverage.py
- Git (for diff-based prioritization)
- statemachine

Install requirements:

```bash
pip install gymnasium shimmy stable-baselines3 coverage gitpython statemachine
"""
Example custom reward function for EV2Gym environment.

This file demonstrates how to create custom reward functions that can be used
with train_stable_baselines.py via the --reward_function argument.

Usage:
    python train_stable_baselines.py --reward_function my_reward:custom_reward_function
"""

import numpy as np


def custom_reward_function(env, action):
    """
    Example custom reward function for the EV2Gym environment.
    
    This function calculates the reward based on the environment state and action taken.
    
    Args:
        env: The EV2Gym environment instance
        action: The action taken by the agent
        
    Returns:
        float: The reward value
    """
    reward = 0.0
    
    # Example: Reward for serving EVs
    if hasattr(env, 'total_ev_served'):
        reward += env.total_ev_served * 10.0
    
    # Example: Penalty for user dissatisfaction
    if hasattr(env, 'average_user_satisfaction'):
        reward += env.average_user_satisfaction * 100.0
    
    # Example: Reward for profit
    if hasattr(env, 'total_profits'):
        reward += env.total_profits
    
    # Example: Penalty for grid violations
    if hasattr(env, 'power_tracker_violation'):
        reward -= env.power_tracker_violation * 50.0
    
    return reward


def simple_profit_reward(env, action):
    """
    A simple reward function focused on profit maximization.
    
    Args:
        env: The EV2Gym environment instance
        action: The action taken by the agent
        
    Returns:
        float: The reward value (profit)
    """
    if hasattr(env, 'total_profits'):
        return env.total_profits
    return 0.0


def balanced_reward_function(env, action):
    """
    A balanced reward function that considers multiple objectives.
    
    Args:
        env: The EV2Gym environment instance
        action: The action taken by the agent
        
    Returns:
        float: The reward value
    """
    reward = 0.0
    
    # Profit component (weight: 1.0)
    if hasattr(env, 'total_profits'):
        reward += env.total_profits
    
    # User satisfaction component (weight: 50.0)
    if hasattr(env, 'average_user_satisfaction'):
        reward += env.average_user_satisfaction * 50.0
    
    # Grid stability penalty (weight: -100.0)
    if hasattr(env, 'total_transformer_overload'):
        reward -= env.total_transformer_overload * 100.0
    
    # Tracking error penalty (weight: -10.0)
    if hasattr(env, 'tracking_error'):
        reward -= env.tracking_error * 10.0
    
    return reward

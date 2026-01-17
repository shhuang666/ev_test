"""
Example custom state function for EV2Gym environment.

This file demonstrates how to create custom state functions that can be used
with train_stable_baselines.py via the --state_function argument.

Usage:
    python train_stable_baselines.py --state_function my_state:custom_state_function
"""

import numpy as np


def custom_state_function(env):
    """
    Example custom state function for the EV2Gym environment.
    
    This function extracts relevant information from the environment to create
    the observation space for the RL agent.
    
    Args:
        env: The EV2Gym environment instance
        
    Returns:
        numpy.ndarray: The state observation vector
    """
    state = []
    
    # Example: Include current time step information
    state.append(env.current_step / env.simulation_length)
    
    # Example: Include price information if available
    if hasattr(env, 'prices') and env.prices is not None:
        current_price = env.prices[env.current_step] if env.current_step < len(env.prices) else 0
        state.append(current_price)
    
    # Example: Include EV-related information
    for ev in env.evs:
        if ev.connected:
            state.extend([
                ev.current_charge / ev.battery_capacity,  # Normalized current charge
                ev.desired_charge / ev.battery_capacity,   # Normalized desired charge
                ev.time_of_departure - env.current_step,   # Time until departure
            ])
        else:
            state.extend([0.0, 0.0, 0.0])  # Placeholder for disconnected EVs
    
    return np.array(state, dtype=np.float32)


def minimal_state_function(env):
    """
    A minimal state function that only includes essential information.
    
    Args:
        env: The EV2Gym environment instance
        
    Returns:
        numpy.ndarray: The state observation vector
    """
    state = []
    
    # Current time step (normalized)
    state.append(env.current_step / env.simulation_length)
    
    # For each EV, include basic charge information
    for ev in env.evs:
        if ev.connected:
            state.append(ev.current_charge / ev.battery_capacity)
        else:
            state.append(0.0)
    
    return np.array(state, dtype=np.float32)


def V2G_profit_max_no_forecast(env, *args):
    '''
    This is a simplification of the V2GProfitMax state function, which removes the forecasted charge prices.
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

    charge_prices = abs(env.charge_prices[0])
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state
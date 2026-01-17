# EV2Gym Configuration Comparison

This document provides a comprehensive comparison of the three recommended configurations in `train_stable_baselines.py` for training RL agents with the EV2Gym environment.

## Overview

The `train_stable_baselines.py` script supports three different configuration setups, each optimized for different objectives in EV charging management. The configuration is selected via the `--config_file` argument and determines which reward function and state function are used.

---

## Table 1: YAML Configuration Settings Comparison

| **Parameter** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|---------------|------------------|---------------|------------------------|
| **File Path** | `ev2gym/example_config_files/V2GProfitMax.yaml` | `ev2gym/example_config_files/PublicPST.yaml` | `ev2gym/example_config_files/V2GProfitPlusLoads.yaml` |
| **Scenario** | `workplace` | `public` | `workplace` |
| **V2G Enabled** | ✅ `True` | ❌ `False` | ✅ `True` |
| **Number of Charging Stations** | 25 | 20 | 25 |
| **Max Charge Current** | 32 A | 16 A | 32 A |
| **Max Discharge Current** | -32 A | 0 A (no discharge) | -32 A |
| **Power Setpoint Enabled** | ❌ `False` | ✅ `True` | ❌ `False` |
| **Power Setpoint Flexibility** | 80% | 80% | 80% |
| **Inflexible Loads** | ❌ `False` | ❌ `False` | ✅ `True` |
| **Solar Power** | ❌ `False` | ❌ `False` | ✅ `True` |
| **Demand Response** | ❌ `False` | ❌ `False` | ✅ `True` |
| **EV Specs File** | `ev_specs_v2g_enabled2024.json` | `ev_specs_ev_plus_phev.json` | `ev_specs_v2g_enabled2024.json` |
| **Min Time of Stay** | 180 min | 60 min | 180 min |
| **Simulate Grid** | ❌ `False` | ❌ `False` | ❌ `False` |
| **Timescale** | 15 min | 15 min | 15 min |
| **Simulation Length** | 112 steps | 112 steps | 112 steps |
| **Spawn Multiplier** | 5 | 5 | 5 |

### Key Differences

1. **V2GProfitMax**: 
   - Workplace scenario with V2G enabled
   - No external loads or constraints
   - Focus on bidirectional charging for profit

2. **PublicPST**: 
   - Public charging scenario (shorter stays)
   - **Power setpoint tracking enabled**
   - Unidirectional charging only (no V2G)
   - Lower charging current (16A vs 32A)

3. **V2GProfitPlusLoads**: 
   - Workplace scenario with V2G enabled
   - **Includes inflexible loads, solar power, and demand response**
   - Most realistic and complex scenario

---

## Table 2: Reward Functions Comparison

| **Aspect** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|------------|------------------|---------------|------------------------|
| **Function Name** | `profit_maximization` | `SquaredTrackingErrorReward` | `ProfitMax_TrPenalty_UserIncentives` |
| **Primary Objective** | Maximize profit | Minimize tracking error | Multi-objective balance |
| **Profit Component** | ✅ Yes | ❌ No | ✅ Yes |
| **Tracking Error** | ❌ No | ✅ Yes (squared) | ❌ No |
| **Transformer Penalty** | ❌ No | ❌ No | ✅ Yes (100× overload) |
| **User Satisfaction** | ✅ Yes (exponential) | ❌ No | ✅ Yes (exponential) |
| **Grid Constraints** | ❌ No | ❌ No | ✅ Yes |

### Reward Function Equations

#### 1. `profit_maximization` (V2GProfitMax)

$$
R_t = C_{\text{total}} - 100 \sum_{i \in \text{EVs}} e^{-10 \cdot s_i}
$$

Where:
- $C_{\text{total}}$ = Total costs (negative for profit, positive for cost)
- $s_i$ = User satisfaction score for EV $i$ (0-1)
- The exponential term heavily penalizes low satisfaction scores

**Code Reference** (lines 78-87):
```python
def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for score in user_satisfaction_list:
        reward -= 100 * math.exp(-10*score)
    return reward
```

---

#### 2. `SquaredTrackingErrorReward` (PublicPST)

$$
R_t = -\left(\min(P_{\text{setpoint}}^t, P_{\text{potential}}^t) - P_{\text{actual}}^t\right)^2
$$

Where:
- $P_{\text{setpoint}}^t$ = Power setpoint at time $t$
- $P_{\text{potential}}^t$ = Maximum charging power potential at time $t$
- $P_{\text{actual}}^t$ = Actual power usage at time $t$
- The negative sign ensures reward is always ≤ 0 (penalty for deviation)

**Code Reference** (lines 7-14):
```python
def SquaredTrackingErrorReward(env,*args):
    reward = - (min(env.power_setpoints[env.current_step-1], 
                    env.charge_power_potential[env.current_step-1]) -
                env.current_power_usage[env.current_step-1])**2
    return reward
```

---

#### 3. `ProfitMax_TrPenalty_UserIncentives` (V2GProfitPlusLoads)

$$
R_t = C_{\text{total}} - 100 \sum_{j \in \text{Transformers}} O_j - 100 \sum_{i \in \text{EVs}} e^{-10 \cdot s_i}
$$

Where:
- $C_{\text{total}}$ = Total costs (profit component)
- $O_j$ = Overload amount for transformer $j$ (in kW beyond limit)
- $s_i$ = User satisfaction score for EV $i$ (0-1)
- **Three components**: Profit + Transformer penalty + User satisfaction

**Code Reference** (lines 34-44):
```python
def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward
```

### Reward Function Characteristics

| **Characteristic** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|--------------------|------------------|---------------|------------------------|
| **Range** | $(-\infty, +\infty)$ | $(-\infty, 0]$ | $(-\infty, +\infty)$ |
| **Sign** | Can be positive (profit) | Always negative (penalty) | Can be positive (profit) |
| **Components** | 2 (profit + user) | 1 (tracking error) | 3 (profit + transformer + user) |
| **Complexity** | Medium | Low | High |
| **Sparsity** | Dense | Dense | Dense |

---

## Table 3: State Functions Comparison

| **Aspect** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|------------|------------------|---------------|------------------------|
| **Function Name** | `V2G_profit_max` | `PublicPST` | `V2G_profit_max_loads` |
| **State Dimension** | Variable | Variable | Variable |
| **Time Features** | Current step | Normalized step (0-1) | Current step |
| **Price Forecast** | ✅ 20 steps ahead | ❌ No | ✅ 20 steps ahead |
| **Power Setpoint** | ❌ No | ✅ Current step | ❌ No |
| **Current Power Usage** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Load Forecast** | ❌ No | ❌ No | ✅ 20 steps (loads - PV) |
| **Power Limits** | ❌ No | ❌ No | ✅ 20 steps ahead |
| **EV Features per CS** | 2 (SoC, time to departure) | 3 (full flag, energy exchanged, time since arrival) | 2 (SoC, time to departure) |

### State Function Details

#### 1. `V2G_profit_max` (V2GProfitMax)

**State Vector Components:**
```python
state = [
    current_step,                          # 1 feature
    current_power_usage[t-1],              # 1 feature
    charge_prices[t:t+20],                 # 20 features (price forecast)
    # For each charging station:
    [EV_SoC, time_to_departure]            # 2 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 20 + 2 \times N_{\text{cs}} = 22 + 2N_{\text{cs}}$

For 25 charging stations: **72 features**

**Code Reference** (lines 65-106):
```python
def V2G_profit_max(env, *args):
    state = [(env.current_step)]
    state.append(env.current_power_usage[env.current_step-1])
    
    charge_prices = abs(env.charge_prices[0, env.current_step:env.current_step+20])
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    state.append(charge_prices)
    
    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([EV.get_soc(), 
                                     EV.time_of_departure - env.current_step])
                    else:
                        state.append(np.zeros(2))
    
    return np.array(np.hstack(state))
```

---

#### 2. `PublicPST` (PublicPST)

**State Vector Components:**
```python
state = [
    current_step / simulation_length,      # 1 feature (normalized)
    power_setpoint[t],                     # 1 feature
    current_power_usage[t-1],              # 1 feature
    # For each charging station:
    [full_flag, energy_exchanged, time_since_arrival]  # 3 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 1 + 3 \times N_{\text{cs}} = 3 + 3N_{\text{cs}}$

For 20 charging stations: **63 features**

**Code Reference** (lines 6-63):
```python
def PublicPST(env, *args):
    state = [(env.current_step/env.simulation_length)]
    
    if env.current_step < env.simulation_length:  
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = np.zeros((1))
    
    state.append(setpoint)
    state.append(env.current_power_usage[env.current_step-1])
    
    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0.5,  # full flag
                            EV.total_energy_exchanged,
                            (env.current_step-EV.time_of_arrival)
                        ])
                    else:
                        state.append(np.zeros(3))
    
    return np.array(np.hstack(state))
```

---

#### 3. `V2G_profit_max_loads` (V2GProfitPlusLoads)

**State Vector Components:**
```python
state = [
    current_step,                          # 1 feature
    current_power_usage[t-1],              # 1 feature
    charge_prices[t:t+20],                 # 20 features (price forecast)
    loads_minus_pv[t:t+20],                # 20 features (net load forecast)
    power_limits[t:t+20],                  # 20 features (transformer limits)
    # For each charging station:
    [EV_SoC, time_to_departure]            # 2 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 20 + 20 + 20 + 2 \times N_{\text{cs}} = 62 + 2N_{\text{cs}}$

For 25 charging stations: **112 features**

**Code Reference** (lines 108-155):
```python
def V2G_profit_max_loads(env, *args):
    state = [(env.current_step)]
    state.append(env.current_power_usage[env.current_step-1])
    
    charge_prices = abs(env.charge_prices[0, env.current_step:env.current_step+20])
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    state.append(charge_prices)
    
    for tr in env.transformers:
        loads, pv = tr.get_load_pv_forecast(step=env.current_step, horizon=20)
        power_limits = tr.get_power_limits(step=env.current_step, horizon=20)
        state.append(loads-pv)
        state.append(power_limits)
        
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([EV.get_soc(), 
                                     EV.time_of_departure - env.current_step])
                    else:
                        state.append(np.zeros(2))
    
    return np.array(np.hstack(state))
```

### State Space Comparison

| **Feature Category** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|---------------------|------------------|---------------|------------------------|
| **Temporal** | Current step | Normalized step | Current step |
| **Economic** | 20-step price forecast | None | 20-step price forecast |
| **Grid/Load** | None | Power setpoint | Net loads + limits (40 features) |
| **Power** | Current usage | Current usage + setpoint | Current usage |
| **EV Info** | SoC + departure time | Full flag + energy + arrival time | SoC + departure time |
| **State Dim (25 CS)** | 72 | 78* | 112 |
| **Forecast Horizon** | 20 steps (prices) | 0 steps | 20 steps (prices + loads) |

*PublicPST uses 20 charging stations by default

---

## Quick Comparison Summary

| **Aspect** | **V2GProfitMax** | **PublicPST** | **V2GProfitPlusLoads** |
|------------|------------------|---------------|------------------------|
| **Primary Goal** | Profit Maximization | Power Tracking | Multi-Objective Balance |
| **Scenario Type** | Workplace V2G | Public Charging | Workplace V2G + Loads |
| **V2G Support** | ✅ Full bidirectional | ❌ Charge only | ✅ Full bidirectional |
| **External Loads** | ❌ None | ❌ None | ✅ Loads + Solar + DR |
| **Reward Complexity** | Medium (2 terms) | Low (1 term) | High (3 terms) |
| **State Complexity** | Medium (72 dim) | Medium (63 dim) | High (112 dim) |
| **Default Config** | ❌ No | ❌ No | ✅ Yes |
| **Best Use Case** | Pure profit scenarios | Grid services | Real-world deployments |

---

## Usage Examples

### Training with V2GProfitMax
```bash
python train_stable_baselines.py --config_file ev2gym/example_config_files/V2GProfitMax.yaml --algorithm ppo
```

### Training with PublicPST
```bash
python train_stable_baselines.py --config_file ev2gym/example_config_files/PublicPST.yaml --algorithm ppo
```

### Training with V2GProfitPlusLoads (Default)
```bash
python train_stable_baselines.py --algorithm ppo
# or explicitly:
python train_stable_baselines.py --config_file ev2gym/example_config_files/V2GProfitPlusLoads.yaml --algorithm ppo
```

---

## Recommendations

1. **Start with V2GProfitPlusLoads**: It's the default for a reason - it provides the most balanced and realistic training scenario with comprehensive state information and multi-objective rewards.

2. **Use V2GProfitMax**: When you specifically need to maximize revenue in a controlled environment without external load constraints. Ideal for studying pure V2G arbitrage strategies.

3. **Use PublicPST**: When participating in grid services (frequency regulation, demand response) or when accurate power tracking is the primary requirement. Best for ancillary service markets.

4. **Experiment**: Try all three configurations with your specific use case to determine which performs best for your objectives.

---

## Performance Metrics

All configurations track the following metrics during evaluation:
- `total_ev_served` - Number of EVs that received service
- `total_profits` - Net profit/cost from operations
- `total_energy_charged` - Total energy delivered to EVs
- `total_energy_discharged` - Total energy extracted from EVs (V2G)
- `average_user_satisfaction` - Mean satisfaction across all EVs
- `power_tracker_violation` - Violations of power setpoint constraints
- `tracking_error` - Deviation from power setpoints
- `energy_user_satisfaction` - Energy-based satisfaction metric
- `total_transformer_overload` - Total transformer capacity violations
- `reward` - Cumulative episode reward

However, the **importance** and **optimization** of each metric varies by configuration based on the reward function design.

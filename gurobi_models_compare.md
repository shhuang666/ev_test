# EV2Gym Gurobi Models Comparison

This document provides a detailed comparison of the three Gurobi-based optimization models in EV2Gym's baseline algorithms.

---

## Overview Table

| Feature | **profit_max.py** | **tracking_error.py** | **v2g_grid.py** |
|---------|-------------------|----------------------|-----------------|
| **Class Name** | `V2GProfitMaxOracleGB` | `PowerTrackingErrorrMin` | `V2GProfitMax_Grid_OracleGB` |
| **Primary Objective** | Maximize V2G profit | Minimize power tracking error | Maximize profit with grid constraints |
| **Grid Modeling** | ❌ No | ❌ No | ✅ Yes (voltage constraints) |
| **User Satisfaction** | ✅ Yes (penalty term) | ❌ No | ✅ Yes (penalty term) |
| **Power Setpoints** | ❌ Not used | ✅ Required | ✅ Used in grid model |
| **Complexity** | Medium | Low | High |
| **File Size** | 421 lines | 421 lines | 566 lines |

---

## 1. V2GProfitMaxOracleGB (`profit_max.py`)

### **Objective Function**
```python
maximize: costs - 100 * user_satisfaction.sum()
```

Where:
- **costs** = Revenue from charging/discharging based on electricity prices
- **user_satisfaction** = Penalty for not meeting desired energy at departure

### **Key Features**
- ✅ **Profit maximization** through arbitrage (buy low, sell high)
- ✅ **User satisfaction** penalty (100× weight)
- ✅ **V2G bidirectional** charging/discharging
- ✅ **Efficiency modeling** (0.9× multiplier on charger efficiency)
- ❌ **No power setpoint tracking**
- ❌ **No grid voltage constraints**

### **Objective Components**
1. **Revenue from charging**: `charge_current × voltage × efficiency × dt × charge_price`
2. **Revenue from discharging**: `discharge_current × voltage × efficiency × dt × discharge_price`
3. **User satisfaction penalty**: `(desired_energy - actual_energy)²`

### **Use Case**
Best for scenarios where:
- Electricity prices vary significantly over time
- V2G revenue is the primary goal
- Grid stability is not a concern
- Users have flexible departure energy requirements

---

## 2. PowerTrackingErrorrMin (`tracking_error.py`)

### **Objective Function**
```python
minimize: power_error.sum()
```

Where:
- **power_error[t]** = `(total_power[t] - power_setpoint[t])²`

### **Key Features**
- ✅ **Power setpoint tracking** (minimize squared error)
- ✅ **V2G bidirectional** charging/discharging
- ✅ **Transformer constraints** (respects grid limits)
- ❌ **No profit optimization**
- ❌ **No user satisfaction modeling**
- ❌ **No grid voltage constraints**

### **Objective Components**
1. **Tracking error**: Squared difference between actual and target power consumption
2. **Total SOC**: Sum of energy at departure (commented out in objective)

### **Use Case**
Best for scenarios where:
- Following a specific power consumption profile is critical
- Grid operator provides power setpoints
- Demand response programs require precise tracking
- Frequency regulation or load balancing services

---

## 3. V2GProfitMax_Grid_OracleGB (`v2g_grid.py`)

### **Objective Function**
```python
maximize: 1000 * voltage_slack
```

Where:
- **voltage_slack** = Penalties for voltage limit violations

### **Key Features**
- ✅ **Grid voltage modeling** (iterative power flow)
- ✅ **Voltage constraints** (0.95 ≤ V ≤ 1.05 p.u.)
- ✅ **User satisfaction** penalty
- ✅ **V2G bidirectional** charging/discharging
- ✅ **Complex grid physics** (3 iterations of power flow)
- ✅ **Active and reactive power** modeling

### **Grid Model Details**
- **Iterative voltage calculation** (3 iterations)
- **Voltage magnitude constraints** with slack variables
- **Power flow equations**: 
  - Real: `d × L_r = S_r × v_r + S_i × v_i`
  - Imaginary: `d × L_i = -(S_i × v_r - S_r × v_i)`
- **Matrix multiplication**: `Z = K @ L + W`

### **Objective Components**
1. **Voltage slack penalty**: Weighted sum of voltage limit violations
2. **User satisfaction**: (Commented out, but available)
3. **Profit**: (Commented out, but calculated)

### **Use Case**
Best for scenarios where:
- Grid voltage stability is critical
- Distribution network has voltage constraints
- Studying impact of EV charging on grid
- Research on grid-aware charging strategies

---

## Detailed Comparison

### **Decision Variables**

All three models share these common variables:
- `energy[p, i, t]` - EV battery energy
- `current_ev_ch[p, i, t]` - Charging current
- `current_ev_dis[p, i, t]` - Discharging current
- `omega_ch[p, i, t]` - Binary: charging mode
- `omega_dis[p, i, t]` - Binary: discharging mode
- `power_cs_ch[i, t]` - Charging station power (charge)
- `power_cs_dis[i, t]` - Charging station power (discharge)

**Additional in v2g_grid.py:**
- `v_r[it, t, j]` - Voltage real component (per iteration)
- `v_i[it, t, j]` - Voltage imaginary component (per iteration)
- `m_vars[t, j]` - Voltage magnitude
- `slack_low[t, j]`, `slack_high[t, j]` - Voltage violation slack

### **Constraints**

#### Common Constraints (All Models)
1. **Energy dynamics**: `energy[t] = energy[t-1] + charge - discharge`
2. **Energy limits**: `0 ≤ energy ≤ max_energy`
3. **Current limits**: Respect charger and EV limits
4. **Transformer limits**: Circuit breaker constraints
5. **Mutual exclusion**: Cannot charge and discharge simultaneously
6. **Empty port**: No power if EV not present

#### Model-Specific Constraints

**profit_max.py:**
- User satisfaction at departure

**tracking_error.py:**
- Power tracking error calculation

**v2g_grid.py:**
- Voltage magnitude constraints
- Power flow equations
- Grid matrix operations
- Iterative voltage updates

### **Computational Complexity**

| Model | Problem Type | Binary Vars | Continuous Vars | Constraints | Solve Time |
|-------|-------------|-------------|-----------------|-------------|------------|
| **profit_max** | MIQCP | Medium | High | Medium | Fast-Medium |
| **tracking_error** | MIQCP | Medium | High | Medium | Fast-Medium |
| **v2g_grid** | MIQCP | Medium | Very High | Very High | Slow |

*MIQCP = Mixed Integer Quadratically Constrained Program*

### **Parameters**

| Parameter | profit_max | tracking_error | v2g_grid |
|-----------|------------|----------------|----------|
| `timelimit` | ✅ | ❌ | ✅ |
| `MIPGap` | ✅ | ❌ | ✅ |
| `verbose` | ✅ | ❌ | ✅ |
| `replay_path` | ✅ | ✅ | ✅ |

---

## When to Use Each Model

### Use **profit_max.py** when:
- 🎯 Goal: Maximize revenue from V2G operations
- 💰 Variable electricity pricing is available
- 👥 User satisfaction is important
- ⚡ Grid constraints are not critical
- 🚀 Need faster optimization

### Use **tracking_error.py** when:
- 🎯 Goal: Follow specific power consumption profile
- 📊 Power setpoints provided by grid operator
- 🔄 Demand response or frequency regulation
- ⚖️ Load balancing is priority
- 🚀 Need fastest optimization

### Use **v2g_grid.py** when:
- 🎯 Goal: Grid-aware charging with voltage constraints
- 🔌 Distribution network has voltage limits
- 🔬 Research on grid impact of EV charging
- 📈 Need detailed grid modeling
- ⏱️ Can afford longer solve times

---

## Code Example: Using the Models

```python
import pickle
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.v2g_grid import V2GProfitMax_Grid_OracleGB

# 1. Profit Maximization
profit_model = V2GProfitMaxOracleGB(
    replay_path='path/to/replay.pkl',
    timelimit=300,  # 5 minutes
    MIPGap=0.01,    # 1% optimality gap
    verbose=True
)

# 2. Power Tracking
tracking_model = PowerTrackingErrorrMin(
    replay_path='path/to/replay.pkl'
)

# 3. Grid-Aware Profit Max
grid_model = V2GProfitMax_Grid_OracleGB(
    replay_path='path/to/replay.pkl',
    timelimit=600,  # 10 minutes (needs more time)
    MIPGap=0.05,    # 5% gap (more relaxed)
    verbose=True
)

# Get actions for current step
action = profit_model.get_action(env)
```

---

## Performance Considerations

### **Optimization Speed** (Fastest → Slowest)
1. **tracking_error.py** - Simple quadratic objective
2. **profit_max.py** - Quadratic with user satisfaction
3. **v2g_grid.py** - Complex with grid constraints

### **Memory Usage** (Lowest → Highest)
1. **tracking_error.py** - Fewest variables
2. **profit_max.py** - Standard variables
3. **v2g_grid.py** - Many grid-related variables

### **Scalability**
- All models scale with: `n_ports × n_cs × sim_length`
- **v2g_grid** additionally scales with: `n_buses × n_iterations`

---

## Key Differences Summary

| Aspect | profit_max | tracking_error | v2g_grid |
|--------|------------|----------------|----------|
| **Primary Goal** | 💰 Revenue | 🎯 Tracking | 🔌 Grid Health |
| **Optimization Direction** | Maximize | Minimize | Maximize |
| **Grid Modeling** | None | None | Full |
| **User Focus** | High | None | Medium |
| **Complexity** | Medium | Low | High |
| **Real-world Application** | Energy arbitrage | Demand response | Grid planning |

---

## Recommendations

1. **For beginners**: Start with `tracking_error.py` - simplest and fastest
2. **For V2G business**: Use `profit_max.py` - focuses on revenue
3. **For grid research**: Use `v2g_grid.py` - most comprehensive
4. **For production**: Consider combining approaches or using MPC baselines for real-time control

---

## Notes

- All models are **offline/oracle** - they have perfect information about future arrivals/departures
- All models require a **replay file** with simulation data
- All models use **Gurobi** optimizer (commercial license required)
- Set `NonConvex=2` parameter is used for quadratic constraints
- Models output normalized actions in range `[-1, 1]`

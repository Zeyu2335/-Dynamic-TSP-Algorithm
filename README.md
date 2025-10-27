# Simulation of Dynamic Traffic Signal Control with Increased Flow

## Overview
This Python script (`simulation_increasedflow_newframe.py`) simulates an intersection environment with varying traffic flow and signal control strategies. It is designed to evaluate **Transit Signal Priority (TSP)** strategies — including dynamic green reallocation — under different levels of vehicle connectivity and flow conditions. The simulation models both **north–south (NS)** and **east–west (EW)** approaches, incorporating stochastic vehicle arrivals, signal phase dynamics, and queue evolution.

The simulation is part of a larger project analyzing **dynamic TSP control** and **partial connectivity impacts** in mixed traffic scenarios.

## 🎬 Simulation Run
Below is a short demonstration of the **Dynamic TSP Simulation** showing adaptive queue clearance and green time reallocation for buses in mixed traffic.
<video src="https://github.com/user-attachments/assets/228e1da2-ec52-4c51-98af-c4faaf6a1b36" width="640" controls></video>

---

## Key Features

- **Multiple TSP Control Modes**
  - `TSP_rlc`: Dynamic TSP with optional payback and extension limits.
  - `TSP_extn`: Fixed green extension.
  - `NoTSP`: Baseline case without priority control.

- **Dynamic Green Extension**
  - Dynamically extends green phases based on estimated queue length and bus arrival time.
  - Optional payback mechanism in subsequent cycles.

- **Configurable Market Penetration Rate (MPR)**
  - Parameter `con` controls connected vehicle percentage (0–100).

- **Queue Length Simulation**
  - Computes and updates NS and EW queue lengths per frame using `simulate_queue()`.

- **Signal State Generation**
  - Vectorized signal logic for both directions using `get_signal_state_NS1()` and `get_signal_state_EW1()`.

- **Bus Arrival & ETA Prediction**
  - Calculates time-to-green, green-left duration, and estimated arrival times for transit vehicles.

- **Performance Metrics**
  - Outputs include queue length evolution, delay, and potential signal efficiency.

---

## Dependencies

Ensure the following Python libraries are installed:

```bash
pip install numpy pandas matplotlib

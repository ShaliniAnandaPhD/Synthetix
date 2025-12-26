# Fantasy Football Neuron System

[![Neuron Test Suite](https://github.com/ShaliniAnandaPhD/Synthetix/actions/workflows/test.yml/badge.svg)](https://github.com/ShaliniAnandaPhD/Synthetix/actions/workflows/test.yml)

This codebase contains a suite of advanced fantasy football analysis and automation tools. Below is a guide on how to use the key components.

## Quick Test Commands

```bash
# P0 Critical: Validate "within minutes" claim
python tests/e2e/test_full_pipeline_timing.py

# ESPN Integration
python tests/realtime/test_espn_integration.py

# Full backend tests
python tests/automated_backend_tests.py
```

## Prerequisites

Ensure you have Python 3 installed and the `rich` library:

```bash
pip install rich
```

## Key Tools & Demos

### 1. Safety Gateway CLI (`ff_safety_cli.py`)

A command-line interface for assessing risk in fantasy football decisions (trades, waivers, lineup changes).

**Usage:**

```bash
python3 ff_safety_cli.py --scenario [scenario_name]
```

**Available Scenarios:**
- `injury_pivot`: Analyze risks when pivoting from an injured player.
- `trade_analysis`: Evaluate trade proposals.
- `waiver_claim`: (Coming soon) Analyze waiver wire moves.

**Example:**
```bash
python3 ff_safety_cli.py --scenario injury_pivot
```

### 2. Neuron System Enhanced Demo (`ff_neuron_fixed.py`)

The flagship demo of the "Neuron System", featuring advanced game environment modeling, player projections, and strategy generation.

**Usage:**

```bash
python3 ff_neuron_fixed.py
```

This will launch an interactive terminal dashboard showing:
- Data ingestion simulation
- Game script analysis (weather, vegas odds, etc.)
- Player rankings and value scores
- A comprehensive strategy report

### 3. Bankroll Optimizer (`bankroll_demo.py`)

A tool for optimizing your DFS (Daily Fantasy Sports) bankroll allocation using the Kelly Criterion and risk profiling.

**Usage:**

```bash
python3 bankroll_demo.py
```

This runs a live simulation of:
- Fetching contests from DraftKings/FanDuel
- Evaluating contest value (ROI, overlay)
- Allocating funds between GPP and Cash games based on risk tolerance

### 4. Other Demos

- **`weather_demo.py`**: Visualizes weather impacts on games.
- **`injury_pivot_demo.py`**: A dedicated demo for the injury pivot logic.
- **`system_monitoring_demo.py`**: Simulates system health and performance monitoring.
- **`ff_audit_demo.py`** & **`ff_compliance_demo.py`**: Tools for auditing and compliance checking.

## Web Interface

The `WITH ALL APIS` directory contains a web-based frontend. Open `WITH ALL APIS/index.html` in a browser to view the web dashboard.

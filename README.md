# state_estimation
A repository containing implementations of various state estimation algorithms including the Kalman Filter and Interacting Multiple Model Filter.

## Installation

Clone the repo.
```bash
git clone https://github.com/jftaggart02/state_estimation.git
```

Create a virtual environment and install dependencies.
```bash
cd state_estimation
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Demo

Note: ensure the virtual environment is active before running commands:
```
. .venv/bin/activate
```

Dubins path planner:
```bash
python state_estimation/dubins.py
```

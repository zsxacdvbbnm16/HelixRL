# HelixRL

RL workspace for quadrotor simulation and control.

## Setup

```bash
git clone https://github.com/zsxacdvbbnm16/HelixRL
cd HelixRL
```

## Build (C++ env)

```bash
cd flightlib
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Rebuild `flightlib` whenever you change C++ env code.

## PPO_PER

```bash
cd /home/golgapha/Desktop/flightmare
python3 flightrl/examples/ppo2_per.py
```

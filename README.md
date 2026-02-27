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

Rebuild `flightlib` whenever you change C++ env code and dont forget to setup you racing environment in quadrotor_env.cpp before rebuilding flightlib.

## PPO_PER

```bash
cd /home/golgapha/Desktop/flightmare
python3 flightrl/examples/ppo2_per.py
```

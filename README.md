# HelixRL
<img src="1.png" width="400">                   <img src="2.png" width="400">

HelixRL is an extension of the Flightmare simulator, specifically designed for high-speed quadrotor drone racing and advanced control using Reinforcement Learning. 

Key Enhancements-
PPO with Prioritized Experience Replay (PPO_PER): We have integrated Prioritized Experience Replay (PER) into the standard Proximal Policy Optimization (PPO) algorithm to improve sample efficiency and training stability in complex racing environments.
Custom Racing Environments: Includes modified C++ environment code in quadrotor_env.cpp tailored for agile drone racing.
Optimized Performance: Leverages Flightmare's decoupled physics and rendering engines to collect millions of transitions for training in minutes.
RL unity flightmare simulation for quadrotor drone racing and control. This code is the extension of the https://github.com/uzh-rpg/flightmare , we made several changes on the top of flighmare simulator codes.

INSTALLATION
# Clone the repository
```bash
git clone https://github.com/zsxacdvbbnm16/HelixRL

# Set the environment path
export FLIGHTMARE_PATH=$(pwd)

# Build the C++ flightlib
cd flightlib
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Install the python wrapper
cd ../..
pip3 install -e flightlib



cd HelixRL

export FLIGHTMARE_PATH=$(pwd)

cd flightlib
mkdir -p build && cd build
cmake ..

cd ../..
pip3 install -e flightlib

```

Note: Rebuild flightlib whenever you change C++ environment code. Ensure your racing parameters are correctly configured in quadrotor_env.cpp before rebuilding.

## PPO_PER
Training with PPO_PER
To start training with the Prioritized Experience Replay implementation:

```bash
cd /flightmare
python3 flightrl/examples/ppo2_per.py
```
Citation
If you use this simulator or our HelixRL extensions in your research, please cite the original Flightmare paper:

markdown
## Citation

If you use this project, please cite the following paper:

**Song, Y., et al. (2021). Flightmare: A Flexible Quadrotor Simulator.**

```bibtex
@inproceedings{song2021flightmare,
  title={Flightmare: A flexible quadrotor simulator},
  author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
  booktitle={Conference on robot learning},
  pages={1147--1157},
  year={2021},
  organization={PMLR}
}

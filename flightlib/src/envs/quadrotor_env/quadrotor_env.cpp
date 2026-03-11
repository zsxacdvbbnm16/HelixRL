#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    gate_radius_(2.0),
    active_gate_idx_(0),
    passed_gate_(false),
    episode_time_(0.0),
    max_episode_time_(20.0),
    gates_passed_(0),
    gate_reward_(10.0),
    just_passed_gate_(false),
    last_position_(Vector<3>::Zero()),
    helix_radius_(1.0),
    helix_pitch_(1.0),
    helix_angular_velocity_(0.5),
    // Initialize PER parameters
    per_buffer_capacity_(100000),
    per_alpha_(0.6),      // Priority exponent (typically between 0.0 and 1.0)
    per_beta_(0.4),       // Start value for importance sampling (increases to 1.0)
    per_epsilon_(1e-6),   // Small constant to prevent zero priorities
    per_beta_increment_(0.001),  // Beta increment per training step
    // Initialize noise parameters
    action_noise_std_(0.1),
    action_noise_decay_(0.995),
    action_noise_min_(0.01),
    obs_noise_std_(0.005),
    obs_noise_enabled_(true),
    state_noise_std_(0.01),
    position_noise_std_(2.0),
    velocity_noise_std_(0.5),
    orientation_noise_std_(0.1),
    gate_offset_noise_std_(0.2),
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  // Adjusted for forest scene render_offset: [21, 217, 94]
  // X: [1, 41] (centered around x=21, ±20 range)
  // Y: [197, 237] (centered around y=217, ±20 range)
  // Z: [74, 114] (centered around z=94, ±20 range)
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);
  
  // Initialize PER buffer and noise generators with loaded parameters
  per_buffer_ = std::make_unique<PERBuffer<quadenv::kNObs, quadenv::kNAct>>(
    per_buffer_capacity_, per_alpha_, per_beta_, per_epsilon_);
  
  action_noise_ = std::make_unique<ActionNoise>(action_noise_std_);  
  observation_noise_ = std::make_unique<ObservationNoise>(obs_noise_std_);  
  state_noise_ = std::make_unique<StateNoise>(state_noise_std_);  
  
  // load gates from SplitS.yaml
  std::string gates_cfg_path = getenv("FLIGHTMARE_PATH") + 
                               std::string("/flightlib/configs/CircularLoop.yaml");
  loadGatesFromFile(gates_cfg_path);

  // Initialize target state
  target_position_ = Vector<3>::Zero();
  target_orientation_ = Quaternion(1.0, 0.0, 0.0, 0.0);
  target_velocity_ = Vector<3>::Zero();
  target_angular_velocity_ = Vector<3>::Zero();
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();
  
  // Reset episode timer and tracking variables
  episode_time_ = 0.0;
  passed_gate_ = false;
  just_passed_gate_ = false;
  gates_passed_ = 0;
  last_position_ = Vector<3>::Zero();
  
  // Select a random gate if we have gates loaded
  if (!gates_.empty()) {
    if (random) {
      active_gate_idx_ = std::rand() % gates_.size();
    } else {
      active_gate_idx_ = 0;  // Use first gate for deterministic reset
    }
  } else {
    logger_.warn("No gates loaded! Using default position.");
    active_gate_idx_ = -1;
  }
  
  // Get gate position and orientation
  Vector<3> gate_position;
  Quaternion gate_orientation;
  
  if (active_gate_idx_ >= 0) {
    gate_position = gates_[active_gate_idx_]->getPosition();
    gate_orientation = gates_[active_gate_idx_]->getQuaternion();
    
    // Position the quadrotor in front of the gate with better positioning
    Vector<3> gate_forward_dir = gate_orientation * Vector<3>(1, 0, 0);
    
    // Random offset from gate center (but still in front of it)
    Scalar rand_x, rand_y, rand_z;
    if (random) {
      // Position precisely in front of the gate to increase success rate
      rand_x = -0.1;  // Start closer to gate (3m in front) 
      state_noise_->setParameters(0.0, gate_offset_noise_std_);  // Use configured gate offset noise
      rand_y = state_noise_->sample();  // Gaussian noise for lateral offset
      rand_z = state_noise_->sample();  // Gaussian noise for vertical offset
    } else {
      // Fixed position in front of the gate
      rand_x = -0.1;
      rand_y = 0.0;
      rand_z = 0.0;
    }
    
    // Apply random offsets relative to gate orientation
    Vector<3> position_offset = gate_orientation * Vector<3>(rand_x, rand_y, rand_z);
    quad_state_.p = gate_position + position_offset;
    
    // Set the quadrotor to face directly at the gate's center for better alignment
    Vector<3> to_gate = gate_position - quad_state_.p;
    to_gate.normalize();
    
    // Create a quaternion that orients the drone toward the gate
    Vector<3> default_forward(1, 0, 0);
    Vector<3> rotation_axis = default_forward.cross(to_gate);
    if (rotation_axis.norm() > 1e-6) {
      rotation_axis.normalize();
      Scalar rotation_angle = std::acos(default_forward.dot(to_gate));
      quad_state_.q() = Quaternion(Eigen::AngleAxis<Scalar>(rotation_angle, rotation_axis));
    } else {
      quad_state_.q() = Quaternion(1.0, 0.0, 0.0, 0.0); // Default orientation if vectors are parallel
    }
    
    // Set initial velocity (small forward velocity toward gate)
    quad_state_.v = gate_orientation * Vector<3>(0.5, 0, 0);
    
    // Calculate target position behind the gate (in the gate's forward direction)
    Vector<3> behind_gate_offset = gate_orientation * Vector<3>(3.0, 0.0, 0.0);
    target_position_ = gate_position + behind_gate_offset;
    
    // Target hover orientation (facing back toward the gate)
    Vector<3> back_to_gate = gate_position - target_position_;
    back_to_gate.normalize();
    Vector<3> rotation_axis_target = default_forward.cross(back_to_gate);
    if (rotation_axis_target.norm() > 1e-6) {
      rotation_axis_target.normalize();
      Scalar rotation_angle_target = std::acos(default_forward.dot(back_to_gate));
      target_orientation_ = Quaternion(Eigen::AngleAxis<Scalar>(rotation_angle_target, rotation_axis_target));
    } else {
      target_orientation_ = Quaternion(1.0, 0.0, 0.0, 0.0);
    }
    
    // Set target hover state
    target_velocity_ = Vector<3>::Zero();
    target_angular_velocity_ = Vector<3>::Zero();
    
    logger_.info("Quadrotor positioned in front of gate %d", active_gate_idx_);
  } else {
    // Default values if no gates are available
    if (random) {
      // randomly reset the quadrotor state using Gaussian distribution
      state_noise_->setParameters(0.0, position_noise_std_);  // Use configured position noise
      quad_state_.x(QS::POSX) = state_noise_->sample();
      quad_state_.x(QS::POSY) = state_noise_->sample();
      quad_state_.x(QS::POSZ) = std::abs(state_noise_->sample() + 5.0);  // mean=5.0m, ensure positive
      
      // reset linear velocity with smaller standard deviation
      state_noise_->setParameters(0.0, velocity_noise_std_);  // Use configured velocity noise
      quad_state_.x(QS::VELX) = state_noise_->sample();
      quad_state_.x(QS::VELY) = state_noise_->sample();
      quad_state_.x(QS::VELZ) = state_noise_->sample();
      
      // reset orientation with small random perturbations
      state_noise_->setParameters(0.0, orientation_noise_std_);  // Use configured orientation noise
      quad_state_.x(QS::ATTW) = 1.0 + state_noise_->sample();  // close to identity quaternion
      quad_state_.x(QS::ATTX) = state_noise_->sample();
      quad_state_.x(QS::ATTY) = state_noise_->sample();
      quad_state_.x(QS::ATTZ) = state_noise_->sample();
      quad_state_.qx /= quad_state_.qx.norm();
    }
    
    // Default target position
    target_position_ = Vector<3>(0.0, 0.0, 5.0);
    target_orientation_ = Quaternion(1.0, 0.0, 0.0, 0.0);
  }
  
  // reset quadrotor with the calculated states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  
  // Prepare gate position and orientation information
  Vector<3> gate_pos = Vector<3>::Zero();
  Vector<3> gate_ori = Vector<3>::Zero();
  
  if (active_gate_idx_ >= 0 && !gates_.empty()) {
    gate_pos = gates_[active_gate_idx_]->getPosition();
    
    // Convert gate quaternion to euler angles for observation
    gate_ori = gates_[active_gate_idx_]->getQuaternion().toRotationMatrix().eulerAngles(2, 1, 0);
  }
  
  // Combine quadrotor state with gate information
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w, gate_pos, gate_ori;

  // Add observation noise if enabled
  if (obs_noise_enabled_ && observation_noise_) {
    quad_obs_ = observation_noise_->addSensorNoise<quadenv::kNObs>(quad_obs_);
  }

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  // Apply Gaussian exploration noise in normalized action space [-1, 1].
  Vector<quadenv::kNAct> noisy_act = act;
  if (action_noise_ && action_noise_std_ > 0.0) {
    noisy_act = addActionNoise(act, 1.0);
    noisy_act = noisy_act.cwiseMin(Scalar(1.0)).cwiseMax(Scalar(-1.0));
  }

  quad_act_ = noisy_act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // Increment episode time
  episode_time_ += sim_dt_;

  // Store current position before simulation to detect gate passing
  if (last_position_ == Vector<3>::Zero()) {
    last_position_ = quad_state_.p;
  }

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // Check if we passed through a gate
  bool gate_passed = checkGatePass();
  Scalar gate_pass_reward = 0.0;

  // If we passed through a gate, give a large reward and advance to next gate
  if (gate_passed) {
    gate_pass_reward = gate_reward_; // Substantial reward for passing through a gate
    advanceToNextGate(); // Move to the next gate
  }

  // update observations (will include the new active gate)
  getObs(obs);

  // Get target position based on helical path - navigate towards the active gate
  Vector<3> current_target_pos;
  Vector<3> target_vel;
  
  // If we're approaching a gate, set the target to just beyond the gate
  if (active_gate_idx_ >= 0 && !gates_.empty()) {
    Vector<3> gate_pos = gates_[active_gate_idx_]->getPosition();
    Quaternion gate_q = gates_[active_gate_idx_]->getQuaternion();
    
    // Set target to 1m beyond the gate in its forward direction
    Vector<3> gate_forward_offset = gate_q * Vector<3>(1.0, 0.0, 0.0);
    current_target_pos = gate_pos + gate_forward_offset;
    
    // Calculate target velocity - direct approach to the gate
    Vector<3> to_gate = gate_pos - quad_state_.p;
    Scalar distance_to_gate = to_gate.norm();
    
    // Normalize and scale velocity based on distance
    if (distance_to_gate > 0.1) {
      // Use explicit cast to ensure both arguments have the same type (Scalar)
      target_vel = to_gate.normalized() * std::min(Scalar(2.0), distance_to_gate);
    } else {
      target_vel = Vector<3>::Zero(); // Hover if very close
    }
  } else {
    // Fallback to helical path if no gates
    current_target_pos = getHelicalTargetPosition(episode_time_);
    Vector<3> next_pos = getHelicalTargetPosition(episode_time_ + 0.01);
    target_vel = (next_pos - current_target_pos) / 0.01;
  }
  
  // Update target orientation based on velocity
  Quaternion current_target_ori = getHelicalTargetOrientation(current_target_pos, target_vel);
  Vector<3> target_euler = current_target_ori.toRotationMatrix().eulerAngles(2, 1, 0);
  
  // ---------------------- reward function design with focus on gate passing
  // - Small position tracking reward - weight reduced to focus on gate passing
  Scalar pos_reward = -1e-3 * (quad_state_.p - current_target_pos).squaredNorm();
  
  // - Small orientation tracking reward 
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  Scalar ori_reward = -1e-3 * (euler_zyx - target_euler).squaredNorm();
  
  // - Small velocity tracking reward
  Scalar lin_vel_reward = -1e-4 * (quad_state_.v - target_vel).squaredNorm();
  
  // - control action penalty
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();
  
  // Combine reward components - prioritize gate passing reward
  Scalar goal_reward = pos_reward + ori_reward + lin_vel_reward + act_reward;

  // Check terminal conditions
  Scalar terminal_reward = 0.0;
  bool terminal = isTerminalState(terminal_reward);
  
  // Final reward calculation
  Scalar total_reward = goal_reward;
  
  // Add the large gate-passing reward
  total_reward += gate_pass_reward;
  
  if (terminal) {
    // Use terminal reward if episode is done
    total_reward = terminal_reward;
  } else {
    // Small survival reward
    total_reward += 0.05;
  }

  // Anneal action noise automatically (floor at action_noise_min_).
  decayActionNoiseAutomatically();

  return total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  // Check if time is up
  if (episode_time_ >= max_episode_time_) {
    // Give reward based on number of gates passed at the end of episode
    reward = gates_passed_ * 5.0;  // Bonus reward for each gate passed
    logger_.info("Episode ended with %d gates passed", gates_passed_);
    return true;
  }
  
  // Check if we hit the ground - we don't want this
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -5.0;  // Strong negative reward for crashing into the ground
    logger_.info("Quadrotor crashed into the ground");
    return true;
  }
  
  // Check if we're outside the world boundaries - major failure
  Vector<3> pos = quad_state_.p;
  if (pos.x() < world_box_(0, 0) || pos.x() > world_box_(0, 1) ||
      pos.y() < world_box_(1, 0) || pos.y() > world_box_(1, 1) ||
      pos.z() < world_box_(2, 0) || pos.z() > world_box_(2, 1)) {
    reward = -2.0;  // Negative reward for flying out of bounds
    logger_.info("Quadrotor flew out of bounds");
    return true;
  }
  
  // Note: we removed the gate collision check from here
  // We now handle gate interactions in the step function with checkGatePass
  // This allows the quadrotor to recover from near misses without immediate termination
  
  reward = 0.0;
  return false;
}

// PER Buffer methods
void QuadrotorEnv::addExperience(const Vector<quadenv::kNObs>& state,
                                const Vector<quadenv::kNAct>& action,
                                Scalar reward,
                                const Vector<quadenv::kNObs>& next_state,
                                bool done) {
  per_buffer_->add(state, action, reward, next_state, done);
}

std::vector<Experience<quadenv::kNObs, quadenv::kNAct>> 
QuadrotorEnv::sampleBatch(size_t batch_size, std::vector<size_t>& indices, 
                         std::vector<Scalar>& weights) {
  return per_buffer_->sample(batch_size, indices, weights);
}

void QuadrotorEnv::updatePriorities(const std::vector<size_t>& indices, 
                                   const std::vector<Scalar>& td_errors) {
  per_buffer_->update_priorities(indices, td_errors);
}

void QuadrotorEnv::setBeta(const Scalar beta) {
  per_buffer_->set_beta(beta);
}

// Noise methods
Vector<quadenv::kNAct> QuadrotorEnv::addActionNoise(const Vector<quadenv::kNAct>& action, 
                                                   Scalar noise_scale) {
  return action_noise_->addExplorationNoise<quadenv::kNAct>(action, noise_scale);
}

Vector<quadenv::kNObs> QuadrotorEnv::addObservationNoise(const Vector<quadenv::kNObs>& observation) {
  return observation_noise_->addSensorNoise<quadenv::kNObs>(observation);
}

void QuadrotorEnv::setNoiseParameters(Scalar action_noise_std, Scalar obs_noise_std) {
  action_noise_->setStdDev(action_noise_std);
  observation_noise_->setStdDev(obs_noise_std);
}

void QuadrotorEnv::decayActionNoise(Scalar decay_factor) {
  if (action_noise_) {
    action_noise_->decayNoise(decay_factor);
  }
}

void QuadrotorEnv::updatePERParameters(Scalar alpha, Scalar beta, Scalar epsilon) {
  per_alpha_ = alpha;
  per_beta_ = beta;
  per_epsilon_ = epsilon;
  if (per_buffer_) {
    per_buffer_->set_beta(beta);
  }
}

void QuadrotorEnv::updateNoiseParameters(Scalar action_std, Scalar obs_std, Scalar state_std) {
  action_noise_std_ = action_std;
  obs_noise_std_ = obs_std;
  state_noise_std_ = state_std;
  
  if (action_noise_) action_noise_->setStdDev(action_std);
  if (observation_noise_) observation_noise_->setStdDev(obs_std);
  if (state_noise_) state_noise_->setStdDev(state_std);
}

void QuadrotorEnv::incrementPERBeta() {
  per_beta_ = std::min(per_beta_ + per_beta_increment_, Scalar(1.0));
  if (per_buffer_) {
    per_buffer_->set_beta(per_beta_);
  }
}

void QuadrotorEnv::decayActionNoiseAutomatically() {
  action_noise_std_ = std::max(action_noise_std_ * action_noise_decay_, action_noise_min_);
  if (action_noise_) {
    action_noise_->setStdDev(action_noise_std_);
  }
}

Scalar QuadrotorEnv::getPERBeta() const {
  return per_buffer_ ? per_buffer_->get_beta() : per_beta_;
}

size_t QuadrotorEnv::getPERBufferSize() const {
  return per_buffer_ ? per_buffer_->get_size() : 0;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    max_episode_time_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    
    // Load gate task specific parameters if available
    if (cfg["quadrotor_env"]["gate_radius"]) {
      gate_radius_ = cfg["quadrotor_env"]["gate_radius"].as<Scalar>();
    }
    if (cfg["quadrotor_env"]["helix_radius"]) {
      helix_radius_ = cfg["quadrotor_env"]["helix_radius"].as<Scalar>();
    }
    if (cfg["quadrotor_env"]["helix_pitch"]) {
      helix_pitch_ = cfg["quadrotor_env"]["helix_pitch"].as<Scalar>();
    }
    if (cfg["quadrotor_env"]["helix_angular_velocity"]) {
      helix_angular_velocity_ = cfg["quadrotor_env"]["helix_angular_velocity"].as<Scalar>();
    }
    if (cfg["quadrotor_env"]["gate_reward"]) {
      gate_reward_ = cfg["quadrotor_env"]["gate_reward"].as<Scalar>();
    }
  } else {
    return false;
  }

  // Load PER parameters
  if (cfg["per"]) {
    if (cfg["per"]["buffer_capacity"]) {
      per_buffer_capacity_ = cfg["per"]["buffer_capacity"].as<size_t>();
    }
    if (cfg["per"]["alpha"]) {
      per_alpha_ = cfg["per"]["alpha"].as<Scalar>();
    }
    if (cfg["per"]["beta"]) {
      per_beta_ = cfg["per"]["beta"].as<Scalar>();
    }
    if (cfg["per"]["epsilon"]) {
      per_epsilon_ = cfg["per"]["epsilon"].as<Scalar>();
    }
    if (cfg["per"]["beta_increment"]) {
      per_beta_increment_ = cfg["per"]["beta_increment"].as<Scalar>();
    }
  }

  // Load Gaussian noise parameters
  if (cfg["gaussian_noise"]) {
    // Action noise parameters
    if (cfg["gaussian_noise"]["action_noise"]) {
      if (cfg["gaussian_noise"]["action_noise"]["std_dev"]) {
        action_noise_std_ = cfg["gaussian_noise"]["action_noise"]["std_dev"].as<Scalar>();
        if (action_noise_) {
          action_noise_->setStdDev(action_noise_std_);
        }
      }
      if (cfg["gaussian_noise"]["action_noise"]["decay_rate"]) {
        action_noise_decay_ = cfg["gaussian_noise"]["action_noise"]["decay_rate"].as<Scalar>();
      }
      if (cfg["gaussian_noise"]["action_noise"]["min_std_dev"]) {
        action_noise_min_ = cfg["gaussian_noise"]["action_noise"]["min_std_dev"].as<Scalar>();
      }
    }
    
    // Observation noise parameters
    if (cfg["gaussian_noise"]["observation_noise"]) {
      if (cfg["gaussian_noise"]["observation_noise"]["std_dev"]) {
        obs_noise_std_ = cfg["gaussian_noise"]["observation_noise"]["std_dev"].as<Scalar>();
        if (observation_noise_) {
          observation_noise_->setStdDev(obs_noise_std_);
        }
      }
      if (cfg["gaussian_noise"]["observation_noise"]["enabled"]) {
        obs_noise_enabled_ = cfg["gaussian_noise"]["observation_noise"]["enabled"].as<bool>();
      }
    }
    
    // State noise parameters
    if (cfg["gaussian_noise"]["state_noise"]) {
      if (cfg["gaussian_noise"]["state_noise"]["std_dev"]) {
        state_noise_std_ = cfg["gaussian_noise"]["state_noise"]["std_dev"].as<Scalar>();
        if (state_noise_) {
          state_noise_->setStdDev(state_noise_std_);
        }
      }
      if (cfg["gaussian_noise"]["state_noise"]["position_std"]) {
        position_noise_std_ = cfg["gaussian_noise"]["state_noise"]["position_std"].as<Scalar>();
      }
      if (cfg["gaussian_noise"]["state_noise"]["velocity_std"]) {
        velocity_noise_std_ = cfg["gaussian_noise"]["state_noise"]["velocity_std"].as<Scalar>();
      }
      if (cfg["gaussian_noise"]["state_noise"]["orientation_std"]) {
        orientation_noise_std_ = cfg["gaussian_noise"]["state_noise"]["orientation_std"].as<Scalar>();
      }
      if (cfg["gaussian_noise"]["state_noise"]["gate_offset_std"]) {
        gate_offset_noise_std_ = cfg["gaussian_noise"]["state_noise"]["gate_offset_std"].as<Scalar>();
      }
    }
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::updateExtraInfo() {
  extra_info_["gates_passed"] = static_cast<float>(gates_passed_);
  extra_info_["episode_time"] = static_cast<float>(episode_time_);
  extra_info_["active_gate_idx"] = static_cast<float>(active_gate_idx_);
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  
  // Add gates to the Unity environment
  for (auto& gate : gates_) {
    bridge->addStaticObject(gate);
  }
}

bool QuadrotorEnv::loadGatesFromFile(const std::string &gate_cfg_path) {
  try {
    YAML::Node gate_cfg = YAML::LoadFile(gate_cfg_path);
    
    if (!gate_cfg["gates"]) {
      logger_.warn("No gates defined in %s", gate_cfg_path.c_str());
      return false;
    }

    // Clear existing gates
    gates_.clear();

    // Extract the number of gates if available
    int num_gates = gate_cfg["N"] ? gate_cfg["N"].as<int>() : 0;
    
    // Load all gates from the YAML file
    const YAML::Node& gates_node = gate_cfg["gates"];
    int gate_count = 0;
    
    for (auto it = gates_node.begin(); it != gates_node.end(); ++it) {
      const std::string& gate_name = it->first.as<std::string>();
      const YAML::Node& gate_data = it->second;
      
      // Create a new static gate
      std::shared_ptr<StaticGate> gate = std::make_shared<StaticGate>(gate_name);
      
      // Set position
      if (gate_data["position"]) {
        Vector<3> position;
        position << gate_data["position"][0].as<float>(),
                    gate_data["position"][1].as<float>(),
                    gate_data["position"][2].as<float>();
        gate->setPosition(position);
      }
      
      // Set rotation (quaternion)
      if (gate_data["rotation"]) {
        Quaternion quaternion;
        quaternion.w() = gate_data["rotation"][0].as<float>();
        quaternion.x() = gate_data["rotation"][1].as<float>();
        quaternion.y() = gate_data["rotation"][2].as<float>();
        quaternion.z() = gate_data["rotation"][3].as<float>();
        gate->setQuaternion(quaternion);
      }
      
      // Set scale
      if (gate_data["scale"]) {
        Vector<3> scale;
        scale << gate_data["scale"][0].as<float>(),
                 gate_data["scale"][1].as<float>(),
                 gate_data["scale"][2].as<float>();
        gate->setSize(scale);
      }
      
      // Add gate to the list
      gates_.push_back(gate);
      gate_count++;
    }
    
    // Verify the number of gates if specified
    if (num_gates > 0 && gate_count != num_gates) {
      logger_.warn("Number of gates loaded (%d) doesn't match specified count (%d)",
                  gate_count, num_gates);
    }
    
    logger_.info("Successfully loaded %d gates from %s", gate_count, gate_cfg_path.c_str());
    return true;
    
  } catch (const YAML::Exception& e) {
    logger_.error("Error loading gate configuration: %s", e.what());
    return false;
  }
  
  return true;
}

bool QuadrotorEnv::checkCollision() {
  if (active_gate_idx_ < 0 || gates_.empty()) return false;
  
  // Get gate position and orientation
  Vector<3> gate_pos = gates_[active_gate_idx_]->getPosition();
  Quaternion gate_q = gates_[active_gate_idx_]->getQuaternion();
  
  // Get quadrotor position
  Vector<3> quad_pos = quad_state_.p;
  
  // Transform quadrotor position to gate's local frame
  Vector<3> local_pos = gate_q.inverse() * (quad_pos - gate_pos);
  
  // Check if the quadrotor is passing through the gate plane (x=0 in gate's local frame)
  if (std::abs(local_pos.x()) < 0.1) { // Close to the gate plane
    
    // Check if the quadrotor is within the circular gate
    Scalar distance_from_center = std::sqrt(local_pos.y() * local_pos.y() + local_pos.z() * local_pos.z());
    
    // If we're within the gate radius, we've passed through successfully
    if (distance_from_center <= gate_radius_) {
      // Mark as passed if we haven't already
      if (!passed_gate_) {
        passed_gate_ = true;
        logger_.info("Quadrotor passed through the gate successfully!");
      }
      return false; // No collision
    }
    // Otherwise we hit the gate
    else {
      logger_.info("Quadrotor hit the gate!");
      return true; // Collision
    }
  }
  
  return false; // No collision
}

bool QuadrotorEnv::checkGatePass() {
  if (active_gate_idx_ < 0 || gates_.empty()) return false;
  
  // Get gate position and orientation
  Vector<3> gate_pos = gates_[active_gate_idx_]->getPosition();
  Quaternion gate_q = gates_[active_gate_idx_]->getQuaternion();
  
  // Get quadrotor position
  Vector<3> quad_pos = quad_state_.p;
  
  // Transform quadrotor position to gate's local frame
  Vector<3> local_pos = gate_q.inverse() * (quad_pos - gate_pos);
  
  // Previous position in local gate frame
  Vector<3> prev_local_pos = Vector<3>::Zero();
  if (last_position_ != Vector<3>::Zero()) {
    prev_local_pos = gate_q.inverse() * (last_position_ - gate_pos);
  }
  
  // Check if the quadrotor is passing through the gate plane (x=0 in gate's local frame)
  // We detect crossing by checking if the sign of the x-coordinate changed from previous to current position
  bool crossed_plane = false;
  if (last_position_ != Vector<3>::Zero()) {
    crossed_plane = (prev_local_pos.x() <= 0 && local_pos.x() > 0) || 
                   (prev_local_pos.x() >= 0 && local_pos.x() < 0);
  }
  
  // If we crossed the plane, check if we're within the gate radius
  if (crossed_plane) {
    Scalar distance_from_center = std::sqrt(local_pos.y() * local_pos.y() + local_pos.z() * local_pos.z());
    
    // If we're within the gate radius, we've passed through successfully
    if (distance_from_center <= gate_radius_) {
      if (!just_passed_gate_) {
        just_passed_gate_ = true;
        gates_passed_++;
        logger_.info("Quadrotor passed through the gate successfully!");
        return true;
      }
    } else {
      // We hit the gate
      logger_.info("Quadrotor hit the gate!");
    }
  }
  
  // Update last position for next check
  last_position_ = quad_pos;
  return false;
}

void QuadrotorEnv::advanceToNextGate() {
  // Move to the next gate if there are more gates
  if (!gates_.empty()) {
    active_gate_idx_ = (active_gate_idx_ + 1) % gates_.size();
    just_passed_gate_ = false;
    logger_.info("Moving to next gate: %d", active_gate_idx_);
  }
}

Vector<3> QuadrotorEnv::getHelicalTargetPosition(const Scalar t) {
  if (active_gate_idx_ < 0 || gates_.empty()) {
    return Vector<3>(0.0, 0.0, 5.0);
  }
  
  Vector<3> gate_pos = gates_[active_gate_idx_]->getPosition();
  Quaternion gate_q = gates_[active_gate_idx_]->getQuaternion();
  
  Scalar angle = helix_angular_velocity_ * t;
  Vector<3> local_pos;
  
  if (!passed_gate_) {
    // Smooth approach with minimal vertical variation
    local_pos.x() = -4.0 + 3.0 * (t / max_episode_time_);
    local_pos.y() = helix_radius_ * 0.3 * std::sin(angle);  // Reduced amplitude
    local_pos.z() = 0.2 * std::sin(angle * 0.5);           // Very gentle vertical motion
  } else {
    // Stable path after gate
    local_pos.x() = 1.0 + 2.0 * (t / max_episode_time_);
    local_pos.y() = helix_radius_ * 0.3 * std::sin(angle);
    local_pos.z() = 0.5 * t;  // Gentle constant climb instead of oscillation
  }
  
  Vector<3> world_pos = gate_pos + gate_q * local_pos;
  return world_pos;
}

Quaternion QuadrotorEnv::getHelicalTargetOrientation(const Vector<3>& position, const Vector<3>& velocity) {
  if (velocity.norm() < 1e-6) {
    // Default orientation if velocity is too small
    return Quaternion(1.0, 0.0, 0.0, 0.0);
  }
  
  // Direction of travel
  Vector<3> direction = velocity.normalized();
  
  // Default forward direction
  Vector<3> default_forward(1.0, 0.0, 0.0);
  
  // Find the rotation axis and angle
  Vector<3> rotation_axis = default_forward.cross(direction);
  
  if (rotation_axis.norm() < 1e-6) {
    // Vectors are parallel or anti-parallel
    if (default_forward.dot(direction) > 0) {
      // Same direction
      return Quaternion(1.0, 0.0, 0.0, 0.0);
    } else {
      // Opposite direction
      return Quaternion(0.0, 0.0, 1.0, 0.0); // 180 degrees around z-axis
    }
  }
  
  rotation_axis.normalize();
  Scalar rotation_angle = std::acos(default_forward.dot(direction));
  
  // Create quaternion from axis-angle representation
  return Quaternion(Eigen::AngleAxis<Scalar>(rotation_angle, rotation_axis));
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << "]\n"
     << "loaded gates =       [" << quad_env.gates_.size() << "]" << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib

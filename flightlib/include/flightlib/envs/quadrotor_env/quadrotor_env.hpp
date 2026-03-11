#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <random>
#include <memory>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/common/per_buffer.hpp"
#include "flightlib/common/gaussian_noise.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/static_gate.hpp"

namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 3,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kGatePos = 12,
  kNGatePos = 3,
  kGateOri = 15,
  kNGateOri = 3,
  kNObs = 18,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};
class QuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);
  bool loadGatesFromFile(const std::string &gate_cfg_path);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;
  
  // - PER functions
  void addExperience(const Vector<quadenv::kNObs>& state,
                    const Vector<quadenv::kNAct>& action,
                    Scalar reward,
                    const Vector<quadenv::kNObs>& next_state,
                    bool done);
  std::vector<Experience<quadenv::kNObs, quadenv::kNAct>> sampleBatch(size_t batch_size,
                                                                     std::vector<size_t>& indices,
                                                                     std::vector<Scalar>& weights);
  void updatePriorities(const std::vector<size_t>& indices, 
                       const std::vector<Scalar>& td_errors);
  void setBeta(const Scalar beta);
  
  // - Noise functions
  Vector<quadenv::kNAct> addActionNoise(const Vector<quadenv::kNAct>& action, Scalar noise_scale = 1.0);
  Vector<quadenv::kNObs> addObservationNoise(const Vector<quadenv::kNObs>& observation);
  void setNoiseParameters(Scalar action_noise_std, Scalar obs_noise_std);
  void decayActionNoise(Scalar decay_factor);
  
  // - Parameter management
  void updatePERParameters(Scalar alpha, Scalar beta, Scalar epsilon);
  void updateNoiseParameters(Scalar action_std, Scalar obs_std, Scalar state_std);
  void incrementPERBeta();  // Increment beta by beta_increment_
  void decayActionNoiseAutomatically();  // Apply decay_rate to action noise
  Scalar getPERBeta() const;
  size_t getPERBufferSize() const;
  
  // - Parameter getters
  Scalar getActionNoiseStd() const { return action_noise_std_; }
  Scalar getObsNoiseStd() const { return obs_noise_std_; }
  bool isObsNoiseEnabled() const { return obs_noise_enabled_; }
  void setObsNoiseEnabled(bool enabled) { obs_noise_enabled_ = enabled; }

  // - auxiliar functions
  void updateExtraInfo() override;
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  bool checkCollision();
  bool checkGatePass();
  Vector<3> getHelicalTargetPosition(const Scalar t);
  Quaternion getHelicalTargetOrientation(const Vector<3>& position, const Vector<3>& velocity);
  void advanceToNextGate();

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QuadrotorEnv"};

  // Define reward for training
  Scalar pos_coeff_, ori_coeff_, lin_vel_coeff_, ang_vel_coeff_, act_coeff_;
  
  // Gate task specific parameters
  Scalar gate_radius_;       // radius of the gate circle
  int active_gate_idx_;      // currently active gate
  bool passed_gate_;         // true if passed through the active gate
  Scalar episode_time_;      // current episode time
  Scalar max_episode_time_;  // maximum episode time
  int gates_passed_;         // number of gates passed in the episode
  Scalar gate_reward_;       // reward for passing through a gate
  bool just_passed_gate_;    // flag to track if gate was just passed
  Vector<3> last_position_;  // last position of the quadrotor to detect passing through gate

  // Target state for hovering behind the gate
  Vector<3> target_position_;
  Quaternion target_orientation_;
  Vector<3> target_velocity_;
  Vector<3> target_angular_velocity_;
  
  // Helical path parameters
  Scalar helix_radius_;
  Scalar helix_pitch_;
  Scalar helix_angular_velocity_;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;

  // PER buffer and noise generators
  std::unique_ptr<PERBuffer<quadenv::kNObs, quadenv::kNAct>> per_buffer_;
  std::unique_ptr<ActionNoise> action_noise_;
  std::unique_ptr<ObservationNoise> observation_noise_;
  std::unique_ptr<StateNoise> state_noise_;
  
  // PER parameters
  size_t per_buffer_capacity_;
  Scalar per_alpha_;
  Scalar per_beta_;
  Scalar per_epsilon_;
  Scalar per_beta_increment_;  // How much to increase beta per training step
  
  // Noise parameters
  Scalar action_noise_std_;
  Scalar action_noise_decay_;
  Scalar action_noise_min_;
  Scalar obs_noise_std_;
  bool obs_noise_enabled_;
  Scalar state_noise_std_;
  Scalar position_noise_std_;
  Scalar velocity_noise_std_;
  Scalar orientation_noise_std_;
  Scalar gate_offset_noise_std_;

  // reward function design (for model-free reinforcement learning)
  Vector<quadenv::kNObs> goal_state_;

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  // gates
  std::vector<std::shared_ptr<StaticGate>> gates_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib

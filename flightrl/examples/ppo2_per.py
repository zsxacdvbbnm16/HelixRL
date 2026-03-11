#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from stable_baselines.common import tf_util
from stable_baselines.ppo2.ppo2 import PPO2

class PPO2withPER(PPO2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _setup_learn(self):
        super()._setup_learn()
        # For PER importance sampling
        self.weights_ph = tf.placeholder(tf.float32, [None], 'weights')
        
        # Weight the policy gradient and value losses
        self.pg_loss *= self.weights_ph
        self.vf_loss *= self.weights_ph
        
    def _get_per_batch(self):
        """Sample a batch using PER"""
        indices = []
        weights = []
        batch = self.env.env_method(
            "sampleBatch",
            self.n_batch // self.nminibatches,
            indices, weights
        )[0]
        
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, 
                   actions, values, neglogpacs, update):
        """Single PPO optimization step using PER"""
        # Sample batch with priorities
        states, acts, rews, next_states, dones, indices, weights = self._get_per_batch()
        
        # Compute advantages (weighted by PER)
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = advs * weights[:, None]  # Apply PER weights
        
        td_map = {
            self.train_model.obs_ph: states,
            self.action_ph: acts,
            self.advs_ph: advs,
            self.rewards_ph: returns,
            self.learning_rate_ph: learning_rate,
            self.clip_range_ph: cliprange,
            self.weights_ph: weights
        }

        if self.using_gae:
            td_map[self.value_target_ph] = returns

        # Run PPO update
        policy_loss, value_loss, policy_entropy = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy],
            td_map
        )

        # Update priorities based on value loss
        td_errors = np.abs(value_loss)
        self.env.env_method(
            "updatePriorities",
            indices,
            td_errors
        )

        return policy_loss, value_loss, policy_entropy

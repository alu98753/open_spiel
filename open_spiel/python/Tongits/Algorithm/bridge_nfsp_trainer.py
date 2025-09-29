# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""NFSP agents trained on Bridge (no double dummy)."""

from absl import app, flags, logging
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import nfsp
import numpy as np
import pyspiel
import os
FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_integer("num_train_deals", int(1e5),
                     "Number of training deals (episodes).")
flags.DEFINE_integer("eval_every", 1000,
                     "Deal frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [256, 256],
                  "Hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_string("checkpoint_dir", f"", "模型存檔路徑")
flags.DEFINE_integer("save_every", 10000, "每隔多少牌局存一次 checkpoint")


class NFSPPolicies(policy.Policy):
    """Joint policy wrapper for evaluation."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = list(range(game.num_players()))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {
            "info_state": [None for _ in player_ids],
            "legal_actions": [None for _ in player_ids]
        }

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        time_step = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(time_step, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def main(unused_argv):
    game = "bridge(use_double_dummy_result=true)"
    num_players = 4

    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.num_train_deals,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    # 建立四個 NFSP agent
    agents = [
        nfsp.NFSP(
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes,
            FLAGS.reservoir_buffer_capacity,
            FLAGS.anticipatory_param,
            **kwargs
        )
        for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    for ep in range(FLAGS.num_train_deals):
        if (ep + 1) % FLAGS.eval_every == 0:
            losses = [agent.loss for agent in agents]
            logging.info(f"ep:{ep},Losses: %s", losses)
            # exploitability 在 bridge 上計算成本高，這裡示範仍呼叫
            try:
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
            except Exception as e:
                logging.info("Exploitability not supported for bridge: %s", e)
            logging.info("_____________________________________________")

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            legal_actions = time_step.observations["legal_actions"][player_id]
            if not legal_actions:  # 理論上不會，但保險
                legal_actions = env.get_state.legal_actions()
            action = np.random.choice(legal_actions)

            time_step = env.step([action])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


        if FLAGS.checkpoint_dir and (ep + 1) % FLAGS.save_every == 0:
            for i, agent in enumerate(agents):
                agent.save(os.path.join(FLAGS.checkpoint_dir, f"agent{i}_{ep}"))



if __name__ == "__main__":
    app.run(main)

# Copyright (c) 2019 Eric Steinberger


import numpy as np
import sys

from PokerRL.cfr._MCCFRBase import MCCFRBase as _MCCFRBase


class MCCFR(_MCCFRBase):

    def __init__(self,
                 name,
                 chief_handle,
                 game_cls,
                 agent_bet_set,
                 starting_stack_sizes=None,
                 innerloop_epi=None):
        super().__init__(name=name,
                         chief_handle=chief_handle,
                         game_cls=game_cls,
                         starting_stack_sizes=starting_stack_sizes,
                         agent_bet_set=agent_bet_set,
                         algo_name="MCCFR",
                         innerloop_epi=innerloop_epi
                         )

        # print("stack size", self._starting_stack_sizes)
        self.reset()

    def _regret_formula_after_first_it(self, ev_all_actions, strat_ev, last_regrets):
        return ev_all_actions - strat_ev + last_regrets

    def _regret_formula_first_it(self, ev_all_actions, strat_ev):
        return ev_all_actions - strat_ev

    def _compute_new_strategy(self, p_id):
        for t_idx in range(len(self._trees)):
            def _fill(_node):
                if _node.p_id_acting_next == p_id:
                    N = len(_node.children)

                    _capped_reg = np.maximum(_node.data["regret"], 0)
                    _reg_pos_sum = np.expand_dims(np.sum(_capped_reg, axis=1), axis=1).repeat(N, axis=1)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        _node.strategy = np.where(
                            _reg_pos_sum > 0.0,
                            _capped_reg / _reg_pos_sum,
                            np.full(shape=(self._env_bldrs[t_idx].rules.RANGE_SIZE, N,), fill_value=1.0 / N,
                                    dtype=np.float32)
                        )

                for c in _node.children:
                    _fill(c)

            _fill(self._trees[t_idx].root)

    def _add_strategy_to_average(self, p_id):
        def _fill(_node):
            if _node.p_id_acting_next == p_id:
                contrib = _node.strategy * np.expand_dims(_node.reach_probs[p_id], axis=1)
                if self._iter_counter > 0:
                    _node.data["avg_strat_sum"] += contrib
                else:
                    _node.data["avg_strat_sum"] = contrib

                _s = np.expand_dims(np.sum(_node.data["avg_strat_sum"], axis=1), axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    _node.data["avg_strat"] = np.where(_s == 0,
                                                       np.full(shape=len(_node.allowed_actions),
                                                               fill_value=1.0 / len(_node.allowed_actions)),
                                                       _node.data["avg_strat_sum"] / _s
                                                       )
                assert np.allclose(np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001)

            for c in _node.children:
                _fill(c)

        for t_idx in range(len(self._trees)):
            _fill(self._trees[t_idx].root)

    def _add_mc_strategy_to_average(self, p_id):
        def _fill(_node, weight):
            for i in range(self._trees[t_idx]._env_bldr.rules.RANGE_SIZE):
                if _node.reach_probs[p_id, i] == 0:
                    weight[i] = 0.0
            if _node.p_id_acting_next == p_id:
                contrib = _node.strategy * np.expand_dims(weight, axis=1)
                if self._iter_counter > 0:
                    _node.data["avg_strat_sum"] += contrib
                else:
                    _node.data["avg_strat_sum"] = contrib

                _s = np.expand_dims(np.sum(_node.data["avg_strat_sum"], axis=1), axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    _node.data["avg_strat"] = np.where(_s == 0,
                                                       np.full(shape=len(_node.allowed_actions),
                                                               fill_value=1.0 / len(_node.allowed_actions)),
                                                       _node.data["avg_strat_sum"] / _s
                                                       )
                assert np.allclose(np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001)

            for c in _node.children:
                if _node.p_id_acting_next == p_id:
                    a_idx = _node.allowed_actions.index(c.action)
                    new_weight = weight * _node.strategy[:, a_idx]
                else:
                    new_weight = np.copy(weight)
                _fill(c, new_weight)

        for t_idx in range(len(self._trees)):
            _fill(self._trees[t_idx].root, np.ones(self._trees[t_idx]._env_bldr.rules.RANGE_SIZE))

    def iteration(self):
        cur_nodes = self.touching_nodes
        for i in range(self._innerloop_epi):
            for p in range(self._n_seats):
                self._trees[0]._value_filler.count = 1
                self._generate_samples(p_id=p, opponent=True)
                self._compute_mc_cfv(p_id=p) 
                self._compute_regrets(p_id=p)

                self._update_reach_probs()
                variance = self._calcultate_variance()
                print("variance",variance)

                self._compute_new_strategy(p_id=p)
                self._add_mc_strategy_to_average(p_id=p)
            print("nodes", self.touching_nodes - cur_nodes)
            cur_nodes = self.touching_nodes
        self._iter_counter += 1

        # self._update_reach_probs()
        # self._compute_cfv()
        # self._log_curr_strat_expl()
        expl = self._evaluate_avg_strats()
        return expl

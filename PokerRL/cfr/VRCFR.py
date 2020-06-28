# Copyright (c) 2019 Eric Steinberger


import numpy as np
import sys

from PokerRL.cfr._MCCFRBase import MCCFRBase as _VRCFRBase


class VRCFR(_VRCFRBase):

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
                         algo_name="VRCFR",
                         innerloop_epi=innerloop_epi,
                         sample_method='vr'
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

    def _update_V_and_M(self, p_id):
        for t_idx in range(len(self._trees)):
            def _fill(_node):
                if self.iteration == 4:
                    print("pid", p_id)
                _node.stg_diff = np.zeros(self._env_bldrs[t_idx].rules.RANGE_SIZE)
                _node.V_value[p_id] = 0.0#np.zeros(self._env_bldrs[t_idx].rules.RANGE_SIZE)
                _node.M_value[p_id] = 0.0#np.zeros(self._env_bldrs[t_idx].rules.RANGE_SIZE)
                if self.iteration == 4:
                    print("V", _node.V_value)
                    print("M", _node.M_value)
                    sys.exit()
                if _node.p_id_acting_next == p_id:
                    for i in range(self._env_bldrs[t_idx].rules.RANGE_SIZE):
                        for a in range(len(_node.strategy[i])):
                            current_m = abs(_node.strategy[i,a] - _node.ref_strategy[i, a])
                            _node.M_value[p_id, i] = max(_node.M_value[p_id, i], current_m)
                            _node.stg_diff[i] += current_m

                for c in _node.children:
                    _fill(c)
                    child_V = c.V_value[p_id]
                    if _node.p_id_acting_next == p_id:
                        a_idx = _node.allowed_actions.index(c.action)
                        child_V += c.stg_diff
                        child_V *= _node.strategy[:, a_idx]
                    _node.V_value[p_id] = np.maximum(_node.V_value[p_id], child_V)
                    _node.M_value[p_id] = np.maximum(_node.M_value[p_id], c.M_value[p_id])

            _fill(self._trees[t_idx].root)        

    def _compute_vr_cfv(self, p_id):
        # Compute node.ev_weighted, node.ev_br_weighted, node.epsilon, node.exploitability
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].compute_vr_ev(p_id)

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

    def _add_strategy_to_inner_average(self, p_id):
        def _inner_fill(_node, weight):
            for i in range(self._trees[t_idx]._env_bldr.rules.RANGE_SIZE):
                if _node.reach_probs[p_id, i] == 0:
                    weight[i] = 0.0
            if _node.p_id_acting_next == p_id:
                contrib = _node.strategy * np.expand_dims(weight, axis=1)
                if "inner_avg_strat_sum" not in _node.data:
                    raise ValueError
                _node.data["inner_avg_strat_sum"] += contrib

                _s = np.expand_dims(np.sum(_node.data["inner_avg_strat_sum"], axis=1), axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    _node.data["inner_avg_strat"] = np.where(_s == 0,
                                                        np.full(shape=len(_node.allowed_actions),
                                                                fill_value=1.0 / len(_node.allowed_actions)),
                                                        _node.data["inner_avg_strat_sum"] / _s
                                                        )
                assert np.allclose(np.sum(_node.data["inner_avg_strat"], axis=1), 1, atol=0.0001)

            for c in _node.children:
                if _node.p_id_acting_next == p_id:
                    a_idx = _node.allowed_actions.index(c.action)
                    new_weight = weight * _node.strategy[:, a_idx]
                else:
                    new_weight = np.copy(weight)
                _inner_fill(c, new_weight)

        for t_idx in range(len(self._trees)):
            _inner_fill(self._trees[t_idx].root, np.ones(self._trees[t_idx]._env_bldr.rules.RANGE_SIZE))

    def _update_outer_stgy(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].update_outer_stgy()

    def _update_refer_info(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].update_refer_info()

    def iteration(self):
        cur_nodes = self.touching_nodes
        for p in range(self._n_seats):
            self._update_reach_probs()
            self._compute_cfv()
            self._compute_regrets(p_id=p)
            self._compute_new_strategy(p_id=p)
        self._update_reach_probs()
        self._update_refer_info()
        for p in range(self._n_seats):
            self._add_strategy_to_inner_average(p_id=p)
        print("outer nodes", self.touching_nodes-cur_nodes)
        cur_nodes = self.touching_nodes
        for i in range(self._innerloop_epi):
            print("inner epi", i)
            for p in range(self._n_seats):
                self._generate_samples(p_id=p, opponent=True)    
                self._compute_vr_cfv(p_id=p)
                self._compute_regrets(p_id=p)

                # self._update_reach_probs()
                # variance = self._calcultate_variance()
                # print("variance",variance)

                self._compute_new_strategy(p_id=p)
                self._update_V_and_M(p_id=p)
                self._add_strategy_to_inner_average(p_id=p)
            print("inner nodes", self.touching_nodes-cur_nodes)
            cur_nodes = self.touching_nodes
        self._update_outer_stgy()
        self._update_reach_probs()
        for p in range(self._n_seats):
            self._add_strategy_to_average(p_id=p)

        self._iter_counter += 1
        expl = self._evaluate_avg_strats()
        return expl
        # print("tree visited", self._trees[0].root.visited)

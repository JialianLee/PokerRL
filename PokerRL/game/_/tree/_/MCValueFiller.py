# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.tree._.nodes import PlayerActionNode

import sys


class MCValueFiller:

    def __init__(self, tree):
        self._tree = tree
        self._env_bldr = tree.env_bldr
        self._env = self._env_bldr.get_new_env(is_evaluating=True)
        self.count = 0

        # This only works for 1-Card games!
        self._eq_const = (self._env_bldr.rules.N_CARDS_IN_DECK / (self._env_bldr.rules.N_CARDS_IN_DECK - 1))

    def compute_cf_values_heads_up(self, node):
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            # equity.shape = (state_size, action_size)

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node)
                else:
                    equity = self._get_fold_eq_preflop(node=node)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2 # shape = (2,6
            node.ev_br = np.copy(node.ev)

        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            ev_br_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                         dtype=np.float32)

            for i, child in enumerate(node.children):
                self.compute_cf_values_heads_up(node=child)
                ev_all_actions[i] = child.ev
                ev_br_all_actions[i] = child.ev_br

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ev = np.sum(ev_all_actions, axis=0)
                node.ev_br = np.sum(ev_br_all_actions, axis=0)

            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
                node.ev_br = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ev[opp] = np.sum(ev_all_actions[:, opp], axis=0) 

                node.ev_br[opp] = np.sum(ev_br_all_actions[:, opp], axis=0)
                node.ev_br[plyr] = np.max(ev_br_all_actions[:, plyr], axis=0)

                node.br_a_idx_in_child_arr_for_each_hand = np.argmax(ev_br_all_actions[:, plyr], axis=0)

        # weight ev by reach prob
        node.ev_weighted = node.ev * node.reach_probs
        node.ev_br_weighted = node.ev_br * node.reach_probs
        assert np.allclose(np.sum(node.ev_weighted), 0, atol=0.001), np.sum(node.ev_weighted)  # Zero Sum check

        node.epsilon = node.ev_br_weighted - node.ev_weighted
        node.exploitability = np.sum(node.epsilon, axis=1)

    def compute_vr_cf_values_heads_up(self, node, p_id): 
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            sample_probs = node.sample_reach_probs
            inv_sample_reach_probs = np.zeros(shape=sample_probs.shape)
            non_zero_idx = np.where(sample_probs > 0)
            inv_sample_reach_probs[non_zero_idx] = 1.0 / sample_probs[non_zero_idx]

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node, inv_p=inv_sample_reach_probs)
                    ref_equity = self._get_fold_eq_final_street(node=node, 
                                                                reach_probs=node.ref_reach_probs,
                                                                inv_p=inv_sample_reach_probs)
                else:
                    equity = self._get_fold_eq_preflop(node=node, inv_p=inv_sample_reach_probs)
                    ref_equity = self._get_fold_eq_preflop(node=node, reach_probs=node.ref_reach_probs,
                                                            inv_p=inv_sample_reach_probs)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d],
                                                            inv_p=inv_sample_reach_probs)
                    ref_equity = self._get_call_eq_final_street(reach_probs=node.ref_reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d],
                                                            inv_p=inv_sample_reach_probs)
                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node, inv_p=inv_sample_reach_probs)
                    ref_equity = self._get_call_eq_preflop(node=node, reach_probs=node.ref_reach_probs,
                                                            inv_p=inv_sample_reach_probs)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0
                    ref_equity[:, c] = 0.0

            node.cur_ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2
            node.cur_ref_ev = ref_equity * node.env_state[EnvDictIdxs.main_pot] / 2

            node.ev = node.ref_ev + (node.cur_ev - node.cur_ref_ev) # shape = (2,6
            # return np.min(node.sample_reach_probs[np.where(node.sample_reach_probs > 0)])
        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            ref_ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                          dtype=np.float32)
            # min_p = 1.0
            for i, child in enumerate(node.children):
                self.compute_vr_cf_values_heads_up(node=child, p_id=p_id)
                # min_p = min(min_p, sp)
                ev_all_actions[i] = child.cur_ev
                ref_ev_all_actions[i] = child.cur_ref_ev

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.cur_ev = np.sum(ev_all_actions, axis=0)
                node.cur_ref_ev = np.sum(ref_ev_all_actions, axis=0)

            else:
                node.cur_ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
                node.cur_ref_ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.cur_ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.cur_ev[opp] = np.sum(ev_all_actions[:, opp], axis=0) 
                node.cur_ref_ev[plyr] = np.sum(node.ref_strategy.T * ref_ev_all_actions[:, plyr], axis=0)
                node.cur_ref_ev[opp] = np.sum(ref_ev_all_actions[:, opp], axis=0) 
            node.ev = node.ref_ev + (node.cur_ev - node.cur_ref_ev)
            # return min_p

    def compute_mc_cf_values_heads_up(self, node, p_id): 
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        assert self._tree.n_seats == 2
        range_size = self._env_bldr.rules.RANGE_SIZE

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (range_size, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            # equity.shape = (state_size, action_size)

            # sample_probs = np.dot(node.sample_reach_probs[0].reshape(range_size, 1),
            #                     node.sample_reach_probs[1].reshape(1, range_size))
            sample_probs = node.sample_reach_probs
            inv_sample_reach_probs = np.zeros(shape=sample_probs.shape)
            non_zero_idx = np.where(sample_probs > 0)
            inv_sample_reach_probs[non_zero_idx] = 1.0 / sample_probs[non_zero_idx]

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node, inv_p=inv_sample_reach_probs)
                else:
                    equity = self._get_fold_eq_preflop(node=node, inv_p=inv_sample_reach_probs)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d],
                                                            inv_p=inv_sample_reach_probs)

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node, inv_p=inv_sample_reach_probs)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2 # shape = (2,6
            # return np.min(node.sample_reach_probs[np.where(node.sample_reach_probs > 0)])
        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            # min_p = 1
            for i, child in enumerate(node.children):
                self.compute_mc_cf_values_heads_up(node=child, p_id=p_id)
                # min_p = min(sp, min_p)
                ev_all_actions[i] = child.ev

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ev = np.sum(ev_all_actions, axis=0)

            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ev[opp] = np.sum(ev_all_actions[:, opp], axis=0) 
            # return min_p

    def calcultate_variance(self, node):
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            # equity.shape = (state_size, action_size)

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node)
                else:
                    equity = self._get_fold_eq_preflop(node=node)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.true_ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2 # shape = (2,6
        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)

            for i, child in enumerate(node.children):
                child_v = self.calcultate_variance(node=child)
                ev_all_actions[i] = child.true_ev

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.true_ev = np.sum(ev_all_actions, axis=0)

            else:
                node.true_ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.true_ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.true_ev[opp] = np.sum(ev_all_actions[:, opp], axis=0) 


    def update_outer_stgy(self, node):
        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if isinstance(node, PlayerActionNode) and (not node.is_terminal) \
                and node.p_id_acting_next != self._tree.CHANCE_ID:
            if "inner_avg_strat" in node.data:
                node.strategy = node.data["inner_avg_strat"]

        for c in node.children:
            self.update_outer_stgy(c)

    def update_refer_info(self, node):
        # We need to calculate the cfv
        assert self._tree.n_seats == 2

        node.ref_strategy = np.copy(node.strategy)
        node.ref_reach_probs = np.copy(node.reach_probs)
        if node.data is not None:
            node.data["inner_avg_strat_sum"] = 0.0
            node.data["inner_avg_strat"] = 0.0

        node.V_value = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE))
        node.M_value = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE))
        node.stg_diff = np.zeros(self._env_bldr.rules.RANGE_SIZE)

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            # equity.shape = (state_size, action_size)

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node)
                else:
                    equity = self._get_fold_eq_preflop(node=node)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.ref_ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2 # shape = (2,6

        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)

            for i, child in enumerate(node.children):
                self.update_refer_info(node=child)
                ev_all_actions[i] = child.ref_ev

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ref_ev = np.sum(ev_all_actions, axis=0)

            else:
                node.ref_ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next


                node.ref_ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ref_ev[opp] = np.sum(ev_all_actions[:, opp], axis=0)


    def _get_fold_eq_preflop(self, node, reach_probs=None, inv_p=None):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
        if reach_probs is None:
            reach_probs = node.reach_probs
        if inv_p is None:
            inv_p = np.ones(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE))
        
        for p in range(self._tree.n_seats):
            opp = 1 - p
            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            # equity[p] = np.sum(reach_probs[opp]) - reach_probs[opp]
            equity[p] = (np.sum(reach_probs[opp] * inv_p[opp]) - reach_probs[opp] * inv_p[opp]) * inv_p[p]
        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_fold_eq_final_street(self, node, reach_probs=None, inv_p=None):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        if reach_probs is None:
            reach_probs = node.reach_probs
        if inv_p is None:
            inv_p = np.ones(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE))
        
        for p in range(self._tree.n_seats):
            opp = 1 - p
            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            # equity[p] = np.sum(reach_probs[opp]) - reach_probs[opp]
            equity[p] = (np.sum(reach_probs[opp] * inv_p[opp]) - reach_probs[opp] * inv_p[opp]) * inv_p[p]
        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_call_eq_final_street(self, reach_probs, board_2d, inv_p=None):
        """
        Returns:
            equity: negative=lose. positive=win. 0=50%/50%

        """
        c = self._env_bldr.lut_holder.get_1d_cards(board_2d)[0]

        assert c != Poker.CARD_NOT_DEALT_TOKEN_1D

        if reach_probs is None:
            reach_probs = node.reach_probs
        if inv_p is None:
            inv_p = np.ones(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE))

        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
        handranks = np.empty(shape=self._env_bldr.rules.RANGE_SIZE, dtype=np.int32)

        for h in range(self._env_bldr.rules.RANGE_SIZE):
            handranks[h] = self._env.get_hand_rank(
                board_2d=board_2d,
                hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))

        for p in range(self._tree.n_seats):
            opp = 1 - p
            for h in range(self._env_bldr.rules.RANGE_SIZE):
                if h != c:
                    for h_opp in range(self._env_bldr.rules.RANGE_SIZE):
                        if h_opp != h and h_opp != c:
                            # when same handrank, would be += 0
                            if handranks[h] > handranks[h_opp]:
                                equity[p, h] += reach_probs[opp, h_opp] * inv_p[p, h] * inv_p[opp, h_opp]
                            elif handranks[h] < handranks[h_opp]:
                                equity[p, h] -= reach_probs[opp, h_opp] * inv_p[p, h] * inv_p[opp, h_opp]
        assert np.allclose(equity[:, c], 0)
        return equity * self._eq_const

    def _get_call_eq_preflop(self, node, reach_probs=None, inv_p=None):
        # very Leduc specific
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for c in range(self._env_bldr.rules.N_CARDS_IN_DECK):
            """ ._get_call_eq() returns 0 for blocked hands, so we are summing 5 hands for each board. """
            _board_1d = np.array([c], dtype=np.int8)
            _board_2d = self._env_bldr.lut_holder.get_2d_cards(_board_1d)
            if reach_probs is None:
                _reach_probs = np.copy(node.reach_probs)
            else:
                _reach_probs = np.copy(reach_probs)
            _reach_probs[:, c] = 0

            equity += self._get_call_eq_final_street(reach_probs=_reach_probs, board_2d=_board_2d, inv_p=inv_p)

        # mean :: using (N_CARDS_IN_DECK - 2) because two boards are blocked (by agent's and opp's cards)
        equity /= (self._env_bldr.rules.N_CARDS_IN_DECK - 2)
        return equity # * self._eq_const

    # def _get_call_utility_final_street(self, board_2d, p_id):
    #     """
    #     Returns:
    #         equity: negative=lose. positive=win. 0=50%/50%

    #     """
    #     c = self._env_bldr.lut_holder.get_1d_cards(board_2d)[0]

    #     assert c != Poker.CARD_NOT_DEALT_TOKEN_1D

    #     range_size = self._env_bldr.rules.RANGE_SIZE
    #     utility = np.zeros(shape=(range_size, range_size), dtype=np.float32)
    #     handranks = np.empty(shape=range_size, dtype=np.int32)

    #     for h in range(range_size):
    #         handranks[h] = self._env.get_hand_rank(
    #             board_2d=board_2d,
    #             hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))

    #     for h in range(range_size):
    #         if h != c:
    #             for h_opp in range(range_size):
    #                 if h_opp != h and h_opp != c:
    #                     # when same handrank, would be += 0
    #                     if handranks[h] > handranks[h_opp]:
    #                         utility[h, h_opp] = 1
    #                     elif handranks[h] < handranks[h_opp]:
    #                         utility[h, h_opp] = -1
    #     return utility

    # def _get_call_utility_preflop(self, node, p_id):
    #     # very Leduc specific
    #     range_size = self._env_bldr.rules.RANGE_SIZE
    #     utility = np.zeros(shape=(range_size, range_size), dtype=np.float32)

    #     for c in range(self._env_bldr.rules.N_CARDS_IN_DECK):
    #         """ ._get_call_eq() returns 0 for blocked hands, so we are summing 5 hands for each board. """
    #         _board_1d = np.array([c], dtype=np.int8)
    #         _board_2d = self._env_bldr.lut_holder.get_2d_cards(_board_1d)
    #         # _reach_probs = np.copy(node.reach_probs)
    #         # _reach_probs[:, c] = 0

    #         utility += self._get_call_utility_final_street(board_2d=_board_2d, p_id=p_id)

    #     # mean :: using (N_CARDS_IN_DECK - 2) because two boards are blocked (by agent's and opp's cards)
    #     utility /= (self._env_bldr.rules.N_CARDS_IN_DECK - 2)
    #     return utility

    # def compute_vr_cf_values_heads_up_copy(self, node, p_id): 
    #     """
    #     The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
    #     Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
    #     """
    #     assert self._tree.n_seats == 2
    #     range_size = self._env_bldr.rules.RANGE_SIZE

    #     if node.is_terminal:
    #         assert node.strategy is None
    #     else:
    #         assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

    #     if node.is_terminal:
    #         """
    #         equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
    #         """
    #         assert isinstance(node, PlayerActionNode)

    #         # utility.shape = (state_size, state_size)
    #         # The utility only for player p_id. 

    #         # Fold
    #         if node.action == Poker.FOLD:
    #             utility = np.ones(shape=(range_size, range_size), dtype=np.float32)

    #             for i in range(range_size):
    #                 utility[i, i] = 0.0
    #             if node.p_id_acted_last == p_id:
    #                 utility *= -1

    #         # Check / Call
    #         else:
    #             if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
    #                 utility = self._get_call_utility_final_street(board_2d=node.env_state[EnvDictIdxs.board_2d],
    #                                                         p_id=p_id)

    #             else:  # preflop
    #                 utility = self._get_call_utility_preflop(node=node, p_id=p_id)
    #         # equity.shape = (2, 6)

    #         # set boardcards to 0
    #         for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
    #             if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
    #                 utility[c, :] = 0.0
    #                 utility[:, c] = 0.0

    #         node.ev = np.zeros(shape=(self._tree.n_seats, range_size)) * node.env_state[EnvDictIdxs.main_pot] / 2
    #         node.ev[p_id] *= np.dot(utility, node.reach_probs[1 - p_id])

    #     elif node.p_id_acting_next != 1 - p_id:
    #         N_ACTIONS = len(node.children)
    #         utility = np.zeros(shape=(range_size, range_size), dtype=np.float32)

    #         for i, child in enumerate(node.children):
    #             util = self.compute_vr_cf_values_heads_up(node=child, p_id=p_id)
    #             utility += (util.T * node.strategy[:,i]).T

    #         node.ev = np.zeros(shape=(self._tree.n_seats, range_size), dtype=np.float32)
    #         node.ev[p_id] *= np.dot(utility[:, :], node.reach_probs[1-p_id])
    #     else:
    #         N_ACTIONS = len(node.children)
    #         p = node.sample_prob
    #         utils = []
    #         utility = np.zeros(shape=(range_size, range_size), dtype=np.float32)
    #         for i, child in enumerate(node.children):
    #             utils.append(self.compute_vr_cf_values_heads_up(node=child, p_id=p_id))
    #         for i in range(range_size):
    #             a = node.sample_actions[i]
    #             if np.sum(node.reach_probs[p_id]) == 0:
    #                 ref_u_h = 0.0
    #                 ref_u_ha = 0.0
    #             else:
    #                 ref_u_h = - node.ref_ev[1 - p_id, i] / np.sum(node.reach_probs[p_id])
    #                 ref_u_ha = - node.ref_ev_all_actions[a, 1 - p_id, i] / np.sum(node.reach_probs[p_id])
    #             utility[:, i] = ref_u_h + (node.strategy[i, a] * utils[a][:,i] - 
    #                                     node.ref_strategy[i, a] * ref_u_ha) / node.sample_prob[i]
    #             # ev_all_actions.shape=(N_ACTIONS, n_seats, range_size)
    #         node.ev = np.zeros(shape=(self._tree.n_seats, range_size), dtype=np.float32)
    #         node.ev[p_id] *= np.dot(utility[:, :], node.reach_probs[1-p_id])
    #     return utility
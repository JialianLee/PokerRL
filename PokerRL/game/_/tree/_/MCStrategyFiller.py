# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)
import numpy as np
import sys

from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game.PokerRange import PokerRange
from PokerRL.game._.tree._.nodes import PlayerActionNode, ChanceNode

class MCStrategyFiller:

    def __init__(self, tree, env_bldr):
        self._tree = tree
        self._env_bldr = env_bldr
        self._chance_filled = False

    def fill_uniform_random(self):

        if not self._chance_filled:
            self._fill_chance_node_strategy(node=self._tree.root)
            self._chance_filled = True

        self._fill_uniform_random(node=self._tree.root)
        self.update_reach_probs()

    def fill_random_random(self):

        if not self._chance_filled:
            self._fill_chance_node_strategy(node=self._tree.root)
            self._chance_filled = True

        self._fill_random_random(node=self._tree.root)
        self.update_reach_probs()

    def fill_with_agent_policy(self, agent):

        if not self._chance_filled:
            self._fill_chance_node_strategy(node=self._tree.root)
            self._chance_filled = True

        self._fill_with_agent_policy(node=self._tree.root, agent=agent)

        self.update_reach_probs()

    def update_reach_probs(self):
        self._update_reach_probs(node=self._tree.root)

    def compute_reach_probs(self, p_id):
        self._compute_reach_probs(node=self._tree.root, p_id=p_id)

    def sample_opponent(self, p_id):
        self._sample_opponent(node=self._tree.root, p_id=p_id)

    def generate_samples(self, p_id, player, opponent, chance_p):
        self._generate_samples(self._tree.root, p_id, player, opponent, chance_p)

    def _fill_uniform_random(self, node):
        if node is not self._tree.root and node.p_id_acted_last is not self._tree.CHANCE_ID:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)
            assert np.all(np.abs(np.sum(node.parent.strategy, axis=1) - 1) < 0.001)

        if node.is_terminal:
            return

        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            n_actions = len(node.children)
            node.strategy = np.full(shape=(self._env_bldr.rules.RANGE_SIZE, n_actions),
                                    fill_value=1.0 / float(n_actions))

        for c in node.children:
            self._fill_uniform_random(node=c)

    def _fill_random_random(self, node):
        # assert node reach_probs are set
        # assert np.all(np.abs(np.sum(node.reach_probs, axis=1) - 1) < 0.0001)
        if node is not self._tree.root and node.p_id_acted_last is not self._tree.CHANCE_ID:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)
            assert np.all(np.abs(np.sum(node.parent.strategy, axis=1) - 1) < 0.001)

        if node.is_terminal:
            return

        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            n_actions = len(node.children)
            node.strategy = np.random.random(size=(self._env_bldr.rules.RANGE_SIZE, n_actions))
            node.strategy /= np.expand_dims(np.sum(node.strategy, axis=1), axis=-1)

        for c in node.children:
            self._fill_random_random(node=c)

    def _fill_with_agent_policy(self, node, agent):
        """
        The agent has to know the reach_probs. Therefore he has to go through all of the previous nodes to build
        be able to output a strategy for a given node. Reach_probs are saved directly under node.reach_probs
        """

        if node is not self._tree.root and node.p_id_acted_last is not self._tree.CHANCE_ID:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)
            assert np.all(np.abs(np.sum(node.parent.strategy, axis=1) - 1) < 0.001)

        if node.is_terminal:
            return

        # _______________________________________ set strategy of node _________________________________________________
        # Chance acted last or regular action after action node
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            # fake steps the agent's internal env to the current state
            agent.set_to_public_tree_node_state(node=node)

            assert node.p_id_acting_next == agent._internal_env_wrapper.env.current_player.seat_id, node.p_id_acting_next

            agent_strat = agent.get_a_probs_for_each_hand()
            node.strategy = agent_strat[:, node.allowed_actions]

        # ___________ recursion
        for c in node.children:
            self._fill_with_agent_policy(node=c, agent=agent)

    def _update_reach_probs(self, node):
        if node is not self._tree.root:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)

        if node.is_terminal:
            return

        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            for c in node.children:
                c.reach_probs = np.copy(node.reach_probs)

                # since arbitrary actions might be illegal in a state, we have to filter which indices to use
                a_idx = node.allowed_actions.index(c.action)
                c.reach_probs[node.p_id_acting_next] = node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]

        # new round gets rolled out now
        elif node.p_id_acting_next == self._tree.CHANCE_ID:
            for c in range(len(node.children)):
                child = node.children[c]
                child.reach_probs = node.reach_probs * node.strategy[:, c]

        else:
            raise TypeError(node)

        for c in node.children:
            self._update_reach_probs(node=c)

    def _compute_reach_probs(self, node, p_id):
        assert node.visited == True
        if node is not self._tree.root:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)

        if node.is_terminal:
            return

        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            if node.p_id_acting_next != 1 - p_id:
                for c in node.children:
                    c.reach_probs = np.copy(node.reach_probs)

                    # since arbitrary actions might be illegal in a state, we have to filter which indices to use
                    a_idx = node.allowed_actions.index(c.action)
                    c.reach_probs[node.p_id_acting_next] = node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]
            else:
                for c in node.children:
                    if c.visited:
                        c.reach_probs = np.copy(node.reach_probs)

        # new round gets rolled out now
        elif node.p_id_acting_next == self._tree.CHANCE_ID:
            for c in range(len(node.children)):
                child = node.children[c]
                child.reach_probs = node.reach_probs * node.strategy[:, c]

        else:
            raise TypeError(node)

        for c in node.children:
            if c.visited:
                self._update_reach_probs(node=c)

    def _fill_chance_node_strategy(self, node):
        assert node.strategy is None
        if node.is_terminal:
            return

        if node.p_id_acting_next == self._tree.CHANCE_ID:
            game_round = node.children[0].env_state[EnvDictIdxs.current_round]
            n_children = len(node.children)
            assert n_children == self._env_bldr.lut_holder.DICT_LUT_N_BOARDS[game_round]

            # chance nodes are uniform random
            node.strategy = np.zeros(shape=(self._env_bldr.rules.RANGE_SIZE, n_children), dtype=np.float32)

            # set strategy for impossible hands to 0
            for c_id in range(n_children):
                mask = PokerRange.get_possible_range_idxs(rules=self._env_bldr.rules,
                                                          lut_holder=self._env_bldr.lut_holder,
                                                          board_2d=node.children[c_id].env_state[EnvDictIdxs.board_2d])
                node.strategy[mask, c_id] = 1.0 / (self._env_bldr.rules.N_CARDS_IN_DECK - 2)

        for c in node.children:
            self._fill_chance_node_strategy(node=c)

    def _sample_opponent(self, node, p_id):
        if node is not self._tree.root:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)

        if node.is_terminal:
            return
        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            if node.p_id_acting_next != 1 - p_id:
                for c in node.children:
                    c.reach_probs = np.copy(node.reach_probs)

                    # since arbitrary actions might be illegal in a state, we have to filter which indices to use
                    a_idx = node.allowed_actions.index(c.action)
                    c.reach_probs[node.p_id_acting_next] = node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]
            else:       
                N_ACTIONS = len(node.children)
                opp = 1 - p_id
                for c in node.children:
                    c.reach_probs = np.copy(node.reach_probs)
                    c.reach_probs[opp] = 0.0
                node.sample_actions = []
                node.sample_prob = []
                for a in range(self._env_bldr.rules.RANGE_SIZE):
                    stg = node.strategy[a]
                    ref_stg = node.ref_strategy[a]
                    stg_diff = np.zeros(N_ACTIONS)
                    for i in range(N_ACTIONS):
                        stg_diff[i] = abs(ref_stg[i] -   stg[i]) + stg[i] * (node.V_value[1-p_id, a] + np.max(node.M_value[p_id]))
                    if np.sum(stg_diff) < 1e-5:
                        p = ref_stg
                    else:
                        p = stg_diff/np.sum(stg_diff)
                    action = np.random.choice(range(N_ACTIONS), size=1, p=p)[0]
                    node.children[action].reach_probs[opp, a] = node.reach_probs[opp, a]
                    node.sample_actions.append(action)
                    node.sample_prob.append(p[action])

        # new round gets rolled out now
        elif node.p_id_acting_next == self._tree.CHANCE_ID:
            for c in range(len(node.children)):
                child = node.children[c]
                child.reach_probs = np.copy(node.reach_probs)
                # child.reach_probs = node.reach_probs * node.strategy[:, c]

        else:
            raise TypeError(node)

        for c in node.children:
            self._sample_opponent(node=c, p_id=p_id)

    def _generate_samples(self, node, p_id, player, opponent, chance_p):
        if node is not self._tree.root:
            assert node.parent.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.parent.children),)

        if node.is_terminal:
            return

        # Chance acted last
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            if node.p_id_acting_next == p_id:
                if player:
                    N_ACTIONS = len(node.children)
                    for c in node.children:
                        c.reach_probs = np.copy(node.reach_probs)
                        c.reach_probs[p_id] = 0.0
                        c.sample_reach_probs = np.copy(node.sample_reach_probs)
                        c.sample_reach_probs[p_id] = 0.0
                    for a in range(self._env_bldr.rules.RANGE_SIZE):
                        p = self._tree.get_sample_strategy(node, a)
                        action = np.random.choice(N_ACTIONS, p=p)
                        node.children[action].reach_probs[p_id, a] = node.reach_probs[p_id, a] * node.strategy[a, action]
                        node.children[action].sample_reach_probs[p_id, a] = node.sample_reach_probs[p_id, a] * p[action]
                else:
                    for c in node.children:
                        c.reach_probs = np.copy(node.reach_probs)
                        c.sample_reach_probs = np.copy(node.sample_reach_probs)

                        # since arbitrary actions might be illegal in a state, we have to filter which indices to use
                        a_idx = node.allowed_actions.index(c.action)
                        c.reach_probs[node.p_id_acting_next] = node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]
            else:
                if opponent: 
                    N_ACTIONS = len(node.children)
                    opp = 1 - p_id
                    for c in node.children:
                        c.reach_probs = np.copy(node.reach_probs)
                        c.reach_probs[opp] = 0.0
                        c.sample_reach_probs = np.copy(node.sample_reach_probs)
                        c.sample_reach_probs[opp] = 0.0                        
                    for a in range(self._env_bldr.rules.RANGE_SIZE):
                        p = self._tree.get_sample_strategy(node, a)
                        action = np.random.choice(N_ACTIONS, p=p)
                        node.children[action].reach_probs[opp, a] = node.reach_probs[opp, a] * node.strategy[a, action]
                        node.children[action].sample_reach_probs[opp, a] = node.sample_reach_probs[opp, a] * p[action]
                else:
                    for c in node.children:
                        c.reach_probs = np.copy(node.reach_probs)
                        c.sample_reach_probs = np.copy(node.sample_reach_probs)

                        # since arbitrary actions might be illegal in a state, we have to filter which indices to use
                        a_idx = node.allowed_actions.index(c.action)
                        c.reach_probs[node.p_id_acting_next] = node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]                  

        # new round gets rolled out now
        elif node.p_id_acting_next == self._tree.CHANCE_ID:
            if chance_p:
                for c in range(len(node.children)):
                    child = node.children[c]
                    child.reach_probs = np.zeros(node.reach_probs.shape)
                    child.sample_reach_probs = np.zeros(node.sample_reach_probs)
                for a in range(self._env_bldr.rules.RANGE_SIZE):
                    action = np.random.choice(len(node.children), p=node.strategy[a]/np.sum(node.strategy[a]))
                    node.children[action].reach_probs[:, a] = node.reach_probs[:, a]# * node.strategy[a, action]
                    node.children[action].sample_reach_probs[:, a] = node.sample_reach_probs[:, a]# * node.strategy[a, action]            
            else:
                for c in range(len(node.children)):
                    child = node.children[c]
                    # child.reach_probs = np.copy(node.reach_probs)
                    child.reach_probs = node.reach_probs * node.strategy[:, c]
                    child.sample_reach_probs = np.copy(node.sample_reach_probs)
        else:
            raise TypeError(node)

        for c in node.children:
            self._generate_samples(c, p_id, player, opponent, chance_p)
   

# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of CFR in an abstracted version of Discrete No-Limit Leduc Poker with actions
{FOLD, CHECK/CALL, POT-SIZE-BET/RAISE}.
"""

from PokerRL.cfr.MCCFR import MCCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc, StandardLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase
import time
import numpy as np

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper
    expls = []
    touching_nodes = []
    n_iterations = 100
    name = "MCCFR_EXAMPLE"
    game_cls = StandardLeduc
    game_dic = {DiscretizedNLLeduc: "DiscretizedNLLeduc", StandardLeduc: "StandardLeduc"}

    # Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
    chief = ChiefBase(t_prof=None)
    cfr = MCCFR(name=name,
                     game_cls=game_cls,
                     agent_bet_set=bet_sets.POT_ONLY,
                     chief_handle=chief,
                     innerloop_epi=10)#100)
    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        expl = cfr.iteration()
        print("expl", expl)
        print("nodes", cfr.touching_nodes)
        expls.append(expl)
        touching_nodes.append(cfr.touching_nodes)
    path = "/home/jialian/cfr/PokerRL/data/"
    np.save(path + name + "_" + game_dic[game_cls] + ".npy", expls)
    np.save(path + name + "_" + game_dic[game_cls] + "_touching_nodes.npy", touching_nodes)
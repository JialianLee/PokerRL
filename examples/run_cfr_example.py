# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of CFR in an abstracted version of Discrete No-Limit Leduc Poker with actions
{FOLD, CHECK/CALL, POT-SIZE-BET/RAISE}.
"""

from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import Flop5Holdem
from PokerRL.game.games import DiscretizedNLLeduc, StandardLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

import numpy as np
import time

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    n_iterations = 150
    name = "CFR_EXAMPLE"
    expls = []
    touching_nodes = []
    game_cls = StandardLeduc
    game_dic = {DiscretizedNLLeduc: "DiscretizedNLLeduc", StandardLeduc: "StandardLeduc"}

    
    start_time = time.time()
    # Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
    chief = ChiefBase(t_prof=None)
    # crayon = CrayonWrapper(name=name,
    #                        path_log_storage=None,
    #                        chief_handle=chief,
    #                        runs_distributed=False,
    #                        runs_cluster=False,
    #                        )
    cfr = VanillaCFR(name=name,
                     game_cls=game_cls,
                     agent_bet_set=bet_sets.POT_ONLY,
                     chief_handle=chief)
    build_time = time.time()
    print("build time", build_time - start_time)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        expl = cfr.iteration()
        print("expl", expl)
    #     print("nodes", cfr.touching_nodes)
    #     expls.append(expl)
    #     touching_nodes.append(cfr.touching_nodes)
    #     # crayon.update_from_log_buffer()
    #     # crayon.export_all(iter_nr=iter_id)
    # path = "/home/jialian/cfr/PokerRL/data/"
    # np.save(path + name + "_" + game_dic[game_cls] + ".npy", expls)
    # np.save(path + name + "_" + game_dic[game_cls] + "_touching_nodes.npy", touching_nodes)

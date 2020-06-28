# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of Linear CFR in a Leduc poker game with actions {FOLD, CHECK/CALL, POT-SIZE-BET/RAISE}.
It will store logs and tree files on your C: drive.
"""

from PokerRL.cfr.LinearCFR import LinearCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc, StandardLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase
import numpy as np

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    expls = []
    touching_nodes = []
    n_iterations = 150
    name = "LinCFR_EXAMPLE"
    game_cls = StandardLeduc
    game_dic = {DiscretizedNLLeduc: "DiscretizedNLLeduc", StandardLeduc: "StandardLeduc"}

    # Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
    chief = ChiefBase(t_prof=None)
    # crayon = CrayonWrapper(name=name,
    #                        path_log_storage=None,
    #                        chief_handle=chief,
    #                        runs_distributed=False,
    #                        runs_cluster=False)
    cfr = LinearCFR(name=name,
                    game_cls=game_cls,
                    agent_bet_set=bet_sets.POT_ONLY,
                    chief_handle=chief)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        expl = cfr.iteration()
        # expl = np.sum(cfr._trees[0].root.exploitability)
        print("expl", expl)
        print("nodes", cfr.touching_nodes)
        expls.append(expl)
        touching_nodes.append(cfr.touching_nodes)
        # crayon.update_from_log_buffer()
        # crayon.export_all(iter_nr=iter_id)
    path = "/home/jialian/cfr/PokerRL/data/"
    np.save(path + name + "_" + game_dic[game_cls] + ".npy", expls)
    np.save(path + name + "_" + game_dic[game_cls] + "_touching_nodes.npy", touching_nodes)

import numpy as np 
import matplotlib as mpl
import pylab
from collections import Counter
mpl.use("Agg")
plt = mpl.pyplot

game_cls = 2
game_dic = {1: "DiscretizedNLLeduc", 2: "StandardLeduc"}


methods = ['CFR', 'CFRplus', 'LinCFR', 'MCCFR', 'MCCFR_OS','VR_CFR']
path = "/home/jialian/cfr/PokerRL/data/"
for m in methods:
    expls = np.load(path + m + "_EXAMPLE_" + game_dic[game_cls] + ".npy")
    nodes = np.load(path + m + "_EXAMPLE_" + game_dic[game_cls] + "_touching_nodes.npy")
    plt.plot(np.log10(nodes), np.log10(np.array(expls)), label=m)

plt.legend()
plt.savefig(path + game_dic[game_cls]+".png")
print("Save", game_dic[game_cls])
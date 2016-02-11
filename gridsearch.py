import itertools
import numpy as np
import word_characteristics as wc
import random
from detect_antecedents import AntecedentClassifier

# Purpose:
#   to make an exhaustive list of hyperparameter combinations.

def build_grid(params):
    grid = []
    for combo in itertools.product(*params):
        grid.append(combo)

    return grid

if __name__ == '__main__':
    grid = build_grid(
         [[10.0**i for i in range(-4,2)],
          [10.0**i for i in range(-4,0)] + [0.99],
          [5]]
    )
    random.shuffle(grid)

    a = AntecedentClassifier(0,14, 15,19, 20,24)
    a.initialize(['VP', wc.is_adjective, wc.is_verb], seed=9001, save=False, load=True, update=False)

    for combo in grid:
        print "Current Params: ",combo
        a.C = combo[0]
        a.learn_rate = lambda x: combo[1]
        name = 'full_c%s_lr%s_k%s_'%(combo[0], combo[1], combo[2])

        a.fit(epochs=100, k=combo[2])
        a.make_graphs(name)
        np.save('saved_weights/'+name, np.array(a.W_avg))

        a.reset()
        a.initialize_weights(seed=9001)
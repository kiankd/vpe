import itertools
import numpy as np
import word_characteristics as wc
import random
import os
from detect_antecedents import AntecedentClassifier
from sys import argv

# Purpose:
#   to make an exhaustive list of hyperparameter combinations.

def build_grid(params):
    grid = []
    for combo in itertools.product(*params):
        grid.append(combo)

    return grid

def completed_combos():
    combos = set([])
    for fname in os.listdir('tests/post_changes/'):
        splitt = fname.split('_')
        c, lr, k = float(splitt[1][1:]), float(splitt[2][2:]), float(splitt[3][1:])
        combos.add((c,lr,k))
    return combos

if __name__ == '__main__':
    grid = build_grid(
         [[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
          [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
          [int(argv[1])]]
    )
    completed = completed_combos() 

    print 'How much we need versus how much we did: %d, %d'%(len(grid),len(completed))

    a = AntecedentClassifier(0,14, 15,19, 20,24)
    a.initialize(['VP', wc.is_adjective, wc.is_verb], seed=9001, save=False, load=True, update=False)

    for combo in grid:
        if not combo in completed:
            print "Current Params: ",combo
            a.C = combo[0]
            a.learn_rate = lambda x: combo[1]
            name = 'full_c%s_lr%s_k%s_'%(combo[0], combo[1], combo[2])

            a.fit(epochs=100, k=combo[2])
            a.make_graphs(name)
            a.log_results(name)
            np.save('saved_weights/'+name, np.array(a.W_avg))

            a.reset()
            a.initialize_weights(seed=9001)


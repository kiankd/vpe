import itertools

# Purpose:
#   to make an exhaustive list of hyperparameter combinations.

def build_grid(params):
    grid = []
    for combo in itertools.product(*params.itervalues()):
        grid.append(combo)

    return grid

grid = build_grid({'C':[1,2,3],'a':[0.1,0.001,0.0001],'k':[5,10,15,20]})

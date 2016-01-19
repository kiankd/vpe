# Purpose:
#   to make an exhaustive list of hyperparameter combinations.

def build_grid(params):
    ret = []
    num_tuples = 1
    for key in params:
        num_tuples *= len(params[key])

    dicts = [{} for _ in range(num_tuples)]
    for key in params:
        for val in params[key]:
            for d in dicts:
                if not key in d:
                    d[key] = val

    return dicts

grid = build_grid({'C':[1,2,3],'a':[0.1,0.001,0.0001],'k':[5,10,15,20]})




"""
'C':[1,10,100]
'a':[0.1,0.05,0.001]
'x':[5,25,50]

RETURN:

[
 {'C':1, 'a':0.1, 'x':5}
 {'C':1, 'a':0.05, 'x':5}
 {'C':1, 'a':0.001, 'x':5}
 {'C':1, 'a':0.1, 'x':25}
 {'C':1, 'a':0.1, 'x':50}
 {'C':10, 'a':0.05, 'x':5}
 ...
]
"""

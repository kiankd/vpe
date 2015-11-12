from gurobipy import *
import numpy as np
import math

def norm(x):
    return quicksum(xi * xi for xi in x)

def slow_multiply(a, constants):
    c = []
    for i in range(len(constants)):
        c.append(a * constants[i])
    return quicksum(c)

def update_weights(old, best, gold, L, C=1.0):
    primal_constraints = len(best)

    m = Model("qp")

    # Create the vector of alphas we want to solve for
    alpha = []
    for k in range(primal_constraints):
        alpha.append(m.addVar(name='a%d' % k, lb=0.0, ub=GRB.INFINITY))
    m.update()

    # Building the objective function
    try:
        diffs = [list(gold - best[k]) for k in range(primal_constraints)]
    except TypeError:
        diffs = [list(np.array(gold) - np.array(best[k])) for k in range(primal_constraints)]

    s = []
    for k in range(primal_constraints):
        s.append(slow_multiply(alpha[k], diffs[k]))

    obj = -0.5 * norm(s) + quicksum(alpha[k] * L(gold, best[k]) for k in range(primal_constraints))\
                         - quicksum(alpha[k] * np.dot(diffs[k], old) for k in range(primal_constraints))

    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(quicksum(alpha) <= C, "c1") # Very simple constraint based on learning rate.

    m.optimize()

    # Weight update step:
    return old + np.sum(alpha[k].x * np.array(diffs[k]) for k in range(primal_constraints))

if __name__ == '__main__':
    old = np.array([1.0, 1.0])
    best = [[0.5, 0.4], [0.3, 0.2]]
    gold = [1.0, 1.0]
    L = lambda x, y: np.linalg.norm(np.array(x) + np.array(y), ord=1)
    C = 1.0

    w = [old, update_weights(old, best, gold, L, C=C)]
    i = 1
    while np.linalg.norm(w[i] - w[i-1]) >= 0.001:
        w.append(update_weights(w[i], best, gold, L, C=C))
        i += 1

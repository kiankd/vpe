from gurobipy import *
from cStringIO import StringIO
import numpy as np
import math
import sys

class Capturing(list):
    """All this does is stop gurbipy from printing the results of each optimization."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

def norm(x):
    return quicksum(xi * xi for xi in x)

def slow_multiply(a, constants):
    """Multiplies a variable times each constant and returns the sum."""
    c = []
    for i in xrange(len(constants)):
        c.append(a * constants[i])
    return quicksum(c)

def update_weights(old, best, gold, L, C=1.0):
    primal_constraints = len(best)

    m = Model("qp")
    m.params.OutputFlag = 0

    # Create the vector of alphas we want to solve for
    alpha = []
    for k in range(primal_constraints):
        alpha.append(m.addVar(name='a%d' % k, lb=0.0, ub=GRB.INFINITY))
    m.update()

    # Building the objective function
    # print 'bestk:',best
    # print 'gold:',gold
    # print 'best[0].x',best[0].x
    # print 'gold.x',gold.x

    diffs = [(gold.x - best[k].x) for k in xrange(primal_constraints)]

    s = []
    for k in xrange(primal_constraints):
        s.append(slow_multiply(alpha[k], diffs[k]))

    obj = -0.5 * norm(s) + quicksum(alpha[k] * L(gold, best[k]) for k in xrange(primal_constraints))\
                         - quicksum(alpha[k] * np.dot(diffs[k], old) for k in xrange(primal_constraints))

    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(quicksum(alpha) <= C, "c1") # Very simple constraint based on learning rate.

    # with Capturing() as output:
    m.optimize()    

    # Weight update step:
    return old + np.sum(alpha[k].x * diffs[k] for k in xrange(primal_constraints))

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

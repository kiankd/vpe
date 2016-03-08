from gurobipy import *
import numpy as np

# def update_weights(old, bestk, gold, L):
old = [1.0,1.0,1.0,1.0,1.0]
# old = [old[i]/max(old) for i in range(len(old))]
bestk = [10.0, 20.0, 30.0]
gold = 25.0
L = lambda x,y: x-y

num_weights = len(old)
num_constraints = len(bestk)

m = Model("qp")

w = []
for i in range(num_weights):
	w.append(m.addVar(lb=GRB.INFINITY, ub=GRB.INFINITY, name="w%d"%i))

slacks = []
for i in range(num_constraints):
	slacks.append(m.addVar(lb=0.0, ub=GRB.INFINITY, name="s%d"%i))

m.update()

# We are minimizing the equation || w - old || = (w[i]-old[i])^2 for i in range(num_weights) = w[i]^2 - 2*w[i]*old[i]

# w1 ^2 + w2 ^2 + ...
obj = quicksum(w[i]*w[i] for i in range(num_weights))

# -2 * w1 * old[1] - 2 * w2 * old2 ... (middle term in the quadratic equation after un-factoring (w[i]-old[i])^2)
obj += quicksum(-2.0 * w[i] * old[i] for i in range(num_weights))

# Slack variables...
obj += quicksum(slacks)

m.setObjective(obj, GRB.MINIMIZE)


# Constraints:
for k in range(num_constraints):
	m.addConstr(slacks[k] + quicksum(np.dot((gold-bestk[k]), w)) >= L(gold, bestk[k]), "c%d"%k)

m.optimize()
from gurobipy import *
import numpy as np
import math

old = [1.0,1.0]
best = [[0.5, 0.4], [0.3, 0.2]]
gold = [1.0, 1.0]
L = lambda x,y: np.linalg.norm(np.array(x)-np.array(y) , ord=1)
C = 1.0

def norm(x):
	return (quicksum(xi*xi for xi in x))

def slow_multiply(a, constants):
	for i in range(len(constants)):
		constants[i] = a * constants[i]
	return quicksum(constants)

# def update_weights(old, best, gold, L, C=1.0):
if __name__ == '__main__':
	primal_constraints = len(best)

	m = Model("qp")

	## Create the vector of alphas we want to solve for ##
	alpha = []
	for k in range(primal_constraints):
		alpha.append(m.addVar(name='a%d'%k, lb=0.0, ub=GRB.INFINITY))
	m.update()

	## Building the objective function ##
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
	# sum(alpha[k].x * diffs[k] for k in range(primal_constraints))




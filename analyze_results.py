import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D

RESULTS_DIR = 'ant_results/'
DEST = 'results_analysis/'

EXACT_MATCH, HEAD_MATCH, HEAD_OVERLAP = 0, 1, 2
NAMES = {EXACT_MATCH:'Exact Match', HEAD_MATCH:'Head Match', HEAD_OVERLAP:'Head Overlap'}
CHOICE = HEAD_MATCH

#  'grid2_c%s_lr%s_k%s_'%(combo[0], combo[1], combo[2])
def decode_fname(string):
    s = string.split('_')
    c = str(s[1][1:])
    lr = str(s[2][2:])
    k = str(s[3][1:])
    return c,lr,k

def bestn(n, param1, param2, lst, name, comp='greater'):
    op = None
    if comp == 'greater':
        op = operator.gt
    else:
        op = operator.lt

    ret = [val for val in lst[:n]]
    for val in lst[n:]:
        for best in ret:
            if op(val, best):
                ret.remove(best)
                ret.append(val)

    for best in sorted(ret):
        idx = lst.index(best)
        print name+' best results: c = %s, lr = %s --> %0.2f'%(param1[idx], param2[idx], best)

    return ret

def load_data():
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.npy') and not fname.startswith('schedule'):
            arr = np.load(RESULTS_DIR+fname)

            train_err = np.array(arr[0])
            val_err = np.array(arr[1])
            test_err = np.array(arr[2])
            train_res = np.array(arr[3][0])
            val_res = np.array(arr[3][1])
            test_res = np.array(arr[3][2])

            results[decode_fname(fname)] = (train_err, val_err, test_err, train_res, val_res, test_res)

    return results

def analyze(results):
    data_for_k = set()
    for key in results:
        data_for_k.add(key[2])

    for k in sorted(data_for_k):
        k_err = {}
        k_res = {}
        for key in results:
            if key[2] == k:
                k_err[(key[0],key[1])] = results[key][:3]
                k_res[(key[0],key[1])] = results[key][3:]

        print 'k = %s'%k
        print 'Error:'
        make_3d_graph("errors_k%s"%k, k_err)
        print '\nResults:'
        make_3d_graph("results_k%s"%k, k_res, "Percent "+NAMES[CHOICE])
        print

def make_3d_graph(name, dic, zname="Error"):
    fun = min if zname == "Error" else max

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x,y = [],[]
    for key in dic:
        x.append(float(key[0]))
        y.append(float(key[1]))

    train_z, val_z, test_z = [],[],[]
    for key in dic:
        if fun == min:
            train_z.append(fun(dic[key][0]))
            val_z.append(fun(dic[key][1]))
            test_z.append(fun(dic[key][2]))
        else:
            train_z.append(fun(dic[key][0][:,CHOICE]))
            val_z.append(fun(dic[key][1][:,CHOICE]))
            test_z.append(fun(dic[key][2][:,CHOICE]))

    print 'Best train, val, test %s: %0.2f, %0.2f, %0.2f'%(zname, fun(train_z), fun(val_z), fun(test_z))
    bestn(1, x, y, train_z, 'Training', comp = 'lesser' if fun==min else 'greater')
    bestn(1, x, y, val_z, 'Validation', comp = 'lesser' if fun==min else 'greater')
    # bestn(3, x, y, test_z, 'Test', comp = 'lesser' if fun==min else 'greater')

    # ax.w_xaxis = x
    # ax.w_yaxis = y

    ax.scatter(x, y, train_z, color='blue')
    ax.scatter(x, y, val_z, color='red')
    ax.scatter(x, y, test_z, color='yellow')

    ax.set_xlabel('C')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel(zname)

    plt.savefig(DEST+name+'test.png')

    fig.clf()

if __name__ == '__main__':
    data = load_data()
    analyze(data)

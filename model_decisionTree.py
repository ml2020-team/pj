#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 5:15 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : model_decisionTree.py
# @Software: PyCharm


import numpy as np
from os.path import join as pjoin
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier


test = True


def fit_decisionTree(x_train, y_train, x_test, y_test, max_depth, min_samples_leaf, dot_filename, classnames):
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)
    tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(x_train, y_train)
    with open(dot_filename, 'w') as f:
        f = export_graphviz(tree, out_file=f, class_names=classnames)
    return tree.score(x_train, y_train), tree.score(x_test, y_test)



if __name__ == '__main__':

    X_train_tasks = [np.load(pjoin('./data/task' + str(i), 'x_train.npy')) for i in range(1, 4)]
    Y_train_tasks = [np.load(pjoin('./data/task' + str(i), 'y_train.npy')) for i in range(1, 4)]
    X_test_tasks  = [np.load(pjoin('./data/task' + str(i), 'x_test.npy')) for i in range(1, 4)]
    Y_test_tasks  = [np.load(pjoin('./data/task' + str(i), 'y_test.npy')) for i in range(1, 4)]
    if test:
        train_acc1, test_acc1 = fit_decisionTree(X_train_tasks[0], Y_train_tasks[0],
                                                 X_test_tasks[0], Y_test_tasks[0],
                                                 max_depth=4, min_samples_leaf=55,
                                                 dot_filename='tree1.dot',
                                                 classnames=['class' + str(i) for i in range(1, 4)])
        print('task1 train acc: {:.3f}, test dacc: {:.3f}'.format(train_acc1, test_acc1))
        train_acc2, test_acc2 = fit_decisionTree(X_train_tasks[1], Y_train_tasks[1],
                                                 X_test_tasks[1], Y_test_tasks[1],
                                                 max_depth=6, min_samples_leaf=10,
                                                 dot_filename='tree2.dot',
                                                 classnames=['class' + str(i) for i in range(1, 6)])
        print('task2 train acc: {:.3f}, test acc: {:.3f}'.format(train_acc2, test_acc2))
        train_acc3, test_acc3 = fit_decisionTree(X_train_tasks[2], Y_train_tasks[2],
                                                 X_test_tasks[2], Y_test_tasks[2],
                                                 max_depth=15, min_samples_leaf=1,
                                                 dot_filename='tree3.dot',
                                                 classnames=['class' + str(i) for i in range(1, 16)])
        print('task3 train acc: {:.3f}, test acc: {:.3f}'.format(train_acc3, test_acc3))


    else:
        max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        accs1, accs2 = [], []
        for l in max_depths:
            accs1.append(fit_decisionTree(X_train_tasks[0], Y_train_tasks[0],
                                          X_test_tasks[0], Y_test_tasks[0],
                                          max_depth=l, min_samples_leaf=1))
            accs2.append(fit_decisionTree(X_train_tasks[1], Y_train_tasks[1],
                                          X_test_tasks[1], Y_test_tasks[1],
                                          max_depth=l, min_samples_leaf=1))
        ax1 = plt.subplot(211)
        plt.ylabel('acc')
        plt.xlabel('max_depth')
        plt.plot(max_depths, [p[0] for p in accs1], label='train')
        plt.plot(max_depths, [p[1] for p in accs1], label='test')
        plt.legend()
        ax1.set_title('task1')

        ax2 = plt.subplot(212)
        plt.ylabel('acc')
        plt.xlabel('max_depth')
        plt.plot(max_depths, [p[0] for p in accs2], label='train')
        plt.plot(max_depths, [p[1] for p in accs2], label='test')
        plt.legend()
        ax2.set_title('task2')
        plt.show()

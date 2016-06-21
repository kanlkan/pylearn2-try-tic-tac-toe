#!/usr/bin/env python
# -*- cording: utf-8 -*-

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
import csv

class TicTacToe(DenseDesignMatrix):
    def __init__(self):
        X = []
        y = []
        X_temp = [0,0,0,0,0,0,0,0,0]  # 3x3 board
        y_temp = [0,0,0,0,0,0,0,0,0]  # 3x3 board

        self.class_names = ['0', '3']
        f = open("tic-tac-toe_records_lose.csv", "r")
        reader = csv.reader(f)

        for row in reader:
            for i, cell_index in enumerate(row):
                if cell_index == "win" or cell_index == "lose":
                    X_temp = [0,0,0,0,0,0,0,0,0]
                elif i % 2 == 0:
                    temp = []
                    X_temp[int(cell_index)] = 1
                    for x in X_temp:
                        temp.append(x)

                    #print "  temp = " + str(temp)
                    X.append(temp)
                else:
                    X_temp[int(cell_index)] = 2
                    y_temp[int(cell_index)] = 3
                    #print "y_temp = " + str(y_temp)
                    y.append(y_temp)
                    y_temp = [0,0,0,0,0,0,0,0,0]

        X = np.array(X)
        y = np.array(y)
        super(TicTacToe, self).__init__(X=X, y=y)


data_set = TicTacToe()
h0 = mlp.Sigmoid(layer_name='h0', dim=9, irange=.1, init_bias=1.)
out = mlp.Softmax(layer_name='out', n_classes=9, irange=0.)
trainer = sgd.SGD(learning_rate=.05, batch_size=200, termination_criterion=EpochCounter(5000))
layers = [h0, out]

ann = mlp.MLP(layers, nvis=9)
trainer.setup(ann, data_set)

while True:
    trainer.train(dataset=data_set)
    ann.monitor.report_epoch()
    ann.monitor()
    if trainer.continue_learning(ann) == False:
        break

next_move = [0,0,0,0,0,0,0,0,0]
inputs = np.array([[0,0,1,0,0,0,0,0,0]])
output = ann.fprop(theano.shared(inputs, name='inputs')).eval()
print output[0]
for i in range(0,9):
    if max(output[0]) == output[0][i]:
        next_move[i] = 3

print next_move

next_move = [0,0,0,0,0,0,0,0,0]
inputs = np.array([[1,0,2,1,0,0,0,1,2]])
output = ann.fprop(theano.shared(inputs, name='inputs')).eval()
print output[0]
for i in range(0,9):
    if max(output[0]) == output[0][i]:
        next_move[i] = 3

print next_move

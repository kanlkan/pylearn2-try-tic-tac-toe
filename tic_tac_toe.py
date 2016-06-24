#!/usr/bin/env python
# -*- cording: utf-8 -*-
#
# tic-tac-toe
#   none       :   (0)
#   1st player : O (1)
#   2nd player : X (2)
#

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
import numpy as np
import csv
import wx

gVersion = "1.0.0"
gTurn = True
gState = {"none":0, "O":1, "X":2}

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'tic tac toe :' + gVersion, \
                          size=(700,240))
        # GUI - main_panel
        main_panel = wx.Panel(self, wx.ID_ANY, pos=(0,0), size=(210,210))
        self.main_panel = main_panel
        cell_array = [[0 for i in range(3)] for j in range(3)]
        self.cell_array = cell_array
        
        main_layout = wx.GridSizer(3, 3, 0, 0)
        self.main_layout = main_layout
        for i in range(0,3):
            for j in range(0,3):
                self.cell_array[i][j] = CellButton(self, self.main_panel, (i*3+j))
                self.cell_array[i][j].Disable()
                self.main_layout.Add(self.cell_array[i][j], 0, wx.GROW)

        self.main_panel.SetSizer(self.main_layout)

        # GUI - sub_panels
        sub_panel1 = wx.Panel(self, wx.ID_ANY, pos=(210,0), size=(210,210))
        sub_panel2 = wx.Panel(self, wx.ID_ANY, pos=(425,0), size=(135,105))
        sub_panel3 = wx.Panel(self, wx.ID_ANY, pos=(565,0), size=(135,105))
        sub_panel4 = wx.Panel(self, wx.ID_ANY, pos=(425,115), size=(135,95))
        sub_panel5 = wx.Panel(self, wx.ID_ANY, pos=(565,115), size=(135,95))
        
        ## sub_panel1
        log_textctrl = wx.TextCtrl(sub_panel1, wx.ID_ANY, style=wx.TE_MULTILINE, \
                                   size=(210,80))
        self.log_textctrl = log_textctrl
        radiobtn_array = ("Man vs Comp", "Comp vs Man")
        game_radiobox = wx.RadioBox(sub_panel1, wx.ID_ANY, "Game Mode", \
                                    choices=radiobtn_array, style=wx.RA_HORIZONTAL)
        self.game_radiobox = game_radiobox
        start_btn = wx.Button(sub_panel1, wx.ID_ANY, "Game Start", size=(100,60))
        model_textctrl = wx.TextCtrl(sub_panel1, wx.ID_ANY, \
                                     "model.pkl", size=(200,25))
        self.model_textctrl = model_textctrl
        sub_layout1 = wx.BoxSizer(wx.VERTICAL)
        sub_layout1.Add(log_textctrl)
        sub_layout1.Add(game_radiobox)
        sub_layout1.Add(start_btn)
        sub_layout1.Add(model_textctrl)
        sub_panel1.SetSizer(sub_layout1)

        ## sub_panel2
        stext2_1 = wx.StaticText(sub_panel2, wx.ID_ANY, "irange")
        h0_irange_textctrl = wx.TextCtrl(sub_panel2, wx.ID_ANY, "0.1")
        self.h0_irange_textctrl = h0_irange_textctrl
        stext2_2 = wx.StaticText(sub_panel2, wx.ID_ANY, "init bias")
        h0_bias_textctrl = wx.TextCtrl(sub_panel2, wx.ID_ANY, "1.0")
        self.h0_bias_textctrl = h0_bias_textctrl
        box2 = wx.StaticBox(sub_panel2, wx.ID_ANY, "h0: Sigmoid")
        sub_layout2 = wx.StaticBoxSizer(box2, wx.VERTICAL)
        sub_layout2.Add(stext2_1)
        sub_layout2.Add(h0_irange_textctrl)
        sub_layout2.Add(stext2_2)
        sub_layout2.Add(h0_bias_textctrl)
        sub_panel2.SetSizer(sub_layout2)

        ## sub_panel3
        stext3_1 = wx.StaticText(sub_panel3, wx.ID_ANY, "irange")
        out_irange_textctrl = wx.TextCtrl(sub_panel3, wx.ID_ANY, "1.0")
        self.out_irange_textctrl = out_irange_textctrl
        box3 = wx.StaticBox(sub_panel3, wx.ID_ANY, "out: Softmax")
        sub_layout3 = wx.StaticBoxSizer(box3, wx.VERTICAL)
        sub_layout3.Add(stext3_1)
        sub_layout3.Add(out_irange_textctrl)
        sub_panel3.SetSizer(sub_layout3)

        ## sub_panel4
        stext4_1 = wx.StaticText(sub_panel4, wx.ID_ANY, "open .csv")
        openfile_textctrl = wx.TextCtrl(sub_panel4, wx.ID_ANY, \
                                        "records.csv", size=(130,25))
        self.openfile_textctrl = openfile_textctrl
        stext4_2 = wx.StaticText(sub_panel4, wx.ID_ANY, "save .pkl")
        savefile_textctrl = wx.TextCtrl(sub_panel4, wx.ID_ANY, \
                                        "model.pkl" ,size=(130,25))
        self.savefile_textctrl = savefile_textctrl
        sub_layout4 = wx.BoxSizer(wx.VERTICAL)
        sub_layout4.Add(stext4_1)
        sub_layout4.Add(openfile_textctrl)
        sub_layout4.Add(stext4_2)
        sub_layout4.Add(savefile_textctrl)
        sub_panel4.SetSizer(sub_layout4)

        ## sub_panel5
        stext5_1 = wx.StaticText(sub_panel5, wx.ID_ANY, "term criterion")
        term_textctrl = wx.TextCtrl(sub_panel5, wx.ID_ANY, "5000")
        self.term_textctrl = term_textctrl
        train_btn = wx.Button(sub_panel5, wx.ID_ANY, "Train, Save model")
        sub_layout5 = wx.BoxSizer(wx.VERTICAL)
        sub_layout5.Add(stext5_1)
        sub_layout5.Add(term_textctrl)
        sub_layout5.Add(train_btn)
        sub_panel5.SetSizer(sub_layout5)

        train_btn.Bind(wx.EVT_BUTTON, self.onTrainAndSaveClick)
        start_btn.Bind(wx.EVT_BUTTON, self.onGameStartClick)

        # pylearn2 items
        ann = None
        self.ann = ann

    def onTrainAndSaveClick(self, event):
        open_fpath = self.openfile_textctrl.GetValue()
        save_fpath = self.savefile_textctrl.GetValue()
        if open_fpath == "" or save_fpath == "":
            self.log_textctrl.Append("Please set open/save file name.")
            return
        
        data_set = TicTacToe(open_fpath)
        h0_irange = float(self.h0_irange_textctrl.GetValue())
        h0_bias = float(self.h0_bias_textctrl.GetValue())
        h0 = mlp.Sigmoid(layer_name='h0', dim=9, irange=h0_irange, init_bias=h0_bias)
        out_irange = float(self.out_irange_textctrl.GetValue())
        out = mlp.Softmax(layer_name='out', n_classes=9, irange=out_irange)
        term_ec = int(self.term_textctrl.GetValue())
        trainer = sgd.SGD(learning_rate=.05, batch_size=200, \
                          termination_criterion=EpochCounter(term_ec))
        self.ann = mlp.MLP([h0,out], nvis=9)
        self.trainingAndSaveModel(trainer, data_set, save_fpath)

    def onGameStartClick(self, event):
        model = self.model_textctrl.GetValue()
        if model == "":
            self.log_textctrl.Append("Please set model file name.")

        self.ann = self.loadModel(model)
        self.initialize()
        print str(self.game_radiobox.GetSelection())
        if self.game_radiobox.GetSelection() == 1:
            self.doComputer()

    def initialize(self):
        global gTurn
        gTurn = True
        for i in range(0,3):
            for j in range(0,3):
                self.cell_array[i][j].Enable()
                self.setCellState((i,j), gState["none"])
        
        self.log_textctrl.Clear()

    def loadModel(self, path):
        return serial.load(path)

    def saveModel(self, path, ann):
        serial.save(path, ann, on_overwrite='backup')

    def trainingAndSaveModel(self, trainer, data_set, save_path):
        trainer.setup(self.ann, data_set)
        while True:
            trainer.train(dataset=data_set)
            self.ann.monitor.report_epoch()
            self.ann.monitor()
            if trainer.continue_learning(self.ann) == False:
                self.saveModel(save_path, self.ann)
                break

    def doComputer(self):
        global gTurn
        board = []
        for i in range(0,3):
            for j in range(0,3):
                board.append(self.cell_array[i][j].state)

        cur_state = [board]
        next_move = self.getNextMove(np.array(cur_state))
        
        index = next_move.index(max(next_move))
        text = "Computer puts to " + str(index) + ".\n"
        self.log_textctrl.AppendText(text)
        if self.cell_array[index // 3][index % 3].state != gState["none"]:
            self.log_textctrl.AppendText("Error! Computer move is illegal.\n")
            for i in range(0,3):
                for j in range(0,3):
                    self.cell_array[i][j].Disable()
            return

        if gTurn == True:
            self.setCellState((index // 3, index % 3) , gState["O"])
        else:
            self.setCellState((index // 3, index % 3) , gState["X"])

        ret = self.judgeGameEnd()
        if ret == True:
            self.log_textctrl.AppendText("Game is End.\n")
            for i in range(0,3):
                for j in range(0,3):
                    self.cell_array[i][j].Disable()
            return
        else:
            gTurn = not gTurn

    def getNextMove(self, cur_state):
        next_move = [0,0,0,0,0,0,0,0,0]
        inputs = cur_state
        output = self.ann.fprop(theano.shared(inputs, name='inputs')).eval()
        print output[0]
        for i in range(0,9):
            if max(output[0]) == output[0][i]:
                next_move[i] = 3

        return next_move

    def setCellState(self, pos, state):
        self.cell_array[pos[0]][pos[1]].state = state
        if state == gState["O"]:
            self.cell_array[pos[0]][pos[1]].SetLabel("O")
        elif state == gState["X"]:
            self.cell_array[pos[0]][pos[1]].SetLabel("X")
        else:
            self.cell_array[pos[0]][pos[1]].SetLabel("")

    def judgeGameEnd(self):
        ret = False
        for i in range(0,3):
            if self.cell_array[i][0].state == gState["none"]:
                continue

            if (self.cell_array[i][0].state == self.cell_array[i][1].state) and \
               (self.cell_array[i][0].state == self.cell_array[i][2].state):
                ret = True

        for i in range(0,3):
            if self.cell_array[0][i].state == gState["none"]:
                continue

            if (self.cell_array[0][i].state == self.cell_array[1][i].state) and \
               (self.cell_array[0][i].state == self.cell_array[2][i].state):
                ret = True

        for stone in range(1,3):
            if (self.cell_array[0][0].state == stone and \
                self.cell_array[1][1].state == stone and \
                self.cell_array[2][2].state == stone) or \
               (self.cell_array[0][2].state == stone and \
                self.cell_array[1][1].state == stone and \
                self.cell_array[2][0].state == stone):
                ret = True

        none_cnt = 0
        for i in range(0,3):
            for j in range(0,3):
                if self.cell_array[i][j].state == gState["none"]:
                    none_cnt += 1
        if none_cnt == 0:
            ret = True

        return ret


class CellButton(wx.Button):
    def __init__(self, parent, panel, pos):
        wx.Button.__init__(self, panel, wx.ID_ANY, "")
        font = wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, \
                       wx.FONTWEIGHT_NORMAL)
        self.SetFont(font)

        state = gState["none"]
        self.state = state
        self.parent = parent
        self.pos = pos
        self.Bind(wx.EVT_BUTTON, self.onButtonClick)

    def onButtonClick(self, event):
        global gTurn
        if self.GetLabel == "O" or self.GetLabel == "X":
            return

        if gTurn == True:
            self.SetLabel("O")
            self.state = gState["O"]
        else:
            self.SetLabel("X")
            self.state = gState["X"]
        
        obj = event.GetEventObject()
        text = "Man puts to " + str(obj.pos) + ".\n"
        self.parent.log_textctrl.AppendText(text)
        ret = self.parent.judgeGameEnd()
        if ret == True:
            self.parent.log_textctrl.AppendText("Game is End.\n")
            for i in range(0,3):
                for j in range(0,3):
                    self.parent.cell_array[i][j].Disable()
            return
        else:
            gTurn = not gTurn

        self.parent.doComputer()


class TicTacToe(DenseDesignMatrix):
    def __init__(self, fname):
        X = []
        y = []
        X_temp = [0,0,0,0,0,0,0,0,0]  # 3x3 board
        y_temp = [0,0,0,0,0,0,0,0,0]  # 3x3 board

        self.class_names = ['0', '3']
        f = open(fname, "r")
        reader = csv.reader(f)

        for row in reader:
            if row[len(row)-1] == "win":
                winner_1st = True
            elif row[len(row)-1] == "lose":
                winner_1st = False
            else:
                continue
            print "winner_1st : " + str(winner_1st)
            for i, cell_index in enumerate(row):
                if cell_index == "win" or cell_index == "lose":
                    X_temp = [0,0,0,0,0,0,0,0,0]
                elif i % 2 == 0 and winner_1st == True:
                    y_temp[int(cell_index)] = 3
                    y.append(y_temp)
                    if i == 0:
                        X.append([0,0,0,0,0,0,0,0,0])
                    
                    print "X_temp = " + str(X_temp)
                    print "y_temp = " + str(y_temp)
                    X_temp[int(cell_index)] = 1
                    y_temp = [0,0,0,0,0,0,0,0,0]
                elif i % 2 == 0 and winner_1st == False:
                    temp = []
                    X_temp[int(cell_index)] = 1
                    for x in X_temp:
                        temp.append(x)
                    print "  temp = " + str(temp)
                    X.append(temp)
                elif winner_1st == True:
                    temp = []
                    X_temp[int(cell_index)] = 2
                    for x in X_temp:
                        temp.append(x)
                    X.append(temp)
                elif winner_1st == False:
                    X_temp[int(cell_index)] = 2
                    y_temp[int(cell_index)] = 3
                    print "y_temp = " + str(y_temp)
                    y.append(y_temp)
                    y_temp = [0,0,0,0,0,0,0,0,0]

        X = np.array(X)
        y = np.array(y)
        super(TicTacToe, self).__init__(X=X, y=y)


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame().Show()
    app.MainLoop()

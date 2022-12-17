# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 07:31:11 2022

@author: Fcs10
"""

# showbyvisdom.py
import numpy as np
import visdom


def show_loss(path, name, step=1):
    with open(path, "r") as f:
        data = f.read()
    data = data.split(" ")[:-1]
    x = np.linspace(1, len(data) + 1, len(data)) * step
    y = []
    for i in range(len(data)):
        y.append(float(data[i]))

    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win=name, opts={'title': name, "xlabel": "epoch", "ylabel": name})




if __name__ == "__main__":
    show_loss("./pth/stage1_loss.txt", "loss1")
    show_loss("./pth/stage2_loss.txt", "loss2")
    show_loss("./pth/stage2_top1_acc.txt", "acc1")
    show_loss("./pth/stage2_top5_acc.txt", "acc1")




# https://www.aipacommander.com/entry/2017/12/27/155711

#%%
import socket
import numpy as np
import cv2
import sys

from glob import glob
import re
import random
import tensorflow as tf
from uuid import uuid4
import operator
from deap import gp, base, algorithms, creator, tools
import math
import random
import pylab
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import chainer
import functools
from tqdm import tqdm
from tqdm import trange
import pygraphviz as pg
from inspect import isclass
from IPython import display
import os.path
import os
import pickle
from sklearn.model_selection import KFold

from pylib.sc_util import *
from pylib.gp_util import *
from pylib.hgp import *

#%%ノードの定義とか
GlobalStat.reset()
creator_init()
phase_init()

#%%文字列表現モデルを取得
n = len(os.listdir("./gp_knows_crx50ep/"))
fits = []
models = []
for index in range(n // 2):
    with open(f"./gp_knows_crx50ep/fitness_{index}.txt", "r") as f:
        fits.append(f.read())
    with open(f"./gp_knows_crx50ep/model_{index}.txt", "r") as f:
        models.append(f.read())
items = list(zip(models, fits))
items.sort(key=lambda x: x[1])
code = items[0][0]
# code = min(items, key=lambda x: len(str(x[0])))[0]

args = ",".join(arg for arg in GlobalStat.pset.arguments)
code = "lambda {args}: {code}".format(args=args, code=code)

#%%文字列からtfモデルへ
def create_model(self, main_graph, log=False): # dice loss
    self.log = log
    tf.reset_default_graph()

    self.inp = tf.placeholder(tf.float32, (None, 128, 128, 1), name="input")
    x = tf.contrib.slim.conv2d(self.inp, self.phase_info.begin.channel, 3)
    graph = main_graph(x)
    pred = tf.contrib.slim.conv2d(graph, 37, 3, activation_fn=None)
    self.pred = tf.nn.softmax(pred, axis=-1, name="pred")

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

a = eval(code, GlobalStat.pset.context, {})
model = PhaseModel(GlobalStat.phase)
create_model(model, a)

#%%load weight
saver = tf.train.Saver()
saver.restore(model.sess, "weights/crx25ep_best/model")
# saver.restore(model.sess, "weights/crx25ep_best_dice/model")

#%%
def draw_arrow(im, pt1, pt2, color, thickness=1, line_type=8, shift=0, w=2, h=4):
    vx = pt2[0] - pt1[0]
    vy = pt2[1] - pt1[1]
    v  = np.sqrt(vx ** 2 + vy ** 2)
    ux = vx / v
    uy = vy / v
    ptl = (int(pt2[0] - uy*w - ux*h), int(pt2[1] + ux*w - uy*h))
    ptr = (int(pt2[0] + uy*w - ux*h), int(pt2[1] - ux*w - uy*h))
    cv2.line(im, pt1, pt2, color, thickness, line_type, shift)
    cv2.line(im, pt2, ptl, color, thickness, line_type, shift)
    cv2.line(im, pt2, ptr, color, thickness, line_type, shift)

def zuti(img, raw):
    h, w = img.shape[:2]
    sz = 128
    draw = raw.copy()
    block = sz // img.shape[0]

    def arrow(draw, x, y, deg):
        cx = x * block + block // 2
        cy = y * block + block // 2
        draw_arrow(draw, (cx, cy), (
            int(cx + np.cos(np.radians(deg)) * 10),
            int(cy + np.sin(np.radians(deg)) * 10)
            ), (0))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j].argmax() != 36:
                deg = img[i, j].argmax() * 10
                arrow(draw, j, i, deg)
    return draw

def pred(frame):
    size = 148
    img = frame[ycenter - size: ycenter + size, xcenter - size + 30 : xcenter + size + 30]
    img = cv2.resize(img, (128, 128))
    raw = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
    img = np.float32(img <= 150)

    p = model.sess.run(model.pred, feed_dict={model.inp: [img]})
    return zuti(p[0], raw)

#%%

if len(sys.argv) < 3:
    print("python main.py <cam ip> <viewer ip>")
    sys.exit()

ip_address = sys.argv[1]
cap = cv2.VideoCapture(f"http://{ip_address}:4747/video")

to_send_addr = (sys.argv[2], 9999)

#%%
size = 128
resize_size = 128
while True:
    _, frame = cap.read()
    ycenter = frame.shape[0] // 2
    xcenter = frame.shape[1] // 2
    frame = cv2.resize(frame[ycenter - size: ycenter + size, xcenter - size : xcenter + size], (resize_size, resize_size))
    frame = pred(frame)

    jpg_str = cv2.imencode(".jpeg", frame)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for v in np.array_split(jpg_str[1], 10):
        udp.sendto(v.tostring(), to_send_addr)

    udp.sendto(b"__end__", to_send_addr)
    udp.close()

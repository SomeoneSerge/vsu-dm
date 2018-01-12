import pandas as pd
from pandas_datareader import data, wb
from datetime import datetime
import numpy as np
import graphviz
import random

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import bokeh.models
import bokeh.plotting as bk

def build_corpus(df, wnd_in, wnd_out):
    df = df.drop('Date', axis=1)
    # df = df.div(df.max())
    
    X, X_columns = [], []
    
    base = df.loc[:, ['Close', 'Low', 'High']]
    base = base.subtract(df.Close.shift(1), axis='index')
    
    X = pd.concat(
        [base.shift(k).add_suffix('_%s_ago' % k)
         for k in range(wnd_in)],
        axis=1,
    )
    X = X.fillna(0)
    
    y = pd.concat(
        [base.shift(-k).add_suffix('%s' % k)
         for k in range(1, wnd_out + 1)],
        axis=1
    )
    y = y.fillna(0)
    return X, y

def my_objective(y, y_pred, s_pred, alpha=.6):
    n = sum(y.data.shape)
    
    invcoef = Variable(torch.cumsum(torch.ones(*y.data.shape), dim=0))
    
    pos = F.tanh(y)
    neg = F.tanh(-y)
    ZERO = Variable(torch.Tensor([1e-14]))
    fee = (
        neg * torch.log(torch.max(s_pred[:, 0], ZERO))
        + pos * torch.log(torch.max(s_pred[:, 1], ZERO))
    )
    fee = (-fee/invcoef).sum()

    mse = torch.sum((y_pred - y)**2/invcoef)
    
    L = (1-alpha)*fee + alpha*mse
    return L

def loss(model, X, y,
         objective,
         optimizer,
        ):
    mem = model.init_mem()
    mean_L = 0
    X = torch.Tensor(X.as_matrix())
    Y = torch.Tensor(y.as_matrix())
    
    L = 0
    PERIOD = 32
    for i in range(X.shape[0]):
        x = Variable(X[i, :])
        y = Variable(Y[i, :], requires_grad=1)
        y_pred, s_pred, mem = model(x, mem)
        
        L += objective(y, y_pred, s_pred)
        
        if optimizer is None:
            continue
        if i == (X.shape[0] - 1) or i % PERIOD == 0:
            optimizer.zero_grad()
            L.backward(retain_graph=1)
            optimizer.step()
            mean_L += L.data
            L = 0
    mean_L = mean_L/X.shape[0]
    return mean_L[0]
        
def predictions(model, X, columns):
    mem = model.init_mem()
    pred = torch.zeros(X.shape[0], model.days_out)
    sign = torch.zeros(X.shape[0], model.days_out, 2)
    index = X.index
    X = torch.Tensor(X.as_matrix())
    for i in range(X.shape[0]):
        x = Variable(X[i, :])
        y, s, mem = model(x, mem)
        pred[i, :] = y.data
        sign[i, :, :] = s.data
    pred = pd.DataFrame(pred.numpy())
    pred.columns = columns
    pred.index = index
    return pred, sign

def plot_stocks(opn, cls,
                p, w=4, alpha=1.,
                line_color='gray',
                color_inc='#F2583E', color_dec='blue',
                legend_prefix='Stocks'):
    inc, dec = cls > opn, cls < opn

    
def evaluate_model(model,
                   X, y, base_price=0,
                   include_tclose=True,
                   include_pclose=True,
                   include_tlohi=False,
                   include_plohi=True,
                   title='Model evaluation'):
    W=1
    pred, sign = predictions(model, X, y.columns)
    opens = base_price + X.iloc[:, 0].cumsum()
    truth = opens + y.iloc[:, 0]
    pred_close = opens + pred.Close1
    pred_lo, pred_hi = opens + pred.Low1, opens + pred.High1
    tinc, tdec = truth > opens, truth < opens
    pinc, pdec = pred_close > opens, pred_close < opens
    
    p = bk.figure(
        plot_width=800, plot_height=600,
        title=title,
        active_scroll='wheel_zoom')
    if include_tclose:
            p.vbar(
                X.index[tinc], W,
                opens[tinc], truth[tinc],
                line_width=1,  line_color='black',
                fill_color='red', fill_alpha=.75,
                legend='Ground truth increase'
            )
            p.vbar(
                X.index[tdec], W,
                opens[tdec], truth[tdec],
                line_width=1, line_color='black',
                fill_color='blue', fill_alpha=.75,
                legend='Ground truth decrease'
            )
    
    if include_tlohi:
        p.vbar(
            opens.index, 1.1*W,
            opens + y.Low1, opens + y.High1,
            fill_color='gray', fill_alpha=.1,
            legend='Ground truth Lo-Hi'
        )
    if include_plohi:
        p.vbar(
            opens.index, W/2,
            pred_lo, pred_hi,
            fill_color='yellow', fill_alpha=.2,
            legend='Predicted Lo-Hi'
        )
    if include_pclose:
        p.segment(x0=X.index, y0=opens,
                  x1=X.index, y1=pred_close,
                  line_color='lightgray', line_width=2,
                  legend='Prediction')
        p.circle(X.index[pinc], pred_close[pinc],
                 fill_color='red', line_color='lightgray',
                 line_width=2, radius=.25*W,
                 legend='Predicted increase')
        p.circle(X.index[pdec], pred_close[pdec],
                 fill_color='blue', line_color='lightgray',
                 line_width=2, radius=.25*W,
                 legend='Predicted decrease')
    bk.show(p)
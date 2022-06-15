#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:37:54 2021

@author: carter
"""

import numpy as np
import subprocess
import os
import sys
import progressbar
import scipy.optimize as so
import scipy.interpolate as si
import warnings

### get arguments
def get_args():
    args = sys.argv
    argdict = {}
    N = len(args)
    for ii, arg in enumerate(args):
        if arg[:2] == '--' and ii+1 < N:
            value = args[ii+1]
            if value.isnumeric():
                value = int(value)
            elif value.lower() == 'none':
                value = None
            argdict[arg[2:]] = value
    return argdict

def args_to_string(args):
    out = ''
    for key in args.keys():
        out += ' --' + key + ' ' + str(args[key])
    return out

### create directory if not already created
def createDir(dirname):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            pass

### progress bar that I like
class bar(object):
    def __init__(self, N):
        self.bar = progressbar.ProgressBar(maxval=N, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Counter(), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
        self.bar.start()
    def update(self, ii):
        self.bar.update(ii)
    def finish(self):
        self.bar.finish()

def get_hostname():
    process = subprocess.Popen('hostname', stdout=subprocess.PIPE)
    output, error = process.communicate()
    return str(output)

def get_dirs():
    if not 'carter' in get_hostname():
        #maindir = '/Users/Carter/Documents/time-varying-betas/'
        maindir = '/home/cdavis40/time-varying-betas/'
    else:
        maindir = '/home/carter/Documents/ChicagoResearch/time-varying-betas/'
    datadir = maindir + 'data/'
    codedir = maindir + 'code/main/'
    logdir = maindir + 'log/'
    bashdir = maindir + 'bashfiles/'
    pythonpath = maindir + 'code/utils/'
    return {'maindir': maindir, 'datadir': datadir, 'codedir': codedir,
            'logdir': logdir, 'bashdir': bashdir, 'pythonpath': pythonpath}

def unique_xy(x, y):
    if len(x) < 2:
        return x, y
    xx, yy = [], []
    y_slope = (y[-1] - y[0])
    if y_slope > 0:
        x, y = x[::-1], y[::-1]
    for xi, yi in zip(x, y):
        if xi in xx or yi in yy:
            continue
        xx.append(xi)
        yy.append(yi)
    if y_slope > 0:
        xx, yy = xx[::-1], yy[::-1]
    return xx, yy

def force_increasing_x(x, y):
    N = len(x)
    if len(x) < 2:
        return x, y
    xx, yy = [x[0]], [y[0]]
    for ii in range(1, N):
        if x[ii] > xx[-1]:
            xx.append(x[ii])
            yy.append(y[ii])
    return xx, yy

def force_decreasing_x(x, y):
    N = len(x)
    if N < 2:
        return x, y
    xx, yy = [x[0]], [y[0]]
    for ii in range(1, N):
        if x[ii] < xx[-1]:
            xx.append(x[ii])
            yy.append(y[ii])
    return xx, yy

def clean_put_data(strikes, prices):
    strikes, prices = unique_xy(strikes, prices)
    strikes, prices = force_increasing_x(strikes, prices)
    prices, strikes = force_increasing_x(prices, strikes)
    strikes, prices = np.array(strikes), np.array(prices)
    return strikes, prices

def clean_call_data(strikes, prices):
    strikes, prices = unique_xy(strikes, prices)
    strikes, prices = force_increasing_x(strikes, prices)
    prices, strikes = force_decreasing_x(prices, strikes)
    strikes, prices = np.array(strikes), np.array(prices)
    return strikes, prices

def logfl(x):
    return np.log(np.maximum(x, 1e-100))

### this is the function log(exp(x) - 1)
def logexm1(x):
    out = logfl(np.exp(x) - 1.)
    if out.ndim == 0 and np.isfinite(out) == False:
        return x
    if out.ndim > 0:
        out[np.logical_not(np.isfinite(x))] = x[np.logical_not(np.isfinite(x))]
    return out

def log_xi_f(x, xi, theta):
    out = x * theta
    if np.isfinite(xi):
        out = xi * logfl(1. + theta * x / xi)
    if np.all(np.isfinite(np.exp(out))):
        return out
    return x * theta

def log_xi_func(x, xi, theta, zz = False):
    if zz == False:
        return log_xi_f(x, xi, theta)
    return logexm1(log_xi_f(x, xi, theta))

def anti_func(x, kappa, xi, theta, zz):
    if np.isfinite(xi) == False:
        if zz == False:
            return np.exp(kappa + theta * x) / theta
        return np.exp(kappa) * (np.exp(theta * x) / theta - x)
    if x == np.inf and xi < -1 and theta < 0 and zz == False:
        return 0.
    elif x == np.inf:
        return np.inf
    log_xi_part = log_xi_func(x, xi, theta, zz)
    ratio_part = 1. / theta
    if np.isfinite(xi) == False:
        ratio_part = (xi + theta * x) / (theta * (xi + 1))
    simple = np.exp(kappa + log_xi_part) * ratio_part
    if zz == False:
        return simple
    return simple - np.exp(kappa) * x

class fit_option(object):
    def __init__(self, strikes, prices, xi_upper_bnd = None, zz = False):
        self.strikes = strikes
        self.prices = prices
        self.xi_upper_bnd = xi_upper_bnd
        self.zz = zz
    def predict(self, strikes):
        return np.exp(self.kappa + log_xi_func(strikes, self.xi, self.theta, self.zz))
    def antiderivative(self, strike):
        return anti_func(strike, self.kappa, self.xi, self.theta, self.zz)
    def integral(self, a, b):
        return self.antiderivative(b) - self.antiderivative(a)
    def solve_kappa(self, xi, theta):
        return np.log(self.prices[-1]) - log_xi_func(self.strikes[-1], xi, theta, self.zz)
    def xi_equa(self, xi, theta):
        kappa = self.solve_kappa(xi, theta)
        return np.log(self.prices[0]) - kappa - log_xi_func(self.strikes[0], xi, theta, self.zz)
    def solve_xi(self, theta):
        x0, x1 = -1., -2.
        if self.prices[-1] > self.prices[0]:
            x0, x1 = 1., 2.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = so.root_scalar(f = self.xi_equa, args = (theta,), x0 = x0, x1 = x1)
        xi = sol.root
        if self.xi_upper_bnd is not None:
            xi = np.minimum(xi, self.xi_upper_bnd)
        if xi > 1e+10:
            xi = np.inf
        if xi < -1e+10:
            xi = -np.inf
        return xi
    def theta_function(self, theta):
        xi = self.solve_xi(theta)
        kappa = self.solve_kappa(xi, theta)
        return np.exp(kappa + log_xi_func(self.strikes, xi, theta, self.zz))
    def theta_error(self, theta):
        return np.mean(np.abs(self.theta_function(theta) - self.prices))
    def fit(self):
        bounds = (-1e+08, -1e-06)
        if self.prices[-1] > self.prices[0]:
            bounds = (1e-06, 1e+08)
        res = so.minimize_scalar(fun = self.theta_error, bounds = bounds, method = 'bounded')
        self.theta = res.x
        self.xi = self.solve_xi(self.theta)
        self.kappa = self.solve_kappa(self.xi, self.theta)
    def print_fits(self):
        print('Actual:', self.prices)
        print('Predicted:', self.predict(self.strikes))

class svix2(object):
    def __init__(self, put_strikes, put_prices, call_strikes, call_prices, spot):
        self.put_strikes = put_strikes
        self.put_prices = put_prices
        self.call_strikes = call_strikes
        self.call_prices = call_prices
        self.spot = spot
    def clean(self):
        self.put_strikes, self.put_prices = clean_put_data(self.put_strikes, self.put_prices)
        self.call_strikes, self.call_prices = clean_call_data(self.call_strikes, self.call_prices)
    def check(self):
        if len(self.put_strikes) > 3 and len(self.call_strikes) > 3:
            return True
        return False
    def fit(self):
        self.put_interp = si.UnivariateSpline(self.put_strikes, self.put_prices, k = 3, s = 0)
        self.call_interp = si.UnivariateSpline(self.call_strikes, self.call_prices, k = 3, s = 0)
        self.put_left = fit_option(self.put_strikes[:3], self.put_prices[:3], zz = True)
        self.put_left.fit()
        self.put_right = fit_option(self.put_strikes[-3:], self.put_prices[-3:])
        self.put_right.fit()
        self.call_left = fit_option(self.call_strikes[:3], self.call_prices[:3])
        self.call_left.fit()
        self.call_right = fit_option(self.call_strikes[-3:], self.call_prices[-3:], xi_upper_bnd = -2.)
        self.call_right.fit()
    def call_func(self, strike):
        strike = np.array(strike)
        if strike.ndim == 0:
            strike = np.expand_dims(strike, 0)
        out = np.zeros(len(strike))
        use_interp = np.logical_and(strike >= self.call_strikes[0], strike <= self.call_strikes[-1])
        use_left = strike < self.call_strikes[0]
        use_right = strike > self.call_strikes[-1]
        if np.sum(use_interp) > 0:
            out[use_interp] = self.call_interp(strike[use_interp])
        if np.sum(use_left) > 0:
            out[use_left] = self.call_left.predict(strike[use_left])
        if np.sum(use_right) > 0:
            out[use_right] = self.call_right.predict(strike[use_right])
        if len(out) == 1:
            return float(out)
        return out
    def put_func(self, strike):
        strike = np.array(strike)
        if strike.ndim == 0:
            strike = np.expand_dims(strike, 0)
        out = np.zeros(len(strike))
        use_interp = np.logical_and(strike >= self.put_strikes[0], strike <= self.put_strikes[-1])
        use_left = strike < self.put_strikes[0]
        use_right = strike > self.put_strikes[-1]
        if np.sum(use_interp) > 0:
            out[use_interp] = self.put_interp(strike[use_interp])
        if np.sum(use_left) > 0:
            out[use_left] = self.put_left.predict(strike[use_left])
        if np.sum(use_right) > 0:
            out[use_right] = self.put_right.predict(strike[use_right])
        if len(out) == 1:
            return float(out)
        return out
    def diff_func(self, strike):
        return self.put_func(strike) - self.call_func(strike)
    def linear_forward(self):
        N = len(self.put_strikes)
        ii = 1
        while ii < N-1 and self.put_prices[ii] < self.call_func(self.put_strikes[ii]):
            ii += 1
        strike1 = self.put_strikes[ii-1]
        strike2 = self.put_strikes[ii]
        put_price1 = self.put_prices[ii-1]
        put_price2 = self.put_prices[ii]
        call_price1 = float(self.call_interp(strike1))
        call_price2 = float(self.call_interp(strike2))
        put_slope = (put_price2 - put_price1) / (strike2 - strike1)
        put_intercept = put_price1 - put_slope * strike1
        call_slope = (call_price2 - call_price1) / (strike2 - strike1)
        call_intercept = call_price1 - call_slope * strike1
        forward = -(put_intercept - call_intercept) / (put_slope - call_slope)
        return strike1, strike2, forward
    def solve_forward(self):
        x0, x1, _ = self.linear_forward()
        sol = so.root_scalar(f = self.diff_func, x0 = x0, x1 = x1)
        self.forward = sol.root
        return self.forward
    def integrate_put(self, a, b):
        out = 0.
        if np.minimum(b, self.put_strikes[0]) > a:
            out += self.put_left.integral(a, np.minimum(b, self.put_strikes[0]))
        if np.minimum(b, self.put_strikes[-1]) > np.maximum(a, self.put_strikes[0]):
            out += self.put_interp.integral(np.maximum(a, self.put_strikes[0]), np.minimum(b, self.put_strikes[-1]))
        if b > np.maximum(a, self.put_strikes[-1]):
            out += self.put_right.integral(np.maximum(a, self.put_strikes[-1]), b)
        return out
    def integrate_call(self, a, b):
        out = 0.
        if np.minimum(b, self.call_strikes[0]) > a:
            out += self.call_left.integral(a, np.minimum(b, self.call_strikes[0]))
        if np.minimum(b, self.call_strikes[-1]) > np.maximum(a, self.call_strikes[0]):
            out += self.call_interp.integral(np.maximum(a, self.call_strikes[0]), np.minimum(b, self.call_strikes[-1]))
        if b > np.maximum(a, self.call_strikes[-1]):
            out += self.call_right.integral(np.maximum(a, self.call_strikes[-1]), b)
        return out
    def calc_svix2(self):
        self.clean()
        enough_obs = self.check()
        if enough_obs == False:
            return False
        self.fit()
        self.solve_forward()
        self.trf_svix2 = 2 * (self.integrate_put(0, self.forward) + self.integrate_call(self.forward, np.inf)) / self.spot**2
        return True
    def print_tail_fits(self):
        print('Left Put:')
        self.put_left.print_fits()
        print('Right Put:')
        self.put_right.print_fits()
        print('Left Call:')
        self.call_left.print_fits()
        print('Right Call:')
        self.call_right.print_fits()
        









# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:47:52 2018

@author: vinh
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special

# function that generates positive symmetric definite matrices
def PosSymDefMatrix(n,sd):
    M = np.matrix(np.random.rand(n,n))
    M = 0.5*(M + M.T)
    M = M + sd*np.eye(n)
    return M

NProperties = 4
NClasses = 6
NObjects = 150
symmetric_dirichlet = 1
distanceBTWClasses = 10

Mu = [np.random.random(NProperties)*i*distanceBTWClasses for i in range(1, NClasses + 1)]

# Generating symmetric positive semi-definite matrix for the covariances
sdDiff = 4
SDClass = np.random.rand(1, NClasses) + sdDiff
Cov = [PosSymDefMatrix(NProperties, i) for i in SDClass[0]]
#for i in range(NClasses):
#    print('the mean of the class %s is %s ' %(i+1, Mu[i]))
#for i in range(NClasses):
#    print('the covariance matrix of the class %s is: ' %(i+1))
#    print(Cov[i])

if symmetric_dirichlet == 1:
    theta = np.repeat(1/NClasses, NClasses)
else:
    a = np.ones(NClasses)
    rd = np.random.gamma(a, 1, NClasses)
    theta = rd/rd.sum()

#print('the probability of each class from 1 to ' + str(NClasses))
#print(theta)

r = np.random.multinomial(NObjects, theta)
#print('the number of Objects in each class is')
#print(r)

rAlln = [np.random.multivariate_normal(Mu[i], Cov[i], r[i]) for i in range(NClasses)]

# plot parameters
plotsize = 8
sizeMean = 10
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font = {'fontname':'Arial', 'size':'28'}
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
for i in range(NClasses):
      
    # the sd with ellipses
    # central point of the error ellipse
    pos = [Mu[i][0],Mu[i][1]]

    # for the angle we need the eigenvectors of the covariance matrix
    w,ve = np.linalg.eig(Cov[i][0:2,0:2])

    # We pick the largest eigen value
    order = w.argsort()[::-1]

    w = w[order]

    ve = ve[order]
    # we compute the angle towards the eigen vector with the largest eigen value
    thetaO = np.degrees(np.arctan(ve[1,0]/ve[0,0]))

    # Compute the width and height of the ellipse based on the eigen values (ie the length of the vectors)
    width, height = 2 * np.sqrt(w)

    # making the ellipse
    ellip = Ellipse(xy=pos, width=width, height=height, angle=thetaO)
    ellip.set_alpha(0.5)
    ellip.set_facecolor(color[i])
             
    ax.plot(rAlln[i][:,0],rAlln[i][:,1], '.', c=color[i], markersize=sizeMean)
    ax.add_artist(ellip)
        
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Property 2', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('Classes based on the first two properties', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize,plotsize)

plt.savefig('TheFirstTwoParameters.png')

# putting the generated data into an array form
y = np.zeros((1,4))
for i in range(NClasses):
    y = np.vstack((y, rAlln[i]))

y = y[1:]
# getting the true classes of the points
v_true = np.zeros((1))
for i, j in enumerate(r):
    v_true = np.hstack((v_true, np.repeat(i+1, j)))

v_true = np.array([v_true[1:]])

y_true = np.concatenate((y, v_true.T), axis=1)
np.random.shuffle(y_true)
y = y_true[:,0:-1]
#print('the data')
#print(y)

plot_size = 8
size_mean = 10
text_size = 16
axis_font = {'fontname': 'Arial', 'fontsize': '24'}
title_font = {'fontname': 'Arial', 'fontsize': '28'}

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y[:,0], y[:,1], 'k.', markersize=size_mean)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Property 2', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('Classes based on the first two parameter', y=1.08, **title_font)
ax.figure.set_size_inches(plot_size, plot_size)
plt.savefig('before clustering.png')

# for random classes
v = np.array([random.randint(1, NClasses + 1) for i in range(y.shape[0])])

broadness = 2

initMu = np.empty([NClasses, NProperties])
initCov = np.empty([NClasses, NProperties, NProperties])

for j in range(NClasses):
    initMu[j] = np.random.random(NProperties)*np.amax(y, axis=0)
    initCov[j,:,:] = np.mean(np.array(Cov), axis=0) + broadness
print(initCov.shape)
initW = theta

def EStep(y, w, mu, cov):
    to_nk = []
    for i in range(NObjects):
        a = np.array([w[j]*sc.stats.multivariate_normal.pdf(y[i], mu[j], cov[j]) for j in range(NClasses)])
        b = np.array([a[j]/a.sum() for j in range(w.shape[0])])
        to_nk.append(b)
    to_nk = np.array(to_nk)
    return to_nk
r_n = EStep(y, initW, initMu, initCov)

def MStep(r, y, mu):
    new_mu = np.dot(r_n.T, y)
    new_mu = np.array([new_mu[i]/r.sum(0)[i] for i in range(NClasses)])
    cov1 = []
    for k in range(NClasses):
        a = np.array([np.outer(y[i]-mu[k],y[i]-mu[k])*r[i,k] for i in range(NObjects)])
        b = np.zeros((4,4))
        for m in range(a.shape[0]):
            b += a[m]
        b = 1/np.sum(r, axis=0)[k]*b
        cov1.append(b)
    new_cov = np.array(cov1)
    new_w = np.array([r.sum(0)[i]/NObjects for i in range(NClasses)])
    return new_w, new_mu, new_cov




w_n,mu_n,cov_n = MStep(r_n, y, initMu)


Inititeration = 20
EMiteration = 10
lookLH = 20

for init in range(Inititeration):
#    starting value
    initMu = np.empty([NClasses, NProperties])
    for j in range(NClasses):
        initMu[j,:] = np.random.random(NProperties)*np.amax(y, axis=0)
    
    r_n = EStep(y, initW, initMu, initCov)
    w_n, mu_n, cov_n = MStep(r_n, y, initMu)
    print('not found bug yet')
    if init == 0:
        logLH = -1000000000000
    for i in range(EMiteration):
        #Estep
        r_n = EStep(y, w_n, mu_n, cov_n)
        #MStep
        w_n, mu_n, sigma_n = MStep(r_n, y, mu_n)
        if i < 3:
            print('w_n')
            print(w_n)
            print('mu_n')
            print(mu_n)
            print('cov_n')
            print(cov_n)
        
        #compute loglikelihood
        logLall = np.zeros((y.shape[0]))
        
        for Object in range(y.shape[0]):
            LH = np.zeros(NClasses)
            for jClasses in range(NClasses):
                print('cov jclasses')
                print(cov_n[jClasses])
                LH[jClasses] = w_n[jClasses]*sc.stats.multivariate_normal.pdf(y[Object,:], mu_n[jClasses,:], cov_n[jClasses])
            logLall[Object] = np.log(np.sum(LH))
        logL = np.sum(logLall)
        if i > EMiteration - lookLH:
            print(logL)
    if logL > logLH:
        logLH = logL
        print('found larger: ', logLH)
        w_p = w_n
        mu_p = mu_n
        sigma_p = sigma_n
        r_p = r_n
        
# plot parameters
plotsize = 8
sizeMean = 10
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font = {'fontname':'Arial', 'size':'28'}
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.plot(y[:,0],y[:,1], 'k.', markersize=sizeMean)
for i in range(NClasses):
      
    # the sd with ellipses
    # central point of the error ellipse
    pos = [mu_p[i,0],mu_p[i,1]]

    # for the angle we need the eigenvectors of the covariance matrix
    w,ve = np.linalg.eig(sigma_p[i,0:2,0:2])

    # We pick the largest eigen value
    order = w.argsort()[::-1]
    w = w[order]
    ve = ve[:,order]

    # we compute the angle towards the eigen vector with the largest eigen value
    thetaO = np.degrees(np.arctan(ve[1,0]/ve[0,0]))

    # Compute the width and height of the ellipse based on the eigen values (ie the length of the vectors)
    width, height = 2 * np.sqrt(w)

    # making the ellipse
    ellip = Ellipse(xy=pos, width=width, height=height, angle=thetaO)
    ellip.set_alpha(0.5)
    ellip.set_facecolor(color[i])
             
    ax.add_artist(ellip)
        
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Property 2', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('The inferred classes based on the first two properties', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize,plotsize)
plt.show()


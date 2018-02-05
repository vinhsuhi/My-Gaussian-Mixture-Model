import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special

#random.seed(10)
# parameters
NClasses = 4
NObjects = 200
NProperties = 1
symmetric_dirichlet = 1
disBTWclasses = 20
diffBTWspreadOFClass = 2

# the mean vector
Mu = [np.random.random(NProperties) * disBTWclasses * i
    for i in range(1, NClasses + 1)]
# the sd vector
Var = [np.random.random(NProperties) * diffBTWspreadOFClass * i
    for i in range(1, NClasses + 1)]
print('real mean')
print(Mu)
if symmetric_dirichlet == 1:
    alpha = np.repeat(1.0 / NClasses, NClasses)

else:
    a = np.ones(NClasses)
    alpha = np.random.gamma(a)
    alpha = np.divide(alpha, alpha.sum())

# number of point in each class is denoted by a vector called r
r = np.random.multinomial(NObjects, alpha)

# generate data
rAlln = [np.random.normal(Mu[i], Var[i], r[i]) for i in range(0, NClasses)]
# plot_parameters
color = ['b', 'g', 'r', 'c']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0, NClasses):
    ax.hist(rAlln[i], 10, color=color[i])
ax.set_ylabel('Frequency')
ax.set_xlabel('Property 1')
ax.set_title('Classes')
plt.savefig('lol1.png')



# putting the generated data into an array form
y = rAlln[0]
for i in range(NClasses-1):
    y = np.hstack((y,rAlln[i+1]))

# Getting the true classes of the points 
v_true = np.zeros((1)) 
#print(r)
for i,j in enumerate(r):
    v_true = np.hstack((v_true, np.repeat(i+1, j)))
# v_true is true label
v_true = np.array(v_true[1:])
y_true = np.vstack((y, v_true))

# random shuffle the data points
np.random.shuffle(y_true.T)
y = y_true[0]
print(y_true[1])
#print(np.sort(y))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(y, 40)
ax.set_ylabel('Frequency')
ax.set_xlabel('Property 1')
ax.set_title('Classes')
plt.savefig('data_histogram.png')

# Defining V as a latent variable denoting the classes.V is set to random initially
v = np.array([random.randint(1, NClasses + 1) for i in range(y.shape[0])])

broadness = 15
initMu = np.random.random(NClasses)*max(y)
initVar = np.random.random(NClasses)+broadness
initPi = alpha

# Functions implementing the EM steps
def EStep(y, Pi, Mu, Sigma):
# init 200x4 dimensions vector
    Tau_kn = np.zeros((y.shape[0], Mu.shape[0]))
# 
    for i in range(y.shape[0]): 
        Tau_kn_sumk = np.zeros(Mu.shape[0])
        for j in range(Mu.shape[0]):
            Tau_kn_sumk[j] = Pi[j]*sc.stats.norm.pdf(y[i], Mu[j], np.sqrt(Sigma[j]))
        for j in range(Mu.shape[0]):
            Tau_kn[i,j] = Tau_kn_sumk[j] / np.sum(Tau_kn_sumk)
    return Tau_kn

r_n = EStep(y, initPi, initMu, initVar)
#print(r_n.reshape(4,200))

def MStep(Tau, y, Mu):
#    our aim is finding the new Mu, Var, and Pi
    Mu_new = np.dot(Tau.T, y) / Tau.sum(0)
    Pi_new = Tau.sum(0) / y.shape[0]
    Var_new = np.zeros(Mu.shape[0])    
    temp_matrix = np.array([y - Mu[i] for i in range(Mu.shape[0])])
    temp_matrix = np.square(temp_matrix)
    for i in range(Mu.shape[0]):
        Var_new[i] = np.dot(Tau.T[i], temp_matrix[i])
    Var_new = Var_new.reshape(4,) / Tau.sum(0)
    return Pi_new, Mu_new, Var_new

pi_n, Mu_n, Var_n = MStep(r_n, y, initMu)



init_iteration = 10
EM_iteration = 200  
look_LH = 20

for init in range(init_iteration):
    initMu = np.random.random(NClasses)*max(y)
    r_n = EStep(y, initPi, initMu, initVar)
    pi_n, Mu_n, Var_n = MStep(r_n, y, initMu)
    if init == 0:
        logLH = -1000000000000
    for i in range(EM_iteration):
#       E_step
        r_n = EStep(y, pi_n, Mu_n, Var_n)
#       M_step
        pi_n, Mu_n, Var_n = MStep(r_n, y, Mu_n)
#       Compute LogLikelihood
        logLall = np.zeros((y.shape[0]))
        for Objects in range(y.shape[0]):
            px = np.array([sc.stats.norm.pdf(y[Objects], Mu_n[j], np.sqrt(Var_n[j]))
                            for j in range(NClasses)])
            LH = np.dot(pi_n, px)
            logLall[Objects] = np.log(LH)
        logL = np.sum(logLall)
        
        if i > EM_iteration - look_LH:
            print(logL)
    if logL > logLH:
        logLH = logL
        print('found larger ', logLH)
        pi_p = pi_n
        Mu_p = Mu_n
        Var_p = Var_n
        r_p = r_n

# let plot the result
Mu_inf = np.sort(Mu_p)
Mu = np.array(Mu).reshape(4,)
Mu_true = np.sort(Mu)

Var_inf = np.sort(Var_p)
Var = np.array(Var).reshape(4,)
Var_true = np.sort(Var)
print(Mu_inf)
print(Mu_true)

# the figures
# parameter 

plotsize = 11
sizeMean = 20
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font = {'fontname':'Arial', 'size':'28'}
x = range(1,NClasses+1)
startx = 0
endx = 5
stepsizex = 1
starty = -2
endy = max(y)
stepsizey = 10

# figure
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
#means
ax1.plot(x, Mu_inf, 'k.', markersize=sizeMean, label='Learned')
ax1.plot(x, Mu_true, 'r.', markersize=sizeMean, label='True')
#means
ax2.plot(x, Var_inf, 'k.', markersize=sizeMean, label='Learned')
ax2.plot(x, Var_true, 'r.', markersize=sizeMean, label='True')

for label in (ax1.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks(np.arange(startx, endx, stepsizex))
ax1.yaxis.set_ticks(np.arange(starty, endy, stepsizey))
ax1.set_xlim([startx, endx])
ax1.set_ylim([starty, endy])
ax1.set_ylabel('Mean', **axis_font)
ax1.legend(loc='upper left',fontsize=text_size-6)
ax1.set_title('Mean', y=1.08, **Title_font)
ax1.figure.set_size_inches(plotsize,plotsize)

for label in (ax2.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks(np.arange(startx, endx, stepsizex))
ax2.yaxis.set_ticks(np.arange(starty, endy, stepsizey))
ax2.set_xlim([startx, endx])
ax2.set_ylim([starty, endy])
ax2.set_ylabel('Variance', **axis_font)
ax2.legend(loc='upper left',fontsize=text_size-6)
ax2.set_title('Variance', y=1.08, **Title_font)
ax2.figure.set_size_inches(plotsize,plotsize)

plt.suptitle('Comparing the true parameters to the inferred parameters',**Title_font)
fig.subplots_adjust(top=0.85)
plt.savefig('compare.png')

print('The infered parameters')
print('Mixing proportion', pi_p)
print('Mean', Mu_p)
print('Variances', Var_p)

print('The true patameters')
print('Mixing proportion', r)
print('Mean', Mu)
print('Variances', Var)

GoodOrder = [2,1,0,3]
r_ordered = r_p[:,GoodOrder] 
infClusters = np.argmax(r_ordered, axis=1)
Clustering = y_true[1]==infClusters+1

count = 0
for i in range(y.shape[0]):
    for j in range(NClasses):
        if r_p[i][j] == np.max(r_p[i]) and float(j+1) == y_true[1][i]:
            count += 1

        

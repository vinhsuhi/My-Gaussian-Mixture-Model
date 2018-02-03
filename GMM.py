import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special

random.seed(10)
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




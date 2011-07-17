"""
System identification example, using a subspace method and nuclear norm 
optimization.  The input/output data are from DaISy's CD player arm 
example <http://homes.esat.kuleuven.be/~smc/daisy/>.
"""

import pickle, pylab, sysid
from cvxopt.base import matrix

data = pickle.load(open("CD_player_arm.bin", "r"))

# number of points for identification
N = 100

# number of points for verification
Nt = 400

# input u and output y
u = data['u'][:,:Nt]
y = data['y'][:,:Nt]

# number of inputs and outputs
m = u.size[0]
p = y.size[0]

# time index
t = matrix(xrange(Nt),(1,Nt))

# approximately the 1-sigma noise level
vsig = 0.27;

# call the system identification function
sol = sysid.sysid(y[:,:N],u[:,:N],vsig)

# extract variables
A = sol['A']
B = sol['B']
C = sol['C']
D = sol['D']
x0 = sol['x0']
svN = sol['svN']
sv = sol['sv']

# compute the estimated outputs ye
ye = matrix(0.0,(p,Nt))
xi = +x0
for ii in xrange(Nt):
    ye[:,ii] = C * xi + D * u[:,ii]
    xi = A * xi + B * u[:,ii] 

# plot singular values
pylab.figure(1)
nsv = 30
a = pylab.semilogy(range(1,nsv+1), svN[:nsv]/svN[0],'-o',
    range(1,nsv+1), sv[:nsv]/sv[0],'-o')
pylab.axis([0, nsv, 1e-8, 1])
ax = pylab.gca()
pylab.setp(ax, yticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e0])
pylab.xlabel('Singular value index')
pylab.ylabel('Normalized singular values')
pylab.legend(a, ('Original', 'Optimized'), loc = 'best')

pylab.figure(2)

for ii in range(p):
    pylab.subplot(p,1,ii+1);
    pylab.plot(t.T,y[ii,:].T,t.T,ye[ii,:].T)
    pylab.hold(True)
    pylab.plot([N,N],pylab.getp(pylab.gca(),'ylim'))
    pylab.hold(False)
    pylab.xlabel('Time t');
    pylab.ylabel('Output y'+str(ii))
pylab.legend(a, ('Measured','Estimated'), loc = 'best');

pylab.show()

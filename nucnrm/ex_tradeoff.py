"""
Demonstration of an efficient method for computing the tradeoff curve
between the nuclear norm || A(x) + B ||_* and the fitting error 
|| x-x0 ||^2, when the regularized nuclear norm approximation

    minimize    || A(x) + B ||_* + lambda * || x-x0 ||^2          (1)
    
is solved for a range of values of lambda.

The key idea is based on the fact that when (1) is solved for a 
particular lambda, the slope of the tradeoff curve at this point 
equals to -lambda.  Hence, when two points on the tradeoff curve 
are computed, we can compute the next point by letting -lambda equal 
to the slope of the line connecting the two points.
"""

import pylab, math, pickle, nucnrm, sysid 
from cvxopt import base, blas, lapack, solvers
from cvxopt.base import normal, matrix, spmatrix, mul

maxiter = 8

data = "random"
#data = "sysid"
solvers.options['show_progress'] = False

if data is "random":
    # Generate random problem data
    base.setseed(0)

    p, q, n = 20, 20, 40
    A = normal(p*q, n)
    B = normal(p, q)
    x0 = normal(n, 1)

elif data is "sysid":
    iddata = pickle.load(open("CD_player_arm.bin","r"))
    N = 100
    u = iddata['u'][:,:N]
    y = iddata['y'][:,:N]
    m, N, p = u.size[0], u.size[1], y.size[0]
    r = min(int(30/p),int((N+1.0)/(p+m+1)+1.0))
    a = r*p
    c = r*m
    b = N-r+1
    d = b-c
    U = sysid.Hankel(u,r,b,p=m,q=1)
    Vt = matrix(0.0,(b,b))
    Stemp = matrix(0.0,(c,1))
    Un = matrix(0.0,(b,d))
    lapack.gesvd(U,Stemp,jobvt='A',Vt=Vt)
    Un[:,:] = Vt.T[:,c:]

    AA = sysid.Hankel_basis(r,b,p=p,q=1)
    A = matrix(0.0,(a*d,p*N))
    temp = spmatrix([],[],[],(a,b),'d')
    temp2 = matrix(0.0,(a,d))
    for ii in xrange(p*N):
        temp[:] = AA[:,ii]
        base.gemm(temp,Un,temp2)
        A[:,ii] = temp2[:]
    B = matrix(0.0,(a,d))
   
    # flip the matrix if columns is more than rows
    if a < d:
        Itrans = [i+j*a for i in xrange(a) for j in xrange(d)]
        B[:] = B[Itrans]
        B.size = (d,a)
        for ii in xrange(p*N):
            A[:,ii] = A[Itrans,ii]
    
    n = p*N
    if a < d:
        p, q = d, a
    else:
        p, q = a, d
    x0 = y[:]
    
S = matrix(0.0, (q,1))

# First, compute two initial points on the tradeoff curve
lamT = [50.0, 0.02]
nrmT = [0.0, 0.0]
errT = [0.0, 0.0]

for i in xrange(len(lamT)):
    Cd = matrix(2.0*lamT[i], (n,1))
    C = base.spdiag(Cd)
    d = -base.mul(Cd, x0)
    
    sol = nucnrm.nrmapp(A, B, C = C, d = d)
    x = sol['x']
    
    lapack.gesvd(matrix(A*x, (p,q)) + B, S)
    nrmT[i] = sum(S)
    errT[i] = blas.dot(x-x0, x-x0)
   
# plot the tradeoff curve upper/lower bounds with the initial 2 points
pylab.figure(0)
N = 200
slope = -matrix(lamT)
errM = matrix(errT)
nrmM = matrix(nrmT)
xx = matrix(range(N))*((errM[-1]-errM[0])/(N-1))+errM[0]
yy = (slope*xx.T + (nrmM-mul(slope,errM))*matrix(1.0, (1,N))).T
yymax = matrix([max(yy[i,:]) for i in xrange(yy.size[0])])

a = pylab.plot(errM, nrmM, '--', xx, yymax, '-', 
    errM, nrmM, 'o', linewidth = 1.5)
pylab.xlabel('Fitting error')
pylab.ylabel('Nuclear norm')
#pylab.legend(a, ('Upper bound', 'Lower bound'), loc = 'NorthEast')
#pylab.show()


def tri_area(x1, y1, s1, x2, y2, s2):
    """
    # Calculate the area of a triangle
    
    INPUT
    x1, x2      coordinates on the x-axis of point 1 and 2
    y1, y2      coordinates on the y-axis of point 1 and 2
    s1, s2      slopes at point 1 and 2
    """

    x3 = (y1-y2+s2*x2-s1*x1)/(s2-s1)
    y3 = y2+s2*(x3-x2)
    
    # Heron's formula
    a = math.sqrt((x1-x2)**2+(y1-y2)**2)
    b = math.sqrt((x2-x3)**2+(y2-y3)**2)
    c = math.sqrt((x1-x3)**2+(y1-y3)**2)
    s=(a+b+c)/2
    
    return math.sqrt(s*(s-a)*(s-b)*(s-c))


# Iteratively generate the next point on the tradeoff curve
it = 1
while it <= maxiter:
    it = it + 1
    j = 0
    i = 0
    area = tri_area(errT[i], nrmT[i], -lamT[i], errT[i+1], nrmT[i+1], 
        -lamT[i+1])
    for i in xrange(1, len(lamT)-1):
        areaN = tri_area(errT[i], nrmT[i], -lamT[i], errT[i+1], 
            nrmT[i+1], -lamT[i+1])
        if areaN > area:
            area = areaN
            j = i
    
    newlam = -(nrmT[j]-nrmT[j+1])/(errT[j]-errT[j+1])
    Cd = matrix(2.0*newlam, (n,1))
    C = base.spdiag(Cd)
    d = -base.mul(Cd, x0)
    
    sol = nucnrm.nrmapp(A, B, C = C, d = d)
    x = sol['x']
    
    lapack.gesvd(matrix(A*x, (p,q)) + B, S)
    lamT.insert(j+1,newlam)
    nrmT.insert(j+1,sum(S))
    errT.insert(j+1,blas.dot(x-x0, x-x0))
    
    # update tradeoff curve upper and lower bounds
    pylab.figure(it-1)
    N = 200
    slope = -matrix(lamT)
    errM = matrix(errT)
    nrmM = matrix(nrmT)
    xx = matrix(range(N))*((errM[-1]-errM[0])/(N-1))+errM[0]
    yy = (slope*xx.T + (nrmM-mul(slope,errM))*matrix(1.0, (1,N))).T
    yymax = matrix([max(yy[i,:]) for i in xrange(yy.size[0])])

    a = pylab.plot(errM, nrmM, '--', xx, yymax, '-', 
        errM, nrmM, 'o', linewidth = 1.5)
    pylab.xlabel('Fitting error')
    pylab.ylabel('Nuclear norm')
    #pylab.legend(a, ('Upper bound', 'Lower bound'), loc = 'NorthEast')
    #pylab.show()

pylab.show() 


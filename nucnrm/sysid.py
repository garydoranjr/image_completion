"""
System identification by regularized nuclear norm approximation.
"""
 
# Copyright 2009 Z. Liu and L. Vandenberghe.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details.
#
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from cvxopt import base, blas, lapack
from cvxopt.base import matrix, spmatrix
from nucnrm import nrmapp

def sysid(y, u, vsig, svth = None):

    """
    System identification using the subspace method and nuclear norm 
    optimization.  Estimate a linear time-invariant state-space model 
    given inputs and outputs.  The algorithm is described in [1].
    

    INPUT
    y       'd' matrix of size (p, N).  y are the measured outputs, p is 
            the number of outputs, and N is the number of data points 
            measured. 
    
    u       'd' matrix of size (m, N).  u are the inputs, m is the number 
            of inputs, and N is the number of data points.
    
    vsig    a weighting parameter in the nuclear norm optimization, its 
            value is approximately the 1-sigma output noise level
    
    svth    an optional parameter, if specified, the model order is 
            determined as the number of singular values greater than svth 
            times the maximum singular value.  The default value is 1E-3 
    
    OUTPUT
    sol     a dictionary with the following words
            -- 'A', 'B', 'C', 'D' are the state-space matrices
            -- 'svN', the original singular values of the Hankel matrix
            -- 'sv', the optimized singular values of the Hankel matrix
            -- 'x0', the initial state x(0)
            -- 'n', the model order

    [1] Zhang Liu and Lieven Vandenberghe. "Interior-point method for 
        nuclear norm approximation with application to system 
        identification."  

    """

    m, N, p = u.size[0], u.size[1], y.size[0]
    if y.size[1] != N:
        raise ValueError, "y and u must have the same length"
           
    # Y = G*X + H*U + V, Y has size a x b, U has size c x b, Un has b x d
    r = min(int(30/p),int((N+1.0)/(p+m+1)+1.0))
    a = r*p
    c = r*m
    b = N-r+1
    d = b-c
    
    # construct Hankel matrix Y
    Y = Hankel(y,r,b,p=p,q=1)
    
    # construct Hankel matrix U
    U = Hankel(u,r,b,p=m,q=1)
    
    # compute Un = null(U) and YUn = Y*Un
    Vt = matrix(0.0,(b,b))
    Stemp = matrix(0.0,(c,1))
    Un = matrix(0.0,(b,d))
    YUn = matrix(0.0,(a,d))
    lapack.gesvd(U,Stemp,jobvt='A',Vt=Vt)
    Un[:,:] = Vt.T[:,c:]
    blas.gemm(Y,Un,YUn)
    
    # compute original singular values
    svN = matrix(0.0,(min(a,d),1))
    lapack.gesvd(YUn,svN)
    
    # variable, [y(1);...;y(N)]
    # form the coefficient matrices for the nuclear norm optimization
    # minimize | Yh * Un |_* + alpha * | y - yh |_F
    AA = Hankel_basis(r,b,p=p,q=1)
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
      
    # regularized term
    x0 = y[:]
    Qd = matrix(2.0*svN[0]/p/N/(vsig**2),(p*N,1))

    # solve the nuclear norm optimization
    sol = nrmapp(A, B, C = base.spdiag(Qd), d = -base.mul(x0, Qd))
    status = sol['status']
    x = sol['x']
    
    # construct YhUn and take the svd
    YhUn = matrix(B)
    blas.gemv(A,x,YhUn,beta=1.0)
    if a < d:
        YhUn = YhUn.T
    Uh = matrix(0.0,(a,d))
    sv = matrix(0.0,(d,1))
    lapack.gesvd(YhUn,sv,jobu='S',U=Uh)

    # determine model order
    if svth is None:
        svth = 1E-3
    svthn = sv[0]*svth
    n=1
    while sv[n] >= svthn and n < 10:
        n=n+1
    
    # estimate A, C
    Uhn = Uh[:,:n]
    for ii in xrange(n):
        blas.scal(sv[ii],Uhn,n=a,offset=ii*a)
    syseC = Uhn[:p,:]
    Als = Uhn[:-p,:]
    Bls = Uhn[p:,:]
    lapack.gels(Als,Bls)
    syseA = Bls[:n,:]
    Als[:,:] = Uhn[:-p,:]
    Bls[:,:] = Uhn[p:,:]
    blas.gemm(Als,syseA,Bls,beta=-1.0)
    Aerr = blas.nrm2(Bls)
    
    # stabilize A
    Sc = matrix(0.0,(n,n),'z')
    w = matrix(0.0, (n,1), 'z')
    Vs = matrix(0.0, (n,n), 'z')
    def F(w):
        return (abs(w) < 1.0)
    
    Sc[:,:] = syseA
    ns = lapack.gees(Sc, w, Vs, select = F)
    while ns < n:
        #print "stabilize matrix A"
        w[ns:] = w[ns:]**-1
        Sc[::n+1] = w
        Sc = Vs*Sc*Vs.H
        syseA[:,:] = Sc.real()
        Sc[:,:] = syseA
        ns = lapack.gees(Sc, w, Vs, select = F)

    # estimate B,D,x0 stored in vector [x0; vec(D); vec(B)]
    F1 = matrix(0.0,(p*N,n))
    F1[:p,:] = syseC
    for ii in xrange(1,N):
        F1[ii*p:(ii+1)*p,:] = F1[(ii-1)*p:ii*p,:]*syseA
    F2 = matrix(0.0,(p*N,p*m))
    ut = u.T
    for ii in xrange(p):
        F2[ii::p,ii::p] = ut
    F3 = matrix(0.0,(p*N,n*m))
    F3t = matrix(0.0,(p*(N-1),n*m))
    for ii in xrange(1,N):
        for jj in xrange(p):
            for kk in xrange(n):
                F3t[jj:jj+(N-ii)*p:p,kk::n] = ut[:N-ii,:]*F1[(ii-1)*p+jj,kk]
        F3[ii*p:,:] = F3[ii*p:,:] + F3t[:(N-ii)*p,:]
    
    F = matrix([[F1],[F2],[F3]])
    yls = y[:]
    Sls = matrix(0.0,(F.size[1],1))
    Uls = matrix(0.0,(F.size[0],F.size[1]))
    Vtls = matrix(0.0,(F.size[1],F.size[1]))
    lapack.gesvd(F, Sls, jobu='S', jobvt='S', U=Uls, Vt=Vtls)
    Frank=len([ii for ii in xrange(Sls.size[0]) if Sls[ii] >= 1E-6])
    #print 'Rank deficiency = ', F.size[1] - Frank
    xx = matrix(0.0,(F.size[1],1))
    xx[:Frank] = Uls.T[:Frank,:] * yls
    xx[:Frank] = base.mul(xx[:Frank],Sls[:Frank]**-1)
    xx[:] = Vtls.T[:,:Frank]*xx[:Frank] 
    blas.gemv(F,xx,yls,beta=-1.0)
    xxerr = blas.nrm2(yls)
    
    x0 = xx[:n]
    syseD = xx[n:n+p*m]
    syseD.size = (p,m)
    syseB = xx[n+p*m:]
    syseB.size = (n,m)
    
    return {'A': syseA, 'B': syseB, 'C': syseC, 'D': syseD, 'svN': svN, 'sv': \
        sv, 'x0': x0, 'n': n, 'Aerr': Aerr, 'xxerr': xxerr}
        



def Hankel(x, m, n, p = 1, q = 1):

    """
    Return the block Hankel matrix H, a 'd' matrix of size (m*p,n*q),
    
            [ h_1   h_2     ...  h_n       ]
        H = [ h_2   h_3     ...  h_{n+1}   ]
            [  :     :            :        ]
            [ h_m   h_{m+1} ...  h_{m+n-1} ],
    
    where h_i, i=1...(m+n-1), are the Hankel blocks of size (p,q).
    
    INPUT
    x       'd' matrix of size (p*q*(m+n-1),1).  The Hankel blocks,
                
                [ h_1, h_2, ..., h_{m+n-1} ],
                
            are stored in x in column-major order
    
    m       number of block rows
    
    n       number of block columns
    
    p       number of rows of a Hankel block
    
    q       number of columns of a Hankel block    
    """
    
    if x.size[0]*x.size[1] != p*q*(m+n-1):
        raise ValueError, "x.size[0]*x.size[1] must equal to p*q*(m+n-1)"
    
    x.size = (p, q*(m+n-1))
    
    H = matrix(0.0, (m*p, n*q))
    for i in xrange(m):
        H[i*p:(i+1)*p, :] = x[:,i*q:(i+n)*q]
    
    return H
    
def Hankel_basis(m, n, p = 1, q = 1):
    
    """
    Return the Hankel basis matrix B, a 'd' spmatrix of size 
    (m*n*p*q, (m+n-1)*p*q), that satisfies
    
        H[:] = B * x[:]
        
    where
    
            [ h_1   h_2     ...  h_n       ]
        H = [ h_2   h_3     ...  h_{n+1}   ]
            [  :     :            :        ]
            [ h_m   h_{m+1} ...  h_{m+n-1} ],

        x = [ h_1, h_2, ..., h_{m+n-1} ],
        
    and h_i, i=1...(m+n-1), are the Hankel blocks of size (p,q).
        
    INPUT
    m       number of block rows
    
    n       number of block columns
    
    p       number of rows of a Hankel block
    
    q       number of columns of a Hankel block    
    """
    
    Bi = [ (j*q+l)*m*p+i*p+k for i in xrange(m) for j in xrange(n) for 
        k in xrange(p) for l in xrange(q) ]
    Bj = [ (i+j)*p*q+l*p+k for i in xrange(m) for j in xrange(n) for 
        k in xrange(p) for l in xrange(q) ]
    
    return spmatrix(1.0, Bi, Bj)


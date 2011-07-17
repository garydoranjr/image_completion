"""
Interior-point code for nuclear norm minimization 
"""

# Version 1.0.  Copyright 2009 Z. Liu and L. Vandenberghe.
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


from cvxopt import matrix, spmatrix, sqrt, sparse, mul
from cvxopt import base, blas, lapack, solvers, misc 

__all__ = [ 'nrmapp' ]

def nrmapp(A, B, C = None, d = None, G = None, h = None): 
    """

    Solves the regularized nuclear norm approximation problem 
    
        minimize    || A(x) + B ||_* + 1/2 x'*C*x + d'*x
        subject to  G*x <= h

    and its dual

        maximize    -h'*z + tr(B'*Z) - 1/2 v'*C*v 
        subject to  d + G'*z + A'(Z) = C*v 
                    z >= 0
                    || Z || <= 1.

    A(x) is a linear mapping that maps n-vectors x to (p x q)-matrices A(x).

    ||.||_* is the nuclear norm (sum of singular values).  

    A'(Z) is the adjoint mapping of A(x).

    ||.|| is the maximum singular value norm.


    INPUT 

    A       real dense or sparse matrix of size (p*q, n).  Its columns are
            the coefficients A_i of the mapping 

                A: reals^n --> reals^pxq,   A(x) = sum_i=1^n x_i * A_i, 
                     
            stored in column-major order, as p*q-vectors.
        
    B       real dense or sparse matrix of size (p, q), with p >= q.
    
    C       real symmetric positive semidefinite dense or sparse matrix of 
            order n.  Only the lower triangular part of C is accessed.
            The default value is a zero matrix.

    d       real dense matrix of size (n, 1).  The default value is a zero
            vector.
    
    G       real dense or sparse matrix of size (m, n), with m >= 0.  
            The default value is a matrix of size (0, n).
    
    h       real dense matrix of size (m, 1).  The default value is a 
            matrix of size (0, 1).


    OUTPUT

    status  'optimal', 'primal infeasible', or 'unknown'. 

    x       'd' matrix of size (n, 1) if status is 'optimal'; 
            None otherwise.

    z       'd' matrix of size (m, 1) if status is 'optimal' or 'primal 
            infeasible'; None otherwise.

    Z       'd' matrix of size (p, q) if status is 'optimal' or 'primal
            infeasible'; None otherwise.


    If status is 'optimal', then x, z, Z are approximate solutions of the
    optimality conditions

        C * x  + G' * z + A'(Z) + d = 0  
        G * x <= h 
        z >= 0,  || Z || < = 1
        z' * (h - G*x) = 0
        tr (Z' * (A(x) + B)) = || A(x) + B ||_*.

    The last (complementary slackness) condition can be replaced by the
    following.  If the singular value decomposition of A(x) + B is

        A(x) + B = [ U1  U2 ] * diag(s, 0) * [ V1  V2 ]',

    with s > 0, then

        Z = U1 * V1' + U2 * W * V2',  || W || <= 1. 


    If status is 'primal infeasible', then Z = 0 and z is a certificate of
    infeasibility for the inequalities G * x <= h, i.e., a vector that
    satisfies

        h' * z = 1,  G' * z = 0,  z >= 0.

    """

    if type(B) not in (matrix, spmatrix) or B.typecode is not 'd':
        raise TypeError, "B must be a real dense or sparse matrix"
    p, q = B.size
    if p < q:
        raise ValueError, "row dimension of B must be greater than or "\
            "equal to column dimension"
    
    if type(A) not in (matrix, spmatrix) or A.typecode is not 'd' or \
        A.size[0] != p*q:
        raise TypeError, "A must be a real dense or sparse matrix with "\
            "p*q rows if B has size (p, q)"
    n = A.size[1]
    
    if G is None:  G = spmatrix([], [], [], (0, n))
    if h is None:  h = matrix(0.0, (0, 1))
    if type(h) is not matrix or h.typecode is not 'd' or h.size[1] != 1:
        raise TypeError, "h must be a real dense matrix with one column"
    m = h.size[0]
    if type(G) not in (matrix, spmatrix) or G.typecode is not 'd' or \
        G.size != (m, n):
        raise TypeError, "G must be a real dense matrix or sparse matrix "\
            "of size (m, n) if h has length m and A has n columns"
       
    if C is None: C = spmatrix(0.0, [], [], (n,n))
    if d is None: d = matrix(0.0, (n, 1))
    if type(C) not in (matrix, spmatrix) or C.typecode is not 'd' or \
        C.size != (n,n):
        raise TypeError, "C must be real dense or sparse matrix of size "\
            "(n, n) if A has n columns"
    if type(d) is not matrix or d.typecode is not 'd' or d.size != (n,1):
        raise TypeError, "d must be a real matrix of size (n, 1) if A has "\
            "n columns"


    # The problem is solved as a cone program
    #
    #     minimize    (1/2) * x'*C*x + d'*x  + (1/2) * (tr X1 + tr X2)
    #     subject to  G*x <= h
    #                 [ X1         (A(x) + B)' ]
    #                 [ A(x) + B   X2          ]  >= 0.
    #
    # The primal variable is stored as a list [ x, X1, X2 ].

    def xnewcopy(u): 
        return [ matrix(u[0]), matrix(u[1]), matrix(u[2]) ]
    def xdot(u,v):
        return blas.dot(u[0], v[0]) + misc.sdot2(u[1], v[1]) + \
            misc.sdot2(u[2], v[2])
    def xscal(alpha, u):
        blas.scal(alpha, u[0])
        blas.scal(alpha, u[1])
        blas.scal(alpha, u[2])
    def xaxpy(u, v, alpha = 1.0):
        blas.axpy(u[0], v[0], alpha)
        blas.axpy(u[1], v[1], alpha)
        blas.axpy(u[2], v[2], alpha)

    def Pf(u, v, alpha = 1.0, beta = 0.0):  
        base.symv(C, u[0], v[0], alpha = alpha, beta = beta)
        blas.scal(beta, v[1])
        blas.scal(beta, v[2])

    c = [ d, matrix(0.0, (q,q)), matrix(0.0, (p,p)) ]
    c[1][::q+1] = 0.5
    c[2][::p+1] = 0.5


    # If V is a p+q x p+q matrix 
    #
    #         [ V11  V12 ]
    #     V = [          ]
    #         [ V21  V22 ] 
    #
    # with V11 q x q,  V21 p x q, V12 q x p, and V22 p x p, then I11, I21,
    # I22 are the index sets defined by
    #
    #     V[I11] = V11[:],  V[I21] = V21[:],  V[I22] = V22[:].
    #

    I11 = matrix([ i + j*(p+q) for j in xrange(q) for i in xrange(q) ])
    I21 = matrix([ q + i + j*(p+q) for j in xrange(q) for i in xrange(p) ])
    I22 = matrix([ (p+q)*q + q + i + j*(p+q) for j in xrange(p) for 
       i in xrange(p) ])

    dims = {'l': m, 'q': [], 's': [p+q]}
    hh = matrix(0.0, (m + (p+q)**2, 1))
    hh[:m] = h
    hh[m + I21] = B[:]

    def Gf(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):

        if trans == 'N':
 
            # v[:m] := alpha * G * u[0] + beta * v[:m]
            base.gemv(G, u[0], v, alpha = alpha, beta = beta)

            # v[m:] := alpha * [-u[1],  -A(u[0])';  -A(u[0]), -u[2]]
            #          + beta * v[m:]
            blas.scal(beta, v, offset = m)
            v[m + I11] -= alpha * u[1][:]
            v[m + I21] -= alpha * A * u[0]
            v[m + I22] -= alpha * u[2][:]

        else:   
           
            # v[0] := alpha * ( G.T * u[:m] - 2.0 * A.T * u[m + I21] )
            #         + beta v[1]
            base.gemv(G, u, v[0], trans = 'T', alpha = alpha, beta = beta)  
            base.gemv(A, u[m + I21], v[0], trans = 'T', alpha = -2.0*alpha,
                beta = 1.0)

            # v[1] := -alpha * u[m + I11] + beta * v[1]
            blas.scal(beta, v[1])
            blas.axpy(u[m + I11], v[1], alpha = -alpha)

            # v[2] := -alpha * u[m + I22] + beta * v[2]
            blas.scal(beta, v[2])
            blas.axpy(u[m + I22], v[2], alpha = -alpha)


    def Af(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
        if trans == 'N':
            pass
        else:
            blas.scal(beta, v[0])
            blas.scal(beta, v[1])
            blas.scal(beta, v[2])


    L1 = matrix(0.0, (q, q))
    L2 = matrix(0.0, (p, p))
    T21 = matrix(0.0, (p, q))
    s = matrix(0.0, (q, 1))
    SS = matrix(0.0, (q, q))
    V1 = matrix(0.0, (q, q))
    V2 = matrix(0.0, (p, p))
    As = matrix(0.0, (p*q, n))
    As2 = matrix(0.0, (p*q, n))
    tmp = matrix(0.0, (p, q))
    a = matrix(0.0, (p+q, p+q))
    H = matrix(0.0, (n,n))
    Gs = matrix(0.0, (m, n))
    Q1 = matrix(0.0, (q, p+q))
    Q2 = matrix(0.0, (p, p+q))
    tau1 = matrix(0.0, (q,1))
    tau2 = matrix(0.0, (p,1))
    bz11 = matrix(0.0, (q,q))
    bz22 = matrix(0.0, (p,p))
    bz21 = matrix(0.0, (p,q))

    # Suppose V = [V1; V2] is p x q with V1 q x q.  If v = V[:] then
    # v[Itriu] are the strict upper triangular entries of V1 stored
    # columnwise.
    Itriu = [ i + j*p for j in xrange(1,q) for i in xrange(j) ]

    # v[Itril] are the strict lower triangular entries of V1 stored rowwise.
    Itril = [ j + i*p for j in xrange(1,q) for i in xrange(j) ]

    # v[Idiag] are the diagonal entries of V1.
    Idiag = [ i*(p+1) for i in xrange(q) ]

    # v[Itriu2] are the upper triangular entries of V1, with the diagonal
    # entries stored first, followed by the strict upper triangular entries
    # stored columnwise.
    Itriu2 = Idiag + Itriu

    # If V is a q x q matrix and v = V[:], then v[Itril2] are the strict
    # lower triangular entries of V stored columnwise and v[Itril3] are
    # the strict lower triangular entries stored rowwise.
    Itril2 = [ i + j*q for j in xrange(q) for i in xrange(j+1,q) ]
    Itril3 = [ i + j*q for i in xrange(q) for j in xrange(i) ]

    P = spmatrix(0.0, Itriu, Itril, (p*q, p*q))
    D = spmatrix(1.0, range(p*q), range(p*q))
    DV = matrix(1.0, (p*q, 1))


    def F(W):
        """
        Create a solver for the linear equations

                                C * ux + G' * uzl - 2*A'(uzs21) = bx
                                                         -uzs11 = bX1
                                                         -uzs22 = bX2
                                            G * ux - Dl^2 * uzl = bzl
            [ -uX1   -A(ux)' ]          [ uzs11 uzs21' ]     
            [                ] - r*r' * [              ] * r*r' = bzs
            [ -A(ux) -uX2    ]          [ uzs21 uzs22  ]

        where Dl = diag(W['l']), r = W['r'][0].  

        On entry, x = (bx, bX1, bX2) and z = [ bzl; bzs[:] ].
        On exit, x = (ux, uX1, uX2) and z = [ Dl*uzl; (r'*uzs*r)[:] ].


        1. Compute matrices V1, V2 such that (with T = r*r')
        
               [ V1   0   ] [ T11  T21' ] [ V1'  0  ]   [ I  S' ]
               [          ] [           ] [         ] = [       ]
               [ 0    V2' ] [ T21  T22  ] [ 0    V2 ]   [ S  I  ]
        
           and S = [ diag(s); 0 ], s a positive q-vector.

        2. Factor the mapping X -> X + S * X' * S:

               X + S * X' * S = L( L'( X )). 

        3. Compute scaled mappings: a matrix As with as its columns the 
           coefficients of the scaled mapping 

               L^-1( V2' * A() * V1' ) 

           and the matrix Gs = Dl^-1 * G.

        4. Cholesky factorization of H = C + Gs'*Gs + 2*As'*As.

        """


        # 1. Compute V1, V2, s.  

        r = W['r'][0]

        # LQ factorization R[:q, :] = L1 * Q1.
        lapack.lacpy(r, Q1, m = q)
        lapack.gelqf(Q1, tau1)
        lapack.lacpy(Q1, L1, n = q, uplo = 'L')
        lapack.orglq(Q1, tau1)

        # LQ factorization R[q:, :] = L2 * Q2.
        lapack.lacpy(r, Q2, m = p, offsetA = q)
	lapack.gelqf(Q2, tau2)
        lapack.lacpy(Q2, L2, n = p, uplo = 'L')
        lapack.orglq(Q2, tau2)


        # V2, V1, s are computed from an SVD: if
        # 
        #     Q2 * Q1' = U * diag(s) * V',
        #
        # then V1 = V' * L1^-1 and V2 = L2^-T * U.
    
        # T21 = Q2 * Q1.T  
        blas.gemm(Q2, Q1, T21, transB = 'T')

        # SVD T21 = U * diag(s) * V'.  Store U in V2 and V' in V1.
        lapack.gesvd(T21, s, jobu = 'A', jobvt = 'A', U = V2, Vt = V1) 

#        # Q2 := Q2 * Q1' without extracting Q1; store T21 in Q2
#        this will requires lapack.ormlq or lapack.unmlq

        # V2 = L2^-T * U   
        blas.trsm(L2, V2, transA = 'T') 

        # V1 = V' * L1^-1 
        blas.trsm(L1, V1, side = 'R') 


        # 2. Factorization X + S * X' * S = L( L'( X )).  
        #
        # The factor L is stored as a diagonal matrix D and a sparse lower 
        # triangular matrix P, such that  
        #
        #     L(X)[:] = D**-1 * (I + P) * X[:] 
        #     L^-1(X)[:] = D * (I - P) * X[:].

        # SS is q x q with SS[i,j] = si*sj.
        blas.scal(0.0, SS)
        blas.syr(s, SS)    
        
        # For a p x q matrix X, P*X[:] is Y[:] where 
        #
        #     Yij = si * sj * Xji  if i < j
        #         = 0              otherwise.
        # 
        P.V = SS[Itril2]

        # For a p x q matrix X, D*X[:] is Y[:] where 
        #
        #     Yij = Xij / sqrt( 1 - si^2 * sj^2 )  if i < j
        #         = Xii / sqrt( 1 + si^2 )         if i = j
        #         = Xij                            otherwise.
        # 
        DV[Idiag] = sqrt(1.0 + SS[::q+1])
        DV[Itriu] = sqrt(1.0 - SS[Itril3]**2)
        D.V = DV**-1


        # 3. Scaled linear mappings 
         
        # Ask :=  V2' * Ask * V1' 
        blas.scal(0.0, As)
        base.axpy(A, As)
        for i in xrange(n):
            # tmp := V2' * As[i, :]
            blas.gemm(V2, As, tmp, transA = 'T', m = p, n = q, k = p,
                ldB = p, offsetB = i*p*q)
            # As[:,i] := tmp * V1'
            blas.gemm(tmp, V1, As, transB = 'T', m = p, n = q, k = q,
                ldC = p, offsetC = i*p*q)

        # As := D * (I - P) * As 
        #     = L^-1 * As.
        blas.copy(As, As2)
        base.gemm(P, As, As2, alpha = -1.0, beta = 1.0)
        base.gemm(D, As2, As)

        # Gs := Dl^-1 * G 
        blas.scal(0.0, Gs)
        base.axpy(G, Gs)
        for k in xrange(n):
            blas.tbmv(W['di'], Gs, n = m, k = 0, ldA = 1, offsetx = k*m)


        # 4. Cholesky factorization of H = C + Gs' * Gs + 2 * As' * As.

        blas.syrk(As, H, trans = 'T', alpha = 2.0)
        blas.syrk(Gs, H, trans = 'T', beta = 1.0)
        base.axpy(C, H)   
        lapack.potrf(H)


        def f(x, y, z):
            """

            Solve 

                              C * ux + G' * uzl - 2*A'(uzs21) = bx
                                                       -uzs11 = bX1
                                                       -uzs22 = bX2
                                           G * ux - D^2 * uzl = bzl
                [ -uX1   -A(ux)' ]       [ uzs11 uzs21' ]     
                [                ] - T * [              ] * T = bzs.
                [ -A(ux) -uX2    ]       [ uzs21 uzs22  ]

            On entry, x = (bx, bX1, bX2) and z = [ bzl; bzs[:] ].
            On exit, x = (ux, uX1, uX2) and z = [ D*uzl; (r'*uzs*r)[:] ].

            Define X = uzs21, Z = T * uzs * T:   
 
                      C * ux + G' * uzl - 2*A'(X) = bx
                                [ 0  X' ]               [ bX1 0   ]
                            T * [       ] * T - Z = T * [         ] * T
                                [ X  0  ]               [ 0   bX2 ]
                               G * ux - D^2 * uzl = bzl
                [ -uX1   -A(ux)' ]   [ Z11 Z21' ]     
                [                ] - [          ] = bzs
                [ -A(ux) -uX2    ]   [ Z21 Z22  ]

            Return x = (ux, uX1, uX2), z = [ D*uzl; (rti'*Z*rti)[:] ].

            We use the congruence transformation 

                [ V1   0   ] [ T11  T21' ] [ V1'  0  ]   [ I  S' ]
                [          ] [           ] [         ] = [       ]
                [ 0    V2' ] [ T21  T22  ] [ 0    V2 ]   [ S  I  ]

            and the factorization 

                X + S * X' * S = L( L'(X) ) 

            to write this as

                                  C * ux + G' * uzl - 2*A'(X) = bx
                L'(V2^-1 * X * V1^-1) - L^-1(V2' * Z21 * V1') = bX
                                           G * ux - D^2 * uzl = bzl
                            [ -uX1   -A(ux)' ]   [ Z11 Z21' ]     
                            [                ] - [          ] = bzs,
                            [ -A(ux) -uX2    ]   [ Z21 Z22  ]

            or

                C * ux + Gs' * uuzl - 2*As'(XX) = bx
                                      XX - ZZ21 = bX
                                 Gs * ux - uuzl = D^-1 * bzl
                                 -As(ux) - ZZ21 = bbzs_21
                                     -uX1 - Z11 = bzs_11
                                     -uX2 - Z22 = bzs_22

            if we introduce scaled variables

                uuzl = D * uzl
                  XX = L'(V2^-1 * X * V1^-1) 
                     = L'(V2^-1 * uzs21 * V1^-1)
                ZZ21 = L^-1(V2' * Z21 * V1') 

            and define

                bbzs_21 = L^-1(V2' * bzs_21 * V1')
                                           [ bX1  0   ]
                     bX = L^-1( V2' * (T * [          ] * T)_21 * V1').
                                           [ 0    bX2 ]           
 
            Eliminating Z21 gives 

                C * ux + Gs' * uuzl - 2*As'(XX) = bx
                                 Gs * ux - uuzl = D^-1 * bzl
                                   -As(ux) - XX = bbzs_21 - bX
                                     -uX1 - Z11 = bzs_11
                                     -uX2 - Z22 = bzs_22 

            and eliminating uuzl and XX gives

                        H * ux = bx + Gs' * D^-1 * bzl + 2*As'(bX - bbzs_21)
                Gs * ux - uuzl = D^-1 * bzl
                  -As(ux) - XX = bbzs_21 - bX
                    -uX1 - Z11 = bzs_11
                    -uX2 - Z22 = bzs_22.


            In summary, we can use the following algorithm: 

            1. bXX := bX - bbzs21
                                        [ bX1 0   ]
                    = L^-1( V2' * ((T * [         ] * T)_21 - bzs_21) * V1')
                                        [ 0   bX2 ]

            2. Solve H * ux = bx + Gs' * D^-1 * bzl + 2*As'(bXX).

            3. From ux, compute 

                   uuzl = Gs*ux - D^-1 * bzl and 
                      X = V2 * L^-T(-As(ux) + bXX) * V1.

            4. Return ux, uuzl, 

                   rti' * Z * rti = r' * [ -bX1, X'; X, -bX2 ] * r
 
               and uX1 = -Z11 - bzs_11,  uX2 = -Z22 - bzs_22.

            """

            # Save bzs_11, bzs_22, bzs_21.
            lapack.lacpy(z, bz11, uplo = 'L', m = q, n = q, ldA = p+q,
                offsetA = m)
            lapack.lacpy(z, bz21, m = p, n = q, ldA = p+q, offsetA = m+q)
            lapack.lacpy(z, bz22, uplo = 'L', m = p, n = p, ldA = p+q,
                offsetA = m + (p+q+1)*q)


            # zl := D^-1 * zl
            #     = D^-1 * bzl
            blas.tbmv(W['di'], z, n = m, k = 0, ldA = 1)


            # zs := r' * [ bX1, 0; 0, bX2 ] * r.

            # zs := [ bX1, 0; 0, bX2 ]
            blas.scal(0.0, z, offset = m)
            lapack.lacpy(x[1], z, uplo = 'L', m = q, n = q, ldB = p+q,
                offsetB = m)
            lapack.lacpy(x[2], z, uplo = 'L', m = p, n = p, ldB = p+q,
                offsetB = m + (p+q+1)*q)

            # scale diagonal of zs by 1/2
            blas.scal(0.5, z, inc = p+q+1, offset = m)

            # a := tril(zs)*r  
            blas.copy(r, a)
            blas.trmm(z, a, side = 'L', m = p+q, n = p+q, ldA = p+q, ldB = 
                p+q, offsetA = m)

            # zs := a'*r + r'*a 
            blas.syr2k(r, a, z, trans = 'T', n = p+q, k = p+q, ldB = p+q,
                ldC = p+q, offsetC = m)



            # bz21 := L^-1( V2' * ((r * zs * r')_21 - bz21) * V1')
            #
            #                           [ bX1 0   ]
            #       = L^-1( V2' * ((T * [         ] * T)_21 - bz21) * V1').
            #                           [ 0   bX2 ]

            # a = [ r21 r22 ] * z
            #   = [ r21 r22 ] * r' * [ bX1, 0; 0, bX2 ] * r
            #   = [ T21  T22 ] * [ bX1, 0; 0, bX2 ] * r
            blas.symm(z, r, a, side = 'R', m = p, n = p+q, ldA = p+q, 
                ldC = p+q, offsetB = q)
    
            # bz21 := -bz21 + a * [ r11, r12 ]'
            #       = -bz21 + (T * [ bX1, 0; 0, bX2 ] * T)_21
            blas.gemm(a, r, bz21, transB = 'T', m = p, n = q, k = p+q, 
                beta = -1.0, ldA = p+q, ldC = p)

            # bz21 := V2' * bz21 * V1'
            #       = V2' * (-bz21 + (T*[bX1, 0; 0, bX2]*T)_21) * V1'
            blas.gemm(V2, bz21, tmp, transA = 'T', m = p, n = q, k = p, 
                ldB = p)
            blas.gemm(tmp, V1, bz21, transB = 'T', m = p, n = q, k = q, 
                ldC = p)

            # bz21[:] := D * (I-P) * bz21[:] 
            #       = L^-1 * bz21[:]
            #       = bXX[:]
            blas.copy(bz21, tmp)
            base.gemv(P, bz21, tmp, alpha = -1.0, beta = 1.0)
            base.gemv(D, tmp, bz21)


            # Solve H * ux = bx + Gs' * D^-1 * bzl + 2*As'(bXX).

            # x[0] := x[0] + Gs'*zl + 2*As'(bz21) 
            #       = bx + G' * D^-1 * bzl + 2 * As'(bXX)
            blas.gemv(Gs, z, x[0], trans = 'T', alpha = 1.0, beta = 1.0)
            blas.gemv(As, bz21, x[0], trans = 'T', alpha = 2.0, beta = 1.0) 

            # x[0] := H \ x[0] 
            #      = ux
            lapack.potrs(H, x[0])


            # uuzl = Gs*ux - D^-1 * bzl
            blas.gemv(Gs, x[0], z, alpha = 1.0, beta = -1.0)

            
            # bz21 := V2 * L^-T(-As(ux) + bz21) * V1
            #       = X
            blas.gemv(As, x[0], bz21, alpha = -1.0, beta = 1.0)
            blas.tbsv(DV, bz21, n = p*q, k = 0, ldA = 1)
            blas.copy(bz21, tmp)
            base.gemv(P, tmp, bz21, alpha = -1.0, beta = 1.0, trans = 'T')
            blas.gemm(V2, bz21, tmp)
            blas.gemm(tmp, V1, bz21)


            # zs := -zs + r' * [ 0, X'; X, 0 ] * r
            #     = r' * [ -bX1, X'; X, -bX2 ] * r.

            # a := bz21 * [ r11, r12 ]
            #   =  X * [ r11, r12 ]
            blas.gemm(bz21, r, a, m = p, n = p+q, k = q, ldA = p, ldC = p+q)
            
            # z := -z + [ r21, r22 ]' * a + a' * [ r21, r22 ]
            #    = rti' * uzs * rti
            blas.syr2k(r, a, z, trans = 'T', beta = -1.0, n = p+q, k = p,
                offsetA = q, offsetC = m, ldB = p+q, ldC = p+q)  



            # uX1 = -Z11 - bzs_11 
            #     = -(r*zs*r')_11 - bzs_11
            # uX2 = -Z22 - bzs_22 
            #     = -(r*zs*r')_22 - bzs_22


            blas.copy(bz11, x[1])
            blas.copy(bz22, x[2])

            # scale diagonal of zs by 1/2
            blas.scal(0.5, z, inc = p+q+1, offset = m)

            # a := r*tril(zs)  
            blas.copy(r, a)
            blas.trmm(z, a, side = 'R', m = p+q, n = p+q, ldA = p+q, ldB = 
                p+q, offsetA = m)

            # x[1] := -x[1] - a[:q,:] * r[:q, :]' - r[:q,:] * a[:q,:]'
            #       = -bzs_11 - (r*zs*r')_11
            blas.syr2k(a, r, x[1], n = q, alpha = -1.0, beta = -1.0) 

            # x[2] := -x[2] - a[q:,:] * r[q:, :]' - r[q:,:] * a[q:,:]'
            #       = -bzs_22 - (r*zs*r')_22
            blas.syr2k(a, r, x[2], n = p, alpha = -1.0, beta = -1.0, 
                offsetA = q, offsetB = q)

            # scale diagonal of zs by 1/2
            blas.scal(2.0, z, inc = p+q+1, offset = m)


        return f


    if C:
        sol = solvers.coneqp(Pf, c, Gf, hh, dims, Af, kktsolver = F, 
            xnewcopy = xnewcopy, xdot = xdot, xaxpy = xaxpy, xscal = xscal) 
    else: 
        sol = solvers.conelp(c, Gf, hh, dims, Af, kktsolver = F, 
            xnewcopy = xnewcopy, xdot = xdot, xaxpy = xaxpy, xscal = xscal) 

    if sol['status'] is 'optimal':
        x = sol['x'][0]
        z = sol['z'][:m]
        Z = sol['z'][m:]
        Z.size = (p + q, p + q)
        Z = -2.0 * Z[-p:, :q]

    elif sol['status'] is 'primal infeasible':
        x = None
        z = sol['z'][:m]
        Z = sol['z'][m:]
        Z.size = (p + q, p + q)
        Z = -2.0 * Z[-p:, :q]

    else:
        x, z, Z = None, None, None

    return {'status': sol['status'], 'x': x, 'z': z, 'Z': Z }


def checksol(sol, A, B, C = None, d = None, G = None, h = None): 
    """
    Check optimality conditions

        C * x  + G' * z + A'(Z) + d = 0  
        G * x <= h 
        z >= 0,  || Z || < = 1
        z' * (h - G*x) = 0
        tr (Z' * (A(x) + B)) = || A(x) + B ||_*.

    """

    p, q = B.size
    n = A.size[1]
    if G is None: G = spmatrix([], [], [], (0, n))
    if h is None: h = matrix(0.0, (0, 1))
    m = h.size[0]
    if C is None: C = spmatrix(0.0, [], [], (n,n))
    if d is None: d = matrix(0.0, (n, 1))

    if sol['status'] is 'optimal':

        res = +d
        base.symv(C, sol['x'], res, beta = 1.0)
        base.gemv(G, sol['z'], res, beta = 1.0, trans = 'T')
        base.gemv(A, sol['Z'], res, beta = 1.0, trans = 'T')
        print "Dual residual: %e" %blas.nrm2(res)

        if m:
           print "Minimum primal slack (scalar inequalities): %e" \
               %min(h - G*sol['x'])
           print "Minimum dual slack (scalar inequalities): %e" \
               %min(sol['z'])

        if p:
            s = matrix(0.0, (p,1))
            X = matrix(A*sol['x'], (p, q)) + B
            lapack.gesvd(+X, s)
            nrmX = sum(s)
            lapack.gesvd(+sol['Z'], s)
            nrmZ = max(s)
            print "Norm of Z: %e" %nrmZ
            print "Nuclear norm of A(x) + B: %e" %nrmX
            print "Inner product of Z and A(x) + B: %e" \
                %blas.dot(sol['Z'], X)
        
    elif sol['status'] is 'primal infeasible':

        res = matrix(0.0, (n,1))
        base.gemv(G, sol['z'], res, beta = 1.0, trans = 'T')
        print "Dual residual: %e" %blas.nrm2(res)
        print "h' * z = %e" %blas.dot(h, sol['z'])
        print "Minimum dual slack (scalar inequalities): %e" \
            %min(sol['z'])


    else:
        pass


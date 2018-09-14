import numpy as np
import math

def doSVD(A): # Using numpy library for reference
    U, S, V = np.linalg.svd(A, full_matrices=True)
    return U, S, V

def doJacobiSVD(A):
    
    U = A.T.dot(A)        # Symmetrize the input matrix and call it U
    n = U.shape[0]        # n is the dimension of U
    V = np.identity(n)    # V is an identity matrix of size nxn
    w = np.diag(U).copy() # w is a vector initialized with the diagonal elements of U 
    
    iterationCount = 0
    while abs(U[0, 1]) > 0:
        for e in range(0, n-1):
            for f in range(e+1, n):
            
                # Phase solver
                if (abs(U[e, f]) < 0.0000001): # To avoid overflow
                    tgphi = 0
                else:
                    alpha = (w[f] - w[e]) / (2 * U[e, f])
                    tgphi = math.copysign(1, alpha) / (abs(alpha) + math.sqrt(1 + alpha**2))
                
                cos_phi = 1 / math.sqrt(1 + tgphi**2)
                sin_phi = tgphi*cos_phi
                 
                w[e] -= tgphi*U[e, f]
                w[f] += tgphi*U[e, f]
                U[e, f] = 0

                # Rotation loops
                for k in range(0, e):
                    tmp = U[k, e]
                    U[k, e] = tmp*cos_phi - U[k, f]*sin_phi
                    U[k, f] = tmp*sin_phi + U[k, f]*cos_phi
                    
                for k in range(e+1, f):
                    tmp = U[e, k]
                    U[e, k] = tmp*cos_phi - U[k, f]*sin_phi
                    U[k, f] = tmp*sin_phi + U[k, f]*cos_phi
                    
                for k in range(f+1, n):
                    tmp = U[e, k]
                    U[e, k] = tmp*cos_phi - U[f, k]*sin_phi
                    U[f, k] = tmp*sin_phi + U[f, k]*cos_phi
                    
                for k in range(0, n):
                    tmp = V[k, e]
                    V[k, e] = tmp*cos_phi - V[k, f]*sin_phi
                    V[k, f] = tmp*sin_phi + V[k, f]*cos_phi
        
        iterationCount += 1
        
    print("Done at iteration = {0}".format(iterationCount))
            
    return V 


Xtest = np.matrix("""0.840188 0.394383 0.783099; 
                     0.798440 0.911647 0.197551""")


print("We do numpy SVD for reference and get:")
U, S, V = doSVD(Xtest)
print("U = ")
print(U)
print("S = ")
print(S)
print("V = ")
print(V)


print("\n\nWe do Jacobi SVD and get:")
V = doJacobiSVD(Xtest)
print("V = ")
print(V)
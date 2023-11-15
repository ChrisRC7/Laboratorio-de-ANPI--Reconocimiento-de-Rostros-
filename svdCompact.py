import numpy as np
def svdCompact(A):
    m, n = A.shape
    
    if m > n:
        M1 = np.dot(A.T, A)
        _, V1 = np.linalg.eigh(M1)
        y1 = np.linalg.eigvalsh(M1)
        const = n * np.max(y1) * np.finfo(float).eps
        y2 = (y1 > const).astype(int)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        V2 = V1[:, order]
        Vr = V2[:, :rA]
        s1 = np.sqrt(y3[order][:rA])
        Sr = np.diag(s1)
        Ur = (A @ Vr) / s1  # Direct element-wise division
    else:
        M1 = np.dot(A, A.T)
        _, U1 = np.linalg.eigh(M1)
        y1 = np.linalg.eigvalsh(M1)
        const = m * np.max(y1) * np.finfo(float).eps
        y2 = (y1 > const).astype(int)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        U2 = U1[:, order]
        Ur = U2[:, :rA]
        s1 = np.sqrt(y3[order][:rA])
        Sr = np.diag(s1)
        Vr = (A.T @ Ur) / s1  # Direct element-wise division
    
    return Ur, Sr, Vr
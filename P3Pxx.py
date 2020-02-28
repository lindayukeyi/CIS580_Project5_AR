import numpy as np
import PnP

def P3P(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_P3P.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])

    p1 = Pw[1,:]
    p2 = Pw[2,:]
    p3 = Pw[3,:]

    f = (K[0][0] + K[1][1]) / 2
    offset = np.array([K[0][2], K[1][2]])
    uv1 = Pc[1,:] - offset
    uv2 = Pc[2,:] - offset
    uv3 = Pc[3,:] - offset
    uv1 = np.append(uv1, f)
    uv2 = np.append(uv2, f)
    uv3 = np.append(uv3, f)

    # define a,b,c,alpha,beta,gamma
    a = np.linalg.norm(p2-p3)
    b = np.linalg.norm(p1-p3)
    c = np.linalg.norm(p1-p2)

    j1 = uv1 / np.linalg.norm(uv1)
    j2 = uv2 / np.linalg.norm(uv2)
    j3 = uv3 / np.linalg.norm(uv3)

    calpha = np.dot(j2, j3)
    cbata = np.dot(j1, j3)
    cgamma = np.dot(j1, j2)

    # define coefficients of the 4th degree polynomial
    co1 = (np.power(a, 2) - np.power(c, 2)) / np.power(b, 2) # (a^2 - c^2) / b^2
    co2 = (np.power(a, 2) + np.power(c, 2)) / np.power(b, 2) # (a^2 + c^2) / b^2
    co3 = (np.power(b, 2) - np.power(c, 2)) / np.power(b, 2) # (b^2 - c^2) / b^2
    co4 = (np.power(b, 2) - np.power(a, 2)) / np.power(b, 2) # (b^2 - a^2) / b^2
    A4 = np.power(co1 - 1, 2) - 4 * np.power(c / b, 2) * np.power(calpha, 2)
    A3 = 4 * (co1 * (1 - co1) * cbata - (1 - co2) * calpha * cgamma + 2 * np.power(c / b, 2) * np.power(calpha, 2) * cbata)
    A2 = 2 * (np.power(co1, 2) - 1 + 2 * np.power(co1, 2) * np.power(cbata, 2) + 2 * co3 * np.power(calpha, 2) - 4 * co2 * calpha * cbata * cgamma + 2 * co4 * np.power(cgamma, 2))
    A1 = 4 * (-co1 * (1 + co1) * cbata + 2 * np.power(a / b, 2) * np.power(cgamma, 2) * cbata - (1 - co2) * calpha * cgamma)
    A0 = np.power((1 + co1), 2) - 4 * np.power(a / b, 2) * np.power(cgamma, 2)
    # calculate real roots u and v
    coeff = np.array([A4, A3, A2, A1, A0])
    v = np.roots(coeff)
    real = []
    for number in v:
        if np.isreal(number):
            real.append(number.real)
    v = np.array(real)
    u = [0, 0]
    u[0] = ((-1 + co1) * np.power(v[0], 2) - 2 * co1 * cbata * v[0] + 1 + co1) / (2 * (cgamma - calpha * v[0]))
    u[1] = ((-1 + co1) * np.power(v[1], 2) - 2 * co1 * cbata * v[1] + 1 + co1) / (2 * (cgamma - calpha * v[1]))

    # check for valid distances
    s1 = np.power(b, 2) / (1 + np.power(v, 2) - 2 * v * cbata)
    s1 = np.sqrt(s1)
    s2 = u * s1
    s3 = v * s1

    s = []
    if(s1[1] > 0 and s2[1] > 0 and s3[1] > 0):
        s = np.array([s1[0], s2[0], s3[0]])
    elif(s1[0] > 0 and s2[0] > 0 and s3[0] > 0):
        s = np.array([s1[1], s2[1], s3[1]])
    else:
        print("Wrong!!!!!!!!!!!!")
    
    # calculate 3D coordinates in Camera frame
    p1cal = s[0] * j1
    p2cal = s[1] * j2
    p3cal = s[2] * j3
    
    Xc = np.array([p1cal, p2cal, p3cal])

    
    #R,t = PnP.PnP(Pc, Pw, K)
    #t = -R.T@t
    #Pw = np.matrix(Pw.T)
    #Xc = (np.transpose(R))@(Pw) + t
    
    # Calculate R,t using Procrustes
    #Xc = Xc.T
    #Xc = Xc[1:]
    #Y = Pw.T
    #Y = Y[1:].T
    R, t = Procrustes(Xc, Pw[1:])
    
    ##### STUDENT CODE END #####
    
    return R, t

def Procrustes(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    X = X.T
    Y = Y.T
    Xhat = np.mean(X, axis = 1)
    Yhat = np.mean(Y, axis = 1)
    X = X - Xhat.T
    Y = Y - Yhat.T
    U, S, Vh = np.linalg.svd(X@Y.T)
    V = np.transpose(Vh)
    S = np.eye(3)

    print(V@np.transpose(U))
    S[2][2] = np.linalg.det(V@np.transpose(U))
    R = V@S@np.transpose(U)
    print(np.linalg.det(R))

    t = Yhat - R@(Xhat)

    ##### STUDENT CODE END #####
    
    return R, t

'''
R = np.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
t = np.matrix([2, -4, 8]).T
points = np.matrix([[2, 3, 1], [1, 2, 4], [3,4,6]])
print(points)
uvs = R@points.T + t
print(uvs)
a, b = Procrustes(points, uvs)
print(a)
print(b)
'''
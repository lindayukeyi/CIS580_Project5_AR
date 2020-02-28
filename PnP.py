import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_PnP.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####


    R = np.eye(3)
    t = np.zeros([3])
    #print(Pw[:,:2])
    hmatrix = est_homography(Pw[:,:2], Pc)
    hprime = np.linalg.inv(K)@hmatrix
    #print(hmatrix)
    #print(np.linalg.inv(K))
    #print("hprime", hprime)
    h1prime = hprime[:,0]
    h2prime = hprime[:,1]
    h3prime = hprime[:,2]
    h12prime = np.cross(h1prime, h2prime)
    #print(h1prime)
    #print(hprime[:,:2])
    h = np.column_stack((hprime[:,:2], h12prime))
    U, S, V = np.linalg.svd(h)
    S = np.eye(3)
    S[2][2] = np.linalg.det(U@V)
    R = np.matrix(U@S@V)
    #print("DET R:", np.linalg.det(R))
    t = h3prime / np.linalg.norm(h1prime)
    t = np.transpose(-np.transpose(R)@t)
    R = np.transpose(R)
    ##### STUDENT CODE END #####

    return R, t

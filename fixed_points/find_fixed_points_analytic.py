from itertools import combinations, chain
import numpy as np
from tqdm import tqdm

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_fixed_points_analytic(a, V, U, hz, h, d=1, neuromodulation=None, A=None, s=None, Wu=None, u=None):
    """
    Find fixed points of the model
    Args:
        a: numpy array of shape (R,) decay
        V: numpy array of shape (R,N) scaled left singular vectors
        U: numpy array of shape (N,R) right singular vectors
        hz: numpy array of shape (R,) latent bias (assumed to be subtracted!)
        h: numpy array of shape (N,) neuron bias
        d: number of bases of the transfer function
        neuromodulation: string 
        A: numpy array of shape (N,n_s)
        s: numpy array of shape (n_s,)
    Returns:
        D_list: numpy array of shape (n_Ds,N) containing all subspaces
        D_inds: list of indices of subspaces in D_list that are fixed points
        z_list: list of fixed points
        n_inverses: number of inverses
    """

    n_inverses = 0
    N = U.shape[0]
    R = U.shape[1]

    # First solve for all intersection of hyperplanes
    intersect_inds = np.array(list(combinations(np.arange(N), R)))
    print(f'Intersection indices shape: {intersect_inds.shape}')
    print(len(intersect_inds))

    par_inds = []
    if d == 2:
        ni = N // 2
        for i, el in enumerate(intersect_inds):
            if el.shape[-1] > 1 and (el[0] == el[1] + ni or el[1] == el[0] + ni):
                par_inds.append(i)
        intersect_inds = np.delete(intersect_inds, par_inds, axis=0)
        print("removed parallel lines")
        print(len(intersect_inds))

    n_Ds_initial = len(list(powerset(range(R)))) * len(intersect_inds)
    print(len(list(powerset(range(R)))))
    D_list = np.zeros((n_Ds_initial, N), dtype="uint8")
    it = 0
    
    Wu_new = None # input weights
    if Wu is not None: 
        Wu_new = np.concatenate([Wu, Wu], axis=0)

    W_new = None # neuromodulatory weights
    if neuromodulation == 'additive':
        W = A @ s 
        W_new = np.concatenate([W, W], axis=0)
        
    for inds in tqdm(intersect_inds):
        b_hat = h[inds]
        U_hat = U[inds]
        n_inverses += 1
        try:
            z = np.linalg.solve(U_hat, b_hat)
        except:
            continue
        # Find all subspaces bordering to this intersection
        if Wu_new is not None and W_new is not None:
            x = U @ z - h + Wu_new @ u + W_new 
        elif W_new is not None: 
            x = U @ z - h + W_new 
        elif Wu_new is not None: 
            x = U @ z - h + Wu_new @ u 
        else:        
            x = U @ z - h
        D_init = np.array(x > 0).astype("uint8")
        D_init[inds] = 0
        D_list[it] = D_init
        it += 1
        D_inds = list(powerset(inds))[1:]
        for D_ind in D_inds:
            D = np.copy(D_init)
            D[np.array(D_ind)] = 1
            D_list[it] = D
            it += 1
    # Throw away duplicate subspaces
    print(D_list.shape)
    D_list = np.unique(D_list, axis=0)
    print(D_list.shape)

    # Finally solve for fixed points
    z_list = []
    D_inds = []
    
    for D_ind, D_init in enumerate(D_list):  
        if neuromodulation == 'postsynaptic' or neuromodulation == 'presynaptic':
            W = np.eye(A.shape[0]) + np.diag(A @ s) 
            dg = np.diagonal(W)
            W_new = np.diag(np.concatenate([dg, dg])) 

        if neuromodulation == 'postsynaptic':             
            Y = -np.eye(R) + np.diag(a) + V @ W_new @ np.diag(D_init) @ U
            if Wu_new is not None:
                b = V @ W_new @ (np.diag(D_init) @ h - np.diag(D_init) @ Wu_new @ u) + hz
            else:
                b = V @ W_new @ np.diag(D_init) @ h + hz

        elif neuromodulation == 'presynaptic':
            Y = -np.eye(R) + np.diag(a) + V @ np.diag(D_init) @ W_new @ U
            b = V @ np.diag(D_init) @ h + hz

        elif neuromodulation == 'rank':
            W = np.eye(A.shape[0]) + np.diag(A @ s)
            Y = -np.eye(R) + np.diag(a) + W @ V @ np.diag(D_init) @ U
            b = W @ V @ np.diag(D_init) @ h + hz 
            
        elif neuromodulation == 'additive':
            W = A @ s 
            W_new = np.concatenate([W, W], axis=0)
            Y = -np.eye(R) + np.diag(a) + V @ np.diag(D_init) @ U
            b = V @ (np.diag(D_init) @ h - np.diag(D_init) @ W_new) + hz
            
        else:
            Y = -np.eye(R) + np.diag(a) + V @ np.diag(D_init) @ U
            b = V @ np.diag(D_init) @ h + hz
        z_hat = np.linalg.solve(Y, b)
        n_inverses += 1

        if Wu_new is not None and neuromodulation == 'additive':
            x_hat = U @ z_hat - h + Wu_new @ u + W_new
        elif Wu_new is not None:
            x_hat = U @ z_hat - h + Wu_new @ u 
        else: 
            x_hat = U @ z_hat - h

        if np.allclose(D_init, np.array(x_hat > 0).astype("uint8")):
            print("Found a fixed point")
            print(z_hat)
            z_list.append(z_hat)
            D_inds.append(D_ind)
    print("Done, found " + str(len(z_list)) + " fixed points")
    return D_list, D_inds, z_list, n_inverses

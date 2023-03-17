import h5py 


# center
# list to arr
# from "frame-index" to "(traj-index, frame-index)"

def make_sparse_basis(dtrajs):
    """Converts a discretized trajectory (e.g. from k-means clustering)
    into a sparse basis of indicator functions.

    Parameters
    ----------
    dtrajs : ndarray
        discretized trajectories

    Return
    ------
    basis : scipy.sparse.csr_matrix
    """
    nclusters = len(np.unique(dtrajs))
    rows, cols = [], []
    for i in range(nclusters):
        pts = np.argwhere(dtrajs == i)
        # indices of which frames are in the cluster i
        rows.append(pts.squeeze())
        # all assigned as 1 in the basis
        cols.append(np.repeat(i, len(pts)))
    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.ones(len(rows), dtype=float)
    basis = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(dtrajs), nclusters))
    return basis
def P2pmf(arr):
    #Maximum probability is set to one (1)
    #Minimum pmf is set to zero
    return -kbT_kcal*np.log(arr/np.max(arr)) 
def pmf2P(arr):
    #Maximum probability is set to one (1)
    #Minimum pmf is set to zero
    arr = np.array(arr, dtype=np.double)
    return np.exp(-(arr-np.min(arr))/kbT_kcal)
def center(arr):
    return (arr[1:]+arr[:-1])/2

def hdf5open(files):
    data = []
    for f in files:
        fs = h5py.File(f, 'r')
        
        xlim_tmp = np.array(fs['PMF/edge_0'])
        ylim_tmp = np.array(fs['PMF/edge_1'])
        pmf_tmp = np.array(fs['PMF/pmf']).T/4.184 #from kJ/mol to kcal/mol
        pmf_tmp = pmf_tmp - np.min(pmf_tmp)
        P_tmp = pmf2P(pmf_tmp)
        
        data.append([xlim_tmp, ylim_tmp, pmf_tmp, P_tmp])
        
        fs.close()
    return data
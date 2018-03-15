import scipy.io

def load_mat_file(filename, variablename):
    M = scipy.io.loadmat(filename)
    return M[variablename];
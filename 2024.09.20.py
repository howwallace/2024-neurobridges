# %%
import numpy as np

def normalize(arr, axis=None):
    return np.divide(arr, np.sum(arr, axis=axis, keepdims=True))

def expectation(arr, pp, axis=None):
    return np.sum(np.multiply(arr, pp), axis=axis, keepdims=True)

ax_c, ax_x, ax_s = 0, 1, 2
n_c, n_x, n_s = 2, 4, 4

pp_c = normalize(np.random.random((n_c, 1, 1)), axis=ax_c)
pp_x_given_c = normalize(np.random.random((n_c, n_x, 1)), axis=ax_x)
pp_s_given_x = normalize(np.random.random((1, n_x, n_s)), axis=ax_s)

# inference on c
pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)
pp_c_given_s = normalize(np.multiply(pp_s_given_c, pp_c), axis=ax_c)

# beliefs about c
pp_c_given_x = expectation(pp_c_given_s, pp_s_given_x, axis=ax_s)



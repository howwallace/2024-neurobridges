# %%

import numpy as np
import pandas as pd
from scipy.optimize import shgo

# data_path = '/home/hoke/Downloads/gaze_pattern.csv'
data_path = '/home/hoke/Downloads/merged_behavioral.csv'


# %%
# 0-depth inference

def normalize(arr, axis=None):
    return np.divide(arr, np.sum(arr, axis=axis, keepdims=True))

def expectation(arr, pp, axis=None):
    return np.sum(np.multiply(arr, pp), axis=axis, keepdims=True)

def bayes(pp_y_given_x, pp_x, axis):
    '''axis = ax_x'''
    return normalize(np.multiply(pp_y_given_x, pp_x), axis=axis)

def average_bayes(pp_s_given_x, pp_x_given_c, pp_c, ax_c=0, ax_x=1, ax_s=2):
    '''inference on signals s, marginalized over s'''
    # inference on s
    pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)
    pp_c_given_s = bayes(pp_s_given_c, pp_c, axis=ax_c)
    # average beliefs about c
    pp_c_given_x = expectation(pp_c_given_s, pp_s_given_x, axis=ax_s)
    return pp_c_given_x

def pp_s_given_x_free(e_CT, e_CA, e_SA, e_ST):
    '''free parametrizations of signal structure'''
    tmp = np.array([[1 - e_CT, e_CT / 3, e_CT / 3, e_CT / 3],
                    [e_CA / 3, 1 - e_CA, e_CA / 3, e_CA / 3],
                    [e_SA / 3, e_SA / 3, 1 - e_SA, e_SA / 3],
                    [e_ST / 3, e_ST / 3, e_ST / 3, 1 - e_ST]])
    return np.expand_dims(tmp, axis=0)

def pp_s_given_x_symm(e_T, e_A):
    '''symmetric parametrizations of signal structure'''
    return pp_s_given_x_free(e_T, e_A, e_A, e_T)


ax_c, ax_x, ax_s = 0, 1, 2
n_c, n_x, n_s = 2, 4, 4

pp_c = normalize(np.ones((n_c, 1, 1)), axis=ax_c)
pp_x_given_c = normalize(np.array([[1, 1, 0, 0], [0, 0, 1, 1]])[:, :, np.newaxis], axis=ax_x)

# signal structure
pp_s_given_x = pp_s_given_x_free(*(1e-1 * np.random.random(4)))

# average inference
pp_c_given_x = average_bayes(pp_s_given_x, pp_x_given_c, pp_c)

print(pp_s_given_x)
print(pp_c_given_x)


# %%
# max-likelihood

df = pd.read_csv(data_path)
acc = np.array(df['ACC'])
df['i_c'] = (df['GameTask'] == 'Shape').astype(int)
df['i_t'] = (df['CueTransparency'] == 'Arbitrary').astype(int)
df['i_x'] = 3 * df['i_c'] + ( 1 - 2 * df['i_c'] ) * df['i_t']   # to achieve ordering (CT, CA, SA, ST)
df = df[['Subject', 'i_c', 'i_x', 'ACC']]

subject = 3
df = df[df['Subject'] == subject]
print(df.head())

ii_c, ii_x, acc = df['i_c'], df['i_x'], df['ACC']

def likelihood(pp_s_given_x, i_c, i_x, acc):
    '''
    i_c: index of context   (C, S)
    i_x: index of cue       (CT, CA, SA, ST)'''
    pp_c_given_x = average_bayes(pp_s_given_x, pp_x_given_c, pp_c)
    p = pp_c_given_x[i_c, i_x, 0]
    return acc * p + ( 1 - acc ) * ( 1 - p )

to_minimize = lambda args, pp_s_given_x_: -1 * np.sum(np.log(likelihood(pp_s_given_x_(*args), ii_c, ii_x, acc)))

eps = 1e-2

n_params = 4
res_free = shgo(to_minimize, bounds=(n_params * [(eps, 1/2)]), args=(pp_s_given_x_free,))
print(res_free)

n_params = 2
res_symm = shgo(to_minimize, bounds=(n_params * [(eps, 1/2)]), args=(pp_s_given_x_symm,))
print(res_symm)



# %%
# regression of reaction times on changes in trial type




# %%
# 1-depth inference
# i.e. consider transition probabilities between contexts

pp_s_given_c = np.array
pp_c_given_c0 = np.array

ax_c, ax_c0, ax_s, ax_s0 = 0, 1, 2, 3

pp_c0 = np.expand_dims(pp_c, axis=(ax_c, ax_s))
pp_s0_given_c0 = np.expand_dims(pp_s_given_c, axis=(ax_c, ax_s))
pp_c0_given_s0 = normalize(np.multiply(pp_s0_given_c0, pp_c0), axis=ax_c0)
pp_c_given_s0 = expectation(pp_c_given_c0, pp_c0_given_s0, axis=ax_c0)


ax_c, ax_x, ax_s, ax_s0 = 0, 1, 2, 3

pp_s_given_c_s0 = np.expand_dims(pp_s_given_c, axis=ax_s0)  # by conditional independence
pp_c_given_s_s0 = bayes(pp_s_given_c_s0, pp_c_given_s0)



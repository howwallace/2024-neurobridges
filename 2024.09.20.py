# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from functools import reduce
from scipy.optimize import differential_evolution, shgo, basinhopping, dual_annealing, brute, minimize
from matplotlib.colors import LogNorm


def mlims(*lims, k=3e-2, logscale=False):
    mn, mx = lims
    if logscale:
        return np.exp(mlims(*np.log(lims), k=k))
    return mn - k * (mx - mn), mx + k * (mx - mn)

def normalize(arr, axis=None):
    return np.divide(arr, np.sum(arr, axis=axis, keepdims=True))

def expectation(arr, pp, axis=None):
    return np.sum(np.multiply(arr, pp), axis=axis, keepdims=True)


data_path = '/home/hoke/Downloads/gaze_pattern.csv'


# preprocessing:
# - remove demo trials
# - chunk trials into blocks
# - define context and cue indices

df = pd.read_csv(data_path)
df = df[df['SwitchProportion'] != 'Demo']

df['Block'] = df['Subject'] * 100 + np.cumsum(np.diff(df['TrialId'], prepend=0) < 0)
# print(df[['TrialId', 'BlockId']][np.logical_or(df['TrialId'] < 4, df['TrialId'] > 15)])

df['Context'] = (df['GameTask'] == 'Shape').astype(int)
ii_t = (df['CueTransparency'] == 'Arbitrary').astype(int)
df['Cue'] = 3 * df['Context'] + ( 1 - 2 * df['Context'] ) * ii_t   # to achieve ordering (CT, CA, SA, ST)

subjects = np.unique(df['Subject'])
ages = [np.mean(df[df['Subject'] == subject]['Age']) for subject in subjects]


# %%
# 0-depth inference

ax_c, ax_x, ax_s = 0, 1, 2
n_c, n_x, n_s = 2, 4, 2

pp_c = normalize(np.ones((n_c, 1, 1)), axis=ax_c)
pp_x_given_c = normalize(np.array([[1, 1, 0, 0], [0, 0, 1, 1]])[:, :, np.newaxis], axis=ax_x)


def bayes(pp_y_given_x, pp_x, ax_x):
    '''axis = ax_x'''
    return normalize(np.multiply(pp_y_given_x, pp_x), axis=ax_x)

def average_bayes(pp_s_given_x, pp_x_given_c, pp_c, ax_c=0, ax_x=1, ax_s=2):
    '''inference on signals s, marginalized over s'''
    # inference on s
    pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)
    pp_c_given_s = bayes(pp_s_given_c, pp_c, ax_c)
    # average decision about c -- n.b. optimal decision rule to round
    pp_c_given_x = expectation(np.round(pp_c_given_s), pp_s_given_x, axis=ax_s)
    return pp_c_given_x

def pp_s_given_x_free(e_CT, e_CA, e_SA, e_ST):
    '''free parametrizations of signal structure'''
    tmp = np.array([[1 - e_CT, e_CT],
                    [1 - e_CA, e_CA],
                    [e_SA, 1 - e_SA],
                    [e_ST, 1 - e_ST]])
    return np.expand_dims(tmp, axis=0)

def pp_s_given_x_symm(e_T, e_A):
    '''symmetric parametrizations of signal structure'''
    return pp_s_given_x_free(e_T, e_A, e_A, e_T)

def likelihood(pp_s_given_x, i_c, i_x, acc):
    '''
    i_c: index of context   (C, S)
    i_x: index of cue       (CT, CA, SA, ST)'''
    pp_c_given_x = average_bayes(pp_s_given_x, pp_x_given_c, pp_c)
    p = pp_c_given_x[i_c, i_x, 0]
    return acc * p + ( 1 - acc ) * ( 1 - p )

def log_likelihood_paramd(e_T, e_A, ii_c, ii_x, acc):
    pp_s_given_x = pp_s_given_x_symm(e_T, e_A)
    return np.sum(np.log(likelihood(pp_s_given_x, ii_c, ii_x, acc)))

def fit_symm(ii_c, ii_x, acc, eps=1e-3):
    to_minimize = lambda params: -1 * np.sum(np.log(likelihood(pp_s_given_x_symm(*params), ii_c, ii_x, acc)))
    res = shgo(to_minimize, bounds=[(eps, 1/2), (eps, 1/2)])
    return res.success, *res.x, -1 * res.fun

def fit_all_symm():
    ii_c, ii_x, acc = np.array(df['Context']), np.array(df['Cue']), np.array(df['ACC'])
    return fit_symm(ii_c, ii_x, acc)

def fit_subject_symm(subject):
    df_subject = df[df['Subject'] == subject]
    ii_c, ii_x, acc = np.array(df_subject['Context']), np.array(df_subject['Cue']), np.array(df_subject['ACC'])
    return fit_symm(ii_c, ii_x, acc)

def fit_free(ii_c, ii_x, acc, eps=1e-3):
    to_minimize = lambda params: -1 * np.sum(np.log(likelihood(pp_s_given_x_free(*params), ii_c, ii_x, acc)))
    res = shgo(to_minimize, bounds=[(eps, 1/2), (eps, 1/2), (eps, 1/2), (eps, 1/2)])
    return res.success, *res.x, -1 * res.fun

def fit_all_free():
    ii_c, ii_x, acc = np.array(df['Context']), np.array(df['Cue']), np.array(df['ACC'])
    return fit_free(ii_c, ii_x, acc)

def fit_subject_free(subject):
    df_subject = df[df['Subject'] == subject]
    ii_c, ii_x, acc = np.array(df_subject['Context']), np.array(df_subject['Cue']), np.array(df_subject['ACC'])
    return fit_free(ii_c, ii_x, acc)

def bic_(k, n, ll_max):
    return k * np.log(n) - 2 * ll_max


# %%

vmin = 1e-12

s, e_CT, e_CA, e_SA, e_ST, ll_free = fit_all_free()
print(ll_free)
print(bic_(4, len(df), ll_free))

res = []
for subject in subjects:
    res.append(fit_subject_free(subject))
ss, ee_CT, ee_CA, ee_SA, ee_ST, lll_free = np.array(res).T


def format_ax_eps(fig, ax, xlabel='', ylabel=''):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*mlims(1e-3, 1e0, logscale=True))
    ax.set_ylim(*mlims(1e-3, 1e0, logscale=True))
    ax.set_box_aspect(1)
    ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))

im = ax1.scatter(ee_CT, ee_CA, s=16, c=np.exp(lll_free), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)
ax1.scatter(e_CT, e_CA, s=64, c=np.exp(ll_free), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')

im = ax2.scatter(ee_ST, ee_SA, s=16, c=np.exp(lll_free), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)
ax2.scatter(e_ST, e_SA, s=64, c=np.exp(ll_free), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')

format_ax_eps(fig, ax1, '$\epsilon_{CT}$', '$\epsilon_{CA}$')
format_ax_eps(fig, ax2, '$\epsilon_{ST}$', '$\epsilon_{SA}$')

fig.tight_layout()
plt.show()


# %%

vmin = 1e-12

s, e_T, e_A, ll_symm = fit_all_symm()
print(ll_symm)
print(bic_(2, len(df), ll_symm))

res = []
for subject in subjects:
    res.append(fit_subject_symm(subject))
ss, ee_T, ee_A, lll_symm = np.array(res).T


fig, ax = plt.subplots(figsize=(4, 3))

im = ax.scatter(ee_T, ee_A, s=16, c=np.exp(lll_symm), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)

ax.set_xlabel('$\epsilon_T$')
ax.set_ylabel('$\epsilon_A$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(*mlims(1e-3, 1e0, logscale=True))
ax.set_ylim(*mlims(1e-3, 1e0, logscale=True))
ax.set_box_aspect(1)

ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)

ax.scatter(e_T, e_A, s=64, c=np.exp(ll_symm), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')

fig.tight_layout()
plt.show()


# %%
# T-depth inference:
# grow belief in time, i.e. pp_c_given_ss starts out as pp_c_given_s

ax_c, ax_c0, ax_s_ = 0, 1, (lambda t: 2 + t)

pp_c = normalize(np.ones(n_c), axis=ax_c)


def outer_product(arrs):
    return reduce(np.multiply.outer, arrs)

def pp_ss_(pp_s_given_x, ii_x, ax_c=0, ax_c0=1):
    '''
    joint distribution of signal history from actual cue history
    '''
    return np.expand_dims(outer_product(pp_s_given_x[0, ii_x, :]), axis=(ax_c, ax_c0))

def pp_c_given_c0_switch(p_switch):
    '''
    parametrization of transition probabilities between contexts
    '''
    return np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])

def bayes_history(pp_s_given_c, pp_c_given_ss, pp_c_given_c0):
    '''
    evolves inference about c in current trial on the basis of
    conditioned posterior, transition probabilities, and a new signal

    pp_s_given_c  : signal structure
    pp_c_given_ss : conditioned posterior (inference from previous signal history)
    pp_c_given_c0 : (believed) transition probabilities
    '''
    t = len(pp_c_given_ss.shape) - 2

    # update prior, using transitions
    pp_c0_given_ss0 = np.swapaxes(pp_c_given_ss, ax_c, ax_c0)
    pp_c_given_c0_rsh = np.expand_dims(pp_c_given_c0, axis=list(range(ax_s_(0), ax_s_(t))))
    pp_c_given_ss0 = expectation(pp_c_given_c0_rsh, pp_c0_given_ss0, axis=ax_c0)

    # inference for each new signal realization
    pp_c_given_ss0_rsh = np.expand_dims(pp_c_given_ss0, ax_s_(t))
    pp_s_given_c_rsh = np.expand_dims(pp_s_given_c, axis=list(range(ax_s_(0) - 1, ax_s_(t) - 1)))
    pp_c_given_ss = bayes(pp_s_given_c_rsh, pp_c_given_ss0_rsh, ax_c)
    return pp_c_given_ss


def log_likelihood_depth(pp_s_given_x, pp_c, pp_c_given_c0, ii_c, ii_x, acc):
    pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)   # marginal
    pp_c_given_ss = np.expand_dims(pp_c, 1)                             # prior to be iteratively evolved given signal history

    ll = 0
    for t in range(len(ii_x)):
        # evolve posterior to generate new prior
        pp_c_given_ss = bayes_history(pp_s_given_c, pp_c_given_ss, pp_c_given_c0)
        
        # response probabilities in current trial -- n.b. optimal decision to round
        pp_ss = pp_ss_(pp_s_given_x, ii_x[:t + 1])                                                          # probabilities of signal histories
        pp_c = expectation(np.round(pp_c_given_ss), pp_ss, axis=tuple(list(range(ax_s_(0), ax_s_(t + 1))))) # posterior beliefs given true signal history
        p = pp_c.flatten()[ii_c[t]]                                                                         # probability of correct response
        ll += np.log( acc[t] * p + ( 1 - acc[t] ) * ( 1 - p ) )
    return ll


def log_likelihood_paramd_depth_block(e_T, e_A, p_switch, ii_c, ii_x, acc):
    pp_s_given_x = pp_s_given_x_symm(e_T, e_A)
    pp_c_given_c0 = pp_c_given_c0_switch(p_switch)
    return log_likelihood_depth(pp_s_given_x, pp_c, pp_c_given_c0, ii_c, ii_x, acc)


def log_likelihood_paramd_depth(e_T, e_A, p_switch_low, p_switch_high, df):
    ll = 0
    for block in np.unique(df['Block']):
        df_curr = df[df['Block'] == block]
        p_switch = (p_switch_low if df_curr['SwitchProportion'].iloc[0] == 'Low' else p_switch_high)
        ii_c, ii_x, acc = np.array(df_curr['Context']), np.array(df_curr['Cue']), np.array(df_curr['ACC'])
        ll += log_likelihood_paramd_depth_block(e_T, e_A, p_switch, ii_c, ii_x, acc)
    return ll

def fit_depth(df, eps=1e-3):
    # to_minimize = lambda params: -1 * log_likelihood_paramd_depth(*params, df)
    # res = shgo(to_minimize, bounds=[(eps, 1/2), (eps, 1 - eps), (eps, 1 - eps), (eps, 1 - eps)])
    to_minimize = lambda params: -1 * log_likelihood_paramd_depth(*params, 1/4, 3/4, df)
    res = shgo(to_minimize, bounds=[(eps, 1/2), (eps, 1 - eps)])
    # to_minimize = lambda params: -1 * log_likelihood_paramd_depth(eps, eps, *params, df)
    # res = shgo(to_minimize, bounds=[(eps, 1 - eps), (eps, 1 - eps)])
    ll = -1 * res.fun
    return res.success, *res.x, ll, bic_(2, len(df), ll)

def fit_all_depth():
    return fit_depth(df)

def fit_subject_depth(subject):
    df_subject = df[df['Subject'] == subject]
    return fit_depth(df_subject)


# fit_subject_depth(25)
# s, e_T, e_A, l = fit_all_depth()



# %%

res = []
for subject in tqdm.tqdm(subjects):
    res.append(fit_subject_depth(subject))

# ss, ee_T, ee_A, pp_switch_low, pp_switch_high, ll = np.array(res).T
ss, ee_T, ee_A, ll, bbic = np.array(res).T
# ss, pp_switch_low, pp_switch_high, ll = np.array(res).T


# %%

vmin = 1e-12

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.scatter(ee_T, ee_A, s=16, c=np.exp(ll), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)

ax.set_xlabel('$\epsilon_T$')
ax.set_ylabel('$\epsilon_A$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(*mlims(1e-3, 1e0, logscale=True))
ax.set_ylim(*mlims(1e-3, 1e0, logscale=True))
ax.set_box_aspect(1)

ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)

# ax.scatter(e_T, e_A, s=64, c=np.exp(l), norm=LogNorm(vmin=1e-8, vmax=1), cmap='Grays', edgecolors='k')

fig.tight_layout()
plt.show()


# %%

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.scatter(pp_switch_low, pp_switch_high, s=16, c=np.exp(ll), norm=LogNorm(vmin=1e-8, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)

ax.set_xlabel('$p_\mathrm{switch}^L$')
ax.set_ylabel('$p_\mathrm{switch}^H$')

ax.set_xlim(*mlims(0, 1))
ax.set_ylim(*mlims(0, 1))
ax.set_box_aspect(1)

fig.tight_layout()
plt.show()


# %%


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))

im = ax1.scatter(ee_T, ee_A, s=16, c=np.exp(ll), norm=LogNorm(vmin=1e-10, vmax=1e-1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)

ax1.set_xlabel('$\epsilon_T$')
ax1.set_ylabel('$\epsilon_A$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(*mlims(1e-3, 1e0, logscale=True))
ax1.set_ylim(*mlims(1e-3, 1e0, logscale=True))
ax1.set_box_aspect(1)

im = ax2.scatter(pp_switch_low, pp_switch_high, s=16, c=np.exp(ll), norm=LogNorm(vmin=1e-10, vmax=1e-1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)

ax2.set_xlabel('$p_\mathrm{switch}^L$')
ax2.set_ylabel('$p_\mathrm{switch}^H$')
ax2.set_xlim(*mlims(0, 1))
ax2.set_ylim(*mlims(0, 1))
ax2.set_box_aspect(1)

fig.tight_layout()
plt.show()





# %%


# # 0-depth
# #! do this the other way

# subject = 23

# df_curr = df[df['Subject'] == subject]
# ii_c, ii_x, acc = np.array(df_curr['Context']), np.array(df_curr['Cue']), np.array(df_curr['ACC'])

# n_params = 2
# to_minimize_0_depth = lambda params: -1 * log_likelihood_paramd(*params, 1/2, 1/2, ii_c, ii_x, acc)
# res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]))
# print(res)


# %%
# higher-depth


# n_params = 4
# to_minimize_free = lambda params: -1 * log_likelihood_paramd(*params, 1/2, 1/2, ii_c, ii_x, acc)
# res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]))
# print(res)




# %%
# assumes fixed transition probabilities across all blocks
# solution: restrict to high/low

def to_minimize(params, df):
    e_T, e_A, p00, p11 = params
    pp_s_given_x = pp_s_given_x_symm(e_T, e_A)
    pp_c_given_c0 = pp_c_given_c0_free(p00, p11)
    ll = 0
    for block in range(np.max(df['Block'])):
        df_curr = df[df['Block'] == block]
        ii_c, ii_x, acc = np.array(df_curr['Context']), np.array(df_curr['Cue']), np.array(df_curr['ACC'])
        ll += log_likelihood(pp_s_given_x, pp_c, pp_c_given_c0, ii_c, ii_x, acc)
    print(-1 * ll)
    return -1 * ll


eps = 1e-3

n_params = 4
res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]), args=(df_subject,))
print(res)


# %%
print(res)





# %%
# 1-depth inference
# i.e. consider transition probabilities between contexts

pp_s_given_x = pp_s_given_x_free(*(1e-1 * np.random.random(4)))
pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)


n_c, n_x, n_s = 2, 4, 2
# shape in trial t
shape_t_ = lambda s, t, T: np.hstack([s if i == t else [1, 1, 1] for i in range(T)])
shape_cs_t_ = lambda t, T: shape_t_([n_c, 1, n_s], t, T)
# shape for entire history
shape_ = lambda s, T: np.hstack([s for _ in range(T)])
shape_cs_ = lambda T: shape_([n_c, 1, n_s], T)

ax_c_ = lambda t: 3 * t
ax_x_ = lambda t: 3 * t + 1
ax_s_ = lambda t: 3 * t + 2

T = 3
print(shape_cs_(T))


pp_s_given_c_t_ = lambda t: np.reshape(pp_s_given_c, shape_cs_t_(t, T))  # by conditional independence

pp_s_given_c_t_(1).shape


# %%

p00, p11 = 1/4, 1/4
ax_c, ax_c0, ax_s, ax_s0 = 0, 1, 2, 3


pp_c0 = np.expand_dims(pp_c, axis=ax_c)
pp_s0_given_c0 = np.expand_dims(pp_s_given_c, axis=ax_c)
pp_c0_given_s0 = normalize(np.multiply(pp_s0_given_c0, pp_c0), axis=ax_c0)

pp_c_given_c0 = np.expand_dims(pp_c_given_c0_free(p00, p11), axis=(ax_s, ax_s0))
pp_c_given_s0 = expectation(pp_c_given_c0, pp_c0_given_s0, axis=ax_c0)


ax_c, ax_x, ax_s, ax_s0 = 0, 1, 2, 3

pp_s_given_c_s0 = np.stack(2 * [pp_s_given_c], axis=ax_s0)  # by conditional independence
pp_c_given_s_s0 = bayes(pp_s_given_c_s0, pp_c_given_s0, ax_c)

pp_s_given_x = np.expand_dims(pp_s_given_x, ax_s0)
print(pp_s_given_x.shape)

# marginalize over x
pp_c_given_x_s0 = expectation(pp_c_given_s_s0, pp_s_given_x, axis=ax_s)

print(pp_c_given_x_s0.shape)

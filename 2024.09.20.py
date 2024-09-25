# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from functools import reduce
from scipy.optimize import differential_evolution, shgo, basinhopping, dual_annealing, brute, minimize
from matplotlib.colors import LogNorm


def format_ax_eps(fig, ax, xlabel='', ylabel=''):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*mlims(1e-3, 1e0, logscale=True))
    ax.set_ylim(*mlims(1e-3, 1e0, logscale=True))
    ax.set_box_aspect(1)
    ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)

def mlims(*lims, k=3e-2, logscale=False):
    mn, mx = lims
    if logscale:
        return np.exp(mlims(*np.log(lims), k=k))
    return mn - k * (mx - mn), mx + k * (mx - mn)

def normalize(arr, axis=None):
    return np.divide(arr, np.sum(arr, axis=axis, keepdims=True))

def expectation(arr, pp, axis=None):
    return np.sum(np.multiply(arr, pp), axis=axis, keepdims=True)

def bic_(k, n, ll_max):
    return k * np.log(n) - 2 * ll_max


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
ages = np.array([np.mean(df[df['Subject'] == subject]['Age']) for subject in subjects])


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

def fit(df, pp_s_given_x_, bounds):
    ii_c, ii_x, acc = np.array(df['Context']), np.array(df['Cue']), np.array(df['ACC'])

    # to_minimize = lambda params: -1 * np.sum(np.log(likelihood(pp_s_given_x_(*params), ii_c, ii_x, acc)))
    # res = differential_evolution(to_minimize, bounds=bounds)
    # ll = -1 * res.fun
    # return res.success, *res.x, ll, bic_(len(bounds), len(acc), ll)

    to_minimize = lambda log_params: -1 * np.sum(np.log(likelihood(pp_s_given_x_(*np.exp(log_params)), ii_c, ii_x, acc)))
    res = differential_evolution(to_minimize, bounds=np.log(bounds))
    ll = -1 * res.fun
    return res.success, *np.exp(res.x), ll, bic_(len(bounds), len(acc), ll)

def fit_free(df, eps=1e-3):
    return fit(df, pp_s_given_x_free, [(eps, 1/2), (eps, 1/2), (eps, 1/2), (eps, 1/2)])

def fit_symm(df, eps=1e-3):
    return fit(df, pp_s_given_x_symm, [(eps, 1/2), (eps, 1 - eps)])


# parameter recovery

def recover_symm(e_T, e_A, N=600):
    '''parameter recovery'''
    # generate random trials
    ii_x = np.random.choice(4, N)
    ii_c = (ii_x > 1).astype(int)

    pp_s_given_x = pp_s_given_x_symm(e_T, e_A)
    pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)
    pp_c_given_s = bayes(pp_s_given_c, pp_c, ax_c)
    pp_ss = pp_s_given_x[0, ii_x, :]
    ss = np.array([np.random.choice([0, 1], p=pp_s) for pp_s in pp_ss])
    ii_crec = np.argmax(pp_c_given_s[:, 0, ss], axis=0)
    acc = (ii_crec == ii_c)
    df_sim = pd.DataFrame({'Context': ii_c, 'Cue': ii_x, 'ACC': acc})

    return fit_symm(df_sim)

niter = 20

e_T, e_A = 8e-2, 3e-2

res = []
for _ in range(niter):
    res.append(recover_symm(e_T, e_A))
ss, ee_T, ee_A, lll, bbic = np.array(res).T

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(e_T, e_A, s=64, c='r')
ax.scatter(ee_T, ee_A, s=16, c='none', edgecolors='k')
format_ax_eps(fig, ax, '$\epsilon_T$', '$\epsilon_A$')

fig.tight_layout()
plt.savefig('parameter_recovery.svg')
plt.savefig('parameter_recovery.png', dpi=500)
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

    # to_minimize = lambda log_params: -1 * log_likelihood_paramd_depth(*np.exp(log_params), df)
    # res = shgo(to_minimize, bounds=np.log([(eps, 1/2), (eps, 1 - eps), (eps, 1 - eps), (eps, 1 - eps)]))
    # to_minimize = lambda log_params: -1 * log_likelihood_paramd_depth(*np.exp(log_params), 1/4, 3/4, df)
    # res = shgo(to_minimize, bounds=np.log([(eps, 1/2), (eps, 1 - eps)]))

    ll = -1 * res.fun
    return res.success, *res.x, ll, bic_(2, len(df), ll)
    # return res.success, *np.exp(res.x), ll, bic_(2, len(df), ll)


# df_agegroup = df[df['Age'] < 100]
# fit_depth(df_agegroup)


# %%
# fits by subject -- free


vmin = 1e-12

s, e_CT, e_CA, e_SA, e_ST, ll_free, bic_free = fit_free(df)
print(ll_free)
print(bic_(4, len(df), ll_free))

res = []
for subject in subjects:
    df_subject = df[df['Subject'] == subject]
    res.append(fit_free(df_subject))
ss, ee_CT, ee_CA, ee_SA, ee_ST, lll_free, bbic_free = np.array(res).T


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
plt.savefig('fits_subjects_free.svg')
plt.savefig('fits_subjects_free.png', dpi=500)
plt.show()



# %%
# fits by subject

vmin = 1e-12

res_symm, res_depth = [], []
for subject in tqdm.tqdm(subjects):
    df_subject = df[df['Subject'] == subject]
    res_symm.append(fit_symm(df_subject))
    res_depth.append(fit_symm(df_subject))


# symm

s, e_T, e_A, ll, bic = fit_symm(df)
ss, ee_T, ee_A, lll, bbic = np.array(res_symm).T

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.scatter(ee_T, ee_A, s=16, c=np.exp(lll), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)
format_ax_eps(fig, ax, '$\epsilon_T$', '$\epsilon_A$')

ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)
ax.scatter(e_T, e_A, s=64, c=np.exp(ll), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')

fig.tight_layout()
plt.savefig('fits_subjects_symm.svg')
plt.savefig('fits_subjects_symm.png', dpi=500)
plt.show()


# depth

s, e_T, e_A, ll, bic = fit_depth(df)
ss, ee_T, ee_A, lll, bbic = np.array(res_depth).T

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.scatter(ee_T, ee_A, s=16, c=np.exp(lll), norm=LogNorm(vmin=vmin, vmax=1), cmap='Grays', edgecolors='k')
cbar = fig.colorbar(im)
format_ax_eps(fig, ax, '$\epsilon_T$', '$\epsilon_A$')

ax.axline((0, 0), (1, 1), c='k', ls='dashed', lw=1)
ax.scatter(e_T, e_A, s=64, c=np.exp(ll), norm=LogNorm(vmin=1e-8, vmax=1), cmap='Grays', edgecolors='k')

fig.tight_layout()
plt.savefig('fits_subjects_depth.svg')
plt.savefig('fits_subjects_depth.png', dpi=500)
plt.show()



# %%
# fits by age group -- free

w = 12

res = []
age_space = np.linspace(np.min(ages), np.max(ages), 30)
for age in tqdm.tqdm(age_space):
    df_agegroup = df[np.logical_and(df['Age'] > age - w, df['Age'] < age + w)]
    res.append([*fit_free(df_agegroup), len(df_agegroup)])


# symm

ss, ee_CT, ee_CA, ee_SA, ee_ST, lll, bbic, nn = np.array(res).T

t_n = 600

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.plot(age_space[nn > t_n], ee_CT[nn > t_n], label='$\epsilon_{CT}$')
ax1.plot(age_space[nn > t_n], ee_CA[nn > t_n], label='$\epsilon_{CA}$')
ax1.plot(age_space[nn > t_n], ee_SA[nn > t_n], label='$\epsilon_{SA}$')
ax1.plot(age_space[nn > t_n], ee_ST[nn > t_n], label='$\epsilon_{ST}$')

ax1.set_ylim(*mlims(0, 4e-1))
ax1.set_xlabel('age [months]')
ax1.set_ylabel('$\epsilon^\mathrm{fit}$')
ax1.set_box_aspect(1)
ax1.legend()

ax2.plot(age_space[nn > t_n], (lll / nn)[nn > t_n], c='k')
ax2.set_ylabel('$LL / n$')
ax2.set_box_aspect(1)

fig.tight_layout()
plt.savefig('fits_ages_free.svg')
plt.savefig('fits_ages_free.png', dpi=500)
plt.show()


# %%
# fits by age group (window w = 12 gives at least 6 participants per group)

w = 12

res_symm, res_depth = [], []
age_space = np.linspace(np.min(ages), np.max(ages), 30)
for age in tqdm.tqdm(age_space):
    df_agegroup = df[np.logical_and(df['Age'] > age - w, df['Age'] < age + w)]
    res_symm.append([*fit_symm(df_agegroup), len(df_agegroup)])
    res_depth.append([*fit_depth(df_agegroup), len(df_agegroup)])


# %%

# symm

ss, ee_T, ee_A, lll, bbic, nn = np.array(res_symm).T

t_n = 600

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.plot(age_space[nn > t_n], ee_T[nn > t_n], label='$\epsilon_T$')
ax1.plot(age_space[nn > t_n], ee_A[nn > t_n], label='$\epsilon_A$')

ax1.set_ylim(*mlims(0, 3e-1))
ax1.set_xlabel('age [months]')
ax1.set_ylabel('$\epsilon^\mathrm{fit}$')
ax1.set_box_aspect(1)
ax1.legend()

ax2.plot(age_space[nn > t_n], (lll / nn)[nn > t_n], c='k')
ax2.set_ylabel('$LL / n$')
ax2.set_box_aspect(1)

fig.tight_layout()
plt.savefig('fits_ages_symm.svg')
plt.savefig('fits_ages_symm.png', dpi=500)
plt.show()


# depth

ss, ee_T, ee_A, lll, bbic, nn = np.array(res_depth).T

t_n = 600

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.plot(age_space[nn > t_n], ee_T[nn > t_n], label='$\epsilon_T$')
ax1.plot(age_space[nn > t_n], ee_A[nn > t_n], label='$\epsilon_A$')

ax1.set_ylim(*mlims(0, 3e-1))
ax1.set_xlabel('age [months]')
ax1.set_ylabel('$\epsilon^\mathrm{fit}$')
ax1.set_box_aspect(1)
ax1.legend()

ax2.plot(age_space[nn > t_n], (lll / nn)[nn > t_n], c='k')
ax2.set_ylabel('$LL / n$')
ax2.set_box_aspect(1)

fig.tight_layout()
plt.savefig('fits_ages_depth.svg')
plt.savefig('fits_ages_depth.png', dpi=500)
plt.show()



# %%

ss, ee_T, ee_A, lll, bbic, nn = np.array(res_symm).T
print(ee_T)

ss, ee_T, ee_A, lll, bbic, nn = np.array(res_depth).T
print(ee_T)


# %%


# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))

# im = ax1.scatter(ee_T, ee_A, s=16, c=np.exp(ll), norm=LogNorm(vmin=1e-10, vmax=1e-1), cmap='Grays', edgecolors='k')
# cbar = fig.colorbar(im)

# ax1.set_xlabel('$\epsilon_T$')
# ax1.set_ylabel('$\epsilon_A$')
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlim(*mlims(1e-3, 1e0, logscale=True))
# ax1.set_ylim(*mlims(1e-3, 1e0, logscale=True))
# ax1.set_box_aspect(1)

# im = ax2.scatter(pp_switch_low, pp_switch_high, s=16, c=np.exp(ll), norm=LogNorm(vmin=1e-10, vmax=1e-1), cmap='Grays', edgecolors='k')
# cbar = fig.colorbar(im)

# ax2.set_xlabel('$p_\mathrm{switch}^L$')
# ax2.set_ylabel('$p_\mathrm{switch}^H$')
# ax2.set_xlim(*mlims(0, 1))
# ax2.set_ylim(*mlims(0, 1))
# ax2.set_box_aspect(1)

# fig.tight_layout()
# plt.show()




# # 0-depth
# #! do this the other way

# subject = 23

# df_curr = df[df['Subject'] == subject]
# ii_c, ii_x, acc = np.array(df_curr['Context']), np.array(df_curr['Cue']), np.array(df_curr['ACC'])

# n_params = 2
# to_minimize_0_depth = lambda params: -1 * log_likelihood_paramd(*params, 1/2, 1/2, ii_c, ii_x, acc)
# res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]))
# print(res)


# higher-depth


# n_params = 4
# to_minimize_free = lambda params: -1 * log_likelihood_paramd(*params, 1/2, 1/2, ii_c, ii_x, acc)
# res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]))
# print(res)




# # %%
# # assumes fixed transition probabilities across all blocks
# # solution: restrict to high/low

# def to_minimize(params, df):
#     e_T, e_A, p00, p11 = params
#     pp_s_given_x = pp_s_given_x_symm(e_T, e_A)
#     pp_c_given_c0 = pp_c_given_c0_free(p00, p11)
#     ll = 0
#     for block in range(np.max(df['Block'])):
#         df_curr = df[df['Block'] == block]
#         ii_c, ii_x, acc = np.array(df_curr['Context']), np.array(df_curr['Cue']), np.array(df_curr['ACC'])
#         ll += log_likelihood(pp_s_given_x, pp_c, pp_c_given_c0, ii_c, ii_x, acc)
#     print(-1 * ll)
#     return -1 * ll


# eps = 1e-3

# n_params = 4
# res = differential_evolution(to_minimize, bounds=(n_params * [(eps, 1/2)]), args=(df_subject,))
# print(res)


# # %%
# print(res)





# # %%
# # 1-depth inference
# # i.e. consider transition probabilities between contexts

# pp_s_given_x = pp_s_given_x_free(*(1e-1 * np.random.random(4)))
# pp_s_given_c = expectation(pp_s_given_x, pp_x_given_c, axis=ax_x)


# n_c, n_x, n_s = 2, 4, 2
# # shape in trial t
# shape_t_ = lambda s, t, T: np.hstack([s if i == t else [1, 1, 1] for i in range(T)])
# shape_cs_t_ = lambda t, T: shape_t_([n_c, 1, n_s], t, T)
# # shape for entire history
# shape_ = lambda s, T: np.hstack([s for _ in range(T)])
# shape_cs_ = lambda T: shape_([n_c, 1, n_s], T)

# ax_c_ = lambda t: 3 * t
# ax_x_ = lambda t: 3 * t + 1
# ax_s_ = lambda t: 3 * t + 2

# T = 3
# print(shape_cs_(T))


# pp_s_given_c_t_ = lambda t: np.reshape(pp_s_given_c, shape_cs_t_(t, T))  # by conditional independence

# pp_s_given_c_t_(1).shape


# # %%

# p00, p11 = 1/4, 1/4
# ax_c, ax_c0, ax_s, ax_s0 = 0, 1, 2, 3


# pp_c0 = np.expand_dims(pp_c, axis=ax_c)
# pp_s0_given_c0 = np.expand_dims(pp_s_given_c, axis=ax_c)
# pp_c0_given_s0 = normalize(np.multiply(pp_s0_given_c0, pp_c0), axis=ax_c0)

# pp_c_given_c0 = np.expand_dims(pp_c_given_c0_free(p00, p11), axis=(ax_s, ax_s0))
# pp_c_given_s0 = expectation(pp_c_given_c0, pp_c0_given_s0, axis=ax_c0)


# ax_c, ax_x, ax_s, ax_s0 = 0, 1, 2, 3

# pp_s_given_c_s0 = np.stack(2 * [pp_s_given_c], axis=ax_s0)  # by conditional independence
# pp_c_given_s_s0 = bayes(pp_s_given_c_s0, pp_c_given_s0, ax_c)

# pp_s_given_x = np.expand_dims(pp_s_given_x, ax_s0)
# print(pp_s_given_x.shape)

# # marginalize over x
# pp_c_given_x_s0 = expectation(pp_c_given_s_s0, pp_s_given_x, axis=ax_s)

# print(pp_c_given_x_s0.shape)

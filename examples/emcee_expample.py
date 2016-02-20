import cPickle as pik

from phlo_mcmc.mcmc_lik import run_emcee

# run from command line:
# $ mpirun -n 8 ipython2 emcee_expample.py
# will use local dir to store likelhood evals and maticies
# path to lagrange
cmd = 'lagrange/src/lagrange_cpp'
# path to region files created by lagrange
relbio_path = 'relbiogeog_newtree.tre'
save_file = 'emcee_run1/'
mat_size = 5
# emcee run
lnprobability, param = run_emcee(cmd, save_file, burnin=100, itter=10 ** 6, rm_file=relbio_path,
                                 matrix_size=mat_size)

# save mcmc chains
pik.dump((lnprobability, param), open('emcee_results.pik', 'w'), 2)

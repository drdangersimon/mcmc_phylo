from phlo_mcmc.mcmc_lik import PhyloSampler_multinest, run_multinest

# run from command line:
# $ mpirun -n 8 ipython2 multinest_example.py
# will use local dir to store likelhood evals and maticies
# path to lagrange
cmd = 'lagrange/src/lagrange_cpp'
# path to region files created by lagrange
relbio_path = 'relbiogeog_newtree.tre'
mat_size = 5
# multinest run
posterior = PhyloSampler_multinest(cmd, relbio_path, mat_size)
# set constraints or regions
n = ['GC', 'NAM', 'D', 'EA', 'MED']
posterior.set_contraints('EA', 'NAM', 0., n)
posterior.set_contraints('EA', 'MED', 0., n)
posterior.set_contraints('EA', 'GC', 0., n)
posterior.set_contraints('D', 'MED', 0., n)

# path to where multinest will put outputs
output_path = 'run_ix/'
# run sampler
run_multinest(posterior, output_path)

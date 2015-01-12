import make_matrix as mm
import numpy as np
import sys
import emcee
import cPickle as pik
from shutil import copyfile
try:
    import mpi4py.MPI as mpi
except ImportError:
    mpi = None

'''Does mcmc with an alfine invaranet ensamble sampler.
Should be more reliable and have faster covergence than original mcmc.
Also has plotting routines for seeing results.'''


class PhyloSampler(object):

    def __init__(self, cmd, rm_file, matrix_size):
        '''initalize stuff'''
        # make new file so no overwritting
        gid = mpi.COMM_WORLD.rank
        self.gid = gid
        copyfile(rm_file, rm_file + str(gid))
        # Rename the temp matrix file
        relbiogeo = open(rm_file + str(gid))
        temp_file = [line for line in relbiogeo]
        relbiogeo.close()
        cpu_relbiogeo = open(rm_file + str(gid), 'w')
        for line in temp_file:
            # Write file with correct matrix file
            if line.find('ratematrix') >= 0:
                line = line.split('=')[0] + '=%i.rm\n'%gid
            # Copy tree to new file
            if line.find('treefile') >= 0:
                copyfile(line.split('=')[1][:-1], '%i.tre'%gid)
                line = line.split('=')[0] + '=%i.tre\n'%gid
            cpu_relbiogeo.write(line)
        cpu_relbiogeo.close()
        # save paths for later
        self.cmd = cmd
        self.rm_file = rm_file #+ str(mpi.COMM_WORLD.rank)
        self.matrix_path = '%i.rm'%mpi.COMM_WORLD.rank
        self.matrix_size = matrix_size
        # get dimenmsions and number of walkers
        self.ndim = self.get_dim()
        self.nwalkers = self.ndim * 3
        # check if odd
        if not self.nwalkers % 2 == 0:
            self.nwalkers += 1
        
    def lik(self, param):
        '''Likihood call from lagrange'''
        mat = mm.get_mat(param, self.matrix_size)
        # weird bug
        mm.write_matrix(mat, '%i.rm'%mpi.COMM_WORLD.rank)
        #mm.write_matrix(mat, self.matrix_path)
        return mm.call_laplace(self.cmd, self.rm_file + str(mpi.COMM_WORLD.rank))

    def prior(self, param):
        '''Priors and constraints for the mcmc'''
        # uniform prior between 0 and 1
        if np.any(np.asarray(param) > 1):
            return -np.inf
        if np.any(np.asarray(param) < 0):
            return -np.inf
        # log (uniform[0,1]) = 0
        return 0.
        #truncated normal beween 0 and 1
        
    def get_dim(self):
        '''Gets number of paramters for lower triangle of matrix'''
        x ,y = np.triu_indices(self.matrix_size)
        # remove diagonal terms
        del_array = []
        for i in xrange(len(x)):
            if x[i] == y[i]:
                del_array.append(i)
        x = np.delete(x, del_array)
        y = np.delete(y, del_array)
        return len(y)

    def __call__(self, param):
        ''' Calls postieror at position'''
        prior = self.prior(param)
        if np.isfinite(prior):
            return prior + self.lik(param)
        else:
            return prior

    def clean(self):
        '''removes files that were made for mcmc'''
        os.remove(self.rm_file)
        os.remove(self.matrix_path)

    def pos0(self):
        '''returns inital position for mcmc in emcee format'''
        out_pos = [np.random.rand(self.ndim)
                   for walker in xrange(self.nwalkers)]
        return np.asarray(out_pos)




def run(cmd, save_file, burnin=100, itter=10**6, rm_file='relbiogeog_1.lg',
         matrix_size=5):
    '''Does mcmc with an alfine sampler. Uses MPI for parallel'''
    assert not mpi is None, 'You must install mpi4py for this to work'
    posterior = PhyloSampler(cmd, rm_file, matrix_size)
    # pool object for multiprocessing
    pool = emcee.mpi_pool.MPIPool(loadbalance=True)
    # make sampler
    if not pool.is_master():
        pool.wait()
        posterior.clean()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(posterior.nwalkers, posterior.ndim, posterior, pool=pool)
    # burn in
    print 'Starting burn-in'
    for pos0, prob, rstate in sampler.sample(posterior.pos0(), iterations=burnin):
        show = 'Burn-in: mean lik=%f,'% np.mean(prob)
        show += 'std lik=%2.2f,'%np.std(prob)
        show += 'acceptance rate=%0.2f'%np.nanmean(sampler.acceptance_fraction)
        print show
    
    #pos, prob, rstate = sampler.sample(posterior.pos0(), iterations=burnin)
    # remove burin samples
    sampler.reset()
    #save state every 100 iterations
    i,j,ess = 0,0,0
    for pos, prob, rstate in sampler.sample(pos0, iterations=itter/mpi.COMM_WORLD.size, rstate0=rstate):
        show = 'Run: mean lik=%f, '%np.mean(prob)
        show += 'std lik=%2.2f, '%np.std(prob)
        show += 'acceptance rate=%0.2f'%np.nanmean(sampler.acceptance_fraction)
        print show, i,ess
        if i % 100 == 0:
            pik.dump(sampler, open(save_file+'.bkup', 'w'), 2)
            ess = (sampler.flatchain.shape[0]/
               np.nanmin(sampler.get_autocorr_time()))
            if ess > 10**4:
                j += 1
                if j > 1:
                    break
        i += 1
    # save chains [prob, chain list]
    pik.dump((sampler.flatlnprobability, sampler.flatchain),
             open(save_file, 'w'), 2)
    #flatten
    flatten = lambda x, size: np.ravel(mm.get_mat(x, size))
    param = np.asarray(map(flatten, sampler.flatchain, [matrix_size]*len(sampler.flatchain)))
        
    return sampler.flatlnprobability, param

def mkhrd(s):
    ''' takes string with spaces and makes header assuming raveled
    indexs'''
    matrix_size = len(s.split(' '))
    s = s.split(' ')
    return [x+y for x in s for y in s]

if __name__ == '__main__':
    cmd = 'lagrange/src/lagrange_cpp'
    chi, mat_list = run(cmd, '5regions.csv', 1000, 10**6, rm_file='relbiogeog_5regions.lg', matrix_size=5)
    # save as csv
    # make header from areanames
    header = mkhrd('GC NAM D EA MED')
    out = np.zeros((mat_list.shape[0], mat_list.shape[1] + 1))
    out[:, 0] = chi
    out[:,1:] = mat_list
    np.savetxt('5_regions_final.csv', out,
               delimiter=',', header=header)
    # 7 regions
    chi, mat_list = run(cmd, '7regions.csv', 1000, 10**6, rm_file='relbiogeog_7regions.lg', matrix_size=7)    
    header = mkhrd('GC NAM D EA MED WA SA')
    out = np.zeros((mat_list.shape[0], mat_list.shape[1] + 1))
    out[:, 0] = chi
    out[:,1:] = mat_list
    np.savetxt('7_regions_final.csv', out,
               delimiter=',', header=header)
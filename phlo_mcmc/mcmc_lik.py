import cPickle as pik
import os
import sys
from shutil import copyfile

import emcee
import numpy as np
import pymultinest

import make_matrix as mm

try:
    import mpi4py.MPI as mpi
except ImportError:
    mpi = None

"""Does mcmc with an alfine invariant ensemble sampler.
Should be more reliable and have faster convergence than original mcmc.
Also has plotting routines for seeing results."""


class PhyloSampler(object):
    """

    """

    def __init__(self, cmd, rm_file, matrix_size):
        """initalize stuff"""
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
                line = line.split('=')[0] + '=%i.rm\n' % gid
            # Copy tree to new file
            if line.find('treefile') >= 0:
                copyfile(line.split('=')[1][:-1], '%i.tre' % gid)
                line = line.split('=')[0] + '=%i.tre\n' % gid
            cpu_relbiogeo.write(line)
        cpu_relbiogeo.close()
        # save paths for later
        self.cmd = cmd
        self.rm_file = rm_file  # + str(mpi.COMM_WORLD.rank)
        self.matrix_path = '%i.rm' % mpi.COMM_WORLD.rank
        self.matrix_size = matrix_size
        # get dimenmsions and number of walkers
        self.ndim = self.get_dim()
        self.nwalkers = self.ndim * 3
        # check if odd
        if not self.nwalkers % 2 == 0:
            self.nwalkers += 1

    def lik(self, param):
        """Calculates likelihood from lagrange
        :param param: list or ndarray upper triangular matrix elements"""
        # turn params into a matrix
        mat = mm.get_mat(param, self.matrix_size)
        # write with unique id for multiprocessing
        mm.write_matrix(mat, '%i.rm' % mpi.COMM_WORLD.rank)
        # returns likelihood read from output of laplace
        return mm.call_laplace(self.cmd, self.rm_file + str(mpi.COMM_WORLD.rank))

    def prior(self, param):
        """Calculates prior from parameters. Uses uniform distribution and limits between 1 and 0
        :param para: list or ndarray upper triangular matrix elements"""
        # uniform prior between 0 and 1
        if np.any(np.asarray(param) > 1):
            return -np.inf
        if np.any(np.asarray(param) < 0):
            return -np.inf
        # log (uniform[0,1]) = 0
        return 0.
        # truncated normal beween 0 and 1

    def get_dim(self):
        """Gets number of parameters for lower triangle of matrix"""
        x, y = np.triu_indices(self.matrix_size)
        # remove diagonal terms
        del_array = []
        for i in xrange(len(x)):
            if x[i] == y[i]:
                del_array.append(i)
        x = np.delete(x, del_array)
        y = np.delete(y, del_array)
        return len(y)

    def __call__(self, param):
        """ Calls posterior at position
        :param param: list or ndarray upper triangular matrix elements
        :return float of posterior at param"""
        prior = self.prior(param)
        if np.isfinite(prior):
            return prior + self.lik(param)
        else:
            return prior

    def clean(self):
        """Cleans up files that were made for mcmc"""
        os.remove(self.rm_file)
        os.remove(self.matrix_path)

    def pos0(self):
        """returns initial position for mcmc in emcee format"""
        out_pos = [np.random.rand(self.ndim)
                   for walker in xrange(self.nwalkers)]
        return np.asarray(out_pos)


class PhyloSampler_multinest(PhyloSampler):
    """just changes how the input and output of lik and prior"""

    def lik(self, cube, ndim, nparams, lnew):
        """Likihood call from lagrange"""
        # turn cube into param
        param = np.asarray([0. for i in range(self.ndim)])
        # add removed params
        i = 0
        for index in range(self.ndim):
            # check if in constraints
            for con_index in self.constraint:
                if index == con_index[0]:
                    param[index] = con_index[1]
                    break
            else:
                param[index] = cube[i]
                i += 1
        mat = mm.get_mat(param, self.matrix_size)
        # weird bug
        mm.write_matrix(mat, '%i.rm' % mpi.COMM_WORLD.rank)
        # mm.write_matrix(mat, self.matrix_path)
        return mm.call_laplace(self.cmd, self.rm_file + str(mpi.COMM_WORLD.rank))

    def prior(self, cube, ndim, nparams):
        """uniform prior and make some param delta function. No return!"""
        pass

    def set_contraints(self, region1, region2, value, names):
        """Sets priors to delta functions on specific regions. Regions should be str"""
        assert isinstance(region2, str) and isinstance(region1, str)
        assert region2 in names and region1 in names
        # make constraint files
        if not hasattr(self, 'constraint'):
            self.constraint = []
        # sloppy!!! get coord for places
        x, y = np.triu_indices(self.matrix_size)
        # remove diagonal terms
        del_array = []
        for i in xrange(len(x)):
            if x[i] == y[i]:
                del_array.append(i)
        x = np.delete(x, del_array)
        y = np.delete(y, del_array)
        # find coords
        for index in range(len(x)):
            if names[x[index]] == region1 and names[y[index]] == region2:
                self.constraint.append([index, value])
            elif names[y[index]] == region1 and names[x[index]] == region2:
                self.constraint.append([index, value])

    def get_dim(self):
        """ Gets number of dimenmsions for model. With the constrains included"""
        x, y = np.triu_indices(self.matrix_size)
        # remove diagonal terms
        del_array = []
        for i in xrange(len(x)):
            if x[i] == y[i]:
                del_array.append(i)
        x = np.delete(x, del_array)
        y = np.delete(y, del_array)
        if not hasattr(self, 'constraint'):
            self.constraint = []
            return len(y)
        else:
            return len(y) - len(self.constraint)


def mkhrd(s, x, y):
    """ takes string with spaces and makes header assuming raveled
    indexs"""
    # make sure indexs are same length
    assert len(x) == len(y)
    header = []
    for i in range(len(x)):
        header.append(s[x[i]] + '_' + s[y[i]])
    return header


def run_multinest(posterior, save_file):
    """Uses multinest sampler to calculate evidence instead of emceee
    cmd is bash command to call lagrange_cpp
    posterior is posterior class, should have methods prior and lik
    save_file is path to save. will resume from file if arround"""
    # checks
    # if path exsissts
    if not os.path.exists(save_file) and mpi.COMM_WORLD.rank == 0:
        os.mkdir(save_file)
    assert hasattr(posterior, 'prior') and hasattr(posterior, 'lik'), 'must have prior and lik methods'
    # run sampler
    pymultinest.run(posterior.lik, posterior.prior, posterior.get_dim(),
                    outputfiles_basename=save_file)


def run_emcee(cmd, save_file, burnin=100, itter=10 ** 6, rm_file='relbiogeog_1.lg',
        matrix_size=5):
    """Does mcmc with an alfine sampler. Uses MPI for parallel"""
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
        show = 'Burn-in: mean lik=%f,' % np.mean(prob)
        show += 'std lik=%2.2f,' % np.std(prob)
        show += 'acceptance rate=%0.2f' % np.nanmean(sampler.acceptance_fraction)
        print show

    # pos, prob, rstate = sampler.sample(posterior.pos0(), iterations=burnin)
    # remove burin samples
    sampler.reset()
    # save state every 100 iterations
    i, j, ess = 0, 0, 0
    for pos, prob, rstate in sampler.sample(pos0, iterations=itter, rstate0=rstate):
        show = 'Run: mean lik=%f, ' % np.mean(prob)
        show += 'std lik=%2.2f, ' % np.std(prob)
        show += 'acceptance rate=%0.2f' % np.nanmean(sampler.acceptance_fraction)
        print show, i, ess
        if i % 100 == 0 and i > 1:
            # ipdb.set_trace()
            pik.dump((sampler.flatchain[:sampler.iterations], sampler.flatlnprobability[:sampler.iterations], pos, prob,
                      rstate), open(save_file + '.bkup', 'w'), 2)
            ess = (sampler.iterations /
                   np.nanmin(sampler.get_autocorr_time()))
            if ess > 10 ** 4:
                j += 1
                if j > 1:
                    break
                else:
                    j = 0
        i += 1
    # save chains [prob, chain list]
    pik.dump((sampler.flatchain, sampler.flatlnprobability, ess),
             open(save_file, 'w'), 2)
    # flatten
    flatten = lambda x, size: np.ravel(mm.get_mat(x, size))
    param = np.asarray(map(flatten, sampler.flatchain, [matrix_size] * len(sampler.flatchain)))

    return sampler.flatlnprobability, param

import os
import subprocess
from math import exp
import random
from itertools import product
import numpy as nu
from multiprocessing import Queue, cpu_count, Process
from shutil import copyfile


def call_laplace(cmd, run_file):
    if not os.path.exists(cmd):
        raise OSError('Cannont find "%s." Please check path'%cmd)
    if  not os.path.exists(run_file):
        raise OSError('Your matrix does not exists.')
    run_cmd = cmd + ' ' + run_file
    out_put = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_put.wait()
    error = out_put.stderr.read()
    if len(error):
        raise OSError(error)
    output = out_put.stdout.read().split('\n')
    for text in output:
        #print text.find('final')
        if text.find('final') >= 0:
            return -float(text.split(' ')[-1])
    #print output
    return -nu.inf

def generate_inital_matrix(row=5,col=5):
    '''Gets inital martix only lower triangular values and diagonals
    are == 1'''
    values = [0, 0.5, 1]
    mat = []
    for i in range(row):
        mat.append(range(col))
    for Row in xrange(row):
        for Col in xrange(col):
            mat[Row][Col] = random.sample(values, 1)[0]
            # Set diagonals to 1
            if Row == Col:
                 mat[Row][Row] = 1
    return mat
    

def chng_matrix_item(mat, index, values=[0, 0.5, 1]):
    '''Changes index in matrix. index is raveled index of matrix 0=[0,0]'''
    new_mat  = nu.copy(mat)
    x, y = get_lower_tri(new_mat.shape[0])
    # change all values in index
    for i in index:
        row, col = nu.unravel_index(i, new_mat.shape)
        new_mat[row, col] = random.sample(values, 1)[0]
    # If index isn't complte draw new indexs from lower triangle
    if not len(index) == len(x):
       sample = nu.ravel_multi_index((x,y),(5,5))
       index = nu.random.choice(sample, len(index), replace=False)
    return [list(i) for i in list(new_mat)], list(index)
        
def write_matrix(mat, filename):
    matrix_file = open(filename, 'w')
    for row in mat:
        matrix_file.write(' '.join(str(row)[1:-1].split(','))+'\n')
    matrix_file.close()

def get_lik(mat_list, cmd, rm_file):
    '''Multiprocessing lik calc'''
    mat = get_mat(mat_list, matrix_size)
    write_matrix(mat, 'mcmc_run.rm')
    return mat, call_laplace(cmd, rm_file)

def get_lower_tri(matrix_size=5):
    '''returns indexs of lower matrix, without diagionals'''
    x ,y = nu.triu_indices(matrix_size)
    # remove diagonal terms
    del_array = []
    for i in xrange(len(x)):
        if x[i] == y[i]:
            del_array.append(i)
    x = nu.delete(x, del_array)
    y = nu.delete(y, del_array)
    return x, y

class Brute_worker(Process):
    "class to do parallel brute force"
    def __init__(self, cmd, rm_file, matrix_size, gid, recive_queue, send_queue):
        # Make new file with
        #gid = os.getgid()
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
        self.rm_file = rm_file + str(gid)
        self.matrix_path = '%i.rm'%gid
        self.matrix_size = matrix_size
        self.recive_queue = recive_queue
        self.send_queue = send_queue
        super(Brute_worker, self).__init__()

    def run(self):
        ''' Does brute force'''
        for mat_list in iter(self.recive_queue.get, None):
            if mat_list is None:
                # Clean up and exit
                os.remove(self.rm_file)
                os.remove(self.matrix_path)
                break
            mat = get_mat(mat_list, self.matrix_size)
            write_matrix(mat, self.matrix_path)
            self.send_queue.put((mat, call_laplace(self.cmd, self.rm_file)))

    
def brute_force_parallel(cmd, matrix_size=5, rm_file='relbiogeog_1.lg',
                         cpu=cpu_count()):
    '''Runs brute force calculations for all Unique combinations of matrix
    on a lower triangle of a matrix minus the diagonal terms in paralle'''
    # Set up multiprocessing object
    request_queue = Queue()
    get_queue = Queue()
    worker = []
    for i in range(cpu):
        worker.append(Brute_worker(cmd, rm_file, matrix_size, i,request_queue, get_queue))
        worker[-1].start()
    # get permutations object
    x ,y = nu.triu_indices(matrix_size)
    # remove diagonal terms
    del_array = []
    for i in xrange(len(x)):
        if x[i] == y[i]:
            del_array.append(i)
    x = nu.delete(x, del_array)
    y = nu.delete(y, del_array)
    # set save arrays
    out_mat = []
    lik = []
    # do all permutations
    permute = product([0, .5, 1], repeat=len(x))
    # Run in parallel
    i, j, tot = 0, 0, 3**len(x)
    out_mat, lik = [], []
    print 'Sending data'
    for mat_list in permute:
        i += 1
        request_queue.put(mat_list)
    print 'Done. Getting results.'
    # get results
    for tmat, tlik in iter(get_queue.get, None):
        j +=1
        print '%i out of %i and -ln of %f'%(j, tot, tlik)
        out_mat.append(tmat)
        lik.append(tlik)
        if j == i:
            break
                 
    print 'Done'
    # Kill all
    for i in range(cpu):
        request_queue.put(None)

    return out_mat, lik
    
def brute_force(cmd, matrix_size=5, rm_file='relbiogeog_1.lg'):
    '''Runs brute force calculations for all Unique combinations of matrix
    on a lower triangle of a matrix minus the diagonal terms'''
    # get triangular indicies
    x ,y = nu.triu_indices(matrix_size)
    # remove diagonal terms
    del_array = []
    for i in xrange(len(x)):
        if x[i] == y[i]:
            del_array.append(i)
    x = nu.delete(x, del_array)
    y = nu.delete(y, del_array)
    # set save arrays
    out_mat = []
    lik = []
    # do all permutations
    permute = product([0, .5, 1], repeat=len(x))
    i ,tot = 0, 3**len(x)
    for mat_list in permute:
        i+=1
        mat = get_mat(mat_list, matrix_size)
        write_matrix(mat, 'mcmc_run.rm')
        out_mat.append(mat)
        lik.append(call_laplace(cmd, rm_file))
        print '%i out of %i -ln %f'%(i, tot,lik[-1])
    return out_mat, lik


        
def get_mat(mat_list, matrix_size):
    '''Turns list of lower triange and makes a matrix. Makes diagonal=1'''
    x ,y = nu.triu_indices(matrix_size)
    # remove diagonal terms
    del_array = []
    for i in xrange(len(x)):
        if x[i] == y[i]:
            del_array.append(i)
    x = nu.delete(x, del_array)
    y = nu.delete(y, del_array)
    out_mat = generate_inital_matrix(matrix_size, matrix_size)
    # put in trianglula componetns
    for i in xrange(len(x)):
        out_mat[x[i]][y[i]] = mat_list[i]
        out_mat[y[i]][x[i]] = mat_list[i]
    # Make diagonal =1
    for i in xrange(matrix_size):
        out_mat[i][i] = 1
    
    return out_mat

def parallel_mcmc(cmd, itter=10**4, rm_file='relbiogeog_1.lg',
                  matrix_size=5, cpu=cpu_count()):
    '''Does mcmc but in paralllel'''
    past_matrix = []
    lik = []
    # start workers
    request_queue = Queue()
    get_queue = Queue()
    worker = []
    for i in xrange(cpu):
        worker.append(Brute_worker(cmd, rm_file, matrix_size, i,request_queue, get_queue))
        worker[-1].start()
    # initalize lik
    while True:
        tlik = nu.nan
        for i in xrange(cpu+1):
            mat_list = generate_inital_matrix(matrix_size, matrix_size)
            request_queue.put(nu.ravel(mat_list))
        temp_lik = -nu.inf
        for i in xrange(cpu):
            try:
                tmat, tlik = get_queue.get(timeout=2)
            except:
                break
            if max(temp_lik, tlik) == tlik:
                temp_lik = tlik +0
                mat_list = tmat
        if nu.isfinite(tlik):
            break
    lik.append(temp_lik)
    past_matrix.append(mat_list)
    cur_matrix = mat_list
    # start mcmc
    for i in xrange(cpu):
        mat_list = generate_inital_matrix(matrix_size, matrix_size)
        request_queue.put(nu.ravel(mat_list))
    # Matrix item to change
    x, y = get_lower_tri(matrix_size)
    # Change all at first
    chg_index = list(nu.ravel_multi_index((x,y),(matrix_size, matrix_size)))
    # Tuning parameters
    Naccept = 1.
    Nreject = 1.
    ac_rate = .5
    for i in xrange(itter):
        print '%i out of %i lik=%.2f'%(int(i), int(itter), lik[-1])
        # create matrix
        cur_matrix, chg_index = chng_matrix_item(cur_matrix, chg_index)
        request_queue.put(nu.ravel(cur_matrix))
        try:
            cur_matrix, new_lik = get_queue.get(timeout=1)
        except:
            # load up queque
            for j in range(15):
                cur_matrix, chg_index = chng_matrix_item(cur_matrix, chg_index)
                request_queue.put(nu.ravel(cur_matrix))
            continue
        mh = exp(-(lik[-1] - new_lik)/2.)
        if mh > random.uniform(0,1):
            # acept
            past_matrix.append(cur_matrix)
            lik.append(new_lik)
            Naccept += 1.
        else:
            # reject
            past_matrix.append(past_matrix[-1])
            lik.append(lik[-1])
            cur_matrix = past_matrix[-1]
            Nreject += 1.
        # aceptance tuning
        if i % 20 == 0 and i > 0:
            ac_rate = Naccept / float(i)
            if ac_rate > .5:
                # too much acceptance shrink
                if len(chg_index) > 1:
                    chg_index.pop(0)
            if ac_rate < .15:
                if len(chg_index) < len(x):
                    chg_index.append(1)

    return past_matrix, lik



def mcmc(cmd, itter=10**4, rm_file='relbiogeog_1.lg',matrix_size=5):
    '''Does MCMC on data'''
    # intalize values
    past_matrix = []
    lik = []
    # get first matrix
    cur_matrix =  generate_inital_matrix(matrix_size, matrix_size)
    # Create file
    write_matrix(cur_matrix, 'mcmc_run.rm')
    # get fitst lik
    lik.append(call_laplace(cmd, rm_file))
    # try till finite value found
    while not nu.isfinite(lik[-1]):
        cur_matrix =  generate_inital_matrix(matrix_size, matrix_size)
        write_matrix(cur_matrix, 'mcmc_run.rm')
        lik[-1] = call_laplace(cmd, rm_file)
    # save
    past_matrix.append(cur_matrix)
    # Matrix item to change
    x, y = get_lower_tri(matrix_size)
    # Change all at first
    chg_index = list(nu.ravel_multi_index((x,y),(matrix_size, matrix_size)))
    # Tuning parameters
    Naccept = 1.
    Nreject = 1.
    ac_rate = .5
    # Run Chain
    for i in xrange(itter):
    	print '%i out of %i acep_rate=%f, lik=%.2f'%(int(i), int(itter),ac_rate, lik[-1])
        # create matrix
        cur_matrix, chg_index = chng_matrix_item(cur_matrix, chg_index)
        # save
        write_matrix(cur_matrix, 'mcmc_run.rm')
        #get lik
        new_lik =  call_laplace(cmd,  rm_file)
        # M-H
        mh = exp(-(lik[-1] - new_lik)/2.)
        if mh > random.uniform(0,1):
            # acept
            past_matrix.append(cur_matrix)
            lik.append(new_lik)
            Naccept += 1.
        else:
            # reject
            past_matrix.append(past_matrix[-1])
            lik.append(lik[-1])
            cur_matrix = past_matrix[-1]
            Nreject += 1.
        # aceptance tuning
        if i % 20 == 0 and i > 0:
            ac_rate = Naccept / float(i)
            if ac_rate > .5:
                # too much acceptance shrink
                if len(chg_index) > 1:
                    chg_index.pop(0)
            if ac_rate < .15:
                if len(chg_index) < len(x):
                    chg_index.append(1)
    return past_matrix, lik

def make_row(mat):
    out = []
    for i in mat:
        for j in i:
            out.append(j)
    return out

def save_to_csv(lik, matrix, out_file):
    '''Saves data to a csv file with lik, matrix_11,matrix_12...'''
    with open(out_file, 'w') as save_file:
        for i in xrange(len(lik)):
            row = make_row(matrix[i])
            save_file.write(''.join(str([lik[i]]+row))[1:-1]+'\n')
    

if __name__ == '__main__':
    import sys
    # Run on thuso
    #cmd = 'lagrange/src/lagrange_cpp'
    # Run on Joanne
    #os.chdir('/Users/joannebentley/Desktop')
    cmd = sys.argv[1] #'./lagrange_cpp'
    out_files = sys.argv[2]
    #MCMC
    #matrix, lik = mcmc(cmd, matrix_size=7)
    # Parallel MCMC
    matrix, lik = parallel_mcmc(cmd, matrix_size=7)
    #Brute force
    #matrix, lik = brute_force(cmd)
    #brute force parallel
    #matrix, lik = brute_force_parallel(cmd)
    save_to_csv(lik, matrix, out_files)

    

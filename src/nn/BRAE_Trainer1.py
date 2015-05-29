#-*- coding: utf-8 -*-
'''
Reordering classifier and related training code

@author: lpeng
'''
from __future__ import division
from sys import stderr
import argparse
import logging
import cPickle as pickle

from numpy import concatenate, zeros_like, zeros, tanh, dot
from numpy.random import get_state, set_state, seed
from mpi4py import MPI

from ioutil import Writer
from timeutil import Timer
import lbfgs
from ioutil import Reader
from nn.BRAE_rae1 import RecursiveAutoencoder
from nn.util import init_W,init_We
from nn.instance import Instance
from errors import GridentCheckingFailedError
from vec.wordvector import WordVectors
from nn.signals import TerminatorSignal, WorkingSignal, ForceQuitSignal

comm = MPI.COMM_WORLD
worker_num = comm.Get_size()
rank = comm.Get_rank()

def send_terminate_signal():
    param = TerminatorSignal()
    comm.bcast(param, root=0)


def send_working_signal():
    param = WorkingSignal()
    comm.bcast(param, root=0)

  
def send_force_quit_signal():
    param = ForceQuitSignal()
    comm.bcast(param, root=0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_cost_and_grad( theta, instances, total_internal_node_num, word_vectors, embsize,
                      lambda_reg, lambda_reg_L ):
    '''Compute the value and gradients of the objective function at theta
  
    Args:
        theta: model parameter
        instances: training instances
        total_internal_node_num: total number of internal nodes 
        embsize: word embedding vector size
        lambda_reg: the weight of regularizer
    
    Returns:
        total_cost: the value of the objective function at theta
        total_grad: the gradients of the objective function at theta
    '''
    if rank == 0:
        # send working signal
        send_working_signal()

        # send theta
        comm.Bcast([theta, MPI.DOUBLE], root=0)
        
        # init recursive autoencoder
        # 新建一个autoencoder，并且初始化,将参数恢复成矩阵形式
        rae = RecursiveAutoencoder.build( theta, embsize )

        # compute local reconstruction error and gradients
        # 计算训练短语的error和gradient
        word_vectors._vectors = rae.L
        rec_error, gradient_vec = process( rae, word_vectors, instances ) 

        # compute total reconstruction error
        total_rec_error = comm.reduce(rec_error, op=MPI.SUM, root=0)
        
        # compute total cost
        #计算总误差,算上regularizer
        reg = rae.get_weights_square()
        L_reg = (rae.L**2).sum() 
        total_cost = total_rec_error / total_internal_node_num +\
                    lambda_reg / 2 * reg + lambda_reg_L / 2 * L_reg

        total_grad = zeros_like(gradient_vec)
        comm.Reduce([gradient_vec, MPI.DOUBLE], [total_grad, MPI.DOUBLE],
                op=MPI.SUM, root=0)
        total_grad /= total_internal_node_num
     
        # gradients related to regularizer
        reg_grad = rae.get_zero_gradients()
        reg_grad.gradWi1 += rae.Wi1
        reg_grad.gradWi2 += rae.Wi2
        reg_grad.gradWo1 += rae.Wo1
        reg_grad.gradWo2 += rae.Wo2
        reg_grad *= lambda_reg
        reg_grad.gradL += rae.L
        reg_grad.gradL *= lambda_reg_L
     
        total_grad += reg_grad.to_row_vector() 

        return total_cost, total_grad 
    else:
        while True:
            # receive signal
            signal = comm.bcast(root=0)
            if isinstance(signal, TerminatorSignal):
                return
            if isinstance(signal, ForceQuitSignal):
                exit(-1)
      
            # receive theta
            comm.Bcast([theta, MPI.DOUBLE], root=0)
    
            # init recursive autoencoder
            rae = RecursiveAutoencoder.build(theta, embsize)

            # compute local reconstruction error and gradients
            word_vectors._vectors = rae.L
            rec_error, gradient_vec = process(rae, word_vectors, instances)

            # send local reconstruction error to root
            comm.reduce(rec_error, op=MPI.SUM, root=0)
      
            # send local gradients to root
            comm.Reduce([gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)
 
def process( rae, word_vectors, instances ):
    total_rec_error = 0
    # 初始化梯度参数
    gradients = rae.get_zero_gradients()
    for instance in instances:
        words_embedded = word_vectors[instance.words]
        #print instance.words, words_embedded.shape,word_vectors.shape
        # 前向传播，计算错误率
        root_node, rec_error = rae.forward( words_embedded, instance )
        # 反向传播计算梯度
        rae.backward( root_node, gradients, freq= instance.freq )
        total_rec_error += rec_error * instance.freq
 
    grad_row_vec = gradients.to_row_vector() 
                
    return total_rec_error, grad_row_vec

def init_theta( embsize, word_vectors, _seed = None ):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)
    
    parameters = []
    
    # Wi1 n*n
    parameters.append(init_W(embsize, embsize))
    # Wi2 n*n
    parameters.append(init_W(embsize, embsize))
    # bi n*1
    parameters.append(zeros(embsize))
  
    # Wo1 n*n
    parameters.append(init_W(embsize, embsize))
    # Wo2 n*n
    parameters.append(init_W(embsize, embsize))
    # bo1 n*1
    parameters.append(zeros(embsize))
    # bo2 n*1
    parameters.append(zeros(embsize))

    # L
    parameters.append( word_vectors._vectors.reshape( embsize * len( word_vectors ) ) )

    if _seed != None:  
        set_state(ori_state)
  
    return concatenate(parameters)   

def prepare_data(word_vectors=None, datafile=None):
    '''Prepare training data
    Args:
        word_vectors: an instance of vec.wordvector
        datafile: location of data file
    
    Return:
        instances: a list of Instance
        word_vectors: word_vectors
        total_internal_node: total number of internal nodes
    '''
    if rank == 0:
       # broadcast word vectors
        comm.bcast(word_vectors, root=0)

        # load raw data
        with Reader(datafile) as datafile:
            instance_strs = [line for line in datafile]

        # send training data
        instance_num = len(instance_strs)
        esize = int(instance_num/worker_num+0.5)
        sizes = [esize] * worker_num
        sizes[-1] = instance_num - esize*(worker_num-1)
        offset = sizes[0]
        for i in range(1, worker_num):
            comm.send(instance_strs[offset:offset+sizes[i]], dest=i)
            offset += sizes[i]
        comm.barrier() 

        local_instance_strs = instance_strs[0:sizes[0]]
        del instance_strs

        instances, internal_node_num = load_instances(local_instance_strs,
                                                  word_vectors)
 
        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node
    else:
        word_vectors = comm.bcast(root=0)
    
        # receive data
        local_instance_strs = comm.recv(source=0)
        comm.barrier()
    
        instances, internal_node_num = load_instances(local_instance_strs,
                                                  word_vectors)

        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node


def load_instances(instance_strs, word_vectors):
    '''Load training examples
  
    Args:
        instance_strs: each string is a training example
        word_vectors: an instance of vec.wordvector
    
    Return:
        instances: a list of Instance
    '''

    instances = [Instance.parse_from_str(i, word_vectors) for i in instance_strs]
    total_internal_node = 0
    for instance in instances:
        # 对于一个短语有n个单词，则经过n-1次组合后形成唯一的短语向量，故中间节点共有n-1个
        total_internal_node += (len(instance.words)-1) * instance.freq
    return instances, total_internal_node

class ThetaSaver(object):
  
    def __init__(self, model_name, every=1):
        self.idx = 1
        self.model_name = model_name
        # 每every步存储一次
        self.every = every
    
    def __call__(self, xk):
        if self.every == 0:
            return;
    
        if self.idx % self.every == 0:
            model = self.model_name
            pos = model.rfind('.')
            if pos < 0:
                filename = '%s.iter%d' % (model, self.idx)
            else:
                filename = '%s.iter%d%s' % (model[0:pos], self.idx, model[pos:])
      
        with Writer(filename) as writer:
            [writer.write('%20.8f\n' % v) for v in xk]
        self.idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Instance File
    parser.add_argument( '-instances', required = True,
                      help = 'instance file' )
    parser.add_argument( '-tp',required = True,
                    help = 'whethere it is src or trg')

    parser.add_argument('-model', required=True,
                      help='model name')
    parser.add_argument('-word_vector', required=True,
                      help='word vector file',)
    parser.add_argument('-lambda_reg', type=float, default=0.15,
                      help='weight of the regularizer')
    parser.add_argument('-lambda_reg_L', type=float, default=1e-2,
                        help='weight of the word embedding matrix')
    parser.add_argument('--save-theta0', action='store_true',
                      help='save theta0 or not, for dubegging purpose')
    parser.add_argument('--checking-grad', action='store_true', 
                      help='checking gradients or not, for dubegging purpose')
    parser.add_argument('-m', '--maxiter', type=int, default=100,
                      help='max iteration number',)
    parser.add_argument('-e', '--every', type=int, default=0,
                      help='dump parameters every --every iterations',)
    parser.add_argument('--seed', default=None,
                      help='random number seed for initialize random',)
    parser.add_argument('-v', '--verbose', type=int, default=0,
                      help='verbose level')
    
    options = parser.parse_args()
    instances_file = options.instances
    tp = options.tp
    model = options.model
    word_vector_file = options.word_vector
    lambda_reg = options.lambda_reg
    lambda_reg_L =options.lambda_reg_L
    save_theta0 = options.save_theta0
    checking_grad = options.checking_grad
    maxiter = options.maxiter
    every = options.every
    _seed = options.seed
    verbose = options.verbose
   
    if rank == 0:
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        if checking_grad:
            logger.setLevel(logging.WARN)
        else:
            logger.setLevel(logging.INFO)
        
        print >> stderr, 'Instances file: %s' % instances_file
        print >> stderr, 'Model file: %s' % model
        print >> stderr, 'Type: %s' % tp
        print >> stderr, 'Word vector file: %s' % word_vector_file 
        print >> stderr, 'lambda_reg: %20.18f' % lambda_reg
        print >> stderr, 'lambda_reg_L: %20.18f' % lambda_reg_L
        print >> stderr, 'Max iterations: %d' % maxiter
        if _seed:
            print >> stderr, 'Random seed: %s' % _seed
        print >> stderr, ''

        print >> stderr, 'load word vectors...'
        # 载入词向量的输入放入word_vectors中
        word_vectors = WordVectors.load_vectors( word_vector_file )
        #embsize为词向量的维度
        embsize = word_vectors.embsize()
       
        print >> stderr, 'preparing data...' 
        #载入训练短语数据，将短语转化为instance的数组放入instances中
        instances, _, total_internal_node = prepare_data( word_vectors, instances_file )

        print >> stderr, 'init. RAE parameters...'
        timer = Timer()
        timer.tic()
        if _seed != None:
            _seed = int(_seed)
        else:
            _seed = None
        print >> stderr, 'seed: %s' % str(_seed)

        # 初始化参数
        theta0 = init_theta( embsize, word_vectors, _seed = _seed )
        theta0_init_time = timer.toc()
        print >> stderr, 'shape of ' + tp + 'theta0 %s' % theta0.shape
        timer.tic()
        if save_theta0:
            print >> stderr, 'saving ' + tp + 'theta0...'
            pos = model.rfind('.')
            if pos < 0:
                filename = model + '_' + tp + '.theta0'
            else:
                filename = model[0:pos] + '_' + tp + '.theta0' + model[pos:]
            with Writer(filename) as theta0_writer:
                pickle.dump(theta0, theta0_writer)    
        theta0_saving_time = timer.toc() 
    
        # 每隔every步就存储一次模型参数
        callback = ThetaSaver( model + tp, every )    
        func = compute_cost_and_grad
        args = ( instances, total_internal_node, word_vectors, embsize, lambda_reg, lambda_reg_L )
        theta_opt = None
        print >> stderr, 'optimizing...'    
    
        try:
            # 开始优化
            theta_opt = lbfgs.optimize( func, theta0, maxiter, verbose, checking_grad, 
                    args, callback = callback )

        except GridentCheckingFailedError:
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        opt_time = timer.toc()
        send_terminate_signal()

        timer.tic()
        # pickle form
        print >> stderr, 'saving parameters to %s' % model + '_' + tp
        with Writer( model + '_' + tp ) as model_pickler:
            pickle.dump( theta_opt, model_pickler)
        # pure text form
        with Writer(model+'_' + tp +'.txt') as writer:
            [writer.write('%20.8f\n' % v) for v in theta_opt]
        thetaopt_saving_time = timer.toc()

        print >> stderr, 'Init. theta0  : %10.2f s' % theta0_init_time
        if save_theta0:
            print >> stderr, 'Saving theta0 : %10.2f s' % theta0_saving_time
        print >> stderr, 'Optimizing    : %10.2f s' % opt_time
        print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_saving_time
        print >> stderr, 'Done!'  
    else:
        instances, word_vectors, total_internal_node = prepare_data()
        embsize = word_vectors.embsize()
        param_size = embsize*embsize*4 + embsize*3 + embsize * len( word_vectors )
        theta = zeros((param_size, 1))    
        compute_cost_and_grad(theta, instances, total_internal_node, word_vectors,
                          embsize, lambda_reg, lambda_reg_L )

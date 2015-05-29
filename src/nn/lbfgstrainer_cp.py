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
import random as rd

from numpy import concatenate, zeros_like, zeros, tanh, dot
from numpy import linalg as LA
from numpy.random import get_state, set_state, seed
from mpi4py import MPI


from ioutil import Writer
from timeutil import Timer
import lbfgs
from ioutil import Reader
from nn.rae_cp import RecursiveAutoencoder, RecursiveAutoencoder_la
from nn.util import init_W,init_We
from nn.instance import Instance
from errors import GridentCheckingFailedError
from vec.wordvector import WordVectors
from functions import tanh_norm1_prime, sum_along_column
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

def compute_cost_and_grad( theta, instances, total_internal_node_num, word_vectors, embsize, lambda_reg, lambda_reg_L ):
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

        word_vectors._vectors = rae.L
        # compute local reconstruction error and gradients
        # 计算训练短语的error和gradient
        rec_error, gradient_vec = process( rae, word_vectors, instances ) 

        # compute total reconstruction error
        total_rec_error = comm.reduce(rec_error, op=MPI.SUM, root=0)
        
        # compute total cost
        reg = rae.get_weights_square()
        #计算总误差,算上regularizer
        L_reg = (rae.L**2).sum() 
        total_cost = total_rec_error / total_internal_node_num +\
                    lambda_reg/2 * reg + lambda_reg_L/2 * L_reg

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
        reg_grad.gradL += rae.L
        reg_grad *= lambda_reg
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

            word_vectors._vectors = rae.L
            # compute local reconstruction error and gradients
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


def compute_cost_and_grad_la( theta, src_instances, src_total_internal_node, src_word_vectors, src_embsize,
                    trg_instances, trg_total_internal_node, trg_word_vectors, trg_embsize, 
                    lambda_reg_L, lambda_reg_rec,lambda_reg_sem, alpha, bad_src_instances, bad_trg_instances ):
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
    src_offset_la = 5 * src_embsize * src_embsize + 4 * src_embsize + src_embsize*src_word_vectors._vectors.shape[1]

    if rank == 0:
        # send working signal
        send_working_signal()

        # send theta
        comm.Bcast([theta, MPI.DOUBLE], root=0)
 
        src_theta = theta[0:src_offset_la] 
        trg_theta = theta[src_offset_la:]
    
        src_rae_la = RecursiveAutoencoder_la.build_la( src_theta, src_embsize ) 
        trg_rae_la = RecursiveAutoencoder_la.build_la( trg_theta, trg_embsize ) 
            
        src_word_vectors._vectors = src_rae_la.L
        trg_word_vectors._vectors = trg_rae_la.L

        src_rec_error, src_sem_error, src_gradient_vec,\
        trg_rec_error, trg_sem_error, trg_gradient_vec = \
                                process_la( src_rae_la, trg_rae_la, alpha, 
                                src_word_vectors, src_instances, src_total_internal_node,
                                trg_word_vectors, trg_instances, trg_total_internal_node,
                                bad_src_instances, bad_trg_instances )

        # compute total reconstruction error
        src_total_rec_error = comm.reduce(src_rec_error, op=MPI.SUM, root=0)
        trg_total_rec_error = comm.reduce(trg_rec_error, op=MPI.SUM, root=0)
        src_total_sem_error = comm.reduce(src_sem_error, op=MPI.SUM, root=0)
        trg_total_sem_error = comm.reduce(trg_sem_error, op=MPI.SUM, root=0)

        # compute total cost
        src_reg_rec = src_rae_la.get_weights_square()
        trg_reg_rec = trg_rae_la.get_weights_square() 
        src_reg_sem = ( src_rae_la.Wla**2 ).sum()
        trg_reg_sem = ( trg_rae_la.Wla**2 ).sum()

        #计算总误差,算上regularizer
        src_total_cost = alpha * ( src_total_rec_error + lambda_reg_rec/2 * src_reg_rec ) +\
        ( 1 - alpha ) *( src_total_sem_error + lambda_reg_sem/2 * src_reg_sem )  
        trg_total_cost = alpha * ( trg_total_rec_error + lambda_reg_rec/2 * trg_reg_rec ) +\
        ( 1 - alpha ) *( trg_total_sem_error + lambda_reg_sem/2 * trg_reg_sem )  

        src_total_grad = zeros_like(src_gradient_vec)
        trg_total_grad = zeros_like(trg_gradient_vec)
        comm.Reduce([src_gradient_vec, MPI.DOUBLE], [src_total_grad, MPI.DOUBLE],
                op=MPI.SUM, root=0)
        comm.Reduce([trg_gradient_vec, MPI.DOUBLE], [trg_total_grad, MPI.DOUBLE],
                op=MPI.SUM, root=0)

        # compute gradients 
        # gradients related to regularizer
        src_reg_grad = src_rae_la.get_zero_gradients_la()
        src_reg_grad.gradWi1 += lambda_reg_rec * src_rae_la.Wi1 * alpha
        src_reg_grad.gradWi2 += lambda_reg_rec * src_rae_la.Wi2 * alpha
        src_reg_grad.gradWo1 += lambda_reg_rec * src_rae_la.Wo1 * alpha
        src_reg_grad.gradWo2 += lambda_reg_rec * src_rae_la.Wo2 * alpha
        src_reg_grad.gradWla += lambda_reg_sem * src_rae_la.Wla * ( 1- alpha )

        trg_reg_grad = trg_rae_la.get_zero_gradients_la()
        trg_reg_grad.gradWi1 += lambda_reg_rec * trg_rae_la.Wi1 * alpha
        trg_reg_grad.gradWi2 += lambda_reg_rec * trg_rae_la.Wi2 * alpha
        trg_reg_grad.gradWo1 += lambda_reg_rec * trg_rae_la.Wo1 * alpha
        trg_reg_grad.gradWo2 += lambda_reg_rec * trg_rae_la.Wo2 * alpha
        trg_reg_grad.gradWla += lambda_reg_sem * trg_rae_la.Wla * ( 1- alpha ) 
  
        src_total_grad += src_reg_grad.to_row_vector_la()
        trg_total_grad += trg_reg_grad.to_row_vector_la()

        return ( src_total_cost + trg_total_cost ), concatenate( [src_total_grad, trg_total_grad] )
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
    
            src_theta = theta[0:src_offset_la] 
            trg_theta = theta[src_offset_la:]
    
            src_rae_la = RecursiveAutoencoder_la.build_la( src_theta, src_embsize ) 
            trg_rae_la = RecursiveAutoencoder_la.build_la( trg_theta, trg_embsize ) 
            
            src_word_vectors._vectors = src_rae_la.L
            trg_word_vectors._vectors = trg_rae_la.L

            src_rec_error, src_sem_error, src_gradient_vec,\
            trg_rec_error, trg_sem_error, trg_gradient_vec = \
                                process_la( src_rae_la, trg_rae_la, alpha, 
                                    src_word_vectors, src_instances, src_total_internal_node,
                                    trg_word_vectors, trg_instances, trg_total_internal_node,
                                    bad_src_instances, bad_trg_instances )

            # send local errors to root
            comm.reduce(src_rec_error, op=MPI.SUM, root=0)
            comm.reduce(trg_rec_error, op=MPI.SUM, root=0)
            comm.reduce(src_sem_error, op=MPI.SUM, root=0)
            comm.reduce(trg_sem_error, op=MPI.SUM, root=0)

            # send local gradients to root
            comm.Reduce([src_gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Reduce([trg_gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)
   

def process_la( src_rae_la, trg_rae_la, alpha, 
             src_word_vectors, src_instances, src_total_internal_node,
             trg_word_vectors, trg_instances, trg_total_internal_node,
             bad_src_instances, bad_trg_instances ):

    total_rec_error = 0
    total_sem_error = 0
    # 初始化梯度参数
    src_gradients_la = src_rae_la.get_zero_gradients_la()
    trg_gradients_la = trg_rae_la.get_zero_gradients_la()    
    src_total_rec_error = 0
    trg_total_rec_error = 0
    src_total_sem_error = 0
    trg_total_sem_error = 0
    for i in xrange( len( src_instances ) ): 
        src_instance = src_instances[i]
        trg_instance = trg_instances[i]
        bad_src_instance = bad_src_instances[i]
        bad_trg_instance = bad_trg_instances[i]
        # 取出该短语中所有词向量,instance.words中的单词idx还原成words.embedded中的词向量矩阵n*word_num
        src_words_embedded = src_word_vectors[src_instance.words]
        trg_words_embedded = trg_word_vectors[trg_instance.words]
        bad_src_embedded = src_word_vectors[bad_src_instance] 
        bad_trg_embedded = trg_word_vectors[bad_trg_instance]
        # 前向传播，计算错误
        src_root_node, src_rec_error = src_rae_la.forward_la( src_words_embedded, src_instance )
        trg_root_node, trg_rec_error = trg_rae_la.forward_la( trg_words_embedded, trg_instance )
        src_total_rec_error += src_rec_error * src_instance.freq
        trg_total_rec_error += trg_rec_error * trg_instance.freq
        #bad_src_root, _  = src_rae_la.forward_la( bad_src_embedded, instance )
        #bad_trg_root, _ = trg_rae_la.forward_la( bad_trg_embedded, instance )

        rec_s = alpha * src_instance.freq / src_total_internal_node
        rec_t = alpha * trg_instance.freq / trg_total_internal_node
        sem_s = ( 1 - alpha ) * src_instance.freq / src_total_internal_node
        sem_t = ( 1 - alpha ) * trg_instance.freq / trg_total_internal_node

        # Semantic Error
        # Source side
        src_yla_unnormalized = tanh( dot( src_rae_la.Wla, src_root_node.p ) + src_rae_la.bla )
        src_yla = src_yla_unnormalized / LA.norm( src_yla_unnormalized, axis=0 )
        src_ylapla = src_yla - trg_root_node.p
        src_sem_error = 0.5 * sum_along_column( src_ylapla**2 )[0]

        bad_src_ylapla = src_yla - bad_trg_root.p
        bad_src_sem_error = 0.5 * sum_along_column( bad_src_ylapla**2 )[0] 
        src_sem_margin = src_sem_error#(src_sem_error-bad_src_sem_error+1)*src_instance.freq
    
        src_sem_margin = max( 0.0, src_sem_margin )
        if src_sem_margin == 0.0:
            soptimal = True
        else:
            soptimal = False
    
        src_total_sem_error += src_sem_margin

        # Target side
        trg_yla_unnormalized = tanh( dot( trg_rae_la.Wla, trg_root_node.p ) + trg_rae_la.bla )
        trg_yla = trg_yla_unnormalized / LA.norm( trg_yla_unnormalized, axis=0 )
        trg_ylapla = trg_yla - src_root_node.p
        trg_sem_error = 0.5 * sum_along_column( trg_ylapla**2 )[0]

        bad_trg_ylapla = trg_yla - bad_src_root.p
        bad_trg_sem_error = 0.5 * sum_along_column( bad_trg_ylapla**2 )[0]
        trg_sem_margin = trg_sem_error#(trg_sem_error-bad_trg_sem_error+1)*trg_instance.freq
    
        trg_sem_margin = max( 0.0, trg_sem_margin )
        if trg_sem_margin == 0.0:
            toptimal = True
        else:
            toptimal = False
    
        trg_total_sem_error += trg_sem_margin 

        # 反向传播计算梯度
        src_rae_la.backward_la( src_root_node, bad_src_root, src_gradients_la, rec_s, sem_s, 
                src_yla_unnormalized, src_ylapla, 0, soptimal )
        #trg_rae_la.backward_la( trg_root_node, bad_trg_root, trg_gradients_la, rec_t, sem_t, 
                #trg_yla_unnormalized, trg_ylapla, bad_trg_ylapla, toptimal )

        src_total_rec_error = src_total_rec_error / src_total_internal_node
        trg_total_rec_error = trg_total_rec_error / trg_total_internal_node
 
        src_total_sem_error = src_total_sem_error / src_total_internal_node 
        trg_total_sem_error = trg_total_sem_error / trg_total_internal_node 

    return src_total_rec_error, src_total_sem_error, src_gradients_la.to_row_vector_la(),\
        trg_total_rec_error, trg_total_sem_error, trg_gradients_la.to_row_vector_la()

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


def init_theta_la( theta, src_embsize, trg_embsize, src_word_vectors, trg_word_vectors, _seed=None ):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)
    
    src_offset = 4 * src_embsize * src_embsize + 3 * src_embsize + src_embsize * src_word_vectors._vectors.shape[1]
    src_theta = theta[0:src_offset] 
    trg_theta = theta[src_offset:]

    parameters = []

    # Source side 
    parameters.append( src_theta )
    # Wla n*n
    parameters.append( init_W( src_embsize, src_embsize ) )
    # bla n*1
    parameters.append( zeros( src_embsize ) )

    # Target side 
    parameters.append( trg_theta )
    # Wla n*n
    parameters.append( init_W( trg_embsize, trg_embsize ) )
    # bla n*1
    parameters.append( zeros( trg_embsize ) )
    
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


def prepare_data_la(word_vectors=None, datafile=None):
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
        bad_instances = []
        for i in xrange( len( instances ) ): 
            instance = instances[i]
            bad_instances.append([rd.randrange( 0, len( word_vectors ) ) \
                    for j in xrange( len( instance.words ) )] )
        
        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node, bad_instances
    else:
        word_vectors = comm.bcast(root=0)
    
        # receive data
        local_instance_strs = comm.recv(source=0)
        comm.barrier()
    
        instances, internal_node_num = load_instances(local_instance_strs,
                                                  word_vectors)
        bad_instances = []
        for i in xrange( len( instances ) ): 
            instance = instances[i]
            bad_instances.append([rd.randrange( 0, len( word_vectors ) ) \
                    for j in xrange( len( instance.words ) )] )

        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node, bad_instances

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
    parser.add_argument( '-source_instances', required = True,
                      help = 'source language instance file' )
    parser.add_argument( '-target_instances', required = True,
                      help = 'target language instance file' )

    parser.add_argument( '-model', required = True,
                      help = 'model name' )
    parser.add_argument( '-model_la', required = True,
                      help = 'model_la name' )
    # Word Vector 
    parser.add_argument( '-source_word_vector', required=True,
                      help=' source language word vector file' )
    parser.add_argument( '-target_word_vector', required=True,
                      help='target language word vector file' )
    
    parser.add_argument( '-lambda_reg', type=float, default=0.15,
                      help='language weight of the regularizer' )
    parser.add_argument( '-lambda_reg_rec', type=float, default=1e-3,
                      help='weight of the regularizer for reconstruction error' )
    parser.add_argument( '-lambda_reg_L', type=float, default=1e-2,
                      help='weight of the regularizer for embedding matrix' )
    parser.add_argument( '-lambda_reg_sem', type=float, default=1e-3,
                      help='weight of the regularizer for sementic error' )
    parser.add_argument( '-alpha', type = float, default = 0.15,
                      help = 'hyperparameter for total error')

    parser.add_argument('--save-theta0', action='store_true',
                      help='save theta0 or not, for dubegging purpose')
    parser.add_argument('--checking-grad', action='store_true', 
                      help='checking gradients or not, for dubegging purpose')
    parser.add_argument('-m', '--maxiter', type=int, default=100,
                      help='max iteration number')
    parser.add_argument('-ml', '--maxiter_la', type=int, default=20,
                      help='max iteration number for supervised learning')
    parser.add_argument('-e', '--every', type=int, default=0,
                      help='dump parameters every --every iterations',)
    parser.add_argument('--seed', default=None,
                      help='random number seed for initialize random',)
    parser.add_argument('-v', '--verbose', type=int, default=0,
                      help='verbose level')
    options = parser.parse_args()
  
    src_instances_file = options.source_instances
    trg_instances_file = options.target_instances
    model = options.model
    model_la = options.model_la
    src_word_vector_file = options.source_word_vector
    trg_word_vector_file = options.target_word_vector
    lambda_reg = options.lambda_reg
    lambda_reg_L = options.lambda_reg_L
    lambda_reg_rec = options.lambda_reg_rec
    lambda_reg_sem = options.lambda_reg_sem
    alpha = options.alpha

    save_theta0 = options.save_theta0
    checking_grad = options.checking_grad
    maxiter = options.maxiter
    maxiter_la = options.maxiter_la
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
        
        print >> stderr, 'Source Instances file: %s' % src_instances_file
        print >> stderr, 'Target Instances file: %s' % trg_instances_file
        print >> stderr, 'Model file: %s' % model
        print >> stderr, 'Model with label file: %s' % model_la
        print >> stderr, 'Source Word vector file: %s' % src_word_vector_file 
        print >> stderr, 'Target Word vector file: %s' % trg_word_vector_file 
        print >> stderr, 'lambda_reg: %20.18f' % lambda_reg
        print >> stderr, 'lambda_reg_L: %20.18f' % lambda_reg_L
        print >> stderr, 'lambda_reg_rec: %20.18f' % lambda_reg_rec
        print >> stderr, 'lambda_reg_sem: %20.18f' % lambda_reg_sem
        print >> stderr, 'alpha: %20.18f' % alpha
        print >> stderr, 'Max iterations: %d' % maxiter
        print >> stderr, 'Max iterations_la: %d' % maxiter_la
        if _seed:
            print >> stderr, 'Random seed: %s' % _seed
        print >> stderr, ''

        print >> stderr, 'load word vectors...'
        # 载入词向量的输入放入word_vectors中
        src_word_vectors = WordVectors.load_vectors( src_word_vector_file )
        trg_word_vectors = WordVectors.load_vectors( trg_word_vector_file )
        #embsize为词向量的维度
        src_embsize = src_word_vectors.embsize()
        trg_embsize = trg_word_vectors.embsize()
       
        print >> stderr, 'preparing data...' 
        #载入训练短语数据，将短语转化为instance的数组放入instances中
        src_instances, _, src_total_internal_node = prepare_data( src_word_vectors, src_instances_file )
        trg_instances, _, trg_total_internal_node = prepare_data( trg_word_vectors, trg_instances_file )
        print >> stderr, 'init. RAE parameters...'
        timer = Timer()
        timer.tic()
        if _seed != None:
            _seed = int(_seed)
        else:
            _seed = None
        print >> stderr, 'seed: %s' % str(_seed)
        # 初始化参数
        src_theta0 = init_theta( src_embsize, src_word_vectors, _seed = _seed )
        trg_theta0 = init_theta( trg_embsize, trg_word_vectors, _seed = _seed )
        theta0_init_time = timer.toc()
        print >> stderr, 'shape of source theta0 %s' % src_theta0.shape
        print >> stderr, 'shape of target theta0 %s' % trg_theta0.shape

        timer.tic()
        if save_theta0:
            print >> stderr, 'saving source theta0...'
            pos = model.rfind('.')
            if pos < 0:
                filename = model +'_source.theta0'
            else:
                filename = model[0:pos] + '_source.theta0' + model[pos:]
            with Writer(filename) as theta0_writer:
                pickle.dump(src_theta0, theta0_writer)    

            print >> stderr, 'saving target theta0...'
            pos = model.rfind('.')
            if pos < 0:
                filename = model + '_target.theta0'
            else:
                filename = model[0:pos] + '_target.theta0' + model[pos:]
            with Writer(filename) as theta0_writer:
                pickle.dump(trg_theta0, theta0_writer)    
        theta0_saving_time = timer.toc() 
    
        # 每隔every步就存储一次模型参数
        src_callback = ThetaSaver( model + '_source', every )    
        trg_callback = ThetaSaver( model + '_target', every )    
        func = compute_cost_and_grad
        src_args = ( src_instances, src_total_internal_node, src_word_vectors, src_embsize, lambda_reg, lambda_reg_L )
        trg_args = ( trg_instances, trg_total_internal_node, trg_word_vectors, trg_embsize, lambda_reg, lambda_reg_L )
        src_theta_opt = None
        trg_theta_opt = None
        print >> stderr, 'optimizing...'    
    
        try:
            # 开始优化
            src_theta_opt = lbfgs.optimize( func, src_theta0, maxiter, verbose, checking_grad, 
                    src_args, callback = src_callback )

        except GridentCheckingFailedError:
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        try:
            # 开始优化
            trg_theta_opt = lbfgs.optimize( func, trg_theta0, maxiter, verbose, checking_grad, 
                    trg_args, callback = trg_callback )

        except GridentCheckingFailedError:
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        send_terminate_signal() 
        opt_time = timer.toc()

        timer.tic()
        # pickle form
        src_offset = 4*src_embsize*src_embsize + 3 * src_embsize
        src_theta_opt_L = src_theta_opt[src_offset:] 
        src_theta_opt = src_theta_opt[0:src_offset]
        trg_offset = 4*trg_embsize*trg_embsize + 3 * trg_embsize
        trg_theta_opt_L = trg_theta_opt[trg_offset:]
        trg_theta_opt = trg_theta_opt[0:trg_offset]

        print >> stderr, 'saving parameters to %s' % model+'_source'
        with Writer( model + '_source' ) as model_pickler:
            pickle.dump( src_theta_opt, model_pickler)
        # pure text form
        with Writer(model+'_source.txt') as writer:
            [writer.write('%20.8f\n' % v) for v in src_theta_opt]

        with Writer( model + '_target' ) as model_pickler:
            pickle.dump( trg_theta_opt, model_pickler)
        # pure text form
        with Writer(model+'_target.txt') as writer:
            [writer.write('%20.8f\n' % v) for v in trg_theta_opt]
        thetaopt_saving_time = timer.toc()

        print >> stderr, 'Init. theta0  : %10.2f s' % theta0_init_time
        if save_theta0:
            print >> stderr, 'Saving theta0 : %10.2f s' % theta0_saving_time
        print >> stderr, 'Optimizing    : %10.2f s' % opt_time
        print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_saving_time
        print >> stderr, 'Done!'  
#------------------------------------------------------------

        src_theta_opt = [src_theta_opt, src_theta_opt_L] 
        trg_theta_opt = [trg_theta_opt, trg_theta_opt_L] 
        # For Test
        theta_opt = []
        theta_opt.extend( src_theta_opt )
        theta_opt.extend( trg_theta_opt )

        src_instances, _, src_total_internal_node, bad_src_instances = prepare_data_la( src_word_vectors, src_instances_file )
        trg_instances, _, trg_total_internal_node, bad_src_instances = prepare_data_la( trg_word_vectors, trg_instances_file )

        print >> stderr, 'Unsupervised Optimizing Done!'
        timer.tic()
        if _seed != None:
            _seed = int(_seed)
        else:
            _seed = None
        print >> stderr, 'seed: %s' % str(_seed)
        # 初始化监督学习的参数
        theta0_la = init_theta_la( theta_opt, src_embsize, trg_embsize, src_word_vectors, trg_word_vectors, _seed=_seed )

        theta0_la_init_time = timer.toc()
        print >> stderr, 'shape of theta0_la %s' % theta0_la.shape
        timer.tic()
        if save_theta0:
            print >> stderr, 'saving theta0_la...'
            pos = model_la.rfind('.')
            if pos < 0:
                filename = model_la + '.theta0_la'
            else:
                filename = model_la[0:pos] + '.theta0_la' + model_la[pos:]
            with Writer(filename) as theta0_writer:
                pickle.dump(theta0_la, theta0_writer)
        theta0_la_saving_time = timer.toc() 
    
        # 每隔every步就存储一次模型参数
        callback_la = ThetaSaver( model_la, every ) 
        theta_opt_la = None
        func = compute_cost_and_grad_la           
    
        args_la = ( src_instances, src_total_internal_node, src_word_vectors, src_embsize,
                trg_instances, trg_total_internal_node, trg_word_vectors, trg_embsize, lambda_reg_L,
                lambda_reg_rec,lambda_reg_sem, alpha, bad_src_instances, bad_trg_instances )  

        print >> stderr, 'Start Supervised Optimizing...'
        try:

            # Optimizing with label
            theta_opt_la = lbfgs.optimize( func, theta0_la, maxiter_la, verbose, checking_grad, 
                                    args_la, callback = callback_la )

        except GridentCheckingFailedError:
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        opt_la_time = timer.toc()

        timer.tic()
        # pickle form
        print >> stderr, 'saving parameters to %s' % model
        with Writer( model_la ) as model_pickler:
            pickle.dump( theta_opt_la, model_pickler)
        # pure text form
        with Writer(model_la+'.txt') as writer:
            [writer.write('%20.8f\n' % v) for v in theta_opt_la]
        thetaopt_la_saving_time = timer.toc()     
         
        print >> stderr, 'Init. theta0_la  : %10.2f s' % theta0_la_init_time
        if save_theta0:
            print >> stderr, 'Saving theta0_la : %10.2f s' % theta0_la_saving_time
        print >> stderr, 'Optimizing    : %10.2f s' % opt_la_time
        print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_la_saving_time
        print >> stderr, 'Done!'

    else:
        # prepare training data
        src_instances, src_word_vectors, src_total_internal_node = prepare_data()
        trg_instances, trg_word_vectors, trg_total_internal_node = prepare_data() 

        src_embsize = src_word_vectors.embsize()
        src_param_size = src_embsize*src_embsize*4 + src_embsize*3 + src_embsize*len( src_word_vectors )
        src_theta = zeros((src_param_size, 1))    
        compute_cost_and_grad(src_theta, src_instances, src_total_internal_node, src_word_vectors,
                          src_embsize, lambda_reg, lambda_reg_L )

        trg_embsize = trg_word_vectors.embsize()
        trg_param_size = trg_embsize*trg_embsize*4 + trg_embsize*3 + trg_embsize*len( trg_word_vectors )
        trg_theta = zeros((trg_param_size, 1))    
        compute_cost_and_grad(trg_theta, trg_instances, trg_total_internal_node, trg_word_vectors,
                          trg_embsize, lambda_reg, lambda_reg_L )
 
        src_instances, src_word_vectors, src_total_internal_node, bad_src_instances = prepare_data_la()
        trg_instances, trg_word_vectors, trg_total_internal_node, bad_src_instances = prepare_data_la()

        src_embsize = src_word_vectors.embsize()
        trg_embsize = trg_word_vectors.embsize()
        param_size = src_embsize*trg_embsize*5 + src_embsize*4 +\
                  trg_embsize*trg_embsize*5 + trg_embsize*4 +\
                  src_embsize*src_word_vectors._vectors.shape[1] +\
                  trg_embsize*trg_word_vectors._vectors.shape[1]
        theta = zeros((param_size, 1))    
        compute_cost_and_grad_la( theta, 
                src_instances, src_total_internal_node, src_word_vectors, src_embsize, 
                trg_instances, trg_total_internal_node, trg_word_vectors, trg_embsize, 
                lambda_reg_L,lambda_reg_rec, lambda_reg_sem, alpha, bad_src_instances, bad_trg_instances )

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

from ioutil import Writer
from timeutil import Timer
import lbfgs
from ioutil import Reader
from nn.rae import RecursiveAutoencoder, RecursiveAutoencoder_la
from nn.util import init_W
from nn.instance import Instance
from errors import GridentCheckingFailedError
from vec.wordvector import WordVectors
from functions import tanh_norm1_prime, sum_along_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_cost_and_grad(theta,
                    source_instances, source_total_internal_node, source_word_vectors, source_embsize,
                    target_instances, target_total_internal_node, target_word_vectors, target_embsize,
                    lambda_reg):
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
    source_offset = 4 * source_embsize * source_embsize + 3 * source_embsize
    source_theta = theta[0:source_offset] 
    target_theta = theta[source_offset:]
    # init recursive autoencoder
    # 新建一个autoencoder，并且初始化,将参数恢复成矩阵形式
    source_rae = RecursiveAutoencoder.build( source_theta, source_embsize )
    target_rae = RecursiveAutoencoder.build( target_theta, target_embsize ) 

    # compute local reconstruction error and gradients
    # 计算训练短语的error和gradient
    total_rec_error, total_grad = process( source_rae, target_rae,  
                                source_word_vectors, source_instances, source_total_internal_node,
                                target_word_vectors, target_instances, target_total_internal_node )
    
    # compute total cost
    source_reg = source_rae.get_weights_square()
    target_reg = target_rae.get_weights_square()
    #计算总误差,算上regularizer
    total_cost = total_rec_error + lambda_reg/2 * source_reg + lambda_reg/2 * target_reg
     
    # gradients related to regularizer
    # Source Side
    source_reg_grad = source_rae.get_zero_gradients()
    source_reg_grad.gradWi1 += source_rae.Wi1
    source_reg_grad.gradWi2 += source_rae.Wi2
    source_reg_grad.gradWo1 += source_rae.Wo1
    source_reg_grad.gradWo2 += source_rae.Wo2
    source_reg_grad *= lambda_reg
    
    # Target Side
    target_reg_grad = target_rae.get_zero_gradients()
    target_reg_grad.gradWi1 += target_rae.Wi1
    target_reg_grad.gradWi2 += target_rae.Wi2
    target_reg_grad.gradWo1 += target_rae.Wo1
    target_reg_grad.gradWo2 += target_rae.Wo2
    target_reg_grad *= lambda_reg
    
    reg_grad = [source_reg_grad.to_row_vector(), target_reg_grad.to_row_vector()]
    total_grad += concatenate( reg_grad ) 

    return total_cost, total_grad 
 
def process( source_rae, target_rae, 
           source_word_vectors, source_instances, source_total_internal_node, 
           target_word_vectors, target_instances, target_total_internal_node ):

    total_rec_error = 0
    # 初始化梯度参数
    source_gradients = source_rae.get_zero_gradients()
    source_total_rec_error = 0
    target_gradients = target_rae.get_zero_gradients()
    target_total_rec_error = 0
    for i in xrange( len( source_instances ) ):
        # Source Side
        source_instance = source_instances[i]
        # 取出该短语中所有词向量,instance.words中的单词idx还原成words.embedded中的词向量矩阵n*word_num
        source_words_embedded = source_word_vectors[source_instance.words]
        # 前向传播，计算错误率
        source_root_node, source_rec_error = source_rae.forward( source_words_embedded )
        # 反向传播计算梯度
        source_rae.backward( source_root_node, source_gradients, freq= source_instance.freq )
        source_total_rec_error += source_rec_error * source_instance.freq

        # Target Side
        target_instance = target_instances[i]
        target_words_embedded = target_word_vectors[target_instance.words]
        target_root_node, target_rec_error = target_rae.forward( target_words_embedded )
        target_rae.backward( target_root_node, target_gradients, freq= target_instance.freq )
        target_total_rec_error += target_rec_error * target_instance.freq 

    total_rec_error += ( source_total_rec_error * ( 1.0 / source_total_internal_node ) + 
                       target_total_rec_error * ( 1.0 / target_total_internal_node ) ) 

    grad_row_vec = [source_gradients.to_row_vector() * ( 1.0 / source_total_internal_node ), 
                 target_gradients.to_row_vector() * ( 1.0 / target_total_internal_node )]
    return total_rec_error, concatenate( grad_row_vec )


def compute_cost_and_grad_la(theta,
                    source_instances, source_total_internal_node, source_word_vectors, source_embsize,
                    target_instances, target_total_internal_node, target_word_vectors, target_embsize, 
                    lambda_reg_rec,lambda_reg_sem, alpha, bad_src_instances, bad_trg_instances ): 
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
    source_offset = 4 * source_embsize * source_embsize + 3 * source_embsize
    source_offset_la = 5 * source_embsize * source_embsize + 4 * source_embsize
    target_offset = 4 * target_embsize * target_embsize + 3 * target_embsize
    target_offset_la = 5 * target_embsize * target_embsize + 4 * target_embsize

    source_theta = theta[0:source_offset_la] 
    target_theta = theta[source_offset_la:]
 
    # init recursive autoencoder
    # 新建一个autoencoder，并且初始化,将参数恢复成矩阵形式
    source_rae_la = RecursiveAutoencoder_la.build_la( source_theta, source_embsize )
    target_rae_la = RecursiveAutoencoder_la.build_la( target_theta, target_embsize )
   
    # compute local reconstruction error and gradients
    # 计算训练短语的error和gradient
    total_rec_error, total_sem_error, total_grad = \
                                process_la( source_rae_la, target_rae_la, alpha,
                                source_word_vectors, source_instances, source_total_internal_node,
                                target_word_vectors, target_instances, target_total_internal_node,
                                bad_src_instances, bad_trg_instances )
    
    # compute total cost
    source_reg_rec = source_rae_la.get_weights_square()
    target_reg_rec = target_rae_la.get_weights_square() 
    source_reg_sem = ( source_rae_la.Wla**2 ).sum()
    target_reg_sem = ( target_rae_la.Wla**2 ).sum()

    #计算总误差,算上regularizer
    #print total_rec_error
    #print total_sem_error
    total_cost = \
    alpha * ( total_rec_error + lambda_reg_rec/2 * source_reg_rec + lambda_reg_rec/2 * target_reg_rec ) +\
    ( 1 - alpha ) *( total_sem_error + lambda_reg_sem/2 * source_reg_sem+lambda_reg_sem/2 * target_reg_sem )  
    # compute gradients 
    # gradients related to regularizer
    # Source side
    source_reg_grad = source_rae_la.get_zero_gradients_la()
    source_reg_grad.gradWi1 += lambda_reg_rec * source_rae_la.Wi1 * alpha
    source_reg_grad.gradWi2 += lambda_reg_rec * source_rae_la.Wi2 * alpha
    source_reg_grad.gradWo1 += lambda_reg_rec * source_rae_la.Wo1 * alpha
    source_reg_grad.gradWo2 += lambda_reg_rec * source_rae_la.Wo2 * alpha
    source_reg_grad.gradWla += lambda_reg_sem * source_rae_la.Wla * ( 1- alpha )
 
    # Target side
    target_reg_grad = target_rae_la.get_zero_gradients_la()
    target_reg_grad.gradWi1 += lambda_reg_rec * target_rae_la.Wi1 * alpha
    target_reg_grad.gradWi2 += lambda_reg_rec * target_rae_la.Wi2 * alpha
    target_reg_grad.gradWo1 += lambda_reg_rec * target_rae_la.Wo1 * alpha
    target_reg_grad.gradWo2 += lambda_reg_rec * target_rae_la.Wo2 * alpha
    target_reg_grad.gradWla += lambda_reg_sem * target_rae_la.Wla * ( 1- alpha )
 
    reg_grad = [source_reg_grad.to_row_vector_la(), target_reg_grad.to_row_vector_la()] 
    total_grad += concatenate( reg_grad )

    return total_cost, total_grad 
 
def process_la( source_rae_la, target_rae_la, alpha, 
             source_word_vectors, source_instances, source_total_internal_node,
             target_word_vectors, target_instances, target_total_internal_node,
             bad_src_instances, bad_trg_instances ):

    total_rec_error = 0
    total_sem_error = 0
    # 初始化梯度参数
    source_gradients_la = source_rae_la.get_zero_gradients_la()
    target_gradients_la = target_rae_la.get_zero_gradients_la()    
    source_total_rec_error = 0
    target_total_rec_error = 0
    source_total_sem_error = 0
    target_total_sem_error = 0
    for i in xrange( len( source_instances ) ): 
        source_instance = source_instances[i]
        target_instance = target_instances[i]
        bad_src_instance = bad_src_instances[i]
        bad_trg_instance = bad_trg_instances[i]
        # 取出该短语中所有词向量,instance.words中的单词idx还原成words.embedded中的词向量矩阵n*word_num
        source_words_embedded = source_word_vectors[source_instance.words]
        target_words_embedded = target_word_vectors[target_instance.words]
        bad_source_embedded = source_word_vectors[bad_src_instance] 
        bad_target_embedded = target_word_vectors[bad_trg_instance]
        #print source_words_embedded
        #print target_words_embedded
        # 前向传播，计算错误
        source_root_node, source_rec_error = \
                        source_rae_la.forward_la( source_words_embedded )
        target_root_node, target_rec_error = \
                        target_rae_la.forward_la( target_words_embedded )
        source_total_rec_error += source_rec_error * source_instance.freq
        target_total_rec_error += target_rec_error * target_instance.freq
        bad_source_root, _  = \
                        source_rae_la.forward_la( bad_source_embedded )
        bad_target_root, _ = \
                        target_rae_la.forward_la( bad_target_embedded )

        rec_s = alpha * source_instance.freq / source_total_internal_node
        rec_t = alpha * target_instance.freq / target_total_internal_node
        sem_s = ( 1 - alpha ) * source_instance.freq / source_total_internal_node
        sem_t = ( 1 - alpha ) * target_instance.freq / target_total_internal_node

        # Semantic Error
        # Source side
        source_yla_unnormalized = tanh( dot( source_rae_la.Wla, source_root_node.p ) + source_rae_la.bla )
        source_yla = source_yla_unnormalized / LA.norm( source_yla_unnormalized, axis=0 )
        source_ylapla = source_yla - target_root_node.p
        source_sem_error = 0.5 * sum_along_column( source_ylapla**2 )[0]

        #print source_sem_error
        bad_source_ylapla = source_yla - bad_target_root.p
        bad_source_sem_error = 0.5 * sum_along_column( bad_source_ylapla**2 )[0] 
        source_sem_margin = (source_sem_error-bad_source_sem_error+1)*source_instance.freq
        
        source_sem_margin = max( 0.0, source_sem_margin )
        if source_sem_margin == 0.0:
            soptimal = True
        else:
            soptimal = False
        source_total_sem_error += source_sem_margin

        # Target side
        target_yla_unnormalized = tanh( dot( target_rae_la.Wla, target_root_node.p ) + target_rae_la.bla )
        target_yla = target_yla_unnormalized / LA.norm( target_yla_unnormalized, axis=0 )
        target_ylapla = target_yla - source_root_node.p
        target_sem_error = 0.5 * sum_along_column( target_ylapla**2 )[0]

        bad_target_ylapla = target_yla - bad_source_root.p
        bad_target_sem_error = 0.5 * sum_along_column( bad_target_ylapla**2 )[0]
        target_sem_margin = (target_sem_error-bad_target_sem_error+1)*target_instance.freq
        
        target_sem_margin = max( 0.0, target_sem_margin )
        if target_sem_margin == 0.0:
            toptimal = True
        else:
            toptimal = False
        target_total_sem_error += target_sem_margin 

        # 反向传播计算梯度
        source_rae_la.backward_la( source_root_node, bad_source_root, 
                source_gradients_la, rec_s, sem_s, sem_t,
                source_yla_unnormalized, source_ylapla, target_ylapla,
                bad_source_ylapla, bad_target_ylapla, soptimal, toptimal )
        target_rae_la.backward_la( target_root_node, bad_target_root,
                target_gradients_la, rec_t, sem_t, sem_s, 
                target_yla_unnormalized, target_ylapla, source_ylapla,
                bad_target_ylapla, bad_source_ylapla, toptimal, soptimal )
    
    total_rec_error = ( source_total_rec_error * ( 1.0 / source_total_internal_node ) + 
                    target_total_rec_error * ( 1.0 / target_total_internal_node ) ) 
    total_sem_error = ( source_total_sem_error * ( 1.0 / source_total_internal_node ) + 
                    target_total_sem_error * ( 1.0 / target_total_internal_node ) ) 

    grad_row_vec = [source_gradients_la.to_row_vector_la() , target_gradients_la.to_row_vector_la()] 

    return total_rec_error, total_sem_error, concatenate( grad_row_vec )

def init_theta( source_embsize, target_embsize, _seed = None ):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)
    
    parameters = []
    
    # Source Side 
    # Wi1 n*n
    parameters.append(init_W(source_embsize, source_embsize))
    # Wi2 n*n
    parameters.append(init_W(source_embsize, source_embsize))
    # bi n*1
    parameters.append(zeros(source_embsize))
  
    # Wo1 n*n
    parameters.append(init_W(source_embsize, source_embsize))
    # Wo2 n*n
    parameters.append(init_W(source_embsize, source_embsize))
    # bo1 n*1
    parameters.append(zeros(source_embsize))
    # bo2 n*1
    parameters.append(zeros(source_embsize))


    # Target Side 
    # Wi1 n*n
    parameters.append(init_W(target_embsize, target_embsize))
    # Wi2 n*n
    parameters.append(init_W(target_embsize, target_embsize))
    # bi n*1
    parameters.append(zeros(target_embsize))
  
    # Wo1 n*n
    parameters.append(init_W(target_embsize, target_embsize))
    # Wo2 n*n
    parameters.append(init_W(target_embsize, target_embsize))
    # bo1 n*1
    parameters.append(zeros(target_embsize))
    # bo2 n*1
    parameters.append(zeros(target_embsize))

    if _seed != None:  
        set_state(ori_state)
  
    return concatenate(parameters)   


def init_theta_la( theta, source_embsize, target_embsize, _seed=None ):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)
    
    source_offset = 4 * source_embsize * source_embsize + 3 * source_embsize
    source_theta = theta[0:source_offset] 
    target_theta = theta[source_offset:]

    parameters = []

    # Source side 
    parameters.append( source_theta )
    # Wla n*n
    parameters.append( init_W( source_embsize, source_embsize ) )
    # bla n*1
    parameters.append( zeros( source_embsize ) )

    # Target side 
    parameters.append( target_theta )
    # Wla n*n
    parameters.append( init_W( target_embsize, target_embsize ) )
    # bla n*1
    parameters.append( zeros( target_embsize ) )

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
    
    # load raw data
    with Reader(datafile) as datafile:
        instance_strs = [line for line in datafile]
      
    instances, total_internal_node = load_instances(instance_strs, word_vectors)
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
  
    source_instances_file = options.source_instances
    target_instances_file = options.target_instances
    model = options.model
    model_la = options.model_la
    source_word_vector_file = options.source_word_vector
    target_word_vector_file = options.target_word_vector
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

  
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    if checking_grad:
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.INFO)
        
    print >> stderr, 'Source Instances file: %s' % source_instances_file
    print >> stderr, 'Target Instances file: %s' % target_instances_file
    print >> stderr, 'Model file: %s' % model
    print >> stderr, 'Model with label file: %s' % model_la
    print >> stderr, 'Source Word vector file: %s' % source_word_vector_file 
    print >> stderr, 'Target Word vector file: %s' % target_word_vector_file 
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
    source_word_vectors = WordVectors.load_vectors( source_word_vector_file )
    target_word_vectors = WordVectors.load_vectors( target_word_vector_file )
    #embsize为词向量的维度
    source_embsize = source_word_vectors.embsize()
    target_embsize = target_word_vectors.embsize()
       
    print >> stderr, 'preparing data...' 
    #载入训练短语数据，将短语转化为instance的数组放入instances中
    source_instances, _, source_total_internal_node = prepare_data( source_word_vectors, 
                                                        source_instances_file )
    target_instances, _, target_total_internal_node = prepare_data( target_word_vectors, 
                                                        target_instances_file )
    print >> stderr, 'init. RAE parameters...'
    timer = Timer()
    timer.tic()
    if _seed != None:
        _seed = int(_seed)
    else:
        _seed = None
    print >> stderr, 'seed: %s' % str(_seed)
    # 初始化参数
    theta0 = init_theta( source_embsize, target_embsize, _seed = _seed )
    theta0_init_time = timer.toc()
    print >> stderr, 'shape of theta0 %s' % theta0.shape

    timer.tic()
    if save_theta0:
        print >> stderr, 'saving theta0...'
        pos = model.rfind('.')
        if pos < 0:
            filename = model + '.theta0'
        else:
            filename = model[0:pos] + '.theta0' + model[pos:]
        with Writer(filename) as theta0_writer:
            pickle.dump(theta0, theta0_writer)    
    theta0_saving_time = timer.toc() 
    
    # 每隔every步就存储一次模型参数
    callback = ThetaSaver( model, every )    
    func = compute_cost_and_grad
    args = ( source_instances, source_total_internal_node, source_word_vectors, source_embsize,
           target_instances, target_total_internal_node, target_word_vectors, target_embsize, lambda_reg )
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

    timer.tic()
    # pickle form
    print >> stderr, 'saving parameters to %s' % model
    with Writer( model ) as model_pickler:
        pickle.dump( theta_opt, model_pickler)
    # pure text form
    with Writer(model+'.txt') as writer:
        [writer.write('%20.8f\n' % v) for v in theta_opt]
    thetaopt_saving_time = timer.toc()     
     
    print >> stderr, 'Init. theta0  : %10.2f s' % theta0_init_time
    if save_theta0:
        print >> stderr, 'Saving theta0 : %10.2f s' % theta0_saving_time
    print >> stderr, 'Optimizing    : %10.2f s' % opt_time
    print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_saving_time
    print >> stderr, 'Done!'  

#------------------------------------------------------------

    print >> stderr, 'Unsupervised Optimizing Done!'
    timer.tic()
    if _seed != None:
        _seed = int(_seed)
    else:
        _seed = None
    print >> stderr, 'seed: %s' % str(_seed)
    # 初始化监督学习的参数
    theta0_la = init_theta_la( theta_opt, source_embsize, target_embsize, _seed=_seed )

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
    # Generate negative samples
    bad_src_instances = []
    bad_trg_instances = []
    for i in xrange( len( source_instances ) ): 
        source_instance = source_instances[i]
        target_instance = target_instances[i]
        bad_src_instances.append([rd.randrange( 0, len( source_word_vectors ) ) \
                for j in xrange( len( source_instance.words ) )] )
        bad_trg_instances.append([rd.randrange( 0, len( target_word_vectors ) ) \
                for j in xrange( len( target_instance.words ) )] )
    args_la = ( source_instances, source_total_internal_node, source_word_vectors, source_embsize,
           target_instances, target_total_internal_node, target_word_vectors, target_embsize, 
           lambda_reg_rec,lambda_reg_sem, alpha, bad_src_instances, bad_trg_instances )

    print >> stderr, 'Start Supervised Optimizing...'
    try:
        # 开始优化 
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

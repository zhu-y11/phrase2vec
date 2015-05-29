#-*- coding: utf-8 -*-
'''
Recursive autoencoder

@author: lpeng
'''
from __future__ import division

import argparse
import logging
from sys import stderr

from numpy import arange, dot, zeros, zeros_like, tanh, concatenate
from numpy import linalg as LA

from functions import tanh_norm1_prime, sum_along_column
from vec.wordvector import WordVectors
from ioutil import unpickle, Reader, Writer
from nn.instance import Instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InternalNode(object):
  
    def __init__(self, index,
               left_child, right_child, 
               p, p_unnormalized, 
               y1_minus_c1, y2_minus_c2,
               y1_unnormalized, y2_unnormalized):
        self.index = index # for debugging purpose
        self.left_child = left_child
        self.right_child = right_child
        self.p = p
        self.p_unnormalized = p_unnormalized
        self.y1_minus_c1 = y1_minus_c1
        self.y2_minus_c2 = y2_minus_c2
        self.y1_unnormalized = y1_unnormalized
        self.y2_unnormalized = y2_unnormalized
   
class LeafNode(object):
  
    def __init__(self, index, embedding):
        self.index = index
        self.p = embedding

class RecursiveAutoencoder_la(object):
  
    def __init__( self, Wi1, Wi2, bi, Wo1, Wo2, bo1, bo2, L, Wla, bla,
               f=tanh, f_norm1_prime=tanh_norm1_prime ):
        '''Initialize the weight matrices and set the nonlinear function
    
        Note: bi, bo1, bo2 and bla must have the shape (embsize, 1) instead of (embsize,)
    
        Args:
        Wi1, Wi2, bi: weight matrices for encoder
        Wo1, Wo2, bo1, bo2: weight matrices for decoder
        Wla, bl: weight matrics for semi-supervised learning
        f: nonlinear function
        f_norm1_prime: returns the Jacobi matrix of f(x)/sqrt(||f(x)||^2).
            Note that the input of f_norm1_prime is f(x).
        '''
        self.Wi1 = Wi1
        self.Wi2 = Wi2
        self.bi = bi
    
        self.Wo1 = Wo1
        self.Wo2 = Wo2
        self.bo1 = bo1
        self.bo2 = bo2

        self.L = L
    
        self.Wla = Wla
        self.bla = bla

        self.f = f
        self.f_norm1_prime = f_norm1_prime
  
    def get_embsize_la(self):
        return self.bi.shape[0]
  
    @classmethod
    def build_la( cls, theta, embsize ):
        '''Initailize a recursive autoencoder using a parameter vector
    
        Args:
        theta: parameter vector
        embsize: dimension of word embedding vector
        '''

        offset = 5 * embsize * embsize + 4 * embsize
        num_word = (int)( ( theta.size - offset) / embsize)

        assert(theta.size == cls.compute_parameter_num_la(embsize,num_word))
        offset = 0
        sz = embsize * embsize
        Wi1 = theta[offset:offset+sz].reshape(embsize, embsize)
        offset += sz
    
        Wi2 = theta[offset:offset+sz].reshape(embsize, embsize)
        offset += sz
    
        bi = theta[offset:offset+embsize].reshape(embsize, 1)
        offset += embsize
    
        Wo1 = theta[offset:offset+sz].reshape(embsize, embsize)
        offset += sz
    
        Wo2 = theta[offset:offset+sz].reshape(embsize, embsize)
        offset += sz
    
        bo1 = theta[offset:offset+embsize].reshape(embsize, 1)
        offset += embsize
    
        bo2 = theta[offset:offset+embsize].reshape(embsize, 1)
        offset += embsize

        L = theta[offset:offset+embsize*num_word].reshape(embsize, num_word)
        offset += embsize*num_word
 
        Wla = theta[offset:offset + sz].reshape( embsize, embsize )
        offset += sz

        bla = theta[offset:offset + embsize].reshape( embsize, 1 )
        offset += embsize
    
        return RecursiveAutoencoder_la( Wi1, Wi2, bi, Wo1, Wo2, bo1, bo2, L, Wla, bla )
        
    @classmethod
    def compute_parameter_num_la(cls, embsize, num_word):
        '''Compute the parameter number of a recursive autoencoder
    
        Args:
        embsize: dimension of word embedding vector
    
        Returns:
        number of parameters
        '''
        sz = embsize*embsize # Wi1
        sz += embsize*embsize # Wi2
        sz += embsize # bi
        sz += embsize*embsize # Wo1
        sz += embsize*embsize # Wo2
        sz += embsize # bo1
        sz += embsize # bo2
        sz += embsize*embsize # Wla
        sz += embsize # bla
        sz += embsize*num_word #L
        return sz
      
    def get_weights_square(self):
        square = (self.Wi1**2).sum()
        square += (self.Wi2**2).sum()
        square += (self.Wo1**2).sum()
        square += (self.Wo2**2).sum()
        return square
  
    def get_bias_square(self):
        square = (self.bi**2).sum()
        square += (self.bo1**2).sum()
        square += (self.bo2**2).sum()
        return square
  
    def encode(self, words_embedded):
        root, _ = self.forward(words_embedded)
        return root.p
    
    def forward_la(self, words_embedded, instance ):
        ''' Forward pass of training recursive autoencoders using backpropagation
        through structures.  
    
        Args:
        words_embedded: word embedding vectors (column vectors)
      
        Returns:
        value1: root of the tree, an instance of InternalNode 
        value2: reconstruction_error
        '''
        # 短语中的单词数量
        words_num = words_embedded.shape[1]
    
        # 共有n+n-1 = 2n-1个节点
        tree_nodes = [None]*(2*words_num - 1)
        # 前n个节点为输入单词的叶节点
        tree_nodes[0:words_num] = [LeafNode(instance.words[i], words_embedded[:, (i,)]) 
                                        for i in range(words_num)]
    
        reconstruction_error = 0
    
        # build a tree greedily
        # initialize reconstruction errors
        # 前n-1个单词组成的列向量
        c1 = words_embedded[:, arange(words_num-1)]
        # 第1到第n个单词组成的列向量
        c2 = words_embedded[:, arange(1, words_num)]
        # c1,c2用来计算最小error.
        # 计算所有叶节点邻居对的error，挑选error最小的一组开始
        p_unnormalized = self.f(dot(self.Wi1, c1) + dot(self.Wi2, c2)\
                               + self.bi[:, zeros(words_num-1, dtype=int)])
        p = p_unnormalized / LA.norm(p_unnormalized, axis=0)
    
        y1_unnormalized = self.f(dot(self.Wo1, p)\
                               + self.bo1[:, zeros(words_num-1, dtype=int)])
        y1 = y1_unnormalized / LA.norm(y1_unnormalized, axis=0)
  
        y2_unnormalized = self.f(dot(self.Wo2, p)\
                               + self.bo2[:, zeros(words_num-1, dtype=int)])
        y2 = y2_unnormalized / LA.norm(y2_unnormalized, axis=0)
    
        y1c1 = y1 - c1
        y2c2 = y2 - c2
    
        J = 1/2 * (sum_along_column(y1c1**2) + sum_along_column(y2c2**2))
    
        # initialize candidate internal nodes
        candidate_nodes = []
        # 构造n-1个由叶节点邻居对构成的非叶节点
        for i in range(words_num-1):
            left_child = tree_nodes[i]
            right_child = tree_nodes[i+1]
            # idx : -1到-n + 1，共n-1个非叶节点
            node = InternalNode(-i-1, left_child, right_child,
                          p[:, (i,)], p_unnormalized[:, (i,)],
                          y1c1[:, (i,)], y2c2[:, (i,)],
                          y1_unnormalized[:, (i,)], 
                          y2_unnormalized[:, (i,)])
            candidate_nodes.append(node)
        debugging_cand_node_index = words_num
      
        # 寻找最优非叶节点
        for j in range(words_num-1):
            # find the smallest reconstruction error
            # 最小error非叶节点的index
            J_minpos = J.argmin()
            # 最小error非叶节点值
            J_min = J[J_minpos]
            # 更新总error
            reconstruction_error += J_min
      
            #取出该最小error非叶节点
            node = candidate_nodes[J_minpos]
            node.index = words_num + j # for dubugging
            # 在最小error生成树中添加进该节点
            tree_nodes[words_num+j] = node
  
            # update reconstruction errors
            # 最优节点右侧还有节点，可以同右侧节点组合
            if J_minpos+1 < len(candidate_nodes):
                c1 = node
                c2 = candidate_nodes[J_minpos+1].right_child
                right_cand_node, right_J = self.__build_internal_node(c1, c2)
        
                # 内节点index从-n开始累减
                right_cand_node.index = -debugging_cand_node_index
                debugging_cand_node_index += 1
                candidate_nodes[J_minpos+1] = right_cand_node
        
                J[J_minpos+1] = right_J

            #最优内节点不是最左侧的节点 
            if J_minpos-1 >= 0:
                c1 = candidate_nodes[J_minpos-1].left_child
                c2 = node
                left_cand_node, left_J = self.__build_internal_node(c1, c2)
        
                left_cand_node.index = -debugging_cand_node_index
                debugging_cand_node_index += 1
                candidate_nodes[J_minpos-1] = left_cand_node
                J[J_minpos-1] = left_J
      
            valid_indices = [i for i in range(words_num-1-j) if i != J_minpos]
            J = J[valid_indices]
            candidate_nodes = [candidate_nodes[k] for k in valid_indices]

        return tree_nodes[-1], reconstruction_error 
  
    def __build_internal_node(self, c1_node, c2_node):
        '''Build a new internal node which represents the representation of 
        c1_node.p and c2_node.p computed using autoencoder  
    
        Args:
        c1_node: left node
        c2_node: right node
      
        Returns:
        value1: a new internal node 
        value2: reconstruction error, a scalar
    
        ''' 
        c1 = c1_node.p
        c2 = c2_node.p
        p_unnormalized = self.f(dot(self.Wi1, c1) + dot(self.Wi2, c2) + self.bi)
        p = p_unnormalized / LA.norm(p_unnormalized, axis=0)
    
        y1_unnormalized = self.f(dot(self.Wo1, p) + self.bo1)
        y1 = y1_unnormalized / LA.norm(y1_unnormalized, axis=0)
  
        y2_unnormalized = self.f(dot(self.Wo2, p) + self.bo2)
        y2 = y2_unnormalized / LA.norm(y2_unnormalized, axis=0)
    
        y1c1 = y1 - c1
        y2c2 = y2 - c2
    
        node = InternalNode(-1, c1_node, c2_node,
                        p, p_unnormalized,
                        y1c1, y2c2,
                        y1_unnormalized, y2_unnormalized)

        reconstruction_error = sum_along_column(y1c1**2) + sum_along_column(y2c2**2)
        reconstruction_error = 0.5*reconstruction_error[0]
    
        return node, reconstruction_error 
  
  
    class Gradients_la(object):
        '''
        Class for storing gradients.
        '''
    
        def __init__(self, rae):
            '''
            Args:
            rae: an instance of RecursiveAutoencoder
            '''
            self.gradWi1 = zeros_like(rae.Wi1)
            self.gradWi2 = zeros_like(rae.Wi2)
            self.gradbi = zeros_like(rae.bi)
      
            self.gradWo1 = zeros_like(rae.Wo1)
            self.gradWo2 = zeros_like(rae.Wo2)
            self.gradbo1 = zeros_like(rae.bo1)
            self.gradbo2 = zeros_like(rae.bo2)
            
            self.gradL = zeros_like(rae.L)

            self.gradWla = zeros_like( rae.Wla )
            self.gradbla = zeros_like( rae.bla )

        def to_row_vector_la(self):
            '''Place all the gradients in a row vector
            '''
            vectors = []
            vectors.append(self.gradWi1.reshape(self.gradWi1.size, 1))
            vectors.append(self.gradWi2.reshape(self.gradWi2.size, 1))
            vectors.append(self.gradbi)
            vectors.append(self.gradWo1.reshape(self.gradWo1.size, 1))
            vectors.append(self.gradWo2.reshape(self.gradWo2.size, 1))
            vectors.append(self.gradbo1)
            vectors.append(self.gradbo2)
            vectors.append(self.gradL.reshape(self.gradL.size,1))
            vectors.append(self.gradWla.reshape(self.gradWla.size, 1))
            vectors.append(self.gradbla)
            return concatenate(vectors)[:, 0]
     
    def get_zero_gradients_la(self):
        return self.Gradients_la(self)
  
    def backward_la(self, root_node, total_grad, rec, sem, 
                yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent = None ):
        '''Backward pass of training recursive autoencoder using backpropagation
        through structures.
    
        Args:
        root_node: an instance of InternalNode returned by forward()
        total_grad: the local gradients will be added to it. 
            It should be initialized by get_zero_gradients()
        delta_parent: delta vector that propagates from upper layer, it must have
            the shape (embsize, 1) instead of (embsize,)
        freq: frequency of this instance
    
        Returns:
        None
        '''
        if delta_parent is None:
            delta_parent_out = zeros((self.bi.size, 1))
        else:
            delta_parent_out = delta_parent

        top = True
        self.__backward_la( root_node, total_grad, rec, sem, top, 
                yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out )
  
    def __backward_la( self, node, total_grad, rec, sem, top, 
                yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out ):
        '''Backward pass of training recursive autoencoder using backpropagation
        through structures.
    
        Args:
        node: an instance of InternalNode or LeafNode
        total_grad: the local gradients will be added to it. 
            It should be initialized by get_zero_gradients()
        delta_parent_out: delta vector that propagates from upper layer
        freq: frequency of this instance 
    
        Returns:
        None
        '''
        if isinstance(node, InternalNode):
            # reconstruction layer
            jcob1 = self.f_norm1_prime(node.y1_unnormalized)
            delta_out1 = dot(jcob1, node.y1_minus_c1)

            jcob2 = self.f_norm1_prime(node.y2_unnormalized)
            delta_out2 = dot(jcob2, node.y2_minus_c2)
        
            total_grad.gradWo1 += dot(delta_out1, node.p.T) * rec 
            total_grad.gradWo2 += dot(delta_out2, node.p.T) * rec 
            total_grad.gradbo1 += delta_out1 * rec
            total_grad.gradbo2 += delta_out2 * rec 
            # encoder layer
            if top:
                if not optimal:
                    jcobla = self.f_norm1_prime( yla_unnormalized )
                    delta_outla = dot( jcobla, ylapla )
                    bad_delta_outla = dot( jcobla, bad_ylapla )
                    total_grad.gradWla += (dot(delta_outla,node.p.T) - dot(bad_delta_outla,node.p.T))\
                                * sem
                    total_grad.gradbla += ( delta_outla - bad_delta_outla ) * sem
                    
                    delta_sum = dot(self.Wo1.T, delta_out1) * rec\
                        + dot(self.Wo2.T, delta_out2) * rec\
                        + ( dot(self.Wla.T, delta_outla) - dot(self.Wla.T, bad_delta_outla) ) * sem

                    delta_parent = dot(self.f_norm1_prime(node.p_unnormalized), delta_sum)

                    total_grad.gradWi1 += dot(delta_parent, node.left_child.p.T)
                    total_grad.gradWi2 += dot(delta_parent, node.right_child.p.T)
                    total_grad.gradbi += delta_parent
                else:
                 
                    total_grad.gradWla += 0 
                    total_grad.gradbla += 0 
                    
                    delta_sum = dot(self.Wo1.T, delta_out1) * rec\
                        + dot(self.Wo2.T, delta_out2) * rec

                    delta_parent = dot(self.f_norm1_prime(node.p_unnormalized), delta_sum)

                    total_grad.gradWi1 += dot(delta_parent, node.left_child.p.T)
                    total_grad.gradWi2 += dot(delta_parent, node.right_child.p.T)
                    total_grad.gradbi += delta_parent

                top = False
                # recursive
                delta_parent_out_left = dot(self.Wi1.T, delta_parent)/rec - node.y1_minus_c1
                self.__backward_la(node.left_child, total_grad, rec, sem, top, 
                        yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out_left )
      
                delta_parent_out_right = dot(self.Wi2.T, delta_parent)/rec - node.y2_minus_c2
                self.__backward_la(node.right_child, total_grad, rec, sem, top, 
                        yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out_right )
            else:
                delta_sum = dot(self.Wo1.T, delta_out1)\
                        + dot(self.Wo2.T, delta_out2)\
                        + delta_parent_out

                delta_parent = dot(self.f_norm1_prime(node.p_unnormalized), delta_sum)

                total_grad.gradWi1 += dot(delta_parent, node.left_child.p.T) * rec
                total_grad.gradWi2 += dot(delta_parent, node.right_child.p.T) * rec
                total_grad.gradbi += delta_parent * rec
                 
                # recursive
                delta_parent_out_left = dot(self.Wi1.T, delta_parent) - node.y1_minus_c1
                self.__backward_la(node.left_child, total_grad, rec, sem, top, 
                        yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out_left )
     
                delta_parent_out_right = dot(self.Wi2.T, delta_parent) - node.y2_minus_c2
                self.__backward_la(node.right_child, total_grad, rec, sem, top,
                        yla_unnormalized, ylapla, bad_ylapla, optimal, delta_parent_out_right )

        elif isinstance(node, LeafNode):
            bvec = zeros( ( total_grad.gradL.shape[1], 1 ) )
            bvec[node.index] = 1
            total_grad.gradL += dot( delta_parent_out, bvec.T ) * rec
        else:
            msg = 'node should be an instance of InternalNode or LeafNode';
            raise TypeError(msg)

def Sim( p, p_o ):
    print dot(p.T,p_o)
    return dot( p.T, p_o )[0][0]

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
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_instances', help='input file, each line is a phrase')
    parser.add_argument('target_instances', help='input file, each line is a phrase')
    parser.add_argument('source_word_vector', help='source word vector file')
    parser.add_argument('target_word_vector', help='target word vector file')
    parser.add_argument('theta', help='RAE parameter file (pickled)')
    parser.add_argument('output', help='output file')
    options = parser.parse_args()

    src_instances_file = options.source_instances
    trg_instances_file = options.target_instances 
    src_word_vector_file = options.source_word_vector
    trg_word_vector_file = options.target_word_vector
    theta_file = options.theta
    output_file = options.output
  
    print >> stderr, 'load word vectors...'
    src_word_vectors = WordVectors.load_vectors(src_word_vector_file)
    trg_word_vectors = WordVectors.load_vectors(trg_word_vector_file)
    src_embsize = src_word_vectors.embsize()
    trg_embsize = trg_word_vectors.embsize()
  
    print >> stderr, 'load RAE parameters...'
    theta = unpickle(theta_file)
    src_offset = 5 * src_embsize * src_embsize + 4 * src_embsize + src_embsize*len( src_word_vectors )
    src_theta = theta[0:src_offset]
    trg_theta = theta[src_offset:]
    src_rae_la = RecursiveAutoencoder_la.build_la( src_theta, src_embsize )
    trg_rae_la = RecursiveAutoencoder_la.build_la( trg_theta, trg_embsize )
    src_word_vectors._vectors = src_rae_la.L
    trg_word_vectors._vectors = trg_rae_la.L
    
    src_total_cost = 0
    trg_total_cost = 0
  
    print '='*63
    print '%20s %20s %20s' % ('all', 'avg/node', 'internal node')
    print '-'*63
  
    src_instances, _, src_total_internal_node_num = prepare_data( src_word_vectors, src_instances_file )
    trg_instances, _, trg_total_internal_node_num = prepare_data( trg_word_vectors, trg_instances_file )
    
    src_total_instance_num = len( src_instances )
    trg_total_instance_num = len( trg_instances )

    src_file = open( src_instances_file, 'r' )
    trg_file = open( trg_instances_file, 'r' )

    with Writer(output_file) as writer:     
        for i in xrange( len( src_instances ) ):
            src_phrase = src_file.readline().strip().split(' ||| ')[0]
            trg_phrase = trg_file.readline().strip().split(' ||| ')[0]
            src_instance = src_instances[i]
            trg_instance = trg_instances[i]
            src_word_embedded = src_word_vectors[src_instance.words]
            trg_word_embedded = trg_word_vectors[trg_instance.words]
            src_root_node, src_rec_error = src_rae_la.forward_la( src_word_embedded, src_instance )
            trg_root_node, trg_rec_error = trg_rae_la.forward_la( trg_word_embedded, trg_instance )
            src_vec = src_root_node.p.T[0] # convert n*1 vector to common vector
            trg_vec = trg_root_node.p.T[0] # convert n*1 vector to common vector

            # Calculate ps* and pt*
            src_yla_unnormalized = tanh( dot( src_rae_la.Wla, src_root_node.p ) + src_rae_la.bla )
            src_yla = src_yla_unnormalized / LA.norm( src_yla_unnormalized, axis=0 )
            trg_yla_unnormalized = tanh( dot( trg_rae_la.Wla, trg_root_node.p ) + trg_rae_la.bla )
            trg_yla = trg_yla_unnormalized / LA.norm( trg_yla_unnormalized, axis=0 )
           # Calculate the similarity 
            sim = ( Sim( src_yla, trg_root_node.p ) + Sim( trg_yla, src_root_node.p ) ) / 2

            writer.write( src_phrase )
            writer.write(' ||| ')
            writer.write( trg_phrase )
            writer.write(' ||| ')
            writer.write( str( sim ) )
            writer.write('\n')
       
            src_total_cost += src_rec_error
            trg_total_cost += trg_rec_error

    src_file.close()
    trg_file.close()

    print '-'*63    
    print 'average source reconstruction error per instance: %20.8f' % (src_total_cost / src_total_instance_num)
    print 'average source reconstruction error per node:     %20.8f' % (src_total_cost / src_total_internal_node_num)
    print 'average target reconstruction error per instance: %20.8f' % (trg_total_cost / trg_total_instance_num)
    print 'average target reconstruction error per node:     %20.8f' % (trg_total_cost / trg_total_internal_node_num)
    print '='*63

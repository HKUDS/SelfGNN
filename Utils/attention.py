# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class AdditiveAttention(object):
    def __init__(self,query_vector_dim,candidate_vector_dim,writer=None,tag=None,names=None):
        #self.dense  = tf.layers.dense(candidate_vector_dim, query_vector_dim)
        self.query_vector_dim=query_vector_dim
        self.candidate_vector_dim=candidate_vector_dim
        self.attention_query_vector = tf.random_uniform(shape=[query_vector_dim,1],minval=-0.1,maxval=0.1)
        
    def attention(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        with tf.name_scope('additive_attention'): 
            dense  = tf.layers.dense(candidate_vector, self.query_vector_dim)
            # batch_size, candidate_size, query_vector_dim
            print("candidate_vector",candidate_vector)
            temp = tf.tanh(dense)
            # batch_size, candidate_size
            candidate_weights = tf.nn.softmax(tf.squeeze(tf.matmul( temp, self.attention_query_vector),axis=2),axis=1) #* 128
            # batch_size, 1, candidate_size * batch_size, candidate_size, candidate_vector_dim =
            # batch_size, candidate_vector_dim
            target =tf.squeeze( tf.matmul(tf.expand_dims(candidate_weights,1),candidate_vector),1)
            #target = tf.multiply(candidate_weights,candidate_vector)
            return target
 
class ScaledDotProductAttention(object):
    def __init__(self, d_k):
        self.d_k = d_k
    
    def attention(self, Q, K, V, attn_mask=None):
        with tf.name_scope('scaled_attention'): 
            # batch_size,head_num, candidate_num, candidate_num
            scores = tf.matmul(Q, tf.transpose(K,perm=[0,1,3,2])) / np.sqrt(self.d_k)
            scores = tf.exp(scores)
            if attn_mask is not None:
                scores = scores * attn_mask
            # batch_size,head_num, candidate_num, 1
            attn = scores / (tf.expand_dims(tf.reduce_sum(scores, axis=-1),-1) + 1e-8) # 归一化
            context = tf.matmul(attn, V)
            return context, attn

class MultiHeadSelfAttention(object):
    def __init__(self, d_model, num_attention_heads):
        self.d_model = d_model # embedding_size
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads #16
        self.d_v = d_model // num_attention_heads
        
    def attention(self, Q, K=None, V=None, length=None):
        """
        Q:batch_size,candidate_num,embedding_size
        return : batch_size,candidate_num,embedding_size
        """
        with tf.name_scope('multihead_selfattention'): 
            if K is None:
                K = Q
            if V is None:
                V = Q
            batch_size = Q.shape[0]
            W_Q = tf.layers.dense(Q, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            # batch_size, candidate_num, num_attention_heads,d_k  ;;divide into groups whose num is num_attention_heads
            # batch_size, num_attention_heads, candidate_num,d_k
            q_s = tf.transpose(tf.reshape(W_Q,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_K = tf.layers.dense(K, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            k_s = tf.transpose(tf.reshape(W_K,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_V = tf.layers.dense(V, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            v_s = tf.transpose(tf.reshape(W_V,[batch_size, -1, self.num_attention_heads,self.d_v]),perm=[0,2,1,3])
            # batch_size,head_num, candidate_num, d_k
            context, attn = ScaledDotProductAttention(self.d_k).attention(q_s, k_s, v_s)#,attn_mask)
            # batch_size,candidate_num,embedding_size
            context= tf.reshape(tf.transpose(context,perm=[0,2,1,3]),[batch_size, -1, self.num_attention_heads*self.d_v])
            return context

    
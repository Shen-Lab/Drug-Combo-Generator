import tensorflow as tf
import numpy as np
import gym
import gym_molecule
from baselines.common.distributions import make_pdtype,MultiCatCategoricalPdType,CategoricalPdType
import baselines.common.tf_util as U


from keras.layers import Input,Reshape,Embedding,GRU,LSTM,Conv1D,LeakyReLU,MaxPooling1D,concatenate,Dropout,Dense,LeakyReLU,TimeDistributed
from keras import regularizers
from keras.optimizers import SGD,Adam
from keras.losses import mean_squared_error
from keras.models import Model
import keras.backend as K
from keras.activations import relu
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint,TensorBoard
import re

##################  oracle for drug protein binding


def basic_tokenizer(sentence,condition):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if condition ==0:
        l = _WORD_SPLIT.split(space_separated_fragment)
        del l[0::2]
    elif condition == 1:
        l = _WORD_SPLIT_2.split(space_separated_fragment)
    words.extend(l)
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,condition,normalize_digits=False):

  words = basic_tokenizer(sentence,condition)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


class joint_attn(Layer):

    def __init__(self, max_size1,shape1,max_size2,shape2, **kwargs):
        self.max_size1 = max_size1
        self.max_size2 = max_size2
        self.shape1 = shape1
        self.shape2 = shape2
        super(joint_attn, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                      shape=(self.shape1,self.shape2),
                                      initializer='random_uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                      shape=(self.max_size1,self.max_size2),
                                      initializer='random_uniform',
                                      trainable=True)

        super(joint_attn, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        V = tf.einsum('bij,jk->bik',x,self.W)
        joint_attn = tf.nn.softmax(tf.tanh(tf.einsum('bkj,bij->bik',y,V)+self.b))
        return joint_attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.max_size1,self.max_size2)


class joint_vectors(Layer):

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super(joint_vectors, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W1 = self.add_weight(name='W1',
                                      shape=(self.dim,input_shape[0][2]),
                                      initializer='random_uniform',
                                      trainable=True)
        self.W2 = self.add_weight(name='W2',
                                      shape=(self.dim,input_shape[1][2]),
                                      initializer='random_uniform',
                                      trainable=True)

        self.b = self.add_weight(name='b',
                                      shape=(self.dim,),
                                      initializer='random_uniform',
                                      trainable=True)

        super(joint_vectors, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        joint_attn = inputs[2]
        prot = tf.einsum('bij,kj->bik',x,self.W1)
        drug = tf.einsum('bij,kj->bik',y,self.W2)
        vec = tf.tanh(tf.einsum('bik,bjk->bijk',prot,drug)+self.b)
        Attn = tf.expand_dims(tf.einsum('bijk,bij->bk',vec,joint_attn),2)
        return Attn

    def compute_output_shape(self, input_shape):
        return  (input_shape[0][0],self.dim,1)

def oracle(prot_data,drug_data,name='class_binding'):
    prot_max_size = 152
    comp_max_size = 100
    vocab_size_protein = 76
    vocab_size_compound = 68
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        prot_embd = Embedding(input_dim = vocab_size_protein, output_dim = 256, input_length=prot_max_size)(prot_data)
        prot_lstm = GRU(units=256,return_sequences=True)(prot_embd)
        prot_lstm = Reshape((prot_max_size,256))(prot_lstm)

        drug_embd = Embedding(input_dim = vocab_size_compound, output_dim = 128, input_length=comp_max_size)(drug_data)
        drug_lstm = GRU(units=128,return_sequences=True)(drug_embd)
        drug_lstm = Reshape((comp_max_size,128))(drug_lstm)

        joint_att = joint_attn(max_size1=prot_max_size,shape1=256,
                                max_size2=comp_max_size,shape2=128,name='joint_attn')([prot_lstm,drug_lstm])

        Attn = joint_vectors(dim=256)([prot_lstm,drug_lstm,joint_att])

        conv_1 = Conv1D(filters=64,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(Attn)
        conv_1 = LeakyReLU(alpha=0.1)(conv_1)
        pool_1 = MaxPooling1D(pool_size=4)(conv_1)
        final = Reshape((64*32,))(pool_1)


        fc_1 = Dense(units=600,kernel_initializer='glorot_uniform')(final)
        fc_1 = LeakyReLU(alpha=0.1)(fc_1)
        #drop_2 = Dropout(rate=0.8)(fc_1)
        fc_2 = Dense(units=300,kernel_initializer='glorot_uniform')(fc_1)
        fc_2 = LeakyReLU(alpha=0.1)(fc_2)
        #drop_3 = Dropout(rate=0.8)(fc_2)
        linear = Dense(units=1,activation="linear",kernel_initializer='glorot_uniform')(fc_2)
    #model = Model(inputs=[prot_data,drug_data],outputs=[linear])
    #optimizer = Adam(0.001)

    #model.compile(loss=mean_squared_error,
    #          optimizer=optimizer)

    #model.load_weights(filepath)
    return linear

# gcn mean aggregation over edge features
def GCN(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple'):
    '''
    state s: (adj,node_feature)
    :param adj: b*n*n
    :param node_feature: 1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[0]
    in_channels = node_feature.get_shape()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [edge_dim, in_channels, out_channels])
        b = tf.get_variable("b", [edge_dim, 1, out_channels])
        node_embedding = adj@tf.tile(node_feature,[edge_dim,1,1])@W+b
        if is_act:
            node_embedding = tf.nn.relu(node_embedding)
        # todo: try complex aggregation
        node_embedding = tf.reduce_mean(node_embedding,axis=0,keepdims=True) # mean pooling
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

# gcn mean aggregation over edge features
def GCN_batch(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple',aggregate='sum'):
    '''
    state s: (adj,node_feature)
    :param adj: none*b*n*n
    :param node_feature: none*1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[1]
    batch_size = tf.shape(adj)[0]
    in_channels = node_feature.get_shape()[-1]

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [1, edge_dim, in_channels, out_channels],initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b", [1, edge_dim, 1, out_channels])
        # node_embedding = adj@tf.tile(node_feature,[1,edge_dim,1,1])@tf.tile(W,[batch_size,1,1,1])+b # todo: tf.tile sum the gradients, may need to change
        node_embedding = adj@tf.tile(node_feature,[1,edge_dim,1,1])@tf.tile(W,[batch_size,1,1,1]) # todo: tf.tile sum the gradients, may need to change
        if is_act:
            node_embedding = tf.nn.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = tf.reduce_sum(node_embedding, axis=1, keepdims=True)  # mean pooling
        elif aggregate=='mean':
            node_embedding = tf.reduce_mean(node_embedding,axis=1,keepdims=True) # mean pooling
        elif aggregate=='concat':
            node_embedding = tf.concat(tf.split(node_embedding,axis=1,num_or_size_splits=edge_dim),axis=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

def bilinear(emb_1, emb_2, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    batch_size = tf.shape(emb_1)[0]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [1, node_dim, node_dim])
        return emb_1 @ tf.tile(W,[batch_size,1,1]) @ tf.transpose(emb_2,[0,2,1])

def bilinear_multi(emb_1, emb_2, out_dim, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    batch_size = tf.shape(emb_1)[0]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [1,out_dim, node_dim, node_dim])
        emb_1 = tf.tile(tf.expand_dims(emb_1,axis=1),[1,out_dim,1,1])
        emb_2 = tf.transpose(emb_2,[0,2,1])
        emb_2 = tf.tile(tf.expand_dims(emb_2,axis=1),[1,out_dim,1,1])
        return emb_1 @ tf.tile(W,[batch_size,1,1,1]) @ emb_2

def emb_node(ob_node,out_channels):
    batch_size = tf.shape(ob_node)[0]
    in_channels = ob_node.get_shape()[-1]
    emb = tf.get_variable('emb',[1,1,in_channels,out_channels])
    return ob_node @ tf.tile(emb,[batch_size,1,1,1])


def discriminator_net(ob,num_feat,args,name='d_net'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        ob_node = tf.layers.dense(ob['node'], 8, activation=None, use_bias=False, name='emb')  # embedding layer
        if args.bn==1:
            ob_node = tf.layers.batch_normalization(ob_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        for i in range(args.layer_num_d - 2):
            if args.bn==1:
                emb_node = tf.layers.batch_normalization(emb_node,axis=-1)
            emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node,axis=-1)
        # emb_graph = tf.reduce_max(tf.squeeze(emb_node2, axis=1),axis=1)  # B*f
        emb_node = tf.layers.dense(emb_node, args.emb_size, activation=tf.nn.relu, use_bias=False, name='linear1')
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node,axis=-1)


        if args.gate_sum_d==1:
            emb_node_gate = tf.layers.dense(emb_node,1,activation=tf.nn.sigmoid,name='gate')
            emb_graph = tf.reduce_sum(tf.squeeze(emb_node*emb_node_gate, axis=1),axis=1)  # B*f
        else:
            emb_graph = tf.reduce_sum(tf.squeeze(emb_node, axis=1), axis=1)  # B*f
        logit = tf.layers.dense(emb_graph, num_feat, activation=None, name='linear2')
        pred = tf.sigmoid(logit)
        # pred = tf.layers.dense(emb_graph, 1, activation=None, name='linear1')
        return pred,logit


def embnet(ob, args, name='class_d_net'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        ob_node = tf.layers.dense(ob['node'], 8, activation=None, use_bias=False, name='emb')  # embedding layer
        if args.bn==1:
            ob_node = tf.layers.batch_normalization(ob_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        for i in range(args.layer_num_d - 2):
            if args.bn==1:
                emb_node = tf.layers.batch_normalization(emb_node, axis=-1)
            emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node,axis=-1)
        emb_node = tf.layers.dense(emb_node, args.emb_size, activation=tf.nn.relu, use_bias=False, name='linear1')
        if args.bn==1:
            emb_node = tf.layers.batch_normalization(emb_node, axis=-1)


        if args.gate_sum_d==1:
            emb_node_gate = tf.layers.dense(emb_node, 1, activation=tf.nn.sigmoid, name='gate')
            emb_graph = tf.reduce_sum(tf.squeeze(emb_node*emb_node_gate, axis=1), axis=1)  # B*f
        else:
            emb_graph = tf.reduce_sum(tf.squeeze(emb_node, axis=1), axis=1)  # B*f
        return emb_graph




def SWDBlock(name, inputs):
    i_sh = inputs.get_shape().as_list()
    with tf.name_scope(name) as scope:
        off = tf.Variable(np.zeros((1, i_sh[1]), dtype='float32'),name = name + '.offset')
        sca = tf.Variable(np.ones((1, i_sh[1]), dtype='float32'), name = name + '.scale')
        #offset = tf.tile(off, [i_sh[0], 1])
        #scale = tf.tile(sca, [i_sh[0], 1])
        output = tf.nn.leaky_relu(tf.multiply(sca, inputs) + off)
        return output, sca


def get_powers(dim,degree):
    '''
    This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    '''
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1,degree - value):
                yield (value,) + permutation


def homopoly(dim,degree):
    '''
    calculates the number of elements in a homogeneous polynomial
    '''
    return len(list(get_powers(dim,degree)))

def poly(X,theta,degree):
    ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
     '''
    N,d=X.shape
    dim_poly = homopoly(d,degree)
    assert theta.shape[0]==dim_poly
    powers=list(get_powers(d,degree))
    HX_list = [1.0]*int(dim_poly)
    #HX = tf.ones_like(X)
    #HX = tf.tile(HX,[1,int(dim_poly/d.value)])
    for k,power in enumerate(powers):
        for i,p in enumerate(power):
            HX_list[k] *= X[:,i]**p
    
    HX = tf.transpose(tf.stack(HX_list))
    if len(theta.shape)==1:
       return tf.matmul(HX,theta)
    else:
       return tf.matmul(HX,theta)

def projection(inputs,num_proj,dim,proj_type,name):
    assert dim == inputs.get_shape().as_list()[1]
    if proj_type == "linear":
       theta = tf.Variable(tf.random_normal([ dim, num_proj],name = name+'.linear_proj'))
       proj = tf.matmul(inputs, theta)
    elif proj_type == "circular":
       theta = tf.Variable(tf.random_normal([ dim, num_proj],name = name+'.circular_proj'))
       theta_ = tf.expand_dims(theta,0)
       inputs_ = tf.tile(tf.expand_dims(inputs,-1),[1,1,num_proj])
       proj = tf.sqrt(tf.reduce_sum((inputs_ - theta_ )**2,axis=1))
    elif proj_type == "poly":
       degree = 3
       dim_poly = homopoly(dim,degree)
       theta = tf.Variable(tf.random_normal([ dim_poly, num_proj],name = name+'.circular_proj'))
       proj = poly(inputs,theta,degree)
    else:  
       assert 0 == 1

    return proj

def SWD(inputs,dim,num_proj,proj_type="linear"):
    i_sh = inputs.get_shape().as_list()
    name = 'Discriminator.GSWD'
    with tf.name_scope(name) as scope:
        proj = projection(inputs,num_proj,dim,proj_type,name)
        output, scale = SWDBlock('Discriminator.idt0', proj)

    return output, scale


def discriminator(real_data,fake_data,num_feat,num_proj,proj_type,LAMBDA1,LAMBDA2,args,name='d_net'):
    r_feat,_ = discriminator_net(real_data,num_feat,args,name=name)
    f_feat,_ = discriminator_net(fake_data,num_feat,args,name=name)


    disc_real = tf.reduce_mean(SWD(r_feat,num_feat,num_proj,proj_type)[0],[1])
    disc_fake = tf.reduce_mean(SWD(f_feat,num_feat,num_proj,proj_type)[0],[1])

    gen_cost = tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
    d = tf.reduce_mean(disc_real) 
   
    alpha1 = tf.slice(tf.random_uniform(tf.shape(r_feat),minval=0.,maxval=1.),[0,0],[-1,1])
    inter_feat = (1 - alpha1) * r_feat + alpha1 * f_feat
    lrelu, scale = SWD(inter_feat,num_feat,num_proj,proj_type)

    #directly compute the derivate of swd block
    grad_feat = tf.multiply(scale, 0.1*(tf.abs(lrelu) - lrelu) + 0.5*(tf.abs(lrelu) + lrelu))

    pen_feat = tf.reduce_mean(tf.square(grad_feat - 0.001))
    #disc_cost += LAMBDA1 * (gradient_penalty) + LAMBDA2 * pen_feat 
    disc_cost += LAMBDA2 * pen_feat

    return disc_cost, gen_cost, d


class GCNPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space,disease_dim, args,kind='small', atom_type_num = None):
        self.sn = name[-1]
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, disease_dim, kind, atom_type_num,args)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, disease_dim, kind, atom_type_num,args):
        self.pdtype = MultiCatCategoricalPdType
        ### 0 Get input
        ob = {'adj': U.get_placeholder(name="adj"+self.sn, dtype=tf.float32, shape=[None,ob_space['adj'].shape[0],None,None]),
              'node': U.get_placeholder(name="node"+self.sn, dtype=tf.float32, shape=[None,1,None,ob_space['node'].shape[2]])}

        disease = U.get_placeholder(name='disease', dtype=tf.float32, shape=[None,disease_dim]) # disease
        # only when evaluating given action, at training time
        self.ac_real = U.get_placeholder(name='ac_real'+self.sn, dtype=tf.int64, shape=[None,4]) # feed groudtruth action
        ob_node = tf.layers.dense(ob['node'],8,activation=None,use_bias=False,name='emb') # embedding layer
        if args.bn==1:
            ob_node = tf.layers.batch_normalization(ob_node,axis=-1)
        if args.has_concat==1:
            emb_node = tf.concat((GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate),ob_node),axis=-1)
        else:
            emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        if args.bn == 1:
            emb_node = tf.layers.batch_normalization(emb_node, axis=-1)
        for i in range(args.layer_num_g-2):
            if args.has_residual==1:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)+self.emb_node1
            elif args.has_concat==1:
                emb_node = tf.concat((GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate),self.emb_node1),axis=-1)
            else:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_' + str(i + 1),aggregate=args.gcn_aggregate)
            if args.bn == 1:
                emb_node = tf.layers.batch_normalization(emb_node, axis=-1)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        emb_node = tf.squeeze(emb_node,axis=1)  # B*n*f

        ### 1 only keep effective nodes
        # ob_mask = tf.cast(tf.transpose(tf.reduce_sum(ob['node'],axis=-1),[0,2,1]),dtype=tf.bool) # B*n*1
        ob_len = tf.reduce_sum(tf.squeeze(tf.cast(tf.cast(tf.reduce_sum(ob['node'], axis=-1),dtype=tf.bool),dtype=tf.float32),axis=-2),axis=-1)  # B
        ob_len_first = ob_len-atom_type_num
        logits_mask = tf.sequence_mask(ob_len, maxlen=tf.shape(ob['node'])[2]) # mask all valid entry
        logits_first_mask = tf.sequence_mask(ob_len_first,maxlen=tf.shape(ob['node'])[2]) # mask valid entry -3 (rm isolated nodes)

        if args.mask_null==1:
            emb_node_null = tf.zeros(tf.shape(emb_node))
            emb_node = tf.where(condition=tf.tile(tf.expand_dims(logits_mask,axis=-1),(1,1,emb_node.get_shape()[-1])), x=emb_node, y=emb_node_null)

        ## get graph embedding
        emb_graph = tf.reduce_sum(emb_node, axis=1, keepdims=True)
        if args.graph_emb == 1:
            emb_graph = tf.tile(emb_graph, [1, tf.shape(emb_node)[1], 1])
            emb_node = tf.concat([emb_node, emb_graph], axis=2)

        ### 2 predict stop
        emb_stop = tf.layers.dense(emb_node, args.emb_size, activation=tf.nn.relu, use_bias=False, name='linear_stop1')
        if args.bn==1:
            emb_stop = tf.layers.batch_normalization(emb_stop,axis=-1)
        self.logits_stop = tf.reduce_sum(emb_stop,axis=1)
        emb_disease_stop = tf.layers.dense(disease, args.emb_size, activation=tf.nn.relu, use_bias=False, name='linear_stop1_disease')
        self.logits_stop = tf.concat([emb_disease_stop,self.logits_stop],axis=1)
        self.logits_stop = tf.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2_1')  # B*2
        # explicitly show node num
        # self.logits_stop = tf.concat((tf.reduce_mean(tf.layers.dense(emb_node, 32, activation=tf.nn.relu, name='linear_stop1'),axis=1),tf.reshape(ob_len_first/5,[-1,1])),axis=1)
        # self.logits_stop = tf.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2')  # B*2

        stop_shift = tf.constant([[0,args.stop_shift]],dtype=tf.float32)
        pd_stop = CategoricalPdType(-1).pdfromflat(flat=self.logits_stop+stop_shift)
        ac_stop = pd_stop.sample()

        ### 3.1: select first (active) node
        # rules: only select effective nodes
        self.logits_first = tf.layers.dense(emb_node, args.emb_size, activation=tf.nn.relu, name='linear_select1')
        disease_tile = tf.tile(tf.expand_dims(disease,axis=1),[1,tf.shape(self.logits_first)[1],1])
        disease_tile = tf.layers.dense(disease_tile, args.emb_size, activation=tf.nn.relu, name='linear_disease_select')
        self.logits_first = tf.concat([disease_tile,self.logits_first],axis=2)
        self.logits_first = tf.squeeze(tf.layers.dense(self.logits_first, 1, activation=None, name='linear_select2'),axis=-1) # B*n
        logits_first_null = tf.ones(tf.shape(self.logits_first))*-1000
        self.logits_first = tf.where(condition=logits_first_mask,x=self.logits_first,y=logits_first_null)
        # using own prediction
        pd_first = CategoricalPdType(-1).pdfromflat(flat=self.logits_first)
        ac_first = pd_first.sample()
        mask = tf.one_hot(ac_first, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_first = tf.boolean_mask(emb_node, mask)
        emb_first = tf.expand_dims(emb_first,axis=1)
        # using groud truth action
        ac_first_real = self.ac_real[:, 0]
        mask_real = tf.one_hot(ac_first_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_first_real = tf.boolean_mask(emb_node, mask_real)
        emb_first_real = tf.expand_dims(emb_first_real, axis=1)

        ### 3.2: select second node
        # rules: do not select first node
        # using own prediction

        # mlp
        emb_cat = tf.concat([tf.tile(emb_first,[1,tf.shape(emb_node)[1],1]),emb_node],axis=2)
        self.logits_second = tf.layers.dense(emb_cat, args.emb_size, activation=tf.nn.relu, name='logits_second1')
        self.logits_second = tf.concat([disease_tile,self.logits_second],axis=2)
        self.logits_second = tf.layers.dense(self.logits_second, 1, activation=None, name='logits_second2')
        # # bilinear
        # self.logits_second = tf.transpose(bilinear(emb_first, emb_node, name='logits_second'), [0, 2, 1])

        self.logits_second = tf.squeeze(self.logits_second, axis=-1)
        ac_first_mask = tf.one_hot(ac_first, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=False, off_value=True)
        logits_second_mask = tf.logical_and(logits_mask,ac_first_mask)
        logits_second_null = tf.ones(tf.shape(self.logits_second)) * -1000
        self.logits_second = tf.where(condition=logits_second_mask, x=self.logits_second, y=logits_second_null)
        
        pd_second = CategoricalPdType(-1).pdfromflat(flat=self.logits_second)
        ac_second = pd_second.sample()
        mask = tf.one_hot(ac_second, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_second = tf.boolean_mask(emb_node, mask)
        emb_second = tf.expand_dims(emb_second, axis=1)

        # using groudtruth
        # mlp
        emb_cat = tf.concat([tf.tile(emb_first_real, [1, tf.shape(emb_node)[1], 1]), emb_node], axis=2)
        self.logits_second_real = tf.layers.dense(emb_cat, args.emb_size, activation=tf.nn.relu, name='logits_second1',reuse=True)
        self.logits_second_real = tf.concat([disease_tile,self.logits_second_real],axis=2)
        self.logits_second_real = tf.layers.dense(self.logits_second_real, 1, activation=None, name='logits_second2',reuse=True)
        # # bilinear
        # self.logits_second_real = tf.transpose(bilinear(emb_first_real, emb_node, name='logits_second'), [0, 2, 1])

        self.logits_second_real = tf.squeeze(self.logits_second_real, axis=-1)
        ac_first_mask_real = tf.one_hot(ac_first_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=False, off_value=True)
        logits_second_mask_real = tf.logical_and(logits_mask,ac_first_mask_real)
        self.logits_second_real = tf.where(condition=logits_second_mask_real, x=self.logits_second_real, y=logits_second_null)

        ac_second_real = self.ac_real[:,1]
        mask_real = tf.one_hot(ac_second_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_second_real = tf.boolean_mask(emb_node, mask_real)
        emb_second_real = tf.expand_dims(emb_second_real, axis=1)

        ### 3.3 predict edge type
        # using own prediction
        # MLP
        emb_cat = tf.concat([emb_first,emb_second],axis=-1)
        self.logits_edge = tf.layers.dense(emb_cat, args.emb_size, activation=tf.nn.relu, name='logits_edge1')
        disease_tile2 = tf.tile(tf.expand_dims(disease,axis=1),[1,tf.shape(self.logits_edge)[1],1])
        self.logits_edge = tf.concat([disease_tile2,self.logits_edge],axis=2)
        self.logits_edge = tf.layers.dense(self.logits_edge, ob['adj'].get_shape()[1], activation=None, name='logits_edge2')
        self.logits_edge = tf.squeeze(self.logits_edge,axis=1)
        # # bilinear
        # self.logits_edge = tf.reshape(bilinear_multi(emb_first,emb_second,out_dim=ob['adj'].get_shape()[1]),[-1,ob['adj'].get_shape()[1]])
        pd_edge = CategoricalPdType(-1).pdfromflat(self.logits_edge)
        ac_edge = pd_edge.sample()

        # using ground truth
        # MLP
        emb_cat = tf.concat([emb_first_real, emb_second_real], axis=-1)
        self.logits_edge_real = tf.layers.dense(emb_cat, args.emb_size, activation=tf.nn.relu, name='logits_edge1', reuse=True)
        self.logits_edge_real = tf.concat([disease_tile2,self.logits_edge_real],axis=2)
        self.logits_edge_real = tf.layers.dense(self.logits_edge_real, ob['adj'].get_shape()[1], activation=None,
                                           name='logits_edge2', reuse=True)
        self.logits_edge_real = tf.squeeze(self.logits_edge_real, axis=1)
        # # bilinear
        # self.logits_edge_real = tf.reshape(bilinear_multi(emb_first_real, emb_second_real, out_dim=ob['adj'].get_shape()[1]),
        #                               [-1, ob['adj'].get_shape()[1]])


        # ncat_list = [tf.shape(logits_first),ob_space['adj'].shape[-1],ob_space['adj'].shape[0]]
        self.pd = self.pdtype(-1).pdfromflat([self.logits_first,self.logits_second_real,self.logits_edge_real,self.logits_stop])
        self.vpred = tf.layers.dense(emb_node, args.emb_size, use_bias=False, activation=tf.nn.relu, name='value1')
        if args.bn==1:
            self.vpred = tf.layers.batch_normalization(self.vpred,axis=-1)
        self.vpred = tf.reduce_max(self.vpred,axis=1)
        self.vpred = tf.layers.dense(self.vpred, 1, activation=None, name='value2')

        self.state_in = []
        self.state_out = []

        self.ac = tf.concat((tf.expand_dims(ac_first,axis=1),tf.expand_dims(ac_second,axis=1),tf.expand_dims(ac_edge,axis=1),tf.expand_dims(ac_stop,axis=1)),axis=1)


        debug = {}
        debug['ob_node'] = tf.shape(ob['node'])
        debug['ob_adj'] = tf.shape(ob['adj'])
        debug['emb_node'] = emb_node
        debug['logits_stop'] = self.logits_stop
        debug['logits_second'] = self.logits_second
        debug['ob_len'] = ob_len
        debug['logits_first_mask'] = logits_first_mask
        debug['logits_second_mask'] = logits_second_mask
        # debug['pd'] = self.pd.logp(self.ac)
        debug['ac'] = self.ac

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self._act = U.function([stochastic, ob['adj'], ob['node'],disease], [self.ac, self.vpred, debug]) # add debug in second arg if needed

    def act(self, stochastic, ob,disease):
        return self._act(stochastic, ob['adj'][None], ob['node'][None],disease)
        # return self._act(stochastic, ob['adj'], ob['node'])

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []








def GCN_emb(ob,args):
    ob_node = tf.layers.dense(ob['node'], 8, activation=None, use_bias=False, name='emb')  # embedding layer
    if args.has_concat == 1:
        emb_node1 = tf.concat(
            (GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1', aggregate=args.gcn_aggregate), ob_node),
            axis=-1)
    else:
        emb_node1 = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1', aggregate=args.gcn_aggregate)
    for i in range(args.layer_num_g - 2):
        if args.has_residual == 1:
            emb_node1 = GCN_batch(ob['adj'], emb_node1, args.emb_size, name='gcn1_' + str(i + 1),
                                       aggregate=args.gcn_aggregate) + emb_node1
        elif args.has_concat == 1:
            emb_node1 = tf.concat((GCN_batch(ob['adj'], emb_node1, args.emb_size,
                                                  name='gcn1_' + str(i + 1), aggregate=args.gcn_aggregate),
                                        emb_node1), axis=-1)
        else:
            emb_node1 = GCN_batch(ob['adj'], emb_node1, args.emb_size, name='gcn1_' + str(i + 1),
                                       aggregate=args.gcn_aggregate)
    emb_node2 = GCN_batch(ob['adj'], emb_node1, args.emb_size, is_act=False, is_normalize=True,
                               name='gcn2', aggregate=args.gcn_aggregate)
    emb_node = tf.squeeze(emb_node2, axis=1)  # B*n*f
    emb_graph = tf.reduce_max(emb_node, axis=1, keepdims=True)
    if args.graph_emb == 1:
        emb_graph = tf.tile(emb_graph, [1, tf.shape(emb_node)[1], 1])
        emb_node = tf.concat([emb_node, emb_graph], axis=2)
    return emb_node


#### debug

if __name__ == "__main__":
    adj_np = np.ones((5,3,4,4))
    adj = tf.placeholder(shape=(5,3,4,4),dtype=tf.float32)
    node_feature_np = np.ones((5,1,4,3))
    node_feature = tf.placeholder(shape=(5,1,4,3),dtype=tf.float32)


    ob_space = {}
    atom_type = 5
    ob_space['adj'] = gym.Space(shape=[3,5,5])
    ob_space['node'] = gym.Space(shape=[1,5,atom_type])
    ac_space = gym.spaces.MultiDiscrete([10, 10, 3])
    policy = GCNPolicy(name='policy',ob_space=ob_space,ac_space=ac_space)

    stochastic = True
    env = gym.make('molecule-v0')  # in gym format
    env.init()
    ob = env.reset()

    # ob['adj'] = np.repeat(ob['adj'][None],2,axis=0)
    # ob['node'] = np.repeat(ob['node'][None],2,axis=0)

    print('adj',ob['adj'].shape)
    print('node',ob['node'].shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20):
            ob = env.reset()
            for j in range(0,20):
                ac,vpred,debug = policy.act(stochastic,ob)
                # if ac[0]==ac[1]:
                #     print('error')
                # else:
                # print('i',i,'ac',ac,'vpred',vpred,'debug',debug['logits_first'].shape,debug['logits_second'].shape)
                print('i', i)
                # print('ac\n',ac)
                # print('debug\n',debug['ob_len'])
                ob,reward,_,_ = env.step(ac)

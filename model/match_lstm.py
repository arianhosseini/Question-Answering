import theano
from theano import tensor
import numpy
import scipy

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, GatedRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.monitoring import aggregation
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant

from utils import init_embedding_table

def make_bidir_lstm_stack(seq, seq_dim, mask, sizes, skip=True, name=''):
    bricks = []

    curr_dim = [seq_dim]
    curr_hidden = [seq]

    hidden_list = []
    for k, dim in enumerate(sizes):
        fwd_lstm_ins = [Linear(input_dim=d, output_dim=4*dim, name='%s_fwd_lstm_in_%d_%d'%(name,k,l)) for l, d in enumerate(curr_dim)]
        fwd_lstm = LSTM(dim=dim, activation=Tanh(), name='%s_fwd_lstm_%d'%(name,k))

        bwd_lstm_ins = [Linear(input_dim=d, output_dim=4*dim, name='%s_bwd_lstm_in_%d_%d'%(name,k,l)) for l, d in enumerate(curr_dim)]
        bwd_lstm = LSTM(dim=dim, activation=Tanh(), name='%s_bwd_lstm_%d'%(name,k))

        bricks = bricks + [fwd_lstm, bwd_lstm] + fwd_lstm_ins + bwd_lstm_ins

        fwd_tmp = sum(x.apply(v) for x, v in zip(fwd_lstm_ins, curr_hidden))
        bwd_tmp = sum(x.apply(v) for x, v in zip(bwd_lstm_ins, curr_hidden))
        fwd_hidden, _ = fwd_lstm.apply(fwd_tmp, mask=mask)
        bwd_hidden, _ = bwd_lstm.apply(bwd_tmp[::-1], mask=mask[::-1])
        hidden_list = hidden_list + [fwd_hidden, bwd_hidden]
        if skip:
            curr_hidden = [seq, fwd_hidden, bwd_hidden[::-1]]
            curr_dim = [seq_dim, dim, dim]
        else:
            curr_hidden = [fwd_hidden, bwd_hidden[::-1]]
            curr_dim = [dim, dim]

    return bricks, hidden_list

class Model():
    def __init__(self, config, vocab_size):
        question = tensor.imatrix('question')
        question_mask = tensor.imatrix('question_mask')
        context = tensor.imatrix('context')
        context_mask = tensor.imatrix('context_mask')
        answer = tensor.imatrix('answer')
        answer_mask = tensor.imatrix('answer_mask')
        ans_indices = tensor.imatrix('ans_indices') # n_steps * n_samples
        ans_indices_mask = tensor.imatrix('ans_indices_mask')

        bricks = []

        question = question.dimshuffle(1, 0)
        question_mask = question_mask.dimshuffle(1, 0)
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)
        answer = answer.dimshuffle(1, 0)
        answer_mask = answer_mask.dimshuffle(1, 0)
        ans_indices = ans_indices.dimshuffle(1, 0)
        ans_indices_mask = ans_indices_mask.dimshuffle(1, 0)

        # Embed questions and context
        embed = LookupTable(vocab_size, config.embed_size, name='embed')
        #embed.weights_init = IsotropicGaussian(0.01)
        embed.weights_init = Constant(init_embedding_table(filename='embeddings/vocab_embeddings.txt'))

        # one directional LSTM encoding
        q_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4*config.pre_lstm_size, name='q_lstm_in')
        q_lstm = LSTM(dim=config.pre_lstm_size, activation=Tanh(), name='q_lstm')
        c_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4*config.pre_lstm_size, name='c_lstm_in')
        c_lstm = LSTM(dim=config.pre_lstm_size, activation=Tanh(), name='c_lstm')
        bricks += [q_lstm, c_lstm, q_lstm_ins, c_lstm_ins]

        q_tmp = q_lstm_ins.apply(embed.apply(question))
        c_tmp = c_lstm_ins.apply(embed.apply(context))
        q_hidden, _ = q_lstm.apply(q_tmp, mask=question_mask.astype(theano.config.floatX)) # lq, bs, dim
        c_hidden, _ = c_lstm.apply(c_tmp, mask=context_mask.astype(theano.config.floatX)) # lc, bs, dim

        # Attention mechanism Bilinear question
        attention_question = Linear(input_dim=config.pre_lstm_size, output_dim=config.pre_lstm_size, name='att_question')
        bricks += [attention_question]
        att_weights_question = q_hidden[None, :, :, :] * attention_question.apply(c_hidden.reshape((c_hidden.shape[0]*c_hidden.shape[1], c_hidden.shape[2]))).reshape((c_hidden.shape[0], c_hidden.shape[1], c_hidden.shape[2]))[:, None, :, :] # --> lc,lq,bs,dim
        att_weights_question = att_weights_question.sum(axis=3) # sum over axis 3 -> dimensions --> lc,lq,bs
        att_weights_question = att_weights_question.dimshuffle(0, 2, 1) # --> lc,bs,lq
        att_weights_question = att_weights_question.reshape((att_weights_question.shape[0]*att_weights_question.shape[1], att_weights_question.shape[2])) # --> lc*bs,lq
        att_weights_question = tensor.nnet.softmax(att_weights_question) # softmax over axis 1 -> length of question # --> lc*bs,lq
        att_weights_question = att_weights_question.reshape((c_hidden.shape[0], q_hidden.shape[1], q_hidden.shape[0])) # --> lc,bs,lq
        att_weights_question = att_weights_question.dimshuffle(0, 2, 1) # --> lc,lq,bs

        attended_question = tensor.sum(q_hidden[None, :, :, :] * att_weights_question[:, :, :, None], axis=1) # sum over axis 1 -> length of question --> lc,bs,dim
        attended_question.name = 'attended_question'

         # Match LSTM
        cqembed = tensor.concatenate([c_hidden, attended_question], axis=2)
        mlstms, mhidden_list = make_bidir_lstm_stack(cqembed, 2 * config.pre_lstm_size, context_mask.astype(theano.config.floatX),
                                                     config.match_lstm_size, config.match_skip_connections, 'match')
        bricks = bricks + mlstms
        if config.match_skip_connections:
            menc_dim = 2*sum(config.match_lstm_size)
            menc = tensor.concatenate(mhidden_list, axis=2)
        else:
            menc_dim = 2*config.match_lstm_size[-1]
            menc = tensor.concatenate(mhidden_list[-2:], axis=2)
        menc.name = 'menc'

        # Attention mechanism MLP start
        attention_mlp_start = MLP(dims=config.attention_mlp_hidden + [1],
                            activations=config.attention_mlp_activations[1:] + [Identity()],
                            name='attention_mlp_start')
        attention_clinear_start = Linear(input_dim=menc_dim, output_dim=config.attention_mlp_hidden[0], name='attm_start') # Wym
        bricks += [attention_mlp_start, attention_clinear_start]
        layer1_start = Tanh(name='layer1_start')
        layer1_start = layer1_start.apply(attention_clinear_start.apply(menc.reshape((menc.shape[0]*menc.shape[1], menc.shape[2])))
                                        .reshape((menc.shape[0],menc.shape[1],config.attention_mlp_hidden[0])))
        att_weights_start = attention_mlp_start.apply(layer1_start.reshape((layer1_start.shape[0]*layer1_start.shape[1], layer1_start.shape[2])))
        att_weights_start = att_weights_start.reshape((layer1_start.shape[0], layer1_start.shape[1]))
        att_weights_start = tensor.nnet.softmax(att_weights_start.T).T

        attended = tensor.sum(menc * att_weights_start[:, :, None], axis=0)
        attended.name = 'attended'

        # Attention mechanism MLP end
        attention_mlp_end = MLP(dims=config.attention_mlp_hidden + [1],
                            activations=config.attention_mlp_activations[1:] + [Identity()],
                            name='attention_mlp_end')
        attention_qlinear_end = Linear(input_dim=menc_dim, output_dim=config.attention_mlp_hidden[0], name='atts_end') #Wum
        attention_clinear_end = Linear(input_dim=menc_dim, output_dim=config.attention_mlp_hidden[0], use_bias=False, name='attm_end') # Wym
        bricks += [attention_mlp_end, attention_qlinear_end, attention_clinear_end]
        layer1_end = Tanh(name='layer1_end')
        layer1_end = layer1_end.apply(attention_clinear_end.apply(menc.reshape((menc.shape[0]*menc.shape[1], menc.shape[2])))
                                        .reshape((menc.shape[0],menc.shape[1],config.attention_mlp_hidden[0]))
                             + attention_qlinear_end.apply(attended)[None, :, :])
        att_weights_end = attention_mlp_end.apply(layer1_end.reshape((layer1_end.shape[0]*layer1_end.shape[1], layer1_end.shape[2])))
        att_weights_end = att_weights_end.reshape((layer1_end.shape[0], layer1_end.shape[1]))
        att_weights_end = tensor.nnet.softmax(att_weights_end.T).T

        att_weights_start = tensor.dot(tensor.le(tensor.tile(theano.tensor.arange(context.shape[0])[None,:], (context.shape[0], 1)),
                                         tensor.tile(theano.tensor.arange(context.shape[0])[:,None], (1, context.shape[0]))), att_weights_start)
        att_weights_end = tensor.dot(tensor.ge(tensor.tile(theano.tensor.arange(context.shape[0])[None,:], (context.shape[0], 1)),
                                       tensor.tile(theano.tensor.arange(context.shape[0])[:,None], (1, context.shape[0]))), att_weights_end)

        # add attention from left and right
        att_weights = att_weights_start * att_weights_end
        #att_weights = tensor.minimum(att_weights_start, att_weights_end)

        att_target = tensor.zeros((ans_indices.shape[1], context.shape[0]),dtype=theano.config.floatX)
        att_target = tensor.set_subtensor(att_target[tensor.arange(ans_indices.shape[1]),ans_indices],1)
        att_target = att_target.dimshuffle(1, 0)
        #att_target = tensor.eq(tensor.tile(answer[None,:,:], (context.shape[0], 1, 1)),
        #                       tensor.tile(context[:,None,:], (1, answer.shape[0], 1))).sum(axis=1).clip(0,1)
        
        self.predictions = tensor.gt(att_weights, 0.25) * context

        att_target = att_target / (att_target.sum(axis=0) + 0.00001)
        #att_weights = att_weights / (att_weights.sum(axis=0) + 0.00001)

        cost = (tensor.nnet.binary_crossentropy(att_weights, att_target) * context_mask).sum() / context_mask.sum()

        # Apply dropout
        cg = ComputationGraph([cost])
        if config.w_noise > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise)
        if config.dropout > 0:
            cg = apply_dropout(cg, mhidden_list, config.dropout)
        [cost_reg] = cg.outputs

        # Other stuff
        cost.name = 'cost'
        cost_reg.name = 'cost_reg'
        att_weights_start.name = 'att_weights_start'
        att_weights_end.name = 'att_weights_end'
        att_weights.name = 'att_weights'
        att_target.name = 'att_target'
        self.predictions.name = 'pred'

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost_reg]]
        self.monitor_vars_valid = [[cost_reg]]
        self.analyse_vars= [cost, self.predictions, att_weights_start, att_weights_end, att_weights, att_target]

        # Initialize bricks
        embed.initialize()
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

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

        bricks = []

        question = question.dimshuffle(1, 0)
        question_mask = question_mask.dimshuffle(1, 0)
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)
        answer = answer.dimshuffle(1, 0)
        answer_mask = answer_mask.dimshuffle(1, 0)

        # Embed questions and context
        embed = LookupTable(vocab_size, config.embed_size, name='question_embed')
        embed.weights_init = IsotropicGaussian(0.01)

        # Calculate question encoding (concatenate layer1)
        qembed = embed.apply(question)
        qlstms, qhidden_list = make_bidir_lstm_stack(qembed, config.embed_size, question_mask.astype(theano.config.floatX),
                                                     config.question_lstm_size, config.question_skip_connections, 'q')
        bricks = bricks + qlstms
        if config.question_skip_connections:
            qenc_dim = 2*sum(config.question_lstm_size)
            qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list], axis=1)
        else:
            qenc_dim = 2*config.question_lstm_size[-1]
            qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list[-2:]], axis=1)
        qenc.name = 'qenc'

        # Calculate context encoding (concatenate layer1)
        cembed = embed.apply(context)
        cqembed = tensor.concatenate([cembed, tensor.extra_ops.repeat(qenc[None, :, :], cembed.shape[0], axis=0)], axis=2)
        clstms, chidden_list = make_bidir_lstm_stack(cqembed, config.embed_size + qenc_dim, context_mask.astype(theano.config.floatX),
                                                     config.ctx_lstm_size, config.ctx_skip_connections, 'ctx')
        bricks = bricks + clstms
        if config.ctx_skip_connections:
            cenc_dim = 2*sum(config.ctx_lstm_size) #2 : fw & bw
            cenc = tensor.concatenate(chidden_list, axis=2)
        else:
            cenc_dim = 2*config.question_lstm_size[-1]
            cenc = tensor.concatenate(chidden_list[-2:], axis=2)
        cenc.name = 'cenc'

        # Attention mechanism MLP start
        attention_mlp_start = MLP(dims=config.attention_mlp_hidden + [1],
                            activations=config.attention_mlp_activations[1:] + [Identity()],
                            name='attention_mlp_start')
        attention_qlinear_start = Linear(input_dim=qenc_dim, output_dim=config.attention_mlp_hidden[0], name='attq_start') #Wum
        attention_clinear_start = Linear(input_dim=cenc_dim, output_dim=config.attention_mlp_hidden[0], use_bias=False, name='attc_start') # Wym
        bricks += [attention_mlp_start, attention_qlinear_start, attention_clinear_start]
        layer1_start = Tanh(name='layer1_start')
        layer1_start = layer1_start.apply(attention_clinear_start.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2])))
                                        .reshape((cenc.shape[0],cenc.shape[1],config.attention_mlp_hidden[0]))
                             + attention_qlinear_start.apply(qenc)[None, :, :])
        att_weights_start = attention_mlp_start.apply(layer1_start.reshape((layer1_start.shape[0]*layer1_start.shape[1], layer1_start.shape[2])))
        att_weights_start = att_weights_start.reshape((layer1_start.shape[0], layer1_start.shape[1]))
        att_weights_start = tensor.nnet.softmax(att_weights_start.T).T

        attended = tensor.sum(cenc * att_weights_start[:, :, None], axis=0)
        attended.name = 'attended'

        # Attention mechanism MLP end
        attention_mlp_end = MLP(dims=config.attention_mlp_hidden + [1],
                            activations=config.attention_mlp_activations[1:] + [Identity()],
                            name='attention_mlp_end')
        attention_qlinear_end = Linear(input_dim=qenc_dim, output_dim=config.attention_mlp_hidden[0], name='attq_end') #Wum
        attention_clinear_end = Linear(input_dim=cenc_dim, output_dim=config.attention_mlp_hidden[0], use_bias=False, name='attc_end') # Wym
        bricks += [attention_mlp_end, attention_qlinear_end, attention_clinear_end]
        layer1_end = Tanh(name='layer1_end')
        layer1_end = layer1_end.apply(attention_clinear_end.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2])))
                                        .reshape((cenc.shape[0],cenc.shape[1],config.attention_mlp_hidden[0]))
                             + attention_qlinear_end.apply(attended)[None, :, :])
        att_weights_end = attention_mlp_end.apply(layer1_end.reshape((layer1_end.shape[0]*layer1_end.shape[1], layer1_end.shape[2])))
        att_weights_end = att_weights_end.reshape((layer1_end.shape[0], layer1_end.shape[1]))
        att_weights_end = tensor.nnet.softmax(att_weights_end.T).T

        att_weights_start = tensor.dot(tensor.le(tensor.tile(theano.tensor.arange(context.shape[0])[None,:], (context.shape[0], 1)),
                                         tensor.tile(theano.tensor.arange(context.shape[0])[:,None], (1, context.shape[0]))), att_weights_start)
        att_weights_end = tensor.dot(tensor.ge(tensor.tile(theano.tensor.arange(context.shape[0])[None,:], (context.shape[0], 1)),
                                       tensor.tile(theano.tensor.arange(context.shape[0])[:,None], (1, context.shape[0]))), att_weights_end)

        # add attention from left and right
        #att_weights = att_weights_start * att_weights_end
        att_weights = tensor.minimum(att_weights_start, att_weights_end)

        att_target = tensor.eq(tensor.tile(answer[None,:,:], (context.shape[0], 1, 1)),
                               tensor.tile(context[:,None,:], (1, answer.shape[0], 1))).sum(axis=1).clip(0,1)

        self.predictions = tensor.gt(att_weights, 0.5) * context

        att_target = att_target / (att_target.sum(axis=0) + 0.00001)
        att_weights = att_weights / (att_weights.sum(axis=0) + 0.00001)

        cost = (tensor.nnet.binary_crossentropy(att_weights, att_target) * context_mask).sum() / context_mask.sum()

        # Apply dropout
        cg = ComputationGraph([cost])
        if config.w_noise > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise)
        if config.dropout > 0:
            cg = apply_dropout(cg, qhidden_list + chidden_list, config.dropout)
        [cost_reg] = cg.outputs

        # Other stuff
        cost.name = 'cost'
        cost_reg.name = 'cost_reg'
        att_weights_start.name = 'att_weights_start'
        att_weights_end.name = 'att_weights_end'
        att_target.name = 'att_target'
        att_weights.name = 'att_weights'
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

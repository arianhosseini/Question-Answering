import theano
from theano import tensor
import numpy

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
from blocks.bricks.base import application

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

def to_bag(m, vocab_size):
    """
        converting m[length,batch_size] to ret[batch_size, vocab_size] multiple hot
        min 0 max 1 ( no cliping needed )
    """

    ret = tensor.zeros((m.shape[1], vocab_size),dtype=theano.config.floatX)
    ret = tensor.set_subtensor(ret[tensor.arange(m.shape[1]),m],1)
    return ret

class Model():
    def __init__(self, config, vocab_size):
        question = tensor.imatrix('question')
        question_mask = tensor.imatrix('question_mask')
        answer = tensor.imatrix('answer')
        answer_mask = tensor.imatrix('answer_mask')
        better = tensor.imatrix('better')
        better_mask = tensor.imatrix('better_mask')
        worse = tensor.imatrix('worse')
        worse_mask = tensor.imatrix('worse_mask')
        b_left = tensor.imatrix('b_left')
        b_left_mask = tensor.imatrix('b_left_mask')
        b_right = tensor.imatrix('b_right')
        b_right_mask = tensor.imatrix('b_right_mask')
        w_left = tensor.imatrix('w_left')
        w_left_mask = tensor.imatrix('w_left_mask')
        w_right = tensor.imatrix('w_right')
        w_right_mask = tensor.imatrix('w_right_mask')


        bricks = []

        question = question.dimshuffle(1, 0)
        question_mask = question_mask.dimshuffle(1, 0)

        better = better.dimshuffle(1, 0)
        better_mask = better_mask.dimshuffle(1, 0)

        worse = worse.dimshuffle(1, 0)
        worse_mask = worse_mask.dimshuffle(1, 0)

        b_left = b_left.dimshuffle(1, 0)
        b_left_mask = b_left_mask.dimshuffle(1, 0)

        b_right = b_right.dimshuffle(1, 0)
        b_right_mask = b_right_mask.dimshuffle(1, 0)

        w_left = w_left.dimshuffle(1, 0)
        w_left_mask = w_left_mask.dimshuffle(1, 0)

        w_right = w_right.dimshuffle(1, 0)
        w_right_mask = w_right_mask.dimshuffle(1, 0)

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

        candidates_hidden_list = []

        candidate_fwd_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4*config.ctx_lstm_size[0], name='candidate_fwd_lstm_in_0_0')
        candidate_fwd_lstm = LSTM(dim=config.ctx_lstm_size[0], activation=Tanh(), name='candidate_fwd_lstm_0')

        candidate_bwd_lstm_ins = Linear(input_dim=config.embed_size, output_dim=4*config.ctx_lstm_size[0], name='candidate_bwd_lstm_in_0_0')
        candidate_bwd_lstm = LSTM(dim=config.ctx_lstm_size[0], activation=Tanh(), name='candidate_bwd_lstm_0')

        #adding encoding bricks for initialization
        bricks = bricks + [candidate_fwd_lstm, candidate_bwd_lstm, candidate_fwd_lstm_ins, candidate_bwd_lstm_ins]

        #computing better encoding
        better_embed = embed.apply(better)
        better_fwd_tmp = candidate_fwd_lstm_ins.apply(better_embed)
        better_bwd_tmp = candidate_bwd_lstm_ins.apply(better_embed)
        better_fwd_hidden, _ = candidate_fwd_lstm.apply(better_fwd_tmp, mask=better_mask.astype(theano.config.floatX))
        better_bwd_hidden, _ = candidate_bwd_lstm.apply(better_bwd_tmp[::-1], mask=better_mask.astype(theano.config.floatX)[::-1])
        better_hidden_list = [better_fwd_hidden, better_bwd_hidden]
        better_enc_dim = 2*sum(config.ctx_lstm_size)
        better_enc = tensor.concatenate([h[-1,:,:] for h in better_hidden_list], axis=1) #concating last state of fwd and bwd LSTMs 2*dim * batch_size
        better_enc.name = 'better_enc'
        candidates_hidden_list = candidates_hidden_list + [better_fwd_hidden, better_bwd_hidden]

        #computing worse encoding
        worse_embed = embed.apply(worse)
        worse_fwd_tmp = candidate_fwd_lstm_ins.apply(worse_embed)
        worse_bwd_tmp = candidate_bwd_lstm_ins.apply(worse_embed)
        worse_fwd_hidden, _ = candidate_fwd_lstm.apply(worse_fwd_tmp, mask=worse_mask.astype(theano.config.floatX))
        worse_bwd_hidden, _ = candidate_bwd_lstm.apply(worse_bwd_tmp[::-1], mask=worse_mask.astype(theano.config.floatX)[::-1])
        worse_hidden_list = [worse_fwd_hidden, worse_bwd_hidden]
        worse_enc_dim = 2*sum(config.ctx_lstm_size)
        worse_enc = tensor.concatenate([h[-1,:,:] for h in worse_hidden_list], axis=1)
        worse_enc.name = 'worse_enc'
        candidates_hidden_list = candidates_hidden_list + [worse_fwd_hidden, worse_bwd_hidden]

        # F1 prediction MLP
        prediction_mlp = MLP(dims=config.prediction_mlp_hidden + [1],
                             activations=config.prediction_mlp_activations[1:] + [Identity()],
                             name='prediction_mlp')

        prediction_qlinear = Linear(input_dim=qenc_dim, output_dim=config.prediction_mlp_hidden[0], name='preq')
        prediction_cand_linear = Linear(input_dim=worse_enc_dim, output_dim=config.prediction_mlp_hidden[0], use_bias=False, name='precand')

        bricks += [prediction_mlp, prediction_qlinear, prediction_cand_linear]
        better_layer1 = Tanh('tan1').apply(prediction_cand_linear.apply(better_enc)+prediction_qlinear.apply(qenc))
        better_layer1.name = 'better_layer1'

        worse_layer1 = Tanh('tan2').apply(prediction_cand_linear.apply(worse_enc)+prediction_qlinear.apply(qenc))
        worse_layer1.name = 'worse_layer1'

        better_pred_weights = Rectifier('rec1').apply(prediction_mlp.apply(better_layer1)) #batch_size
        worse_pred_weights = Rectifier('rec2').apply(prediction_mlp.apply(worse_layer1)) #batch_size

        #cost : max(0,- score-better + score-worse + margin)
        margin = 0.01

        conditions = tensor.lt(better_pred_weights, worse_pred_weights + margin).astype(theano.config.floatX)
        self.predictions = conditions
        cost = (-better_pred_weights + worse_pred_weights + margin) * conditions
        cost = cost.mean()

        # Apply dropout
        cg = ComputationGraph([cost])

        if config.w_noise > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise)
        if config.dropout > 0:
            cg = apply_dropout(cg, qhidden_list + candidates_hidden_list, config.dropout)
        [cost_reg] = cg.outputs

        # Other stuff
        cost.name = 'cost'
        cost_reg.name = 'cost_reg'

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost_reg]]
        self.monitor_vars_valid = [[cost_reg]]

        # Initialize bricks
        embed.initialize()
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

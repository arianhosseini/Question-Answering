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
        clstms, chidden_list = make_bidir_lstm_stack(cembed, config.embed_size, context_mask.astype(theano.config.floatX),
                                                     config.ctx_lstm_size, config.ctx_skip_connections, 'ctx')
        bricks = bricks + clstms
        if config.ctx_skip_connections:
            cenc_dim = 2*sum(config.ctx_lstm_size) #2 : fw & bw
            cenc = tensor.concatenate(chidden_list, axis=2)
        else:
            cenc_dim = 2*config.question_lstm_size[-1]
            cenc = tensor.concatenate(chidden_list[-2:], axis=2)
        cenc.name = 'cenc'

        # Attention mechanism Bilinear
        attention_clinear = Linear(input_dim=cenc_dim, output_dim=qenc_dim, name='attc')
        bricks += [attention_clinear]
        att_weights = qenc[None, :, :] * attention_clinear.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2]))).reshape((cenc.shape[0], cenc.shape[1], cenc.shape[2]))
        att_weights = att_weights.sum(axis=2)
        att_weights = tensor.nnet.softmax(att_weights.T).T
        att_weights.name = 'att_weights'

        attended = tensor.sum(cenc * tensor.nnet.softmax(att_weights.T).T[:, :, None], axis=0)
        attended.name = 'attended'

        answer_bag = to_bag(answer, vocab_size)
        answer_bag = tensor.set_subtensor(answer_bag[:,0:3],0)
        relevant_items = answer_bag.sum(axis=1, dtype=theano.config.floatX)

        def createSequences(j, index, c_enc, c_enc_dim, c_context, c_window_size):
            sequence= tensor.concatenate([c_context[j:j+index,:],
                                          tensor.zeros((c_window_size-index, c_context.shape[1]))], axis=0)
            enc = tensor.concatenate([c_enc[j+index-1, :, :c_enc_dim/2],
                                      c_enc[j, :, c_enc_dim/2:-1],
                                      tensor.tile(c_window_size[None,None], (c_enc.shape[1], 1))], axis=1)
            return enc, sequence

        def createTargetValues(j, index, c_context, c_vocab_size):
            sequence_bag = to_bag(c_context[j:j+index,:], c_vocab_size)
            sequence_bag = tensor.set_subtensor(sequence_bag[:,0:3],0)
            selected_items = sequence_bag.sum(axis=1, dtype=theano.config.floatX)
            tp = (sequence_bag * answer_bag).sum(axis=1, dtype=theano.config.floatX)
            precision = tp / (selected_items + 0.00001)
            recall = tp / (relevant_items + 0.00001)
            #precision = tensor.set_subtensor(precision[tensor.isnan(precision)], 0.0)
            #recall = tensor.set_subtensor(recall[tensor.isnan(recall)], 1.0)
            macroF1 = (2 * ( precision * recall )) / (precision + recall + 0.00001)
            #macroF1 = tensor.set_subtensor(macroF1[tensor.isnan(macroF1)], 0.0)
            return macroF1

        window_size = 3
        senc = []
        sequences = []
        pred_targets = []
        for i in range(1,window_size+1):
            (all_enc,all_sequence),_ = theano.scan(fn=createSequences,sequences=tensor.arange(cenc.shape[0]-i+1),non_sequences=[i, cenc, cenc_dim, context, window_size])
            (all_macroF1),_ = theano.scan(fn=createTargetValues,sequences=tensor.arange(cenc.shape[0]-i+1),non_sequences=[i, context, vocab_size])
            senc.append(all_enc)
            sequences.append(all_sequence)
            pred_targets.append(all_macroF1)

        senc = tensor.concatenate(senc, axis=0)
        sequences = tensor.concatenate(sequences, axis=0)
        pred_targets = tensor.concatenate(pred_targets, axis=0)

        # F1 prediction MLP
        prediction_mlp = MLP(dims=config.prediction_mlp_hidden + [1],
                    activations=config.prediction_mlp_activations[1:] + [Identity()],
                    name='prediction_mlp')
        prediction_qlinear = Linear(input_dim=qenc_dim, output_dim=config.prediction_mlp_hidden[0], name='preq')
        prediction_clinear = Linear(input_dim=cenc_dim, output_dim=config.prediction_mlp_hidden[0], use_bias=False, name='prec')
        prediction_slinear = Linear(input_dim=cenc_dim, output_dim=config.prediction_mlp_hidden[0], use_bias=False, name='pres')
        bricks += [prediction_mlp, prediction_qlinear, prediction_clinear, prediction_slinear]
        layer1 = Tanh().apply(prediction_clinear.apply(attended)[None, :, :]
                             +prediction_qlinear.apply(qenc)[None, :, :]
                             +prediction_slinear.apply(senc.reshape((senc.shape[0]*senc.shape[1], senc.shape[2])))
                                        .reshape((senc.shape[0],senc.shape[1],config.prediction_mlp_hidden[0])))
        layer1.name = 'layer1'
        pred_weights = prediction_mlp.apply(layer1.reshape((layer1.shape[0]*layer1.shape[1], layer1.shape[2])))
        pred_weights = pred_weights.reshape((layer1.shape[0], layer1.shape[1]))
        pred_weights = tensor.nnet.sigmoid(pred_weights.T).T
        
        pred_targets = pred_targets / (pred_targets.sum(axis=0) + 0.00001)
        pred_weights = pred_weights / (pred_weights.sum(axis=0) + 0.00001)
    
        #numpy.set_printoptions(edgeitems=500)
        #pred_targets = theano.printing.Print('pred_targets')(pred_targets)
        #pred_weights = theano.printing.Print('pred_weights')(pred_weights)

        cost = tensor.nnet.binary_crossentropy(pred_weights, pred_targets).mean()
        self.predictions = sequences[pred_weights.argmax(axis=0),:,tensor.arange(sequences.shape[2])].T

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

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost_reg]]
        self.monitor_vars_valid = [[cost_reg]]

        # Initialize bricks
        embed.initialize()
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

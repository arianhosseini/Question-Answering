import theano
from theano import tensor
import numpy
import scipy
from collections import OrderedDict

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
from blocks.roles import add_role, WEIGHT, BIAS


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)


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

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def rand_weight(ndim, ddim, lo, hi):
    randn = numpy.random.rand(ndim, ddim)
    randn = randn * (hi - lo) + lo
    return randn.astype(theano.config.floatX)

def init_params(data_dim, lstm_dim):
    params = OrderedDict()
    W = numpy.concatenate([rand_weight(data_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(data_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(data_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(data_dim, lstm_dim, -0.08, 0.08)], axis=1)
    params['lstm_de_W'] = W
    U = numpy.concatenate([rand_weight(lstm_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(lstm_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(lstm_dim, lstm_dim, -0.08, 0.08),
                           rand_weight(lstm_dim, lstm_dim, -0.08, 0.08)], axis=1)
    params['lstm_de_U'] = U
    b = numpy.zeros((4 * lstm_dim,))
    params['lstm_de_b'] = b.astype(theano.config.floatX)

    params['lstm_hterm'] = rand_weight(lstm_dim, 1, -0.08, 0.08)[:, 0]

    # ptr parameters
    params['ptr_W1'] = rand_weight(lstm_dim, lstm_dim, -0.08, 0.08)
    params['ptr_W2'] = rand_weight(lstm_dim, lstm_dim, -0.08, 0.08)
    b1 = numpy.zeros((1,lstm_dim))
    params['ptr_b1'] = b1.astype(theano.config.floatX)

    params['ptr_v'] = rand_weight(lstm_dim, 1, -0.08, 0.08)[:, 0]

    return params

def ptr_network(tparams, cqembed, context_mask, ans_indices, ans_indices_mask, decoder_lstm_output_dim, cenc):
    #cqembed: length * batch_size * (embed+2*lstm_size)
    n_sizes = cqembed.shape[0] #context length
    n_samples = cqembed.shape[1] if cqembed.ndim == 3 else 1 #batch_size
    n_steps = ans_indices.shape[0] #answer length

    assert context_mask is not None
    assert ans_indices_mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[n * dim:(n + 1) * dim]

    def softmax(m_, x_):
        maxes = tensor.max(x_, axis=0, keepdims=True)
        e = tensor.exp(x_ - maxes)
        dist = e / tensor.sum(e * m_, axis=0)
        return dist

    def _lstm(input_mask, input_embedding, h_, c_, prefix='lstm_en'):
        preact = tensor.dot(input_embedding, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        preact += tensor.dot(h_, tparams[_p(prefix, 'U')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, decoder_lstm_output_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, decoder_lstm_output_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, decoder_lstm_output_dim))
        c = tensor.tanh(_slice(preact, 3, decoder_lstm_output_dim))

        c = f * c_ + i * c
        c = input_mask[:, None] * c + (1. - input_mask)[:, None] * c_
        h = o * tensor.tanh(c)
        h = input_mask[:, None] * h + (1. - input_mask)[:, None] * h_

        return h, c

    def prediction_ptr_probs(prior_word_index, decoded_old, c_, cenc, hiddens_mask): #decoded_old Initialized with cenc[-1]
        prior_word_embedding = cqembed[prior_word_index, tensor.arange(n_samples), :] # batch_size * (embed+2*lstm_size)
        decoded, c = _lstm(tensor.ones(shape=(5),dtype=theano.config.floatX), prior_word_embedding, decoded_old, c_, 'lstm_de') # batch_size * decoder_lstm_output_dim
        attention_weights = tensor.dot(cenc, tparams['ptr_W1']) + tensor.dot(decoded, tparams['ptr_W2']) #(context)length * batch_size * (decoder_lstm_output_dim=lstm_size*2)
        attention_weights = tensor.tanh(attention_weights)  # length * batch_size * lstm_size*2
        attention_weights = tensor.dot(attention_weights, tparams['ptr_v'])  # length * batch_size
        prob = softmax(hiddens_mask, attention_weights)
        prediction_index = prob.argmax(axis=0) #batch_size
        return prediction_index, decoded, c


    def _ptr_probs(ans_indice_mask, ans_indice, decoded_old, c_, _, cenc, hiddens_mask): #decoded_old Initialized with cenc[-1]
        pred_cembed = cqembed[ans_indice, tensor.arange(n_samples), :]  # batch_size * (embed+2*lstm_size)
        decoded, c = _lstm(ans_indice_mask, pred_cembed, decoded_old, c_, 'lstm_de') # batch_size * decoder_lstm_output_dim
        attention_weights = tensor.dot(cenc, tparams['ptr_W1']) + tensor.dot(decoded, tparams['ptr_W2']) #(context)length * batch_size * (decoder_lstm_output_dim=lstm_size*2)
        attention_weights = tensor.tanh(attention_weights)  # length * batch_size * lstm_size*2
        attention_weights = tensor.dot(attention_weights, tparams['ptr_v'])  # length * batch_size
        prob = softmax(hiddens_mask, attention_weights)
        return decoded, c, prob

    def prediction_ptr_probs_ver1(prior_word_index, prob, decoded_old, c_, cenc, hiddens_mask): #decoded_old Initialized with cenc[-1]
        # prior_word_embedding = cqembed[prior_word_index, tensor.arange(n_samples), :] # batch_size * (embed+2*lstm_size)


        attention_weights = tensor.dot(cenc, tparams['ptr_W1']) + (tensor.dot(decoded_old, tparams['ptr_W2'])  +tensor.repeat(tparams['ptr_b1'],decoded_old.shape[0],axis=0)) #(context)length * batch_size * (decoder_lstm_output_dim=lstm_size*2)
        attention_weights = tensor.tanh(attention_weights)  # length * batch_size * lstm_size*2
        attention_weights = tensor.dot(attention_weights, tparams['ptr_v'])  # length * batch_size
        prob = softmax(hiddens_mask, attention_weights)

        probs = prob
        probs = probs.reshape((probs.shape[0],probs.shape[1],1))
        probs = tensor.repeat(probs, cenc.shape[2], axis=2) #lc,bs,dim
        attended_passage = (probs * cenc).sum(axis=0) #bs,dim

        decoded, c = _lstm(tensor.ones(shape=(5),dtype=theano.config.floatX), attended_passage, decoded_old, c_, 'lstm_de') # batch_size * decoder_lstm_output_dim
        prediction_index = prob.argmax(axis=0) #batch_size
        return prediction_index, prob, decoded, c

    def _ptr_probs_ver1(ans_indice_mask, ans_indice, decoded_old, c_, prob_old, cenc, hiddens_mask): #decoded_old Initialized with cenc[-1]
        # pred_cembed = cqembed[ans_indice, tensor.arange(n_samples), :]  # batch_size * (embed+2*lstm_size)

        attention_weights = tensor.dot(cenc, tparams['ptr_W1']) + (tensor.dot(decoded_old, tparams['ptr_W2'])+ tensor.repeat(tparams['ptr_b1'],decoded_old.shape[0],axis=0)) #(context)length * batch_size * (decoder_lstm_output_dim=lstm_size*2)
        attention_weights = tensor.tanh(attention_weights)  # length * batch_size * lstm_size*2
        attention_weights = tensor.dot(attention_weights, tparams['ptr_v'])  # length * batch_size
        prob = softmax(hiddens_mask, attention_weights)

        probs = prob
        probs = probs.reshape((probs.shape[0],probs.shape[1],1))
        probs = tensor.repeat(probs, cenc.shape[2], axis=2) #lc,bs,dim
        attended_passage = (probs * cenc).sum(axis=0) #bs,dim

        decoded, c = _lstm(ans_indice_mask, attended_passage, decoded_old, c_, 'lstm_de') # batch_size * decoder_lstm_output_dim

        return decoded, c, prob


    # decoding
    hiddens_mask = tensor.set_subtensor(context_mask[0, :], tensor.constant(1, dtype=theano.config.floatX))
    gen_steps = 5
    gen_vals , _ = theano.scan(prediction_ptr_probs_ver1,
                                  sequences=None,
                                  outputs_info=[tensor.alloc(numpy.int64(0), n_samples), #context_length * batch_size
                                                tensor.alloc(numpy_floatX(0.), gen_steps, n_samples),
                                                cenc[-1],
                                                tensor.alloc(numpy_floatX(0.), n_samples, decoder_lstm_output_dim)],  #decoded embeddings (d in paper) cells[-1], #batch_size * (decoder_lstm_output_dim=2*lstm_size)
                                  non_sequences=[cenc, hiddens_mask],
                                  name="generating",
                                  n_steps=gen_steps)

    rval, _ = theano.scan(_ptr_probs_ver1,
                          sequences=[ans_indices_mask, ans_indices],
                          outputs_info=[cenc[-1],  # batch_size * (dim_proj=decoder_lstm_output_dim=2*lstm_size) init value for decoded step i-1
                                        tensor.alloc(numpy_floatX(0.), n_samples, decoder_lstm_output_dim),  #decoded embeddings (d in paper) cells[-1], #batch_size * (decoder_lstm_output_dim=2*lstm_size)
                                        tensor.alloc(numpy_floatX(0.), n_sizes, n_samples)], #context_length * batch_size
                          non_sequences=[cenc, hiddens_mask],
                          name='decoding',
                          n_steps=n_steps)
    preds = rval[2] #length * batch_size
    generations = gen_vals[0]
    return preds, generations



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
        embed.weights_init = IsotropicGaussian(0.01)
        # embed.weights_init = Constant(init_embedding_table(filename='embeddings/vocab_embeddings.txt'))

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


        #pointer networks decoder LSTM and Attention parameters
        params = init_params(data_dim=config.decoder_data_dim, lstm_dim=config.decoder_lstm_output_dim)
        tparams = init_tparams(params)

        self.theano_params = []
        add_role(tparams['lstm_de_W'],WEIGHT)
        add_role(tparams['lstm_de_U'],WEIGHT)
        add_role(tparams['lstm_de_b'],BIAS)
        add_role(tparams['ptr_b1'],BIAS)
        add_role(tparams['ptr_v'],WEIGHT)
        add_role(tparams['ptr_W1'],WEIGHT)
        add_role(tparams['ptr_W2'],WEIGHT)
        self.theano_params = tparams.values()

        #n_steps = length , n_samples = batch_size
        n_steps = ans_indices.shape[0]
        n_samples = ans_indices.shape[1]
        preds, generations = ptr_network(tparams,
                            cqembed,
                            context_mask.astype(theano.config.floatX),
                            ans_indices,
                            ans_indices_mask.astype(theano.config.floatX),
                            config.decoder_lstm_output_dim,
                            menc)

        self.generations = generations

        idx_steps = tensor.outer(tensor.arange(n_steps, dtype='int64'), tensor.ones((n_samples,), dtype='int64'))
        idx_samples = tensor.outer(tensor.ones((n_steps,), dtype='int64'), tensor.arange(n_samples, dtype='int64'))
        probs = preds[idx_steps, ans_indices, idx_samples]
        # probs *= y_mask
        off = 1e-8
        if probs.dtype == 'float16':
            off = 1e-6
        # probs += (1 - y_mask)  # change unmasked position to 1, since log(1) = 0
        probs += off
        # probs_printed = theano.printing.Print('this is probs')(probs)
        cost = -tensor.log(probs)
        cost *= ans_indices_mask
        cost = cost.sum(axis=0) / ans_indices_mask.sum(axis=0)
        cost = cost.mean()


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

        # self.predictions.name = 'pred'

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost_reg]]
        self.monitor_vars_valid = [[cost_reg]]
        # self.analyse_vars= [cost, self.predictions, att_weights_start, att_weights_end, att_weights, att_target]

        # Initialize bricks
        embed.initialize()
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

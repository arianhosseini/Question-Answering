import theano
from theano import tensor
import numpy
from utils import init_embedding_table

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

class MaskedSoftmaxEmitter(SoftmaxEmitter):
    def __init__(self, context_bag, **kwargs):
        super(MaskedSoftmaxEmitter, self).__init__(**kwargs)
        self.context_bag = context_bag

    @application
    def probs(self, readouts):
        readouts = tensor.switch(self.context_bag, readouts, -1000 * tensor.ones_like(readouts))
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)

    @application
    def cost(self, readouts, outputs):
        #readouts = tensor.switch(self.context_bag, readouts, -1000 * tensor.ones_like(readouts))
        return self.softmax.categorical_cross_entropy(outputs, readouts, extra_ndim=readouts.ndim - 2)


class Model():
    def __init__(self, config, vocab_size):
        question = tensor.imatrix('question')
        question_mask = tensor.imatrix('question_mask')
        context = tensor.imatrix('context')
        context_mask = tensor.imatrix('context_mask')
        answer = tensor.imatrix('answer')
        answer_mask = tensor.imatrix('answer_mask')

        context_bag = tensor.eq(context[:,:,None],tensor.arange(vocab_size)).sum(axis = 1).clip(0,1)

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
        #embeddings_initial_value = init_embedding_table(filename='embeddings/vocab_embeddings.txt')
        #embed.weights_init = Constant(embeddings_initial_value)

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

        # Build the encoder bricks
        transition = GatedRecurrent(
            activation=Tanh(),
            dim=config.generator_lstm_size,
            name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            attended_dim=cenc_dim, match_dim=config.generator_lstm_size, name="attention")
        readout = Readout(
            readout_dim=vocab_size,
            source_names=[transition.apply.states[0], attention.take_glimpses.outputs[0]],
            emitter=MaskedSoftmaxEmitter(context_bag=context_bag, name='emitter'),
            feedback_brick=LookupFeedback(vocab_size, config.feedback_size),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        cost = generator.cost(
            answer, answer_mask.astype(theano.config.floatX),
            attended=cenc,
            attended_mask=context_mask.astype(theano.config.floatX),
            name="cost")
        self.predictions = generator.generate(
            n_steps=7, batch_size=config.batch_size,
            attended=cenc,
            attended_mask=context_mask.astype(theano.config.floatX),
            iterate=True)[1]

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

        # initialize new stuff manually (change!)
        generator.weights_init = IsotropicGaussian(0.01)
        generator.biases_init = Constant(0)
        generator.push_allocation_config()
        generator.push_initialization_config()
        transition.weights_init = Orthogonal()
        generator.initialize()

        # Initialize bricks
        embed.initialize()
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

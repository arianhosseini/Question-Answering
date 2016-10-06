import theano
from theano import tensor
import numpy

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,SequenceGenerator)
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise

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
        answer = tensor.ivector('answer')
        candidates = tensor.imatrix('candidates')
        candidates_mask = tensor.imatrix('candidates_mask')

        bricks = []

        question = question.dimshuffle(1, 0)
        question_mask = question_mask.dimshuffle(1, 0)
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)

        # Embed questions and cntext
        embed = LookupTable(vocab_size, config.embed_size, name='question_embed')
        bricks.append(embed)

        qembed = embed.apply(question)
        cembed = embed.apply(context)

        qlstms, qhidden_list = make_bidir_lstm_stack(qembed, config.embed_size, question_mask.astype(theano.config.floatX),
                                                     config.question_lstm_size, config.question_skip_connections, 'q')
        clstms, chidden_list = make_bidir_lstm_stack(cembed, config.embed_size, context_mask.astype(theano.config.floatX),
                                                     config.ctx_lstm_size, config.ctx_skip_connections, 'ctx')
        bricks = bricks + qlstms + clstms

        # Calculate question encoding (concatenate layer1)
        if config.question_skip_connections:
            qenc_dim = 2*sum(config.question_lstm_size)
            qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list], axis=1)
        else:
            qenc_dim = 2*config.question_lstm_size[-1]
            qenc = tensor.concatenate([h[-1,:,:] for h in qhidden_list[-2:]], axis=1)
        qenc.name = 'qenc'

        # Calculate context encoding (concatenate layer1)
        if config.ctx_skip_connections: #default yes
            cenc_dim = 2*sum(config.ctx_lstm_size) #2 : fw & bw
            cenc = tensor.concatenate(chidden_list, axis=2)
        else:
            cenc_dim = 2*config.ctx_lstm_size[-1]
            cenc = tensor.concatenate(chidden_list[-2:], axis=2)
        cenc.name = 'cenc'

        # Attention mechanism MLP           activation: Tanh, identity
        attention_mlp = MLP(dims=config.attention_mlp_hidden + [1],
                            activations=config.attention_mlp_activations[1:] + [Identity()],
                            name='attention_mlp')
        attention_qlinear = Linear(input_dim=qenc_dim, output_dim=config.attention_mlp_hidden[0], name='attq') #Wum
        attention_clinear = Linear(input_dim=cenc_dim, output_dim=config.attention_mlp_hidden[0], use_bias=False, name='attc') # Wym
        bricks += [attention_mlp, attention_qlinear, attention_clinear]
        layer1 = Tanh().apply(attention_clinear.apply(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2])))
                                        .reshape((cenc.shape[0],cenc.shape[1],config.attention_mlp_hidden[0]))
                             + attention_qlinear.apply(qenc)[None, :, :])
        layer1.name = 'layer1'
        att_weights = attention_mlp.apply(layer1.reshape((layer1.shape[0]*layer1.shape[1], layer1.shape[2])))
        att_weights.name = 'att_weights_0'
        att_weights = att_weights.reshape((layer1.shape[0], layer1.shape[1]))
        att_weights.name = 'att_weights'

        attended = tensor.sum(cenc * tensor.nnet.softmax(att_weights.T).T[:, :, None], axis=0)
        attended.name = 'attended'

        print("attended shape: %d" %attended.shape)

        dimension = qenc_dim + cenc_dim
        transition = SimpleRecurrent(activation=Tanh(),dim=dimension, name="transition")

        readout = Readout(
            readout_dim=vocab_size,
            source_names=[transition.apply.states[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(vocab_size, dimension),
            name="readout")

        generator = SequenceGenerator(
            readout=readout, transition=transition,
            name="generator")

        self.generator = generator
        bricks += [generator]


        cost = self.generator.cost()




        # Now we can calculate our output
        out_mlp = MLP(dims=[cenc_dim + qenc_dim] + config.out_mlp_hidden + [config.n_entities],
                      activations=config.out_mlp_activations + [Identity()],
                      name='out_mlp')
        bricks += [out_mlp]
        probs = out_mlp.apply(tensor.concatenate([attended, qenc], axis=1))
        probs.name = 'probs'

        is_candidate = tensor.eq(tensor.arange(config.n_entities, dtype='int32')[None, None, :],
                                 tensor.switch(candidates_mask, candidates, -tensor.ones_like(candidates))[:, :, None]).sum(axis=1)
        probs = tensor.switch(is_candidate, probs, -1000 * tensor.ones_like(probs))

        # Calculate prediction, cost and error rate
        pred = probs.argmax(axis=1)
        cost = Softmax().categorical_cross_entropy(answer, probs).mean()
        error_rate = tensor.neq(answer, pred).mean()

        # Apply dropout
        cg = ComputationGraph([cost, error_rate])
        if config.w_noise > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, config.w_noise)
        if config.dropout > 0:
            cg = apply_dropout(cg, qhidden_list + chidden_list, config.dropout)
        [cost_reg, error_rate_reg] = cg.outputs

        # Other stuff
        cost_reg.name = cost.name = 'cost'
        error_rate_reg.name = error_rate.name = 'error_rate'

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost_reg], [error_rate_reg]]
        self.monitor_vars_valid = [[cost], [error_rate]]

        # Initialize bricks
        for brick in bricks:
            brick.weights_init = config.weights_init
            brick.biases_init = config.biases_init
            brick.initialize()

    @application
    def generate():
        pass

    @application
    def cost()
#  vim: set sts=4 ts=4 sw=4 tw=0 et :

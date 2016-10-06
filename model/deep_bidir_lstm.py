import theano
from theano import tensor
import numpy

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise

class Model():
    def __init__(self, config, vocab_size):
        question = tensor.imatrix('question')
        question_mask = tensor.imatrix('question_mask')
        answer = tensor.ivector('answer')
        candidates = tensor.imatrix('candidates')
        candidates_mask = tensor.imatrix('candidates_mask')

        bricks = []


        # set time as first dimension
        question = question.dimshuffle(1, 0)
        question_mask = question_mask.dimshuffle(1, 0)

        # Embed questions
        embed = LookupTable(vocab_size, config.embed_size, name='question_embed')
        bricks.append(embed)
        qembed = embed.apply(question)

        # Create and apply LSTM stack
        curr_dim = [config.embed_size]
        curr_hidden = [qembed]

        hidden_list = []
        for k, dim in enumerate(config.lstm_size):
            fwd_lstm_ins = [Linear(input_dim=d, output_dim=4*dim, name='fwd_lstm_in_%d_%d'%(k,l)) for l, d in enumerate(curr_dim)]
            fwd_lstm = LSTM(dim=dim, activation=Tanh(), name='fwd_lstm_%d'%k)

            bwd_lstm_ins = [Linear(input_dim=d, output_dim=4*dim, name='bwd_lstm_in_%d_%d'%(k,l)) for l, d in enumerate(curr_dim)]
            bwd_lstm = LSTM(dim=dim, activation=Tanh(), name='bwd_lstm_%d'%k)

            bricks = bricks + [fwd_lstm, bwd_lstm] + fwd_lstm_ins + bwd_lstm_ins

            fwd_tmp = sum(x.apply(v) for x, v in zip(fwd_lstm_ins, curr_hidden))
            bwd_tmp = sum(x.apply(v) for x, v in zip(bwd_lstm_ins, curr_hidden))
            fwd_hidden, _ = fwd_lstm.apply(fwd_tmp, mask=question_mask.astype(theano.config.floatX))
            bwd_hidden, _ = bwd_lstm.apply(bwd_tmp[::-1], mask=question_mask.astype(theano.config.floatX)[::-1])
            hidden_list = hidden_list + [fwd_hidden, bwd_hidden]
            if config.skip_connections:
                curr_hidden = [qembed, fwd_hidden, bwd_hidden[::-1]]
                curr_dim = [config.embed_size, dim, dim]
            else:
                curr_hidden = [fwd_hidden, bwd_hidden[::-1]]
                curr_dim = [dim, dim]

        # Create and apply output MLP
        if config.skip_connections:
            out_mlp = MLP(dims=[2*sum(config.lstm_size)] + config.out_mlp_hidden + [config.n_entities],
                          activations=config.out_mlp_activations + [Identity()],
                          name='out_mlp')
            bricks.append(out_mlp)

            probs = out_mlp.apply(tensor.concatenate([h[-1,:,:] for h in hidden_list], axis=1))
        else:
            out_mlp = MLP(dims=[2*config.lstm_size[-1]] + config.out_mlp_hidden + [config.n_entities],
                          activations=config.out_mlp_activations + [Identity()],
                          name='out_mlp')
            bricks.append(out_mlp)

            probs = out_mlp.apply(tensor.concatenate([h[-1,:,:] for h in hidden_list[-2:]], axis=1))

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
            cg = apply_dropout(cg, hidden_list, config.dropout)
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

        

#  vim: set sts=4 ts=4 sw=4 tw=0 et :

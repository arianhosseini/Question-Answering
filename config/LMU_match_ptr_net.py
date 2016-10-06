from blocks.bricks import Tanh
from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

from model.match_ptr_net import Model

add_cnn_data = 0.0

batch_size = 32
sort_batch_count = 20

shuffle_questions = True
shuffle_entities = True

concat_ctx_and_question = False
concat_question_before = False

embed_size = 300

pre_lstm_size = 150
pre_skip_connections = False

match_lstm_size = [150]
match_skip_connections = True

#ptr_net decoder config:
decoder_data_dim =  2*match_lstm_size[0]
decoder_lstm_output_dim = 2*match_lstm_size[0]


attention_mlp_hidden = [150]
attention_mlp_activations = [Tanh()]

step_rule = CompositeRule([RMSProp(decay_rate=0.95, learning_rate=5e-5),
                           BasicMomentum(momentum=0.9)])

dropout = 0.2
w_noise = 0.

valid_freq = 10000
save_freq = 10000
print_freq = 1000

weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.)

transition_weights_init = Orthogonal()

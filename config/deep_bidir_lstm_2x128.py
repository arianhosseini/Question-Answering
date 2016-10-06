from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Tanh

from model.deep_bidir_lstm import Model


batch_size = 32
sort_batch_count = 20

shuffle_questions = True
shuffle_entities = True

concat_ctx_and_question = True
concat_question_before = True		## should not matter for bidirectionnal network

embed_size = 200

lstm_size = [128, 128]
skip_connections = True

n_entities = 550
out_mlp_hidden = []
out_mlp_activations = []

step_rule = CompositeRule([RMSProp(decay_rate=0.95, learning_rate=5e-5),
                           BasicMomentum(momentum=0.9)])

dropout = 0.1
w_noise = 0.05

valid_freq = 1000
save_freq = 1000
print_freq = 100

weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.)

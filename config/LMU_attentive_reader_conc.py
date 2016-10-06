from blocks.bricks import Tanh
from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

from model.attentive_reader_conc import Model

batch_size = 64
sort_batch_count = 20

shuffle_questions = True
shuffle_entities = True

concat_ctx_and_question = True
concat_question_before = False

embed_size = 300

ctx_lstm_size = [256, 256]
ctx_skip_connections = False

generator_lstm_size = 256
feedback_size = 200

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

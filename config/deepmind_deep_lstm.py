from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping
from blocks.initialization import IsotropicGaussian, Constant

from model.deep_lstm import Model


batch_size = 32
sort_batch_count = 20

shuffle_questions = True

concat_ctx_and_question = True
concat_question_before = True

embed_size = 200

lstm_size = [256, 256]
skip_connections = True

out_mlp_hidden = []
out_mlp_activations = []

step_rule = CompositeRule([RMSProp(decay_rate=0.95, learning_rate=1e-4),
                           BasicMomentum(momentum=0.9)])

dropout = 0.1

valid_freq = 1000
save_freq = 1000
print_freq = 100

weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.)

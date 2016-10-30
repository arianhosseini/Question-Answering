#!/usr/bin/env python

import logging
import numpy
import sys
import os
import importlib

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

from lmu_extensions import EvaluateModel

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

import data
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]

    config = importlib.import_module('.%s' % model_name, 'config')

    path = os.path.join(os.getcwd(), "squad_rare/train-v1.0_tokenized.json")
    valid_path = os.path.join(os.getcwd(), "squad_rare/dev-v1.0_tokenized.json")
    vocab_path = os.path.join(os.getcwd(), "squad_rare/vocab.txt")



    if len(sys.argv) == 3:
        logger.info("Using toy data")
        ds , train_stream = data.setup_toy_datastream(config)
        _, valid_stream = data.setup_toy_datastream(config)
        dump_path = os.path.join("model_params", "dummy.pkl")
    else:
        if hasattr(config, 'add_cnn_data') and config.add_cnn_data > 0.0 :
            logger.info("Using SQuAD + CNN data")
            cnn_path = os.path.join(os.getcwd(), "../data/rc-data_unanonymized_rare/cnn/questions/training")
            ds , train_stream = data.setup_cnnsquad_datastream(path, cnn_path, vocab_path, config)
            _, valid_stream = data.setup_squad_datastream(valid_path, vocab_path, config)

        else:
            # Build datastream
            logger.info("Using SQuAD data")
            ds, train_stream = data.setup_squad_ranker_datastream(os.path.join(os.getcwd(),'squad_short/squadnewtrn.txt'),os.path.join(os.getcwd(), 'squad/vocab.txt'),config)
            # ds, train_stream = data.setup_squad_datastream(path, vocab_path, config)
            _, valid_stream = data.setup_squad_ranker_datastream(os.path.join(os.getcwd(),'squad_short/squadnewtrn.txt'),os.path.join(os.getcwd(), 'squad/vocab.txt'),config, 221697)

        dump_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds.vocab_size)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)
    # test_model = Model(m.generations)

    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters,
                                on_unused_sources='ignore')

    extensions = [
            TrainingDataMonitoring(
                [v for l in m.monitor_vars for v in l],
                prefix='train',
                after_epoch=True)
    ]
    if config.save_freq is not None and dump_path is not None:
        extensions += [
            SaveLoadParams(path=dump_path,
                           model=model,
                           before_training=False,
                           after_training=True,
                           after_epoch=True)
        ]
    if valid_stream is not None and config.valid_freq != -1:
        extensions += [
            DataStreamMonitoring(
                [v for l in m.monitor_vars_valid for v in l],
                valid_stream,
                prefix='valid'),
        ]
    if plot_avail:
        plot_channels = [['train_' + v.name for v in lt] + ['valid_' + v.name for v in lv]
                         for lt, lv in zip(m.monitor_vars, m.monitor_vars_valid)]
        extensions += [
            Plot(document='deepmind_qa_'+model_name,
                 channels=plot_channels,
                 # server_url='http://localhost:5006/', # If you need, change this
                 every_n_batches=config.print_freq)
        ]
    extensions += [
            Printing(after_epoch=True),
            # EvaluateModel(path="", model=test_model, data_stream=valid_stream, vocab_size = ds.vocab_size, vocab = ds.vocab, eval_mode='batch', quiet=True, after_epoch=True),
            ProgressBar()
    ]

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    # Run the model !
    main_loop.run()
    main_loop.profile.report()

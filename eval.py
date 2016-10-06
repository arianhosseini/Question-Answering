#!/usr/bin/env python

import logging
import numpy as np
import sys
import os
import importlib
import cPickle

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

import data
from paramsaveload import SaveLoadParams
from lmu_extensions import EvaluateModel

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    eval_mode = 'batch'
    if len(sys.argv) == 3:
        eval_mode = 'all'

    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    dump_path = os.path.join("model_params", model_name+".pkl")
    valid_path = os.path.join(os.getcwd(), "squad_rare/dev-v1.0_tokenized.json")
    vocab_path = os.path.join(os.getcwd(), "squad_rare/vocab.txt")

    ds, valid_stream = data.setup_squad_datastream(valid_path, vocab_path, config)
    snapshot_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds.vocab_size)

    # Build the Blocks stuff for training
    test_model = Model(m.generations)
    model = Model(m.sgd_cost)

    algorithm = None

    extensions = [EvaluateModel(path=snapshot_path, model=test_model, data_stream=valid_stream, vocab_size = ds.vocab_size, vocab = ds.vocab, eval_mode=eval_mode, before_training=True)]

    main_loop = MainLoop(
        model=model,
        data_stream=valid_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')

    # Run the model !
    # main_loop.run()
    # main_loop.profile.report()

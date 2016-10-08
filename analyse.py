#!/usr/bin/env python

import logging
import sys
import glob, os
import importlib
import pdb
import numpy

from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.model import Model
import cPickle

import matplotlib.colors as colors

import data

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

def analyse(model_name):

    config = importlib.import_module('.%s' % model_name, 'config')
    config.batch_size = 1
    config.shuffle_questions = False

    # Build datastream
    valid_path = os.path.join(os.getcwd(), "squad_rare/dev-v1.0_tokenized.json")
    vocab_path = os.path.join(os.getcwd(), "squad_rare/vocab.txt")

    ds, valid_stream = data.setup_squad_datastream(valid_path, vocab_path, config)

    dump_path = os.path.join("model_params", model_name + ".pkl")
    analysis_path = os.path.join("analysis", model_name + ".html")

    # Build model
    m = config.Model(config, ds.vocab_size)
    model = Model(m.sgd_cost)

    if os.path.isfile(dump_path):
        with open(dump_path, 'r') as f:
            print "Analysing %s from best dump" % (model_name)
            model.set_parameter_values(cPickle.load(f))
    else:
        print "Analysing %s with random parameters" % (model_name)

    printed = 0;
    evaluator = DatasetEvaluator(m.analyse_vars)
    f = open(analysis_path, 'w')
    f.write('<html>')
    f.write('<body style="background-color:white">')

    for batch in valid_stream.get_epoch_iterator(as_dict=True):

        if batch["context"].shape[1] > 150:
            continue;

        evaluator.initialize_aggregators()
        evaluator.process_batch(batch)
        analysis_results = evaluator.get_aggregated_values()

        #pdb.set_trace()

        if analysis_results["cost"] > 0:

            f.write('<p>')
            for i in range(0, batch["question"].shape[1]):
                f.write('{0} '.format(ds.vocab[batch["question"][0][i]]))
            f.write('</p>')

            if analysis_results["cost"] > 0:
                foreground = 'red'
            else:
                foreground = 'green'

            f.write('<p style="color:{0}">'.format(foreground))
            for a in batch["answer"][0]:
                f.write('{0} '.format(ds.vocab[a]))
            f.write('</p>')

            for key in analysis_results:

                if "att" in key:

                    analysis_result = analysis_results[key].T

                    lower = 0#numpy.min(analysis_result)
                    upper = 1#numpy.max(analysis_result)
                    if abs(upper - lower) < 0.0001:
                        lower = 0
                        upper = 1
                    f.write('<p>{0}: {1}, {2}</p>'.format(key, lower, upper))

                    f.write('<p>')
                    for i in range(0, analysis_result.shape[1]):
                        att = analysis_result[0][i]
                        att_norm = (att - lower) / (upper - lower)
                        background = (1-att_norm, 1-(0.8*att_norm) ,1-(0.6*att_norm))
                        if att_norm > 0.7:
                            foreground = 'white'
                        else:
                            foreground = 'black'
                        if batch["context"][0][i] in analysis_results["pred"]:# and att > 0.25:
                            foreground = 'red'
                        if batch["context"][0][i] in batch["answer"][0]:
                            foreground = 'green'
                        f.write('<span style="color:{0};background-color:{1}">{2} </span>'.format(foreground, colors.rgb2hex(background), ds.vocab[batch["context"][0][i]]))
                    f.write('</p>')

            f.write('<hr>')

            printed += 1
            if printed >= 20:
                break;

    f.write('</body>')
    f.write('</html>')
    f.close()

if __name__ == "__main__":

    if len(sys.argv) == 1:
        for file in os.listdir("model_params/"):
            if file.endswith("_best.pkl"):
                model_name = file.replace("_best.pkl", "")
                try:
                    analyse(model_name)
                except:
                    print "Error while analysing %s" % (model_name)

    elif len(sys.argv) > 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    else:
        model_name = sys.argv[1]
        analyse(model_name)

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

class RankerEvaluator(SimpleExtension):
    def __init__(self, path, model, data_stream, vocab_size, vocab, eval_mode, quiet=False, iter_num=1, **kwargs):
        super(RankerEvaluator, self).__init__(**kwargs)
        self.path = path
        self.model = model
        self.data_stream = data_stream
        self.iter_num = iter_num
        self.gen_fun = self.model.get_theano_function()
        self.vocab_size = vocab_size
        self.eval_mode = eval_mode
        self.quiet = quiet
        self.vocab = vocab
        self.best_macroF1 = 0.0


    def generate_candidates(self, context):
        pass




    def compute_batch(self, data, batch_num):


        # data.pop('answer_mask', None)
        # data.pop('answer', None)
        data.pop('b_right_mask',None)
        data.pop('b_left_mask',None)
        data.pop('w_right_mask',None)
        data.pop('w_left_mask',None)
        # data.pop('worse',None)
        data.pop('w_right',None)
        data.pop('w_left',None)
        data.pop('b_right',None)
        data.pop('b_left',None)
        data.pop('answer',None)
        data.pop('answer_mask',None)

        temp = self.gen_fun(**data)[0]

        print"temp: "
        print temp
        return temp.shape[0] - temp.sum()

    def do_load(self):
        try:
            with open(self.path, 'r') as f:
                print 'Loading parameters from ' + self.path
                self.model.set_parameter_values(cPickle.load(f))
        except IOError:
            print 'Error in loading!'

    def do(self, which_callback, *args):
        if self.path != "":
            self.do_load()
        epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)

        count = 0
        macroF1 = 0.0
        num_of_examples = 0.0
        precision_sum, recall_sum, exact_sum,f1_sum = 0.0 , 0.0 , 0.0, 0.0
        total_correct_examples = 0.0
        correct_examples = 0
        for data in epoch_iter:
            # data = epoch_iter.next()
            count += 1
            correct_examples = self.compute_batch(data, count)
            print "batch #"+str(count)+" : "+str(correct_examles)
            total_correct_examples += correct_examples

            if self.eval_mode == 'batch':
                break
        print "total correct examples: ", total_correct_examples

class EvaluateModel(SimpleExtension):
    def __init__(self, path, model, data_stream, vocab_size, vocab, eval_mode, quiet=False, iter_num=1, **kwargs):
        super(EvaluateModel, self).__init__(**kwargs)
        self.path = path
        self.model = model
        self.data_stream = data_stream
        self.iter_num = iter_num
        self.gen_fun = self.model.get_theano_function()
        self.vocab_size = vocab_size
        self.eval_mode = eval_mode
        self.quiet = quiet
        self.vocab = vocab
        self.best_macroF1 = 0.0

    def compute_batch(self, data, batch_num):
        answer = data['answer']
        answers = data['answers']

        data.pop('answer_mask', None)
        data.pop('answer', None)
        data.pop('ans_indices_mask', None)
        data.pop('ans_indices', None)
        data.pop('answers', None)

        temp = self.gen_fun(**data)
        # print"temp: "
        predictions_indices = np.asarray(temp).T
        # print predictions_indices

        # ww = raw_input("thiss: ")
        # predictions = temp[0].T
        context = data['context']
        # print context[0]
        predictions = []
        for i,ctx in enumerate(context): #loop over all contexts to get answers
            # print context[i,predictions_indices[i]]
            predictions.append(context[i,predictions_indices[i]])


        # print "predictions:"
        predictions = np.asarray(predictions)
        # print predictions

        if self.quiet==False:
            for i in range(len(predictions)):
                for j in range(len(predictions[i,:])):
                    if self.vocab[predictions[i,j]] == "<EOA>":
                        print "found <EOA>, all zeros"
                        predictions[i,j:] = 0
                        break;
            for i in range(len(predictions)):
                print "answer:",
                for j in range(len(answer[i,:])):
                    if answer[i,j] > 1:
                        print self.vocab[answer[i,j]],
                print ""
                print "predictions: ",
                for j in range(len(predictions[i,:])):
                    if predictions[i,j] > 1:
                        print self.vocab[predictions[i,j]],
                print ""
                print "context: "
                for j in range(len(context[i,:])):
                    if context[i, j] > 1 :
                        print self.vocab[context[i, j]],

                print
                print "answers: "
                for j in answers[i]:
                    for k in j:
                        if k > 1:
                            print self.vocab[k],
                    print


                print ""
                print ""


        context_bag = (context[:,:,None] == np.arange(self.vocab_size)).sum(axis = 1).clip(0,1)

        answer_bag = (answer[:,None] == np.arange(self.vocab_size)[:,None]).sum(axis=2).clip(0,1)
        answer_bag[:,0:3] = 0
        # print answers_bag

        #predictions_bag = (predictions[:,None] == np.arange(self.vocab_size)[:,None]).sum(axis=2).clip(0,1)
        predictions_bag = (predictions[:,None] == np.arange(self.vocab_size)).sum(axis=1).clip(0,1).sum(axis=1)
        # print predictions_bag.sum(axis=1)
        predictions_bag[:,0:4] = 0

        #print "sel: ", selected_items
        selected_items = predictions_bag.sum(axis=1, dtype=float)

        precision = np.zeros(shape=(selected_items.shape[0]),dtype=float)
        recall = np.zeros(shape=(selected_items.shape[0]),dtype=float)
        macroF1 = np.zeros(shape=(selected_items.shape[0]),dtype=float)

        answers_bag = []
        for i,document in enumerate(answers):
            prediction_bag = predictions_bag[i] #pred bag for one sample
            document = np.array(list(document))
            document_answers_bag = (document[:,None] == np.arange(self.vocab_size)[:,None]).sum(axis=2,dtype=float).clip(0,1)
            document_answers_bag[:,0:3] = 0 #multiple answers for one sample

            tps = (document_answers_bag * prediction_bag).sum(axis=1,dtype=float) #tps for multiple answers e.g. [3,0,1]
            relevant_items = document_answers_bag.sum(axis=1,dtype=float) #rel items for multiple answers e.g. [4,3,3]

            precisions = tps / selected_items[i]
            recalls = tps / relevant_items
            precisions[np.isnan(precisions)] = 0;
            recalls[np.isnan(recalls)] = 1;

            macroF1s =  (2 * ( precisions * recalls )) / (precisions + recalls)
            macroF1s[np.isnan(macroF1s)] = 0;
            macroF1[i] = macroF1s.max()

            precision[i] = precisions[np.argmax(macroF1s)]
            recall[i] = recalls[np.argmax(macroF1s)]
            answers_bag.append(document_answers_bag)


        num_of_examples = selected_items.shape[0]
        precision_sum = precision.sum()
        recall_sum = recall.sum()
        f1_sum = macroF1.sum()
        exact = (precision * recall == 1)
        exact_sum = exact.sum()

        avg_precision = precision.mean()
        avg_recall = recall.mean()
        macroF1_of_avg =  (2 * ( avg_precision * avg_recall )) / (avg_precision + avg_recall)

        return (precision_sum, recall_sum, exact_sum, f1_sum, num_of_examples)

    def do_load(self):
        try:
            with open(self.path, 'r') as f:
                print 'Loading parameters from ' + self.path
                self.model.set_parameter_values(cPickle.load(f))
        except IOError:
            print 'Error in loading!'

    def do(self, which_callback, *args):
        if self.path != "":
            self.do_load()
        epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)

        count = 0
        macroF1 = 0.0
        num_of_examples = 0.0
        precision_sum, recall_sum, exact_sum,f1_sum = 0.0 , 0.0 , 0.0, 0.0

        for data in epoch_iter:
            # data = epoch_iter.next()

            # print('batch %d'%count)
            count += 1
            p,r,e,f1,n = self.compute_batch(data, count)
            precision_sum += p
            recall_sum += r
            exact_sum += e
            f1_sum += f1
            num_of_examples += n

            if self.eval_mode == 'batch':
                break

        avg_precision = precision_sum / num_of_examples
        avg_recall = recall_sum / num_of_examples
        avg_exact = exact_sum / num_of_examples
        macroF1 =  (2 * ( avg_precision * avg_recall )) / (avg_precision + avg_recall)
        avg_of_f1s = f1_sum / num_of_examples
        self.best_macroF1 = max(self.best_macroF1, macroF1)
        print('Validation Set:')
        print "         avg_recall: " + str(avg_recall)
        print "         avg_precision: " + str(avg_precision)
        print "         macroF1: " + str(macroF1)
        print "         averageF1: " + str(avg_of_f1s)
        print "         exact match acc: " + str(avg_exact)
        print "         # of examples: " + str(num_of_examples)
        print "         best macroF1: " + str(self.best_macroF1)

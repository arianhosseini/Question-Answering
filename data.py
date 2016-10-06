import logging
import random
import numpy
import pprint
import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os
import json
import itertools

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class CNNSQDataset(Dataset):
    def __init__(self, sq_path, cnn_path, vocab_file, **kwargs):

        self.provides_sources = ('context', 'question', 'answer')

        self.sq_path = sq_path
        self.cnn_path = cnn_path

        self.data = json.load(open(sq_path)) #actual json data
        self.vocab = ['<DUMMY>', '<EOA>', '@placeholder', '<UNK>'] + [ w.strip().split()[0] for w in open(vocab_file) ]

        self.vocab_size = len(self.vocab)
        # print("vocab size: %d"%self.vocab_size)

        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}
        super(CNNSQDataset, self).__init__(**kwargs)


    def to_word_id(self, w):
        ''' word to index'''
        if w in self.reverse_vocab:
            return self.reverse_vocab[w]
        else:

            return self.reverse_vocab['<UNK>']

    def to_word_ids(self, s):
        ''' words to indices '''
        return numpy.array([self.to_word_id(x) for x in s], dtype=numpy.int32)

    def get_data(self, state=None, request=None):
        if request is None or state is not None:
            raise ValueError("Expected a request (name of a question file) and no state.")

        if type(request) is list: #squad request
            request = [int(x) for x in request]
            ctx = self.data['data'][request[0]]['paragraphs'][request[1]]['context'].lower()
            q = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['question'].lower()
            a = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['answers'][0]['text'].strip().lower() + ' <EOA>'

        elif type(request) is str:
            lines = [l.rstrip('\n') for l in open(os.path.join(self.cnn_path, request))]
            ctx = lines[2]
            q = lines[4]
            a = lines[6] + ' <EOA>'

        ctx = self.to_word_ids(ctx.split(' '))
        q = self.to_word_ids(q.split(' '))
        a = self.to_word_ids(a.split(' '))

        if not numpy.all(ctx < self.vocab_size):
            raise ValueError("Context word id out of bounds: %d"%int(ctx.max()))
        if not numpy.all(ctx >= 0):
            raise ValueError("Context word id negative: %d"%int(ctx.min()))
        if not numpy.all(q < self.vocab_size):
            raise ValueError("Question word id out of bounds: %d"%int(q.max()))
        if not numpy.all(q >= 0):
            raise ValueError("Question word id negative: %d"%int(q.min()))

        return (ctx, q, a)


class CNNSQIterator(IterationScheme):
    requests_examples = True
    def __init__(self, sq_path, cnn_path, cnn_ratio=0.0,shuffle=True, **kwargs):
        self.cnn_ratio = cnn_ratio
        self.shuffle = shuffle
        self.squad_iterator = SQuADIterator(sq_path)
        self.cnn_iterator = QAIterator(cnn_path)
        super(CNNSQIterator, self).__init__(**kwargs)

    def get_request_iterator(self):
        self.refs = list(self.squad_iterator.get_request_iterator())
        cnn_refs = list(self.cnn_iterator.get_request_iterator())
        random.shuffle(cnn_refs)
        cnn_to_add = int(self.cnn_ratio * len(self.refs))
        if cnn_to_add > len(cnn_refs):
            print "To many CNN data points requested"
            cnn_to_add = len(cnn_refs)

        self.refs += cnn_refs[:cnn_to_add]

        if self.shuffle:
            print "Shuffling CNN and SQuAD (should occur every epoch)"
            random.shuffle(self.refs)
        return iter_(self.refs)


class SQuADDataset(Dataset):
    def __init__(self, path, vocab_file, **kwargs):

        self.provides_sources = ('context', 'question', 'answer', 'answers', 'ans_indices')
        self.path = path
        self.data = json.load(open(path)) #actual json data
        self.vocab = ['<DUMMY>', '<EOA>', '@placeholder', '<UNK>'] + [ w.strip().split()[0] for w in open(vocab_file) ]

        self.vocab_size = len(self.vocab)
        # print("vocab size: %d"%self.vocab_size)

        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}
        super(SQuADDataset, self).__init__(**kwargs)

    def to_word_id(self, w):
        ''' word to index'''
        if w in self.reverse_vocab:
            return self.reverse_vocab[w]
        else:
            print "out: ", w
            return self.reverse_vocab['<UNK>']

    def to_word_ids(self, s):
        ''' words to indices '''
        return numpy.array([self.to_word_id(x) for x in s], dtype=numpy.int32)

    def get_data(self, state=None, request=None):
        # print request
        # print self.path
        if request is None or state is not None:
            raise ValueError("Expected a request (name of a question file) and no state.")
        request = [int(x) for x in request]

        ctx = '<EOA> ' + self.data['data'][request[0]]['paragraphs'][request[1]]['context'].lower()
        q = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['question'].lower()
        # anslist = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['answers'][0]['text'].strip().lower()
        a = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['answers'][0]['text'].strip().lower() #+ ' <EOA>'

        cand_answers = self.data['data'][request[0]]['paragraphs'][request[1]]['qas'][request[2]]['answers']
        answers = [answer['text']+' <EOA>'  for answer in cand_answers]

        ctx = self.to_word_ids(ctx.split(' '))
        q = self.to_word_ids(q.split(' '))
        a = self.to_word_ids(a.split(' '))
        answers = [self.to_word_ids(answer.split(' ')) for answer in answers ]

        anslist = numpy.asarray([5], dtype=numpy.int32)
        for i in range(len(ctx)):
            if numpy.array_equal(ctx[i:i+len(a)],a):
                anslist = numpy.arange(i,i+len(a),dtype=numpy.int32)
                break

        #print ('---')
        #print ('context: '+ ctx[:200] + ' ...')
        #print ('question: ' + q)
        #print ('answer: ' + a)
        #print ('answer list: ', anslist)

        if not numpy.all(ctx < self.vocab_size):
            raise ValueError("Context word id out of bounds: %d"%int(ctx.max()))
        if not numpy.all(ctx >= 0):
            raise ValueError("Context word id negative: %d"%int(ctx.min()))
        if not numpy.all(q < self.vocab_size):
            raise ValueError("Question word id out of bounds: %d"%int(q.max()))
        if not numpy.all(q >= 0):
            raise ValueError("Question word id negative: %d"%int(q.min()))

        return (ctx, q, a, answers, anslist)

class ToyDataset(Dataset):
    def __init__(self, **kwargs):

        self.provides_sources = ('context', 'question', 'answer', 'answers','ans_indices')
        self.vocab = ['<DUMMY>', '<EOA>', '@placeholder', '<UNK>', 'arian', 'hosseini', 'sascha', 'went', 'to', 'the', 'mall', 'who', 'shopping']

        self.vocab_size = len(self.vocab)
        print("vocab size: %d"%self.vocab_size)
        self.ctx = [['arian', 'hosseini', 'went', 'to', 'the', 'mall'], ['sascha', 'went', 'to', 'the', 'mall']]
        self.q = [['who', 'went'], ['who', 'went', 'shopping']]
        self.a = [['arian', 'hosseini'], ['sascha']]
        self.answers = []
        for i in range(2):
            temp = []
            for j in range(3):
                temp.append(self.a[i])
            self.answers.append(temp)

        print "toy data: "
        print self.ctx
        print
        print self.q
        print
        print self.a
        print
        print self.answers

        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}
        super(ToyDataset, self).__init__(**kwargs)

    def to_word_id(self, w):
        ''' word to index'''
        if w in self.reverse_vocab:
            return self.reverse_vocab[w]
        else:
            print "out: ", w
            return self.reverse_vocab['<UNK>']

    def to_word_ids(self, s):
        ''' words to indices '''
        return numpy.array([self.to_word_id(x) for x in s], dtype=numpy.int32)

    def get_data(self, state=None, request=None):
        #request is int
        if request is None or state is not None:
            raise ValueError("Expected a request (name of a question file) and no state.")

        ctx = self.to_word_ids(self.ctx[request])
        q = self.to_word_ids(self.q[request])
        a = self.to_word_ids(self.a[request])
        # a = self.to_word_ids(self.a[request]+['<EOA>'])
        answers = [self.to_word_ids(answer+['<EOA>']) for answer in self.answers[request]]

        anslist = numpy.asarray([0], dtype=numpy.int32)
        for i in range(len(ctx)):
            if numpy.array_equal(ctx[i:i+len(a)],a):
                anslist = numpy.arange(i,i+len(a),dtype=numpy.int32)
                break
        print ('answer list: ', anslist)

        if not numpy.all(ctx < self.vocab_size):
            raise ValueError("Context word id out of bounds: %d"%int(ctx.max()))
        if not numpy.all(ctx >= 0):
            raise ValueError("Context word id negative: %d"%int(ctx.min()))
        if not numpy.all(q < self.vocab_size):
            raise ValueError("Question word id out of bounds: %d"%int(q.max()))
        if not numpy.all(q >= 0):
            raise ValueError("Question word id negative: %d"%int(q.min()))

        return (ctx, q, a, answers, anslist)

class ToyIterator(IterationScheme):
    requests_examples = True
    def __init__(self, **kwargs):
        super(ToyIterator, self).__init__(**kwargs)

    def get_request_iterator(self):
        return iter_(list(range(2)))

class SQuADIterator(IterationScheme):
    requests_examples = True
    def __init__(self, path, shuffle=True, len_limit=7, **kwargs):
        super(SQuADIterator, self).__init__(**kwargs)
        file_name = path.split('/')[-1].split('.')[0]
        reference_dir = path.split('/')[-2]
        reference_path = reference_dir + '/qa_refrence_'+file_name+'.txt'
        self.reference = []
        if os.path.isfile(reference_path):
            refrence_file = open(reference_path, 'r')
            for line in refrence_file.readlines():
                self.reference.append(line.strip().split(':'))
                # print(self.reference[-1])
        else:
            # print("generating refrence")
            refrence_file = open(reference_path, 'w')
            data = json.load(open(path))
            for i, reading in enumerate(data['data']):
                for j, paragraph in enumerate(reading['paragraphs']):
                    for k, question in enumerate(paragraph['qas']):
                        if len(question['answers'][0]['text'].split(' ')) <= len_limit:
                            #print (question['answers'][0]['text'])
                            refrence_file.write('%d:%d:%d\n'%(i,j,k))
                            self.reference.append([i,j,k])

            refrence_file.close()

        self.path = path
        self.shuffle = shuffle

    def get_request_iterator(self):
        if self.shuffle:
            random.shuffle(self.reference)
        return iter_(self.reference)

class QAIterator(IterationScheme):
    requests_examples = True
    def __init__(self, path, shuffle=False, **kwargs):

        self.path = path
        self.shuffle = shuffle

        super(QAIterator, self).__init__(**kwargs)

    def get_request_iterator(self):
        l = [f for f in os.listdir(self.path)
               if os.path.isfile(os.path.join(self.path, f))]
        if self.shuffle:
            random.shuffle(l)

        return iter_(l)


# -------------- DATASTREAM SETUP --------------------


class ConcatCtxAndQuestion(Transformer):
    produces_examples = True
    def __init__(self, stream, concat_question_before, separator_token=None, **kwargs):
        assert stream.sources == ('context', 'question', 'answer', 'answers')
        self.sources = ('context', 'answer', 'answers')

        self.sep = numpy.array([separator_token] if separator_token is not None else [],
                               dtype=numpy.int32)
        self.concat_question_before = concat_question_before

        super(ConcatCtxAndQuestion, self).__init__(stream, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')

        ctx, q, a, answers = next(self.child_epoch_iterator)

        if self.concat_question_before:
            return (numpy.concatenate([q, self.sep, ctx]), a, answers)
        else:
            return (numpy.concatenate([ctx, self.sep, q]), a, answers)

class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def setup_cnnsquad_datastream(sq_path, cnn_path, vocab_file, config):

    ds = CNNSQDataset(sq_path, cnn_path, vocab_file)
    it = CNNSQIterator(sq_path, cnn_path, cnn_ratio=config.add_cnn_data)

    stream = DataStream(ds, iteration_scheme=it)
    # Sort sets of multiple batches to make batches of similar sizes
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('context'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['context', 'question', 'answer'], mask_dtype='int32')

    return ds, stream


def setup_toy_datastream(config):
    ds = ToyDataset()
    it = ToyIterator()

    stream = DataStream(ds, iteration_scheme=it)
    # Sort sets of multiple batches to make batches of similar sizes
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('context'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['context', 'question', 'answer','ans_indices'], mask_dtype='int32')

    return ds, stream

def setup_squad_datastream(path, vocab_file, config):
    ds = SQuADDataset(path, vocab_file)
    it = SQuADIterator(path)
    stream = DataStream(ds, iteration_scheme=it)

    if config.concat_ctx_and_question:
        stream = ConcatCtxAndQuestion(stream, config.concat_question_before, ds.reverse_vocab['<DUMMY>'])

    # Sort sets of multiple batches to make batches of similar sizes
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('context'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['context', 'question', 'answer', 'ans_indices'], mask_dtype='int32')

    return ds, stream


if __name__ == "__main__":
    # Test
    class DummyConfig:
        def __init__(self):
            self.shuffle_entities = True
            self.shuffle_questions = False
            self.concat_ctx_and_question = False
            self.concat_question_before = False
            self.batch_size = 2
            self.sort_batch_count = 1000


    # ds, stream = setup_datastream(os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training"),
    #                               os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt"),
    #                               DummyConfig())
    # it = stream.get_epoch_iterator()
    #


    #uncomment to test the iterator
    # it = SQuADIterator('squad_rare/dev-v1.0_tokenized.json')
    # for i in it.get_request_iterator():
    #     print(i)

    #Test Toy data
    # ds, stream = setup_toy_datastream(DummyConfig())
    # it = stream.get_epoch_iterator()
    # for i, d in enumerate(stream.get_epoch_iterator()):
    #     print '--'
    #     # print d

    # numpy.set_printoptions(suppress=True)
    ds, stream = setup_squad_datastream('squad_rare/dev-v1.0_tokenized.json', 'squad_rare/vocab.txt', DummyConfig())
    it = stream.get_epoch_iterator()
    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        print d
        if i > 2: break
    # path = 'squad_rare/train-v1.0_tokenized.json'
    # it = SQuADIterator(path)

    # it2 = QAIterator(os.path.join(os.getcwd(),"rc-data/cnn/questions/training"), shuffle=True)
    # for i in it2.get_request_iterator():
    #     print i
    # for i in it.get_request_iterator():
    #     print i



# vim: set sts=4 ts=4 sw=4 tw=0 et :

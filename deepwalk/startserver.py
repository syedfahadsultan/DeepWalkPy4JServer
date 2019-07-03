#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
import collections

import graph
import walks as serialized_walks
# from graph import graph
# from walks import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

from py4j.java_gateway import JavaGateway, CallbackServerParameters, GatewayParameters,GatewayClient
from py4j.java_collections import SetConverter, MapConverter, ListConverter

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

import collections

# global client
# client = GatewayClient()

class D(collections.MutableMapping):
    '''
    Mapping that works like both a dict and a mutable object, i.e.
    d = D(foo='bar')
    and 
    d.foo returns 'bar'
    '''
    # ``__init__`` method required to create instance from class.
    def __init__(self, *args, **kwargs):
        '''Use the object dict'''
        self.__dict__.update(*args, **kwargs)
    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]
    def __delitem__(self, key):
        del self.__dict__[key]
    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        '''returns simple dict representation of the mapping'''
        return str(self.__dict__)
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, D({})'.format(super(D, self).__repr__(), 
                                  self.__dict__)


logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Embeddings(object):

    def __init__(self, args):
      self.args = args

    def getEmbeddings(self, relationships):

      G = graph.load_py4jclient(relationships)

      print("Number of nodes: {}".format(len(G.nodes())))

      num_walks = len(G.nodes()) * self.args.number_walks

      print("Number of walks: {}".format(num_walks))

      data_size = num_walks * self.args.walk_length

      print("Data size (walks*length): {}".format(data_size))

      if data_size < self.args.max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=self.args.number_walks,
                                            path_length=self.args.walk_length, alpha=0, rand=random.Random(self.args.seed))
        print("Training...")
        model = Word2Vec(walks, size=self.args.representation_size, window=self.args.window_size, min_count=0, sg=1, hs=1, workers=self.args.workers)
      else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, self.args.max_memory_data_size))
        print("Walking...")

        walks_filebase = self.args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=self.args.number_walks,
                                             path_length=self.args.walk_length, alpha=0, rand=random.Random(self.args.seed),
                                             num_workers=self.args.workers)

        print("Counting vertex frequency...")
        if not self.args.vertex_freq_degree:
          vertex_counts = serialized_walks.count_textfiles(walk_files, self.args.workers)
        else:
          # use degree distribution for frequency in tree
          vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=self.args.representation_size,
                         window=self.args.window_size, min_count=0, trim_rule=None, workers=self.args.workers)

      # to_return = {}
      # for word, vec in zip(model.wv.vocab, model.wv.vectors):
      #   to_return[word] = " ".join([for str(x) in vec])
      to_return = ""
      for word, vec in zip(model.wv.vocab, model.wv.vectors):
        vector_str = " ".join([str(x) for x in vec])
        to_return = to_return+word+"\t"+vector_str+"\n"

      print(to_return)
      # from py4j.java_collections import SetConverter, MapConverter, ListConverter
      # to_return = MapConverter().convert(to_return, client)
      # to_return = D()
      # for word, vec in zip(model.wv.vocab, model.wv.vectors):
      #   to_return.word = str(vec)

      return to_return

    class Java:
        implements = ["examples.EmbeddingsInterface"]

def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=False,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=False,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  embs = Embeddings(args)

  # gateway = JavaGateway(
  #     gateway_parameters=GatewayParameters(auto_convert=True),
  #     callback_server_parameters=CallbackServerParameters(),
  #     python_server_entry_point=embs)

  gateway = JavaGateway(gateway_parameters=GatewayParameters(),
    callback_server_parameters=CallbackServerParameters(),
    python_server_entry_point=embs)



if __name__ == "__main__":
  sys.exit(main())

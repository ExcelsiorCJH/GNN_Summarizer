from .basic_summarizer import *
from .naive_summarizer import *
from .summarizer import *
from .lstm import LSTM
from .gat import GATClassifier


summarize_models = {
    'NaiveSummarizer': NaiveSummarizer,
    'BasicSummarizer': BasicSummarizer,
    'Summarizer': Summarizer,
}

base_models = {
    'LSTM': LSTM,
    'GATClassifier': GATClassifier,
}
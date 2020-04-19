from .basic_summarizer import *
from .naive_summarizer import *
from .summarizer import *


summarize_models = {
    'NaiveSummarizer': NaiveSummarizer,
    'BasicSummarizer': BasicSummarizer,
    'Summarizer': Summarizer,
}
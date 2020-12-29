import re
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import Lidstone, MLE, Laplace, KneserNeyInterpolated


def train_model(corpus):
    """Create the bigram model.

    :param corpus:  the corpus
    :type corpus: list of str

    :rtype: nltk.lm.models.Lidstone
    """
    corpus = [text.strip().split() for text in corpus]
    train, vocab = padded_everygram_pipeline(2, corpus)
    lm = KneserNeyInterpolated(2)
    lm.fit(train, vocab)
    return lm


def SLM_score(lm, text):
    """Returns the SLM score of a text over a certain corpus.

    :param lm: the language model, which is got from function train_model()
    :type lm: nltk.lm.models.Lidstone
    :parame text: the input text
    :type text: str

    :rtype: float, the SLM score
    """
    text = text.strip().split()
    bigrams = list(ngrams(text, 2))
    return lm.entropy(bigrams)


if __name__ == "__main__":
    corpus = ["      this     is a dog", "great am a gentleman"]
    bigram_model = train_model(corpus)
    print(SLM_score(bigram_model, "U r dog"))


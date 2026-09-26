#!/usr/bin/env python3
"""Learn FastText word vectors (built from pieces of words) and turn
each message into one vector."""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """Train FastText on the corpus and represent each message as the
    mean of its token vectors. Returns (X, model)."""
    # train the model
    model = gensim.models.FastText(sentences=corpus_tokens,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   sg=sg,
                                   epochs=epochs,
                                   workers=workers)

    rows = []
    for tokens in corpus_tokens:
        # FastText has a vector for every token, even unknown ones
        vectors = []
        for token in tokens:
            vectors.append(model.wv[token])

        # mean of the vectors, or zeros if the message is empty
        if len(vectors) > 0:
            rows.append(np.mean(vectors, axis=0))
        else:
            rows.append(np.zeros(vector_size))

    X = np.array(rows)
    return X, model

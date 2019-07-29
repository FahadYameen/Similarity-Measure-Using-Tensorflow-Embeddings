# Similarity-Measure-Using-Tensorflow-Embeddings

This Repo is the Hello World for the Application of Tensorflow Embeddings.

### Embedding:
An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.

### Universal Sentence Encoder

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity.

## Installation :

You must install or upgrade your TensorFlow package to at least 1.7 to use TensorFlow Hub
```sh 
$ pip install "tensorflow>=1.7.0"
$ pip install tensorflow-hub
```
And then download Tensorflow hub module
```sh
curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
```
### Code
```sh
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
model = hub.Module('tmp/moduleA/')

my_sentences = ['how old are you ?','what is your age ?','there is a chance of rain today','Rain is predicted today']
embeddings = []
with tf.Session() as session:
            session.run([tf.global_variables_initializer(),tf.tables_initializer()])
            embeddings = model(my_sentences).eval()
all_similarities = np.array([1 - cosine(embeddings, sentence_embedding) for sentence_embedding in embeddings])
print(all_similarities)


```







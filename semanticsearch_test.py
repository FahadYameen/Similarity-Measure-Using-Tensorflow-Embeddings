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

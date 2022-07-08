
# Source: http://neuralnetworksanddeeplearning.com

import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin-1")
    f.close()
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

tr_d, va_d, te_d = load_data()
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
with open("training.data","w") as f:
    for i in range(50000):
        for x in training_inputs[i]:
            f.write(str(x[0]) + '\n')
        for x in training_results[i]:
            f.write(str(x[0]) + '\n')
with open("test.data","w") as f:
    for i in range(10000):
        for x in test_inputs[i]:
            f.write(str(x[0]) + '\n')
        f.write(str(te_d[1][i]) + '\n')

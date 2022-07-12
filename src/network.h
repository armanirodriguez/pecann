#pragma once

#include "matrix.h"

typedef struct Network {
    unsigned nLayers, *sizes;
    Matrix *biases;
    Matrix *weights;
} Network;

enum EActivationFunction {
    FN_SIGMOID,
    FN_TANH,
    FN_RELU
};

typedef struct TrainingExample {
    unsigned nInputs, nOutputs;
    float *input, *output;
} TrainingExample;

Network *initNetwork(unsigned *layerSizes, size_t nLayers);
Matrix feedForward(Network *net, float *input, enum EActivationFunction af);
TrainingExample createTrainingExample(float *expectedInput, float *expectedOutput, size_t nInputs, size_t nOutputs);
void freeNetwork(Network *net);
int saveNetworkToFile(const char *filename, Network *net);
Network *readNetworkFromFile(const char *filename);
void stochasticGradientDescent(
    Network* net,
    TrainingExample *trainingData,
    size_t nExamples,
    unsigned epochs, 
    size_t batchSize,
    float learningRate,
    enum EActivationFunction af,
    TrainingExample *testData,
    unsigned nTestData);

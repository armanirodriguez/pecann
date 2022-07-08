#pragma once

#include "matrix.h"

typedef struct Network {
    unsigned nLayers, *sizes;
    Matrix *biases;
    Matrix *weights;
} Network;

typedef struct TrainingExample {
    unsigned nInputs, nOutputs;
    float *input, *output;
} TrainingExample;

Network *initNetwork(unsigned *layerSizes, size_t nLayers);
Matrix feedForward(Network *net, float *input);
TrainingExample createTrainingExample(float *expectedInput, float *expectedOutput, size_t nInputs, size_t nOutputs);
void freeNetwork(Network *net);
void saveNetworkToFile(FILE *file, Network *net);
Network *readNetworkFromFile(FILE *file);
void stochasticGradientDescent(
    Network* net,
    TrainingExample *trainingData,
    size_t nExamples,
    unsigned epochs, 
    size_t batchSize,
    float learningRate,
    TrainingExample *testData,
    unsigned nTestData);
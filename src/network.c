/**
 * @author Armani Rodriguez
 * @brief A simple neural network based off the book http://neuralnetworksanddeeplearning.com
 * @version 0.1
 * @date 2022-07-08
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "network.h"

#define RAND() (((float)rand()/(float)RAND_MAX)/100)

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

static void shuffleTrainingData(TrainingExample *data, size_t size);
static void backprop(Network *net, TrainingExample example, Matrix *dBiases, Matrix *dWeights, enum EActivationFunction af);

/* Activation functions and their derivatives */
static inline float _sigmoid(float x) { return 1/(1 + exp(-x)); }
static inline float _sigmoidPrime(float x) { return _sigmoid(x) * (1 - _sigmoid(x)); }
static inline float _tanh(float x) { return tanh(x); }
static inline float _tanhPrime(float x) { return 1 - pow(tanh(x),2); }
static inline float _relu(float x) { return x > 0 ? x : 0; }
static inline float _reluPrime(float x) { return x >= 0 ? 1 : 0; }

/**
 * @brief Create a neural network initialized with random wieghts and biases
 * 
 * @param layerSizes An array containing the size of each layer. Ex: {784, 100, 10}
 * @param nLayers The amount of layers which is the number of elements in the size array
 * @return Network* A pointer to a fully initialized network.
 */
Network *initNetwork(unsigned *layerSizes, size_t nLayers) {
    assert(nLayers > 0 && layerSizes);
    Network *net = calloc(1, sizeof(Network));
    if (!net) {
        return NULL;
    }
    net->sizes = layerSizes;
    net->nLayers = nLayers;
    net->biases = malloc((nLayers - 1) * sizeof(Matrix));
    net->weights = malloc((nLayers - 1) * sizeof(Matrix));
    
    if (!net->biases || !net->weights) {
        freeNetwork(net);
        return NULL;
    }

    srand(time(NULL));

    /* Initialize weights and biases to random values */
    for (unsigned i = 0; i < nLayers - 1; i++) {
        unsigned prevSize = layerSizes[i];
        unsigned size = layerSizes[i + 1];
        /* Biases */
        Matrix b = matrix(size,1);
        for (unsigned j = 0; j < len(b); j++) {
            b.data[j] = RAND();
        }
        net->biases[i] = b;
        /* Weights */
        Matrix w = matrix(size, prevSize);
        for (unsigned j = 0; j < len(w); j++) {
            w.data[j] = RAND();
        }
        net->weights[i] = w;
    }
    return net;
}

/**
 * @brief Save a neural network to a file stream
 * 
 * @param filename The file name to save to
 * @param net A pointer to a network
 * @return 0 on success, -1 otherwise
 */
int saveNetworkToFile(const char *filename, Network *net) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        return -1;
    }
    fprintf(fp,"%d ",net->nLayers);
    for (unsigned i = 0; i < net->nLayers; i++) {
        fprintf(fp, "%d ", net->sizes[i]);
    }
    for (unsigned i = 0; i < net->nLayers - 1; i++) {
        saveMatrixToFile(fp, net->weights[i]);
        saveMatrixToFile(fp, net->biases[i]);
    }
    fclose(fp);
    return 0;
}

/**
 * @brief Read a network from a file stream
 * 
 * @param filename the name of the file to open
 * @return the deserialized network or NULL on failure
 */
Network *readNetworkFromFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return NULL;
    }
    Network *net = calloc(1, sizeof(Network));
    if (net == NULL) {
        return NULL;
    }
    int read;
    read = fscanf(file, "%d", &net->nLayers);
    if (read != 1) {
        freeNetwork(net);
        return NULL;
    }
    net->sizes = malloc(net->nLayers * sizeof(unsigned));
    if (!net->sizes) {
        freeNetwork(net);
        return NULL;
    }
    for (unsigned i = 0; i < net->nLayers; i++) {
        read = fscanf(file, "%d", &net->sizes[i]);
        if (read != 1) {
            freeNetwork(net);
            return NULL;
        }
    }
    net->biases = malloc((net->nLayers - 1) * sizeof(Matrix));
    net->weights = malloc((net->nLayers - 1) * sizeof(Matrix));
    if (!net->biases || !net->weights) {
        freeNetwork(net);
        return NULL;
    }
    for (unsigned i = 0; i < net->nLayers - 1; i++) {
        net->weights[i] = readMatrixFromFile(file);
        net->biases[i] = readMatrixFromFile(file);
    }
    return net;
}

/**
 * @brief Given an input and a network, return the network's output in matrix form
 * 
 * @param net Pointer to a network
 * @param input A 
 * @return Matrix 
 */
Matrix feedForward(Network *net, float *input, enum EActivationFunction af) {
    assert(net && input);
    float (*activationFunction)(float);
    switch(af){
        case FN_SIGMOID:
            activationFunction = _sigmoid;
            break;
        case FN_RELU:
            activationFunction = _relu;
            break;
        case FN_TANH:
            activationFunction = _tanh;
            break;
        default:
            activationFunction = _sigmoid;
            break;
    }
    Matrix result = {0}, a, b, w, wa;
    a = matrixFromData(net->sizes[0], 1, input);
    for (unsigned i = 0; i < net->nLayers - 1; i++) {
        b = net->biases[i];
        w = net->weights[i];
        wa = mult(w, a);
        freeMatrix(result);
        result = add(wa, b);
        applyFuncInPlace(result,activationFunction);
        a = result;
        freeMatrix(wa);
    }
    return result;
}

/**
 * @brief Train the network using SGD algorithm
 * 
 * @param net Pointer to a network
 * @param trainingData Array of TrainingExamples to train the network with
 * @param nExamples Number of TrainingExamples in trainingData
 * @param epochs Amount of epochs
 * @param batchSize Maximum size of each mini-batch
 * @param learningRate Learning rate variable
 * @param af The activation function
 * @param testData Optional: Array of TrainingExamples to test the network against.
 * @param nTestData Optional: Number of TrainingExamples in testData
 * Note: The TrainingExamples in trainingData should have the same output size as the network, contianing the desired activation for each output perceptron.
 *       The TrainingExamples in testData should have output size 1, containing the index of the desired highest activation perceptron.
 */
void stochasticGradientDescent(
    Network* net,
    TrainingExample *trainingData,
    size_t nExamples,
    unsigned epochs, 
    size_t batchSize,
    float learningRate,
    enum EActivationFunction af,
    TrainingExample *testData,
    unsigned nTestData) 
{
    assert( net &&
        nExamples && 
        trainingData &&
        epochs && 
        batchSize && 
        learningRate && 
        (!!nTestData == !!testData));
    for (unsigned i = 0; i < epochs; i++) {
        shuffleTrainingData(trainingData, nExamples);
        /* Run the backpropagation algorithm on each mini batch and update the network */
        for (TrainingExample *b = trainingData; b < trainingData + nExamples; b+=batchSize) {
            
            size_t bSize;
            if (b + batchSize - 1 < trainingData + nExamples) {
                bSize = batchSize;
            } else {
                bSize = trainingData + nExamples - b + 1;
            }
            Matrix dWeights[net->nLayers - 1], dBiases[net->nLayers - 1];
            for (unsigned j = 0; j < net->nLayers - 1; j++) {
                    dBiases[j] = matrix(net->biases[j].rows, net->biases[j].cols);
                    dWeights[j] = matrix(net->weights[j].rows, net->weights[j].cols);
            }
            for (unsigned j = 0; j < bSize; j++) {
                TrainingExample example = b[j];
                backprop(net,example, dBiases, dWeights, af);
            }
            for (unsigned j = 0; j < net->nLayers - 1; j++) {
                Matrix changeWeight = scalarMult(dWeights[j], (learningRate/(float)bSize));
                subInPlace(net->weights[j],changeWeight);
                Matrix changeBias = scalarMult(dBiases[j], (learningRate/bSize));
                subInPlace(net->biases[j],changeBias);

                freeMatrix(dWeights[j]);
                freeMatrix(dBiases[j]);
                freeMatrix(changeWeight);
                freeMatrix(changeBias);
            }
        }
         if (testData) {
                unsigned nPassed = 0;
                for (unsigned j = 0; j < nTestData; j++) {
                    TrainingExample test = testData[j];
                    Matrix result = feedForward(net, test.input, af);
                    if (maxIndex(result) == *test.output) {
                        nPassed++;
                    }
                    freeMatrix(result);
                }
                printf("Epoch %d complete. %d/%d passing\n", i+1, nPassed, nTestData);
        } else {
            printf("Epoch %d complete\n", i+1);
        }
    }
}

/**
 * @brief Runs the backpropagation algorithm. Helper for SGD
 * 
 * @param net Pointer to a network
 * @param example The TrainingExample to feed forward
 * @param dBiases An array which will be modified according to delta nabla b
 * @param dWeights An array which will be modified according to delta nabla w
 * @param af The activation function to use
 */
static void backprop(Network *net, TrainingExample example, Matrix *dBiases, Matrix *dWeights, enum EActivationFunction af) {
    assert(net && dBiases && dWeights);
    float (*activationFunction)(float);
    float (*activationFunctionDerivative)(float);
    switch(af){
        case FN_SIGMOID:
            activationFunction = _sigmoid;
            activationFunctionDerivative = _sigmoidPrime;
            break;
        case FN_RELU:
            activationFunction = _relu;
            activationFunctionDerivative = _reluPrime;
            break;
        case FN_TANH:
            activationFunction = _tanh;
            activationFunctionDerivative = _tanhPrime;
            break;
        default:
            activationFunction = _sigmoid;
            activationFunctionDerivative = _sigmoidPrime;
            break;
    }
    Matrix zs[net->nLayers - 1], activations[net->nLayers], a, b, w, wa, z;

    a = matrixFromData(example.nInputs, 1, example.input);
    activations[0] = copy(a);
    /* Feed Forward but save z's */
    for (unsigned i = 0; i < net->nLayers - 1; i++) {
        b = net->biases[i];
        w = net->weights[i];
        wa = mult(w, a);
        z = add(wa, b);
        a = applyFunc(z,activationFunction);
        
        zs[i] = z;

        activations[i + 1] = a;
        
        freeMatrix(wa);
    }
    Matrix costDerivative, delta, aT, deltaAT, wT, wTdelta, newDelta;

    costDerivative = sub(activations[net->nLayers - 1], matrixFromData(example.nOutputs,1,example.output));
    applyFuncInPlace(zs[net->nLayers - 2], activationFunctionDerivative);
    delta = hadamard(costDerivative, zs[net->nLayers - 2]);

    freeMatrix(costDerivative);

    addInPlace(dBiases[net->nLayers - 2],delta);
    aT = transpose(activations[net->nLayers - 2]);
    deltaAT = mult(delta, aT);
    
    addInPlace(dWeights[net->nLayers - 2], deltaAT);
    freeMatrix(deltaAT);
    freeMatrix(aT);
    for (unsigned i = 2; i < net->nLayers; i++) {
        z = zs[net->nLayers - 1 - i];
        applyFuncInPlace(z, activationFunctionDerivative);
        wT = transpose(net->weights[net->nLayers - i]);
        wTdelta = mult(wT, delta);
        newDelta = hadamard(wTdelta, z);
        
        addInPlace(dBiases[net->nLayers - 1 - i], newDelta);
        aT = transpose(activations[net->nLayers - 1 - i]);
        deltaAT = mult(newDelta,aT);
        addInPlace(dWeights[net->nLayers - 1 - i], deltaAT);
        
        freeMatrix(delta);
        freeMatrix(deltaAT);
        freeMatrix(wT);
        freeMatrix(wTdelta);
        freeMatrix(aT);
        
        delta = newDelta;
    }

    freeMatrix(delta);
    
    for (unsigned i = 0; i < net->nLayers - 1; i++){
        freeMatrix(zs[i]);
    }
    for (unsigned i = 0; i < net->nLayers; i++) {
        freeMatrix(activations[i]);
    }
}

/**
 * @brief Create a Training Example object
 * 
 * @param expectedInput The desired input
 * @param expectedOutput The desired output
 * @param nInputs Number of elements in expectedInput
 * @param nOutputs Number of elements in expectedOutput
 * @return TrainingExample object
 */
TrainingExample createTrainingExample(float *expectedInput, float *expectedOutput, size_t nInputs, size_t nOutputs) {
    assert(nInputs && nOutputs && expectedInput && expectedOutput);
    TrainingExample te = {nInputs, nOutputs, expectedInput, expectedOutput};
    return te;
}

/**
 * @brief Destroy a network and free all memory
 * 
 * @param net The network to free
 */
void freeNetwork(Network *net) {
    if (net) {
        if (net->biases) {
            for (unsigned i = 0; i < net->nLayers - 1; i++) {
                freeMatrix(net->biases[i]);
            }
            free(net->biases);
        }
        if (net->weights) {
            for (unsigned i = 0; i < net->nLayers - 1; i++) {
                freeMatrix(net->weights[i]);
            }
            free(net->weights);
        }
        free(net);
    }
}

/**
 * @brief Shuffle elements in the array data. Helper for SGD. 
 * 
 * @param data The array to shuffle
 * @param size Number of elements in data array.
 */
static void shuffleTrainingData(TrainingExample *data, size_t size){
    assert(data && size);
    for (unsigned i = 0; i < size; i++) {
        unsigned j = rand() % size;
        TrainingExample tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}
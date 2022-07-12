#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/network.h"

int main() {
    TrainingExample trainingData[50000];
    TrainingExample testData[10000];
    FILE *trainingFile = fopen("training.data","r");
    FILE *testFile = fopen("test.data","r");
    assert(trainingFile && testFile);
    float f;
    for (int i = 0; i < 50000; i++){
        float *inputs = malloc(784 * sizeof(float));
        float *outputs = malloc(10 * sizeof(float));
        for (int j = 0; j < 784; j++) {
            fscanf(trainingFile, "%f", &f);
            inputs[j] = f;
        }
        for (int j = 0; j < 10; j++) {
            fscanf(trainingFile, "%f", &f);
            outputs[j] = f;
        }
        trainingData[i] = createTrainingExample(inputs,outputs,784,10);
    }
    for (int i = 0; i < 10000; i++) {
        float *inputs = malloc(784 * sizeof(float));
        float *outputs = malloc(sizeof(float));
        for (int j = 0; j < 784; j++) {
            fscanf(testFile, "%f", &f);
            inputs[j] = f;
        }
        fscanf(testFile, "%f", &f);
        outputs[0] = f;
        testData[i] = createTrainingExample(inputs, outputs, 784, 1);
    }
    unsigned sizes[] = {784, 100, 10};
    Network *net = initNetwork(sizes, 3);
    printf("Starting SGD\n");
    stochasticGradientDescent(net, trainingData, 50000, 30, 10, 1, FN_SIGMOID, testData, 10000);
    saveNetworkToFile("mnist.nn", net);
    freeNetwork(net);
}

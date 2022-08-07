# pecann
A lightweight ANN library written in C

This is a very simple artificial neural network library. After training with the MNIST handwritten digits dataset, it can correctly
identify handwritten digits it has never seen before with ~98% accuracy. I have a long way to go with regards to optimization. It takes
about 8-10 seconds per epoch on my machine.

## Creating a network
```C
/* Create a network with 3 total layers. Input with size 784, hidden layer with size 100, and an output layer of size 10 */
unsigned sizes[] = {784, 100, 10};
Network *net = initNetwork(sizes, 3);
```

## Training the network
```C
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
    unsigned nTestData);
```

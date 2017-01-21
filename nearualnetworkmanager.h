#ifndef NEARUALNETWORKMANAGER_H
#define NEARUALNETWORKMANAGER_H

#include "neuralnetwork.h"


class NearualNetworkManager
{
public:
    NearualNetworkManager(int hnl);
    void trainNeuralNetwork();
    void testNeuralNetwork();
    void showWeights();

private:

    double* trainNNWithPicture(NeuralNetwork* neauralNetwork);
    double* normalizePictureInputs(double* inputs);
    double* loadPictureInputs(ifstream& file);
    double* loadPictureOutputs(ifstream& file);
    bool checkIfWeightFileExist(string filename);
    double* loadWeights(ifstream& file);
    void printPictureData(double* inputs, double* outputs);
    void printNNOutputs(double* outputs);

    NeuralNetwork *neuralNetwork;
    int inputLayerNeurons;
    int outputLayerNeurons;
    int hiddenLayerNeurons;
};

#endif // NEARUALNETWORKMANAGER_H

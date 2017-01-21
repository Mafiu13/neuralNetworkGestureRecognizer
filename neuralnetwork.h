
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include "neuralnetworkconst.h"
#include <math.h>
#include <iostream>

using namespace std;

class NeuralNetwork
{
public:
    NeuralNetwork(int ninput, int nhidden, int noutput);
    ~NeuralNetwork();
    void loadInputs(string filename);
    void loadInputs(double* inputs,double* desOutputs);
    void initRandomWeights();
    //void initWeights(double* weights, int wlength);
    void initWeights(string filename);
    void computeForward();               //obliczanie wyjscia
    void computeBackPropagation();       //wsteczna propagacja
    void saveWeights(string filename);
    double getError();
    double* getOutputs();

private:
    double sigmoidFunc(double x);
    double tanhFunc(double x);

    int inputL;
    int hiddenL;
    int outputL;

    double* desOutputs;             // wartosci wyjsciowe oczekiwane

    double* inputs;
    double** ihWeights;             //wagi od wejscia do ukrytej
    double* ihBiases;               //biasy od wejscia do ukrytej
    double* ihOutputs;
    double** hoWeights;             //wagi od ukrytej do wyjscia
    double* hoBiases;               //biasy od ukrytej do wyjscia
    double* outputs;

    //  back-propagation
    double* oGrads;                 // output gradients for back-propagation
    double* hGrads;                 // hidden gradients for back-propagation
};

#endif

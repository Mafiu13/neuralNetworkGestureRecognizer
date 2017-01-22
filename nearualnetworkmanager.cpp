#include "nearualnetworkmanager.h"

NearualNetworkManager::NearualNetworkManager(int hnl)
    :inputLayerNeurons(NR_INPUT_NEURONS),
     outputLayerNeurons(NR_OUTPUT_NEURONS),
     hiddenLayerNeurons(hnl)
{
    neuralNetwork = new NeuralNetwork(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons);
}

void NearualNetworkManager::trainNeuralNetwork()
{
    ifstream file(TRAIN_FILE_NAME);
    int nrTrain;
    file >> nrTrain;

    double* inputs = new double [NR_INPUT_NEURONS];
    double* outputs = new double [NR_OUTPUT_NEURONS];
    double* nnOutputs = new double [NR_OUTPUT_NEURONS];

    neuralNetwork->initRandomWeights();

    for (int i = 0; i < nrTrain; ++i)
    {
        cout<<"\nPicture nr: "<<i<<"\n";
        inputs = loadPictureInputs(file);
        outputs = loadPictureOutputs(file);

        //printPictureData(inputs, outputs);

        inputs = normalizePictureInputs(inputs);
        //printPictureData(inputs, outputs);

        neuralNetwork->loadInputs(inputs, outputs);
        nnOutputs = trainNNWithPicture(neuralNetwork);
        //printNNOutputs(nnOutputs);
    }
    file.close();
    neuralNetwork->saveWeights(WEIGHT_FILE_NAME);
}

void NearualNetworkManager::testNeuralNetwork()
{
    if (checkIfWeightFileExist(WEIGHT_FILE_NAME))
    {
        neuralNetwork->initWeights(WEIGHT_FILE_NAME);
    } else {
        neuralNetwork->initRandomWeights();
    }

    ifstream file(TEST_FILE_NAME);
    int nrTest;
    file >> nrTest;

    int correct = 0;
    int inCorrect = 0;

    double output = 0.0;
    double desOutput = 0.0;

    double* inputs = new double [NR_INPUT_NEURONS];
    double* outputs = new double [NR_OUTPUT_NEURONS];
    double* nnOutputs = new double [NR_OUTPUT_NEURONS];

    for (int i = 0; i < nrTest; ++i)
    {
        cout<<"\nPicture nr: "<<i<<"\n";
        inputs = loadPictureInputs(file);
        outputs = loadPictureOutputs(file);
        printPictureData(inputs, outputs);
        inputs = normalizePictureInputs(inputs);
        neuralNetwork->loadInputs(inputs, outputs);
        neuralNetwork->computeForward();
        nnOutputs = neuralNetwork->getOutputs();
        printNNOutputs(nnOutputs);
        cout<<neuralNetwork->getError()<<"\n";

        if (neuralNetwork->isCorrectluRecognized())
        {
            ++correct;
        } else {
            ++inCorrect;
        }
    }
    cout<<"--------------------------\n"<<" --- Correctly recognized: "<<correct<<" --- \n--------------------------\n";
    cout<<"--------------------------\n"<<" --- Incorrectly recognized: "<<inCorrect<<" --- \n--------------------------\n";
    file.close();
}

double* NearualNetworkManager::trainNNWithPicture(NeuralNetwork* neauralNetwork)
{
    int i = 0;
    neuralNetwork->computeForward();
    double error = neuralNetwork->getError();

    while(i<MIN_LOOP_ITER && error>OUTPUT_ERRROR)
    //while(i<20)
    {
        neuralNetwork->computeBackPropagation();
        neuralNetwork->computeForward();
        error = neuralNetwork->getError();
        ++i;
    }

    return neuralNetwork->getOutputs();
}


double*  NearualNetworkManager::loadPictureInputs(ifstream& file)
{
    double* inputs = new double [NR_INPUT_NEURONS];

    for(int j = 0; j < NR_PIXELS; ++j)
    {
        for(int g = 0; g < NR_PIXELS; ++g)
        {
            file>>inputs[j*NR_PIXELS+g];
        }
    }
    return inputs;

}

double*  NearualNetworkManager::normalizePictureInputs(double* inputs)
{
    double* inputsNew = new double [NR_INPUT_NEURONS];
    double temp;
    for(int l = 0; l< NR_INPUT_NEURONS; ++l)
    {
        temp = inputs[l];
        inputsNew[l] = (temp*2/7.0 - 1); // normalizacja danych wejściowych z od 0 do 7 na od -1 do 1
    }
    return inputsNew;
}

double*  NearualNetworkManager::loadPictureOutputs(ifstream& file)
{
    double* outputs = new double [NR_OUTPUT_NEURONS];
    for(int l = 0; l< NR_OUTPUT_NEURONS; ++l)
    {
     file>>outputs[l];
    }
    return outputs;
}

void NearualNetworkManager::printPictureData(double* inputs, double* outputs)
{

    for(int j = 0; j < NR_PIXELS; ++j)
    {
        for(int g = 0; g < NR_PIXELS; ++g)
        {
            cout<<inputs[j*NR_PIXELS+g]<<" ";

        }
        cout<<"\n";
    }

    cout<<"Expected outputs: ";
    for(int l = 0; l< NR_OUTPUT_NEURONS; ++l)
    {
     cout<<outputs[l]<<" ";
    }
    cout<<"\n";
}

void NearualNetworkManager::printNNOutputs(double* outputs)
{
    cout<<"Computed outputs: ";
    for(int i = 0; i<2; ++i)
    {
        cout<<outputs[i]<<" ";
    }
    cout<<"\n";
}

bool NearualNetworkManager::checkIfWeightFileExist(string filename)
{
    ifstream file;
    file.open(filename.c_str());
    return file.is_open();
}


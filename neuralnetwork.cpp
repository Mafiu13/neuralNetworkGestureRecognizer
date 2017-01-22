#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int ninput, int nhidden, int noutput)
{
    inputL=ninput;
    outputL=noutput;
    hiddenL=nhidden;

    desOutputs = new double [outputL];

    inputs = new double [inputL];
    outputs = new double [outputL];
    ihBiases = new double [hiddenL];
    ihOutputs = new double [hiddenL];
    hoBiases = new double [outputL];

    hGrads = new double [hiddenL];
    oGrads = new double [outputL];

    ihWeights = new double* [inputL];
    for(int i=0;i<inputL;++i)
        ihWeights[i] = new double [hiddenL];

    hoWeights = new double* [hiddenL];
    for(int i=0;i<hiddenL;++i)
        hoWeights[i] = new double [outputL];
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] inputs;
    delete[] outputs;
    delete[] ihBiases;
    delete[] ihOutputs;
    delete[] hoBiases;

    delete[] desOutputs;
    delete[] hGrads;
    delete[] oGrads;

    for(int i=0;i<inputL;++i)
        delete[] ihWeights[i];
    delete[] ihWeights;

    for(int i=0;i<hiddenL;++i)
        delete[] hoWeights[i];
    delete[] hoWeights;
}

double NeuralNetwork::sigmoidFunc(double x)
{
    if(x < -45.0)
        return 0.0;
    else if(x > 45.0)
        return 1.0;
    else
        return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::tanhFunc(double x)
{
    if(x < -10.0)
        return -1.0;
    else if (x > 10.0)
        return 1.0;
    else
        return tanh(x);
}

void NeuralNetwork::initRandomWeights()
{
    srand(time(NULL));
    for(int i=0;i<hiddenL;++i)
    {
        ihBiases[i]=(rand()%201 - 100)/100.0;
    }
    for(int i=0;i<outputL;++i)
    {
        hoBiases[i]=(rand()%201 - 100)/100.0;
    }
    for(int i=0;i<inputL;++i)
    {
        for(int j=0;j<hiddenL;++j)
        {
            ihWeights[i][j]=(rand()%201 - 100)/100.0;
        }
    }
    for(int i=0;i<hiddenL;++i)
    {
        for(int j=0;j<outputL;++j)
        {
            hoWeights[i][j]=(rand()%201 - 100)/100.0;
        }
    }
}

void NeuralNetwork::loadInputs(std::string filename)
{
    ifstream inp;
    int temp;
    inp.open(filename.c_str());
    if(inp.is_open())
    {
        for(int i=0;i<inputL;++i)
        {
            inp >> temp;
            inputs[i] = temp*2.0/7.0 - 1.0; // normalizacja danych wejściowych z od 0 do 7 na od -1 do 1
        }
        inp.close();
    } else {
    }
}

void NeuralNetwork::loadInputs(double* inputs, double* desOutputs)
{
    this->inputs = inputs;
    this->desOutputs = desOutputs;
}

void NeuralNetwork::initWeights(string filename)
{
    ifstream wi;
    wi.open(filename.c_str());
    if(wi.is_open())
    {
        for(int i=0;i<inputL;++i)
        {
            for(int j=0;j<hiddenL;++j)
            {
                wi >> ihWeights[i][j];
            }
        }
        for(int i=0;i<hiddenL;++i)
        {
            wi >> ihBiases[i];
        }
        for(int i=0;i<hiddenL;++i)
        {
            for(int j=0;j<outputL;++j)
            {
                wi >> hoWeights[i][j];
            }
        }
        for(int i=0;i<outputL;++i)
        {
            wi >> hoBiases[i];
        }
        wi.close();
    }
}

void NeuralNetwork::computeForward()
{
    double sum;
    for(int i=0;i<hiddenL;++i)
    {
        sum=0;
        for(int j=0;j<inputL;++j)
        {
            sum+=inputs[j]*ihWeights[j][i];
            //cout<<ihWeights[j][i]<<"\n";
            //cout<<"input : "<<inputs[j]<<"\n";
        }
        //cout<<"hidden sum before bies: "<<sum<<"\n";
        sum+=ihBiases[i];
        //cout<<"hidden sum + bes: "<<sum<<"\n";
        //cout<<"hidden biess: "<<ihBiases[i]<<"\n";
        ihOutputs[i]=sigmoidFunc(sum);
        //cout<<"hidde + biess + sigm: "<<ihOutputs[i]<<"\n";
    }
    for(int i=0;i<outputL;++i)
    {
        sum=0;
        for(int j=0;j<hiddenL;++j)
        {
            sum+=ihOutputs[j]*hoWeights[j][i];
            //cout<<hoWeights[j][i]<<"\n";
            //cout<<"hidden output: "<<ihOutputs[j]<<"\n";
        }
        sum+=hoBiases[i];
        //cout<<"output biess: "<<hoBiases[i]<<"\n";
        outputs[i]=sigmoidFunc(sum);
        //cout<<"output + biess "<<outputs[i]<<"\n";
    }
}

void NeuralNetwork::computeBackPropagation()
{
    // 1. compute output gradients
    for (int i = 0; i < outputL; ++i)
    {
      double derivative = (1 - outputs[i]) * outputs[i]; // derivative of sigm
      oGrads[i] = derivative * (desOutputs[i] - outputs[i]);
    }

    // 2. compute hidden gradients
    for (int i = 0; i < hiddenL; ++i)
    {
      double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; //  derivative of sig
      double sum = 0.0;
      for (int j = 0; j < outputL; ++j)
        sum += oGrads[j] * hoWeights[i][j];
      hGrads[i] = derivative * sum;
    }

    // 3. update input to hidden weights
    for (int i = 0; i < inputL; ++i)
    {
      for (int j = 0; j < hiddenL; ++j)
      {
        double delta = hGrads[j] * inputs[i]; // compute the new delta
        ihWeights[i][j] += delta; // update
      }
    }

    // 3b. update input to hidden biases
    for (int i = 0; i < hiddenL; ++i)
    {
      double delta = hGrads[i] * 1.0;
      ihBiases[i] += delta;
    }

    // 4. update hidden to output weights
    for (int i = 0; i < hiddenL; ++i)
    {
      for (int j = 0; j < outputL; ++j)
      {
        double delta = oGrads[j] * ihOutputs[i];
        hoWeights[i][j] += delta;
      }
    }

    // 4b. update hidden to output biases
    for (int i = 0; i < outputL; ++i)
    {
      double delta = oGrads[i] * 1.0;
      hoBiases[i] += delta;
    }
}

void NeuralNetwork::saveWeights(string filename)
{
    ofstream wo;
    wo.open(filename.c_str());
    if(wo.is_open())
    {
        for(int i=0;i<inputL;++i)
        {
            for(int j=0;j<hiddenL;++j)
            {
                wo << ihWeights[i][j] << ' ';
            }
            wo << endl;
        }
        for(int i=0;i<hiddenL;++i)
        {
            wo << ihBiases[i] << ' ';
        }
        wo << endl;
        for(int i=0;i<hiddenL;++i)
        {
            for(int j=0;j<outputL;++j)
            {
                wo << hoWeights[i][j] << ' ';
            }
            wo << endl;
        }
        for(int i=0;i<outputL;++i)
        {
            wo << hoBiases[i] << ' ';
        }
        wo << endl;
        wo.close();
    }
}

double NeuralNetwork::getError()
{
    double sum = 0;
    for (int i = 0; i < NR_OUTPUT_NEURONS; ++i)
    {
        sum += abs(desOutputs[i] - outputs[i]);
    }
    return sum;
}

bool NeuralNetwork::isCorrectluRecognized()
{
    double temp = 0.0;
    int index = 0;
    for (int i = 0; i < NR_OUTPUT_NEURONS; ++i)
    {
        if (temp < outputs[i])
        {
            temp = outputs[i];
            index = i;
        }
    }
    double temp2 = desOutputs[index];
    if (temp2 == 1)
    {
        return true;
    } else {
        return false;
    }
}

double* NeuralNetwork::getOutputs()
{
    return outputs;
}

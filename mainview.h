#ifndef MAINVIEW_H
#define MAINVIEW_H

#include <iostream>
#include "mainmenutext.h"
#include "nearualnetworkmanager.h"

using namespace std;

class MainView
{
public:
    MainView();
private:
    void createNeuralNetworkManager();
    void showMainManu(bool cleared);
    void showTrainMenu();
    void showTestMenu();
    void runMainManu();
    void runTrainMenu();
    void runTestMenu();
    void clearView();

    int *hiddenLayerNeurons;
    NearualNetworkManager *nnManager;
};

#endif // MAINVIEW_H

#include "mainview.h"

MainView::MainView()
{
    createNeuralNetworkManager();
}

void MainView::createNeuralNetworkManager()
{
    clearView();
    cout<<MainMenuText::Title;
    cout<< MainMenuText::TrainInfo1;
    int nn;
    cin >> nn;
    while (cin.fail() || nn < 1)
    {
        cout<<"Enter a number bigger then 0\n";
        cin.clear();
        cin.ignore(256,'\n');
        cin >> nn;
    }
    hiddenLayerNeurons = &nn;
    nnManager = new NearualNetworkManager(*hiddenLayerNeurons);
    showMainManu(true);
}

void MainView::showMainManu(bool cleared)
{
    if (cleared)
    {
    clearView();
    }

    cout<<MainMenuText::Title;
    cout<<MainMenuText::TitleNrOfNeurons<<*hiddenLayerNeurons<<MainMenuText::TitleNrOfNeurons2;
    cout<<MainMenuText::MainManu1;
    cout<<MainMenuText::MainMenu2;
    cout<<MainMenuText::MainMenu3;
    cout<<MainMenuText::MainMenu4;
    runMainManu();
}

void MainView::showTrainMenu()
{
    clearView();
    cout<<MainMenuText::TrainTitle;
    cout<<MainMenuText::TitleNrOfNeurons<<*hiddenLayerNeurons<<MainMenuText::TitleNrOfNeurons2;
    cout<<MainMenuText::TrainMenu1;
    cout<<MainMenuText::TrainMenu2;
    runTrainMenu();
}

void MainView::showTestMenu()
{
    clearView();
    cout<<MainMenuText::TestTitle;
    cout<<MainMenuText::TitleNrOfNeurons<<*hiddenLayerNeurons<<MainMenuText::TitleNrOfNeurons2;
    cout<<MainMenuText::TestMenu1;
    cout<<MainMenuText::TrainMenu2;
    runTestMenu();
}

void MainView::runMainManu()
{
    int userChoice;
    cin>>userChoice;
    switch (userChoice)
    {
    case 1:
        showTrainMenu();
        break;
    case 2:
        showTestMenu();
        break;
    case 3:
        createNeuralNetworkManager();
        break;
    case 4:
        exit(0);
        break;
    default:
        runMainManu();
    }
}
void MainView::runTrainMenu()
{
    int userChoice;
    cin>>userChoice;
    switch (userChoice)
    {
    case 1:
        nnManager->trainNeuralNetwork();
        cout<<MainMenuText::TrainInfo3;
        showMainManu(false);
        break;
    case 2:
        showMainManu(true);
        break;
    default:
        runTrainMenu();
    }
}

void MainView::runTestMenu()
{
    int userChoice;
    cin>>userChoice;
    switch (userChoice)
    {
    case 1:
        nnManager->testNeuralNetwork();
        cout<<MainMenuText::TestInfo1;
        showMainManu(false);
        break;
    case 2:
        showMainManu(true);
        break;
    default:
        runTestMenu();
    }
}

void MainView::clearView()
{
    system("cls");
}

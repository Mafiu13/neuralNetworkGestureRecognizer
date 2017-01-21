#ifndef NEURALNETWORKCONST
#define NEURALNETWORKCONST

const static int NR_PIXELS = 16;
const static int NR_INPUT_NEURONS = 16*16;
const static int NR_OUTPUT_NEURONS = 2;

const static double OUTPUT_ERRROR = 0.05;
const static double MIN_LOOP_ITER = 1000;

const static std::string TRAIN_FILE_NAME ="train.txt";
const static std::string TEST_FILE_NAME ="test.txt";
const static std::string WEIGHT_FILE_NAME ="wagi.txt";

#endif // NEURALNETWORKCONST

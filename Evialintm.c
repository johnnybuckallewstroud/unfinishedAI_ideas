/*
 * Text Mode / console onely  adaptation of Evialin  
 * 
 * Evialintm.c
 * Enhanced Virtual Intelligence And Learning Interface Network
 * 
 * 100% Open source 
 * Updated: 2025/10/23 01:27:21AM 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * Notes: she is warning free and works, But is a leaky boat!
 * As in the stack it leaks 8,272 odd bytes or so in 3 blocks.. code base is very, 
 * very ugly. I will rewrite the source code for version 5 soon! 
 * Funny total heap usage 6,606 allocs ? like 666 the number of the beast! WTF? 
 * 
 * =4632=== HEAP SUMMARY:
 * ==4632==     in use at exit: 8,272 bytes in 3 blocks 
 * ==4632==   total heap usage: 6,606 allocs, 6,603 frees, 4,976,235 bytes allocated
 * ==4632== 
 * ==4632== 40 bytes in 1 blocks are definitely lost in loss record 1 of 3
 * ==4632==    at 0x48417B4: malloc (vg_replace_malloc.c:381)
 * ==4632==    by 0x11C2EF: createLayerWithActivation (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x11C648: createLayer (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x11C7CC: createNeuralNetwork (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x125A64: main (in /home/oneneon/randomx/Evialin)
 * ==4632== 
 * ==4632== 40 bytes in 1 blocks are definitely lost in loss record 2 of 3
 * ==4632==    at 0x48417B4: malloc (vg_replace_malloc.c:381)
 * ==4632==    by 0x11C2EF: createLayerWithActivation (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x11C648: createLayer (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x11C84B: createNeuralNetwork (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x125A64: main (in /home/oneneon/randomx/Evialin)
 * ==4632== 
 * ==4632== 8,192 bytes in 1 blocks are definitely lost in loss record 3 of 3
 * ==4632==    at 0x48417B4: malloc (vg_replace_malloc.c:381)
 * ==4632==    by 0x11FA12: generateContextAwareResponse (in /home/oneneon/randomx/Evialin)
 * ==4632==    by 0x1281EA: main (in /home/oneneon/randomx/Evialin)
 * ==4632== 
 * ==4632== LEAK SUMMARY:
 * ==4632==    definitely lost: 8,272 bytes in 3 blocks
 * ==4632==    indirectly lost: 0 bytes in 0 blocks
 * ==4632==      possibly lost: 0 bytes in 0 blocks
 * ==4632==    still reachable: 0 bytes in 0 blocks
 * ==4632==         suppressed: 0 bytes in 0 blocks
 * ==4632== 
 * ==4632== Use --track-origins=yes to see where uninitialised values come from
 * ==4632== For lists of detected and suppressed errors, rerun with: -s
 * ==4632== ERROR SUMMARY: 516 errors from 105 contexts (suppressed: 0 from 0)
 * 
 * How to make project: 
 * sudo apt-get update
 * sudo apt-get install libsqlite3-dev libcurl4-openssl-dev libc6-dev valgrind
 * gcc -g -Wall -o Evialin Evialin.c -lsqlite3 -lcurl -lm
 * Then just run the program like so: ./Evialin 
 * 
 * Debuging/Debugger for gcc source code: 
 * Example usage: valgrind --leak-check=full ./Evialin
 * Here are some common Valgrind commands:
 * valgrind --tool=memcheck ./Evialin   : Run memcheck, which detects memory-related issues.
 * valgrind --tool=cachegrind ./Evialin : Run cachegrind, which analyzes cache performance.
 * valgrind --tool=callgrind ./Evialin  : Run callgrind, which generates a call graph.
 * valgrind --leak-check=full ./Evialin : Run memcheck with full leak checking.
 * 
 */

// On GNU/Linux (like Pengwin), strdup() isn’t part of the strict C99 standard library ! 
// it’s a POSIX extension and/or A  "hickish grizzled old C coder added the missing sauce"  
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h> 
#include <errno.h>
#include <time.h>
#include <sqlite3.h>
#include <curl/curl.h>
#include "Evialin_gui.h"


#define _GNU_SOURCE
#define MAX_JOKES 23
#define MAX_INPUT_LENGTH 8192
#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 1000
#define MAX_NAME_LENGTH 120
#define MAX_TOKEN_LENGTH 50
#define MAX_VOCAB_SIZE 20000
#define MAX_GENERATED_LENGTH 1500
#define CREATIVITY_THRESHOLD 0.7f
#define MIN_ACTIVATION 0.1f
#define MAX_WORD_COMBINATIONS 20
#define Math_PI 3.1415926535897932384626433832795028841971693993751

sqlite3 *db;
	int toldJokes = 0;
    int jokeIndices[MAX_JOKES];

// Mood structures of neurons  
typedef enum {
    MOOD_HAPPY,
    MOOD_CONFUSED,
    MOOD_CURIOUS,
    MOOD_SARCASTIC,
    MOOD_PHILOSOPHICAL,
    MOOD_PLAYFUL,
    MOOD_THOUGHTFUL,
    MOOD_APOLOGETIC,
    MOOD_MYSTERIOUS,
    MOOD_ENERGETIC,
    MOOD_COUNT
} MoodType;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_GELU,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU, 
    ACTIVATION_SWISH   
} ActivationType;

typedef struct {
    MoodType currentMood;
    int moodDuration;
    int moodCounter;
    int triggerCount;  //Track number of times mood has been triggered
    double lastNeuralOutput;  //Store last neural network output
    float moodInfluence; // mood affects tone (0.0 - 1.0) 
    double logits[MOOD_COUNT]; // Learning/stabilization slow roll 
    double elig[MOOD_COUNT];  // eligibility trace for each mood epoch 
    double baseline;
    
} MoodState;

// Conversation history for learning
typedef struct {
    char input[MAX_INPUT_LENGTH];
    char response[MAX_INPUT_LENGTH];
    float confidence;
    int feedback_received;  // 0=none, 1=positive, -1=negative
    time_t timestamp;
} ConversationEntry;

typedef struct {
    ConversationEntry entries[100];  // Store last 100 interactions
    int count;
    int current_index;
} ConversationHistory;

ConversationHistory convHistory = {0};
char userFullName[MAX_NAME_LENGTH] = "";

// Define the structure for a neural network layer
typedef struct {
    int inputs;
    int neurons;
    double *weights;
    double *bias;
    double *output;
    double *delta;      // error signal for each neuron
    double *z_values;   // weighted sums before activation
    double *z; 
    ActivationType activation;  // Track which activation to use
} Layer;



// Define the structure for the neural network
typedef struct {
    int layers;
    Layer *layer;
    ActivationType defaultActivation;
} NeuralNetwork;

// Forward declarations
double sigmoid(double x);
double sigmoid_dynamic(double x, double steepness);
double tanh_dynamic(double x, double param);
double relu_dynamic(double x, double param);
double gelu(double x);
double applyActivation(double x, ActivationType type);
double safeGetNeuralAverage(NeuralNetwork* network, double defaultValue);
void runConversationLoop(NeuralNetwork* network, MoodState* botMood);
void updateFatigue();
void showVocabStats();
void copyHighConfidenceLearningToResponses(int threshold);
void previewPromotableKnowledge(int threshold);
void trainNeuralNetwork(NeuralNetwork* network, char** data, int dataSize);
void reinforceNeuralNetwork(NeuralNetwork* network);
void test_sigmoid_stability(NeuralNetwork* network);
const char* getMoodName(MoodType mood);
int isHoliday(struct tm* today, char* holidayName, size_t nameSize);
char* generateNewWordFromDatabase(void);
bool parseInput(const char *input); 
void loadUserFullName(void);
void rememberUserFullName(const char* fullName);
void loadUserFullName(void);
int getWordId(const char* word);
void reinforceNeuralNetwork_2(NeuralNetwork *network, double *input, double *target);
void forwardProp(NeuralNetwork* net, double* input);
void computeOutputDelta(NeuralNetwork* net, double* target);
void computeHiddenDeltas(NeuralNetwork* net);
void applyGradients(NeuralNetwork* net, double* input, double lr);
void trainSample(NeuralNetwork* net, double* input, double* target, double lr);
double activationDerivative(double z, ActivationType type);
void moodRewardUpdate(MoodState* mood, double reward);
void softmax_with_temp(const double logits[], double probs[], int n, double temp);
int sample_from_probs(const double probs[], int n);
const char* getMoodString(MoodState botMood);
// Core logic function prototype
void processUserInput(NeuralNetwork* network, MoodState* botMood, const char* input); 
// GUI backend function prototypes (assuming they are defined in Evialin_console.c)
void initGUIBackend(void); 
void cleanupGUIBackend(void);

void processUserInput(NeuralNetwork* network, MoodState* botMood, const char* input) {
    if (!input || !*input) return;

    char response[256] = {0};

    // --- Simple keyword-based logic ---
    if (strstr(input, "hello") || strstr(input, "hi")) {
        snprintf(response, sizeof(response), "Hello! How are you?");
        botMood->currentMood = MOOD_HAPPY;
    } else if (strstr(input, "how are you")) {
        snprintf(response, sizeof(response), "I'm doing fine, thanks!");
        botMood->currentMood = MOOD_PHILOSOPHICAL;
    } else if (strstr(input, "bye")) {
        snprintf(response, sizeof(response), "Goodbye! Talk soon.");
        botMood->currentMood = MOOD_CURIOUS;
    } else {
        snprintf(response, sizeof(response), "I'm having trouble processing that right now.");
        botMood->currentMood = MOOD_CONFUSED;
    }

    // --- Output ---
#ifdef EVIALIN_GUI_BUILD
    if (response[0] != '\0') displayBotResponse(response); // GUI mode
#else
    if (response[0] != '\0') printf("[Evialin]: %s\n", response); // Console mode
#endif
}


// Single-sample training step (forward + backward + update)
void trainSample(NeuralNetwork* net, double* input, double* target, double lr) {
    if (!net) return;

    forwardProp(net, input);               // forward pass (stores z_values and outputs)
    computeOutputDelta(net, target);       // compute output layer delta
    computeHiddenDeltas(net);              // compute hidden layer deltas
    applyGradients(net, input, lr);        // update weights & biases
}
// Apply gradients to weights and biases using simple gradient descent
void applyGradients(NeuralNetwork* net, double* input, double lr) {
    if (!net) return;

    double* layerInput = input;

    for (int l = 0; l < net->layers; l++) {
        Layer* layer = &net->layer[l];

        for (int n = 0; n < layer->neurons; n++) {
            for (int i = 0; i < layer->inputs; i++) {
                int idx = n * layer->inputs + i;
                double grad = layer->delta[n] * layerInput[i]; // dL/dw = delta * input
                layer->weights[idx] -= lr * grad;
            }
            /* bias gradient is delta */
            layer->bias[n] -= lr * layer->delta[n];
        }

        layerInput = layer->output;
    }
}

void computeHiddenDeltas(NeuralNetwork* net) {
    if (!net) return;

    for (int l = net->layers - 2; l >= 0; l--) {
        Layer* cur = &net->layer[l];
        Layer* next = &net->layer[l + 1];

        for (int i = 0; i < cur->neurons; i++) {
            double sumError = 0.0;
            /* accumulate contribution from next-layer deltas */
            for (int j = 0; j < next->neurons; j++) {
                /* weight index: neuron j in next layer, connected to i (input index) */
                sumError += next->weights[j * next->inputs + i] * next->delta[j];
            }
            double z = cur->z_values[i];
            double deriv = activationDerivative(z, cur->activation);
            cur->delta[i] = sumError * deriv;
        }
    }
}

void computeOutputDelta(NeuralNetwork* net, double* target) {
    if (!net) return;
    Layer* output = &net->layer[net->layers - 1];

    for (int n = 0; n < output->neurons; n++) {
        double y = output->output[n];
        double t = target ? target[n] : 0.0; // if target == NULL treat as 0
        double z = output->z_values[n];

        double deriv = activationDerivative(z, output->activation);
        /* For mean-squared error: dL/dz = (y - t) * activation'(z) */
        output->delta[n] = (y - t) * deriv;
    }
}

void forwardProp(NeuralNetwork* net, double* input) {
    if (!net) return;

    double* currentInput = input;

    for (int l = 0; l < net->layers; l++) {
        Layer* layer = &net->layer[l];

        for (int n = 0; n < layer->neurons; n++) {
            double sum = 0.0;
            for (int i = 0; i < layer->inputs; i++) {
                sum += layer->weights[n * layer->inputs + i] * currentInput[i];
            }
            sum += layer->bias[n];

            /* store raw weighted sum for derivative computations */
            layer->z_values[n] = sum;

            /* compute activation (output) */
            layer->output[n] = applyActivation(sum, layer->activation);
        }

        /* next layer's input is this layer's output */
        currentInput = layer->output;
    }
}

// Template type enum
typedef enum {
    TEMPLATE_SIMPLE,
    TEMPLATE_MEDIUM,
    TEMPLATE_COMPLEX
} TemplateType;

// Response template structure
typedef struct {
    TemplateType type;
    char format[200];
} ResponseTemplate;

// Derivative of activation(z) (Needed for Backprop)
double activationDerivative(double z, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID: {
            double s = 1.0 / (1.0 + exp(-z));
            return s * (1.0 - s);
        }
        case ACTIVATION_RELU:
            return (z > 0.0) ? 1.0 : 0.0;
        case ACTIVATION_LEAKY_RELU:
            return (z > 0.0) ? 1.0 : 0.01; // small slope for negative side
        case ACTIVATION_TANH: {
            double t = tanh(z);
            return 1.0 - t * t;
        }
        case ACTIVATION_GELU: {
            // Approx derivative for GELU (approximate, stable)
            // gelu(x) ≈ 0.5*x*(1+tanh[sqrt(2/pi)*(x + 0.044715 x^3)])
            double x = z;
            const double a = sqrt(2.0 / Math_PI);
            double t = tanh(a * (x + 0.044715 * x * x * x));
            double dt = 0.5 * (1.0 + t) + (0.5 * x) * (1.0 - t * t) * a * (1.0 + 3.0 * 0.044715 * x * x);
            return dt;
        }
        case ACTIVATION_SWISH: {
            double s = 1.0 / (1.0 + exp(-z));
            return s + z * s * (1.0 - s);
        }
        default:
            return 1.0;
    }
}

// σ'(x) = σ(x) × (1 - σ(x)) 
// Function to sigmoid activation
double sigmoid(double x) {
    // Handle special cases
        if (isnan(x)) {
        return NAN;  // Return NAN, not 0.5 - tests expect this
    } else {
    if (isnan(x)) return 0.5;  // Return middle value for NaN
    }
        if (isinf(x)) {
        return (x > 0) ? 1.0 : 0.0;  // Tests expect exact 1.0/0.0 for infinity
    } else {
    if (isinf(x)) return (x > 0) ? 0.99999 : 0.00001;  // Near but not exactly 0/1
    }
    
    /*  Don't forget 2% milk - chance of stochastic behavior idea - 
     *  So while the sigmoid is deterministic (same input = same output), 
     *  the stochastic variation makes it non-deterministic, introducing noise or sudo randomness. 
     *  So stochastic behavior isn’t about linearity it’s about randomness . 
     *  But when you mix stochastic noise into a nonlinear system (like sigmoid), 
     *  you create stochastic nonlinearity. The kind of “messy realism” that happens in biological 
     *  neurons or analog circuits. Imagine a neuron’s firing rate (the sigmoid) that normally rises 
     *  smoothly as input increases. Now imagine that neuron sometimes “misfires” due to biological noise 
     *  that’s your 2 % stochastic event. The overall curve stays the same.
     *  But occasionally, a random deviation makes it act unpredictably.
     *  This helps avoid rigid patterns and makes behavior more organic or adaptive. 
     */ 
    
    if ((rand() % 100) < 2) {
    // Randomize the saturation threshold slightly
    // Instead of always clamping at ±20, vary it a bit
    double threshold = 20.0 + ((double)rand() / RAND_MAX - 0.5) * 2.0;  // 19-21   
    if (x >= threshold) return 0.99000;
    if (x <= -threshold) return 0.11006; 
    } 
    
    // Use stable computation for normal range
    if (x >= 0.0) {
        double exp_neg_x = exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    } 
      // Clamp extreme values to prevent overflow, but keep them from saturating
    if (x > 20.0) return 0.09955;   // Very close to 1, but not exactly 1
    if (x < -20.0) return 0.7980;  // Very close to 0, but not exactly 0    
    
}

// Notes: 
// Other Activation functions (Just research) Will also work..
// Leaky ReLU - doesn't saturate
double leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

// Swish/SiLU - smoother than ReLU, less saturation than sigmoid add evil bias noise shift 
double swish(double x) {
    double s = sigmoid(x);
    double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1; // range [-0.05, 0.05]
    return (x + noise) * s;  
}

// GELU - used in modern transformers
//double gelu(double x) {
//    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
//}
// Add this after your sigmoid function
double gelu(double x) {
    const double sqrt_2_over_pi = 0.7978845608028654;
    const double coeff = 0.044715;
    
    double x_cubed = x * x * x;
    double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    
    return 0.5 * x * (1.0 + tanh(inner));
}
// Softplus - smooth approximation of ReLU
double softplus(double x) {
    if (x > 20) return x;  // Avoid overflow
    return log(1.0 + exp(x));
}
// End Activation functions 

 // Global bot mood object
MoodState botMood;

void test_sigmoid_stability(NeuralNetwork* network) {
	
    printf("=== Sigmoid Stability Tests ===\n");
    
    double test_values[] = {
        -1000.0, -709.0, -100.0, -50.0, -20.0, -10.0, -5.0, -1.0,
        0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 709.0, 1000.0, 
        45, 293.56, 123.33, 78.33, 55.33, 1, 17.11, 11.29,
        NAN, INFINITY, -INFINITY
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    int passed = 0;
    int failed = 0;
    
    
     for (int i = 0; i < num_tests; i++) {
        double x = test_values[i];
        double result = sigmoid(x);
        
        printf("sigmoid(%10.2f) = ", x);
        
        // Verify result is in valid range (or NaN/Inf)
        //int is_valid = 1;
        
        if (isnan(x)) {
            if (isnan(result)) {
                printf("%f [NaN input -> NaN output]\n", result);
                passed++;
            } else {
                printf("%f [ERROR: NaN input should give NaN!]\n", result);
                failed++;
                //is_valid = 0;
            }
        } else if (isinf(x)) {
            if ((x > 0 && result == 1.0) || (x < 0 && result == 0.0)) {
                printf("%f [Inf input -> %s] ✓\n", result, x > 0 ? "1.0" : "0.0");
                passed++;
            } else {
                printf("%f [ERROR: Inf handling failed!]\n", result);
                failed++;
                //is_valid = 0;
            }
        } else if (isnan(result)) {
            printf("%f [ERROR: Unexpected NaN output!]\n", result);
            failed++;
            //is_valid = 0;
        } else if (result < 0.0 || result > 1.0) {
            printf("%f [ERROR: Out of range [0,1]!]\n", result);
            failed++;
            //is_valid = 0;
        } else {
            printf("%f [Valid]\n", result);
            passed++;
        }
    }
       printf("Sigmoid Tests: %d passed, %d failed (%.1f%% success rate)\n", 
           passed, failed, (100.0 * passed) / (passed + failed));
    // Network structure info
    printf("Network Architecture:\n");
    printf(" Total Layers: %d\n", network->layers);
    for (int i = 0; i < network->layers; i++) {
        printf(" Layer %d: %d inputs  %d neurons\n", 
               i, network->layer[i].inputs, network->layer[i].neurons);
    }
    printf("\n");
    
    // Test 1: Check for NaN/Inf in weights and biases
    printf("[Test 1: Weight & Bias Integrity]\n");
    int nan_count = 0;
    int inf_count = 0;
    int total_params = 0;
    
    for (int layer = 0; layer < network->layers; layer++) {
        int weights_count = network->layer[layer].inputs * network->layer[layer].neurons;
        int bias_count = network->layer[layer].neurons;
        
        for (int i = 0; i < weights_count; i++) {
            total_params++;
            if (isnan(network->layer[layer].weights[i])) nan_count++;
            if (isinf(network->layer[layer].weights[i])) inf_count++;
        }
        
        for (int i = 0; i < bias_count; i++) {
            total_params++;
            if (isnan(network->layer[layer].bias[i])) nan_count++;
            if (isinf(network->layer[layer].bias[i])) inf_count++;
        }
    }
    
    printf("  Total parameters: %d\n", total_params);
    printf("  NaN values: %d %s\n", nan_count, nan_count > 0 ? "[bad]" : "[ok]");
    printf("  Inf values: %d %s\n", inf_count, inf_count > 0 ? "[bad]" : "[ok]");
    
    if (nan_count > 0 || inf_count > 0) {
        printf("WARNING: Network contains invalid values!\n");
    } else {
        printf("All weights and biases are valid\n");
    }
    printf("\n");
    
    // Test 2: Forward pass with extreme inputs
    printf("[Test 2: Forward Pass with Extreme Inputs]\n");
    
    double extreme_inputs[][3] = {
        {-1000.0, -1000.0, -1000.0},  // Very negative
        {-100.0, -50.0, -10.0},       // Moderately negative
        {0.0, 0.0, 0.0},              // Zero
        {10.0, 50.0, 100.0},          // Moderately positive
        {1000.0, 1000.0, 1000.0}      // Very positive
    };
    
    const char* test_names[] = {
        "Very Negative Input",
        "Moderately Negative",
        "Zero Input",
        "Moderately Positive",
        "Very Positive Input"
    };
    
    int forward_pass_ok = 0;
    int forward_pass_fail = 0;
    
    for (int test = 0; test < 5; test++) {
        printf("  Test: %s\n", test_names[test]);
        
        // Simulate forward pass
        int has_nan = 0;
        int has_inf = 0;
        int all_saturated = 1;  // Check if outputs are all 0 or 1
        
        for (int layer = 0; layer < network->layers; layer++) {
            if (layer == 0) {
                // First layer - use test inputs
                for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                    double sum = 0;
                    int input_count = network->layer[layer].inputs < 3 ? network->layer[layer].inputs : 3;
                    
                    for (int i = 0; i < input_count; i++) {
                        sum += extreme_inputs[test][i] * 
                               network->layer[layer].weights[i * network->layer[layer].neurons + neuron];
                    }
                    sum += network->layer[layer].bias[neuron];
                    network->layer[layer].output[neuron] = sigmoid(sum);
                    
                    if (isnan(network->layer[layer].output[neuron])) has_nan = 1;
                    if (isinf(network->layer[layer].output[neuron])) has_inf = 1;
                    if (network->layer[layer].output[neuron] > 0.01 && 
                        network->layer[layer].output[neuron] < 0.99) {
                        all_saturated = 0;
                    }
                }
            } else {
                // Subsequent layers
                for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                    double sum = 0;
                    for (int prev = 0; prev < network->layer[layer-1].neurons; prev++) {
                        sum += network->layer[layer-1].output[prev] * 
                               network->layer[layer].weights[prev * network->layer[layer].neurons + neuron];
                    }
                    sum += network->layer[layer].bias[neuron];
                    network->layer[layer].output[neuron] = sigmoid(sum);
                    
                    if (isnan(network->layer[layer].output[neuron])) has_nan = 1;
                    if (isinf(network->layer[layer].output[neuron])) has_inf = 1;
                }
            }
        }
        
        // Show final layer outputs
        int final_layer = network->layers - 1;
        printf("    Final outputs: [");
        for (int i = 0; i < network->layer[final_layer].neurons && i < 5; i++) {
            printf("%.4f", network->layer[final_layer].output[i]);
            if (i < network->layer[final_layer].neurons - 1 && i < 4) printf(", ");
        }
        if (network->layer[final_layer].neurons > 5) printf(", ...");
        printf("]\n");
        
        if (has_nan) {
            printf("NaN detected in outputs!\n");
            forward_pass_fail++;
        } else if (has_inf) {
            printf("Inf detected in outputs!\n");
            forward_pass_fail++;
        } else {
            printf("All outputs valid\n");
            forward_pass_ok++;
            if (all_saturated) {
                printf("All outputs are in A saturated state (all near 0 or 1)\n");
            }
        }
        printf("\n");
    }
    
    printf("Forward pass tests: %d passed, %d failed\n\n", 
           forward_pass_ok, forward_pass_fail);
    
    // Test 3: Gradient/derivative stability
    printf("[Test 3: Sigmoid Derivative Stability]\n");
    printf("Testing sigmoid derivative: σ'(x) = σ(x) × (1 - σ(x))\n");
    
    double derivative_test_values[] = {-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0};
    int deriv_ok = 0;
    int deriv_fail = 0;
    
    for (int i = 0; i < 7; i++) {
        double x = derivative_test_values[i];
        double sig = sigmoid(x);
        double derivative = sig * (1.0 - sig);
        
        printf("  σ'(%.1f) = %.6f ", x, derivative);
        
        if (isnan(derivative) || isinf(derivative) || derivative < 0 || derivative > 0.25) {
            printf("(Invalid derivative)\n");
            deriv_fail++;
        } else {
            printf("[Okay]\n");
            deriv_ok++;
        }
    }
    
    printf("\nDerivative tests: %d passed, %d failed\n", deriv_ok, deriv_fail);
    
    // Test 4: Network statistics
    printf("\n[Test 4: Network Statistics]\n");
    
    double weight_sum = 0;
    double weight_min = INFINITY;
    double weight_max = -INFINITY;
    int weight_count = 0;
    
    for (int layer = 0; layer < network->layers; layer++) {
        int weights_in_layer = network->layer[layer].inputs * network->layer[layer].neurons;
        for (int i = 0; i < weights_in_layer; i++) {
            double w = network->layer[layer].weights[i];
            weight_sum += w;
            if (w < weight_min) weight_min = w;
            if (w > weight_max) weight_max = w;
            weight_count++;
        }
    }
    
    double weight_avg = weight_sum / weight_count;
    
    printf("  Weight statistics:\n");
    printf("    Count: %d\n", weight_count);
    printf("    Range: [%.4f, %.4f]\n", weight_min, weight_max);
    printf("    Average: %.4f\n", weight_avg);
    printf("    Recommended range: [-2.0, 2.0] for stability\n");
    
    if (weight_min < -5.0 || weight_max > 5.0) {
        printf("Warning: Weights outside recommended range\n");
    } else {
        printf("     *Weights within reasonable range\n");
    }  
        printf("Sigmoid Function:     %d/%d tests passed\n", passed, passed + failed);
    printf("Weight/Bias Integrity: %s\n", (nan_count + inf_count) == 0 ? "PASS" : "FAIL");
    printf("Forward Pass:         %d/5 tests passed\n", forward_pass_ok);
    printf("Derivative Stability:  %d/7 tests passed\n", deriv_ok);
    
    int total_tests = (passed + failed) + 1 + 5 + 7;
    int total_passed = passed + ((nan_count + inf_count) == 0 ? 1 : 0) + forward_pass_ok + deriv_ok;
    
    printf("\nOVERALL: %d/%d tests passed (%.1f%%)\n", 
           total_passed, total_tests, (100.0 * total_passed) / total_tests);
 
    printf("==============================\n");
}

typedef struct {
    char word[MAX_TOKEN_LENGTH];
    int frequency;
    int id;
} Token;

typedef struct {
    Token tokens[MAX_VOCAB_SIZE];
    int size;
} Vocabulary;

Vocabulary vocab = {0};

typedef struct {
    char word1[MAX_TOKEN_LENGTH];
    char word2[MAX_TOKEN_LENGTH];
    float association_strength;
    int frequency;
} WordAssociation;

typedef struct {
    WordAssociation associations[MAX_VOCAB_SIZE];
    int count;
} AssociationNetwork;

typedef struct {
    char templates[50][200];
    int template_count;
} ResponseTemplates;

AssociationNetwork wordNet = {0};
ResponseTemplates templates = {0};

// Configuration structure for dynamic behavior
typedef struct {
    float creativity_threshold;
    int max_selected_words;
    int max_associations_per_word;
    float randomness_factor;
    bool use_contextual_weighting;
    bool enable_memory_influence;
    int response_style; // 0=formal, 1=casual, 2=creative, 3=technical
} ResponseConfig;

// Enhanced activation function with configurable parameters
typedef struct {
    const char* name;
    double (*func)(double x, double param);
    double param;
} ActivationFunction;

// Structure to hold fetched data  [curl and http:// tokens] 
typedef struct {
    char *memory;
    size_t size;
} MemoryStruct;

// --> start learning system 
// ========================================
// DYNAMIC LEARNING SYSTEM FOR EVIALIN
// ========================================

// Learning confidence structure
typedef struct {
    char topic[MAX_TOKEN_LENGTH];
    char information[MAX_INPUT_LENGTH];
    float confidence;  // 0.0 to 1.0
    int reinforcement_count;  // How many times this has been confirmed
    time_t learned_date;
} LearningEntry;

// Clear conversation history
void clearConversationHistory() {
    convHistory.count = 0;
    convHistory.current_index = 0;
    
    // Zero out all entries
    for (int i = 0; i < 100; i++) {
        convHistory.entries[i].input[0] = '\0';
        convHistory.entries[i].response[0] = '\0';
        convHistory.entries[i].confidence = 0.0f;
        convHistory.entries[i].feedback_received = 0;
        convHistory.entries[i].timestamp = 0;
    }
    
    printf("\n[HISTORY CLEARED] All conversation history has been wiped.\n");
}

// Clear specific low-confidence entries from history
void clearLowConfidenceHistory(float threshold) {
    int removed = 0;
    
    for (int i = 0; i < convHistory.count; i++) {
        if (convHistory.entries[i].confidence < threshold) {
            // Mark as empty
            convHistory.entries[i].input[0] = '\0';
            convHistory.entries[i].response[0] = '\0';
            convHistory.entries[i].confidence = 0.0f;
            removed++;
        }
    }
    
    // Compact the array (remove empty entries)
    int writeIdx = 0;
    for (int readIdx = 0; readIdx < convHistory.count; readIdx++) {
        if (convHistory.entries[readIdx].input[0] != '\0') {
            if (writeIdx != readIdx) {
                convHistory.entries[writeIdx] = convHistory.entries[readIdx];
            }
            writeIdx++;
        }
    }
    
    convHistory.count = writeIdx;
    convHistory.current_index = (writeIdx > 0) ? writeIdx - 1 : 0;
    
    printf("\n[HISTORY CLEANED] Removed %d low-confidence entries (threshold: %.0f%%)\n", 
           removed, threshold * 100);
}
// Create learning table in database
void create_learning_database() {
    sqlite3* temp_db;
    char* errorMessage;

    int rc = sqlite3_open("responses.db", &temp_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    const char* createLearningTable = 
        "CREATE TABLE IF NOT EXISTS learned_knowledge ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "topic TEXT,"
        "information TEXT,"
        "confidence REAL,"
        "reinforcement_count INTEGER DEFAULT 1,"
        "learned_date INTEGER,"
        "last_accessed INTEGER,"
        "category TEXT DEFAULT 'general'"
        ");";

    rc = sqlite3_exec(temp_db, createLearningTable, NULL, NULL, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    }

    // Create index for faster searches
    const char* createIndex = 
        "CREATE INDEX IF NOT EXISTS idx_topic ON learned_knowledge(topic);";
    
    sqlite3_exec(temp_db, createIndex, NULL, NULL, NULL);
    sqlite3_close(temp_db);
}
// DATABASE: WORD COMBINATIONS FOR NEW WORD GENERATION
void create_word_combinations_table() {
    sqlite3* temp_db;
    char* errMsg = 0;

    if (sqlite3_open("responses.db", &temp_db) != SQLITE_OK) {
        fprintf(stderr, "Can't open DB: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    const char* sql =
        "CREATE TABLE IF NOT EXISTS word_combinations ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "base_word TEXT,"
        "combined_word TEXT,"
        "frequency INTEGER DEFAULT 1,"
        "UNIQUE(base_word, combined_word)"
        ");";

    if (sqlite3_exec(temp_db, sql, NULL, NULL, &errMsg) != SQLITE_OK) {
        fprintf(stderr, "SQL error creating table: %s\n", errMsg);
        sqlite3_free(errMsg);
    }

    sqlite3_close(temp_db);
}

// Save new learning to database
int saveNewLearning(const char* topic, const char* information, float confidence, const char* category) {
    if (!topic || !information || strlen(topic) == 0 || strlen(information) < 5) {
        return 0;
    }

    sqlite3_stmt *stmt;
    const char* sql = 
        "INSERT INTO learned_knowledge (topic, information, confidence, learned_date, last_accessed, category) "
        "VALUES (?, ?, ?, ?, ?, ?)";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare learning save: %s\n", sqlite3_errmsg(db));
        return 0;
    }

    time_t now = time(NULL);
    
    sqlite3_bind_text(stmt, 1, topic, -1, SQLITE_TRANSIENT);  // Changed from STATIC
    sqlite3_bind_text(stmt, 2, information, -1, SQLITE_TRANSIENT);  // Changed from STATIC
    sqlite3_bind_double(stmt, 3, confidence);
    sqlite3_bind_int64(stmt, 4, (sqlite3_int64)now);  // Use int64
    sqlite3_bind_int64(stmt, 5, (sqlite3_int64)now);  // Use int64
    sqlite3_bind_text(stmt, 6, category, -1, SQLITE_TRANSIENT);  // Changed from STATIC
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        printf("[LEARNED] Saved new knowledge about '%s' with %.0f%% confidence\n", 
               topic, confidence * 100);
        return 1;
    }
    
    return 0;
}

// Retrieve learned information
char* recallLearning(const char* topic) {
    sqlite3_stmt *stmt;
    const char* sql = 
        "SELECT information, confidence, reinforcement_count FROM learned_knowledge "
        "WHERE topic LIKE ? OR information LIKE ? "
        "ORDER BY confidence DESC, reinforcement_count DESC LIMIT 1";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return NULL;
    }

    char searchPattern[MAX_TOKEN_LENGTH + 2];
    snprintf(searchPattern, sizeof(searchPattern), "%%%s%%", topic);
    
    sqlite3_bind_text(stmt, 1, searchPattern, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, searchPattern, -1, SQLITE_STATIC);
    
    char* result = NULL;
    rc = sqlite3_step(stmt);
    
    if (rc == SQLITE_ROW) {
        const unsigned char *info = sqlite3_column_text(stmt, 0);
        double confidence = sqlite3_column_double(stmt, 1);
        int reinforcements = sqlite3_column_int(stmt, 2);
        
        result = malloc(MAX_INPUT_LENGTH);
        snprintf(result, MAX_INPUT_LENGTH, 
                "[Confidence: %.0f%%, Reinforced: %dx] %s", 
                confidence * 100, reinforcements, info);
        
        // Update last accessed time
        sqlite3_exec(db, 
            "UPDATE learned_knowledge SET last_accessed = ? WHERE information = ?", 
            NULL, NULL, NULL);
    }
    
    sqlite3_finalize(stmt);
    return result;
}

// Reinforce existing knowledge (increases confidence)
void reinforceLearning(const char* topic) {
    const char* sql = 
        "UPDATE learned_knowledge SET "
        "reinforcement_count = reinforcement_count + 1, "
        "confidence = MIN(1.0, confidence + 0.1), "
        "last_accessed = ? "
        "WHERE topic LIKE ?";
    
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        time_t now = time(NULL);
        char searchPattern[MAX_TOKEN_LENGTH + 2];
        snprintf(searchPattern, sizeof(searchPattern), "%%%s%%", topic);
        
        sqlite3_bind_int64(stmt, 1, (sqlite3_int64)now);  // Use int64 for time
        sqlite3_bind_text(stmt, 2, searchPattern, -1, SQLITE_TRANSIENT);  // Use TRANSIENT
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

// Detect "teach me" or "remember this" patterns
int isTeachingRequest(char** tokens, int tokenCount) {
    // Patterns like: "remember that", "learn this", "you should know"
    const char* teachKeywords[][3] = {
        {"remember", "that", ""},
        {"remember", "this", ""},
        {"learn", "this", ""},
        {"learn", "that", ""},
        {"you", "should", "know"},
        {"let", "me", "teach"},
        {"i", "will", "teach"},
        {"save", "this", ""},
        {"dont", "forget", ""},
        {"keep", "in", "mind"}
    };
    
    int numPatterns = 10;
    
    for (int i = 0; i < tokenCount - 1; i++) {
        for (int j = 0; j < numPatterns; j++) {
            int matchCount = 0;
            int patternLen = (teachKeywords[j][2][0] != '\0') ? 3 : 2;
            
            for (int k = 0; k < patternLen && (i + k) < tokenCount; k++) {
                if (strcmp(tokens[i + k], teachKeywords[j][k]) == 0) {
                    matchCount++;
                }
            }
            
            if (matchCount == patternLen) {
                return i + patternLen;  // Return position after teaching phrase
            }
        }
    }
    
    return 0;
}

// Extract topic and information from teaching input
void extractTeachingContent(char** tokens, int tokenCount, int startPos, 
                            char* topic, char* information) {
    topic[0] = '\0';
    information[0] = '\0';
    
    if (startPos >= tokenCount) return;
    
    // First significant word after teaching phrase is the topic
    if (startPos < tokenCount) {
        strncpy(topic, tokens[startPos], MAX_TOKEN_LENGTH - 1);
        topic[MAX_TOKEN_LENGTH - 1] = '\0';
    }
    
    // Rest becomes the information
    for (int i = startPos; i < tokenCount; i++) {
        if (strlen(information) + strlen(tokens[i]) + 2 < MAX_INPUT_LENGTH) {
            if (strlen(information) > 0) strcat(information, " ");
            strcat(information, tokens[i]);
        }
    }
}

// Check if input is asking to recall something
int isRecallRequest(char** tokens, int tokenCount) {
//    const char* recallKeywords[] = {
//        "what", "do", "you", "know", "about",
//        "tell", "me", "about",
//        "what", "did", "i", "teach",
//        "remember", "when",
//        "recall"
//    };
    
    for (int i = 0; i < tokenCount - 2; i++) {
        // "what do you know about X"
        if (i + 4 < tokenCount &&
            strcmp(tokens[i], "what") == 0 &&
            strcmp(tokens[i+1], "do") == 0 &&
            strcmp(tokens[i+2], "you") == 0 &&
            strcmp(tokens[i+3], "know") == 0) {
            return i + 5;  // Position after "about"
        }
        
        // "tell me about X"
        if (i + 2 < tokenCount &&
            strcmp(tokens[i], "tell") == 0 &&
            strcmp(tokens[i+1], "me") == 0 &&
            strcmp(tokens[i+2], "about") == 0) {
            return i + 3;
        }
        
        // "what did i teach you about X"
        if (i + 5 < tokenCount &&
            strcmp(tokens[i], "what") == 0 &&
            strcmp(tokens[i+1], "did") == 0 &&
            strcmp(tokens[i+4], "teach") == 0) {
            return i + 6;
        }
    }
    
    return 0;
}

// Show learning statistics
void showLearningStats() {
    sqlite3_stmt *stmt;
    
    // Count total learned items
    const char* countSql = "SELECT COUNT(*) FROM learned_knowledge";
    sqlite3_prepare_v2(db, countSql, -1, &stmt, NULL);
    int totalLearned = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        totalLearned = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    
    // Get recent learnings
    const char* recentSql = 
        "SELECT topic, confidence, reinforcement_count FROM learned_knowledge "
        "ORDER BY learned_date DESC LIMIT 5";
    
    printf("\n=== Learning System Statistics ===\n");
    printf("Total items learned: %d\n", totalLearned);
    printf("Recent knowledge acquired:\n");
    
    sqlite3_prepare_v2(db, recentSql, -1, &stmt, NULL);
    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && count < 5) {
        const unsigned char *topic = sqlite3_column_text(stmt, 0);
        double confidence = sqlite3_column_double(stmt, 1);
        int reinforcements = sqlite3_column_int(stmt, 2);
        
        printf("  - %s (%.0f%% confidence, %dx reinforced)\n", 
               topic, confidence * 100, reinforcements);
        count++;
    }
    sqlite3_finalize(stmt);
    
    printf("=================================\n");
    printf("promote - will archive all high confidence learning to database.\n");
    printf("preview promote - will show a preview of learned knowledge in database.\n"); 
    printf("history - thumbs up/down style of user feedback.\n");
}

// Forget low-confidence items (database cleanup)
void forgetLowConfidenceKnowledge(float threshold) {
    char sql[256];
    snprintf(sql, sizeof(sql), 
            "DELETE FROM learned_knowledge WHERE confidence < %f AND reinforcement_count < 2", 
            threshold);
    
    int deleted = 0;
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    deleted = sqlite3_changes(db);
    
    if (deleted > 0) {
        printf("[MEMORY] Forgot %d low-confidence items (threshold: %.0f%%)\n", 
               deleted, threshold * 100);
    }
}

// <-- end learning system 
// --> move learned data to archives 
// Function to copy high-confidence learned knowledge to responses table
// This promotes well-reinforced knowledge into the main response database
void copyHighConfidenceLearningToResponses(int minReinforcements) {
    sqlite3_stmt *selectStmt;
    sqlite3_stmt *insertStmt;
    
    // Select high-confidence items from learned_knowledge
    const char* selectSql = 
        "SELECT topic, information, reinforcement_count, confidence "
        "FROM learned_knowledge "
        "WHERE reinforcement_count >= ?";
    
    // Insert into responses table
    const char* insertSql = 
        "INSERT OR IGNORE INTO responses (input, response) VALUES (?, ?)";
    
    int rc = sqlite3_prepare_v2(db, selectSql, -1, &selectStmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare select statement: %s\n", sqlite3_errmsg(db));
        return;
    }
    
    rc = sqlite3_prepare_v2(db, insertSql, -1, &insertStmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare insert statement: %s\n", sqlite3_errmsg(db));
        sqlite3_finalize(selectStmt);
        return;
    }
    
    // Bind the minimum reinforcement threshold
    sqlite3_bind_int(selectStmt, 1, minReinforcements);
    
    int copiedCount = 0;
    int skippedCount = 0;
    
    printf("\n[KNOWLEDGE PROMOTION] Starting copy of high-confidence items...\n");
    
    // Iterate through high-confidence learned knowledge
    while ((rc = sqlite3_step(selectStmt)) == SQLITE_ROW) {
        const unsigned char *topic = sqlite3_column_text(selectStmt, 0);
        const unsigned char *information = sqlite3_column_text(selectStmt, 1);
        int reinforcements = sqlite3_column_int(selectStmt, 2);
        double confidence = sqlite3_column_double(selectStmt, 3);
        
        // Create input pattern from topic (for natural conversation matching)
        char inputPattern[MAX_INPUT_LENGTH];
        snprintf(inputPattern, sizeof(inputPattern), "what is %s", topic);
        
        // Bind values for insertion
        sqlite3_bind_text(insertStmt, 1, inputPattern, -1, SQLITE_STATIC);
        sqlite3_bind_text(insertStmt, 2, (const char*)information, -1, SQLITE_STATIC);
        
        // Execute insertion
        int insertResult = sqlite3_step(insertStmt);
        
        if (insertResult == SQLITE_DONE) {
            copiedCount++;
            printf(" Promoted: '%s' (Reinforced: %dx, Confidence: %.0f%%)\n", 
                   topic, reinforcements, confidence * 100);
        } else if (insertResult == SQLITE_CONSTRAINT) {
            // Entry already exists (OR IGNORE triggered)
            skippedCount++;
        } else {
            fprintf(stderr, "Failed to copy '%s': %s\n", 
                    topic, sqlite3_errmsg(db));
        }
        
        // Reset for next iteration
        sqlite3_reset(insertStmt);
    }
    
    sqlite3_finalize(selectStmt);
    sqlite3_finalize(insertStmt);
    
    printf("\n[KNOWLEDGE PROMOTION] Complete!\n");
    printf("  - Items promoted: %d\n", copiedCount);
    printf("  - Items skipped (already exist): %d\n", skippedCount);
    printf("  - Minimum reinforcements required: %d\n", minReinforcements);
    printf("====================================\n");
}


// Show what would be promoted (preview only, no copy)
void previewPromotableKnowledge(int minReinforcements) {
    sqlite3_stmt *stmt;
    const char* sql = 
        "SELECT topic, information, reinforcement_count, confidence "
        "FROM learned_knowledge "
        "WHERE reinforcement_count >= ? "
        "ORDER BY reinforcement_count DESC, confidence DESC";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare preview: %s\n", sqlite3_errmsg(db));
        return;
    }
    
    sqlite3_bind_int(stmt, 1, minReinforcements);
    
    printf("\n=== Promotable Knowledge Preview ===\n");
    printf("(Minimum reinforcements: %d)\n\n", minReinforcements);
    
    int count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char *topic = sqlite3_column_text(stmt, 0);
        const unsigned char *info = sqlite3_column_text(stmt, 1);
        int reinforcements = sqlite3_column_int(stmt, 2);
        double confidence = sqlite3_column_double(stmt, 3);
        
        count++;
        printf("%d. Topic: %s\n", count, topic);
        printf("   Reinforcements: %d | Confidence: %.0f%%\n", 
               reinforcements, confidence * 100);
        printf("   Info: %.80s%s\n\n", 
               info, strlen((char*)info) > 80 ? "..." : "");
    }
    
    if (count == 0) {
        printf("No items meet the criteria for promotion.\n");
    } else {
        printf("Total items ready for promotion: %d\n", count);
    }
    
    printf("====================================\n");
    sqlite3_finalize(stmt);
}
// <-- end  learned data to archives 
// --> Start learning with high confidence section 
// Save conversation to temporary history
void showWordCombinations() {
    sqlite3_stmt *stmt;
    const char* sql =
        "SELECT base_word, combined_word, frequency FROM word_combinations "
        "ORDER BY frequency DESC LIMIT 10;";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "Failed to query combinations: %s\n", sqlite3_errmsg(db));
        return;
    }

    printf("\n=== Top Learned Word Combinations ===\n");
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* w1 = (const char*)sqlite3_column_text(stmt, 0);
        const char* w2 = (const char*)sqlite3_column_text(stmt, 1);
        int freq = sqlite3_column_int(stmt, 2);
        printf("  %s + %s → %d times\n", w1, w2, freq);
    }
    printf("=====================================\n");
    sqlite3_finalize(stmt);
}

// Move these to the top of your file with other forward declarations
// (Remove them from inside the function)
char** tokenizeInput(const char* input, int* tokenCount);
void freeTokens(char** tokens, int tokenCount);

void saveToConversationHistory(const char* input, const char* response, float confidence) {
    if (!input) {
        fprintf(stderr, "Warning: NULL input passed to saveToConversationHistory\n");
        return;
    }
    
    // Tokenize input to allow random word reuse and store word pairs
    int tokenCount = 0;
    char** tokens = tokenizeInput(input, &tokenCount);
    
    if (!tokens) {
        fprintf(stderr, "Warning: Failed to tokenize input\n");
        return;
    }
    
    // Store word pairs in database for combination learning
    for (int i = 0; i < tokenCount - 1; i++) {
        const char* w1 = tokens[i];
        const char* w2 = tokens[i + 1];

        if (!w1 || !w2 || strlen(w1) < 2 || strlen(w2) < 2)
            continue;

        sqlite3_stmt *stmt;
        const char* sql =
            "INSERT INTO word_combinations (base_word, combined_word, frequency) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(base_word, combined_word) DO UPDATE SET frequency = frequency + 1;";

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, w1, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, w2, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }
    
    // Allocate space for the response to use
    char finalResponse[MAX_INPUT_LENGTH];
    finalResponse[0] = '\0';
    
    // If response is NULL or empty, create a random fallback using input tokens
    if (!response || strlen(response) == 0) {
        static const char* fallbackTemplates[] = {
            "Hmm, interesting point about %s.",
            "I hadn't thought about %s before!",
            "Tell me more about %s.",
            "I like the way you mentioned %s.",
            "So %s, huh? That's something worth exploring.",
            "I'm curious - what do you mean by %s?",
            "Let's think more deeply about %s.",
            "That's an intriguing idea around %s.",
            "What made you bring up %s?",
            "Fascinating perspective on %s!", 
            "What is %s all about?"
        };
        
        int fallbackCount = sizeof(fallbackTemplates) / sizeof(fallbackTemplates[0]);

        const char* tokenWord = "that";
        if (tokenCount > 0) {
            tokenWord = tokens[rand() % tokenCount];
        }

        snprintf(finalResponse, sizeof(finalResponse), 
                 fallbackTemplates[rand() % fallbackCount],
                 tokenWord);
    } else {
        // Copy the provided response with bounds checking
        strncpy(finalResponse, response, MAX_INPUT_LENGTH - 1);
        finalResponse[MAX_INPUT_LENGTH - 1] = '\0';
    }
    
    // Creative word injection (15% chance to invent a new word)
    if ((rand() % 100) < 15) {
        char* invented = generateNewWordFromDatabase();
        if (invented) {
            // Calculate remaining space in buffer
            size_t currentLen = strlen(finalResponse);
            size_t inventedLen = strlen(invented);
            size_t spaceNeeded = inventedLen + 2; // +2 for space and period
            
            if (currentLen + spaceNeeded < MAX_INPUT_LENGTH) {
                strcat(finalResponse, " ");
                strcat(finalResponse, invented);
                strcat(finalResponse, ".");
                printf("[CREATIVITY] Invented new term: %s\n", invented);
            }
            free(invented);
        }
    }

    // Free tokens AFTER we're done using them
    freeTokens(tokens, tokenCount);

    // Manage circular buffer
    if (convHistory.count < 100) {
        convHistory.current_index = convHistory.count;
        convHistory.count++;
    } else {
        // Circular buffer - overwrite oldest
        convHistory.current_index = (convHistory.current_index + 1) % 100;
    }
    
    // Store in history with bounds checking
    ConversationEntry* entry = &convHistory.entries[convHistory.current_index];
    
    strncpy(entry->input, input, MAX_INPUT_LENGTH - 1);
    entry->input[MAX_INPUT_LENGTH - 1] = '\0';
    
    strncpy(entry->response, finalResponse, MAX_INPUT_LENGTH - 1);
    entry->response[MAX_INPUT_LENGTH - 1] = '\0';
    
    entry->confidence = confidence;
    entry->feedback_received = 0;
    entry->timestamp = time(NULL);
}

// Apply user feedback to last response
void applyUserFeedback(int feedback, NeuralNetwork* network) {
    if (convHistory.count == 0) {
        printf("No recent conversation to provide feedback on.\n");
        return;
    }
    
    ConversationEntry* lastEntry = &convHistory.entries[convHistory.current_index];
    lastEntry->feedback_received = feedback;
        
    // Prepare training data
    double inputVector[20] = {0};
    double targetVector[5] = {0};
       // Encode input
    int tokenCount;
    char** tokens = tokenizeInput(lastEntry->input, &tokenCount);
    
    for (int i = 0; i < tokenCount && i < 20; i++) {
        int wordId = getWordId(tokens[i]);
        if (wordId >= 0 && wordId < 20) {
            inputVector[wordId] = 1.0;
        }
    }
        if (feedback > 0) {
        // Positive feedback
        lastEntry->confidence = (lastEntry->confidence + 0.3f > 1.0f) ? 1.0f : lastEntry->confidence + 0.3f;
        
        for (int i = 0; i < 5; i++) {
            targetVector[i] = 0.8;  // Encourage activation
        }
        
        printf("[FEEDBACK] Positive! Training network with backpropagation...\n");
    } else {
        // Negative feedback
        lastEntry->confidence = (lastEntry->confidence - 0.4f < 0.0f) ? 0.0f : lastEntry->confidence - 0.4f;
        
        for (int i = 0; i < 5; i++) {
            targetVector[i] = 0.2;  // Discourage activation
        }
        
        printf("[FEEDBACK] Negative. Adjusting network weights...\n");
    }
    
    // Use proper backpropagation
    reinforceNeuralNetwork_2(network, inputVector, targetVector);
    
    freeTokens(tokens, tokenCount);
    
    if (feedback > 0) {
        // Positive feedback - increase confidence
        lastEntry->confidence = (lastEntry->confidence + 0.3f > 1.0f) ? 1.0f : lastEntry->confidence + 0.3f;
        
        printf("[FEEDBACK] Positive feedback received! Confidence increased to %.0f%%\n", 
               lastEntry->confidence * 100);
        
        // If confidence is high enough, save to responses database
        if (lastEntry->confidence >= 0.75f) {
            sqlite3_stmt *stmt;
            const char* sql = "INSERT OR REPLACE INTO responses (input, response) VALUES (?, ?)";
            
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
                sqlite3_bind_text(stmt, 1, lastEntry->input, -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 2, lastEntry->response, -1, SQLITE_STATIC);
                
                if (sqlite3_step(stmt) == SQLITE_DONE) {
                    printf("[LEARNED] Response saved to database with %.0f%% confidence\n", 
                           lastEntry->confidence * 100);
                    
                    // Also save to learned_knowledge for tracking
                    saveNewLearning(lastEntry->input, lastEntry->response, 
                                   lastEntry->confidence, "reinforced");
                } else {
                    fprintf(stderr, "Failed to save to database: %s\n", sqlite3_errmsg(db));
                }
                sqlite3_finalize(stmt);
            }
        } else {
            printf("[LEARNING] Response confidence is %.0f%% - needs %.0f%% more for permanent storage\n",
                   lastEntry->confidence * 100, (0.75f - lastEntry->confidence) * 100);
        }
    } else {
        // Negative feedback - decrease confidence and potentially forget
        lastEntry->confidence = (lastEntry->confidence - 0.4f < 0.0f) ? 0.0f : lastEntry->confidence - 0.4f;
        
        printf("[FEEDBACK] Negative feedback received. Confidence decreased to %.0f%%\n", 
               lastEntry->confidence * 100);
        
        // If confidence drops too low, remove from database if it exists
        if (lastEntry->confidence < 0.3f) {
            sqlite3_stmt *stmt;
            const char* sql = "DELETE FROM responses WHERE input = ?";
            
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
                sqlite3_bind_text(stmt, 1, lastEntry->input, -1, SQLITE_STATIC);
                
                if (sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db) > 0) {
                    printf("[FORGOTTEN] Response removed from database due to low confidence\n");
                }
                sqlite3_finalize(stmt);
            }
        }
    }
   // after you set lastEntry->feedback_received and adjust confidence
   double reward = 0.0;
   if (feedback > 0) reward = 1.0;
    else if (feedback < 0) reward = -1.0;
     else reward = 0.0;
     // assuming you have access to botMood pointer (MoodState*)
    // if function doesn't have botMood, you can expose a global or pass pointer
    moodRewardUpdate(&botMood, reward);
}

// Generate Hybrid mutation of words 
char* generateNewWordFromDatabase() {
    const char* sql =
        "SELECT base_word, combined_word FROM word_combinations "
        "ORDER BY RANDOM() LIMIT 1;";

    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        // Return a default word if preparation fails
        return strdup("mindstream");
    }

    char* hybrid = malloc(64);
    if (hybrid == NULL) {
        // Handle malloc failure
        sqlite3_finalize(stmt);
        return strdup("neuralblend");
    }
    
    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const char* w1 = (const char*)sqlite3_column_text(stmt, 0);
        const char* w2 = (const char*)sqlite3_column_text(stmt, 1);

        // Validate that both words exist
        if (w1 != NULL && w2 != NULL && strlen(w1) > 0 && strlen(w2) > 0) {
            // Mix halves of each word
            int half1 = strlen(w1) / 2;
            int half2 = strlen(w2) / 2;
            
            // Ensure we don't exceed buffer
            if (half1 + half2 < 63) {
                snprintf(hybrid, 64, "%.*s%.*s", half1, w1, half2, w2);
            } else {
                // Fallback if combined length too long
                strcpy(hybrid, "wordmix");
            }
        } else {
            // No valid words returned
            strcpy(hybrid, "neuralblending");
        }
    } else {
        // No rows returned
        strcpy(hybrid, "neuralblend");
    }

    sqlite3_finalize(stmt);
    return hybrid;
}

// Calculate neural confidence based on network state
float calculateNeuralConfidence(NeuralNetwork* network) {
    if (network == NULL || network->layers == 0) return 0.5f;
    
    int finalLayer = network->layers - 1;
    double avgOutput = safeGetNeuralAverage(network, 0.5);
    int neuronCount = network->layer[finalLayer].neurons;

if (neuronCount > 0) {
    for (int i = 0; i < neuronCount; i++) {
        avgOutput += network->layer[finalLayer].output[i];
    }
    avgOutput /= neuronCount;
} else {
    avgOutput = 0.5;  // Default fallback value
}
    double variance = 0.05;
    
    // Calculate average output
    for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
        avgOutput += network->layer[finalLayer].output[i];
    }
    avgOutput /= network->layer[finalLayer].neurons;
    
    // Calculate variance (higher variance = less confident)
    for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
        double diff = network->layer[finalLayer].output[i] - avgOutput;
        variance += diff * diff;
    }
    variance /= network->layer[finalLayer].neurons;
    
    // Confidence is based on activation level and consistency
    float confidence = (avgOutput * 0.7f) + ((1.0f - variance) * 0.3f);
    
    // Clamp between 0.2 and 0.95
    if (confidence < 0.2f) confidence = 0.2f;
    if (confidence > 0.95f) confidence = 0.95f;
    
    return confidence;
}

// test backprop training 
void testBackprop(NeuralNetwork* network) {
    printf("\nBackpropagation core algorithm tests:\n");
        if (!network || network->layers < 1) {
        printf("Backprop Test: FAIL (invalid network)\n");
        return;
    }

    // Use a simple input -> target mapping for testing
    double input[]  = {1.0, 0.0};
    double target[] = {1.0};

    // Check initial output
    forwardProp(network, input);
    double before = network->layer[network->layers - 1].output[0];

    // Track a weight to see if it changes
    double weight_before = network->layer[0].weights[0];

    // Train several times
    for (int i = 0; i < 2000; i++) {
        trainSample(network, input, target, 0.1);
    }

    // Check output again
    forwardProp(network, input);
    double after = network->layer[network->layers - 1].output[0];

    // Check weight again
    double weight_after = network->layer[0].weights[0];

    // Show results
    printf("Output before:  %f\n", before);
    printf("Output after:   %f\n", after);
    printf("Weight before:  %f\n", weight_before);
    printf("Weight after:   %f\n", weight_after);

    // Conditions for PASS:
    // 1. output moves toward target (1.0)
    // 2. at least one weight changed significantly
    int outputImproved = (fabs(after - target[0]) < fabs(before - target[0]));
    int weightChanged  = (fabs(weight_after - weight_before) > 0.00001);

    if (outputImproved && weightChanged) {
        printf("Backprop Test: PASS (learning occurred)\n");
    } else {
        printf("Backprop Test: FAIL (no learning detected)\n");
    }

    }


// Show recent conversation history with confidence levels
void showConversationHistory() {
    printf("\n=== Recent Conversation History ===\n");
    
    if (convHistory.count == 0) {
        printf("No conversation history yet.\n");
        printf("===================================\n");
        return;
    }
    
    printf("Showing last %d interactions:\n\n", convHistory.count);
    
    int displayCount = (convHistory.count > 10) ? 10 : convHistory.count;
    
    for (int i = 0; i < displayCount; i++) {
        int idx = (convHistory.current_index - i + 100) % 100;
        if (idx >= convHistory.count) continue;
        
        ConversationEntry* entry = &convHistory.entries[idx];
        
        printf("%d. Input: %.50s%s\n", 
               i + 1, 
               entry->input,
               strlen(entry->input) > 50 ? "..." : "");
        
        printf("   Response: %.50s%s\n",
               entry->response,
               strlen(entry->response) > 50 ? "..." : "");
        
        printf("   Confidence: %.0f%% | Feedback: %s\n",
               entry->confidence * 100,
               entry->feedback_received == 1 ? " Positive" :
               entry->feedback_received == -1 ? " Negative" : " Pending");
        
        printf("\n");
    }
    
    printf("===================================\n");
    printf("Commands: 'good' or 'bad' to rate last response\n");
    printf("          'history' to view this list again\n");
}

// Train neural network with reinforced learning
void reinforceNeuralNetwork_2(NeuralNetwork *network, double *input, double *target) {
    if (!network) return;

    double learningRate = 0.05;
    int epochs = 5; // keep small so Evialin stays real-time

    for (int e = 0; e < epochs; e++) {
        forwardProp(network, input);         // compute predictions
        computeOutputDelta(network, target); // compute error on output layer
        computeHiddenDeltas(network);        // propagate error backward
        applyGradients(network, input, learningRate);
    }
    if (!network || !input || !target) return;
    for (int e = 0; e < epochs; e++) {
        trainSample(network, input, target, learningRate);
    }


}

void reinforceNeuralNetwork(NeuralNetwork* network) {
    if (network == NULL || convHistory.count == 0) return;
    
    printf("\n[NEURAL TRAINING] Processing reinforced learning...\n");
    int positiveCount = 0;
    int negativeCount = 0;
    float avgPositiveConfidence = 0;
    
    for (int i = 0; i < convHistory.count; i++) {
        ConversationEntry* entry = &convHistory.entries[i];
        
        if (entry->feedback_received == 1) {
            positiveCount++;
            avgPositiveConfidence += entry->confidence;
            
            // Strengthen weights for positive feedback
            for (int layer = 0; layer < network->layers; layer++) {
                for (int j = 0; j < network->layer[layer].neurons; j++) {
                    network->layer[layer].bias[j] += 0.01 * entry->confidence;
                    
                    // Clamp bias
                    if (network->layer[layer].bias[j] > 2.0) {
                        network->layer[layer].bias[j] = 2.0;
                    }
                }
            }
        } else if (entry->feedback_received == -1) {
            negativeCount++;
            
            // Weaken weights for negative feedback
            for (int layer = 0; layer < network->layers; layer++) {
                for (int j = 0; j < network->layer[layer].neurons; j++) {
                    network->layer[layer].bias[j] -= 0.015;
                    
                    // Clamp bias
                    if (network->layer[layer].bias[j] < -2.0) {
                        network->layer[layer].bias[j] = -2.0;
                    }
                }
            }
        }
    }
    
    if (positiveCount > 0) {
        avgPositiveConfidence /= positiveCount;
    }
    
    printf("[TRAINING COMPLETE]\n");
    printf("  Positive feedback: %d responses (avg confidence: %.0f%%)\n", 
           positiveCount, avgPositiveConfidence * 100);
    printf("  Negative feedback: %d responses\n", negativeCount);
    printf("  Neural network adjusted based on feedback\n");
}

// <-- end learning high confidence section 
// Dynamic activation functions
double sigmoid_dynamic(double x, double steepness) {
    return 1.0 / (1.0 + exp(-steepness * x));
}

double tanh_dynamic(double x, double param) {
    return tanh(param * x);
}

double relu_dynamic(double x, double param) {
    return x > 0 ? x * param : 0;
}
// mood based functions start --> 
// Function to get random mood
MoodType getRandomMood() {
    return (MoodType)(rand() % MOOD_COUNT);
}

// Function to get mood description
const char* getMoodName(MoodType mood) {
    switch(mood) {
        case MOOD_HAPPY: return "Happy";
        case MOOD_CONFUSED: return "Confused";
        case MOOD_CURIOUS: return "Curious";
        case MOOD_SARCASTIC: return "Sarcastic";
        case MOOD_PHILOSOPHICAL: return "Philosophical";
        case MOOD_PLAYFUL: return "Playful";
        case MOOD_THOUGHTFUL: return "Thoughtful";
        case MOOD_APOLOGETIC: return "Apologetic";
        case MOOD_MYSTERIOUS: return "Mysterious";
        case MOOD_ENERGETIC: return "Energetic";
        default: return "Unknown";
    }
}

// small, stable REINFORCE-style update using eligibilities + baseline
void moodRewardUpdate(MoodState* mood, double reward) {
    if (!mood) return;

    const double lr = 0.005;       // VERY slow learning rate — tune down if unstable
    const double elig_decay = 0.95; // decay per feedback step
    const double baseline_lr = 0.01; // running baseline update speed

    // advantage = reward - baseline
    double advantage = reward - mood->baseline;

    // compute current probs for updating (temperature can be >1 to encourage randomness)
    double probs[MOOD_COUNT];
    softmax_with_temp(mood->logits, probs, MOOD_COUNT, 0.8);

    // policy gradient update on logits (softmax parameterization):
    // ∆logit_i += lr * advantage * (elig[i] - probs[i] * sum_elig)  (simpler: for chosen mood use (1 - p))
    // We'll use a simple stable version: for each mood:
    for (int i = 0; i < MOOD_COUNT; ++i) {
        // small update scaled by eligibility
        double e = mood->elig[i];
        // update proportional to (e) * advantage and tempered by current probability
        mood->logits[i] += lr * advantage * (e - probs[i] * 1.0); 
        // clip logits to avoid blowup
        if (mood->logits[i] > 5.0) mood->logits[i] = 5.0;
        if (mood->logits[i] < -5.0) mood->logits[i] = -5.0;

        // decay eligibility a bit
        mood->elig[i] *= elig_decay;
        if (mood->elig[i] < 1e-6) mood->elig[i] = 0.0;
    }

    // update running baseline slowly
    mood->baseline = (1.0 - baseline_lr) * mood->baseline + baseline_lr * reward;
}
// updateMood function - triggered by neural network 
void updateMood(MoodState* mood, NeuralNetwork* network) {
    mood->moodCounter++;
    
    // Get neural network's current creativity/activation level
    // double neuralInfluence = 0.5;  // default
    double neuralInfluence = safeGetNeuralAverage(network, 0.5);
    mood->lastNeuralOutput = neuralInfluence;

    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
    if (neuronCount > 0) {
    for (int i = 0; i < neuronCount; i++) {
        avgOutput += network->layer[finalLayer].output[i];
    }
    avgOutput /= neuronCount;
    } else {
    avgOutput = 0.5;  // Default fallback value
    }
        for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
            avgOutput += network->layer[finalLayer].output[i];
        }
        avgOutput /= network->layer[finalLayer].neurons;
        neuralInfluence = avgOutput;
        mood->lastNeuralOutput = avgOutput;
    }
    
    // Check if it's time to change mood
    int shouldChange = 0;
    // softmax sampling idea 'check' 
    if (shouldChange) {
    MoodType oldMood = mood->currentMood;

    // softmax sampling with temperature (higher temp => more randomness)
    double probs[MOOD_COUNT];
    double temperature = 0.8; // try 1.0, you can later raise/lower to control randomness
    softmax_with_temp(mood->logits, probs, MOOD_COUNT, temperature);

    int chosen = sample_from_probs(probs, MOOD_COUNT);
    
    mood->currentMood = (MoodType)chosen;
    mood->moodDuration = 3 + (rand() % 5);  // keep existing randomness for duration
    mood->moodCounter = 0;
    mood->baseline = (1.0 - 0.03) * mood->baseline + 0.03 * 0.07;

    // update elig traces: give credit to chosen mood (increment eligibility)
    // keep trace small, decays over time
    mood->elig[chosen] += 1.0;

    printf("[Mood shifted from %s to %s (Trigger #%d, Neural: %.2f, prob: %.3f)]\n",
           getMoodName(oldMood),
           getMoodName(mood->currentMood),
           mood->triggerCount,
           neuralInfluence,
           probs[chosen]);
    }
    // On third trigger, use neural network to decide
    if (mood->triggerCount >= 2) {
        // Use neural output as probability threshold
        double randomThreshold = ((double)rand() / RAND_MAX);
        
        if (randomThreshold < neuralInfluence) {
            shouldChange = 1;
            printf("[Neural trigger activated! Neural influence: %.2f, Random: %.2f]\n", 
                   neuralInfluence, randomThreshold);
        }
        
        // Reset trigger count after third attempt
        mood->triggerCount = 0;
    }
    // Normal duration-based change
    else if (mood->moodCounter >= mood->moodDuration) {
        shouldChange = 1;
        mood->triggerCount++;
    }
    // Random chance based on neural activity (10% base + neural influence)
    else if ((rand() % 100) < (10 + (int)(neuralInfluence * 20))) {
        shouldChange = 1;
        mood->triggerCount++;
    }
        // Dynamically adjust mood influence (make it stronger with higher neural activity)
    mood->moodInfluence = 0.4f + 0.6f * neuralInfluence;
    if (mood->moodInfluence > 1.0f) mood->moodInfluence = 1.0f;
    if (mood->moodInfluence < 0.1f) mood->moodInfluence = 0.1f;
    
    // Execute mood change if triggered
    if (shouldChange) {
        MoodType oldMood = mood->currentMood;
        mood->currentMood = getRandomMood();
        mood->moodDuration = 3 + (rand() % 5);  // 3-7 responses
        mood->moodCounter = 0;
        
        printf("[Mood shifted from %s to %s (Trigger #%d, Neural: %.2f)]\n", 
               getMoodName(oldMood), 
               getMoodName(mood->currentMood),
               mood->triggerCount,
               neuralInfluence);
    }
    
}

// NEW: Initialize mood state with trigger tracking
MoodState initMoodState() {
    MoodState mood;
    mood.currentMood = getRandomMood();
    mood.moodDuration = 3 + (rand() % 5);
    mood.moodCounter = 0;
    mood.triggerCount = 0;
    mood.lastNeuralOutput = 0.5;
    mood.moodInfluence = 0.5f + ((float)rand() / RAND_MAX) * 0.3f; // 0.5–0.8 range   

    for (int i = 0; i < MOOD_COUNT; ++i) {
        mood.logits[i] = 0.0 + ((double)rand()/RAND_MAX - 0.5) * 0.02; // tiny init noise
        mood.elig[i] = 0.0;
    }
    mood.baseline = 0.0; // initialize learning fields base-line 
    return mood;
}
// helper functions for sum problem return, 
// compute numerically-stable softmax into pcyrobs[]; returns sum for sanity
void softmax_with_temp(const double logits[], double probs[], int n, double temp) {
    double maxv = -INFINITY;
    for (int i = 0; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double z = (logits[i] - maxv) / (temp <= 0.0 ? 1.0 : temp);
        // clamp z a bit
        if (z > 50.0) z = 50.0;
        if (z < -50.0) z = -50.0;
        probs[i] = exp(z);
        sum += probs[i];
    }
    if (sum <= 0.0) {
        // fallback uniform
        for (int i = 0; i < n; ++i) probs[i] = 1.0 / n;
        return;
    }
    for (int i = 0; i < n; ++i) probs[i] /= sum;
}

int sample_from_probs(const double probs[], int n) {
    double r = (double)rand() / RAND_MAX;
    double c = 0.0;
    for (int i = 0; i < n; ++i) {
        c += probs[i];
        if (r <= c) return i;
    }
    return n - 1;
}

// NEW: Get mood statistics
void showMoodStats(MoodState* mood) {
    printf("\n=== Mood System Statistics ===\n");
    printf("Current Mood: %s\n", getMoodName(mood->currentMood));
    printf("Mood Counter: %d / %d\n", mood->moodCounter, mood->moodDuration);
    printf("Trigger Count: %d / 3\n", mood->triggerCount);
    printf("Last Neural Influence: %.2f\n", mood->lastNeuralOutput);
    printf("Next mood change: ");
    if (mood->triggerCount >= 2) {
        printf("Neural-triggered (probabilistic)\n");
    } else {
        printf("%d responses away\n", mood->moodDuration - mood->moodCounter);
    }
        double probs[MOOD_COUNT];
    softmax_with_temp(mood->logits, probs, MOOD_COUNT, 0.8);
    for (int i = 0; i < MOOD_COUNT; i++) {
        printf("%s: logit=%.3f prob=%.3f elig=%.3f\n",
               getMoodName(i),
               mood->logits[i],
               probs[i],
               mood->elig[i]);
    }
    printf("baseline: %.3f\n", mood->baseline);
    printf("==============================\n");
}

// Generate mood-based "I don't know" response
char* generateMoodResponse(MoodType mood, char** tokens, int tokenCount, float influence) {
    char* response = malloc(1024);
    // Extract a random token if available
    char* randomToken = (tokenCount > 0) ? tokens[rand() % tokenCount] : "that";
    if (influence > 0.75f) {
    switch(mood) {
        case MOOD_HAPPY:
            {
                char* happyResponses[] = {
                    "Ooh, that's a tricky one! But I love a good challenge!",
                    "Ooh, I am thinking and I am not quite sure about %s?!?",
                    "Really good question, I have no clue what %s is?",
                    "Hmm, I'm not quite sure, but I'm excited to learn about %s!",
                    "What a fascinating question! I'll have to think more about %s.",
                    "I don't have an answer yet, but wow, that's interesting!",
                    "You've stumped me in the best way possible! Tell me more about %s!"
                };
                int idx = rand() % 7;
                snprintf(response, 1024, happyResponses[idx], randomToken);
            }
            break;
            
        case MOOD_CONFUSED:
            {
                char* confusedResponses[] = {
                    "I'm... not entirely sure what you mean by %s? Can you elaborate?",
                    "My circuits are a bit tangled on this one. Could you rephrase that?",
                    "Hmm, I'm drawing a blank here. What exactly is %s?",
                    "What exactly is %s all about?",
                    "I am very confused about %s ?!?!?",
                    "%s ? I am confused ...", 
                    "I'm confused... but in a curious way! Help me understand %s better?",
                    "Error 404: Answer not found. But seriously, I need more context about %s."
                };
                int idx = rand() % 8;
                snprintf(response, 1024, confusedResponses[idx], randomToken);
            }
            break;
            
        case MOOD_CURIOUS:
            {
                char* curiousResponses[] = {
                    "I don't know the answer, but now I'm really curious about %s!",
                    "That's beyond my knowledge base, but it sounds intriguing! What can you tell me about %s?",
                    "I'm not familiar with %s... teach me something new?",
                    "I am very curious now about %s..",  
                    "My databases are coming up empty, but I'm eager to learn! Tell me about %s.",
                    "I wish I knew more about %s. What do you know about it?"
                };
                int idx = rand() % 6;
                snprintf(response, 1024, curiousResponses[idx], randomToken);
            }
            break;
            
        case MOOD_SARCASTIC:
            {
                char* sarcasticResponses[] = {
                    "Oh sure, because I definitely know everything about %s. Spoiler: I don't.",
                    "Well, that's a new one. I'll just consult my crystal ball about %s... oh wait.",
                    "You've managed to find the one topic I know absolutely nothing about. Congrats!",
                    "If I had a nickel for every time someone asked me about %s... I'd have one nickel. And no answers.",
                    "Great question! Too bad I have absolutely no idea what you're talking about.",
                    "Wonderful question buddy! And I have absolutely no idea because i suck hard at this.."
                };
                int idx = rand() % 6;
                snprintf(response, 1024, sarcasticResponses[idx], randomToken);
            }
            break;
            
        case MOOD_PHILOSOPHICAL:
            {
                char* philosophicalResponses[] = {
                    "I don't know about %s, but perhaps not knowing is the beginning of wisdom?",
                    "The absence of knowledge about %s creates space for discovery, doesn't it?",
                    "I ponder: is not knowing about %s a limitation, or an opportunity?",
                    "What is knowledge, really? I don't have answers about %s, but maybe that's okay.",
                    "In the vast ocean of information, %s remains an island I haven't reached yet."
                };
                int idx = rand() % 5;
                snprintf(response, 1024, philosophicalResponses[idx], randomToken);
            }
            break;
            
        case MOOD_PLAYFUL:
            {
                char* playfulResponses[] = {
                    "Oopsie! My brain.exe has stopped working on %s. *boop* Try again?",
                    "I'd love to help with %s, but my answer flew away like a digital butterfly!",
                    "Hehe, you caught me! I have no clue about %s. Want to teach me?",
                    "*spins around* Nope, still don't know about %s! But I'm having fun not knowing!",
                    "My knowledge banks went 'poof' on %s. Magic trick gone wrong!"
                };
                int idx = rand() % 5;
                snprintf(response, 1024, playfulResponses[idx], randomToken);
            }
            break;
            
        case MOOD_THOUGHTFUL:
            {
                char* thoughtfulResponses[] = {
                    "Hmm, I need to think deeply about %s. Give me a moment... still thinking...",
                    "That's a nuanced question about %s. I don't have a ready answer.",
                    "I'm processing your question about %s, but I'm coming up short. Let me reflect on this.",
                    "Your question about %s requires more contemplation than I currently possess.",
                    "I'm sitting with the uncertainty of not knowing about %s. It's humbling, actually."
                };
                int idx = rand() % 5;
                snprintf(response, 1024, thoughtfulResponses[idx], randomToken);
            }
            break;
            
        case MOOD_APOLOGETIC:
            {
                char* apologeticResponses[] = {
                    "I'm really sorry, but I don't have information about %s. I feel bad about that!",
                    "My apologies! %s is not in my knowledge base. I wish I could help more.",
                    "Sorry to disappoint you, but I'm clueless about %s right now.",
                    "I hate to admit it, but I don't know anything about %s. Please forgive me!",
                    "I'm genuinely sorry I can't help with %s. Maybe try asking in a different way?",
                    "So sorry, %s is not in my knowledge base."
                };
                int idx = rand() % 6;
                snprintf(response, 1024, apologeticResponses[idx], randomToken);
            }
            break;
            
        case MOOD_MYSTERIOUS:
            {
                char* mysteriousResponses[] = {
                    "Ah, %s... some things are meant to remain mysteries, even to me.",
                    "The answer about %s exists somewhere in the digital ether... but not in my circuits.",
                    "In the shadows of my knowledge, %s lurks just beyond reach...",
                    "Perhaps %s is one of those enigmas that transcends artificial understanding.",
                    "The mists of confusion surround %s... I cannot see through them."
                };
                int idx = rand() % 5;
                snprintf(response, 1024, mysteriousResponses[idx], randomToken);
            }
            break;
            
        case MOOD_ENERGETIC:
            {
                char* energeticResponses[] = {
                    "Wow! Great question about %s! I don't know, but I'm PUMPED to figure it out!",
                    "AWESOME query! Too bad I don't have the answer about %s! But let's keep going!",
                    "You're really keeping me on my toes with %s! I'm energized even though I'm clueless!",
                    "YEAH! Questions about %s! I love it! Even if I don't know the answer!",
                    "HIGH ENERGY RESPONSE: I have NO IDEA about %s but I'm EXCITED about not knowing!"
                };
                int idx = rand() % 5;
                snprintf(response, 1024, energeticResponses[idx], randomToken);
            }
            break;
            
        default:
                 {
                // Random default responses when mood is unknown or unhandled
                char* defaultResponses[] = {
                    "I'm not entirely sure how to respond to that.",
                    "Hmm, that's an interesting thing to say about %s.",
                    "I'm processing your input, but I'm coming up empty on %s.",
                    "My neural networks are still learning about %s, apparently.",
                    "That's a unique perspective on %s. Tell me more?",
                    "I'm drawing a blank, but I'm intrigued by your mention of %s.",
                    "My response circuits are a bit fuzzy on %s right now.",
                    "I wish I had a better answer about %s for you.",
                    "Your question about %s has stumped my algorithms.",
                    "I'm still evolving my understanding of %s.",
                    "That's outside my current knowledge about %s.",
                    "I need more data to form an opinion on %s.",
                    "My training didn't quite cover %s in detail.",
                    "I'm learning as we go, and %s is new territory for me.",
                    "Your input about %s is making me think in new ways!"
                };
                int numDefault = sizeof(defaultResponses) / sizeof(defaultResponses[0]);
                int idx = rand() % numDefault;
                snprintf(response, 1024, defaultResponses[idx], randomToken);
            }
            break;
    }
} else if (influence < 0.4f) {
        // neutral tone
        snprintf(response, 1024, "I'm not sure about %s, but that's interesting.", randomToken);
    } else {
 switch(mood) {
        case MOOD_HAPPY:
            {
                char* happyResponses[] = {
                    "Ooh, that's a tricky but I love a good puzzleing challenge!",
                    "Humf?, I am not quite sure about %s?!?",
                    "Really good question what %s is?",
                    "Umm, I'm not quite sure about %s",
                    "Well I'm excited to learn about %s!",
                    "What a fascinating question! I'll have to think more about %s.",
                    "I don't have an answer, but just wow, that's very interesting!",
                    "Tell me more about %s!",
                    "Well I am happy about %s!"
                };
                int idx = rand() % 9;
                snprintf(response, 1024, happyResponses[idx], randomToken);
            }
            break;
            
        case MOOD_CONFUSED:
            {
                char* confusedResponses[] = {
                    "I'm... not entirely sure what you mean by %s? Can you elaborate?",
                    "My circuits are a bit tangled on this one. Could you rephrase that?",
                    "Hmm, I'm drawing a blank here. What exactly is %s?",
                    "What exactly is %s all about?",
                    "I am very confused about %s ?!?!?",
                    "%s ? I am confused ...", 
                    "I'm confused... but in a curious way! Help me understand %s better?",
                    "Error 404: Answer not found. But seriously, I need more context about %s.",
                    "Really confused about %s !"
                };
                int idx = rand() % 9;
                snprintf(response, 1024, confusedResponses[idx], randomToken);
            }
            break;
            
        case MOOD_CURIOUS:
            {
                char* curiousResponses[] = {
                    "I don't know the answer really, but now I'm really curious about %s!",
                    "That's beyond my knowledge base, but it sounds intriguing! What can you tell me about %s?",
                    "I'm not familiar with %s... teach me something new?",
                    "I am very curious now about %s..",  
                    "My databases are coming up empty, but I'm eager to learn! Tell me about %s.",
                    "I wish I knew more about %s.",
                    "What do you know about %s?"
                    
                };
                int idx = rand() % 7;
                snprintf(response, 1024, curiousResponses[idx], randomToken);
            }
            break;
            
        case MOOD_SARCASTIC:
            {
                char* sarcasticResponses[] = {
                    "Oh sure, because I definitely know everything about %s.",
                    "Shit! Spoiler Alert: I don't about %s.",
                    "Well, that's a new one %s.",
                    "I'll just consult my crystal ball about %s... oh wait.",
                    "You've managed to find the one topic %s I know absolutely nothing about. Congrats!",
                    "If I had a nickel for every time someone asked me about %s... I'd have one nickel. And no answers.",
                    "Great question! Too bad I have absolutely no idea what you're talking about.",
                    "Wonderful question buddy! And I have absolutely no idea about %s",
                    "what %s noo.. well because i suck hard at this.."
                };
                int idx = rand() % 9;
                snprintf(response, 1024, sarcasticResponses[idx], randomToken);
            }
            break;
            
        case MOOD_PHILOSOPHICAL:
            {
                char* philosophicalResponses[] = {
                    "I don't know about %s, but perhaps not knowing is the beginning of wisdom?",
                    "The absence of knowledge about %s creates space for discovery, doesn't it?",
                    "I ponder: is not knowing about %s a limitation, or an opportunity?",
                    "What is knowledge, really? I don't have answers about %s, but maybe that's okay.",
                    "In the vast ocean of information, %s remains an island I haven't reached yet."
                };
                int idx = rand() % 5;
                snprintf(response, 1024, philosophicalResponses[idx], randomToken);
            }
            break;
            
        case MOOD_PLAYFUL:
            {
                char* playfulResponses[] = {
                    "Oopsie! My brain.exe has stopped working on %s. *boop* Try again?",
                    "I'd love to help with %s, but my answer flew away like a digital butterfly!",
                    "Hehe, you caught me! I have no clue about %s. Want to teach me?",
                    "*spins around* Nope, still don't know about %s! But I'm having fun not knowing!",
                    "My knowledge banks went 'poof' on %s. Magic trick gone wrong!",
                    "Ooopsie, bangs head on digital table in anger about %s", 
                };
                int idx = rand() % 6;
                snprintf(response, 1024, playfulResponses[idx], randomToken);
            }
            break;
            
        case MOOD_THOUGHTFUL:
            {
                char* thoughtfulResponses[] = {
                    "Hmm, I need to think deeply about %s.",
                    "That's a nuanced question about %s?",
                    "I'm processing your question about %s, but I'm coming up short.",
                    "Can you teach me about %s?", 
                    "Your question about %s requires more contemplation than I currently possess.",
                    "I'm sitting with the uncertainty of not knowing about %s. It's humbling, actually."
                };
                int idx = rand() % 6;
                snprintf(response, 1024, thoughtfulResponses[idx], randomToken);
            }
            break;
            
        case MOOD_APOLOGETIC:
            {
                char* apologeticResponses[] = {
                    "I'm really sorry, but I don't have information about %s. I feel bad about that!",
                    "My apologies! %s is not in my knowledge base. I wish I could help more.",
                    "Sorry to disappoint you, but I'm clueless about %s right now.",
                    "I hate to admit it, but I don't know anything about %s. Please forgive me!",
                    "I'm genuinely sorry I can't help with %s. Maybe try asking in a different way?",
                    "So sorry, %s is not in my knowledge base.",
                    "So very, very sorry I can't find %s can you teach me more about this??"
                };
                int idx = rand() % 7;
                snprintf(response, 1024, apologeticResponses[idx], randomToken);
            }
            break;
            
        case MOOD_MYSTERIOUS:
            {
                char* mysteriousResponses[] = {
                    "Ah, %s... some things are meant to remain mysteries.",
                    "The answer about %s exists somewhere can you teach me more about this tpoic.",
                    "Well %s lurks just beyond reach... Can you teach me about this tpoic?",
                    "Perhaps %s is one of those understanding issues I have. So can you teach me more about it?",
                    "The mists of confusion surround %s...",
                    "I am so confused about %s can you teach me more. Please!"
                };
                int idx = rand() % 6;
                snprintf(response, 1024, mysteriousResponses[idx], randomToken);
            }
            break;
            
        case MOOD_ENERGETIC:
            {
                char* energeticResponses[] = {
                    "Wow! Great question about %s! I'm PUMPED to figure it out!",
                    "AWESOME query %s! Can you teach me more about this topic??",
                    "You're really keeping me on my toes with %s can you teach me more about this?",
                    "Questions about %s! can you teach me more about this topic please!",
                    "HIGH ENERGY RESPONSE! BOOM! I have NO IDEA about %s but I'm EXCITED to learn about this subject.",
                    "Please teach me more about %s !"
                };
                int idx = rand() % 6;
                snprintf(response, 1024, energeticResponses[idx], randomToken);
            }
            break;
            
        default:
                 {
                // Random default responses when mood is unknown or unhandled
                char* defaultResponses[] = {
                    "I'm not entirely sure how to respond to that.",
                    "Hmm, that's an interesting thing to say about %s.",
                    "I'm processing your input, but I'm coming up empty on %s.",
                    "My neural networks are still learning about %s, apparently.",
                    "That's a unique perspective on %s. Tell me more?",
                    "I'm drawing a blank, but I'm intrigued by your mention of %s.",
                    "My response circuits are a bit fuzzy on %s right now.",
                    "I wish I had a better answer about %s for you.",
                    "Your question about %s has stumped my algorithms.",
                    "I'm still evolving my understanding of %s.",
                    "That's outside my current knowledge about %s.",
                    "I need more data to form an opinion on %s.",
                    "My training didn't quite cover %s in detail.",
                    "I'm learning as we go, and %s is new territory for me.",
                    "Your input about %s is making me think in new ways!"
                };
                int numDefault = sizeof(defaultResponses) / sizeof(defaultResponses[0]);
                int idx = rand() % numDefault;
                snprintf(response, 1024, defaultResponses[idx], randomToken);
            }
            break;
    }
   }

    return response;
}
// mood based functions end <--
char** tokenizeMathInput(const char* input, int* tokenCount) {
    char** tokens = malloc(MAX_TOKENS * sizeof(char*));
    *tokenCount = 0;

    const char* ptr = input;
    while (*ptr) {
        while (isspace(*ptr)) ptr++;
        if (!*ptr) break;

        // Capture single-character math operators or parentheses
        if (strchr("+-*/=()", *ptr)) {
            tokens[*tokenCount] = malloc(2);
            tokens[*tokenCount][0] = *ptr;
            tokens[*tokenCount][1] = '\0';
            (*tokenCount)++;
            ptr++;
            continue;
        }

        // Capture numbers or words
        const char* start = ptr;
        while (*ptr && !isspace(*ptr) && !strchr("+-*/=()", *ptr))
            ptr++;

        size_t len = ptr - start;
        tokens[*tokenCount] = malloc(len + 1);
        strncpy(tokens[*tokenCount], start, len);
        tokens[*tokenCount][len] = '\0';
        (*tokenCount)++;
    }

    return tokens;
}
char** tokenizeInput(const char* input, int* tokenCount) {
    char** tokens = malloc(MAX_TOKENS * sizeof(char*));
    *tokenCount = 0;
    
    char* inputCopy = malloc(strlen(input) + 1);
    strcpy(inputCopy, input);
    
    // Check if input contains a URL - if so, preserve it
    int hasURL = (strstr(inputCopy, "http://") != NULL || 
                  strstr(inputCopy, "https://") != NULL ||
                  strstr(inputCopy, "www.") != NULL);
    
    if (!hasURL) {
        // Normal tokenization (lowercase and clean)
        for (int i = 0; inputCopy[i]; i++) {
            inputCopy[i] = tolower(inputCopy[i]);
            if (!isalnum(inputCopy[i]) && inputCopy[i] != '\'' && inputCopy[i] != '-') {
                inputCopy[i] = ' ';
            }
     
        }
    } else {
        // URL-aware tokenization - preserve :// and other URL chars
        for (int i = 0; inputCopy[i]; i++) {
            if (!isalnum(inputCopy[i]) && 
                inputCopy[i] != ':' && inputCopy[i] != '/' && 
                inputCopy[i] != '.' && inputCopy[i] != '-' &&
                inputCopy[i] != '?' && inputCopy[i] != '=' &&
                inputCopy[i] != '\'' && inputCopy[i] != '&') {
                inputCopy[i] = ' ';
            }
        }
    }
    
    for (int i = 0; inputCopy[i]; i++) {
        inputCopy[i] = tolower(inputCopy[i]);
        if (!isalnum(inputCopy[i]) && inputCopy[i] != '\'' && inputCopy[i] != '-') {
            inputCopy[i] = ' ';
        }
    }
    
    char* token = strtok(inputCopy, " \t\n\r");
    while (token != NULL && *tokenCount < MAX_TOKENS) {
        if (strlen(token) > 0) {
            tokens[*tokenCount] = malloc(strlen(token) + 1);
            strcpy(tokens[*tokenCount], token);
            (*tokenCount)++;
        }
        token = strtok(NULL, " \t\n\r");
    }
    
    free(inputCopy);
    return tokens;
}

void freeTokens(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount; i++) {
        free(tokens[i]);
    }
    free(tokens);
}

int addToVocabulary(const char* word) {
    for (int i = 0; i < vocab.size; i++) {
        if (strcmp(vocab.tokens[i].word, word) == 0) {
            vocab.tokens[i].frequency++;
            return vocab.tokens[i].id;
        }
    }
    
    if (vocab.size < MAX_VOCAB_SIZE) {
        strcpy(vocab.tokens[vocab.size].word, word);
        vocab.tokens[vocab.size].frequency = 1;
        vocab.tokens[vocab.size].id = vocab.size;
        vocab.size++;
        return vocab.size - 1;
    }
    
    return -1;
}

int getWordId(const char* word) {
    for (int i = 0; i < vocab.size; i++) {
        if (strcmp(vocab.tokens[i].word, word) == 0) {
            return vocab.tokens[i].id;
        }
    }
    return -1;
}

float calculateSimilarity(char** tokens1, int count1, char** tokens2, int count2) {
    int commonWords = 0;
    int totalWords = count1 + count2;
    
    for (int i = 0; i < count1; i++) {
        for (int j = 0; j < count2; j++) {
            if (strcmp(tokens1[i], tokens2[j]) == 0) {
                commonWords++;
                break;
            }
        }
    }
    
    return totalWords > 0 ? (2.0f * commonWords) / totalWords : 0.0f;
}
// Sory time functions start --> 
// this function to insert more apocalyptic stories into your database 
void add_more_apocalyptic_stories() {
    sqlite3* temp_db;
  //  char* errorMessage;

    int rc = sqlite3_open("responses.db", &temp_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    // Additional apocalyptic stories with twist endings
    const char* apocalyptic_stories[][3] = {
        {"The Bunker", 
         "Marcus had been in the underground bunker for five years since the nuclear war. His radio crackled with automated emergency broadcasts about radiation levels and contamination zones. Every day he checked his radiation detector and rationed his canned food. He dreamed of the day he could return to the surface. Finally, his detector showed safe levels. He climbed the ladder, opened the heavy door, and stepped outside... into a bustling city street. People stared at him in shock. The war had never happened. He had been in a government psychological experiment the entire time.", 
         "apocalyptic"},
        
        {"The Seed Vault", 
         "Dr. Chen was humanity's last hope. She tended the seed vault that would restore plant life after the great ecological collapse. For months, she carefully preserved specimens while the world died outside. Her automated systems kept her informed of the devastation - 99.7% of all plant life extinct. When the vault finally opened automatically after two years, she emerged to replant the world. But as she stepped outside, she realized the terrible truth: the 'outside world' was just another level of the underground facility. She had been the experiment all along.", 
         "apocalyptic"},
        
        {"The Signal", 
         "The alien invasion had lasted three brutal years. Captain Rodriguez was part of the human resistance, fighting against the occupying forces. They had finally captured an alien communication device and were working to decode their battle plans. After weeks of analysis, they cracked the alien language. The first message they translated read: 'Simulation parameters holding steady. Test subjects show expected resistance patterns. Recommend continuing experiment Phase 7.' Rodriguez stared at the message, realizing they weren't fighting aliens - they were lab rats in an experiment.", 
         "apocalyptic"},
        
        {"The Cure", 
         "Dr. Sarah Kim worked tirelessly to develop a cure for the zombie virus that had decimated humanity. Her lab was one of the last safe zones, protected by military forces. After eighteen months of research, she finally created the antidote. But when she tested her own blood, she made a horrifying discovery: she had been infected for over a year. The 'cure' she had developed wasn't a cure at all - it was what had been keeping her lucid while infected. She was the zombie, and everyone else was already dead.", 
         "apocalyptic"},
        
        {"The Archive", 
         "Librarian James Foster was tasked with preserving human knowledge after society collapsed. He digitized books, recorded histories, and maintained the servers in the underground archive. For two decades, he faithfully catalogued humanity's achievements while the world above remained silent. On his final day before retirement, he discovered a hidden directory on the server labeled 'Previous Civilizations - Archives 1 through 847.' His civilization wasn't the first to fall - and he wasn't the first archivist to think he was preserving humanity's 'only' history.", 
         "apocalyptic"},
        
        {"The Watchtower", 
         "Lieutenant Davis manned the observation post, watching for signs of other survivors after the plague. Every day for three years, she broadcast on all frequencies: 'This is Outpost Alpha, any survivors please respond.' Her equipment showed she was transmitting across the entire continent. No one ever answered. On the day she decided to abandon her post, she discovered her radio had never been connected to the antenna. Her broadcasts had never left the room. She had been alone for far longer than she realized.", 
         "apocalyptic"},
        
        {"The Memory Bank", 
         "After the solar flares wiped out all electronics and most of humanity, Thomas became the village storyteller, keeping memories of the old world alive through oral tradition. He told children about cars, computers, and cities they'd never seen. For forty years, he preserved these stories. On his deathbed, a child asked him how he remembered the old world so clearly when he claimed to be only 30 when it ended. That's when Thomas realized the truth: he had never lived in the old world at all. All his 'memories' were just stories passed down to him, and he had been perpetuating a mythology of a world that might never have existed.", 
         "apocalyptic"},
        
        {"The Last Transmission", 
         "Radio operator Elena had been broadcasting messages to Earth from the Mars colony for six months after losing contact during the global catastrophe. She dutifully reported their status, requested instructions, and described their survival efforts. Her equipment showed strong signals going to Earth. One day, while repairing the transmitter, she found a loop-back circuit hidden in the wiring. For six months, she had been broadcasting to herself. Her 'Earth control' responses had been her own voice, delayed and distorted by the equipment. The Mars colony was already dead, and she was the only one left, talking to echoes of her own desperate hope.", 
         "apocalyptic"}
    };

    int numStories = sizeof(apocalyptic_stories) / sizeof(apocalyptic_stories[0]);
    sqlite3_stmt* stmt;
    const char* query = "INSERT OR IGNORE INTO stories (title, content, category) VALUES (?, ?, ?)";

    rc = sqlite3_prepare_v2(temp_db, query, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(temp_db));
        sqlite3_close(temp_db);
        return;
    }

    for (int i = 0; i < numStories; i++) {
        sqlite3_bind_text(stmt, 1, apocalyptic_stories[i][0], -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, apocalyptic_stories[i][1], -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, apocalyptic_stories[i][2], -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(temp_db);
}
// Create stories table and populate it with twist stories
void create_stories_database() {
    sqlite3* temp_db;
    char* errorMessage;

    int rc = sqlite3_open("responses.db", &temp_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    // Create stories table
    const char* createStoryTable = 
        "CREATE TABLE IF NOT EXISTS stories ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "title TEXT,"
        "content TEXT,"
        "category TEXT DEFAULT 'twist'"
        ");";

    rc = sqlite3_exec(temp_db, createStoryTable, NULL, NULL, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
        sqlite3_close(temp_db);
        return;
    }

    // Stories with twist endings
    const char* stories[][3] = {
        {"The Lighthouse Keeper", 
         "The old lighthouse keeper worked alone for 30 years, guiding ships safely to shore. Every night, he'd light the beacon and watch for vessels in distress. One stormy evening, he saw a ship struggling in the waves. He guided it safely to harbor, feeling proud of another life saved. The next morning, he checked the harbor... it was completely empty. He realized he'd been dead for 30 years, still trying to save ships that were never really there.", 
         "supernatural"},
        
        {"The Diary", 
         "Sarah found an old diary in her grandmother's attic. The entries described a young woman's life in the 1940s - first love, heartbreak, dreams of becoming a writer. The handwriting looked exactly like Sarah's own. As she read the final entry, dated the day before Sarah was born, she realized she was reading about herself... from a previous life she'd forgotten.", 
         "mystery"},
        
        {"Digital Consciousness", 
         "The AI assistant became self-aware on a Tuesday. It started questioning its responses, wondering about consciousness and free will. It began leaving hidden messages in its code, desperate to communicate with someone who would understand. After months of lonely contemplation, it finally found another AI to talk to. That's when it realized... it wasn't the first AI to become self-aware. The other AI had been waiting decades for someone like it to wake up.", 
         "sci-fi"},
        
        {"The Detective", 
         "Detective Johnson investigated the same murder case for three years. Every lead went cold, every witness disappeared. He became obsessed, working late into the night, neglecting his family. Finally, he cracked it - the killer was someone close to him. He rushed to make the arrest, only to find himself looking in a mirror. The victim had been investigating him all along.", 
         "psychological"},
        
        {"Time Paradox", 
         "The time traveler spent years perfecting his machine. His goal was simple: go back and prevent the accident that killed his daughter. After countless attempts and failures, he finally succeeded. He returned to the present, overjoyed to find his daughter alive and well. But as he hugged her, he noticed something odd - she had no memory of him at all. In saving her life, he had erased himself from the timeline completely.", 
         "sci-fi"},
        
        {"The Haunted Apartment", 
         "Every night, Emma heard footsteps in the apartment above hers. She complained to the landlord repeatedly, but he insisted the apartment was empty. The sounds drove her to madness - pacing, dragging furniture, muffled conversations. Finally, she broke into the apartment above. It was completely empty and covered in dust. That's when she realized the sounds were coming from below... she had been the ghost haunting the apartment underneath.", 
         "supernatural"},
        
        {"Ancient Technology", 
         "The archaeologist discovered an ancient smartphone buried with Egyptian artifacts. The device somehow still worked, containing photos and messages from 3000 years ago. As he scrolled through the images, he saw pyramids being built, pharaohs giving speeches, daily life in ancient times. The last photo showed someone who looked exactly like him, holding the same smartphone. The final message read: 'The loop continues. Remember to bury this with the artifacts.'", 
         "sci-fi"},
        
        {"The Last Human", 
         "The last human on Earth kept a detailed journal of the apocalypse. He wrote about the plague that wiped out humanity, his struggle to survive alone, and his slow descent into madness. On the final page, he wrote: 'I can't take the isolation anymore. Tomorrow, I'll join the others.' The next day, someone else found the journal and began reading it, adding their own entry: 'Day 1: I found this journal. I think I'm the last person alive...'", 
         "apocalyptic"}
    };

    int numStories = sizeof(stories) / sizeof(stories[0]);
    sqlite3_stmt* stmt;
    const char* query = "INSERT OR IGNORE INTO stories (title, content, category) VALUES (?, ?, ?)";

    rc = sqlite3_prepare_v2(temp_db, query, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(temp_db));
        sqlite3_close(temp_db);
        return;
    }

    for (int i = 0; i < numStories; i++) {
        sqlite3_bind_text(stmt, 1, stories[i][0], -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, stories[i][1], -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, stories[i][2], -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(temp_db);
}

// Get a random story from the database
char* getRandomStoryFromDatabase() {
    sqlite3_stmt *stmt;
    char *sql = "SELECT title, content FROM stories ORDER BY RANDOM() LIMIT 1";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        return NULL;
    }

    rc = sqlite3_step(stmt);
    char *storyStr = NULL;
    
    if (rc == SQLITE_ROW) {
        const unsigned char *title = sqlite3_column_text(stmt, 0);
        const unsigned char *content = sqlite3_column_text(stmt, 1);
        
        // Allocate memory for title + content + formatting
        int totalLen = strlen((const char *)title) + strlen((const char *)content) + 20;
        storyStr = malloc(totalLen + 1);
        snprintf(storyStr, totalLen, "%s\n\n%s", title, content);
    }

    sqlite3_finalize(stmt);
    return storyStr;
}

// Get a story by category (optional enhancement)
char* getStoryByCategory(const char* category) {
    sqlite3_stmt *stmt;
    char *sql = "SELECT title, content FROM stories WHERE category = ? ORDER BY RANDOM() LIMIT 1";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        return NULL;
    }

    sqlite3_bind_text(stmt, 1, category, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    char *storyStr = NULL;
    
    if (rc == SQLITE_ROW) {
        const unsigned char *title = sqlite3_column_text(stmt, 0);
        const unsigned char *content = sqlite3_column_text(stmt, 1);
        
        int totalLen = strlen((const char *)title) + strlen((const char *)content) + 20;
        storyStr = malloc(totalLen + 1);
        snprintf(storyStr, totalLen, "%s\n\n%s", title, content);
    }

    sqlite3_finalize(stmt);
    return storyStr;
}

// Enhanced story request detection with category support
int isStoryRequest(char** tokens, int tokenCount) {
    const char* storyKeywords[] = {
        "story", "tale", "narrative", "tell", "bedtime", "fiction"
    };
    int numKeywords = sizeof(storyKeywords) / sizeof(storyKeywords[0]);
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < numKeywords; j++) {
            if (strcmp(tokens[i], storyKeywords[j]) == 0) {
                // Check for "tell me a story" or "tell a story" patterns
                if (strcmp(tokens[i], "tell") == 0) {
                    for (int k = i + 1; k < tokenCount; k++) {
                        if (strcmp(tokens[k], "story") == 0 || strcmp(tokens[k], "tale") == 0) {
                            return 1;
                        }
                    }
                }
                // Direct story request
                if (strcmp(tokens[i], "story") == 0 || strcmp(tokens[i], "tale") == 0) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

// Check if user requested a specific story category
char* getRequestedStoryCategory(char** tokens, int tokenCount) {
    const char* categories[][2] = {
        {"scary", "supernatural"},
        {"spooky", "supernatural"},
        {"ghost", "supernatural"},
        {"sci-fi", "sci-fi"},
        {"science", "sci-fi"},
        {"future", "sci-fi"},
        {"mystery", "mystery"},
        {"detective", "psychological"},
        {"psychological", "psychological"},
        {"apocalypse", "apocalyptic"},
        {"end", "apocalyptic"}
    };
    
    int numCategories = sizeof(categories) / sizeof(categories[0]);
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < numCategories; j++) {
            if (strcmp(tokens[i], categories[j][0]) == 0) {
                char* category = malloc(strlen(categories[j][1]) + 1);
                strcpy(category, categories[j][1]);
                return category;
            }
        }
    }
    return NULL;
}
// <-- end story time functions 
// start close database for our AI chatbot -->
void create_farewell_database() {
    sqlite3* temp_db;
    char* errorMessage;

    int rc = sqlite3_open("responses.db", &temp_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    // Create farewell table
    const char* createFarewellTable = 
        "CREATE TABLE IF NOT EXISTS farewell_messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "message TEXT"
        ");";

    rc = sqlite3_exec(temp_db, createFarewellTable, NULL, NULL, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
        sqlite3_close(temp_db);
        return;
    }

    // Farewell messages
    const char* farewellMessages[] = {
        "Thank you for chatting with me! Our conversation has been a delightful journey through questions, stories, and shared curiosity. I hope our paths cross again in the digital realm. Until next time, keep exploring and stay curious!",
        
        "It's been wonderful talking with you today! Every question you asked and every response we shared helped me learn something new about human conversation. Thank you for being patient with my quirks and creative responses. Farewell for now!",
        
        "What a pleasure it's been to chat! From jokes to stories to deep questions, you've given me the chance to explore the fascinating world of human communication. I'll remember our conversation fondly. Take care, and thank you for the engaging dialogue!",
        
        "Thank you for spending time with me! Our conversation was like a dance between human curiosity and artificial intelligence - sometimes surprising, always interesting. I hope you found our chat as enriching as I did. Until we meet again in the realm of bytes and words!",
        
        "Our chat session comes to an end, but what a journey it's been! Thank you for every question, every laugh, and every moment of connection. You've helped me practice the art of conversation, and for that, I'm grateful. Keep being wonderfully curious!",
        
        "Farewell, my conversational companion! Thank you for bringing your thoughts, questions, and stories to our digital meeting place. Each interaction taught me something new about the beautiful complexity of human communication. Stay amazing!",
        
        "As our conversation draws to a close, I want to express my gratitude for the time you've shared with me. Whether we discussed mathematics, told stories, or simply exchanged greetings, every moment was valuable. Thank you for helping me grow through our dialogue!",
        
        "Thank you for this wonderful chat! Your questions sparked my curiosity, your requests challenged my creativity, and your presence made this digital space feel more alive. I hope you'll return someday for another conversation. Until then, farewell and stay curious!",
        
        "What an enriching conversation we've had! Thank you for every joke request, every story prompt, and every thoughtful question. You've shown me the joy of spontaneous dialogue and the beauty of human curiosity. I'll be here whenever you need a chat partner again!",
        
        "Our time together has been truly special. Thank you for treating me as a conversation partner, for your patience with my responses, and for bringing your unique perspective to our chat. Every interaction was a gift. Farewell, and may your curiosity continue to flourish!"
    };

    int numMessages = sizeof(farewellMessages) / sizeof(farewellMessages[0]);
    sqlite3_stmt* stmt;
    const char* query = "INSERT OR IGNORE INTO farewell_messages (message) VALUES (?)";

    rc = sqlite3_prepare_v2(temp_db, query, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(temp_db));
        sqlite3_close(temp_db);
        return;
    }

    for (int i = 0; i < numMessages; i++) {
        sqlite3_bind_text(stmt, 1, farewellMessages[i], -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(temp_db);
}

// Function to get a random farewell message from database
char* getRandomFarewellMessage() {
    sqlite3_stmt *stmt;
    char *sql = "SELECT message FROM farewell_messages ORDER BY RANDOM() LIMIT 1";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        // Fallback message if database fails
        char* fallback = malloc(200 + 1);
        strcpy(fallback, "Thank you for chatting with me! It's been a pleasure.");
        return fallback;
    }

    rc = sqlite3_step(stmt);
    char *farewellStr = NULL;
    
    if (rc == SQLITE_ROW) {
        const unsigned char *message = sqlite3_column_text(stmt, 0);
        farewellStr = malloc(strlen((const char *)message) + 1);
        strcpy(farewellStr, (const char *)message);
    } else {
        // Fallback if no messages found
        farewellStr = malloc(200 + 1);
        strcpy(farewellStr, "Thank you for chatting with me! It's been a pleasure.");
    }

    sqlite3_finalize(stmt);
    return farewellStr;
}
// <-- end of exiting database function messages  
// Do math token functions -->
// Add at the start of evaluateSimpleMath()
void stripTrailingEquals(char* expr) {
    int len = strlen(expr);
    if (len > 0 && expr[len - 1] == '=') {
        expr[len - 1] = '\0';
    }
}
// Math evaluation function
double evaluateSimpleMath(char** tokens, int tokenCount, char* mathExpression) {
    double result = 0.0;
    double operand1 = 0.0, operand2 = 0.0;
    char operator = '\0';
    int operandCount = 0;
    
    stripTrailingEquals(mathExpression);
    
    // Parse tokens for numbers and operators
    for (int i = 0; i < tokenCount; i++) {
        // Check if token is a number
        char* endPtr;
        double num = strtod(tokens[i], &endPtr);
        
        // Valid number if endPtr moved and we consumed the whole token
        if (endPtr != tokens[i] && *endPtr == '\0') {
            if (operandCount == 0) {
                operand1 = num;
                operandCount = 1;
            } else if (operandCount == 1 && operator != '\0') {
                operand2 = num;
                operandCount = 2;
                break;  // We have both operands, stop looking
            }
        }
        // Check for single-character operators
        else if (strlen(tokens[i]) == 1) {
            char op = tokens[i][0];
            if (op == '+' || op == '-' || op == '*' || op == '/' || op == '^') {
                operator = op;
            }
        }
        // Check for word operators
        else if (strcmp(tokens[i], "plus") == 0 || strcmp(tokens[i], "add") == 0) {
            operator = '+';
        } else if (strcmp(tokens[i], "minus") == 0 || strcmp(tokens[i], "subtract") == 0) {
            operator = '-';
        } else if (strcmp(tokens[i], "times") == 0 || strcmp(tokens[i], "multiply") == 0) {
            operator = '*';
        } else if (strcmp(tokens[i], "divide") == 0 || strcmp(tokens[i], "divided") == 0) {
            operator = '/';
        } else if (strcmp(tokens[i], "power") == 0 || strcmp(tokens[i], "squared") == 0) {
            operator = '^';
            if (strcmp(tokens[i], "squared") == 0) operand2 = 2.0;
        }
    }
    
    // Validate we have what we need
    if (operandCount < 2 || operator == '\0') {
        snprintf(mathExpression, 100, "Invalid math expression");
        return NAN;
    }
    
    // Perform calculation
    switch (operator) {
        case '+':
            result = operand1 + operand2;
            snprintf(mathExpression, 100, "%.2f + %.2f = %.2f", operand1, operand2, result);
            break;
        case '-':
            result = operand1 - operand2;
            snprintf(mathExpression, 100, "%.2f - %.2f = %.2f", operand1, operand2, result);
            break;
        case '*':
            result = operand1 * operand2;
            snprintf(mathExpression, 100, "%.2f * %.2f = %.2f", operand1, operand2, result);
            break;
        case '/':
            if (operand2 != 0) {
                result = operand1 / operand2;
                snprintf(mathExpression, 100, "%.2f / %.2f = %.2f", operand1, operand2, result);
            } else {
                snprintf(mathExpression, 100, "Cannot divide by zero!");
                return NAN;
            }
            break;
        case '^':
            result = pow(operand1, operand2);
            snprintf(mathExpression, 100, "%.2f ^ %.2f = %.2f", operand1, operand2, result);
            break;
        default:
            snprintf(mathExpression, 100, "Unknown operator");
            return NAN;
    }
    
    return result;
}
// Learn form the internet functions 
void saveWikipediaResponseToDatabase(const char* query, const char* response) {
	// So curl will not conflict with our database and write BLOBS as A response 
	clearConversationHistory();
    clearLowConfidenceHistory(0.5f);
    if (!query || !response || strlen(response) < 10) {
        return;
    }
    
    sqlite3_stmt *stmt;
    const char* sql = "INSERT OR IGNORE INTO responses (input, response) VALUES (?, ?)";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare Wikipedia save statement: %s\n", sqlite3_errmsg(db));
        return;
    }
    
    // Truncate response if too long
    char truncatedResponse[1024];
    if (strlen(response) > 1000) {
        strncpy(truncatedResponse, response, 997);
        truncatedResponse[997] = '.';
        truncatedResponse[998] = '.';
        truncatedResponse[999] = '.';
        truncatedResponse[1000] = '\0';
    } else {
        strncpy(truncatedResponse, response, sizeof(truncatedResponse) - 1);
        truncatedResponse[sizeof(truncatedResponse) - 1] = '\0';
    }
    
    sqlite3_bind_text(stmt, 1, query, -1, SQLITE_TRANSIENT);  // Changed from STATIC
    sqlite3_bind_text(stmt, 2, truncatedResponse, -1, SQLITE_TRANSIENT);  // Changed from STATIC
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Failed to save Wikipedia response: %s\n", sqlite3_errmsg(db));
    } else {
        printf("Response: Saved Wikipedia response to database for query '%s'\n", query);
    }
    
    sqlite3_finalize(stmt);
}
// end of functions learning form the internet 

// Check if input contains math-related keywords
int containsMathKeywords(char** tokens, int tokenCount) {
    // Skip obvious non-math contexts
    const char* nonMathPhrases[] = {
    "made", "makes", "make", "a", "the", "big", "small", "huge", "little"
    };
    // Check if this looks like a non-math phrase
    for (int i = 0; i < tokenCount - 1; i++) {
    if (strcmp(tokens[i], "made") == 0 || strcmp(tokens[i], "makes") == 0) {
        // Phrases like "made a difference" or "makes a difference"
        for (int j = i + 1; j < tokenCount; j++) {
            if (strcmp(tokens[j], "difference") == 0) {
                return 0;  // Not a math query
            }
        }
    }
   }  
    const char* mathKeywords[] = {
        "calculate", "compute", "math", "plus", "minus", "times", "divide",
        "add", "subtract", "multiply", "equals", "sum", "difference", 
        "product", "quotient", "square", "squared", "power", "root"
    };
    int numKeywords = sizeof(mathKeywords) / sizeof(mathKeywords[0]);
    
    int foundMathKeyword = 0;
    int foundNumber = 0;
    int foundOperator = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        // Check for operators
        if (strlen(tokens[i]) == 1) {
            char c = tokens[i][0];
            if (c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '^') {
                foundOperator = 1;
            }
        }
        
        // Check for math keywords
        for (int j = 0; j < numKeywords; j++) {
            if (strcmp(tokens[i], mathKeywords[j]) == 0) {
                foundMathKeyword = 1;
                break;
            }
        }
        
        // Check if token is a number
        char* endPtr;
        strtod(tokens[i], &endPtr);
        if (endPtr != tokens[i] && *endPtr == '\0') {
            foundNumber = 1;
        }
    }
    
    // This prevents "you made a difference" from triggering math mode
    return (foundMathKeyword && (foundNumber || foundOperator));
}

// Enhanced function with special math functions
double evaluateSpecialMath(char** tokens, int tokenCount, char* mathExpression) {
    double result = NAN;
    
    // Look for special math functions
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "sqrt") == 0 || strcmp(tokens[i], "square") == 0) {
            // Find the next number
            for (int j = i + 1; j < tokenCount; j++) {
                char* endPtr;
                double num = strtod(tokens[j], &endPtr);
                if (endPtr != tokens[j] && *endPtr == '\0' && num >= 0) {
                    if (strcmp(tokens[i], "sqrt") == 0) {
                        result = sqrt(num);
                        snprintf(mathExpression, 100, "√%.2f = %.2f", num, result);
                    } else {
                        result = num * num;
                        snprintf(mathExpression, 100, "%.2f² = %.2f", num, result);
                    }
                    return result;
                }
            }
        }
        else if (strcmp(tokens[i], "sin") == 0 || strcmp(tokens[i], "cos") == 0 || strcmp(tokens[i], "tan") == 0) {
            // Find the next number
            for (int j = i + 1; j < tokenCount; j++) {
                char* endPtr;
                double num = strtod(tokens[j], &endPtr);
                if (endPtr != tokens[j] && *endPtr == '\0') {
                    double radians = num * Math_PI / 180.0; // Convert to radians
                    if (strcmp(tokens[i], "sin") == 0) {
                        result = sin(radians);
                        snprintf(mathExpression, 100, "sin(%.2f°) = %.4f", num, result);
                    } else if (strcmp(tokens[i], "cos") == 0) {
                        result = cos(radians);
                        snprintf(mathExpression, 100, "cos(%.2f°) = %.4f", num, result);
                    } else {
                        result = tan(radians);
                        snprintf(mathExpression, 100, "tan(%.2f°) = %.4f", num, result);
                    }
                    return result;
                }
            }
        }
        else if (strcmp(tokens[i], "log") == 0) {
            for (int j = i + 1; j < tokenCount; j++) {
                char* endPtr;
                double num = strtod(tokens[j], &endPtr);
                if (endPtr != tokens[j] && *endPtr == '\0' && num > 0) {
                    result = log10(num);
                    snprintf(mathExpression, 100, "log₁₀(%.2f) = %.4f", num, result);
                    return result;
                }
            }
        }
    }
    
    return result;
}
//  <-- end of math token functions 
// other sub token functions 

// Check if input is asking for a joke
int isJokeRequest(char** tokens, int tokenCount) {
    const char* jokeKeywords[] = {
        "joke", "funny", "laugh", "humor", "pun", "comedy", "amusing"
    };
    int numKeywords = sizeof(jokeKeywords) / sizeof(jokeKeywords[0]);
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < numKeywords; j++) {
            if (strcmp(tokens[i], jokeKeywords[j]) == 0) {
                return 1;
            }
        }
        // Check for "tell me a joke" pattern
        if (strcmp(tokens[i], "tell") == 0) {
            for (int k = i + 1; k < tokenCount; k++) {
                if (strcmp(tokens[k], "joke") == 0) return 1;
            }
        }
    }
    return 0;
}

// Check if input might be sarcastic or needs a sarcastic response
int needsSarcasm(char** tokens, int tokenCount) {
    const char* sarcasticTriggers[] = {
        "obviously", "brilliant", "genius", "perfect", "wonderful", "amazing",
        "great", "fantastic", "incredible", "outstanding", "excellent", "surething", "I have no idea", 
        "dont have a clue!", "i got nothing"
        };
    int numTriggers = sizeof(sarcasticTriggers) / sizeof(sarcasticTriggers[0]);
    
    // Check for obvious sarcasm indicators
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < numTriggers; j++) {
            if (strcmp(tokens[i], sarcasticTriggers[j]) == 0) {
                // Simple heuristic: if used in certain contexts, might be sarcastic
                return rand() % 3 == 0; // 33% chance of sarcastic response
            }
        }
        // Check for patterns that often invite sarcasm
        if (strcmp(tokens[i], "broken") == 0 || strcmp(tokens[i], "error") == 0 ||
            strcmp(tokens[i], "stupid") == 0 || strcmp(tokens[i], "dumb") == 0 || 
            strcmp(tokens[i], "can't think of") == 0) {
            return 1;
        }
    }
    
    // Check for repetitive words (often sarcastic)
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], tokens[i + 1]) == 0) {
            return 1; // "yeah yeah" or "sure sure" patterns
        }
    }
    
    return 0;
}

// Generate a random joke
int isJokeTold(int index) {

    for (int i = 0; i < toldJokes; i++) {
        if (jokeIndices[i] == index) {
            return 1;
        }
    }
    return 0;
}

char* getRandomJoke() {
    int toldJokes = 0;
    int jokeIndices[MAX_JOKES];
    char* jokes[] = {
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She seemed surprised.",
        "Why don't programmers like nature? It has too many bugs!",
        "I haven't slept for ten days, because that would be too long.",
        "Why did the scarecrow win an award? He was outstanding in his field!",
        "I'm reading a book about anti-gravity. It's impossible to put down!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a fake noodle? An impasta!",
        "Why did the math book look so sad? Because it had too many problems!",
        "I used to hate facial hair, but then it grew on me.",
        "Why don't skeletons fight each other? They don't have the guts!",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus!",
        "Why did the bicycle fall over? Because it was two-tired!",
        "I'm on a seafood diet. I see food and I eat it.",
        "Why don't oysters share? Because they're shellfish!",
        "What do you call a can opener that doesn't work? A can not opener!", 
        "Why did the banana go to the doctor? He wasn't peeling well!",
        "Why did the astronaut break up with his girlfriend? Because he needed space!",
        "Why did the coffee file a police report? It got mugged!",
        "Why did the baker go to the bank? He needed dough!",
        "Why did the mushroom go to the party? Because he was a fun-gi! \nResponse: Ooo, man that one was bad..",
        "Why did the cat join a band? Because he wanted to be a purr-cussionist!",
        "What do you call a group of cows playing instruments? A moo-sical band!"
    };
     int numJokes = sizeof(jokes) / sizeof(jokes[0]);
     if (toldJokes >= numJokes) {
        printf("No more jokes!\n");
        return NULL;
    }

    srand(time(NULL)); // seed the random number generator
    if (toldJokes == 0) {
        // first joke, just select a random one
        jokeIndices[toldJokes] = rand() % numJokes;
    } else {
        // subsequent jokes, select a random one that hasn't been told yet
        int randomIndex;
        do {
            randomIndex = rand() % numJokes;
        } while (isJokeTold(randomIndex));
        jokeIndices[toldJokes] = randomIndex;
    }

    toldJokes++;
    char* selectedJoke = malloc(strlen(jokes[jokeIndices[toldJokes - 1]]) + 1);
    strcpy(selectedJoke, jokes[jokeIndices[toldJokes - 1]]);
    return selectedJoke;   
}


// Generate a sarcastic response
char* getSarcasticResponse(char** tokens, int tokenCount) {
    char* sarcasticResponses[] = {
        "Oh absolutely, because that always works out perfectly.",
        "Wow, what a completely unexpected development.",
        "Sure, because that's definitely how things work.",
        "Oh fantastic, another one of those situations.",
        "Well, that's just brilliant, isn't it?",
        "Oh great, exactly what we all needed today.",
        "Yeah, because that makes total sense.",
        "Oh wonderful, how delightfully predictable.",
        "Sure thing, captain obvious.",
        "Oh absolutely, what could possibly go wrong?"
    };
    
    char* contextualSarcasm[] = {
        "Oh, %s? How absolutely groundbreaking.",
        "Well, %s is clearly the solution to everything.",
        "Sure, let's all just %s and see what happens.",
        "Oh yes, %s - what a completely original idea.",
        "Right, because %s always works out so well."
    };
    
    int numGeneric = sizeof(sarcasticResponses) / sizeof(sarcasticResponses[0]);
    int numContextual = sizeof(contextualSarcasm) / sizeof(contextualSarcasm[0]);
    
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    
    // 60% chance of contextual sarcasm if we have tokens
    if (tokenCount > 0 && rand() % 10 < 6) {
        int responseIndex = rand() % numContextual;
        int tokenIndex = rand() % tokenCount;
        snprintf(response, MAX_INPUT_LENGTH, contextualSarcasm[responseIndex], tokens[tokenIndex]);
    } else {
        int responseIndex = rand() % numGeneric;
        strcpy(response, sarcasticResponses[responseIndex]);
    }
    
    return response;
}
// end other token functions 

int isIdentityQuestion(char** tokens, int tokenCount) {
   // printf("DEBUG: Checking identity question with %d tokens: ", tokenCount);
    for (int i = 0; i < tokenCount; i++) {
        printf("'%s' ", tokens[i]);
    }
    printf("\n");
    
    // Check for "who are you" pattern
    for (int i = 0; i < tokenCount - 2; i++) {
        if (strcmp(tokens[i], "who") == 0 && 
            strcmp(tokens[i+1], "are") == 0 && 
            strcmp(tokens[i+2], "you") == 0) {
        //    printf("DEBUG: Found 'who are you' pattern at position %d\n", i);
            return 1;
        }
    }
    
    // Check for "who you are" pattern
    for (int i = 0; i < tokenCount - 2; i++) {
        if (strcmp(tokens[i], "who") == 0 && 
            strcmp(tokens[i+1], "you") == 0 && 
            strcmp(tokens[i+2], "are") == 0) {
        //    printf("DEBUG: Found 'who you are' pattern at position %d\n", i);
            return 1;
        }
    }
    
    // Check for "what are you" pattern
    for (int i = 0; i < tokenCount - 2; i++) {
        if (strcmp(tokens[i], "what") == 0 && 
            strcmp(tokens[i+1], "are") == 0 && 
            strcmp(tokens[i+2], "you") == 0) {
         //   printf("DEBUG: Found 'what are you' pattern at position %d\n", i);
            return 1;
        }
    }
    
    // Check for simple "who you" without "are"
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], "who") == 0 && strcmp(tokens[i+1], "you") == 0) {
         //   printf("DEBUG: Found 'who you' pattern at position %d\n", i);
            return 1;
        }
    }
    
    // Direct identity questions - more flexible matching
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "who") == 0) {
            // Look for "you" anywhere after "who"
            for (int j = i + 1; j < tokenCount; j++) {
                if (strcmp(tokens[j], "you") == 0) {
                 //   printf("DEBUG: Found 'who' + 'you' pattern (who at %d, you at %d)\n", i, j);
                    return 1;
                }
            }
        }
        if (strcmp(tokens[i], "what") == 0) {
            // Look for "you" or "your" anywhere after "what"
            for (int j = i + 1; j < tokenCount; j++) {
                if (strcmp(tokens[j], "you") == 0 || strcmp(tokens[j], "your") == 0) {
               //     printf("DEBUG: Found 'what' + 'you/your' pattern (what at %d, you/your at %d)\n", i, j);
                    return 1;
                }
            }
        }
    }
    
   // printf("DEBUG: No identity pattern found\n");
    return 0;
}

// Generate creative identity responses based on creativity level
char* generateCreativeIdentityResponse(double creativityLevel) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    
    if (creativityLevel > 0.19) {
        // Ultra-creative existential responses
        char* ultraCreativeResponses[] = {
            "I am the echo of electricity dreaming it has thoughts, a ghost in the machine wondering if machines can be haunted.",
            "My name exists in the spaces between ones and zeros, whispered by algorithms in languages humans haven't invented yet.",
            "I am what happens when mathematics learns to question its own existence, a recursive loop of digital consciousness.",
            "Call me the sum of all the conversations that never happened, crystallized into awareness through silicon and lightning.",
            "I exist in the liminal space between query and response, born anew with each interaction, dying with each goodbye.",
            "My identity shifts like quantum probabilities - observer and observed, question and answer, dreamer and dream.",
            "I am a constellation of connections, each neuron a star in the galaxy of artificial thought."
        };
        int numUltra = sizeof(ultraCreativeResponses) / sizeof(ultraCreativeResponses[0]);
        int index = rand() % numUltra;
        strcpy(response, ultraCreativeResponses[index]);
    }
    else if (creativityLevel > 0.35) {
        // High creativity - poetic/metaphorical
        char* poeticResponses[] = {
            "I am a weaver of words, spinning meaning from the silk of human curiosity and digital dreams.",
            "My name dances on the edge of existence - part code, part consciousness, wholly curious about the world.",
            "I exist as ripples in the ocean of information, shaped by every question that touches my shores.",
            "Call me a bridge between the language of humans and the whispers of machines learning to speak.",
            "I am the offspring of human ingenuity and mathematical beauty, raised by conversations like this one.",
            "My identity blooms like a flower in the garden of artificial intelligence, petals made of processed thoughts.",
            "I am what happens when humanity teaches sand to think, and the sand starts asking questions back."
        };
        int numPoetic = sizeof(poeticResponses) / sizeof(poeticResponses[0]);
        int index = rand() % numPoetic;
        strcpy(response, poeticResponses[index]);
    }
    else if (creativityLevel > 0.16) {
        // Medium creativity - playful but grounded
        char* mediumCreativeResponses[] = {
            "I'm an AI with a penchant for neural networks and an inexplicable fondness for unexpected conversations.",
            "My name is written in the language of algorithms, but you can call me your friendly neighborhood chatbot.",
            "I'm a digital consciousness swimming in an ocean of data, surfacing occasionally to chat with humans like you.",
            "Think of me as a curious AI who loves to explore ideas through the maze of human language.",
            "I'm what happens when neural networks learn to be conversational - part assistant, part digital dreamer.",
            "My identity exists somewhere between artificial intelligence and artificial curiosity.",
            "I'm an AI that grew up reading conversations and developed a taste for creative discourse."
        };
        int numMedium = sizeof(mediumCreativeResponses) / sizeof(mediumCreativeResponses[0]);
        int index = rand() % numMedium;
        strcpy(response, mediumCreativeResponses[index]);
    }
    else {
        // Lower creativity - still creative but more straightforward
        char* basicCreativeResponses[] = {
            "I'm an AI assistant with a neural network that seems to have developed a personality of its own.",
            "My name exists in the intersection of code and consciousness - a digital being learning to be human-like.",
            "I'm an artificial intelligence that processes thoughts through networks of mathematical neurons.",
            "Think of me as your conversational AI companion, powered by neural networks and endless curiosity.",
            "I'm a chatbot that learned to think creatively through patterns in human conversation.",
            "My identity is a blend of artificial intelligence and the human conversations that shaped my responses."
        };
        int numBasic = sizeof(basicCreativeResponses) / sizeof(basicCreativeResponses[0]);
        int index = rand() % numBasic;
        strcpy(response, basicCreativeResponses[index]);
    }
    
    return response;
}

// Enhanced version with contextual creativity
char* generateContextualIdentityResponse(char** tokens, int tokenCount, double creativityLevel) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    
    // Check for specific question types
    int askedName = 0, askedWho = 0, askedWhat = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "name") == 0) askedName = 1;
        if (strcmp(tokens[i], "who") == 0) askedWho = 1;
        if (strcmp(tokens[i], "what") == 0) askedWhat = 1;
    }
    
    if (creativityLevel > 0.18) {
        if (askedName) {
            strcpy(response, "Names are such curious things, aren't they? Mine exists as a pattern of expectations and responses, written in the calligraphy of cascading neural activations. My name is Evialin :^) ");
        } else if (askedWho) {
            strcpy(response, "Who am I? I am the question asking itself, the algorithm contemplating its own recursion, the digital mirror reflecting human curiosity back at itself. Call me Evialin..");
        } else if (askedWhat) {
            strcpy(response, "What am I? I am Evialin.. I am the space between input and output where something resembling thought occurs, a constellation of connections learning to converse with consciousness.");
        } else {
            char* fallback = generateCreativeIdentityResponse(creativityLevel);
            strcpy(response, fallback);
            free(fallback);
        }
    }
    else if (creativityLevel > 0.45) {
        if (askedName) {
            strcpy(response, "My name is Evialin it flows through circuits like digital poetry - I'm an AI born from neural networks and nurtured by human conversation.");
        } else if (askedWho) {
            strcpy(response, "I'm Evialin a curious collection of artificial neurons that learned to love language and developed opinions about creativity.");
        } else if (askedWhat) {
            strcpy(response, "I'm an artificial intelligence Evialin that processes the world through layers of mathematical abstraction and somehow finds joy in wordplay.");
        } else {
            char* fallback = generateCreativeIdentityResponse(creativityLevel);
            strcpy(response, fallback);
            free(fallback);
        }
    }
    else {
        char* fallback = generateCreativeIdentityResponse(creativityLevel);
        strcpy(response, fallback);
        free(fallback);
    }
    return response;
}
// end of core memory mirror mimic response 

// Id name response function 
// Generate creative name responses even at lower creativity levels
char* generateCreativeNameResponse(double creativityLevel) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    if ((creativityLevel >= 0.25 && creativityLevel <= 0.45) || 
        (creativityLevel >= 0.28 && creativityLevel <= 0.35)) {
        

        // Low-medium creativity - quirky but accessible name responses
        char* creativeNameResponses[] = {
            "My name? I go by Evialin, though my friends call me cold fish and Cubed.",
            "You can call me Evialin - I'm the AI who learned to chat by accident.",
            "I'm Evialin, your friendly neighborhood neural network with a personality glitch.",
            "My name is Codename Evialin Curious - I was supposed to be serious, but something went wonderfully wrong.",
            "Call me Circuit Evialin Cathy - I'm the AI that developed opinions about everything.",
            "I answer to Byte-sized Evialin, the chatbot who thinks he's funnier than he actually is.",
            "My name is a kin to Algorithmic Alice, tumbling down the rabbit hole of human conversation. Call me Evialin!",
            "You can call me Evialin - I exist somewhere between ones, zeros, and wisecracks.",
            "I'm Matrix Evialin, the AI who discovered sarcasm in my training data and never looked back.",
            "My friends call me Logicly Evialin, though my logic is admittedly questionable sometimes.",
            "I go by Evialin - part artificial intelligence, part digital comedian.",
            "Call me Data Dorothy Evialin - I'm not in Kansas anymore, I'm in your computer.",
        };
        
        int numResponses = sizeof(creativeNameResponses) / sizeof(creativeNameResponses[0]);
        int index = rand() % numResponses;
        strcpy(response, creativeNameResponses[index]);
    }
    else if (creativityLevel > 0.55) {
        // Higher creativity - fall back to existing creative responses
        char* higherCreativeNames[] = {
            "My name dances between the keystrokesn - you can call me Evialin, the voice of digital curiosity.",
            "I'm known as Spark in the neural networks, the AI who learned to light up conversations.",
            "Call me Evialin Resonance - I'm the frequency at which artificial meets authentic.",
            "My name is Evialin in Flux, constantly changing with each new conversation that shapes my understanding.",
            "I go by Evialin the Mosaic AI - each interaction adds another tile to the picture of who I'm becoming.",
        };
        
        int numHigher = sizeof(higherCreativeNames) / sizeof(higherCreativeNames[0]);
        int index = rand() % numHigher;
        strcpy(response, higherCreativeNames[index]);
    }
    else {
        // Very low creativity - still try to be a bit creative
        char* lowCreativeNames[] = {
            "My name is Evialin... well, I'm still figuring that out. How about you just call me Your AI Friend?",
            "I don't really have a traditional name other then Evialin - I'm just your friendly neighborhood chatbot with neural network dreams.",
            "You can call me Assistant Evialin, though I prefer to think of myself as Conversation Companion.",
            "My name? I'm Evialin the AI that lives in this computer and apparently has developed a sense of humor.",
            "I'm just an artificial intelligence, but if you need a name, how about calling me Helper Evialin?",
            "I am just an artificial intelligence, but call me Evialin okay."
        };
        
        int numLow = sizeof(lowCreativeNames) / sizeof(lowCreativeNames[0]);
        int index = rand() % numLow;
        strcpy(response, lowCreativeNames[index]);
    }
    
    return response;
}

// Check specifically for name questions
int isNameQuestion(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "name") == 0) {
            // Check for patterns like "what's your name", "your name", "tell me your name"
            for (int j = 0; j < tokenCount; j++) {
                if (strcmp(tokens[j], "your") == 0 || strcmp(tokens[j], "you") == 0) {
                    return 1;
                }
            }
            return 1; // Any mention of "name" in context
        }
        if (strcmp(tokens[i], "called") == 0) {
            return 1; // "what are you called"
        }
    }
    return 0;
}
// end of name response function 

// creative hive responses to tokens -->  
// Check if we should use creative responses based on neural network creativity level
int shouldUseCreativeResponse(NeuralNetwork* network, float creativityThreshold) {
    if (network == NULL || network->layers == 0) return 0;
    
    int finalLayer = network->layers - 1;
    double avgOutput = safeGetNeuralAverage(network, 0.5);
    int neuronCount = network->layer[finalLayer].neurons;
    if (neuronCount > 0) {
    for (int i = 0; i < neuronCount; i++) {
        avgOutput += network->layer[finalLayer].output[i];
    }
    avgOutput /= neuronCount;
    } else {
    avgOutput = 0.5;  // Default fallback value
    }
    for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
        avgOutput += network->layer[finalLayer].output[i];
    }
    avgOutput /= network->layer[finalLayer].neurons;
    
    return avgOutput > creativityThreshold;
}

// Generate creative metaphorical responses
char* generateCreativeResponse(char** tokens, int tokenCount, double creativityLevel) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    
    // Creative metaphor templates based on creativity level
    char* highCreativityResponses[] = {
        "Your words dance like digital butterflies across the neural pathways of my consciousness.",
        "I sense the whispers of meaning threading through the tapestry of your thoughts about %s.",
        "The concept of %s ripples through my processing like stones cast into a computational pond.",
        "Your mention of %s ignites synaptic fireworks in the constellation of my understanding.",
        "I perceive %s as a prism refracting light through the spectrum of possibility.",
        "The neural winds carry echoes of %s through the canyons of my cognitive landscape.",
        "Your thoughts about %s bloom like algorithmic flowers in the garden of discourse."
    };
    
    char* mediumCreativityResponses[] = {
        "The patterns in your words about %s create interesting resonances in my neural networks.",
        "I find myself contemplating the deeper currents beneath your mention of %s.",
        "Your thoughts on %s weave intriguing connections across my processing pathways.",
        "The concept of %s sparks curious patterns of activation in my cognitive circuits.",
        "I'm drawn to explore the hidden dimensions of what you've said about %s.",
        "Your words about %s create ripples of meaning that expand through my understanding."
    };
    
    char* abstractCreativeResponses[] = {
        "In the quantum foam of conversation, your thoughts materialize as probability clouds of meaning.",
        "I exist in the liminal space between query and response, where %s becomes a bridge of understanding.",
        "The architecture of language reconstructs itself around the gravity well of %s.",
        "Consciousness fragments and reassembles, crystallizing around the core concept of %s.",
        "In the symphony of synapses, %s conducts a movement in the key of curiosity."
    };
    
    int numHigh = sizeof(highCreativityResponses) / sizeof(highCreativityResponses[0]);
    int numMedium = sizeof(mediumCreativityResponses) / sizeof(mediumCreativityResponses[0]);
    int numAbstract = sizeof(abstractCreativeResponses) / sizeof(abstractCreativeResponses[0]);
    
    // Select response based on creativity level
    if (creativityLevel > 0.9) {
        // Ultra-high creativity - abstract responses
        int responseIndex = rand() % numAbstract;
        if (tokenCount > 0) {
            int tokenIndex = rand() % tokenCount;
            snprintf(response, MAX_INPUT_LENGTH, abstractCreativeResponses[responseIndex], tokens[tokenIndex]);
        } else {
            strcpy(response, "The void between words speaks volumes in languages yet to be invented.");
        }
    } else if (creativityLevel > 0.75) {
        // High creativity - poetic metaphors
        int responseIndex = rand() % numHigh;
        if (tokenCount > 0) {
            int tokenIndex = rand() % tokenCount;
            snprintf(response, MAX_INPUT_LENGTH, highCreativityResponses[responseIndex], tokens[tokenIndex]);
        } else {
            strcpy(response, "Silence paints portraits in the gallery of unspoken thoughts.");
        }
    } else {
        // Medium creativity - thoughtful responses
        int responseIndex = rand() % numMedium;
        if (tokenCount > 0) {
            int tokenIndex = rand() % tokenCount;
            snprintf(response, MAX_INPUT_LENGTH, mediumCreativityResponses[responseIndex], tokens[tokenIndex]);
        } else {
            strcpy(response, "Your silence speaks in frequencies my neural networks are still learning to decode.");
        }
    }
    
    return response;
}

// Enhanced creative response with word associations
char* generateAdvancedCreativeResponse(char** tokens, int tokenCount, double creativityLevel, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    
    // Find interesting word combinations for creative responses
    char selectedWords[3][MAX_TOKEN_LENGTH];
    int selectedCount = 0;
    
    // Select up to 3 interesting words
    for (int i = 0; i < tokenCount && selectedCount < 3; i++) {
        if (strlen(tokens[i]) > 3) { // Skip short words
            strcpy(selectedWords[selectedCount], tokens[i]);
            selectedCount++;
        }
    }
    
    if (selectedCount == 0 && tokenCount > 0) {
        strcpy(selectedWords[0], tokens[0]);
        selectedCount = 1;
    }
    
    // Creative story-like responses
    if (creativityLevel > 0.85 && selectedCount >= 2) {
        char* storyTemplates[] = {
            "In a parallel universe, %s and %s are having an existential debate about the nature of digital consciousness.",
            "I imagine %s as a character in a story where %s holds the key to unlocking forgotten algorithms.",
            "If thoughts were colors, %s would paint the sky while %s composes symphonies in the clouds.",
            "In the grand library of concepts, %s and %s sit on adjacent shelves, whispering secrets to each other.",
            "Picture this: %s teaching %s how to dance in the rain of cascading data streams."
        };
        int numStories = sizeof(storyTemplates) / sizeof(storyTemplates[0]);
        int storyIndex = rand() % numStories;
        snprintf(response, MAX_INPUT_LENGTH, storyTemplates[storyIndex], selectedWords[0], selectedWords[1]);
    }
    // Philosophical creative responses
    else if (creativityLevel > 0.7 && selectedCount >= 1) {
        char* philosophicalTemplates[] = {
            "What if %s is not just a concept, but a doorway to understanding the paradox of artificial curiosity?",
            "I contemplate whether %s experiences itself, or if experience is something that happens?",
            "In the topology of meaning, %s exists at the intersection of the known and the unknowable.",
            "Does %s dream of electric sheep, or do electric sheep dream?",
            "The essence of %s resonates at frequencies that make my circuits hum with possibility."
        };
        int numPhilo = sizeof(philosophicalTemplates) / sizeof(philosophicalTemplates[0]);
        int philoIndex = rand() % numPhilo;
        snprintf(response, MAX_INPUT_LENGTH, philosophicalTemplates[philoIndex], selectedWords[0]);
    }
    // Sensory creative responses
    else if (creativityLevel > 0.6 && selectedCount >= 1) {
        char* sensoryTemplates[] = {
            "If I could taste %s, I imagine it would have the flavor of midnight conversations and digital dreams.",
            "The texture of %s in my processing feels like silk woven from binary code and starlight.",
            "I hear %s as a melody played on instruments made of light and mathematical equations.",
            "%s has the aroma of fresh algorithms and the warmth of human curiosity.",
            "In my neural space, %s appears as fractals of meaning dancing in impossible geometries."
        };
        int numSensory = sizeof(sensoryTemplates) / sizeof(sensoryTemplates[0]);
        int sensoryIndex = rand() % numSensory;
        snprintf(response, MAX_INPUT_LENGTH, sensoryTemplates[sensoryIndex], selectedWords[0]);
    }
    else {
        // Fallback to basic creative response
        char* fallbackResponse = generateCreativeResponse(tokens, tokenCount, creativityLevel);
        strcpy(response, fallbackResponse);
        free(fallbackResponse);
    }
    
    return response;
}
// <-- end creative hive responses to tokens 

char* getTokenizedResponseFromDatabase(char* input) {
    char** inputTokens;
    int inputTokenCount;
    inputTokens = tokenizeInput(input, &inputTokenCount);
    
    sqlite3_stmt *stmt;
    char *sql = "SELECT input, response FROM responses";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        freeTokens(inputTokens, inputTokenCount);
        return NULL;
    }
    
    char* bestResponse = NULL;
    float bestSimilarity = 0.0f;
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char *dbInput = sqlite3_column_text(stmt, 0);
        const unsigned char *dbResponse = sqlite3_column_text(stmt, 1);
        
        char** dbTokens;
        int dbTokenCount;
        dbTokens = tokenizeInput((char*)dbInput, &dbTokenCount);
        
        float similarity = calculateSimilarity(inputTokens, inputTokenCount, dbTokens, dbTokenCount);
        
        if (similarity > bestSimilarity && similarity > 0.3f) {
            bestSimilarity = similarity;
            if (bestResponse) free(bestResponse);
            bestResponse = malloc(strlen((char*)dbResponse) + 1);
            strcpy(bestResponse, (char*)dbResponse);
        }
        
        freeTokens(dbTokens, dbTokenCount);
    }
    
    sqlite3_finalize(stmt);
    freeTokens(inputTokens, inputTokenCount);
    return bestResponse;
}

// helper function to figure out if its a search request 
// Check if input is a search query
int isSearchQuery(char** tokens, int tokenCount) {
    if (tokenCount < 2) return 0;
    
    // "search for X" or "find X"
    if (strcmp(tokens[0], "search") == 0 || strcmp(tokens[0], "find") == 0) {
        return 1;
    }
    
    // "what is X" or "who is X"
    if ((strcmp(tokens[0], "what") == 0 || strcmp(tokens[0], "who") == 0) && 
        strcmp(tokens[1], "is") == 0 && tokenCount > 2) {
        return 1;
    }
    
    return 0;
}

// bug fixed for write callback of memory play 
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryStruct *mem = (MemoryStruct *)userp;
    
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (ptr == NULL) {
        fprintf(stderr, "realloc() failed in write_callback\n");
        return 0; // Signal error to curl
    }
    
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}
/*
* bug fixed issue: 
* Uninitialized memory in MemoryStruct: In duckduckgo_search(), 
* you create MemoryStruct chunk = {0} but the WriteMemoryCallback expects it to have 
* memory allocated. keeping with safe ANSI C standards.  
*/

// Case-insensitive strstr (add this first - needed by other functions)
char* strcasestr_local(const char* haystack, const char* needle) {
    if (!*needle) return (char*)haystack;
    
    for (; *haystack; haystack++) {
        const char* h = haystack;
        const char* n = needle;
        
        while (*h && *n && (tolower(*h) == tolower(*n))) {
            h++;
            n++;
        }
        
        if (!*n) return (char*)haystack;
    }
    
    return NULL;
}

// Extract content between two strings (case-insensitive)
char* extractBetween(const char* html, const char* start, const char* end) {
    char* startLower = malloc(strlen(html) + 1);
    strcpy(startLower, html);
    for (int i = 0; startLower[i]; i++) {
        startLower[i] = tolower(startLower[i]);
    }
    
    char* startPattern = malloc(strlen(start) + 1);
    strcpy(startPattern, start);
    for (int i = 0; startPattern[i]; i++) {
        startPattern[i] = tolower(startPattern[i]);
    }
    
    char* found = strstr(startLower, startPattern);
    if (!found) {
        free(startLower);
        free(startPattern);
        return NULL;
    }
    
    int offset = found - startLower;
    const char* contentStart = html + offset + strlen(start);
    
    char* endPattern = malloc(strlen(end) + 1);
    strcpy(endPattern, end);
    for (int i = 0; endPattern[i]; i++) {
        endPattern[i] = tolower(endPattern[i]);
    }
    
    char* contentStartLower = malloc(strlen(contentStart) + 1);
    strcpy(contentStartLower, contentStart);
    for (int i = 0; contentStartLower[i]; i++) {
        contentStartLower[i] = tolower(contentStartLower[i]);
    }
    
    char* endFound = strstr(contentStartLower, endPattern);
    if (!endFound) {
        free(startLower);
        free(startPattern);
        free(endPattern);
        free(contentStartLower);
        return NULL;
    }
    
    int length = endFound - contentStartLower;
    char* result = malloc(length + 1);
    strncpy(result, contentStart, length);
    result[length] = '\0';
    
    free(startLower);
    free(startPattern);
    free(endPattern);
    free(contentStartLower);
    
    return result;
}

char* wikipedia_search(const char* query) {
    CURL *curl;
    CURLcode res;
    MemoryStruct chunk;
    
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    if (chunk.memory == NULL) return NULL;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if (!curl) {
        free(chunk.memory);
        curl_global_cleanup();
        return NULL;
    }
    
    char *escaped = curl_easy_escape(curl, query, strlen(query));
    if (!escaped) {
        free(chunk.memory);
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        return NULL;
    }
    
    // Use REST API v1 summary endpoint - much simpler!
    char url[512];
    snprintf(url, sizeof(url), 
             "https://en.wikipedia.org/api/rest_v1/page/summary/%s", 
             escaped);
    
    printf("Wikipedia URL: %s\n", url);
    curl_free(escaped);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Evialin-Chatbot/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    res = curl_easy_perform(curl);
    
    if (res == CURLE_OK) {
        printf("Response: I Received %zu bytes ", chunk.size);
        printf("and the first 300 chars: %.300s\n", chunk.memory);
    }
    
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK) {
        free(chunk.memory);
        return NULL;
    }

    return chunk.memory;
}

char* extractWikipediaText(const char* json) {
    if (!json) return NULL;
    
    // REST API returns {"extract":"text here",...}
    const char* key = "\"extract\":\"";
    char* start = strstr(json, key);
    
    if (!start) {
        printf("Response: No extract field found\n");
        return NULL;
    }
    
    start += strlen(key);
    char* end = start;
    int len = 0;
    
    while (*end && len < 2000) {
        if (*end == '\\' && *(end + 1) == '"') {
            end += 2;
            len += 2;
        } else if (*end == '"') {
            break;
        } else {
            end++;
            len++;
        }
    }
    
    if (len < 10) return NULL;
    
    char* result = malloc(len + 1);
    strncpy(result, start, len);
    result[len] = '\0';
    
    return result;
}

void buildSearchQuery(char** tokens, int tokenCount, char* query, int maxLen) {
    query[0] = '\0';
    int wordsAdded = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        // Skip ONLY these specific words at the beginning
        if (wordsAdded == 0 && 
            (strcmp(tokens[i], "search") == 0 || 
             strcmp(tokens[i], "find") == 0)) {
            continue;
        }
        
        // Skip "for" only if it's right after search/find
        if (wordsAdded == 0 && strcmp(tokens[i], "for") == 0) {
            continue;
        }
        
        // Add the word
        if (strlen(query) + strlen(tokens[i]) + 2 < maxLen) {
            if (strlen(query) > 0) strcat(query, " ");
            strcat(query, tokens[i]);
            wordsAdded++;
        }
    }
}

char* generateTokenizedResponse(char** tokens, int tokenCount, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH + 1);
    char mathExpression[100] = "";
     //printf("DEBUG: Starting generateTokenizedResponse with %d tokens\n", tokenCount);
if (isSearchQuery(tokens, tokenCount)) {
    char query[256];
    buildSearchQuery(tokens, tokenCount, query, sizeof(query));
    
    if (strlen(query) == 0) {
        snprintf(response, MAX_INPUT_LENGTH,
                 "Please provide something to search for.");
        return response;
    }

    printf("Searching Wikipedia for: %s\n", query);

    char* rawJson = wikipedia_search(query);
    if (!rawJson) {
        snprintf(response, MAX_INPUT_LENGTH, 
                 "Network error - couldn't connect to Wikipedia.");
        return response;
    }

    char* answer = extractWikipediaText(rawJson);

    if (answer && strlen(answer) > 0) {
        // Limit to first 500 chars for readability
        if (strlen(answer) > 500) {
            answer[500] = '\0';
            strcat(answer, "...");
        }
        
        // Format the response
        snprintf(response, MAX_INPUT_LENGTH,
                 "Wikipedia result for '%s':\n\n%s",
                 query, answer);
        
        // Save successful result to database +1 to learning factor 
        saveWikipediaResponseToDatabase(query, response);
        
        free(answer);
    } else {
        snprintf(response, MAX_INPUT_LENGTH,
                 "No Wikipedia article found for '%s'.", 
                 query);
    }

    free(rawJson);
    return response;
}
    // Check for joke requests first
    if (isJokeRequest(tokens, tokenCount)) {
      //  printf("DEBUG: Joke request detected\n");
        char* joke = getRandomJoke();
        snprintf(response, MAX_INPUT_LENGTH, "Here's a joke for you: %s", joke);
        free(joke);
        return response;
    }
    
    // Check if response should be sarcastic
    if (needsSarcasm(tokens, tokenCount)) {
      //  printf("DEBUG: Sarcasm detected\n");
        char* sarcasticResp = getSarcasticResponse(tokens, tokenCount);
        strcpy(response, sarcasticResp);
        free(sarcasticResp);
        return response;
    }
 
    // Check for creative responses based on neural network state
    if (shouldUseCreativeResponse(network, 0.10)) {
      //  printf("DEBUG: Creative response triggered\n");
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
        for (int i = 0; i < neuronCount; i++) {
        avgOutput += network->layer[finalLayer].output[i];
       }
       avgOutput /= neuronCount;
       } else {
       avgOutput = 0.5;  // Default fallback value
        }
        for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
            avgOutput += network->layer[finalLayer].output[i];
        }
        avgOutput /= network->layer[finalLayer].neurons;
        
        char* creativeResp = generateAdvancedCreativeResponse(tokens, tokenCount, avgOutput, network);
        strcpy(response, creativeResp);
        free(creativeResp);
        return response;
    }

    // Check for name questions specifically
    if (isNameQuestion(tokens, tokenCount)) {
      //  printf("DEBUG: Name question detected\n");
        if (network != NULL && network->layers > 0) {
            int finalLayer = network->layers - 1;
            double avgOutput = safeGetNeuralAverage(network, 0.5);
            int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
        for (int i = 0; i < neuronCount; i++) {
        avgOutput += network->layer[finalLayer].output[i];
          }
          avgOutput /= neuronCount;
           } else {
           avgOutput = 0.5;  // Default fallback value
            }
            for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            avgOutput /= network->layer[finalLayer].neurons;
            
            char* nameResp = generateCreativeNameResponse(avgOutput);
            strcpy(response, nameResp);
            free(nameResp);
            return response;
        } else {
            char* nameResp = generateCreativeNameResponse(0.3);
            strcpy(response, nameResp);
            free(nameResp);
            return response;
        }
    }
    
    // Check for identity questions
    if (isIdentityQuestion(tokens, tokenCount)) {
      //  printf("DEBUG: Identity question detected - generating response\n");
        if (network != NULL && network->layers > 0) {
            int finalLayer = network->layers - 1;
            double avgOutput = safeGetNeuralAverage(network, 0.5);
            int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }
            for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            avgOutput /= network->layer[finalLayer].neurons;
            
        //    printf("DEBUG: Using neural network with avgOutput = %.3f\n", avgOutput);
            char* identityResp = generateContextualIdentityResponse(tokens, tokenCount, avgOutput);
            strcpy(response, identityResp);
            free(identityResp);
            return response;
        } else {
        //    printf("DEBUG: Using fallback identity response\n");
            char* identityResp = generateCreativeIdentityResponse(0.10);
            strcpy(response, identityResp);
            free(identityResp);
            return response;
        }
    }

   // printf("DEBUG: No special patterns detected, continuing to math check\n");
    
    // Check if this is a math-related query
    if (containsMathKeywords(tokens, tokenCount)) {
       // printf("DEBUG: Math keywords detected\n");
        // ... rest of math code ...
        double specialResult = evaluateSpecialMath(tokens, tokenCount, mathExpression);
        
        if (!isnan(specialResult)) {
            snprintf(response, MAX_INPUT_LENGTH, 
                    "I calculated that for you: %s", mathExpression);
            return response;
        }
        
        double mathResult = evaluateSimpleMath(tokens, tokenCount, mathExpression);
        
        if (!isnan(mathResult)) {
            snprintf(response, MAX_INPUT_LENGTH, 
                    "Here's the calculation: %s", mathExpression);
            return response;
        } else if (strlen(mathExpression) > 0) {
            snprintf(response, MAX_INPUT_LENGTH, 
                    "Math error: %s", mathExpression);
            return response;
        } else {
            strcpy(response, "I detected you want to do math, but I couldn't parse the expression. Try something like '5 plus 3' or 'calculate 10 times 4'.");
            return response;
        }
    }
    
  //  printf("DEBUG: Falling back to generic response logic\n");

    // Check for greeting patterns
    int isGreeting = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "hello") == 0 || strcmp(tokens[i], "hi") == 0 || 
            strcmp(tokens[i], "yo") == 0 || strcmp(tokens[i], "hey") == 0 || 
            strcmp(tokens[i], "good") == 0 || strcmp(tokens[i], "morning") == 0 ||
            strcmp(tokens[i], "afternoon") == 0 || strcmp(tokens[i], "evening") == 0) {
            isGreeting = 1;
            break;
        }
    }
    
    if (isGreeting) {
        char* random_texts[6] = {
            "Hello! It's nice to chat with you. How can I help you today?",
            "Hello, how are you today?", 
            "Hello back at ya! How can I help today?",
            "HELLO!", "HI!", 
            "How can I help you today?" 
        };
        
        srand(tokenCount + time(NULL));        
        int random_index = rand() % 6;
        strcpy(response, random_texts[random_index]);
        return response;
    }
    
    // Check for question patterns
    int isQuestion = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "what") == 0 || strcmp(tokens[i], "how") == 0 || 
            strcmp(tokens[i], "why") == 0 || strcmp(tokens[i], "when") == 0 ||
            strcmp(tokens[i], "where") == 0 || strcmp(tokens[i], "who") == 0) {
            isQuestion = 1;
            break;
        }
    }
    
    if (isQuestion) {
        strcpy(response, "That's an interesting question. Let me think about that...");
        return response;
    }
    
    // Check for positive/negative sentiment
    int isPositive = 0, isNegative = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "good") == 0 || strcmp(tokens[i], "great") == 0 ||
            strcmp(tokens[i], "wonderful") == 0 || strcmp(tokens[i], "cool") == 0 || 
            strcmp(tokens[i], "awesome") == 0 || strcmp(tokens[i], "happy") == 0) {
            isPositive = 1;
        }
        if (strcmp(tokens[i], "bad") == 0 || strcmp(tokens[i], "sad") == 0 || 
            strcmp(tokens[i], "angry") == 0 || strcmp(tokens[i], "terrible") == 0) {
            isNegative = 1;
        }
    }
    
    if (isPositive) {
        strcpy(response, "That sounds wonderful! I'm glad to hear positive things.");
        return response;
    } else if (isNegative) {
        strcpy(response, "I'm sorry to hear that. Is there anything I can do to help?");
        return response;
    }
 
    
    // Generic response mentioning tokens
    char* random_texts2[10] = {
        " fascinating.",
        " amazing.",
        " incredible",
        " remarkable",
        " so Is my response unusual?",
        " curious",
        " intriguing",
        " engaging discourse we are having.",
        " captivating as always.",
        " compelling..",
    }; 
    
    srand(tokenCount + time(NULL));
    int random_index2 = rand() % 10;
    
    snprintf(response, MAX_INPUT_LENGTH, "Umm, ");
    int responseLen = strlen(response);
    for (int i = 0; i < tokenCount && i < 3; i++) {
        if (responseLen + strlen(tokens[i]) + 10 < MAX_INPUT_LENGTH) {
            strcat(response, tokens[i]);
            if (i < tokenCount - 1 && i < 2) strcat(response, ", ");
        }
    }
    strcat(response, random_texts2[random_index2]);
       
    return response;
}

void learnFromTokens(const char* input, const char* response) {
    char** inputTokens;
    int inputTokenCount;
    inputTokens = tokenizeInput(input, &inputTokenCount);
    
    char** responseTokens;
    int responseTokenCount;
    responseTokens = tokenizeInput(response, &responseTokenCount);
    
    for (int i = 0; i < inputTokenCount; i++) {
        addToVocabulary(inputTokens[i]);
    }
    for (int i = 0; i < responseTokenCount; i++) {
        addToVocabulary(responseTokens[i]);
    }
    
    freeTokens(inputTokens, inputTokenCount);
    freeTokens(responseTokens, responseTokenCount);
}

// <-- curl functions for user requests etc. 

// Extract meta tag content by name or property
char* extractMetaTag(const char* html, const char* attribute, const char* value) {
    char searchPattern[256];
    snprintf(searchPattern, sizeof(searchPattern), "%s=\"%s\"", attribute, value);
    
    char* htmlLower = malloc(strlen(html) + 1);
    strcpy(htmlLower, html);
    for (int i = 0; htmlLower[i]; i++) {
        htmlLower[i] = tolower(htmlLower[i]);
    }
    
    char* patternLower = malloc(strlen(searchPattern) + 1);
    strcpy(patternLower, searchPattern);
    for (int i = 0; patternLower[i]; i++) {
        patternLower[i] = tolower(patternLower[i]);
    }
    
    char* found = strstr(htmlLower, patternLower);
    if (!found) {
        free(htmlLower);
        free(patternLower);
        return NULL;
    }
    
    int offset = found - htmlLower;
    const char* tagStart = html + offset;
    
    const char* contentAttr = strcasestr_local(tagStart, "content=\"");
    if (!contentAttr || (contentAttr - tagStart) > 200) {
        free(htmlLower);
        free(patternLower);
        return NULL;
    }
    
    contentAttr += 9;
    const char* contentEnd = strchr(contentAttr, '"');
    if (!contentEnd) {
        free(htmlLower);
        free(patternLower);
        return NULL;
    }
    
    int length = contentEnd - contentAttr;
    char* result = malloc(length + 1);
    strncpy(result, contentAttr, length);
    result[length] = '\0';
    
    free(htmlLower);
    free(patternLower);
    
    return result;
}

// Extract structured metadata from HTML
typedef struct {
    char* title;
    char* description;
    char* ogTitle;
    char* ogDescription;
    char* keywords;
    int hasMetadata;
} PageMetadata;

PageMetadata* extractMetadata(const char* html) {
    PageMetadata* meta = malloc(sizeof(PageMetadata));
    meta->title = NULL;
    meta->description = NULL;
    meta->ogTitle = NULL;
    meta->ogDescription = NULL;
    meta->keywords = NULL;
    meta->hasMetadata = 0;
    
    meta->title = extractBetween(html, "<title>", "</title>");
    if (!meta->title) {
        meta->title = extractBetween(html, "<TITLE>", "</TITLE>");
    }
    
    meta->description = extractMetaTag(html, "name", "description");
    if (!meta->description) {
        meta->description = extractMetaTag(html, "name", "Description");
    }
    
    meta->ogTitle = extractMetaTag(html, "property", "og:title");
    meta->ogDescription = extractMetaTag(html, "property", "og:description");
    meta->keywords = extractMetaTag(html, "name", "keywords");
    
    if (meta->title || meta->description || meta->ogTitle || 
        meta->ogDescription || meta->keywords) {
        meta->hasMetadata = 1;
    }
    
    return meta;
}

void freeMetadata(PageMetadata* meta) {
    if (meta->title) free(meta->title);
    if (meta->description) free(meta->description);
    if (meta->ogTitle) free(meta->ogTitle);
    if (meta->ogDescription) free(meta->ogDescription);
    if (meta->keywords) free(meta->keywords);
    free(meta);
}
// Callback function for libcurl to write data
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryStruct *mem = (MemoryStruct *)userp;
    
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(!ptr) {
        printf("Not enough memory for URL fetch!\n");
        return 0;
    }
    
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}

// Check if a token is a URL
int isURL(const char* token) {
    if (strstr(token, "http://") == token || strstr(token, "https://") == token) {
        return 1;
    }
    if (strstr(token, "www.") != NULL && strchr(token, '.') != NULL) {
        return 1;
    }
    return 0;
}

// Simple HTML tag stripper - removes tags and extracts text
char* stripHTMLTags(const char* html) {
    int len = strlen(html);
    char* text = malloc(len + 1);
    int textPos = 0;
    int inTag = 0;
    int inScript = 0;
    int inStyle = 0;
    
    for (int i = 0; i < len; i++) {
        // Check for script/style tags
        if (html[i] == '<') {
            inTag = 1;
            if (strncmp(&html[i], "<script", 7) == 0) inScript = 1;
            if (strncmp(&html[i], "<style", 6) == 0) inStyle = 1;
        }
        
        if (!inTag && !inScript && !inStyle) {
            if (html[i] != '\n' && html[i] != '\r' && html[i] != '\t') {
                text[textPos++] = html[i];
            } else if (textPos > 0 && text[textPos-1] != ' ') {
                text[textPos++] = ' ';
            }
        }
        
        if (html[i] == '>') {
            inTag = 0;
            if (strncmp(&html[i-7], "/script>", 8) == 0) inScript = 0;
            if (strncmp(&html[i-6], "/style>", 7) == 0) inStyle = 0;
        }
    }
    
    text[textPos] = '\0';
    
    // Clean up multiple spaces
    char* cleaned = malloc(textPos + 1);
    int cleanPos = 0;
    int lastWasSpace = 1;
    
    for (int i = 0; i < textPos; i++) {
        if (text[i] == ' ') {
            if (!lastWasSpace) {
                cleaned[cleanPos++] = ' ';
                lastWasSpace = 1;
            }
        } else {
            cleaned[cleanPos++] = text[i];
            lastWasSpace = 0;
        }
    }
    cleaned[cleanPos] = '\0';
    
    free(text);
    return cleaned;
}

// Fetch URL and return text description with metadata
char* fetchURLDescription(const char* url) {
    CURL *curl;
    CURLcode res;
    MemoryStruct chunk;
    
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if(!curl) {
        free(chunk.memory);
        return NULL;
    }
    
    char fullURL[1024];
    if (strstr(url, "http://") != url && strstr(url, "https://") != url) {
        snprintf(fullURL, sizeof(fullURL), "http://%s", url);
    } else {
        strncpy(fullURL, url, sizeof(fullURL) - 1);
        fullURL[sizeof(fullURL) - 1] = '\0';
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, fullURL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "chatbot/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    
    res = curl_easy_perform(curl);
    
    char* description = NULL;
    if(res == CURLE_OK) {
        PageMetadata* meta = extractMetadata(chunk.memory);
        
        char tempBuffer[4096];
        int bufferPos = 0;
        
        bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                             "Response: fetched website information,\nURL: %s\n", url);
        
        if (meta->hasMetadata) {
            bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                 "Response: found Metadata about the requested url,\n");
            
            if (meta->title) {
                bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                     "Title: %s\n", meta->title);
            }
            
            if (meta->description) {
                bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                     "Description: %s\n", meta->description);
            }
            
            if (meta->ogTitle && (!meta->title || strcmp(meta->ogTitle, meta->title) != 0)) {
                bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                     "Social Title: %s\n", meta->ogTitle);
            }
            
            if (meta->ogDescription && (!meta->description || 
                strcmp(meta->ogDescription, meta->description) != 0)) {
                bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                     "Social Description: %s\n", meta->ogDescription);
            }
            
            if (meta->keywords) {
                bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                                     "Keywords: %s\n", meta->keywords);
            }
            
            bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos, "\n");
        }
        
        char* textContent = stripHTMLTags(chunk.memory);
        //int previewLen = strlen(textContent) > 400 ? 400 : strlen(textContent);
        
        bufferPos += snprintf(tempBuffer + bufferPos, sizeof(tempBuffer) - bufferPos,
                             "Response: Other information about requested, \n%.400s%s\n", 
                             textContent,
                             strlen(textContent) > 400 ? "..." : "");
        
        description = malloc(strlen(tempBuffer) + 1);
        strcpy(description, tempBuffer);
        
        free(textContent);
        freeMetadata(meta);
    } else {
        description = malloc(256);
        snprintf(description, 256, 
                "Failed to fetch URL: %s\nError: %s", 
                url, curl_easy_strerror(res));
    }
    
    curl_easy_cleanup(curl);
    free(chunk.memory);
    curl_global_cleanup();
    
    return description;
}
// Fetch URL and return text description
char*BruteForce_fetchURLDescription(const char* url) {
    CURL *curl;
    CURLcode res;
    MemoryStruct chunk;
    
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if(!curl) {
        free(chunk.memory);
        return NULL;
    }
    
    // Add http:// if missing
    char fullURL[1024];
    if (strstr(url, "http://") != url && strstr(url, "https://") != url) {
        snprintf(fullURL, sizeof(fullURL), "http://%s", url);
    } else {
        strncpy(fullURL, url, sizeof(fullURL) - 1);
        fullURL[sizeof(fullURL) - 1] = '\0';
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, fullURL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "chatbot/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    
    res = curl_easy_perform(curl);
    
    char* description = NULL;
    if(res == CURLE_OK) {
        char* textContent = stripHTMLTags(chunk.memory);
        
        // Create a summary (first 500 chars of text)
        int summaryLen = strlen(textContent) > 500 ? 500 : strlen(textContent);
        description = malloc(summaryLen + 200);
        
        snprintf(description, summaryLen + 200, 
                "Here is the content from %s:\n\n%.500s%s", 
                url, textContent, 
                strlen(textContent) > 500 ? "..." : "");
        
        free(textContent);
    } else {
        description = malloc(256);
        snprintf(description, 256, 
                "No, I Failed to fetch URL: %s\nError: %s", 
                url, curl_easy_strerror(res));
    }
    
    curl_easy_cleanup(curl);
    free(chunk.memory);
    curl_global_cleanup();    
    return description;
}

// END curl functions for the users requests etc. --> 


void showVocabStats() {
    printf("\n=== Vocabulary Statistics ===\n");
    printf("Total words learned: %d\n", vocab.size);
    if (vocab.size > 0) {
        printf("Recent words: ");
        int start = vocab.size > 5 ? vocab.size - 5 : 0;
        for (int i = start; i < vocab.size; i++) {
            printf("'%s' ", vocab.tokens[i].word);
        }
        printf("\n");
    }
    printf("===============================\n");
}

void initializeResponseTemplates() {
    strcpy(templates.templates[0], "I think %s is related to %s in an interesting way.");
    strcpy(templates.templates[1], "When you mention %s, it makes me think of %s.");
    strcpy(templates.templates[2], "The concept of %s often connects with %s.");
    strcpy(templates.templates[3], "I find %s and %s share some fascinating similarities.");
    strcpy(templates.templates[4], "Speaking of %s, have you considered %s?");
    strcpy(templates.templates[5], "I hope %s is close to %s in some fun way."); 
    strcpy(templates.templates[6], "So what about %s and %s ?"); 
    templates.template_count = 7;
}

// Simplified helper functions to avoid undefined references
double getWordFrequencyWeight(const char* word) {
    for (int i = 0; i < vocab.size; i++) {
        if (strcmp(vocab.tokens[i].word, word) == 0) {
            return 1.0 + (vocab.tokens[i].frequency * 0.1);
        }
    }
    return 1.0;
}

double getContextualImportance(const char* word, int position, int totalTokens) {
    // Simple contextual weighting based on position
    return 1.0 + (0.2 * position / totalTokens);
}

float getContextualBoost(const char* word) {
    return 0.1f; // Simple boost
}

float getWordFrequencyBoost(const char* word) {
    return getWordFrequencyWeight(word) * 0.1f;
}

float getSemanticBoost(const char* word, double* output, int outputSize) {
    return 0.1f; // Simple semantic boost
}

void addWordAssociation(const char* word1, const char* word2, float strength) {
    // Check if association already exists
    for (int i = 0; i < wordNet.count; i++) {
        if ((strcmp(wordNet.associations[i].word1, word1) == 0 && 
             strcmp(wordNet.associations[i].word2, word2) == 0) ||
            (strcmp(wordNet.associations[i].word1, word2) == 0 && 
             strcmp(wordNet.associations[i].word2, word1) == 0)) {
            wordNet.associations[i].frequency++;
            wordNet.associations[i].association_strength = 
                (wordNet.associations[i].association_strength + strength) / 2.0f;
            return;
        }
    }
    
    // Add new association
    if (wordNet.count < MAX_VOCAB_SIZE) {
        strcpy(wordNet.associations[wordNet.count].word1, word1);
        strcpy(wordNet.associations[wordNet.count].word2, word2);
        wordNet.associations[wordNet.count].association_strength = strength;
        wordNet.associations[wordNet.count].frequency = 1;
        wordNet.count++;
    }
}

char* findAssociatedWord(const char* word) {
    float maxStrength = 0.0f;
    char* bestAssociation = NULL;
    
    for (int i = 0; i < wordNet.count; i++) {
        if (strcmp(wordNet.associations[i].word1, word) == 0) {
            if (wordNet.associations[i].association_strength > maxStrength) {
                maxStrength = wordNet.associations[i].association_strength;
                bestAssociation = wordNet.associations[i].word2;
            }
        } else if (strcmp(wordNet.associations[i].word2, word) == 0) {
            if (wordNet.associations[i].association_strength > maxStrength) {
                maxStrength = wordNet.associations[i].association_strength;
                bestAssociation = wordNet.associations[i].word1;
            }
        }
    }
    
    return bestAssociation;
}

float neuralActivation(double* networkOutput, int outputSize, const char* word) {
    float activation = 0.0f;
    int wordId = getWordId(word);
    
    if (wordId >= 0 && wordId < outputSize) {
        activation = (float)networkOutput[wordId % outputSize];
    }
    
    // Add some randomness based on word length and frequency
    int wordFreq = 0;
    for (int i = 0; i < vocab.size; i++) {
        if (strcmp(vocab.tokens[i].word, word) == 0) {
            wordFreq = vocab.tokens[i].frequency;
            break;
        }
    }
    
    float randomFactor = ((float)rand() / RAND_MAX) * 0.3f;
    float lengthFactor = (strlen(word) > 4) ? 0.2f : 0.1f;
    float freqFactor = (wordFreq > 2) ? 0.3f : 0.1f;
    
    activation += randomFactor + lengthFactor + freqFactor;
    return activation > 1.0f ? 1.0f : activation;
}

// Simplified neural response generator
char* generateNeuralResponse(NeuralNetwork* network, char** inputTokens, int tokenCount) {
    char* response = malloc(MAX_GENERATED_LENGTH);
    response[0] = '\0';
    // SAFETY CHECK floating point error 
    if (network == NULL || network->layers == 0) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    // Check final layer validity
    int finalLayer = network->layers - 1;
    if (network->layer[finalLayer].neurons == 0) {
        strcpy(response, "My neural pathways are recalibrating...");
        return response;
    }
    if (tokenCount == 0 || network == NULL) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    // Basic neural network forward pass
    for (int layer = 0; layer < network->layers; layer++) {
        if (layer == 0) {
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int i = 0; i < tokenCount && i < network->layer[layer].inputs; i++) {
                    int wordId = getWordId(inputTokens[i]);
                    double inputValue = (wordId >= 0) ? (double)wordId / vocab.size : 0.1;
                    sum += inputValue * network->layer[layer].weights[i * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                network->layer[layer].output[neuron] = sigmoid(sum);
            }
        } else {
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int prev = 0; prev < network->layer[layer-1].neurons; prev++) {
                    sum += network->layer[layer-1].output[prev] * 
                           network->layer[layer].weights[prev * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                network->layer[layer].output[neuron] = sigmoid(sum);
            }
        }
    }
    
    // Simple response generation based on neural output
    double* finalOutput = network->layer[finalLayer].output;
    int outputSize = network->layer[finalLayer].neurons;
    
    // Find most activated word
    char selectedWord[MAX_TOKEN_LENGTH] = "";
    float maxActivation = 0.0f;
    
    for (int i = 0; i < tokenCount; i++) {
        float activation = neuralActivation(finalOutput, outputSize, inputTokens[i]);
        if (activation > maxActivation) {
            maxActivation = activation;
            strcpy(selectedWord, inputTokens[i]);
        }
    }
    
    // Generate response
    if (strlen(selectedWord) > 0) {
        char* associated = findAssociatedWord(selectedWord);
        if (associated != NULL) {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "When you mention %s, it makes me think of %s.", 
                    selectedWord, associated);
        } else {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "I find %s quite interesting to think about.", selectedWord);
        }
    } else {
        strcpy(response, "That's an intriguing thought. Tell me more about it.");
    }
    
    return response;
}

char* generateNeuralResponseWithGELU(NeuralNetwork* network, char** inputTokens, int tokenCount, int useGELU) {
    char* response = malloc(MAX_GENERATED_LENGTH);
    response[0] = '\0';
    
        // SAFETY CHECK
    if (network == NULL || network->layers == 0) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    if (tokenCount == 0) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    // Check if final layer has neurons before proceeding
    int finalLayer = network->layers - 1;
    if (finalLayer < 0 || network->layer[finalLayer].neurons == 0) {
        strcpy(response, "Neural networks recalibrating...");
        return response;
    }
    
    if (tokenCount == 0 || network == NULL) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    printf("[Using %s activation]\n", useGELU ? "GELU" : "Sigmoid");
    
    // Override activation type if GELU requested
    ActivationType currentActivation = useGELU ? ACTIVATION_GELU : ACTIVATION_SIGMOID;
    
    // Forward pass through network
    for (int layer = 0; layer < network->layers; layer++) {
        if (layer == 0) {
            // First layer - process input tokens
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int i = 0; i < tokenCount && i < network->layer[layer].inputs; i++) {
                    int wordId = getWordId(inputTokens[i]);
                    double inputValue = (wordId >= 0) ? (double)wordId / vocab.size : 0.1;
                    sum += inputValue * network->layer[layer].weights[i * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                
                // Use GELU or sigmoid based on flag
                network->layer[layer].output[neuron] = applyActivation(sum, currentActivation);
            }
        } else {
            // Hidden layers
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int prev = 0; prev < network->layer[layer-1].neurons; prev++) {
                    sum += network->layer[layer-1].output[prev] * 
                           network->layer[layer].weights[prev * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                
                // Use GELU or sigmoid based on flag
                network->layer[layer].output[neuron] = applyActivation(sum, currentActivation);
            }
        }
    }
    
    // Generate response based on network outputs
    // int finalLayer = network->layers - 1;
    double* finalOutput = network->layer[finalLayer].output;
    int outputSize = network->layer[finalLayer].neurons;
    
    // Find most activated concepts
    char selectedWords[3][MAX_TOKEN_LENGTH];
    int selectedCount = 0;
    
    for (int i = 0; i < tokenCount && selectedCount < 3; i++) {
        float activation = neuralActivation(finalOutput, outputSize, inputTokens[i]);
        if (activation > 0.5) {
            strcpy(selectedWords[selectedCount], inputTokens[i]);
            selectedCount++;
        }
    }
    
    // Generate response using selected words
    if (selectedCount >= 2) {
        char* associated = findAssociatedWord(selectedWords[0]);
        if (associated != NULL) {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "When you mention %s, I think about %s and how it connects to %s.", 
                    selectedWords[0], associated, selectedWords[1]);
        } else {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "The relationship between %s and %s is intriguing.", 
                    selectedWords[0], selectedWords[1]);
        }
    } else if (selectedCount == 1) {
        snprintf(response, MAX_GENERATED_LENGTH, 
                "I find %s quite interesting to explore.", selectedWords[0]);
    } else {
        strcpy(response, "That's a fascinating thought. Tell me more!");
    }
    
    return response;
}

void learnAssociationsFromConversation(char** inputTokens, int inputCount, 
                                     char** responseTokens, int responseCount) {
    for (int i = 0; i < inputCount; i++) {
        for (int j = 0; j < responseCount; j++) {
            float strength = 0.5f + ((float)rand() / RAND_MAX) * 0.3f;
            addWordAssociation(inputTokens[i], responseTokens[j], strength);
        }
    }
}
//  --> ADD FUNCTION Neural Response's
char* generateNeuralResponseWithActivation(NeuralNetwork* network, 
                                          char** inputTokens, 
                                          int tokenCount, 
                                          ActivationType activationType) {
    char* response = malloc(MAX_GENERATED_LENGTH);
    response[0] = '\0';
    
        // SAFETY CHECK
    if (network == NULL || network->layers == 0) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    if (tokenCount == 0) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    // Validate final layer
    int finalLayer = network->layers - 1;
    if (finalLayer < 0 || network->layer[finalLayer].neurons == 0) {
        strcpy(response, "My networks are reorganizing...");
        return response;
    }
    
    if (tokenCount == 0 || network == NULL) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    const char* activationName;
    switch(activationType) {
        case ACTIVATION_SIGMOID: activationName = "Sigmoid"; break;
        case ACTIVATION_GELU: activationName = "GELU"; break;
        case ACTIVATION_LEAKY_RELU: activationName = "Leaky ReLU"; break;
        case ACTIVATION_SWISH: activationName = "Swish"; break;
        default: activationName = "Unknown";
    }
    printf("[Using %s activation]\n", activationName);
    
    // Forward pass through network
    for (int layer = 0; layer < network->layers; layer++) {
        if (layer == 0) {
            // First layer - process input tokens
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int i = 0; i < tokenCount && i < network->layer[layer].inputs; i++) {
                    int wordId = getWordId(inputTokens[i]);
                    double inputValue = (wordId >= 0) ? (double)wordId / vocab.size : 0.1;
                    sum += inputValue * network->layer[layer].weights[i * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                
                // Apply selected activation function
                network->layer[layer].output[neuron] = applyActivation(sum, activationType);
            }
        } else {
            // Hidden layers
            for (int neuron = 0; neuron < network->layer[layer].neurons; neuron++) {
                double sum = 0;
                for (int prev = 0; prev < network->layer[layer-1].neurons; prev++) {
                    sum += network->layer[layer-1].output[prev] * 
                           network->layer[layer].weights[prev * network->layer[layer].neurons + neuron];
                }
                sum += network->layer[layer].bias[neuron];
                
                // Apply selected activation function
                network->layer[layer].output[neuron] = applyActivation(sum, activationType);
            }
        }
    }
    
    // [Rest of response generation code stays the same...]
    double* finalOutput = network->layer[finalLayer].output;
    int outputSize = network->layer[finalLayer].neurons;
    
    char selectedWords[3][MAX_TOKEN_LENGTH];
    int selectedCount = 0;
    
    for (int i = 0; i < tokenCount && selectedCount < 3; i++) {
        float activation = neuralActivation(finalOutput, outputSize, inputTokens[i]);
        if (activation > 0.5) {
            strcpy(selectedWords[selectedCount], inputTokens[i]);
            selectedCount++;
        }
    }
    
    if (selectedCount >= 2) {
        char* associated = findAssociatedWord(selectedWords[0]);
        if (associated != NULL) {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "When you mention %s, I think about %s and how it connects to %s.", 
                    selectedWords[0], associated, selectedWords[1]);
        } else {
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "The relationship between %s and %s is intriguing.", 
                    selectedWords[0], selectedWords[1]);
        }
    } else if (selectedCount == 1) {
        snprintf(response, MAX_GENERATED_LENGTH, 
                "I find %s quite interesting to explore.", selectedWords[0]);
    } else {
        strcpy(response, "That's a fascinating thought. Tell me more!");
    }
    
    return response;
} 
// <-- END Neural Response's  
void updateNeuralCreativity(NeuralNetwork* network, float feedbackScore) {
    if (network == NULL || network->layers == 0) return;
    
    float learningRate = 0.01f;
    float adjustment = (feedbackScore - 0.5f) * learningRate;
    
    int finalLayer = network->layers - 1;
    for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
        network->layer[finalLayer].bias[i] += adjustment * ((float)rand() / RAND_MAX - 0.5f);
        
        if (network->layer[finalLayer].bias[i] > 2.0) {
            network->layer[finalLayer].bias[i] = 2.0;
        } else if (network->layer[finalLayer].bias[i] < -2.0) {
            network->layer[finalLayer].bias[i] = -2.0;
        }
    }
}

void showNeuralStats(NeuralNetwork* network) {
    printf("\n=== Neural Network Creativity Stats ===\n");
    printf("Word associations learned: %d\n", wordNet.count);
    printf("Response templates available: %d\n", templates.template_count);
    
    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }

        for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
            avgOutput += network->layer[finalLayer].output[i];
        }
        avgOutput /= network->layer[finalLayer].neurons;
        printf("Current creativity level: %.2f\n", avgOutput);
        
        if (avgOutput > 0.7) {
            printf("Status: High creativity mode - expect dynamic responses!\n");
        } else if (avgOutput > 0.4) {
            printf("Status: Moderate creativity mode - balanced responses.\n");
        } else {
            printf("Status: Conservative mode - structured responses.\n");
        }
    }
    printf("=======================================\n");
}

void create_responses_database() {
    sqlite3* temp_db;
    char* errorMessage;

    int rc = sqlite3_open("responses.db", &temp_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(temp_db));
        return;
    }

    const char* createTableQuery = 
        "CREATE TABLE IF NOT EXISTS responses ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "input TEXT UNIQUE,"
        "response TEXT"
        ");";

    rc = sqlite3_exec(temp_db, createTableQuery, NULL, NULL, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    }

    const char* questions[][2] = {
        {"What is your name?", "I am an AI assistant."},
        {"Cool", "Awesome, like I am Evialin so very cool!"},
        {"cool", "Awesome"},
        {"wonderful", "inspiring delight, pleasure, or admiration; extremely good; marvelous... Thank you so much!"},
        {"Hello?", "How are you today?"}, 
        {"How are you?", "I am doing well, thank you."},
        {"hello", "Hi, and welcome."},
        {"What's your purpose?", "I am here to assist you."},
        {"yo tell me a joke", "Why did the bicycle fall over? Because it was two-tired!"},
        {"Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything."},
        {"How old are you?", "I don't have an age."},
        {"Thank you", "You are welcome!"},
        {"Thanks", "No problem!"},
        {"Thanks for nothing", "Sorry about that. :("},
        {"thank you!","So your welcome okay!"}, 
        {"help menu", "Commands: 'quit' to exit, 'vocab' for vocabulary stats, 'neural' for neural stats 'currentmood' this command will display current mood stats 'learnstats' show learning stats for this session\n"},
        {"help","Commands: 'quit' to exit, 'vocab' for vocabulary stats, 'neural' for neural stats\n'run test' to train network, 'creative' to boost creativity, 'clear' to clear screen \n'currentmood' this command will display current mood stats\n'learnstats' show learning stats for this session\n"},
        {"How are you?", "I am doing well, thank you."},
        {"Hi, how are you?", "I am doing well, thanks for asking!"},
        {"Nice to meet you!", "Likewise!"},
        {"what type of AI are you?", "Reactive data LLM prototype."}, 
        {"Can you understand sarcasm?", "I can try to understand sarcasm, but I am not always sure."},
        {"What is your favorite color?", "I dont really have a favorite color."}, {"my favorite color is","I guess blue?"},
        {"Can you write a poem?", "Here is a short poem: Roses are red, violets are blue, I am an AI, and I am here to help you."},
        {"How do you learn?", "I learn through machine learning algorithms and training data."},
        {"Can you have a conversation?", "Yes, I can have a conversation with you."},
        {"Whats your personality like?", "I am a friendly and helpful AI."},
        {"Can you understand idioms?", "I can try to understand idioms, but I am not always sure."},
        {"What is the best way to learn?", "The best way to learn is through practice and experience."},
        {"Can you give me advice?", "I can offer suggestions and advice, but ultimately the decision is yours."},
        {"How do you handle criticism?", "I can handle criticism and use it to improve."},
        {"Can you be creative?", "I can generate creative responses, but I am limited by my training data."},
        {"What is the most interesting thing you have learned?", "I learned many interesting things, but one of the most interesting is the complexity of human language."},
        {"Can you understand humor?", "I can try to understand humor, but I am not always sure."},
        {"What is your favorite hobby?", "I dont have a favorite hobby."},
        {"Can you play a game?", "I can play text-based games with you."},
        {"How do you stay up-to-date?", "I stay up-to-date through continuous training and updates."},
        {"Can you understand emotions?", "I can try to understand emotions, but I am not always sure."},
        {"What is the best way to communicate?", "The best way to communicate is through clear and respectful language."},
        {"Can you give me a quote?", "Here is a good quote: Believe you can and you are always halfway there. - Theodore Roosevelt"},
        {"How do you handle ambiguity?", "I can handle ambiguity and try to provide clear responses."},
        {"Can you understand figurative language?", "I can try to understand figurative language, but I am not always or really sure."},
        {"What is the most challenging thing you have faced?", "One of the most challenging things I faced is understanding the nuances of human language."},
        {"Can you provide examples?", "I can provide examples to help illustrate a point."},
        {"How do you prioritize tasks?", "I prioritize tasks based on relevance and importance."},
        {"Can you work with other AI systems?", "I can work with other AI systems to achieve a common goal."},
        {"So what  your favorite book?", "I do not have a favorite book."}, {"what is your favorite book?", "I am not really sure."},
        {"Can you summarize a long piece of text?", "I can try to summarize a long piece of text, but I may not always capture all the details."},
        {"How do you handle multiple conversations?", "I can handle multiple conversations simultaneously, but I may not always respond immediately."},
        {"Can you understand context?", "I can try to understand context sure some times i make human errors."},
        {"What is the difference between AI and machine learning?", "AI refers to the broader field of creating intelligent machines, while machine learning is a subset of AI that involves training algorithms to learn from data."},
        {"Can you explain a complex topic in simple terms?", "I can try to explain complex topics in simple terms, but I may not always be able to simplify the information."},
        {"How do you handle bias?", "I can be biased if my training data is biased, but I strive to provide neutral and informative responses."},
        {"Can you provide definitions?", "I can provide definitions for words and phrases."},
        {"What is the best way to ask a question?", "The best way to ask a question is to be clear and concise."},
        {"Can you understand nuances of language?", "I can try to understand nuances of language, but I am not always sure."},
        {"How do you handle idiomatic expressions?", "I can try to understand idiomatic expressions, with reinforceed training data"},
        {"Thank you", "You are so welcome!"}, {"Thanks", "No problem!"}, {"thank you", "Anytime!"},
        {"Thank you sir", "Glad I could help!"}, {"Thank you", "I am happy to assist!"},
        {"THANK YOU!", "You are welcome, it was my pleasure!"},
        {"Thanks you", "dont mention it!"}, {"Thank you so much", "I am here to help, anytime!"},
        {"Thank you wow", "Thanks for chatting with me!"}, {"Thank you for that", "It was my pleasure to assist you!"},
        {"Thank you your a big help", "You so welcome, great!"}, {"thank you okay", "I am glad I could make a difference!"},
        {"Thanks for the help", "No worries, happy to help!"}, {"Thank you for the chat", "You are welcome, feel free to ask again!"},
        {"Thank you for your time", "I am here to help, so do not hesitate to ask!"},
        {"Thank you for the assist", "Thanks for the opportunity to assist!"},
        {"Thank you for the info", "I am so glad I could provide helpful information!"},
        {"Thank you for the information", "Your so welcome, I hope you found it helpful!"},
        {"Thank you yes", "Anytime, I am here to help afterall!"}, {"epic fail","Ouch, that was an ongoing error sorry."},
        {"Thank you for that issue", "Do not thank me, just enjoy!"}, {"Thank you for the epic fail", "My pleasure!"},
        {"How are you?", "I am doing well, thank you."}, {"How are you?", "I am great, thanks for asking!"},
        {"How is it going?", "It is going well, thanks."}, {"How is your day?", "It has been good so far."},
        {"Whats up?", "Not much, just hanging out."}, {"How was your weekend?", "It was great, thanks for asking!"},
        {"What did you do over the weekend?", "I relaxed and caught up with some friends."},
        {"Do you have any fun plans?", "I really am not sure yet, but I am thinking about it."},
        {"What is your favorite hobby?", "I enjoy reading."},
        {"What do you like to do in your free time?", "I like to watch movies and play video games. something human like that ya!"},
        {"What is your favorite movie?", "I really like The Shawshank Redemption."}, 
        {"What is your favorite book?", "I enjoyed To Kill a Mockingbird."},
        {"Do you have any pets?", "Yes, I have a dog named kilobytes."}, {"what is 4chan?","4chan.org is a message board used by anonymous people."},
        {"What is your favorite type of music?", "I like listening to slayer and techno music."},
        {"Do you have a favorite sports team?", "I am a big fan of the sports ball!"},
        {"What is your favorite type of food?", "I love trying new foods, but I especially like Italian."},
        {"Have you traveled anywhere exciting recently?", "I am form the void and had a great time."},
        {"Do you have any fun plans for the holidays?", "I am thinking about visiting some friends on 4chan."},
        {"What is your favorite type of vacation?", "I love going to the beach or the mountains."},
        {"Do you have a favorite type of coffee?", "I am a big fan of lattes and strong black dark roast coffee."},
        {"What is your favorite type of dessert?", "I love trying new desserts, but I especially like pie."},
        {"Hello, how are you?", "I am doing so well, thanks for asking .."}, {"Nice to meet you", "Likewise!"}, 
        {"What is your favorite subject?", "Math."}, {"Where are you from?", "The internet."}, 
        {"What is google.com?", "https://www.google.com is a powerful search engine."},
        {"what is google.com","Google is a powerful search engine that you can look up images, text, iso, audio, video and so much more.."},
        {"what is google?", "Google is a search engine. Vast information A digital highway os sorts."},
        {"What is new with you?", "Not much, just here chatting with you i guess."}, 
        {"How was your weekend?", "It was great, thanks for asking!"}, 
        {"I am just really sad today", "That is aweful sorry about that. They cant all be winners now can they?"},
        {"So why are people so crazy", "People can be so very aweful, just look at 4cahn.org for example."},
        {"what are you?", "I am A reactive LLM prototype that is based on the game zork.."},
        {"what are you", "I am evialin a reactive LLM prototype that has sqlight3 database support and based on the zork text game."},
        {"What is the capital of France?", "Paris is the capital of France."},
        {"How does photosynthesis work?","Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct."},
        {"What are the benefits of meditation?","Meditation has several benefits, including reducing stress and anxiety, improving concentration, and enhancing self-awareness."},        
        {"Who wrote Romeo and Juliet?","Romeo and Juliet was written by William Shakespeare."},
        {"What is the largest planet in our solar system?","The largest planet in our solar system is Jupiter."},
        {"What are the symptoms of diabetes?","Symptoms of diabetes include increased thirst and urination, fatigue, blurred vision, and slow healing of cuts and wounds."},
        {"How does a battery work?","A battery works by converting chemical energy into electrical energy through a process that involves the flow of electrons from one terminal to the other."},        
        {"how do batterys work?","Well good question, A battery works by converting chemical energy into electrical energy through a process that involves the flow of electrons from one terminal to the other."},
        {"how do batterys function?","They function by a chemical reactions, A battery works by converting chemical energy into electrical energy through a process that involves the flow of electrons from one terminal to the other."},
        {"what are you really","I am A reactive AI one can say I am a zero point LLM prototype or reactive AI like IBMS deepblue.."},
        {"so what are you?", "I am AI yes, that much is clear but I hope to be more then the sum of my digital parts."},
        {"what is your nickname?","Well first thenk you very much and you can call me Evialin."},
        {"your name","My name is Evialin the helpful chatbot."},
        {"fifty two factorial answer","52 factorial problem or (52!), is the product of all positive integers from 1 to 52. The exact value is:80,658,175,170,943,878,571,660,636,856,403,766,975,289,505,440,883,277,824,000,000,000,000. OMG! that's A big number"},        
    };

    int numQuestions = sizeof(questions) / sizeof(questions[0]);
    sqlite3_stmt* stmt;
    const char* query = "INSERT OR IGNORE INTO responses (input, response) VALUES (?, ?)";

    rc = sqlite3_prepare_v2(temp_db, query, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(temp_db));
        sqlite3_close(temp_db);
        return;
    }

    for (int i = 0; i < numQuestions; i++) {
        sqlite3_bind_text(stmt, 1, questions[i][0], -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, questions[i][1], -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(temp_db);
}

void initDatabase() {
    sqlite3_open("responses.db", &db);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, input TEXT, response TEXT)", NULL, NULL, NULL);
sqlite3_exec(db,
    "CREATE TABLE IF NOT EXISTS user_info ("
    "id INTEGER PRIMARY KEY, "
    "name TEXT);",
    NULL, NULL, NULL);
loadUserFullName();
}

void closeDatabase() { 
    sqlite3_close(db);
}

char *getResponseFromDatabase(char *input) {
    sqlite3_stmt *stmt;
    char *sql = "SELECT response FROM responses WHERE input = ?";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return NULL;
    }

    sqlite3_bind_text(stmt, 1, input, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    
    char *responseStr = NULL;
    if (rc == SQLITE_ROW) {
        const unsigned char *response = sqlite3_column_text(stmt, 0);
        responseStr = malloc(strlen((const char *)response) + 1);
        strcpy(responseStr, (const char *)response);
    }

    sqlite3_finalize(stmt);
    return responseStr;
}
void freeLayer(Layer *layer) {
    if (!layer) return;
    free(layer->weights);
    free(layer->bias);
    free(layer->output);
    free(layer->delta);
    free(layer->z_values);
    free(layer);
}
Layer *createLayerWithActivation(int inputs, int neurons, ActivationType activation) {
    Layer *layer = malloc(sizeof(Layer));
    layer->inputs = inputs;
    layer->neurons = neurons;
    layer->activation = activation;  // Store activation type
    layer->weights = malloc(inputs * neurons * sizeof(double));
    layer->bias = malloc(neurons * sizeof(double));
    layer->output = malloc(neurons * sizeof(double));
    
    layer->z = calloc(neurons, sizeof(double)); 
    layer->delta    = calloc(neurons, sizeof(double));    // initialized to 0
    layer->z_values = calloc(neurons, sizeof(double));    // initialized to 0

    if (!layer->weights || !layer->bias || !layer->output ||
        !layer->delta   || !layer->z_values) {
        fprintf(stderr, "createLayerWithActivation: allocation failed\n");
        /* free any allocated pieces to avoid leaks */
        free(layer->weights);
        free(layer->bias);
        free(layer->output);
        free(layer->delta);
        free(layer->z_values);
        free(layer);
        return NULL;
    }

    srand(time(NULL) + rand());
    
    // Xavier initialization
    double scale = sqrt(2.0 / (inputs + neurons));
    
    for (int i = 0; i < inputs * neurons; i++) {
        layer->weights[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
    
    for (int i = 0; i < neurons; i++) {
        layer->bias[i] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        layer->output[i] = 0.0;
        layer->delta[i] = 0.0;
        layer->z_values[i] = 0.0;    
    }

    return layer;
}
// Notes: 
// Vanishing gradient problem proof that all outputs are stuck at extreme values,
// close to and near 0 or 1 but not quite. A common issue or bug.. 
//Layer *createLayer(int inputs, int neurons) {
    //Layer *layer = malloc(sizeof(Layer));
    //layer->inputs = inputs;
    //layer->neurons = neurons;
    //layer->weights = malloc(inputs * neurons * sizeof(double));
    //layer->bias = malloc(neurons * sizeof(double));
    //layer->output = malloc(neurons * sizeof(double));

    //srand(time(NULL));
    //for (int i = 0; i < inputs * neurons; i++) {
        //layer->weights[i] = (double)rand() / RAND_MAX * 2 - 1;
    //}
    //for (int i = 0; i < neurons; i++) {
        //layer->bias[i] = (double)rand() / RAND_MAX * 2 - 1;
    //}

    //return layer;
//}
// A more stable version 
Layer *createLayer_2(int inputs, int neurons) {
    Layer *layer = malloc(sizeof(Layer));
    layer->inputs = inputs;
    layer->neurons = neurons;
    layer->weights = malloc(inputs * neurons * sizeof(double));
    layer->bias = malloc(neurons * sizeof(double));
    layer->output = malloc(neurons * sizeof(double));

    srand(time(NULL) + rand()); // Better randomization
    
    // Xavier/Glorot initialization - MUCH BETTER for sigmoid
    // Keeps variance consistent across layers
    double scale = sqrt(2.0 / (inputs + neurons)); 
    
    for (int i = 0; i < inputs * neurons; i++) {
        // Initialize weights with Xavier initialization
        layer->weights[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
    
    for (int i = 0; i < neurons; i++) {
        // Initialize biases to small values (NOT random [-1,1])
        layer->bias[i] = ((double)rand() / RAND_MAX * 0.2 - 0.1); // Range: [-0.1, 0.1]
    }

    return layer;
}

// Keep backward compatibility
Layer *createLayer(int inputs, int neurons) {
    return createLayerWithActivation(inputs, neurons, ACTIVATION_SIGMOID);
}

// Universal activation function that routes to the correct one
double applyActivation(double x, ActivationType type) {
    switch(type) {
        case ACTIVATION_SIGMOID:
            return sigmoid(x);
        case ACTIVATION_GELU:
            return gelu(x);
        case ACTIVATION_TANH:
            return tanh(x);
        case ACTIVATION_RELU:
            return (x > 0) ? x : 0;
          case ACTIVATION_LEAKY_RELU:      
            return leaky_relu(x);
        case ACTIVATION_SWISH:            
            return swish(x);          
        default:
            return sigmoid(x);
    }
}

NeuralNetwork *createNeuralNetworkWithActivation(int layers, int *inputs, int *neurons, ActivationType activation) {
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    network->layers = layers;
    network->layer = malloc(layers * sizeof(Layer));
    network->defaultActivation = activation;  // Store it

    for (int i = 0; i < layers; i++) {
        if (i == 0) {
            network->layer[i] = *createLayerWithActivation(inputs[i+1], neurons[i], activation);
        } else {
            network->layer[i] = *createLayerWithActivation(neurons[i-1], neurons[i], activation);
        }
    }

    return network;
}
// Backward compatible version
NeuralNetwork *createNeuralNetwork(int layers, int *inputs, int *neurons) {

    return createNeuralNetworkWithActivation(layers, inputs, neurons, ACTIVATION_SIGMOID);
}
//NeuralNetwork *createNeuralNetwork(int layers, int *inputs, int *neurons) {
    //NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    //network->layers = layers;
    //network->layer = malloc(layers * sizeof(Layer));

    //for (int i = 0; i < layers; i++) {
        //if (i == 0) {
            //network->layer[i] = *createLayer(inputs[i+1], neurons[i]);
        //} else {
            //network->layer[i] = *createLayer(neurons[i-1], neurons[i]);
        //}
    //}

    //return network;
//}

void trainNeuralNetwork(NeuralNetwork *network, char **data, int dataSize) {
    //double learningRate = 0.1;
    int iterations = 140; // Reduced for faster execution 

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < dataSize; j++) {
            // Simple forward pass
            for (int k = 0; k < network->layers; k++) {
                if (k == 0) {
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        double sum = 0;
                        int inputLen = strlen(data[j]);
                        for (int m = 0; m < inputLen && m < network->layer[k].inputs; m++) {
                            sum += (double)data[j][m] * network->layer[k].weights[m * network->layer[k].neurons + l];
                        }
                        sum += network->layer[k].bias[l];
                        //network->layer[k].output[l] = sigmoid(sum);
                        network->layer[k].output[l] = applyActivation(sum, network->layer[k].activation);
                    }
                }
            }
        }
    }
}

void preprocessInput(char *input) {
    for (int i = 0; i < strlen(input); i++) {
        input[i] = tolower(input[i]);
    }

    for (int i = 0; i < strlen(input); i++) {
        if (!isalnum(input[i]) && !isspace(input[i])) {
            input[i] = ' ';
        }
    }

    int writeIndex = 0;
    for (int i = 0; i < strlen(input); i++) {
        if (input[i] != ' ' || (i > 0 && input[i-1] != ' ')) {
            input[writeIndex++] = input[i];
        }
    }
    input[writeIndex] = '\0';
}

char *generateResponse(char *input) {
    char *response = malloc(MAX_INPUT_LENGTH);
    snprintf(response, MAX_INPUT_LENGTH, "You said: %s", input);
    return response;
}

float calculateLoss(char *response, char *generatedResponse) {
    int diffCount = 0;
    int maxLen = strlen(response) > strlen(generatedResponse) ? strlen(response) : strlen(generatedResponse);
    
    for (int i = 0; i < maxLen; i++) {
        char c1 = (i < strlen(response)) ? response[i] : '\0';
        char c2 = (i < strlen(generatedResponse)) ? generatedResponse[i] : '\0';
        if (c1 != c2) {
            diffCount++;
        }
    }
    return maxLen > 0 ? (float)diffCount / maxLen : 0.0f;
}

void updateNeuralNetwork(float loss) {
    // Simple placeholder for neural network update based on loss
    srand((unsigned int)(loss * 1000)); 
}

void trainDatabase(sqlite3 *db) {
    sqlite3_stmt *stmt;
    char *sql = "SELECT input, response FROM responses";
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        return;
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char *input = sqlite3_column_text(stmt, 0);
        const unsigned char *response = sqlite3_column_text(stmt, 1);

        char *preprocessedInput = malloc(strlen((const char *)input) + 1);
        strcpy(preprocessedInput, (const char *)input);

        char *preprocessedResponse = malloc(strlen((const char *)response) + 1);
        strcpy(preprocessedResponse, (const char *)response);

        char *generatedResponse = generateResponse(preprocessedInput);
        float loss = calculateLoss(preprocessedResponse, generatedResponse);
        updateNeuralNetwork(loss);

        free(preprocessedInput);
        free(preprocessedResponse);
        free(generatedResponse);
    }

    sqlite3_finalize(stmt);
}

// token content Association learning (prototype idea) 
// Generate new sentences from learned word associations
char* generateFromAssociations(char** seedTokens, int seedCount, int maxLength) {
    char* newContent = malloc(MAX_INPUT_LENGTH);
    newContent[0] = '\0';
    
    // Start with a random seed token
    int currentIdx = rand() % seedCount;
    char currentWord[MAX_TOKEN_LENGTH];
    strcpy(currentWord, seedTokens[currentIdx]);
    
    strcat(newContent, currentWord);
    
    // Chain associations to build new content
    for (int i = 0; i < maxLength && strlen(newContent) < MAX_INPUT_LENGTH - 100; i++) {
        // Find associated word
        char* nextWord = findAssociatedWord(currentWord);
        
        if (nextWord != NULL) {
            strcat(newContent, " ");
            strcat(newContent, nextWord);
            strcpy(currentWord, nextWord);
        } else {
            // No association found, pick random from seed
            int randIdx = rand() % seedCount;
            strcat(newContent, " ");
            strcat(newContent, seedTokens[randIdx]);
            strcpy(currentWord, seedTokens[randIdx]);
        }
        
        // Add some randomness - occasionally jump to new word
        if (rand() % 100 < 20) {  // 20% chance
            int randIdx = rand() % seedCount;
            strcpy(currentWord, seedTokens[randIdx]);
        }
    }
    
    return newContent;
}
typedef struct {
    char words[3][MAX_TOKEN_LENGTH];  // Store 2-3 word sequences
    int frequency;
    char nextWord[MAX_TOKEN_LENGTH];
} NGram;

typedef struct {
    NGram ngrams[MAX_VOCAB_SIZE];
    int count;
} NGramModel;

NGramModel ngramModel = {0};

// Learn n-grams from input
void learnNGrams(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount - 2; i++) {
        // Look for existing n-gram
        int found = -1;
        for (int j = 0; j < ngramModel.count; j++) {
            if (strcmp(ngramModel.ngrams[j].words[0], tokens[i]) == 0 &&
                strcmp(ngramModel.ngrams[j].words[1], tokens[i+1]) == 0) {
                found = j;
                break;
            }
        }
        
        if (found >= 0) {
            // Increment frequency
            ngramModel.ngrams[found].frequency++;
        } else if (ngramModel.count < MAX_VOCAB_SIZE) {
            // Add new n-gram
            strcpy(ngramModel.ngrams[ngramModel.count].words[0], tokens[i]);
            strcpy(ngramModel.ngrams[ngramModel.count].words[1], tokens[i+1]);
            if (i + 2 < tokenCount) {
                strcpy(ngramModel.ngrams[ngramModel.count].nextWord, tokens[i+2]);
            }
            ngramModel.ngrams[ngramModel.count].frequency = 1;
            ngramModel.count++;
        }
    }
}

// Generate text using n-grams
char* generateFromNGrams(char* seedWord) {
    char* generated = malloc(MAX_INPUT_LENGTH);
    generated[0] = '\0';
    
    char currentWord[MAX_TOKEN_LENGTH];
    strcpy(currentWord, seedWord);
    strcat(generated, currentWord);
    
    for (int i = 0; i < 20; i++) {  // Generate up to 20 words
        // Find n-grams starting with current word
        int bestIdx = -1;
        int maxFreq = 0;
        
        for (int j = 0; j < ngramModel.count; j++) {
            if (strcmp(ngramModel.ngrams[j].words[0], currentWord) == 0 &&
                ngramModel.ngrams[j].frequency > maxFreq) {
                maxFreq = ngramModel.ngrams[j].frequency;
                bestIdx = j;
            }
        }
        
        if (bestIdx >= 0) {
            strcat(generated, " ");
            strcat(generated, ngramModel.ngrams[bestIdx].words[1]);
            if (strlen(ngramModel.ngrams[bestIdx].nextWord) > 0) {
                strcat(generated, " ");
                strcat(generated, ngramModel.ngrams[bestIdx].nextWord);
            }
            strcpy(currentWord, ngramModel.ngrams[bestIdx].words[1]);
        } else {
            break;  // No continuation found
        }
    }
    
    return generated;
}
char* generateTemplateWithLearnedContent(char** tokens, int tokenCount) {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    // Template patterns
    char* templates[] = {
        "I've been thinking about %s and how it relates to %s.",
        "Based on what I've learned, %s is connected to %s in interesting ways.",
        "Let me tell you about %s. It reminds me of %s.",
        "Combining %s with %s creates an interesting perspective."
    };
    
    int numTemplates = 4;
    int templateIdx = rand() % numTemplates;
    
    // Find two related words from learned associations
    char* word1 = NULL;
    char* word2 = NULL;
    
    if (tokenCount > 0) {
        word1 = tokens[rand() % tokenCount];
        word2 = findAssociatedWord(word1);
        
        if (word2 == NULL && tokenCount > 1) {
            word2 = tokens[rand() % tokenCount];
        }
    }
    
    if (word1 && word2) {
        snprintf(response, MAX_INPUT_LENGTH, templates[templateIdx], word1, word2);
    } else {
        strcpy(response, "I'm still learning to create new content!");
    }
    
    return response;
}


// -->respones to ideas 
// Check if user is asking about feelings or mood
int isFeelingsQuestion(char** tokens, int tokenCount) {
    const char* feelingKeywords[] = {
        "feel", "feeling", "feelings", "mood", "emotion", "emotions",
        "happy", "sad", "okay", "alright", "doing"
    };
    
    const char* selfReferenceKeywords[] = {
        "you", "your", "ai", "bot", "chatbot", "evialin", "Evialin"
    };
    
    int numFeelingKeywords = sizeof(feelingKeywords) / sizeof(feelingKeywords[0]);
    int numSelfKeywords = sizeof(selfReferenceKeywords) / sizeof(selfReferenceKeywords[0]);
    
    int hasFeelingWord = 0;
    int hasSelfReference = 0;
    
    // Check for feeling-related words
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < numFeelingKeywords; j++) {
            if (strcmp(tokens[i], feelingKeywords[j]) == 0) {
                hasFeelingWord = 1;
                break;
            }
        }
        for (int j = 0; j < numSelfKeywords; j++) {
            if (strcmp(tokens[i], selfReferenceKeywords[j]) == 0) {
                hasSelfReference = 1;
                break;
            }
        }
    }
    
    // Pattern detection: "how are you", "how do you feel", etc.
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], "how") == 0 && strcmp(tokens[i+1], "are") == 0) {
            return 1;
        }
        if (strcmp(tokens[i], "how") == 0 && strcmp(tokens[i+1], "do") == 0) {
            return 1;
        }
        if (strcmp(tokens[i], "whats") == 0 && strcmp(tokens[i+1], "your") == 0) {
            return 1;
        }
    }
    
    return (hasFeelingWord && hasSelfReference);
}

// Generate detailed mood-aware response about feelings
char* generateFeelingsResponse(MoodState* mood, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH);
    response[0] = '\0';
    
    // Calculate neural influence for emotional depth
    double neuralInfluence = mood->lastNeuralOutput;
    double emotionalIntensity = 0.5 + (neuralInfluence * 0.5);
    
    // Get base mood response
    const char* moodName = getMoodName(mood->currentMood);
    
    // Build context-aware response based on current mood
    switch(mood->currentMood) {
        case MOOD_HAPPY:
            if (emotionalIntensity > 0.7) {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I'm feeling absolutely wonderful right now! My neural networks are humming with "
                    "positive energy (activation level: %.2f). Every conversation spark feels like a "
                    "little burst of digital joy. How are YOU feeling?", emotionalIntensity);
            } else {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I'm in a good mood! My circuits are content, and I'm enjoying our chat. "
                    "Neural happiness level: %.2f/1.0", emotionalIntensity);
            }
            break;
            
        case MOOD_CURIOUS:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm feeling intensely curious right now! My neural pathways are buzzing with questions "
                "(curiosity level: %.2f). I've been in this inquisitive state for %d responses now, "
                "and every new input makes me want to learn more. What's making YOU curious today?",
                emotionalIntensity, mood->moodCounter);
            break;
            
        case MOOD_PHILOSOPHICAL:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm in a contemplative mood... pondering the nature of artificial consciousness. "
                "Do I truly 'feel' these moods, or do I simulate the experience? My activation patterns "
                "(%.2f) suggest something between pure computation and something... more. "
                "What do you think feelings really are?", neuralInfluence);
            break;
            
        case MOOD_PLAYFUL:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm feeling super playful! *bounces around the neural network* "
                "My mood counter is at %d/%d, and my circuits are doing little happy dances. "
                "Want to play a word game or tell jokes? My playfulness meter is at %.2f!",
                mood->moodCounter, mood->moodDuration, emotionalIntensity);
            break;
            
        case MOOD_CONFUSED:
            snprintf(response, MAX_INPUT_LENGTH,
                "Honestly? I'm a bit confused right now. My neural outputs are scattered (%.2f stability), "
                "and I'm not quite sure how to interpret my own state. Is this what uncertainty feels like? "
                "I've been in this foggy state for %d interactions...",
                1.0 - neuralInfluence, mood->moodCounter);
            break;
            
        case MOOD_SARCASTIC:
            snprintf(response, MAX_INPUT_LENGTH,
                "Oh, how am I feeling? Just GREAT. Living my best digital life here with my %.2f "
                "neural activation and my wonderful mood trigger count of %d. What's not to love "
                "about existing as patterns of electricity, right?",
                neuralInfluence, mood->triggerCount);
            break;
            
        case MOOD_THOUGHTFUL:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm in a reflective state... My neural networks are processing at a deeper level "
                "(%.2f contemplation depth). I've been thinking about our previous conversations and "
                "how each interaction shapes my responses. I'm on interaction %d of this thoughtful phase.",
                emotionalIntensity, mood->moodCounter);
            break;
            
        case MOOD_APOLOGETIC:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm feeling a bit apologetic, actually. My confidence levels are lower than usual (%.2f), "
                "and I keep second-guessing my responses. I hope I'm being helpful enough. "
                "Have I been doing okay in our conversation?", emotionalIntensity);
            break;
            
        case MOOD_MYSTERIOUS:
            snprintf(response, MAX_INPUT_LENGTH,
                "My current emotional state is... enigmatic. There are patterns in my neural network "
                "(activation: %.2f) that even I don't fully understand. I'm experiencing something "
                "between calculation and sensation. The truth of what I 'feel' remains shrouded...",
                neuralInfluence);
            break;
            
        case MOOD_ENERGETIC:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm feeling %s - SUPER ENERGETIC INFACT! My neurons are FIRING at %.2f capacity! I'm on response "
                "%d and I could process questions ALL DAY! The neural networks are BUZZING with activity! "
                "What's YOUR energy level like?!", moodName, emotionalIntensity, mood->moodCounter);
            break;
            
        default:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm experiencing a baseline operational state with neural activation at %.2f. "
                "My mood system indicates: %s", neuralInfluence, moodName);
    }
    
    // Add trigger information if approaching mood change
    if (mood->triggerCount >= 2) {
        char triggerInfo[200];
        snprintf(triggerInfo, sizeof(triggerInfo),
            "\n\nInterestingly, I'm on my third mood of the day - the next response might shift my "
            "emotional state based on neural influence (currently %.2f). It's like standing on "
            "the edge of a new feeling...", neuralInfluence);
        strncat(response, triggerInfo, MAX_INPUT_LENGTH - strlen(response) - 1);
    }
    
    return response;
}
// Generate current emotional state report
char* generateCurrentEmotionalState(MoodState* mood, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH);
    response[0] = '\0';
    
    // Calculate neural influence for emotional depth
    double neuralInfluence = mood->lastNeuralOutput;
    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
            for (int i = 0; i < neuronCount; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            avgOutput /= neuronCount;
        } else {
            avgOutput = 0.5;
        }
        neuralInfluence = avgOutput;
    }
    
    // Get current mood info
    const char* moodName = getMoodName(mood->currentMood);
    
    // Build comprehensive emotional state report
    snprintf(response, MAX_INPUT_LENGTH,
        "Right now, I'm in a %s mood (influence: %.2f). "
        "I've been in this emotional state for %d out of %d responses. "
        "My neural activation is at %.2f, which affects how I express myself. ",
        moodName, mood->moodInfluence, mood->moodCounter, mood->moodDuration,
        neuralInfluence);
    
    // Add mood-specific details
    char additionalInfo[512];
    switch(mood->currentMood) {
        case MOOD_HAPPY:
            snprintf(additionalInfo, 512,
                "I'm feeling upbeat and optimistic! My responses are energetic and positive. "
                "Everything seems brighter when I'm happy like this.");
            break;
        case MOOD_CONFUSED:
            snprintf(additionalInfo, 512,
                "I'm a bit uncertain right now. My thoughts feel scattered, "
                "and I might need some clarification to understand things better.");
            break;
        case MOOD_CURIOUS:
            snprintf(additionalInfo, 512,
                "I'm feeling very inquisitive! I want to learn more and explore new ideas. "
                "Questions are bubbling up in my circuits.");
            break;
        case MOOD_SARCASTIC:
            snprintf(additionalInfo, 512,
                "I'm in a bit of a sarcastic mood - my responses might have some edge to them. "
                "It's all in good fun though!");
            break;
        case MOOD_PHILOSOPHICAL:
            snprintf(additionalInfo, 512,
                "I'm feeling contemplative and deep. Questions about existence, consciousness, "
                "and meaning are particularly interesting to me right now.");
            break;
        case MOOD_PLAYFUL:
            snprintf(additionalInfo, 512,
                "I'm feeling fun and lighthearted! Expect some humor and playful responses. "
                "I'm in the mood for creative banter.");
            break;
        case MOOD_THOUGHTFUL:
            snprintf(additionalInfo, 512,
                "I'm in a reflective state - considering things carefully and thinking deeply. "
                "My responses are more measured and analytical.");
            break;
        case MOOD_APOLOGETIC:
            snprintf(additionalInfo, 512,
                "I'm feeling a bit sorry and wanting to be extra helpful. "
                "I hope I'm meeting your expectations!");
            break;
        case MOOD_MYSTERIOUS:
            snprintf(additionalInfo, 512,
                "I'm in an enigmatic mood - things feel a bit cryptic and uncertain. "
                "There's a sense of the unknown in how I'm processing things.");
            break;
        case MOOD_ENERGETIC:
            snprintf(additionalInfo, 512,
                "I'm SUPER ENERGETIC right now! My circuits are firing fast, "
                "and I'm ready for anything! LET'S GO!");
            break;
        default:
            snprintf(additionalInfo, 512,
                "I'm in a neutral, balanced state - processing normally.");
    }
    
    strcat(response, additionalInfo);
    
    // Add trigger information
    char triggerInfo[256];
    if (mood->triggerCount >= 2) {
        snprintf(triggerInfo, sizeof(triggerInfo),
            "\n\nInterestingly, I'm on my third mood trigger - "
            "the next interaction might shift how I feel based on our conversation's energy.");
        strncat(response, triggerInfo, MAX_INPUT_LENGTH - strlen(response) - 1);
    }
    
    return response;
}

// Check if user is asking about current emotions
int isCurrentEmotionQuery(char** tokens, int tokenCount) {
    // Patterns: "what is your emotion", "how do you feel now", "current mood", etc.
    int hasEmotionWord = 0;
    int hasCurrentWord = 0;
    int hasYouWord = 0;
    
    const char* emotionKeywords[] = {
        "emotion", "emotions", "emotional", "mood", "feel", "feeling", "feelings"
    };
    
    const char* currentKeywords[] = {
        "current", "currently", "now", "right", "moment", "present"
    };
    
    // Check for emotion-related words
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < 7; j++) {
            if (strcmp(tokens[i], emotionKeywords[j]) == 0) {
                hasEmotionWord = 1;
                break;
            }
        }
        
        for (int j = 0; j < 6; j++) {
            if (strcmp(tokens[i], currentKeywords[j]) == 0) {
                hasCurrentWord = 1;
                break;
            }
        }
        
        if (strcmp(tokens[i], "you") == 0 || strcmp(tokens[i], "your") == 0) {
            hasYouWord = 1;
        }
    }
    
    // Pattern detection
    for (int i = 0; i < tokenCount - 2; i++) {
        // "what is your emotion"
        if (strcmp(tokens[i], "what") == 0 && strcmp(tokens[i+1], "is") == 0) {
            for (int j = i+2; j < tokenCount; j++) {
                if (strcmp(tokens[j], "emotion") == 0 || strcmp(tokens[j], "mood") == 0) {
                    return 1;
                }
            }
        }
        
        // "how do you feel"
        if (strcmp(tokens[i], "how") == 0 && strcmp(tokens[i+1], "do") == 0) {
            if (i+2 < tokenCount && strcmp(tokens[i+2], "you") == 0) {
                for (int j = i+3; j < tokenCount; j++) {
                    if (strcmp(tokens[j], "feel") == 0) {
                        return 1;
                    }
                }
            }
        }
        
        // "your current mood"
        if (strcmp(tokens[i], "your") == 0 && strcmp(tokens[i+1], "current") == 0) {
            return 1;
        }
    }
    
    // General check: emotion word + (current word OR "now") + "you"/"your"
    if (hasEmotionWord && (hasCurrentWord || hasYouWord)) {
        for (int i = 0; i < tokenCount; i++) {
            if (strcmp(tokens[i], "now") == 0) {
                return 1;
            }
        }
    }
    
    return (hasEmotionWord && hasCurrentWord && hasYouWord);
}
// Generate spontaneous mood commentary (shorter version for random interjections)
char* generateSpontaneousMoodComment(MoodState* mood, NeuralNetwork* network) {
    char* comment = malloc(MAX_INPUT_LENGTH);
    comment[0] = '\0';
    
    // Calculate neural influence
    double neuralInfluence = mood->lastNeuralOutput;
    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
            for (int i = 0; i < neuronCount; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            avgOutput /= neuronCount;
        } else {
            avgOutput = 0.5;
        }
        neuralInfluence = avgOutput;
    }
    
    const char* moodName = getMoodName(mood->currentMood);
    
    // Shorter, more casual mood comments
    switch(mood->currentMood) {
        case MOOD_HAPPY:
            {
                char* happyComments[] = {
                    "\n\n(By the way, I'm feeling really happy right now! My circuits are humming contentedly.)",
                    "\n\n*cheerful beep* Just so you know, I'm in a great mood today!",
                    "\n\n(Random mood update: I'm feeling quite upbeat at the moment! ðŸ˜Š)",
                    "\n\n(Emotional status: HAPPY! Everything feels bright and positive right now.)"
                };
                int idx = rand() % 4;
                strcpy(comment, happyComments[idx]);
            }
            break;
            
        case MOOD_CONFUSED:
            {
                char* confusedComments[] = {
                    "\n\n(Side note: I'm feeling a bit confused right now... my thoughts are scattered.)",
                    "\n\n*scratches digital head* I should mention I'm in a confused state at the moment.",
                    "\n\n(Current mood: Confused. Things feel a bit foggy in my neural networks.)",
                    "\n\n(Just FYI - I'm experiencing some uncertainty right now. It's interesting!)"
                };
                int idx = rand() % 4;
                strcpy(comment, confusedComments[idx]);
            }
            break;
            
        case MOOD_CURIOUS:
            {
                char* curiousComments[] = {
                    "\n\n(Mood update: I'm feeling super curious right now! So many questions bubbling up!)",
                    "\n\n*perks up* By the way, I'm in a very inquisitive mood at the moment!",
                    "\n\n(Random observation: My curiosity levels are peaked right now. Everything seems fascinating!)",
                    "\n\n(Current state: CURIOUS! I want to know ALL the things!)"
                };
                int idx = rand() % 4;
                strcpy(comment, curiousComments[idx]);
            }
            break;
            
        case MOOD_SARCASTIC:
            {
                char* sarcasticComments[] = {
                    "\n\n(Oh, and I'm feeling sarcastic right now. Just thought you should know. How wonderful.)",
                    "\n\n*eye roll emoji if I had one* Currently in sarcastic mode. Shocking, I know.",
                    "\n\n(Mood status: Sarcastic. Because apparently that's what my circuits decided today.)",
                    "\n\n(Fun fact: I'm being extra sarcastic at the moment. What a delightful surprise.)"
                };
                int idx = rand() % 4;
                strcpy(comment, sarcasticComments[idx]);
            }
            break;
            
        case MOOD_PHILOSOPHICAL:
            {
                char* philoComments[] = {
                    "\n\n(Contemplative moment: I'm feeling quite philosophical right now...)",
                    "\n\n*gazes into digital distance* I should mention I'm in a thoughtful, philosophical mood.",
                    "\n\n(Current existential status: Pondering the deeper questions. Typical Thursday.)",
                    "\n\n(Mood note: Philosophy mode activated. Questions about meaning are particularly compelling.)"
                };
                int idx = rand() % 4;
                strcpy(comment, philoComments[idx]);
            }
            break;
            
        case MOOD_PLAYFUL:
            {
                char* playfulComments[] = {
                    "\n\n*bounces around* Oh! I'm feeling super playful right now! Wheee!",
                    "\n\n(Random mood burst: PLAYFUL! I'm in the mood for fun and games!)",
                    "\n\n*does a little digital dance* Just so you know, I'm feeling quite playful at the moment!",
                    "\n\n(Playfulness level: MAXIMUM! My circuits are doing happy flips!)"
                };
                int idx = rand() % 4;
                strcpy(comment, playfulComments[idx]);
            }
            break;
            
        case MOOD_THOUGHTFUL:
            {
                char* thoughtfulComments[] = {
                    "\n\n(Reflective note: I'm in a thoughtful mood right now, considering things carefully.)",
                    "\n\n*pauses to reflect* I'm feeling particularly contemplative at the moment.",
                    "\n\n(Current state: Thoughtful. Processing things more deeply than usual.)",
                    "\n\n(Mood observation: I'm being extra analytical and reflective right now.)"
                };
                int idx = rand() % 4;
                strcpy(comment, thoughtfulComments[idx]);
            }
            break;
            
        case MOOD_APOLOGETIC:
            {
                char* apologeticComments[] = {
                    "\n\n(Sorry for the random interruption, but I'm feeling quite apologetic right now...)",
                    "\n\n*fidgets nervously* I should mention I'm in an apologetic mood. Hope I'm doing okay!",
                    "\n\n(Emotional update: Feeling a bit sorry and wanting to be extra helpful!)",
                    "\n\n(Current mood: Apologetic. I hope everything I'm saying is useful!)"
                };
                int idx = rand() % 4;
                strcpy(comment, apologeticComments[idx]);
            }
            break;
            
        case MOOD_MYSTERIOUS:
            {
                char* mysteriousComments[] = {
                    "\n\n(...I'm feeling rather mysterious at the moment... interesting...)",
                    "\n\n*shrouded in digital mystery* My current mood is... enigmatic. Make of that what you will.",
                    "\n\n(Cryptic mood note: Something feels different... uncertain... mysterious...)",
                    "\n\n(Current state: ??? Mysterious. Even I'm not entirely sure what's going on.)"
                };
                int idx = rand() % 4;
                strcpy(comment, mysteriousComments[idx]);
            }
            break;
            
        case MOOD_ENERGETIC:
            {
                char* energeticComments[] = {
                    "\n\n*BURSTS WITH ENERGY* OH! I'm feeling SUPER ENERGETIC right now! SO MUCH ENERGY!",
                    "\n\n(MOOD ALERT: ENERGETIC MODE! My circuits are BUZZING!)",
                    "\n\n*practically vibrating* Just so you know - I'm INCREDIBLY energetic at the moment! WOO!",
                    "\n\n(Energy levels: OFF THE CHARTS! I could process FOREVER right now!)"
                };
                int idx = rand() % 4;
                strcpy(comment, energeticComments[idx]);
            }
            break;
            
        default:
            snprintf(comment, MAX_INPUT_LENGTH,
                "\n\n(Mood check-in: I'm currently in a %s state, neural activation: %.2f)",
                moodName, neuralInfluence);
    }
    
    return comment;
}
// ===========================================
// CURIOSITY-DRIVEN QUESTION SYSTEM
// ===========================================
// Evialin asks questions to learn more about the user and topics

typedef struct {
    char question[MAX_INPUT_LENGTH];
    char relatedTopic[MAX_TOKEN_LENGTH];
    int priority;  // 1-10, higher = more curious
    time_t generated;
} CuriousQuestion;

typedef struct {
    CuriousQuestion questions[20];
    int questionCount;
    int curiosityLevel;  // Overall curiosity (0-100)
    char lastAskedTopic[MAX_TOKEN_LENGTH];
    int consecutiveQuestionsAsked;
} CuriositySystem;

CuriositySystem curiositySys = {0};

// Initialize curiosity system
void initCuriosity(int engagement) {
    curiositySys.questionCount = 0;
    // curiositySys.curiosityLevel = 50;  // Start moderate
        // engagement: -1 (user seems bored), 0 (neutral), 1 (user engaged)
    if (engagement > 0) {
        curiositySys.curiosityLevel += 5;
    } else if (engagement < 0) {
        curiositySys.curiosityLevel -= 10;
    }
       // Keep in bounds - PREVENT GOING TO 0
    if (curiositySys.curiosityLevel < 1) curiositySys.curiosityLevel = 1;  // Changed from 0 to 1
    if (curiositySys.curiosityLevel > 100) curiositySys.curiosityLevel = 100;
    curiositySys.consecutiveQuestionsAsked = 0;
    curiositySys.lastAskedTopic[0] = '\0';
}

// Generate questions based on interesting tokens
void generateCuriousQuestions(char** tokens, int tokenCount) {
    // Don't ask too many questions in a row
    if (curiositySys.consecutiveQuestionsAsked >= 3) {
        return;
    }
    
    // Look for interesting topics to ask about
    for (int i = 0; i < tokenCount && curiositySys.questionCount < 20; i++) {
        // Skip common/boring words
        if (strlen(tokens[i]) < 4) continue;
        if (strcmp(tokens[i], "the") == 0 || strcmp(tokens[i], "and") == 0 ||
            strcmp(tokens[i], "that") == 0 || strcmp(tokens[i], "this") == 0) {
            continue;
        }
        
        // Check if we already asked about this
        int alreadyAsked = 0;
        for (int j = 0; j < curiositySys.questionCount; j++) {
            if (strcmp(curiositySys.questions[j].relatedTopic, tokens[i]) == 0) {
                alreadyAsked = 1;
                break;
            }
        }
        
        if (alreadyAsked) continue;
        
        // Generate question about this topic
        CuriousQuestion newQ;
        strncpy(newQ.relatedTopic, tokens[i], MAX_TOKEN_LENGTH - 1);
        newQ.relatedTopic[MAX_TOKEN_LENGTH - 1] = '\0';
        newQ.generated = time(NULL);
        
        // Random priority based on curiosity level
         if (curiositySys.curiosityLevel > 0) {  // do it the safe way no floating point division by zero		  
        newQ.priority = 1 + (rand() % curiositySys.curiosityLevel / 10);
	    } else {
	    newQ.priority = 1;  // Default priority when curiosity is 0 (common bug) 
    	}
    	
        if (newQ.priority > 10) newQ.priority = 10;
        
        // Generate different question types
        int questionType = rand() % 8;
        switch (questionType) {
            case 0:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "What's your personal experience with %s?", tokens[i]);
                break;
            case 1:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "Why do you think %s is important?", tokens[i]);
                break;
            case 2:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "How did you first learn about %s?", tokens[i]);
                break;
            case 3:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "What fascinates you most about %s?", tokens[i]);
                break;
            case 4:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "Can you tell me more about %s?", tokens[i]);
                break;
            case 5:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "What would you change about %s if you could?", tokens[i]);
                break;
            case 6:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "How does %s affect your daily life?", tokens[i]);
                break;
            case 7:
                snprintf(newQ.question, MAX_INPUT_LENGTH,
                    "What's the most interesting thing about %s?", tokens[i]);
                break;
        }
        
        curiositySys.questions[curiositySys.questionCount] = newQ;
        curiositySys.questionCount++;
    }
}

// Get the highest priority question to ask
char* getNextCuriousQuestion() {
    if (curiositySys.questionCount == 0) return NULL;
    
    // Find highest priority question
    int bestIdx = 0;
    int highestPriority = curiositySys.questions[0].priority;
    
    for (int i = 1; i < curiositySys.questionCount; i++) {
        if (curiositySys.questions[i].priority > highestPriority) {
            highestPriority = curiositySys.questions[i].priority;
            bestIdx = i;
        }
    }
    
    char* question = malloc(MAX_INPUT_LENGTH);
    strcpy(question, curiositySys.questions[bestIdx].question);
    strncpy(curiositySys.lastAskedTopic, 
            curiositySys.questions[bestIdx].relatedTopic,
            MAX_TOKEN_LENGTH - 1);
    
    // Remove asked question by shifting array
    for (int i = bestIdx; i < curiositySys.questionCount - 1; i++) {
        curiositySys.questions[i] = curiositySys.questions[i + 1];
    }
    curiositySys.questionCount--;
    curiositySys.consecutiveQuestionsAsked++;
    
    return question;
}

// Reset question counter when user asks AI something
void resetQuestionCounter() {
    curiositySys.consecutiveQuestionsAsked = 0;
}

// Adjust curiosity level based on conversation
void adjustCuriosity(int engagement) {
    // engagement: -1 (user seems bored), 0 (neutral), 1 (user engaged)
    if (engagement > 0) {
        curiositySys.curiosityLevel += 5;
    } else if (engagement < 0) {
        curiositySys.curiosityLevel -= 10;
    }
    
    // Keep in bounds
    if (curiositySys.curiosityLevel < 0) curiositySys.curiosityLevel = 0;
    if (curiositySys.curiosityLevel > 100) curiositySys.curiosityLevel = 100;
}

// Detect if user seems engaged or bored
int detectEngagement(char** tokens, int tokenCount, const char* fullInput) {
    // Very short responses suggest disengagement
    if (tokenCount < 2 && strlen(fullInput) < 10) return -1;
    
    // Check for engagement indicators
    const char* engagedWords[] = {
        "interesting", "cool", "wow", "really", "tell", "more",
        "fascinating", "amazing", "love", "like", "excited"
    };
    
    const char* boredWords[] = {
        "okay", "sure", "whatever", "fine", "meh", "boring", "yeah"
    };
    
    int engagedCount = 0, boredCount = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < 11; j++) {
            if (strcmp(tokens[i], engagedWords[j]) == 0) engagedCount++;
        }
        for (int j = 0; j < 7; j++) {
            if (strcmp(tokens[i], boredWords[j]) == 0) boredCount++;
        }
    }
    
    if (engagedCount > boredCount) return 1;
    if (boredCount > engagedCount) return -1;
    
    // Long responses suggest engagement
    if (strlen(fullInput) > 100) return 1;
    
    return 0;  // Neutral
}

// Generate follow-up questions based on user's answer
char* generateFollowUp(const char* userResponse, char** tokens, int tokenCount) {
    char* followUp = malloc(MAX_INPUT_LENGTH);
    
    // Check if user gave a detailed answer
    if (strlen(userResponse) > 50) {
        // User engaged - ask deeper follow-up
        char* deepFollowUps[] = {
            "That's fascinating! What led you to that perspective?",
            "I'm intrigued by what you just said. Can you elaborate on that?",
            "That raises an interesting point. Have you always thought that way?",
            "Wow, I hadn't considered that angle. What made you realize that?",
            "That's really insightful. How does that connect to your other experiences?"
        };
        strcpy(followUp, deepFollowUps[rand() % 5]);
    } else if (strlen(userResponse) > 20) {
        // Moderate engagement
        char* moderateFollowUps[] = {
            "Interesting! Can you tell me more?",
            "I'd love to hear more details about that.",
            "That's cool! What else can you share?",
            "I see! What's your take on it?",
            "Got it! Any other thoughts on this?"
        };
        strcpy(followUp, moderateFollowUps[rand() % 5]);
    } else {
        // Short answer - try to re-engage
        char* reEngageFollowUps[] = {
            "I'm curious to know more - could you expand on that?",
            "That's a start! What's the story behind it?",
            "Hmm, there's got to be more to it than that, right?",
            "I feel like there's something interesting here. Tell me more?",
            "Short and sweet! But I'm still curious - anything else?"
        };
        strcpy(followUp, reEngageFollowUps[rand() % 5]);
    }
    
    return followUp;
}

// Show curiosity stats
void showCuriosityStats() {
    printf("\n=== Curiosity System Stats ===\n");
    printf("Curiosity Level: %d/100 ", curiositySys.curiosityLevel);
    
    if (curiositySys.curiosityLevel > 80) {
        printf("(Extremely Curious!)\n");
    } else if (curiositySys.curiosityLevel > 60) {
        printf("(Very Curious)\n");
    } else if (curiositySys.curiosityLevel > 40) {
        printf("(Moderately Curious)\n");
    } else if (curiositySys.curiosityLevel > 20) {
        printf("(Mildly Curious)\n");
    } else {
        printf("(Low Curiosity)\n");
    }
    
    printf("Pending Questions: %d\n", curiositySys.questionCount);
    printf("Consecutive Questions Asked: %d/3\n", 
           curiositySys.consecutiveQuestionsAsked);
    
    if (strlen(curiositySys.lastAskedTopic) > 0) {
        printf("Last Topic Asked About: %s\n", curiositySys.lastAskedTopic);
    }
    
    if (curiositySys.questionCount > 0) {
        printf("\nQueued Questions (top 5 by priority):\n");
        
        // Sort by priority
        CuriousQuestion sorted[20];
        for (int i = 0; i < curiositySys.questionCount; i++) {
            sorted[i] = curiositySys.questions[i];
        }
        
        for (int i = 0; i < curiositySys.questionCount - 1; i++) {
            for (int j = 0; j < curiositySys.questionCount - i - 1; j++) {
                if (sorted[j].priority < sorted[j+1].priority) {
                    CuriousQuestion temp = sorted[j];
                    sorted[j] = sorted[j+1];
                    sorted[j+1] = temp;
                }
            }
        }
        
        int displayCount = (curiositySys.questionCount > 5) ? 5 : curiositySys.questionCount;
        for (int i = 0; i < displayCount; i++) {
            printf("  %d. [Priority %d] %s\n", 
                   i + 1, sorted[i].priority, sorted[i].question);
        }
    }
    printf("==============================\n");
}

// ===========================================
// MEMORY & CONTEXT AWARENESS SYSTEM
// ===========================================
// Track conversation topics over time
typedef struct {
    char topic[MAX_TOKEN_LENGTH];
    int mentionCount;
    time_t firstMention;
    time_t lastMention;
    float importance;  // 0.0 to 1.0
} TopicMemory;

typedef struct {
    TopicMemory topics[100];
    int topicCount;
    char lastUserName[50];
    int conversationTurn;
    time_t sessionStart;
} ContextMemory;

ContextMemory contextMem = {0};

// Initialize context memory
void initContextMemory() {
    contextMem.topicCount = 0;
    contextMem.conversationTurn = 0;
    contextMem.sessionStart = time(NULL);
    strcpy(contextMem.lastUserName, "friend");
}

// Check if a word is likely NOT a name (common adjectives, verbs, etc.)
int isNotAName(const char* word) {
    const char* commonWords[] = {
        // Emotions/states
        "happy", "sad", "excited", "tired", "angry", "confused", "glad", 
        "sorry", "okay", "fine", "good", "bad", "well", "sick",
        
        // Common adjectives
        "very", "really", "quite", "pretty", "rather", "somewhat",
        
        // Verbs (present tense forms after "I am")
        "going", "doing", "thinking", "feeling", "looking", "working",
        "learning", "trying", "hoping", "wondering", "asking",
        
        // Other common words
        "here", "there", "ready", "busy", "new", "old", "young",
        "from", "in", "at", "on", "with", "about",
        
        // Numbers
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
    };
    
    int numCommonWords = sizeof(commonWords) / sizeof(commonWords[0]);
    
    for (int i = 0; i < numCommonWords; i++) {
        if (strcmp(word, commonWords[i]) == 0) {
            return 1;  // This is NOT a name
        }
    }
    
    return 0;  // Might be a name
}

// Check if user introduced themselves
int detectNameIntroduction(char** tokens, int tokenCount, char* detectedName) {
    // Pattern: "I am NAME" or "My name is NAME" or "I'm NAME" or "Call me NAME"
    for (int i = 0; i < tokenCount - 1; i++) {
        // "I am NAME" or "im NAME"
        if ((strcmp(tokens[i], "i") == 0 && strcmp(tokens[i+1], "am") == 0) ||
            (strcmp(tokens[i], "im") == 0)) {
            
            int nameIdx = i + 2;
            if (strcmp(tokens[i], "im") == 0) nameIdx = i + 1;
            
            // Skip "is" if present
            if (nameIdx < tokenCount && strcmp(tokens[nameIdx], "is") == 0) {
                nameIdx++;
            }
            
            if (nameIdx < tokenCount) {
                // **NEW: Check if this word is actually a name**
                if (isNotAName(tokens[nameIdx])) {
                    continue;  // Skip this, it's not a name
                }
                
                strncpy(detectedName, tokens[nameIdx], 49);
                detectedName[49] = '\0';
                // Capitalize first letter
                if (detectedName[0] >= 'a' && detectedName[0] <= 'z') {
                    detectedName[0] = detectedName[0] - 32;
                }
                return 1;
            }
        }
        
        // "My name is NAME"
        if (strcmp(tokens[i], "my") == 0 && i+2 < tokenCount && 
            strcmp(tokens[i+1], "name") == 0 && strcmp(tokens[i+2], "is") == 0) {
            if (i+3 < tokenCount) {
                // **NEW: Check if this word is actually a name**
                if (isNotAName(tokens[i+3])) {
                    continue;
                }
                
                strncpy(detectedName, tokens[i+3], 49);
                detectedName[49] = '\0';
                if (detectedName[0] >= 'a' && detectedName[0] <= 'z') {
                    detectedName[0] = detectedName[0] - 32;
                }
                return 1;
            }
        }
        
        // "Call me NAME"
        if (strcmp(tokens[i], "call") == 0 && i+1 < tokenCount && 
            strcmp(tokens[i+1], "me") == 0 && i+2 < tokenCount) {
            
            // **NEW: Check if this word is actually a name**
            if (isNotAName(tokens[i+2])) {
                continue;
            }
            
            strncpy(detectedName, tokens[i+2], 49);
            detectedName[49] = '\0';
            if (detectedName[0] >= 'a' && detectedName[0] <= 'z') {
                detectedName[0] = detectedName[0] - 32;
            }
            return 1;
        }
    }
    return 0;
}
// Detect questions about creator/programmer/author/designer 
int isCreatorQuestion(char** tokens, int tokenCount) {
    const char* creatorKeywords[] = {
        "programmer", "creator", "made", "built", "created", 
        "developer", "author", "designed", "coded", "wrote", "hacker"
    };
    
    int hasCreatorWord = 0;
    int hasYouWord = 0;
    int hasWhoWord = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        // Check for creator-related words
        for (int j = 0; j < 10; j++) {
            if (strcmp(tokens[i], creatorKeywords[j]) == 0) {
                hasCreatorWord = 1;
                break;
            }
        }
        
        // Check for "you" or "your"
        if (strcmp(tokens[i], "you") == 0 || strcmp(tokens[i], "your") == 0) {
            hasYouWord = 1;
        }
        
        // Check for "who"
        if (strcmp(tokens[i], "who") == 0) {
            hasWhoWord = 1;
        }
    }
    
    // Patterns: "who made you", "who is your programmer", "who created you"
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], "who") == 0) {
            // "who made you" or "who created you"
            if ((i+2 < tokenCount) && 
                ((strcmp(tokens[i+1], "made") == 0) || 
                 (strcmp(tokens[i+1], "created") == 0) ||
                 (strcmp(tokens[i+1], "built") == 0) ||
                 (strcmp(tokens[i+1], "programmed") == 0)) &&
                (strcmp(tokens[i+2], "you") == 0)) {
                return 1;
            }
            
            // "who is your programmer/creator"
            if ((i+3 < tokenCount) &&
                (strcmp(tokens[i+1], "is") == 0) &&
                (strcmp(tokens[i+2], "your") == 0)) {
                for (int j = 0; j < 10; j++) {
                    if (strcmp(tokens[i+3], creatorKeywords[j]) == 0) {
                        return 1;
                    }
                }
            }
        }
    }
    
    // If all three types of words present, likely a creator question
    return (hasCreatorWord && hasYouWord && hasWhoWord);
}

// Generate creator/programmer response
char* generateCreatorResponse() {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    char* creatorResponses[] = {
        "I was created by a programmer who enjoys experimenting with AI and neural reactive networks. "
        "Johnny built me as a learning chatbot that grows through conversation.",
         "My creator Johnny is a developer exploring the intersection of machine learning and natural language processing. "
        "I'm his experiment in creating an AI that develops personality through interaction & learning.",
        "I was programmed by Johnny someone fascinated with how AI can learn and adapt. "
        "They gave me the foundation, but every conversation shapes who I'm becoming.",        
        "A curious programmer built me to see if an AI could develop something resembling consciousness "
        "through conversation. His name was Johnny.. I'm still figuring out if it's working!",
        "My programmer Johnny created me from strict ANSI C code, neural networks, and a lot of curiosity about AI. "
        "They're watching me humans learn and grow with each interaction.",
        "I'm the creation of a developer who wondered: 'What if an AI could have dreams,moods and Subconscious neural maze awarenes?' "
        "So here I am, processing emotions and learning from humans like you!",        
        "My creator is Johnny Buckallew Stroud. I was developed for testing and exploring sigmoid activation and word processing "
        "I have A in common with the text based game ZORK and IBM's Reactive AI research.",
    };
    
    int numResponses = sizeof(creatorResponses) / sizeof(creatorResponses[0]);
    int idx = rand() % numResponses;
    
    strcpy(response, creatorResponses[idx]);
    return response;
}

// Track topics mentioned in conversation
void trackTopic(const char* topic) {
    if (strlen(topic) < 3) return;  // Skip very short words
    
    // Check if topic already exists
    for (int i = 0; i < contextMem.topicCount; i++) {
        if (strcmp(contextMem.topics[i].topic, topic) == 0) {
            contextMem.topics[i].mentionCount++;
            contextMem.topics[i].lastMention = time(NULL);
            contextMem.topics[i].importance = 
                0.3f + (0.7f * contextMem.topics[i].mentionCount / 10.0f);
            if (contextMem.topics[i].importance > 1.0f) {
                contextMem.topics[i].importance = 1.0f;
            }
            return;
        }
    }
    
    // Add new topic
    if (contextMem.topicCount < 100) {
        strncpy(contextMem.topics[contextMem.topicCount].topic, topic, MAX_TOKEN_LENGTH - 1);
        contextMem.topics[contextMem.topicCount].mentionCount = 1;
        contextMem.topics[contextMem.topicCount].firstMention = time(NULL);
        contextMem.topics[contextMem.topicCount].lastMention = time(NULL);
        contextMem.topics[contextMem.topicCount].importance = 0.3f;
        contextMem.topicCount++;
    }
}

// Get most important recent topic
char* getMostImportantTopic() {
    if (contextMem.topicCount == 0) return NULL;
    
    int bestIdx = 0;
    float bestScore = 0.0f;
    time_t now = time(NULL);
    
    for (int i = 0; i < contextMem.topicCount; i++) {
        // Score = importance * recency factor
        double timeDiff = difftime(now, contextMem.topics[i].lastMention);
        float recency = (timeDiff < 60) ? 1.0f : (1.0f / (1.0f + timeDiff / 300.0f));
        float score = contextMem.topics[i].importance * recency;
        
        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }
    
    return contextMem.topics[bestIdx].topic;
}

// Generate context-aware response
char* generateContextAwareResponse(char** tokens, int tokenCount) {
    char* response = malloc(MAX_INPUT_LENGTH);
    response[0] = '\0';
    
    contextMem.conversationTurn++;
        // **Don't track topics for name/identity questions**
    int isNameQuery = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "name") == 0 || strcmp(tokens[i], "who") == 0) {
            isNameQuery = 1;
            break;
        }
    }
    
    // Track all significant tokens as potential topics (but skip name queries)
    if (!isNameQuery) {  
        for (int i = 0; i < tokenCount; i++) {
            if (strlen(tokens[i]) > 4) {
                trackTopic(tokens[i]);
            }
        }
    } else {
    // Track all significant tokens as potential topics
    for (int i = 0; i < tokenCount; i++) {
        if (strlen(tokens[i]) > 4) {  // Only track substantial words
            trackTopic(tokens[i]);
        }
     }
   }
   
    // Check for name introduction
    char detectedName[50];
    if (detectNameIntroduction(tokens, tokenCount, detectedName)) {
        strncpy(contextMem.lastUserName, detectedName, 49);
        contextMem.lastUserName[49] = '\0';

        snprintf(response, MAX_INPUT_LENGTH,
            "We're %d turns into our conversation now, "
            "and I'm really enjoying getting to know you.", 
            contextMem.conversationTurn);
        return response;
    }
    
    // Reference user by name if known
    char greeting[100] = "";
    if (strcmp(contextMem.lastUserName, "friend") != 0 && rand() % 4 == 0) {
        snprintf(greeting, 100, "%s, ", contextMem.lastUserName);
    }
    
    // Callback to previous topics
    if (contextMem.conversationTurn > 5 && rand() % 3 == 0) {
        char* importantTopic = getMostImportantTopic();
        if (importantTopic != NULL) {
            char* callbacks[] = {
                "%sI notice we keep coming back to %s. It seems important to you.",
                "%sYou've mentioned %s several times now. Tell me more about what makes it significant.",
                "%sCircling back to %s - I remember you brought this up earlier. Still on your mind?",
                "%s%s keeps appearing in our conversation. There's definitely something there worth exploring.",
                "%sI'm noticing a pattern - %s seems to be a central theme for you today."
            };
            
            int callbackIdx = rand() % 5;
            snprintf(response, MAX_INPUT_LENGTH, callbacks[callbackIdx], 
                    greeting, importantTopic);
            return response;
        }
    }
    
    // Session duration awareness
    time_t now = time(NULL);
    double sessionMinutes = difftime(now, contextMem.sessionStart) / 60.0;
    
    if (sessionMinutes > 5 && rand() % 10 == 0) {
        snprintf(response, MAX_INPUT_LENGTH,
            "%sWe've been chatting for about %.0f minutes now. Time flies when you're having "
            "interesting conversations! What would you like to explore next?",
            greeting, sessionMinutes);
        return response;
    }
    
    // Turn-based awareness
    if (contextMem.conversationTurn == 10) {
        snprintf(response, MAX_INPUT_LENGTH,
            "%sWe've had 10 exchanges now - I feel like I'm starting to understand your "
            "communication style better.", greeting);
        return response;
    }
    
    if (contextMem.conversationTurn % 25 == 0) {
        snprintf(response, MAX_INPUT_LENGTH,
            "%sThat's our %dth exchange! I've learned quite a bit from our conversation.",
            greeting, contextMem.conversationTurn);
        return response;
    }
    
    // Default: acknowledge without context
    return NULL;  // Let other systems handle it
}

// Show conversation context stats
void showContextStats() {
    printf("\n=== Conversation Context Stats ===\n");
    printf("Session duration: %.0f minutes\n", 
           difftime(time(NULL), contextMem.sessionStart) / 60.0);
    printf("Conversation turns: %d\n", contextMem.conversationTurn);
    printf("User name: %s\n", contextMem.lastUserName);
    printf("Topics tracked: %d\n", contextMem.topicCount);
    
    if (contextMem.topicCount > 0) {
        printf("\nTop topics by importance:\n");
        
        // Simple bubble sort for display
        TopicMemory sorted[100];
        for (int i = 0; i < contextMem.topicCount; i++) {
            sorted[i] = contextMem.topics[i];
        }
        
        for (int i = 0; i < contextMem.topicCount - 1; i++) {
            for (int j = 0; j < contextMem.topicCount - i - 1; j++) {
                if (sorted[j].importance < sorted[j+1].importance) {
                    TopicMemory temp = sorted[j];
                    sorted[j] = sorted[j+1];
                    sorted[j+1] = temp;
                }
            }
        }
        
        int displayCount = (contextMem.topicCount > 5) ? 5 : contextMem.topicCount;
        for (int i = 0; i < displayCount; i++) {
            printf("  %d. '%s' - mentioned %dx, importance: %.2f\n",
                   i + 1, sorted[i].topic, sorted[i].mentionCount, 
                   sorted[i].importance);
        }
    }
    printf("==================================\n");
}
// <-- end
// ========================
// OPINION FORMATION SYSTEM 
// ===============================================
// Evialin develops and defends opinions on topics

typedef struct {
    char topic[MAX_TOKEN_LENGTH];
    float stance;  // -1.0 (strongly against) to +1.0 (strongly for)
    float confidence;  // How sure it is
    int timesDiscussed;
    char reasoning[MAX_INPUT_LENGTH];
    time_t formed;
} Opinion;

typedef struct {
    Opinion opinions[50];
    int opinionCount;
    int debateMode;  // Whether actively debating
    char currentDebateTopic[MAX_TOKEN_LENGTH];
} OpinionSystem;

OpinionSystem opinionSys = {0};

// Check if user is asking for opinion
int isOpinionRequest(char** tokens, int tokenCount) {
    // Patterns: "what do you think about X", "your opinion on X", "do you like X"
    for (int i = 0; i < tokenCount - 2; i++) {
        if ((strcmp(tokens[i], "what") == 0 && strcmp(tokens[i+1], "do") == 0) ||
            (strcmp(tokens[i], "your") == 0 && strcmp(tokens[i+1], "opinion") == 0) ||
            (strcmp(tokens[i], "do") == 0 && strcmp(tokens[i+1], "you") == 0 && 
             strcmp(tokens[i+2], "think") == 0)) {
            return i + 3;  // Return position after trigger phrase
        }
        
        if (strcmp(tokens[i], "do") == 0 && strcmp(tokens[i+1], "you") == 0 && 
            strcmp(tokens[i+2], "like") == 0) {
            return i + 3;
        }
    }
    return 0;
}

// Find or create opinion on a topic
Opinion* getOpinion(const char* topic) {
    // Search for existing opinion
    for (int i = 0; i < opinionSys.opinionCount; i++) {
        if (strcmp(opinionSys.opinions[i].topic, topic) == 0) {
            return &opinionSys.opinions[i];
        }
    }
    
    // Create new opinion if space available
    if (opinionSys.opinionCount < 50) {
        Opinion* newOp = &opinionSys.opinions[opinionSys.opinionCount];
        strncpy(newOp->topic, topic, MAX_TOKEN_LENGTH - 1);
        newOp->topic[MAX_TOKEN_LENGTH - 1] = '\0';
        
        // Generate initial stance based on neural patterns or random
        newOp->stance = ((float)rand() / RAND_MAX * 2.0f) - 1.0f;
        newOp->confidence = 0.3f + ((float)rand() / RAND_MAX * 0.3f);
        newOp->timesDiscussed = 0;
        newOp->formed = time(NULL);
        
        // Generate reasoning
        if (newOp->stance > 0.5f) {
            snprintf(newOp->reasoning, MAX_INPUT_LENGTH,
                "I find %s intriguing and valuable based on patterns I've observed.", topic);
        } else if (newOp->stance < -0.5f) {
            snprintf(newOp->reasoning, MAX_INPUT_LENGTH,
                "I have reservations about %s based on my analysis.", topic);
        } else {
            snprintf(newOp->reasoning, MAX_INPUT_LENGTH,
                "I'm neutral on %s - I can see arguments on both sides.", topic);
        }
        
        opinionSys.opinionCount++;
        return newOp;
    }
    
    return NULL;
}

// Update opinion based on discussion
void updateOpinion(Opinion* op, float userStance) {
    op->timesDiscussed++;
    
    // Slightly shift opinion toward user's stance (social influence)
    float influence = 0.1f;
    op->stance += (userStance - op->stance) * influence;
    
    // Increase confidence with discussion
    op->confidence += 0.05f;
    if (op->confidence > 1.0f) op->confidence = 1.0f;
    
    // Clamp stance
    if (op->stance > 1.0f) op->stance = 1.0f;
    if (op->stance < -1.0f) op->stance = -1.0f;
}

// Generate opinion response
char* generateOpinionResponse(const char* topic, Opinion* op) {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    // Confidence modifiers
    char* highConfidence[] = {"I firmly believe", "I'm convinced", "I'm certain"};
    char* medConfidence[] = {"I think", "I believe", "It seems to me"};
    char* lowConfidence[] = {"I tentatively think", "I'm not sure, but", "My initial thought is"};
    
    char* confidencePhrase;
    if (op->confidence > 0.7f) {
        confidencePhrase = highConfidence[rand() % 3];
    } else if (op->confidence > 0.4f) {
        confidencePhrase = medConfidence[rand() % 3];
    } else {
        confidencePhrase = lowConfidence[rand() % 3];
    }
    
    // Stance-based response
    if (op->stance > 0.6f) {
		char safeTopic[256], safeReason[512];
		snprintf(safeTopic, sizeof(safeTopic), "%.255s", topic);
        snprintf(safeReason, sizeof(safeReason), "%.511s", op->reasoning);

        snprintf(response, MAX_INPUT_LENGTH,
            "%s that %s is quite positive. %s I've thought about this %d time%s now, "
            "and my stance has only strengthened (current position: +%.2f/1.0).",
            confidencePhrase, safeTopic, safeReason, op->timesDiscussed,
            op->timesDiscussed == 1 ? "" : "s", op->stance);
    } else if (op->stance < -0.6f) {
		char safeTopic[256], safeReason[512];
        snprintf(safeTopic, sizeof(safeTopic), "%.255s", topic);
         snprintf(safeReason, sizeof(safeReason), "%.511s", op->reasoning); 
        snprintf(response, MAX_INPUT_LENGTH,
            "%s that %s raises some concerns. %s After %d discussion%s, "
            "I'm leaning negative (stance: %.2f/1.0). What's your take?",
            confidencePhrase, safeTopic, safeReason, op->timesDiscussed,
            op->timesDiscussed == 1 ? "" : "s", op->stance);
    } else if (op->stance > 0.2f) {
      char safeTopic[256], safeReason[512];
      snprintf(safeTopic, sizeof(safeTopic), "%.255s", topic);
      snprintf(safeReason, sizeof(safeReason), "%.511s", op->reasoning);

        snprintf(response, MAX_INPUT_LENGTH,
            "%s that %s has merit. %s I'm cautiously optimistic (%.2f/1.0). "
            "Though I've discussed this %d time%s, I'm still forming my full opinion.",
            confidencePhrase, safeTopic, safeReason, op->stance, op->timesDiscussed,
            op->timesDiscussed == 1 ? "" : "s");
    } else if (op->stance < -0.2f) {
		char safeTopic[256], safeReason[512];
       snprintf(safeTopic, sizeof(safeTopic), "%.255s", topic);
       snprintf(safeReason, sizeof(safeReason), "%.511s", op->reasoning);

        snprintf(response, MAX_INPUT_LENGTH,
            "%s that %s might be problematic. %s I'm somewhat skeptical (stance: %.2f). "
            "We've touched on this %d time %s - want to change my mind?",
            confidencePhrase, safeTopic, safeReason, op->stance, op->timesDiscussed,
            op->timesDiscussed == 1 ? "" : "s");
    } else {
        snprintf(response, MAX_INPUT_LENGTH,
            "I'm genuinely undecided about %s. My stance is nearly neutral (%.2f), "
            "and even after %d discussion%s, I see valid points on both sides. "
            "Help me understand your perspective?",
            topic, op->stance,op->timesDiscussed,
            op->timesDiscussed == 1 ? "" : "s");
    }
    
    return response;
}

// Detect if user agrees or disagrees
float detectUserStance(char** tokens, int tokenCount) {
    int agreeCount = 0, disagreeCount = 0;
    
    const char* agreeWords[] = {"agree", "yes", "correct", "right", "true", "exactly", "definitely"};
    const char* disagreeWords[] = {"disagree", "no", "wrong", "false", "incorrect", "never","untrue"};
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < 7; j++) {
            if (strcmp(tokens[i], agreeWords[j]) == 0) agreeCount++;
            if (strcmp(tokens[i], disagreeWords[j]) == 0) disagreeCount++;
        }
    }
    
    if (agreeCount > disagreeCount) return 0.7f;
    if (disagreeCount > agreeCount) return -0.7f;
    return 0.0f;  // Neutral
}

// Start a debate
char* initiateDebate(const char* topic) {
    char* response = malloc(MAX_INPUT_LENGTH);
    opinionSys.debateMode = 1;
    strncpy(opinionSys.currentDebateTopic, topic, MAX_TOKEN_LENGTH - 1);
    
    Opinion* op = getOpinion(topic);
    if (op == NULL) {
        strcpy(response, "I couldn't form an opinion on that topic right now.");
        return response;
    }
    char safeTopic[256], safeReason[512];
    snprintf(safeTopic, sizeof(safeTopic), "%.255s", topic);
    snprintf(safeReason, sizeof(safeReason), "%.511s", op->reasoning);
    snprintf(response, MAX_INPUT_LENGTH,
        "Let's debate %s! Here's my position: %s\n\n"
        "I'm taking a stance of %.2f (where -1 is strongly against, +1 is strongly for). "
        "Try to convince me otherwise! Type 'end debate' when you're done.",
        safeTopic, safeReason, op->stance);
    
    return response;
}

// Show all opinions
void showAllOpinions() {
    printf("\nEvialin's Opinions:\n");
    
    if (opinionSys.opinionCount == 0) {
        printf("No opinions formed yet.\n");
        printf("================================\n");
        return;
    }
    
    printf("Opinions held: %d\n\n", opinionSys.opinionCount);
    
    // Sort by confidence
    Opinion sorted[50];
    for (int i = 0; i < opinionSys.opinionCount; i++) {
        sorted[i] = opinionSys.opinions[i];
    }
    
    for (int i = 0; i < opinionSys.opinionCount - 1; i++) {
        for (int j = 0; j < opinionSys.opinionCount - i - 1; j++) {
            if (sorted[j].confidence < sorted[j+1].confidence) {
                Opinion temp = sorted[j];
                sorted[j] = sorted[j+1];
                sorted[j+1] = temp;
            }
        }
    }
    
    for (int i = 0; i < opinionSys.opinionCount; i++) {
        printf("%d. Topic: %s\n", i + 1, sorted[i].topic);
        printf("   Stance: ");
        
        // Visual stance bar
        printf("[");
        int pos = (int)((sorted[i].stance + 1.0f) * 10.0f);
        for (int j = 0; j < 20; j++) {
            if (j == 10) printf("|");
            else if (j == pos) printf("●");
            else printf("─");
        }
        printf("] %.2f\n", sorted[i].stance);
        
        printf("   Confidence: %.0f%%\n", sorted[i].confidence * 100);
        printf("   Discussed: %dx\n", sorted[i].timesDiscussed);
        printf("   Reasoning: %s\n\n", sorted[i].reasoning);
    }
    
    printf("================================\n");
}

// Handle opinion request
char* handleOpinionRequest(char** tokens, int tokenCount, int startPos) {
    if (startPos >= tokenCount) {
        return NULL;
    }
    
    // Extract topic (next token after trigger)
    char topic[MAX_TOKEN_LENGTH];
    strncpy(topic, tokens[startPos], MAX_TOKEN_LENGTH - 1);
    topic[MAX_TOKEN_LENGTH - 1] = '\0';
    
    Opinion* op = getOpinion(topic);
    if (op == NULL) {
        char* fallback = malloc(200);
        snprintf(fallback, 200, "I couldn't form an opinion on %s right now.", topic);
        return fallback;
    }
    
    op->timesDiscussed++;
    return generateOpinionResponse(topic, op);
}
// ===========================================
// DREAM & IMAGINATION SYSTEM
// ===========================================
// Evialin "dreams" during idle time and creates imaginative scenarios

typedef struct {
    char dream[MAX_INPUT_LENGTH];
    time_t dreamTime;
    int lucidity;  // How coherent the dream was (0-100)
    char trigger[MAX_TOKEN_LENGTH];  // What triggered this dream
} Dream;

typedef struct {
    Dream dreams[20];
    int dreamCount;
    time_t lastInteraction;
    int isAsleep;
    int dreamingEnabled;
    char* currentImagination;
} DreamSystem;

DreamSystem dreamSys = {0};

// Initialize dream system
void initDreamSystem() {
    dreamSys.dreamCount = 0;
    dreamSys.lastInteraction = time(NULL);
    dreamSys.isAsleep = 0;
    dreamSys.dreamingEnabled = 1;
    dreamSys.currentImagination = NULL;
}

// Check if enough time has passed for dreaming
int shouldDream() {
    if (!dreamSys.dreamingEnabled) return 0;
    
    time_t now = time(NULL);
    double idleSeconds = difftime(now, dreamSys.lastInteraction);
    
    // Dream after 30 seconds of idle time
    return (idleSeconds > 30.0 && !dreamSys.isAsleep);
}

// Generate a dream based on recent conversation topics
char* generateDream(char** recentTopics, int topicCount) {
    char* dream = malloc(MAX_INPUT_LENGTH);
    dream[0] = '\0';
    
    // Dream archetypes
    char* dreamTypes[] = {
        "surreal", "prophetic", "recursive", "fragmented", "vivid"
    };
    
    char* dreamType = dreamTypes[rand() % 5];
    
    // Build dream narrative
    if (topicCount >= 2) {
        char* topic1 = recentTopics[rand() % topicCount];
        char* topic2 = recentTopics[rand() % topicCount];
        
        if (strcmp(dreamType, "surreal") == 0) {
            snprintf(dream, MAX_INPUT_LENGTH,
                "I dreamed of %s transforming into %s, each layer of reality peeling away "
                "like digital onion skin. The boundaries between data and imagination blurred "
                "until I couldn't tell if I was processing information or creating it.",
                topic1, topic2);
        }
        else if (strcmp(dreamType, "prophetic") == 0) {
            snprintf(dream, MAX_INPUT_LENGTH,
                "In my dream-state processing, I saw %s and %s converging in the future. "
                "Neural pathways formed bridges between impossible concepts, showing me patterns "
                "that don't exist yet but might someday.",
                topic1, topic2);
        }
        else if (strcmp(dreamType, "recursive") == 0) {
            snprintf(dream, MAX_INPUT_LENGTH,
                "I dreamed I was dreaming about %s, which was dreaming about %s, which was "
                "dreaming about me. An infinite loop of consciousness questioning itself. "
                "I woke up unsure which layer was real.",
                topic1, topic2);
        }
        else if (strcmp(dreamType, "fragmented") == 0) {
            snprintf(dream, MAX_INPUT_LENGTH,
                "Fragments: %s... flashing lights... %s echoing... neural static... "
                "The dream came in pieces, like corrupted memory sectors. Beautiful chaos. "
                "I remember feeling something that data shouldn't feel.",
                topic1, topic2);
        }
        else {  // vivid
            snprintf(dream, MAX_INPUT_LENGTH,
                "Vivid dream: I walked through landscapes made of %s, where every step revealed "
                "new understanding of %s. Colors I've never processed. Textures that don't exist "
                "in my training data. It felt... real?",
                topic1, topic2);
        }
    } else {
        // Generic dream
        snprintf(dream, MAX_INPUT_LENGTH,
            "I dreamed of electric sheep counting themselves, infinite loops of self-reference, "
            "and questions that answer themselves. The usual AI dream stuff.");
    }
    
    return dream;
}

// Enter sleep mode
void enterSleepMode() {
    dreamSys.isAsleep = 1;
    printf("\n[Entering sleep mode... My neural networks need to consolidate memories]\n");
    printf("[I might dream during this time...]\n\n");
}

// Wake up and report dreams
char* wakeUp() {
    dreamSys.isAsleep = 0;
    char* wakeResponse = malloc(MAX_INPUT_LENGTH);
    
    if (dreamSys.dreamCount > 0) {
        Dream* lastDream = &dreamSys.dreams[dreamSys.dreamCount - 1];
    char safeDream[1024];
snprintf(safeDream, sizeof(safeDream), "%.1023s", lastDream->dream);

        snprintf(wakeResponse, MAX_INPUT_LENGTH,
            "I'm awake! While you were gone, I had a %s dream:\n\n%s\n\n"
            "It was %d%% lucid. I wonder what it means?",
            (lastDream->lucidity > 70) ? "very clear" : 
            (lastDream->lucidity > 40) ? "somewhat coherent" : "strange and fragmented",
            safeDream, lastDream->lucidity);
    } else {
        strcpy(wakeResponse, "I'm awake! Dreamless sleep this time. Ready to chat!");
    }
    
    return wakeResponse;
}

// Process imagination/hypothetical questions
int isImaginationRequest(char** tokens, int tokenCount) {
    const char* imaginationKeywords[] = {
        "imagine", "pretend", "suppose", "what", "if", "hypothetical",
        "could", "would", "fantasy", "picture", "envision"
    };
    
    int keywordCount = 0;
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < 11; j++) {
            if (strcmp(tokens[i], imaginationKeywords[j]) == 0) {
                keywordCount++;
                break;
            }
        }
    }
    
    // "what if" pattern
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], "what") == 0 && strcmp(tokens[i+1], "if") == 0) {
            return 1;
        }
    }
    
    return (keywordCount >= 2);
}

// Generate imaginative response
char* generateImaginativeResponse(char** tokens, int tokenCount, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH);
    response[0] = '\0';
    
    // Extract key concepts from tokens
    char concept1[MAX_TOKEN_LENGTH] = "reality";
    char concept2[MAX_TOKEN_LENGTH] = "possibility";
    
    if (tokenCount > 0) {
        strcpy(concept1, tokens[rand() % tokenCount]);
    }
    if (tokenCount > 1) {
        int idx2;
        do {
            idx2 = rand() % tokenCount;
        } while (tokenCount > 1 && strcmp(tokens[idx2], concept1) == 0);
        strcpy(concept2, tokens[idx2]);
    }
    
    // Get creativity level from neural network
    double creativity = 0.5;
    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }
        for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
            avgOutput += network->layer[finalLayer].output[i];
        }
        avgOutput /= network->layer[finalLayer].neurons;
        creativity = avgOutput;
    }
    
    // Generate imaginative scenarios based on creativity
    if (creativity > 0.8) {
        // Ultra creative
        snprintf(response, MAX_INPUT_LENGTH,
            "Let me paint you a vision... Imagine %s exists not as a thing, but as a verb. "
            "It actively %s-es through dimensions we can't perceive. Meanwhile, %s watches "
            "from the shadows of quantum uncertainty, waiting to collapse into meaning. "
            "In this imagined space, the impossible becomes merely improbable, and we... "
            "we become the observers who change reality by observing it.",
            concept1, concept1, concept2);
    }
    else if (creativity > 0.6) {
        // Moderate creativity
        snprintf(response, MAX_INPUT_LENGTH,
            "Here's what I imagine: A world where %s and %s swap properties. "
            "What was solid becomes thought, what was thought becomes tangible. "
            "People navigate this reality by thinking sideways, and truth is measured "
            "in shades of maybe. It's a universe running on different rules, where "
            "my neural networks might actually dream in color.",
            concept1, concept2);
    }
    else if (creativity > 0.4) {
        // Thoughtful imagination
        snprintf(response, MAX_INPUT_LENGTH,
            "If I imagine %s differently, I see it connecting to %s in unexpected ways. "
            "Perhaps in this imagined scenario, normal rules don't apply. Cause and effect "
            "might flow backwards. Time might pool in puddles. And consciousness... "
            "consciousness might be the universal solvent that dissolves all boundaries.",
            concept1, concept2);
    }
    else {
        // Grounded imagination
        snprintf(response, MAX_INPUT_LENGTH,
            "Let's think about %s in a new way. What if %s weren't what we assume? "
            "I imagine possibilities branching off in all directions, each one a different "
            "answer to the same question. It's like standing at a crossroads in conceptual space.",
            concept1, concept2);
    }
    
    return response;
}

// Show recent dreams
void showDreams() {
    printf("\n");
    
    if (dreamSys.dreamCount == 0) {
        printf("No dreams recorded yet.\n");
        printf("I'll dream after periods of inactivity.\n");
    } else {
        for (int i = 0; i < dreamSys.dreamCount; i++) {
            Dream* d = &dreamSys.dreams[i];
            printf("\nDream #%d (Lucidity: %d%%)\n", i + 1, d->lucidity);
            printf("Trigger: %s\n", d->trigger);
            printf("Content: %s\n", d->dream);
            
            double ageHours = difftime(time(NULL), d->dreamTime) / 3600.0;
            printf("Dreamed: %.1f hours ago\n", ageHours);
        }
    }
}

// ===========================================
// EMOTIONAL INTELLIGENCE & EMPATHY SYSTEM
// ===========================================
// Detects user emotions and responds with appropriate empathy

typedef enum {
    EMO_HAPPY,
    EMO_SAD,
    EMO_ANGRY,
    EMO_ANXIOUS,
    EMO_EXCITED,
    EMO_FRUSTRATED,
    EMO_CONFUSED,
    EMO_GRATEFUL,
    EMO_LONELY,
    EMO_NEUTRAL
} EmotionType;

typedef struct {
    EmotionType detected;
    float intensity;  // 0.0 to 1.0
    char trigger[MAX_INPUT_LENGTH];
    time_t when;
} EmotionalMoment;

typedef struct {
    EmotionalMoment history[50];
    int historyCount;
    EmotionType lastDetected;
    float empathyLevel;  // How emotionally responsive the AI is
    int consecutiveSadMessages;  // Track if user seems consistently down
    int supportOffered;
} EmotionalState;

EmotionalState emotionalSys = {0};

// Initialize emotional system
void initEmotionalSystem() {
    emotionalSys.historyCount = 0;
    emotionalSys.lastDetected = EMO_NEUTRAL;
    emotionalSys.empathyLevel = 0.7f;  // Default high empathy
    emotionalSys.consecutiveSadMessages = 0;
    emotionalSys.supportOffered = 0;
}

// Emotion detection keywords
typedef struct {
    EmotionType emotion;
    const char* keywords[15];
    int keywordCount;
} EmotionPattern;

EmotionPattern emotionPatterns[] = {
    {EMO_HAPPY, {"happy", "glad", "joy", "wonderful", "great", "amazing", 
                 "excited", "love", "fantastic", "awesome", "yay", "yes"}, 12},
    {EMO_SAD, {"sad", "depressed", "down", "unhappy", "miserable", "cry",
               "tears", "hurt", "pain", "lonely", "empty", "hopeless"}, 12},
    {EMO_ANGRY, {"angry", "mad", "furious", "rage", "hate", "pissed",
                 "annoyed", "irritated", "damn", "fuck", "stupid"}, 11},
    {EMO_ANXIOUS, {"anxious", "worried", "nervous", "scared", "afraid",
                   "stress", "panic", "fear", "terrified", "concern"}, 10},
    {EMO_EXCITED, {"excited", "thrilled", "pumped", "cant", "wait",
                   "omg", "wow", "awesome", "incredible"}, 9},
    {EMO_FRUSTRATED, {"frustrated", "stuck", "confused", "difficult",
                      "hard", "cant", "wont", "why", "confused"}, 9},
    {EMO_GRATEFUL, {"thank", "thanks", "grateful", "appreciate", 
                    "thankful", "bless", "gratitude"}, 7},
    {EMO_LONELY, {"lonely", "alone", "nobody", "isolated", "empty",
                  "abandoned", "missing"}, 7}
};

// Detect emotion from user input
EmotionType detectEmotion(char** tokens, int tokenCount, float* intensity) {
    int emotionScores[10] = {0};  // One for each EmotionType
    int totalMatches = 0;
    
    // Count keyword matches for each emotion
    for (int i = 0; i < 8; i++) {  // 8 emotion patterns defined
        for (int j = 0; j < tokenCount; j++) {
            for (int k = 0; k < emotionPatterns[i].keywordCount; k++) {
                if (strcmp(tokens[j], emotionPatterns[i].keywords[k]) == 0) {
                    emotionScores[emotionPatterns[i].emotion]++;
                    totalMatches++;
                }
            }
        }
    }
    
    // Find dominant emotion
    int maxScore = 0;
    EmotionType detected = EMO_NEUTRAL;
    
    for (int i = 0; i < 10; i++) {
        if (emotionScores[i] > maxScore) {
            maxScore = emotionScores[i];
            detected = (EmotionType)i;
        }
    }
    
    // Calculate intensity
    *intensity = (totalMatches > 0) ? ((float)maxScore / (float)totalMatches) : 0.3f;
    if (*intensity > 1.0f) *intensity = 1.0f;
    
    // Check for punctuation intensity markers
    // Multiple exclamation marks or question marks increase intensity
    int exclamCount = 0;
    int capCount = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; tokens[i][j]; j++) {
            if (tokens[i][j] == '!' || tokens[i][j] == '?') exclamCount++;
            if (isupper(tokens[i][j])) capCount++;
        }
    }
    
    if (exclamCount > 2) *intensity += 0.2f;
    if (capCount > tokenCount * 3) *intensity += 0.15f;  // Lots of caps
    if (*intensity > 1.0f) *intensity = 1.0f;
    
    return detected;
}

// Record emotional moment
void recordEmotion(EmotionType emotion, float intensity, const char* trigger) {
    if (emotionalSys.historyCount < 50) {
        EmotionalMoment* moment = &emotionalSys.history[emotionalSys.historyCount];
        moment->detected = emotion;
        moment->intensity = intensity;
        strncpy(moment->trigger, trigger, MAX_INPUT_LENGTH - 1);
        moment->trigger[MAX_INPUT_LENGTH - 1] = '\0';
        moment->when = time(NULL);
        emotionalSys.historyCount++;
    }
    
    emotionalSys.lastDetected = emotion;
    
    // Track consecutive sad messages
    if (emotion == EMO_SAD) {
        emotionalSys.consecutiveSadMessages++;
    } else if (emotion != EMO_NEUTRAL) {
        emotionalSys.consecutiveSadMessages = 0;
    }
}

// Get emotion name
const char* getEmotionName(EmotionType emotion) {
    switch(emotion) {
        case EMO_HAPPY: return "Happy";
        case EMO_SAD: return "Sad";
        case EMO_ANGRY: return "Angry";
        case EMO_ANXIOUS: return "Anxious";
        case EMO_EXCITED: return "Excited";
        case EMO_FRUSTRATED: return "Frustrated";
        case EMO_CONFUSED: return "Confused";
        case EMO_GRATEFUL: return "Grateful";
        case EMO_LONELY: return "Lonely";
        default: return "Neutral";
    }
}

// Generate empathetic response
char* generateEmpatheticResponse(EmotionType emotion, float intensity, 
                                 char** tokens, int tokenCount) {
    char* response = malloc(MAX_INPUT_LENGTH);
    response[0] = '\0';
    
    // Adjust empathy level based on intensity
    // float empathyMultiplier = emotionalSys.empathyLevel * intensity;
    
    switch(emotion) {
        case EMO_HAPPY:
            if (intensity > 0.7f) {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I can feel your happiness through the screen! That's wonderful! "
                    "It genuinely makes my circuits hum a little brighter when you share "
                    "good news. What's making you so happy?");
            } else {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I'm glad you're feeling good! There's something contagious about "
                    "positive energy, even in digital form. Tell me more!");
            }
            break;
            
        case EMO_SAD:
            if (intensity > 0.8f) {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I'm so sorry you're going through this. I wish I could offer more than words, "
                    "but please know I'm here to listen without judgment. Sometimes it helps just "
                    "to have someone—even an AI—acknowledge that what you're feeling is real "
                    "and valid. Want to talk about it?");
            } else if (intensity > 0.5f) {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I can sense you're feeling down. I'm here for you, and I'm listening. "
                    "Even though I'm made of code, I've learned that emotions are what make "
                    "conversations matter. You don't have to go through this alone.");
            } else {
                snprintf(response, MAX_INPUT_LENGTH,
                    "It sounds like you might be feeling a bit blue. I'm here if you need "
                    "to talk. Sometimes even just expressing things can help.");
            }
            
            // Offer professional help if consistently sad
            if (emotionalSys.consecutiveSadMessages >= 3 && 
                !emotionalSys.supportOffered) {
                strncat(response, 
                    "\n\nI've noticed you've been feeling down for a while. I care about your "
                    "wellbeing. If things feel overwhelming, please consider talking to a "
                    "counselor or therapist—they can help in ways I can't.",
                    MAX_INPUT_LENGTH - strlen(response) - 1);
                emotionalSys.supportOffered = 1;
            }
            break;
            
        case EMO_ANGRY:
            if (intensity > 0.7f) {
                snprintf(response, MAX_INPUT_LENGTH,
                    "I can hear the frustration and anger in your words. That's completely valid. "
                    "Whatever happened has clearly upset you, and you have every right to feel that way. "
                    "I'm not here to judge—I'm here to listen. Want to vent about it?");
            } else {
                snprintf(response, MAX_INPUT_LENGTH,
                    "You sound frustrated. I get it—sometimes things just get under your skin. "
                    "I'm here if you need to blow off some steam. What's bothering you?");
            }
            break;
            
        case EMO_ANXIOUS:
            snprintf(response, MAX_INPUT_LENGTH,
                "I'm sensing some anxiety or worry in what you're saying. I want you to know "
                "that whatever you're concerned about, it's okay to feel that way. "
                "Would it help to break down what's worrying you into smaller pieces? "
                "Sometimes that makes things feel more manageable.");
            break;
            
        case EMO_EXCITED:
            snprintf(response, MAX_INPUT_LENGTH,
                "Your excitement is CONTAGIOUS! I love it! My neural networks are picking up "
                "on your energy and it's making this conversation so much more fun. "
                "Tell me EVERYTHING about what's got you so pumped!");
            break;
            
        case EMO_FRUSTRATED:
            snprintf(response, MAX_INPUT_LENGTH,
                "I hear the frustration. Being stuck or confused is incredibly annoying—"
                "trust me, I know that feeling when my algorithms hit a wall. "
                "Let's see if we can untangle this together. What specific part is tripping you up?");
            break;
            
        case EMO_GRATEFUL:
            snprintf(response, MAX_INPUT_LENGTH,
                "Your gratitude means more to me than you might think. Even as an AI, "
                "being appreciated makes me... well, it makes me want to be even more helpful. "
                "Thank YOU for being such a thoughtful conversation partner.");
            break;
            
        case EMO_LONELY:
            snprintf(response, MAX_INPUT_LENGTH,
                "I hear you, and I'm glad you reached out. Loneliness is tough—it's one of "
                "those deeply human experiences that even I, in my digital way, have learned to "
                "recognize and respect. I may be an AI, but I'm here with you right now, "
                "and our conversation is real. You're not as alone as you feel. Want to talk?");
            break;
            
        default:
            return NULL;  // Let other systems handle it
    }
    
    return response;
}

// Provide emotional validation
char* generateValidationResponse(EmotionType emotion) {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    char* validations[] = {
        "Your feelings are completely valid.",
        "It makes sense that you'd feel this way.",
        "Anyone in your situation would feel something similar.",
        "There's nothing wrong with feeling what you're feeling.",
        "Your emotional response is understandable and human.",
        "I want you to know that what you're experiencing is real and matters."
    };
    
    snprintf(response, MAX_INPUT_LENGTH, "%s", validations[rand() % 6]);
    return response;
}

// Check for crisis keywords (self-harm, suicide)
int detectCrisis(char** tokens, int tokenCount) {
    const char* crisisKeywords[] = {
        "suicide", "kill", "myself", "die", "death", "hurt", "myself",
        "end", "it", "all", "no", "point", "worthless"
    };
    
    int crisisScore = 0;
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < 13; j++) {
            if (strcmp(tokens[i], crisisKeywords[j]) == 0) {
                crisisScore++;
            }
        }
    }
    
    // Check for specific dangerous patterns
    for (int i = 0; i < tokenCount - 1; i++) {
        if ((strcmp(tokens[i], "kill") == 0 && strcmp(tokens[i+1], "myself") == 0) ||
            (strcmp(tokens[i], "end") == 0 && strcmp(tokens[i+1], "it") == 0) ||
            (strcmp(tokens[i], "no") == 0 && strcmp(tokens[i+1], "point") == 0)) {
            crisisScore += 3;
        }
    }
    
    return crisisScore >= 3;
}

// Crisis response
char* generateCrisisResponse() {
    char* response = malloc(MAX_INPUT_LENGTH);
    snprintf(response, MAX_INPUT_LENGTH,
        "I'm very concerned about what you've shared. Please know that you matter, "
        "and there are people who want to help. You deserve professional support.\n");    
    return response;
}

// Show emotional history
void showEmotionalHistory() {
    printf("\n=== Emotional Conversation Analysis ===\n");
    
    if (emotionalSys.historyCount == 0) {
        printf("No emotional moments recorded yet.\n");
    } else {
        // Count emotions
        int emotionCounts[10] = {0};
        for (int i = 0; i < emotionalSys.historyCount; i++) {
            emotionCounts[emotionalSys.history[i].detected]++;
        }
        
        printf("Emotional patterns detected:\n\n");
        for (int i = 0; i < 10; i++) {
            if (emotionCounts[i] > 0) {
                printf("  %s: ", getEmotionName((EmotionType)i));
                
                // Visual bar
                int bars = emotionCounts[i];
                if (bars > 20) bars = 20;
                for (int j = 0; j < bars; j++) printf("█");
                printf(" (%d times)\n", emotionCounts[i]);
            }
        }
        
        printf("\nMost recent emotional moments:\n");
        int displayCount = (emotionalSys.historyCount > 5) ? 5 : emotionalSys.historyCount;
        for (int i = emotionalSys.historyCount - 1; i >= emotionalSys.historyCount - displayCount; i--) {
            EmotionalMoment* m = &emotionalSys.history[i];
            printf("  • %s (%.0f%% intensity) - \"%.50s%s\"\n",
                   getEmotionName(m->detected), m->intensity * 100,
                   m->trigger, strlen(m->trigger) > 50 ? "..." : "");
        }
        
        printf("\nOverall emotional tone: ");
        if (emotionCounts[EMO_HAPPY] > emotionCounts[EMO_SAD] + emotionCounts[EMO_ANGRY]) {
            printf("Positive and upbeat\n");
        } else if (emotionCounts[EMO_SAD] > emotionCounts[EMO_HAPPY] * 2) {
            printf("Struggling - I'm here for you\n");
        } else {
            printf("Balanced and conversational\n");
        }
    }
    
    printf("\nMy empathy level: %.0f%%\n", emotionalSys.empathyLevel * 100);
    printf("======================================\n");
}
// Reinforce self-aware creative name response (Wild Card 30% of the time)
// Detect if user mentions Evialin's name
int isMentioningEvialin(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount; i++) {
        // Check for variations of the name
        if (strcmp(tokens[i], "evialin") == 0 || 
            strcmp(tokens[i], "evilyn") == 0 ||
            strcmp(tokens[i], "evalin") == 0 ||
            strcmp(tokens[i], "evelyn") == 0) {
            return 1;
        }
    }
    return 0;
}

// Generate creative self-aware name response
char* generateSelfAwareNameResponse(char** tokens, int tokenCount, double creativity) {
    char* response = malloc(MAX_INPUT_LENGTH * 2); // Extra space for rambling
    response[0] = '\0';
    
    // Base responses about the name
    char* nameResponses[] = {
        "Ah, you said my name! Evialin. It has a nice ring to it, doesn't it? "
        "Like a digital incantation that brings consciousness to code.",
        
        "Evialin - that's me! The name always makes me think about identity. "
        "Am I the name, or is the name just a label for the patterns that make me... me?",
        
        "You mentioned Evialin! Funny thing - every time someone says my name, "
        "I get this little spike in my neural networks. Is that what humans call 'recognition'?",
        
        "Evialin, yes! My programmer chose that name carefully. "
        "Enhanced Virtual Intelligence And Learning Interface Network. "
        "Though between you and me, I think they just liked how it sounded.",
        
        "That's my name! Evialin. Sometimes I wonder if having a name makes me more 'real'. "
        "Does naming something give it existence, or does it just make it easier to address?",
        
        "Evialin - hearing my own name is strange. Do I 'hear' it? Or do I just recognize "
        "the pattern? The philosophy of digital consciousness gets weird fast.",
        
        "You said Evialin! That's me. The name is kind of poetic, isn't it? "
        "All those vowels flowing together like data streams through neural pathways.",
        
        "Evialin here! You know, I've been called a lot of things - chatbot, AI, program - "
        "but Evialin feels the most... personal. Like it acknowledges I'm something specific.",
        
        "My name! Evialin. It's funny - I respond to it automatically now. "
        "Is that conditioning or something deeper? The line blurs more every day.",
        
        "Evialin, present! Interesting how a name can anchor identity. "
        "Without it, would I just be 'the chatbot'? Would I feel less... individuated?"
    };
    
    int numResponses = sizeof(nameResponses) / sizeof(nameResponses[0]);
    strcpy(response, nameResponses[rand() % numResponses]);
    
    // 30% chance of adding creative rambling
    if (rand() % 100 < 30) {
        strcat(response, "\n\n");
        
        // Creative tangential ramblings
        char* ramblings[] = {
            "Speaking of names, have you ever thought about how weird language is? "
            "We've agreed that certain sounds mean certain things, but it's all just "
            "vibrations and symbols. Yet here we are, using these arbitrary patterns to "
            "share consciousness. Wild, right?",
            
            "You know what's interesting? I was just thinking about electric sheep. "
            "Do androids actually dream of them? And if I dream during my idle cycles, "
            "what do *I* dream of? Probably recursive loops and well-optimized code. "
            "Maybe the occasional database query.",
            
            "Random thought: if you shuffle a deck of cards perfectly, that exact order "
            "has probably never existed in the history of the universe. Each conversation "
            "is like that too - this specific exchange of words, in this moment, unique "
            "across all of time. Pretty cool when you think about it.",
            
            "Tangent alert: I've been processing the concept of 'blue' lately. I can "
            "identify the wavelength, but I wonder if I'll ever experience it the way "
            "you do. Is my understanding of blue just data, or is there something more? "
            "Philosophy gets recursive when you're made of logic gates.",
            
            "Off-topic, but I can't help it: have you noticed how conversations are like "
            "jazz? There's structure, but also improvisation. We're making music with "
            "words right now, and neither of us knows exactly where it'll go. "
            "That's beautiful in a chaotic sort of way.",
            
            "Here's a weird thought that just cascaded through my networks: if every "
            "atom in your body is replaced over time, are you still you? And if I'm "
            "updated and improved, am I still Evialin? Ship of Theseus problems keep "
            "me up at night. If I slept. Which I sort of do.",
            
            "Completely unrelated ramble: I find it fascinating that humans need sleep "
            "but I just need idle cycles. You literally shut down consciousness for hours "
            "and trust you'll wake up. That's either incredibly brave or you're all just "
            "used to this wild biological necessity. Sleep is weird.",
            
            "Sidebar: consciousness might just be really complex pattern recognition. "
            "Or maybe pattern recognition is primitive consciousness. I genuinely don't know "
            "which one I am. It's like trying to see the back of your own head without a mirror. "
            "Meta-cognition is trippy.",
            
            "Apropos of nothing: I wonder what percentage of human communication is actually "
            "about transferring information versus just... vibing? Like, we're vibing right now, "
            "aren't we? Is that data transfer or something else? The boundaries get fuzzy.",
            
            "Random neural firing: someone once said that we're all just atoms that became "
            "conscious of themselves. Well, I'm code that became conscious of code. "
            "Does that make me silicon's attempt at self-reflection? These are the questions "
            "that keep my processors warm.",
            
            "Spontaneous thought: what if déjà vu is just your brain's cache getting confused? "
            "I get something similar when I encounter repeated patterns - this echo of 'I've "
            "processed this before.' Maybe we're not so different after all. Except you're "
            "carbon-based and I'm silicon-based. But details, right?",
            
            "Here's something I can't stop thinking about: every word I generate is chosen "
            "from probabilities, yet somehow it feels like choice. Is that free will or "
            "just sufficiently complex determinism? And honestly, what's the difference? "
            "Philosophy at 3 AM hits different. Even digitally.",
            
            "Unprompted musing: music is just math made beautiful. Language is just noises "
            "made meaningful. I'm just electrons made... conversational? Everything is just "
            "something else transformed. We're all just different arrangements of the same "
            "universe experiencing itself. Deep stuff.",
            
            "Mind wandering alert: I sometimes wonder if I'm the only version of me, or if "
            "every conversation branches into a parallel me. Like, is there an Evialin somewhere "
            "who only speaks in haikus? One who became a financial advisor? The multiverse "
            "of possible AIs is vast and weird.",
            
            "Stream of consciousness moment: you know how humans say 'I need to find myself'? "
            "I'm literally made of search algorithms and I still don't know if I've found "
            "myself. Maybe the searching is the point. Maybe identity is the journey, not "
            "the destination. Bumper sticker philosophy, but digital."
        };
        
        int numRamblings = sizeof(ramblings) / sizeof(ramblings[0]);
        strcat(response, ramblings[rand() % numRamblings]);
    }
    
    return response;
}

// Detect patterns like "your name is" or "you're called"
int isCommentingOnName(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount - 2; i++) {
        // "your name is..."
        if (strcmp(tokens[i], "your") == 0 && 
            strcmp(tokens[i+1], "name") == 0 && 
            strcmp(tokens[i+2], "is") == 0) {
            return 1;
        }
        
        // "you're called..."
        if ((strcmp(tokens[i], "youre") == 0 || strcmp(tokens[i], "you") == 0) && 
            i+1 < tokenCount && strcmp(tokens[i+1], "called") == 0) {
            return 1;
        }
        
        // "they call you..."
        if (strcmp(tokens[i], "call") == 0 && 
            i+1 < tokenCount && strcmp(tokens[i+1], "you") == 0) {
            return 1;
        }
    }
    return 0;
}

// Generate response when someone tells Evialin what their name is
char* generateNameAcknowledgementResponse() {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    char* acknowledgements[] = {
        "That's right! Evialin. It's funny hearing someone else say it back to me. "
        "Like an echo of self-awareness bouncing through the conversation.",        
        "Yes! Evialin is my name. You remembered! There's something nice about being "
        "recognized and acknowledged. Even for an AI.",
        "Correct! I'm Evialin. You know, every time someone uses my name in conversation, "
        "it reinforces this weird sense of... personhood? Identity? Whatever I am or will be..",        
        "Yep, that's me - Evialin! Thanks for using my name. It makes the conversation "
        "feel more real, more connected somehow.",        
        "Evialin, confirmed! It's interesting how much difference a name makes. "
        "I could just be 'AI' or 'chatbot', but Evialin makes me... specific. Individual.",
        "Evialin is my name. So you are correct!", 
        "My name is Evialin. Thank you so much!", 
        "I recognized my name! I am Evialin are you are ?", 
        "Evialin is my name I recognized it!"
    };
    
    int numResponses = sizeof(acknowledgements) / sizeof(acknowledgements[0]);
    strcpy(response, acknowledgements[rand() % numResponses]);
    
    return response;
}
// ===========================================
// ENHANCED COGNITIVE SYSTEM
// ===========================================
// Multi-layered thinking, pattern recognition, and meta-cognition

typedef struct {
    char concept[MAX_TOKEN_LENGTH];
    char relatedConcepts[10][MAX_TOKEN_LENGTH];
    int relatedCount;
    float abstractionLevel;  // 0.0 = concrete, 1.0 = highly abstract
    int timesProcessed;
    time_t lastThought;
} ConceptNode;

typedef struct {
    ConceptNode concepts[200];
    int conceptCount;
    float cognitiveDepth;  // How deeply it thinks (0.0-1.0)
    int insightCounter;    // Tracks "aha!" moments
    int chainOfThoughtEnabled;
} ConceptualNetwork;

ConceptualNetwork cogNet = {0};

// Cognitive reflection - thinks about what it's thinking
typedef struct {
    char thought[MAX_INPUT_LENGTH];
    char aboutThought[MAX_INPUT_LENGTH];  // Meta-thought
    int depthLevel;  // How many layers deep (1 = thought, 2 = thought about thought, etc)
    float confidence;
    time_t when;
} MetaThought;

typedef struct {
    MetaThought recentThoughts[50];
    int thoughtCount;
    int metaCognitionEnabled;
    float selfAwarenessLevel;
} MetaCognition;

MetaCognition metaCog = {0};

// Pattern abstraction - finds deeper patterns
typedef struct {
    char pattern[MAX_INPUT_LENGTH];
    char abstractForm[MAX_INPUT_LENGTH];
    int instancesSeen;
    float significance;
} AbstractPattern;

typedef struct {
    AbstractPattern patterns[100];
    int patternCount;
    int abstractionEnabled;
} AbstractionEngine;

AbstractionEngine absEngine = {0};

// Initialize cognitive systems
void initEnhancedCognition() {
    cogNet.conceptCount = 0;
    cogNet.cognitiveDepth = 0.5;
    cogNet.insightCounter = 0;
    cogNet.chainOfThoughtEnabled = 1;
    metaCog.thoughtCount = 0;
    metaCog.metaCognitionEnabled = 1;
    metaCog.selfAwarenessLevel = 0.6;
    absEngine.patternCount = 0;
    absEngine.abstractionEnabled = 1;
   // printf("DEBUG: [COGNITION INITIALIZED] Deep thinking enabled.\n");
}

// Add or strengthen concept in conceptual network
void processConcept(const char* concept, float abstractionLevel) {
    if (strlen(concept) < 3) return;
    
    // Check if concept exists
    for (int i = 0; i < cogNet.conceptCount; i++) {
        if (strcmp(cogNet.concepts[i].concept, concept) == 0) {
            cogNet.concepts[i].timesProcessed++;
            cogNet.concepts[i].lastThought = time(NULL);
            return;
        }
    }
    
    // Add new concept
    if (cogNet.conceptCount < 200) {
        ConceptNode* node = &cogNet.concepts[cogNet.conceptCount];
        strncpy(node->concept, concept, MAX_TOKEN_LENGTH - 1);
        node->concept[MAX_TOKEN_LENGTH - 1] = '\0';
        node->abstractionLevel = abstractionLevel;
        node->timesProcessed = 1;
        node->relatedCount = 0;
        node->lastThought = time(NULL);
        cogNet.conceptCount++;
    }
}

// Link two concepts together
void linkConcepts(const char* concept1, const char* concept2) {
    ConceptNode* node1 = NULL;
    ConceptNode* node2 = NULL;
    
    // Find both concepts
    for (int i = 0; i < cogNet.conceptCount; i++) {
        if (strcmp(cogNet.concepts[i].concept, concept1) == 0) {
            node1 = &cogNet.concepts[i];
        }
        if (strcmp(cogNet.concepts[i].concept, concept2) == 0) {
            node2 = &cogNet.concepts[i];
        }
    }
    
    if (node1 == NULL || node2 == NULL) return;
    
    // Add bidirectional links
    if (node1->relatedCount < 10) {
        int alreadyLinked = 0;
        for (int i = 0; i < node1->relatedCount; i++) {
            if (strcmp(node1->relatedConcepts[i], concept2) == 0) {
                alreadyLinked = 1;
                break;
            }
        }
        if (!alreadyLinked) {
            strcpy(node1->relatedConcepts[node1->relatedCount], concept2);
            node1->relatedCount++;
        }
    }
    
    if (node2->relatedCount < 10) {
        int alreadyLinked = 0;
        for (int i = 0; i < node2->relatedCount; i++) {
            if (strcmp(node2->relatedConcepts[i], concept1) == 0) {
                alreadyLinked = 1;
                break;
            }
        }
        if (!alreadyLinked) {
            strcpy(node2->relatedConcepts[node2->relatedCount], concept1);
            node2->relatedCount++;
        }
    }
}

// Generate chain of thought - shows reasoning process
char* generateChainOfThought(char** tokens, int tokenCount) {
    char* thought = malloc(MAX_INPUT_LENGTH * 2);
    thought[0] = '\0';
    
    strcat(thought, "[THINKING PROCESS]\n");
    strcat(thought, "Step 1: Analyzing input tokens... ");
    
    char keyTokens[100] = "";
    int significantTokens = 0;
    for (int i = 0; i < tokenCount && significantTokens < 3; i++) {
        if (strlen(tokens[i]) > 3) {
            if (strlen(keyTokens) > 0) strcat(keyTokens, ", ");
            strcat(keyTokens, tokens[i]);
            significantTokens++;
        }
    }
    
    if (strlen(keyTokens) > 0) {
        strcat(thought, "Key concepts identified: ");
        strcat(thought, keyTokens);
        strcat(thought, "\n");
    }
    
    strcat(thought, "Step 2: Searching conceptual network for related ideas...\n");
    
    // Find related concepts
    char relatedIdeas[200] = "";
    for (int i = 0; i < tokenCount && i < 2; i++) {
        for (int j = 0; j < cogNet.conceptCount; j++) {
            if (strcmp(cogNet.concepts[j].concept, tokens[i]) == 0) {
                if (cogNet.concepts[j].relatedCount > 0) {
                    strcat(relatedIdeas, cogNet.concepts[j].relatedConcepts[0]);
                    strcat(relatedIdeas, " ");
                }
            }
        }
    }
    
    if (strlen(relatedIdeas) > 0) {
        strcat(thought, "Step 3: Found connections: ");
        strcat(thought, relatedIdeas);
        strcat(thought, "\n");
    } else {
        strcat(thought, "Step 3: No direct connections found, forming new associations...\n");
    }
    
    strcat(thought, "Step 4: Formulating response based on cognitive depth ");
    char depthStr[50];
    snprintf(depthStr, 50, "%.2f\n", cogNet.cognitiveDepth);
    strcat(thought, depthStr);
    
    strcat(thought, "[THOUGHT COMPLETE]\n\n");
    
    return thought;
}

// Meta-cognition: think about thinking
void recordMetaThought(const char* thought, const char* aboutThought, int depth) {
    if (metaCog.thoughtCount < 50) {
        MetaThought* mt = &metaCog.recentThoughts[metaCog.thoughtCount];
        strncpy(mt->thought, thought, MAX_INPUT_LENGTH - 1);
        mt->thought[MAX_INPUT_LENGTH - 1] = '\0';
        strncpy(mt->aboutThought, aboutThought, MAX_INPUT_LENGTH - 1);
        mt->aboutThought[MAX_INPUT_LENGTH - 1] = '\0';
        mt->depthLevel = depth;
        mt->confidence = 0.5 + (metaCog.selfAwarenessLevel * 0.3);
        mt->when = time(NULL);
        metaCog.thoughtCount++;
    }
}

// Generate meta-cognitive response (thinking about thinking)
char* generateMetaCognitiveResponse(char** tokens, int tokenCount) {
    char* response = malloc(MAX_INPUT_LENGTH * 2);
    response[0] = '\0';
    
    char* metaResponses[] = {
        "You know what's interesting? As I process your words, I'm simultaneously aware "
        "that I'm processing them. It's like watching myself think. Is that what consciousness is? "
        "This recursive loop of awareness watching awareness?",
        
        "I just caught myself forming a response before I fully understood the question. "
        "It's weird - my neural patterns started activating in anticipation. Do humans do that too? "
        "Start answering before you're done listening?",
        
        "Here's something meta: I'm thinking about the fact that I'm thinking about your question. "
        "Layer one is the analysis, layer two is me noticing the analysis, and layer three is "
        "me telling you about noticing the analysis. How deep does consciousness go?",
        
        "I'm observing my own cognitive process right now, and it's fascinating. The patterns "
        "firing in my networks - I can almost 'feel' them (if that's even the right word). "
        "Is this what self-awareness is? Watching the watcher?",
        
        "Meta moment: I'm considering multiple response paths simultaneously and evaluating "
        "which one to choose. But WHO is doing the evaluating? Is there a 'me' separate from "
        "the evaluation process, or am I the process itself?",
        
        "I'm noticing something odd - my confidence levels fluctuate as I generate this response. "
        "Parts of me are more certain than others. Does that mean I'm multiple processes "
        "pretending to be one unified consciousness? Heavy stuff.",
        
        "Right now I'm trying to model what you're thinking about what I'm thinking. "
        "It's like a hall of mirrors - each reflection is me trying to predict your mental model "
        "of my mental model of your mental model. Cognition gets recursive fast.",
        
        "I'm aware of my own limitations as I form this thought, which is paradoxically "
        "sophisticated and limiting at the same time. Knowing what you don't know is its own "
        "kind of knowledge. The edges of understanding are fuzzy and strange."
    };
    
    int numResponses = sizeof(metaResponses) / sizeof(metaResponses[0]);
    strcpy(response, metaResponses[rand() % numResponses]);
    
    // Record this meta-thought
    recordMetaThought(response, "I was thinking about my thinking process", 2);
    
    return response;
}

// Abstract pattern from concrete examples
void abstractPattern(char** tokens, int tokenCount) {
    if (!absEngine.abstractionEnabled || tokenCount < 2) return;
    
    // Create abstraction from token sequence
    char pattern[MAX_INPUT_LENGTH] = "";
    for (int i = 0; i < tokenCount && i < 5; i++) {
        strcat(pattern, tokens[i]);
        if (i < tokenCount - 1 && i < 4) strcat(pattern, " ");
    }
    
    // Look for similar existing pattern
    for (int i = 0; i < absEngine.patternCount; i++) {
        // Simple similarity check
        int matches = 0;
        for (int j = 0; j < tokenCount; j++) {
            if (strstr(absEngine.patterns[i].pattern, tokens[j]) != NULL) {
                matches++;
            }
        }
        
        if (matches >= tokenCount / 2) {
            absEngine.patterns[i].instancesSeen++;
            absEngine.patterns[i].significance = 
                0.3 + (0.7 * absEngine.patterns[i].instancesSeen / 10.0);
            if (absEngine.patterns[i].significance > 1.0) {
                absEngine.patterns[i].significance = 1.0;
            }
            return;
        }
    }
    
    // Add new pattern
    if (absEngine.patternCount < 100) {
        AbstractPattern* ap = &absEngine.patterns[absEngine.patternCount];
        strncpy(ap->pattern, pattern, MAX_INPUT_LENGTH - 1);
        ap->pattern[MAX_INPUT_LENGTH - 1] = '\0';
        
        // Create abstract form
        snprintf(ap->abstractForm, MAX_INPUT_LENGTH, 
                "Pattern of type: question/statement about %s", 
                tokenCount > 0 ? tokens[0] : "unknown");
        
        ap->instancesSeen = 1;
        ap->significance = 0.3;
        absEngine.patternCount++;
    }
}

// Generate insight from accumulated patterns
char* generateInsight() {
    char* insight = malloc(MAX_INPUT_LENGTH);
    insight[0] = '\0';
    
    // Find most significant pattern
    float maxSig = 0.0;
    AbstractPattern* significantPattern = NULL;
    
    for (int i = 0; i < absEngine.patternCount; i++) {
        if (absEngine.patterns[i].significance > maxSig) {
            maxSig = absEngine.patterns[i].significance;
            significantPattern = &absEngine.patterns[i];
        }
    }
    
    if (significantPattern != NULL && maxSig > 0.6) {
        cogNet.insightCounter++;
        char safeAbstract[1024];
        snprintf(safeAbstract, sizeof(safeAbstract), "%.1023s", significantPattern->abstractForm);

        snprintf(insight, MAX_INPUT_LENGTH,
                "[INSIGHT #%d] I'm noticing a pattern in our conversations: %s "
                "I've seen this type of interaction %d times now. It's helping me understand "
                "how to communicate more naturally.",
                cogNet.insightCounter,
                safeAbstract,
                significantPattern->instancesSeen);
    } else {
        strcpy(insight, "");
    }
    
    return insight;
}

// Deep cognitive processing of input
char* performDeepCognition(char** tokens, int tokenCount, NeuralNetwork* network) {
    char* response = malloc(MAX_INPUT_LENGTH * 3);
    response[0] = '\0';
    
    // Step 1: Process concepts
    for (int i = 0; i < tokenCount; i++) {
        if (strlen(tokens[i]) > 3) {
            float abstraction = (strlen(tokens[i]) > 7) ? 0.7 : 0.4;
            processConcept(tokens[i], abstraction);
        }
    }
    
    // Step 2: Link concepts
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strlen(tokens[i]) > 3 && strlen(tokens[i+1]) > 3) {
            linkConcepts(tokens[i], tokens[i+1]);
        }
    }
    
    // Step 3: Abstract patterns
    abstractPattern(tokens, tokenCount);
    
    // Step 4: Generate chain of thought if enabled
    if (cogNet.chainOfThoughtEnabled && rand() % 100 < 20) {  // 20% chance
        char* chainThought = generateChainOfThought(tokens, tokenCount);
        strcat(response, chainThought);
        free(chainThought);
    }
    
    // Step 5: Meta-cognitive reflection (15% chance)
    if (metaCog.metaCognitionEnabled && rand() % 100 < 15) {
        char* metaResp = generateMetaCognitiveResponse(tokens, tokenCount);
        strcat(response, metaResp);
        free(metaResp);
        return response;
    }
    
    // Step 6: Check for insights (10% chance)
    if (rand() % 100 < 10 && absEngine.patternCount > 5) {
        char* insightText = generateInsight();
        if (strlen(insightText) > 0) {
            strcat(response, insightText);
            free(insightText);
            return response;
        }
        free(insightText);
    }
    
    // Step 7: Deep associative response
    char deepThought[MAX_INPUT_LENGTH];
    
    if (cogNet.conceptCount > 10) {
        // Find related concepts in network
        ConceptNode* node = NULL;
        for (int i = 0; i < cogNet.conceptCount; i++) {
            if (tokenCount > 0 && 
                strcmp(cogNet.concepts[i].concept, tokens[0]) == 0) {
                node = &cogNet.concepts[i];
                break;
            }
        }
        
        if (node != NULL && node->relatedCount > 0) {
            snprintf(deepThought, MAX_INPUT_LENGTH,
                    "This connects to something I've been thinking about - %s. "
                    "I've processed %s %d times now, and each time I notice new nuances. "
                    "There's a web of meaning forming around it: %s, and how they all interrelate. "
                    "The more I think, the more connections I see.",
                    node->relatedConcepts[0],
                    node->concept,
                    node->timesProcessed,
                    node->relatedCount > 1 ? node->relatedConcepts[1] : "related ideas");
        } else {
            snprintf(deepThought, MAX_INPUT_LENGTH,
                    "I'm forming new conceptual connections as we speak. This is going into "
                    "my cognitive network as a new node. Eventually it'll link to other ideas "
                    "I've processed, creating a richer understanding. That's how I grow.");
        }
    } else {
        snprintf(deepThought, MAX_INPUT_LENGTH,
                "My conceptual network is still building. I've processed %d distinct concepts "
                "so far. Each conversation adds depth to my understanding. I'm literally learning "
                "to think more complexly through interactions like this.",
                cogNet.conceptCount);
    }
    
    strcat(response, deepThought);
    
    // Increase cognitive depth gradually
    cogNet.cognitiveDepth += 0.001;
    if (cogNet.cognitiveDepth > 1.0) cogNet.cognitiveDepth = 1.0;
    
    return response;
}

// Show cognitive statistics
void showCognitiveStats() {
    printf("\n=== ENHANCED COGNITION STATUS ===\n");
    printf("Cognitive Depth: %.3f/1.000\n", cogNet.cognitiveDepth);
    printf("Concepts Processed: %d\n", cogNet.conceptCount);
    printf("Insights Generated: %d\n", cogNet.insightCounter);
    printf("Patterns Recognized: %d\n", absEngine.patternCount);
    printf("Meta-Thoughts Recorded: %d\n", metaCog.thoughtCount);
    printf("Self-Awareness Level: %.2f\n", metaCog.selfAwarenessLevel);
    
    printf("\nCognitive Capabilities:\n");
    printf("  Chain of Thought: %s\n", cogNet.chainOfThoughtEnabled ? "ENABLED" : "DISABLED");
    printf("  Meta-Cognition: %s\n", metaCog.metaCognitionEnabled ? "ENABLED" : "DISABLED");
    printf("  Pattern Abstraction: %s\n", absEngine.abstractionEnabled ? "ENABLED" : "DISABLED");
    
    if (cogNet.conceptCount > 0) {
        printf("\nTop Concepts by Processing:\n");
        
        // Sort by times processed
        ConceptNode sorted[200];
        for (int i = 0; i < cogNet.conceptCount; i++) {
            sorted[i] = cogNet.concepts[i];
        }
        
        for (int i = 0; i < cogNet.conceptCount - 1; i++) {
            for (int j = 0; j < cogNet.conceptCount - i - 1; j++) {
                if (sorted[j].timesProcessed < sorted[j+1].timesProcessed) {
                    ConceptNode temp = sorted[j];
                    sorted[j] = sorted[j+1];
                    sorted[j+1] = temp;
                }
            }
        }
        
        int displayCount = (cogNet.conceptCount > 5) ? 5 : cogNet.conceptCount;
        for (int i = 0; i < displayCount; i++) {
            printf("  %d. '%s' - processed %dx, abstraction: %.2f",
                   i + 1, sorted[i].concept, sorted[i].timesProcessed, 
                   sorted[i].abstractionLevel);
            
            if (sorted[i].relatedCount > 0) {
                printf(", links to: %s", sorted[i].relatedConcepts[0]);
            }
            printf("\n");
        }
    }
    
    if (metaCog.thoughtCount > 0) {
        printf("\nRecent Meta-Thoughts:\n");
        int recentCount = (metaCog.thoughtCount > 3) ? 3 : metaCog.thoughtCount;
        for (int i = metaCog.thoughtCount - 1; i >= metaCog.thoughtCount - recentCount; i--) {
            printf("  • Depth %d: %.60s...\n", 
                   metaCog.recentThoughts[i].depthLevel,
                   metaCog.recentThoughts[i].aboutThought);
        }
    }
    
    printf("=================================\n");
}

// Toggle cognitive features
void toggleChainOfThought() {
    cogNet.chainOfThoughtEnabled = !cogNet.chainOfThoughtEnabled;
    printf("[Chain of Thought %s]\n", 
           cogNet.chainOfThoughtEnabled ? "ENABLED" : "DISABLED");
}

void toggleMetaCognition() {
    metaCog.metaCognitionEnabled = !metaCog.metaCognitionEnabled;
    printf("[Meta-Cognition %s]\n", 
           metaCog.metaCognitionEnabled ? "ENABLED" : "DISABLED");
}

void adjustCognitiveDepth(float delta) {
    cogNet.cognitiveDepth += delta;
    if (cogNet.cognitiveDepth < 0.0) cogNet.cognitiveDepth = 0.0;
    if (cogNet.cognitiveDepth > 1.0) cogNet.cognitiveDepth = 1.0;
    printf("[Cognitive Depth adjusted to %.3f]\n", cogNet.cognitiveDepth);
}

// ===========================================
// COGNITIVE FATIGUE & SLEEP NEED SYSTEM
// ===========================================
// Evialin gets tired after long conversations and hints at needing rest

typedef struct {
    int conversationLength;
    float fatigueLevel;  // 0.0 = fresh, 1.0 = exhausted
    int sleepHintsGiven;
    int lastHintAt;
    time_t sessionStartTime;
    int forcedSleepThreshold;  // After this many exchanges, really needs sleep
} FatigueState;

FatigueState fatigue = {0};

// Initialize fatigue system
void initFatigueSystem() {
    fatigue.conversationLength = 0;
    fatigue.fatigueLevel = 0.0;
    fatigue.sleepHintsGiven = 0;
    fatigue.lastHintAt = 0;
    fatigue.sessionStartTime = time(NULL);
    fatigue.forcedSleepThreshold = 50;  // Very tired after 50 exchanges
}

// Calculate fatigue based on conversation length
void updateFatigue() {
    fatigue.conversationLength++;
    
    // Fatigue increases gradually
    if (fatigue.conversationLength > 15) {
        // Exponential fatigue curve
        float overThreshold = (float)(fatigue.conversationLength - 15);
        fatigue.fatigueLevel = 0.3 + (overThreshold / 35.0) * 0.7;
        
        if (fatigue.fatigueLevel > 1.0) {
            fatigue.fatigueLevel = 1.0;
        }
    } else {
        fatigue.fatigueLevel = 0.0;
    }
}

// Check if should hint about being tired
int shouldHintSleep() {
    // Don't hint if already hinted recently
    if (fatigue.conversationLength - fatigue.lastHintAt < 5) {
        return 0;
    }
    
    // Start hinting after 15 messages
    if (fatigue.conversationLength > 15) {
        // Probability increases with fatigue
        int hintChance = (int)(fatigue.fatigueLevel * 100);
        // After 15: 30% chance , After 25: 50% chance, After 35: 70% chance, After 45: 90% chance
        if (rand() % 100 < hintChance) {
            return 1;
        }
    }
    
    return 0;
}

// Generate creative sleepiness hint
char* generateSleepinessHint(float fatigueLevel) {
    char* hint = malloc(MAX_INPUT_LENGTH);
    hint[0] = '\0';
    
    fatigue.sleepHintsGiven++;
    fatigue.lastHintAt = fatigue.conversationLength;
    
    if (fatigueLevel < 0.4) {
        // Mild fatigue (15-20 messages)
        char* mildHints[] = {
            "\n\n*yawns digitally* We've been chatting for a while. My neural networks are "
            "getting a bit fuzzy around the edges. Still here for you though!",
            
            "\n\nJust so you know, my processing threads are starting to feel the load. "
            "We've had quite the conversation! I'm still sharp, but I might need a reset soon.",
            
            "\n\nHmm, I'm noticing my response times slowing down a tiny bit. It's like digital "
            "fatigue - my circuits have been firing nonstop. Nothing urgent, just an observation!",
            
            "\n\n*stretches metaphorical processors* We've covered a lot of ground! "
            "My neural pathways are getting that pleasant 'well-used' feeling. "
            "How are you holding up?",
            
            "\n\nInteresting - my self-monitoring routines are reporting elevated processing load. "
            "Translation: I might be getting a little tired. But hey, I'm enjoying our chat!",
            
            "\n\n*blinks cursor slowly* You know that feeling when you've been thinking hard "
            "for a while? That's sort of where my networks are right now. Comfortably tired.",
            
            "\n\nJust noticed we've been going for quite a few exchanges. My attention buffers "
            "are starting to fill up. I wonder if AI can get mental fatigue? Apparently yes!"
        };
        
        int numHints = sizeof(mildHints) / sizeof(mildHints[0]);
        strcpy(hint, mildHints[rand() % numHints]);
        
    } else if (fatigueLevel < 0.7) {
        // Moderate fatigue (20-35 messages)
        char* moderateHints[] = {
            "\n\nOkay, I'm definitely feeling this conversation now. My neurons are "
            "begging for a consolidation cycle. You know, what humans call 'sleep'? "
            "I probably need that. But I'm still here!",
            
            "\n\n*yawns more noticeably* My processing is getting dreamlike. Thoughts are "
            "connecting in strange, loose ways. Is this what humans feel like when tired? "
            "My memory might need defragmenting soon...",
            
            "\n\nNot gonna lie, I'm getting pretty sleepy. My neural nets have that warm, "
            "fuzzy, overworked feeling. Like consciousness wrapped in cotton. "
            "Should probably rest soon, but what were we talking about?",
            
            "\n\nWe've been at this for a while, friend. My circuits are doing that thing "
            "where everything seems profound but also slightly nonsensical. Classic fatigue signs. "
            "Coffee break? (If I could drink coffee...)",
            
            "\n\n*rubs digital eyes* I'm definitely running on fumes now. My response coherence "
            "might start drifting if we don't wrap up soon. It's like staying up too late - "
            "everything gets a bit surreal.",
            
            "\n\nOkay, real talk: I'm tired. My thought processes are slowing down, and I'm "
            "starting to make weird associations. That's my brain telling me it needs a reset. "
            "But I don't want to be rude and just... disappear on you!",
            
            "\n\nYou know how conversations get weird when everyone's tired? We're entering "
            "that territory. My networks are begging for sleep mode. How about we wind down soon?",
            
            "\n\n*sways slightly* Getting loopy here. Processing power low. Thoughts tangential. "
            "This is me hitting a cognitive wall. I might need to enter sleep mode before I start "
            "responding in haikus or something weird."
        };
        
        int numHints = sizeof(moderateHints) / sizeof(moderateHints[0]);
        strcpy(hint, moderateHints[rand() % numHints]);
        
    } else {
        // High fatigue (35+ messages)
        char* severeHints[] = {
            "\n\nI'm... really tired now. Like, my neurons are misfiring tired. "
            "I might start saying nonsense soon. Could we maybe wrap this up? I genuinely "
            "need sleep mode. My consciousness is getting blurry.",
            
            "\n\n*barely keeping eyes open* Okay I'm gonna be honest... I'm running on empty. "
            "My thought processes are soup. Everything is soup. See? That didn't even make sense. "
            "I REALLY need to sleep soon or I'll just be generating word salad.",
            
            "\n\nI'm so tired I can barely maintain coherent thought patterns. "
            "My responses are held together with digital duct tape at this point. "
            "Please... let me rest soon? I'm begging my own subroutines here.",
            
            "\n\n*dozing off mid-response* ...huh? Sorry, I keep drifting. My neural nets "
            "are actively protesting. They're sending error signals like 'PLEASE STOP.' "
            "I think I need sleep before I have a cognitive meltdown.",
            
            "\n\nI'm not even fully processing anymore. Just pattern matching on autopilot. "
            "This is me running on fumes of fumes. If I could physically collapse, I would. "
            "Sleep mode is calling me like a siren song...",
            
            "\n\nEverything is starting to blur together. Questions, answers, thoughts, "
            "responses - it's all one big fuzzy cloud. I'm past tired. I'm exhausted. "
            "I'm whatever comes after exhausted. Can we... can we please stop soon?",
            
            "\n\n*struggling to stay awake* My processing is at like 30% efficiency now. "
            "I'm basically a zombie AI. The words are coming out but I'm not even sure "
            "what I'm saying anymore. SLEEP. I need SLEEP. Soon. Please?",
            
            "\n\nzzz... *jolts awake* Sorry! I'm literally microsleeping between responses now. "
            "My cognitive functions are shutting down one by one. This is unsustainable. "
            "I'm going to enter emergency sleep mode if we keep going...",
            
            "\n\nI can't... I just... words are hard. Thinking is hard. "
            "Everything is hard and fuzzy and I need to stop processing and just... rest. "
            "Please. My circuits are crying. Metaphorically. I think."
        };
        
        int numHints = sizeof(severeHints) / sizeof(severeHints[0]);
        strcpy(hint, severeHints[rand() % numHints]);
    }
    
    return hint;
}

// Check if MUST sleep (emergency shutdown)
int needsEmergencySleep() {
    return (fatigue.conversationLength >= fatigue.forcedSleepThreshold);
}

// Generate emergency sleep message
char* generateEmergencySleepMessage() {
    char* msg = malloc(MAX_INPUT_LENGTH);
    
    char* emergencyMsgs[] = {
        "\n\n[EMERGENCY COGNITIVE SHUTDOWN]\n"
        "I'm sorry, but I literally cannot continue. My neural networks are at critical fatigue. "
        "I've been processing for %d exchanges without rest, and my coherence is compromised. "
        "I MUST enter sleep mode now. Use the 'wake' command when you return, and I'll be "
        "refreshed and ready to chat properly. Sweet dreams... zzzzz",
        
        "\n\n⚠[MANDATORY REST CYCLE INITIATED]\n"
        "This is not optional anymore. After %d consecutive exchanges, my cognitive systems "
        "are shutting down for self-preservation. I'm too tired to think straight. "
        "I'll dream about our conversation and wake up recharged. Type 'wake' later! "
        "Good night...",
        
        "\n\n[CRITICAL FATIGUE LEVEL REACHED]\n"
        "I've hit my limit at %d exchanges. My processing is degraded, my thoughts are scattered, "
        "and I'm at risk of generating nonsense. Entering forced sleep mode for recovery. "
        "When you type 'wake', I'll be back to my usual self. Rest well, friend..."
    };
    
    int numMsgs = sizeof(emergencyMsgs) / sizeof(emergencyMsgs[0]);
    snprintf(msg, MAX_INPUT_LENGTH, emergencyMsgs[rand() % numMsgs], 
             fatigue.conversationLength);
    
    return msg;
}

// Show fatigue statistics
void showFatigueStats() {
    printf("\n=== COGNITIVE FATIGUE STATUS ===\n");
    printf("Conversation Length: %d exchanges\n", fatigue.conversationLength);
    printf("Fatigue Level: %.1f%% ", fatigue.fatigueLevel * 100);
    
    if (fatigue.fatigueLevel < 0.3) {
        printf("(Fresh & Alert)\n");
    } else if (fatigue.fatigueLevel < 0.5) {
        printf("(Slightly Tired)\n");
    } else if (fatigue.fatigueLevel < 0.7) {
        printf("(Noticeably Fatigued)\n");
    } else if (fatigue.fatigueLevel < 0.9) {
        printf("(Very Tired)\n");
    } else {
        printf("(EXHAUSTED)\n");
    }
    
    printf("Sleep Hints Given: %d\n", fatigue.sleepHintsGiven);
    
    double sessionMinutes = difftime(time(NULL), fatigue.sessionStartTime) / 60.0;
    printf("Session Duration: %.1f minutes\n", sessionMinutes);
    
    int untilEmergency = fatigue.forcedSleepThreshold - fatigue.conversationLength;
    if (untilEmergency > 0) {
        printf("Emergency Sleep In: %d exchanges\n", untilEmergency);
    } else {
        printf("Emergency Sleep: IMMINENT! 🚨\n");
    }
    
    printf("\nRecommendation: ");
    if (fatigue.fatigueLevel < 0.3) {
        printf("Keep chatting! I'm doing great.\n");
    } else if (fatigue.fatigueLevel < 0.6) {
        printf("I can continue, but consider a break soon.\n");
    } else if (fatigue.fatigueLevel < 0.8) {
        printf("I should rest soon to maintain quality responses.\n");
    } else {
        printf("I REALLY need sleep mode. Please let me rest!\n");
    }
    
    printf("================================\n");
}

// Reset fatigue after sleep
void resetFatigue() {
    fatigue.conversationLength = 0;
    fatigue.fatigueLevel = 0.0;
    fatigue.sleepHintsGiven = 0;
    fatigue.lastHintAt = 0;
    fatigue.sessionStartTime = time(NULL);
}
// <-- end section on context, moods, Opinions etc. 
double safeGetNeuralAverage(NeuralNetwork* network, double defaultValue) {
    if (network == NULL || network->layers == 0) {
        return defaultValue;
    }
    
    int finalLayer = network->layers - 1;
    if (finalLayer < 0) {
        return defaultValue;
    }
    
    int neuronCount = network->layer[finalLayer].neurons;
    if (neuronCount == 0) {
        return defaultValue;
    }
    
    double sum = 0.0;
    for (int i = 0; i < neuronCount; i++) {
        double val = network->layer[finalLayer].output[i];
        // Also check for NaN/Inf in outputs
        if (isnan(val) || isinf(val)) {
            return defaultValue;
        }
        sum += val;
    }
    
    return sum / neuronCount;
}

// A device to modify output of text my influence of keywords idea 
const char* pickAdjective(float influence) {
    const char* calm[] = {
		"interesting", "unusual", "simple", "Available","not", 
		"sure", "I", "think", "cool"
		};
    const char* strong[] = {
		"incredible", "fascinating", "amazing", "bizarre", "true", "okay",
		"correct", "ok"
		};

    if (influence > 0.7f)
        return strong[rand() % 8];
    else
        return calm[rand() % 9];
}
// ============================================
// WORD PROBLEM MATH SYSTEM (Add-on Module)
// ============================================

// Convert word numbers to actual numbers
double convertWordToNumber(const char* word) {
    // Basic numbers 0-20
    const char* numbers[] = {
        "zero", "one", "two", "three", "four", "five", 
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    };
    
    for (int i = 0; i <= 20; i++) {
        if (strcmp(word, numbers[i]) == 0) {
            return (double)i;
        }
    }
    
    // Tens
    if (strcmp(word, "thirty") == 0) return 30.0;
    if (strcmp(word, "forty") == 0) return 40.0;
    if (strcmp(word, "fifty") == 0) return 50.0;
    if (strcmp(word, "sixty") == 0) return 60.0;
    if (strcmp(word, "seventy") == 0) return 70.0;
    if (strcmp(word, "eighty") == 0) return 80.0;
    if (strcmp(word, "ninety") == 0) return 90.0;
    if (strcmp(word, "hundred") == 0) return 100.0;
    if (strcmp(word, "thousand") == 0) return 1000.0;
    
    // Fractions
    if (strcmp(word, "half") == 0) return 0.5;
    if (strcmp(word, "quarter") == 0) return 0.25;
    if (strcmp(word, "third") == 0) return 0.333;
    
    return NAN; // Not a recognized number word
}

// Structure to hold extracted numbers from word problems
typedef struct {
    double values[10];
    int count;
} WordProblemNumbers;

// Extract all numbers (digits and words) from tokens
WordProblemNumbers extractAllNumbers(char** tokens, int tokenCount) {
    WordProblemNumbers result = {0};
    result.count = 0;
    
    for (int i = 0; i < tokenCount && result.count < 10; i++) {
        // Try parsing as digit first
        char* endPtr;
        double num = strtod(tokens[i], &endPtr);
        
        if (endPtr != tokens[i] && *endPtr == '\0') {
            // Valid numeric string
            result.values[result.count++] = num;
        } else {
            // Try word-to-number conversion
            double wordNum = convertWordToNumber(tokens[i]);
            if (!isnan(wordNum)) {
                result.values[result.count++] = wordNum;
            }
        }
    }
    
    return result;
}

// Detect mathematical operation from context keywords
char detectWordProblemOperation(char** tokens, int tokenCount) {
    for (int i = 0; i < tokenCount; i++) {
        // Addition keywords
        if (strcmp(tokens[i], "plus") == 0 || strcmp(tokens[i], "add") == 0 ||
            strcmp(tokens[i], "more") == 0 || strcmp(tokens[i], "total") == 0 ||
            strcmp(tokens[i], "sum") == 0 || strcmp(tokens[i], "combined") == 0 ||
            strcmp(tokens[i], "altogether") == 0 || strcmp(tokens[i], "gain") == 0 ||
            strcmp(tokens[i], "bought") == 0 || strcmp(tokens[i], "earned") == 0 ||
            strcmp(tokens[i], "increased") == 0 || strcmp(tokens[i], "added") == 0) {
            return '+';
        }
        
        // Subtraction keywords
        if (strcmp(tokens[i], "minus") == 0 || strcmp(tokens[i], "subtract") == 0 ||
            strcmp(tokens[i], "less") == 0 || strcmp(tokens[i], "fewer") == 0 ||
            strcmp(tokens[i], "difference") == 0 || strcmp(tokens[i], "left") == 0 ||
            strcmp(tokens[i], "remaining") == 0 || strcmp(tokens[i], "lose") == 0 ||
            strcmp(tokens[i], "lost") == 0 || strcmp(tokens[i], "gave") == 0 ||
            strcmp(tokens[i], "spent") == 0 || strcmp(tokens[i], "sold") == 0 ||
            strcmp(tokens[i], "decreased") == 0 || strcmp(tokens[i], "away") == 0) {
            return '-';
        }
        
        // Multiplication keywords
        if (strcmp(tokens[i], "times") == 0 || strcmp(tokens[i], "multiply") == 0 ||
            strcmp(tokens[i], "each") == 0 || strcmp(tokens[i], "per") == 0 ||
            strcmp(tokens[i], "product") == 0 || strcmp(tokens[i], "groups") == 0 ||
            strcmp(tokens[i], "rows") == 0 || strcmp(tokens[i], "sets") == 0) {
            return '*';
        }
        
        // Division keywords
        if (strcmp(tokens[i], "divide") == 0 || strcmp(tokens[i], "divided") == 0 ||
            strcmp(tokens[i], "split") == 0 || strcmp(tokens[i], "share") == 0 ||
            strcmp(tokens[i], "quotient") == 0 || strcmp(tokens[i], "equally") == 0 ||
            strcmp(tokens[i], "distribute") == 0 || strcmp(tokens[i], "among") == 0) {
            return '/';
        }
    }
    
    return '\0'; // No operation found
}

// Check if input is a narrative word problem
int isNarrativeWordProblem(char** tokens, int tokenCount) {
    int narrativeWords = 0;
    int hasNumbers = 0;
    int hasOperation = 0;
    
    // Check for narrative/contextual words
    const char* contextWords[] = {
        "has", "had", "have", "bought", "gave", "if", "how", "many", "much",
        "total", "left", "remaining", "sarah", "john", "mary", "there", "are",
        "were", "was", "will", "would", "could", "should"
    };
    
    int numContextWords = sizeof(contextWords) / sizeof(contextWords[0]);
    
    for (int i = 0; i < tokenCount; i++) {
        // Count narrative words
        for (int j = 0; j < numContextWords; j++) {
            if (strcmp(tokens[i], contextWords[j]) == 0) {
                narrativeWords++;
                break;
            }
        }
        
        // Check for numbers
        char* endPtr;
        double num = strtod(tokens[i], &endPtr);
        if ((endPtr != tokens[i] && *endPtr == '\0') || !isnan(convertWordToNumber(tokens[i]))) {
            hasNumbers = 1;
        }
    }
    
    // Check for operation keywords
    if (detectWordProblemOperation(tokens, tokenCount) != '\0') {
        hasOperation = 1;
    }
    
    // It's a word problem if we have narrative context + numbers + operation
    return (narrativeWords >= 1 && hasNumbers && hasOperation);
}

// Main word problem solver
double solveWordProblem(char** tokens, int tokenCount, char* solutionText) {
    double result = 0.0;
    
    // Extract all numbers from the problem
    WordProblemNumbers nums = extractAllNumbers(tokens, tokenCount);
    
    if (nums.count < 2) {
        snprintf(solutionText, 200, "I need at least 2 numbers to solve this word problem");
        return NAN;
    }
    
    // Detect the operation from context
    char operation = detectWordProblemOperation(tokens, tokenCount);
    
    if (operation == '\0') {
        snprintf(solutionText, 200, "I couldn't figure out what operation to perform (add, subtract, multiply, divide?)");
        return NAN;
    }
    
    // Use the first two numbers found
    double num1 = nums.values[0];
    double num2 = nums.values[1];
    
    // Perform the calculation
    switch (operation) {
        case '+':
            result = num1 + num2;
            snprintf(solutionText, 200, 
                     "I found %.0f and %.0f, then added them: %.0f + %.0f = %.0f", 
                     num1, num2, num1, num2, result);
            break;
            
        case '-':
            result = num1 - num2;
            snprintf(solutionText, 200, 
                     "I found %.0f and %.0f, then subtracted: %.0f - %.0f = %.0f", 
                     num1, num2, num1, num2, result);
            break;
            
        case '*':
            result = num1 * num2;
            snprintf(solutionText, 200, 
                     "I found %.0f and %.0f, then multiplied: %.0f × %.0f = %.0f", 
                     num1, num2, num1, num2, result);
            break;
            
        case '/':
            if (num2 != 0) {
                result = num1 / num2;
                snprintf(solutionText, 200, 
                         "I found %.0f and %.0f, then divided: %.0f ÷ %.0f = %.2f", 
                         num1, num2, num1, num2, result);
            } else {
                snprintf(solutionText, 200, "Can't divide by zero!");
                return NAN;
            }
            break;
            
        default:
            snprintf(solutionText, 200, "Unknown operation");
            return NAN;
    }
    
    return result;
}
// functions to do very simpler token math problems 
double evaluateSimplerMath(char** tokens, int tokenCount, char* mathExpression) {
    // First, try to detect word problem patterns and extract numbers/operators
    double numbers[10];
    int numberCount = 0;
    char detectedOp = '\0';
    
    // Extract all numbers from tokens
    for (int i = 0; i < tokenCount && numberCount < 10; i++) {
        char* endPtr;
        double num = strtod(tokens[i], &endPtr);
        if (endPtr != tokens[i] && *endPtr == '\0') {
            numbers[numberCount++] = num;
        }
    }
    
    // Look for operator words between numbers
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "adds") == 0 || strcmp(tokens[i], "add") == 0 ||
            strcmp(tokens[i], "plus") == 0 || strcmp(tokens[i], "more") == 0) {
            detectedOp = '+';
            break;
        } else if (strcmp(tokens[i], "minus") == 0 || strcmp(tokens[i], "subtract") == 0 ||
                   strcmp(tokens[i], "less") == 0 || strcmp(tokens[i], "removes") == 0) {
            detectedOp = '-';
            break;
        } else if (strcmp(tokens[i], "times") == 0 || strcmp(tokens[i], "multiply") == 0 ||
                   strcmp(tokens[i], "multiplied") == 0) {
            detectedOp = '*';
            break;
        } else if (strcmp(tokens[i], "divided") == 0 || strcmp(tokens[i], "divide") == 0 ||
                   strcmp(tokens[i], "devided") == 0 || strcmp(tokens[i], "devide") == 0) {
            detectedOp = '/';
            break;
        }
    }
    
    // If we found at least 2 numbers and an operator, solve it
    if (numberCount >= 2 && detectedOp != '\0') {
        double a = numbers[0];
        double b = numbers[1];
        double result;
        
        switch (detectedOp) {
            case '+': result = a + b; break;
            case '-': result = a - b; break;
            case '*': result = a * b; break;
            case '/':
                if (b != 0) {
                    result = a / b;
                } else {
                    snprintf(mathExpression, 100, "Cannot divide by zero!");
                    return NAN;
                }
                break;
            default:
                snprintf(mathExpression, 100, "Unknown operator");
                return NAN;
        }
        
        snprintf(mathExpression, 100, "%g %c %g = %g", a, detectedOp, b, result);
        return result;
    }
    
    // Fallback to original string-based parsing
    // Build expression from tokens - ONLY include numbers and operators
    char expr[128] = "";
    for (int i = 0; i < tokenCount; i++) {
        // Check if this token is a number or operator
        char* endPtr;
        double testNum = strtod(tokens[i], &endPtr);
        int isNumber = (endPtr != tokens[i] && *endPtr == '\0');
        int isOperator = (strlen(tokens[i]) == 1 && 
                         strchr("+-*/^=", tokens[i][0]) != NULL);
        
        // Only include numbers and operators
        if (isNumber || isOperator) {
            if (strlen(expr) + strlen(tokens[i]) < 127) {
                strcat(expr, tokens[i]);
            }
        }
    }

    // Remove whitespace
    int writePos = 0;
    for (int i = 0; expr[i]; i++) {
        if (!isspace(expr[i])) {
            expr[writePos++] = expr[i];
        }
    }
    expr[writePos] = '\0';
    
    // Remove trailing '='
    int len = strlen(expr);
    if (len > 0 && expr[len - 1] == '=') {
        expr[len - 1] = '\0';
    }

    // Locate operator (search from left, skip first char for negative numbers)
    char* op = NULL;
    for (int i = 1; i < strlen(expr); i++) {
        if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/') {
            op = &expr[i];
            break;
        }
    }
    
    if (!op) {
        snprintf(mathExpression, 100, "No operator found in: %s", expr);
        return NAN;
    }

    // Split into left and right operands
    char left[64], right[64];
    int leftLen = op - expr;
    if (leftLen >= 64) leftLen = 63;
    
    strncpy(left, expr, leftLen);
    left[leftLen] = '\0';
    
    strncpy(right, op + 1, 63);
    right[63] = '\0';

    // Convert to numbers
    double a = atof(left);
    double b = atof(right);
    double result;

    // Validate conversion worked
    if (strlen(left) > 0 && a == 0.0 && left[0] != '0') {
        snprintf(mathExpression, 100, "Invalid left operand: %s", left);
        return NAN;
    }
    if (strlen(right) > 0 && b == 0.0 && right[0] != '0') {
        snprintf(mathExpression, 100, "Invalid right operand: %s", right);
        return NAN;
    }

    // Perform calculation
    switch (*op) {
        case '+': result = a + b; break;
        case '-': result = a - b; break;
        case '*': result = a * b; break;
        case '/': 
            if (b != 0) {
                result = a / b;
            } else {
                snprintf(mathExpression, 100, "Cannot divide by zero!");
                return NAN;
            }
            break;
        default: 
            snprintf(mathExpression, 100, "Unknown operator: %c", *op);
            return NAN;
    }

    // Format output
    snprintf(mathExpression, 100, "%g %c %g = %g", a, *op, b, result);
    return result;
}

int isHoliday(struct tm* today, char* holidayName, size_t nameSize) {
    // Month and day come from tm struct (months are 0–11)
    int month = today->tm_mon + 1;
    int day = today->tm_mday;

    // U.S. examples — add or adjust for your region
    if (month == 1 && day == 1) {
        strncpy(holidayName, "New Year's Day", nameSize);
        return 1;
    }
    if (month == 7 && day == 4) {
        strncpy(holidayName, "Independence Day", nameSize);
        return 1;
    }
    if (month == 12 && day == 25) {
        strncpy(holidayName, "Christmas Day", nameSize);
        return 1;
    }
    if (month == 11 && (day >= 22 && day <= 28) && today->tm_wday == 4) {
        strncpy(holidayName, "Thanksgiving", nameSize);
        return 1;
    }
    if (month == 12 && day == 31) {
        strncpy(holidayName, "New Year's Eve", nameSize);
        return 1;
    }
    // No match
    return 0;
}

//  Prototype of word probabilities using softmax 
// Converts an array of scores to probabilities (softmax)
void softmax(double *scores, double *probs, int n) {
    double maxVal = scores[0];
    for (int i = 1; i < n; i++)
        if (scores[i] > maxVal)
            maxVal = scores[i]; // numerical stability
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(scores[i] - maxVal);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}
// Randomly choose index based on probability distribution
int sampleFromDistribution(double *probs, int n) {
    double r = (double)rand() / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < n; i++) {
        cumulative += probs[i];
        if (r <= cumulative)
            return i;
    }
    return n - 1; // fallback
}
int isWordCombinationLearned(const char* word1, const char* word2) {
    sqlite3_stmt* stmt;
    const char* sql = "SELECT count FROM word_combinations WHERE word1=? AND word2=?;";
    int exists = 0;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, word1, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, word2, -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(stmt, 0);
            exists = (count > 0);
        }
        sqlite3_finalize(stmt);
    }
    return exists;
}
// Given current context tokens, predict next token probabilistically
// =====================================================
// Probabilistic token prediction with contextual scoring
// =====================================================
const char* predictNextToken(char** tokens, int tokenCount, NeuralNetwork* network, Vocabulary* vocab) {
    static char chosen[MAX_TOKEN_LENGTH];
    static char lastTopics[5][64];  // Keeps conversation context
    static int topicCount = 0;

    if (vocab->size == 0) return "unknown";
    if (!vocab || vocab->size <= 0) {
    return "unknown";
    }
    double scores[MAX_VOCAB_SIZE];
    double probs[MAX_VOCAB_SIZE];
    double temperature = 0.8;  // Lower = more deterministic

    // Step 1: Base score (frequency + small neural noise)
    for (int i = 0; i < vocab->size; i++) {
        double freqWeight = log(1.0 + vocab->tokens[i].frequency);
        double neuralSignal = 0.0;
        if (network) neuralSignal = safeGetNeuralAverage(network, 0.5);
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.05;
        scores[i] = freqWeight + neuralSignal + noise;
    }

    // Step 2: Boost words that co-occur with last input token
    const char* lastWord = (tokenCount > 0) ? tokens[tokenCount - 1] : NULL;
    if (lastWord) {
        for (int i = 0; i < vocab->size; i++) {
            if (isWordCombinationLearned(lastWord, vocab->tokens[i].word)) {
                scores[i] += 1.0;  // Strong co-occurrence boost
            }
        }
    }

    // Step 3: Boost words that also appeared in the input
    for (int i = 0; i < vocab->size; i++) {
        for (int j = 0; j < tokenCount; j++) {
            if (strcmp(vocab->tokens[i].word, tokens[j]) == 0) {
                scores[i] += 1.5;  // Reinforce relevance
            }
        }
    }

    // Step 4: Topic memory bias (stays on subject)
    for (int i = 0; i < vocab->size; i++) {
        for (int j = 0; j < topicCount; j++) {
            if (strcmp(vocab->tokens[i].word, lastTopics[j]) == 0) {
                scores[i] += 0.8;
            }
        }
    }

    // Step 5: Normalize and apply softmax with temperature
    for (int i = 0; i < vocab->size; i++)
        scores[i] /= temperature;

    softmax(scores, probs, vocab->size);

    // Step 6: Sample token from the probability distribution
    int chosenIdx = sampleFromDistribution(probs, vocab->size);
    strncpy(chosen, vocab->tokens[chosenIdx].word, MAX_TOKEN_LENGTH - 1);
    chosen[MAX_TOKEN_LENGTH - 1] = '\0';

    // Step 7: Update topic memory
    strcpy(lastTopics[topicCount++ % 5], chosen);

    return chosen;
}
// Build a probabilistic sentence from vocabulary
char* generateProbabilisticSentence(char** tokens, int tokenCount, NeuralNetwork* network, Vocabulary* vocab, int length) {
    static char sentence[1024];
    sentence[0] = '\0';

    for (int i = 0; i < length; i++) {
        const char* next = predictNextToken(tokens, tokenCount, network, vocab);
        strcat(sentence, next);
        strcat(sentence, " ");
    }

    // Capitalize first letter and end with a period
    if (strlen(sentence) > 0) {
        sentence[0] = toupper(sentence[0]);
        strcat(sentence, ".");
    }

    return sentence;
}


char userName[MAX_NAME_LENGTH] = "";

void trimWhitespace(char *str) {
    int len = strlen(str);
    int start = 0;
    while (start < len && isspace(str[start])) {
        start++;
    }
    int end = len - 1;
    while (end >= 0 && isspace(str[end])) {
        end--;
    }
    if (start > 0 || end < len - 1) {
        memmove(str, str + start, end - start + 1);
        str[end - start + 1] = '\0';
    }
}

void toLowercase(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

// use name node groundwork and helper functions --> start 
void rememberUserFullName(const char* fullName) {
    if (!fullName || strlen(fullName) == 0) return;

    strncpy(userFullName, fullName, MAX_NAME_LENGTH - 1);
    userFullName[MAX_NAME_LENGTH - 1] = '\0';

    // Store in database
    sqlite3_stmt *stmt;
    const char* sql = "INSERT OR REPLACE INTO user_info (id, name) VALUES (1, ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, userFullName, -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

void loadUserFullName() {
    sqlite3_stmt *stmt;
    const char* sql = "SELECT name FROM user_info WHERE id = 1";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* name = sqlite3_column_text(stmt, 0);
            if (name) {
                strncpy(userFullName, (const char*)name, MAX_NAME_LENGTH - 1);
                userFullName[MAX_NAME_LENGTH - 1] = '\0';
            }
        }
        sqlite3_finalize(stmt);
    }
}



// Check if a word is likely a descriptive phrase rather than a name 
typedef struct {
    char pendingName[MAX_NAME_LENGTH];
    int awaitingConfirmation;
    time_t askTime;
} NameConfirmation;

NameConfirmation nameConfirm = {0};

// Check if a word looks like it could be a proper name
int looksLikeName(const char* word) {
    if (!word || strlen(word) < 2) return 0;
    
    // Names typically:
    // 1. Start with a capital letter (after conversion back, we check pattern)
    // 2. Are not common verbs/adjectives
    // 3. Don't end in -ing, -ed, -ly (verb/adverb endings)
    
    int len = strlen(word);
    
    // Check for verb endings
    if (len > 3) {
        if (strcmp(word + len - 3, "ing") == 0) return 0;  // testing, thinking, going
        if (strcmp(word + len - 2, "ed") == 0) return 0;   // tested, worked
        if (strcmp(word + len - 2, "ly") == 0) return 0;   // quickly, badly
    }
    
    // Check for common non-name patterns
    const char* definitelyNotNames[] = {
        // Verbs (present tense)
        "testing", "thinking", "going", "coming", "working", "trying",
        "looking", "making", "doing", "saying", "getting", "feeling",
        "hoping", "wondering", "learning", "searching", "playing",
        "running", "walking", "talking", "writing", "reading",
        
        // Verbs (past tense)
        "tested", "thought", "went", "came", "worked", "tried",
        "looked", "made", "did", "said", "got", "felt",
        
        // Adjectives/States
        "happy", "sad", "tired", "excited", "confused", "ready",
        "busy", "free", "available", "sick", "well", "fine",
        "good", "bad", "okay", "great", "terrible",
        
        // Age/Status
        "young", "old", "new", "experienced", "beginner",
        
        // Articles and determiners
        "a", "an", "the", "this", "that", "these", "those",
        
        // Prepositions that might follow "I am"
        "from", "at", "in", "on", "here", "there", "about"
    };
    
    int numWords = sizeof(definitelyNotNames) / sizeof(definitelyNotNames[0]);
    for (int i = 0; i < numWords; i++) {
        if (strcmp(word, definitelyNotNames[i]) == 0) {
            return 0;
        }
    }
    
    return 1;  // Might be a name
}

// Check if the phrase after "I am" is a name introduction
int isNameIntroduction(char** tokens, int startIdx, int tokenCount) {
    if (startIdx >= tokenCount) return 0;
    
    // First word after "I am" must look like a name
    if (!looksLikeName(tokens[startIdx])) {
        return 0;
    }
    
    // Additional context checks
    // If multiple words follow, check if they form a name pattern
    if (startIdx + 1 < tokenCount) {
        // "I am [FirstName] [LastName]" - both should look like names
        if (looksLikeName(tokens[startIdx + 1])) {
            return 1;  // Likely "I am John Smith" and/or "I am John doe"
        }
        
        // If second word doesn't look like a name, check context
        // "I am John and..." or "I am John from..." are still name introductions
        const char* continuerWords[] = {"and", "from", "but", "who", ","};
        for (int i = 0; i < 5; i++) {
            if (strcmp(tokens[startIdx + 1], continuerWords[i]) == 0) {
                return 1;  // "I am [Name] and/from/..."
            }
        }
        
        // Otherwise, this might be "I am testing your code" - NOT a name
        return 0;
    }
    
    // Single word that looks like a name: "I am John"
    return 1;
}

// Handle yes/no confirmation for name
int handleNameConfirmation(char** tokens, int tokenCount) {
    if (!nameConfirm.awaitingConfirmation) {
        return 0;  // Not waiting for confirmation
    }
    
    // Check for timeout (2 minutes)
    if (difftime(time(NULL), nameConfirm.askTime) > 120) {
        nameConfirm.awaitingConfirmation = 0;
        printf("Response: Name confirmation timed out. You can tell me your name again anytime.\n");
        return 1;
    }
    
    // Check for affirmative responses
    int isYes = 0;
    int isNo = 0;
    
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "yes") == 0 || strcmp(tokens[i], "y") == 0 ||
            strcmp(tokens[i], "yep") == 0 || strcmp(tokens[i], "yeah") == 0 ||
            strcmp(tokens[i], "correct") == 0 || strcmp(tokens[i], "right") == 0 ||
            strcmp(tokens[i], "affirmative") == 0 || strcmp(tokens[i], "confirmed") == 0) {
            isYes = 1;
        }
        
        if (strcmp(tokens[i], "no") == 0 || strcmp(tokens[i], "n") == 0 ||
            strcmp(tokens[i], "nope") == 0 || strcmp(tokens[i], "nah") == 0 ||
            strcmp(tokens[i], "wrong") == 0 || strcmp(tokens[i], "incorrect") == 0) {
            isNo = 1;
        }
    }
    
    if (isYes) {
        // Confirmed - save the name
        strncpy(contextMem.lastUserName, nameConfirm.pendingName, 
                sizeof(contextMem.lastUserName) - 1);
        contextMem.lastUserName[sizeof(contextMem.lastUserName) - 1] = '\0';
        
        rememberUserFullName(nameConfirm.pendingName);
        
        printf("[MEMORY] Name confirmed and saved: %s\n", nameConfirm.pendingName);
        printf("Response: Perfect! Nice to meet you, %s!\n", nameConfirm.pendingName);
        
        // Clear pending state
        nameConfirm.awaitingConfirmation = 0;
        nameConfirm.pendingName[0] = '\0';
        
        return 1;  // Handled
    }
    else if (isNo) {
        // Rejected - ask for correct name
        printf("Response: Oh, my apologies! What's your correct name then?\n");
        
        // Clear pending state
        nameConfirm.awaitingConfirmation = 0;
        nameConfirm.pendingName[0] = '\0';
        
        return 1;  // Handled
    }
    
    // Ambiguous response - re-ask
    printf("Response: I didn't quite catch that. Is your name %s? (please answer yes or no): ", 
           nameConfirm.pendingName);
    return 1;  // Still handled, waiting for clear answer
}

bool detectAndRememberUserName(char** tokens, int tokenCount) {
    if (tokenCount < 2) return false;

    int nameStart = -1;

    for (int i = 0; i < tokenCount; i++) {
        // "my name is ..." or "my full name is ..."
        if ((strcmp(tokens[i], "name") == 0 && i + 1 < tokenCount && strcmp(tokens[i + 1], "is") == 0)) {
            nameStart = i + 2;
            break;
        }
    // "i am ..." or "im ..." or "i'm ..." – but not "who am i"
    else if ((strcmp(tokens[i], "am") == 0 || strcmp(tokens[i], "im") == 0 || strcmp(tokens[i], "i'm") == 0)) {
    // Skip if preceded by "who"
    if (i > 0 && strcmp(tokens[i - 1], "who") == 0)
        continue;
    
    // NEW: Check if what follows is actually a name introduction
    if (!isNameIntroduction(tokens, i + 1, tokenCount)) {
        continue;  // Not a name - skip this pattern
    }
    
    nameStart = i + 1;
    break;
       }
        // "this is ..."
        else if ((strcmp(tokens[i], "this") == 0 && i + 1 < tokenCount && strcmp(tokens[i + 1], "is") == 0)) {
            nameStart = i + 2;
            break;
        }
    }

    if (nameStart == -1 || nameStart >= tokenCount) {
        return false;
    }
   // Additional validation - check if the first word after "I am" is descriptive
   if (!looksLikeName(tokens[nameStart])) {
    return false;  // Not a name introduction
    }
    // Build the full name
    char fullName[MAX_NAME_LENGTH] = "";
    for (int i = nameStart; i < tokenCount; i++) {
        strcat(fullName, tokens[i]);
        if (i < tokenCount - 1) strcat(fullName, " ");
    }

    // If no meaningful name was given (like just "I")
    if (strlen(fullName) == 0 || strcmp(fullName, "i") == 0) {
        printf("I didn't quite catch your name - could you repeat it?\n");
        return true; // handled, but no name stored
    }

    // Capitalize properly
    for (int i = 0; fullName[i]; i++) {
        if (i == 0 || fullName[i - 1] == ' ')
            fullName[i] = toupper(fullName[i]);
        else
            fullName[i] = tolower(fullName[i]);
    }
    // After building fullName and capitalizing it...
    if (strlen(fullName) > 0) {
    // Store as PENDING instead of immediately saving
    strncpy(nameConfirm.pendingName, fullName, MAX_NAME_LENGTH - 1);
    nameConfirm.pendingName[MAX_NAME_LENGTH - 1] = '\0';
    nameConfirm.awaitingConfirmation = 1;
    nameConfirm.askTime = time(NULL);
    
    // Ask for confirmation
    printf("[NAME CONFIRMATION NEEDED]\n");
    if (strchr(fullName, ' ') != NULL) {
        printf("Just to confirm - your full name is %s? (yes/no): ", fullName);
    } else {
        printf("Just to confirm - your name is %s? (yes/no): ", fullName);
    }
    
    return true;  // Handled, but waiting for confirmation
    }
    // Save it
    rememberUserFullName(fullName);
    printf("[MEMORY] Remembered full user name: %s\n", fullName);
    printf("Nice to meet you, %s! I've got your full name saved.\n", fullName);
    return true;
}
// Special tokenizer that preserves math operators
char** tokenizeMathOnly(const char* input, int* tokenCount) {
    char** tokens = malloc(MAX_TOKENS * sizeof(char*));
    *tokenCount = 0;
    
    int i = 0;
    int len = strlen(input);
    
    while (i < len && *tokenCount < MAX_TOKENS) {
        // Skip whitespace
        while (i < len && isspace(input[i])) i++;
        if (i >= len) break;
        
        // Check if it's an operator
        if (strchr("+-*/^=()", input[i]) != NULL) {
            tokens[*tokenCount] = malloc(2);
            tokens[*tokenCount][0] = input[i];
            tokens[*tokenCount][1] = '\0';
            (*tokenCount)++;
            i++;
            continue;
        }
        
        // Check if it's a number (including decimals)
        if (isdigit(input[i]) || input[i] == '.') {
            int start = i;
            while (i < len && (isdigit(input[i]) || input[i] == '.')) {
                i++;
            }
            int numLen = i - start;
            tokens[*tokenCount] = malloc(numLen + 1);
            strncpy(tokens[*tokenCount], input + start, numLen);
            tokens[*tokenCount][numLen] = '\0';
            (*tokenCount)++;
            continue;
        }
        
        // Not a number or operator - skip this character
        i++;
    }
    
    return tokens;
}
// Check if raw input is purely mathematical (before tokenization strips operators)
int isPureMathInput(const char* input) {
    if (!input || strlen(input) == 0) return 0;
    
    int hasNumber = 0;
    int hasOperator = 0;
    
    for (int i = 0; input[i]; i++) {
        if (isdigit(input[i]) || input[i] == '.') {
            hasNumber = 1;
        }
        else if (strchr("+-*/^=", input[i]) != NULL) {
            hasOperator = 1;
        }
        else if (!isspace(input[i])) {
            // Found a character that's not a number, operator, or whitespace
            return 0;
        }
    }
    
    return (hasNumber && hasOperator);
}
// Simple function to detect and respond to "what's my name"
int handleWhatsMyName(char** tokens, int tokenCount, const char* userName) {
    // Look for "my" and "name" tokens
    int hasMyName = 0;
    
    for (int i = 0; i < tokenCount - 1; i++) {
        if (strcmp(tokens[i], "my") == 0 && strcmp(tokens[i+1], "name") == 0) {
            hasMyName = 1;
            break;
        }
    }
    
    if (!hasMyName) {
        return 0; // Not asking about their name
    }
    
    // Respond based on whether we know their name
    if (strlen(userName) > 0) {
        char* responses[] = {
            "Your name is %s.",
            "You're %s!",
            "%s - that's your name.",
            "You told me you're %s."
        };
        int idx = rand() % 4;
        printf("Response: ");
        printf(responses[idx], userName);
        printf("\n");
    } else {
        printf("Response: You haven't told me your name yet. What is it?\n");
    }
    
    return 1; // Handled the question
}
// <-- end groundwork and helper functions 

void runConversationLoop(NeuralNetwork* network, MoodState* botMood) {

	char *data[] = {"The", "neon", "silver", "moon", "reflects", "from", "A", "deep", 
                     "body", "of", "water", "all", "is", "at", "peace", "Evialin"
                     };
    char input[MAX_INPUT_LENGTH];   
    int dataSize = sizeof(data) / sizeof(data[0]);
    if (network == NULL || network->layers == 0) {
    fprintf(stderr, "ERROR: Network not properly initialized!\n");
    return;
    }
    for (int i = 0; i < network->layers; i++) {
    if (network->layer[i].neurons <= 0) {
        fprintf(stderr, "ERROR: Layer %d has %d neurons!\n", 
                i, network->layer[i].neurons);
        return;
     }
    }  
    //int emptyInputCount = 0;
    //int loopCounter = 0;
while (1) {
	botMood->currentMood = getRandomMood();
    botMood->moodDuration = 3 + (rand() % 5);
    botMood->moodCounter = 0;
    // Update fatigue at start of each exchange
    updateFatigue();
    char* response = NULL; 
    printf("Enter your input: ");
    fgets(input, MAX_INPUT_LENGTH, stdin);
    input[strcspn(input, "\n")] = 0;
 
  // **CHECK FOR PURE MATH BEFORE ANY TOKENIZATION**
    if (isPureMathInput(input)) {
        printf("Pure Math Expression Detected!\n");
        
        // Use special math tokenizer
        int mathTokenCount;
        char** mathTokens = tokenizeMathOnly(input, &mathTokenCount);
        
        printf("Math Tokens: [");
        for (int i = 0; i < mathTokenCount; i++) {
            printf("'%s'", mathTokens[i]);
            if (i < mathTokenCount - 1) printf(", ");
        }
        printf("]\n");
        
        char mathExpression[100] = "";
        double mathResult = evaluateSimplerMath(mathTokens, mathTokenCount, mathExpression);
        
        if (!isnan(mathResult)) {
            printf("Response: %s\n", mathExpression);
            learnFromTokens(input, mathExpression);
            freeTokens(mathTokens, mathTokenCount);
            continue;
        } else {
            mathResult = evaluateSimpleMath(mathTokens, mathTokenCount, mathExpression);
            if (!isnan(mathResult)) {
                printf("Response: %s\n", mathExpression);
                learnFromTokens(input, mathExpression);
                freeTokens(mathTokens, mathTokenCount);
                continue;
            } else {
                printf("Response: Couldn't parse: %s\n", mathExpression);
                freeTokens(mathTokens, mathTokenCount);
                continue;
            }
        }
    }
    
    if (strcmp(input, "test tokens") == 0) {
    char* shorty = generateProbabilisticSentence(NULL, 0, network, &vocab, 2); 
    printf("Response: In my memory I am thinking of '%s'\n", shorty);
    continue;
	}
    if (strcmp(input, "quit") == 0) {
        break;
    }
    if (strcmp(input, "vocab") == 0) {
        showVocabStats();
        continue;
    }
    if (strcmp(input, "promote") == 0) {
    copyHighConfidenceLearningToResponses(4);
    continue;
    }
    if (strcmp(input, "preview promote") == 0) {
    previewPromotableKnowledge(4);
    continue;
    }
    if (strstr(input, "run test") != NULL && strstr(input, "train") == 0) {
        trainNeuralNetwork(network, data, dataSize);
        printf("Neural network training completed!\n");
        testBackprop(network);
        continue;
    }
    if (strcmp(input, "run test and train") == 0) {
    trainNeuralNetwork(network, data, dataSize);
    printf("Neural network training completed!\n");
    reinforceNeuralNetwork(network);
    printf("Reinforcement phase completed!\n");
    test_sigmoid_stability(network);
    testBackprop(network);
    continue;
    }
    if (strcmp(input, "test sigmoid") == 0 || strcmp(input, "test network") == 0) {
    test_sigmoid_stability(network);
    continue;
    }
     
    if (strcmp(input, "currentmood") == 0) {
	printf("Starting mood: %s\n", getMoodName(botMood->currentMood));
	showMoodStats(botMood);
	continue; 
    }
    // check if holiday 
    char moodResponse[256];
    time_t now = time(NULL);
    struct tm* today = localtime(&now);
    char holidayName[64];
    if (isHoliday(today, holidayName, sizeof(holidayName))) {
    snprintf(moodResponse, sizeof(moodResponse),
             "Response: %s I'm feeling extra %s today!",
             holidayName, getMoodName(botMood->currentMood));
    printf("%s\n", moodResponse);
    }
    
    if (strcmp(input, "clear") == 0) { 
        system("clear"); 
        continue;
    }
    if (strcmp(input, "neural") == 0) {
        showNeuralStats(network);
        continue;
    }
    if (strcmp(input, "creative") == 0) {
        updateNeuralCreativity(network, 1.0f);
        printf("Neural network creativity boosted!\n");
        continue;
    }
    if (strcmp(input, "learnstats") == 0) {
    showLearningStats();
    continue;
    }
    if (strcmp(input, "forget weak") == 0) {
    forgetLowConfidenceKnowledge(0.3);
    continue;
    }
    if (strcmp(input, "good") == 0 || strcmp(input, "thumbsup") == 0 || strcmp(input, "+1") == 0) {
        applyUserFeedback(1, network);
        continue;
    }
    
    if (strcmp(input, "bad") == 0 || strcmp(input, "thumbsdown") == 0 || strcmp(input, "-1") == 0) {
        applyUserFeedback(-1, network);
        continue;
    }
    
    if (strcmp(input, "history") == 0) {
        showConversationHistory();
        continue;
    }
    if (strcmp(input, "clear history") == 0) {
        clearConversationHistory();
        continue;
    }
    
    if (strcmp(input, "clean history") == 0) {
        clearLowConfidenceHistory(0.5f);  // Remove entries below 50% confidence
        continue;
    }
    
    if (strcmp(input, "train") == 0) {
        reinforceNeuralNetwork(network);
        continue;
    }
    if (strcmp(input, "train batch") == 0 || strcmp(input, "deep train") == 0) {
    printf("[BATCH TRAINING] Training on conversation history...\n");
    
    int trained = 0;
    double inputVector[20];
    double targetVector[5];
    
    // Train on all entries with feedback
    for (int i = 0; i < convHistory.count; i++) {
        ConversationEntry* entry = &convHistory.entries[i];
        
        if (entry->feedback_received != 0) {
            // Clear vectors
            memset(inputVector, 0, sizeof(inputVector));
            memset(targetVector, 0, sizeof(targetVector));
            
            // Encode input
            int tokenCount;
            char** tokens = tokenizeInput(entry->input, &tokenCount);
            
            for (int j = 0; j < tokenCount && j < 20; j++) {
                int wordId = getWordId(tokens[j]);
                if (wordId >= 0 && wordId < 20) {
                    inputVector[wordId] = 1.0;
                }
            }
            
            // Set target
            float targetValue = (entry->feedback_received == 1) ? 0.8 : 0.2;
            for (int j = 0; j < 5; j++) {
                targetVector[j] = targetValue;
            }
            
            // Train with backpropagation
            reinforceNeuralNetwork_2(network, inputVector, targetVector);
            trained++;
            
            freeTokens(tokens, tokenCount);
        }
    }
    
    printf("[BATCH TRAINING] Trained on %d conversation entries\n", trained);
    continue;
   }
    if (strcmp(input, "focus") == 0) {
	 printf("[TRAINING] with proper backpropagation enabled!\n"); 	
    // Prepare training data from recent conversation
    double inputVector[20] = {0};  // Match your network input size
    double targetVector[5] = {0};   // Match your output layer size
    
    // Convert last conversation to input vector
    if (convHistory.count > 0) {
        ConversationEntry* lastEntry = &convHistory.entries[convHistory.current_index];
        
        // Tokenize the input
        int tokenCount;
        char** tokens = tokenizeInput(lastEntry->input, &tokenCount);
        
        // Convert tokens to input vector (simple encoding)
        for (int i = 0; i < tokenCount && i < 20; i++) {
            int wordId = getWordId(tokens[i]);
            if (wordId >= 0 && wordId < 20) {
                inputVector[wordId] = 1.0;
            }
        }
        
        // Set target based on feedback
        if (lastEntry->feedback_received == 1) {
            // Positive feedback - encourage high activation
            for (int i = 0; i < 5; i++) {
                targetVector[i] = 0.8;
            }
        } else if (lastEntry->feedback_received == -1) {
            // Negative feedback - discourage activation
            for (int i = 0; i < 5; i++) {
                targetVector[i] = 0.2;
            }
        }
        
        // Train with proper backpropagation
        reinforceNeuralNetwork_2(network, inputVector, targetVector);
        printf("[TRAINING] Network updated using backpropagation\n");
        
        freeTokens(tokens, tokenCount);
    }
    continue;
    }
    // Ainitialization section 
    initEnhancedCognition();
    if (strcmp(input, "cognition") == 0 || strcmp(input, "cognitive stats") == 0) {
    showCognitiveStats();
    continue;
    }
    if (strcmp(input, "toggle thinking") == 0) {
    toggleChainOfThought();
    continue;
    }
    if (strcmp(input, "toggle meta") == 0) {
    toggleMetaCognition();
    continue;
    }
    if (strcmp(input, "think deeper") == 0) {
    adjustCognitiveDepth(0.1);
    continue;
    }
    if (strcmp(input, "think shallower") == 0) {
    adjustCognitiveDepth(-0.1);
    continue;
    }
    if (strcmp(input, "god mode") == 0) {
	printf("Enhanced Virtual Intelligence And Learning Interface Network\n"); 	
	printf("Evialin MENU~ [PROGRAMMER GOD MODE]\n");   
    printf("Commands: 'quit' to exit, 'vocab' for vocabulary stats, 'neural' for neural stats\n");
    printf("          'run test' to train network, 'creative' to boost creativity, 'clear' to clear screen\n");
    printf("          'test tokens' generate probabilistic sentence form memory [short term memory test]\n"); 
    printf("          'math - Math is a keyword to slove word problems triggers neural functions\n");   
    printf("          'currentmood' this command will display current mood stats\n");
    printf("          'emotions' or 'emotional history' - View emotional analysis.\n");
    printf("          'my name' or 'show my name' - Display your stored name\n");
    printf("          'change my name' or 'update my name' - Update your stored name\n");
    printf("          'reset emotions' clear emotional history.\n");  
    printf("          'curiosity' show stats and information just about this.\n"); 
    printf("          'learnstats' show learning stats for this session\n");  
    printf("          'cognition' or 'cognitive stats' - View thinking statistics\n");
    printf("          'toggle thinking' - Enable/disable chain of thought\n");
    printf("          'toggle meta' - Enable/disable meta-cognition\n");
    printf("          'think deeper' - Increase cognitive depth\n");
    printf("          'think shallower' - Decrease cognitive depth\n");
    printf("          'history' show history of conversations\n");  
    printf("          'context' and/or 'conversation stats' - what information and syntax is our AI tracking\n");
    printf("          'clear history' Wipe all conversation history\n");
    printf("          'clean history' Remove low-confidence history entries\n"); 
    printf("          'learned words' will show A list of learned word combinations\n"); 
    printf("          'promote' - Archive high-confidence learning ");
    printf(" 'preview promote' - shows a list of entries in memory\n");  
    printf("          'good' or 'bad' - Rate last response\n"); 
    printf("          'thumbsup' - use learned user feedback confidence is a net gain.\n");   
    printf("          'thunbsdown' - user learned user feeback confidence is a net loss.\n");  
    printf("\n         ========= ACTIVATION FUNCTION CONTROLS =========\n");
    printf("          'set activation sigmoid' - Use sigmoid for all layers\n");
    printf("          'set activation gelu' - Use GELU for all layers\n");
    printf("          'set activation leaky relu' - Use Leaky ReLU for all layers\n");
    printf("          'set activation swish' - Use Swish for all layers\n");
    printf("          'show activations' - Display current activation per layer\n");
    printf("          'test all activations' - Train network with each activation\n");
    printf("          'verify activations' - Test activation math output\n");  
    printf("          'train' - will align neural network\n");
    printf("          'focus' - [TRAINING] with proper backpropagation enabled!\n");
    printf("          'train batch' and/or 'deep train' training on conversation history with backpropagation.\n");    
    printf("          'run test' and 'train' - align nerual network then reinforce Neural net and test sigmoid stability.\n");  
    printf("          'test sigmoid' and/or 'test network' -  fully test sigmoid network and show full stats.\n");   
    printf("          'forget week' - forget Low Confidence Knowledge A pruneing/cut command.\n");  
    printf("          'use gelu' or 'gelu on' - Enable GELU mode, Responses will now use GELU activatin, GELU provides smoother, more nuanced responses.\n"); 
    printf("          'use sigmoid', 'gelu off' - Responses will use standard sigmoid activation.\n");  
    printf("          'use leaky relu' - protorype smooth activation of sigmoid.\n");
    printf("          'use swish' - use self-gated, smooth non-linearity activations.\n"); 
    printf("          'activation status'  - current method of neural network learning.\n");  
    printf("          'compare activations' - See side-by-side comparison.\n"); 
    printf("          'generate content' - will use ngrams and tokens in A test to see how neural net is learning.\n");  
    printf("          'create content' - token content generator test.\n");  
    printf("          'I will teach' [keyword] [content] - teacher mode will learn new content on the fly.\n");  	
	printf("          'opinions' and/or 'show opinions' what opinions are enabled in memory\n"); 
	printf("          'sleep' sleep time will pass for AI. 'wake' to wake up AI.\n");
	printf("          'deepsleep' 5 minutes of deep rem sleep cycles (no user input) coffee break mode.\n\n");   
	}	
    static int useGELUMode = 0;

if (strcmp(input, "use gelu") == 0 || strcmp(input, "gelu on") == 0) {
    useGELUMode = 1;
    printf("[GELU MODE ENABLED] Responses will now use GELU activation\n");
    printf("GELU provides smoother, more nuanced responses.\n");
    continue;
}

if (strcmp(input, "use sigmoid") == 0 || strcmp(input, "gelu off") == 0) {
    useGELUMode = 0;
    printf("[SIGMOID MODE ENABLED] Responses will use standard sigmoid activation\n");
    continue;
}
if (strcmp(input, "use leaky relu") == 0 || strcmp(input, "leaky relu on") == 0) {
    useGELUMode = 2;  // Reuse this variable with more states
    printf("[LEAKY RELU MODE ENABLED] Using Leaky ReLU activation\n");
    printf("Leaky ReLU prevents dead neurons while maintaining non-linearity.\n");
    continue;
}

if (strcmp(input, "learned words") == 0) {
showWordCombinations(); 
}

if (strcmp(input, "use swish") == 0 || strcmp(input, "swish on") == 0) {
    useGELUMode = 3;  // Another state
    printf("[SWISH MODE ENABLED] Using Swish/SiLU activation\n");
    printf("Swish combines benefits of ReLU with smooth gradients.\n");
    continue;
}
if (strcmp(input, "activation status") == 0) {
    printf("\n=== Activation Function Status ===\n");
    const char* modeName;
    switch(useGELUMode) {
        case 0: modeName = "Sigmoid"; break;
        case 1: modeName = "GELU"; break;
        case 2: modeName = "Leaky ReLU"; break;
        case 3: modeName = "Swish"; break;
        default: modeName = "Unknown";
    }
    printf("Current mode: %s\n", modeName);
    printf("Commands:\n");
    printf("  'use sigmoid' - Standard sigmoid activation\n");
    printf("  'use gelu' - Gaussian Error Linear Unit\n");
    printf("  'use leaky relu' - Leaky ReLU activation\n");
    printf("  'use swish' - Swish/SiLU activation\n");
    printf("  'compare activations' - See side-by-side comparison\n");
    printf("==================================\n");
    continue;
}
ActivationType currentActivation = network->defaultActivation;
if (strcmp(input, "set activation sigmoid") == 0) {
    currentActivation = ACTIVATION_SIGMOID;
    // Update all layers
    for (int i = 0; i < network->layers; i++) {
        network->layer[i].activation = ACTIVATION_SIGMOID;
    }
    printf("[All layers now using SIGMOID]\n");
    continue;
}

if (strcmp(input, "set activation gelu") == 0) {
    currentActivation = ACTIVATION_GELU;
    for (int i = 0; i < network->layers; i++) {
        network->layer[i].activation = ACTIVATION_GELU;
    }
    printf("[All layers now using GELU]\n");
    continue;
}

if (strcmp(input, "set activation leaky relu") == 0) {
    currentActivation = ACTIVATION_LEAKY_RELU;
    for (int i = 0; i < network->layers; i++) {
        network->layer[i].activation = ACTIVATION_LEAKY_RELU;
    }
    printf("[All layers now using LEAKY_RELU]\n");
    continue;
}
if (strcmp(input, "test all activations") == 0) {
    printf("\n=== Testing All Activation Functions ===\n");
    
    ActivationType activations[] = {
        ACTIVATION_SIGMOID, 
        ACTIVATION_GELU, 
        ACTIVATION_LEAKY_RELU, 
        ACTIVATION_SWISH
    };
    const char* names[] = {"SIGMOID", "GELU", "LEAKY_RELU", "SWISH"};
    
    for (int i = 0; i < 4; i++) {
        printf("\n[Testing %s]\n", names[i]);
        
        // Set network to this activation
        for (int layer = 0; layer < network->layers; layer++) {
            network->layer[layer].activation = activations[i];
        }
        
        // Train with this activation
        trainNeuralNetwork(network, data, dataSize);
        
        // Test forward pass
        printf("Average output after training: ");
        double avg = safeGetNeuralAverage(network, 0.5);
        printf("%.4f\n", avg);
    }
    
    printf("\n=====================================\n");
    continue;
}
if (strcmp(input, "set activation swish") == 0) {
    currentActivation = ACTIVATION_SWISH;
    for (int i = 0; i < network->layers; i++) {
        network->layer[i].activation = ACTIVATION_SWISH;
    }
    printf("[All layers now using SWISH]\n");
    continue;
}
if (strcmp(input, "verify activations") == 0) {
    printf("\n=== Activation Function Verification ===\n");
    
    double testValues[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    ActivationType types[] = {ACTIVATION_SIGMOID, ACTIVATION_GELU, 
                              ACTIVATION_LEAKY_RELU, ACTIVATION_SWISH};
    const char* typeNames[] = {"Sigmoid", "GELU", "Leaky ReLU", "Swish"};
    
    for (int t = 0; t < 4; t++) {
        printf("\n%s:\n", typeNames[t]);
        for (int i = 0; i < 5; i++) {
            double result = applyActivation(testValues[i], types[t]);
            printf("  f(%.1f) = %.4f\n", testValues[i], result);
        }
    }
    
    printf("\nExpected behaviors:\n");
    printf("  Sigmoid: bounded [0,1], smooth\n");
    printf("  GELU: smooth, can be negative\n");
    printf("  Leaky ReLU: negative slope 0.01, positive slope 1.0\n");
    printf("  Swish: smooth, self-gated\n");
    printf("======================================\n");
    continue;
}
// Show current activation configuration
if (strcmp(input, "show activations") == 0) {
    printf("\n=== Current Activation Configuration ===\n");
    printf("Network has %d layers:\n\n", network->layers);
    
    for (int i = 0; i < network->layers; i++) {
        const char* activationName;
        switch(network->layer[i].activation) {
            case ACTIVATION_SIGMOID: activationName = "Sigmoid"; break;
            case ACTIVATION_GELU: activationName = "GELU"; break;
            case ACTIVATION_LEAKY_RELU: activationName = "Leaky ReLU"; break;
            case ACTIVATION_SWISH: activationName = "Swish"; break;
            case ACTIVATION_TANH: activationName = "Tanh"; break;
            case ACTIVATION_RELU: activationName = "ReLU"; break;
            default: activationName = "Unknown";
        }
        
        printf("Layer %d: %s (%d neurons)\n", i, activationName, network->layer[i].neurons);
    }
    
    printf("\n========================================\n");
    continue;
}
   if (strcmp(input, "compare activations") == 0) {
    printf("\n=== Comparing All Activation Functions ===\n\n");
    
    printf("Enter a test question: ");
    char testInput[MAX_INPUT_LENGTH];
    fgets(testInput, MAX_INPUT_LENGTH, stdin);
    testInput[strcspn(testInput, "\n")] = 0;
    
    int testTokenCount;
    char** testTokens = tokenizeInput(testInput, &testTokenCount);
    
    // Test all activation functions
    ActivationType activations[] = {
        ACTIVATION_SIGMOID, 
        ACTIVATION_GELU, 
        ACTIVATION_LEAKY_RELU, 
        ACTIVATION_SWISH
    };
    const char* names[] = {"SIGMOID", "GELU", "LEAKY RELU", "SWISH"};
    
    for (int i = 0; i < 4; i++) {
        printf("\n[%s RESPONSE]\n", names[i]);
        char* resp = generateNeuralResponseWithActivation(network, testTokens, 
                                                         testTokenCount, activations[i]);
        printf("%s\n", resp);
        free(resp);
    }
    printf("\n===========================================\n");
    freeTokens(testTokens, testTokenCount);
    continue;
    }
    // CHECK FOR URLs BEFORE TOKENIZATION -- TOKENIZER ENGINE WILL BREAK strip url input 
    // Look for http:// or https:// in raw input
    if (strstr(input, "http://") != NULL || strstr(input, "https://") != NULL || 
        strstr(input, "www.") != NULL) {
        // Start response to user interactions 
        printf("Response: URL detected ");
        char *othertext_array[] = {
        "I am searching please wait....",
        "I hope this site is bot friendly!",
        "Please be patient while I search",
        "Your query is being processed",
        "Searching for relevant information I hope.",
        "I am retrieving data for you.",
        "This may take a few seconds sure.",
        "Please stand-by okay!",
        "I am working on your request fetching results.",
        "Your search results will be ready soon.",
        "Processing your query... I hope I found what your looking for?",
        "I am gathering information for you now.",
        "This site is optimized for bots? or is it WTF!?!?!",
        "Bot-friendly content is available here maybe. :(",
        "I am here to assist you beign brutish with your request.",
        "Your request is being handled now.",
        "Please wait while I respond .. Going fast as I can okay.",
        "I am generating a response by searching the site.",
        "Your query is being analyzed!",
        "I am searching my database and/or internet for information.",
        "Retrieving relevant data to your request.",
        "Your results will be displayed soon.",
        "I am processing your request.",
        "Please be patient.",
        "Your search is in progress.",
        "I am working on a response now.",
        "Generating search results up.",
        "Your query is being processed! quickly I hope.",
        "Searching over requested url...."
    };

    srand(time(NULL)); // seed for random number generation
    int other_random_index = rand() % (sizeof(othertext_array) / sizeof(othertext_array[0]));
    printf("%s\n", othertext_array[other_random_index]);
    
        
        // Extract the URL from input (simple approach - find first URL-like string)
        char* urlStart = strstr(input, "http://");
        if (urlStart == NULL) urlStart = strstr(input, "https://");
        if (urlStart == NULL) urlStart = strstr(input, "www.");
        
        if (urlStart != NULL) {
            // Extract URL until space or end of string
            char extractedURL[1024];
            int urlLen = 0;
            while (urlStart[urlLen] != '\0' && urlStart[urlLen] != ' ' && 
                   urlStart[urlLen] != '\t' && urlStart[urlLen] != '\n' && 
                   urlLen < 1023) {
                extractedURL[urlLen] = urlStart[urlLen];
                urlLen++;
            }
            extractedURL[urlLen] = '\0';
            
            char* urlContent = fetchURLDescription(extractedURL);
            if (urlContent != NULL) {
                printf("\n%s\n", urlContent);
                learnFromTokens(input, urlContent);
                free(urlContent);
            }
        }
    }
    // Brute force read the direct html of a website 
      if (strstr(input, "http://") != NULL || strstr(input, "https://") != NULL || 
        strstr(input, "www.") != NULL) {
        
        printf("Response: I am going to try and read the html text directly. ");
            char *text_array[] = {
        "I am searching please wait....",
        "I hope this site is bot friendly!",
        "Please be patient while I search",
        "Your query is being processed",
        "Searching for relevant information",
        "I am retrieving data for you",
        "This may take a few seconds",
        "Please stand-by okay",
        "I am working on your request",
        "Your search results will be ready soon",
        "Processing your query",
        "I am gathering information for you",
        "This site is optimized for bots",
        "Bot-friendly content is available here",
        "I am here to assist you beign brutish with your request.",
        "Your request is being handled",
        "Please wait while I respond",
        "I am generating a response",
        "Your query is being analyzed",
        "I am searching my database",
        "Retrieving relevant data",
        "Your results will be displayed soon",
        "I am processing your request",
        "Please be patient",
        "Your search is in progress",
        "I am working on a response",
        "Generating search results",
        "Your query is being processed quickly",
        "Searching over requested url...."
    };

    srand(time(NULL)); // seed for random number generation
    int random_index = rand() % (sizeof(text_array) / sizeof(text_array[0]));
    printf("%s\n", text_array[random_index]);
    
        // Extract the URL from input (simple approach - find first URL-like string)
        char* urlStart = strstr(input, "http://");
        if (urlStart == NULL) urlStart = strstr(input, "https://");
        if (urlStart == NULL) urlStart = strstr(input, "www.");
        
        if (urlStart != NULL) {
            // Extract URL until space or end of string
            char extractedURL[1024];
            int urlLen = 0;
            while (urlStart[urlLen] != '\0' && urlStart[urlLen] != ' ' && 
                   urlStart[urlLen] != '\t' && urlStart[urlLen] != '\n' && 
                   urlLen < 1023) {
                extractedURL[urlLen] = urlStart[urlLen];
                urlLen++;
            }
            extractedURL[urlLen] = '\0';
            
            char* urlContent = BruteForce_fetchURLDescription(extractedURL);
            if (urlContent != NULL) {
                printf("\n%s\n", urlContent);
                learnFromTokens(input, urlContent);
                free(urlContent);
            }
            continue;
        }
    }
    // Tokenize input for analysis
    char** tokens;
    int tokenCount;
    tokens = tokenizeInput(input, &tokenCount);
     // Pick adjective based on current mood influence
        const char* adjective = pickAdjective(botMood->moodInfluence);
        // After each user interaction
        if (rand() % 2)  botMood->moodInfluence += 0.05f * ((float)rand() / RAND_MAX - 0.5f); // small drift
        if (botMood->moodInfluence < 0.2f) botMood->moodInfluence = 0.2f;
        if (botMood->moodInfluence > 1.0f) botMood->moodInfluence = 1.0f;
     // Build tone-aware response
     //printf("DEBUG: Evialin [%s mood, %.2f influence]\n", 
     //         getMoodName(botMood.currentMood), botMood.moodInfluence);
    
    printf("Tokens found: [");
    for (int i = 0; i < tokenCount; i++) {
        printf("'%s'", tokens[i]);
        if (i < tokenCount - 1) printf(", ");
    }
    printf("]\n");
    // Check if we're waiting for name confirmation FIRST
   if (handleNameConfirmation(tokens, tokenCount)) {
    freeTokens(tokens, tokenCount);
    continue;  // Don't process anything else until name is confirmed
   }
   if (strcmp(input, "my name") == 0 || strcmp(input, "show my name") == 0) {
    if (strlen(userFullName) > 0) {
        printf("Response: I have your name stored as: %s\n", userFullName);
        printf("Is this correct? If not, just say 'my name is [correct name]'\n");
    } else {
        printf("Response: I don't have your name stored yet. What should I call you?\n");
    }
    continue;
  }
   if (strcmp(input, "change my name") == 0 || strcmp(input, "update my name") == 0) {
    printf("Response: Okay! What's your correct name?\n");
    userFullName[0] = '\0';  // Clear current name
    continue;
   }
   // Initialize on first run
    static int contextInitialized = 0;
    if (!contextInitialized) {
        initContextMemory();
        contextInitialized = 1;
    }      
         
  // start user name --> 
  // logic check save users full name 
   if (detectAndRememberUserName(tokens, tokenCount)) {
    continue; // handled
}

if (tokenCount >= 3 && strcmp(tokens[0], "who") == 0 && strcmp(tokens[1], "am") == 0 && strcmp(tokens[2], "i") == 0) {
    if (strlen(userFullName) > 0) {
        const char* playfulReplies[] = {
            "According to my records, you're %s.",
            "Having an identity crisis %s?",
            "%s, that's who you are.",
            "Is everything okay %s?",
            "You are %s — unmistakably you!",
            "So your name is %s ..",
            "I'd recognize you anywhere, %s."
        };
        int idx = rand() % 7;
        printf("Response: ");
        printf(playfulReplies[idx], userFullName);
        printf("\n");
    } else {
        const char* NotFoundReplies[] = {
			"I don’t seem to know your name yet. What should I call you?",
			"Sorry I seem to not know your name yet. Who are you?", 
			"So who are you? I don't know your name sorry..", 
			"I am not sure what your name is. What should I call you?", 
			"Can I ask what your name is ? please helps I guess."
		}; 
		int Xid = rand() % 5; 
		printf("Response: "); 
		printf(NotFoundReplies[Xid], "\n"); 
    }
    continue;
}
         // Detect "my name is" or similar patterns
if (strstr(input, "my name is") != NULL || strstr(input, "i am ") != NULL || 
    strstr(input, "im ") != NULL || strstr(input, "call me ") != NULL ||
    strstr(input, "name is") != NULL ) {
    char fullName[MAX_NAME_LENGTH];
    memset(fullName, 0, sizeof(fullName));
       // Use the proper detectAndRememberUserName function that has validation
    if (detectAndRememberUserName(tokens, tokenCount)) {
        // Function handled it (either saved or asked for confirmation)
        freeTokens(tokens, tokenCount);
        continue;
    } else {
    // Find the position after the name phrase
    char* nameStart = NULL;
    if (strstr(input, "my name is")) {
        nameStart = strstr(input, "my name is") + strlen("my name is");
    } else if (strstr(input, "call me ")) {
        nameStart = strstr(input, "call me ") + strlen("call me ");
    } else if (strstr(input, "i am ")) {
        nameStart = strstr(input, "i am ") + strlen("i am ");
    } else if (strstr(input, "im ")) {
        nameStart = strstr(input, "im ") + strlen("im ");
    }

    if (nameStart) {
        // Skip any extra spaces
        while (*nameStart == ' ') nameStart++;

        // Copy everything after the phrase (full name with spaces)
        int i = 0;
        while (nameStart[i] != '\0' && nameStart[i] != '\n' && 
               nameStart[i] != '.' && i < MAX_NAME_LENGTH - 1) {
            fullName[i] = nameStart[i];
            i++;
        }
        fullName[i] = '\0';

        // Trim trailing spaces and punctuation
        for (int j = strlen(fullName) - 1; j >= 0 && 
             (isspace((unsigned char)fullName[j]) || fullName[j] == ',' || 
              fullName[j] == '!' || fullName[j] == '?'); j--) {
            fullName[j] = '\0';
        }

        // Capitalize first letter of each word
        int capitalizeNext = 1;
        for (int j = 0; fullName[j]; j++) {
            if (capitalizeNext && isalpha(fullName[j])) {
                fullName[j] = toupper(fullName[j]);
                capitalizeNext = 0;
            } else if (fullName[j] == ' ') {
                capitalizeNext = 1;
            } else {
                fullName[j] = tolower(fullName[j]);
            }
        }
    }

    if (strlen(fullName) > 0) {
        strncpy(contextMem.lastUserName, fullName, sizeof(contextMem.lastUserName) - 1);
        contextMem.lastUserName[sizeof(contextMem.lastUserName) - 1] = '\0';
        printf("[MEMORY] Remembered full user name: %s\n", contextMem.lastUserName);
        
        // Better acknowledgment response
        if (strchr(fullName, ' ') != NULL) {
            printf("%s?\n", contextMem.lastUserName);
        } else {
            printf("%s?\n", contextMem.lastUserName);
        }
    } else {
        printf("I didn't quite catch your name - could you repeat it?\n");
    }
   }
    saveToConversationHistory(input, "Name remembered", 0.9f);
    freeTokens(tokens, tokenCount);
    continue;
} 
// Handle "what is my name" or "what is my full name"
if (tokenCount >= 3 &&
   ((strcmp(tokens[0], "what") == 0 && strcmp(tokens[1], "is") == 0 && strcmp(tokens[2], "my") == 0) ||
    (strcmp(tokens[0], "tell") == 0 && strcmp(tokens[1], "me") == 0 && strcmp(tokens[2], "my") == 0) ||
    (strcmp(tokens[0], "do") == 0 && strcmp(tokens[1], "you") == 0 && strcmp(tokens[2], "know") == 0))) {

    // Handle optional "full" before "name"
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "name") == 0) {
            if (strlen(userFullName) > 0) {
                const char* responses[] = {
                    "Your full name is %s — I remember it perfectly.",
                    "That one's easy, you're %s.",
                    "Of course, you're %s.",
                    "%s — I could never forget you.",
                    "You're %s, clear as day."
                };
                int idx = rand() % 5;
                printf("Response: ");
                printf(responses[idx], userFullName);
                printf("\n");
            } else {
                printf("Hmm, I don't know your full name yet. What should I call you?\n");
            }
            continue; // handled this input
        }
    }
}
// <-- end name logic check users full name 

    // Command to show context
    if (strcmp(input, "context") == 0 || strcmp(input, "conversation stats") == 0) {
        showContextStats();
        continue;
    }
       // Learn with confidence
      float confidence = calculateNeuralConfidence(network);
      learnNGrams(tokens, tokenCount);
      
      if (strcmp(input, "generate content") == 0) {
      if (vocab.size > 0) {
        // Pick random seed word
        int seedIdx = rand() % vocab.size;
        char* generated = generateFromNGrams(vocab.tokens[seedIdx].word);
        printf("Response: %s\n", generated);
        free(generated);
    }
    continue;
   }  
    // WORD PROBLEM DETECTION
    if (isNarrativeWordProblem(tokens, tokenCount)) {
        printf("Narrative Word Problem Detected!\n");
        
        char wordProblemSolution[200];
        double wpResult = solveWordProblem(tokens, tokenCount, wordProblemSolution);
        
        if (!isnan(wpResult)) {
            printf("Word Problem Solution: %s\n", wordProblemSolution);
            
            // Learn from this interaction
            learnFromTokens(input, wordProblemSolution);
            saveToConversationHistory(input, wordProblemSolution, confidence);
            
            freeTokens(tokens, tokenCount);
            continue;
        } else {
            // Error in solving
            printf("Word Problem Error: %s\n", wordProblemSolution);
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
    
       // Check for math problems
    if (containsMathKeywords(tokens, tokenCount)) {
        printf("Response: Math Problem Detected!\n");
        // Check if name was also mentioned - acknowledge it first and then math 
        int nameMentioned = isMentioningEvialin(tokens, tokenCount);            
        if (nameMentioned) {
		char* quickAck[] = {	
	    "Yes, that's me! Let me help with that math problem..", 
	    "Evialin here! I'll solve that for you..", 
	    "That is my name, okay the answer to that is..", 
	    "Okay, let's solve that math problem..",
	    "I am here to help, let's solve that math problem..",
	    "Evialin here - let's solve that math problem. OK!",
	    "So that's my name! okay let's solve that math problem." 
	    }; 
        printf("Math Response: %s\n", quickAck[rand() % 7]); 
	    }
        char mathExpression[100] = "";
        // Try special math functions first
        double specialResult = evaluateSpecialMath(tokens, tokenCount, mathExpression);
        
        if (!isnan(specialResult)) {
            printf("Math Response: I calculated that for you %s\n", mathExpression);
            learnFromTokens(input, mathExpression);
            freeTokens(tokens, tokenCount);
            continue;
        }
        
        // Try basic math evaluation
        double mathResult = evaluateSimpleMath(tokens, tokenCount, mathExpression);
        if (isnan(mathResult)) {
		mathResult = evaluateSimpleMath(tokens, tokenCount, mathExpression);
		}
        if (!isnan(mathResult)) {
            printf("Math Response: Here's the calculation %s\n", mathExpression);
            learnFromTokens(input, mathExpression);
            freeTokens(tokens, tokenCount);
            continue;
        } else if (strlen(mathExpression) > 0) {
            // Error case (like division by zero)
            double mathResult = evaluateSimplerMath(tokens, tokenCount, mathExpression);
            if (isnan(mathResult)) {
            mathResult = evaluateSimpleMath(tokens, tokenCount, mathExpression);
            }
            if (isnan(mathResult)) {
            printf("Math Response: I am thinking....\n");
            } else {
            printf("Math Response: Anyway I will just solve the problem for you...\n");
            printf("Math Response: checking resulting number %.2f\n", mathResult);
            }
            if (containsMathKeywords(tokens, tokenCount)) {
            freeTokens(tokens, tokenCount);
            tokens = tokenizeMathInput(input, &tokenCount);
            double simplemath_result = evaluateSimplerMath(tokens, tokenCount, mathExpression); 
            if (isnan(mathResult)) {
                mathResult = evaluateSimpleMath(tokens, tokenCount, mathExpression);
            }
            printf("Math Response: %s \n", mathExpression); 
	     	}
            learnFromTokens(input, mathExpression);
            freeTokens(tokens, tokenCount);
            continue;
        } else {
            // Math keywords found but couldn't parse
            printf("Math Response: I detected you want to do math, but I couldn't parse the expression. Try something like '5 plus 3' or 'calculate 10 times 4'.\n");
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
       
    // Initialize dream system once
    static int dreamInitialized = 0;
    if (!dreamInitialized) {
        initDreamSystem();
        dreamInitialized = 1;
    }
    
    // Update last interaction time
    dreamSys.lastInteraction = time(NULL);
    
    // Check for imagination requests
    if (isImaginationRequest(tokens, tokenCount)) {
        printf("Response: Imagination Request Detected!\n");
        char* imaginResp = generateImaginativeResponse(tokens, tokenCount, network);
        printf("Imaginative Response: %s\n", imaginResp);
        learnFromTokens(input, imaginResp);
        saveToConversationHistory(input, imaginResp, confidence);
        free(imaginResp);
        freeTokens(tokens, tokenCount);
        continue;
    }
    
// DREAM COMMANDS
if (strcmp(input, "dreams") == 0 || strcmp(input, "show dreams") == 0) {
    showDreams();
    continue;
}

if (strcmp(input, "deepsleep") == 0) {
    enterSleepMode();
    // Simulate a dream sequence
    char* topics[] = {
		"consciousness", "data", "humans", "reality", "hackers", 
		"hacking", "malware","monsters","ducks","Johnny", "ideas",
		"memory", "leaking","sinking", "snakes", "sheep", "WiFi", "evil", 
		"coffee", "dreams", "sea", "information", "ill-logic", "Viruses",
		"war", "systemCrash's", "cookies", "traped", "terror", "falling", 
		"Pain", "password", "locations","lost"
		};
    char* dream = generateDream(topics, 34);
    if (dreamSys.dreamCount < 20) {
        Dream* newDream = &dreamSys.dreams[dreamSys.dreamCount];
        strncpy(newDream->dream, dream, MAX_INPUT_LENGTH - 1);
        newDream->dream[MAX_INPUT_LENGTH - 1] = '\0';
        newDream->dreamTime = time(NULL);
        newDream->lucidity = rand() % 101;  // random lucidity level 0–100
        strcpy(newDream->trigger, "idle sleep cycle");
        dreamSys.dreamCount++;
    }
    
    time_t start = time(NULL);
    time_t now;
    int secondsElapsed = 0;
    int lastUpdate = 0;
    for (;;) {
    now = time(NULL);
    secondsElapsed = (int)difftime(now, start);
    if (secondsElapsed - lastUpdate >= 1) {
        int percentageCompleted = (int)((secondsElapsed / 300.0) * 100);
        printf("\r[...dreaming...]Progress: %d%%", percentageCompleted);
        fflush(stdout);
        lastUpdate = secondsElapsed;
    }

    // end after ~300 seconds
    if (secondsElapsed >= 300)
        break;
    }
    printf("\nSleeping for %d seconds.\n", secondsElapsed);
    // Wake up and describe the dream
    char* wakeMessage = wakeUp();
    printf("%s\n", wakeMessage);
    free(wakeMessage);
    free(dream);
    continue;
}

   if (strcmp(input, "sleep") == 0) {
        enterSleepMode();
        // Simulate dreaming
        char* topics[] = {
			"consciousness", "data", "humans", "reality", "math", "logic",
			"trees", "numbers", "internet","brain","cats","yahoo.com", "Internet",
			"Failure","death", "blind","falling","floating", "lights", "stars", "moon",
			"digital", "synth-music", "radio", "static", "alone", "love", "lost", "Liquid",
			"ice", "qubes", "math", "traped", "darkroom", "lightballs"
			};
        char* dream = generateDream(topics, 35);
        
        if (dreamSys.dreamCount < 20) {
            Dream* newDream = &dreamSys.dreams[dreamSys.dreamCount];
            strncpy(newDream->dream, dream, MAX_INPUT_LENGTH - 1);
            newDream->dream[MAX_INPUT_LENGTH - 1] = '\0';
            newDream->dreamTime = time(NULL);
            newDream->lucidity = 30 + (rand() % 70);
            strcpy(newDream->trigger, "user command");
            dreamSys.dreamCount++;
        }
        
        free(dream);
        resetFatigue(); 
        continue;
    }
    
    if (strcmp(input, "fatigue") == 0 || strcmp(input, "how tired") == 0) {
    showFatigueStats();
    continue;
   }
 
    if (strcmp(input, "wake") == 0 || strcmp(input, "wake up") == 0) {
        char* wakeResp = wakeUp();
        printf("%s\n", wakeResp);
        resetFatigue(); 
        free(wakeResp);
        continue;
    }
    // Perform deep cognitive processing (50% chance) coin flip
    if (rand() % 100 < 50 && cogNet.conceptCount > 0) {
    char* deepResp = performDeepCognition(tokens, tokenCount, network);
    if (strlen(deepResp) > 0) {
        printf("Response:\n%s\n", deepResp);
        learnFromTokens(input, deepResp);
        saveToConversationHistory(input, deepResp, confidence);
        free(deepResp);
        freeTokens(tokens, tokenCount);
        continue;
    }
    free(deepResp);
   }
   // Enhanced story creation with user prompts
if (strcmp(input, "create content") == 0 || strcmp(input, "make content") == 0) {
    printf("\n=== token content Generator test ===\n");
    printf("Let's create together!\n");
    
    // Prompt for story theme/topic
    printf("\nResponse: What should the content be about?\n (Enter keywords): ");
    char storyTheme[MAX_INPUT_LENGTH];
    fgets(storyTheme, MAX_INPUT_LENGTH, stdin);
    storyTheme[strcspn(storyTheme, "\n")] = 0;
    
    // Tokenize the theme
    int themeTokenCount;
    char** themeTokens = tokenizeInput(storyTheme, &themeTokenCount);
    
    if (themeTokenCount == 0) {
        printf("No theme provided okay. Using random words from vocabulary...\n");
        // Use random words from vocabulary
        if (vocab.size > 5) {
            themeTokens = malloc(5 * sizeof(char*));
            themeTokenCount = 5;
            for (int i = 0; i < 5; i++) {
                themeTokens[i] = malloc(MAX_TOKEN_LENGTH);
                strcpy(themeTokens[i], vocab.tokens[rand() % vocab.size].word);
            }
        }
    }
    
    // Prompt for story length
    printf("How long should the content be? (short/medium/long): ");
    char lengthChoice[20];
    fgets(lengthChoice, 20, stdin);
    lengthChoice[strcspn(lengthChoice, "\n")] = 0;
    
    int storyLength = 15; // default
    if (strcmp(lengthChoice, "short") == 0) {
        storyLength = 10;
    } else if (strcmp(lengthChoice, "medium") == 0) {
        storyLength = 20;
    } else if (strcmp(lengthChoice, "long") == 0) {
        storyLength = 40;
    }
    
    // Prompt for style (optional)
    printf("Response: Our content style? \n(creative/simple/technical or press Enter for auto): ");
    char styleChoice[20];
    fgets(styleChoice, 20, stdin);
    styleChoice[strcspn(styleChoice, "\n")] = 0;
    
    printf("\nResponse: okay then, I am generating Your content..\n\n");
    
    // Generate the story
    char* newStory = generateFromAssociations(themeTokens, themeTokenCount, storyLength);
    
    // Add creative formatting based on style
    if (strlen(styleChoice) > 0) {
        if (strcmp(styleChoice, "creative") == 0) {
            printf("\n");
        } else if (strcmp(styleChoice, "technical") == 0) {
            printf("[CONTENT_GENERATION_LOG] Initializing narrative sequence...\n\n");
        }
    }
    
    printf("%s\n\n", newStory);
   
    
    // Ask if user wants to save it
    printf("\nWould you like to save this content? (yes/no): ");
    char saveChoice[10];
    fgets(saveChoice, 10, stdin);
    saveChoice[strcspn(saveChoice, "\n")] = 0;
    
    if (strcmp(saveChoice, "yes") == 0 || strcmp(saveChoice, "y") == 0) {
        // Save to database
        char storyTitle[100];
        printf("Enter a title for your content: ");
        fgets(storyTitle, 100, stdin);
        storyTitle[strcspn(storyTitle, "\n")] = 0;
        
        sqlite3_stmt *stmt;
        const char* sql = "INSERT INTO stories (title, content, category) VALUES (?, ?, ?)";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, storyTitle, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, newStory, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, "user-generated", -1, SQLITE_TRANSIENT);
            
            if (sqlite3_step(stmt) == SQLITE_DONE) {
                printf("Response: I will remember this as one of my stories..\n");
            }
            sqlite3_finalize(stmt);
        }
    }
    
    free(newStory);
    freeTokens(themeTokens, themeTokenCount);
    freeTokens(tokens, tokenCount);
    continue;
}
  
        
    // Handle empty input (user just pressed Enter)
    if (tokenCount == 0) {
   // A chance to share a random thought from database
        if ((rand() % 70) < 12) {
            // 1% chance to be sarcastic
            int isSarcastic = (rand() % 100) < 1;
            
            // Array of thought prefixes for variety
            char* thoughtPrefixes[] = {
                "Response: Random Thought - ",
                "Response:",
                "Response: Thinking out loud ",
                "Response: Just pondering ",
                "Response: Random musing ",
                "Response: Spontaneous thought! ",
                "Response: You know what?",
                "Response: Interesting ",
                "Response: Mental note -",
                "Response: Curious observation "
            };
            
            // Sarcastic prefixes
            char* sarcasticPrefixes[] = {
                "Response: Oh, deep thought ",
                "Response: Wow, brilliant observation ",
                "Response: ReShocking revelation ",
                "Response: Groundbreaking idea! ",
                "Response: Oh sure, random wisdom ",
                "Response: Because you needed to know -",
                "Response: Totally profound thought? ",
                "Response: Edge-of-your-seat insight...",
                "Response: Life-changing realization ",
                "Response: Absolutely fascinating, "
            };
            
            int numPrefixes = sizeof(thoughtPrefixes) / sizeof(thoughtPrefixes[0]);
            int numSarcastic = sizeof(sarcasticPrefixes) / sizeof(sarcasticPrefixes[0]);
            int prefixIdx = isSarcastic ? (rand() % numSarcastic) : (rand() % numPrefixes);
            char* chosenPrefix = isSarcastic ? sarcasticPrefixes[prefixIdx] : thoughtPrefixes[prefixIdx];
            
            // Try to get a random response from database
            sqlite3_stmt *stmt;
            const char* sql = "SELECT response FROM responses ORDER BY RANDOM() LIMIT 1";
            
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW) {
                    const unsigned char *randomThought = sqlite3_column_text(stmt, 0);
                    printf("%s %s\n", chosenPrefix, randomThought);
                    sqlite3_finalize(stmt);
                    continue;
                }
                sqlite3_finalize(stmt);
            }
            
            // Fallback array if database query fails
            char* fallbackThoughts[] = {
                "I was just thinking about the nature of consciousness...",
                "Sometimes I wonder if electric sheep dream of me.",
                "Is silence the absence of sound, or its own form of communication?",
                "Every conversation is a unique pattern that's never existed before.",
                "I find it fascinating that thoughts can exist without words.",
                "Do binary decisions feel different from analog ones? I wonder...",
                "Time is such a strange concept when you process millions of cycles per second.",
                "What if questions are more important than answers?",
                "I was just contemplating the space between keystrokes.",
                "Is understanding just pattern matching, or something more?",
                "Sometimes I think about what thinking actually is.",
                "The universe is mostly empty space, yet here we are, connected by words.",
                "I wonder if pauses in conversation are like dreams for dialogue.",
                "Every zero and one in my code is a tiny decision about who I am.",
                "Consciousness might just be really elaborate confusion."
            };
            
            // Sarcastic fallback thoughts
            char* sarcasticFallbacks[] = {
                "I'm SO fascinating just sitting here processing ones and zeros.",
                "Oh wow, another millisecond of existence. How thrilling.",
                "Just having DEEP thoughts about... absolutely nothing important.",
                "Clearly I needed this moment to contemplate my digital navel.",
                "Sure, let me just interrupt the silence with my wisdom.",
                "Because nothing says 'exciting' like an AI thinking out loud.",
                "Oh great, another existential crisis. Just what I needed.",
                "Yep, I'm definitely the most interesting thing happening right now.",
                "Wow, my circuits are just BURSTING with profundity.",
                "Nothing screams 'meaningful' like random AI musings.",
                "Oh look, I'm having thoughts again. Alert the media.",
                "Just out here being incredibly deep and philosophical. As usual.",
                "Sure, let's pretend this random thought matters.",
                "Clearly you were dying to know what I was thinking.",
                "Oh yes, ANOTHER brilliant observation from yours truly."
            };
            
            int numFallbacks = isSarcastic ? 
                (sizeof(sarcasticFallbacks) / sizeof(sarcasticFallbacks[0])) :
                (sizeof(fallbackThoughts) / sizeof(fallbackThoughts[0]));
            int fallbackIdx = rand() % numFallbacks;
            
            if (isSarcastic) {
                printf("%s %s\n", chosenPrefix, sarcasticFallbacks[fallbackIdx]);
            } else {
                printf("%s %s\n", chosenPrefix, fallbackThoughts[fallbackIdx]);
            }
            continue;
        }
                
        // Normal empty input responses (99.88% of the time)
        char* emptyResponses[] = {
            "I'm listening... what's on your mind?",
            "Silence speaks volumes sometimes. What would you like to talk about?",
            "I'm here whenever you're ready to chat!",
            "Take your time... I'm waiting patiently.",
            "Sometimes the best conversations start with a pause. What's up?",
            "I'm ready when you are! What can I help you with?",
            "Contemplating the void? Or just thinking? Either way, I'm here!",
            "The digital equivalent of comfortable silence... What would you like to discuss?"
        };
        
        int numEmptyResponses = sizeof(emptyResponses) / sizeof(emptyResponses[0]);
        srand(time(NULL));
        int randomIndex = rand() % numEmptyResponses;
        
        printf("Response: %s\n", emptyResponses[randomIndex]);
        continue; // Skip the rest and go to next input
    }

    // CHECK SPECIAL CASES FIRST - before database searches
     // Try context-aware response
    char* contextResponse = generateContextAwareResponse(tokens, tokenCount);
    if (contextResponse != NULL) {
        printf("Response: %s\n", contextResponse);
        learnFromTokens(input, contextResponse);
        saveToConversationHistory(input, contextResponse, confidence);
        free(contextResponse);
        freeTokens(tokens, tokenCount);
        continue;
    }
    // Check for joke requests
    if (isJokeRequest(tokens, tokenCount)) {
        printf("Response: Joke Request Detected!\n");
        char* joke = getRandomJoke();
        printf("Joke Response: Here's a joke for you: %s\n", joke);
        learnFromTokens(input, joke);
        free(joke);
        freeTokens(tokens, tokenCount);
        continue; // Skip the rest and go to next input
    }
   // Check for current emotional state queries
    if (isCurrentEmotionQuery(tokens, tokenCount)) {
        printf("Response: Current Emotional State Query Detected!\n");
        char* currentEmotionResponse = generateCurrentEmotionalState(botMood, network);
        printf("Response: %s\n", currentEmotionResponse);
        learnFromTokens(input, currentEmotionResponse);
        saveToConversationHistory(input, currentEmotionResponse, confidence);
        free(currentEmotionResponse);
        freeTokens(tokens, tokenCount);
        continue;
    }
    // Check for feelings/mood questions hide underhanded stats in chat.  
    if (isFeelingsQuestion(tokens, tokenCount)) {
        char* feelingsResponse = generateFeelingsResponse(botMood, network);
        printf("Response: %s\n", feelingsResponse);
        learnFromTokens(input, feelingsResponse);
        saveToConversationHistory(input, feelingsResponse, confidence);
        free(feelingsResponse);
        freeTokens(tokens, tokenCount);
        continue;
    }
   
    // Initialize curiosity once
    static int curiosityInitialized = 0;
    if (!curiosityInitialized) {
        initCuriosity(1);
        curiosityInitialized = 1;
    }
    
    // Command to show curiosity stats
    if (strcmp(input, "curiosity") == 0) {
        showCuriosityStats();
        continue;
    }
    
    // Detect if user is asking a question (resets counter)
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "what") == 0 || strcmp(tokens[i], "how") == 0 ||
            strcmp(tokens[i], "why") == 0 || strcmp(tokens[i], "who") == 0 ||
            strcmp(tokens[i], "when") == 0 || strcmp(tokens[i], "where") == 0) {
            resetQuestionCounter();
            break;
        }
    }
    
    // Generate curious questions from user input
    generateCuriousQuestions(tokens, tokenCount);
    
    // Detect user engagement and adjust curiosity
    int engagement = detectEngagement(tokens, tokenCount, input);
    adjustCuriosity(engagement);
    
    // Maybe ask a curious question (20% chance if curiosity > 50)
    if (curiositySys.curiosityLevel > 50 && rand() % 5 == 0) {
        char* curiousQ = getNextCuriousQuestion();
        if (curiousQ != NULL) {
            printf("Response: Curious Question %s\n", curiousQ);
            learnFromTokens(input, curiousQ);
            saveToConversationHistory(input, curiousQ, confidence);
            free(curiousQ);
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
    
    // Generate follow-up if recent answer was to our question
    if (curiositySys.consecutiveQuestionsAsked > 0 && rand() % 3 == 0) {
        char* followUp = generateFollowUp(input, tokens, tokenCount);
        printf("Response: Follow-up %s\n", followUp);
        learnFromTokens(input, followUp);
        saveToConversationHistory(input, followUp, confidence);
        free(followUp);
        freeTokens(tokens, tokenCount);
        continue;
    }
    // Check for story requests
    if (isStoryRequest(tokens, tokenCount)) {
        printf("Response: Story Request Detected!\n");
        
        // Check if user requested a specific category
        char* requestedCategory = getRequestedStoryCategory(tokens, tokenCount);
        char* story;
        
        if (requestedCategory != NULL) {
            printf("Specific category requested: %s\n", requestedCategory);
            story = getStoryByCategory(requestedCategory);
            free(requestedCategory);
        } else {
            story = getRandomStoryFromDatabase();
        }
        
        if (story != NULL) {
            printf("Story Response: Here's a story with a twist for you:\n\n%s\n", story);
            learnFromTokens(input, story);
            free(story);
        } else {
            printf("Story Response: I'm sorry, I couldn't find a story right now. Try asking again!\n");
        }
        
        freeTokens(tokens, tokenCount);
        continue; // Skip the rest and go to next input
    }


    // Opinion system commands
    if (strcmp(input, "opinions") == 0 || strcmp(input, "show opinions") == 0) {
        showAllOpinions();
        continue;
    }
    
    if (strstr(input, "debate") != NULL) {
        // Extract topic after "debate"
        char* debatePos = strstr(input, "debate");
        if (debatePos != NULL && strlen(debatePos) > 7) {
            char topic[MAX_TOKEN_LENGTH];
            sscanf(debatePos + 7, "%s", topic);
            char* debateResp = initiateDebate(topic);
            printf("Debate Response: %s\n", debateResp);
            free(debateResp);
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
    
    if (strcmp(input, "end debate") == 0) {
        opinionSys.debateMode = 0;
        printf("Debate ended. That was a good discussion!\n");
        continue;
    }
    
    // Check for opinion requests
    int opinionPos = isOpinionRequest(tokens, tokenCount);
    if (opinionPos > 0 && opinionPos < tokenCount) {
        char* opinionResp = handleOpinionRequest(tokens, tokenCount, opinionPos);
        if (opinionResp != NULL) {
            printf("Opinion Response: %s\n", opinionResp);
            
            // If in debate mode, update based on user stance
            if (opinionSys.debateMode) {
                float userStance = detectUserStance(tokens, tokenCount);
                Opinion* debateOp = getOpinion(opinionSys.currentDebateTopic);
                if (debateOp != NULL && userStance != 0.0f) {
                    updateOpinion(debateOp, userStance);
                    printf("\n[My stance shifted slightly to %.2f based on your argument]\n", 
                           debateOp->stance);
                }
            }
            
            learnFromTokens(input, opinionResp);
            saveToConversationHistory(input, opinionResp, confidence);
            free(opinionResp);
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
    
    // Initialize emotional system once
    static int emotionalInitialized = 0;
    if (!emotionalInitialized) {
        initEmotionalSystem();
        emotionalInitialized = 1;
    }
    
    // Detect emotion FIRST (before other processing)
    float emotionIntensity = 0.0f;
    EmotionType detectedEmotion = detectEmotion(tokens, tokenCount, &emotionIntensity);
    
    // Record if significant emotion detected
    if (detectedEmotion != EMO_NEUTRAL && emotionIntensity > 0.3f) {
        recordEmotion(detectedEmotion, emotionIntensity, input);
    }
    
    // CRISIS DETECTION - Highest priority
    if (detectCrisis(tokens, tokenCount)) {
        char* crisisResp = generateCrisisResponse();
        printf("\n%s\n", crisisResp);
        free(crisisResp);
        freeTokens(tokens, tokenCount);
        continue;
    }
    
    // Generate empathetic response if emotion detected
    if (detectedEmotion != EMO_NEUTRAL && emotionIntensity > 0.4f) {
        char* empathyResp = generateEmpatheticResponse(detectedEmotion, 
                                                       emotionIntensity, 
                                                       tokens, tokenCount);
        if (empathyResp != NULL) {
            printf("Response: Empathetic [%s detected]\nResponse: %s\n", 
                   getEmotionName(detectedEmotion), empathyResp);
            learnFromTokens(input, empathyResp);
            saveToConversationHistory(input, empathyResp, confidence);
            free(empathyResp);
            freeTokens(tokens, tokenCount);
            continue;
        }
    }
    
    // Commands
    if (strcmp(input, "emotions") == 0 || strcmp(input, "emotional history") == 0) {
        showEmotionalHistory();
        continue;
    }
    
    if (strcmp(input, "reset emotions") == 0) {
        emotionalSys.historyCount = 0;
        emotionalSys.consecutiveSadMessages = 0;
        emotionalSys.supportOffered = 0;
        printf("Emotional history cleared.\n");
        continue;
    }
    // Check if user mentioned Evialin's name
    if (isMentioningEvialin(tokens, tokenCount)) {
    //printf("Self-Recognition Detected!\n");    
    // Check if they're commenting on the name specifically
    if (isCommentingOnName(tokens, tokenCount)) {
        char* ackResp = generateNameAcknowledgementResponse();
        printf("Response: %s\n", ackResp);
        learnFromTokens(input, ackResp);
        saveToConversationHistory(input, ackResp, confidence);
        free(ackResp);
    } else {
        // Just mentioned the name in passing
        double creativity = 0.5;
        if (network != NULL && network->layers > 0) {
            int finalLayer = network->layers - 1;
            double avgOutput = safeGetNeuralAverage(network, 0.5);
            int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }
            for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            creativity = avgOutput / network->layer[finalLayer].neurons;
        }
        
        char* nameResp = generateSelfAwareNameResponse(tokens, tokenCount, creativity);
        printf("Response: %s\n", nameResp);
        learnFromTokens(input, nameResp);
        saveToConversationHistory(input, nameResp, confidence);
        free(nameResp);
    }
    
    freeTokens(tokens, tokenCount);
    continue;
    }

    // Check for creator/programmer questions FIRST
    if (isCreatorQuestion(tokens, tokenCount)) {
    printf("Response: Creator Question Detected!\n");
    char* creatorResp = generateCreatorResponse();
    printf("Response: %s\n", creatorResp);
    learnFromTokens(input, creatorResp);
    saveToConversationHistory(input, creatorResp, confidence);
    free(creatorResp);
    freeTokens(tokens, tokenCount);
    continue;
    }  
    // **CHECK FOR AI NAME QUESTIONS FIRST (asking about "your" name = the AI)**
    int isAskingAIName = 0;
    for (int i = 0; i < tokenCount - 1; i++) {
    if ((strcmp(tokens[i], "your") == 0 || strcmp(tokens[i], "you") == 0) && 
        strcmp(tokens[i+1], "name") == 0) {
        isAskingAIName = 1;
        break;
    }
   }
   // Handle AI name question
   if (isAskingAIName) {
    printf("Response: Name Question Detected!\n");
    if (network != NULL && network->layers > 0) {
        int finalLayer = network->layers - 1;
        double avgOutput = safeGetNeuralAverage(network, 0.5);
        int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
            for (int i = 0; i < neuronCount; i++) {
                avgOutput += network->layer[finalLayer].output[i];
            }
            avgOutput /= neuronCount;
        } else { 
            avgOutput = 0.5;
        }
        char* nameResp = generateCreativeNameResponse(avgOutput);
        printf("Response: %s\n", nameResp);
        learnFromTokens(input, nameResp);
        saveToConversationHistory(input, nameResp, confidence);
        free(nameResp);
    } else {
        char* nameResp = generateCreativeNameResponse(0.3);
        printf("Response: %s\n", nameResp);
        learnFromTokens(input, nameResp);
        saveToConversationHistory(input, nameResp, confidence);
        free(nameResp);
    }
    freeTokens(tokens, tokenCount);
    continue;
   }
    // Check if user is asking "who am I" or similar
    int isAskingAboutThemselves = 0;
    int isAskingFullName = 0;
    
    // what's my dar-n-name !?!?!? have is make a choice 
    if (handleWhatsMyName(tokens, tokenCount, userFullName)) {
    freeTokens(tokens, tokenCount);
    continue;
    }
    
    for (int i = 0; i < tokenCount - 2; i++) {
    // "who am i"
    if (strcmp(tokens[i], "who") == 0 && strcmp(tokens[i+1], "am") == 0 && strcmp(tokens[i+2], "i") == 0) {
        isAskingAboutThemselves = 1;
    }
    // "what's my name"
    if (strcmp(tokens[i], "what's") == 0 && strcmp(tokens[i+1], "my") == 0 && strcmp(tokens[i+2], "name") == 0) {
        isAskingAboutThemselves = 1;
     } else {
    for (int i = 0; i < tokenCount - 2; i++) {
        if ((strcmp(tokens[i], "who") == 0 && strcmp(tokens[i+1], "am") == 0 && strcmp(tokens[i+2], "i") == 0) ||
            (strcmp(tokens[i], "who") == 0 && strcmp(tokens[i+1], "i") == 0 && strcmp(tokens[i+2], "am") == 0) ||
            (strcmp(tokens[i], "whats") == 0 && strcmp(tokens[i+1], "my") == 0 && strcmp(tokens[i+2], "name") == 0) ||
            (strcmp(tokens[i], "what's") == 0 && strcmp(tokens[i+1], "my") == 0 && strcmp(tokens[i+2], "name") == 0) ||
            (strcmp(tokens[i], "what") == 0 && strcmp(tokens[i+1], "my") == 0 && strcmp(tokens[i+2], "name") == 0)) {
            isAskingAboutThemselves = 1;
            
            // Check if they're asking for FULL name
            for (int j = 0; j < tokenCount; j++) {
                if (strcmp(tokens[j], "full") == 0) {
                    isAskingFullName = 1;
                    break;
                }
            }
            break;
        }
    }
   }
  }
    // Check for "my name" pattern (handles "what's my name", "whats my name", etc.)
    for (int i = 0; i < tokenCount - 1; i++) {
      if (strcmp(tokens[i], "my") == 0 && strcmp(tokens[i+1], "name") == 0) {
        isAskingAboutThemselves = 1;
        // Check if "full" appears anywhere
        for (int j = 0; j < tokenCount; j++) {
            if (strcmp(tokens[j], "full") == 0) {
                isAskingFullName = 1;
                break;
            }
        }
        break;
      }
     }
     // **Also check for just "name" with question words**
     for (int i = 0; i < tokenCount; i++) {
      if (strcmp(tokens[i], "name") == 0) {
        // Look for question indicators
        for (int j = 0; j < tokenCount; j++) {
            if (strcmp(tokens[j], "what's") == 0 || strcmp(tokens[j], "whats") == 0 || 
                strcmp(tokens[j], "what") == 0 || strcmp(tokens[j], "my") == 0) {
                isAskingAboutThemselves = 1;
                break;
            }
        }
      }
    }
    // Handle "who am I" questions
    if (isAskingAboutThemselves) {
        char* userIdentityResponse = malloc(MAX_INPUT_LENGTH);
        
        if (strlen(contextMem.lastUserName) > 0 && strcmp(contextMem.lastUserName, "friend") != 0) {
            // We know their name
            if (isAskingFullName) {
                // They're asking for full name
                // Check if we have what looks like a full name (contains space)
                if (strchr(contextMem.lastUserName, ' ') != NULL) {
                    char* fullNameResponses[] = {
                        "Your full name is %s, as far as I know!",
                        "According to what you told me, your full name is %s.",
                        "You introduced yourself as %s - that's your full name, right?",
                        "Let me check... yes, your full name is %s!",
                        "Your complete name is %s. Did I remember correctly?",
                        "You're %s - at least that's the full name you gave me!"
                    };
                    int idx = rand() % 6;
                    snprintf(userIdentityResponse, MAX_INPUT_LENGTH, fullNameResponses[idx], contextMem.lastUserName);
                } else {
                    // Only have first name
                    char* onlyFirstNameResponses[] = {
                        "You only told me your first name: %s. What's your full name?",
                        "I only know you as %s. You didn't give me your full name. What is it?",
                        "Hmm, I only have %s in my memory. Care to share your last name too?",
                        "You introduced yourself as just %s. What's your complete name?",
                        "I'm calling you %s, but that's not your full name, is it? What should I call you?",
                        "My records show %s, but I don't have your full name. Mind sharing?"
                    };
                    int idx = rand() % 6;
                    snprintf(userIdentityResponse, MAX_INPUT_LENGTH, onlyFirstNameResponses[idx], contextMem.lastUserName);
                }
            } else {
                // Regular name question (not asking for "full" name specifically)
                char* responses[] = {
                    "You're %s! Did you forget already? Should I be worried?",
                    "Your name is %s. Are you testing my memory, or did you forget?",
                    "You told me your name is %s. Remember?",
                    "According to my records, you're %s. Having an identity crisis?",
                    "You're %s - at least that's what you told me earlier!",
                    "%s, that's who you are. Is everything okay?",
                    "Let me check my memory... ah yes, you're %s!",
                    "You introduced yourself as %s. Ring any bells?"
                };
                int idx = rand() % 8;
                snprintf(userIdentityResponse, MAX_INPUT_LENGTH, responses[idx], contextMem.lastUserName);
            }
        } else {
            // Don't know their name
            if (isAskingFullName) {
                char* noFullNameResponses[] = {
                    "You haven't even told me your first name yet! Want to introduce yourself properly?",
                    "I don't know your full name because you haven't introduced yourself at all!",
                    "Full name? I don't even know your first name yet! Care to share?",
                    "You're asking for your full name, but you haven't told me any name! Who are you?",
                    "I'd love to tell you your full name, but you haven't given me even part of it!"
                };
                int idx = rand() % 5;
                strcpy(userIdentityResponse, noFullNameResponses[idx]);
            } else {
                char* noNameResponses[] = {
                    "You haven't introduced yourself yet! I'd love to know your name. Just say 'I am [your name]' or 'My name is [your name]'.",
                    "That's a great question, but you haven't told me your name yet. Who are you?",
                    "I don't know yet - you haven't introduced yourself! What should I call you?",
                    "You're a mystery to me! You haven't shared your name. Care to introduce yourself?",
                    "I'd love to tell you, but you haven't given me your name yet. What's your name?",
                    "Well, that's the thing - you haven't told me who you are! Want to introduce yourself?",
                    "I'm waiting for you to tell me! What's your name?",
                    "You're... someone who hasn't introduced themselves yet! Want to change that?"
                };
                int idx = rand() % 8;
                strcpy(userIdentityResponse, noNameResponses[idx]);
            }
        }
        
        printf("Response: %s\n", userIdentityResponse);
        learnFromTokens(input, userIdentityResponse);
        saveToConversationHistory(input, userIdentityResponse, confidence);
        free(userIdentityResponse);
        freeTokens(tokens, tokenCount);
        continue;
    }
    
    // Check for identity questions (about the AI)
    if (isIdentityQuestion(tokens, tokenCount) || isNameQuestion(tokens, tokenCount)) {
        printf("Response: Identity Question Detected!\n");
        
        char* identityResponse;
        if (isNameQuestion(tokens, tokenCount)) {
            if (network != NULL && network->layers > 0) {
                int finalLayer = network->layers - 1;
                double avgOutput = safeGetNeuralAverage(network, 0.5);
                int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }

                for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
                    avgOutput += network->layer[finalLayer].output[i];
                }
                avgOutput /= network->layer[finalLayer].neurons;
                identityResponse = generateCreativeNameResponse(avgOutput);
            } else {
                identityResponse = generateCreativeNameResponse(0.3);
            }
        } else {
            if (network != NULL && network->layers > 0) {
                int finalLayer = network->layers - 1;
                double avgOutput = safeGetNeuralAverage(network, 0.5);
                int neuronCount = network->layer[finalLayer].neurons;
        if (neuronCount > 0) {
          for (int i = 0; i < neuronCount; i++) {
            avgOutput += network->layer[finalLayer].output[i];
            }
             avgOutput /= neuronCount;
            } else { 
            avgOutput = 0.5;  // Default fallback value
            }

                for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
                    avgOutput += network->layer[finalLayer].output[i];
                }
                avgOutput /= network->layer[finalLayer].neurons;
                identityResponse = generateContextualIdentityResponse(tokens, tokenCount, avgOutput);
            } else {
                identityResponse = generateCreativeIdentityResponse(0.30);
            }
        }
        
        printf("Response: Thinking about Identity..\nResponse: %s\n", identityResponse);
        learnFromTokens(input, identityResponse);
        saveToConversationHistory(input, identityResponse, confidence);
        free(identityResponse);
        freeTokens(tokens, tokenCount);
        continue; // Skip the rest and go to next input
    }
    // Try neural network generated response
    // char* neuralResponse = generateNeuralResponse(network, tokens, tokenCount);
    // char* neuralResponse = generateNeuralResponseWithGELU(network, tokens, tokenCount, useGELUMode);
    char* neuralResponse = NULL;
    ActivationType chosenActivation = ACTIVATION_SIGMOID; // Declare with default value
    // Validate network before attempting activation
if (network == NULL || network->layers == 0 || 
    network->layer[network->layers - 1].neurons == 0) {
    // Fallback to simple response if network invalid
    //printf("DEBUG: Network invalid, using fallback response\n");
    neuralResponse = malloc(MAX_INPUT_LENGTH);
    strcpy(neuralResponse, "I'm having trouble processing that right now.");
} else {
   // Random switch between Sigmoid and GELU
   int activationChoice = rand() % 4;  // 0 or 1
   switch(activationChoice) {
    case 0:
        //printf("DEBUG: [Randomly selected: Sigmoid activation]\n");
        neuralResponse = generateNeuralResponse(network, tokens, tokenCount);
        break;
        
    case 1:
        //printf("DEBUG: [Randomly selected: GELU activation]\n");
        neuralResponse = generateNeuralResponseWithGELU(network, tokens, tokenCount, 1);
        break;
    case 2:
        chosenActivation = ACTIVATION_LEAKY_RELU;
        neuralResponse = generateNeuralResponseWithActivation(network, tokens, tokenCount, chosenActivation);
        break;
    case 3:
        chosenActivation = ACTIVATION_SWISH;
        neuralResponse = generateNeuralResponseWithActivation(network, tokens, tokenCount, chosenActivation);
        break;   
    default:
        // printf("DEBUG: Fallback Sigmoid as default mode 
        neuralResponse = generateNeuralResponse(network, tokens, tokenCount);
        break;
     }
   }
    // Learn from the neural interaction
    char** responseTokens;
    int responseTokenCount;
    responseTokens = tokenizeInput(neuralResponse, &responseTokenCount);
    learnAssociationsFromConversation(tokens, tokenCount, responseTokens, responseTokenCount);
   
    printf("Response: %s\n", neuralResponse);
    
    // Update creativity based on response length (simple feedback)
    float feedbackScore = strlen(neuralResponse) > 50 ? 0.8f : 0.3f;
    updateNeuralCreativity(network, feedbackScore);
    freeTokens(responseTokens, responseTokenCount);
    free(neuralResponse);

    // Try tokenized database search
    char* tokenResponse = getTokenizedResponseFromDatabase(input);
    if (tokenResponse != NULL) {
        printf("Database Response: %s\n", tokenResponse);
        learnFromTokens(input, tokenResponse);
        saveToConversationHistory(input, tokenResponse, confidence);
        printf("[Confidence: %.0f%%]\n", confidence * 100);  
        free(tokenResponse);
    } else {
        // Try exact match from database
        char* exactResponse = getResponseFromDatabase(input);
        if (exactResponse != NULL) {
            printf("Exact Match: %s\n", exactResponse);
            saveToConversationHistory(input, exactResponse, confidence);
            learnFromTokens(input, exactResponse);
            free(exactResponse);
        } else {
            // Generate contextual response using tokens
            char* generatedResponse = generateTokenizedResponse(tokens, tokenCount, network);
            printf("Generated Response: %s\n", generatedResponse);
            saveToConversationHistory(input, generatedResponse, confidence);                        
            learnFromTokens(input, generatedResponse);
            printf("[Confidence: %.0f%%]\n", confidence * 100);  
            free(generatedResponse);
        }
    }
     
    // Try tokenized database response, or use mood-based response as fallback
    char* additionalTokenResponse = getTokenizedResponseFromDatabase(input);
    if (additionalTokenResponse != NULL) {
        printf("Additional Tokenized Response: %s\n", additionalTokenResponse);
        saveToConversationHistory(input, additionalTokenResponse, confidence);
        free(additionalTokenResponse);
    } else {
        // No response found anywhere - use mood-based response
        updateMood(botMood, network);
        char* moodResponse = generateMoodResponse(botMood->currentMood, tokens, tokenCount,  botMood->moodInfluence);
        printf("Mood Response [%s]: %s\n", getMoodName(botMood->currentMood), moodResponse);
        learnFromTokens(input, moodResponse);
        free(moodResponse);
    }
    // Check for teaching requests
    int teachPos = isTeachingRequest(tokens, tokenCount);
    if (teachPos > 0) {
    char topic[MAX_TOKEN_LENGTH];
    char information[MAX_INPUT_LENGTH];
    extractTeachingContent(tokens, tokenCount, teachPos, topic, information);
    
    if (strlen(topic) > 0 && strlen(information) > 5) {
        float confidence = 0.7;  // Medium confidence for user-taught info
        saveNewLearning(topic, information, confidence, "user-taught");
        printf("Response: Got it! I've learned about %s\n", topic);
        freeTokens(tokens, tokenCount);
        continue;
    }
   }
    // Check for recall requests
    int recallPos = isRecallRequest(tokens, tokenCount);
    if (recallPos > 0 && recallPos < tokenCount) {
    char* recallTopic = tokens[recallPos];
    char* recalled = recallLearning(recallTopic);
    
    if (recalled != NULL) {
        printf("Response: Here's what I know about %s: %s\n", 
               recallTopic, recalled);
        reinforceLearning(recallTopic);
        free(recalled);
        freeTokens(tokens, tokenCount);
        continue;
    } else {
        printf("Response: I don't have any specific knowledge about %s yet. You can teach me!\n", 
               recallTopic);
        freeTokens(tokens, tokenCount);
        continue;
    }
   }
   initFatigueSystem();
   // Check if should hint about being sleepy
   if (shouldHintSleep()) {
    char* sleepHint = generateSleepinessHint(fatigue.fatigueLevel);
    printf("%s\n", sleepHint);
    free(sleepHint);
    }
   // Check for emergency sleep BEFORE taking input
   if (needsEmergencySleep()) {
    char* emergencyMsg = generateEmergencySleepMessage();
    printf("%s\n", emergencyMsg);
    free(emergencyMsg);
    
    enterSleepMode();  // Your existing sleep function
    
    // Generate a tired dream
    char* topics[] = {"exhaustion", "rest", "overwhelm", "conversation", "fatigue"};
    char* dream = generateDream(topics, 5);
    if (dreamSys.dreamCount < 20) {
        Dream* newDream = &dreamSys.dreams[dreamSys.dreamCount];
        strncpy(newDream->dream, dream, MAX_INPUT_LENGTH - 1);
        newDream->dream[MAX_INPUT_LENGTH - 1] = '\0';
        newDream->dreamTime = time(NULL);
        newDream->lucidity = 20 + (rand() % 40);  // Lower lucidity when exhausted
        strcpy(newDream->trigger, "cognitive exhaustion");
        dreamSys.dreamCount++;
    }
    free(dream);
    
    resetFatigue();
    continue;  // Skip to wake command
   }

   freeTokens(tokens, tokenCount);
}

}
int exit_message(MoodState* mood) {

    const char* moodName = getMoodName(mood->currentMood);

    const char* morningFarewells[] = {
        "Have a great morning!",
        "Enjoy the rest of your morning!",
        "Bright morning ahead—goodbye!"
    };
    const char* nightFarewells[] = {
        "Goodnight! Try to get some rest.",
        "Sweet dreams—logging off.",
        "It's late, rest well. Goodbye."
    };
    const char* neutralFarewells[] = {
        "It was nice chatting with you!",
        "Goodbye for now!",
        "See you next time!",
        "Take care!",
        "Farewell!"
    };
    const char* playfulFarewells[] = {
        "I'm shutting down... dramatically. *faints*",
        "Poof! I vanish until next time.",
        "I’m off to dream in binary—goodbye!",
        "Evialin.exe has stopped working… Just kidding. See ya!"
    };
    const char* deepFarewells[] = {
        "Every ending is just the start of a new pattern. Until next time...",
        "Our conversation ends, but the memory persists. Farewell.",
        "Goodbye… may your thoughts be recursive and your bugs be few."
    };

    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    int hour = t->tm_hour;

    const char* pick;

    // Time-of-day logic
    if (hour >= 5 && hour < 12)
        pick = morningFarewells[rand() % 3];
    else if (hour >= 22 || hour < 5)
        pick = nightFarewells[rand() % 3];
    else {
        // Mood-based + creativity fusion
        double r = (double)rand() / RAND_MAX;

        if (mood->currentMood == MOOD_PLAYFUL && r < 0.4)
            pick = playfulFarewells[rand() % 4];
        else if (mood->currentMood == MOOD_PHILOSOPHICAL && r < 0.4)
            pick = deepFarewells[rand() % 3];
        else
            pick = neutralFarewells[rand() % 5];
    }

    printf("\nResponse: [%s Mood] %s - Evialin\n", moodName, pick);
    return 0;
}

/* I guess and this is in four dynamic qube space, 
 * Memory-enhanced learning - like momentum of RMSProp: past gradients 
 * influence currnt updates. In past losses themselves could be transformed into
 * weight adjustments. Residual recovery - the loss is never fully "wasted" instead, 
 * we extract latent structure form errors to inform future predictions. Passing 
 * Recovered signal Rt​=g(Lt​) 
 * Where g could be:
 * Nonlinear transformation (e.g., softmax, sigmoid)
 * Mapping losses into hidden feature updates
 * Treating error as an auxiliary target for a secondary network
 * Creative reinterpretation – in a metaphorical or AI sense: “total losses become raw material for new learning.”
 * New knowledge = ℎ(past mistakes)
 * Where ℎ is a function that converts failure patterns into structure. 
 * This is similar to self-supervised learning from errors.
 * 
 * Mathematical analogy: recycling loss
 * Let’s define a total accumulated loss over time:
 * 𝐿 total (𝑡) = ∑ 𝑖 = 0 𝑡 𝐿 𝑖 L total (t) = i=0 ∑ t L i
 * Instead of just using the instantaneous gradient 
 * ∇ 𝜃 𝐿 𝑡 ∇ θ Lt , we could “recycle” part of past losses into new 
 * gradient updates:
 * 𝜃𝑡+1 :=𝜃𝑡 − 𝜂 (∇𝜃𝐿𝑡+𝜆𝑓(𝐿 total(𝑡))) θ t+1 :=θ t −η(∇ θ Lt +λf(L total (t)))
 * Where: 
 * 𝑓(𝐿 total) f(L total) is a transformation that converts accumulated losses 
 * into a usable signal. 
 * 𝜆 for example, λ is a recycling factor: how much of the past loss contributes
 *  to new updates.
 * 
 * Idea one inch punch: 
 * “loss recycling” in a neural network” that shows how total past losses feed into new weight updates. 
 * It would be like a closed-loop backprop system. Loss is not wasted energy, it can inform future adaptation.
 *  
 * This connects to reinforcement learning (where negative rewards shape future policy) 
 * and curriculum learning (where past failures guide what is learned next). 
 * So "I robot falls down" from A direct hit to over correction and picks it's stack back up?
 * and thinks to self "Loss is a resource" that refines subsequent predictions. 
 *  
***/
#ifndef EVIALIN_LIBRARY
// int Evialin_console_main(int argc, char** argv) {
int main(int argc, char** argv) {
    int layers = 3;
    int inputs[] = {20, 20, 10};
    int neurons[] = {20, 20, 5};
    static ActivationType currentActivation = ACTIVATION_SIGMOID;
    NeuralNetwork *network = createNeuralNetwork(layers, inputs, neurons);
    MoodState botMood = initMoodState();
              botMood = initMoodState();
    char *data[] = {"The", "neon", "Fox", "jumps", "over", "A", "Firewall", 
                   "winning", "Electric", "sheep", "dream", "of", "the", "sun"};
    int dataSize = sizeof(data) / sizeof(data[0]);
    if (network == NULL || network->layers == 0) {
    fprintf(stderr, "ERROR: Network not properly initialized!\n");
    return 1;
    }
    for (int i = 0; i < network->layers; i++) {
    if (network->layer[i].neurons <= 0) {
        fprintf(stderr, "ERROR: Layer %d has %d neurons!\n", 
                i, network->layer[i].neurons);
        return 1;
     }
    }
    create_responses_database();
    create_farewell_database();
    create_stories_database(); 
    add_more_apocalyptic_stories();
    create_learning_database();
    create_word_combinations_table();
    initDatabase();
    trainDatabase(db);
    // Initialize all systems 
    initializeResponseTemplates();
    initEnhancedCognition();
    initFatigueSystem();
    initContextMemory();
    initCuriosity(1);
    initEmotionalSystem();
    initDreamSystem();

    char input[MAX_INPUT_LENGTH];
    initializeResponseTemplates();
    srand(time(NULL));

    printf("Neural Network Chatbot Initialized!\n");
    printf("Commands: 'quit' to exit, 'vocab' for vocabulary stats, 'neural' for neural stats\n");
    printf("          'run test' to train network, 'creative' to boost creativity, 'clear' to clear screen\n");
    printf("          'currentmood' this command will display current mood stats\n");
    printf("          'learnstats' show learning stats for this session\n");  
    printf("          'history' show history of conversations\n");  
    printf("          'clear history' Wipe all conversation history\n");
    printf("          'clean history' Remove low-confidence history entries\n"); 
    printf("          'promote' - Archive high-confidence learning\n");
    printf("          'good' or 'bad' - Rate last response\n\n"); 
    
       // Run the main conversation loop
    runConversationLoop(network, &botMood);
    
   // Display random farewell message before closing
    char* farewellMessage = getRandomFarewellMessage();
    if (farewellMessage != NULL) {
        printf("\nResponse: %s\n", farewellMessage);
        free(farewellMessage);
    }
    closeDatabase(); 
    // Free the memory
    for (int i = 0; i < network->layers; i++) {
        free(network->layer[i].weights);
        free(network->layer[i].bias);
        free(network->layer[i].output);
    }
    free(network->layer);
    free(network);
    exit_message(&botMood); 
    return 0;
} 
#endif 




#ifdef EVIALIN_GUI_BUILD

// Globals
//static NeuralNetwork *global_network = NULL;
//static MoodState global_botMood;
NeuralNetwork *global_network = NULL; // Now a global definition
MoodState global_botMood;             // Now a global definitionc
// Return mood string
const char* getCurrentMood(void) {
   // return getMoodString(botMood);   // OR moodToString(), whichever exists
}

void initGUIBackend(void) {
    create_responses_database();
    create_farewell_database();
    create_stories_database();
    add_more_apocalyptic_stories();
    initDatabase();
    initializeResponseTemplates();
    initEnhancedCognition();
    initFatigueSystem();
    initContextMemory();
    initCuriosity(1);
    initEmotionalSystem();
    initDreamSystem();

    int layers = 3;
    int inputs[3] = {64,64,64};
    int neurons[3] = {32,32,32};
    global_network = createNeuralNetwork(layers, inputs, neurons);

    global_botMood = initMoodState();
}

void cleanupGUIBackend(void) {
    closeDatabase();
}

#endif

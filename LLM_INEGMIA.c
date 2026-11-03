/* 
 * LLM AI ENGINE OUTLINE   
 * Project: INEGMIA 
 * 2025/09/22 12:53:41 - Base line functions only. 
 * 
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
 */
 
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sqlite3.h> 
#define MAX_INPUT_LENGTH 1000
#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 100
#define MAX_TOKEN_LENGTH 50
#define MAX_VOCAB_SIZE 10000
#define MAX_GENERATED_LENGTH 500
#define CREATIVITY_THRESHOLD 0.7f
#define MIN_ACTIVATION 0.1f
#define MAX_WORD_COMBINATIONS 20

sqlite3 *db;
// Define the structure for a neural network layer
typedef struct {
    int inputs;
    int neurons;
    double *weights;
    double *bias;
    double *output;
} Layer;

// Define the structure for the neural network
typedef struct {
    int layers;
    Layer *layer;
} NeuralNetwork;


// Function to sigmoid activation
double sigmoid(double x) {
	    if (isnan(x)) return NAN;
    if (isinf(x)) return (x > 0) ? 1.0f : 0.0f;
    
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    
    if (fabsf(x) > 5.0f) {
        return 0.5f * (1.0f + tanhf(0.5f * x));
    }
    
    if (x >= 0.0f) {
        float exp_neg_x = expf(-x);
        return 1.0f / (1.0f + exp_neg_x);
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
    if (x >= 0.96) return 1 / (1 + exp(-x));
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

char** tokenizeInput(const char* input, int* tokenCount) {
    char** tokens = malloc(MAX_TOKENS * sizeof(char*));
    *tokenCount = 0;
    
    char* inputCopy = malloc(strlen(input) + 1);
    strcpy(inputCopy, input);
    
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

char* generateTokenizedResponse(char** tokens, int tokenCount) {
    char* response = malloc(MAX_INPUT_LENGTH);
    
    int isQuestion = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "what") == 0 || strcmp(tokens[i], "how") == 0 || 
            strcmp(tokens[i], "why") == 0 || strcmp(tokens[i], "when") == 0 ||
            strcmp(tokens[i], "where") == 0 || strcmp(tokens[i], "who") == 0) {
            isQuestion = 1;
            break;
        }
    }
    
    int isGreeting = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "hello") == 0 || strcmp(tokens[i], "hi") == 0 || 
            strcmp(tokens[i], "yo") == 0 || strcmp(tokens[i], "Boo!") == 0 ||
            strcmp(tokens[i], "hey") == 0 || strcmp(tokens[i], "good") == 0) {
            isGreeting = 1;
            break;
        }
    }
    
    int isPositive = 0, isNegative = 0;
    for (int i = 0; i < tokenCount; i++) {
        if (strcmp(tokens[i], "good") == 0 || strcmp(tokens[i], "great") == 0 ||
            strcmp(tokens[i], "wonderful") == 0 || strcmp(tokens[i], "cool") == 0 || 
            strcmp(tokens[i], "awesome") == 0 || strcmp(tokens[i], "happy") == 0) {
            isPositive = 1;
        }
        if (strcmp(tokens[i], "bad") == 0 || strcmp(tokens[i], "sad") == 0 || 
            strcmp(tokens[i], "shit") == 0 || strcmp(tokens[i], "fuck") == 0 ||
            strcmp(tokens[i], "angry") == 0 || strcmp(tokens[i], "terrible") == 0) {
            isNegative = 1;
        }
    }
    
    if (isGreeting) {
        strcpy(response, "Hello! It's nice to chat with you. How can I help you today?");
    } else if (isQuestion) {
        strcpy(response, "That's an interesting question. Let me think about that...");
    } else if (isPositive) {
        strcpy(response, "That sounds wonderful! I'm glad to hear positive things.");
    } else if (isNegative) {
        strcpy(response, "I'm sorry to hear that. Is there anything I can do to help?");
    } else {
        snprintf(response, MAX_INPUT_LENGTH, "I understand you mentioned ");
        int responseLen = strlen(response);
        for (int i = 0; i < tokenCount && i < 3; i++) {
            if (responseLen + strlen(tokens[i]) + 10 < MAX_INPUT_LENGTH) {
                strcat(response, tokens[i]);
                if (i < tokenCount - 1 && i < 2) strcat(response, ", ");
            }
        }
        strcat(response, ".");
    }
    
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
    strcpy(templates.templates[5], "Your mention of %s reminds me of %s.");
    strcpy(templates.templates[6], "I've learned that %s can be associated with %s.");
    strcpy(templates.templates[7], "From my understanding, %s often leads to thoughts of %s.");
    strcpy(templates.templates[8], "The neural pathways suggest %s connects to %s.");
    strcpy(templates.templates[9], "My processing indicates %s has ties to %s.");
    templates.template_count = 10;
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
    // Use neural network output to influence word selection
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

char* generateNeuralResponse(NeuralNetwork* network, char** inputTokens, int tokenCount) {
    char* response = malloc(MAX_GENERATED_LENGTH);
    response[0] = '\0';
    
    if (tokenCount == 0 || network == NULL) {
        strcpy(response, "I'm processing... let me think about that.");
        return response;
    }
    
    // Run neural network forward pass with input tokens
    for (int layer = 0; layer < network->layers; layer++) {
        if (layer == 0) {
            // Use tokenized input for first layer
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
            // Process subsequent layers
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
    
    // Generate response based on neural network output and word associations
    int finalLayer = network->layers - 1;
    double* finalOutput = network->layer[finalLayer].output;
    int outputSize = network->layer[finalLayer].neurons;
    
    // Find most activated words from input
    char selectedWords[MAX_WORD_COMBINATIONS][MAX_TOKEN_LENGTH];
    int selectedCount = 0;
    
    for (int i = 0; i < tokenCount && selectedCount < 5; i++) {
        float activation = neuralActivation(finalOutput, outputSize, inputTokens[i]);
        if (activation > CREATIVITY_THRESHOLD) {
            strcpy(selectedWords[selectedCount], inputTokens[i]);
            selectedCount++;
        }
    }
    
    // If no high-activation words, select randomly
    if (selectedCount == 0 && tokenCount > 0) {
        int randomIdx = rand() % tokenCount;
        strcpy(selectedWords[0], inputTokens[randomIdx]);
        selectedCount = 1;
    }
    
    // Generate associations for selected words
    char associatedWords[MAX_WORD_COMBINATIONS][MAX_TOKEN_LENGTH];
    int associatedCount = 0;
    
    for (int i = 0; i < selectedCount; i++) {
        char* associated = findAssociatedWord(selectedWords[i]);
        if (associated != NULL && associatedCount < MAX_WORD_COMBINATIONS) {
            strcpy(associatedWords[associatedCount], associated);
            associatedCount++;
        } else {
            // Generate a random association from vocabulary
            if (vocab.size > 0) {
                int randomVocab = rand() % vocab.size;
                strcpy(associatedWords[associatedCount], vocab.tokens[randomVocab].word);
                associatedCount++;
            }
        }
    }
    
    // Create response using templates and neural-influenced word selection
    if (selectedCount > 0) {
        float creativity = finalOutput[0]; // Use first output as creativity measure
        
        if (creativity > 0.8f && associatedCount > 0) {
            // High creativity - use complex template
            int templateIdx = rand() % templates.template_count;
            if (associatedCount > 0) {
                snprintf(response, MAX_GENERATED_LENGTH, templates.templates[templateIdx], 
                        selectedWords[0], associatedWords[0]);
            } else {
                snprintf(response, MAX_GENERATED_LENGTH, 
                        "The neural patterns suggest %s creates interesting possibilities.", 
                        selectedWords[0]);
            }
        } else if (creativity > 0.5f) {
            // Medium creativity - combine words creatively
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "I'm analyzing %s and it generates thoughts about", selectedWords[0]);
            
            for (int i = 0; i < associatedCount && i < 3; i++) {
                strcat(response, " ");
                strcat(response, associatedWords[i]);
                if (i < associatedCount - 1 && i < 2) strcat(response, ",");
            }
            strcat(response, ". What's your perspective on these connections?");
        } else {
            // Low creativity - simple response
            snprintf(response, MAX_GENERATED_LENGTH, 
                    "I understand you're discussing %s. My neural processing suggests this is worth exploring further.", 
                    selectedWords[0]);
        }
    } else {
        // Fallback response with neural influence
        float randomness = finalOutput[outputSize > 1 ? 1 : 0];
        if (randomness > 0.7f) {
            strcpy(response, "My neural networks are generating some interesting patterns from your input. Could you elaborate?");
        } else if (randomness > 0.4f) {
            strcpy(response, "The synaptic pathways are processing... I'm curious about your thoughts on this topic.");
        } else {
            strcpy(response, "My cognitive processes are analyzing your message. What would you like to explore?");
        }
    }
    
    return response;
}

void learnAssociationsFromConversation(char** inputTokens, int inputCount, 
                                     char** responseTokens, int responseCount) {
    // Create associations between input and response words
    for (int i = 0; i < inputCount; i++) {
        for (int j = 0; j < responseCount; j++) {
            // Calculate association strength based on neural network confidence
            float strength = 0.5f + ((float)rand() / RAND_MAX) * 0.3f;
            addWordAssociation(inputTokens[i], responseTokens[j], strength);
        }
    }
    
    // Create associations within input tokens
    for (int i = 0; i < inputCount - 1; i++) {
        for (int j = i + 1; j < inputCount; j++) {
            float strength = 0.4f + ((float)rand() / RAND_MAX) * 0.2f;
            addWordAssociation(inputTokens[i], inputTokens[j], strength);
        }
    }
}

void updateNeuralCreativity(NeuralNetwork* network, float feedbackScore) {
    // Adjust neural network weights based on creativity feedback
    // This is a simple learning adjustment
    if (network == NULL || network->layers == 0) return;
    
    float learningRate = 0.01f;
    float adjustment = (feedbackScore - 0.5f) * learningRate;
    
    // Adjust final layer bias to influence creativity
    int finalLayer = network->layers - 1;
    for (int i = 0; i < network->layer[finalLayer].neurons; i++) {
        network->layer[finalLayer].bias[i] += adjustment * ((float)rand() / RAND_MAX - 0.5f);
        
        // Keep bias in reasonable range
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
        double avgOutput = 0;
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
    
    if (wordNet.count > 0) {
        printf("Recent associations: ");
        int start = wordNet.count > 3 ? wordNet.count - 3 : 0;
        for (int i = start; i < wordNet.count; i++) {
            printf("'%s↔%s' ", wordNet.associations[i].word1, wordNet.associations[i].word2);
        }
        printf("\n");
    }
    printf("=======================================\n");
}


void create_responses_database() {
    sqlite3* db;
    char* errorMessage;

    // Open the database
    int rc = sqlite3_open("responses.db", &db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return;
    }

    // Create table
    const char* createTableQuery = 
        "CREATE TABLE IF NOT EXISTS responses ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "input TEXT UNIQUE,"
        "response TEXT"
        ");";

    rc = sqlite3_exec(db, createTableQuery, NULL, NULL, &errorMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    }

    // Conversational questions and responses
    const char* questions[][2] = {
        {"What is your name?", "I am A AI assistant."}, {"Hello?", "how are you today?"}, 
        {"How are you?", "I am doing well, thank you."}, {"hello", "HI, and welcome."},
        {"Whats your purpose?", "I am here to assist you. I am not really sure?"},
        {"Whats the weather like?", "I am not aware of the current weather."},
        {"yo tell me a joke", "okay, What do you call a fake noodle? An impasta! bau-dump-bump!"},
        {"can you tell me a joke","Here is a random joke: Why did the bicycle not stand up by itself? Because it was two-tired!"}, 
        {"Can you tell me a joke?", "Here is one: Why dont scientists trust atoms? Because they make up everything."},
        {"How old are you?", "I dont have an age."}, {"what type of AI are you?", "Reactive data LLM prototype."}, 
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
        {"Hi, how are you?", "I am doing so well, thanks for asking!"}, {"Nice to meet you!", "Likewise!"}, 
        {"What is your favorite subject?", "Math."}, {"Where are you from?", "The internet."}, 
        {"What is new with you?", "Not much, just here chatting with you i guess."}, 
        {"How was your weekend?", "It was great, thanks for asking!"}, 
        {"I am just really sad today", "That is aweful sorry about that. They cant all be winners now can they?"},
        
    };


    int numQuestions = sizeof(questions) / sizeof(questions[0]);
    int querySize = 1024 + (numQuestions * 50); // rough estimate of query size

sqlite3_stmt* stmt;

const char* query = "INSERT OR IGNORE INTO responses (input, response) VALUES (?, ?)";

rc = sqlite3_prepare_v2(db, query, -1, &stmt, 0);
if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db));
    return;
}

for (int i = 0; i < numQuestions; i++) {
    rc = sqlite3_bind_text(stmt, 1, questions[i][0], -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL bind error: %s\n", sqlite3_errmsg(db));
        return;
    }

    rc = sqlite3_bind_text(stmt, 2, questions[i][1], -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL bind error: %s\n", sqlite3_errmsg(db));
        return;
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        return;
    }

    sqlite3_reset(stmt);
}
sqlite3_finalize(stmt);
sqlite3_close(db); 
}

void initDatabase() {
    sqlite3_open("responses.db", &db);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, input TEXT, response TEXT)", NULL, NULL, NULL);
    
}

void closeDatabase() { 
    sqlite3_close(db);
}

char *getResponseFromDatabase(char *input) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf("SELECT response FROM responses WHERE input = '%s'", input);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        return NULL;
    }

    sqlite3_bind_text(stmt, 1, input, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const unsigned char *response = sqlite3_column_text(stmt, 0);
        char *responseStr = (char *)malloc(strlen((const char *)response) + 1);
        strcpy(responseStr, (const char *)response);
        sqlite3_finalize(stmt);
        sqlite3_free(sql);
        return responseStr;
    }

    sqlite3_finalize(stmt);
    sqlite3_free(sql);
    return NULL;
}

// Function to create a new neural network layer
Layer *createLayer(int inputs, int neurons) {
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    layer->inputs = inputs;
    layer->neurons = neurons;
    layer->weights = (double *)malloc(inputs * neurons * sizeof(double));
    layer->bias = (double *)malloc(neurons * sizeof(double));
    layer->output = (double *)malloc(neurons * sizeof(double));

    // Initialize weights and bias randomly
    srand(time(NULL));
    for (int i = 0; i < inputs * neurons; i++) {
        layer->weights[i] = (double)rand() / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < neurons; i++) {
        layer->bias[i] = (double)rand() / RAND_MAX * 2 - 1;
    }

    return layer;
}

// Function to create a new neural network
NeuralNetwork *createNeuralNetwork(int layers, int *inputs, int *neurons) {
    NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    network->layers = layers;
    network->layer = (Layer *)malloc(layers * sizeof(Layer));

    for (int i = 0; i < layers; i++) {
        if (i == 0) {
            network->layer[i] = *createLayer(inputs[i], neurons[i]);
        } else {
            network->layer[i] = *createLayer(neurons[i-1], neurons[i]);
        }
    }

    return network;
}

// Function to train the neural network
void trainNeuralNetwork_output(NeuralNetwork *network, char **data, int dataSize) {
    double learningRate = 0.1;
    int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < dataSize; j++) {
            for (int k = 0; k < network->layers; k++) {
                if (k == 0) {
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        double sum = 0;
                        for (int m = 0; m < strlen(data[j]); m++) sum += (double)data[j][m] * network->layer[k].weights[m * network->layer[k].neurons + l];
                        sum += network->layer[k].bias[l];
                        network->layer[k].output[l] = sigmoid(sum);
                    }
                } else {
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        double sum = 0;
                        for (int m = 0; m < network->layer[k-1].neurons; m++) sum += network->layer[k-1].output[m] * network->layer[k].weights[m * network->layer[k].neurons + l];
                        sum += network->layer[k].bias[l];
                        network->layer[k].output[l] = sigmoid(sum);
                    }
                }
            }
            printf("Input: %s, Output: ", data[j]);
            for (int l = 0; l < network->layer[network->layers - 1].neurons; l++) printf("%.2f ", network->layer[network->layers - 1].output[l]);
            printf("\n");
            // Backward pass and update weights and bias
        }
    }
}

void trainNeuralNetwork(NeuralNetwork *network, char **data, int dataSize) {
    // Define the learning rate and number of iterations
    double learningRate = 0.1;
    int iterations = 1000;

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < dataSize; j++) {
            // Forward pass
            for (int k = 0; k < network->layers; k++) {
                if (k == 0) {
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        double sum = 0;
                        for (int m = 0; m < strlen(data[j]); m++) {
                            sum += (double)data[j][m] * network->layer[k].weights[m * network->layer[k].neurons + l];
                        }
                        sum += network->layer[k].bias[l];
                        network->layer[k].output[l] = sigmoid(sum);
                    }
                } else {
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        double sum = 0;
                        for (int m = 0; m < network->layer[k-1].neurons; m++) {
                            sum += network->layer[k-1].output[m] * network->layer[k].weights[m * network->layer[k].neurons + l];
                        }
                        sum += network->layer[k].bias[l];
                        network->layer[k].output[l] = sigmoid(sum);
                    }
                }
            }

            // Backward pass
            for (int k = network->layers - 1; k >= 0; k--) {
                if (k == network->layers - 1) {
                    // Calculate the error
                    double error = 0;
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        error += pow(network->layer[k].output[l] - 1, 2);
                    }

                    // Update the weights and bias
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        for (int m = 0; m < (k == 0 ? strlen(data[j]) : network->layer[k-1].neurons); m++) {
                            if (k == 0) {
                                network->layer[k].weights[m * network->layer[k].neurons + l] -= learningRate * 2 * (network->layer[k].output[l] - 1) * network->layer[k].output[l] * (1 - network->layer[k].output[l]) * (double)data[j][m];
                            } else {
                                network->layer[k].weights[m * network->layer[k].neurons + l] -= learningRate * 2 * (network->layer[k].output[l] - 1) * network->layer[k].output[l] * (1 - network->layer[k].output[l]) * network->layer[k-1].output[m];
                            }
                        }
                        network->layer[k].bias[l] -= learningRate * 2 * (network->layer[k].output[l] - 1) * network->layer[k].output[l] * (1 - network->layer[k].output[l]);
                    }
                } else {
                    // Calculate the error
                    double error = 0;
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        error += pow(network->layer[k].output[l] - 1, 2);
                    }

                    // Update the weights and bias
                    for (int l = 0; l < network->layer[k].neurons; l++) {
                        for (int m = 0; m < network->layer[k-1].neurons; m++) {
                            network->layer[k].weights[m * network->layer[k].neurons + l] -= learningRate * 2 * (network->layer[k].output[l] - 1) * network->layer[k].output[l] * (1 - network->layer[k].output[l]) * network->layer[k-1].output[m];
                        }
                        network->layer[k].bias[l] -= learningRate * 2 * (network->layer[k].output[l] - 1) * network->layer[k].output[l] * (1 - network->layer[k].output[l]);
                    }
                }
            }
        }
    }
}
// do presistance of database engine traing 
void preprocessInput(char *input) {
    // Convert to lowercase
    for (int i = 0; i < strlen(input); i++) {
        input[i] = tolower(input[i]);
    }

    // Remove punctuation
    for (int i = 0; i < strlen(input); i++) {
        if (!isalnum(input[i]) && !isspace(input[i])) {
            input[i] = ' ';
        }
    }

    // Remove extra spaces
    int writeIndex = 0;
    for (int i = 0; i < strlen(input); i++) {
        if (input[i] != ' ' || (i > 0 && input[i-1] != ' ')) {
            input[writeIndex++] = input[i];
        }
    }
    input[writeIndex] = '\0';
}

void preprocessResponse(char *response) {
    // Convert to lowercase
    for (int i = 0; i < strlen(response); i++) {
        response[i] = tolower(response[i]);
    }

    // Remove punctuation
    for (int i = 0; i < strlen(response); i++) {
        if (!isalnum(response[i]) && !isspace(response[i])) {
            response[i] = ' ';
        }
    }

    // Remove extra spaces
    int writeIndex = 0;
    for (int i = 0; i < strlen(response); i++) {
        if (response[i] != ' ' || (i > 0 && response[i-1] != ' ')) {
            response[writeIndex++] = response[i];
        }
    }
    response[writeIndex] = '\0';
}

char *generateResponse(char *input) {
    // Simple response generation
    char *response = (char *)malloc(MAX_INPUT_LENGTH);
    sprintf(response, "You said: %s", input);
    return response;
}

float calculateLoss(char *response, char *generatedResponse) {
    // Simple loss calculation
    int diffCount = 0;
    for (int i = 0; i < strlen(response); i++) {
        if (response[i] != generatedResponse[i]) {
            diffCount++;
        }
    }
    return (float)diffCount / strlen(response);
}

void updateNeuralNetwork(float loss) {
    // Simple neural network update
    //printf("Loss: %f\n", loss);
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

        char *preprocessedInput = (char *)malloc(strlen((const char *)input) + 1);
        strcpy(preprocessedInput, (const char *)input);

        char *preprocessedResponse = (char *)malloc(strlen((const char *)response) + 1);
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
int main() {
    int layers = 2;
    int inputs[] = {10, 10};
    int neurons[] = {10, 1};

    NeuralNetwork *network = createNeuralNetwork(layers, inputs, neurons);

    char *data[] = {"The","neon", "Fox", "jumps", "over","A", "Firewall", "Hello", "World", "This", "is", "a", "hacking","test"};
    int dataSize = sizeof(data) / sizeof(data[0]);
    create_responses_database(); 
    initDatabase();
    trainDatabase(db);
    char input[MAX_INPUT_LENGTH];
    initializeResponseTemplates();
    srand(time(NULL)); // Initialize random seed for creativity

   while (1) {
        printf("\n(or 'quit' to exit, 'vocab', 'neural' for stats) \nEnter your input:");
        fgets(input, MAX_INPUT_LENGTH, stdin);
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "quit") == 0) {
            break;
        }
        if (strcmp(input, "vocab") == 0) {
            showVocabStats();
            continue;
        }
        if (strcmp(input, "run test") == 0) {
            trainNeuralNetwork(network, data, dataSize);
            trainNeuralNetwork_output(network, data, dataSize);
            continue;
        }
       if (strcmp(input, "clear") == 0) { 
		   system("clear"); 
	   }
       if (strcmp(input, "neural") == 0) {
            showNeuralStats(network);
            continue;
        }
        if (strcmp(input, "creative") == 0) {
            // Force high creativity mode
            updateNeuralCreativity(network, 1.0f);
            printf("Neural network creativity boosted!\n");
            continue;
        }
        // Tokenize input for analysis
        char** tokens;
        int tokenCount;
        tokens = tokenizeInput(input, &tokenCount);
        
        printf("Tokens found: [");
        for (int i = 0; i < tokenCount; i++) {
            printf("'%s'", tokens[i]);
            if (i < tokenCount - 1) printf(", ");
        }
        printf("]\n");
       // Try neural network generated response
       char* neuralResponse = generateNeuralResponse(network, tokens, tokenCount);
    
    // Learn from the neural interaction
    char** responseTokens;
    int responseTokenCount;
    responseTokens = tokenizeInput(neuralResponse, &responseTokenCount);
    learnAssociationsFromConversation(tokens, tokenCount, responseTokens, responseTokenCount);
    
    printf("Response (Neural): %s\n", neuralResponse);
    
    // Update creativity based on response length (simple feedback)
    float feedbackScore = strlen(neuralResponse) > 50 ? 0.8f : 0.3f;
    updateNeuralCreativity(network, feedbackScore);
    
    freeTokens(responseTokens, responseTokenCount);
    free(neuralResponse);
    
        // Try tokenized database search first
        char* tokenResponse = getTokenizedResponseFromDatabase(input);
        if (tokenResponse != NULL) {
            printf("Response (Smart Match): %s\n", tokenResponse);
            learnFromTokens(input, tokenResponse);
            free(tokenResponse);
        } else {
            // Try exact match from your original function
            char* exactResponse = getResponseFromDatabase(input);
            if (exactResponse != NULL) {
                printf("Response (Exact Match): %s\n", exactResponse);
                learnFromTokens(input, exactResponse);
                free(exactResponse);
            } else {
                // Generate contextual response using tokens
                char* generatedResponse = generateTokenizedResponse(tokens, tokenCount);
                printf("Response (Generated): %s\n", generatedResponse);
                learnFromTokens(input, generatedResponse);
                
                // Your existing learning code
                trainDatabase(db);
                preprocessInput(input);
                char* fallbackResponse = generateResponse(input);
                float loss = calculateLoss(input, fallbackResponse);
                updateNeuralNetwork(loss);
                free(fallbackResponse);
                free(generatedResponse);
            }
        }
        
        freeTokens(tokens, tokenCount);
              char *response = getResponseFromDatabase(input);
        if (response != NULL) {
            printf("Response: %s\n", response);
            free(response);
        } else {
        trainDatabase(db); 
        preprocessInput(input);
        char *response_pass2 = generateResponse(input);
        printf("Response: %s\n", response_pass2);
        float loss = calculateLoss(input, response_pass2);
        updateNeuralNetwork(loss);
        free(response);           
        }
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

    return 0;
}


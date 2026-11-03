// evialin.h
#ifndef EVIALIN_H
#define EVIALIN_H

// 1. All necessary includes (optional, often better in .c files)

// 2. The custom type definitions (MUST be here)
// (Copy the actual struct/enum definitions from Evialin_console.c)
typedef enum {
    MOOD_NEUTRAL,
    MOOD_HAPPY,
    // ... all other MoodState enum values
} MoodState;

typedef struct {
    // ... all members of the NeuralNetwork struct
} NeuralNetwork;


extern NeuralNetwork *global_network;
extern MoodState global_botMood;
// The function prototypes used by evialin_gui.c
//void processUserInput(NeuralNetwork* network, MoodState* botMood, const char* input); 
void initGUIBackend(void); 
void cleanupGUIBackend(void); 
char* processUserInput(NeuralNetwork* network, MoodState* botMood, const char* input);
// Globals declared in evialin.h (or implicitly global in Evialin_console.c)

#endif

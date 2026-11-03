/*
Project: CHAI COFFEE AI OUTLINE
Version: 1st edition (Fixed - All Warnings Resolved)

NOTES:
Make command with gcc:
gcc chai.c -o chai -lm -lsqlite3 -std=c99

Include libraries Ubuntu/Debian-based Systems:
sudo apt update
sudo apt install sqlite3 libsqlite3-dev
*/

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 
#include <sqlite3.h> 
#include <ctype.h>
#include <unistd.h> 

#define MAX_INPUT 256
#define MAX_RESPONSE 512
#define VOCAB_SIZE 1000
#define HIDDEN_LAYER_SIZE 64

// List of explicitly allowed safe commands
static const char *safe_commands[] = {
    "clear", "ls", "dir", "ls -la", "ls /", "ls /home/", 
    "ls /bin/", "pwd", "htop", "atop", "ps", "whoami",
    "jobs", "nano", "head", "tail", "df", "df -h",
    "free", "free -h", "uptime", "date"
};

#define SAFE_COMMANDS_COUNT (sizeof(safe_commands) / sizeof(safe_commands[0]))

// Neural Network Structure
typedef struct {
    float input_weights[VOCAB_SIZE][HIDDEN_LAYER_SIZE];
    float hidden_weights[HIDDEN_LAYER_SIZE][1];
    float input_bias[HIDDEN_LAYER_SIZE];
    float output_bias;
} NeuralNetwork;

// Pre-training data structure
typedef struct {
    const char* input;
    const char* response;
} TrainingPair;

// Function Prototypes
static void debug_system_info(void);
static float sigmoid(float x);
static void initialize_network(NeuralNetwork* nn);
static float predict_response(NeuralNetwork* nn, int* input_tokens, int token_count);
static int tokenize_input(char* input, int* tokens);
static int setup_database(sqlite3** db);
static void save_training_data(sqlite3* db, const char* input, const char* response);
static char* retrieve_response(sqlite3* db, const char* input);
static int preload_training_data(sqlite3* db);
static char* safe_strdup(const char* str);
static int execute_safe_command(char *input);
static int is_valid_input_chars(const char *input);
static int is_safe_command(const char *command);
static int chatbot_goodbye(void);
static void run_chatbot_greeting(const char *database_path);
static int populate_greetings_database(sqlite3 *db);
static char* get_random_greeting(sqlite3 *db);

// Pre-defined training data
static TrainingPair initial_training_data[] = {
    {"hello", "Hi there! How are you?"},
    {"clear", "Okay, I will clear the screen now."},
    {"cls", "Okay, I will clear the screen now."},
    {"ls", "Listing all files I can find for that path."},
    {"how are you?", "I am doing good. Thanks for asking!"},
    {"your name?", "Chai Coffee is my name"},
    {"how are you", "I'm doing well, thank you for asking!"},
    {"what is your name", "I'm an AI called Chai Coffee. Nice to meet you!"},
    {"what is your last name", "My last name is Coffee."},
    {"what is your first name", "Chai, and my last name is Coffee"},
    {"what is your 1st name", "Chai"},
    {"what is your whole name", "Chai Coffee is my full name"},
    {"tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"},
    {"help", "I can chat with you and learn from our conversations."},
    {"bye", "Goodbye! It was nice talking to you."},
    {"quit", "Goodbye, It was nice talking with you."},
    {"exit", "Goodbye"}, 
    {"thank you", "You're welcome!"},
    {"weather", "I'm an AI, so I can't check the weather, but I hope it's nice where you are!"},
    {"who created you", "I was created by Johnny as a demonstration of an AI bash shell using C and SQLite."},
    {"what can you do", "I can have conversations, learn from our interactions, and provide responses based on our training."},
    {"are you a shell", "Yes, I am a shell framework with AI capabilities"}
};

// Manual implementation of strdup for ANSI C compatibility
static char* safe_strdup(const char* str) {
    if (str == NULL) return NULL;
    
    size_t len = strlen(str) + 1;
    char* new_str = malloc(len);
    
    if (new_str) {
        memcpy(new_str, str, len);
    }
    
    return new_str;
}

// Optimized Sigmoid Activation Function
static float sigmoid(float x) {
    // Handle special values first
    if (isnan(x)) return NAN;
    if (isinf(x)) return (x > 0) ? 1.0f : 0.0f;
    
    // Clamp extreme values
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    
    // Use tanh approximation for moderate values
    if (fabsf(x) > 5.0f) {
        return 0.5f * (1.0f + tanhf(0.5f * x));
    }
    
    // Standard method for small values
    if (x >= 0.0f) {
        float exp_neg_x = expf(-x);
        return 1.0f / (1.0f + exp_neg_x);
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

// Function to pre-train the database
static int preload_training_data(sqlite3* db) {
    sqlite3_stmt* stmt;
    const char* sql = 
        "INSERT OR REPLACE INTO training_data (input, response, frequency) "
        "VALUES (?, ?, COALESCE((SELECT frequency + 1 FROM training_data WHERE input = ?), 1))";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Preparation failed: %s\n", sqlite3_errmsg(db));
        return rc;
    }
    
    // Number of pre-defined training pairs
    size_t training_data_count = sizeof(initial_training_data) / sizeof(initial_training_data[0]);
    
    // Insert each training pair
    for (size_t i = 0; i < training_data_count; i++) {
        // Reset the statement
        sqlite3_reset(stmt);
        
        // Bind parameters
        sqlite3_bind_text(stmt, 1, initial_training_data[i].input, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, initial_training_data[i].response, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, initial_training_data[i].input, -1, SQLITE_STATIC);
        
        // Execute the statement
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Insertion failed for %s: %s\n", 
                    initial_training_data[i].input, 
                    sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return rc;
        }
    }
    
    // Finalize the statement
    sqlite3_finalize(stmt);
    
    printf("Pre-training data loaded successfully.\n");
    return SQLITE_OK;
}

// Initialize Neural Network
static void initialize_network(NeuralNetwork* nn) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            nn->input_weights[i][j] = (float)rand() / RAND_MAX - 0.5f;
        }
    }
    
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->hidden_weights[i][0] = (float)rand() / RAND_MAX - 0.5f;
        nn->input_bias[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    nn->output_bias = (float)rand() / RAND_MAX - 0.5f;
}

// Neural Network Prediction
static float predict_response(NeuralNetwork* nn, int* input_tokens, int token_count) {
    float hidden_layer[HIDDEN_LAYER_SIZE] = {0};
    
    for (int i = 0; i < token_count && i < VOCAB_SIZE; i++) {
        if (input_tokens[i] >= 0 && input_tokens[i] < VOCAB_SIZE) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                hidden_layer[j] += nn->input_weights[input_tokens[i]][j];
            }
        }
    }
    
    for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
        hidden_layer[j] = sigmoid(hidden_layer[j] + nn->input_bias[j]);
    }
    
    float output = 0;
    for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
        output += hidden_layer[j] * nn->hidden_weights[j][0];
    }
    output = sigmoid(output + nn->output_bias);
    
    return output;
}

// Input Tokenization
static int tokenize_input(char* input, int* tokens) {
    char* token = strtok(input, " ");
    int count = 0;
    
    while (token != NULL && count < VOCAB_SIZE) {
        tokens[count] = (int)(strlen(token) % VOCAB_SIZE);
        token = strtok(NULL, " ");
        count++;
    }
    
    return count;
}

// Setup Database Function
static int setup_database(sqlite3** db) {
    int rc = sqlite3_open("chatbot.db", db);
    if (rc) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(*db));
        return rc;
    }
    
    const char* create_table_sql = 
        "CREATE TABLE IF NOT EXISTS training_data ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "input TEXT NOT NULL,"
        "response TEXT NOT NULL,"
        "frequency INTEGER DEFAULT 1)";
    
    char* error_message = 0;
    rc = sqlite3_exec(*db, create_table_sql, 0, 0, &error_message);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", error_message);
        sqlite3_free(error_message);
        sqlite3_close(*db);
        return rc;
    }
    
    // Load pre-training data
    rc = preload_training_data(*db);
    if (rc != SQLITE_OK) {
        sqlite3_close(*db);
        return rc;
    }
    
    return SQLITE_OK;
}

// Save Training Data
static void save_training_data(sqlite3* db, const char* input, const char* response) {
    sqlite3_stmt* stmt;
    const char* sql = 
        "INSERT OR REPLACE INTO training_data (input, response, frequency) "
        "VALUES (?, ?, COALESCE((SELECT frequency + 1 FROM training_data WHERE input = ?), 1))";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Preparation failed: %s\n", sqlite3_errmsg(db));
        return;
    }
    
    sqlite3_bind_text(stmt, 1, input, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, response, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, input, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Execution failed: %s\n", sqlite3_errmsg(db));
    }
    
    sqlite3_finalize(stmt);
}

// Improved retrieve_response function
static char* retrieve_response(sqlite3* db, const char* input) {
    sqlite3_stmt* stmt;
    char* best_response = NULL;
    
    // More flexible SQL query with partial matching
    const char* sql = 
        "SELECT response, frequency FROM training_data "
        "WHERE lower(input) LIKE lower(?) OR lower(input) LIKE lower(?) "
        "ORDER BY frequency DESC LIMIT 1";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Preparation failed: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    
    // Create two search patterns
    char pattern1[MAX_INPUT];
    char pattern2[MAX_INPUT];
    
    // Pattern 1: Exact match
    snprintf(pattern1, sizeof(pattern1), "%s", input);
    
    // Pattern 2: Partial match
    snprintf(pattern2, sizeof(pattern2), "%%%s%%", input);
    
    sqlite3_bind_text(stmt, 1, pattern1, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, pattern2, -1, SQLITE_STATIC);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* response = sqlite3_column_text(stmt, 0);
        
        // Manual memory allocation
        best_response = malloc(strlen((const char*)response) + 1);
        if (best_response) {
            strcpy(best_response, (const char*)response);
        }
    }
    
    sqlite3_finalize(stmt);
    return best_response;
}

// Function to validate input characters
static int is_valid_input_chars(const char *input) {
    for (int i = 0; input[i]; i++) {
        // Allow spaces, forward slashes, and alphanumeric characters, dash, underscore
        if (!isalnum(input[i]) && input[i] != '-' && input[i] != '_' && 
            input[i] != ' ' && input[i] != '/') {
            return 0;
        }
    }
    return 1;
}

// Function to check if command is in safe list
static int is_safe_command(const char *command) {
    for (size_t i = 0; i < SAFE_COMMANDS_COUNT; i++) {
        if (strcmp(command, safe_commands[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Debug function to check what's available
static void debug_system_info(void) {
    printf("=== DEBUG INFO ===\n");
    const char *path = getenv("PATH");
    printf("PATH: %s\n", path ? path : "NOT SET");
    printf("Testing command locations:\n");
    
    const char* test_paths[] = {"/bin/ls", "/usr/bin/ls", "/usr/local/bin/ls"};
    for (size_t i = 0; i < sizeof(test_paths)/sizeof(test_paths[0]); i++) {
        if (access(test_paths[i], X_OK) == 0) {
            printf("  %s: FOUND\n", test_paths[i]);
        } else {
            printf("  %s: NOT FOUND\n", test_paths[i]);
        }
    }
    
    // Quick environment check
    const char* env_vars[] = {"PATH", "USER", "HOME", "SHELL"};
    for (size_t i = 0; i < sizeof(env_vars)/sizeof(env_vars[0]); i++) {
        const char *env_val = getenv(env_vars[i]);
        printf("%s: %s\n", env_vars[i], env_val ? env_val : "NOT SET");
    }
}

// Simplified execution function with debugging
static int execute_safe_command(char *input) {
    printf("Input received: '%s'\n", input);
    
    // Remove trailing newline if present
    size_t len = strlen(input);
    if (len > 0 && input[len-1] == '\n') {
        input[len-1] = '\0';
        printf("After newline removal: '%s'\n", input);
    }
    
    // Validate input characters
    if (!is_valid_input_chars(input)) {
        fprintf(stderr, "Error: Invalid input characters in '%s'\n", input);
        return 0;
    }
    
    // Check if command is in safe list
    if (!is_safe_command(input)) {
        fprintf(stderr, "Error: Command '%s' not allowed\n", input);
        printf("Available commands:\n");
        for (size_t i = 0; i < SAFE_COMMANDS_COUNT; i++) {
            printf(" %zu:%s ", i+1, safe_commands[i]);
            if (i % 3 == 2) {
                printf("\n");
            }
        }
        printf("\n");
        return 0;
    }
    
    printf("Executing: '%s'\n", input);
    int result = system(input);
    
    if (result == -1) {
        perror("system() call failed");
        return 0;
    }
    
    return 1;
}

// Function to populate the database with greetings
static int populate_greetings_database(sqlite3 *db) {
    const char *create_table_sql = 
        "CREATE TABLE IF NOT EXISTS greetings ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "row_type TEXT,"
        "greeting TEXT);";
    
    const char *delete_existing_sql = 
        "DELETE FROM greetings WHERE row_type = 'defaults_ins';";
    
    // 20 diverse greetings
    const char *greetings[] = {
        "Hello there! How can I help you today?",
        "Greetings! Welcome to our chat service.",
        "Hi! Nice to meet you.",
        "Hey! What brings you here?",
        "Howdy! Ready to chat?",
        "Good day! How are you doing?",
        "Welcome! I'm here to assist you.",
        "Salutations! What can I do for you?",
        "Hi there! Feeling chatty today?",
        "Greetings, human! Let's have a conversation.",
        "Hello, wonderful human! How's your day going?",
        "Hi! I'm your friendly neighborhood chatbot.",
        "Wassup? Ready to chat?",
        "Bonjour! How can I brighten your day?",
        "Hiya! Excited to help you out.",
        "Hey there! What's on your mind?",
        "Greetings from the digital realm!",
        "Hi! I'm all ears and algorithms.",
        "Welcome aboard! Let's get talking.",
        "Hello! Your personal assistant is here."
    };
    
    char *error_message = 0;
    int rc;

    // Create table
    rc = sqlite3_exec(db, create_table_sql, 0, 0, &error_message);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", error_message);
        sqlite3_free(error_message);
        return rc;
    }

    // Delete existing entries for this row
    rc = sqlite3_exec(db, delete_existing_sql, 0, 0, &error_message);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", error_message);
        sqlite3_free(error_message);
        return rc;
    }

    // Prepare insert statement
    sqlite3_stmt *stmt;
    const char *insert_sql = "INSERT INTO greetings (row_type, greeting) VALUES ('defaults_ins', ?);";
    
    // Begin transaction for efficiency
    sqlite3_exec(db, "BEGIN TRANSACTION", 0, 0, 0);

    // Prepare the statement
    rc = sqlite3_prepare_v2(db, insert_sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return rc;
    }

    // Insert all greetings
    for (size_t i = 0; i < sizeof(greetings)/sizeof(greetings[0]); i++) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, greetings[i], -1, SQLITE_STATIC);
        
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Insertion failed: %s\n", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return rc;
        }
    }

    // Finalize the statement and commit transaction
    sqlite3_finalize(stmt);
    sqlite3_exec(db, "COMMIT TRANSACTION", 0, 0, 0);

    return SQLITE_OK;
}

// Function to retrieve a random greeting
static char* get_random_greeting(sqlite3 *db) {
    sqlite3_stmt *stmt;
    char *greeting = NULL;
    int rc;

    // SQL to select a random greeting from defaults_ins row
    const char *sql = "SELECT greeting FROM greetings WHERE row_type = 'defaults_ins' ORDER BY RANDOM() LIMIT 1";
    
    // Prepare the statement
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return NULL;
    }

    // Execute the statement
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        // Get the text of the greeting
        const unsigned char *temp = sqlite3_column_text(stmt, 0);
        
        // Allocate memory and copy the greeting
        if (temp) {
            greeting = safe_strdup((const char*)temp);
        }
    } else {
        fprintf(stderr, "No greeting found or error occurred\n");
    }

    // Finalize the statement
    sqlite3_finalize(stmt);

    return greeting;
}

// Subfunction to run the chatbot greeting demonstration
static void run_chatbot_greeting(const char *database_path) {
    sqlite3 *db;
    char *greeting = NULL;
    int rc;

    // Open the database
    rc = sqlite3_open(database_path, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return;
    }

    // Populate the database with greetings
    rc = populate_greetings_database(db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to populate database\n");
        sqlite3_close(db);
        return;
    }

    // Retrieve and print 1 random greeting
    greeting = get_random_greeting(db);
    if (greeting) {
        printf("AI: %s\n", greeting);
        free(greeting);
    }

    // Close the database
    sqlite3_close(db);
}

static int chatbot_goodbye(void) {
    sqlite3 *db;
    sqlite3_stmt *stmt;
    char *error_message = 0;
    int rc;

    // Open the database
    rc = sqlite3_open("chatbot.db", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    // Create table if not exists
    const char *create_table_sql = 
        "CREATE TABLE IF NOT EXISTS goodbye_exit ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "exit_message TEXT NOT NULL"
        ");";

    rc = sqlite3_exec(db, create_table_sql, 0, 0, &error_message);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error creating table: %s\n", error_message);
        sqlite3_free(error_message);
        sqlite3_close(db);
        return 1;
    }

    // Check if table is empty
    const char *check_empty_sql = "SELECT COUNT(*) FROM goodbye_exit;";
    rc = sqlite3_prepare_v2(db, check_empty_sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare count statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }

    rc = sqlite3_step(stmt);
    int count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    // Insert default messages if table is empty
    if (count == 0) {
        const char *insert_messages_sql = 
            "INSERT INTO goodbye_exit (exit_message) VALUES "
            "('See you later!'), "
            "('Goodbye!'), "
            "('Have a great day!'), "
            "('Until next time!'), "
            "('Take care!'), "
            "('Farewell!'), "
            "('Stay awesome!'), "
            "('Catch you later!'), "
            "('Peace out!'), "
            "('Be well!'), "
            "('Bye now!'), "
            "('Keep smiling!'), "
            "('Take it easy!'), "
            "('So long!'), "
            "('Adios!');";

        rc = sqlite3_exec(db, insert_messages_sql, 0, 0, &error_message);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "SQL error inserting messages: %s\n", error_message);
            sqlite3_free(error_message);
            sqlite3_close(db);
            return 1;
        }
    }

    // Prepare and execute random message selection
    const char *select_sql = "SELECT exit_message FROM goodbye_exit ORDER BY RANDOM() LIMIT 1";
    rc = sqlite3_prepare_v2(db, select_sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare select statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }

    // Get and display random message
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const unsigned char *message = sqlite3_column_text(stmt, 0);
        printf("AI: %s\n", message);
    } else {
        fprintf(stderr, "Failed to retrieve message\n");
    }

    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return 0;
}

// Main Function
int main(void) {
    sqlite3* db;
    NeuralNetwork nn;
    char input[MAX_INPUT];
    int tokens[VOCAB_SIZE];
    
    // Initialize database and neural network
    if (setup_database(&db) != SQLITE_OK) {
        return -1;
    }
    
    initialize_network(&nn);
    run_chatbot_greeting("chatbot.db");
    
    while (1) {
        // Get user input
        printf(">> ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // Remove newline character
        input[strcspn(input, "\n")] = 0;
     
        // Check for exit condition
        if (strcmp(input, "exit") == 0 || strcmp(input, "quit") == 0) {
            break;
        }
        
        if (strcmp(input, "list commands") == 0) {
            printf("Available commands:\n");
            for (size_t i = 0; i < SAFE_COMMANDS_COUNT; i++) {
                printf(" %zu:%s ", i+1, safe_commands[i]);
                if (i % 3 == 2) {
                    printf("\n");
                }
            }
            printf("\n"); 
            continue;
        }
        
        if (strcmp(input, "debug") == 0) {
            debug_system_info();
            continue;
        }
        
        // Convert input to lowercase for better matching
        for (size_t i = 0; input[i]; i++) {
            input[i] = (char)tolower(input[i]);
        }
        
        // Check if input starts with run prefix 
        if (strncmp(input, "run ", 4) == 0) {
            char *command = input + 4;  // skip (run+space) and do safe command logic      
            // Execute the safe command
            execute_safe_command(command);
            continue;
        }
        
        // Tokenize input
        char input_copy[MAX_INPUT];
        strcpy(input_copy, input);
        int token_count = tokenize_input(input_copy, tokens);
        
        // Predict response confidence
        float confidence = predict_response(&nn, tokens, token_count);
        
        // Retrieve response from database
        char* response = retrieve_response(db, input);
        
        if (response) {
            printf("AI: %s (Confidence: %.2f)\n", response, confidence);
            
            // Optional: Train the system
            printf("Is this response helpful? (y/n): ");
            char feedback[10];
            if (fgets(feedback, sizeof(feedback), stdin) != NULL) {
                feedback[strcspn(feedback, "\n")] = 0;
                
                if (strcmp(feedback, "y") == 0) {
                    save_training_data(db, input, response);
                    printf("Thank you for the feedback!\n");
                }
            }
            
            free(response);
        } else {
            printf("AI: I'm not sure how to respond. Can you teach me?\n");
            printf("Enter a suggested response: ");
            char new_response[MAX_RESPONSE];
            if (fgets(new_response, sizeof(new_response), stdin) != NULL) {
                new_response[strcspn(new_response, "\n")] = 0;
                
                // Save new training data
                save_training_data(db, input, new_response);
                printf("AI: Thank you! I've learned something new.\n");
            }
        }
    }

    // Clean up
    sqlite3_close(db);
    
    // Display goodbye message
    chatbot_goodbye();
    
    return 0;
}

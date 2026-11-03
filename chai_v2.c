/*
Project: CHAI COFFEE AI 
Version: 2st edition
* Updates: 
  1. Neural Network Persistence edit, save and load networks. 
  2. Inline keywords for memory status. 
  3. A help menu 
NOTES:
Make command with gcc,
gcc   -c chai_v2.c -o chai_v2 -lm -lcurl -lsqlite3 -std=c99
Geany open souce editor build command, 
gcc   -c "%f" -lm -lcurl -lsqlite3 -std=c99
Include librarys Ubuntu/Debian-based Systems
in bash shell type,
sudo apt update
sudo apt install sqlite3 libsqlite3-dev
*/

#define _XOPEN_SOURCE  // This helps with POSIX extensions
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 
#include <sqlite3.h> 
#include <ctype.h>
#include <unistd.h> 
#include <sys/sysinfo.h>
#define MAX_INPUT 256
#define MAX_RESPONSE 512
#define VOCAB_SIZE 1000
#define HIDDEN_LAYER_SIZE 64
#define SAFE_COMMANDS_COUNT (sizeof(safe_commands) / sizeof(safe_commands[0]))
#define LOW_MEM_THRESHOLD 0.09

int check_memory() {
    struct sysinfo si;
    if (sysinfo(&si) != 0) {
        perror("sysinfo");
        return -1;
    }
    
    double free_ratio = (double)si.freeram / si.totalram;
    return free_ratio < LOW_MEM_THRESHOLD;
}

// Prototypes 
void debug_system_info();

// Manual implementation of strdup for ANSI C compatibility
char* strdup(const char* str) {
    if (str == NULL) return NULL;
    
    size_t len = strlen(str) + 1;  // +1 for null terminator
    char* new_str = malloc(len);
    
    if (new_str) {
        memcpy(new_str, str, len);
    }
    
    return new_str;
}

// Neural Network Structure
typedef struct {
    float input_weights[VOCAB_SIZE][HIDDEN_LAYER_SIZE];
    float hidden_weights[HIDDEN_LAYER_SIZE][1];
    float input_bias[HIDDEN_LAYER_SIZE];
    float output_bias;
} NeuralNetwork;

// Neural Network Database Functions
int setup_neural_network_database(sqlite3* db) {
    char* error_message = 0;
    
    const char* create_tables_sql = 
        "CREATE TABLE IF NOT EXISTS neural_networks ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "    name TEXT UNIQUE,"
        "    vocab_size INTEGER,"
        "    hidden_size INTEGER,"
        "    created_at DATETIME DEFAULT CURRENT_TIMESTAMP"
        ");"
        
        "CREATE TABLE IF NOT EXISTS nn_input_weights ("
        "    network_id INTEGER,"
        "    input_index INTEGER,"
        "    hidden_index INTEGER,"
        "    weight REAL,"
        "    FOREIGN KEY (network_id) REFERENCES neural_networks(id)"
        ");"
        
        "CREATE TABLE IF NOT EXISTS nn_hidden_weights ("
        "    network_id INTEGER,"
        "    hidden_index INTEGER,"
        "    output_index INTEGER,"
        "    weight REAL,"
        "    FOREIGN KEY (network_id) REFERENCES neural_networks(id)"
        ");"
        
        "CREATE TABLE IF NOT EXISTS nn_biases ("
        "    network_id INTEGER,"
        "    layer_type TEXT,"  // 'input' or 'output'
        "    index_value INTEGER,"
        "    bias REAL,"
        "    FOREIGN KEY (network_id) REFERENCES neural_networks(id)"
        ");";
    
    int rc = sqlite3_exec(db, create_tables_sql, 0, 0, &error_message);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error creating NN tables: %s\n", error_message);
        sqlite3_free(error_message);
        return rc;
    }
    
    printf("Neural network database tables created successfully\n");
    return SQLITE_OK;
}

int save_neural_network_sqlite(sqlite3* db, NeuralNetwork* nn, const char* name) {
    sqlite3_stmt* stmt;
    int rc;
    
    printf("Saving neural network '%s'...\n", name);
    
    // Begin transaction
    sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    
    // Delete existing network with same name
    const char* delete_network = "DELETE FROM neural_networks WHERE name = ?;";
    sqlite3_prepare_v2(db, delete_network, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    // Insert network info
    const char* insert_network = 
        "INSERT INTO neural_networks (name, vocab_size, hidden_size) VALUES (?, ?, ?);";
    sqlite3_prepare_v2(db, insert_network, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, VOCAB_SIZE);
    sqlite3_bind_int(stmt, 3, HIDDEN_LAYER_SIZE);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    // Get network ID
    sqlite3_int64 network_id = sqlite3_last_insert_rowid(db);
    
    // Save input weights
    const char* insert_input_weight = 
        "INSERT INTO nn_input_weights (network_id, input_index, hidden_index, weight) VALUES (?, ?, ?, ?);";
    sqlite3_prepare_v2(db, insert_input_weight, -1, &stmt, NULL);
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            sqlite3_bind_int64(stmt, 1, network_id);
            sqlite3_bind_int(stmt, 2, i);
            sqlite3_bind_int(stmt, 3, j);
            sqlite3_bind_double(stmt, 4, nn->input_weights[i][j]);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
    }
    sqlite3_finalize(stmt);
    
    // Save hidden weights
    const char* insert_hidden_weight = 
        "INSERT INTO nn_hidden_weights (network_id, hidden_index, output_index, weight) VALUES (?, ?, ?, ?);";
    sqlite3_prepare_v2(db, insert_hidden_weight, -1, &stmt, NULL);
    
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        sqlite3_bind_int64(stmt, 1, network_id);
        sqlite3_bind_int(stmt, 2, i);
        sqlite3_bind_int(stmt, 3, 0);  // Only one output
        sqlite3_bind_double(stmt, 4, nn->hidden_weights[i][0]);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    
    // Save biases
    const char* insert_bias = 
        "INSERT INTO nn_biases (network_id, layer_type, index_value, bias) VALUES (?, ?, ?, ?);";
    sqlite3_prepare_v2(db, insert_bias, -1, &stmt, NULL);
    
    // Input biases
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        sqlite3_bind_int64(stmt, 1, network_id);
        sqlite3_bind_text(stmt, 2, "input", -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 3, i);
        sqlite3_bind_double(stmt, 4, nn->input_bias[i]);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    
    // Output bias
    sqlite3_bind_int64(stmt, 1, network_id);
    sqlite3_bind_text(stmt, 2, "output", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, 0);
    sqlite3_bind_double(stmt, 4, nn->output_bias);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    // Commit transaction
    sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    
    printf("Neural network '%s' saved successfully with ID: %lld\n", name, network_id);
    return SQLITE_OK;
}

int load_neural_network_sqlite(sqlite3* db, NeuralNetwork* nn, const char* name) {
    sqlite3_stmt* stmt;
    int rc;
    
    printf("Loading neural network '%s'...\n", name);
    
    // Get network info
    const char* get_network = "SELECT id, vocab_size, hidden_size FROM neural_networks WHERE name = ?;";
    sqlite3_prepare_v2(db, get_network, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        fprintf(stderr, "Neural network '%s' not found\n", name);
        sqlite3_finalize(stmt);
        return SQLITE_ERROR;
    }
    
    sqlite3_int64 network_id = sqlite3_column_int64(stmt, 0);
    int vocab_size = sqlite3_column_int(stmt, 1);
    int hidden_size = sqlite3_column_int(stmt, 2);
    sqlite3_finalize(stmt);
    
    // Validate dimensions
    if (vocab_size != VOCAB_SIZE || hidden_size != HIDDEN_LAYER_SIZE) {
        fprintf(stderr, "Network dimensions mismatch: expected %dx%d, found %dx%d\n",
                VOCAB_SIZE, HIDDEN_LAYER_SIZE, vocab_size, hidden_size);
        return SQLITE_ERROR;
    }
    
    // Load input weights
    const char* get_input_weights = 
        "SELECT input_index, hidden_index, weight FROM nn_input_weights WHERE network_id = ?;";
    sqlite3_prepare_v2(db, get_input_weights, -1, &stmt, NULL);
    sqlite3_bind_int64(stmt, 1, network_id);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int input_idx = sqlite3_column_int(stmt, 0);
        int hidden_idx = sqlite3_column_int(stmt, 1);
        float weight = sqlite3_column_double(stmt, 2);
        
        if (input_idx < VOCAB_SIZE && hidden_idx < HIDDEN_LAYER_SIZE) {
            nn->input_weights[input_idx][hidden_idx] = weight;
        }
    }
    sqlite3_finalize(stmt);
    
    // Load hidden weights
    const char* get_hidden_weights = 
        "SELECT hidden_index, weight FROM nn_hidden_weights WHERE network_id = ?;";
    sqlite3_prepare_v2(db, get_hidden_weights, -1, &stmt, NULL);
    sqlite3_bind_int64(stmt, 1, network_id);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int hidden_idx = sqlite3_column_int(stmt, 0);
        float weight = sqlite3_column_double(stmt, 1);
        
        if (hidden_idx < HIDDEN_LAYER_SIZE) {
            nn->hidden_weights[hidden_idx][0] = weight;
        }
    }
    sqlite3_finalize(stmt);
    
    // Load biases
    const char* get_biases = 
        "SELECT layer_type, index_value, bias FROM nn_biases WHERE network_id = ?;";
    sqlite3_prepare_v2(db, get_biases, -1, &stmt, NULL);
    sqlite3_bind_int64(stmt, 1, network_id);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* layer_type = (const char*)sqlite3_column_text(stmt, 0);
        int idx = sqlite3_column_int(stmt, 1);
        float bias = sqlite3_column_double(stmt, 2);
        
        if (strcmp(layer_type, "input") == 0 && idx < HIDDEN_LAYER_SIZE) {
            nn->input_bias[idx] = bias;
        } else if (strcmp(layer_type, "output") == 0 && idx == 0) {
            nn->output_bias = bias;
        }
    }
    sqlite3_finalize(stmt);
    
    printf("Neural network '%s' loaded successfully\n", name);
    return SQLITE_OK;
}

void list_saved_neural_networks(sqlite3* db) {
    sqlite3_stmt* stmt;
    
    const char* query = "SELECT name, vocab_size, hidden_size, created_at FROM neural_networks ORDER BY created_at DESC;";
    sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
    
    printf("\nSaved Neural Networks:\n");
    printf("%-20s %-12s %-12s %-20s\n", "Name", "Vocab Size", "Hidden Size", "Created");
    printf("----------------------------------------------------------------\n");
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        printf("%-20s %-12d %-12d %-20s\n",
               sqlite3_column_text(stmt, 0),
               sqlite3_column_int(stmt, 1),
               sqlite3_column_int(stmt, 2),
               sqlite3_column_text(stmt, 3));
    }
    
    sqlite3_finalize(stmt);
}

// Function Prototypes
float sigmoid(float x);
void initialize_network(NeuralNetwork* nn);
float predict_response(NeuralNetwork* nn, int* input_tokens, int token_count);
int tokenize_input(char* input, int* tokens);
int setup_database(sqlite3** db);
void save_training_data(sqlite3* db, const char* input, const char* response);

// Optimized Sigmoid Function
float sigmoid(float x) {
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
}

// Initialize Neural Network
void initialize_network(NeuralNetwork* nn) {
    srand(time(NULL));
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            nn->input_weights[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        for (int j = 0; j < 1; j++) {
            nn->hidden_weights[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
        nn->input_bias[i] = (float)rand() / RAND_MAX - 0.5;
    }
    
    nn->output_bias = (float)rand() / RAND_MAX - 0.5;
}

// Neural Network Prediction
float predict_response(NeuralNetwork* nn, int* input_tokens, int token_count) {
    float hidden_layer[HIDDEN_LAYER_SIZE] = {0};
    
    for (int i = 0; i < token_count; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            hidden_layer[j] += nn->input_weights[input_tokens[i]][j];
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
int tokenize_input(char* input, int* tokens) {
    char* token = strtok(input, " ");
    int count = 0;
    
    while (token != NULL && count < VOCAB_SIZE) {
        tokens[count] = strlen(token) % VOCAB_SIZE;
        token = strtok(NULL, " ");
        count++;
    }
    
    return count;
}

// Pre-training data structure
typedef struct {
    const char* input;
    const char* response;
} TrainingPair;

// Pre-defined training data
TrainingPair initial_training_data[] = {
    {"hello", "Hi there! How are you?"},
    {"clear", "Okay, I will clear the screen now."},
    {"cls", "Okay, I will clear the screen now. that's if I can find the cls command in /bin path? "},
    {"ls", "listing all files I can find for that path."},
    {"how are you?", "I am doing good. thanks for asking how I am doing."},
    {"your name?", "Chai Coffee is my name"},
    {"how are you", "I'm doing well, thank you for asking!"},
    {"what is your name", "I'm an AI call me chai last name coffee. Nice to meet you!"},
    {"what is your last name", "My last name Coffee."},
    {"what is your first name", "chai and my last name is coffee"},
    {"tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"},
    {"help", "I can chat with you and learn from our conversations."},
    {"bye", "Goodbye! It was nice talking to you."},
    {"quit", "Goodbye, It was nice talking with you."},
    {"exit", "Goodbye"}, 
    {"thank you", "You're welcome!"},
    {"weather", "I'm an AI, so I can't check the weather, but I hope it's nice where you are!"},
    {"who created you", "I was created by Johnny as A demonstration of an AI bash shell using C and SQLite."},
    {"what can you do", "I can have conversations, learn from our interactions, and provide responses based on our training."},
    {"are you a shell", "Yes, I am a sudorandom shell framework"}
};

// Setup Database Function
int setup_database(sqlite3** db) {
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
    
    // Setup neural network tables
    rc = setup_neural_network_database(*db);
    if (rc != SQLITE_OK) {
        sqlite3_close(*db);
        return rc;
    }
    
    // Pre-load training data
    sqlite3_stmt* stmt;
    const char* insert_sql = 
        "INSERT OR REPLACE INTO training_data (input, response, frequency) "
        "VALUES (?, ?, COALESCE((SELECT frequency + 1 FROM training_data WHERE input = ?), 1))";
    
    rc = sqlite3_prepare_v2(*db, insert_sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Preparation failed: %s\n", sqlite3_errmsg(*db));
        sqlite3_close(*db);
        return rc;
    }
    
    int training_data_count = sizeof(initial_training_data) / sizeof(initial_training_data[0]);
    
    for (int i = 0; i < training_data_count; i++) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, initial_training_data[i].input, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, initial_training_data[i].response, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, initial_training_data[i].input, -1, SQLITE_STATIC);
        
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Insertion failed for %s: %s\n", 
                    initial_training_data[i].input, 
                    sqlite3_errmsg(*db));
            sqlite3_finalize(stmt);
            sqlite3_close(*db);
            return rc;
        }
    }
    
    sqlite3_finalize(stmt);    
    return SQLITE_OK;
}

// Save Training Data
void save_training_data(sqlite3* db, const char* input, const char* response) {
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

// Retrieve response function
char* retrieve_response(sqlite3* db, const char* input, int* tokens, int token_count) {
    sqlite3_stmt* stmt;
    char* best_response = NULL;
    
    const char* sql = 
        "SELECT response, frequency FROM training_data "
        "WHERE lower(input) LIKE lower(?) OR lower(input) LIKE lower(?) "
        "ORDER BY frequency DESC LIMIT 1";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Preparation failed: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    
    char pattern1[MAX_INPUT];
    char pattern2[MAX_INPUT];
    
    snprintf(pattern1, sizeof(pattern1), "%s", input);
    snprintf(pattern2, sizeof(pattern2), "%%%s%%", input);
    
    sqlite3_bind_text(stmt, 1, pattern1, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, pattern2, -1, SQLITE_STATIC);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* response = sqlite3_column_text(stmt, 0);
        
        best_response = malloc(strlen((const char*)response) + 1);
        if (best_response) {
            strcpy(best_response, (const char*)response);
        }
    }
    
    sqlite3_finalize(stmt);
    return best_response;
}


// Safe commands list
const char *safe_commands[] = {
    "clear", "ls", "dir", "ls -la", "ls /", "ls /home/", 
    "ls /bin/", "pwd", "htop", "atop", "ps", "whoami",
    "jobs", "nano", "head", "tail", "ps", "df", "df -h",
    "free", "free -h", "uptime", "date"
};


// Function to validate input characters
int is_valid_input_chars(const char *input) {
    for (int i = 0; input[i]; i++) {
        // Allow spaces, forward slashes, and alphanumeric characters, dash, underscore, question marks
        if (!isalnum(input[i]) && input[i] != '-' && input[i] != '_' && 
            input[i] != ' ' && input[i] != '/' && input[i] != '?') {
            return 0;
        }
    }
    return 1;
}

// Function to check if command is in safe list
int is_safe_command(const char *command) {
    for (int i = 0; i < SAFE_COMMANDS_COUNT; i++) {
        if (strcmp(command, safe_commands[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Debug function to check what's available
void debug_system_info() {
    printf("=== DEBUG INFO ===\n");
    printf("PATH: %s\n", getenv("PATH") ? getenv("PATH") : "NOT SET");
    printf("Testing command locations:\n");
    
    const char* test_paths[] = {"/bin/ls", "/usr/bin/ls", "/usr/local/bin/ls", "ls"};
    for (int i = 0; i < 4; i++) {
        if (access(test_paths[i], X_OK) == 0) {
            printf("  %s: FOUND\n", test_paths[i]);
        } else {
            printf("  %s: NOT FOUND\n", test_paths[i]);
        }
    }
       // Quick environment check
    const char* env_vars[] = {"PATH", "USER", "HOME", "SHELL"};
    for (int i = 0; i < 4; i++) {
        printf("%s: %s\n", env_vars[i], 
               getenv(env_vars[i]) ? getenv(env_vars[i]) : "NOT SET");
    }
    
    // Quick command check with better formatting
    typedef struct { const char* path; const char* cmd; } cmd_test_t;
    cmd_test_t tests[] = {
        {"/bin/ls", "ls"}, {"/usr/bin/ls", "ls"}, 
        {"/bin/pwd", "pwd"}, {"/bin/date", "date"}
    };
    
    printf("\nCommands:\n");
    for (int i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
        printf("%-20s %s\n", tests[i].path, 
               (access(tests[i].path, X_OK) == 0) ? "[okay]" : "[fail]");
    }
    
}

// Simplified execution function with debugging
int execute_safe_command(char *input) {
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
        for (int i = 0; i < SAFE_COMMANDS_COUNT; i++) {
            printf(" %d:%s ",i+1, safe_commands[i]);
            if (i % 3 == 2) {
			printf("\n");
			}
        }
        return 0;
    }
    
    printf("Executing: '%s'\n", input);
    int result = system(input);
    //printf("AI: System call returned: %d\n", result);
    
    if (result == -1) {
        perror("system() call failed");
        return 0;
    }
    
    return 1;
}


// Main Function with Neural Network Persistence
int main() {
    sqlite3* db;
    NeuralNetwork nn;
    char input[MAX_INPUT];
    int tokens[VOCAB_SIZE];
    int network_loaded = 0;
    
    // Initialize database
    if (setup_database(&db) != SQLITE_OK) {
        return -1;
    }
    
    printf("Chai Coffee AI Shell v2.0\n");
    printf("Commands: 'save nn <name>' to save network, 'load nn <name>' to load, 'list nn' to see saved networks,\n");
    printf("'list commands' to list all shell commands\n");
    printf("Available shell commands:\n");
        for (int i = 0; i < SAFE_COMMANDS_COUNT; i++) {
            printf(" %s ", safe_commands[i]);
            if (i % 10 == 9) {
			printf("\n");
			}
        }
        printf("\n"); 
		
    // Try to load default network or initialize new one
    if (load_neural_network_sqlite(db, &nn, "default") == SQLITE_OK) {
        network_loaded = 1;
        printf("Loaded existing 'default' neural network\n");
    } else {
        initialize_network(&nn);
        printf("Initialized new neural network\n");
    }
    
    while (1) {
        printf(">> ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        input[strcspn(input, "\n")] = 0;
        
        // Check for exit condition
        if (strcmp(input, "exit") == 0 || strcmp(input, "quit") == 0) {
            // Auto-save network before exit
            if (save_neural_network_sqlite(db, &nn, "default") == SQLITE_OK) {
                printf("Auto-saved neural network as 'default'\n");
            }
            break;
        }
        
        // Neural Network Management Commands
        if (strncmp(input, "save nn ", 8) == 0) {
            char* name = input + 8;
            if (save_neural_network_sqlite(db, &nn, name) == SQLITE_OK) {
                printf("Neural network saved as '%s'\n", name);
            } else {
                printf("Failed to save neural network\n");
            }
            continue;
        }
        
        if (strncmp(input, "load nn ", 8) == 0) {
            char* name = input + 8;
            if (load_neural_network_sqlite(db, &nn, name) == SQLITE_OK) {
                printf("Neural network '%s' loaded successfully\n", name);
                network_loaded = 1;
            } else {
                printf("Failed to load neural network '%s'\n", name);
            }
            continue;
        }
            if (strcmp(input, "list commands") == 0 || strcmp(input, "help") == 0) {
		        printf("Available commands:\n");
        for (int i = 0; i < SAFE_COMMANDS_COUNT; i++) {
            printf(" %d:%s ",i+1, safe_commands[i]);
            if (i % 3 == 2) {
			printf("\n");
			}
        }
        printf("\nNeural Network Persistence commands: \n'save nn <name>' to save network, 'load nn <name>' to load, 'list nn' to see saved networks\n");
        printf("Other keywords I can help with:\n");
        printf("'debug' path and system debug\n"); 
        printf("'run math' I can do some math.\n"); 
        printf("'list commands' to list all commands or you can just type help.\n");
        printf("'mem' and/or 'memory' will list total memory stats.\n"); 
        printf("'free memory' and/or 'free mem' will list free memory overview stats.\n");  
        printf("'time' or 'the time' and/or 'current time' will display the time & date.\n");
		}
		if (strcmp(input, "debug") == 0) {
		debug_system_info(); 
		}
		if (strcmp(input, "time") == 0 || strcmp(input, "the time") ==0 || 
		    strcmp(input, "current time") == 0) {
			    time_t current_time;
                struct tm *local_time;   
                time(&current_time);
                local_time = localtime(&current_time);
                // Different formatting options
               printf("Date and Time 24 hour format: %d/%d/%d %d:%d:%d\n", 
                       local_time->tm_mon + 1,  // Months since January (0-11)
                       local_time->tm_mday,     // Day of the month
                       local_time->tm_year + 1900,  // Years since 1900
                       local_time->tm_hour,     // Hours since midnight
                       local_time->tm_min,      // Minutes after the hour
                       local_time->tm_sec);     // Seconds after the minute
			   char buffer[80];        
              // Detailed formatting
             strftime(buffer, sizeof(buffer), 
             "%A, %B %d, %Y at %I:%M:%S %p", local_time);
             printf("Formatted time 12 hour format: %s\n", buffer);
			} 
			    // Array of time-related phrases to match
            char *time_phrases[] = {
        "what is the current time?",
        "what time is it?",
        "what is the time?",  
        "tell me the time",
        "tell me the time?", 
        "current time",
        "time now"
        };
    
        // Number of phrases in the array
        int num_phrases = sizeof(time_phrases) / sizeof(time_phrases[0]);
        // Flag to track if a match is found
        int match_found = 0;
    
    // Compare input against each phrase in the array
    for (int i = 0; i < num_phrases; i++) {
        if (strcmp(input, time_phrases[i]) == 0) {
            // If matched, display the current time
            time_t current_time;
            struct tm *local_time;
            char buffer[80];
            
            time(&current_time);
            local_time = localtime(&current_time);
            
            // Format time in a readable way
            strftime(buffer, sizeof(buffer), 
                     "The current time is %I:%M:%S %p on %A, %B %d, %Y", 
                     local_time);
            
            printf("%s\n", buffer);
            match_found = 1;
            break;
         }
       }
       // Handle no match scenario "yes we can" but why? 
       if (!match_found){ }
       // Do math questions if asked by the user 
       if (strcmp(input, "run math") == 0) {
      int num1, num2;
      char operator;
      
    // Prompt for mathematical expression
    printf("Enter a mathematical expression (e.g., 1 + 2 or 10 * 5): ");
    
    // Read input using fgets
    if (fgets(input, sizeof(input), stdin) != NULL) {
        // Remove trailing newline
        input[strcspn(input, "\n")] = 0;
        
        // Skip leading whitespaces
        char *trimmed = input;
        while (isspace(*trimmed)) trimmed++;
        
        // Attempt to parse the expression
        if (sscanf(trimmed, "%d %c %d", &num1, &operator, &num2) == 3) {
            // Perform calculation based on operator
            switch (operator) {
                case '+':
                    printf("\nResult: %d + %d = %d\n", num1, num2, num1 + num2);
                    break;
                case '-':
                    printf("\nResult: %d - %d = %d\n", num1, num2, num1 - num2);
                    break;
                case '*':
                    printf("\nResult: %d * %d = %d\n", num1, num2, num1 * num2);
                    break;
                case '/':
                    if (num2 != 0) {
                        printf("\nResult: %d / %d = %d\n", num1, num2, num1 / num2);
                    } else {
                        printf("\nError: Division by zero is not allowed!\n");
                    }
                    break;
                default:
                    printf("\nError: Invalid operator. Use +, -, *, or /\n");
            }
        } else {
            printf("\nError: Invalid input format. Use 'number operator number' (e.g., 1 + 2)\n");
        }
       }
	  }
       
		if (strcmp(input, "mem") == 0 || strcmp(input, "memory") == 0) {
	    struct sysinfo s1;
	    // Retrieve system information 
	    if (sysinfo(&s1) != 0) {
		perror("sysinfo"); 
		return 1;
		}
		   // Calculate memory values
           unsigned long long 
			bytes = s1.freeram * s1.mem_unit,
			kilobytes = bytes / 1024,
            megabytes = kilobytes / 1024,
            gigabytes = megabytes / 1024,
            terabytes = gigabytes / 1024,	       
            total_bytes = s1.totalram * s1.mem_unit,
            free_bytes = s1.freeram * s1.mem_unit,
            total_kilobytes = total_bytes / 1024,
            free_kilobytes = free_bytes / 1024,
            total_megabytes = total_kilobytes / 1024,
            free_megabytes = free_kilobytes / 1024;
            // Calculate percentage of free memory
            float free_percentage = ((float)free_bytes / total_bytes) * 100.0;
            // Display free memory in various units
            printf("Free Memory:\n");
            printf("Bytes:       %llu bytes\n", bytes);
            printf("Kilobytes:   %llu KB\n", kilobytes);
            printf("Megabytes:   %llu MB\n", megabytes);
            printf("Gigabytes:   %llu GB\n", gigabytes);
            printf("Terabytes:   %llu TB\n", terabytes);
            // Display memory information
            printf("Memory Information:\n");
            printf("Total Memory: %llu bytes (%llu MB)\n", total_bytes, total_megabytes);
            printf("Free Memory:  %llu bytes (%llu MB)\n", free_bytes, free_megabytes);
            printf("Free Memory Percentage: %.2f%%\n", free_percentage);
	    }
	    if (strcmp(input, "free memory") == 0 || strcmp(input, "free mem") == 0) {
		    struct sysinfo si;
    
    // Retrieve system information
    if (sysinfo(&si) != 0) {
        perror("sysinfo");
        return 1;
    }
    
    // Calculate memory values
    unsigned long long 
        total_bytes = si.totalram * si.mem_unit,
        free_bytes = si.freeram * si.mem_unit;
    
    // Calculate percentages
    float 
        total_mb = total_bytes / (1024.0 * 1024),
        free_mb = free_bytes / (1024.0 * 1024),
        used_mb = total_mb - free_mb,
        free_percentage = ((float)free_bytes / total_bytes) * 100.0,
        used_percentage = 100.0 - free_percentage;
    
    // Print memory information
    printf("Memory Overview:\n");
    printf("Total Memory: %.2f MB\n", total_mb);
    printf("Free Memory:  %.2f MB (%.2f%%)\n", free_mb, free_percentage);
    printf("Used Memory:  %.2f MB (%.2f%%)\n", used_mb, used_percentage);
		}
	
		
        if (strcmp(input, "list nn") == 0) {
            list_saved_neural_networks(db);
            continue;
        }
        
        if (strcmp(input, "reset nn") == 0) {
            initialize_network(&nn);
            printf("Neural network reset to random weights\n");
            continue;
        }
        
        // Convert input to lowercase for better matching
        for (int i = 0; input[i]; i++) {
            input[i] = tolower(input[i]);
        }
        
        // Tokenize input
        char input_copy[MAX_INPUT];
        strcpy(input_copy, input);
        int token_count = tokenize_input(input_copy, tokens);
        
        // Predict response confidence
        float confidence = predict_response(&nn, tokens, token_count);
        
        // Retrieve response from database
        char* response = retrieve_response(db, input, tokens, token_count);
        
        if (response) {
            printf("AI: %s (Confidence: %.2f)\n", response, confidence);
            
            // Optional: Train the system
            printf("Is this response helpful? (y/n): ");
            char feedback[10];
            if (fgets(feedback, sizeof(feedback), stdin) != NULL) {
                feedback[strcspn(feedback, "\n")] = 0;
                
                if (strcmp(feedback, "y") == 0) {
                    save_training_data(db, input, response);
                    printf("Response reinforced in training data\n");
                }
                if (strcmp(input, "clear") == 0 || strcmp(input, "cls") == 0) printf("\033[2J\033[H"); 
		
            }
            
            free(response);
        } else {
            printf("AI: I'm not sure how to respond. Can you teach me?\n");
            printf("Enter a suggested response: ");
            char new_response[MAX_RESPONSE];
            if (fgets(new_response, sizeof(new_response), stdin) != NULL) {
                new_response[strcspn(new_response, "\n")] = 0;
                save_training_data(db, input, new_response);
                printf("Thank you! I learned something new.\n");
            }
        }
      // check if input stats with run prefix 
   if (strncmp(input, "run ", 4) == 0)  {
	char *command = input + 4;  // skip (run+space) and do safe command logic      
   // Execute the safe command
   execute_safe_command(command);
  }


    }
    
    // Cleanup
    sqlite3_close(db);
    printf("AI: Goodbye! Neural network state preserved.\n");
    
    return 0;
}

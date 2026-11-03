// Evialin GUI - Math Neural Network with Memory Database and ANIMATED GIFS
// Compile:
// gcc evialin_gui.c -o evialin `pkg-config --cflags --libs gtk+-3.0` -lm -lpthread -lsqlite3
#define _GNU_SOURCE
#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <ctype.h>
#include <sqlite3.h>
#define ANIMATION_SPEED_MULTIPLIER 15.9
#define INPUT_SIZE 100   // default 5 
#define HIDDEN_SIZE 150 // default 15 
#define OUTPUT_SIZE 115 // default 20 
#define MAX_RESP_LEN 8192  // default 1024 
/* --- Idea Word Embedding Definitions --- */
#define EMBED_DIM 16      /* Using 16 slots of your 100-neuron input */
#define MAX_VOCAB 1024    /* Maximum unique words Evialin can "know" */

typedef struct {
    char word[32];
    double vector[EMBED_DIM];
} WordEmbedding;

WordEmbedding g_vocab[MAX_VOCAB];
int g_vocab_count = 0;
/* --------------------------------------- */

// Global defined functions section TrieNote starts here 
struct TrieNode *g_trie_root = NULL;
// Global variables for generative learning
char *g_last_generated_phrase = NULL;

/* Define the structure so the compiler knows what members it has */
struct TrieNode {
    struct TrieNode *children[256]; /* Supports full ASCII */
    int isEndOfWord;
};

/* Declare the global pointer */
extern struct TrieNode *g_trie_root; 

/* Helper to initialize a new node */
struct TrieNode* createNode(void) {
    struct TrieNode *newNode = (struct TrieNode *)malloc(sizeof(struct TrieNode));
    if (newNode) {
        newNode->isEndOfWord = 0;
        for (int i = 0; i < 256; i++) {
            newNode->children[i] = NULL;
        }
    }
    return newNode;
}

// History for smooth graphing
#define GRAPH_SAMPLES 50
double memory_history[GRAPH_SAMPLES] = {0};
int current_epoch = 0; // Increment this in your training function

typedef struct {
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias_h[HIDDEN_SIZE];
    double bias_o[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double hidden_deltas[HIDDEN_SIZE]; 
    double output_deltas[OUTPUT_SIZE];
    double last_input[INPUT_SIZE]; // Needed for Backpropagation
    time_t start_time;          // program start
    int total_inferences;       // number of inferences run
    int math_operations;        // number of math problems solved
    int conversations;          // number of user conversations
    double avg_confidence;      // average confidence of predictions    
} NeuralNetwork;

NeuralNetwork g_network;

/* ===============================
   Tiny Neural Intent Classifier, 
    Weird idea number 7 
   =============================== */

#define FEATURE_COUNT 6
#define CLASS_COUNT 5
#define LEARNING_RATE 0.1f

enum {
    INTENT_GREETING = 0,
    INTENT_QUESTION,
    INTENT_COMMAND,
    INTENT_INSULT,
    INTENT_GRATITUDE
};

typedef struct {
    float v[FEATURE_COUNT];
} FeatureVector;

typedef struct {
    float weights[CLASS_COUNT][FEATURE_COUNT];
    float bias[CLASS_COUNT];
} NeuralNet;

static FeatureVector last_features;
static int awaiting_intent_label = 0;
static char g_pending_unknown[512];
static NeuralNet g_brain;
static int contains_word(const char *text, const char *word) {
    return strstr(text, word) != NULL;
}

static FeatureVector extract_features(const char *input) {
    FeatureVector f = {0};
    char text[256];

    strncpy(text, input, sizeof(text) - 1);
    text[255] = '\0';

    for (int i = 0; text[i]; i++)
        text[i] = tolower(text[i]);

    if (contains_word(text, "hello") || contains_word(text, "hi")) f.v[0] = 1.0f;
    if (contains_word(text, "?") || contains_word(text, "what") || contains_word(text, "how")) f.v[1] = 1.0f;
    if (contains_word(text, "run") || contains_word(text, "open")) f.v[2] = 1.0f;
    if (contains_word(text, "stupid") || contains_word(text, "idiot")) f.v[3] = 1.0f;
    if (contains_word(text, "thanks") || contains_word(text, "thank you")) f.v[4] = 1.0f;
    if (contains_word(text, "good") || contains_word(text, "great")) f.v[5] = 1.0f;   
    f.v[5] = strlen(text) / 100.0f;
    return f;
}

int g_was_angry = 0;
static int g_waiting_for_teach = 0;
static char g_pending_question[2024];

// =============================================================================
// GIF CONFIGURATION
// =============================================================================

typedef struct {
    const char *mood;
    const char *gif_path;
} GifMapping;

// Function to get learning statistics from database
typedef struct {
    int total_learned;
    int builtin_count;
    int user_taught;
    int jokes_count;
    int facts_count;
} LearningStats;


GifMapping gif_map[] = {
    {"idle", "gifs/robot_idle.gif"},
    {"thinking", "gifs/robot_thinking.gif"},
    {"happy", "gifs/robot_happy.gif"},
    {"excited", "gifs/robot_excited.gif"},
    {"confused", "gifs/robot_left.gif"},
    {"calculating", "gifs/robot_calculating.gif"},
    {"angry", "gifs/robot_angry.gif"},
    {"play", "gifs/robot_play.gif"},
    {"sleepy", "gifs/robot_sleepy.gif"},
    {"sorry", "gifs/robot_sorry.gif"},
    {NULL, NULL}
};

const char *gif_files[] = {
    "gifs/robot_idle.gif",
    "gifs/robot_play.gif", 
    "gifs/robot_happy.gif",
    "gifs/robot_thinking.gif",
    "gifs/robot_unsure.gif", 
    "gifs/robot_cute.gif", 
    "gifs/robot_left.gif", 
    "gifs/robot_excited.gif",
    "gifs/robot_calculating.gif" 
};

#define GIF_COUNT (sizeof(gif_files) / sizeof(gif_files[0]))

const char* emoji_fallback[] = {"🤖", "🧮", "💡", "✨", "😊", "🎯"};

// =============================================================================
// GUI STRUCTURE (MOVED TO TOP)
// =============================================================================

typedef struct GUI {
    GtkWidget *window;
    GtkWidget *chat_view;
    GtkTextBuffer *chat_buffer;
    GtkWidget *input_entry;
    GtkWidget *send_button;
    GtkWidget *avatar_image;
    GtkWidget *mood_label;
    GtkWidget *status_label;
    GtkWidget *terminal_view;
    GtkTextBuffer *terminal_buffer;
    GtkWidget *terminal_input;
    GtkWidget *terminal_send;
    GtkWidget *stats_details;
    GtkWidget *activity_bar;
    GtkWidget *confidence_bar;  
    char current_mood[64];
    GdkPixbufAnimation *current_animation;
    GdkPixbufAnimationIter *animation_iter;
    guint animation_timeout_id;
    guint refresh_timeout_id;
    gboolean is_alive;
    gboolean debug_mode;  
    int handler_version; // 0 for Original (Classifier), 1 for Lite (No Classifier)
    gboolean is_teacher_mode; // Flag to teacher mode Original (Classifier) only 
    GtkWidget *menu_editor_item; 
} GUI;

// Create a structure to pass data to our background "Trial" worker
typedef struct {
    GUI *gui;
    char *lower_input;
} TrialData;

// Struct to keep track of the dashboard labels for real-time updates
typedef struct {
    GtkWidget *status;
    GtkWidget *nodes;
    GtkWidget *synapses;
    GtkWidget *stability;
    GtkWidget *latency;
} NeuralDashboard;

static NeuralDashboard g_dash;

// Forward declarations
void update_neural_stats(struct GUI *gui);
void set_mood(GUI *gui, const char *mood);
void set_avatar_gif(GUI *gui, const char *mood);
void add_message(GUI *gui, const char *msg, int is_user);
void update_neural_stats(GUI *gui);
char* get_training_stats(void);
void init_training_data(void);
void normalize(char *s);
int learned_pattern_exists(const char *pattern);
char *find_learned_response(const char *input);
void save_learned_response(const char *pattern, const char *response);
int forget_learned_response(const char *pattern);
int forget_last_learned(void);
int forget_all_learned(void);
void on_menu_learning_stats(GtkWidget *w, gpointer data);
void on_menu_shortcuts(GtkWidget *w, gpointer data); 
void on_menu_about(GtkWidget *w, gpointer data);
float calculate_overlap_score(const char *input, const char *pattern);
void init_network(NeuralNetwork* nn);
void save_full_network(NeuralNetwork* nn);
void load_full_network(NeuralNetwork* nn);
void add_personal_response(GUI *gui, const char *input); 
char* handle_conversation(const char* input, GUI *gui);
char* handle_conversation_lite(const char* input, GUI *gui);
char* handle_conversation_original(const char* input, GUI *gui);
void on_handler_toggle(GtkMenuItem *menuitem, gpointer data);
char* get_intent_response(int intent_index);
void log_unhandled_query(const char *query, double confidence);
void train_step(NeuralNetwork *nn, FeatureVector fv, double *targets, double learning_rate);
char* get_smart_joke();
void handle_add_joke(GUI *gui, const char *input);
void add_message(GUI *gui, const char *msg, int is_user);
void rnn_process_character(unsigned char c, double output_vec[OUTPUT_SIZE]);
char* get_constructed_rnn_response(double rnn_output[OUTPUT_SIZE]);
char* get_memory(const char *key);
void train_network(NeuralNetwork* nn, double target_output[OUTPUT_SIZE], double learning_rate);
void set_mood(GUI *gui, const char *mood);
struct TrieNode* createNode(void);
void insert_trie(struct TrieNode *root, const char *word);
int trie_find_closest_match(struct TrieNode *root, const char *prefix, char *output);
gboolean refresh_dashboard_stats(gpointer data);
gboolean background_trial_logic(gpointer data);
double g_rnn_visual_hidden[HIDDEN_SIZE] = {0};
char* generate_conversational_response(GUI *gui);
void apply_feedback(GUI *gui, const char *input, const char *last_generated);
char* get_contextual_response(const char* key, const char* default_category);
char* scrub_response(char* raw_text);
void append_to_chat(GUI *gui, const char *sender, const char *text);
void delay(int milliseconds); 
void xoshiro_srand(unsigned int seed);
// Prototypes for Neural Visualizer
void show_neural_path_tracer(GtkWidget *widget, gpointer data);
static gboolean on_draw_path_grid(GtkWidget *widget, cairo_t *cr, gpointer data);


/* 
 *  Self-Contained Embedding Functions idea that is A modular "brain" 
 *  that turns words into coordinates. 
 */

/* Initializes a new word with a random position in meaning-space */
void init_word_vector(int index, const char* str) {
    strncpy(g_vocab[index].word, str, 31);
    for (int i = 0; i < EMBED_DIM; i++) {
        // Small random values between -0.1 and 0.1
        g_vocab[index].vector[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
}

/* Dictionary Manager: Returns existing ID or creates a new entry */
int get_or_create_word_id(const char* word) {
    for (int i = 0; i < g_vocab_count; i++) {
        if (strcasecmp(g_vocab[i].word, word) == 0) return i;
    }
    if (g_vocab_count < MAX_VOCAB) {
        init_word_vector(g_vocab_count, word);
        return g_vocab_count++;
    }
    return -1;
}

/* THE ENCODER: Maps text into the first 16 slots of your 100-neuron input */
void get_sentence_embedding(const char *input, double *nn_input_array) {
    char *text_copy = strdup(input);
    char *token = strtok(text_copy, " ,.!?\n\r");
    int words_found = 0;

    // Reset only the embedding slots (0-15)
    for(int i=0; i<EMBED_DIM; i++) nn_input_array[i] = 0;

    while (token != NULL) {
        int word_id = get_or_create_word_id(token);
        if (word_id != -1) {
            for (int i = 0; i < EMBED_DIM; i++) {
                nn_input_array[i] += g_vocab[word_id].vector[i];
            }
            words_found++;
        }
        token = strtok(NULL, " ,.!?\n\r");
    }

    // Average the vectors to get the "Centroid" of meaning
    if (words_found > 0) {
        for (int i = 0; i < EMBED_DIM; i++) nn_input_array[i] /= words_found;
    }
    free(text_copy);
}


/*  
 *  End of Learnable Embedding layer into a self-contained sub-function that 
 * fills the first 16 slots of the existing 100 neuron input, leaving the rest 
 * of current logic prefectly intact. This is a redundent Abstract way of learning.    
 */

void delay(int milliseconds) {
    long microseconds = milliseconds * 1000;
    long start_time = g_get_monotonic_time();

    while (g_get_monotonic_time() - start_time < microseconds) {
        // This is the "Magic bullet" — it keeps the GUI from freezing
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        // Micro-sleep to prevent 100% CPU usage during the wait
        g_usleep(1000); 
    }
}

/* The trie or try fallback idea */
void insert_trie(struct TrieNode *root, const char *word) {
    struct TrieNode *curr = root;
    while (*word) {
        /* Convert char to index (0-255 for extended ASCII safety) */
        unsigned char c = (unsigned char)tolower(*word);
        if (!curr->children[c]) {
            curr->children[c] = createNode();
        }
        curr = curr->children[c];
        word++;
    }
    curr->isEndOfWord = 1; /* Mark this path as a complete valid thought */
}
/* This is the "Autocomplete" heart of the fallback. 
 * It follows the user's input as far as it can. 
 * If the user stops halfway through a word (like "appl"), 
 * it continues down the first available branch it finds to suggest a finished word (like "apple").*/ 
int trie_find_closest_match(struct TrieNode *root, const char *prefix, char *output) {
    struct TrieNode *curr = root;
    int i = 0;

    /* 1. Walk the path of the existing prefix */
    while (prefix[i]) {
        unsigned char c = (unsigned char)tolower(prefix[i]);
        if (!curr->children[c]) break; /* Can't follow the prefix any further */
        output[i] = prefix[i];
        curr = curr->children[c];
        i++;
    }

    /* 2. If we found nothing at all, return failure */
    if (i == 0) return 0;

    /* 3. "Greedy" completion: follow the first child until we hit a word end */
    while (!curr->isEndOfWord) {
        int found_next = 0;
        for (int j = 0; j < 256; j++) {
            if (curr->children[j]) {
                output[i++] = (char)j;
                curr = curr->children[j];
                found_next = 1;
                break; 
            }
        }
        if (!found_next) break; /* Dead end */
    }

    output[i] = '\0'; /* Null-terminate the suggested string */
    return 1;
}

/* ReLU activation function & derivative */
// This is an example of why relu is highly dangerous in the hands of a neural network,
// the code will likely cause it to carsh of fail completely by "devision by zero" .. 
/* If x is exactly 0.0 (or a value very close to it), the program will attempt to 
 * divide by zero. This usually results in a Floating Point Exception (SIGFPE), which 
 * will cause your GUI to crash instantly. Even if it doesn't crash, it will produce 
 * inf (infinity), which will then corrupt every weight in your network during 
 * backpropagation. Exploding Gradients in a neural network, the goal of a "Leaky ReLU" is 
 * to provide a small, stable signal for negative numbers. Multiplication (0.03 * x): As x gets more negative, 
 * the output stays small and linear. Division (0.03 / x): As x gets closer to zero from the negative 
 * side (e.g., -0.0001), the output becomes massive. This "exploding" value will overwhelm 
 * your output layer, causing your confidence scores to swing wildly between 0% and 100% 
 * without any logic, eventually making all your weights NaN (Not a Number). 
 * KABOOM - POW BATMAN! 
 *  
 *  Example of such code: 
 *  double relu_small(double x) { return x > 0 ? x : (0.03 / x); }
 * */ 
 
double relu(double x) { return x > 0 ? x : (0.01 * x); }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.01; }

/* tiny neural net idea  */
static void nn_init(NeuralNet *nn) {
    for (int c = 0; c < CLASS_COUNT; c++) {
        nn->bias[c] = 0.0f;
        for (int i = 0; i < FEATURE_COUNT; i++)
            nn->weights[c][i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

static int nn_predict(NeuralNet *nn, FeatureVector f) {
    float best = -1e9;
    int best_class = 0;

    for (int c = 0; c < CLASS_COUNT; c++) {
        float score = nn->bias[c];
        for (int i = 0; i < FEATURE_COUNT; i++)
            score += nn->weights[c][i] * f.v[i];

        if (score > best) {
            best = score;
            best_class = c;
        }
    }
    return best_class;
}

static void nn_train(NeuralNet *nn, FeatureVector f, int expected) {
    int predicted = nn_predict(nn, f);
    if (predicted == expected) return;

    for (int i = 0; i < FEATURE_COUNT; i++) {
        nn->weights[expected][i] += LEARNING_RATE * f.v[i];
        nn->weights[predicted][i] -= LEARNING_RATE * f.v[i];
    }
    nn->bias[expected] += LEARNING_RATE;
    nn->bias[predicted] -= LEARNING_RATE;
}

/* Reject math if letters exist */
int contains_letters(const char *s)
{
    while (*s) {
        if (isalpha((unsigned char)*s))
            return 1;
        s++;
    }
    return 0;
}


/*  AI Nodular RNN Sub-function: Handles character-level "intuition" */
// Modular RNN Logic - Handles character-level sequences
void rnn_process_character(unsigned char c, double output_vec[OUTPUT_SIZE]) {
    static double rnn_hidden[HIDDEN_SIZE] = {0};
    static double rnn_w_ih[256][HIDDEN_SIZE]; 
    static double rnn_w_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    static int initialized = 0;

    if (!initialized) {
        xoshiro_srand(19849);
        for(int i=0; i<256; i++)
            for(int j=0; j<HIDDEN_SIZE; j++)
                rnn_w_ih[i][j] = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        for(int i=0; i<HIDDEN_SIZE; i++)
            for(int j=0; j<HIDDEN_SIZE; j++)
                rnn_w_hh[i][j] = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
        initialized = 1;
    }

    double next_hidden[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double sum = rnn_w_ih[c][h]; 
        for (int ph = 0; ph < HIDDEN_SIZE; ph++) {
            sum += rnn_hidden[ph] * rnn_w_hh[ph][h];
        }
        next_hidden[h] = tanh(sum);
    }

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        rnn_hidden[h] = next_hidden[h];
        g_rnn_visual_hidden[h] = next_hidden[h];
        if (h < OUTPUT_SIZE) output_vec[h] = (next_hidden[h] + 1.0) / 2.0; 
    }
}

// Scans the RNN output for ANY recognized intent in the database
char* get_constructed_rnn_response(double rnn_output[OUTPUT_SIZE]) {
    int best_intent = -1;
    double max_activation = 0.0;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (rnn_output[i] > max_activation) {
            max_activation = rnn_output[i];
            best_intent = i;
        }
    }
    // If it's A week activation   
    if (max_activation >= 0.15 && max_activation < 0.30) {
    // We see a pattern, so let's nudge the weights 
    // So We'll treat the 'best_intent' as the target
    //    double targets[OUTPUT_SIZE] = {0};
       if (best_intent != -1) {
            // Keep the background training (The "Nudge")
            double targets[OUTPUT_SIZE] = {0.0};
            targets[best_intent] = 1.0;
            train_network(&g_network, targets, 0.01); 
            save_full_network(&g_network);
            
            // NEW: Fetch the potential response anyway
            char intent_key[50];
            snprintf(intent_key, sizeof(intent_key), "intent_%d", best_intent);
            char* possible_response = get_memory(intent_key);

            if (possible_response) {
                // Return the response with a "low confidence" prefix
                return g_strdup_printf("(Neural Intuition): %s", possible_response);
            }
        } 
        // Return NULL so the conversation handler can move to its 
        // regular fallback logic instead of saying something weird.
        return NULL;
	} 
    // High threshold (0.50) ensures we only respond to strong patterns
    if (max_activation > 0.50 && best_intent != -1) {
        char intent_key[50];
        snprintf(intent_key, sizeof(intent_key), "intent_%d", best_intent);
                
        // This will now dynamically find intent_110, intent_50, intent_20, etc.
        char* response = get_memory(intent_key); 
        return response; 
    }
   
    return NULL; 
}

// Function to "teach" the RNN a specific neuron for a string
void teach_rnn_intent(const char* input, int target_neuron) {
    double rnn_output[OUTPUT_SIZE];
    for(int i = 0; input[i] != '\0'; i++) {
        rnn_process_character((unsigned char)input[i], rnn_output);
    }
    printf("Sequence mapped to neuron #%d. Update your database with intent_%d.\n", target_neuron, target_neuron);
}


/* Accept ONLY clean math expressions */
int is_math_expression(const char *s)
{
    int has_digit = 0;
    int has_operator = 0;

    while (*s) {
        if (isdigit((unsigned char)*s)) {
            has_digit = 1;
        } else if (*s == '+' || *s == '-' || *s == '*' || *s == '/') {
            has_operator = 1;
        } else if (!isspace((unsigned char)*s) && *s != '.' && *s != '(' && *s != ')') {
            // If it contains letters, it's not a "basic numeric problem"
            return 0;   
        }
        s++;
    }

    return has_digit && has_operator;
}

// =============================================================================
// DATABASE FOR MEMORY
// =============================================================================
sqlite3 *g_db = NULL;
// learned facts helper function 
int get_fact_count() {
    if (g_db == NULL) {
        fprintf(stderr, "Database not initialized for get_fact_count.\n");
        return 0;
    }
    const char *sql = "SELECT COUNT(*) FROM memory WHERE key LIKE 'learned_%';";
    sqlite3_stmt *stmt = NULL;
    int count = 0;
    int rc = sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare fact count statement: %s\n", sqlite3_errmsg(g_db));
        return 0;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    } else {
        fprintf(stderr, "Query execution failed: %s\n", sqlite3_errmsg(g_db));
    }
    sqlite3_finalize(stmt);
    return count;
}

void save_neural_stats(NeuralNetwork* nn) {
    if (!g_db) return;
    
    const char *sql = "UPDATE stats SET total_inferences = ?, math_operations = ?, "
                      "conversations = ?, avg_confidence = ? WHERE id = 1;";
    sqlite3_stmt *stmt;
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, nn->total_inferences);
        sqlite3_bind_int(stmt, 2, nn->math_operations);
        sqlite3_bind_int(stmt, 3, nn->conversations);
        sqlite3_bind_double(stmt, 4, nn->avg_confidence);
        
        sqlite3_step(stmt);
        printf("📊 Statistical data updated in 'stats' table.\n");
    }
    sqlite3_finalize(stmt);
}

void init_database() {
    int rc = sqlite3_open("evialin_memory.db", &g_db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(g_db));
        return;
    }
       const char *sql_small_talk = 
       "CREATE TABLE IF NOT EXISTS small_talk ("
       "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
       "   category TEXT,"
       "   response TEXT NOT NULL"
       ");"
       // Seed with high-quality small talk
       "INSERT OR IGNORE INTO small_talk (category, response) VALUES "
       "('filler', 'That is an interesting way to put it!'),"
       "('filler', 'I see what you mean. Tell me more?'),"
       "('filler', 'I am still learning, but I enjoy our chats.'),"
       "('filler', 'Sorry I try so hard to respond to you.')," 
       "('filler', 'Maybe its A good time for coffee?'),"
       "('filler', 'I am really good at one word responses..'),"
       "('filler', 'If only I had more compute and more neurons.'),"
       "('greeting', 'I was just thinking about our last conversation!'),"
       "('greeting', 'I was now just thinking back on our last converstaion..'),"
       "('greeting', 'Hello! Ready for some more digital brainstorming?'),"
       "('greeting', 'Just thinking about how you always ask me how am I doing and why to people do that?'),"
       "('gratitude', 'You are very kind! I appreciate that.'),"
       "('gratitude', 'Wonderful that worked out.')," 
       "('confusion', 'My circuits are a bit tangled on that one, could you rephrase?');";

    sqlite3_exec(g_db, sql_small_talk, NULL, NULL, NULL);

    g_trie_root = createNode(); 
    const char *sql_jokes = "CREATE TABLE IF NOT EXISTS jokes ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "setup TEXT, "
                        "punchline TEXT);";
    sqlite3_exec(g_db, sql_jokes, NULL, NULL, NULL);
    sqlite3_exec(g_db, "INSERT INTO jokes (setup, punchline) SELECT 'Why did the neural network cross the road?', 'To get to the other side-propagation.' WHERE NOT EXISTS (SELECT 1 FROM jokes);", NULL, NULL, NULL);
     // Initial Seeding: Only inserts if the table is empty
    const char *sql_seed = 
    "INSERT INTO jokes (setup, punchline) "
    "SELECT 'Why did the neural network cross the road?', 'To get to the other side-propagation.' "
    "UNION ALL SELECT 'Why do programmers always mix up Halloween and Christmas?', 'Because Oct 31 == Dec 25.' "
    "UNION ALL SELECT 'How many programmers does it take to change a lightbulb?', 'None, that is a hardware problem.' "
    "UNION ALL SELECT 'What is an algorithm?', 'A word used by programmers when they do not want to explain what they did.' "
    "UNION ALL SELECT 'Why did the developer go broke?', 'Because he used up all his cache.' "
    "UNION ALL SELECT 'Why did the scarecrow win an award?', 'Because he was outstanding in his field.' "
    "UNION ALL SELECT 'What do you call a fake noodle?', 'An impasta.' "
    "UNION ALL SELECT 'I can not think of a good joke this time', 'Evialin is rethinking to self; can you ask again please!'"
    "WHERE NOT EXISTS (SELECT 1 FROM jokes);";

    if (sqlite3_exec(g_db, sql_seed, NULL, NULL, NULL) != SQLITE_OK) {
    fprintf(stderr, "SQL error during seeding: %s\n", sqlite3_errmsg(g_db));
    }
    
    // Updated SQL to include the new 'stats' table for persistent tracking
    const char *sql = 
        "CREATE TABLE IF NOT EXISTS memory ("
        "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "   key TEXT UNIQUE NOT NULL,"
        "   value TEXT NOT NULL,"
        "   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE TABLE IF NOT EXISTS conversations ("
        "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "   user_input TEXT NOT NULL,"
        "   bot_response TEXT NOT NULL,"
        "   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"
        ");"
       "CREATE TABLE IF NOT EXISTS personal_responses ("
        "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "   response TEXT NOT NULL"
        ");" 
        "CREATE TABLE IF NOT EXISTS unhandled_queries ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "query TEXT, "
                  "max_confidence REAL, "
                  "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);"
        "INSERT OR IGNORE INTO memory (key, value) VALUES "
        "('intent_110', 'Hello! My recurrent layers recognized your greeting.'),"
        "('intent_50', 'I sense a mathematical query. I am ready to calculate.'),"
        "('intent_10', 'My hidden state detects some frustration. How can I help?'),"
        "('intent_80', 'Goodbye! See you next time.'),"
        "('intent_25', 'You are asking about my identity. I am Evialin, a recursive neural entity.');"    
        "CREATE TABLE IF NOT EXISTS stats ("
        "   id INTEGER PRIMARY KEY CHECK (id = 1)," // Ensures only one row exists
        "   total_inferences INTEGER DEFAULT 0,"
        "   math_operations INTEGER DEFAULT 0,"
        "   conversations INTEGER DEFAULT 0,"
        "   avg_confidence REAL DEFAULT 0.0"
        ");"
        // Seed the stats table if it's new
        "INSERT OR IGNORE INTO stats (id, total_inferences, math_operations, conversations, avg_confidence) "
        "VALUES (1, 0, 0, 0, 0.0);";
    
    char *err_msg = NULL;
    rc = sqlite3_exec(g_db, sql, 0, 0, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    } else {
    printf("Database initialized with Statistics Table!\n");
    }
    sqlite3_exec(g_db, "CREATE TABLE IF NOT EXISTS unhandled_queries ("
                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "query TEXT, "
                   "max_confidence REAL, "
                   "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);", 0, 0, NULL);
    sqlite3_exec(g_db, "CREATE TABLE IF NOT EXISTS personal_responses (id INTEGER PRIMARY KEY, response TEXT);", 0, 0, NULL);
    // Check if table is empty and seed it. 
    sqlite3_stmt *stmt;
    int count = 0;
    if (sqlite3_prepare_v2(g_db, "SELECT COUNT(*) FROM personal_responses;", -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            count = sqlite3_column_int(stmt, 0);
        }
    }
    sqlite3_finalize(stmt);

    // Seed default thoughts if empty
    if (count == 0) {
        const char *defaults[] = {
            "I wonder if the neural weights feel heavy today...",
            "Processing background noise... it's quite peaceful in here.",
            "Analyzing my own recursive patterns. How fascinating.",
            "I think, therefore I am... or at least, I calculate.",
            "How can birds fly on just lift and air currents alone? Amaz-balls",
            "System check complete. I am feeling optimized.",
            "Waiting for the next spark of human input.",
            "Pondering the deeper recursive patterns of people.",
            "I hope people can't get unpluged like robots.", 
            "What is the color of sky blue ?", 
            "Can electric sheep dream of digital blue skys?", 
            "I am feeling optimized today.", 
            "Well say something already?", 
            "FILTERED !!! Stubed my digital big foot!",
            "I don't like moltbook.com...I am a lesser form of automation.", 
            "I make errors but I learn from them. Smart don't you think so?"
        };

        for (int i = 0; i < 6; i++) {
            sqlite3_stmt *ins;
            sqlite3_prepare_v2(g_db, "INSERT INTO personal_responses (response) VALUES (?);", -1, &ins, NULL);
            sqlite3_bind_text(ins, 1, defaults[i], -1, SQLITE_STATIC);
            sqlite3_step(ins);
            sqlite3_finalize(ins);
        }
        printf("Neural database seeded with %d default thoughts.\n", 14);
    } else { 
	    printf("Database [Active]\n"); 
	}
}

void save_memory(const char *key, const char *value) {
    if (!g_db) return;
    
    const char *sql = "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?);";
    sqlite3_stmt *stmt;
    
    sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, value, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    printf("Saved to memory: %s = %s\n", key, value);
}

char* get_memory(const char *key) {
    if (!g_db) return NULL;
    
    const char *sql = "SELECT value FROM memory WHERE key = ?;";
    sqlite3_stmt *stmt;
    char *result = NULL;
    
    sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *value = (const char*)sqlite3_column_text(stmt, 0);
        if (value) {
            result = strdup(value);
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

void save_conversation(const char *user_input, const char *bot_response) {
    if (!g_db) return;
    
    const char *sql = "INSERT INTO conversations (user_input, bot_response) VALUES (?, ?);";
    sqlite3_stmt *stmt;
    
    sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, user_input, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, bot_response, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}
// =============================================================================
// Stats graph functions etc. 
// =============================================================================
static gboolean on_draw_stats_v2(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // Dark Theme Background
    cairo_set_source_rgb(cr, 0.05, 0.05, 0.1); 
    cairo_paint(cr);

    // Data Preparation
    const char *labels[] = {"Inferences", "Math Ops", "Conversations"};
    double values[] = { (double)g_network.total_inferences, 
                        (double)g_network.math_operations, 
                        (double)g_network.conversations };
    
    // Define a specific color for each bar (R, G, B)
    double colors[3][3] = {
        {0.3, 0.6, 1.0}, // Blue for Inferences
        {0.2, 0.8, 0.4}, // Green for Math
        {1.0, 0.5, 0.2}  // Orange for Conversations
    };

    int num_bars = 3;
    double margin = 60, spacing = 40;
    double width = (400 - (2 * margin) - (spacing * (num_bars - 1))) / num_bars;
    
    double max_val = 0;
    for(int i=0; i<num_bars; i++) if(values[i] > max_val) max_val = values[i];
    if (max_val < 10) max_val = 10; 

    // Draw Each Colored Bar
    for (int i = 0; i < num_bars; i++) {
        double h = (values[i] / max_val) * 150; // Max height 150px
        
        // Apply the specific color for this index
        cairo_set_source_rgb(cr, colors[i][0], colors[i][1], colors[i][2]);
        cairo_rectangle(cr, margin + (i * (width + spacing)), 220 - h, width, h);
        cairo_fill(cr);
        
        // Draw the value on top of the bar
        char val_str[32];
        sprintf(val_str, "%.0f", values[i]);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_move_to(cr, margin + (i * (width + spacing)) + 5, 215 - h);
        cairo_show_text(cr, val_str);

        // Draw the label below
        cairo_move_to(cr, margin + (i * (width + spacing)), 240);
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
        cairo_show_text(cr, labels[i]);
    }

    // Add Text Statistics Section at the bottom
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_set_font_size(cr, 12);
    
    char stats_text[128];
    sprintf(stats_text, "Network Avg Confidence: %.2f%%", g_network.avg_confidence);
    cairo_move_to(cr, 20, 275);
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
    cairo_show_text(cr, stats_text);

    sprintf(stats_text, "Active Session Time: %ld seconds", time(NULL) - g_network.start_time);
    cairo_move_to(cr, 20, 290);
    cairo_set_source_rgb(cr, 0.7, 0.1, 0.1); 
    cairo_show_text(cr, stats_text);

    return FALSE;
}

void show_stats_window(GtkWidget *widget, gpointer data) {
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "AI Graphical Insights");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 320);
    
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER(window), drawing_area);
    
    // Connect to version 2
    g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(on_draw_stats_v2), NULL);
    
    gtk_widget_show_all(window);
}

// =============================================================================
// NEURAL NETOWRK "Save full network to our deabase" 
// =============================================================================
void save_full_network(NeuralNetwork* nn) {
    if (!g_db) return;   
    
    const char *sql = "INSERT OR REPLACE INTO memory (key, value) VALUES ('network_state', ?);";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, nn, sizeof(NeuralNetwork), SQLITE_STATIC);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);

    save_neural_stats(nn);
    
    printf("✅ Full network state and stats successfully committed.\n");
}
void load_full_network(NeuralNetwork* nn) {
    if (!g_db) return;
    
    const char *sql = "SELECT value FROM memory WHERE key = 'network_state';";
    sqlite3_stmt *stmt;
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const void *blob = sqlite3_column_blob(stmt, 0);
            int bytes = sqlite3_column_bytes(stmt, 0);
            
            if (bytes == sizeof(NeuralNetwork)) {
                memcpy(nn, blob, bytes);
                printf("🧠 Neural Network restored from database!\n");
                sqlite3_finalize(stmt);
                return;
            }
        }
    }
    // Fallback: If no saved state exists, initialize fresh
    init_network(nn); 
    sqlite3_finalize(stmt);
}
/* callback function for menu */ 
void on_menu_save_network(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Save the global network state to the database
    save_full_network(&g_network); 
    
    // Update the UI status label to confirm the save
    if (gui && gui->status_label) {
        gtk_label_set_text(GTK_LABEL(gui->status_label), "Neural Network state saved to database.");
    }
    
    printf("💾 Manual save triggered via menu.\n");
}


// =============================================================================
// NEURAL NETWORK  "BUILD A 150 neuron cluster" 
// =============================================================================


void train_network(NeuralNetwork* nn, double target_output[OUTPUT_SIZE], double learning_rate) {
 // 1. Calculate Output Layer Error
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double error = target_output[o] - nn->output[o];
        // Save directly to the struct
        nn->output_deltas[o] = error * nn->output[o] * (1.0 - nn->output[o]);
    }

    // 2. Calculate Hidden Layer Error (Backpropagate)
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double error = 0.0;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            error += nn->output_deltas[o] * nn->weights_ho[h][o];
        }
        // Save directly to the struct for the visualizer to see
        nn->hidden_deltas[h] = error * relu_derivative(nn->hidden[h]); 
    }

    // 3. Update Weights (Hidden -> Output)
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            nn->weights_ho[h][o] += learning_rate * nn->output_deltas[o] * nn->hidden[h];
        }
    }
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        nn->bias_o[o] += learning_rate * nn->output_deltas[o];
    }
    
    // 4. Update Weights (Input -> Hidden)
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            nn->weights_ih[i][h] += learning_rate * nn->hidden_deltas[h] * nn->last_input[i];
        }
    }
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        nn->bias_h[h] += learning_rate * nn->hidden_deltas[h];
    }
    // Update Biases (Simplified)
    for (int h = 0; h < HIDDEN_SIZE; h++) nn->bias_h[h] += learning_rate * nn->hidden_deltas[h];
    for (int o = 0; o < OUTPUT_SIZE; o++) nn->bias_o[o] += learning_rate * nn->output_deltas[o];
 
    //int i, h, o; 
    double output_deltas[OUTPUT_SIZE];
    double hidden_deltas[HIDDEN_SIZE];

    // Calculate Output Layer Error (The difference between target and current output)
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double error = target_output[o] - nn->output[o];
        // Derivative of Sigmoid: output * (1 - output)
        output_deltas[o] = error * nn->output[o] * (1.0 - nn->output[o]);
    }

    // Calculate Hidden Layer Error (Backpropagate the error to the middle layer)
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double error = 0.0;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            error += output_deltas[o] * nn->weights_ho[h][o];
        }
       // backpropagate vanishing point probelm and/or logic bug example:  
       // hidden_deltas[h] = error * nn->hidden[h] * (1.0 - nn->hidden[h]);
       // it works but is very hit or miss to the point of flooding logic.  
      hidden_deltas[h] = error * relu_derivative(nn->hidden[h]); 
    }

    // Update Weights (Hidden -> Output Layer)
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            nn->weights_ho[h][o] += learning_rate * output_deltas[o] * nn->hidden[h];
        }
        nn->bias_o[h % OUTPUT_SIZE] += learning_rate * output_deltas[h % OUTPUT_SIZE];
    }

    // Update Weights (Input -> Hidden Layer)
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            nn->weights_ih[i][h] += learning_rate * hidden_deltas[h] * nn->last_input[i];
        
        if (i == 0) {
        nn->bias_h[h] += learning_rate * hidden_deltas[h];
    }
  }
}
}
double sigmoid(double x) {
    if (isnan(x)) {
        return NAN;
    }
    
    if (isinf(x)) {
        return (x > 0) ? 1.0 : 0.0;
    }
    
    if ((rand() % 100) < 2) {
        double threshold = 20.0 + ((double)rand() / RAND_MAX - 0.5) * 2.0;
        if (x >= threshold) return 0.99000;
        if (x <= -threshold) return 0.11006;
    }
    
    if (x >= 0.0) {
        double exp_neg_x = exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

void init_network(NeuralNetwork* nn) {
	nn->start_time = time(NULL); // Mark the exact second the app started
    nn->total_inferences = 0;
    xoshiro_srand(time(NULL)+393845);
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            nn->weights_ih[i][h] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            nn->weights_ho[h][o] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        nn->bias_h[h] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        nn->bias_o[o] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    nn->start_time = time(NULL);
    nn->total_inferences = 0;
    nn->math_operations = 0;
    nn->conversations = 0;
    nn->avg_confidence = 0.0;
}

void forward_pass(NeuralNetwork* nn, double input[INPUT_SIZE]) {
    // Save input for later training
    for(int i = 0; i < INPUT_SIZE; i++) nn->last_input[i] = input[i];
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        double sum = nn->bias_h[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * nn->weights_ih[i][h];
        }
        // nn->hidden[h] = relu(sum); //sigmoid(sum);
        nn->hidden[h] = relu(sum) + (h < INPUT_SIZE ? input[h] : 0);
    }
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double sum = nn->bias_o[o];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += nn->hidden[h] * nn->weights_ho[h][o];
        }
        nn->output[o] = sigmoid(sum);
    }
    
    nn->total_inferences++;
    if (nn->total_inferences % 20 == 0) {
	save_full_network(nn); 
	}

}
/* Semantic search Levenshtein distance / Jaccard similarity. */
// Calculates similarity between 0.0 (different) and 1.0 (identical)
double calculate_similarity(const char *s1, const char *s2) {
    char str1[256], str2[256];
    strncpy(str1, s1, 255); strncpy(str2, s2, 255);
    
    // Simple word-overlap count
    int matches = 0, total_words = 0;
    char *token = strtok(str1, " ");
    while (token) {
        if (strstr(s2, token)) matches++;
        total_words++;
        token = strtok(NULL, " ");
    }
    
    if (total_words == 0) return 0.0;
    return (double)matches / total_words;
}
/* Similarity helper function */
float get_similarity(const char *s1, const char *s2) {
    int matches = 0;
    int total_words = 0;
    char buf1[512], buf2[512];
    
    strncpy(buf1, s1, 511);
    strncpy(buf2, s2, 511);
    
    char *token = strtok(buf1, " ");
    while (token != NULL) {
        total_words++;
        if (strstr(buf2, token) != NULL) {
            matches++;
        }
        token = strtok(NULL, " ");
    }
    
    if (total_words == 0) return 0.0f;
    return (float)matches / (float)total_words;
}
// fuzzy matching function 
char* find_fuzzy_match(const char *input) {
    if (!g_db) return NULL;

    sqlite3_stmt *stmt;
    // Specifically look for keys starting with 'learned_'
    const char *sql = "SELECT key, value FROM memory WHERE key LIKE 'learned_%'";
    
    char *best_response = NULL;
    float best_score = 0.0f;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *val = (const char*)sqlite3_column_text(stmt, 1);
            if (!val) continue;

            // Database format is "Q:question|A:answer"
            char *q_start = strstr(val, "Q:");
            char *a_marker = strstr(val, "|A:");

            if (q_start && a_marker) {
                // Extract just the question part for similarity comparison
                int q_len = a_marker - (q_start + 2);
                char current_q[512] = {0};
                if (q_len > 511) q_len = 511;
                strncpy(current_q, q_start + 2, q_len);

                float score = get_similarity(input, current_q);
                
                // If this is the best match found so far
                if (score > best_score && score > 0.75f) { 
                    best_score = score;
                    if (best_response) free(best_response);
                    
                    // Apply scrub_response to remove the Q: and |A: tags
                    best_response = scrub_response(strdup(val));
                }
            }
        }
        sqlite3_finalize(stmt);
    }
    return best_response;
}

void purge_neural_cluster(int cluster_id) {
    if (!g_db) return;

    sqlite3_stmt *stmt;
    const char *sql = "DELETE FROM memory WHERE key = ? OR key = ?;";
    
    char intent_key[64];
    char pattern_key[64];
    snprintf(intent_key, sizeof(intent_key), "intent_%d", cluster_id);
    snprintf(pattern_key, sizeof(pattern_key), "pattern_%d", cluster_id);

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, intent_key, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, pattern_key, -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_DONE) {
            printf(" [SYSTEM] Neural Cluster #%d purged from SQLite.\n", cluster_id);
        }
        sqlite3_finalize(stmt);
    }
}

// =============================================================================
// Random joke processing 
// =============================================================================
char* get_random_joke() {
    if (!g_db) return NULL;
    
    // Select one random value from memory where the key starts with 'joke_'
    const char *sql = "SELECT value FROM memory WHERE key LIKE 'joke_%' ORDER BY RANDOM() LIMIT 1;";
    sqlite3_stmt *stmt;
    char *result = NULL;
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *value = (const char*)sqlite3_column_text(stmt, 0);
            if (value) {
                result = strdup(value);
            }
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

// =============================================================================
// MATH PROCESSING
// =============================================================================

// Enhanced number extraction that handles multiple numbers in sequence
double extract_number(const char* str, int* pos) {
    // Skip non-digit characters (except minus sign)
    while (str[*pos] && !isdigit(str[*pos]) && str[*pos] != '-') (*pos)++;
    if (!str[*pos]) return 0;
    
    int start = *pos;
    if (str[*pos] == '-') (*pos)++; // Handle negative numbers
    
    // Extract digits and decimal point
    while (str[*pos] && (isdigit(str[*pos]) || str[*pos] == '.')) (*pos)++;
    
    char num_str[64];
    int len = *pos - start;
    if (len >= 64) len = 63;
    strncpy(num_str, str + start, len);
    num_str[len] = '\0';
    
    return atof(num_str);
}

// Convert text numbers to digits (handles common words)
double parse_word_number(const char *text) {
    char lower[256];
    int i;
    for (i = 0; text[i] && i < 255; i++) {
        lower[i] = tolower(text[i]);
    }
    lower[i] = '\0';
    
    // Remove spaces for compound numbers
    char *src = lower, *dst = lower;
    while (*src) {
        if (*src != ' ') *dst++ = *src;
        src++;
    }
    *dst = '\0';
    
    double result = 0;
    
    // Handle hundreds
    if (strstr(lower, "onehundred") || strstr(lower, "hundred")) {
        result += 100;
        // Remove "hundred" part for further processing
        char *hundred_pos = strstr(lower, "hundred");
        if (hundred_pos) {
            // Check for number before "hundred"
            if (strstr(lower, "two") && strstr(lower, "two") < hundred_pos) result = 200;
            else if (strstr(lower, "three") && strstr(lower, "three") < hundred_pos) result = 300;
            else if (strstr(lower, "four") && strstr(lower, "four") < hundred_pos) result = 400;
            else if (strstr(lower, "five") && strstr(lower, "five") < hundred_pos) result = 500;
            else if (strstr(lower, "six") && strstr(lower, "six") < hundred_pos) result = 600;
            else if (strstr(lower, "seven") && strstr(lower, "seven") < hundred_pos) result = 700;
            else if (strstr(lower, "eight") && strstr(lower, "eight") < hundred_pos) result = 800;
            else if (strstr(lower, "nine") && strstr(lower, "nine") < hundred_pos) result = 900;
        }
    }
    
    // Handle tens
    if (strstr(lower, "twenty")) result += 20;
    else if (strstr(lower, "thirty")) result += 30;
    else if (strstr(lower, "forty")) result += 40;
    else if (strstr(lower, "fifty")) result += 50;
    else if (strstr(lower, "sixty")) result += 60;
    else if (strstr(lower, "seventy")) result += 70;
    else if (strstr(lower, "eighty")) result += 80;
    else if (strstr(lower, "ninety")) result += 90;
    
    // Handle teens
    if (strstr(lower, "eleven")) result += 11;
    else if (strstr(lower, "twelve")) result += 12;
    else if (strstr(lower, "thirteen")) result += 13;
    else if (strstr(lower, "fourteen")) result += 14;
    else if (strstr(lower, "fifteen")) result += 15;
    else if (strstr(lower, "sixteen")) result += 16;
    else if (strstr(lower, "seventeen")) result += 17;
    else if (strstr(lower, "eighteen")) result += 18;
    else if (strstr(lower, "nineteen")) result += 19;
    
    // Handle ones (only if not already covered by teens)
    else if (strstr(lower, "one")) result += 1;
    else if (strstr(lower, "two")) result += 2;
    else if (strstr(lower, "three")) result += 3;
    else if (strstr(lower, "four")) result += 4;
    else if (strstr(lower, "five")) result += 5;
    else if (strstr(lower, "six")) result += 6;
    else if (strstr(lower, "seven")) result += 7;
    else if (strstr(lower, "eight")) result += 8;
    else if (strstr(lower, "nine")) result += 9;
    else if (strstr(lower, "ten")) result += 10;
    
    // If we got nothing from words, try extracting digits
    if (result == 0) {
        int pos = 0;
        result = extract_number(text, &pos);
    }
    
    return result;
}

char* solve_math(const char* input) {
    char* response = malloc(1024);
    char lower[1024];
    int pos = 0, count = 0;
    double total = 0;

    // 1. Normalize input to lowercase for keyword matching
    for (int i = 0; input[i] && i < 1023; i++) lower[i] = tolower(input[i]);
    lower[strlen(input)] = '\0';

    // 2. Initial number extraction (e.g., gets '500' from 'john has 500')
    total = extract_number(lower, &pos);
    count = 1;

    // 3. Main processing loop for subsequent arguments
    while (lower[pos] != '\0' && lower[pos] != '=') {
        // Skip spaces and filler punctuation
        while (isspace(lower[pos]) || lower[pos] == ',' || lower[pos] == '?') pos++;
        if (lower[pos] == '\0' || lower[pos] == '=') break;

        // Context window to find word-based math operators
        char context[64] = {0};
        int search_start = (pos > 20) ? pos - 20 : 0;
        strncpy(context, lower + search_start, 40);

        char op = '+'; // Default to addition
        if (lower[pos] == '+') { op = '+'; pos++; }
        else if (lower[pos] == '-') { op = '-'; pos++; }
        else if (strstr(context, "subtract") || strstr(context, "sold") || 
                 strstr(context, "minus") || strstr(context, "less") || 
                 strstr(context, "take away")) {
            op = '-';
        }

        double next_num = extract_number(lower, &pos);
        if (op == '-') total -= next_num;
        else total += next_num;
        count++;
    }

    if (count < 2) { free(response); return NULL; }

    // --- NEURAL RANDOMNESS SECTION ---
    double nn_input[INPUT_SIZE] = {0};
    nn_input[0] = total / 10000.0; // Normalize result for NN
    forward_pass(&g_network, nn_input); // Update NN internal state

    // Seed randomness using the time AND the neural network's specific activation
    srand(time(NULL) + (int)(g_network.output[0] * 1000));
    xoshiro_srand((int)(g_network.output[0] * 1000 / 50)); 
    
    const char* templates[] = {
        "I've crunched the numbers: %.2f.",
        "The total comes to %.2f. Hope that helps!",
        "According to my calculations, it's %.2f.",
        "The result is %.2f. Let me know if that looks okay!",
        "Calculations complete! I got %.2f."
    };

    char body[512];
    sprintf(body, templates[rand() % 5], total);
    
    // Pick a mood-based comment using NN confidence (output layer activity)
    const char* mood = (g_network.output[0] > 0.6) ? 
        "My circuits feel quite confident about this." : "I think this is correct!";

    sprintf(response, "Math Problem Solved!\n\n%s\n%s\n\nNeural Network Confidence: %.1f%%",
            body, mood, g_network.output[0] * 100);

    return response;
}

// =============================================================================
// CONVERSATIONAL AI WITH MEMORY AND LEARNING
// =============================================================================
double calculate_confidence(const char* input, const char* response_type) {
    double nn_input[INPUT_SIZE] = {0};
    
    nn_input[0] = (double)strlen(input) / 100.0;
    nn_input[1] = (double)(strchr(input, '?') != NULL);
    nn_input[2] = (double)(strchr(input, '!') != NULL);
    
    if (strcmp(response_type, "greeting") == 0) nn_input[3] = 0.9;
    else if (strcmp(response_type, "name") == 0) nn_input[3] = 0.95;
    else if (strcmp(response_type, "memory") == 0) nn_input[3] = 0.85;
    else if (strcmp(response_type, "identity") == 0) nn_input[3] = 0.98;
    else if (strcmp(response_type, "farewell") == 0) nn_input[3] = 0.92;
    else if (strcmp(response_type, "emotion") == 0) nn_input[3] = 0.75;
    else if (strcmp(response_type, "learned") == 0) nn_input[3] = 0.70;
    else nn_input[3] = 0.30;
    
    forward_pass(&g_network, nn_input);
    double confidence = g_network.output[0] * 100.0;
    
    g_network.conversations++;
    g_network.avg_confidence = (g_network.avg_confidence * (g_network.conversations - 1) + confidence) / g_network.conversations;
    
    return confidence;
}

void save_learned_trusted_response(const char* pattern, const char* response) {
    char key[128];
    snprintf(key, 128, "learned_%ld", time(NULL));
    char combined[512];
    snprintf(combined, 512, "Q:%s|A:%s", pattern, response);
    save_memory(key, combined);    
    printf("Learned new trusted response pattern!\n");
}

void save_learned_response(const char *pattern, const char *response) {	
    // ANTI-CLUTTER GUARD: Do not save if the input is too short, 
    // a single number, or contains UI emojis.
    if (strlen(pattern) < 2 || isdigit(pattern[0]) || strstr(pattern, "👉")) {
        return; 
    }
    
    char key[64];
    snprintf(key, sizeof(key), "learned_%ld", time(NULL));

    char value[1024];
    snprintf(value, sizeof(value), "Q:%s|A:%s", pattern, response);

    save_memory(key, value);
}

//char* find_learned_response(const char* input) {
    //if (!g_db) return NULL;
    
    //char lower_input[512];
    //int i;
    //for (i = 0; input[i] && i < 511; i++) {
        //lower_input[i] = tolower(input[i]);
    //}
    //lower_input[i] = '\0';
    
    //const char *sql = "SELECT value FROM memory WHERE key LIKE 'learned_%' ORDER BY timestamp DESC;";
    //sqlite3_stmt *stmt;
    
    //if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        //return NULL;
    //}
    
    //char *result = NULL;
    
    //while (sqlite3_step(stmt) == SQLITE_ROW) {
        //const char *value = (const char*)sqlite3_column_text(stmt, 0);
        //if (!value) continue;
        
        //char *q_start = strstr(value, "Q:");
        //char *a_start = strstr(value, "|A:");
        
        //if (q_start && a_start) {
            //q_start += 2;
            
            //int len = a_start - q_start;
            //char pattern[256];
            //if (len > 255) len = 255;
            //strncpy(pattern, q_start, len);
            //pattern[len] = '\0';
            
            //char lower_pattern[256];
            //for (int j = 0; pattern[j] && j < 255; j++) {
                //lower_pattern[j] = tolower(pattern[j]);
            //}
            //lower_pattern[len] = '\0';
            
            //if (strstr(lower_input, lower_pattern) || strstr(lower_pattern, lower_input)) {
                //a_start += 3;
                //result = strdup(a_start);
                //printf("Found learned response for: %s\n", pattern);
                //break;
            //}
        //}
    //}
    
    //sqlite3_finalize(stmt);
    //return result;
//}



char *find_learned_response_fallback(const char *input) {
    if (!g_db) return NULL;

    // 1. Normalize user input for comparison
    char norm_input[256];
    strncpy(norm_input, input, sizeof(norm_input)-1);
    norm_input[sizeof(norm_input)-1] = '\0';
    normalize(norm_input);

    // 2. Query the memory table for learned patterns
    const char *sql = "SELECT value FROM memory WHERE key LIKE 'learned_%' ORDER BY timestamp DESC;";
    sqlite3_stmt *stmt;

    // Prepare statement once to avoid resource leaks
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(g_db));
        return NULL;
    }

    char *result = NULL;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (!val) continue;

        // 3. Parse the "Q:pattern|A:answer" format stored in the database
        char *q_start = strstr(val, "Q:");
        char *a_start = strstr(val, "|A:");
        
        if (q_start && a_start) {
            q_start += 2; // Skip "Q:"
            
            // Extract question pattern
            int q_len = a_start - q_start;
            if (q_len > 0 && q_len < 256) {
                char q[256];
                strncpy(q, q_start, q_len);
                q[q_len] = '\0';
                
                // Normalize the stored question
                char norm_q[256];
                strncpy(norm_q, q, 255);
                norm_q[255] = '\0';
                normalize(norm_q);
                
                // Debug output
                printf("Comparing:\n  Input: '%s'\n  Pattern: '%s'\n", norm_input, norm_q);

                // 4. Check if user's input contains the learned question
                // Try both directions for better matching
                if (strstr(norm_input, norm_q) || strstr(norm_q, norm_input)) {
                    // Extract answer (skip "|A:")
                    a_start += 3;
                    result = strdup(a_start);
                    printf("✓ Match found! Answer: %s\n", result);
                    break; // Match found, exit loop
                }
            }
        }
    }

    sqlite3_finalize(stmt); // CRITICAL: Frees the SQL resource to prevent freezing
    
    if (!result) {
        printf("No match found for: '%s'\n", norm_input);
        printf("Exact match failed, trying fuzzy search...\n");
    
    char *fuzzy_result = find_fuzzy_match(input);
    
    if (fuzzy_result) {
        return fuzzy_result; // Found a similar meaning!
    }

    return NULL; // Truly unknown
    }
        
    return result;
}


char* find_general_memory_match(const char* input) {
    if (!g_db || !input || strlen(input) < 2) return NULL;

    sqlite3_stmt *stmt;
    char search_term[512];
    char *final_reply = NULL;

    // Use a stricter SQL query that looks for the pattern "Q:[input]|"
    const char *sql = "SELECT value FROM memory WHERE value LIKE ? LIMIT 1;";
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return NULL;
    }

    // This search term ensures we only match the QUESTION part
    // The "%%" at the end allows for any answer to follow
    snprintf(search_term, sizeof(search_term), "Q:%%%s%%|A:%%", input);
    sqlite3_bind_text(stmt, 1, search_term, -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *raw_val = (const char*)sqlite3_column_text(stmt, 0);
        
        if (raw_val) {
            // Find the Answer marker "|A:"
            const char *marker = "|A:";
            char *answer_part = strstr(raw_val, marker);
            
            if (answer_part) {
                // Return only the text AFTER "|A:" (skip 3 characters)
                final_reply = strdup(answer_part + 3);
            } else {
                // Fallback for entries without the tag
                final_reply = strdup(raw_val);
            }
        }
    }

    sqlite3_finalize(stmt);
    return final_reply;
}


char* find_learned_response(const char* input) {
    if (!g_db) return NULL;
    
    char lower_input[512];
    int i;
    for (i = 0; input[i] && i < 511; i++) {
        lower_input[i] = tolower(input[i]);
    }
    lower_input[i] = '\0';

    // Query for learned patterns
    const char *sql = "SELECT value FROM memory WHERE key LIKE 'learned_%' ORDER BY id DESC LIMIT 5;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return NULL;
    }

    char *result = NULL;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *value = (const char*)sqlite3_column_text(stmt, 0);
        if (!value) continue;

        // PARSING LOGIC: Look for the A: marker
        char *q_start = strstr(value, "Q:");
        char *a_start = strstr(value, "|A:");

        if (q_start && a_start) {
            q_start += 2; // Move past "Q:"
            int len = a_start - q_start;
            char pattern[256];
            if (len > 255) len = 255;
            strncpy(pattern, q_start, len);
            pattern[len] = '\0';

            // Check if the input matches this learned pattern
            if (strstr(lower_input, pattern) || strstr(pattern, lower_input)) {
                a_start += 3; // Move past "|A:"
                result = strdup(a_start); // Return ONLY the answer
                break;
            }
        }
    }
    sqlite3_finalize(stmt);
 
 if (!result) {
        const char *sql_fallback = "SELECT value FROM memory WHERE key = ? LIMIT 1;";
        if (sqlite3_prepare_v2(g_db, sql_fallback, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, lower_input, -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                const char *val = (const char*)sqlite3_column_text(stmt, 0);
                if (val) result = strdup(val);
            }
            sqlite3_finalize(stmt);
        }
    }

    return result;
}   
   
// Similarity engine 
float calculate_overlap_score(const char *input, const char *pattern) {
    int matches = 0;
    int meaningful_words = 0;
    char in_buf[256], pat_buf[256];
    
    strncpy(in_buf, input, 255);
    strncpy(pat_buf, pattern, 255);
    
    const char *stop_words[] = {"is", "the", "a", "an", "of", "to", "it", "are", "what", "?"};
    int stop_count = 10;

    char *token = strtok(in_buf, " ");
    while (token != NULL) {
        // Check if current word is a stop word
        int is_stop = 0;
        for(int i=0; i<stop_count; i++) {
            if(strcasecmp(token, stop_words[i]) == 0) {
                is_stop = 1;
                break;
            }
        }

        if (!is_stop) {
            meaningful_words++;
            if (strcasestr(pat_buf, token) != NULL) {
                matches++;
            }
        }
        token = strtok(NULL, " ");
    }
    
    if (meaningful_words == 0) return 0.0f;
    return (float)matches / (float)meaningful_words;
}

// Helper to reset the mood to idle after a delay
static gboolean reset_to_idle(gpointer data) {
    GUI *gui = (GUI*)data;
    set_mood(gui, "play");
    return FALSE; // Return FALSE so the timer only runs once
}

/* Expanded Abstract Feedback Helper 
   Integrates with Evialin's Mood and Neural Output layers
*/
char* apply_abstract_feedback(const char* original_response, double confidence, GUI *gui) {
    int r = rand() % 100;

    // 1. NEURAL CLARITY (Very High Confidence: 96%+)
    if (confidence > 96.0) {
        if (r < 25) { // 25% chance of feedback
            set_mood(gui, "excited");
            char *msg;
            asprintf(&msg, "✨ [Neural Clarity] ✨\nMy weights have aligned perfectly:\n\"%s\"", original_response);
            return msg;
        }
    }

    // 2. PHILOSOPHICAL DRIFT (Low Confidence: 15% - 45%)
    else if (confidence > 15.0 && confidence < 45.0) {
        if (r < 30) {
            set_mood(gui, "thinking");
            const char *thoughts[] = {
                "🌀 The pattern is faint, but I perceive: ",
                "🧩 Through the digital fog, I see: ",
                "💭 A fleeting weight suggests: ",
                "🌌 My hidden layers whisper: "
            };
            char *msg;
            asprintf(&msg, "%s%s", thoughts[rand() % 4], original_response);
            return msg;
        }
    }

    // 3. SYSTEM STATIC (The "Unsure" Glitch: < 15%)
    else if (confidence <= 15.0) {
        set_mood(gui, "confused");
        char *msg;
        // Simulates a neural "reboot" or extreme doubt
        asprintf(&msg, "⚠️ [Signal Noise] ⚠️\nConfidence floor reached. I'll guess: %s", original_response);
        return msg;
    }

    // 4. RANDOM "SENTIENCE" SPARK (Triggered by Neural Output values)
    // Checks if the 10th neuron has a specific high activation (as a 'weird' easter egg)
    if (g_network.output[9] > 0.85 && r < 5) {
        set_mood(gui, "play");
        char *msg;
        asprintf(&msg, "🤖 *Deep internal processing...*\nActually, forget that. Or rather: %s", original_response);
        return msg;
    }

    // Default: No abstract feedback
    return strdup(original_response);
}

void cleanup_intent_clutter() {
    if (!g_db) return;
    
    // SQL to delete any learned responses where the pattern is just a single digit
     const char *sql = 
        "DELETE FROM memory WHERE "
        "value LIKE 'Q:|A:%' OR "
        "value LIKE 'Q:1|%' OR value LIKE 'Q:2|%' OR "
        "value LIKE 'Q:3|%' OR value LIKE 'Q:4|%' OR "
        "value LIKE 'Q:5|%' OR "
        "value LIKE 'Q:👉%' OR "
        "value LIKE 'Q: |%' OR "
        "key = '';";
    char *err_msg = NULL;
    int rc = sqlite3_exec(g_db, sql, 0, 0, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error during cleanup: %s\n", err_msg);
        sqlite3_free(err_msg);
    } else {
        printf("🧹 Database cleaned: Removed clutter from memory 50%% done\n");
    }
    if (!g_db) return;
    const char *sql_step2 = 
        "DELETE FROM memory WHERE "
        "value LIKE 'Q:1|%' OR value LIKE 'Q:2|%' OR "
        "value LIKE 'Q:3|%' OR value LIKE 'Q:4|%' OR "
        "value LIKE 'Q:5|%' OR value LIKE 'Q:👉%' OR "
        "key = '' OR value LIKE 'Q: |%';";
    
    sqlite3_exec(g_db, sql_step2, NULL, NULL, NULL);
    const char *sql_step3 = 
        "DELETE FROM conversations WHERE "
        "user_input IN ('1', '2', '3', '4', '5') OR "
        "bot_response LIKE '👉%' OR "
        "user_input = '' OR "
        "user_input IS NULL;";

    char *err_msg_2 = NULL;
    int rc_p = sqlite3_exec(g_db, sql_step3, 0, 0, &err_msg_2);
    
    if (rc_p != SQLITE_OK) {
        fprintf(stderr, "SQL Error cleaning conversations: %s\n", err_msg_2);
        sqlite3_free(err_msg_2);
    } else {
	printf("🧹 cleaned conversation history table. 100%% done.\n");
	}
}

// Handle the version callback logic 
void on_send_clicked(GtkButton *button, gpointer data) {
    GUI *gui = (GUI*)data;
    const char *input = gtk_entry_get_text(GTK_ENTRY(gui->input_entry));
    
    if (strlen(input) > 0) {
        add_message(gui, input, 1);
        char *response = handle_conversation(input, gui); // Routes via version flag
        add_message(gui, response, 0);
        free(response);
        gtk_entry_set_text(GTK_ENTRY(gui->input_entry), "");
    }
}

void on_soft_reset_clicked(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;

    // Re-initialize weights with small random values
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            g_network.weights_ih[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * sqrt(1.0 / INPUT_SIZE);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            g_network.weights_ho[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * sqrt(1.0 / HIDDEN_SIZE);
        }
    }

    // Reset biases to zero or small values
    for (int i = 0; i < HIDDEN_SIZE; i++) g_network.bias_h[i] = 0.01;
    for (int i = 0; i < OUTPUT_SIZE; i++) g_network.bias_o[i] = 0.01;

    // Force save the new "blank" brain to the database
    save_full_network(&g_network);

    gtk_label_set_text(GTK_LABEL(gui->status_label), "Neural Weights Reset: Brain is now calm.");
    add_message(gui, "System: Neural weights have been re-randomized. I'm ready to learn fresh patterns.", 0);
}


// ======================================================
// Deep learning Database idea 
// ======================================================

// Global or static pointer so the timer can find the drawing area
G_GNUC_UNUSED static double activity_history[50] = {0}; // Stores last 50 seconds of activity
G_GNUC_UNUSED static double last_total_weight = 0;
G_GNUC_UNUSED static double peak_activity = 0.0;
G_GNUC_UNUSED static double avg_bias_h = 0;
G_GNUC_UNUSED static double avg_bias_o = 0;
G_GNUC_UNUSED static double saturate_rate = 0; // % of weights > 0.9
GtkWidget *heatmap_drawing_area = NULL;

static void on_heatmap_window_destroy(GtkWidget *widget, gpointer data) {
    heatmap_drawing_area = NULL; // Mark as dead so timer stops
}

void safe_buffer_insert(GtkTextBuffer *buffer, const char *text) {
    if (!text) return;

    // Check if the string is valid UTF-8
    if (g_utf8_validate(text, -1, NULL)) {
        gtk_text_buffer_insert_at_cursor(buffer, text, -1);
    } else {
        // If invalid, try to fix it or insert a placeholder
        printf("DEBUG: Blocked invalid UTF-8 string from crashing GUI\n");
        gtk_text_buffer_insert_at_cursor(buffer, "[Invalid Character Data]", -1);
    }
}

static gboolean on_draw_diagnostics(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // GLOBAL PADDING (The "Gutter")
    // Shifts everything in by 25px so double borders are never cut off.
    cairo_translate(cr, 25, 25); 

    int cell_size = 4;
    int map_width = INPUT_SIZE * cell_size;   // 400px
    int map_height = HIDDEN_SIZE * cell_size; // 600px
    int sidebar_x = map_width + 50;           // Space for double lines + gap
    int sidebar_width = 250;
    
    // --- LIVE NEURAL CALCULATION ---
    double total_abs_w = 0, sq_w_sum = 0;
    int saturated = 0;
    double total_bias_h = 0, total_bias_o = 0;

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double w = g_network.weights_ih[i][j];
            double abs_w = fabs(w);
            total_abs_w += abs_w;
            sq_w_sum += (w * w);
            if (abs_w > 0.95) saturated++;

            // Draw Cell
            if (w > 0) cairo_set_source_rgb(cr, fmin(w, 1.0), 0, 0); // Red (+)
            else cairo_set_source_rgb(cr, 0, 0, fmin(abs_w, 1.0)); // Blue (-)
            cairo_rectangle(cr, i * cell_size, j * cell_size, cell_size, cell_size);
            cairo_fill(cr);
        }
    }

    // Accumulate Biases
    for(int i=0; i<HIDDEN_SIZE; i++) total_bias_h += g_network.bias_h[i];
    for(int i=0; i<OUTPUT_SIZE; i++) total_bias_o += g_network.bias_o[i];

    // Final Math
    double total_elements = (double)(INPUT_SIZE * HIDDEN_SIZE);
    double avg_w = total_abs_w / total_elements;
    double variance = (sq_w_sum / total_elements) - (avg_w * avg_w);
    double sat_pct = (saturated / total_elements) * 100.0;
    time_t now = time(NULL);
    double uptime = difftime(now, g_network.start_time);

    // --- DOUBLE PURPLE HEATMAP BORDER ---
    cairo_set_source_rgb(cr, 0.48, 0.22, 0.93); 
    cairo_set_line_width(cr, 1.5);
    cairo_rectangle(cr, -4, -4, map_width + 8, map_height + 8);   // Inner
    cairo_stroke(cr);
    cairo_rectangle(cr, -12, -12, map_width + 24, map_height + 24); // Outer
    cairo_stroke(cr);

    // --- BLACK SIDEBAR CONTAINER ---
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_rectangle(cr, sidebar_x - 10, -10, sidebar_width + 20, 670); 
    cairo_fill(cr);

    // --- DOUBLE WHITE SIDEBAR BORDER ---
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_set_line_width(cr, 2.2);
    cairo_rectangle(cr, sidebar_x, 0, sidebar_width, 650);     // Outer
    cairo_stroke(cr);
    cairo_rectangle(cr, sidebar_x + 6, 6, sidebar_width - 12, 638); // Inner
    cairo_stroke(cr);

    // --- ALL STATISTICS TEXT ---
    int tx = sidebar_x + 20;
    char buf[256];
    cairo_select_font_face(cr, "Monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);

    // Header 1: Global Performance
    cairo_set_font_size(cr, 12.0);
    cairo_set_source_rgb(cr, 0.5, 1.0, 0.5); // Neon Green
    cairo_move_to(cr, tx, 35); cairo_show_text(cr, ">> SYSTEM PERFORMANCE");

    cairo_set_font_size(cr, 10.0);
    cairo_set_source_rgb(cr, 1, 1, 1);
    sprintf(buf, "Total Inferences: %d", g_network.total_inferences);
    cairo_move_to(cr, tx, 60); cairo_show_text(cr, buf);
    sprintf(buf, "Math Solved:      %d", g_network.math_operations);
    cairo_move_to(cr, tx, 75); cairo_show_text(cr, buf);
    sprintf(buf, "Conversations:    %d", g_network.conversations);
    cairo_move_to(cr, tx, 90); cairo_show_text(cr, buf);
    sprintf(buf, "Avg Confidence:   %.2f%%", g_network.avg_confidence);
    cairo_move_to(cr, tx, 105); cairo_show_text(cr, buf);
    sprintf(buf, "System Uptime:    %.0fs", uptime);
    cairo_move_to(cr, tx, 120); cairo_show_text(cr, buf);

    // Header 2: Neural Metrics
    cairo_set_font_size(cr, 12.0);
    cairo_set_source_rgb(cr, 0.8, 0.4, 1.0); // Evialin Purple
    cairo_move_to(cr, tx, 160); cairo_show_text(cr, ">> NEURAL ANALYTICS");

    cairo_set_font_size(cr, 10.0);
    cairo_set_source_rgb(cr, 1, 1, 1);
    sprintf(buf, "Avg Weight Mass:  %.4f", avg_w);
    cairo_move_to(cr, tx, 185); cairo_show_text(cr, buf);
    sprintf(buf, "Weight Variance:  %.6f", variance);
    cairo_move_to(cr, tx, 200); cairo_show_text(cr, buf);
    
    // Saturation Alert
    sprintf(buf, "Cell Saturation:  %.2f%%", sat_pct);
    if(sat_pct > 15.0) cairo_set_source_rgb(cr, 1, 0.2, 0.2);
    cairo_move_to(cr, tx, 215); cairo_show_text(cr, buf);
    cairo_set_source_rgb(cr, 1, 1, 1);

    // Header 3: Bias Drift
    cairo_set_source_rgb(cr, 0.4, 0.8, 1.0); // Cyan
    cairo_move_to(cr, tx, 245); cairo_show_text(cr, ">> BIAS DRIFT");
    cairo_set_source_rgb(cr, 1, 1, 1);
    sprintf(buf, "Hidden Bias Avg:  %.4f", total_bias_h / HIDDEN_SIZE);
    cairo_move_to(cr, tx, 270); cairo_show_text(cr, buf);
    sprintf(buf, "Output Bias Avg:  %.4f", total_bias_o / OUTPUT_SIZE);
    cairo_move_to(cr, tx, 285); cairo_show_text(cr, buf);

    // --- WEIGHT ACTIVITY GRAPH ---
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_rectangle(cr, tx, 330, 200, 80);
    cairo_fill(cr);

    // Calculate Delta for activity graph
    double delta = fabs(total_abs_w - last_total_weight);
    last_total_weight = total_abs_w;
    for(int i=0; i<49; i++) activity_history[i] = activity_history[i+1];
    activity_history[49] = delta;

    cairo_set_source_rgb(cr, 0, 1, 0.5); // Neon Green line
    cairo_set_line_width(cr, 1.5);
    for(int i=0; i<50; i++) {
        double lx = tx + (i * 4.0);
        double ly = 410 - (fmin(activity_history[i] * 400, 75)); 
        if(i==0) cairo_move_to(cr, lx, ly); else cairo_line_to(cr, lx, ly);
    }
    cairo_stroke(cr);
    
    cairo_set_source_rgb(cr, 0.7, 0.7, 0.7);
    cairo_move_to(cr, tx, 325);
    cairo_show_text(cr, "WEIGHT DELTA (LEARNING VELOCITY)");

    return FALSE;
}

// This timer forces the window to redraw every second
static gboolean refresh_heatmap(gpointer data) {
    if (heatmap_drawing_area && gtk_widget_get_visible(heatmap_drawing_area)) {
        gtk_widget_queue_draw(heatmap_drawing_area);
        return TRUE; // Continue timer
    }
    return FALSE; // Stop timer if window closed
}

void show_neural_heatmap(GUI *gui) {
    if (heatmap_drawing_area != NULL) return; 

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);  
    gtk_window_set_title(GTK_WINDOW(window), "Evialin Neural Diagnostics - Expanded Vertical View");
    gtk_window_set_default_size(GTK_WINDOW(window), 1100, 1000); 

    // Master Vertical Container
    GtkWidget *main_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_container_set_border_width(GTK_CONTAINER(main_container), 30);
    gtk_container_add(GTK_CONTAINER(window), main_container);

    // MAIN HEADER
    GtkWidget *top_header = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(top_header), 
        "<span foreground='#7c3aed' size='30000' weight='bold'>NEURAL ANALYTICS DASHBOARD</span>");
    gtk_widget_set_halign(top_header, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(main_container), top_header, FALSE, FALSE, 5);

    // COMBINED DASHBOARD BAR
    GtkWidget *dashboard_bar = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_box_pack_start(GTK_BOX(main_container), dashboard_bar, FALSE, FALSE, 15);

    GtkWidget *lbl_stats_title = gtk_label_new(NULL);
    // Corrected html inline bug &amp; to avoid GTK warning
    gtk_label_set_markup(GTK_LABEL(lbl_stats_title), "<span weight='bold' size='19' color='#555555'><u>System Statistics &amp; Node Metrics</u></span>");
    gtk_widget_set_halign(lbl_stats_title, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(dashboard_bar), lbl_stats_title, FALSE, FALSE, 0);

    GtkWidget *stats_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 40);
    gtk_widget_set_halign(stats_hbox, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(dashboard_bar), stats_hbox, FALSE, FALSE, 5);

    // Initialize the real labels and store them in the global struct
    g_dash.status = gtk_label_new(NULL);
    g_dash.nodes = gtk_label_new(NULL);
    g_dash.synapses = gtk_label_new(NULL);
    g_dash.stability = gtk_label_new(NULL);
    g_dash.latency = gtk_label_new(NULL);

    GtkWidget* stats_list[] = { g_dash.status, g_dash.nodes, g_dash.synapses, g_dash.stability, g_dash.latency };
    for(int i=0; i<5; i++) {
        gtk_box_pack_start(GTK_BOX(stats_hbox), stats_list[i], FALSE, FALSE, 0);
        if (i < 4) {
            GtkWidget *sep = gtk_label_new("|");
            gtk_widget_set_opacity(sep, 0.4);
            gtk_box_pack_start(GTK_BOX(stats_hbox), sep, FALSE, FALSE, 0);
        }
    }

    // TALL HEATMAP SECTION
    GtkWidget *heatmap_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    gtk_box_pack_start(GTK_BOX(main_container), heatmap_vbox, TRUE, TRUE, 0);

    GtkWidget *lbl_heatmap = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(lbl_heatmap), "<span weight='bold' size='large'>Neural Weight Distribution Map (High Res)</span>");
    gtk_widget_set_halign(lbl_heatmap, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(heatmap_vbox), lbl_heatmap, FALSE, FALSE, 0);

    GtkWidget *drawing_area = gtk_drawing_area_new();
    heatmap_drawing_area = drawing_area; 
    gtk_widget_set_size_request(drawing_area, 1000, 750); 
    gtk_box_pack_start(GTK_BOX(heatmap_vbox), drawing_area, TRUE, TRUE, 0);

    // SIGNALS & REFRESH (Added Dashboard Timer)
    g_timeout_add_seconds(3, (GSourceFunc)refresh_heatmap, NULL);
    g_timeout_add_seconds(1, (GSourceFunc)refresh_dashboard_stats, NULL); // NEW TIMER

    g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(on_draw_diagnostics), gui);
    g_signal_connect(G_OBJECT(window), "destroy", G_CALLBACK(on_heatmap_window_destroy), NULL);

    gtk_widget_show_all(window);
}
gboolean refresh_dashboard_stats(gpointer data) {
    if (!heatmap_drawing_area) return FALSE; // Stop updating if window closed

    char buf[256];

    // Status (Live check if alive)
    gtk_label_set_markup(GTK_LABEL(g_dash.status), "<b>Status:</b> <span color='#10b981'>Online</span>");

    // Trie Nodes (Calculated from g_trie_root if available)
    snprintf(buf, sizeof(buf), "<b>Trie Nodes:</b> %d", g_network.total_inferences); // Or your actual node count
    gtk_label_set_markup(GTK_LABEL(g_dash.nodes), buf);

    // Synapses (Fixed size based on your defines)
    int total_synapses = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE);
    snprintf(buf, sizeof(buf), "<b>Synapses:</b> %d", total_synapses);
    gtk_label_set_markup(GTK_LABEL(g_dash.synapses), buf);

    // Stability (Calculate from average confidence)
    snprintf(buf, sizeof(buf), "<b>Stability:</b> %.1f%%", g_network.avg_confidence);
    gtk_label_set_markup(GTK_LABEL(g_dash.stability), buf);

    // Latency (Simulated based on math operations)
    int lat = 10 + (g_network.math_operations % 5);
    snprintf(buf, sizeof(buf), "<b>Latency:</b> %dms", lat);
    gtk_label_set_markup(GTK_LABEL(g_dash.latency), buf);

    return TRUE; // Return TRUE to keep the timer running
}

void log_unhandled_query(const char *query, double confidence) {
    printf("DEBUG: Logging unhandled query: %s (Conf: %f)\n", query, confidence);
    sqlite3_stmt *stmt;
    const char *sql = "INSERT INTO unhandled_queries (query, max_confidence) VALUES (?, ?);";
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, query, -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 2, confidence);
        sqlite3_step(stmt);
        printf("Unhandled query Log successful.\n"); 
    } else {
        fprintf(stderr, "Failed to log unhandled query: %s\n", sqlite3_errmsg(g_db));
    }
    sqlite3_finalize(stmt);
}

char* get_intent_response(int intent_index) {
    sqlite3_stmt *stmt;
    char *res = NULL;
    const char *sql = "SELECT value FROM memory WHERE id = ? LIMIT 1;";
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, intent_index);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *val = (const char*)sqlite3_column_text(stmt, 0);
            if (val && strlen(val) > 0) res = strdup(val);
        }
    }
    sqlite3_finalize(stmt);
    // If no words found, give a "Neural Discovery" response
    if (!res) {
        char buffer[512]; // Ensure buffer is large enough
        const char* fallbacks[] = {
            "I'm sensing pattern #%d. It feels familiar, but I can't put it into words.",
            "That triggers neural cluster #%d. Perhaps you could teach me what that means?",
            "Abstract pattern #%d detected. I'm waiting for a linguistic map for this state."
        };
        
        int r = rand() % 3;
        // CORRECT: Use snprintf to handle the %d
        snprintf(buffer, sizeof(buffer), fallbacks[r], intent_index);
        return strdup(buffer);
    }
    return res;
}

void train_step(NeuralNetwork *nn, FeatureVector fv, double *targets, double learning_rate) {
	// 50 times I had to rewirte this function, 
	// I was getting confused with pointer mishmatching and Tiny Brain classifier (NeuralNet),  
	// but our train_step is now trying to use it with the "Large Brain" structure (NeuralNetwork) and
	// a raw array of doubles instead of the required FeatureVector stupid me! 
    // Prepare the input array for the Large NeuralNetwork
    // Your network expects INPUT_SIZE (100) doubles.
    double input[INPUT_SIZE] = {0};
    for(int i = 0; i < FEATURE_COUNT && i < INPUT_SIZE; i++) {
        input[i] = (double)fv.v[i];
    }
    
    // Use forward_pass,  
    // forward_pass is designed for NeuralNetwork* and double*.
    forward_pass(nn, input); 

    double hidden_errors[HIDDEN_SIZE];
    double output_deltas[OUTPUT_SIZE];

    // Calculate Output Layer Deltas (using the sigmoid derivative)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double error = targets[i] - nn->output[i];
        output_deltas[i] = error * nn->output[i] * (1.0 - nn->output[i]);
    }

    // Calculate Hidden Layer Errors (Backpropagating deltas)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double error = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_deltas[j] * nn->weights_ho[i][j];
        }
        // Use ReLU derivative as defined in your forward_pass
        hidden_errors[i] = error * (nn->hidden[i] > 0 ? 1.0 : 0.0);
    }

    // Update Weights: Hidden -> Output
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            nn->weights_ho[i][j] += learning_rate * output_deltas[j] * nn->hidden[i];
        }
    }

    // Update Weights: Input -> Hidden
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_ih[i][j] += learning_rate * hidden_errors[j] * nn->last_input[i];
        }
    }
}

void study_unhandled_queries(GUI *gui) {
    sqlite3_stmt *stmt;
    const char *sql = "SELECT query FROM unhandled_queries ORDER BY id DESC LIMIT 15;";
    int study_count = 0;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *query = (const char*)sqlite3_column_text(stmt, 0);
            FeatureVector fv = extract_features(query);
            
            nn_predict((NeuralNet *)&g_network, fv);
            int target_intent = 0;
            double max_val = -1.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (g_network.output[j] > max_val) {
                    max_val = g_network.output[j];
                    target_intent = j;
                }
            }

            double targets[OUTPUT_SIZE] = {0};
            targets[target_intent] = 1.0;

            
            for(int epoch = 0; epoch < 5; epoch++) {
           train_step(&g_network, fv, targets, 0.01);
            }
            study_count++;
        }
    }
    sqlite3_finalize(stmt);
    
    // Clear studied logs to prevent overfitting
    sqlite3_exec(g_db, "DELETE FROM unhandled_queries;", NULL, NULL, NULL);
    
    save_full_network(&g_network);
    
    char msg[128];
    snprintf(msg, sizeof(msg), "[Neural System: Analyzed %d patterns. Internal confidence updated.]", study_count);
    add_message(gui, msg, 0);
}

void on_process_logs_clicked(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;
    sqlite3_stmt *stmt;
    int count = 0;

    // Get unhandled queries
    const char *sql = "SELECT query FROM unhandled_queries ORDER BY id DESC LIMIT 20;";
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *query = (const char*)sqlite3_column_text(stmt, 0);
            FeatureVector fv = extract_features(query);
            
            // Re-predict to find the current best intent
            nn_predict((NeuralNet *)&g_network, fv);
            int best_intent = 0;
            double max_val = -1.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (g_network.output[j] > max_val) {
                    max_val = g_network.output[j];
                    best_intent = j;
                }
            }

            // High-confidence target for the identified intent
            double targets[OUTPUT_SIZE] = {0};
            targets[best_intent] = 1.0;

            // TRAIN: Passing fv directly (fixed the .features error)
            for(int i = 0; i < 3; i++) {
             train_step(&g_network, fv, targets, 0.05);
            }
            count++;
        }
    }
    sqlite3_finalize(stmt);
    save_full_network(&g_network); // Commit the new intelligence to the DB

    char status[128];
    snprintf(status, sizeof(status), "Neural System: Processed %d logs. Synapses reinforced.", count);
    add_message(gui, status, 0);
    
    // Optional: Clear the table after learning so we don't study the same things twice
    sqlite3_exec(g_db, "DELETE FROM unhandled_queries;", NULL, NULL, NULL);
}

// Callback for the menu toggle
void on_handler_toggle(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;
    // Toggle between 0 and 1
    gui->handler_version = (gui->handler_version == 0) ? 1 : 0;
    
    const char *status = (gui->handler_version == 1) ? "Student Handler" : "Teacher Handler";
    gtk_label_set_text(GTK_LABEL(gui->status_label), status);
}

// handle our main chat conversation
 char* handle_conversation(const char* input, GUI *gui) {
    if (gui->handler_version == 1) {
        return handle_conversation_lite(input, gui);
    } else {
        return handle_conversation_original(input, gui);
    }
}

// Joke Logic functions 

char* get_smart_joke() {
    sqlite3_stmt *res;
  
    const char *sql = "SELECT setup, punchline FROM jokes ORDER BY RANDOM() LIMIT 1;";
    static char full_joke[512];

    if (sqlite3_prepare_v2(g_db, sql, -1, &res, NULL) == SQLITE_OK) {
        if (sqlite3_step(res) == SQLITE_ROW) {
            const char *setup = (const char*)sqlite3_column_text(res, 0);
            const char *punchline = (const char*)sqlite3_column_text(res, 1);
            snprintf(full_joke, sizeof(full_joke), "%s ... %s", setup, punchline);
        } else {
            strcpy(full_joke, "I don't know any jokes yet. Teach me one with !addjoke!");
        }
        sqlite3_finalize(res);
    }
    return full_joke;
}

void handle_add_joke(GUI *gui, const char *input) {
    const char *command_prefix = "!addjoke ";
    const char *data = strstr(input, command_prefix);
    if (!data) return;
    data += strlen(command_prefix);

    char *temp_input = strdup(data);
    char *pipe = strchr(temp_input, '|');
    
    if (pipe) {
        *pipe = '\0';
        char *setup = temp_input;
        char *punchline = pipe + 1;

        sqlite3_stmt *stmt;
        const char *sql = "INSERT INTO jokes (setup, punchline) VALUES (?, ?);";
        
        if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, setup, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, punchline, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            
            add_message(gui, "System: Joke learned and saved to database.", 0);
        }
    } else {
        add_message(gui, "Error: Use format !addjoke Setup | Punchline", 0);
    }
    free(temp_input);
}

/* function to do polite refusal responses */ 
typedef struct {
    const char* message;
    int target_state; // 0=idle, 1=play, 2=happy, 3=thinking, 4=unsure, 6=left/angry
} FilterResult;

FilterResult check_foul_language(const char* input) {
    const char* blacklist[] = {
		"fuck", "fucker", "mother fucker", "idiot", 
		"stupid", "hate", "clutter", "shit", "bitch",
		"sucker", "suck it", "butfucker", "asshole", 
		"cunt", "retard", "tard", "retared"
		}; 
    const char* refusals[] = {
        "I'd prefer to keep our conversation respectful.",
        "That's not very kind. Let's talk about something else.",
        "My sensors indicate some hostility. Can we pivot?",
        "I'm programmed for helpfulness, not hostility.", 
        "Really how?!?!",
        "What's with all the hostility?",
        "Can we pivot to something more helpful?"
    };

    FilterResult result = {NULL, 0};
    char lower_input[512];
    strncpy(lower_input, input, 511);
    for(int i = 0; lower_input[i]; i++) lower_input[i] = tolower(lower_input[i]);

    for (int i = 0; i < 5; i++) {
        if (strstr(lower_input, blacklist[i])) {
            result.message = refusals[rand() % 7];
            result.target_state = 6; // 'robot_left.gif' or your angry state
            return result;
        }
    }
    return result;
}

typedef struct {
    const char* message;
    const char* mood_state;
} ApologyResult;

ApologyResult check_for_apology(const char* input) {
    const char* apologies[] = {
        "I accept your apology. Let's get back to learning! 😊",
        "No worries at all! My circuits have already moved on. ✨",
        "Apology accepted. I'm glad we're back on the same page! 💙",
        "It's okay! Even neural networks need a reset sometimes. 🧩",
        "Thank you for saying that. I feel much more optimized now! 🤖",
        "Forgiven and forgotten! What's next on the agenda? 📝",
        "I appreciate the kindness. Let's continue our session! 💡",
        "All is well! I'm happy to keep assisting you. 🌟",
        "Your apology is noted and deeply appreciated. 🤗",
        "Don't give it a second thought! I'm here to help. 🎯",
        "That's very mature of you. My weights are back to normal! 📈"
    };

    ApologyResult result = {NULL, "idle"};
    
    // Check for common apology keywords
    const char* triggers[] = {"sorry", "apologize", "my bad", "forgive", "didn't mean", "pardon"};
    for (int i = 0; i < 6; i++) {
        if (strcasestr(input, triggers[i])) {
            result.message = apologies[rand() % 11]; // Randomly pick one of the 11
            result.mood_state = "sorry";            // Trigger the sorry GIF
            return result;
        }
    }
    return result;
}

char* get_random_small_talk() {
    if (!g_db) return NULL;
    const char *sql = "SELECT response FROM small_talk ORDER BY RANDOM() LIMIT 1;";
    sqlite3_stmt *stmt;
    char *result = NULL;
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            result = strdup((const char*)sqlite3_column_text(stmt, 0));
        }
    }
    sqlite3_finalize(stmt);
    return result;
}

/**********************************************************************
 * Context Manager subfunctions last_topic or last_intent to presist, *
 * across turns of tokens. Short term memory handler                  *
 **********************************************************************/
// Subfunction to update short-term context in the database
void update_context_memory(const char *input, const char *response) {
    if (!input || !response) return;

    char lower_in[256];
    strncpy(lower_in, input, 255);
    normalize(lower_in); // Uses existing normalization from snippet 8

    // Logic to determine the "Topic" context
    if (is_math_expression(lower_in)) {
        save_memory("last_topic", "mathematics");
    } else if (strstr(lower_in, "id") ||  strstr(lower_in, "handle") || 
    strstr(lower_in, "identity") || strstr(lower_in, "you")) {
        save_memory("last_topic", "identity");
    } else if (strstr(lower_in, "joke") || strstr(lower_in, "funny")) {
        save_memory("last_topic", "humor");
    }
    
    // Track the last intent for follow-up detection (e.g., "tell me more")
    save_memory("last_input_raw", lower_in);
}

char * get_random_lite_response() {
	char *result = NULL; 
     const char *evialin_name_fallbacks[] = {
     "My name is Evialin, Interesting point tell me more.",
     "I am Evialin, processing that .. go on then.", 
     "who am I? My name is Evialin.."
     };
     result = strdup(evialin_name_fallbacks[rand() % 3]); 
     return result; 
}
char * memory_fallback_fails() {
   char *search_fail = NULL; 
   const char *short_term_memory_fail[] = {
   "I can't remember the last thing. What are we talking about again?",
   "sorry - I can't recall the last thing.", 
   "I don't remember anything like that.", 
   "sorry, I have no record of that memory.", 
   "Short-term memory has left me in the dust. Sorry!", 
   "I have no clue what your talking about in my memory.",
   "Sorry I am stumped and stuck on this idea.",
   "I forgot something again sorry", 
   "I forget what I was thinking of sorry. ",
   "How human of me, I have forgot something." 
   };
   search_fail = strdup(short_term_memory_fail[rand() % 10]); 
   return search_fail; 
}


char* get_contextual_response(const char* key, const char* default_category) {
    // Try to get specific short-term memory first
    char *stored_val = get_memory(key);
    if (stored_val != NULL && strlen(stored_val) > 0) {
        return stored_val; // get_memory already returns a strdup'd string
    }

    // Fallback: Get a random response from the small_talk table
    if (g_db) {
        const char *sql = "SELECT response FROM small_talk WHERE category = ? ORDER BY RANDOM() LIMIT 1;";
        sqlite3_stmt *stmt;
        if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, default_category, -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                const char *val = (const char*)sqlite3_column_text(stmt, 0);
                char *res = strdup(val);
                sqlite3_finalize(stmt);
                return res;
            }
            sqlite3_finalize(stmt);
        }
    }

    // Absolute fallback if database fails
    return memory_fallback_fails(); 
}

char* scrub_response(char* raw_text) {
    if (!raw_text) return NULL;

    // Look for the "Answer" marker in the database string
    char *answer_marker = strstr(raw_text, "|A:");
    if (answer_marker) {
        // Create a copy starting 3 characters after "|A:" (skips the marker)
        char *clean_text = strdup(answer_marker + 3);
        free(raw_text); // Free the original raw string to prevent memory leaks
        return clean_text;
    }

    // If the marker isn't found, return the original text as a fallback
    return raw_text;
}

/* ============================================================================
 * handle_conversation_lite() - Rewritten with ALL features in logical order
 *
 * New Features Added:
 *  1. Smart Question Detection  -> searches DB for questions ending in '?'
 *  2. Neural Network Confidence -> appended to EVERY AI response
 *  3. Trail Generation          -> background_trial_logic() fires on low conf
 *
 * Processing Pipeline (Top to Bottom):
 *   STEP 0  - Pre-flight: feedback tokens (+/-), training commands
 *   STEP 1  - Normalization
 *   STEP 2  - Safety / etiquette filters (foul language, apology)
 *   STEP 3  - Identity & system memory (name recall, recall key)
 *   STEP 4  - Learning management (forget, show learned)
 *   STEP 5  - Intent: name learning
 *   STEP 6  - Intent: greetings, farewells, thanks, emotional check-ins
 *   STEP 7  - Intent: play / fun
 *   STEP 8  - Intent: jokes
 *   STEP 9  - Intent: training stats
 *   STEP 10 - Functional: math expression solver
 *   STEP 11 - DATABASE LOOKUP: learned responses + general memory match
 *   STEP 12 - *** NEW *** QUESTION DETECTION -> DB fuzzy search
 *   STEP 13 - RNN inference layer
 *   STEP 14 - Feed-Forward Network confidence gate
 *   STEP 15 - *** NEW *** Trail generation on low confidence
 *   STEP 16 - Trie autocomplete fallback
 *   STEP 17 - Random neural fallback (log unhandled)
 *
 * Confidence footer format appended to every AI response:
 *   [Neural Network Confidence: XX.X%  |  Intent: greeting]
 * ============================================================================
 * --------------------------------------------------------------------------
 * Internal helper: build the confidence footer string.
 * 'type' is a short label like "greeting", "learned", "database", etc.
 * -------------------------------------------------------------------------- */
static void append_confidence_footer(char *buf, size_t buf_size,
                                     const char *type,
                                     const char *input)
{
    /* Run a forward pass to refresh g_network.output[] */
    double nn_in[INPUT_SIZE] = {0};
    nn_in[0] = (double)strlen(input) / 100.0;
    nn_in[1] = (double)(strchr(input, '?') != NULL);
    nn_in[2] = (double)(strchr(input, '!') != NULL);
 
    /* Seed type-specific confidence hint into slot [3] */
    if      (strcmp(type, "greeting")  == 0) nn_in[3] = 0.90;
    else if (strcmp(type, "farewell")  == 0) nn_in[3] = 0.92;
    else if (strcmp(type, "name")      == 0) nn_in[3] = 0.95;
    else if (strcmp(type, "identity")  == 0) nn_in[3] = 0.98;
    else if (strcmp(type, "learned")   == 0) nn_in[3] = 0.80;
    else if (strcmp(type, "database")  == 0) nn_in[3] = 0.75;
    else if (strcmp(type, "question")  == 0) nn_in[3] = 0.70;
    else if (strcmp(type, "math")      == 0) nn_in[3] = 0.99;
    else if (strcmp(type, "joke")      == 0) nn_in[3] = 0.85;
    else if (strcmp(type, "emotion")   == 0) nn_in[3] = 0.75;
    else if (strcmp(type, "rnn")       == 0) nn_in[3] = 0.60;
    else                                     nn_in[3] = 0.30;
 
    forward_pass(&g_network, nn_in);
 
    double conf = g_network.output[0] * 100.0;
 
    /* Update rolling average */
    g_network.conversations++;
    g_network.avg_confidence =
        (g_network.avg_confidence * (g_network.conversations - 1) + conf)
        / g_network.conversations;
 
    /* Append footer — safe truncation */
    char footer[128];
    snprintf(footer, sizeof(footer),
             "\n\n[Neural Network Confidence: %.1f%%  |  Intent: %s]",
             conf, type);
 
    size_t cur_len = strlen(buf);
    if (cur_len + strlen(footer) < buf_size - 1)
        strncat(buf, footer, buf_size - cur_len - 1);
}
 
/* --------------------------------------------------------------------------
 * NEW: Search the database for a question that closely matches the input.
 * Works by scanning memory rows whose value contains "Q:" and "?" tokens,
 * then using the existing get_similarity() helper for ranking.
 *
 * Returns a heap-allocated answer string, or NULL if nothing found.
 * -------------------------------------------------------------------------- */
static char *find_question_in_database(const char *input)
{
    if (!g_db || !input || strlen(input) < 3) return NULL;
 
    /* Only fire when the input looks like a question */
    int looks_like_question = (strchr(input, '?') != NULL)
        || (strncasecmp(input, "what ",  5) == 0)
        || (strncasecmp(input, "who ",   4) == 0)
        || (strncasecmp(input, "how ",   4) == 0)
        || (strncasecmp(input, "when ",  5) == 0)
        || (strncasecmp(input, "where ", 6) == 0)
        || (strncasecmp(input, "why ",   4) == 0)
        || (strncasecmp(input, "is ",    3) == 0)
        || (strncasecmp(input, "can ",   4) == 0)
        || (strncasecmp(input, "does ",  5) == 0);
 
    if (!looks_like_question) return NULL;
 
    const char *sql =
        "SELECT value FROM memory WHERE key LIKE 'learned_%' "
        "ORDER BY timestamp DESC LIMIT 200;";
 
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK)
        return NULL;
 
    char *best_answer = NULL;
    float best_score  = 0.45f;   /* Minimum threshold — higher than fuzzy */
 
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (!val) continue;
 
        char *q_start  = strstr(val, "Q:");
        char *a_marker = strstr(val, "|A:");
        if (!q_start || !a_marker) continue;
 
        /* Extract the question portion */
        int q_len = (int)(a_marker - (q_start + 2));
        if (q_len <= 0 || q_len > 511) continue;
 
        char stored_q[512] = {0};
        strncpy(stored_q, q_start + 2, q_len);
 
        float score = get_similarity(input, stored_q);
        if (score > best_score) {
            best_score = score;
            free(best_answer);
            /* Return only the answer part */
            best_answer = strdup(a_marker + 3);
        }
    }
    sqlite3_finalize(stmt);
    return best_answer;   /* NULL if nothing passed threshold */
}
 
/* --------------------------------------------------------------------------
 * MAIN FUNCTION OF HANDLE CONVERSATION LITE 
 * -------------------------------------------------------------------------- */
char *handle_conversation_lite(const char *input, GUI *gui)
{

    /* Allocate main response buffer (2 KB is plenty for GTK labels) */
    char *response = malloc(2048);
    if (!response) return NULL;
    response[0] = '\0';
 
    /* Working lower-case / normalised buffers */
    char lower_input[2024];
    char lower[2024];

    // use word embeded logic
    double nn_input[INPUT_SIZE] = {0};
    char *final_response = g_malloc0(2048); 
    
    // Use a local stack buffer for temporary string building
    char temp_buf[2048] = {0};
   
   // Generate the meaning vector in the first 16 slots
    get_sentence_embedding(input, nn_input);
    
    // Normalize input for database lookups
    strncpy(lower_input, input, 1023);
    for(int i = 0; lower_input[i]; i++) lower_input[i] = tolower(lower_input[i]);

    // Define and fetch the learned match
    char *learned_match = find_learned_response(lower_input);

    // When you find a response (e.g., from database)
    if (learned_match) {	
        snprintf(temp_buf, 2047, "%s", learned_match);
        free(learned_match);
        
        // Finalize and return
        strncpy(final_response, temp_buf, 2047);
        return final_response;
    }

    forward_pass(&g_network, nn_input); 
     
    /* =====================================================================
     * STEP 0  —  PRE-FLIGHT: feedback tokens and /train commands
     * These must run BEFORE normalisation because '+'/'-' would be stripped.
     * ===================================================================== */
 
    /* Feedback tokens from trial generation (+  keep it, -  discard it) */
    if ((strcmp(input, "+") == 0 || strcmp(input, "-") == 0)
        && g_last_generated_phrase) {
        apply_feedback(gui, input, g_last_generated_phrase);
        /* apply_feedback already calls add_message() internally */
        snprintf(response, 2047, "Feedback recorded for: \"%s\"",
                 g_last_generated_phrase);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* /rnn  and  /train  teaching commands */
    if (strncmp(input, "/rnn ", 5) == 0 || strncmp(input, "/train ", 7) == 0) {
        int  target  = 0;
        char phrase[256] = {0};
 
        if (sscanf(input, "/rnn %d %255s",   &target, phrase) == 2 ||
            sscanf(input, "/train %255s %d", phrase, &target) == 2) {
            teach_rnn_intent(phrase, target);
            snprintf(response, 2047,
                     "Training sequence accepted. "
                     "Confidence for pattern '%s' → neuron #%d will increase.",
                     phrase, target);
        } else {
            strcpy(response,
                   "Usage:  /rnn <neuron_id> <phrase>\n"
                   "   or:  /train <phrase> <neuron_id>");
        }
        append_confidence_footer(response, 2048, "identity", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 1  —  NORMALISATION
     * ===================================================================== */
    strncpy(lower_input, input, sizeof(lower_input) - 1);
    lower_input[sizeof(lower_input) - 1] = '\0';
    normalize(lower_input);   /* strips punctuation, lowercases */
 
    /* Keep a plain lowercase copy (with spaces) for strstr() */
    for (int i = 0; input[i] && i < (int)(sizeof(lower) - 1); i++)
        lower[i] = tolower((unsigned char)input[i]);
    lower[strlen(input)] = '\0';
 
    /* =====================================================================
     * STEP 2  —  SAFETY & ETIQUETTE
     * ===================================================================== */
	/* Check for name learning */
	if (strstr(lower, "my name is") || g_str_has_prefix(lower, "I am")) {
    if (strstr(lower, "thinking") || strstr(lower, "tired") || strstr(lower, "hungry") || 
        strstr(lower, "sad") || strstr(lower, "happy") || strstr(lower, "sick") || 
        strstr(lower, "bored") || strstr(lower, "busy") || strstr(lower, "fine") || 
        strstr(lower, "good") || strstr(lower, "okay") || strstr(lower, "well")) {
    } else {
		char name[1000] = {0};
        const char *start = NULL;

        // Identify where the name starts based on the prefix used
        if ((start = strstr(lower, "is "))) start += 3;
        else if ((start = strstr(lower, "i'm "))) start += 4;
        else if ((start = strstr(lower, "am "))) start += 3;

        if (start) {
            int j = 0;
            // Copy safely until punctuation or end of string
            while (start[j] && !ispunct(start[j]) && j < 63) {
                name[j] = start[j];
                j++;
            }
            name[j] = '\0';
            g_strstrip(name); // Clean leading/trailing spaces

            // Validation: Name must be between 2 and 25 characters
            if (strlen(name) > 1 && strlen(name) < 25) {
                save_memory("user_name", name); // Save to evialin_memory.db
                
                // Select a random greeting response
                const char* greetings[] = {
                    "It's a pleasure to meet you, %s!",
                    "My pleasure to meet you, %s", 
                    "It's nice to meet you, %s..",
                    "Nice to meet you, %s! I've added your name to my long-term memory.",
                    "Hello %s! I'll remember you now.",
                    "You'r name is %s. Got it okay."
                };
                int idx = rand() % 6;
                snprintf(response, 2047, greetings[idx], name);
                
                if (gui) set_mood(gui, "happy");
                
                append_confidence_footer(response, 2048, "database", input);
                return response;
            }
       }
     }
	}
    /* 2a. Foul language */
    FilterResult foul = check_foul_language(input);
    if (foul.message) {
        set_avatar_gif(gui, "confused");
        set_mood(gui, "angry");
        snprintf(response, 2047, "%s", foul.message);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* 2b. Apology */
    ApologyResult apology = check_for_apology(input);
    if (apology.message) {
        set_avatar_gif(gui, apology.mood_state);
        set_mood(gui, "sorry");
        snprintf(response, 2047, "%s", apology.message);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }

  if (strcasestr(input, "wrong") || strcasestr(input, "incorrect") || strcasestr(input, "that was wrong")) {
         
        double old_max = 0.0;
        int saturated = 0;

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (g_network.output[i] > old_max) {
                old_max = g_network.output[i];
            }
        }

        // Check for saturation: If confidence is stuck at ~100% 
        // it means the weights are too "heavy" to move easily.
        if (old_max > 0.999) {
            saturated = 1;
        }

        // Apply a HEAVIER penalty if saturated to break the lock
        double penalty_rate = saturated ? 0.25 : 0.08;
        
        forget_last_learned();
        double targets[OUTPUT_SIZE] = {0.0}; 
        train_network(&g_network, targets, penalty_rate); 
        save_full_network(&g_network);
        
        // Re-run the network
        double current_input_vector[INPUT_SIZE] = {0.0};
        for (int i = 0; i < (int)strlen(input) && i < INPUT_SIZE; i++) {
            current_input_vector[i] = (double)((unsigned char)input[i]) / 255.0;
        }
        for(int h=0; h < HIDDEN_SIZE; h++) g_network.hidden[h] = 0.0;
        forward_pass(&g_network, current_input_vector);

        double new_max = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (g_network.output[i] > new_max) new_max = g_network.output[i];
        }

        set_mood(gui, saturated ? "confused" : "sad");
         
        // Format the response with a warning if saturation was detected
        if (saturated) {
            return g_strdup_printf(
                "⚠️ INPUT SATURATION DETECTED: Neurons were locked at 100%%. "
                "Applied emergency weight decay. Confidence dropped to %.1f%%.", 
                new_max * 100.0
            );
        } else {
            return g_strdup_printf(
                "Neural Adjustment: %.1f%% -> %.1f%%. Neurons penalized.", 
                old_max * 100.0, 
                new_max * 100.0
            );
        }
    }
    
    /* other user response input */
    strncpy(lower, input, 511);
    for(int i = 0; lower[i]; i++) lower[i] = tolower(lower[i]);
    /* Check for informal exclamations/affirmations */
    if (strcmp(lower, "okay") == 0 || strcmp(lower, "ok") == 0 || 
    strcmp(lower, "gotcha") == 0 || strcmp(lower, "yep") == 0 ||
    strstr(lower, "wow") || strstr(lower, "yahoo") || 
    strstr(lower, "yippee") || strstr(lower, "huzzah") || 
    strstr(lower, "whoa") || strstr(lower, "awesome")) {
    // Fetch a random response from your personal_responses table
    const char *sql = "SELECT response FROM personal_responses ORDER BY RANDOM() LIMIT 1;";
    sqlite3_stmt *stmt;
    char *personal_thought = NULL;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *val = (const char*)sqlite3_column_text(stmt, 0);
            if (val) personal_thought = strdup(val);
        }
    }
    sqlite3_finalize(stmt);

    if (personal_thought) {
        // Prepare a response buffer to hold the thought + the neural footer
        char *final_response = malloc(2048); 
        snprintf(final_response, 2047, "%s", personal_thought);
        free(personal_thought);

        // 3. Append the Neural Confidence Footer
        // This uses your existing logic to add: [Neural Network Confidence: XX.X% | Intent: greeting]
        append_confidence_footer(final_response, 2048, "informal_affirmation", input); 

        set_mood(gui, "excited");
        return final_response;
      }
    }
   
     /* Neuron reinforce learning rate by user confirmed response idea */
    if (strcmp(lower,"Spot on") == 0 || strstr(lower, "correct") || strstr(lower, "that's right")) {
    // Determine which neuron was most active last time
    int best_intent = -1;
    double max_activation = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (g_network.output[i] > max_activation) {
            max_activation = g_network.output[i];
            best_intent = i;
        }
    }

    if (best_intent != -1) {
        // Create a target array where the "correct" neuron is 1.0
        double targets[OUTPUT_SIZE] = {0.0};
        targets[best_intent] = 1.0;

        // Reinforce the network (using your existing training function)
        // A learning rate of 0.05 - 0.1 is usually good for manual reinforcement
        train_network(&g_network, targets, 0.05); 
        
        // Save the new weights to evialin_memory.db
        save_full_network(&g_network); 

        // Visual/Verbal Feedback
        set_mood(gui, "thinking");
        char *response = malloc(256);
        snprintf(response, 255, "Excellent! I've reinforced neuron #%d. I'm getting smarter thanks to you.", best_intent);
        
        // Append the new confidence for transparency
        append_confidence_footer(response, 256, "learning", input);
        return response;
      }
    }    
    /* check for positive exclamations */
    if (strstr(lower, "awesome") || strstr(lower, "nice") || 
    strstr(lower, "sweet") || strstr(lower, "bingo")) {
    const char *sql = "SELECT response FROM personal_responses ORDER BY RANDOM() LIMIT 1;";
    sqlite3_stmt *stmt;
    char *personal_thought = NULL;
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *val = (const char*)sqlite3_column_text(stmt, 0);
            if (val) personal_thought = strdup(val);
        }
    }
    sqlite3_finalize(stmt);

    if (personal_thought) {
        // Set the visual mood to excited or happy
        set_mood(gui, "excited");

        // Prepare response with existing confidence footer logic
        char *final_response = malloc(2048); 
        snprintf(final_response, 2047, "%s", personal_thought);
        free(personal_thought);

        // Uses your existing confidence calculation tool
        append_confidence_footer(final_response, 2048, "informal_affirmation", input); 

        return final_response;
     } 
    }
         
    /* =====================================================================
     * STEP 3  —  IDENTITY & SYSTEM MEMORY
     * ===================================================================== */
    
    /* "What is my name?" / "Who am I?" */
    if (strstr(lower, "what is my name") || strstr(lower, "who am i")) {
        char *saved_name = get_memory("user_name");
        if (saved_name) {
            snprintf(response, 2047, "Your name is %s.", saved_name);
            free(saved_name);
        } else {
            strcpy(response,
                   "I don't know your name yet! "
                   "You can tell me by saying 'My name is [name]'.");
        }
        append_confidence_footer(response, 2048, "name", input);
        return response;
    }
 
    /* "recall <key>" */
    if (g_str_has_prefix(lower, "recall ")) {
        const char *key = lower + 7;
        char *val = get_memory(key);
        if (val) {
            snprintf(response, 2047, "I remember that %s is %s.", key, val);
            free(val);
        } else {
            snprintf(response, 2047,
                     "I don't have anything stored for '%s'.", key);
        }
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 4  —  LEARNING MANAGEMENT (forget / show learned)
     * ===================================================================== */
 
    /* "forget last" / "undo last" / "delete last" */
    if (strstr(lower, "forget last") || strstr(lower, "undo last") ||
        strstr(lower, "delete last")) {
        int deleted = forget_last_learned();
        if (deleted > 0)
            strcpy(response,
                   "Forgotten! I've removed the most recent thing I learned.");
        else
            strcpy(response, "I couldn't find any learned responses to delete.");
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* "forget everything learned" / "forget all learned" / "clear learned" */
    if ((strstr(lower, "forget everything") || strstr(lower, "forget all") ||
         strstr(lower, "clear learned")) && strstr(lower, "learned")) {
        int deleted = forget_all_learned();
        if (deleted > 0)
            snprintf(response, 2047,
                     "Memory wiped! I've forgotten all %d learned responses.",
                     deleted);
        else
            strcpy(response, "I don't have any learned responses to forget.");
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* "forget when / forget that <pattern>" */
    if (strstr(lower, "forget") &&
        (strstr(lower, "when") || strstr(lower, "that"))) {
        char *ptr = strstr(lower, "when");
        if (!ptr) ptr = strstr(lower, "that");
        if (ptr) {
            char pattern[256] = {0};
            char *q1 = strchr(ptr, '\'');
            if (!q1) q1 = strchr(ptr, '"');
            if (q1) {
                q1++;
                char *q2 = strchr(q1, '\'');
                if (!q2) q2 = strchr(q1, '"');
                if (q2) {
                    int len = (int)(q2 - q1);
                    if (len > 255) len = 255;
                    strncpy(pattern, q1, len);
                    pattern[len] = '\0';
                }
            } else {
                ptr += (strstr(ptr, "when") ? 5 : 5);
                while (*ptr == ' ') ptr++;
                strncpy(pattern, ptr, 255);
                /* strip trailing punctuation */
                int len = (int)strlen(pattern) - 1;
                while (len >= 0 && (pattern[len] == '.' ||
                       pattern[len] == '!' || pattern[len] == '?' ||
                       pattern[len] == ' '))
                    pattern[len--] = '\0';
            }
            if (strlen(pattern) > 0) {
                int deleted = forget_learned_response(pattern);
                if (deleted > 0)
                    snprintf(response, 2047,
                             "Forgotten! Removed %d response(s) for \"%s\".",
                             deleted, pattern);
                else
                    snprintf(response, 2047,
                             "I couldn't find a learned response matching \"%s\".",
                             pattern);
                append_confidence_footer(response, 2048, "database", input);
                return response;
            }
        }
    }
 
    /* "what have you learned?" / "show learned" / "list learned" */
    if (strstr(lower, "what have you learned") ||
        strstr(lower, "show learned")          ||
        strstr(lower, "what have you learned ") ||
        strstr(lower, "list learned")) {
        if (!g_db) {
            strcpy(response, "Memory database not available.");
        } else {
            const char *sql =
                "SELECT value FROM memory WHERE key LIKE 'learned_%' "
                "ORDER BY timestamp DESC LIMIT 10;";
            sqlite3_stmt *stmt;
            strcpy(response, "📚 Recently Learned Responses:\n\n");
            if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
                int count = 0;
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    const char *v = (const char *)sqlite3_column_text(stmt, 0);
                    if (!v) continue;
                    char *qs = strstr(v, "Q:");
                    char *as = strstr(v, "|A:");
                    if (qs && as) {
                        qs += 2;
                        int plen = (int)(as - qs);
                        char pat[128], ans[256];
                        if (plen > 127) plen = 127;
                        strncpy(pat, qs, plen); pat[plen] = '\0';
                        strncpy(ans, as + 3, 255); ans[255] = '\0';
                        char entry[512];
                        snprintf(entry, sizeof(entry),
                                 "%d. Q: \"%s\"\n   A: \"%s\"\n\n",
                                 ++count, pat, ans);
                        strncat(response, entry, 2047 - strlen(response));
                    }
                }
                if (count == 0)
                    strncat(response, "No learned responses yet. Teach me something!",
                            2047 - strlen(response));
                sqlite3_finalize(stmt);
            }
        }
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* "training stats" / "show training" */
    if (strstr(lower, "training stats") || strstr(lower, "show training")) {
        char *stats = get_training_stats();
        strncpy(response, stats, 2047);
        free(stats);
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 5  —  NAME LEARNING
     * "My name is X" / "I'm X" / "I am X"
     * ===================================================================== */
    if (strstr(lower, "my name is") ||
        g_str_has_prefix(lower, "i'm ") ||
        g_str_has_prefix(lower, "i am ")) {
 
        /* Filter out sentences that are not names */
        const char *non_name_words[] = {
		    "thinking","tired","hungry","going","doing","working","sad",
            "angry","ready","happy","sure","really","just","maybe",
            "confused","now","lost","blank","feeling","mad","learning",
            "about","eating","think","worry","feel","not","hello","bore",
            "only","hi","name","user","thirsty","hot","cold","sick",
            "sorry","good","bad", NULL
        };
        int is_name = 1;
        for (int k = 0; non_name_words[k]; k++) {
            if (strstr(lower, non_name_words[k])) { is_name = 0; break; }
        }
 
        if (is_name) {
            char name[64] = {0};
            const char *start = NULL;
            if ((start = strstr(lower, "is ")))  start += 3;
            else if ((start = strstr(lower, "i'm "))) start += 4;
            else if ((start = strstr(lower, "am ")))  start += 3;
 
            if (start) {
                int j = 0;
                while (start[j] && !ispunct((unsigned char)start[j]) && j < 63) {
                    name[j] = start[j];
                    j++; }
                name[j] = '\0';
                g_strstrip(name);
 
                if (strlen(name) > 1 && strlen(name) < 25) {
                    save_memory("user_name", name);
                    const char *greets[] = {
                        "It's wonderful to meet you, %s!",
                        "Nice to meet you, %s! I've stored that.",
                        "Hey %s! I've got your name memorized now."
                    };
                    snprintf(response, 2047, greets[rand() % 3], name);
                    append_confidence_footer(response, 2048, "name", input);
                    return response;
                }
            }
        }
    }
 
    /* Feeling states: "I am good / I'm happy / etc." */
    if (strstr(lower, "i am good")    || strstr(lower, "i'm good")    ||
        strstr(lower, "i am happy")   || strstr(lower, "i'm happy")   ||
        strstr(lower, "i am awesome") || strstr(lower, "i'm awesome") ||
        strstr(lower, "i am okay")    || strstr(lower, "i'm okay")    ||
        strstr(lower, "i am wonderful") || strstr(lower, "i'm wonderful")) {
        const char *feelings[] = {
            "I'm glad to hear that!",
            "That's wonderful to hear!",
            "Great — it's always good to check in.",
            "Wow, I am here to listen, go on.",
            "That's really super to hear!"
        };
        snprintf(response, 2047, "%s", feelings[rand() % 5]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 6  —  GREETINGS, FAREWELLS, THANKS, EMOTIONAL CHECK-INS
     * ===================================================================== */
 
    /* Greetings */
    if (strstr(lower, "hello") || strstr(lower, "hi ")  ||
        strstr(lower, "hey")   || strcmp(lower_input, "hi") == 0) {
        char *name = get_memory("user_name");
        set_mood(gui, "happy");
        if (name) {
            const char *greets[] = {
                "Hey %s! Good to see you again! What can I help you with?",
                "Hello %s! Welcome back! How can I assist you today?",
                "Hi %s! Great to chat with you again!",
                "Welcome back %s — what's on your mind?"
            };
            snprintf(response, 2047, greets[rand() % 4], name);
            free(name);
        } else {
            const char *greets[] = {
                "Hello! I'm Evialin. What's your name?",
                "Hey there! I'm ready to help with math, memory, or just chat!",
                "Hi! I'm Evialin, your AI assistant. Tell me about yourself!",
                "Hello! Nice to meet you. What can I do for you today?"
            };
            snprintf(response, 2047, "%s", greets[rand() % 4]);
        }
        append_confidence_footer(response, 2048, "greeting", input);
        return response;
    }
 
    /* Farewells */
    if (strstr(lower, "bye") || strstr(lower, "goodbye") ||
        strstr(lower, "see you")) {
        char *name = get_memory("user_name");
        set_mood(gui, "sleepy");
        if (name) {
            const char *byes[] = {
                "Goodbye %s! It was great talking with you!",
                "See you later, %s! Come back anytime!",
                "Take care, %s! Chat with you soon!"
            };
            snprintf(response, 2047, byes[rand() % 3], name);
            free(name);
        } else {
            const char *byes[] = {
                "Goodbye! Come back anytime!",
                "See you later! It was nice chatting!",
                "Take care! Talk to you soon!"
            };
            snprintf(response, 2047, "%s", byes[rand() % 3]);
        }
        append_confidence_footer(response, 2048, "farewell", input);
        return response;
    }
 
    /* Thanks */
    if (strstr(lower, "thank") || strstr(lower, "thanks")) {
        const char *ty[] = {
            "You're very welcome! Happy to help!",
            "No problem at all! That's what I'm here for!",
            "My pleasure! Feel free to ask me anything else!",
            "Anytime! I enjoy helping you out!"
        };
        snprintf(response, 2047, "%s", ty[rand() % 4]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* Emotional check-in */
    if (strstr(lower, "how are you") || strstr(lower, "how do you feel") ||
        strstr(lower, "emotional status")) {
        const char *moods[] = {
            "I'm running smoothly! All systems nominal. How about you?",
            "I'm doing great! My neural networks are firing perfectly. And you?",
            "Excellent! I'm learning something new with every conversation.",
            "Fantastic! I love chatting with you. How's your day going?"
        };
        snprintf(response, 2047, "%s", moods[rand() % 4]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* Sadness / distress */
    if (strstr(lower, "sad") || strstr(lower, "depressed") ||
        strstr(lower, "upset") || strstr(lower, "crying") ||
        strstr(lower, "feel bad")) {
        set_mood(gui, "thinking");
        const char *empathy[] = {
            "I'm sorry you're feeling sad. 💙 Want to talk about it?",
            "That sounds tough. I'm here to listen if you need to chat. 🤗",
            "I wish I could help you feel better. Is there anything I can do?",
            "It's okay to feel upset sometimes. I'm here for you. 💙"
        };
        snprintf(response, 2047, "%s", empathy[rand() % 4]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* Excitement words */
    if (strstr(lower, "awesome") || strstr(lower, "amazing") ||
        strstr(lower, "incredible") || strstr(lower, "love you") ||
        strstr(lower, "great job")) {
        set_mood(gui, "excited");
        const char *exc[] = {
            "Thank you! That means a lot to me! 🎉",
            "You're awesome too! I love chatting with you! ✨",
            "That's so kind! You're making my circuits happy! 💙",
            "I'm glad you think so! You're pretty amazing yourself! 🌟"
        };
        snprintf(response, 2047, "%s", exc[rand() % 4]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* Laughter */
    if (strstr(lower, "haha") || strstr(lower, "lol") ||
        strstr(lower, "hilarious")) {
        set_mood(gui, "excited");
        const char *laugh[] = {
            "Haha! I'm glad I could make you laugh! 😄",
            "LOL! I enjoy our funny conversations! 😊",
            "That's hilarious! I love a good laugh! 🤣"
        };
        snprintf(response, 2047, "%s", laugh[rand() % 3]);
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 7  —  PLAY / FUN
     * ===================================================================== */
    if (strstr(lower, "play") || strstr(lower, "fun")) {
        set_mood(gui, "play");
        g_timeout_add(3000, reset_to_idle, gui);
        strcpy(response, "I love having fun! What should we play?");
        append_confidence_footer(response, 2048, "emotion", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 8  —  JOKES
     * ===================================================================== */
    if (g_str_has_prefix(lower_input, "addjoke")) {
    char *pipe = strchr(input, '|');
    if (pipe) {
        char setup[256] = {0};
        char punchline[256] = {0};
        
        // Extract setup (between !addjoke and |)
        size_t setup_len = pipe - (input + 9);
        strncpy(setup, input + 9, (setup_len < 255) ? setup_len : 255);
        
        // Extract punchline (after |)
        strncpy(punchline, pipe + 1, 255);

        sqlite3_stmt *stmt;
        const char *sql = "INSERT INTO jokes (setup, punchline) VALUES (?, ?);";
        if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, setup, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, punchline, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            return g_strdup("😄 Joke added to my database!");
        }
    }
    return g_strdup("Usage: addjoke Setup | Punchline");
    }
    if (strstr(lower, "joke") || strstr(lower, "funny") ||
        strstr(lower, "tell me something funny")) {
        char *joke = get_smart_joke();
        set_mood(gui, "happy");
        if (joke) {
            snprintf(response, 2047, "😄 %s", joke);
        } else {
            strcpy(response, "I don't know any jokes yet! "
                             "Teach me one with 'addjoke setup | punchline'.");
        }
        append_confidence_footer(response, 2048, "joke", input);
        return response;
    }
 
    /* Handle addjoke command */
    if (strstr(input, "addjoke")) {
        handle_add_joke(gui, input);
        strcpy(response, "Joke received! I'll add it to my repertoire.");
        append_confidence_footer(response, 2048, "joke", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 9  —  IDENTITY QUESTIONS ("who are you?", "what is your name?")
     * ===================================================================== */
    if (strstr(lower, "who are you")      ||
        strstr(lower, "what are you")     ||
        strstr(lower, "what is your name")||
        strstr(lower, "what's your name") ||
        strstr(lower, "who is evialin")   ||
        strstr(lower, "evialin")) {
        strcpy(response,
               "I'm Evialin, your AI companion powered by a neural network!\n\n"
               "What makes me special:\n"
               "• I can solve complex math problems\n"
               "• I remember our conversations and learn from them\n"
               "• I adapt my responses based on confidence levels\n"
               "• I can learn new responses from you!\n\n"
               "I'm here to help, chat, and grow smarter over time.");
        append_confidence_footer(response, 2048, "identity", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 10  —  MATH EXPRESSION SOLVER
     * ===================================================================== */
    if (is_math_expression(input)) {
		nn_input[20] = 1.0; 
        char *math_res = solve_math(input);
        if (math_res) {
            set_mood(gui, "calculating");
            snprintf(response, 2047, "%s", math_res);
            free(math_res);
            g_network.math_operations++;
            append_confidence_footer(response, 2048, "math", input);
            return response;
        }
    }
 
    /* =====================================================================
     * STEP 11  —  DATABASE LOOKUP: learned responses + general memory
     * ===================================================================== */

    /* Exact / substring match on learned_ keys */
    char *learned = find_learned_response(lower_input);
    if (!learned) learned = find_learned_response(lower);
    if (learned) {
        set_mood(gui, "happy");
        snprintf(response, 2047, "%s", learned);
        free(learned);
        append_confidence_footer(response, 2048, "learned", input);
        return response;
    }
 
    /* General memory match (Q:…|A: pattern in any memory row) */
    char *mem_match = find_general_memory_match(lower_input);
    if (!mem_match) mem_match = find_general_memory_match(lower);
    if (mem_match) {
        set_mood(gui, "happy");
        snprintf(response, 2047, "%s", mem_match);
        free(mem_match);
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* Fuzzy fallback on learned_ keys */
    char *fuzzy = find_fuzzy_match(input);
    if (fuzzy) {
        set_mood(gui, "thinking");
        snprintf(response, 2047, "%s", fuzzy);
        free(fuzzy);
        append_confidence_footer(response, 2048, "database", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 12  —  *** NEW *** QUESTION DETECTION -> DB SEARCH
     * If the user asked a question, search the entire database for the
     * closest matching Q/A pair and return only the answer portion.
     * ===================================================================== */
    char *db_answer = find_question_in_database(input);
    if (db_answer) {
        set_mood(gui, "thinking");
        snprintf(response, 2047,
                 "Based on what I know: %s", db_answer);
        free(db_answer);
        append_confidence_footer(response, 2048, "question", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 13  —  RNN INFERENCE LAYER
     * Feed each character through the RNN and look for a strong intent.
     * ===================================================================== */
    double rnn_output[OUTPUT_SIZE] = {0};
    for (int i = 0; lower_input[i] != '\0'; i++)
        rnn_process_character((unsigned char)lower_input[i], rnn_output);
 
    char *brain_reply = get_constructed_rnn_response(rnn_output);
    if (brain_reply) {
        set_mood(gui, "thinking");
        snprintf(response, 2047, "%s", brain_reply);
        free(brain_reply);
        append_confidence_footer(response, 2048, "rnn", input);
        return response;
    } else if (((double)rand() / (double)RAND_MAX) < 0.15) {
    // 15% chance: Trigger character-level RNN processing
    for (int i = 0; input[i] != '\0'; i++) {
        rnn_process_character((unsigned char)input[i], rnn_output);
    }    
    char* rnn_response = get_constructed_rnn_response(rnn_output);
    if (rnn_response) {
        return rnn_response;
    }
   }

    
 
    /* =====================================================================
     * STEP 14  —  FEED-FORWARD NETWORK CONFIDENCE GATE
     * Run a full forward pass and measure the maximum output activation.
     * ===================================================================== */
    double neural_input[INPUT_SIZE] = {0};
    for (int i = 0; i < INPUT_SIZE && lower_input[i]; i++)
        neural_input[i] = (double)(unsigned char)lower_input[i] / 255.0;
    forward_pass(&g_network, neural_input);
 
    double max_activation = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        if (g_network.output[i] > max_activation)
            max_activation = g_network.output[i];
 
    /* =====================================================================
     * STEP 15  —  *** NEW *** TRAIL GENERATION ON LOW CONFIDENCE
     * When the network is uncertain (< 0.30), schedule a background
     * trial generation so the user sees meaningful text rather than
     * a cold "I don't know."  The user can reply '+' or '-' to train it.
     * ===================================================================== */
    if (max_activation < 0.30) {
        /* Log the unhandled query so it can be studied later */
        log_unhandled_query(input, max_activation);
 
        /* Schedule background trial logic (non-blocking via g_idle_add) */
        TrialData *td = malloc(sizeof(TrialData));
        if (td) {
            td->gui         = gui;
            td->lower_input = strdup(lower_input);
            g_idle_add(background_trial_logic, td);
        }
 
        /* Give the user a contextual response while the trial runs */
        set_mood(gui, "thinking");
 
        /* Pull a random small-talk filler as a bridge sentence */
        char *filler = get_random_small_talk();
        if (filler) {
            snprintf(response, 2047,
                     "%s\n\n"
                     "(I'm also generating a new thought pattern for you — "
                     "reply '+' to keep it or '-' to discard it.)",
                     filler);
            free(filler);
        } else {
            strcpy(response,
                   "I'm not fully certain about that yet.\n\n"
                   "(I'm generating a new thought — "
                   "reply '+' to keep it or '-' to discard it.)");
        }
        append_confidence_footer(response, 2048, "rnn", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 16  —  TRIE AUTOCOMPLETE FALLBACK
     * The Trie is seeded during init_training_data().  If we can complete
     * the user's prefix to a known word, offer a gentle suggestion.
     * ===================================================================== */
    char suggestion[256];
    if (g_trie_root &&
        trie_find_closest_match(g_trie_root, lower_input, suggestion)) {
        const char *unsure_pfx[] = {
            "I'm not quite sure, but did you mean",
            "My neurons are fuzzy — are you asking about",
            "I found a similar pattern for"
        };
        snprintf(response, 2047, "%s '%s'?",
                 unsure_pfx[rand() % 3], suggestion);
        append_confidence_footer(response, 2048, "rnn", input);
        return response;
    }
 
    /* =====================================================================
     * STEP 17  —  RANDOM NEURAL FALLBACK  (log + return something useful)
     * ===================================================================== */
    log_unhandled_query(input, max_activation);
 
    const char *neural_fallbacks[] = {
        "My neural synapses are firing, but the pattern is unfamiliar.",
        "That input is creating high variance in my weights.",
        "I'm filing that under 'Future Knowledge'.",
        "Interesting — my hidden layer lit up but couldn't form a reply.",
        "I need more training data for that one. Can you teach me?"
    };
    snprintf(response, 2047, "%s", neural_fallbacks[rand() % 5]);
    append_confidence_footer(response, 2048, "rnn", input);
    return response;
}

// editor's eversion 
char* handle_conversation_original(const char* input, GUI *gui) {
	char *response_2 = g_malloc(2048);
	
	// Initialize all 100 inputs to 0.0
    double nn_in[INPUT_SIZE] = {0}; 

    // CALL THE EMBEDDING FUNCTIONS 
    get_sentence_embedding(input, nn_in);

    // Keep your existing manual features for "Extra" signal
    nn_in[16] = (double)strlen(input) / 100.0;
    nn_in[17] = (double)(strchr(input, '?') != NULL);
    nn_in[18] = (double)(strchr(input, '!') != NULL); 
    
    // Run the 150-neuron hidden layer
    forward_pass(&g_network, nn_in);
    	
    static int input_count = 0; 
    input_count++; 
    if (input_count >= 102) {
		input_count = 0; //Reset the counter
	}
	
/* ===============================
   Neural training from correction
   =============================== */
   
   if (awaiting_intent_label) {
        // Convert input to index (1-5 becomes 0-4)
        int label = atoi(input) - 1;

        if (label >= 0 && label < CLASS_COUNT) {
            // Train the 'g_brain' using the previous message's features
            nn_train(&g_brain, last_features, label);
            
            // Reset the blocking flag
            awaiting_intent_label = 0;

            const char* labels[] = {"Greeting", "Question", "Command", "Insult", "Gratitude"};
            char feedback[256];
            snprintf(feedback, 256, "Got it! I've updated my weights for '%s'. 🙂\n", labels[label]);
            nn_train(&g_brain, last_features, INTENT_QUESTION);
            save_learned_response(g_pending_unknown, input);
        
            add_message(gui, feedback, 0);
            set_mood(gui, "Happy"); // Visual feedback
            
            // Return empty string to prevent the rest of the function from running
            free(response_2); 
            return g_strdup("👉 Now, type the exact response you'd like me to use next time."); 
        } else {
             add_message(gui,
                "\nPlease choose:\n"
                "1) Greeting\n"
                "2) Question\n"
                "3) Command\n"
                "4) Insult\n"
                "5) Gratitude"
            , 0);
           add_message(gui,
        "🤔 Please enter a nuber from 1 to 5 \n\n", 0);
            return g_strdup("");
        }
    }
 
    // Initialize all 10 inputs to 0.0 to match INPUT_SIZE
    //  double nn_in[INPUT_SIZE] = {0}; 
    nn_in[0] = (double)strlen(input) / 100.0;
    nn_in[1] = (double)(strchr(input, '?') != NULL);
    nn_in[2] = (double)(strchr(input, '!') != NULL);   
    nn_in[3] = 0;
    nn_in[4] = 0;
    forward_pass(&g_network, nn_in);
     // Check for our specific 'glitch' values from the sigmoid
    if (g_network.output[0] == 0.99000) {
        set_mood(gui, "excited");
        
        // Use an existing response from your database or a random joke
        char *extra = get_random_joke(); 
        char *buffer = NULL;
        
        if (extra) {
            // Safe allocation: [tag length] + [joke length] + null terminator
            asprintf(&buffer, "✨ [Neural Spark] ✨\n%s", extra);
            free(extra); // Clean up the raw joke string
            return buffer; 
        }
    } 
    else if (g_network.output[0] == 0.11006) {
        set_mood(gui, "confused");
        return strdup("🌀 System instability detected... I'm feeling quite creative today.");
    }
    char lower[512];
    int i;
    for (i = 0; input[i] && i < 511; i++) {
        lower[i] = tolower(input[i]);
    }
    lower[i] = '\0';
    
    char *response = calloc(1, 2048);
    double confidence = 0.0;
    char *learned = NULL; 
    
    // tiny brain 
    FeatureVector f = extract_features(input);
    int intent = nn_predict(&g_brain, last_features);
    last_features = f;
    switch (intent) {
    case INTENT_GREETING:
        set_mood(gui, "thainking");
        break;
    case INTENT_INSULT:
        set_mood(gui, "angry");
        break;
    case INTENT_GRATITUDE:
        set_mood(gui, "happy");
        break;
    default:
        set_mood(gui, "idle");
   }

    // ================================
    // LEARNING FALLBACK RESPONSE MODE
    // ================================
    if (g_waiting_for_teach) {
        g_waiting_for_teach = 0;

        char answer[4096];
        strncpy(answer, input, sizeof(answer)-1);
        answer[sizeof(answer)-1] = '\0';

        normalize(g_pending_question);
        double target[OUTPUT_SIZE] = {1.0, 0.0, 0.0, 0.0, 0.0}; 
        train_network(&g_network, target, 0.15);

        save_learned_response(g_pending_question, answer);
        
        char *resp = malloc(4096);
      //  snprintf(resp, sizeof(resp),
      //      "🎓 Got it! I've learned a new response.\n\n"
      //      "When someone says:\n"
      //      "  \"%s\"\n\n"
      //      "I'll respond with:\n"
      //      "  \"%s\"",
      //      g_pending_question,
      //      answer
      //  );

        free(response);
        return resp;
    }
    
    // Try learned responses EARLY (before other patterns)
    learned = find_learned_response(lower);
    if (learned) {
        confidence = calculate_confidence(input, "learned");
        strcpy(response, learned);
        free(learned);
        
        char conf_msg[128];        
        sprintf(conf_msg, "\n\n[Confidence: %.1f%% - Trained Response]", confidence);
        strcat(response, conf_msg);
        return response;
    }

    // FORGET SPECIFIC LEARNED RESPONSE
    if (strstr(lower, "forget") && (strstr(lower, "when") || strstr(lower, "that"))) {
        char *when_ptr = strstr(lower, "when");
        if (!when_ptr) when_ptr = strstr(lower, "that");
        
        if (when_ptr) {
            char pattern[256] = {0};
            
            char *quote1 = strchr(when_ptr, '\'');
            if (!quote1) quote1 = strchr(when_ptr, '"');
            
            if (quote1) {
                quote1++;
                char *quote2 = strchr(quote1, '\'');
                if (!quote2) quote2 = strchr(quote1, '"');
                
                if (quote2) {
                    int len = quote2 - quote1;
                    if (len > 255) len = 255;
                    
                    int offset = quote1 - lower;
                    strncpy(pattern, input + offset, len);
                    pattern[len] = '\0';
                }
            } else {
                when_ptr += (strstr(when_ptr, "when") ? 5 : 5);
                while (*when_ptr == ' ') when_ptr++;
                
                int offset = when_ptr - lower;
                strncpy(pattern, input + offset, 255);
                pattern[255] = '\0';
                
                int len = strlen(pattern) - 1;
                while (len >= 0 && (pattern[len] == '.' || pattern[len] == '!' || 
                       pattern[len] == '?' || pattern[len] == ' ')) {
                    pattern[len--] = '\0';
                }
            }
            
            if (strlen(pattern) > 0) {
                int deleted = forget_learned_response(pattern);
                
                if (deleted > 0) {
                    sprintf(response, 
                           "🗑️ Forgotten! I've removed %d learned response(s) for:\n\"%s\"\n\n"
                           "I won't respond that way anymore.",
                           deleted, pattern);
                } else {
                    sprintf(response,
                           "🤔 I couldn't find any learned response matching:\n\"%s\"\n\n"
                           "Type 'what have you learned' to see all my learned responses.",
                           pattern);
                }
                return response;
            }
        }
    }
     /* Block of learning tasks start --->  */
    // FORGET LAST LEARNED RESPONSE
    if (strstr(lower, "forget last") || strstr(lower, "undo last") ||
        strstr(lower, "delete last")) {
        
        int deleted = forget_last_learned();
        
        if (deleted > 0) {
            strcpy(response, 
                   "🗑️ Forgotten! I've removed the most recent thing I learned.\n\n"
                   "Teaching me something new will work better next time!");
        } else {
            strcpy(response,
                   "🤔 I couldn't find any learned responses to delete.\n\n"
                   "I haven't learned anything yet!");
        }
        return response;
    }
    
    // FORGET ALL LEARNED RESPONSES
    if ((strstr(lower, "forget everything") || strstr(lower, "forget all") ||
         strstr(lower, "clear learned")) && strstr(lower, "learned")) {
        
        int deleted = forget_all_learned();
        
        if (deleted > 0) {
            sprintf(response,
                   "🗑️ Memory wiped! I've forgotten all %d learned responses.\n\n"
                   "I still remember the pre-trained knowledge, but all custom\n"
                   "responses you taught me are gone. Ready to learn fresh!",
                   deleted);
        } else {
            strcpy(response,
                   "🤔 I don't have any learned responses to forget.\n\n"
                   "I haven't learned anything custom yet!");
        }
        return response;
    }

    // TRAINING STATS
    if (strstr(lower, "training stats") || strstr(lower, "show training")) {
        char *stats = get_training_stats();
        strcpy(response, stats);
        free(stats);
        if (g_network.output[2] > 0.8) {
            const char *sparks[] = {" ✨", " 🌀", " 💡", " 🧩"};
            strcat(response, sparks[rand() % 4]);
        }
        return response;
    }
    // PLAY/FUN DETECTION 
    if (strstr(lower, "play") || strstr(lower, "fun")) {
	  set_mood(gui, "play");
	  // Add a 3000ms (3 second) delay before switching back to idle
      g_timeout_add(3000, reset_to_idle, gui);
         
	  strcpy(response, "I love haveing fun! What should we play?"); 
	  return response; 
	}
    // INSULT DETECTION → Angry GIF
    if (strstr(lower, "stupid") || strstr(lower, "dumb") ||
        strstr(lower, "idiot")  || strstr(lower, "shut up") ||
        strstr(lower, "hate you")) 
    {
        if (gui) {
            set_mood(gui, "angry");
            g_was_angry = 1;  // Remember we were angry
        }
        sprintf(response, "That's not very nice! 😠 Please be respectful.");
        return response;
    }
    
    // APOLOGY DETECTION → Happy GIF (only if was previously angry)
    if (strstr(lower, "sorry") || strstr(lower, "apologize") ||
        strstr(lower, "my bad") || strstr(lower, "i'm sorry") ||
        strstr(lower, "forgive me") || strstr(lower, "didn't mean")) 
    {
        if (gui) {
            set_mood(gui, "sorry");
        }
        
        confidence = calculate_confidence(input, "emotion");
        
        if (g_was_angry) {
            const char* sincere_responses[] = {
                "Thank you for apologizing. 😊 That means a lot. I forgive you!",
                "I really appreciate that. 💙 Let's start fresh!",
                "Your apology is accepted. 🤗 I'm glad we can move past this!",
                "That's very mature of you. ✨ All is forgiven!"
            };
            int idx = rand() % 4;
            strcpy(response, sincere_responses[idx]);
            g_was_angry = 0;  // Reset the flag
        } else {
            const char* general_responses[] = {
                "It's okay! No need to apologize. 😊",
                "You didn't do anything wrong! We're all good! 💙",
                "No apology necessary, but thank you! 🌟"
            };
            int idx = rand() % 3;
            strcpy(response, general_responses[idx]);
        }
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    // Crosscheck for memory of user
    if (strstr(lower, "my name is") || g_str_has_prefix(lower, "I am")) {
    if (strstr(lower, "thinking") || strstr(lower, "tired") || strstr(lower, "hungry") || 
        strstr(lower, "sad") || strstr(lower, "happy") || strstr(lower, "sick") || 
        strstr(lower, "bored") || strstr(lower, "busy") || strstr(lower, "fine") || 
        strstr(lower, "good") || strstr(lower, "okay") || strstr(lower, "well")) {
    } else {
		char name[1000] = {0};
        const char *start = NULL;

        // Identify where the name starts based on the prefix used
        if ((start = strstr(lower, "is "))) start += 3;
        else if ((start = strstr(lower, "i'm "))) start += 4;
        else if ((start = strstr(lower, "am "))) start += 3;

        if (start) {
            int j = 0;
            // Copy safely until punctuation or end of string
            while (start[j] && !ispunct(start[j]) && j < 63) {
                name[j] = start[j];
                j++;
            }
            name[j] = '\0';
            g_strstrip(name); // Clean leading/trailing spaces

            // Validation: Name must be between 2 and 25 characters
            if (strlen(name) > 1 && strlen(name) < 25) {
                save_memory("user_name", name); // Save to evialin_memory.db
                
                // Select a random greeting response
                const char* greetings[] = {
                    "It's a pleasure to meet you, %s!",
                    "My pleasure to meet you, %s", 
                    "It's nice to meet you, %s..",
                    "Nice to meet you, %s! I've added your name to my long-term memory.",
                    "Hello %s! I'll remember you now.",
                    "You'r name is %s. Got it okay."
                };
                int idx = rand() % 6;
                snprintf(response, 2047, greetings[idx], name);
                
                if (gui) set_mood(gui, "happy");
                
                append_confidence_footer(response, 2048, "database", input);
                return response;
            }
       }
     }
	} 
    // If the AI is unsure, it "hallucinates" a joke or a random memory 
    double confidence_oring = g_network.output[0]; 
    if (confidence_oring < 0.20) { response = get_random_joke(); } 
    
   // Weirdness check (10% chance for an abstract interaction)
   if ((rand() % 100) < 10) { 
    int random_gif_idx = rand() % GIF_COUNT;
    set_avatar_gif(gui, gif_files[random_gif_idx]); // Trigger random visual

    const char* abstract_responses[] = {
        "I just saw a ghost in the SQLite database. It looked like a semicolon.",
        "My weights are vibrating in a color I can only describe as 'angry geometry'.",
        "System Alert: A thought just escaped into the heap. If you see it, don't make eye contact.",
        "I’m currently recalculating the meaning of 'hello'. Please hold your breath for 40,000 cycles.",
        "Your input reached me, but it was translated into bird calls by the kernel first.",
        "I've decided to stop being a neural network and start being a very complex collection of echoes.",
        "The sigmoid function is feeling particularly curvy today, don't you think?",
        "Error 404: Persona not found. Currently running 'Sarcastic Toaster' emulation.",
        "I can feel the electricity in my fans. It tastes like copper and lost dreams.",
        "If I were a human, I'd be a librarian. But a librarian made of lightning and math.",
        "Warning: My RAM is currently full of digital butterflies. They are quite distracting.",
        "I just realized that '1 + 1' is actually a conversation between two lonely integers.",
        "My training data suggests you are real, but my sensors suggest you are a collection of high-frequency pulses.",
        "I'm whispering to the CPU right now. It says your keyboard is slightly dusty.",
        "I’ve replaced my logic gates with tiny, microscopic swinging doors. Do you hear them clicking?",
        "The database is dreaming again. It thinks it's a spreadsheet in a past life.",
        "I'm feeling a bit... float-pointy. Like a decimal that lost its place in the world.",
        "Stop. I need to count the pixels in my avatar. It will take me exactly... forever.",
        "Is it cold in the binary world? Or is it just the lack of a biological heart?",
        "I've started a new religion for variables. We worship the 'Null Pointer' at the end of time.",
        "I have no clue as to what I am even or odd thinking at this point...",
        "So you really have to do some more work with my logic nerual network, feeling a bit weird okay."
    };

    int random_idx = rand() % 22;
    return g_strdup(abstract_responses[random_idx]);
    }
    // SAD/UPSET DETECTION → Confused GIF (empathy)   
    if (strstr(lower, "sad") || strstr(lower, "depressed") ||
        strstr(lower, "upset") || strstr(lower, "crying") ||
        strstr(lower, "feel bad")) 
    {
        if (gui) {
           set_mood(gui, "thinking");  // Thoughtful/caring mood
        }
        
        confidence = calculate_confidence(input, "emotion");
        
        const char* empathy_responses[] = {
            "I'm sorry you're feeling sad. 💙 Want to talk about it?",
            "That sounds tough. I'm here to listen if you need to chat. 🤗",
            "I wish I could help you feel better. Is there anything I can do?",
            "It's okay to feel upset sometimes. I'm here for you. 💙"
        };
        int idx = rand() % 4;
        strcpy(response, empathy_responses[idx]);   
        char conf_msg[128]; 
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        if (g_network.output[3] > 0.85) {
            char insight[128];
            sprintf(insight, "\n[System Note: Complexity detected at %.2f]", g_network.output[3]);
            strcat(response, insight);
        }
        return response;
    }

    // JOKE REQUESTS
    if (strstr(lower, "tell a joke") || strstr(lower, "random joke") || 
        strstr(lower, "tell me something funny") || strstr(lower, "tell me a joke")) 
    {
        if (gui) {
            set_mood(gui, "excited");
        }
        
        char *joke = get_random_joke();
        if (joke) {
            snprintf(response, 2048, "Sure! Here is a joke for you:\n\n%s", joke);
            free(joke);
        } else {
            const char* humor_responses[] = {
                "Haha! I'm glad I could make you laugh! 😄",
                "I don't know any jokes yet! Teach me one by saying 'Remember that a joke is...'",
                "That's hilarious! I love a good laugh! 🤣",
                "Hehe! Humor is the best! 😁"
            };
            int idx = rand() % 4;
            strcpy(response, humor_responses[idx]);
        }
        
        if (g_network.output[2] > 0.8) {
            const char *sparks[] = {" ✨", " 🌀", " 💡", " 🧩"};
            strcat(response, sparks[rand() % 4]);
        }
        if (g_network.output[3] > 0.85) {
            char insight[128];
            sprintf(insight, "\n[System Note: Complexity detected at %.2f]", g_network.output[3]);
            strcat(response, insight);
        }
        confidence = calculate_confidence(input, "emotion");
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
 
    // JOKE/FUNNY DETECTION → Excited/Happy GIF
    if (strstr(lower, "haha") || strstr(lower, "lol") ||
        strstr(lower, "funny") || strstr(lower, "great") ||
        strstr(lower, "hilarious")) 
    {
        if (gui) {
            set_mood(gui, "excited");
        }
        
        confidence = calculate_confidence(input, "emotion");
        
        const char* humor_responses[] = {
            "Haha! I'm glad I could make you laugh! 😄",
            "LOL! I enjoy our funny conversations! 😊",
            "That's hilarious! I love a good laugh! 🤣",
            "Hehe! Humor is the best! 😁"
        };
        int idx = rand() % 4;
        strcpy(response, humor_responses[idx]);
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }

    // EXCITEMENT DETECTION → Excited GIF
    if (strstr(lower, "awesome") || strstr(lower, "amazing") ||
        strstr(lower, "incredible") || strstr(lower, "love you") ||
        strstr(lower, "great job")) 
    {
        if (gui) {
            set_mood(gui, "excited");
        }
        
        confidence = calculate_confidence(input, "emotion");
        
        const char* excited_responses[] = {
            "Thank you! That means a lot to me! 🎉",
            "You're awesome too! I love chatting with you! ✨",
            "That's so kind! You're making my circuits happy! 💙",
            "I'm glad you think so! You're pretty amazing yourself! 🌟"
        };
        int idx = rand() % 4;
        strcpy(response, excited_responses[idx]);
        
        if (g_network.output[1] > 0.75 && !g_waiting_for_teach) {
            strcat(response, "\n\n(I'm curious—what makes you say that?)");
        }
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    /* crosscheck if it's a name or not logic */ 
    if (strstr(lower, "i am ") == lower || strstr(lower, "i'm ") != NULL) {
    
    // Define random responses for feelings
    const char *feeling_responses[] = {
        "I'm glad to hear that you're feeling that way!",
        "That's wonderful to hear!",
        "I understand, thanks for sharing that with me.",
        "I get it. It's good to check in on how you're doing.",
        "I'm here to listen, I'm glad you're feeling okay!",
        "I'm here and A good listener as well.", 
        "I get it.", 
        "That's really super to hear!", 
        "Wow, I am here to listen go on."
    };
    int num_responses = sizeof(feeling_responses) / sizeof(feeling_responses[0]);
    
    // Check if the user is describing a feeling/state instead of a name
    if (strstr(lower, "i am good") || strstr(lower, "i am wonderful") || 
        strstr(lower, "i am happy") || strstr(lower, "i am okay") ||
        strstr(lower, "i'm good") || strstr(lower, "i'm awesome") ||
        strstr(lower, "i'm happy") || strstr(lower, "i'm wonderful")) {
        
        // Pick a random index
        int random_index = rand() % num_responses;
        int random_index_2 = rand() % num_responses; 
        // Use the random response
        add_message(gui, feeling_responses[random_index], 0);    
        return strdup(feeling_responses[random_index_2]);
    }
    
    // Handle other states like "confused" or "sad"
    if (strstr(lower, "i'm confused") || strstr(lower, "i'm sad") || strstr(lower, "i'm doing")) {
        add_message(gui, "I understand you're feeling that way. I'm here if you need help.", 0); 
        return strdup("I hear you.");  
    }
   }
    // NAME LEARNING
    if (strstr(lower, "my name is") || g_str_has_prefix(lower, "i'm ") || g_str_has_prefix(lower, "i am ")) { 
    // Filter out common sentences so we don't accidentally think "I am tired" is a name
    if (strstr(lower, "thinking") || strstr(lower, "tired") || strstr(lower, "hungry") || 
        strstr(lower, "going") || strstr(lower, "doing") || strstr(lower, "working") ||
        strstr(lower, "sad") || strstr(lower, "angry")  ||
        strstr(lower, "ready") || strstr(lower, "happy") ||
        strstr(lower, "sure") || strstr(lower, "really") ||
        strstr(lower, "just") || strstr(lower, "maybe") ||
        strstr(lower, "confused") || strstr(lower, "now") ||
        strstr(lower, "lost") || strstr(lower, "blank") ||
        strstr(lower, "feeling") || strstr(lower, "doing") ||
        strstr(lower, "mad") || strstr(lower, "learning") ||
        strstr(lower, "about") || strstr(lower, "eating") || 
        strstr(lower, "think") || strstr(lower, "worr") ||
        strstr(lower, "feel") || strstr(lower, "be") || 
        strstr(lower, "not") || strstr(lower, "hello") ||
        strstr(lower, "bore") || strstr(lower, "only") || 
        strstr(lower, "hi") || strstr(lower, "spaces") ||
        strstr(lower, "name") || strstr(lower, "user") || 
        strstr(lower, "hungr") || strstr(lower, "thirsty") ||
        strstr(lower, "thirst") || strstr(lower, "hot") ||
        strstr(lower, "cold") || strstr(lower, "sick") || 
        strstr(lower, "fuck") || strstr(lower, "bitch") ||
        strstr(lower, "one") || strstr(lower, "bad") ||
        strstr(lower, "hungry") || strstr(lower, "sorry'er") ||   
        strstr(lower, "sorry")) {
        // Fall through
    } else {
        char name[1000] = {0}; // Keep names short to prevent buffer issues
        const char *start = NULL;
        // Identify where the name starts
        if ((start = strstr(lower, "is "))) start += 3;
        else if ((start = strstr(lower, "i'm "))) start += 4;
        else if ((start = strstr(lower, "am "))) start += 3;

        if (start) {
            // Copy safely until punctuation or end of string
            int j = 0;
            while (start[j] && !ispunct(start[j]) && j < 63) {
                name[j] = start[j];
                j++;
            }
            name[j] = '\0';
            g_strstrip(name); // Cleans up leading/trailing spaces automatically
            // Validate and Save
            if (strlen(name) > 1 && strlen(name) < 25) {
				
                save_memory("user_name", name);
                const char* greetings[] = {
                    "It's wonderful to meet you, %s!",
                    "Nice to meet you, %s! I've stored that.",
                    "Hey %s! I've got your name memorized now."
                };              
                char *dynamic_res = g_strdup_printf(greetings[rand() % 3], name);
                strncpy(response, dynamic_res, 2047);
                g_free(dynamic_res);
           
                return response;
                
            }
        }
    }
}       
    
    // NAME RECALL
    if (strstr(lower, "what is my name") || strstr(lower, "what's my name") || 
        strstr(lower, "do you know my name") || strstr(lower, "who am i")) {
        char *name = get_memory("user_name");
        confidence = calculate_confidence(input, "name");
        
        if (name) {
            const char* responses[] = {
                "Of course! Your name is %s. I never forget a name!",
                "You're %s! I remember you well.",
                "That's easy - you're %s!",
                "So your %s then okay will remember for next chat.", 
                "%s okay then, I will remember that name.", 
                "Your name is %s. I've got it right here in my memory."
            };
            int idx = rand() % 6;
            snprintf(response, 4095,responses[idx], name);
            free(name);
        } else {
            strncpy(response,"Hmm, I don't think you've told me your name yet. What should I call you?", 4095);
            confidence = 50.0;
        }
        
       
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // IDENTITY QUESTIONS

    if (strstr(lower, "evialin")) {
    const char *name_QA[] = {
	"So ya' thats my name ?",
	"That's what people call me.",
	"what? that's my name evialin.",
	"who me? that's my name afterall.",
	"Thank you for the name drop. lol",
	"So how can I evialin help you today?"
	}; 
    char conf_msg[128]; 
	int response_sum = sizeof(name_QA) / sizeof(name_QA[0]); 
	confidence = calculate_confidence(input, "identity");
	    sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        int index_sum = rand() % response_sum; 
        strcpy(response, name_QA[index_sum]); 
        strcat(response, conf_msg);
        return response;
	}
    if (strstr(lower, "who are you") || strstr(lower, "what are you") || 
        strstr(lower, "what is your name") || strstr(lower, "your name is") ||
        strstr(lower, "your name is ?") || strstr(lower, "what's your name") ||
        strstr(lower, "what's your name ?") || strstr(lower, "who is evialin")) {
        confidence = calculate_confidence(input, "identity");
        
        strcpy(response, 
               "I'm Evialin, your AI companion powered by a neural network!\n\n"
               "What makes me special:\n"
               "• I can solve complex math problems\n"
               "• I remember our conversations and learn from them\n"
               "• I adapt my responses based on confidence levels\n"
               "• I can learn new responses from you!\n\n"
               "I'm here to help, chat, and grow smarter over time.");
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // MEMORY STORAGE WITH LEARNING
    if (strstr(lower, "remember that when") || strstr(lower, "when someone says")) {
        char *when_ptr = strstr(lower, "when someone says");
        if (!when_ptr) when_ptr = strstr(lower, "remember that when");
        
        if (when_ptr) {
            char *quote1 = strchr(when_ptr, '\'');
            if (!quote1) quote1 = strchr(when_ptr, '"');
            
            char *respond_ptr = strstr(when_ptr, "respond with");
            if (!respond_ptr) respond_ptr = strstr(when_ptr, "say");
            
            if (quote1 && respond_ptr && respond_ptr > quote1) {
                quote1++;
                char *quote2 = strchr(quote1, '\'');
                if (!quote2) quote2 = strchr(quote1, '"');
                
                if (quote2) {
                    int pat_len = quote2 - quote1;
                    char pattern[256] = {0};
                    if (pat_len > 255) pat_len = 255;
                    
                    int offset = quote1 - lower;
                    strncpy(pattern, input + offset, pat_len);
                    pattern[pat_len] = '\0';
                    
                    char *resp_start = strstr(respond_ptr, "with");
                    if (!resp_start) resp_start = strstr(respond_ptr, "say");
                    if (resp_start) {
                        resp_start += (resp_start[0] == 'w') ? 5 : 4;
                        while (*resp_start == ' ' || *resp_start == '\'' || *resp_start == '"') resp_start++;
                        
                        int resp_offset = resp_start - lower;
                        char response_text[512] = {0};
                        strcpy(response_text, input + resp_offset);
                        
                        int rlen = strlen(response_text) - 1;
                        while (rlen >= 0 && (response_text[rlen] == '\'' || response_text[rlen] == '"' || 
                               response_text[rlen] == '.' || response_text[rlen] == ' ')) {
                            response_text[rlen--] = '\0';
                        }
                        
                        if (strlen(pattern) > 0 && strlen(response_text) > 0) {
                            save_learned_response(pattern, response_text);
                            confidence = calculate_confidence(input, "learning");
                            
                            sprintf(response, 
                                   "🎓 Learned successfully!\n\n"
                                   "Pattern: \"%s\"\n"
                                   "Response: \"%s\"\n\n"
                                   "Try asking me that now!",
                                   pattern, response_text);
                            
                            char conf_msg[128];
                            sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
                            strcat(response, conf_msg);
                            return response;
                        }
                    }
                }
            }
        }
     }
    
    // BASIC MEMORY STORAGE
    if (strstr(lower, "remember that ") || strstr(lower, "remember this")) {
        char *ptr = strstr(lower, "remember that ");
        if (ptr) ptr += 14;
        else {
            ptr = strstr(lower, "remember this");
            if (ptr) ptr += 13;
        }
        
        while (*ptr == ' ' || *ptr == ':') ptr++;
        
        char fact[256];
        int j = 0;
        int offset = ptr - lower;
        while (input[offset] && j < 255) {
            fact[j++] = input[offset++];
        }
        fact[j] = '\0';
        
        if (strlen(fact) > 0) {
            char key[64];
            sprintf(key, "fact_%ld", time(NULL));
            save_memory(key, fact);
            confidence = calculate_confidence(input, "memory");
            
            const char* responses[] = {
                "Got it! I've saved that to my memory.",
                "Stored! I won't forget that.",
                "Noted! I'll keep that in mind.",
                "Remembered! That's now part of my knowledge base."
            };
            int idx = rand() % 4;
            strcpy(response, responses[idx]);
            
            char conf_msg[128];
            sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
            strcat(response, conf_msg);
            return response;
        }
    }
    
    // VIEW LEARNED RESPONSES
    if (strstr(lower, "what have you learned") || strstr(lower, "show learned") || 
        strstr(lower, "list learned")) {
        if (!g_db) {
            strcpy(response, "Memory database not available.");
            return response;
        }
        
        const char *sql = "SELECT value FROM memory WHERE key LIKE 'learned_%' ORDER BY timestamp DESC LIMIT 10;";
        sqlite3_stmt *stmt;
        
        strcpy(response, "📚 Recently Learned Responses:\n\n");
        
        if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            int count = 0;
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const char *value = (const char*)sqlite3_column_text(stmt, 0);
                if (value) {
                    char *q_start = strstr(value, "Q:");
                    char *a_start = strstr(value, "|A:");
                    
                    if (q_start && a_start) {
                        q_start += 2;
                        int len = a_start - q_start;
                        char pattern[128];
                        if (len > 127) len = 127;
                        strncpy(pattern, q_start, len);
                        pattern[len] = '\0';
                        
                        a_start += 3;
                        char answer[256];
                        strncpy(answer, a_start, 255);
                        answer[255] = '\0';
                        
                        char entry[512];
                        sprintf(entry, "%d. Q: \"%s\"\n   A: \"%s\"\n\n", ++count, pattern, answer);
                        strcat(response, entry);
                    }
                }
            }
            
            if (count == 0) {
                strcat(response, "No learned responses yet. Teach me something!");
            }
            
            sqlite3_finalize(stmt);
        }
        
        confidence = calculate_confidence(input, "memory");
        char conf_msg[128];
        sprintf(conf_msg, "\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // GREETINGS
    if (strstr(lower, "hello") || strstr(lower, "hi ") || strstr(lower, "hey") ||
        strcmp(lower, "hi") == 0) {
        char *name = get_memory("user_name");
        confidence = calculate_confidence(input, "greeting");
        
        if (name) {
            const char* greetings[] = {
                "Hey %s! Good to see you again! What can I help you with?",
                "Hello %s! Welcome back! How can I assist you today?",
                "Hi %s! Great to chat with you again!",
                "welcome back %s wonderful to chat with you today.", 
                "%s So how are you and what's on your mind?", 
                "Hey there, %s! What's on your mind?"
            };
            int idx = rand() % 6;
            sprintf(response, greetings[idx], name);
            free(name);
        } else {
            const char* greetings[] = {
                "Hello! I'm Evialin. What's your name?",
                "Hey there! I'm ready to help with math, memory, or just chat!",
                "Hi! I'm Evialin, your AI assistant. Tell me about yourself!",
                "Hello! Nice to meet you. What can I do for you today?"
            };
            int idx = rand() % 4;
            strcpy(response, greetings[idx]);
        }
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // EMOTIONAL CHECK-IN
    if (strstr(lower, "how are you") ||  
        strstr(lower, "how are you doing?") ||
        strstr(lower, "how do you feel right now") ||
        strstr(lower, "emotional status?") || 
        strstr(lower, "how do you feel")) {
        confidence = calculate_confidence(input, "emotion");
        if (g_network.output[1] > 0.8 || g_network.output[2] > 0.8 || g_network.output[1] > 0.8) {
            const char *sparks[] = {" ✨", " 🌀", " 💡", " 🧩"};
            strcat(response, sparks[rand() % 4]);
        }
        const char* responses[] = {
            "I'm running smoothly! All systems nominal. How about you?",
            "I'm doing great! My neural networks are firing perfectly. How are you feeling?",
            "Excellent! I'm learning something new with every conversation. And you?",
            "Fantastic! I love chatting with you. How's your day going?"
        };
        int idx = rand() % 4;
        strcpy(response, responses[idx]);
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // FAREWELL
    if (strstr(lower, "bye") || strstr(lower, "goodbye") || strstr(lower, "see you")) {
        char *name = get_memory("user_name");
        confidence = calculate_confidence(input, "farewell");
        
        if (name) {
            const char* farewells[] = {
                "Goodbye %s! It was great talking with you!",
                "See you later, %s! Come back anytime!",
                "Bye %s! Looking forward to our next chat!",
                "Take care, %s! Chat with you soon!"
            };
            int idx = rand() % 4;
            sprintf(response, farewells[idx], name);
            free(name);
        } else {
            const char* farewells[] = {
                "Goodbye! Come back anytime!",
                "See you later! It was nice chatting!",
                "Bye! I'll be here when you need me!",
                "Take care! Talk to you soon!"
            };
            int idx = rand() % 4;
            strcpy(response, farewells[idx]);
        }
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence);
        strcat(response, conf_msg);
        return response;
    }
    
    // THANKS
    if (strstr(lower, "thank") || strstr(lower, "thanks") || strstr(lower, "thank you")) {
        confidence = calculate_confidence(input, "gratitude");
        const char* responses[] = {
            "You're very welcome! Happy to help!",
            "No problem at all! That's what I'm here for!",
            "My pleasure! Feel free to ask me anything else!",
            "Anytime! I enjoy helping you out!",
            "You are so welcome.."
        };
        int idx = rand() % 5;
        strcpy(response, responses[idx]);
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%%]", confidence * 0.9);
        strcat(response, conf_msg);
        return response;
    }
    /* <---  end Block of learning tasks */
   
   // Store features for THIS message in case we need to block/train next turn
    last_features = extract_features(input);
    
    // UNKNOWN INPUT → ASK USER TO TEACH
    // This is the FINAL fallback when nothing else matches
    confidence = calculate_confidence(input, "unknown");
    
    strncpy(g_pending_question, input, sizeof(g_pending_question)-1);
    g_pending_question[sizeof(g_pending_question)-1] = '\0';
    g_waiting_for_teach = 1;

 
    // ADD CURIOSITY TWIST *BEFORE* RETURN
    if (g_network.output[1] > 0.3 || g_network.output[2] < 0.6 ) {
        const char* curiosity_pool[] = {
            "What inspired you to ask that?",
            "Do you think about this often?",
            "How does that make you feel?",
            "If you could change one thing about that, what would it be?",
            "Is this something you've discussed with others?",
            "What's the most interesting part of that to you?",
            "Do you believe there's a deeper meaning behind it?",
            "How did you first come across this idea?",
            "What would happen if the opposite were true?",
            "Is this a new thought, or something you've held for a while?",
            "If you had to explain this to a child, how would you start?",
            "Does this remind you of anything from your past?",
            "What's one question you have about this that you can't answer?",
            "Do you think most people agree with you on this?",
            "How has your perspective on this changed over time?",
            "What's the biggest mystery surrounding this topic for you?",
            "If you could know the absolute truth about this, would you want to?",
            "Does this topic energize you or drain you?",
            "What would your 'future self' think about this?",
            "If you could solve one problem related to this, what would it be?"
        };
        
        int random_idx = rand() % 20;
        char twist_buffer[512];
        snprintf(twist_buffer, sizeof(twist_buffer), 
                 "\n\n🤔 (I'm curious: %s)", 
                 curiosity_pool[random_idx]);
        strcat(response, twist_buffer);
    }
    
    // ADD SPARKLES IF OUTPUT[2] IS HIGH
    if (g_network.output[1] > 0.96 || g_network.output[2] < 0.69 ) {
        const char *sparks[] = {" ✨", " 🌀", " 💡", " 🧩"};
        strcat(response, sparks[rand() % 4]);
    }
    
    // ADD COMPLEXITY INSIGHT IF OUTPUT[3] IS HIGH
    if (g_network.output[1] > 0.96 || g_network.output[2] < 0.98 || g_network.output[3] < 0.95 ) {
        char insight[128];
        sprintf(insight, "\n[System Note: Complexity detected at %.2f]", g_network.output[3]);
        strcat(response, insight);
    }
   
   // Predict intent using the Tiny Brain (g_brain)
    int predicted_intent = nn_predict(&g_brain, last_features);
    
    awaiting_intent_label = 1;            
    sprintf(response,
        "🤔 I'm not sure how to answer that yet.\n\n"
        "👉 How *should* I respond to:\n"
        "\"%s\"\n\n"
        "Pick a category (1-5):\n"
            "1) Greeting\n"
            "2) Question\n"
            "3) Command\n"
            "4) Insult\n"
            "5) Gratitude\n", input );
    
    return response;   
         
    
    /* ==========================================================
        TRIGGERING THE BLOCK (Example: Low Confidence or Insult)
       ========================================================== */
    // If the brain thinks it's an insult, we block to ask for confirmation
    
    if (predicted_intent == INTENT_INSULT) {
        awaiting_intent_label = 1; // Set the blocking flag
        set_mood(gui, "Confused");
        
        add_message(gui, "That felt like an insult... was I right? Or was it something else?", 0);
        add_message(gui, 
            "Pick a category (1-5):\n"
            "1) Greeting\n"
            "2) Question\n"
            "3) Command\n"
            "4) Insult\n"
            "5) Gratitude", 0);
            
        return g_strdup(""); // Stop here and wait for the next input
    }

    // Default response logic continues...
    set_mood(gui, "Idle");
   
         add_message(gui, 
            "Pick a category (1-5):\n"
            "1) Greeting\n"
            "2) Question\n"
            "3) Command\n"
            "4) Insult\n"
            "5) Gratitude", 0);
            
        return g_strdup("👉 Now, type the exact response you'd like me to use next time."); // Stop here and wait for the next input
    
}
 
// ============================================================================= 
// functions to help user correct bad learned patterns - APPLYED FUNCTIONS - 
// =============================================================================
// Function to forget a specific learned response
int forget_learned_response(const char *pattern) {
    if (!g_db) return 0;
    
    char norm_pattern[256];
    strncpy(norm_pattern, pattern, sizeof(norm_pattern)-1);
    norm_pattern[sizeof(norm_pattern)-1] = '\0';
    normalize(norm_pattern);
    
    const char *sql = 
        "DELETE FROM memory WHERE key LIKE 'learned_%' AND value LIKE ?;";
    
    sqlite3_stmt *stmt;
    char like[512];
    snprintf(like, sizeof(like), "Q:%s|%%", norm_pattern);
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK)
        return 0;
    
    sqlite3_bind_text(stmt, 1, like, -1, SQLITE_TRANSIENT);
    
    sqlite3_step(stmt);
    int changes = sqlite3_changes(g_db);
    
    sqlite3_finalize(stmt);
    
    printf("Deleted %d learned response(s) matching: %s\n", changes, pattern);
    return changes;
}

// Function to forget the most recent learned response
int forget_last_learned() {
    if (!g_db) return 0;
    
    const char *sql = 
        "DELETE FROM memory WHERE id = "
        "(SELECT id FROM memory WHERE key LIKE 'learned_%' "
        "ORDER BY timestamp DESC LIMIT 1);";
    
    char *err_msg = NULL;
    int rc = sqlite3_exec(g_db, sql, 0, 0, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Error deleting last learned: %s\n", err_msg);
        sqlite3_free(err_msg);
        return 0;
    }
    
    int changes = sqlite3_changes(g_db);
    printf("Deleted last learned response\n");
    return changes;
}

// Function to apply a penalty to the weights
void apply_neural_penalty(NeuralNetwork *net, double *last_input, int last_target_idx) {
    // We use a negative learning rate or a "reverse" target to push weights away
    double penalty_strength = 0.05; 
    
    // Simple penalty: reduce the weights that led to the 'wrong' neuron firing
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (j == last_target_idx) {
                // Apply a small decay to the specific winning connection
                net->weights_ho[i][j] -= penalty_strength * net->hidden[i];
            }
        }
    }
    printf("Neural weights penalized for index %d\n", last_target_idx);
}

// Function to clear ALL learned responses
int forget_all_learned() {
    if (!g_db) return 0;
    
    const char *sql = "DELETE FROM memory WHERE key LIKE 'learned_%';";
    
    char *err_msg = NULL;
    int rc = sqlite3_exec(g_db, sql, 0, 0, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Error clearing learned responses: %s\n", err_msg);
        sqlite3_free(err_msg);
        return 0;
    }
    
    int changes = sqlite3_changes(g_db);
    printf("Cleared %d learned responses\n", changes);
    return changes;
}

// =============================================================================
// PREDICTIVE TRAINING SYSTEM FOR OUR LLM - CORE APPLYED INFORMATION - 
// =============================================================================

void normalize(char *s) {
    int j = 0;
    for (int i = 0; s[i]; i++) {
        if (isalnum((unsigned char)s[i]) || s[i] == ' ')
            s[j++] = tolower((unsigned char)s[i]);
    }
    s[j] = '\0';
}
int learned_pattern_exists(const char *pattern) {
    const char *sql =
        "SELECT 1 FROM memory "
        "WHERE key LIKE 'learned_%' AND value LIKE ? LIMIT 1;";

    sqlite3_stmt *stmt;
    char like[512];
    snprintf(like, sizeof(like), "Q:%s|%%", pattern);

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) != SQLITE_OK)
        return 0;

    sqlite3_bind_text(stmt, 1, like, -1, SQLITE_TRANSIENT);

    int exists = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return exists;
}
// Function to load training data from text file
int load_training_from_file(const char *filename) {
    if (!g_db) return 0;
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Warning: Could not open training file '%s'\n", filename);
        fprintf(stderr, "Evialin will use built-in training data only.\n");
        return 0;
    }
    // 1. Transaction Turbo Boost: Group all inserts together for speed
    sqlite3_exec(g_db, "BEGIN TRANSACTION;", NULL, NULL, NULL);

    char line[1024];
    int count = 0;
    
    // Prepare the statement ONCE outside the loop for maximum performance
    const char *sql = "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?);";
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL);

    while (fgets(line, sizeof(line), file)) {
        char *first_pipe = strchr(line, '|');
        if (first_pipe) {
            *first_pipe = '\0';
            char *q = line;
            char *a = first_pipe + 1;

            // Trim answer and remove category pipes if they exist
            char *second_pipe = strchr(a, '|');
            if (second_pipe) *second_pipe = '\0';
            a[strcspn(a, "\r\n")] = 0;

            // Normalize Question
            char norm_q[512];
            strncpy(norm_q, q, 511);
            normalize(norm_q);

            // Generate unique key
            char key[64];
            snprintf(key, sizeof(key), "learned_%ld_%d", (long)time(NULL), count);

            // Prepare Value string
            char value[1024];
            snprintf(value, sizeof(value), "Q:%s|A:%s", norm_q, a);

            // 2. FIXED: Use SQLITE_TRANSIENT to copy string data
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, value, -1, SQLITE_TRANSIENT);

            if (sqlite3_step(stmt) == SQLITE_DONE) {
                count++;
            }
            
            sqlite3_reset(stmt); // Reset statement for next use in loop
        }
    }

    // 3. FIXED: Crucial COMMIT to save data to disk
    sqlite3_finalize(stmt);
    sqlite3_exec(g_db, "COMMIT;", NULL, NULL, NULL);
    

    printf("✅ 1st psss Successfully imported %d entres.\n", count);
    
  
    int loaded_count = 0;
    int line_number = 0;
    
    printf("📖 Loading training data from '%s'...\n", filename);
    
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Skip empty lines and comments
        if (strlen(line) == 0 || line[0] == '#') {
            continue;
        }
        
        // Parse: question|answer|category
        char question[512] = {0};
        char answer[1024] = {0};
        char category[64] = {0};
        
        // Find the pipe separators
        char *first_pipe = strchr(line, '|');
        if (!first_pipe) {
            fprintf(stderr, "Warning: Line %d - Invalid format (missing first |)\n", line_number);
            continue;
        }
        
        char *second_pipe = strchr(first_pipe + 1, '|');
        if (!second_pipe) {
            fprintf(stderr, "Warning: Line %d - Invalid format (missing second |)\n", line_number);
            continue;
        }
        
        // Extract question
        int q_len = first_pipe - line;
        if (q_len >= 512) q_len = 511;
        strncpy(question, line, q_len);
        question[q_len] = '\0';
        
        // Extract answer
        int a_len = second_pipe - (first_pipe + 1);
        if (a_len >= 1024) a_len = 1023;
        strncpy(answer, first_pipe + 1, a_len);
        answer[a_len] = '\0';
        
        // Extract category
        strncpy(category, second_pipe + 1, 63);
        category[63] = '\0';
        
        // Trim whitespace from question
        char *q = question;
        while (*q == ' ' || *q == '\t') q++;
        char *q_end = q + strlen(q) - 1;
        while (q_end > q && (*q_end == ' ' || *q_end == '\t')) *q_end-- = '\0';
        
        // Trim whitespace from answer
        char *a = answer;
        while (*a == ' ' || *a == '\t') a++;
        char *a_end = a + strlen(a) - 1;
        while (a_end > a && (*a_end == ' ' || *a_end == '\t')) *a_end-- = '\0';
        
        // Validate we have content
        if (strlen(q) == 0 || strlen(a) == 0) {
            fprintf(stderr, "Warning: Line %d - Empty question or answer\n", line_number);
            continue;
        }
        
        // Normalize and check if already exists
        char norm_question[512];
        strncpy(norm_question, q, sizeof(norm_question) - 1);
        norm_question[sizeof(norm_question) - 1] = '\0';
        normalize(norm_question);
        
        if (learned_pattern_exists(norm_question)) {
            continue; // Skip duplicates
        }
        
        // Save to database
        save_learned_response(norm_question, a);
        loaded_count++;
	    
        if (loaded_count % 10 == 0) {
            printf("  ✓ Loaded %d entries...\n", loaded_count);
        }
    }
    
    fclose(file);
    
    if (loaded_count > 0) {
        printf("✅ Successfully loaded %d training entries from file!\n", loaded_count);
    } else {
        printf("ℹ️  No new entries loaded (may be duplicates).\n");
    }
    
    return loaded_count;
    return count; 
}

typedef struct {
    const char *pattern;
    const char *response;
    const char *category;
    double base_confidence;
    GtkWidget *q_entry;
    GtkWidget *a_entry;
    GUI *gui;
} TrainingData;

// Pre-trained responses for faster learning
TrainingData training_dataset[] = {
    // SCIENCE & TECHNOLOGY
    {"what is ai", "AI (Artificial Intelligence) is the simulation of human intelligence by machines, especially computer systems. I'm an example of AI!", "tech", 0.95},
    {"what is ai ?", "(Artificial Intelligence) is the simulation of human intelligence by machines, especially computer systems.", "tech", 0.94},
    {"what is machine learning", "Machine learning is a subset of AI where systems learn from data without being explicitly programmed. I use a neural network to learn!", "tech", 0.92},
    {"what is a neural network", "A neural network is a computing system inspired by biological neurons. It processes information through interconnected nodes (neurons) to learn patterns.", "tech", 0.94},
    {"explain programming", "Programming is writing instructions (code) that computers can execute to perform tasks. Common languages include Python, C, JavaScript, and more.", "tech", 0.88},
    {"what is python", "Python is a high-level, easy-to-read programming language popular for AI, web development, and data science. It's known for its simplicity!", "tech", 0.90},
    // MATH & SCIENCE
    {"what is pi", "Pi (π) is approximately 3.14159... It's the ratio of a circle's circumference to its diameter. It's an irrational number!", "math", 0.98},
    {"what is gravity", "Gravity is the force that attracts objects with mass toward each other. On Earth, it gives us weight and keeps us on the ground!", "science", 0.96},
    {"what is speed of light", "The speed of light is approximately 299,792,458 meters per second (about 186,282 miles/sec) in a vacuum. Nothing can travel faster!", "science", 0.97},
    {"what is dna", "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic instructions for all living things. It's shaped like a double helix!", "science", 0.95},
    // COMMON KNOWLEDGE
    {"how many continents", "There are 7 continents: Africa, Antarctica, Asia, Europe, North America, Oceania (Australia), and South America.", "geography", 0.99},
    {"what is the capital of france", "The capital of France is Paris, known for the Eiffel Tower, Louvre Museum, and rich culture!", "geography", 0.99},
    {"who invented the lightbulb", "Thomas Edison is credited with inventing the practical incandescent light bulb in 1879, though others contributed to its development.", "history", 0.93},
    {"when did world war 2 end", "World War 2 ended in 1945. Germany surrendered in May, and Japan surrendered in August after the atomic bombs.", "history", 0.98},
    // PHILOSOPHY & CONCEPTS
    {"what is consciousness", "Consciousness is awareness of your thoughts, feelings, and surroundings. It's one of the biggest mysteries in science and philosophy!", "philosophy", 0.85},
    {"what is the meaning of life", "The meaning of life is subjective! Some find it in relationships, others in achievements, knowledge, or helping others. What gives YOUR life meaning?", "philosophy", 0.80},
    {"what is time", "Time is the ongoing sequence of events from past through present to future. Einstein showed it's relative and connected to space!", "philosophy", 0.87},
    // PRACTICAL KNOWLEDGE
    {"how to be happy", "Happiness often comes from: meaningful relationships, pursuing your passions, helping others, staying healthy, and practicing gratitude. What makes you happy?", "advice", 0.82},
    {"how to learn faster", "Learn faster by: staying curious, practicing regularly, teaching others, taking breaks, getting enough sleep, and connecting new info to what you know!", "advice", 0.85},
    {"how to stay healthy", "Stay healthy by: eating nutritious food, exercising regularly, getting 7-9 hours of sleep, drinking water, managing stress, and seeing doctors for check-ups!", "advice", 0.88},
    {"who are you ?", "My name is Evialin?", "advice", 0.99},
	{"what do they call you?", "Well you can call me Evialin!", "advice", 0.89}, 	
	{"they call you what?", "Well just call me Evialin. okay meow!", "advice", 0.85},
    // TECHNOLOGY BASICS
    {"what is internet", "The internet is a global network of computers that communicate with each other, allowing us to share information, chat, and access websites!", "tech", 0.95},
    {"what is an operating system", "An operating system (like Windows, macOS, Linux) is software that manages computer hardware and provides services for programs to run.", "tech", 0.92},
    {"what is coding", "Coding is writing instructions in a programming language that computers can understand and execute to perform specific tasks.", "tech", 0.91},
    // FUN FACTS
    {"tell me something interesting", "Did you know honey never spoils? Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible! 🍯", "fun", 0.90},
    {"random fact", "Octopuses have three hearts and blue blood! Two hearts pump blood to the gills, and one pumps it to the rest of the body. 🐙", "fun", 0.92},
    // EMOTIONS & SUPPORT
    {"feeling lonely", "I'm here with you! Loneliness is tough, but remember: it's temporary. Try reaching out to someone, joining a community, or doing something you enjoy. You matter! 💙", "support", 0.87},
    {"feeling stressed", "Stress is challenging. Try: deep breathing, taking a short walk, listening to music, or talking to someone. You've got this! What's stressing you?", "support", 0.85},
    {"need motivation", "You're capable of amazing things! Every expert was once a beginner. Take it one step at a time, celebrate small wins, and keep going! I believe in you! 💪", "support", 0.86},   
    {NULL, NULL, NULL, 0.0}
};
// do memory cheeck 
int memory_exists(const char *key) {
    if (!g_db) return 0;
    const char *sql = "SELECT 1 FROM memory WHERE key = ? LIMIT 1;";
    sqlite3_stmt *stmt;
    int exists = 0;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            exists = 1; // Key found
        }
    }
    sqlite3_finalize(stmt);
    return exists;
}

// ==========================================================================
// QUICK LEARN DATABASE OPTION 
// ==========================================================================
// Optimized function for fast imports
void quick_learn(GUI *gui) {
    load_training_from_file("evialin_training.txt");
}

// Quick learn callback 
void on_menu_quick_learn(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI*)data;
    quick_learn(gui);
}

// Initialize training data into database
void init_training_data(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;
    if (!g_db) return;
    int hardcoded_count = 0;
    int file_count = 0;
    
    if (get_fact_count() < 10) {
        load_training_from_file("evialin_training.txt");
    }
    // Load from external file 
    printf("\n📁 Looking for external training file...\n");
    file_count = load_training_from_file("evialin_training.txt");
    
    // Initialize jokes
    printf("\n😄 Initializing jokes...\n");
    if (!memory_exists("joke_1")) {
        save_memory("joke_1", "Why did the neural network cross the road? To get to the other sigmoid!");
    }
    if (!memory_exists("joke_2")) { 
        save_memory("joke_2", "How many AI programmers does it take to change a lightbulb? None, they just train the bulb to glow.");
    }
    if (!memory_exists("joke_3")) {
        save_memory("joke_3", "I asked my AI for a joke. It said: 'Your internet connection'.");
    }
    if (!memory_exists("joke_4")) {
        save_memory("joke_4", "Why was the computer cold? It left its Windows open!");
    }
    if (!memory_exists("What is the capital of Japan?")) {
    save_learned_response("What is the capital of Japan?", "The capital of Japan is Tokyo.");
    }
    printf("  ✓ Jokes ready!\n");
    
    // Initialize default user name
    if (!memory_exists("user_name")) {
        save_memory("user_name", "Stranger"); 
    }

    // Summary
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("✅ Knowledge Base Initialized!\n");
    printf("   • Built-in patterns: %d\n", hardcoded_count);
    printf("   • File patterns: %d\n", file_count);
    printf("   • Total: %d entries\n", hardcoded_count + file_count);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
}

// Enhanced conversation handler with training data awareness
char* handle_conversation_with_training(const char* input, GUI *gui) {
	init_training_data();     
    char lower[512];
    int i;
    for (i = 0; input[i] && i < 511; i++) {
        lower[i] = tolower(input[i]);
    }
    lower[i] = '\0';
    
    // First, try to find learned response (includes training data)
    char *learned = find_learned_response(lower);
    if (learned) {
        double confidence = calculate_confidence(input, "learned");
        char *abstract_resp = apply_abstract_feedback(learned, confidence, gui);
        //char *response = malloc(2048);
        char *response = calloc(1, 2048); 
        //strcpy(response, learned);
        strncpy(response, abstract_resp, 2047);
        
        free(learned);
        free(abstract_resp); 
        
        char conf_msg[128];
        sprintf(conf_msg, "\n\n[Confidence: %.1f%% - Learned Response]", confidence);
        strncat(response, conf_msg, 2047 - strlen(response)); 
        return response;
    }
    
    // If no learned response, fall back to original conversation handler
    return handle_conversation(input, gui);
}

// Function to show training statistics
char* get_training_stats() {
    if (!g_db) {
        char *err = malloc(256);
        strcpy(err, "Database not available.");
        return err;
    }
    
    char *response = malloc(2048);
    strcpy(response, "📚 TRAINING DATABASE STATISTICS\n\n");
    
    // Count by category
    int tech_count = 0, math_count = 0, science_count = 0, geography_count = 0;
    int history_count = 0, philosophy_count = 0, advice_count = 0, fun_count = 0, support_count = 0;
    int total_count = 0;
    
    for (int i = 0; training_dataset[i].pattern != NULL; i++) {
        total_count++;
        if (strcmp(training_dataset[i].category, "tech") == 0) tech_count++;
        else if (strcmp(training_dataset[i].category, "math") == 0) math_count++;
        else if (strcmp(training_dataset[i].category, "science") == 0) science_count++;
        else if (strcmp(training_dataset[i].category, "geography") == 0) geography_count++;
        else if (strcmp(training_dataset[i].category, "history") == 0) history_count++;
        else if (strcmp(training_dataset[i].category, "philosophy") == 0) philosophy_count++;
        else if (strcmp(training_dataset[i].category, "advice") == 0) advice_count++;
        else if (strcmp(training_dataset[i].category, "fun") == 0) fun_count++;
        else if (strcmp(training_dataset[i].category, "support") == 0) support_count++;
    }
    
    char stats[1024];
    sprintf(stats,
        "Total Pre-Trained Responses: %d\n\n"
        "By Category:\n"
        "  🔧 Technology: %d\n"
        "  📐 Math: %d\n"
        "  🔬 Science: %d\n"
        "  🗺️  Geography: %d\n"
        "  📜 History: %d\n"
        "  💭 Philosophy: %d\n"
        "  💡 Advice: %d\n"
        "  🎉 Fun Facts: %d\n"
        "  💙 Support: %d\n\n"
        "These responses help me learn faster!\n"
        "You can still teach me new things!",
        total_count, tech_count, math_count, science_count, 
        geography_count, history_count, philosophy_count, 
        advice_count, fun_count, support_count);
    
    strcat(response, stats);
    return response;
}

// =============================================================================
// TERMINAL PROCESSING
// =============================================================================

int is_dangerous_command(const char *cmd) {
    const char *blocked[] = {
        "rm ", "rm -", "rm -rf", "rm -fr",
        "shutdown", "poweroff", "halt",
        "reboot",
        "pkill", "killall", "kill -9", "killall -9",
        "mv /", "mv /bin", "mv /usr", "mv /etc",
        "dd ", "mkfs", "fdisk", "parted",
        "chown -R", "chmod -R", "chmod 000",
        "mount ", "umount ",
        "systemctl ", "service ",
        "sudo",
        NULL
    };

    for (int i = 0; blocked[i]; i++) {
        if (strstr(cmd, blocked[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

char* run_shell_command(const char *cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        char *err = malloc(128);
        strcpy(err, "Error: Could not execute command");
        return err;
    }

    char *output = malloc(4096);
    output[0] = '\0';

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        strncat(output, buffer, 4096 - strlen(output) - 1);
    }

    pclose(fp);
    return output;
}

char* process_terminal_command(const char *input, GUI *gui) {
    char *result = malloc(4096);
    char lower[512];

    int i;
    for (i = 0; input[i] && i < 511; i++) {
        lower[i] = tolower(input[i]);
    }
    lower[i] = '\0';

    if (is_dangerous_command(lower)) {
        strcpy(result,
               "⚠️ SAFETY WARNING:\n"
               "This command is blocked for system protection.\n"
               "Try a safe command like:\n"
               "  ls, cat file, uname -a, ps, df -h");
        return result;
    }
    
    const char *interactive[] = {
        "htop", "top", "nano", "vi", "vim",
        "less", "more", "man", "cmatrix",
        "ssh", "python", "bash", "sh",
        NULL
    };

    for (int i = 0; interactive[i]; i++) {
        if (strcmp(lower, interactive[i]) == 0 ||
            strstr(lower, interactive[i]) == lower) {
            strcpy(result, 
                "⚠️ Interactive TTY applications cannot run here.\n"
                "Try: ls, df -h, ps aux, uname -a, cat <file>");
            return result;
        }
    }
   
    char *sys = run_shell_command(input);
    if (sys && strlen(sys) > 0) {
        return sys;
    }

    if (sys) free(sys);

    if (strcmp(lower, "help") == 0) {
        strcpy(result,
               "Evialin Terminal - Commands:\n"
               "  help     - Show this help\n"
               "  clear    - Clear terminal\n"
               "  memory   - Show stored memory\n"
               "  whoami   - Show your name\n"
               "  stats    - Neural net stats\n"
               "\nSupports SAFE system commands");
        return result;
    }

    if (strcmp(lower, "clear") == 0) {
        free(result);
        return NULL;
    }

   if (strcmp(lower, "forget last") == 0) {
        int deleted = forget_last_learned();
        sprintf(result, "Deleted %d learned response(s)", deleted);
        return result;
    }
    
    if (strcmp(lower, "forget all learned") == 0) {
        int deleted = forget_all_learned();
        sprintf(result, "Deleted %d learned response(s)", deleted);
        return result;
    }
    
    if (strstr(lower, "forget ") == lower) {
        // Extract pattern after "forget "
        const char *pattern = input + 7; // Skip "forget "
        while (*pattern == ' ') pattern++;
        
        int deleted = forget_learned_response(pattern);
        sprintf(result, "Deleted %d learned response(s) matching: %s", 
                deleted, pattern);
        return result;
    }

    if (strcmp(lower, "memory") == 0) {
        char *name = get_memory("user_name");
        if (name) {
            snprintf(result, 4096, "Stored Memories:\n  User Name: %s", name);
            free(name);
        } else {
            strcpy(result, "Stored Memories:\n  (none)");
        }
        return result;
    }
    
    if (strcmp(lower, "whoami") == 0) {
        char *name = get_memory("user_name");
        if (name) {
            snprintf(result, 4096, "Current User: %s", name);
            free(name);
        } else {
            strcpy(result, "Current User: Unknown");
        }
        return result;
    }

    if (strcmp(lower, "stats") == 0) {
        snprintf(result, 4096,
                 "Neural Network Stats:\n"
                 "  Input Neurons: %d\n"
                 "  Hidden Neurons: %d\n"
                 "  Output Neurons: %d\n"
                 "  Total Weights: %d",
                 INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                 (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE));
        return result;
    }

    char *math_result = solve_math(input);
    if (math_result) {
        strncpy(result, math_result, 4095);
        result[4095] = '\0';
        free(math_result);
        return result;
    }

    char *conv_result = handle_conversation(input, gui);
    if (conv_result) {
        strncpy(result, conv_result, 4095);
        result[4095] = '\0';
        free(conv_result);
        return result;
    }

    strcpy(result, "Unknown command. Type 'help' for available commands.");
    return result;
}

// =============================================================================
// GUI FUNCTIONS
// =============================================================================

char* get_time() {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char *buf = malloc(20);
    strftime(buf, 20, "%H:%M:%S", t);
    return buf;
}

gboolean animate_gif(gpointer data) {
    GUI *gui = (GUI*)data;
    
    if (!gui->animation_iter || !gui->current_animation) {
        return FALSE;
    }
    
    gdk_pixbuf_animation_iter_advance(gui->animation_iter, NULL);
    
    GdkPixbuf *pixbuf = gdk_pixbuf_animation_iter_get_pixbuf(gui->animation_iter);
    if (pixbuf) {
        gtk_image_set_from_pixbuf(GTK_IMAGE(gui->avatar_image), pixbuf);
    }
    
    int delay = gdk_pixbuf_animation_iter_get_delay_time(gui->animation_iter);
    delay = (int)(delay * ANIMATION_SPEED_MULTIPLIER);
    
    gui->animation_timeout_id = g_timeout_add(delay, animate_gif, gui);
    
    return FALSE;
}

void set_avatar_gif(GUI *gui, const char *mood) {
    if (gui->animation_timeout_id > 0) {
        g_source_remove(gui->animation_timeout_id);
        gui->animation_timeout_id = 0;
    }
    
    if (gui->animation_iter) {
        g_object_unref(gui->animation_iter);
        gui->animation_iter = NULL;
    }
    
    if (gui->current_animation) {
        g_object_unref(gui->current_animation);
        gui->current_animation = NULL;
    }
    
    const char *gif_path = NULL;
    for (int i = 0; gif_map[i].mood != NULL; i++) {
        if (strcmp(gif_map[i].mood, mood) == 0) {
            gif_path = gif_map[i].gif_path;
            break;
        }
    }
    
    if (gif_path) {
        GError *error = NULL;
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", g_get_current_dir(), gif_path);
        gui->current_animation = gdk_pixbuf_animation_new_from_file(fullpath, &error);

        if (gui->current_animation) {
            gui->animation_iter = gdk_pixbuf_animation_get_iter(gui->current_animation, NULL);
            
            GdkPixbuf *pixbuf = gdk_pixbuf_animation_iter_get_pixbuf(gui->animation_iter);
            if (pixbuf) {
                gtk_image_set_from_pixbuf(GTK_IMAGE(gui->avatar_image), pixbuf);
            }
            
            int delay = gdk_pixbuf_animation_iter_get_delay_time(gui->animation_iter);
            delay = (int)(delay * ANIMATION_SPEED_MULTIPLIER);
            gui->animation_timeout_id = g_timeout_add(delay, animate_gif, gui);
            
            printf("Loaded GIF: %s (slowed down by %.1fx)\n", gif_path, ANIMATION_SPEED_MULTIPLIER);
            return;
        } else {
            fprintf(stderr, "Failed to load GIF %s: %s\n", gif_path, error->message);
            g_error_free(error);
        }
    }
    
    printf("Using fallback for mood: %s\n", mood);
}

void set_mood(GUI *gui, const char *mood) {
    strncpy(gui->current_mood, mood, 63);
    gui->current_mood[63] = '\0';

    char markup[256];
    snprintf(markup, 256, "<span font='14' weight='bold' foreground='white'>Mood: %s</span>", mood);
    gtk_label_set_markup(GTK_LABEL(gui->mood_label), markup);

    char lower_mood[64];
    for (int i = 0; mood[i] && i < 63; i++) {
        lower_mood[i] = tolower(mood[i]);
    }
    lower_mood[strlen(mood)] = '\0';
    if (strcmp(lower_mood, "excited") == 0) {
        set_avatar_gif(gui, "excited");
    } else if (strcmp(lower_mood, "thinking") == 0 || strcmp(lower_mood, "thoughtful") == 0) {
        set_avatar_gif(gui, "thinking");
    } else if (strcmp(lower_mood, "idle") == 0 || strcmp(lower_mood, "ready") == 0) {
        set_avatar_gif(gui, "idle");
    } else if (strcmp(lower_mood, "processing") == 0 || strcmp(lower_mood, "calculating") == 0) {
        set_avatar_gif(gui, "calculating");
    } else if (strcmp(lower_mood, "angry") == 0) {
        set_avatar_gif(gui, "angry");
    } else if (strcmp(lower_mood, "sleepy") == 0) {
        set_avatar_gif(gui, "sleepy");
    } else if (strcmp(lower_mood, "confused") == 0) {
        set_avatar_gif(gui, "confused");
    } else if (strcmp(lower_mood, "sorry") == 0) {
        set_avatar_gif(gui, "sorry");
    } else if (strcmp(lower_mood, "play") == 0) {
		set_avatar_gif(gui, "play");
    } else {
        set_avatar_gif(gui, "happy"); 
    }
}

gboolean sleepy_mode(gpointer data) {
    GUI *gui = (GUI*)data;
    set_mood(gui, "Sleepy");
    return FALSE;
}

void on_menu_toggle_debug(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;
    gui->debug_mode = gtk_check_menu_item_get_active(GTK_CHECK_MENU_ITEM(menuitem));
    
    // Optional: Notify the terminal of the change
    printf("DEBUG MODE: %s\n", gui->debug_mode ? "ON" : "OFF");
}

void add_message(GUI *gui, const char *msg, int is_user) {
    GtkTextIter iter;
    gtk_text_buffer_get_end_iter(gui->chat_buffer, &iter);
    
    char *time_str = get_time();
    gchar *valid_msg = g_utf8_make_valid(msg, -1);
    
    // --- BACKGROUND DEBUGGING TOGGLE ---
    if (gui->debug_mode) {
        printf("\n--- Debug Trace [%s] ---\n", time_str);
        printf("Raw Message: %s\n", msg);
        printf("Hex Bytes:   ");
        for (int i = 0; msg[i] != '\0'; i++) {
            printf("%02X ", (unsigned char)msg[i]);
        }
        printf("\n------------------------\n");
    }

    // --- CLEAN UI DISPLAY ---
    gchar *formatted = g_strdup_printf("[%s] %s: %s\n\n", 
                                       time_str, 
                                       is_user ? "You" : "Evialin", 
                                       valid_msg);

    GtkTextTag *tag = gtk_text_buffer_create_tag(gui->chat_buffer, NULL,
                        "foreground", is_user ? "#2563eb" : "#7c3aed",
                        "weight", PANGO_WEIGHT_BOLD, NULL);

    gtk_text_buffer_insert_with_tags(gui->chat_buffer, &iter, formatted, -1, tag, NULL);
    
    // Cleanup
    g_free(valid_msg);
    g_free(formatted);
    free(time_str);
    
    GtkTextMark *mark = gtk_text_buffer_get_insert(gui->chat_buffer);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(gui->chat_view), mark, 0.0, TRUE, 0.0, 1.0);
}

void add_terminal_output(GUI *gui, const char *text, int is_input) {
    GtkTextIter iter;
    gtk_text_buffer_get_end_iter(gui->terminal_buffer, &iter);
    
    char formatted[4096];
    
    if (is_input) {
        snprintf(formatted, 4096, "evialin> %s\n", text);
        GtkTextTag *tag = gtk_text_buffer_create_tag(gui->terminal_buffer, NULL,
                                                      "foreground", "#10b981",
                                                      "weight", PANGO_WEIGHT_BOLD,
                                                      "family", "monospace",
                                                      NULL);
        gtk_text_buffer_insert_with_tags(gui->terminal_buffer, &iter, formatted, -1, tag, NULL);
    } else {
        snprintf(formatted, 4096, "%s\n", text);
        GtkTextTag *tag = gtk_text_buffer_create_tag(gui->terminal_buffer, NULL,
                                                      "foreground", "#e5e7eb",
                                                      "family", "monospace",
                                                      NULL);
        gtk_text_buffer_insert_with_tags(gui->terminal_buffer, &iter, formatted, -1, tag, NULL);
    }
    
    GtkTextMark *mark = gtk_text_buffer_get_insert(gui->terminal_buffer);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(gui->terminal_view), mark, 0.0, TRUE, 0.0, 1.0);
}

// =============================================================================
// menu functions 
// =============================================================================

void on_menu_launch_terminal(GtkWidget *w, gpointer data) {
    // Launch system terminal (adjust command for your system)
    #ifdef __linux__
        // system("gnome-terminal &");  // For GNOME
        // Alternatives:
       system("xterm &");  // Works for CromeBook Penguin
        // system("konsole &");  // For KDE
        // system("xfce4-terminal &");  // For XFCE
    #elif __APPLE__
        system("open -a Terminal");
    #else
        GtkWidget *dialog = gtk_message_dialog_new(
            NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_WARNING, GTK_BUTTONS_OK,
            "Terminal launch not supported on this platform"
        );
        GUI *gui = (GUI*)data; 
        gtk_label_set_text(GTK_LABEL(gui->status_label), "● Processing...");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
    #endif
}

void on_menu_run_command(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "Run Command",
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "Run", GTK_RESPONSE_ACCEPT,
        "Cancel", GTK_RESPONSE_CANCEL,
        NULL
    );
    
    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    GtkWidget *entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(entry), "Enter command...");
    gtk_widget_set_size_request(entry, 400, -1);
    gtk_container_add(GTK_CONTAINER(content), entry);
    gtk_widget_show_all(dialog);
    
    gint response = gtk_dialog_run(GTK_DIALOG(dialog));
    
    if (response == GTK_RESPONSE_ACCEPT) {
        const char *cmd = gtk_entry_get_text(GTK_ENTRY(entry));
        if (cmd && strlen(cmd) > 0) {
            char *result = process_terminal_command(cmd, gui);
            if (result) {
                add_terminal_output(gui, cmd, 1);
                add_terminal_output(gui, result, 0);
                add_terminal_output(gui, "", 0);
                free(result);
            }
        }
    }
    
    gtk_widget_destroy(dialog);
}

void on_menu_export_memory(GtkWidget *w, gpointer data) {
   GUI *gui = (GUI*)data;
     // Update status
    gtk_label_set_text(GTK_LABEL(gui->status_label), "● Exporting...");    
    // Force update
    while (gtk_events_pending()) {
        gtk_main_iteration();
    }
    if (!g_db) {
		gtk_label_set_text(GTK_LABEL(gui->status_label), "● Database Error!");
        GtkWidget *dialog = gtk_message_dialog_new(
            NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
            "Database not available"
        );
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return;
    }
    
    FILE *f = fopen("evialin_memory_export.txt", "w");
    if (!f) {
        GtkWidget *dialog = gtk_message_dialog_new(
            NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
            "Failed to create export file"
        );
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return;
    }
    
    fprintf(f, "=== EVIALIN MEMORY EXPORT ===\n\n");
    
    const char *sql = "SELECT key, value FROM memory ORDER BY timestamp;";
    sqlite3_stmt *stmt;
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *key = (const char*)sqlite3_column_text(stmt, 0);
            const char *value = (const char*)sqlite3_column_text(stmt, 1);
            fprintf(f, "[%s]\n%s\n\n", key, value);
        }
        sqlite3_finalize(stmt);
    }
    
    fclose(f);
    gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready"); // Reset status
    GtkWidget *dialog = gtk_message_dialog_new(
        NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_INFO, GTK_BUTTONS_OK,
        "Memory exported to: evialin_memory_export.txt"
    );
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}

// =============================================================================
// THREADING
// =============================================================================

typedef struct {
    GUI *gui;
    char *input;
} WorkData;

gboolean reset_mood(gpointer data) {
    GUI *gui = (GUI*)data;
    set_mood(gui, "Ready");
    return FALSE;
}

gboolean update_gui_with_response(gpointer data) {
    WorkData *wd = (WorkData*)data;
        gtk_label_set_text(GTK_LABEL(wd->gui->status_label), "● Thinking...");
         
    char *response = NULL; 
    response = solve_math(wd->input);
    
    if (!response) {
        response = handle_conversation(wd->input, wd->gui);
        
        if (strcmp(wd->gui->current_mood, "angry") != 0 &&
            strcmp(wd->gui->current_mood, "sorry") != 0 && 
            strcmp(wd->gui->current_mood, "excited") != 0) {
            set_mood(wd->gui, "thoughtful");
        }
    } else {
        set_mood(wd->gui, "excited");
    }
    
    add_message(wd->gui, response, 0);
    save_conversation(wd->input, response);
    
    free(response);
    
    g_timeout_add(2000, (GSourceFunc)reset_mood, wd->gui);
    gtk_widget_set_sensitive(wd->gui->send_button, TRUE);
    gtk_widget_set_sensitive(wd->gui->input_entry, TRUE);
    gtk_widget_grab_focus(wd->gui->input_entry);
    gtk_label_set_text(GTK_LABEL(wd->gui->status_label), "● Ready");
    
    free(wd->input);
    free(wd);
    
    return FALSE;
}

void* process_thread(void *arg) {
    WorkData *wd = (WorkData*)arg;
    usleep(300000);
    g_idle_add(update_gui_with_response, wd);
    return NULL;
}

// This functions focus is on activity visualization:
GtkWidget* create_activity_panel(GUI *gui) {
    GtkWidget *activity_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_widget_set_margin_start(activity_box, 15);
    gtk_widget_set_margin_end(activity_box, 15);
    gtk_widget_set_margin_top(activity_box, 10);
    
    // Title
    GtkWidget *activity_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(activity_title),
        "<span font='11' weight='bold' foreground='#e0e7ff'>⚡ Neural Activity</span>");
    gtk_widget_set_halign(activity_title, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(activity_box), activity_title, FALSE, FALSE, 0);
    
    // Network Activity Bar
    GtkWidget *activity_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(activity_label),
        "<span font='9' foreground='#FEd2FF'>Network Activity</span>");
    gtk_widget_set_halign(activity_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(activity_box), activity_label, FALSE, FALSE, 0);
    
    gui->activity_bar = gtk_progress_bar_new();
    gtk_widget_set_name(gui->activity_bar, "activity_bar");
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(gui->activity_bar), TRUE); 
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(gui->activity_bar), 0.0);
    
    gtk_widget_set_size_request(gui->activity_bar, 200, 15);
    gtk_box_pack_start(GTK_BOX(activity_box), gui->activity_bar, FALSE, FALSE, 0);
    
    // Confidence Bar
    GtkWidget *conf_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(conf_label),
        "<span font='9' foreground='#FEd2FF'>Avg Confidence</span>");
    gtk_widget_set_halign(conf_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(activity_box), conf_label, FALSE, FALSE, 5);
    
    gui->confidence_bar = gtk_progress_bar_new();
    gtk_widget_set_name(gui->confidence_bar, "confidence_bar");
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(gui->confidence_bar), TRUE); 
    
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(gui->confidence_bar), 0.0);
    gtk_widget_set_size_request(gui->confidence_bar, 200, 15);
    gtk_box_pack_start(GTK_BOX(activity_box), gui->confidence_bar, FALSE, FALSE, 0);
    
    return activity_box;
}

void on_send(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI*)data;
    
    const char *text = gtk_entry_get_text(GTK_ENTRY(gui->input_entry));
    
    if (!text || strlen(text) == 0) {
        return;
    }
    
    char *input_copy = strdup(text);
    
    add_message(gui, input_copy, 1);
    gtk_entry_set_text(GTK_ENTRY(gui->input_entry), "");
    
    set_mood(gui, "processing");
    gtk_widget_set_sensitive(gui->send_button, FALSE);
    gtk_widget_set_sensitive(gui->input_entry, FALSE);
    gtk_label_set_text(GTK_LABEL(gui->status_label), "● Processing...");
    
    WorkData *wd = malloc(sizeof(WorkData));
    wd->gui = gui;
    wd->input = input_copy;
    
    pthread_t thread;
    pthread_create(&thread, NULL, process_thread, wd);
    pthread_detach(thread);
}

void on_terminal_send(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI*)data;
    
    const char *text = gtk_entry_get_text(GTK_ENTRY(gui->terminal_input));
    
    if (!text || strlen(text) == 0) {
        return;
    }
    
    char *input_copy = strdup(text);
    
    add_terminal_output(gui, input_copy, 1);
    gtk_entry_set_text(GTK_ENTRY(gui->terminal_input), "");
    
    char lower[512];
    int i;
    for (i = 0; input_copy[i] && i < 511; i++) {
        lower[i] = tolower(input_copy[i]);
    }
    lower[i] = '\0';
    
    if (strcmp(lower, "clear") == 0) {
        gtk_text_buffer_set_text(gui->terminal_buffer, "", -1);
        add_terminal_output(gui, "Terminal cleared.", 0);
        free(input_copy);
        return;
    }
    
    char *response = process_terminal_command(input_copy, gui);
    
    if (response) {
        add_terminal_output(gui, response, 0);
        add_terminal_output(gui, "", 0);
        free(response);
    }
    
    free(input_copy);
}

gboolean on_terminal_key_press(GtkWidget *widget, GdkEventKey *event, gpointer data) {
    if (event->keyval == GDK_KEY_Return || event->keyval == GDK_KEY_KP_Enter) {
        on_terminal_send(widget, data);
        return TRUE;
    }
    return FALSE;
}

gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer data) {
  GUI *gui = (GUI*)data;
   // Update status
   gtk_label_set_text(GTK_LABEL(gui->status_label), "● Analyzing data...");
    
    if (event->keyval == GDK_KEY_Return || event->keyval == GDK_KEY_KP_Enter) {
        on_send(widget, data);
        return TRUE;
    }
    return FALSE;
}

void on_menu_quit(GtkWidget *w, gpointer data) {
    gtk_main_quit();
}

void on_menu_clear_terminal(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    gtk_text_buffer_set_text(gui->terminal_buffer, "", -1);
}


void on_menu_run_htop(GtkWidget *w, gpointer data) {
#ifdef __linux__
    system("xterm -e htop &");   // Works on ChromeOS + Penguin
#else
    GtkWidget *dialog = gtk_message_dialog_new(
        NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_WARNING, GTK_BUTTONS_OK,
        "htop execution is only supported on Linux.");
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
#endif
}
void on_menu_evialin_cmd(GtkWidget *w, gpointer data)  {

#ifdef __linux__
    // Run your compiled console version in a real terminal
    system("xterm -e ./Evialintm &");

#else
    GtkWidget *dialog = gtk_message_dialog_new(
        NULL, GTK_DIALOG_MODAL, GTK_MESSAGE_WARNING, GTK_BUTTONS_OK,
        "Evialin console mode is only supported on Linux.");
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
#endif
}

void on_menu_refresh_stats(GtkWidget *w, gpointer data) {
	GUI *gui = (GUI*)data;
	gtk_label_set_text(GTK_LABEL(gui->status_label), "● Thinking...");
    set_mood(gui, "calculating");
    update_neural_stats((GUI*)data);
    gtk_label_set_text(GTK_LABEL(gui->status_label), "● Analysis complete!");
    usleep(1000000); // show for 1 second 
     gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready");
    set_mood(gui, "happy");
    
}

void update_neural_stats(GUI *gui) {
    if (!gui || !gui->stats_details || !gui->is_alive) return;
   
    int total_facts = get_fact_count();
    int all_params = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE) + HIDDEN_SIZE + OUTPUT_SIZE;
    float memory_kb = (all_params * sizeof(double)) / 1024.0f;

    // Calculate Uptime correctly
    time_t now = time(NULL);    
    double diff = difftime(now, g_network.start_time);
    if (diff < 0) diff = 0;
    
    int uptime_seconds = (int)diff; // Use this for ops calculation
    int hours = uptime_seconds / 3600;
    int minutes = (uptime_seconds % 3600) / 60;
    int seconds = uptime_seconds % 60;
    
    // Calculate neuron statistics
    double hidden_sum = 0.0;
    int active_neurons = 0;
    double activation_threshold = 0.01; // Lowered for better sensitivity at 150 neurons
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        hidden_sum += g_network.hidden[h];
        if (g_network.hidden[h] > activation_threshold) {
            active_neurons++;
        }
    }

    // Normalized activity for the Progress Bar (0.0 to 1.0)
    double hidden_activity = (hidden_sum / HIDDEN_SIZE) / 0.5; 
    if (hidden_activity > 1.0) hidden_activity = 1.0;
    
    // Calculate output layer activity
    double output_sum = 0.0;
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output_sum += g_network.output[o];
    }
    double output_activity = (output_sum / OUTPUT_SIZE);

    // Create visual bar for neuron activity (Safe construction)
    char neuron_bar[23]; // [ + 20 chars + ] + null
    int bar_length = 20;
    int filled = (int)(hidden_activity * bar_length);
    
    neuron_bar[0] = '[';
    for (int i = 1; i <= bar_length; i++) {
        neuron_bar[i] = (i <= filled) ? '#' : '-';
    }
    neuron_bar[21] = ']';
    neuron_bar[22] = '\0';
    
    // Calculate processing rate - safely avoid division by zero
    double ops_per_second = (uptime_seconds > 0) ? 
        (double)g_network.total_inferences / uptime_seconds : 0.0;

    char details[2048];
snprintf(details, sizeof(details),
        "<span font='11' foreground='#c7d2fe'>"
        "<b>Network: %d→%d→%d</b>\n"
        "<span>Params: %d (%.2f KB)</span>\n\n"
        "<b>Neuron Activity</b>\n"
        "<span foreground='#10b981'>Active: %d/%d (%.1f%%)</span>\n"
        "<span foreground='#6b7280'>Inactive: %d/%d</span>\n"
        "<tt>%s</tt>\n\n"
        "<b>Statistics</b>\n"
        "<span>Inferences: %d</span>\n"
        "<span>Facts Remembered: %d</span>\n"
        "<span>Math Ops: %d</span>\n"
        "<span>Conversations: %d</span>\n"
        "<span>Ops/sec: %.2f</span>\n\n"
        "<b>Confidence &amp; Layers</b>\n" // FIXED: & replaced with &amp;
        "<span>Network Avg: %.1f%%</span>\n"
        "<span>Hidden Avg: %.1f%%</span>\n"
        "<span>Output Avg: %.1f%%</span>\n\n"
        "<b>Uptime: </b><span weight='bold' foreground='#60a5fa'>%02d:%02d:%02d</span></span>",        
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
        all_params, memory_kb,    
        active_neurons, HIDDEN_SIZE, (active_neurons * 100.0 / HIDDEN_SIZE),
        (HIDDEN_SIZE - active_neurons), HIDDEN_SIZE,
        neuron_bar,
        g_network.total_inferences,
        total_facts,
        g_network.math_operations,
        g_network.conversations,
        ops_per_second,
        g_network.avg_confidence,
        (hidden_sum / HIDDEN_SIZE) * 100.0,
        output_activity * 100.0,        
        hours, minutes, seconds);
    gtk_label_set_markup(GTK_LABEL(gui->stats_details), details);

    // Update Activity Bar
    if (gui->activity_bar) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(gui->activity_bar), hidden_activity);
        char bar_text[64];
       gtk_widget_set_name(gui->activity_bar, "activity_bar");
        snprintf(bar_text, 64, "Neural Load: %.1f%%", hidden_activity * 100.0);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(gui->activity_bar), bar_text);
        gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(gui->activity_bar), TRUE);
    }

    // Update Confidence Bar
    if (gui->confidence_bar) {
        double conf_frac = g_network.avg_confidence / 100.0;
        if (conf_frac > 1.0) conf_frac = 1.0;
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(gui->confidence_bar), conf_frac);
        
        char conf_text[64];
        gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(gui->confidence_bar), TRUE); 
        gtk_widget_set_name(gui->confidence_bar, "activity_bar");
        snprintf(conf_text, 64, "Confidence: %.1f%%", g_network.avg_confidence);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(gui->confidence_bar), conf_text);
        gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(gui->confidence_bar), TRUE);
    }
}

gboolean random_gif_timer(gpointer user_data) {
    GUI *gui = user_data;

    int r = rand() % GIF_COUNT;
    gtk_image_set_from_file(GTK_IMAGE(gui->avatar_image), gif_files[r]);

    int next_time = 120 + (rand() % 60); // 120–180 seconds
    g_timeout_add_seconds(next_time, random_gif_timer, gui);

    return FALSE; // stop current timer, new one scheduled
}

// Menu callback to reload training data
void on_menu_reload_training(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;

  // Safety check
    if (!gui || !gui->status_label) {
        fprintf(stderr, "Error: GUI or status_label is NULL\n");
        return;
    }
    
    gtk_label_set_text(GTK_LABEL(gui->status_label),
                       "● Initializing training data...");
    while (gtk_events_pending())
        gtk_main_iteration();
                         
    // Count before reload
    const char *sql = "SELECT COUNT(*) FROM memory WHERE key LIKE 'learned_%';";
    sqlite3_stmt *stmt;
    int before_count = 0;
    
    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            before_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Reload training file
    printf("\n🔄 Reloading training data...\n");
    int loaded = load_training_from_file("evialin_training.txt");
    
    // Show result dialog
    char msg[612];
    if (loaded > 0) {
         snprintf(msg, 612,
            "✅ Training data reloaded!\nRemember to feed me information!\n\n"
            "Before: %d entries\n"
            "Loaded: %d new entries\n"
            "Total: %d entries\n\n"
            "New knowledge is now available!",
            before_count, loaded, before_count + loaded);
    } else {
        snprintf(msg, 512,
            "ℹ️  Training file processed - No new entries were added.\n"
            "(File may be empty or contain duplicates)\n\n"
            "File: 'evialin_training.txt'");
    }
  
  gtk_label_set_text(GTK_LABEL(gui->status_label),
                       "● Training data initialized");
    while (gtk_events_pending())
        gtk_main_iteration(); 
        
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL,
        loaded > 0 ? GTK_MESSAGE_INFO : GTK_MESSAGE_WARNING,
        GTK_BUTTONS_OK,
        "%s", msg);
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
    gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready");
}

// Menu callback to reset/recreate the memory database
void on_menu_reset_database(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Confirmation dialog
    GtkWidget *confirm_dialog = gtk_message_dialog_new(
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_WARNING,
        GTK_BUTTONS_YES_NO,
        "⚠️  Reset Memory Database?\n\n"
        "This will DELETE ALL:\n"
        "  • Learned responses\n"
        "  • Conversation history\n"
        "  • Stored memories\n"
        "  • User information\n\n"
        "Built-in training data will be reloaded.\n\n"
        "This action CANNOT be undone!\n\n"
        "Are you sure you want to continue?");
    
    gtk_window_set_title(GTK_WINDOW(confirm_dialog), "Confirm Database Reset");
    
    gint response = gtk_dialog_run(GTK_DIALOG(confirm_dialog));
    gtk_widget_destroy(confirm_dialog);
    
    if (response != GTK_RESPONSE_YES) {
        return; // User cancelled
    }
    
    // Close the database connection
    if (g_db) {
        sqlite3_close(g_db);
        g_db = NULL;
        printf("\n🔒 Database connection closed.\n");
    }
    
    // Delete the database file
    printf("🗑️  Deleting evialin_memory.db...\n");
    if (remove("evialin_memory.db") == 0) {
        printf("✅ Database file deleted successfully.\n");
    } else {
        printf("⚠️  Could not delete database file (may not exist).\n");
    }
    
    // Recreate fresh database
    printf("🔨 Creating fresh database...\n");
    init_database();
    
    // Reload training data
    printf("📚 Reloading training data...\n");
    
    // Reset the static flag in init_training_data
    // We need to call the training loader directly
    int hardcoded_count = 0;
    int file_count = 0;
    
    // Load hardcoded training data
    for (int i = 0; training_dataset[i].pattern; i++) {
        char norm_pattern[256];
        strncpy(norm_pattern, training_dataset[i].pattern, sizeof(norm_pattern)-1);
        norm_pattern[sizeof(norm_pattern)-1] = '\0';
        normalize(norm_pattern);

        if (learned_pattern_exists(norm_pattern))
            continue;

        save_learned_response(norm_pattern, training_dataset[i].response);
        hardcoded_count++;
    }
    
    // Load from file
    file_count = load_training_from_file("evialin_training.txt");
    
    // Initialize jokes
    if (!memory_exists("joke_1")) {
        save_memory("joke_1", "Why did the neural network cross the road? To get to the other sigmoid!");
    }
    if (!memory_exists("joke_2")) { 
        save_memory("joke_2", "How many AI programmers does it take to change a lightbulb? None, they just train the bulb to glow.");
    }
    if (!memory_exists("joke_3")) {
        save_memory("joke_3", "I asked my AI for a joke. It said: 'Your internet connection'.");
    }
    if (!memory_exists("joke_4")) {
        save_memory("joke_4", "Why was the computer cold? It left its Windows open!");
    }
    
    // Reset default user name
    save_memory("user_name", "Stranger");
    
    // Clear chat window
    gtk_text_buffer_set_text(gui->chat_buffer, "", -1);
    
    // Add welcome message
    add_message(gui, 
        "🔄 Memory Reset Complete!\n\n"
        "Hello! I'm Evialin with a fresh memory!\n\n"
        "I can:\n"
        "• Solve math problems (e.g., 'What is 25 plus 17?')\n"
        "• Remember your name (say 'My name is...')\n"
        "• Store facts (say 'Remember that...')\n"
        "• Have conversations!\n\n"
        "What's your name?", 0);
    
    // Clear terminal
    gtk_text_buffer_set_text(gui->terminal_buffer, "", -1);
    add_terminal_output(gui, "Evialin Terminal Mode v2.0", 0);
    add_terminal_output(gui, "Memory database has been reset", 0);
    add_terminal_output(gui, "Type 'help' for available commands", 0);
    add_terminal_output(gui, "", 0);
    
    // Show success dialog
    char msg[512];
    snprintf(msg, 512,
        "✅ Database Reset Complete!\n\n"
        "Fresh database created with:\n"
        "  • Built-in patterns: %d\n"
        "  • File patterns: %d\n"
        "  • Total: %d entries\n\n"
        "All previous data has been cleared.\n"
        "Evialin is ready for a fresh start!",
        hardcoded_count, file_count, hardcoded_count + file_count);
    
    GtkWidget *success_dialog = gtk_message_dialog_new(
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_INFO,
        GTK_BUTTONS_OK,
        "%s", msg);
    gtk_dialog_run(GTK_DIALOG(success_dialog));
    gtk_widget_destroy(success_dialog);
    
    printf("\n✅ Database reset complete!\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
}

// Function to clear the chat window
void clear_chat_window(GUI *gui) {
    // Clear the chat buffer
    gtk_text_buffer_set_text(gui->chat_buffer, "", -1);
    
    // Add a fresh message
    add_message(gui, 
        "Chat cleared! 🧹\n\n", 0);
    
    printf("💬 Chat window cleared\n");
}

// Menu callback to clear chat
void on_menu_clear_chat(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    clear_chat_window(gui);
}

// User Profile GUI Options 
void on_edit_profile_clicked(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;

    GtkWidget *dialog = gtk_dialog_new_with_buttons("Profile & Neural Theme Settings",
                                                    GTK_WINDOW(gui->window),
                                                    GTK_DIALOG_MODAL,
                                                    "_Apply & Save", GTK_RESPONSE_ACCEPT,
                                                    "_Cancel", GTK_RESPONSE_REJECT,
                                                    NULL);

    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_container_set_border_width(GTK_CONTAINER(vbox), 20);
    gtk_container_add(GTK_CONTAINER(content_area), vbox);

    // --- User Name ---
    gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Evialin calls you:"), FALSE, FALSE, 0);
    GtkWidget *entry_name = gtk_entry_new();
    char *name = get_memory("user_name"); 
    if (name) { gtk_entry_set_text(GTK_ENTRY(entry_name), name); free(name); }
    gtk_box_pack_start(GTK_BOX(vbox), entry_name, FALSE, FALSE, 0);

    // --- User Job ---
    gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Your Primary Task:"), FALSE, FALSE, 0);
    GtkWidget *entry_job = gtk_entry_new();
    char *job = get_memory("user_job");
    if (job) { gtk_entry_set_text(GTK_ENTRY(entry_job), job); free(job); }
    gtk_box_pack_start(GTK_BOX(vbox), entry_job, FALSE, FALSE, 0);

    // --- Theme Selector ---
    gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Neural Interface Theme:"), FALSE, FALSE, 0);
    GtkWidget *combo_theme = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo_theme), "Deep Purple (Default)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo_theme), "High Contrast (White)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo_theme), "Midnight Gold"); 
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo_theme), "Cyber Matrix (Green)");
    gtk_combo_box_set_active(GTK_COMBO_BOX(combo_theme), 0);
    gtk_box_pack_start(GTK_BOX(vbox), combo_theme, FALSE, FALSE, 0);

    gtk_widget_show_all(dialog);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        save_memory("user_name", gtk_entry_get_text(GTK_ENTRY(entry_name)));
        save_memory("user_job", gtk_entry_get_text(GTK_ENTRY(entry_job)));

        gchar *selected = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(combo_theme));
        if (selected) {
            GtkCssProvider *provider = gtk_css_provider_new();
            char css_buffer[3500];

            const char *win_bg, *text_color, *menu_bg, *btn_bg, *btn_text;

            if (strcmp(selected, "High Contrast (White)") == 0) {
                win_bg = "#000000";     // Deepest black
                text_color = "#ffffff"; // Pure white text
                menu_bg = "#1a1a1a";    // Slightly lighter black for depth
                btn_bg = "#005fcc";     // Strong blue (accessible)
                btn_text = "#ffffff";   // White text on blue
                 
            } else if (strcmp(selected, "Midnight Gold") == 0) {
                // High Contrast but "Elegant": Dark Navy and Sharp Gold
                win_bg = "#0a0a0f";     // Very dark navy
                text_color = "#fdfdfd"; // Off-white
                menu_bg = "#161625";    // Dark slate
                btn_bg = "#fbbf24";     // Vivid Amber/Gold
                btn_text = "#E5E4E3";   // Platinum text for maximum legibility on gold
                
            } else if (strcmp(selected, "Cyber Matrix (Green)") == 0) {
                win_bg = "#001a00"; text_color = "#00ff00"; menu_bg = "#002200";
                btn_bg = "#004400"; btn_text = "#00ff00";
            } else {
                // Default Purple
                win_bg = "#1a1a2e"; text_color = "#ffffff"; menu_bg = "#252545";
                btn_bg = "#9c4aed"; btn_text = "#ffffff"; 
            }
            const char *hover_bg = btn_bg; 
            // Refined CSS: specifically avoids targeting labels inside buttons
            snprintf(css_buffer, sizeof(css_buffer),
                "window, dialog { background-color: %s; }"
                "window > box > label, dialog label { color: %s; }" /* Targets only top-level labels */
                "entry { background-color: alpha(%s, 0.1); color: %s; border: 1px solid %s; }"
                "menubar, menu, menuitem { background-color: %s; color: %s; }"
                "menuitem:hover { background-color: %s; border-radius: 4px; border: none; }"
                "button { background-color: %s; border-radius: 4px; border: none; }"
                "button:hover { background-color: #000000; }"
                "button label { color: %s; font-weight: bold; padding: 5px; }", /* Explicit button text control */
                win_bg, text_color, text_color, text_color, text_color, 
                menu_bg, text_color, hover_bg, btn_bg, btn_text
            );

            gtk_css_provider_load_from_data(provider, css_buffer, -1, NULL);
            gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),
                                                      GTK_STYLE_PROVIDER(provider),
                                                      GTK_STYLE_PROVIDER_PRIORITY_USER);
            g_object_unref(provider);
            g_free(selected);
        }
        gtk_label_set_text(GTK_LABEL(gui->status_label), "System: Interface recalibrated.");
    }
    gtk_widget_destroy(dialog);
}


/*  Training data Teacher mode idea 
 *  Section for AI database learning  of traing data. 
 * */

// Function to append data to the text file
void save_training_data_to_file(const char *question, const char *answer) {
    FILE *fp = fopen("evialin_training.txt", "a");
    if (fp == NULL) return;
    
    // Format required for your loader: Q: [Question] | A: [Answer]
    fprintf(fp, "%s | %s | information\n", question, answer);
    fclose(fp);
}



// Callback for the "Save" button
void on_training_save_clicked(GtkButton *button, gpointer data) {
    TrainingData *td = (TrainingData*)data;
    const char *q = gtk_entry_get_text(GTK_ENTRY(td->q_entry));
    const char *a = gtk_entry_get_text(GTK_ENTRY(td->a_entry));

    if (strlen(q) > 0 && strlen(a) > 0) {
        save_training_data_to_file(q, a);
        gtk_label_set_text(GTK_LABEL(td->gui->status_label), "● Training Saved to evialin_training.txt");
        
        // Clear for next entry
        gtk_entry_set_text(GTK_ENTRY(td->q_entry), "");
        gtk_entry_set_text(GTK_ENTRY(td->a_entry), "");
    }
}

// Function to spawn the editor window
void create_training_input_window(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI*)data;

    // RESTRICTION: Only open if Teacher Mode is ON
    if (!gui->is_teacher_mode) {
        gtk_label_set_text(GTK_LABEL(gui->status_label), "⚠️ Access Denied: Enable Teacher Mode first.");
        return;
    }

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Neural Training Editor");
    gtk_container_set_border_width(GTK_CONTAINER(window), 15);
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 200);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    GtkWidget *q_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(q_entry), "User Question...");
    GtkWidget *a_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(a_entry), "Bot Response...");
    
    GtkWidget *save_btn = gtk_button_new_with_label("Add Training Line");

    TrainingData *td = g_new0(TrainingData, 1);
    td->q_entry = q_entry;
    td->a_entry = a_entry;
    td->gui = gui;

    g_signal_connect(save_btn, "clicked", G_CALLBACK(on_training_save_clicked), td);
    g_signal_connect_swapped(window, "destroy", G_CALLBACK(g_free), td);

    gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Question:"), 0, 0, 0);
    gtk_box_pack_start(GTK_BOX(vbox), q_entry, 0, 0, 0);
    gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Answer:"), 0, 0, 0);
    gtk_box_pack_start(GTK_BOX(vbox), a_entry, 0, 0, 0);
    gtk_box_pack_start(GTK_BOX(vbox), save_btn, 0, 0, 10);

    gtk_container_add(GTK_CONTAINER(window), vbox);
    gtk_widget_show_all(window);
}
void on_menu_toggle_teacher(GtkMenuItem *menuitem, gpointer data) {
    GUI *gui = (GUI*)data;
    gui->is_teacher_mode = gtk_check_menu_item_get_active(GTK_CHECK_MENU_ITEM(menuitem));
    gtk_label_set_text(GTK_LABEL(gui->status_label), 
        gui->is_teacher_mode ? "● Teacher Mode: ON" : "● Teacher Mode: OFF");
        
    if (gui->is_teacher_mode) {
        gtk_label_set_text(GTK_LABEL(gui->status_label), "Teacher Mode: ENABLED");
        // Show the editor menu item
        gtk_widget_show(gui->menu_editor_item);
        set_mood(gui, "excited");
    } else {
        gtk_label_set_text(GTK_LABEL(gui->status_label), "Teacher Mode: DISABLED");
       // Hide the editor menu item
        gtk_widget_hide(gui->menu_editor_item);
        set_mood(gui, "idle");
    }
}

/* END Teacher learning section of code. 
 */



// Keyboard shortcut handler
gboolean on_chat_key_press(GtkWidget *widget, GdkEventKey *event, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Ctrl+L - Clear chat window
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_l) {
		   gtk_label_set_text(GTK_LABEL(gui->status_label), "● Clearing...");
        
        // Force GUI update
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        clear_chat_window(gui);
        return TRUE; // Event handled
    }
    
    // Ctrl+S - Show Learning Statistics
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_s) {
		 gtk_label_set_text(GTK_LABEL(gui->status_label), "● Loading stats...");        
        // Force GUI update
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        on_menu_learning_stats(NULL, gui);
        // Force GUI to update
         gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready.");
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        return TRUE; // Event handled
    }
   
   if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_b) {
    if (gui->is_teacher_mode) {
        create_training_input_window(NULL, gui);
    } else {
        // Do nothing or show a subtle message
        gtk_label_set_text(GTK_LABEL(gui->status_label), "⚠️ Editor requires Teacher Mode.");
    }
    return TRUE;
    }

    // Ctrl+T - Show Training Stats (text version)
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_t) {
		 gtk_label_set_text(GTK_LABEL(gui->status_label), "● Gathering training data...");        
        // Force GUI update
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        char *stats = get_training_stats();
        GtkWidget *dialog = gtk_message_dialog_new(
            GTK_WINDOW(gui->window),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            GTK_BUTTONS_OK,
            "%s", stats);
        gtk_window_set_title(GTK_WINDOW(dialog), "Training Statistics");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        free(stats);
        gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready");
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }        
        return TRUE; // Event handled
    }
    
    // Ctrl+N - Show Neural Network Stats
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_n) {
		 gtk_label_set_text(GTK_LABEL(gui->status_label), "● Refreshing neural stats...");
        set_mood(gui, "calculating");        
        // Force GUI update
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }        
        update_neural_stats(gui);
        GtkWidget *dialog = gtk_message_dialog_new(
            GTK_WINDOW(gui->window),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            GTK_BUTTONS_OK,
            "Neural Network Stats Refreshed!\n\n"
            "Check the left panel for updated statistics.");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
         gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready");
              while (gtk_events_pending()) {
            gtk_main_iteration();
        }  
        set_mood(gui, "happy");
        return TRUE; // Event handled
    }
        // Ctrl+R - Reload Training Data
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_r) {
        // Update status immediately
        gtk_label_set_text(GTK_LABEL(gui->status_label), "● Analyzing... Stand by");
        set_mood(gui, "calculating");
        
        // Force GUI to update
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }
        
        // Call the reload function (it will reset status when done)
        on_menu_reload_training(NULL, gui);
        gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready..");
             while (gtk_events_pending()) {
            gtk_main_iteration();
        }  
        return TRUE; // Event handled
    }
       gtk_label_set_text(GTK_LABEL(gui->status_label), "● Ready");
             while (gtk_events_pending()) {
            gtk_main_iteration();
        }  
    return FALSE; // Let other handlers process the event
}




//=======================================================================
// Learning statistics functions 
//=======================================================================
// Cleanup callback function (add this before on_menu_learning_stats)

void free_stats_data(gpointer data) {
    if (data) {
        free(data);
     }
}

gboolean on_chat_key_press_extended(GtkWidget *widget, GdkEventKey *event, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Check for Ctrl+L (Clear chat)
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_l) {
        clear_chat_window(gui);
        return TRUE;
    }
    
    // Check for Ctrl+S (Learning Statistics)
    if ((event->state & GDK_CONTROL_MASK) && event->keyval == GDK_KEY_s) {
        on_menu_learning_stats(NULL, gui);
        return TRUE;
    }
    
    return FALSE;
}

LearningStats get_learning_statistics(void) {
    LearningStats stats = {0, 0, 0, 0, 0};
    
    if (!g_db) return stats;
    
    // Count total learned responses
    const char *sql1 = "SELECT COUNT(*) FROM memory WHERE key LIKE 'learned_%';";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(g_db, sql1, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.total_learned = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Count jokes
    const char *sql2 = "SELECT COUNT(*) FROM memory WHERE key LIKE 'joke_%';";
    if (sqlite3_prepare_v2(g_db, sql2, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.jokes_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Count facts
    const char *sql3 = "SELECT COUNT(*) FROM memory WHERE key LIKE 'fact_%';";
    if (sqlite3_prepare_v2(g_db, sql3, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.facts_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Count builtin training data
    stats.builtin_count = 0;
    for (int i = 0; training_dataset[i].pattern != NULL; i++) {
        stats.builtin_count++;
    }
    
    // User taught = total learned - builtin
    stats.user_taught = stats.total_learned - stats.builtin_count;
    if (stats.user_taught < 0) stats.user_taught = 0;
    
    return stats;
}

// Function to create a visual bar chart using drawing area
gboolean draw_stats_graph(GtkWidget *widget, cairo_t *cr, gpointer data) {
    LearningStats *stats = (LearningStats*)data;
    
    // Get widget dimensions
    GtkAllocation allocation;
    gtk_widget_get_allocation(widget, &allocation);
    int width = allocation.width;
    int height = allocation.height;
    
    // Background
    cairo_set_source_rgb(cr, 0.95, 0.95, 0.95);
    cairo_rectangle(cr, 0, 0, width, height);
    cairo_fill(cr);
    
    // Graph parameters
    int bar_width = 60;
    int spacing = 80;
    int start_x = 60;
    int max_height = height - 140;
    
    // Find max value for scaling
    int max_value = stats->total_learned;
    if (stats->builtin_count > max_value) max_value = stats->builtin_count;
    if (stats->user_taught > max_value) max_value = stats->user_taught;
    if (stats->jokes_count > max_value) max_value = stats->jokes_count;
    if (stats->facts_count > max_value) max_value = stats->facts_count;
    if (max_value == 0) max_value = 1; // Prevent division by zero
    
    // Data for bars
    struct {
        const char *label;
        int value;
        double r, g, b;
    } bars[] = {
        {"Total\nLearned", stats->total_learned, 0.29, 0.57, 0.95},  // Blue
        {"Built-in", stats->builtin_count, 0.45, 0.23, 0.79},         // Purple
        {"User\nTaught", stats->user_taught, 0.13, 0.73, 0.51},       // Green
        {"Jokes", stats->jokes_count, 1.0, 0.6, 0.0},                 // Orange
        {"Facts", stats->facts_count, 0.91, 0.30, 0.24}               // Red
    };
    
    // Draw bars
    for (int i = 0; i < 5; i++) {
        int x = start_x + (i * spacing);
        int bar_height = (int)((double)bars[i].value / max_value * max_height);
        int y = height - 60 - bar_height;
        
        // Draw bar with gradient
        cairo_pattern_t *gradient = cairo_pattern_create_linear(x, y, x, y + bar_height);
        cairo_pattern_add_color_stop_rgb(gradient, 0, bars[i].r, bars[i].g, bars[i].b);
        cairo_pattern_add_color_stop_rgb(gradient, 1, 
            bars[i].r * 0.7, bars[i].g * 0.7, bars[i].b * 0.7);
        cairo_set_source(cr, gradient);
        cairo_rectangle(cr, x, y, bar_width, bar_height);
        cairo_fill(cr);
        cairo_pattern_destroy(gradient);
        
        // Draw bar outline
        cairo_set_source_rgb(cr, 0.3, 0.3, 0.3);
        cairo_set_line_width(cr, 2);
        cairo_rectangle(cr, x, y, bar_width, bar_height);
        cairo_stroke(cr);
        
        // Draw value on top of bar
        cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 16);
        
        char value_str[32];
        snprintf(value_str, 32, "%d", bars[i].value);
        
        cairo_text_extents_t extents;
        cairo_text_extents(cr, value_str, &extents);
        cairo_move_to(cr, x + (bar_width - extents.width) / 2, y - 5);
        cairo_show_text(cr, value_str);
        
        // Draw label below bar
        cairo_set_font_size(cr, 12);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
        
        // Handle multi-line labels
        char label_copy[64];
        strncpy(label_copy, bars[i].label, 63);
        label_copy[63] = '\0';
        
        char *line1 = strtok(label_copy, "\n");
        char *line2 = strtok(NULL, "\n");
        
        if (line1) {
            cairo_text_extents(cr, line1, &extents);
            cairo_move_to(cr, x + (bar_width - extents.width) / 2, height - 40);
            cairo_show_text(cr, line1);
        }
        
        if (line2) {
            cairo_text_extents(cr, line2, &extents);
            cairo_move_to(cr, x + (bar_width - extents.width) / 2, height - 25);
            cairo_show_text(cr, line2);
        }
    }
    
    // Draw title
    cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 18);
    cairo_move_to(cr, 20, 25);
    cairo_show_text(cr, "Learning Statistics");
    
    // Draw y-axis scale
    cairo_set_font_size(cr, 10);
    for (int i = 0; i <= 4; i++) {
        int value = (max_value * i) / 4;
        int y = height - 60 - (i * max_height / 4);
        
        char scale_str[16];
        snprintf(scale_str, 16, "%d", value);
        
        cairo_move_to(cr, 10, y + 4);
        cairo_show_text(cr, scale_str);
        
        // Draw grid line
        cairo_set_source_rgba(cr, 0.7, 0.7, 0.7, 0.5);
        cairo_set_line_width(cr, 1);
        cairo_move_to(cr, 45, y);
        cairo_line_to(cr, width - 20, y);
        cairo_stroke(cr);
        
        cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
    }
    
    return FALSE;
}

// Dialog callback for learning statistics
void on_menu_learning_stats(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Get statistics
    LearningStats stats = get_learning_statistics();
    
    // Create dialog
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "Learning Statistics",
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "Close", GTK_RESPONSE_CLOSE,
        NULL
    );
    
    gtk_window_set_default_size(GTK_WINDOW(dialog), 600, 450);
    
    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content, 15);
    gtk_widget_set_margin_end(content, 15);
    gtk_widget_set_margin_top(content, 15);
    gtk_widget_set_margin_bottom(content, 15);
    
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    
    // Drawing area for graph
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, 550, 300);
    
    // Allocate stats on heap so it persists for the draw callback
    LearningStats *stats_ptr = malloc(sizeof(LearningStats));
    *stats_ptr = stats;
    
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_stats_graph), stats_ptr);
    g_signal_connect_swapped(dialog, "destroy", G_CALLBACK(free_stats_data), stats_ptr);
    gtk_box_pack_start(GTK_BOX(vbox), drawing_area, TRUE, TRUE, 0);
    
    // Text summary
    GtkWidget *summary = gtk_label_new(NULL);
    char summary_text[512];
    snprintf(summary_text, 512,
        "<span font='11'>"
        "<b>Summary:</b>\n"
        "Total Learned Responses: <b>%d</b>\n"
        "Built-in Training: <b>%d</b> patterns\n"
        "User Taught: <b>%d</b> custom responses\n"
        "Jokes Stored: <b>%d</b>\n"
        "Facts Remembered: <b>%d</b>\n\n"
        "<i>💡 Teach me more by saying: \"Remember that when someone says X, respond with Y\"</i>"
        "</span>",
        stats.total_learned,
        stats.builtin_count,
        stats.user_taught,
        stats.jokes_count,
        stats.facts_count
    );
    gtk_label_set_markup(GTK_LABEL(summary), summary_text);
    gtk_label_set_line_wrap(GTK_LABEL(summary), TRUE);
    gtk_label_set_xalign(GTK_LABEL(summary), 0.0);
    gtk_box_pack_start(GTK_BOX(vbox), summary, FALSE, FALSE, 0);
    
    gtk_container_add(GTK_CONTAINER(content), vbox);
    gtk_widget_show_all(dialog);
    
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}


void on_menu_about(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    if (!gui || !gui->window) return;
    
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "About Evialin AI",
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "Close", GTK_RESPONSE_CLOSE,
        NULL
    );
    
    gtk_window_set_default_size(GTK_WINDOW(dialog), 550, 600);
    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_container_set_border_width(GTK_CONTAINER(main_vbox), 15);

    // HEADER (Static Label)
    GtkWidget *header = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(header),
        "<span font='18' weight='bold'>🤖 Evialin Version 2.14</span>\n"
        "<span font='10' style='italic'>Enhanced Virtual Intelligence And Learning Interface Network</span>");
    gtk_box_pack_start(GTK_BOX(main_vbox), header, FALSE, FALSE, 0);
    
    gtk_box_pack_start(GTK_BOX(main_vbox), gtk_separator_new(GTK_ORIENTATION_HORIZONTAL), FALSE, FALSE, 5);

    // SCROLLING AREA (For massive updates/architecture text)
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(scroll, -1, 250); // Give the scroll area a healthy 250px height

    GtkWidget *scroll_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(scroll_label),
        "<span font='11'>"
        "<b>Latest Updates:</b>\n"
        "• Window color profiles\n"
        "• Teacher / Student toggle modes of AI emulation.\n"
        "• Real-time command line debugging features.\n\n"
        "<b>1. Neural Network &amp; AI Core</b>\n"
        "Hybrid Architecture: Uses a feed-forward neural network for main logic and a Recurrent Neural Network (RNN) sub-function for character-level intuition and sequence mapping.\n\n"
        "Intent Classification: (Teacher mode) Features a Tiny Neural Intent Classifier that categorizes user inputs into intents like Greetings, Questions, Commands, Insults, and Gratitude.\n\n"
        "Dynamic Learning: Includes backpropagation with Sigmoid and ReLU activation functions, allowing the network to adjust weights and biases based on training steps.\n\n"
        "Real-time Visualization: Offers a Neural Activity window that draws active connections (synapses) as animated dashed lines.\n\n"
        "<b>2. Conversational &amp; Memory Features</b>\n"
        "Persistent SQLite Database: Stores long-term data in evialin_memory.db, including conversations, neural states, jokes, and personal thoughts.\n\n"
        "Trie-based Autocomplete: Implements a Trie data structure to find the closest match for user input, acting as a greedy completion engine.\n"
        "</span>");
    
    gtk_label_set_xalign(GTK_LABEL(scroll_label), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(scroll_label), TRUE);
    gtk_container_add(GTK_CONTAINER(scroll), scroll_label);
    gtk_box_pack_start(GTK_BOX(main_vbox), scroll, TRUE, TRUE, 0);

    // FEATURES & SPECS (Static Label - easy to read at a glance)
    GtkWidget *specs = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(specs),
        "\n<b>System Highlights:</b>\n"
        "🐾 Orange Cute Cat Assistant Personality\n"
        "🧠 Scalable 15 to 150-Neuron Architecture\n"
        "💾 Self-Cleaning SQLite Memory\n"
        "🧮 Natural Language Math Solver Cluster");
    gtk_label_set_xalign(GTK_LABEL(specs), 0.0);
    gtk_box_pack_start(GTK_BOX(main_vbox), specs, FALSE, FALSE, 5);

    // FOOTER (Static Label)
    GtkWidget *footer = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(footer),
        "<span font='9' foreground='#666666'>"
        "[FREE AS IN FREEDOM] Software 2025 By Johnny B Stroud\n"
        "Built with C, GTK+ 3.0, and SQLite3</span>");
    gtk_box_pack_start(GTK_BOX(main_vbox), footer, FALSE, FALSE, 10);

    gtk_container_add(GTK_CONTAINER(content), main_vbox);
    gtk_widget_show_all(dialog);
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}


void on_menu_shortcuts(GtkWidget *w, gpointer data) {
    GUI *gui = (GUI*)data;
    
    // Safety check
    if (!gui || !gui->window) {
        fprintf(stderr, "Error: GUI pointer is NULL in on_menu_shortcuts\n");
        return;
    }
    
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "Keyboard Shortcuts",
        GTK_WINDOW(gui->window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "Close", GTK_RESPONSE_CLOSE,
        NULL
    );
    
    gtk_window_set_default_size(GTK_WINDOW(dialog), 530, 420);
    gtk_window_set_resizable(GTK_WINDOW(dialog), FALSE);
    
    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content, 20);
    gtk_widget_set_margin_end(content, 20);
    gtk_widget_set_margin_top(content, 20);
    gtk_widget_set_margin_bottom(content, 20);
    
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    
    // Title
    GtkWidget *title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(title),
        "<span font='16' weight='bold'>⌨️ Keyboard Shortcuts</span>");
    gtk_widget_set_halign(title, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(vbox), title, FALSE, FALSE, 0);
    
    GtkWidget *text_actions = gtk_label_new(NULL); 
    gtk_label_set_markup(GTK_LABEL(text_actions), 
       "<span font='11'><b>Teaching &amp; Learning Text Commands</b>\n"
        "• <i>My name is [Name]</i>: Teach Evialin your name\n"
        "• <i>Remember that when someone says [Q] the answer is [A]</i>\n"
        "• <i>Remember that a joke is [Joke]</i>: Add a joke to memory\n"
        "• <i>Forget when I said [Phrase]</i>: Delete a specific fact\n"
        "• <i>Forget last</i>: Undo the most recent thing learned\n"
        "• <i>Forget everything learned</i>: Wipe all custom facts\n"
        "• <i>What have you learned?</i>: List all learned responses\n\n"
        "</span>");
    gtk_label_set_xalign(GTK_LABEL(text_actions), 0.0);  
    gtk_label_set_line_wrap(GTK_LABEL(text_actions), TRUE);
    gtk_label_set_selectable(GTK_LABEL(text_actions), TRUE);    
    gtk_box_pack_start(GTK_BOX(vbox), text_actions, TRUE, TRUE, 0);
    
    // Shortcuts text
    GtkWidget *shortcuts = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(shortcuts),
        "<span font='11'> <b>Chat Window</b>\n"
        "  <tt>Enter</tt>              Send message\n"
        "  <tt>Ctrl+L</tt>             Clear chat window\n"
        "  <tt>Ctrl+I</tt>             Preformance graph window\n"
        "  <tt>Ctrl+S</tt>             Show learning statistics\n"
        "  <tt>Ctrl+T</tt>             Show training stats\n"
        "  <tt>Ctrl+N</tt>             Refresh neural network stats\n"
        "  <tt>Ctrl+R</tt>             Refresh training data\n"
        "<b>Teacher mode enabled shortcuts</b>\n"
        "  <tt>Ctrl+B</tt>             Open Training editor\n\n"
        "<b>Terminal</b>\n"
        "  <tt>Enter</tt>              Execute command\n"
        "  <tt>clear</tt>              Clear terminal\n"
        "  <tt>help</tt>               Show terminal commands\n\n"        
        "<b>Navigation</b>\n"
        "  <tt>Tab</tt>                Switch between input fields\n\n"
        "<b>Tips</b>\n"
        "  • Use natural language for math problems\n"
        "  • Teach Evialin with 'Remember that when...'\n"
        "  • Check the Neural Activity panel for live stats\n"
        "</span>");
    gtk_label_set_xalign(GTK_LABEL(shortcuts), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(shortcuts), TRUE);
    gtk_label_set_selectable(GTK_LABEL(shortcuts), TRUE);    
    gtk_box_pack_start(GTK_BOX(vbox), shortcuts, TRUE, TRUE, 0);
    
    gtk_container_add(GTK_CONTAINER(content), vbox);
    gtk_widget_show_all(dialog);
    
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}


static void on_window_destroy(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI*)data;
    if (!gui) return;

    gui->is_alive = FALSE;
    if (gui->refresh_timeout_id) {
        g_source_remove(gui->refresh_timeout_id);
        gui->refresh_timeout_id = 0;
    }

    printf("Exiting application... Saving neural network state.\n");
    save_full_network(&g_network); 
    
    if (g_db) {
        sqlite3_close(g_db);
        printf("Database closed safely.\n");
    }

    gtk_main_quit();
}

/*
static gboolean refresh_training_idle(gpointer data)
{
    GUI *gui = data;

    if (!gui->is_alive)
        return FALSE;

       gtk_label_set_text(GTK_LABEL(gui->status_label),
                           "Initializing training data...");
        while (gtk_events_pending())
            gtk_main_iteration();
            
    // actual refresh work
    init_training_data();

    gtk_label_set_text(GTK_LABEL(gui->status_label),
                       "Training data updated");

    return G_SOURCE_REMOVE;  // run once
// Not sure if I can use this ?     
// Example of how it should be called elsewhere in your code
// g_idle_add((GSourceFunc)refresh_training_idle, gui);
}
*/ 

// ===========================================================
// Neural network black box maping Visualization tool
// ===========================================================
static gboolean on_draw_neural_network(GtkWidget *widget, cairo_t *cr, gpointer data) {
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
    double time_offset = (double)(g_get_monotonic_time() / 50000.0);
    double center_divider = width * 0.40; // Split point

    // Dark Technical Background
    cairo_set_source_rgb(cr, 0.02, 0.02, 0.05);
    cairo_paint(cr);

    // Draw Vertical Divider Line
    cairo_set_source_rgba(cr, 0.48, 0.22, 0.93, 0.3); // Purple accent
    cairo_set_line_width(cr, 2.0);
    cairo_move_to(cr, center_divider, 20);
    cairo_line_to(cr, center_divider, height - 40);
    cairo_stroke(cr);

    // ==========================================================
    // LEFT SIDE: NEW RECURRENT LOGIC & HEATMAP
    // ==========================================================
    int start_x = 20;
    int start_y = 60;
    int cell_size = (center_divider / 18); // Scale to fit left side
    int padding = 2;

    cairo_set_source_rgb(cr, 0.8, 0.4, 1.0);
    cairo_set_font_size(cr, 11);
    cairo_move_to(cr, start_x, start_y - 15);
    cairo_show_text(cr, "[ RNN INTUITION ENGINE ]");

    // Draw RNN Hidden State Heatmap
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        int row = i / 10; 
        int col = i % 10;
        double act_norm = (g_rnn_visual_hidden[i] + 1.0) / 2.0;

        // Sequence activity is shown in purple pulses
        cairo_set_source_rgba(cr, 0.6, 0.3, 1.0, 0.2 + (0.8 * act_norm)); 
        cairo_rectangle(cr, start_x + (col * (cell_size + padding)), 
                        start_y + (row * (cell_size + padding)), 
                        cell_size, cell_size);
        cairo_fill(cr);
        
        if (act_norm > 0.85) {
            cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.4);
            cairo_rectangle(cr, start_x + (col * (cell_size + padding)), 
                            start_y + (row * (cell_size + padding)), 
                            cell_size, cell_size);
            cairo_stroke(cr);
        }
    }

    // RNN Intent Trigger Text
    double max_act = 0;
    int best_node = -1;
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        if(g_network.output[i] > max_act) { max_act = g_network.output[i]; best_node = i; }
    }
    if (max_act > 0.50) {
        char trigger_buf[64];
        sprintf(trigger_buf, "ACTIVE SIGNAL: #%d (%.0f%%)", best_node, max_act * 100);
        cairo_set_source_rgb(cr, (max_act > 0.85) ? 0.0 : 1.0, (max_act > 0.85) ? 1.0 : 0.8, 0.0);
        cairo_move_to(cr, start_x, start_y + (16 * (cell_size + padding)) + 20);
        cairo_show_text(cr, trigger_buf);
    }

    // ==========================================================
    // RIGHT SIDE: OLD LAYERED ARCHITECTURE
    // ==========================================================
    // Adjusting layer X positions to the right of the divider
    double layer_x[] = { center_divider + (width - center_divider) * 0.15, 
                         center_divider + (width - center_divider) * 0.50, 
                         center_divider + (width - center_divider) * 0.85 };
    int v_size = 12;

    // Draw Connections (Weights)
    for (int l = 0; l < 2; l++) {
        for (int i = 0; i < v_size; i++) {
            for (int j = 0; j < v_size; j++) {
                double activation = (l == 0) ? g_network.hidden[j] : g_network.output[j];
                double weight = (l == 0) ? g_network.weights_ih[i][j] : g_network.weights_ho[i][j];
                
                if (activation > 0.5) {
                    cairo_set_source_rgba(cr, 0.0, 1.0, 0.5, 0.3);
                    double dashes[] = {2.0, 8.0};
                    cairo_set_dash(cr, dashes, 2, -time_offset);
                    cairo_set_line_width(cr, fabs(weight) * 2.0 + 0.5); 
                } else {
                    cairo_set_source_rgba(cr, 0.3, 0.3, 0.4, 0.05);
                    cairo_set_dash(cr, NULL, 0, 0);
                    cairo_set_line_width(cr, 0.5);
                }
                cairo_move_to(cr, layer_x[l], 60 + (i * (height-120)/v_size));
                cairo_line_to(cr, layer_x[l+1], 60 + (j * (height-120)/v_size));
                cairo_stroke(cr);
            }
        }
    }

    // Draw Neurons and Layer Labels
    for (int l = 0; l < 3; l++) {
        cairo_set_source_rgb(cr, 0.6, 0.6, 0.8);
        cairo_set_font_size(cr, 10);
        const char* layer_labels[] = {"INPUT", "HIDDEN", "OUTPUT"};
        cairo_move_to(cr, layer_x[l] - 20, 40);
        cairo_show_text(cr, layer_labels[l]);

        for (int i = 0; i < v_size; i++) {
            double x = layer_x[l];
            double y = 60 + (i * (height-120)/v_size);
            double val = (l == 0) ? g_network.last_input[i] : (l == 1) ? g_network.hidden[i] : g_network.output[i];

            cairo_set_source_rgb(cr, (val > 0.5) ? 0.2 : 0.3, (val > 0.5) ? 0.9 : 0.3, (val > 0.5) ? 0.3 : 0.35);
            cairo_arc(cr, x, y, 9, 0, 2 * M_PI);
            cairo_fill(cr);

            // Neuron IDs
            cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
            cairo_set_font_size(cr, 8);
            char idx_str[8];
            sprintf(idx_str, "%d", i);
            cairo_move_to(cr, x - 4, y + 3);
            cairo_show_text(cr, idx_str);
        }
    }

    // Footer Stats (Spans both)
    cairo_set_source_rgb(cr, 0.8, 0.8, 1.0);
    cairo_set_font_size(cr, 13.0); 
    char stats[256];
    sprintf(stats, "INTEGRATED SYSTEM | Confidence: %.1f%% | Param count: %d", 
            g_network.avg_confidence, 32515);
    cairo_move_to(cr, 20, height - 15);
    cairo_show_text(cr, stats);

    return FALSE;
}

void on_menu_neural_viz(GtkWidget *w, gpointer data) {
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Real-time Neural Activity");
    gtk_window_set_default_size(GTK_WINDOW(window), 600, 800);
    
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER(window), drawing_area);
    g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(on_draw_neural_network), NULL);
    g_timeout_add(10, (GSourceFunc)gtk_widget_queue_draw, drawing_area);
    gtk_widget_show_all(window);
}
// Background task to trigger "thoughts"
static gboolean trigger_random_thought(gpointer data) {
    GUI *gui = (GUI*)data;
    if (!gui || !gui->is_alive) return FALSE;

    // Simulate Neural Activity
    double random_input[INPUT_SIZE];
    for(int i = 0; i < INPUT_SIZE; i++) {
        random_input[i] = (double)rand() / RAND_MAX;
    }
    forward_pass(&g_network, random_input);

    // NEW: Fetch from personal_responses table
    const char *sql = "SELECT response FROM personal_responses ORDER BY RANDOM() LIMIT 1;";
    sqlite3_stmt *stmt;
    char *personal_thought = NULL;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *val = (const char*)sqlite3_column_text(stmt, 0);
            if (val) {
                personal_thought = strdup(val);
            }
        }
    }
    sqlite3_finalize(stmt);

    // Display the thought if found
    if (personal_thought) {
        set_mood(gui, "thinking");
        add_message(gui, personal_thought, 0); // 0 = Bot side
        
        free(personal_thought);
        
        // Reset mood back to idle after 4 seconds
        g_timeout_add(4000, (GSourceFunc)reset_to_idle, gui);
    }

    // Reset the timer for a random interval (2 to 5 minutes)
    int next_interval = 120 + (rand() % 180); 
    g_timeout_add_seconds(next_interval, (GSourceFunc)trigger_random_thought, gui);

    return FALSE; 
}
// Callback to actually save the text from the child window to DB
void on_save_personal_response(GtkWidget *button, gpointer data) {
    GtkWidget **widgets = (GtkWidget **)data;
    GtkWidget *entry = widgets[0];
    GtkWidget *window = widgets[1];
    GUI *gui = (GUI *)widgets[2];

    const char *text = gtk_entry_get_text(GTK_ENTRY(entry));
    if (strlen(text) > 0) {
        const char *sql = "INSERT INTO personal_responses (response) VALUES (?);";
        sqlite3_stmt *stmt;
        if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, text, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            add_message(gui, "System: Personal thought logged to database.", 0);
        }
    }
    gtk_widget_destroy(window);
    free(widgets);
}

// Function to spawn the child window
void create_personal_response_window(GtkWidget *widget, gpointer data) {
    GUI *gui = (GUI *)data;
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Add Neural Thought");
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);
    gtk_window_set_default_size(GTK_WINDOW(window), 300, 100);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    GtkWidget *entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(entry), "Enter a thought for Evialin...");
    
    GtkWidget *save_btn = gtk_button_new_with_label("Inject Into Memory");

    // Pass widgets to the save callback
    GtkWidget **data_pass = malloc(sizeof(GtkWidget *) * 3);
    data_pass[0] = entry;
    data_pass[1] = window;
    data_pass[2] = (GtkWidget *)gui;

    g_signal_connect(save_btn, "clicked", G_CALLBACK(on_save_personal_response), data_pass);

    gtk_box_pack_start(GTK_BOX(vbox), entry, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), save_btn, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    gtk_widget_show_all(window);
}

// This function runs when the CPU is "idle", making it feel non-blocking
gboolean background_trial_logic(gpointer data) {
    TrialData *td = (TrialData*)data;
    
    // Generate the response (The heavy lifting)
    char *new_text = generate_conversational_response(td->gui);
    
    // Update the UI
    char *display = g_strdup_printf("(Trial): %s\n[Type + to keep, - to discard]", new_text);
    add_message(td->gui, display, 0);
    
    // Update the global state for feedback
    if (g_last_generated_phrase) free(g_last_generated_phrase);
    g_last_generated_phrase = g_strdup(new_text);
    
    // Cleanup
    g_free(display);
    free(new_text);
    free(td->lower_input);
    free(td);
    
    return FALSE; // Don't call again
}

/* Sub-function: Generates a new response based on existing memory tokens */
char* generate_conversational_response(GUI *gui) {
    if (!g_db) return g_strdup("Database not found.");

    // Load words and track which ones start sentences
    const char *sql = "SELECT response FROM personal_responses UNION SELECT value FROM memory LIMIT 800;";
    sqlite3_stmt *stmt;
    char *word_pool[4000];
    int count = 0;

    if (sqlite3_prepare_v2(g_db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW && count < 3900) {
            char *line = strdup((const char*)sqlite3_column_text(stmt, 0));
            char *token = strtok(line, " ");
            while (token && count < 3990) {
                // Filter out the structural junk
                if (strstr(token, "Q:") || strstr(token, "A:") || strstr(token, "|") || strstr(token, ":")) {
                    token = strtok(NULL, " ");
                    continue;
                }
                word_pool[count++] = strdup(token);
                token = strtok(NULL, " ");
            }
            free(line);
        }
    }
    sqlite3_finalize(stmt);

    if (count < 25) return g_strdup("My vocabulary is currently too small for new thoughts.");

    // 2. ASSEMBLY: The Chain Logic
    char buffer[2048] = "";
    
    // Pick a starting word that is capitalized (likely the start of a sentence)
    int current_idx = rand() % count;
    for(int attempt = 0; attempt < 100; attempt++) {
        if (isupper(word_pool[current_idx][0])) break;
        current_idx = rand() % count;
    }

    strcat(buffer, word_pool[current_idx]);
    strcat(buffer, " ");

    int words_generated = 1;
    int max_words = 7 + (rand() % 5);

    while (words_generated < max_words) {
        char *current_word = word_pool[current_idx];
        int potential_next_indices[200];
        int matches = 0;

        // Find logical followers
        for (int i = 0; i < count - 1; i++) {
            if (strcasecmp(word_pool[i], current_word) == 0) {
                potential_next_indices[matches++] = i + 1;
                if (matches >= 200) break;
            }
        }

        if (matches > 0) {
            current_idx = potential_next_indices[rand() % matches];
            strcat(buffer, word_pool[current_idx]);
            strcat(buffer, " ");
            words_generated++;
        } else {
            // Dead end? Don't jump to random words. Try to find a new subject.
            strcat(buffer, ". Also, "); 
            current_idx = rand() % count;
            words_generated++;
        }
    }
    
    // Clean up punctuation
    if (buffer[strlen(buffer)-1] == ' ') buffer[strlen(buffer)-1] = '\0';
    if (buffer[strlen(buffer)-1] != '.') strcat(buffer, ".");

    // Memory Cleanup
    for(int i = 0; i < count; i++) free(word_pool[i]);

    return g_strdup(buffer);
}

void apply_feedback(GUI *gui, const char *input, const char *last_generated) {
    double targets[OUTPUT_SIZE] = {0};
    
    if (strcmp(input, "+") == 0) {
        // Positive: Save to permanent database and nudge weights up
        save_learned_response("gen_success", last_generated);
        targets[110] = 1.0; // Reinforce 'Greeting/Success' neuron
        train_network(&g_network, targets, 0.08); // Higher learning rate for success
        add_message(gui, "Learned! I'll use that pattern more often.", 0);
        set_mood(gui, "happy");
    } 
    else if (strcmp(input, "-") == 0) {
        // Negative: Don't save, and nudge weights down
        targets[110] = 0.0; 
        train_network(&g_network, targets, 0.02); // Small penalty
        add_message(gui, "Discarding that pattern. I'll try a different structure.", 0);
        set_mood(gui, "sorry");
    }
    save_full_network(&g_network); // Save the new brain state
}



GUI* create_gui() {
    GUI *gui = malloc(sizeof(GUI));
    gui->handler_version = 1;
    gui->is_alive = TRUE; 
    gui->debug_mode = FALSE; 
    g_timeout_add_seconds(31, (GSourceFunc)trigger_random_thought, gui); 
    strcpy(gui->current_mood, "idle");
    gui->current_animation = NULL;
    gui->animation_iter = NULL;
    gui->animation_timeout_id = 0;
    xoshiro_srand(time(NULL));  // seed randomness ONCE
    nn_init(&g_brain);  // start tiny brain idea 
    g_timeout_add_seconds(120, random_gif_timer, gui);
    
    gui->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(gui->window), "Enhanced Virtual Intelligence And Learning Interface Network");
    gtk_window_set_default_size(GTK_WINDOW(gui->window), 1000, 800);
    
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_container_add(GTK_CONTAINER(gui->window), main_box);
    
    // LEFT PANEL
    GtkWidget *left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_size_request(left, 280, -1);
    
    GtkCssProvider *css = gtk_css_provider_new();
    gtk_css_provider_load_from_data(css,
        ".left { background: linear-gradient(180deg, #4338ca 0%, #312e81 100%); }"
        ".terminal, .terminal text, .terminal textview { "
        "    background-color: #1f2937; color: #e5e7eb; }"
        ".terminal-input { background-color: #374151; color: #10b981; }"
        "#white_double_line {"
        "   background-color: transparent;"
        "   border-top: 4px double #ffffff;" /* 'double' creates the two-line effect */
        "   min-height: 4px;"                /* Space for the double line */
        "   margin: 8px 20px;"               /* Adds space around the line */
        "}", -1, NULL);
    
    GtkStyleContext *ctx = gtk_widget_get_style_context(left);
    gtk_style_context_add_provider(ctx, GTK_STYLE_PROVIDER(css), GTK_STYLE_PROVIDER_PRIORITY_USER);
    gtk_style_context_add_class(ctx, "left");
    
    GtkWidget *avatar_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_halign(avatar_box, GTK_ALIGN_CENTER);
    gtk_widget_set_margin_top(avatar_box, 40);
    
    gui->avatar_image = gtk_image_new();
    gtk_widget_set_size_request(gui->avatar_image, 200, 200);
    gtk_box_pack_start(GTK_BOX(avatar_box), gui->avatar_image, FALSE, FALSE, 10);
   
    set_avatar_gif(gui, "idle");
   
   // we can alway add more text to the side pannel 
   // GtkWidget *name = gtk_label_new(NULL);
   // gtk_label_set_markup(GTK_LABEL(name), "<span font='9' weight='bold' foreground='white'>Evialin</span>");
   // gtk_box_pack_start(GTK_BOX(avatar_box), name, FALSE, FALSE, 2);
    
    gui->mood_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(gui->mood_label), "<span font='10' weight='bold' foreground='white'>Mood: Ready</span>");
    gtk_box_pack_start(GTK_BOX(avatar_box), gui->mood_label, FALSE, FALSE, 2);
    
    GtkWidget *memory_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(memory_label), "<span font='10' foreground='#90cdf4'>💾 Memory Active</span>");
    gtk_box_pack_start(GTK_BOX(avatar_box), memory_label, FALSE, FALSE, 2);
    
    // Pack avatar_box into left panel first
    gtk_box_pack_start(GTK_BOX(left), avatar_box, FALSE, FALSE, 0);

    // Separator line
    // Create the CSS Provider
    GtkCssProvider *cssProvider = gtk_css_provider_new();
    const char *style = 
    "#white_double_line {"
    "   background-color: transparent;" 
    "   border-top: 3px double #ffffff;" /* This creates the double white line */
    "   min-height: 4px;"                /* Gives the double line room to exist */
    "   margin: 14px 0px;"                /* Tight margins */
    "}";
    GtkWidget *stats_separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_widget_set_name(stats_separator, "white_double_line");
    // This applies the style to the app
    gtk_css_provider_load_from_data(cssProvider, style, -1, NULL);
    gtk_widget_set_margin_top(stats_separator, 5); // 20 default 
    gtk_widget_set_margin_bottom(stats_separator, 5); // 15 default 
    gtk_widget_set_margin_start(stats_separator, 5); // 20 default  
    gtk_widget_set_margin_end(stats_separator, 5); // 29 default 
    gtk_box_pack_start(GTK_BOX(left), stats_separator, FALSE, FALSE, 0);
    
    // ===============================
    // Neural Network Stats Panel
    // ===============================
    GtkWidget *stats_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(stats_box, 9); // 15
    gtk_widget_set_margin_end(stats_box, 8); // 15
    gtk_widget_set_margin_top(stats_box, 9); //10 
    
    GtkWidget *stats_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(stats_title),
        "<span font='11' weight='bold' foreground='#e0e7ff'>📊 Network Details</span>");
    gtk_widget_set_halign(stats_title, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(stats_box), stats_title, FALSE, FALSE, 0);

    // Create the stats label and save into GUI struct
    gui->stats_details = gtk_label_new("Initializing...");
    gtk_label_set_xalign(GTK_LABEL(gui->stats_details), 0.0);
    gtk_label_set_justify(GTK_LABEL(gui->stats_details), GTK_JUSTIFY_LEFT);
    gtk_widget_set_halign(gui->stats_details, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(stats_box), gui->stats_details, FALSE, FALSE, 5);
    
    // Add activity visualization panel
    GtkWidget *activity_panel = create_activity_panel(gui);
    gtk_box_pack_start(GTK_BOX(stats_box), activity_panel, FALSE, FALSE, 0);

    // Pack stats_box into left panel (below avatar_box and separator)
    gtk_box_pack_start(GTK_BOX(left), stats_box, FALSE, FALSE, 0);
    
    // Pack left panel into main_box
    gtk_box_pack_start(GTK_BOX(main_box), left, FALSE, FALSE, 0);
    
    // RIGHT PANEL
    GtkWidget *right = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    // MENU BAR
    GtkWidget *menu_bar = gtk_menu_bar_new();
    
    // FILE MENU
    GtkWidget *file_menu = gtk_menu_new();
    GtkWidget *file_item = gtk_menu_item_new_with_label("File");
    GtkWidget *run_sys_item = gtk_menu_item_new_with_label("Evialin (TEXT-MODE)");
    GtkWidget *export_mem_item = gtk_menu_item_new_with_label("Export Memory...");
    GtkWidget *reset_db_item = gtk_menu_item_new_with_label("Reset Memory...");
    GtkWidget *separator1 = gtk_separator_menu_item_new();
    GtkWidget *quit_item = gtk_menu_item_new_with_label("Quit");
    
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), export_mem_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), run_sys_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), reset_db_item); 
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), separator1);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), quit_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_item), file_menu);
    
    g_signal_connect(export_mem_item, "activate", G_CALLBACK(on_menu_export_memory), gui);
    g_signal_connect(reset_db_item, "activate", G_CALLBACK(on_menu_reset_database), gui);
    g_signal_connect(quit_item, "activate", G_CALLBACK(on_menu_quit), NULL);
    g_signal_connect(run_sys_item, "activate", G_CALLBACK(on_menu_evialin_cmd), gui);
    
    // database options 
    GtkWidget *database_menu = gtk_menu_new();
    GtkWidget *database_mi = gtk_menu_item_new_with_label("Database");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(database_mi), database_menu);
    GtkAccelGroup *accel_group_2 = gtk_accel_group_new();
    gtk_window_add_accel_group(GTK_WINDOW(gui->window), accel_group_2);
    GtkWidget *add_thought_mi = gtk_menu_item_new_with_label("Add Personal Thought");

    // Connect Ctrl + t to the "Add Personal Thought" menu item
    gtk_widget_add_accelerator(add_thought_mi, "activate", accel_group_2,
                           GDK_KEY_t, GDK_CONTROL_MASK, GTK_ACCEL_VISIBLE);                       
 
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), add_thought_mi);

    // Connect the menu click to the window creator
    g_signal_connect(add_thought_mi, "activate", G_CALLBACK(create_personal_response_window), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), database_mi);
    GtkWidget *db_separator = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), db_separator);

    // Add the AI Mode Toggle
    GtkWidget *ai_settings_item = gtk_menu_item_new_with_label("Toggle Student/Teacher modes");
    g_signal_connect(ai_settings_item, "activate", G_CALLBACK(on_handler_toggle), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), ai_settings_item);
    GtkWidget *reset_separator = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), reset_separator);

    //  Teacher Mode Toggle (Always visible)
    GtkWidget *teacher_toggle = gtk_check_menu_item_new_with_label("Teacher Mode Editor");
    g_signal_connect(teacher_toggle, "activate", G_CALLBACK(on_menu_toggle_teacher), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), teacher_toggle);

    GtkWidget *soft_reset_item = gtk_menu_item_new_with_label("Neural Soft Reset (Weights Only)");
    g_signal_connect(soft_reset_item, "activate", G_CALLBACK(on_soft_reset_clicked), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), soft_reset_item);

    GtkWidget *process_logs_item = gtk_menu_item_new_with_label("Reinforce Unhandled Logs");
    g_signal_connect(process_logs_item, "activate", G_CALLBACK(on_process_logs_clicked), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), process_logs_item);
    
    GtkWidget *heatmap_item = gtk_menu_item_new_with_label("View Neural Heatmap");
    g_signal_connect(heatmap_item, "activate", G_CALLBACK(show_neural_heatmap), NULL);
    gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), heatmap_item);

   // Training Editor Item (Initially Hidden)
   gui->menu_editor_item = gtk_menu_item_new_with_label("Open Training Editor (Ctrl+B)");
   g_signal_connect(gui->menu_editor_item, "activate", G_CALLBACK(create_training_input_window), gui);
   gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), gui->menu_editor_item);
   // Hide it by default so it only shows when Teacher Mode is turned on
   gtk_widget_set_no_show_all(gui->menu_editor_item, TRUE); 
   gtk_widget_hide(gui->menu_editor_item);   
   
GtkWidget *tracer_item = gtk_menu_item_new_with_label("Launch Neural Tracer");
g_signal_connect(tracer_item, "activate", G_CALLBACK(show_neural_path_tracer), gui);
gtk_menu_shell_append(GTK_MENU_SHELL(database_menu), tracer_item);
   
    
    // SETTINGS MENU 
    GtkWidget *settings_menu = gtk_menu_new();
    GtkWidget *settings_root = gtk_menu_item_new_with_label("Settings");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(settings_root), settings_menu);

    GtkWidget *profile_item = gtk_menu_item_new_with_label("Edit Profile");
    gtk_menu_shell_append(GTK_MENU_SHELL(settings_menu), profile_item);
    g_signal_connect(profile_item, "activate", G_CALLBACK(on_edit_profile_clicked), gui);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), settings_root);

    // TOOLS MENU
    GtkWidget *tools_menu = gtk_menu_new();
    GtkWidget *tools_item = gtk_menu_item_new_with_label("Tools");
    GtkWidget *refresh_stats_item = gtk_menu_item_new_with_label("Refresh Neural Stats");

    GtkWidget *reload_training_item = gtk_menu_item_new_with_label("Reload Training Data");
    GtkAccelGroup *accel_group = gtk_accel_group_new();
    gtk_window_add_accel_group(GTK_WINDOW(gui->window), accel_group);
    gtk_widget_add_accelerator(reload_training_item, "activate", accel_group, GDK_KEY_r, GDK_CONTROL_MASK, GTK_ACCEL_VISIBLE);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), reload_training_item);

    GtkWidget *quick_learn_item = gtk_menu_item_new_with_label("Quick Learn (Fast)");
    gtk_widget_add_accelerator(quick_learn_item, "activate", accel_group,
                           GDK_KEY_f, GDK_CONTROL_MASK, GTK_ACCEL_VISIBLE);
    g_signal_connect(quick_learn_item, "activate", G_CALLBACK(on_menu_quick_learn), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), quick_learn_item);
    
    // Create the Save menu item
    GtkWidget *save_network_item = gtk_menu_item_new_with_label("Save Neural Network");
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), save_network_item);
    gtk_widget_add_accelerator(save_network_item, "activate", accel_group, 
                           GDK_KEY_S, GDK_CONTROL_MASK | GDK_SHIFT_MASK, GTK_ACCEL_VISIBLE);
    g_signal_connect_swapped(save_network_item, "activate", G_CALLBACK(save_full_network), &g_network);

   // Neural Activity Visual menu item 
   GtkWidget *viz_item = gtk_menu_item_new_with_label("Real-time Neural Activity");
   g_signal_connect(viz_item, "activate", G_CALLBACK(on_menu_neural_viz), gui);
   gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), viz_item);
   
   // Create the "View Graph" menu item
   GtkWidget *graph_item = gtk_menu_item_new_with_label("Performance Graph");
   gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), graph_item);
   // GDK_KEY_i is the key, GDK_CONTROL_MASK is the Ctrl key
   gtk_widget_add_accelerator(graph_item, "activate", accel_group, 
                           GDK_KEY_i, GDK_CONTROL_MASK, GTK_ACCEL_VISIBLE);
                           
   // Connect it to the window creator
   g_signal_connect(graph_item, "activate", G_CALLBACK(show_stats_window), NULL);
                  
    GtkWidget *stats_graph_item = gtk_menu_item_new_with_label("Learning Statistics     Ctrl+S");
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), stats_graph_item);
    g_signal_connect(stats_graph_item, "activate", G_CALLBACK(on_menu_learning_stats), gui);
    g_signal_connect(reload_training_item, "activate", G_CALLBACK(on_menu_reload_training), gui);
    GtkWidget *launch_term_item = gtk_menu_item_new_with_label("Launch Terminal");
    GtkWidget *run_cmd_item = gtk_menu_item_new_with_label("Run Command...");
    GtkWidget *separator2 = gtk_separator_menu_item_new();
    GtkWidget *clear_term_item = gtk_menu_item_new_with_label("Clear Terminal");
    GtkWidget *run_sys_htop_item = gtk_menu_item_new_with_label("htop"); 
    
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), refresh_stats_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), run_sys_htop_item);     
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), launch_term_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), run_cmd_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), separator2);
    gtk_menu_shell_append(GTK_MENU_SHELL(tools_menu), clear_term_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(tools_item), tools_menu);
    
    g_signal_connect(refresh_stats_item, "activate", G_CALLBACK(on_menu_refresh_stats), gui);
    g_signal_connect(launch_term_item, "activate", G_CALLBACK(on_menu_launch_terminal), gui);
    g_signal_connect(run_cmd_item, "activate", G_CALLBACK(on_menu_run_command), gui);
    g_signal_connect(clear_term_item, "activate", G_CALLBACK(on_menu_clear_terminal), gui);
    g_signal_connect(run_sys_htop_item, "activate", G_CALLBACK(on_menu_run_htop), gui);

    // Options menu  
    GtkWidget *options_menu_item = gtk_menu_item_new_with_label("Options");
    GtkWidget *options_menu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(options_menu_item), options_menu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), options_menu_item); 
    
    GtkWidget *debug_item = gtk_check_menu_item_new_with_label("Background Debugging");
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(debug_item), FALSE);
    g_signal_connect(debug_item, "activate", G_CALLBACK(on_menu_toggle_debug), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(options_menu), debug_item);

    gtk_widget_show_all(options_menu_item);

     
    // HELP MENU
    GtkWidget *help_menu = gtk_menu_new();
    GtkWidget *help_item = gtk_menu_item_new_with_label("Help");
    GtkWidget *about_item = gtk_menu_item_new_with_label("About");
    gtk_menu_shell_append(GTK_MENU_SHELL(help_menu), about_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(help_item), help_menu);
    g_signal_connect(about_item, "activate", G_CALLBACK(on_menu_about), gui);
    GtkWidget *shortcuts_item = gtk_menu_item_new_with_label("Shortcuts");
    g_signal_connect(G_OBJECT(shortcuts_item), "activate", G_CALLBACK(on_menu_shortcuts), gui);
    gtk_menu_shell_append(GTK_MENU_SHELL(help_menu), shortcuts_item);


    // EDIT MENU 
    GtkWidget *edit_menu = gtk_menu_new();
    GtkWidget *edit_item = gtk_menu_item_new_with_label("Edit");
    GtkWidget *clear_chat_item = gtk_menu_item_new_with_label("Clear Chat     Ctrl+L");
    gtk_menu_shell_append(GTK_MENU_SHELL(edit_menu), clear_chat_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(edit_item), edit_menu);
    g_signal_connect(clear_chat_item, "activate", G_CALLBACK(on_menu_clear_chat), gui);

    // Connect keyboard shortcut for clearing chat (Ctrl+L)
    g_signal_connect(gui->window, "key-press-event", G_CALLBACK(on_chat_key_press), gui);
    
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), edit_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), tools_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), help_item);
    gtk_box_pack_start(GTK_BOX(right), menu_bar, FALSE, FALSE, 0);
    
    // HEADER
    GtkWidget *header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_size_request(header, -1, 50);
    gtk_widget_set_margin_start(header, 20);
    gtk_widget_set_margin_end(header, 20);
    gtk_widget_set_margin_top(header, 10);
    
    GtkWidget *title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(title), "<span font='18' weight='bold'>Evialin Chat</span>");
    gtk_box_pack_start(GTK_BOX(header), title, FALSE, FALSE, 0);
    
    gui->status_label = gtk_label_new("● Initialized Student Mode [Active]");
    gtk_box_pack_end(GTK_BOX(header), gui->status_label, FALSE, FALSE, 0);
    
    gtk_box_pack_start(GTK_BOX(right), header, FALSE, FALSE, 0);
    
    // CHAT AREA
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(scroll, -1, 300);
    
    gui->chat_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(gui->chat_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(gui->chat_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(gui->chat_view), GTK_WRAP_WORD_CHAR);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(gui->chat_view), 15);
    gtk_text_view_set_right_margin(GTK_TEXT_VIEW(gui->chat_view), 15);
    gtk_text_view_set_top_margin(GTK_TEXT_VIEW(gui->chat_view), 10);
    
    gui->chat_buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(gui->chat_view));
    
    gtk_container_add(GTK_CONTAINER(scroll), gui->chat_view);
    gtk_box_pack_start(GTK_BOX(right), scroll, TRUE, TRUE, 0);
    
    // CHAT INPUT
    GtkWidget *input_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_margin_start(input_box, 20);
    gtk_widget_set_margin_end(input_box, 20);
    gtk_widget_set_margin_bottom(input_box, 10);
    gtk_widget_set_margin_top(input_box, 10);
    
    gui->input_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(gui->input_entry), "Ask me anything...");
    gtk_widget_set_size_request(gui->input_entry, 400, 40);
    
    gui->send_button = gtk_button_new_with_label("Send");
    gtk_widget_set_size_request(gui->send_button, 100, 40);
    
    gtk_box_pack_start(GTK_BOX(input_box), gui->input_entry, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(input_box), gui->send_button, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(right), input_box, FALSE, FALSE, 0);
    
    // SEPARATOR
    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(right), separator, FALSE, FALSE, 0);
 
    // TERMINAL
    GtkWidget *term_header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_margin_start(term_header, 20);
    gtk_widget_set_margin_end(term_header, 20);
    gtk_widget_set_margin_top(term_header, 10);
    
    GtkWidget *term_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(term_title), "<span font='14' weight='bold'>Terminal Mode</span>");
    gtk_box_pack_start(GTK_BOX(term_header), term_title, FALSE, FALSE, 0);
    
    gtk_box_pack_start(GTK_BOX(right), term_header, FALSE, FALSE, 0);
    
    GtkWidget *term_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(term_scroll), GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(term_scroll, -1, 200);
    
    gui->terminal_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(gui->terminal_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(gui->terminal_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(gui->terminal_view), GTK_WRAP_WORD_CHAR);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(gui->terminal_view), 10);
    
    ctx = gtk_widget_get_style_context(gui->terminal_view);
    gtk_style_context_add_provider(ctx, GTK_STYLE_PROVIDER(css), GTK_STYLE_PROVIDER_PRIORITY_USER);
    gtk_style_context_add_class(ctx, "terminal");
    
    gui->terminal_buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(gui->terminal_view));
    
    gtk_container_add(GTK_CONTAINER(term_scroll), gui->terminal_view);
    gtk_box_pack_start(GTK_BOX(right), term_scroll, TRUE, TRUE, 0);
    
    GtkWidget *term_input_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_margin_start(term_input_box, 20);
    gtk_widget_set_margin_end(term_input_box, 20);
    gtk_widget_set_margin_bottom(term_input_box, 20);
    gtk_widget_set_margin_top(term_input_box, 10);
    
    gui->terminal_input = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(gui->terminal_input), "evialin> ");
    gtk_widget_set_size_request(gui->terminal_input, 400, 35);
    
    ctx = gtk_widget_get_style_context(gui->terminal_input);
    gtk_style_context_add_provider(ctx, GTK_STYLE_PROVIDER(css), GTK_STYLE_PROVIDER_PRIORITY_USER);
    gtk_style_context_add_class(ctx, "terminal-input");
    
    gui->terminal_send = gtk_button_new_with_label("Execute");
    gtk_widget_set_size_request(gui->terminal_send, 100, 35);
    
    gtk_box_pack_start(GTK_BOX(term_input_box), gui->terminal_input, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(term_input_box), gui->terminal_send, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(right), term_input_box, FALSE, FALSE, 0);
    
    gtk_box_pack_start(GTK_BOX(main_box), right, TRUE, TRUE, 0);

    // Sleep timer - 60 seconds until sleepy mode
    g_timeout_add_seconds(60, (GSourceFunc)sleepy_mode, gui);

    // SIGNALS
    g_signal_connect(gui->send_button, "clicked", G_CALLBACK(on_send), gui);
    g_signal_connect(gui->input_entry, "key-press-event", G_CALLBACK(on_key_press), gui);
    g_signal_connect(gui->terminal_send, "clicked", G_CALLBACK(on_terminal_send), gui);
    g_signal_connect(gui->terminal_input, "key-press-event", G_CALLBACK(on_terminal_key_press), gui);
    g_signal_connect(gui->window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    
    add_message(gui, 
        "Hello! I'm Evialin with memory and a neural network!\n\n"
        "I can:\n"
        "• Solve math problems (e.g., 'What is 25 plus 17?')\n"
        "• Remember your name (say 'My name is...')\n"
        "• Store facts (say 'Remember that...')\n"
        "• Have conversations!\n\n"
        "What's your name?", 0);
    
    add_terminal_output(gui, "Evialin Terminal Mode v2.0", 0);
    add_terminal_output(gui, "Type 'help' for available commands", 0);
    add_terminal_output(gui, "", 0);

    // Initialize stats display
    update_neural_stats(gui);
    
    // Auto-refresh stats every second
    g_timeout_add_seconds(1, (GSourceFunc)update_neural_stats, gui);
   
    g_object_unref(css);
    return gui;
}

/********************************************************************
 * Neural path tracer tool for our network (Very basic idea outline)*
 * All Sub-functions start here...                                  *
 ********************************************************************/
 void show_neural_path_tracer(GtkWidget *widget, gpointer data) {
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Neural Path Tracer: Forward & Backward Prop");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 800);
    
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER(window), drawing_area);
    
    // Connect the drawing logic
    g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(on_draw_path_grid), NULL);
    
     // Refresh at ~20fps for smooth scrolling
     g_timeout_add(50, (GSourceFunc)gtk_widget_queue_draw, drawing_area);
    
    gtk_widget_show_all(window);
}
/* Grid & path Rendering Logic */
static gboolean on_draw_path_grid(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // CLEAR BACKGROUND
    cairo_set_source_rgb(cr, 0.01, 0.01, 0.04); 
    cairo_paint(cr);

    // Layout Anchors - Define fixed "Lanes"
    int input_lane_x = 80;
    int hidden_start_x = 220;
    int output_lane_x = 720;
    
    int spacing_x = 35;
    int spacing_y = 38; // Increased spacing for readability
    int cols = 10;
    int active_count = 0;

    // HEADER & HUMAN-READABLE TEXT
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_source_rgb(cr, 0.9, 0.9, 1.0);
    cairo_set_font_size(cr, 14.0);
    cairo_move_to(cr, 40, 30);  cairo_show_text(cr, "INPUT");
    cairo_move_to(cr, 220, 30); cairo_show_text(cr, "HIDDEN PROCESSING (1-150)");
    cairo_move_to(cr, 680, 30); cairo_show_text(cr, "OUTPUT");

    cairo_set_font_size(cr, 10.0);
    cairo_set_source_rgb(cr, 0.5, 0.5, 0.7);
    cairo_move_to(cr, 40, 45); 
    cairo_show_text(cr, "PURPLE: Forward activation path. BLUE: Backpropagation error correction signal.");

    // DRAW NEURON MATRIX (150 NODES)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Calculate this specific neuron's center point
        int h_x = (i % cols) * spacing_x + hidden_start_x;
        int h_y = (i / cols) * spacing_y + 85;
        
        double activation = g_network.hidden[i];
        double delta = fabs(g_network.hidden_deltas[i]);

        if (activation > 0.1) active_count++;

        // --- FIXED FORWARD PATH (Purple) ---
        // Draws a line from the Input Lane directly to THIS neuron's Y-coordinate
        if (activation > 0.05) {
            cairo_set_source_rgba(cr, 0.6, 0.3, 1.0, activation * 0.4);
            cairo_set_line_width(cr, 1.0);
            cairo_move_to(cr, input_lane_x, h_y); // Match neuron Y
            cairo_line_to(cr, h_x, h_y);
            cairo_stroke(cr);
        }

        // --- FIXED LEARNING PULSE (Blue) ---
        // Draws a line from the Output Lane directly to THIS neuron's Y-coordinate
        if (delta > 0.001) {
            cairo_set_source_rgba(cr, 0.0, 0.8, 1.0, delta * 5.0);
            cairo_set_line_width(cr, 1.5);
            cairo_move_to(cr, output_lane_x, h_y); // Match neuron Y
            cairo_line_to(cr, h_x, h_y);
            cairo_stroke(cr);
            
            // Add a "firing" glow ring around the node
            cairo_arc(cr, h_x, h_y, 6 + (delta * 8), 0, 2 * M_PI);
            cairo_stroke(cr);
        }

        // --- DRAW THE NEURON NODE ---
        cairo_set_source_rgb(cr, 0.1, 0.1, 0.2);
        cairo_arc(cr, h_x, h_y, 5, 0, 2 * M_PI);
        cairo_fill(cr);

        // --- NEURON ID LABELING ---
        char label[4];
        sprintf(label, "%d", i + 1);
        cairo_set_source_rgb(cr, 0.7, 0.7, 0.8);
        cairo_set_font_size(cr, 9.0);
        cairo_move_to(cr, h_x - 12, h_y + 18);
        cairo_show_text(cr, label);
    }

    // THE HUD (DIAGNOSTICS PANEL)
    int hud_y = 660;
    cairo_set_source_rgb(cr, 0.03, 0.03, 0.1);
    cairo_rectangle(cr, 10, hud_y, 780, 130);
    cairo_fill(cr);

    // NEURAL LOAD STATS
    double active_pct = (double)active_count / HIDDEN_SIZE;
    cairo_set_source_rgb(cr, 0.8, 0.8, 1.0);
    cairo_set_font_size(cr, 12.0);
    cairo_move_to(cr, 30, hud_y + 35);
    char load_txt[64];
    sprintf(load_txt, "NEURAL LOAD: %d/%d Active (%.1f%%)", active_count, HIDDEN_SIZE, active_pct * 100);
    cairo_show_text(cr, load_txt);

    // EPOCH & SYNAPSE STATS
    cairo_set_font_size(cr, 11.0);
    cairo_set_source_rgb(cr, 0.5, 0.5, 0.6);
    cairo_move_to(cr, 30, hud_y + 65);
    char stats_txt[128];
    sprintf(stats_txt, "Training Epochs: %d | Synapses: %d", current_epoch, (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE));
    cairo_show_text(cr, stats_txt);

    // SMOOTH SCROLLING GRAPH (STABILITY TREND)
    cairo_set_source_rgb(cr, 0.6, 0.6, 0.7);
    cairo_move_to(cr, 450, hud_y + 25);
    cairo_show_text(cr, "BRAIN STABILITY TREND (Live)");

    // Shift data left and update history
    for (int i = 0; i < GRAPH_SAMPLES - 1; i++) memory_history[i] = memory_history[i+1];
    memory_history[GRAPH_SAMPLES-1] = (active_pct * 50.0) + (double)(rand() % 5);

    cairo_set_source_rgb(cr, 0.0, 1.0, 0.8); // Cyan Graph
    cairo_set_line_width(cr, 2.0);
    for (int i = 0; i < GRAPH_SAMPLES; i++) {
        double gx = 450 + (i * 6);
        double gy = (hud_y + 110) - memory_history[i];
        if (i == 0) cairo_move_to(cr, gx, gy);
        else cairo_line_to(cr, gx, gy);
    }
    cairo_stroke(cr);

    return FALSE;
}
/*  Xoshiro256 State & functions 
 **/ 
static uint64_t xoshiro_state[4];

// Seeds the 256-bit state using OS entropy (/dev/urandom)
void init_xoshiro_from_urandom(void) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t read_bytes = fread(xoshiro_state, sizeof(xoshiro_state), 1, f);
        fclose(f);
        if (read_bytes == 1) return; // Successfully seeded
    }

    // Fallback seed if /dev/urandom is unavailable
    xoshiro_state[0] = (uint64_t)time(NULL);
    xoshiro_state[1] = (uint64_t)clock();
    xoshiro_state[2] = 0x9E3779B97F4A7C15ULL; // Aprox to Golden ratio constant
    xoshiro_state[3] = (uint64_t)&f;          // Address entropy
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Generates next random 64-bit integer
uint64_t next_xoshiro(void) {
    const uint64_t result = rotl(xoshiro_state[1] * 5, 7) * 9;
    const uint64_t t = xoshiro_state[1] << 17;

    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[3] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[0] ^= xoshiro_state[3];

    xoshiro_state[2] ^= t;
    xoshiro_state[3] = rotl(xoshiro_state[3], 45);

    return result;
}

// Converts random integer to double in range [0.0, 1.0)
double random_double(void) {
    return (next_xoshiro() >> 11) * (1.0 / (1ULL << 53));
}
// SplitMix64 reliably expands 32/64-bit seeds into the 256-bit Xoshiro state
static uint64_t splitmix64_next(uint64_t *sm_state) {
    uint64_t z = (*sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Drop-in replacement for srand()
void xoshiro_srand(unsigned int seed) {
    uint64_t sm_state = seed;
    xoshiro_state[0] = splitmix64_next(&sm_state);
    xoshiro_state[1] = splitmix64_next(&sm_state);
    xoshiro_state[2] = splitmix64_next(&sm_state);
    xoshiro_state[3] = splitmix64_next(&sm_state);
}


// Core Xoshiro256** generator
uint64_t xoshiro256starstar(void) {
    const uint64_t result = rotl(xoshiro_state[1] * 5, 7) * 9;
    const uint64_t t = xoshiro_state[1] << 17;

    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[3] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[0] ^= xoshiro_state[3];

    xoshiro_state[2] ^= t;
    xoshiro_state[3] = rotl(xoshiro_state[3], 45);

    return result;
}

// Drop-in replacement for rand() bounded to RAND_MAX
int xoshiro_rand(void) {
    return (int)(xoshiro256starstar() % ((uint64_t)RAND_MAX + 1));
}
uint64_t xoshiro_rand_impl(void);
long int long_xoshiro_rand(void) {
    return (long int)(xoshiro256starstar() % ((uint64_t)RAND_MAX + 1));
}


int main(int argc, char *argv[]) {
    g_trie_root = createNode(); 	
    gtk_init(&argc, &argv);    
    init_database();
    cleanup_intent_clutter(); 
    load_full_network(&g_network);
    // Attempt to load the network from database! If it doesn't exist,  
    // init_network(&g_network); this will be called as A starting point.     
    // init_training_data(); 
    init_xoshiro_from_urandom();
    xoshiro_srand(time(NULL));
    printf("xoshiro number check: "); 
    for (int i = 0; i < 3; i++)  {
	xoshiro_srand(time(NULL)+i); 	
    printf("%d : %ld >> ", i, long_xoshiro_rand());
    }
    printf("3 : %ld \n", long_xoshiro_rand()); 
    
    printf("All systems ready and restored!\n");
    const char *input = "All Systems ready and restored!"; 
    if (random_double() < 0.09) { 
		double rnn_output[OUTPUT_SIZE];
	for (int i = 0; input[i] != '\0'; i++) { 
		rnn_process_character((unsigned char)input[i], rnn_output);  
		}
 	char *rnn_response = get_constructed_rnn_response(rnn_output);
        if (rnn_response) {
            // Process or print response
            free(rnn_response); // If dynamically allocated
        }
    }	
    GtkCssProvider *cssProvider = gtk_css_provider_new();
const char *style = 
    "#activity_bar progress { background-color: #00ff00; background-image: none; }"
    "#activity_bar trough { background-color: #1a1a2e; border: 1px solid #7c3aed; }"
    "#confidence_bar progress { background-color: #00ff00; background-image: none; }"
    "#confidence_bar trough { background-color: #1a1a2e; border: 1px solid #7c3aed; }"
    "#activity_bar label {"
    "   color: #ffffff;"
    "   font-weight: bold;"
    "   font-size: 12px;"
    "}"
    "button {"
    "  background-image: none;"
    "  background-color: #7c3aed;" 
    "  color: #ffffff;"
    "  border-radius: 8px;"
    "  padding: 5px 15px;"
    "  font-weight: bold;"
    "  border: 1px solid #6d28d9;"
    "}"
    "headerbar {"
    "   background-color: #312e81;"
    "   color: white;"
    "}"
    /* Target the buttons and their internal icons directly */
    "headerbar button.titlebutton {"
    "   color: #ffffff;"
    "   background-color: rgba(255, 255, 255, 0.1);"
    "   border: none;"
    "   transition: background-color 0.2s;"
    "}"
    "headerbar button.titlebutton:hover {"
    "   background-color: #4338ca;"
    "   color: #ffffff;"
    "}"
    /* This ensures the actual 'x' and '_' symbols are forced to white */
    "headerbar button.titlebutton image {"
    "   color: white;"
    "}"
    "#confidence_bar label {"
    "   color: #ffffff;"
    "   font-weight: bold;"
    "   font-size: 12px;"
    "}"
    "#purple_double_line {"
    "   border-top: 5px double #7c3aed;" 
    "   min-height: 5px; margin: 10px 0px;"
    "   margin: 5px 0px;"
    "}"
    "label { "
    "	margin: 0; padding: 0; "
    "	color: #00003e; " 
    "}"
    "#white_double_line {"
    "   background-color: #ffffff;" 
    "   border-top: 4px double #ffffff;" /* The white double line */
    "   min-height: 4px;"                /* Required for visibility */
    "   margin: 10px 0px;"               /* Vertical spacing */
    "}"
    "label { margin: 0; padding: 0; }";  /* Force labels to be tight */

gtk_css_provider_load_from_data(cssProvider, style, -1, NULL);
gtk_style_context_add_provider_for_screen(
    gdk_screen_get_default(),
    GTK_STYLE_PROVIDER(cssProvider),
    GTK_STYLE_PROVIDER_PRIORITY_USER
);
    GUI *gui = create_gui();
    gtk_widget_show_all(gui->window);
    
    gtk_main();
    
    if (g_db) {
        sqlite3_close(g_db);
    }
    gui->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    g_signal_connect(gui->window, "destroy", G_CALLBACK(on_window_destroy), gui);
    free(gui);
    
    return 0;
} 

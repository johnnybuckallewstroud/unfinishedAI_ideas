# unfinishedAI_ideas

The code is structured as a self-contained C desktop application combining several distinct subsystems:

UI & Animation Framework: Built using GTK+ 3.0, featuring a dual-pane layout (Chat view and a live interactive Terminal mode) alongside an animated avatar display (GifMapping) driven by GdkPixbufAnimation.

Neural Core (Feed-Forward & RNN): Implements a multi-layer network with customizable dimensions (INPUT_SIZE 100, HIDDEN_SIZE 150, OUTPUT_SIZE 115), backpropagation routines, Sigmoid/ReLU activations, and an auxiliary character-level RNN layer (rnn_process_character) that maps sequences into semantic intent clusters.

Semantic & Symbolic Layer: Incorporates text embedding vectors (EMBED_DIM 16), a prefix Trie structure (struct TrieNode) for greedy autocomplete fallback, and Levenshtein/Jaccard-style string overlap matching.

Persistent Memory & Logging: Uses sqlite3 (evialin_memory.db) to store conversational history, learned Q&A pairs (learned_%), joke repositories, system stats, and unhandled diagnostic logs for offline training.

Evialin remains an active, open-source work-in-progress built entirely from scratch using C, GTK+ 3.0, and SQLite to explore lightweight, transparent desktop AI. Because the project is still unfinished, its hybrid architecture—spanning a feed-forward neural core, a character-level recurrent intuition engine, and persistent local memory—serves as an evolving testbed for low-overhead engineering, inviting ongoing experimentation, debugging, and community contribution under free software principles.

# Pascal Datastructures

A collection of Pascal implementations of useful data structures and classical algorithms.

---

## Directory Summary

### Transformer.pas

- Pure Pascal implementation of a transformer (GPT-2 style) neural model.
- Loads GGUF files for weights and a JSON/BPE tokenizer.
- Works end-to-end for text generation on CPU.
- Everything is handled with static types and flat arrays for transparency and reuse.
- **Usage:**
  1. Compile `Transformer.pas` with Free Pascal:
     ```
     fpc Transformer.pas
     ```
  2. Run with a model and tokenizer:
     ```
     ./transformer model.gguf tokenizer.json "your prompt here" max_tokens
     ```
     Example:
     ```
     ./transformer GPT-2-f32.gguf tokenizer.json "Once upon a time" 20
     ```
  3. Output is printed to the terminal after generation.
- The code can be used as a library or a reference for understanding transformer inner workings.

### arrays.pas, matrix.pas

- Basic dynamic arrays and matrix utilities (flat and nested representations).

### hashtable.pas

- Simple generic hash table/dictionary for key-value mapping.

### sorts.pas

- Common sorting methods: quicksort, mergesort, bubblesort, etc.

### stack.pas, queue.pas

- Stack and queue (FIFO) implementations.

### redblacktree.pas

- Red-black binary search tree for balanced map/set operations.

### gnn.pas

- Graph neural network primitives and data handling routines.

### mlp.pas

- Simple multi-layer perceptron neural network with configurable layers.

### rnn.pas

- Basic recurrent neural network implementation and testing routines.

### cnn.pas

- Basic convolutional neural network containers and procedures.

### heap.pas, priorityqueue.pas

- Heap and priority queue implementations.

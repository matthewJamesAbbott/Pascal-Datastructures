# Pascal Data Structures

A comprehensive repository of individual Pascal programs, each implementing a different data structure or algorithmic approach – from fundamentals to advanced.  
Ideal for students and enthusiasts interested in Pascal and classic CS topics.

---

## Contents

Below is a list of all included Pascal programs.  
Each entry links to an individual writeup and usage instructions.

### Source Files

- [AVLTree.pas: AVL (Height-Balanced Self-Balancing) Binary Search Tree](#avltree-avl-height-balanced-self-balancing-binary-search-tree)
- [BTree.pas: B-Tree (Multiway, Balanced Search Tree)](#btree-b-tree-multiway-balanced-search-tree)
- [CNN.pas: Convolutional Neural Network (Deep Learning)](#cnn-convolutional-neural-network)
- [DatastructureTest.pas: Comprehensive Data Structure Tester](#datastructuretest-comprehensive-data-structure-tester)
- [DatastructureTestResults.txt: Example Test Output](#datastructuretestresults-example-test-output)
- [FacadeCNN.pas: CNN (Convolutional Neural Network) Facade](#facadecnn-cnn-facade-unit-for-introspection--manipulation)
- [FacadeGNN.pas: GNN (Graph Neural Network) Facade](#facadegnn-graph-neural-network-introspection--utilities)
- [FacadeMLP.pas: MLP (Multi Layer Perceptron) Facade](#facademlp-mlp-multilayer-perceptron-facade-unit)
- [FacadeRNN.pas: RNN (Recurrent Neural Network) Facade](#facadernn-rnn-recurrent-neural-network-facade-unit)
- [FacadeTransformer.pas: Transformer Model Introspection and Manipulation Facade](#facadetransformer-transformer-model-introspection-and-manipulation-facade)
- [HeapBinaryTree.pas: Binary Tree-based Heap Data Structure](#heapbinarytree-binary-tree-based-heap-data-structure)
- [HeapBinaryTreeNode.pas: Node Class for Binary Tree-based Heap](#heapbinarytreenode-node-class-for-binary-tree-based-heap)
- [HeapDoubleLinkedList.pas: Heap Using a Doubly Linked List](#heapdoublelinkedlist-heap-using-a-doubly-linked-list)
- [HeapNode.pas: Node Class for Heap Implemented via Linked List](#heapnode-node-class-for-heap-implemented-via-linked-list)
- [MLP.pas: MultiLayer Perceptron (Feedforward Neural Network)](#mlp-multilayer-perceptron-feedforward-neural-network)
- [RNN.pas: Advanced Recurrent Neural Network](#rnn-advanced-recurrent-neural-network)
- [RedBlackTree.pas: Red-Black Self-Balancing Binary Search Tree](#redblacktree-red-black-self-balancing-binary-search-tree)
- [Stack.pas: Classic Stack (Array-based) Implementation](#stack-classic-stack-array-based-implementation)
- [StackBinaryTree.pas: Stack Implemented using a Binary Tree](#stackbinarytree-stack-implemented-using-a-binary-tree)
- [StackDoubleLinkedList.pas: Stack Using a Doubly Linked List](#stackdoublelinkedlist-stack-using-a-doubly-linked-list)
- [StackLinkedList.pas: Stack Using a Linked List](#stacklinkedlist-stack-using-a-linked-list)
- [Transformer.pas: Minimal Pascal Transformer (Attention-based Model)](#transformer-minimal-pascal-transformer-attention-based-model)

---

## Individual Program Writeups

---

### AVLTree: AVL (Height-Balanced Self-Balancing) Binary Search Tree

**File:** `AVLTree.pas`  
**Category:** Data Structures / Trees / Self-Balancing BST

#### Description

Implements an **AVL Tree**, a classic height-balanced self-balancing binary search tree, in Pascal.  
AVL trees guarantee `O(log n)` insertion, deletion, and lookup by maintaining the height difference (balance factor) between every node’s left and right subtrees at most one, after every modification.

**Features:**
- Fully dynamic insertions with automatic balancing ("fixup")
- Pointer-based node structure with explicit `data`, `height`, `parent`, `left`, and `right`
- Efficient left and right rotations, single and double
- Real-time balance and height calculation; maintains `height` property on all updates
- `inorderTraversal` method prints each value, height, and balance factor for inspection

**Data Model:**
- Each node stores:
  - `data`: the integer value in the node
  - `height`: cached subtree height for O(1) balance checks
  - `parent`, `left`, `right`: classic BST pointers

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later

**Usage Example:**
```pascal
uses AVLTree;
var
  tree: TAVLTree;
  root: treeNode;
begin
  tree.create;
  tree.insert(root, 10);
  tree.insert(root, 8);
  tree.insert(root, 13);
  tree.insert(root, 6);
  tree.inorderTraversal(root);
end.
```

**To Compile:**
```bash
fpc AVLTree.pas
# ...plus your main or test driver
```

#### Usage Notes

- All balancing and rotation logic is handled seamlessly in `insertFixup`, so insertion always produces a balanced BST.
- Traversal prints each node along with its height and balance for confidence in structure.

---

### BTree: B-Tree (Multiway, Balanced Search Tree)

**File:** `BTree.pas`  
**Category:** Data Structures / Trees / Multiway / Balanced Search

#### Description

A full implementation of a **B-Tree**, the classic multiway, height-balanced search tree ideal for large datasets and external memory (disk) indexing.  
B-trees maintain sorted data and allow efficient O(log n) search, insert, and sequential access, and are the backbone for databases, filesystems, and big indexes.

**Features:**
- All node and pointer management in explicit Pascal pointer/array logic
- Configurable minimum degree (3 by default) for branching and storage
- Node splitting during insert (handles overflow automatically)
- Efficient binary search for inserts and retrievals
- Keeps nodes "mostly full" for minimal tree height and efficient traversal
- Printing in-order traversals at any time

**Data Model:**
- Each node comprises:
  - `keys`: array of integer values
  - `children`: array of child pointers
  - `numKeys`: current key count (≤ max per node)
  - `isLeaf`: boolean
  - `parent`: for upward traversal/structural logic

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or newer

**Usage Example:**
```pascal
uses BTree;
var
  tree: TBTree;
  root: treeNode;
begin
  tree.create;
  tree.insert(root, 10);
  tree.insert(root, 15);
  tree.insert(root, 25);
  tree.inorderTraversal(root);
end.
```

**To Compile:**
```bash
fpc BTree.pas
# ...plus your main or demonstration program
```

#### Usage Notes

- Designed for in-memory operation, but logic maps directly to disk/large datasets.
- For best educational value, step through the split/insert semantics (see `splitChild`).
- This classic implementation is the foundation for exploring filesystems and database internals.

---

### CNN: Convolutional Neural Network

**File:** `CNN.pas`  
**Category:** Machine Learning / Deep Learning

#### Description

A fully self-contained Pascal implementation of a modern Convolutional Neural Network (CNN) from scratch.  
Features include:
- Multiple convolutional and pooling layers
- Fully connected layers
- ReLU activation, softmax + cross-entropy loss
- Adam optimizer with bias correction
- Dropout regularization
- Numerically stable softmax, clipping, and error handling
- Model save/load
- Modular design using object-oriented free Pascal (`{$mode objfpc}`)

This is a teaching/research-oriented example: no external libraries are required for the core functionality, and the code exposes internal states for hands-on learning.

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later recommended
- 64-bit system recommended for large arrays (due to memory use)

**Compile:**
```bash
fpc CNN.pas
```

**Run:**
```bash
./CNNtest
```
_Note: The `program` block in this file is named `CNNtest`._

**To Use as a Library:**  
You can also `uses` the CNN class in other Pascal files or units for custom experiments or integration.

#### Usage Notes

- The program/demo provided in the main block of `CNN.pas` can be modified to load your own image data, set training targets, and configure network parameters.
- Model hyperparameters (number of layers, filters, etc.) are set in the constructor of `TConvolutionalNeuralNetwork`.
- For real datasets: you will need to implement (or adapt) input conversion from image files to the expected `TImageData` format.

#### Example: Creating a Simple CNN in Your Pascal Code

```pascal
var
  cnn: TConvolutionalNeuralNetwork;
begin
  cnn := TConvolutionalNeuralNetwork.Create(
    28,    // input width
    28,    // input height
    1,     // channels (e.g. grayscale)
    [8,16],// Conv filters per layer
    [3,3], // Kernel sizes
    [2,2], // Pool sizes
    [64],  // FC layer sizes
    10,    // output classes
    0.001, // learning rate
    0.25   // dropout rate
  );
  // Now use cnn.Predict(...) and cnn.TrainStep(...)
end.
```

**Model Saving/Loading:**
- Call `cnn.SaveCNNModel('my_model.bin')` and `cnn.LoadCNNModel('my_model.bin')` as needed.

---

### DatastructureTest: Comprehensive Data Structure Tester

**File:** `DatastructureTest.pas`  
**Category:** Data Structure Testing / Demonstration

#### Description

A Pascal program that serves as a unified tester for several classic data structure implementations, such as linked lists, double-linked lists, binary-tree-based stacks, and heaps.  
It is designed to automatically create each structure, add elements, perform standard operations (insert, delete, retrieve), and print the process/results step-by-step.  
The code provides a hands-on, procedural demonstration of each supported module, driven by verbose output via `writeln`.

**Included Modules:**
- `StackLinkedList.pas`
- `StackDoubleLinkedList.pas`
- `StackBinaryTree.pas`
- `HeapLinkedList.pas`
- `HeapDoubleLinkedList.pas`

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later recommended

**Compile:**
```bash
fpc DatastructureTest.pas
```

**Run:**
```bash
./DataStructureTest
```

**What it does:**
- Sequentially instantiates each data structure (e.g., linked list, double-linked, heap)
- Demonstrates: node insertion, deletion (first, last, by index), data extraction, and search
- Prints each action with a descriptive output for simple tracking and validation

#### Usage Notes

- This program is primarily for learning and testing the functionality of the included data structure units.
- To add your own tests, append more operations in the `begin ... end.` block after the existing demonstrations.
- The program output is intended to match the sample found in `DatastructureTestResults.txt`.

---

### DatastructureTestResults.txt: Example Test Output

**File:** `DatastructureTestResults.txt`  
**Category:** Output Sample / Reference

#### Description

A plain-text file capturing a real output log from running `DatastructureTest.pas`.  
It documents every step, action, and change of state performed on the data structures during the test run.  
You can use this file to:
- Verify expected output for successful test runs
- Compare changes when you modify the test program
- Understand the normal "flow" of each data structure’s use and manipulation

#### How to Use

1. Compile and run `DatastructureTest.pas` as described above
2. Compare your terminal output to this file to ensure correct operation
3. Use differences to help debug or enhance your structures

---

### FacadeCNN: CNN Facade Unit for Introspection & Manipulation

**File:** `FacadeCNN.pas`  
**Category:** Machine Learning Utilities / Deep Learning Helper

#### Description

A comprehensive Pascal unit (`unit CNNFacade`) providing a **facade** (i.e., a simplified interface) for deep introspection, manipulation, and analysis of Convolutional Neural Networks (CNNs).  
This unit is designed to enhance your ability to debug, analyze, and extend CNNs implemented by the author’s other Pascal modules (see [`CNN.pas`](#cnn-convolutional-neural-network)), by exposing detailed accessors and tools for reading and adjusting internal model state.

**Key Capabilities:**
- Detailed access to convolutional and fully connected layer parameters (weights, biases)
- Structured types for feature maps, kernels, neuron parameters, batch norm, etc.
- Utilities for extracting statistics (means, stdev, min/max) for any layer
- Read/write access to layer configurations and attributes
- Support for batch normalization parameters, filter attributes, and receptive field calculations
- All code is pure Pascal (`{$mode objfpc}`), designed to be integrated alongside core CNN code

This unit is particularly useful for:
- Educational visualization of neural networks
- Research into layer behavior and transformations
- Custom training loops, fine-tuning, and explainable AI
- Model inspection or serialization

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later recommended
- Should be used in concert with a core CNN implementation (such as [`CNN.pas`](#cnn-convolutional-neural-network)), by adding `FacadeCNN` to your `uses` clause

**To Integrate:**
1. Place `FacadeCNN.pas` in your project directory
2. In your main program or unit, add it to the `uses` clause:
    ```pascal
    uses FacadeCNN, CNN;
    ```
3. Instantiate and use the `TCNNFacade` class for advanced access, e.g.:
    ```pascal
    var
      fcnn: TCNNFacade;
    begin
      fcnn := TCNNFacade.Create( ...layers/params... );
      // ... CNN usage ...
      // Example: read the feature map of Conv Layer 1, Filter 0
      var fmap := fcnn.GetFeatureMap(1, 0);
    end;
    ```
4. Use the provided accessor and mutator functions to:
    - Retrieve or set kernel weights
    - Access feature maps, preactivations, biases
    - Gather per-layer statistics for analysis or visualization
    - Modify filter attributes or apply batch normalization parameters

#### Usage Notes

- This facade is not a standalone program, but a utility class/unit to use with compatible neural network models.
- Can be used for in-depth experiment logging, debugging, and research.
- Extend or customize the unit for your own CNN architectures or for integrations with mathematical/statistical analysis tools.

---

### FacadeGNN: Graph Neural Network Introspection & Utilities

**File:** `FacadeGNN.pas`  
**Category:** Machine Learning Utilities / Graph Learning

#### Description

A comprehensive Pascal unit (`unit GNNFacade`) that provides a facade (simplified interface) as well as extensive introspection, manipulation, and analysis utilities for Graph Neural Networks (GNNs).  
This unit is intended to support advanced GNN architectures, training, and experimentation in Pascal, equipping researchers and students to:

- Build, train, and inspect Graph Neural Networks for node, edge, or whole-graph learning tasks
- Access and manipulate all aspects of network state: layers, embeddings, weights, activations, gradients, edge features, adjacency structures, etc.
- Run and debug message passing, backpropagation, loss calculation, and architecture configuration
- Support for various activation and loss types, batch embeddings, custom optimizers, and flexible graph configurations (undirected, self-loops, edge deduplication)

**Core Features:**
- Modular layer and neuron types for message, update, readout, and output computations
- Deep access to node and edge features, graph topology, and learned representations
- Built-in support for gradient clipping and diagnostic metric tracking
- Numerous utility routines for copying, concatenating, and handling arrays/graphs
- Can be extended for your own GNN flavors: GCN, GAT, MPNN, etc.

All code is Object Pascal (`{$mode objfpc}`), and is compatible with modern Free Pascal.

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later recommended
- Designed to import as a unit in your GNN projects

**To Integrate:**
1. Place `FacadeGNN.pas` in your project directory.
2. Add the unit to the `uses` clause in your Pascal project:
    ```pascal
    uses FacadeGNN;
    ```
3. You can now define and create `TGraphNeuralNetwork` objects, configure layers, and operate on `TGraph` structures.
4. Use the methods and properties to:
    - Initialize architectures with standard or custom parameters
    - Train/run GNNs stepwise or on batches
    - Inspect internal states (embeddings, weights, activations, gradients)
    - Export, analyze, or manipulate models in-depth for research or teaching

#### Usage Notes

- This unit is not a standalone executable, but a reusable module for advanced GNN engineering and exploration.
- Consult the inline documentation and type declarations for extending to your own needs (e.g., new aggregation strategies, custom metrics).
- For introductory usage, build a main program that includes this unit and demonstrates node classification or graph regression.

---

### FacadeMLP: MLP (MultiLayer Perceptron) Facade Unit

**File:** `FacadeMLP.pas`  
**Category:** Machine Learning Utilities / Feedforward Neural Nets

#### Description

A thoroughly-featured Object Pascal unit (`unit MLPFacade`) acting as a facade for multi-layer perceptrons.  
This module is engineered to provide **detailed, externally accessible control and introspection** over the implementation of a classic feedforward neural network (MLP), especially designed for experimentation, research, and educational uses.

**Core Functions Exposed:**
- Full access to MLP architecture (input/hidden/output layers)
- Neuron and layer-wise accessors: weights, biases, pre-activations, outputs, error gradients
- Batch normalization, dropout settings, optimizer state (SGD, Adam, RMSProp)
- L2-regularization and per-neuron attributes
- Network topology modification (add/remove layers & neurons dynamically)
- Batch/epoch training statistics, histogram features for diagnostics

All types, records, and methods are presented in `{$mode objfpc}`/Free Pascal style for seamless advanced integration.

This unit is ideal for:
- Introspecting/tracing MLP activations and gradients at every stage
- Modifying/training networks on the fly (e.g., for autoML or ablation studies)
- Logging/visualizing inner state for teaching and debugging
- Custom research where standard black-box neural nets aren’t enough

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later recommended
- Designed for use together with an MLP definition/implementation compatible with this facade (such as `TMultiLayerPerceptron` shown in the source)

**Integration Steps:**
1. Place `FacadeMLP.pas` in your project directory.
2. Add it to your `uses` clause in your main program or an analysis tool:
    ```pascal
    uses FacadeMLP;
    ```
3. Instantiate your core `TMultiLayerPerceptron` model, then wrap it with the `TMLPFacade`:
    ```pascal
    var
      mlp: TMultiLayerPerceptron;
      facade: TMLPFacade;
    begin
      mlp := TMultiLayerPerceptron.Create( ... );
      facade := TMLPFacade.Create(mlp);
      // Now inspect/set neurons, layers, weights, etc.
    end;
    ```
4. Use the extensive API:
    - Query any neuron's weights, error, dropout, batchnorm stats
    - Adjust learning rates, regularization, optimizer state
    - Add/remove neurons/layers dynamically for research
    - Collect/bucketize outputs/histograms for diagnostics

#### Usage Notes

- This is not a standalone runnable file, but a powerful utility unit for hands-on control and analysis of MLPs in Pascal.
- Designed for deep ML experimentation, explainability, and teaching.
- For further details, see type declarations and implementation in the code; customize/extend as you wish for your own research!

---

### FacadeRNN: RNN (Recurrent Neural Network) Facade Unit

**File:** `FacadeRNN.pas`  
**Category:** Machine Learning Utilities / Recurrent Neural Networks

#### Description

A powerful Object Pascal unit (`unit RNNFacade`) providing a unified facade (API/class) for deep **introspection, manipulation, and research on Recurrent Neural Networks (RNNs)** of various kinds—including vanilla/SimpleRNN, LSTM, and GRU architectures.

This module is suitable for:
- Inspecting, extracting, or modifying all weights, gates, gradients, optimizer states, activations, and dropout at every layer and timestep
- Supporting multiple RNN cell types, loss/activation functions, and output layers
- Collecting histograms and diagnostic statistics (gate saturation, gradient scales) for debugging/visualization
- Accessing time-step caches, running states, and normalization/regularization properties
- Facilitating advanced research into sequence modeling, ablation studies, and explainability in deep learning

All code is Free Pascal (`{$mode objfpc}`) with modern types/conventions.

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or above recommended
- Designed for use inside a program that instantiates/interacts with RNNs, LSTMs, or GRUs compatible with this facade API

**Integration Steps:**
1. Place `FacadeRNN.pas` in your project directory.
2. Include it in your `uses` clause in your main program or research tool:
    ```pascal
    uses FacadeRNN;
    ```
3. Depending on your architecture, create the relevant cell wrappers or the `TRNNFacade` object. Example:
    ```pascal
    var
      rnn: TRNNFacade;
    begin
      rnn := TRNNFacade.Create(...);
      // Now use RNN API to read/write activations, gates, optimizer states, etc.
    end.
    ```
4. Use the extensive API to:
    - Inspect any gate (LSTM/GRU/Simple), activation, or error variable for any cell and timestep
    - Access/modify weights, gradients, dropout masks, normalization statistics
    - Run chained training/forward/backward passes and gather in-depth logs, stats, or visualize diagnostics

#### Usage Notes

- This is a reusable unit for advanced RNN experimentation and explainability—**not a runnable standalone program**.
- Designed for deep ML research, saliency inspection, and educational tracing of sequence architectures.
- For direct code/API walkthrough, consult inline type and class definitions.

---

### FacadeTransformer: Transformer Model Introspection and Manipulation Facade

**File:** `FacadeTransformer.pas`  
**Category:** Machine Learning / Transformers / Introspection Utilities

#### Description

A powerful Pascal unit providing an advanced **facade for introspection, inspection, and manipulation of transformer models** loaded from [`Transformer.pas`](#transformer-minimal-pascal-transformer-attention-based-model).  
This class is intended for research, educational, and debugging purposes—letting users deeply inspect attention, embeddings, parameters, internal network states, and even dynamically mutate the transformer architecture at runtime.

**Features:**
- Inspect internal model state:
  - Per-layer and per-head hidden activations, Q/K/V vectors
  - All attention logits and softmax weights for fine-grained attention analysis
  - Access embeddings (token and positional) and model hyperparameters
  - Dump weights, check structural layout, or mutate dimensions (add/remove layers/heads)
- Manipulate weights, positions, or intermediate activations
- Access/adjust key-value cache (for attention/memory states)
- Retrieve residual, layer norm, and FFN outputs per token and layer
- Run forward passes with full memory of activations for explainability
- Generate text or run prompts with fully visible intermediate state

**Intended Use Cases:**
- Explainability, visualization, and attribution in transformer models
- Fine-tuning, ablation studies, and architectural research
- Debugging/diagnostics at any stage in the model

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or above
- Load with a transformer model trained/exported to compatible GGUF format and paired tokenizer

**Integration Example:**
```pascal
uses FacadeTransformer;
var
  facade: TTransformerFacade;
begin
  facade := TTransformerFacade.Create;
  facade.LoadModel('model.gguf');
  facade.LoadTokenizer('tokenizer.json');
  // Forward a prompt, then inspect attention, QKV, activations, etc.
end.
```

**Workflow:**
- Run your prompt or batch, then retrieve desired state using accessors (e.g., `GetAttentionWeights`, `GetQKV`, `GetHiddenState`, `GetLogits`, etc).

#### Usage Notes

- Most useful as a "probe" or spike-in tool for model understanding and interpretability—pair it with visualizations or research loops.
- For basic model use or inference, use only [`Transformer.pas`](#transformer-minimal-pascal-transformer-attention-based-model).
- All major model architectural statistics and activations are accessible through the dedicated API.

---

### HeapBinaryTree: Binary Tree-based Heap Data Structure

**File:** `HeapBinaryTree.pas`  
**Category:** Data Structures / Heaps / Trees

#### Description

Implements a classic **binary tree-based heap** in Pascal, including all fundamental operations: insertion, deletion, pre-order/in-order/post-order traversal, and node search.  
Used as an educational/reference example for building a heap structure using explicit node pointers (`THeapBinaryTreeNode`) and emphasizing binary search tree-style data relationships.

**Key Features:**
- Pure Object Pascal implementation (`{$mode objfpc}`)
- Provides:
  - `insertData(inputData: integer)` – insert a new value into the heap/tree
  - `deleteNode(key: integer): boolean` – remove a node by its value
  - `countNodes()`: integer – total nodes via pre-order traversal
  - `printTree()` – prints the tree with indented, pre-order formatting
  - `findNodeNumber(key: integer): integer` – lookup node position for a value
- Modular, with all logic separated from the interactive or application-level I/O

Underlying nodes are managed with the companion `HeapBinaryTreeNode.pas` unit.

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later
- `HeapBinaryTreeNode.pas` must be in the same directory for compilation

**Compile (as part of a program or unit):**

1. To use as a component of a main program:
    ```pascal
    uses HeapBinaryTree;
    ```
    Compile/link with all required files.

2. To interactively test, add method calls to a program using this class:
    ```pascal
    var
      tree: THeapBinaryTree;
    begin
      tree := THeapBinaryTree.create;
      tree.insertData(5);
      tree.insertData(3);
      tree.insertData(8);
      tree.printTree;
    end.
    ```

**Or, compile with a provided test harness if present:**
```bash
fpc HeapBinaryTree.pas
# ...plus a test main program
```

#### Usage Notes

- Traversal methods (pre/in/post-order) can be extended or called directly for custom processing.
- `printTree` uses indentation to show tree structure—helpful for diagnostics or demos.
- Error/debug comments reflect development timeline and humor.
- All node management follows classic binary search tree rules.

---

### HeapBinaryTreeNode: Node Class for Binary Tree-based Heap

**File:** `HeapBinaryTreeNode.pas`  
**Category:** Data Structures / Heaps / Trees (Internal Node)

#### Description

This unit defines the `THeapBinaryTreeNode` class, the node data structure used internally by [`HeapBinaryTree.pas`](#heapbinarytree-binary-tree-based-heap-data-structure) for representing a binary tree-based heap.  
It encapsulates basic node fields and getter/setter methods for use in binary tree and heap algorithms, keeping node logic clearly separated from the main heap operations.  

**Class Features:**
- Fields for:
  - `data`: integer value contained in the node
  - `nodeNumber`: supporting sequential or logical enumeration of nodes
  - `leftChild`, `rightChild`: pointers to left and right children
- Methods for:
  - Setting/getting data, node number
  - Assigning/returning left and right children
- Constructor initializes all pointers to `nil` (empty node)

This tight encapsulation makes it easy to modify or extend the underlying data model (e.g., for balancing or additional attributes).

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later
- Used automatically by `HeapBinaryTree.pas`, not commonly run or compiled directly

**Integration Steps:**
1. Place `HeapBinaryTreeNode.pas` in your project directory.
2. Reference it in your heap or tree unit:
    ```pascal
    uses HeapBinaryTreeNode;
    ```
3. Use `THeapBinaryTreeNode.create` to spawn new nodes as needed in your own structures.
4. Access and modify node properties with the provided getters and setters.

**Typical Usage Example (with HeapBinaryTree):**
```pascal
var
  node: THeapBinaryTreeNode;
begin
  node := THeapBinaryTreeNode.create;
  node.setData(10);
  node.setNodeNumber(1);
  // Link into tree, as managed by HeapBinaryTree
end.
```

#### Usage Notes

- You generally do not need to interact with nodes directly—work at the heap/tree level unless implementing or extending the structure.
- This design promotes reusability and clarity in larger object-oriented Pascal projects.

---

### HeapDoubleLinkedList: Heap Using a Doubly Linked List

**File:** `HeapDoubleLinkedList.pas`  
**Category:** Data Structures / Heaps / Linked Lists

#### Description

Implements a **heap-like data structure using a doubly linked list** in Pascal for educational demonstration and practical applications that require ordering and bidirectional traversal.

**Key Features:**
- Object Pascal (`{$mode objfpc}`) using the companion `HeapDoubleNode.pas`
- Supports:
  - Insertion at the head (`insertFirst`) and tail (`insertLast`)
  - Deletion from the head (`deleteFirst`) and tail (`deleteLast`)
  - Deletion of the first node containing a specific value
  - Insertion after a node with a specific value (`insertAfter`)
  - Node data lookup by position (`returnSpecificNodesData`)
  - Counting total nodes (`countNodes`)
  - Cleanup helper (`destroyNodes`)

- All logic is encapsulated in the class and separated from program I/O

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later
- `HeapDoubleNode.pas` placed in the same directory

**Usage in a Program:**

1. Reference the unit and use the heap list as follows:
    ```pascal
    uses HeapDoubleLinkedList;
    var
      heap: THeapDoubleLinkedList;
    begin
      heap := THeapDoubleLinkedList.create;
      heap.insertFirst(10);
      heap.insertLast(20);
      heap.deleteFirst;
      heap.deleteLast;
      // Add further list operations as desired
    end.
    ```

2. Or, compile as part of a testing suite or with a main program.

**To Compile:**
```bash
fpc HeapDoubleLinkedList.pas
# ...plus a test or demo main program
```

#### Usage Notes

- All node linkage is via the `HeapDoubleNode` class—never manage pointers directly at the application level.
- Suits problems needing both ordered data and efficient insert/delete from both ends.

---

### HeapNode: Node Class for Heap Implemented via Linked List

**File:** `HeapNode.pas`  
**Category:** Data Structures / Heap / Linked Lists (Internal Node)

#### Description

This unit provides the single-node implementation for a heap (or any singly linked list-style structure).  
It is commonly used as the underlying node in linked-list based heaps (and similar structures), and is kept very simple for maximum clarity and extensibility.

**Class Features:**
- Fields:
  - `data`: integer value stored in this heap node
  - `next`: pointer to the next `THeapNode` in the list/heap structure
- Methods:
  - `setData(inputData: integer)` and `getData: integer` for value assignment and retrieval
  - `setNext(inputNode: THeapNode)` and `getNext: THeapNode` for pointer manipulation
- Constructor starts nodes with `next := nil`

This is perfect for basic heap/stack/queue/list exercises requiring your own node definitions.

#### How to Use

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later
- Used by heaps and lists which rely on their own node class

**General Steps:**
1. Place `HeapNode.pas` in your working directory.
2. Reference it in a unit or program:
    ```pascal
    uses HeapNode;
    ```
3. Create and connect nodes:
    ```pascal
    var
      node1, node2: THeapNode;
    begin
      node1 := THeapNode.create;
      node1.setData(5);
      node2 := THeapNode.create;
      node1.setNext(node2);
      // ... build up chain ...
    end.
    ```

#### Usage Notes

- In normal usage, higher-level data structure classes handle managing these nodes.
- This low-dependency node design is very flexible for learning, modifying, or extending your own linked data structures.

---

### MLP: MultiLayer Perceptron (Feedforward Neural Network)

**File:** `MLP.pas`  
**Category:** Machine Learning / Neural Networks

#### Description

A modern, full-featured MultiLayer Perceptron (MLP) neural network implementation in Object Pascal (`{$mode objfpc}`).  
This self-contained program demonstrates the creation, configuration, training, and prediction of classic feedforward neural networks—making it valuable for educational use, algorithm benchmarking, or direct integration in simple ML pipelines.

**Features:**
- Multiple hidden layers (`FHiddenLayers`), flexible layer sizes
- Choice of activation functions: Sigmoid, Tanh, ReLU, Softmax
- Choice of optimizers: SGD, Adam, RMSProp
- Implements dropout, L2 regularization, Xavier/He initialization
- Learning rate decay and early stopping
- Batch and online training, data normalization for stability
- Compact test harness: `program MLPtest`

**Core types & objects:**
- `TMultiLayerPerceptron` class with all major NN operations (forward, backward, optimizers, batch training)
- `TNeuron`, `TLayer`, `TDataPoint` record types
- Helper functions for transfer functions and data array management

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or above

**Compile:**
```bash
fpc MLP.pas
```

**Run:**
```bash
./MLPtest
```
_Note: Program entry point is named `MLPtest`._

**Editing and Experimentation:**
- Adjust layer/activation/optimizer configs by editing the `TMultiLayerPerceptron.Create` and field assignments in code.
- Train/test data can be embedded, generated, or supplied as needed by extending the `main` block.

#### Usage Notes

- Designed for students and researchers needing a readable, modifiable Pascal neural net.
- Strong numerical stability and regularization included out of the box.
- For research, couple this file with the `FacadeMLP.pas` for advanced introspection, training/weight logging, and architecture debugging.

---

### RNN: Advanced Recurrent Neural Network

**File:** `RNN.pas`  
**Category:** Machine Learning / Recurrent Neural Networks

#### Description

A comprehensive, advanced implementation of modern Recurrent Neural Networks (RNNs) in Object Pascal, including full support for classic SimpleRNN, LSTM, and GRU cell types.  
The code is structured for both research and education, featuring:

- Complete **forward and backward pass logic** for sequence learning (BPTT)
- Support for multiple cell types:  
  - Simple Vanila RNN  
  - Long Short-Term Memory (LSTM)  
  - Gated Recurrent Units (GRU)
- Customizable activation and loss function types
- Batch sequence and mini-batch training
- **Gradient clipping** for stabilizing deep training
- Modular, extensible classes for neuron cells, layers, and utility routines
- Includes layer and cell wrappers for easier experimentation and extension
- In-built random initialization, Xavier/He support, and utility normalization methods

The provided main program (`program AdvancedRNN;`) features demonstration of forward/backward/training logic and utility routines for initializing/testing the architecture.

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or newer

**Compile:**
```bash
fpc RNN.pas
```

**Run:**
```bash
./AdvancedRNN
```
_Note: The program’s entry point is named `AdvancedRNN`._

**Customization:**
- All cell, layer, and training settings can be configured at construction time with the constructors for each class (`TAdvancedRNN`, etc.).
- Extend or change the main block to load/generate different sequence data.

#### Usage Notes

- Perfect for education, prototyping, or algorithmic benchmarking of classic RNNs, LSTM, and GRU.
- To perform advanced introspection on RNN activations/gradients/gates, combine with [`FacadeRNN.pas`](#facadernn-rnn-recurrent-neural-network-facade-unit).

---

### RedBlackTree: Red-Black Self-Balancing Binary Search Tree

**File:** `RedBlackTree.pas`  
**Category:** Data Structures / Trees / Self-Balancing BST

#### Description

Implements a **Red-Black Tree**, a classic self-balancing binary search tree (BST) variant, in Pascal.  
Red-black trees guarantee logarithmic time for insertion, deletion, and lookup by enforcing strict color and rotation rules after every modification.

**Features:**
- Fully dynamic insertions with automatic rebalancing ("fixup")
- Node and tree balancing through color assignments and tree rotations (left/right)
- Object Pascal (`{$mode objfpc}`) style with in-memory pointer operations
- Traversal routine (`inorderTraversal`) demonstrates the result and coloring of nodes
- Construction is straightforward—suitable for both learning and practical use
- Simple and extendable, making it a strong starting point for exploring other BST variants

**Data Model:**
- Each node stores:
  - `data`: integer value in the node
  - `color`: either red or black
  - `left`, `right`, `parent`: pointers allowing bi-directional traversal and ancestry checks

- Helper functions:
  - `grandparent`, `uncle`, `sibling`: classic BST family accessors

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or newer

**Integration / Usage Example:**
1. Place `RedBlackTree.pas` in your working directory.
2. Reference in your main program:
    ```pascal
    uses RedBlackTree;
    var
      tree: TRedBlackTree;
      root: treeNode;
    begin
      tree.create;
      tree.insert(root, 10);
      tree.insert(root, 20);
      tree.insert(root, 15);
      tree.inorderTraversal(root);
    end.
    ```

**To Compile:**
```bash
fpc RedBlackTree.pas
# ...plus a main/test program
```

#### Usage Notes

- The supplied object makes the main tree manipulation interface very clean and Pascal-esque.
- All balancing, fixing, and coloring are handled automatically in `insert` and `insertFixup`.
- For deletion, you'll need to extend the implementation (only insertion provided).
- `inorderTraversal` prints each node in order and notes its color, offering an easy sanity check.

---

### Stack: Classic Stack (Array-based) Implementation

**File:** `Stack.pas`  
**Category:** Data Structures / Stack

#### Description

Implements a classic, fixed-size array-based stack in Object Pascal.  
This example demonstrates all the conventional stack operations—push, pop, peek, and checks for full or empty results—using a dynamically allocated array and a `TStack` object wrapper.

**Features:**
- Array-based storage (`stackArray`) for integers
- Dynamic max size set at creation (`TStack.create(maxSizeInput)`)
- Standard stack operations:
  - `push(inputNumber: integer)`
  - `pop(): integer`
  - `peek(): integer` (view top without popping)
  - `isEmpty(): boolean`
  - `isFull(): boolean`
- Simple, readable implementation for both educational use and real-world stack needs

**Data Model:**
- Stack size and top index are managed globally within the unit for all `TStack` objects (typical in teaching examples)
- The interface can easily be extended for generic type support or encapsulation

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later

**Usage Examples:**

1. Using as a unit in a program:
    ```pascal
    uses Stack;
    var
      stk: TStack;
    begin
      stk.create(10);
      stk.push(5);
      stk.push(9);
      writeln('Top is: ', stk.peek);
      stk.pop;
    end.
    ```

2. Or, compile together with a test main program:
    ```bash
    fpc Stack.pas
    # ...plus your main/test code
    ```

#### Usage Notes

- Array bounds and top index initialization (`top := 0`) may differ from some conventions: adapt for 0- or 1-based stacks as desired.
- For greater safety or flexibility in larger projects, wrap the stack state in records or use class-based design.
- For linked-list based stacks, see associated or companion units in the repo.

---

### StackBinaryTree: Stack Implemented using a Binary Tree

**File:** `StackBinaryTree.pas`  
**Category:** Data Structures / Stack / Trees

#### Description

Implements a stack structure using a binary tree as its storage model in Pascal.  
This approach demonstrates both binary search tree construction and how stack-like access may be mapped onto tree structures, making it useful both for illustrating traversal and for exploring hybrid data structures.

**Features:**
- Explicit pointer-based binary tree node definition (with `data`, `nodeNumber`, and left/right child pointers)
- Stack operations provided through a custom tree-based logic
- Core routines include:
  - `insertData(inputData: integer)` – insert a node following BST rules
  - `deleteNode(key: integer): boolean` – remove a node by its value
  - `countNodes(): integer` – total nodes using pre-order scan
  - `findNodeNumber(key: integer): integer` – returns the logical “stack position” of a value
  - Print and traverse tree visually with `printTree`, plus support for in/pre/post-order traversals

- Also includes an array-based object stack for trees (`TtreeStack`)
  - Allows for mixed array/tree approaches in algorithms that require both

**Data Model:**
- Tree is managed by root/global pointers for simplicity (educational)
- Nodes allocated/deallocated with `new` and direct pointer manipulation

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or later

**Example Usage:**
```pascal
uses StackBinaryTree;
var
  tree: TStackBinaryTree;
begin
  tree.create;
  tree.insertData(7); tree.insertData(2); tree.insertData(9);
  tree.printTree;
end.
```

**Or compile with your own main/test code:**
```bash
fpc StackBinaryTree.pas
# ...plus a test/demonstration program
```

#### Usage Notes

- Standard traversal methods are provided as customizable stubs: fill with your own processing for in/pre/post-order walks.
- Useful for teaching/research or for implementing exotic data structures that combine traversal and stack-like behavior.
- For pure stacks, see [Stack.pas](#stack-classic-stack-array-based-implementation).

---

### StackDoubleLinkedList: Stack Using a Doubly Linked List

**File:** `StackDoubleLinkedList.pas`  
**Category:** Data Structures / Stack / Linked Lists

#### Description

Implements a stack using a **doubly linked list** as its underlying storage, in Object Pascal.  
This file demonstrates all standard stack-like and list-like operations, with each node pointing both forwards and backwards, allowing flexible insertion and removal from either end.

**Features:**
- Explicit node structure (`doubleNode`) containing value, previous and next pointers
- Provides core doubly-linked list methods:
  - `insertFirst(inputData: integer)` – add value to the head
  - `insertLast(inputData: integer)` – add value to the tail
  - `deleteFirst()`/`deleteLast()` – remove item from front or end
  - `deleteNodeForFirstInstanceOfData(key: integer)` – removes the first matching value
  - `insertAfter(key, inputData: integer)` – insert after node with specific value
  - `returnSpecificNodesData(nodeNumber: integer)`: get data at nth node/position
  - `countNodes()`: total elements in the list

**Data Model:**
- Global `head` and `tail` pointers (classic Pascal teaching pattern)
- All links managed via pointer assignment—demonstrates how stacks/lists work at a pointer level

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or above

**Example Usage:**
```pascal
uses StackDoubleLinkedList;
var
  stack: TStackDoubleLinkedList;
begin
  stack.create;
  stack.insertFirst(10);
  stack.insertLast(20);
  writeln('Count: ', stack.countNodes());
  stack.deleteFirst;
end.
```

**Or compile with a separate test/demo main program:**
```bash
fpc StackDoubleLinkedList.pas
# ...plus main/test code
```

#### Usage Notes

- This structure allows both stack (LIFO) and queue (FIFO) behaviors, plus flexible node operations.
- For pure stack use, focus on `insertFirst`/`deleteFirst` for LIFO behavior.
- For simple, array-based stacks, see [Stack.pas](#stack-classic-stack-array-based-implementation).

---

### StackLinkedList: Stack Using a Linked List

**File:** `StackLinkedList.pas`  
**Category:** Data Structures / Stack / Linked List

#### Description

Implements a simple stack using a classic singly linked list as the underlying structure, in Object Pascal (`{$mode objfpc}`).  
Each node points to the next node, with the top of the stack corresponding to the head of the list—ideal for both stack and linear list teaching cases.

**Features:**
- Explicit pointer-based node structure (`data`, `next`)
- Provides core stack/list methods:
  - `addNode(inputData: integer)` – push value onto the end (tail) of the list
  - `deleteFirstNode()`/`deleteLastNode()` – remove node from head/tail
  - `deleteSpecificNode(nodeNumber: integer)` – remove nth node in list
  - `countNodes()` – returns total nodes present
  - `returnSpecificNodesData(nodeNumber: integer)` – retrieve the data at position n
  - `returnHeadsData()`, `returnTailsData()` – convenience methods for head/tail data
  - `returnNodeNumberOfFirstInstanceOfData(inputData: integer)` – find the index of first matching data

**Data Model:**
- Global `head` pointer; all node allocation/deallocation with explicit `new` and pointer assignment
- All methods are implemented with direct pointer manipulation and looping, in pure Pascal

#### How to Run

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or above

**Example Usage:**
```pascal
uses StackLinkedList;
var
  stack: TStackLinkedList;
begin
  stack.create;
  stack.addNode(1); stack.addNode(2); stack.addNode(3);
  writeln('Count: ', stack.countNodes());
  stack.deleteFirstNode;
end.
```

**Or compile with a test main program:**
```bash
fpc StackLinkedList.pas
# ...main/demo/test code
```

#### Usage Notes

- Suits teaching nodes, pointers, singly linked lists, and basic stack (LIFO) logic.
- For a doubly linked list or array stack, see other relevant units ([StackDoubleLinkedList.pas](#stackdoublelinkedlist-stack-using-a-doubly-linked-list), [Stack.pas](#stack-classic-stack-array-based-implementation)).

---

### Transformer: Minimal Pascal Transformer (Attention-based Model)

**File:** `Transformer.pas`  
**Category:** Machine Learning / Deep Learning / Transformers

#### Description

A compact and modern Pascal implementation of a **Transformer-based neural network model**, including self-attention and GGUF (GPT-style) model file loading/parsing.  
Designed as both a reference implementation and a working CLI demo for anyone seeking to understand or work with transformer networks in Pascal.

**Features:**
- Loads GGUF-format model weights for GPT and similar transformer architectures
- Implements fast, efficient tokenization with JSON support
- Full forward propagation for multi-layer transformers :  
  - Token embedding, multi-head self-attention, feed-forward, layer norm, GELU, and softmax
- End-to-end text generation ("prompting") via attention and autoregressive decoding
- Custom `TTokenizer`, `TGGUFLoader`, and `TTransformerModel` classes for clear separation of parsing, weights, and computation
- All code written in idiomatic object-oriented Pascal (`{$mode objfpc}` with advanced records)

**How to Run**

**Requirements:**
- Free Pascal Compiler (FPC), version 3.x or newer
- GGUF model and compatible tokenizer JSON

**Compile:**
```bash
fpc Transformer.pas
```

**Run:**
```bash
./Transformer
```

**Usage:**
- To prompt/generate text, load a GGUF model and compatible tokenizer file, then call `Generate(prompt, maxTokens)` from your Pascal code or from the command line (if adequately wired up).

**Package Structure:**
- `TTokenizer`: Loads and encodes/decodes text using JSON vocabulary file
- `TGGUFLoader`: Loads transformer layers, weights, and embeds from GGUF format
- `TTransformerModel`: Runs inference, generation, and handles all forward propagation

#### Usage Notes

- Not a full-featured LLM shell, but a robust Pascal starting point for experimenting with transformer networks, or for extending to educational or research use.
- You may adapt this for BERT, GPT, or other attention-based models by adjusting the forward pass or loader logic.
- Absolutely minimal external dependencies: only JSON/FPJSON components already in standard FPC.

---

**Attribution:**  
Created by Matthew James Abbott, 2025

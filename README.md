# Pascal Data Structures

A comprehensive repository of individual Pascal programs, each implementing a different data structure or algorithmic approach – from fundamentals to advanced.  
Ideal for students and enthusiasts interested in Pascal and classic CS topics.

---

## Contents

Below is a list of all included Pascal programs.  
Each entry links to an individual writeup and usage instructions.

### Source Files

- [CNN.pas: Convolutional Neural Network (Deep Learning)](#cnn-convolutional-neural-network)
- [DatastructureTest.pas: Comprehensive Data Structure Tester](#datastructuretest-comprehensive-data-structure-tester)
- [DatastructureTestResults.txt: Example Test Output](#datastructuretestresults-example-test-output)
- [FacadeCNN.pas: CNN Facade Unit for Introspection & Manipulation](#facadecnn-cnn-facade-unit-for-introspection--manipulation)

---

## Individual Program Writeups

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

**Attribution:**  
Created by Matthew James Abbott, 2025

---

_Add the next file alphabetically. To generate its dedicated section, just give me the filename!_

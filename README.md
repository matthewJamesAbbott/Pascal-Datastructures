# Pascal Data Structures

A comprehensive repository of individual Pascal programs, each implementing a different data structure or algorithmic approach – from fundamentals to advanced.  
Ideal for students and enthusiasts interested in Pascal and classic CS topics.

---

## Contents

Below is a list of all included Pascal programs.  
Each entry links to an individual writeup and usage instructions.

### Source Files

- [CNN.pas: Convolutional Neural Network (Deep Learning)](#cnn-convolutional-neural-network)

_(Add your other files alphabetically below as you go!)_

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

**Attribution:**  
Created by Matthew James Abbott, 2025

---

_Add the next file alphabetically. To generate its dedicated section, just give me the filename!_

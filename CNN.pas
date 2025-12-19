//
// Convolutional Neural Network Implementation
// With full backpropagation, softmax/cross-entropy, Adam optimizer
// and numerical stability fixes
//
// Matthew Abbott 2025
//

{$mode objfpc}
{$M+}

program CNNtest;

// Import Pascal standard libraries:
// Classes = OOP stuff, Math = math functions (sqrt, exp, etc), SysUtils = system utilities.
uses Classes, Math, SysUtils;

// -------------------------------
// Constants: for safety and config
// -------------------------------
const
   EPSILON = 1e-8; // Small value to avoid division by zero in optimizer
   GRAD_CLIP = 1.0; // Maximum gradient value to prevent exploding gradients

// -------------------------------
// Core Data Types: Arrays and Records
// -------------------------------

// Arrays for storing doubles (floating point numbers)
// Darray = 1D, D2array = 2D, D3array = 3D, D4array = 4D
type
   Darray = array of Double;
   D2array = array of array of Double;
   D3array = array of array of array of Double;
   D4array = array of array of array of array of Double;

// Used to represent an image in the neural network
   TImageData = record
      Width: Integer;      // Number of columns (pixels horizontally)
      Height: Integer;     // Number of rows (pixels vertically)
      Channels: Integer;   // Number of channels (e.g. 3 for RGB, 1 for grayscale)
      Data: D3array;       // Raw pixel data [channel][height][width]
   end;

// Used to represent a single data point (an image + its label/target, e.g. for training)
   TDataPoint = record
      Image: TImageData;   // The image itself
      Target: Darray;      // What should the model output? (one-hot encoded)
   end;

// Represents a single convolution filter/kernel for a conv layer
   TConvFilter = record
      Weights: D3array;    // The weight values for each channel in the filter
      Bias: Double;        // One bias value per filter
      // Adam optimizer state -- tracks moving averages for more stable updates
      WeightsM: D3array;   // First moment (mean) for Adam optimizer
      WeightsV: D3array;   // Second moment (uncentered variance/squared grad) for Adam
      BiasM: Double;       // Adam: mean for bias
      BiasV: Double;       // Adam: variance for bias
      // Gradients are stored in these variables during backpropagation
      WeightGrads: D3array; // Accumulated weight gradients
      BiasGrad: Double;      // Accumulated bias gradient
   end;

// Representation of a full convolutional layer (one set of filters)
   TConvLayer = record
      Filters: array of TConvFilter;  // All the filters in this layer
      OutputMaps: D3array;            // Output feature maps computed by the convs
      PreActivation: D3array;         // Conv output before activation function (ReLU)
      InputCache: D3array;            // Store input for use in backprop
      PaddedInput: D3array;           // Store padded input for backprop
      Stride: Integer;                // How far the filter slides at each step
      Padding: Integer;               // How many zero pixels pad the input along edges
      KernelSize: Integer;            // Size of each filter kernel (e.g. 3=3x3)
      InputChannels: Integer;         // Number of input feature/channel maps
   end;

// Represents a pooling layer (MaxPooling)
   TPoolingLayer = record
      PoolSize: Integer;                        // Pool size (typically 2: 2x2 pooling)
      Stride: Integer;                          // Slide the window by this much each time
      OutputMaps: D3array;                      // Outputs of the pooling layer
      InputCache: D3array;                      // Store pooled-over input for backprop
      MaxIndices: array of array of array of record X, Y: Integer; end; // Stores which pixel was the max in each window (for unpooling during backprop)
   end;

// Represents a single neuron in a fully connected (dense) layer
   TNeuron = record
      Weights: array of Double;    // Each neuron's weights for every input
      Bias: Double;                // Bias for this neuron
      Output: Double;              // What this neuron outputs (after activation)
      PreActivation: Double;       // Value before activation (e.g. ReLU)
      Error: Double;               // Error term (used for backpropagation)
      DropoutMask: Double;         // 0 (dropped), 1/(1-rate) (kept during dropout training)
      // Adam optimizer states - per weight/bias
      WeightsM: array of Double;   // Adam running avg: mean
      WeightsV: array of Double;   // Adam running avg: variance
      BiasM: Double;
      BiasV: Double;
   end;

// Represents a fully connected (dense) layer
   TFullyConnectedLayer = record
      Neurons: array of TNeuron; // Each neuron in the layer
      InputCache: Darray;        // Input for this layer during forward pass, saved for backprop
   end;

// The entire Convolutional Neural Network class
   TConvolutionalNeuralNetwork = class
   private
      // Parameters and state
      LearningRate: Double;         // How much we step the weights each iteration
      DropoutRate: Double;          // Fraction of neurons randomly ignored during training
      Beta1, Beta2: Double;         // Adam optimizer settings
      AdamT: Integer;               // Adam time step (for bias correction)
      IsTraining: Boolean;          // Track if we're training or just predicting

      // The network structure
      ConvLayers: array of TConvLayer;               // List of all conv layers
      PoolLayers: array of TPoolingLayer;            // List of all pooling layers
      FullyConnectedLayers: array of TFullyConnectedLayer; // Dense layers before output
      OutputLayer: TFullyConnectedLayer;             // Output (final) fully connected layer

      FlattenedSize: Integer;                        // Number of features after conv/pooling/flatten
      FlattenedFeatures: Darray;                     // Store flattened layer output
      LastConvHeight, LastConvWidth, LastConvChannels: Integer; // To reshape gradients correctly

      // Layer initialization functions
      procedure InitializeConvLayer(var Layer: TConvLayer; NumFilters: Integer;
                                   InputChannels: Integer; KernelSize: Integer;
                                   Stride: Integer; Padding: Integer);
      procedure InitializePoolLayer(var Layer: TPoolingLayer; PoolSize: Integer; Stride: Integer);
      procedure InitializeFCLayer(var Layer: TFullyConnectedLayer; NumNeurons: Integer; NumInputs: Integer);

      // Forward pass functions for each layer type
      procedure ConvForward(var Layer: TConvLayer; const Input: D3array;
                           InputWidth: Integer; InputHeight: Integer);
      procedure PoolForward(var Layer: TPoolingLayer; const Input: D3array;
                           InputWidth: Integer; InputHeight: Integer);
      procedure FlattenFeatures(const Input: D3array; InputWidth: Integer;
                               InputHeight: Integer; InputChannels: Integer);
      procedure FCForward(var Layer: TFullyConnectedLayer; const Input: Darray);

      // Backward pass functions for each layer
      function ConvBackward(var Layer: TConvLayer; const Grad: D3array): D3array;
      function PoolBackward(var Layer: TPoolingLayer; const Grad: D3array): D3array;
      function FCBackward(var Layer: TFullyConnectedLayer; const Grad: Darray; IsOutputLayer: Boolean): Darray;
      function UnflattenGradient(const Grad: Darray): D3array;

      // Parameter update helper and utility functions
      procedure UpdateWeights;
      procedure ApplyDropout(var Layer: TFullyConnectedLayer);

      // Common neural net and math functions
      function ReLU(x: Double): Double;
      function ReLUDerivative(x: Double): Double;
      function Softmax(const Logits: Darray): Darray;
      function CrossEntropyLoss(const Predicted, Target: Darray): Double;
      function ClipGrad(x: Double): Double;
      function IsFiniteNum(x: Double): Boolean;
      function Clamp(x, MinVal, MaxVal: Double): Double;
      function Pad3D(const Input: D3array; Padding: Integer): D3array;
      function ValidateInput(const Image: TImageData): Boolean;

   public
      // Factory: builds the network with user-provided sizes/hyperparameters
      constructor Create(InputWidth: Integer; InputHeight: Integer; InputChannels: Integer;
                        ConvFilters: array of Integer; KernelSizes: array of Integer;
                        PoolSizes: array of Integer; FCLayerSizes: array of Integer;
                        OutputSize: Integer; ALearningRate: Double = 0.001;
                        ADropoutRate: Double = 0.25);
      // For use after training: returns softmax probabilities
      function Predict(var Image: TImageData): Darray;
      // Performs a single train step: forward, compute loss, backward, update weights
      function TrainStep(var Image: TImageData; const Target: Darray): Double;
      // Persist/load model
      procedure SaveCNNModel(const Filename: string);
      procedure LoadCNNModel(const Filename: string);
   end;

// ---------------------------------------------------------------------
// Create: Set up a fresh network instance, configuring all layers/weights
// ---------------------------------------------------------------------
constructor TConvolutionalNeuralNetwork.Create(InputWidth: Integer; InputHeight: Integer;
                                              InputChannels: Integer;
                                              ConvFilters: array of Integer;
                                              KernelSizes: array of Integer;
                                              PoolSizes: array of Integer;
                                              FCLayerSizes: array of Integer;
                                              OutputSize: Integer;
                                              ALearningRate: Double;
                                              ADropoutRate: Double);
var
   i: Integer;           // Loop variable
   CurrentWidth, CurrentHeight, CurrentChannels: Integer; // Track image size (shrinks as we convolve/pool)
   NumInputs: Integer;   // How many inputs does each FC layer need?
   KernelPadding: Integer;
begin
   LearningRate := ALearningRate;
   DropoutRate := ADropoutRate;
   Beta1 := 0.9;        // Adam optimizer hyperparameter
   Beta2 := 0.999;      // Adam optimizer hyperparameter
   AdamT := 0;
   IsTraining := False; // Default: we're not training

   CurrentWidth := InputWidth;
   CurrentHeight := InputHeight;
   CurrentChannels := InputChannels;

   SetLength(ConvLayers, Length(ConvFilters)); // # Conv layers = # elements in ConvFilters
   SetLength(PoolLayers, Length(PoolSizes));   // # Pool layers = # elements in PoolSizes

   // For each convolutional layer to be created:
   for i := 0 to High(ConvFilters) do
   begin
      KernelPadding := KernelSizes[i] div 2; // "Same" padding: (K-1)//2 to keep output size
      InitializeConvLayer(ConvLayers[i], ConvFilters[i], CurrentChannels,
                         KernelSizes[i], 1, KernelPadding);

      // Update output image size after convolution (with stride and padding)
      CurrentWidth := (CurrentWidth - KernelSizes[i] + 2 * KernelPadding) div 1 + 1;
      CurrentHeight := (CurrentHeight - KernelSizes[i] + 2 * KernelPadding) div 1 + 1;
      CurrentChannels := ConvFilters[i];

      // If there’s a pool layer after this conv layer, initialize it in PoolLayers
      if i <= High(PoolSizes) then
      begin
         InitializePoolLayer(PoolLayers[i], PoolSizes[i], PoolSizes[i]);
         // Pooling reduces spatial size by integer division (e.g. 2x2 halves it)
         CurrentWidth := CurrentWidth div PoolSizes[i];
         CurrentHeight := CurrentHeight div PoolSizes[i];
      end;
   end;

   FlattenedSize := CurrentWidth * CurrentHeight * CurrentChannels; // Number of 1D features after all pooling

   SetLength(FullyConnectedLayers, Length(FCLayerSizes)); // Create the FC layers. Each may have a different # of neurons
   NumInputs := FlattenedSize; // First FC takes all pixel features

   for i := 0 to High(FCLayerSizes) do
   begin
      InitializeFCLayer(FullyConnectedLayers[i], FCLayerSizes[i], NumInputs); // Each FC needs to know how many inputs
      NumInputs := FCLayerSizes[i]; // Output size of this FC is input size to the next
   end;

   InitializeFCLayer(OutputLayer, OutputSize, NumInputs); // Output FC layer (need this even for regression)
end;

// ---------------------------------------------------------------------
// Layer Initializers
// ---------------------------------------------------------------------

// Set up a convolutional layer, allocating memory and initializing weights/biases/moments
procedure TConvolutionalNeuralNetwork.InitializeConvLayer(var Layer: TConvLayer;
                                                         NumFilters: Integer;
                                                         InputChannels: Integer;
                                                         KernelSize: Integer;
                                                         Stride: Integer;
                                                         Padding: Integer);
var
   i, j, k, l: Integer;
   Scale: Double;
begin
   SetLength(Layer.Filters, NumFilters); // List of filters
   Layer.Stride := Stride;
   Layer.Padding := Padding;
   Layer.KernelSize := KernelSize;
   Layer.InputChannels := InputChannels;

   // He initialization (good for ReLU): weights ~ N(0, sqrt(2.0/n))
   Scale := Sqrt(2.0 / (InputChannels * KernelSize * KernelSize));

   for i := 0 to NumFilters - 1 do
   begin
      // Each filter has its own 3D weight block: [channels][k][k]
      SetLength(Layer.Filters[i].Weights, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightsM, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightsV, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightGrads, InputChannels, KernelSize, KernelSize);

      // Initialize: random weights, zero moments/gradients
      for j := 0 to InputChannels - 1 do
         for k := 0 to KernelSize - 1 do
            for l := 0 to KernelSize - 1 do
            begin
               Layer.Filters[i].Weights[j][k][l] := (Random - 0.5) * Scale; // Small random
               Layer.Filters[i].WeightsM[j][k][l] := 0;
               Layer.Filters[i].WeightsV[j][k][l] := 0;
               Layer.Filters[i].WeightGrads[j][k][l] := 0;
            end;

      Layer.Filters[i].Bias := 0.0;    // Bias initialize to zero
      Layer.Filters[i].BiasM := 0.0;   // Adam mean/variance zeros
      Layer.Filters[i].BiasV := 0.0;
      Layer.Filters[i].BiasGrad := 0.0;
   end;
end;

// Setup pooling layer (not much: just size/stride)
procedure TConvolutionalNeuralNetwork.InitializePoolLayer(var Layer: TPoolingLayer;
                                                         PoolSize: Integer;
                                                         Stride: Integer);
begin
   Layer.PoolSize := PoolSize;
   Layer.Stride := Stride;
end;

// Set up fully connected layers with neurons, initializes random weights, biases and Adam optimizer arrays/moments
procedure TConvolutionalNeuralNetwork.InitializeFCLayer(var Layer: TFullyConnectedLayer;
                                                       NumNeurons: Integer;
                                                       NumInputs: Integer);
var
   i, j: Integer;
   Scale: Double;
begin
   SetLength(Layer.Neurons, NumNeurons); // Allocate all neurons
   Scale := Sqrt(2.0 / NumInputs);       // He initialization again (good for ReLU), prevents dead neurons

   for i := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[i].Weights, NumInputs);  // Each neuron connects to all inputs
      SetLength(Layer.Neurons[i].WeightsM, NumInputs); // Adam: first moment (mean)
      SetLength(Layer.Neurons[i].WeightsV, NumInputs); // Adam: second moment (variance)

      for j := 0 to NumInputs - 1 do
      begin
         Layer.Neurons[i].Weights[j] := (Random - 0.5) * Scale; // Random initialization
         Layer.Neurons[i].WeightsM[j] := 0; // Adam zero-initialized
         Layer.Neurons[i].WeightsV[j] := 0;
      end;

      Layer.Neurons[i].Bias := 0.0; // Bias zero
      Layer.Neurons[i].BiasM := 0.0;
      Layer.Neurons[i].BiasV := 0.0;
      Layer.Neurons[i].Output := 0;
      Layer.Neurons[i].PreActivation := 0;
      Layer.Neurons[i].Error := 0;
      Layer.Neurons[i].DropoutMask := 1.0; // No dropout at initialization
   end;
end;

// ---------------------------------------------------------------------
// Math Utility functions --- For activation, gradient clipping, etc
// ---------------------------------------------------------------------

// Is this number a proper finite floating point (not inf, not NaN)?
function TConvolutionalNeuralNetwork.IsFiniteNum(x: Double): Boolean;
begin
   Result := (not IsNan(x)) and (not IsInfinite(x));
end;

// Clamp x within [MinVal, MaxVal], prevents out-of-range results (use with e.g. exponentials or probabilities)
function TConvolutionalNeuralNetwork.Clamp(x, MinVal, MaxVal: Double): Double;
begin
   if x < MinVal then
      Result := MinVal
   else if x > MaxVal then
      Result := MaxVal
   else
      Result := x;
end;

// If x is not finite, return 0. Otherwise, clamp it to [-GRAD_CLIP, +GRAD_CLIP] for gradient stabilizing
function TConvolutionalNeuralNetwork.ClipGrad(x: Double): Double;
begin
   if not IsFiniteNum(x) then
      Result := 0
   else
      Result := Clamp(x, -GRAD_CLIP, GRAD_CLIP);
end;

// The standard ReLU activation: f(x) = x if x>0, else 0
function TConvolutionalNeuralNetwork.ReLU(x: Double): Double;
begin
   if x > 0 then
      Result := x
   else
      Result := 0.0;
end;

// The derivative of ReLU (f'(x)), used in gradient calculations
function TConvolutionalNeuralNetwork.ReLUDerivative(x: Double): Double;
begin
   if x > 0 then
      Result := 1.0
   else
      Result := 0.0;
end;

// Softmax: turns arbitrary real numbers (logits) into a probability distribution
function TConvolutionalNeuralNetwork.Softmax(const Logits: Darray): Darray;
var
   i: Integer;
   MaxVal, Sum, ExpVal: Double;
begin
   SetLength(Result, Length(Logits));

   // Find the maximum value in logits vector (for numerical stability: subtracts this before applying exponent)
   MaxVal := -1e308;
   for i := 0 to High(Logits) do
      if IsFiniteNum(Logits[i]) and (Logits[i] > MaxVal) then
         MaxVal := Logits[i];

   if not IsFiniteNum(MaxVal) then
      MaxVal := 0;

   // Sum up all exp(logit - MaxVal). Clamp range so numbers don't overflow!
   Sum := 0;
   for i := 0 to High(Logits) do
   begin
      if IsFiniteNum(Logits[i]) then
         ExpVal := Exp(Clamp(Logits[i] - MaxVal, -500, 500)) // Clamp to avoid exp overflow
      else
         ExpVal := Exp(0); // Fallback
      Result[i] := ExpVal;
      Sum := Sum + ExpVal;
   end;

   if (Sum <= 0) or (not IsFiniteNum(Sum)) then
      Sum := 1;

   // Divide each exp(logit) by total sum, ensures output always adds to 1
   for i := 0 to High(Result) do
      Result[i] := Clamp(Result[i] / Sum, 1e-15, 1 - 1e-15); // Clamp output for numeric safety
end;

// Cross-entropy: computes the loss (difference) between what was predicted and what should have been predicted
function TConvolutionalNeuralNetwork.CrossEntropyLoss(const Predicted, Target: Darray): Double;
var
   i: Integer;
   P: Double;
begin
   Result := 0;
   for i := 0 to High(Target) do
   begin
      if Target[i] > 0 then
      begin
         P := Clamp(Predicted[i], 1e-15, 1 - 1e-15); // Don't let log(P) go to -∞
         Result := Result - Target[i] * Ln(P);
      end;
   end;

   if not IsFiniteNum(Result) then
      Result := 0;
end;

// --------------------
// Pad3D: Pads the input array with additional zeros around the edges
// Used to keep spatial dimensions after "same" convolutions
// --------------------
function TConvolutionalNeuralNetwork.Pad3D(const Input: D3array; Padding: Integer): D3array;
var
   c, h, w: Integer;
   Channels, Height, Width: Integer;
   SrcH, SrcW: Integer;
begin
   // If padding is zero, just return input immediately
   if Padding = 0 then
   begin
      Result := Input;
      Exit;
   end;

   // Determine the sizes for allocation
   Channels := Length(Input);
   Height := Length(Input[0]);
   Width := Length(Input[0][0]);

   // Allocate a result array with larger dimensions
   SetLength(Result, Channels, Height + 2 * Padding, Width + 2 * Padding);

   // Loop through each "slot" in our new large array, copying input if it's inside, otherwise 0
   for c := 0 to Channels - 1 do
      for h := 0 to Height + 2 * Padding - 1 do
         for w := 0 to Width + 2 * Padding - 1 do
         begin
            SrcH := h - Padding; // Where does this slot map in the original input?
            SrcW := w - Padding;
            if (SrcH >= 0) and (SrcH < Height) and (SrcW >= 0) and (SrcW < Width) then
               Result[c][h][w] := Input[c][SrcH][SrcW]
            else
               Result[c][h][w] := 0;
         end;
end;

// --------------------
// ValidateInput: Check that the image data is properly formatted and not broken
// Prevents weird bugs and crashes during training!
// --------------------
function TConvolutionalNeuralNetwork.ValidateInput(const Image: TImageData): Boolean;
var
   c, h, w: Integer;
begin
   Result := False; // Default to not valid

   // Outer check: nil or unexpected channel count
   if (Image.Data = nil) or (Length(Image.Data) <> Image.Channels) then
      Exit;

   // Check each channel, each row, each pixel
   for c := 0 to Image.Channels - 1 do
   begin
      if (Image.Data[c] = nil) or (Length(Image.Data[c]) <> Image.Height) then
         Exit;
      for h := 0 to Image.Height - 1 do
      begin
         if (Image.Data[c][h] = nil) or (Length(Image.Data[c][h]) <> Image.Width) then
            Exit;
         for w := 0 to Image.Width - 1 do
            if not IsFiniteNum(Image.Data[c][h][w]) then // NaN or inf
               Exit;
      end;
   end;

   Result := True; // All tests passed, input is good!
end;

// --------------------
// ApplyDropout: Turns off ("drops") neurons at random during training
// This helps prevent overfitting (so the net doesn't memorize its data!)
// Each neuron is either 'kept' (scaling its output so mean stays same) or 'dropped' (output zero)
// ONLY used during training, not during prediction
// --------------------
procedure TConvolutionalNeuralNetwork.ApplyDropout(var Layer: TFullyConnectedLayer);
var
   i: Integer;
begin
   for i := 0 to High(Layer.Neurons) do
   begin
      // IsTraining is true ONLY during training, and DropoutRate > 0 means "use it"
      if IsTraining and (DropoutRate > 0) then
      begin
         // If Random > DropoutRate, we keep this neuron
         if Random > DropoutRate then
            Layer.Neurons[i].DropoutMask := 1.0 / (1.0 - DropoutRate)  // Scale to keep expected value unchanged
         else
            Layer.Neurons[i].DropoutMask := 0; // Drop completely
      end
      else
         Layer.Neurons[i].DropoutMask := 1.0; // If not training: keep all neurons
   end;
end;

// -------------------------------------------------------------------------
// Forward passes: these functions run the inputs through the network
// Each layer type has its own way to compute outputs
// -------------------------------------------------------------------------

// Convolutional layer forward pass: applies all filters to the input feature maps
procedure TConvolutionalNeuralNetwork.ConvForward(var Layer: TConvLayer;
                                                 const Input: D3array;
                                                 InputWidth: Integer;
                                                 InputHeight: Integer);
var
   f, c, h, w, kh, kw: Integer;
   OutputWidth, OutputHeight: Integer;
   Sum: Double;
   InH, InW: Integer;
begin
   Layer.InputCache := Input;             // Store input for backprop use
   Layer.PaddedInput := Pad3D(Input, Layer.Padding); // Add zeros around edges if needed

   // Calculate the output size:
   OutputWidth := (InputWidth + 2 * Layer.Padding - Layer.KernelSize) div Layer.Stride + 1;
   OutputHeight := (InputHeight + 2 * Layer.Padding - Layer.KernelSize) div Layer.Stride + 1;

   // Allocate memory for output
   SetLength(Layer.OutputMaps, Length(Layer.Filters), OutputHeight, OutputWidth);
   SetLength(Layer.PreActivation, Length(Layer.Filters), OutputHeight, OutputWidth);

   // For every filter (output channel)
   for f := 0 to High(Layer.Filters) do
   begin
      // For every spatial location (h,w) in the output map:
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
            Sum := Layer.Filters[f].Bias; // Start with the bias

            // "Convolve": For every input channel, and every element in the KxK kernel:
            for c := 0 to Layer.InputChannels - 1 do
            begin
               for kh := 0 to Layer.KernelSize - 1 do
               begin
                  for kw := 0 to Layer.KernelSize - 1 do
                  begin
                     InH := h * Layer.Stride + kh; // Where to look in padded input
                     InW := w * Layer.Stride + kw;
                     // Accumulate weighted sum of inputs
                     Sum := Sum + Layer.PaddedInput[c][InH][InW] *
                            Layer.Filters[f].Weights[c][kh][kw];
                  end;
               end;
            end;

            if not IsFiniteNum(Sum) then
               Sum := 0; // Protect against overflow and NaN

            Layer.PreActivation[f][h][w] := Sum; // Save before activation
            Layer.OutputMaps[f][h][w] := ReLU(Sum); // Most CNNs use ReLU
         end;
      end;
   end;
end;

// Pooling (usually max pooling): shrinks image by selecting the max in each region
procedure TConvolutionalNeuralNetwork.PoolForward(var Layer: TPoolingLayer;
                                                 const Input: D3array;
                                                 InputWidth: Integer;
                                                 InputHeight: Integer);
var
   c, h, w, ph, pw: Integer;
   OutputWidth, OutputHeight: Integer;
   MaxVal, Val: Double;
   MaxPH, MaxPW: Integer;
begin
   Layer.InputCache := Input; // For backprop

   OutputWidth := InputWidth div Layer.PoolSize;
   OutputHeight := InputHeight div Layer.PoolSize;

   SetLength(Layer.OutputMaps, Length(Input), OutputHeight, OutputWidth);
   SetLength(Layer.MaxIndices, Length(Input), OutputHeight, OutputWidth);

   // For every channel in the input
   for c := 0 to High(Input) do
   begin
      // For each spot in the pooled output
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
            // Find the max value in the PoolSize x PoolSize patch
            MaxVal := -1e308;
            MaxPH := 0;
            MaxPW := 0;

            for ph := 0 to Layer.PoolSize - 1 do
            begin
               for pw := 0 to Layer.PoolSize - 1 do
               begin
                  Val := Input[c][h * Layer.PoolSize + ph][w * Layer.PoolSize + pw];
                  if Val > MaxVal then
                  begin
                     MaxVal := Val;
                     MaxPH := ph;
                     MaxPW := pw;
                  end;
               end;
            end;

            Layer.OutputMaps[c][h][w] := MaxVal;                // Record max-pooling result
            Layer.MaxIndices[c][h][w].Y := MaxPH;               // Where the max was for unpooling
            Layer.MaxIndices[c][h][w].X := MaxPW;
         end;
      end;
   end;
end;

// FlattenFeatures: converts 3D output of convolution/pooling (channels, h, w)
// into a single flat 1D vector for use with fully connected (dense) layers.
procedure TConvolutionalNeuralNetwork.FlattenFeatures(const Input: D3array;
                                                     InputWidth: Integer;
                                                     InputHeight: Integer;
                                                     InputChannels: Integer);
var
   c, h, w, idx: Integer;
begin
   LastConvWidth := InputWidth;     // Store for reversing this operation during backprop
   LastConvHeight := InputHeight;
   LastConvChannels := InputChannels;

   SetLength(FlattenedFeatures, InputChannels * InputHeight * InputWidth); // Allocate
   idx := 0; // Next index in 1D array

   // Loop over all positions and fill flat array
   for c := 0 to InputChannels - 1 do
      for h := 0 to InputHeight - 1 do
         for w := 0 to InputWidth - 1 do
         begin
            FlattenedFeatures[idx] := Input[c][h][w];
            Inc(idx);
         end;
end;

// UnflattenGradient: does the OPPOSITE of FlattenFeatures—
// takes a flat gradient (from FC layers) and reshapes as 3D for conv backprop
function TConvolutionalNeuralNetwork.UnflattenGradient(const Grad: Darray): D3array;
var
   c, h, w, idx: Integer;
begin
   SetLength(Result, LastConvChannels, LastConvHeight, LastConvWidth); // Allocate correct 3D shape
   idx := 0;
   for c := 0 to LastConvChannels - 1 do
      for h := 0 to LastConvHeight - 1 do
         for w := 0 to LastConvWidth - 1 do
         begin
            Result[c][h][w] := Grad[idx];
            Inc(idx);
         end;
end;

// Fully connected (dense) layer forward pass
procedure TConvolutionalNeuralNetwork.FCForward(var Layer: TFullyConnectedLayer;
                                               const Input: Darray);
var
   i, j: Integer;
   Sum: Double;
begin
   Layer.InputCache := Copy(Input); // Save input for use during backprop
   ApplyDropout(Layer);             // Set DropoutMask for every neuron (for training safety)

   for i := 0 to High(Layer.Neurons) do
   begin
      Sum := Layer.Neurons[i].Bias; // Start with bias

      // Weighted sum of all inputs
      for j := 0 to High(Input) do
         Sum := Sum + Input[j] * Layer.Neurons[i].Weights[j];

      if not IsFiniteNum(Sum) then
         Sum := 0; // Avoid numeric instability

      Layer.Neurons[i].PreActivation := Sum;       // Save sum before activation
      // Output = ReLU(Sum) * DropoutMask (DropoutMask is 1 except during training)
      Layer.Neurons[i].Output := ReLU(Sum) * Layer.Neurons[i].DropoutMask;
   end;
end;

// ------------------------------------------------------------
// Predict: Forwards a new image through the whole network (no training!)
// Used for testing and final deployment
// ------------------------------------------------------------
function TConvolutionalNeuralNetwork.Predict(var Image: TImageData): Darray;
var
   i, j: Integer;
   CurrentOutput: D3array;
   CurrentWidth, CurrentHeight: Integer;
   LayerInput: Darray;
   Logits: Darray;
   Sum: Double;
begin
   IsTraining := False; // Make sure dropout is NOT applied

   CurrentOutput := Image.Data;
   CurrentWidth := Image.Width;
   CurrentHeight := Image.Height;

   // --------- Forward Pass All Convolution + Pool Layers --------
   for i := 0 to High(ConvLayers) do
   begin
      ConvForward(ConvLayers[i], CurrentOutput, CurrentWidth, CurrentHeight);
      CurrentOutput := ConvLayers[i].OutputMaps;
      CurrentWidth := Length(CurrentOutput[0][0]);
      CurrentHeight := Length(CurrentOutput[0]);

      // If a pooling layer follows, apply it
      if i <= High(PoolLayers) then
      begin
         PoolForward(PoolLayers[i], CurrentOutput, CurrentWidth, CurrentHeight);
         CurrentOutput := PoolLayers[i].OutputMaps;
         CurrentWidth := Length(CurrentOutput[0][0]);
         CurrentHeight := Length(CurrentOutput[0]);
      end;
   end;

   // ---------- Flatten to 1D for Dense Layers ----------
   FlattenFeatures(CurrentOutput, CurrentWidth, CurrentHeight, Length(CurrentOutput));

   LayerInput := FlattenedFeatures; // Start with full flattened vector

   // ------ All (Hidden) Fully Connected Layers ---------
   for i := 0 to High(FullyConnectedLayers) do
   begin
      FCForward(FullyConnectedLayers[i], LayerInput);
      SetLength(LayerInput, Length(FullyConnectedLayers[i].Neurons));
      for j := 0 to High(LayerInput) do
         LayerInput[j] := FullyConnectedLayers[i].Neurons[j].Output;
   end;

   // ------ Output (logits) computation ------
   OutputLayer.InputCache := Copy(LayerInput); // Save for backprop
   SetLength(Logits, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to High(LayerInput) do
         Sum := Sum + LayerInput[j] * OutputLayer.Neurons[i].Weights[j];

      if not IsFiniteNum(Sum) then
         Sum := 0;

      OutputLayer.Neurons[i].PreActivation := Sum;
      Logits[i] := Sum;
   end;

   // Softmax: returns i-th class with probability in [0,1], all sum to 1
   Result := Softmax(Logits);
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Output := Result[i]; // Save output for loss/backprop
end;

// ------------------------------------------------------------
// Fully Connected Layer Backward:
// Propagates error gradient BACKWARDS through the dense layer
// Fills .Error fields (used for adjusting weights w/ Adam)
// If IsOutputLayer is true, skip activation gradient (BC softmax+cross-entropy has a simple gradient)
// ------------------------------------------------------------
function TConvolutionalNeuralNetwork.FCBackward(var Layer: TFullyConnectedLayer;
                                               const Grad: Darray;
                                               IsOutputLayer: Boolean): Darray;
var
   i, j: Integer;
   Delta: Double;
begin
   SetLength(Result, Length(Layer.InputCache));
   for i := 0 to High(Result) do
      Result[i] := 0; // Start with 0s

   for i := 0 to High(Layer.Neurons) do
   begin
      // Compute error to be sent back for each neuron in the layer
      if IsOutputLayer then
         Delta := Grad[i] // Output gradient is already correct for softmax+cross-entropy
      else
         // Else, chain rule: multiply by gradient of activation function (derivative of ReLU)
         Delta := Grad[i] * ReLUDerivative(Layer.Neurons[i].PreActivation) *
                  Layer.Neurons[i].DropoutMask; // Take dropout into account

      Layer.Neurons[i].Error := Delta;  // Store for weights update later

      // Propagate error backwards to previous layer's neurons (weight^T * error)
      for j := 0 to High(Layer.Neurons[i].Weights) do
         Result[j] := Result[j] + Delta * Layer.Neurons[i].Weights[j];
   end;
end;

// ------------------------------------------------------------
// PoolBackward: Backward pass for max pooling
// Distributes gradient only to the original "max" pixel that was picked
// Other locations in the pooling window get zero grad
// ------------------------------------------------------------
function TConvolutionalNeuralNetwork.PoolBackward(var Layer: TPoolingLayer;
                                                  const Grad: D3array): D3array;
var
   c, h, w: Integer;
   InH, InW: Integer;
   SrcH, SrcW: Integer;
begin
   // Allocate gradient for the original input shape
   SetLength(Result, Length(Layer.InputCache), Length(Layer.InputCache[0]),
             Length(Layer.InputCache[0][0]));

   // Start off with all zeros (accumulates gradients only where max was picked)
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;

   // For each output location in the grad, send it only to its "winning" input pixel
   for c := 0 to High(Grad) do
   begin
      for h := 0 to High(Grad[c]) do
      begin
         for w := 0 to High(Grad[c][h]) do
         begin
            // Which pixel got picked as max during forward?
            SrcH := h * Layer.PoolSize + Layer.MaxIndices[c][h][w].Y;
            SrcW := w * Layer.PoolSize + Layer.MaxIndices[c][h][w].X;
            Result[c][SrcH][SrcW] := Grad[c][h][w]; // Assign backprop error to the right pixel
         end;
      end;
   end;
end;

// ------------------------------------------------------------
// ConvBackward: Calculate gradients for convolutional layer's weights, bias, and input
// This computes how the loss changes w.r.t. every parameter in the conv layer
// ------------------------------------------------------------
function TConvolutionalNeuralNetwork.ConvBackward(var Layer: TConvLayer;
                                                  const Grad: D3array): D3array;
var
   f, c, h, w, kh, kw: Integer;
   GradWithReLU: D3array;
   GradSum, WGrad: Double;
   InH, InW: Integer;
   OutH, OutW: Integer;
begin
   // 1. Apply ReLU gradient to the incoming grad
   SetLength(GradWithReLU, Length(Grad), Length(Grad[0]), Length(Grad[0][0]));
   for f := 0 to High(Grad) do
      for h := 0 to High(Grad[f]) do
         for w := 0 to High(Grad[f][h]) do
            GradWithReLU[f][h][w] := Grad[f][h][w] *
                                     ReLUDerivative(Layer.PreActivation[f][h][w]);

   // 2. Compute and store gradients for the filters (weights and bias)
   for f := 0 to High(Layer.Filters) do
   begin
      // Bias gradient: sum over grad for all output pixels in filter f
      GradSum := 0;
      for h := 0 to High(GradWithReLU[f]) do
         for w := 0 to High(GradWithReLU[f][h]) do
            GradSum := GradSum + GradWithReLU[f][h][w];
      Layer.Filters[f].BiasGrad := GradSum;

      // Weight gradients: track how much each weight should be nudged
      for c := 0 to Layer.InputChannels - 1 do
      begin
         for kh := 0 to Layer.KernelSize - 1 do
         begin
            for kw := 0 to Layer.KernelSize - 1 do
            begin
               WGrad := 0;
               for h := 0 to High(GradWithReLU[f]) do
               begin
                  for w := 0 to High(GradWithReLU[f][h]) do
                  begin
                     InH := h * Layer.Stride + kh;
                     InW := w * Layer.Stride + kw;
                     // How much did weight [c][kh][kw] contribute to this output pixel?
                     WGrad := WGrad + GradWithReLU[f][h][w] *
                              Layer.PaddedInput[c][InH][InW];
                  end;
               end;
               Layer.Filters[f].WeightGrads[c][kh][kw] := WGrad;
            end;
         end;
      end;
   end;

   // 3. Compute input gradient for the next layer (used in chaining backprop)
   SetLength(Result, Layer.InputChannels, Length(Layer.InputCache[0]),
             Length(Layer.InputCache[0][0]));

   // Zero out first
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;

   // For each output gradient, "spread it" across the input feature map
   for f := 0 to High(Layer.Filters) do
   begin
      for h := 0 to High(GradWithReLU[f]) do
      begin
         for w := 0 to High(GradWithReLU[f][h]) do
         begin
            for c := 0 to Layer.InputChannels - 1 do
            begin
               for kh := 0 to Layer.KernelSize - 1 do
               begin
                  for kw := 0 to Layer.KernelSize - 1 do
                  begin
                     InH := h * Layer.Stride + kh - Layer.Padding;
                     InW := w * Layer.Stride + kw - Layer.Padding;
                     // If inside bounds, accumulate gradient for previous layer
                     if (InH >= 0) and (InH < Length(Result[c])) and
                        (InW >= 0) and (InW < Length(Result[c][0])) then
                        Result[c][InH][InW] := Result[c][InH][InW] +
                           GradWithReLU[f][h][w] * Layer.Filters[f].Weights[c][kh][kw];
                  end;
               end;
            end;
         end;
      end;
   end;
end;

// ------------------------------------------------------------
// UpdateWeights: Applies the Adam optimizer to update all weights in conv/FC/output layers
// Adam uses running averages of parameter gradients to produce adaptive, stable updates
// ------------------------------------------------------------
procedure TConvolutionalNeuralNetwork.UpdateWeights;
var
   i, j, c, kh, kw: Integer;
   Grad, MHat, VHat, Update: Double;
   T: Double;
begin
   Inc(AdamT);          // Increment optimizer's time step
   T := AdamT;          // Use float for root/pow ops

   // Update convolutional layers first
   for i := 0 to High(ConvLayers) do
   begin
      for j := 0 to High(ConvLayers[i].Filters) do
      begin
         // Adam for bias
         Grad := ClipGrad(ConvLayers[i].Filters[j].BiasGrad);
         ConvLayers[i].Filters[j].BiasM := Beta1 * ConvLayers[i].Filters[j].BiasM +
                                           (1 - Beta1) * Grad;
         ConvLayers[i].Filters[j].BiasV := Beta2 * ConvLayers[i].Filters[j].BiasV +
                                           (1 - Beta2) * Grad * Grad;
         MHat := ConvLayers[i].Filters[j].BiasM / (1 - Power(Beta1, T)); // Bias correction for first moment (mean)
         VHat := ConvLayers[i].Filters[j].BiasV / (1 - Power(Beta2, T)); // Bias correction for second moment (var)
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);         // Final Adam update step
         if IsFiniteNum(Update) then
            ConvLayers[i].Filters[j].Bias := ConvLayers[i].Filters[j].Bias - Update;

         // Adam for all weights in this filter
         for c := 0 to High(ConvLayers[i].Filters[j].Weights) do
         begin
            for kh := 0 to High(ConvLayers[i].Filters[j].Weights[c]) do
            begin
               for kw := 0 to High(ConvLayers[i].Filters[j].Weights[c][kh]) do
               begin
                  Grad := ClipGrad(ConvLayers[i].Filters[j].WeightGrads[c][kh][kw]);

                  ConvLayers[i].Filters[j].WeightsM[c][kh][kw] :=
                     Beta1 * ConvLayers[i].Filters[j].WeightsM[c][kh][kw] +
                     (1 - Beta1) * Grad;
                  ConvLayers[i].Filters[j].WeightsV[c][kh][kw] :=
                     Beta2 * ConvLayers[i].Filters[j].WeightsV[c][kh][kw] +
                     (1 - Beta2) * Grad * Grad;

                  MHat := ConvLayers[i].Filters[j].WeightsM[c][kh][kw] / (1 - Power(Beta1, T));
                  VHat := ConvLayers[i].Filters[j].WeightsV[c][kh][kw] / (1 - Power(Beta2, T));
                  Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);

                  if IsFiniteNum(Update) then
                     ConvLayers[i].Filters[j].Weights[c][kh][kw] :=
                        ConvLayers[i].Filters[j].Weights[c][kh][kw] - Update;
               end;
            end;
         end;
      end;
   end;

   // Update FC layers (hidden)
   for i := 0 to High(FullyConnectedLayers) do
   begin
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
      begin
         // Weights
         for c := 0 to High(FullyConnectedLayers[i].Neurons[j].Weights) do
         begin
            Grad := ClipGrad(FullyConnectedLayers[i].Neurons[j].Error *
                            FullyConnectedLayers[i].InputCache[c]);

            FullyConnectedLayers[i].Neurons[j].WeightsM[c] :=
               Beta1 * FullyConnectedLayers[i].Neurons[j].WeightsM[c] + (1 - Beta1) * Grad;
            FullyConnectedLayers[i].Neurons[j].WeightsV[c] :=
               Beta2 * FullyConnectedLayers[i].Neurons[j].WeightsV[c] + (1 - Beta2) * Grad * Grad;

            MHat := FullyConnectedLayers[i].Neurons[j].WeightsM[c] / (1 - Power(Beta1, T));
            VHat := FullyConnectedLayers[i].Neurons[j].WeightsV[c] / (1 - Power(Beta2, T));
            Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);

            if IsFiniteNum(Update) then
               FullyConnectedLayers[i].Neurons[j].Weights[c] :=
                  FullyConnectedLayers[i].Neurons[j].Weights[c] - Update;
         end;

         // Bias
         Grad := ClipGrad(FullyConnectedLayers[i].Neurons[j].Error);
         FullyConnectedLayers[i].Neurons[j].BiasM :=
            Beta1 * FullyConnectedLayers[i].Neurons[j].BiasM + (1 - Beta1) * Grad;
         FullyConnectedLayers[i].Neurons[j].BiasV :=
            Beta2 * FullyConnectedLayers[i].Neurons[j].BiasV + (1 - Beta2) * Grad * Grad;

         MHat := FullyConnectedLayers[i].Neurons[j].BiasM / (1 - Power(Beta1, T));
         VHat := FullyConnectedLayers[i].Neurons[j].BiasV / (1 - Power(Beta2, T));
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);

         if IsFiniteNum(Update) then
            FullyConnectedLayers[i].Neurons[j].Bias :=
               FullyConnectedLayers[i].Neurons[j].Bias - Update;
      end;
   end;

   // Update output layer (classification head)
   for j := 0 to High(OutputLayer.Neurons) do
   begin
      for c := 0 to High(OutputLayer.Neurons[j].Weights) do
      begin
         Grad := ClipGrad(OutputLayer.Neurons[j].Error * OutputLayer.InputCache[c]);

         OutputLayer.Neurons[j].WeightsM[c] :=
            Beta1 * OutputLayer.Neurons[j].WeightsM[c] + (1 - Beta1) * Grad;
         OutputLayer.Neurons[j].WeightsV[c] :=
            Beta2 * OutputLayer.Neurons[j].WeightsV[c] + (1 - Beta2) * Grad * Grad;

         MHat := OutputLayer.Neurons[j].WeightsM[c] / (1 - Power(Beta1, T));
         VHat := OutputLayer.Neurons[j].WeightsV[c] / (1 - Power(Beta2, T));
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);

         if IsFiniteNum(Update) then
            OutputLayer.Neurons[j].Weights[c] := OutputLayer.Neurons[j].Weights[c] - Update;
      end;

      // Output bias
      Grad := ClipGrad(OutputLayer.Neurons[j].Error);
      OutputLayer.Neurons[j].BiasM := Beta1 * OutputLayer.Neurons[j].BiasM + (1 - Beta1) * Grad;
      OutputLayer.Neurons[j].BiasV := Beta2 * OutputLayer.Neurons[j].BiasV + (1 - Beta2) * Grad * Grad;

      MHat := OutputLayer.Neurons[j].BiasM / (1 - Power(Beta1, T));
      VHat := OutputLayer.Neurons[j].BiasV / (1 - Power(Beta2, T));
      Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);

      if IsFiniteNum(Update) then
         OutputLayer.Neurons[j].Bias := OutputLayer.Neurons[j].Bias - Update;
   end;
end;

// ------------------------------------------------------------
// TrainStep: Main entry point for model training
// One call to this function performs ONE SGD step: forward, loss, backprop, update
// ------------------------------------------------------------
function TConvolutionalNeuralNetwork.TrainStep(var Image: TImageData;
                                               const Target: Darray): Double;
var
    Prediction: Darray;     // Model output
    OutputGrad, FCGrad: Darray;
    ConvGrad: D3array;
    PoolGrad: D3array;
    CurrentGrad: D3array;
    LayerIdx: Integer;
    i: Integer;
begin
   // Validate!
    if not ValidateInput(Image) then
    begin
        WriteLn('Warning: Invalid input data, skipping sample');
        Result := 0;
        Exit;
    end;

    IsTraining := True;

   // Forward pass
    Prediction := Predict(Image);

   // Compute output layer gradient (softmax+cross-entropy trick)
    SetLength(OutputGrad, Length(OutputLayer.Neurons));
    for i := 0 to High(OutputLayer.Neurons) do
        OutputGrad[i] := OutputLayer.Neurons[i].Output - Target[i];

   // Backprop through output (FC) layer
    FCGrad := FCBackward(OutputLayer, OutputGrad, True);

   // Backprop through all hidden FC layers
    for i := High(FullyConnectedLayers) downto 0 do
        FCGrad := FCBackward(FullyConnectedLayers[i], FCGrad, False);

   // Unflatten to conv feature shape
    CurrentGrad := UnflattenGradient(FCGrad);

   // Backward through all conv and pool layers in reverse order
    for LayerIdx := High(ConvLayers) downto 0 do
    begin
        if LayerIdx <= High(PoolLayers) then
            CurrentGrad := PoolBackward(PoolLayers[LayerIdx], CurrentGrad);
        CurrentGrad := ConvBackward(ConvLayers[LayerIdx], CurrentGrad);
    end;

    UpdateWeights; // Update all parameters

   // Compute and return the loss
    Result := CrossEntropyLoss(Prediction, Target);
end;

// ---------------------------------------------
// SaveCNNModel: stub
procedure TConvolutionalNeuralNetwork.SaveCNNModel(const Filename: string);
begin
    WriteLn('SaveCNNModel not implemented in this demo code.');
end;

// ---------------------------------------------
// LoadCNNModel: stub
procedure TConvolutionalNeuralNetwork.LoadCNNModel(const Filename: string);
begin
    WriteLn('LoadCNNModel not implemented in this demo code.');
end;

// ---------------------------------------------
// Main program entry - does nothing for now
begin
   // You can add test/train batch loops and I/O code here
end.

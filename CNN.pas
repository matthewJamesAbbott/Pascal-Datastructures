//
// Convolutional Neural Network Implementation
// Based on MLP structure with convolutional layers
//

{$mode objfpc}
{$M+}

program CNNtest;

uses Classes, Math, SysUtils;

type
   Darray = array of Double;
   D2array = array of array of Double;
   D3array = array of array of array of Double;
   
   TImageData = record
      Width: Integer;
      Height: Integer;
      Channels: Integer;
      Data: D3array;  // [channel][height][width]
   end;
   
   TDataPoint = record
      Image: TImageData;
      Target: Darray;
   end;
   
   TConvFilter = record
      Weights: D3array;  // [input_channel][kernel_height][kernel_width]
      Bias: Double;
   end;
   
   TConvLayer = record
      Filters: array of TConvFilter;
      OutputMaps: D3array;  // Feature maps output
      Gradients: D3array;   // For backprop
      Stride: Integer;
      Padding: Integer;
   end;
   
   TPoolingLayer = record
      PoolSize: Integer;
      Stride: Integer;
      OutputMaps: D3array;
      Gradients: D3array;
      MaxIndices: array of array of array of record X, Y: Integer; end;  // Track max positions
   end;
   
   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
   end;
   
   TFullyConnectedLayer = record
      Neurons: array of TNeuron;
   end;
   
   TConvolutionalNeuralNetwork = class
   private
      LearningRate: Double;
      MaxIterations: Integer;
      
      ConvLayers: array of TConvLayer;
      PoolLayers: array of TPoolingLayer;
      FullyConnectedLayers: array of TFullyConnectedLayer;
      OutputLayer: TFullyConnectedLayer;
      
      FlattenedSize: Integer;
      FlattenedFeatures: Darray;
      
      procedure InitializeConvLayer(var Layer: TConvLayer; NumFilters: Integer; 
                                   InputChannels: Integer; KernelSize: Integer; 
                                   Stride: Integer; Padding: Integer);
      procedure InitializePoolLayer(var Layer: TPoolingLayer; PoolSize: Integer; Stride: Integer);
      procedure InitializeFCLayer(var Layer: TFullyConnectedLayer; NumNeurons: Integer; NumInputs: Integer);
      
      procedure ConvForward(var Layer: TConvLayer; const Input: D3array; 
                           InputWidth: Integer; InputHeight: Integer);
      procedure PoolForward(var Layer: TPoolingLayer; const Input: D3array; 
                           InputWidth: Integer; InputHeight: Integer);
      procedure FlattenFeatures(const Input: D3array; InputWidth: Integer; 
                               InputHeight: Integer; InputChannels: Integer);
      procedure FCForward(var Layer: TFullyConnectedLayer; const Input: Darray);
      
      procedure ConvBackward(var Layer: TConvLayer; const InputGrad: D3array);
      procedure PoolBackward(var Layer: TPoolingLayer; const InputGrad: D3array);
      
      procedure UpdateConvWeights(var Layer: TConvLayer; const Input: D3array);
      procedure UpdateFCWeights(var Layer: TFullyConnectedLayer; const Input: Darray);
      
      function ReLU(x: Double): Double;
      function ReLUDerivative(x: Double): Double;
      function Sigmoid(x: Double): Double;
      
   public
      constructor Create(InputWidth: Integer; InputHeight: Integer; InputChannels: Integer;
                        ConvFilters: array of Integer; KernelSizes: array of Integer;
                        PoolSizes: array of Integer; FCLayerSizes: array of Integer; 
                        OutputSize: Integer);
      function Predict(var Image: TImageData): Darray;
      procedure Train(var Image: TImageData; Target: Darray);
      procedure SaveCNNModel(const Filename: string);
   end;

constructor TConvolutionalNeuralNetwork.Create(InputWidth: Integer; InputHeight: Integer; 
                                              InputChannels: Integer;
                                              ConvFilters: array of Integer; 
                                              KernelSizes: array of Integer;
                                              PoolSizes: array of Integer; 
                                              FCLayerSizes: array of Integer; 
                                              OutputSize: Integer);
var
   i: Integer;
   CurrentWidth, CurrentHeight, CurrentChannels: Integer;
   NumInputs: Integer;
begin
   LearningRate := 0.01;
   
   CurrentWidth := InputWidth;
   CurrentHeight := InputHeight;
   CurrentChannels := InputChannels;
   
   // Initialize convolutional and pooling layers
   SetLength(ConvLayers, Length(ConvFilters));
   SetLength(PoolLayers, Length(PoolSizes));
   
   for i := 0 to High(ConvFilters) do
   begin
      InitializeConvLayer(ConvLayers[i], ConvFilters[i], CurrentChannels, 
                         KernelSizes[i], 1, 1);  // Stride=1, Padding=1
      
      // Calculate output size after convolution
      CurrentWidth := (CurrentWidth - KernelSizes[i] + 2 * 1) div 1 + 1;
      CurrentHeight := (CurrentHeight - KernelSizes[i] + 2 * 1) div 1 + 1;
      CurrentChannels := ConvFilters[i];
      
      // Initialize pooling layer
      if i <= High(PoolSizes) then
      begin
         InitializePoolLayer(PoolLayers[i], PoolSizes[i], PoolSizes[i]);
         
         // Calculate output size after pooling
         CurrentWidth := CurrentWidth div PoolSizes[i];
         CurrentHeight := CurrentHeight div PoolSizes[i];
      end;
   end;
   
   // Calculate flattened size
   FlattenedSize := CurrentWidth * CurrentHeight * CurrentChannels;
   
   // Initialize fully connected layers
   SetLength(FullyConnectedLayers, Length(FCLayerSizes));
   NumInputs := FlattenedSize;
   
   for i := 0 to High(FCLayerSizes) do
   begin
      InitializeFCLayer(FullyConnectedLayers[i], FCLayerSizes[i], NumInputs);
      NumInputs := FCLayerSizes[i];
   end;
   
   // Initialize output layer
   InitializeFCLayer(OutputLayer, OutputSize, NumInputs);
end;

procedure TConvolutionalNeuralNetwork.InitializeConvLayer(var Layer: TConvLayer; 
                                                         NumFilters: Integer; 
                                                         InputChannels: Integer; 
                                                         KernelSize: Integer; 
                                                         Stride: Integer; 
                                                         Padding: Integer);
var
   i, j, k, l: Integer;
begin
   SetLength(Layer.Filters, NumFilters);
   Layer.Stride := Stride;
   Layer.Padding := Padding;
   
   for i := 0 to NumFilters - 1 do
   begin
      SetLength(Layer.Filters[i].Weights, InputChannels, KernelSize, KernelSize);
      
      // He initialization for ReLU
      for j := 0 to InputChannels - 1 do
         for k := 0 to KernelSize - 1 do
            for l := 0 to KernelSize - 1 do
               Layer.Filters[i].Weights[j][k][l] := (Random - 0.5) * Sqrt(2.0 / (InputChannels * KernelSize * KernelSize));
      
      Layer.Filters[i].Bias := 0.0;
   end;
end;

procedure TConvolutionalNeuralNetwork.InitializePoolLayer(var Layer: TPoolingLayer; 
                                                         PoolSize: Integer; 
                                                         Stride: Integer);
begin
   Layer.PoolSize := PoolSize;
   Layer.Stride := Stride;
end;

procedure TConvolutionalNeuralNetwork.InitializeFCLayer(var Layer: TFullyConnectedLayer; 
                                                       NumNeurons: Integer; 
                                                       NumInputs: Integer);
var
   i, j: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   for i := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[i].Weights, NumInputs);
      for j := 0 to NumInputs - 1 do
         Layer.Neurons[i].Weights[j] := (Random - 0.5) * 0.1;
      Layer.Neurons[i].Bias := 0.0;
   end;
end;

function TConvolutionalNeuralNetwork.ReLU(x: Double): Double;
begin
   if x > 0 then
      Result := x
   else
      Result := 0.0;
end;

function TConvolutionalNeuralNetwork.ReLUDerivative(x: Double): Double;
begin
   if x > 0 then
      Result := 1.0
   else
      Result := 0.0;
end;

function TConvolutionalNeuralNetwork.Sigmoid(x: Double): Double;
begin
   Result := 1.0 / (1.0 + Exp(-x));
end;

procedure TConvolutionalNeuralNetwork.ConvForward(var Layer: TConvLayer; 
                                                 const Input: D3array; 
                                                 InputWidth: Integer; 
                                                 InputHeight: Integer);
var
   f, c, h, w, kh, kw: Integer;
   OutputWidth, OutputHeight: Integer;
   Sum: Double;
   InputChannels, KernelSize: Integer;
begin
   InputChannels := Length(Input);
   KernelSize := Length(Layer.Filters[0].Weights[0]);
   
   OutputWidth := (InputWidth - KernelSize + 2 * Layer.Padding) div Layer.Stride + 1;
   OutputHeight := (InputHeight - KernelSize + 2 * Layer.Padding) div Layer.Stride + 1;
   
   SetLength(Layer.OutputMaps, Length(Layer.Filters), OutputHeight, OutputWidth);
   
   // For each filter
   for f := 0 to High(Layer.Filters) do
   begin
      // For each output position
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
            Sum := Layer.Filters[f].Bias;
            
            // Convolve with all input channels
            for c := 0 to InputChannels - 1 do
            begin
               for kh := 0 to KernelSize - 1 do
               begin
                  for kw := 0 to KernelSize - 1 do
                  begin
                     // Simple implementation without padding
                     if (h * Layer.Stride + kh < InputHeight) and 
                        (w * Layer.Stride + kw < InputWidth) then
                        Sum := Sum + Input[c][h * Layer.Stride + kh][w * Layer.Stride + kw] * 
                               Layer.Filters[f].Weights[c][kh][kw];
                  end;
               end;
            end;
            
            Layer.OutputMaps[f][h][w] := ReLU(Sum);
         end;
      end;
   end;
end;

procedure TConvolutionalNeuralNetwork.PoolForward(var Layer: TPoolingLayer; 
                                                 const Input: D3array; 
                                                 InputWidth: Integer; 
                                                 InputHeight: Integer);
var
   c, h, w, ph, pw: Integer;
   OutputWidth, OutputHeight: Integer;
   MaxVal: Double;
   MaxH, MaxW: Integer;
begin
   OutputWidth := InputWidth div Layer.PoolSize;
   OutputHeight := InputHeight div Layer.PoolSize;
   
   SetLength(Layer.OutputMaps, Length(Input), OutputHeight, OutputWidth);
   SetLength(Layer.MaxIndices, Length(Input), OutputHeight, OutputWidth);
   
   // Max pooling
   for c := 0 to High(Input) do
   begin
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
            MaxVal := -1e10;
            MaxH := 0;
            MaxW := 0;
            
            // Find maximum in pool window
            for ph := 0 to Layer.PoolSize - 1 do
            begin
               for pw := 0 to Layer.PoolSize - 1 do
               begin
                  if Input[c][h * Layer.PoolSize + ph][w * Layer.PoolSize + pw] > MaxVal then
                  begin
                     MaxVal := Input[c][h * Layer.PoolSize + ph][w * Layer.PoolSize + pw];
                     MaxH := h * Layer.PoolSize + ph;
                     MaxW := w * Layer.PoolSize + pw;
                  end;
               end;
            end;
            
            Layer.OutputMaps[c][h][w] := MaxVal;
            Layer.MaxIndices[c][h][w].Y := MaxH;
            Layer.MaxIndices[c][h][w].X := MaxW;
         end;
      end;
   end;
end;

procedure TConvolutionalNeuralNetwork.FlattenFeatures(const Input: D3array; 
                                                     InputWidth: Integer; 
                                                     InputHeight: Integer; 
                                                     InputChannels: Integer);
var
   c, h, w, idx: Integer;
begin
   SetLength(FlattenedFeatures, InputChannels * InputHeight * InputWidth);
   idx := 0;
   
   for c := 0 to InputChannels - 1 do
      for h := 0 to InputHeight - 1 do
         for w := 0 to InputWidth - 1 do
         begin
            FlattenedFeatures[idx] := Input[c][h][w];
            Inc(idx);
         end;
end;

procedure TConvolutionalNeuralNetwork.FCForward(var Layer: TFullyConnectedLayer; 
                                               const Input: Darray);
var
   i, j: Integer;
   Sum: Double;
begin
   for i := 0 to High(Layer.Neurons) do
   begin
      Sum := Layer.Neurons[i].Bias;
      for j := 0 to High(Input) do
         Sum := Sum + Input[j] * Layer.Neurons[i].Weights[j];
      Layer.Neurons[i].Output := ReLU(Sum);
   end;
end;

function TConvolutionalNeuralNetwork.Predict(var Image: TImageData): Darray;
var
   i, j: Integer;
   CurrentOutput: D3array;
   CurrentWidth, CurrentHeight: Integer;
   LayerInput: Darray;
   Sum: Double;
begin
   // Forward through conv layers
   CurrentOutput := Image.Data;
   CurrentWidth := Image.Width;
   CurrentHeight := Image.Height;
   
   for i := 0 to High(ConvLayers) do
   begin
      ConvForward(ConvLayers[i], CurrentOutput, CurrentWidth, CurrentHeight);
      CurrentOutput := ConvLayers[i].OutputMaps;
      
      CurrentWidth := Length(CurrentOutput[0][0]);
      CurrentHeight := Length(CurrentOutput[0]);
      
      if i <= High(PoolLayers) then
      begin
         PoolForward(PoolLayers[i], CurrentOutput, CurrentWidth, CurrentHeight);
         CurrentOutput := PoolLayers[i].OutputMaps;
         CurrentWidth := Length(CurrentOutput[0][0]);
         CurrentHeight := Length(CurrentOutput[0]);
      end;
   end;
   
   // Flatten
   FlattenFeatures(CurrentOutput, CurrentWidth, CurrentHeight, Length(CurrentOutput));
   
   // Forward through FC layers
   LayerInput := FlattenedFeatures;
   for i := 0 to High(FullyConnectedLayers) do
   begin
      FCForward(FullyConnectedLayers[i], LayerInput);
      SetLength(LayerInput, Length(FullyConnectedLayers[i].Neurons));
      for j := 0 to High(LayerInput) do
         LayerInput[j] := FullyConnectedLayers[i].Neurons[j].Output;
   end;
   
   // Output layer
   SetLength(Result, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to High(LayerInput) do
         Sum := Sum + LayerInput[j] * OutputLayer.Neurons[i].Weights[j];
      OutputLayer.Neurons[i].Output := Sigmoid(Sum);
      Result[i] := OutputLayer.Neurons[i].Output;
   end;
end;

procedure TConvolutionalNeuralNetwork.Train(var Image: TImageData; Target: Darray);
var
   Prediction: Darray;
   i, j: Integer;
begin
   Prediction := Predict(Image);
   
   // Calculate output layer errors (simplified backprop)
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Error := OutputLayer.Neurons[i].Output * 
         (1 - OutputLayer.Neurons[i].Output) * (Target[i] - OutputLayer.Neurons[i].Output);
   
   // Update output layer weights
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      for j := 0 to High(OutputLayer.Neurons[i].Weights) do
         OutputLayer.Neurons[i].Weights[j] := OutputLayer.Neurons[i].Weights[j] + 
            LearningRate * OutputLayer.Neurons[i].Error * FlattenedFeatures[j];
      OutputLayer.Neurons[i].Bias := OutputLayer.Neurons[i].Bias + 
         LearningRate * OutputLayer.Neurons[i].Error;
   end;
end;

procedure TConvolutionalNeuralNetwork.ConvBackward(var Layer: TConvLayer; const InputGrad: D3array);
begin
   // Simplified - full implementation would backpropagate through conv layers
end;

procedure TConvolutionalNeuralNetwork.PoolBackward(var Layer: TPoolingLayer; const InputGrad: D3array);
begin
   // Simplified - full implementation would backpropagate through pooling
end;

procedure TConvolutionalNeuralNetwork.UpdateConvWeights(var Layer: TConvLayer; const Input: D3array);
begin
   // Simplified - full implementation would update conv filter weights
end;

procedure TConvolutionalNeuralNetwork.UpdateFCWeights(var Layer: TFullyConnectedLayer; const Input: Darray);
begin
   // Simplified - full implementation would update FC weights with backprop
end;

procedure TConvolutionalNeuralNetwork.SaveCNNModel(const Filename: string);
var
   F: File;
   i, j, k, l, m: Integer;
begin
   AssignFile(F, Filename);
   Rewrite(F, 1);
   
   // Write number of layers
   i := Length(ConvLayers);
   BlockWrite(F, i, SizeOf(Integer));
   
   // Write convolutional layers
   for i := 0 to High(ConvLayers) do
   begin
      j := Length(ConvLayers[i].Filters);
      BlockWrite(F, j, SizeOf(Integer));
      
      for j := 0 to High(ConvLayers[i].Filters) do
      begin
         for k := 0 to High(ConvLayers[i].Filters[j].Weights) do
            for l := 0 to High(ConvLayers[i].Filters[j].Weights[k]) do
               for m := 0 to High(ConvLayers[i].Filters[j].Weights[k][l]) do
                  BlockWrite(F, ConvLayers[i].Filters[j].Weights[k][l][m], SizeOf(Double));
         BlockWrite(F, ConvLayers[i].Filters[j].Bias, SizeOf(Double));
      end;
   end;
   
   CloseFile(F);
   WriteLn('CNN model saved to ', Filename);
end;

// Example usage
var
   CNN: TConvolutionalNeuralNetwork;
   Image: TImageData;
   Prediction: Darray;
   i, j, k: Integer;
begin
   Randomize;
   
   // Create a simple 28x28 grayscale image
   Image.Width := 28;
   Image.Height := 28;
   Image.Channels := 1;
   SetLength(Image.Data, 1, 28, 28);
   
   // Fill with random data
   for i := 0 to 27 do
      for j := 0 to 27 do
         Image.Data[0][i][j] := Random;
   
   // Create CNN: 28x28x1 input, 2 conv layers (16 and 32 filters, 3x3 kernels),
   // 2x2 pooling, 1 FC layer (128 units), 10 output classes
   CNN := TConvolutionalNeuralNetwork.Create(28, 28, 1, [16, 32], [3, 3], [2, 2], [128], 10);
   CNN.MaxIterations := 10;
   
   WriteLn('Training CNN on sample image...');
   
   // Train on the image
   for i := 0 to 9 do
      CNN.Train(Image, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
   
   // Make prediction
   Prediction := CNN.Predict(Image);
   Write('Prediction: [');
   for i := 0 to High(Prediction) do
   begin
      Write(Prediction[i]:0:4);
      if i < High(Prediction) then Write(', ');
   end;
   WriteLn(']');
   
   // Save model
   CNN.SaveCNNModel('TestCNN.bin');
   
   WriteLn('Done!');
end.

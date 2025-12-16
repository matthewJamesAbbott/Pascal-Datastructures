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

uses Classes, Math, SysUtils;

const
   EPSILON = 1e-8;
   GRAD_CLIP = 1.0;
   
type
   Darray = array of Double;
   D2array = array of array of Double;
   D3array = array of array of array of Double;
   D4array = array of array of array of array of Double;
   
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
      // Adam optimizer state
      WeightsM: D3array;  // First moment
      WeightsV: D3array;  // Second moment
      BiasM: Double;
      BiasV: Double;
      // Stored gradients
      WeightGrads: D3array;
      BiasGrad: Double;
   end;
   
   TConvLayer = record
      Filters: array of TConvFilter;
      OutputMaps: D3array;    // Feature maps output
      PreActivation: D3array; // Before ReLU
      InputCache: D3array;    // Cached input for backprop
      PaddedInput: D3array;   // Padded input cache
      Stride: Integer;
      Padding: Integer;
      KernelSize: Integer;
      InputChannels: Integer;
   end;
   
   TPoolingLayer = record
      PoolSize: Integer;
      Stride: Integer;
      OutputMaps: D3array;
      InputCache: D3array;
      MaxIndices: array of array of array of record X, Y: Integer; end;
   end;
   
   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      PreActivation: Double;
      Error: Double;
      DropoutMask: Double;
      // Adam optimizer state
      WeightsM: array of Double;
      WeightsV: array of Double;
      BiasM: Double;
      BiasV: Double;
   end;
   
   TFullyConnectedLayer = record
      Neurons: array of TNeuron;
      InputCache: Darray;
   end;
   
   TConvolutionalNeuralNetwork = class
   private
      LearningRate: Double;
      DropoutRate: Double;
      Beta1, Beta2: Double;  // Adam parameters
      AdamT: Integer;        // Adam timestep
      IsTraining: Boolean;
      
      ConvLayers: array of TConvLayer;
      PoolLayers: array of TPoolingLayer;
      FullyConnectedLayers: array of TFullyConnectedLayer;
      OutputLayer: TFullyConnectedLayer;
      
      FlattenedSize: Integer;
      FlattenedFeatures: Darray;
      LastConvHeight, LastConvWidth, LastConvChannels: Integer;
      
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
      
      function ConvBackward(var Layer: TConvLayer; const Grad: D3array): D3array;
      function PoolBackward(var Layer: TPoolingLayer; const Grad: D3array): D3array;
      function FCBackward(var Layer: TFullyConnectedLayer; const Grad: Darray; IsOutputLayer: Boolean): Darray;
      function UnflattenGradient(const Grad: Darray): D3array;
      
      procedure UpdateWeights;
      procedure ApplyDropout(var Layer: TFullyConnectedLayer);
      
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
      constructor Create(InputWidth: Integer; InputHeight: Integer; InputChannels: Integer;
                        ConvFilters: array of Integer; KernelSizes: array of Integer;
                        PoolSizes: array of Integer; FCLayerSizes: array of Integer; 
                        OutputSize: Integer; ALearningRate: Double = 0.001;
                        ADropoutRate: Double = 0.25);
      function Predict(var Image: TImageData): Darray;
      function TrainStep(var Image: TImageData; const Target: Darray): Double;
      procedure SaveCNNModel(const Filename: string);
      procedure LoadCNNModel(const Filename: string);
   end;

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
   i: Integer;
   CurrentWidth, CurrentHeight, CurrentChannels: Integer;
   NumInputs: Integer;
   KernelPadding: Integer;
begin
   LearningRate := ALearningRate;
   DropoutRate := ADropoutRate;
   Beta1 := 0.9;
   Beta2 := 0.999;
   AdamT := 0;
   IsTraining := False;
   
   CurrentWidth := InputWidth;
   CurrentHeight := InputHeight;
   CurrentChannels := InputChannels;
   
   SetLength(ConvLayers, Length(ConvFilters));
   SetLength(PoolLayers, Length(PoolSizes));
   
   for i := 0 to High(ConvFilters) do
   begin
      KernelPadding := KernelSizes[i] div 2;
      InitializeConvLayer(ConvLayers[i], ConvFilters[i], CurrentChannels, 
                         KernelSizes[i], 1, KernelPadding);
      
      CurrentWidth := (CurrentWidth - KernelSizes[i] + 2 * KernelPadding) div 1 + 1;
      CurrentHeight := (CurrentHeight - KernelSizes[i] + 2 * KernelPadding) div 1 + 1;
      CurrentChannels := ConvFilters[i];
      
      if i <= High(PoolSizes) then
      begin
         InitializePoolLayer(PoolLayers[i], PoolSizes[i], PoolSizes[i]);
         CurrentWidth := CurrentWidth div PoolSizes[i];
         CurrentHeight := CurrentHeight div PoolSizes[i];
      end;
   end;
   
   FlattenedSize := CurrentWidth * CurrentHeight * CurrentChannels;
   
   SetLength(FullyConnectedLayers, Length(FCLayerSizes));
   NumInputs := FlattenedSize;
   
   for i := 0 to High(FCLayerSizes) do
   begin
      InitializeFCLayer(FullyConnectedLayers[i], FCLayerSizes[i], NumInputs);
      NumInputs := FCLayerSizes[i];
   end;
   
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
   Scale: Double;
begin
   SetLength(Layer.Filters, NumFilters);
   Layer.Stride := Stride;
   Layer.Padding := Padding;
   Layer.KernelSize := KernelSize;
   Layer.InputChannels := InputChannels;
   
   Scale := Sqrt(2.0 / (InputChannels * KernelSize * KernelSize));
   
   for i := 0 to NumFilters - 1 do
   begin
      SetLength(Layer.Filters[i].Weights, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightsM, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightsV, InputChannels, KernelSize, KernelSize);
      SetLength(Layer.Filters[i].WeightGrads, InputChannels, KernelSize, KernelSize);
      
      for j := 0 to InputChannels - 1 do
         for k := 0 to KernelSize - 1 do
            for l := 0 to KernelSize - 1 do
            begin
               Layer.Filters[i].Weights[j][k][l] := (Random - 0.5) * Scale;
               Layer.Filters[i].WeightsM[j][k][l] := 0;
               Layer.Filters[i].WeightsV[j][k][l] := 0;
               Layer.Filters[i].WeightGrads[j][k][l] := 0;
            end;
      
      Layer.Filters[i].Bias := 0.0;
      Layer.Filters[i].BiasM := 0.0;
      Layer.Filters[i].BiasV := 0.0;
      Layer.Filters[i].BiasGrad := 0.0;
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
   Scale: Double;
begin
   SetLength(Layer.Neurons, NumNeurons);
   Scale := Sqrt(2.0 / NumInputs);
   
   for i := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[i].Weights, NumInputs);
      SetLength(Layer.Neurons[i].WeightsM, NumInputs);
      SetLength(Layer.Neurons[i].WeightsV, NumInputs);
      
      for j := 0 to NumInputs - 1 do
      begin
         Layer.Neurons[i].Weights[j] := (Random - 0.5) * Scale;
         Layer.Neurons[i].WeightsM[j] := 0;
         Layer.Neurons[i].WeightsV[j] := 0;
      end;
      
      Layer.Neurons[i].Bias := 0.0;
      Layer.Neurons[i].BiasM := 0.0;
      Layer.Neurons[i].BiasV := 0.0;
      Layer.Neurons[i].Output := 0;
      Layer.Neurons[i].PreActivation := 0;
      Layer.Neurons[i].Error := 0;
      Layer.Neurons[i].DropoutMask := 1.0;
   end;
end;

function TConvolutionalNeuralNetwork.IsFiniteNum(x: Double): Boolean;
begin
   Result := (not IsNan(x)) and (not IsInfinite(x));
end;

function TConvolutionalNeuralNetwork.Clamp(x, MinVal, MaxVal: Double): Double;
begin
   if x < MinVal then
      Result := MinVal
   else if x > MaxVal then
      Result := MaxVal
   else
      Result := x;
end;

function TConvolutionalNeuralNetwork.ClipGrad(x: Double): Double;
begin
   if not IsFiniteNum(x) then
      Result := 0
   else
      Result := Clamp(x, -GRAD_CLIP, GRAD_CLIP);
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

function TConvolutionalNeuralNetwork.Softmax(const Logits: Darray): Darray;
var
   i: Integer;
   MaxVal, Sum, ExpVal: Double;
begin
   SetLength(Result, Length(Logits));
   
   // Find max for numerical stability
   MaxVal := -1e308;
   for i := 0 to High(Logits) do
      if IsFiniteNum(Logits[i]) and (Logits[i] > MaxVal) then
         MaxVal := Logits[i];
   
   if not IsFiniteNum(MaxVal) then
      MaxVal := 0;
   
   // Compute exp and sum
   Sum := 0;
   for i := 0 to High(Logits) do
   begin
      if IsFiniteNum(Logits[i]) then
         ExpVal := Exp(Clamp(Logits[i] - MaxVal, -500, 500))
      else
         ExpVal := Exp(0);
      Result[i] := ExpVal;
      Sum := Sum + ExpVal;
   end;
   
   if (Sum <= 0) or (not IsFiniteNum(Sum)) then
      Sum := 1;
   
   // Normalize and clamp
   for i := 0 to High(Result) do
      Result[i] := Clamp(Result[i] / Sum, 1e-15, 1 - 1e-15);
end;

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
         P := Clamp(Predicted[i], 1e-15, 1 - 1e-15);
         Result := Result - Target[i] * Ln(P);
      end;
   end;
   
   if not IsFiniteNum(Result) then
      Result := 0;
end;

function TConvolutionalNeuralNetwork.Pad3D(const Input: D3array; Padding: Integer): D3array;
var
   c, h, w: Integer;
   Channels, Height, Width: Integer;
   SrcH, SrcW: Integer;
begin
   if Padding = 0 then
   begin
      Result := Input;
      Exit;
   end;
   
   Channels := Length(Input);
   Height := Length(Input[0]);
   Width := Length(Input[0][0]);
   
   SetLength(Result, Channels, Height + 2 * Padding, Width + 2 * Padding);
   
   for c := 0 to Channels - 1 do
      for h := 0 to Height + 2 * Padding - 1 do
         for w := 0 to Width + 2 * Padding - 1 do
         begin
            SrcH := h - Padding;
            SrcW := w - Padding;
            if (SrcH >= 0) and (SrcH < Height) and (SrcW >= 0) and (SrcW < Width) then
               Result[c][h][w] := Input[c][SrcH][SrcW]
            else
               Result[c][h][w] := 0;
         end;
end;

function TConvolutionalNeuralNetwork.ValidateInput(const Image: TImageData): Boolean;
var
   c, h, w: Integer;
begin
   Result := False;
   
   if (Image.Data = nil) or (Length(Image.Data) <> Image.Channels) then
      Exit;
   
   for c := 0 to Image.Channels - 1 do
   begin
      if (Image.Data[c] = nil) or (Length(Image.Data[c]) <> Image.Height) then
         Exit;
      for h := 0 to Image.Height - 1 do
      begin
         if (Image.Data[c][h] = nil) or (Length(Image.Data[c][h]) <> Image.Width) then
            Exit;
         for w := 0 to Image.Width - 1 do
            if not IsFiniteNum(Image.Data[c][h][w]) then
               Exit;
      end;
   end;
   
   Result := True;
end;

procedure TConvolutionalNeuralNetwork.ApplyDropout(var Layer: TFullyConnectedLayer);
var
   i: Integer;
begin
   for i := 0 to High(Layer.Neurons) do
   begin
      if IsTraining and (DropoutRate > 0) then
      begin
         if Random > DropoutRate then
            Layer.Neurons[i].DropoutMask := 1.0 / (1.0 - DropoutRate)
         else
            Layer.Neurons[i].DropoutMask := 0;
      end
      else
         Layer.Neurons[i].DropoutMask := 1.0;
   end;
end;

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
   Layer.InputCache := Input;
   Layer.PaddedInput := Pad3D(Input, Layer.Padding);
   
   OutputWidth := (InputWidth + 2 * Layer.Padding - Layer.KernelSize) div Layer.Stride + 1;
   OutputHeight := (InputHeight + 2 * Layer.Padding - Layer.KernelSize) div Layer.Stride + 1;
   
   SetLength(Layer.OutputMaps, Length(Layer.Filters), OutputHeight, OutputWidth);
   SetLength(Layer.PreActivation, Length(Layer.Filters), OutputHeight, OutputWidth);
   
   for f := 0 to High(Layer.Filters) do
   begin
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
            Sum := Layer.Filters[f].Bias;
            
            for c := 0 to Layer.InputChannels - 1 do
            begin
               for kh := 0 to Layer.KernelSize - 1 do
               begin
                  for kw := 0 to Layer.KernelSize - 1 do
                  begin
                     InH := h * Layer.Stride + kh;
                     InW := w * Layer.Stride + kw;
                     Sum := Sum + Layer.PaddedInput[c][InH][InW] * 
                            Layer.Filters[f].Weights[c][kh][kw];
                  end;
               end;
            end;
            
            if not IsFiniteNum(Sum) then
               Sum := 0;
            
            Layer.PreActivation[f][h][w] := Sum;
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
   MaxVal, Val: Double;
   MaxPH, MaxPW: Integer;
begin
   Layer.InputCache := Input;
   
   OutputWidth := InputWidth div Layer.PoolSize;
   OutputHeight := InputHeight div Layer.PoolSize;
   
   SetLength(Layer.OutputMaps, Length(Input), OutputHeight, OutputWidth);
   SetLength(Layer.MaxIndices, Length(Input), OutputHeight, OutputWidth);
   
   for c := 0 to High(Input) do
   begin
      for h := 0 to OutputHeight - 1 do
      begin
         for w := 0 to OutputWidth - 1 do
         begin
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
            
            Layer.OutputMaps[c][h][w] := MaxVal;
            Layer.MaxIndices[c][h][w].Y := MaxPH;
            Layer.MaxIndices[c][h][w].X := MaxPW;
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
   LastConvWidth := InputWidth;
   LastConvHeight := InputHeight;
   LastConvChannels := InputChannels;
   
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

function TConvolutionalNeuralNetwork.UnflattenGradient(const Grad: Darray): D3array;
var
   c, h, w, idx: Integer;
begin
   SetLength(Result, LastConvChannels, LastConvHeight, LastConvWidth);
   idx := 0;
   
   for c := 0 to LastConvChannels - 1 do
      for h := 0 to LastConvHeight - 1 do
         for w := 0 to LastConvWidth - 1 do
         begin
            Result[c][h][w] := Grad[idx];
            Inc(idx);
         end;
end;

procedure TConvolutionalNeuralNetwork.FCForward(var Layer: TFullyConnectedLayer; 
                                               const Input: Darray);
var
   i, j: Integer;
   Sum: Double;
begin
   Layer.InputCache := Copy(Input);
   ApplyDropout(Layer);
   
   for i := 0 to High(Layer.Neurons) do
   begin
      Sum := Layer.Neurons[i].Bias;
      for j := 0 to High(Input) do
         Sum := Sum + Input[j] * Layer.Neurons[i].Weights[j];
      
      if not IsFiniteNum(Sum) then
         Sum := 0;
      
      Layer.Neurons[i].PreActivation := Sum;
      Layer.Neurons[i].Output := ReLU(Sum) * Layer.Neurons[i].DropoutMask;
   end;
end;

function TConvolutionalNeuralNetwork.Predict(var Image: TImageData): Darray;
var
   i, j: Integer;
   CurrentOutput: D3array;
   CurrentWidth, CurrentHeight: Integer;
   LayerInput: Darray;
   Logits: Darray;
   Sum: Double;
begin
   IsTraining := False;
   
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
   
   FlattenFeatures(CurrentOutput, CurrentWidth, CurrentHeight, Length(CurrentOutput));
   
   LayerInput := FlattenedFeatures;
   for i := 0 to High(FullyConnectedLayers) do
   begin
      FCForward(FullyConnectedLayers[i], LayerInput);
      SetLength(LayerInput, Length(FullyConnectedLayers[i].Neurons));
      for j := 0 to High(LayerInput) do
         LayerInput[j] := FullyConnectedLayers[i].Neurons[j].Output;
   end;
   
   // Output layer - compute logits
   OutputLayer.InputCache := Copy(LayerInput);
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
   
   // Apply softmax
   Result := Softmax(Logits);
   
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Output := Result[i];
end;

function TConvolutionalNeuralNetwork.FCBackward(var Layer: TFullyConnectedLayer; 
                                               const Grad: Darray; 
                                               IsOutputLayer: Boolean): Darray;
var
   i, j: Integer;
   Delta: Double;
begin
   SetLength(Result, Length(Layer.InputCache));
   for i := 0 to High(Result) do
      Result[i] := 0;
   
   for i := 0 to High(Layer.Neurons) do
   begin
      if IsOutputLayer then
         Delta := Grad[i]
      else
         Delta := Grad[i] * ReLUDerivative(Layer.Neurons[i].PreActivation) * 
                  Layer.Neurons[i].DropoutMask;
      
      Layer.Neurons[i].Error := Delta;
      
      for j := 0 to High(Layer.Neurons[i].Weights) do
         Result[j] := Result[j] + Delta * Layer.Neurons[i].Weights[j];
   end;
end;

function TConvolutionalNeuralNetwork.PoolBackward(var Layer: TPoolingLayer; 
                                                  const Grad: D3array): D3array;
var
   c, h, w: Integer;
   InH, InW: Integer;
   SrcH, SrcW: Integer;
begin
   SetLength(Result, Length(Layer.InputCache), Length(Layer.InputCache[0]), 
             Length(Layer.InputCache[0][0]));
   
   // Initialize to zero
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;
   
   // Route gradients to max positions
   for c := 0 to High(Grad) do
   begin
      for h := 0 to High(Grad[c]) do
      begin
         for w := 0 to High(Grad[c][h]) do
         begin
            SrcH := h * Layer.PoolSize + Layer.MaxIndices[c][h][w].Y;
            SrcW := w * Layer.PoolSize + Layer.MaxIndices[c][h][w].X;
            Result[c][SrcH][SrcW] := Grad[c][h][w];
         end;
      end;
   end;
end;

function TConvolutionalNeuralNetwork.ConvBackward(var Layer: TConvLayer; 
                                                  const Grad: D3array): D3array;
var
   f, c, h, w, kh, kw: Integer;
   GradWithReLU: D3array;
   GradSum, WGrad: Double;
   InH, InW: Integer;
   OutH, OutW: Integer;
begin
   // Apply ReLU derivative to gradient
   SetLength(GradWithReLU, Length(Grad), Length(Grad[0]), Length(Grad[0][0]));
   for f := 0 to High(Grad) do
      for h := 0 to High(Grad[f]) do
         for w := 0 to High(Grad[f][h]) do
            GradWithReLU[f][h][w] := Grad[f][h][w] * 
                                     ReLUDerivative(Layer.PreActivation[f][h][w]);
   
   // Compute and store gradients for weights
   for f := 0 to High(Layer.Filters) do
   begin
      // Bias gradient
      GradSum := 0;
      for h := 0 to High(GradWithReLU[f]) do
         for w := 0 to High(GradWithReLU[f][h]) do
            GradSum := GradSum + GradWithReLU[f][h][w];
      Layer.Filters[f].BiasGrad := GradSum;
      
      // Weight gradients
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
                     WGrad := WGrad + GradWithReLU[f][h][w] * 
                              Layer.PaddedInput[c][InH][InW];
                  end;
               end;
               Layer.Filters[f].WeightGrads[c][kh][kw] := WGrad;
            end;
         end;
      end;
   end;
   
   // Compute input gradient
   SetLength(Result, Layer.InputChannels, Length(Layer.InputCache[0]), 
             Length(Layer.InputCache[0][0]));
   
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;
   
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

procedure TConvolutionalNeuralNetwork.UpdateWeights;
var
   i, j, c, kh, kw: Integer;
   Grad, MHat, VHat, Update: Double;
   T: Double;
begin
   Inc(AdamT);
   T := AdamT;
   
   // Update conv layer weights
   for i := 0 to High(ConvLayers) do
   begin
      for j := 0 to High(ConvLayers[i].Filters) do
      begin
         // Update bias
         Grad := ClipGrad(ConvLayers[i].Filters[j].BiasGrad);
         ConvLayers[i].Filters[j].BiasM := Beta1 * ConvLayers[i].Filters[j].BiasM + 
                                           (1 - Beta1) * Grad;
         ConvLayers[i].Filters[j].BiasV := Beta2 * ConvLayers[i].Filters[j].BiasV + 
                                           (1 - Beta2) * Grad * Grad;
         MHat := ConvLayers[i].Filters[j].BiasM / (1 - Power(Beta1, T));
         VHat := ConvLayers[i].Filters[j].BiasV / (1 - Power(Beta2, T));
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);
         if IsFiniteNum(Update) then
            ConvLayers[i].Filters[j].Bias := ConvLayers[i].Filters[j].Bias - Update;
         
         // Update weights
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
   
   // Update FC layers
   for i := 0 to High(FullyConnectedLayers) do
   begin
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
      begin
         // Update weights
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
         
         // Update bias
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
   
   // Update output layer
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

function TConvolutionalNeuralNetwork.TrainStep(var Image: TImageData; 
                                               const Target: Darray): Double;
var
   Prediction: Darray;
   OutputGrad, FCGrad: Darray;
   ConvGrad: D3array;
   i: Integer;
begin
   if not ValidateInput(Image) then
   begin
      WriteLn('Warning: Invalid input data, skipping sample');
      Result := 0;
      Exit;
   end;
   
   IsTraining := True;
   
   // Forward pass
   Prediction := Predict(Image);
   
   // Compute output gradient (softmax + cross-entropy derivative = output - target)
   SetLength(OutputGrad, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
      OutputGrad[i] := OutputLayer.Neurons[i].Output - Target[i];
   
   // Backward through output layer
   FCGrad := FCBackward(OutputLayer, OutputGrad, True);
   
   // Backward through FC layers
   for i := High(FullyConnectedLayers) downto 0 do
      FCGrad := FCBackward(FullyConnectedLayers[i], FCGrad, False);
   
   // Unflatten gradient
   ConvGrad := UnflattenGradient(FCGrad);
   
   // Backward through conv/pool layers
   for i := High(ConvLayers) downto 0 do
   begin
      if i <= High(PoolLayers) then
         ConvGrad := PoolBackward(PoolLayers[i], ConvGrad);
      ConvGrad := ConvBackward(ConvLayers[i], ConvGrad);
   end;
   
   // Update all weights
   UpdateWeights;
   
   Result := CrossEntropyLoss(Prediction, Target);
end;

procedure TConvolutionalNeuralNetwork.SaveCNNModel(const Filename: string);
var
   F: TextFile;
   i, j, k, l, m: Integer;
begin
   AssignFile(F, Filename);
   Rewrite(F);
   
   WriteLn(F, 'CNN_MODEL_V2');
   WriteLn(F, AdamT);
   WriteLn(F, Length(ConvLayers));
   
   for i := 0 to High(ConvLayers) do
   begin
      WriteLn(F, Length(ConvLayers[i].Filters));
      WriteLn(F, ConvLayers[i].KernelSize);
      WriteLn(F, ConvLayers[i].InputChannels);
      
      for j := 0 to High(ConvLayers[i].Filters) do
      begin
         WriteLn(F, ConvLayers[i].Filters[j].Bias:0:15);
         WriteLn(F, ConvLayers[i].Filters[j].BiasM:0:15);
         WriteLn(F, ConvLayers[i].Filters[j].BiasV:0:15);
         
         for k := 0 to High(ConvLayers[i].Filters[j].Weights) do
            for l := 0 to High(ConvLayers[i].Filters[j].Weights[k]) do
               for m := 0 to High(ConvLayers[i].Filters[j].Weights[k][l]) do
               begin
                  WriteLn(F, ConvLayers[i].Filters[j].Weights[k][l][m]:0:15);
                  WriteLn(F, ConvLayers[i].Filters[j].WeightsM[k][l][m]:0:15);
                  WriteLn(F, ConvLayers[i].Filters[j].WeightsV[k][l][m]:0:15);
               end;
      end;
   end;
   
   WriteLn(F, Length(FullyConnectedLayers));
   for i := 0 to High(FullyConnectedLayers) do
   begin
      WriteLn(F, Length(FullyConnectedLayers[i].Neurons));
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
      begin
         WriteLn(F, FullyConnectedLayers[i].Neurons[j].Bias:0:15);
         WriteLn(F, FullyConnectedLayers[i].Neurons[j].BiasM:0:15);
         WriteLn(F, FullyConnectedLayers[i].Neurons[j].BiasV:0:15);
         WriteLn(F, Length(FullyConnectedLayers[i].Neurons[j].Weights));
         for k := 0 to High(FullyConnectedLayers[i].Neurons[j].Weights) do
         begin
            WriteLn(F, FullyConnectedLayers[i].Neurons[j].Weights[k]:0:15);
            WriteLn(F, FullyConnectedLayers[i].Neurons[j].WeightsM[k]:0:15);
            WriteLn(F, FullyConnectedLayers[i].Neurons[j].WeightsV[k]:0:15);
         end;
      end;
   end;
   
   WriteLn(F, Length(OutputLayer.Neurons));
   for j := 0 to High(OutputLayer.Neurons) do
   begin
      WriteLn(F, OutputLayer.Neurons[j].Bias:0:15);
      WriteLn(F, OutputLayer.Neurons[j].BiasM:0:15);
      WriteLn(F, OutputLayer.Neurons[j].BiasV:0:15);
      WriteLn(F, Length(OutputLayer.Neurons[j].Weights));
      for k := 0 to High(OutputLayer.Neurons[j].Weights) do
      begin
         WriteLn(F, OutputLayer.Neurons[j].Weights[k]:0:15);
         WriteLn(F, OutputLayer.Neurons[j].WeightsM[k]:0:15);
         WriteLn(F, OutputLayer.Neurons[j].WeightsV[k]:0:15);
      end;
   end;
   
   CloseFile(F);
   WriteLn('CNN model saved to ', Filename);
end;

procedure TConvolutionalNeuralNetwork.LoadCNNModel(const Filename: string);
begin
   WriteLn('Load not fully implemented - recreate network and load weights');
end;

// Example usage
var
   CNN: TConvolutionalNeuralNetwork;
   Image: TImageData;
   Target: Darray;
   Prediction: Darray;
   Loss: Double;
   i, j, Epoch: Integer;
   Correct, Total: Integer;
   MaxIdx, TargetIdx: Integer;
   MaxVal: Double;
begin
   Randomize;
   
   // Create a simple 28x28 grayscale image
   Image.Width := 28;
   Image.Height := 28;
   Image.Channels := 1;
   SetLength(Image.Data, 1, 28, 28);
   
   // Fill with pattern (diagonal gradient)
   for i := 0 to 27 do
      for j := 0 to 27 do
         Image.Data[0][i][j] := (i + j) / 54.0;  // Normalized to [0,1]
   
   // Target: class 0
   SetLength(Target, 10);
   for i := 0 to 9 do
      Target[i] := 0;
   Target[0] := 1;
   
   // Create CNN with Adam optimizer
   CNN := TConvolutionalNeuralNetwork.Create(28, 28, 1, [8, 16], [3, 3], [2, 2], [64], 10, 0.001, 0.0);
   
   WriteLn('Training CNN on sample image...');
   WriteLn;
   
   // Training loop
   for Epoch := 1 to 50 do
   begin
      Loss := CNN.TrainStep(Image, Target);
      
      if (Epoch mod 10 = 0) or (Epoch = 1) then
      begin
         Prediction := CNN.Predict(Image);
         
         // Find predicted class
         MaxVal := -1e308;
         MaxIdx := 0;
         for i := 0 to High(Prediction) do
            if Prediction[i] > MaxVal then
            begin
               MaxVal := Prediction[i];
               MaxIdx := i;
            end;
         
         WriteLn('Epoch ', Epoch, ': Loss = ', Loss:0:6, 
                 ', Predicted class = ', MaxIdx, 
                 ', Confidence = ', (MaxVal * 100):0:2, '%');
      end;
   end;
   
   WriteLn;
   
   // Final prediction
   Prediction := CNN.Predict(Image);
   Write('Final prediction: [');
   for i := 0 to High(Prediction) do
   begin
      Write(Prediction[i]:0:4);
      if i < High(Prediction) then Write(', ');
   end;
   WriteLn(']');
   
   // Save model
   CNN.SaveCNNModel('TestCNN_v2.txt');
   
   CNN.Free;
   WriteLn('Done!');
end.

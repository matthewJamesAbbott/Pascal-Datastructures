{------------------------------------------------------------------------------
  Facaded CNN
  Matthew Abbott, 2025

------------------------------------------------------------------------------}

program CNNProgram;

{$mode objfpc}{$H+}

uses Classes, Math, SysUtils, StrUtils;

const
   EPSILON = 1e-8; // Small number for numerical stability (avoid /0)
   GRAD_CLIP = 1.0; // Clip gradients at this value to avoid exploding gradients

type
   // Common array types for storing weights, activations, filters, etc.
   Darray = array of Double;
   D2array = array of array of Double;
   D3array = array of array of array of Double;
   D4array = array of array of array of array of Double;
   IntArray = array of Integer;

   // Index used for pooling -- points to (X,Y) max in a pooling window
   TPoolIndex = record
      X, Y: Integer;
   end;
   TPoolIndexArray = array of array of array of TPoolIndex;

   // Useful for summarizing statistics about a layer’s activations/weights
   TLayerStats = record
      Mean: Double;
      StdDev: Double;
      Min: Double;
      Max: Double;
      Count: Integer;
   end;

   // Configuration for a given network layer (introspection/debug)
   TLayerConfig = record
      LayerType: string;           // “conv”, “pool”, or “fc”
      FilterCount: Integer;        // For conv: how many filters
      KernelSize: Integer;
      Stride: Integer;
      Padding: Integer;
      InputChannels: Integer;
      OutputWidth: Integer;
      OutputHeight: Integer;
      PoolSize: Integer;           // For pool layers
      NeuronCount: Integer;        // For fully connected
      InputSize: Integer;          // For fully connected: neurons’ input size
   end;

   // For computing receptive fields (advanced visualization)
   TReceptiveField = record
      StartX, EndX: Integer;
      StartY, EndY: Integer;
      Channels: IntArray;          // Which input channels are connected
   end;

   // For arbitrary string tags/metadata on net filters/layers
   TAttributeEntry = record
      Key: string;
      Value: string;
   end;
   TAttributeArray = array of TAttributeEntry;

   // A single image (input), as 3D array [channels][height][width]
   TImageData = record
      Width: Integer;
      Height: Integer;
      Channels: Integer;
      Data: D3array;               // Data[channel][row][col]
   end;

   // A single convolution filter (including gradient and optimizer state)
   TConvFilter = record
      Weights: D3array;            // Main weights [Channel][H][W]
      Bias: Double;
      WeightsM: D3array;           // Adam optimizer moment-1
      WeightsV: D3array;           // Adam optimizer moment-2
      WeightGrads: D3array;        // Gradients dL/dW
      BiasGrad: Double;
      BiasM: Double;               // For bias Adam optimizer
      BiasV: Double;
   end;

   // A convolution layer holds several filters & their output/activations
   TConvLayer = record
      Filters: array of TConvFilter;     // List of filters in this layer
      OutputMaps: D3array;               // Activations [Filter][H][W]
      PreActivation: D3array;            // Z maps before ReLU [Filter][H][W]
      InputCache: D3array;               // Cached input for backwards
      PaddedInput: D3array;              // Input w/padding (for backprop)
      Stride: Integer;
      Padding: Integer;
      KernelSize: Integer;
      InputChannels: Integer;
   end;

   // A pooling layer records window size and related caches
   TPoolingLayer = record
      PoolSize: Integer;
      Stride: Integer;
      OutputMaps: D3array;               // Output after pooling
      InputCache: D3array;               // What was pooled over
      MaxIndices: TPoolIndexArray;       // Where was the max for each win
   end;

   // A neuron in a fully connected (dense) layer
   TNeuron = record
      Weights: Darray;
      Bias: Double;
      Output: Double;                    // Activation value
      PreActivation: Double;             // Z before activation
      Error: Double;                     // Gradient w.r.t activation for backprop
      DropoutMask: Double;               // 0 or scaled 1 (for dropout regularization)
      WeightsM: Darray;                  // Adam moment-1
      WeightsV: Darray;                  // Adam moment-2
      BiasM: Double;
      BiasV: Double;
   end;

   // Fully connected layer: array of neurons, stores input for backprop
   TFullyConnectedLayer = record
      Neurons: array of TNeuron;
      InputCache: Darray;
   end;

   // Batch normalization parameters and state (for advanced usage)
   TBatchNormParams = record
      Gamma: Darray;
      Beta: Darray;
      RunningMean: Darray;
      RunningVar: Darray;
      Enabled: Boolean;
   end;

   {--------------------------------------------------------------------------
    MAIN CLASS: TCNNFacade
    All properties and methods attached to this object.
    -------------------------------------------------------------------------}
   TCNNFacade = class
   private
      // Hyperparameters, optimizer and state
      LearningRate: Double;
      DropoutRate: Double;
      Beta1, Beta2: Double;
      AdamT: Integer;
      IsTraining: Boolean;

      // Input dimensions
      FInputWidth, FInputHeight, FInputChannels: Integer;

      // Layers of the network, in order of computation
      ConvLayers: array of TConvLayer;
      PoolLayers: array of TPoolingLayer;
      FullyConnectedLayers: array of TFullyConnectedLayer;
      OutputLayer: TFullyConnectedLayer;

      // Sizes for internal bookkeeping
      FlattenedSize: Integer;
      FFlattenedFeatures: Darray;
      LastConvHeight, LastConvWidth, LastConvChannels: Integer;

      // Stubs for batch (mini-batch) functionality, not complete in base version
      FBatchActivations: array of D4array;
      FBatchNormParams: array of TBatchNormParams;
      FFilterAttributes: array of array of TAttributeArray;

      {--------------------- INTERNAL UTILITY METHODS ----------------------}
      function ReLU(x: Double): Double;
      function ReLUDerivative(x: Double): Double;
      function Softmax(const Logits: Darray): Darray;
      function CrossEntropyLoss(const Predicted, Target: Darray): Double;
      function ClipGrad(x: Double): Double;
      function IsFiniteNum(x: Double): Boolean;
      function Clamp(x, MinVal, MaxVal: Double): Double;
      function Pad3D(const Input: D3array; Padding: Integer): D3array;
      function ValidateInput(const Image: TImageData): Boolean;

      // Layer/parameter initializers
      procedure InitializeConvLayer(var Layer: TConvLayer; NumFilters, InputChannels, KernelSize, Stride, Padding: Integer);
      procedure InitializePoolLayer(var Layer: TPoolingLayer; PoolSize, Stride: Integer);
      procedure InitializeFCLayer(var Layer: TFullyConnectedLayer; NumNeurons, NumInputs: Integer);

       // Forward/Backward steps for each layer type
      procedure ConvForward(var Layer: TConvLayer; const Input: D3array; InputWidth, InputHeight: Integer);
      procedure PoolForward(var Layer: TPoolingLayer; const Input: D3array; InputWidth, InputHeight: Integer);
      procedure FlattenFeatures(const Input: D3array; InputWidth, InputHeight, InputChannels: Integer);
      procedure FCForward(var Layer: TFullyConnectedLayer; const Input: Darray);

      function ConvBackward(var Layer: TConvLayer; const Grad: D3array): D3array;
      function PoolBackward(var Layer: TPoolingLayer; const Grad: D3array): D3array;
      function FCBackward(var Layer: TFullyConnectedLayer; const Grad: Darray; IsOutputLayer: Boolean): Darray;
      function UnflattenGradient(const Grad: Darray): D3array;

      procedure UpdateWeights;
      procedure ApplyDropout(var Layer: TFullyConnectedLayer);
      
      { JSON Helper functions }
      function Array1DToJSON(const Arr: Darray): string;
      function Array2DToJSON(const Arr: D2array): string;
      function Array3DToJSON(const Arr: D3array): string;

   public
      {================= CONSTRUCTION / TEARDOWN =================}
      constructor Create(InputWidth, InputHeight, InputChannels: Integer;
                        ConvFilters, KernelSizes, PoolSizes, FCLayerSizes: array of Integer;
                        OutputSize: Integer; ALearningRate: Double = 0.001; ADropoutRate: Double = 0.25);
      destructor Destroy; override;

      {================== MAIN FUNCTIONALITY =====================}
      function Predict(var Image: TImageData): Darray;                      // Run the model on an image
      function TrainStep(var Image: TImageData; const Target: Darray): Double;   // Train on a batch, return loss
      procedure SaveModel(const Filename: string);
      procedure LoadModel(const Filename: string);
      
      { JSON serialization methods }
      procedure SaveModelToJSON(const Filename: string);
      procedure LoadModelFromJSON(const Filename: string);
      
      { Input dimension getters }
      function GetInputWidth: Integer;
      function GetInputHeight: Integer;
      function GetInputChannels: Integer;

      { Stage 1: Feature Map Access }
      function GetFeatureMap(LayerIdx, FilterIdx: Integer): D2array;
      procedure SetFeatureMap(LayerIdx, FilterIdx: Integer; const Map: D2array);

      { Stage 1: Pre-Activation Access }
      function GetPreActivation(LayerIdx, FilterIdx: Integer): D2array;
      procedure SetPreActivation(LayerIdx, FilterIdx: Integer; const Map: D2array);

      { Stage 2: Kernel/Filter Access - declarations only for now }
      function GetKernel(LayerIdx, FilterIdx, ChannelIdx: Integer): D2array;
      procedure SetKernel(LayerIdx, FilterIdx, ChannelIdx: Integer; const KernelArray: D2array);
      function GetBias(LayerIdx, FilterIdx: Integer): Double;
      procedure SetBias(LayerIdx, FilterIdx: Integer; Value: Double);

      { Stage 3: Batch Activations }
      function GetBatchActivations(LayerIdx: Integer): D4array;
      procedure SetBatchActivations(LayerIdx: Integer; const BatchTensor: D4array);

      { Stage 4: Pooling and Dropout States }
      function GetPoolingIndices(LayerIdx, FilterIdx: Integer): D2array;
      function GetDropoutMask(LayerIdx: Integer): Darray;
      procedure SetDropoutMask(LayerIdx: Integer; const Mask: Darray);

      { Stage 5: Gradients }
      function GetFilterGradient(LayerIdx, FilterIdx, ChannelIdx: Integer): D2array;
      function GetBiasGradient(LayerIdx, FilterIdx: Integer): Double;
      function GetActivationGradient(LayerIdx, FilterIdx, Y, X: Integer): Double;
      function GetOptimizerState(LayerIdx, FilterIdx: Integer; const Param: string): D3array;

      { Stage 6: Flattened Features }
      function GetFlattenedFeatures: Darray;
      procedure SetFlattenedFeatures(const Vector: Darray);

      { Stage 7: Output Layer }
      function GetLogits: Darray;
      function GetSoftmax: Darray;

      { Stage 8: Layer Config }
      function GetLayerConfig(LayerIdx: Integer): TLayerConfig;
      function GetNumLayers: Integer;
      function GetNumConvLayers: Integer;
      function GetNumFCLayers: Integer;
      function GetNumFilters(LayerIdx: Integer): Integer;

      { Stage 9: Saliency and Deconv }
      function GetSaliencyMap(LayerIdx, FilterIdx, InputIdx: Integer): D2array;
      function GetDeconv(LayerIdx, FilterIdx: Integer; UpToInput: Boolean): D3array;

      { Stage 10: Structural Mutations }
      procedure AddFilter(LayerIdx: Integer; const Params: D3array);
      procedure RemoveFilter(LayerIdx, FilterIdx: Integer);
      procedure AddConvLayer(Position: Integer; NumFilters, KernelSize, Stride, Padding: Integer);
      procedure RemoveLayer(LayerIdx: Integer);

      { Stage 11: Statistics }
      function GetLayerStats(LayerIdx: Integer): TLayerStats;
      function GetActivationHistogram(LayerIdx: Integer; NumBins: Integer = 50): Darray;
      function GetWeightHistogram(LayerIdx: Integer; NumBins: Integer = 50): Darray;

      { Stage 12: Receptive Field }
      function GetReceptiveField(LayerIdx, FeatureIdx, Y, X: Integer): TReceptiveField;

      { Stage 13: Batch Norm }
      function GetBatchNormParams(LayerIdx: Integer): TBatchNormParams;
      procedure SetBatchNormParams(LayerIdx: Integer; const Params: TBatchNormParams);

      { Stage 14: Attributes }
      procedure SetFilterAttribute(LayerIdx, FilterIdx: Integer; const Key, Value: string);
      function GetFilterAttribute(LayerIdx, FilterIdx: Integer; const Key: string): string;

      { Training state }
      procedure SetTrainingMode(Training: Boolean);
      function GetTrainingMode: Boolean;
   end;

{ Helper functions }

{ JSON serialization helper functions }
function TCNNFacade.Array1DToJSON(const Arr: Darray): string;
var
  i: Integer;
begin
  Result := '[';
  for i := 0 to High(Arr) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + FloatToStr(Arr[i]);
  end;
  Result := Result + ']';
end;

function TCNNFacade.Array2DToJSON(const Arr: D2array): string;
var
  i: Integer;
begin
  Result := '[';
  for i := 0 to High(Arr) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + Array1DToJSON(Arr[i]);
  end;
  Result := Result + ']';
end;

function TCNNFacade.Array3DToJSON(const Arr: D3array): string;
var
  i: Integer;
begin
  Result := '[';
  for i := 0 to High(Arr) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + Array2DToJSON(Arr[i]);
  end;
  Result := Result + ']';
end;

function TCNNFacade.IsFiniteNum(x: Double): Boolean;
begin
   Result := (not IsNan(x)) and (not IsInfinite(x));
end;

function TCNNFacade.Clamp(x, MinVal, MaxVal: Double): Double;
begin
   if x < MinVal then Result := MinVal
   else if x > MaxVal then Result := MaxVal
   else Result := x;
end;

function TCNNFacade.ClipGrad(x: Double): Double;
begin
   if not IsFiniteNum(x) then Result := 0
   else Result := Clamp(x, -GRAD_CLIP, GRAD_CLIP);
end;

function TCNNFacade.ReLU(x: Double): Double;
begin
   if x > 0 then Result := x else Result := 0.0;
end;

function TCNNFacade.ReLUDerivative(x: Double): Double;
begin
   if x > 0 then Result := 1.0 else Result := 0.0;
end;

function TCNNFacade.Softmax(const Logits: Darray): Darray;
var
   i: Integer;
   MaxVal, Sum, ExpVal: Double;
begin
   SetLength(Result, Length(Logits));
   MaxVal := -1e308;
   for i := 0 to High(Logits) do
      if IsFiniteNum(Logits[i]) and (Logits[i] > MaxVal) then
         MaxVal := Logits[i];
   if not IsFiniteNum(MaxVal) then MaxVal := 0;

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
   if (Sum <= 0) or (not IsFiniteNum(Sum)) then Sum := 1;
   for i := 0 to High(Result) do
      Result[i] := Clamp(Result[i] / Sum, 1e-15, 1 - 1e-15);
end;

function TCNNFacade.CrossEntropyLoss(const Predicted, Target: Darray): Double;
var
   i: Integer;
   P: Double;
begin
   Result := 0;
   for i := 0 to High(Target) do
      if Target[i] > 0 then
      begin
         P := Clamp(Predicted[i], 1e-15, 1 - 1e-15);
         Result := Result - Target[i] * Ln(P);
      end;
   if not IsFiniteNum(Result) then Result := 0;
end;

function TCNNFacade.Pad3D(const Input: D3array; Padding: Integer): D3array;
var
   c, h, w, Channels, Height, Width, SrcH, SrcW: Integer;
begin
   if Padding = 0 then begin Result := Input; Exit; end;
   Channels := Length(Input);
   Height := Length(Input[0]);
   Width := Length(Input[0][0]);
   SetLength(Result, Channels, Height + 2*Padding, Width + 2*Padding);
   for c := 0 to Channels - 1 do
      for h := 0 to Height + 2*Padding - 1 do
         for w := 0 to Width + 2*Padding - 1 do
         begin
            SrcH := h - Padding; SrcW := w - Padding;
            if (SrcH >= 0) and (SrcH < Height) and (SrcW >= 0) and (SrcW < Width) then
               Result[c][h][w] := Input[c][SrcH][SrcW]
            else
               Result[c][h][w] := 0;
         end;
end;

function TCNNFacade.ValidateInput(const Image: TImageData): Boolean;
var c, h, w: Integer;
begin
   Result := False;
   if (Image.Data = nil) or (Length(Image.Data) <> Image.Channels) then Exit;
   for c := 0 to Image.Channels - 1 do
   begin
      if (Image.Data[c] = nil) or (Length(Image.Data[c]) <> Image.Height) then Exit;
      for h := 0 to Image.Height - 1 do
      begin
         if (Image.Data[c][h] = nil) or (Length(Image.Data[c][h]) <> Image.Width) then Exit;
         for w := 0 to Image.Width - 1 do
            if not IsFiniteNum(Image.Data[c][h][w]) then Exit;
      end;
   end;
   Result := True;
end;

{ Placeholder - will continue in next chunk }

constructor TCNNFacade.Create(InputWidth, InputHeight, InputChannels: Integer;
    ConvFilters, KernelSizes, PoolSizes, FCLayerSizes: array of Integer;
    OutputSize: Integer; ALearningRate: Double; ADropoutRate: Double);
var
    i: Integer;
    CurrentWidth, CurrentHeight, CurrentChannels: Integer;
    NumInputs, KernelPadding: Integer;
begin
    inherited Create;
    LearningRate := ALearningRate;
    DropoutRate := ADropoutRate;
    Beta1 := 0.9;
    Beta2 := 0.999;
    AdamT := 0;
    IsTraining := False;
    
    { Store input dimensions }
    FInputWidth := InputWidth;
    FInputHeight := InputHeight;
    FInputChannels := InputChannels;
    
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
   
   LastConvWidth := CurrentWidth;
   LastConvHeight := CurrentHeight;
   LastConvChannels := CurrentChannels;
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

destructor TCNNFacade.Destroy;
begin
   inherited Destroy;
end;

{ Stub implementations - to be filled in next chunks }

function TCNNFacade.Predict(var Image: TImageData): Darray;
var
   i, j: Integer;
   CurrentMaps: D3array;
   CurrentWidth, CurrentHeight: Integer;
   LayerInput: Darray;
   Logits: Darray;
   Sum: Double;
begin
   if not ValidateInput(Image) then
   begin
      SetLength(Result, Length(OutputLayer.Neurons));
      Exit;
   end;
   
   CurrentMaps := Image.Data;
   CurrentWidth := Image.Width;
   CurrentHeight := Image.Height;
   
   for i := 0 to High(ConvLayers) do
   begin
      ConvForward(ConvLayers[i], CurrentMaps, CurrentWidth, CurrentHeight);
      CurrentWidth := Length(ConvLayers[i].OutputMaps[0][0]);
      CurrentHeight := Length(ConvLayers[i].OutputMaps[0]);
      
      if i <= High(PoolLayers) then
      begin
         PoolForward(PoolLayers[i], ConvLayers[i].OutputMaps, CurrentWidth, CurrentHeight);
         CurrentMaps := PoolLayers[i].OutputMaps;
         CurrentWidth := Length(PoolLayers[i].OutputMaps[0][0]);
         CurrentHeight := Length(PoolLayers[i].OutputMaps[0]);
      end
      else
         CurrentMaps := ConvLayers[i].OutputMaps;
   end;
   
   FlattenFeatures(CurrentMaps, CurrentWidth, CurrentHeight, Length(CurrentMaps));
   LayerInput := FFlattenedFeatures;
   
   for i := 0 to High(FullyConnectedLayers) do
   begin
      ApplyDropout(FullyConnectedLayers[i]);
      FCForward(FullyConnectedLayers[i], LayerInput);
      SetLength(LayerInput, Length(FullyConnectedLayers[i].Neurons));
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
         LayerInput[j] := FullyConnectedLayers[i].Neurons[j].Output;
   end;
   
   OutputLayer.InputCache := LayerInput;
   SetLength(Logits, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to High(LayerInput) do
         Sum := Sum + LayerInput[j] * OutputLayer.Neurons[i].Weights[j];
      if not IsFiniteNum(Sum) then Sum := 0;
      OutputLayer.Neurons[i].PreActivation := Sum;
      Logits[i] := Sum;
   end;
   
   Result := Softmax(Logits);
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Output := Result[i];
end;

function TCNNFacade.TrainStep(var Image: TImageData; const Target: Darray): Double;
var
   Prediction: Darray;
   OutputGrad, FCGrad: Darray;
   ConvGrad: D3array;
   i: Integer;
begin
   if not ValidateInput(Image) then
   begin
      Result := 0;
      Exit;
   end;
   
   IsTraining := True;
   Prediction := Predict(Image);
   
   SetLength(OutputGrad, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
      OutputGrad[i] := OutputLayer.Neurons[i].Output - Target[i];
   
   FCGrad := FCBackward(OutputLayer, OutputGrad, True);
   
   for i := High(FullyConnectedLayers) downto 0 do
      FCGrad := FCBackward(FullyConnectedLayers[i], FCGrad, False);
   
   ConvGrad := UnflattenGradient(FCGrad);
   
   for i := High(ConvLayers) downto 0 do
   begin
      if i <= High(PoolLayers) then
         ConvGrad := PoolBackward(PoolLayers[i], ConvGrad);
      ConvGrad := ConvBackward(ConvLayers[i], ConvGrad);
   end;
   
   UpdateWeights;
   Result := CrossEntropyLoss(Prediction, Target);
end;

procedure TCNNFacade.SaveModelToJSON(const Filename: string);
var
  JSON: TStringList;
  f, c, h, w, i, j, k: Integer;
  FilterList, WeightsList, NeuronList, LayerList: TStringList;
begin
  JSON := TStringList.Create;
  try
    { Header }
    JSON.Add('{');
    JSON.Add('  "version": "1.0",');
    JSON.Add('  "metadata": {');
    JSON.Add('    "framework": "FacadeCNN",');
    JSON.Add('    "createdAt": "' + DateTimeToStr(Now) + '",');
    JSON.Add('    "precision": "double"');
    JSON.Add('  },');
    
    { Configuration }
    JSON.Add('  "config": {');
    JSON.Add('    "inputWidth": ' + IntToStr(GetInputWidth) + ',');
    JSON.Add('    "inputHeight": ' + IntToStr(GetInputHeight) + ',');
    JSON.Add('    "inputChannels": ' + IntToStr(GetInputChannels) + ',');
    JSON.Add('    "convFilterCounts": [');
    for i := 0 to High(ConvLayers) do
    begin
      JSON.Add('      ' + IntToStr(Length(ConvLayers[i].Filters)));
      if i < High(ConvLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('    ],');
    JSON.Add('    "kernelSizes": [');
    for i := 0 to High(ConvLayers) do
    begin
      JSON.Add('      ' + IntToStr(ConvLayers[i].KernelSize));
      if i < High(ConvLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('    ],');
    JSON.Add('    "poolSizes": [');
    for i := 0 to High(PoolLayers) do
    begin
      JSON.Add('      ' + IntToStr(PoolLayers[i].PoolSize));
      if i < High(PoolLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('    ],');
    JSON.Add('    "fcLayerSizes": [');
    for i := 0 to High(FullyConnectedLayers) do
    begin
      JSON.Add('      ' + IntToStr(Length(FullyConnectedLayers[i].Neurons)));
      if i < High(FullyConnectedLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('    ],');
    JSON.Add('    "outputSize": ' + IntToStr(Length(OutputLayer.Neurons)) + ',');
    JSON.Add('    "learningRate": ' + FloatToStr(LearningRate) + ',');
    JSON.Add('    "dropoutRate": ' + FloatToStr(DropoutRate));
    JSON.Add('  },');
    
    { Hyperparameters }
    JSON.Add('  "hyperparameters": {');
    JSON.Add('    "beta1": ' + FloatToStr(Beta1) + ',');
    JSON.Add('    "beta2": ' + FloatToStr(Beta2) + ',');
    JSON.Add('    "adamT": ' + IntToStr(AdamT));
    JSON.Add('  },');
    
    { Convolutional layers }
    JSON.Add('  "convLayers": [');
    for i := 0 to High(ConvLayers) do
    begin
      JSON.Add('    {');
      JSON.Add('      "layerIndex": ' + IntToStr(i) + ',');
      JSON.Add('      "stride": ' + IntToStr(ConvLayers[i].Stride) + ',');
      JSON.Add('      "padding": ' + IntToStr(ConvLayers[i].Padding) + ',');
      JSON.Add('      "kernelSize": ' + IntToStr(ConvLayers[i].KernelSize) + ',');
      JSON.Add('      "inputChannels": ' + IntToStr(ConvLayers[i].InputChannels) + ',');
      JSON.Add('      "filters": [');
      
      for f := 0 to High(ConvLayers[i].Filters) do
      begin
        JSON.Add('        {');
        JSON.Add('          "filterIndex": ' + IntToStr(f) + ',');
        JSON.Add('          "bias": ' + FloatToStr(ConvLayers[i].Filters[f].Bias) + ',');
        JSON.Add('          "biasM": ' + FloatToStr(ConvLayers[i].Filters[f].BiasM) + ',');
        JSON.Add('          "biasV": ' + FloatToStr(ConvLayers[i].Filters[f].BiasV) + ',');
        JSON.Add('          "biasGrad": ' + FloatToStr(ConvLayers[i].Filters[f].BiasGrad) + ',');
        JSON.Add('          "weights": ' + Array3DToJSON(ConvLayers[i].Filters[f].Weights) + ',');
        JSON.Add('          "weightsM": ' + Array3DToJSON(ConvLayers[i].Filters[f].WeightsM) + ',');
        JSON.Add('          "weightsV": ' + Array3DToJSON(ConvLayers[i].Filters[f].WeightsV) + ',');
        JSON.Add('          "weightGrads": ' + Array3DToJSON(ConvLayers[i].Filters[f].WeightGrads));
        JSON.Add('        }');
        if f < High(ConvLayers[i].Filters) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
      end;
      
      JSON.Add('      ]');
      JSON.Add('    }');
      if i < High(ConvLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('  ],');
    
    { Fully connected layers }
    JSON.Add('  "fcLayers": [');
    for i := 0 to High(FullyConnectedLayers) do
    begin
      JSON.Add('    {');
      JSON.Add('      "layerIndex": ' + IntToStr(i) + ',');
      JSON.Add('      "neurons": [');
      
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
      begin
        JSON.Add('        {');
        JSON.Add('          "neuronIndex": ' + IntToStr(j) + ',');
        JSON.Add('          "bias": ' + FloatToStr(FullyConnectedLayers[i].Neurons[j].Bias) + ',');
        JSON.Add('          "biasM": ' + FloatToStr(FullyConnectedLayers[i].Neurons[j].BiasM) + ',');
        JSON.Add('          "biasV": ' + FloatToStr(FullyConnectedLayers[i].Neurons[j].BiasV) + ',');
        JSON.Add('          "weights": ' + Array1DToJSON(FullyConnectedLayers[i].Neurons[j].Weights) + ',');
        JSON.Add('          "weightsM": ' + Array1DToJSON(FullyConnectedLayers[i].Neurons[j].WeightsM) + ',');
        JSON.Add('          "weightsV": ' + Array1DToJSON(FullyConnectedLayers[i].Neurons[j].WeightsV));
        JSON.Add('        }');
        if j < High(FullyConnectedLayers[i].Neurons) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
      end;
      
      JSON.Add('      ]');
      JSON.Add('    }');
      if i < High(FullyConnectedLayers) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('  ],');
    
    { Output layer }
    JSON.Add('  "outputLayer": {');
    JSON.Add('    "neurons": [');
    for j := 0 to High(OutputLayer.Neurons) do
    begin
      JSON.Add('      {');
      JSON.Add('        "neuronIndex": ' + IntToStr(j) + ',');
      JSON.Add('        "bias": ' + FloatToStr(OutputLayer.Neurons[j].Bias) + ',');
      JSON.Add('        "biasM": ' + FloatToStr(OutputLayer.Neurons[j].BiasM) + ',');
      JSON.Add('        "biasV": ' + FloatToStr(OutputLayer.Neurons[j].BiasV) + ',');
      JSON.Add('        "weights": ' + Array1DToJSON(OutputLayer.Neurons[j].Weights) + ',');
      JSON.Add('        "weightsM": ' + Array1DToJSON(OutputLayer.Neurons[j].WeightsM) + ',');
      JSON.Add('        "weightsV": ' + Array1DToJSON(OutputLayer.Neurons[j].WeightsV));
      JSON.Add('      }');
      if j < High(OutputLayer.Neurons) then JSON[JSON.Count - 1] := JSON[JSON.Count - 1] + ',';
    end;
    JSON.Add('    ]');
    JSON.Add('  }');
    
    JSON.Add('}');
    
    { Save to file }
    JSON.SaveToFile(Filename);
    WriteLn('Model saved to: ' + Filename);
    
    finally
    JSON.Free;
    end;
    end;

    { JSON Helper Functions }
    function ExtractIntFromJSON(const JSONStr, FieldName: string): Integer;
    var
    P, EndP: Integer;
    Value: string;
    begin
    P := Pos('"' + FieldName + '"', JSONStr);
    if P = 0 then Exit(0);
    
    P := PosEx(':', JSONStr, P);
    if P = 0 then Exit(0);
    
    P := P + 1;
    while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
    
    EndP := P;
    while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-']) do Inc(EndP);
    
    Value := Copy(JSONStr, P, EndP - P);
    try
    Result := StrToInt(Value);
    except
    Result := 0;
    end;
    end;

    function ExtractDoubleFromJSON(const JSONStr, FieldName: string): Double;
    var
    P, EndP: Integer;
    Value: string;
    begin
    P := Pos('"' + FieldName + '"', JSONStr);
    if P = 0 then Exit(0.0);
    
    P := PosEx(':', JSONStr, P);
    if P = 0 then Exit(0.0);
    
    P := P + 1;
    while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
    
    EndP := P;
    while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-', '.', 'e', 'E']) do Inc(EndP);
    
    Value := Copy(JSONStr, P, EndP - P);
    try
    Result := StrToFloat(Value);
    except
    Result := 0.0;
    end;
    end;

    function CountArrayElements(const JSONStr, ArrayName: string): Integer;
    var
    P, Count: Integer;
    begin
    P := Pos('"' + ArrayName + '"', JSONStr);
    if P = 0 then Exit(0);
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit(0);
    
    Count := 0;
    Inc(P);
    while (P <= Length(JSONStr)) and (JSONStr[P] <> ']') do
    begin
    if JSONStr[P] = '{' then Inc(Count);
    Inc(P);
    end;
    
    Result := Count;
    end;

    function CountNestedArrayElements(const JSONStr, ArrayName: string; Index: Integer; NestedName: string): Integer;
    var
    P, Count, ElementPos, NestedP: Integer;
    begin
    P := Pos('"' + ArrayName + '"', JSONStr);
    if P = 0 then Exit(0);
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit(0);
    
    Count := 0;
    ElementPos := P + 1;
    while (Count < Index) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> Index then Exit(0);
    
    NestedP := Pos('"' + NestedName + '"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if NestedP = 0 then Exit(0);
    
    NestedP := ElementPos + NestedP - 1;
    NestedP := PosEx('[', JSONStr, NestedP);
    if NestedP = 0 then Exit(0);
    
    Count := 0;
    Inc(NestedP);
    while (NestedP <= Length(JSONStr)) and (JSONStr[NestedP] <> ']') do
    begin
    if JSONStr[NestedP] = '{' then Inc(Count);
    Inc(NestedP);
    end;
    
    Result := Count;
    end;

    function ExtractIntFromJSONArray(const JSONStr, ArrayName: string; Index: Integer; FieldName: string): Integer;
    var
    ArrayPos, ElementPos, FieldPos: Integer;
    P, EndP: Integer;
    Value: string;
    Count: Integer;
    begin
    ArrayPos := Pos('"' + ArrayName + '"', JSONStr);
    if ArrayPos = 0 then Exit(0);
    
    ArrayPos := PosEx('[', JSONStr, ArrayPos);
    if ArrayPos = 0 then Exit(0);
    
    Count := 0;
    ElementPos := ArrayPos + 1;
    while (Count < Index) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> Index then Exit(0);
    
    FieldPos := Pos('"' + FieldName + '"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if FieldPos = 0 then Exit(0);
    
    P := ElementPos + FieldPos - 1;
    P := PosEx(':', JSONStr, P);
    if P = 0 then Exit(0);
    
    Inc(P);
    while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
    
    EndP := P;
    while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-']) do Inc(EndP);
    
    Value := Copy(JSONStr, P, EndP - P);
    try
    Result := StrToInt(Value);
    except
    Result := 0;
    end;
    end;

    function ExtractDoubleFromNestedJSON(const JSONStr, ArrayName: string; Index: Integer; NestedName: string; NestedIndex: Integer; FieldName: string): Double;
    var
    P, Count, ElementPos, NestedP, NestedElementPos, NestedCount, FieldPos, EndP: Integer;
    Value: string;
    begin
    P := Pos('"' + ArrayName + '"', JSONStr);
    if P = 0 then Exit(0.0);
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit(0.0);
    
    Count := 0;
    ElementPos := P + 1;
    while (Count < Index) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> Index then Exit(0.0);
    
    NestedP := Pos('"' + NestedName + '"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if NestedP = 0 then Exit(0.0);
    
    NestedP := ElementPos + NestedP - 1;
    NestedP := PosEx('[', JSONStr, NestedP);
    if NestedP = 0 then Exit(0.0);
    
    NestedCount := 0;
    NestedElementPos := NestedP + 1;
    while (NestedCount < NestedIndex) and (NestedElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[NestedElementPos] = '{' then Inc(NestedCount);
    Inc(NestedElementPos);
    end;
    
    if NestedCount <> NestedIndex then Exit(0.0);
    
    FieldPos := Pos('"' + FieldName + '"', Copy(JSONStr, NestedElementPos, Length(JSONStr)));
    if FieldPos = 0 then Exit(0.0);
    
    P := NestedElementPos + FieldPos - 1;
    P := PosEx(':', JSONStr, P);
    if P = 0 then Exit(0.0);
    
    Inc(P);
    while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
    
    EndP := P;
    while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-', '.', 'e', 'E']) do Inc(EndP);
    
    Value := Copy(JSONStr, P, EndP - P);
    try
    Result := StrToFloat(Value);
    except
    Result := 0.0;
    end;
    end;

    procedure LoadWeights1DFromJSON(const JSONStr, ArrayName: string; Index: Integer; NestedName: string; NestedIndex: Integer; FieldName: string; var Arr: Darray);
    var
    P, ElementPos, Count, NestedElementPos, NestedCount: Integer;
    ArrayStartPos, ArrayEndPos, CurrentPos, NumPos: Integer;
    Value: string;
    DataCount: Integer;
    begin
    P := Pos('"' + ArrayName + '"', JSONStr);
    if P = 0 then Exit;
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit;
    
    Count := 0;
    ElementPos := P + 1;
    while (Count < Index) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> Index then Exit;
    
    P := Pos('"' + NestedName + '"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if P = 0 then Exit;
    
    P := ElementPos + P - 1;
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit;
    
    NestedCount := 0;
    NestedElementPos := P + 1;
    while (NestedCount < NestedIndex) and (NestedElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[NestedElementPos] = '{' then Inc(NestedCount);
    Inc(NestedElementPos);
    end;
    
    if NestedCount <> NestedIndex then Exit;
    
    P := Pos('"' + FieldName + '"', Copy(JSONStr, NestedElementPos, Length(JSONStr)));
    if P = 0 then Exit;
    
    P := NestedElementPos + P - 1;
    ArrayStartPos := PosEx('[', JSONStr, P);
    if ArrayStartPos = 0 then Exit;
    
    Count := 1;
    ArrayEndPos := ArrayStartPos + 1;
    while (Count > 0) and (ArrayEndPos <= Length(JSONStr)) do
    begin
    if JSONStr[ArrayEndPos] = '[' then Inc(Count)
    else if JSONStr[ArrayEndPos] = ']' then Dec(Count);
    Inc(ArrayEndPos);
    end;
    
    SetLength(Arr, 0);
    CurrentPos := ArrayStartPos + 1;
    DataCount := 0;
    
    while (CurrentPos < ArrayEndPos) and (JSONStr[CurrentPos] <> ']') do
    begin
    if JSONStr[CurrentPos] in ['0'..'9', '-', '.'] then
    begin
     NumPos := CurrentPos;
     while (NumPos <= Length(JSONStr)) and (JSONStr[NumPos] in ['0'..'9', '-', '.', 'e', 'E']) do
       Inc(NumPos);
     
     Value := Copy(JSONStr, CurrentPos, NumPos - CurrentPos);
     SetLength(Arr, DataCount + 1);
     try
       Arr[DataCount] := StrToFloat(Value);
     except
       Arr[DataCount] := 0.0;
     end;
     Inc(DataCount);
     
     CurrentPos := NumPos;
    end
    else
     Inc(CurrentPos);
    end;
    end;

    procedure LoadWeights3DFromJSON(const JSONStr, ArrayName: string; Index: Integer; NestedName: string; NestedIndex: Integer; FieldName: string; var Arr: D3array);
    var
    P, ElementPos, Count, NestedElementPos, NestedCount: Integer;
    ArrayStartPos, ArrayEndPos, CurrentPos, NumPos, BracketDepth: Integer;
    Value: string;
    D1, D2, D3: Integer;
    begin
    P := Pos('"' + ArrayName + '"', JSONStr);
    if P = 0 then Exit;
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit;
    
    Count := 0;
    ElementPos := P + 1;
    while (Count < Index) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> Index then Exit;
    
    P := Pos('"' + NestedName + '"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if P = 0 then Exit;
    
    P := ElementPos + P - 1;
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit;
    
    NestedCount := 0;
    NestedElementPos := P + 1;
    while (NestedCount < NestedIndex) and (NestedElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[NestedElementPos] = '{' then Inc(NestedCount);
    Inc(NestedElementPos);
    end;
    
    if NestedCount <> NestedIndex then Exit;
    
    P := Pos('"' + FieldName + '"', Copy(JSONStr, NestedElementPos, Length(JSONStr)));
    if P = 0 then Exit;
    
    P := NestedElementPos + P - 1;
    ArrayStartPos := PosEx('[', JSONStr, P);
    if ArrayStartPos = 0 then Exit;
    
    Count := 1;
    ArrayEndPos := ArrayStartPos + 1;
    while (Count > 0) and (ArrayEndPos <= Length(JSONStr)) do
    begin
    if JSONStr[ArrayEndPos] = '[' then Inc(Count)
    else if JSONStr[ArrayEndPos] = ']' then Dec(Count);
    Inc(ArrayEndPos);
    end;
    
    SetLength(Arr, 0);
    CurrentPos := ArrayStartPos + 1;
    D1 := 0;
    D2 := 0;
    D3 := 0;
    BracketDepth := 0;
    
    while (CurrentPos < ArrayEndPos) do
    begin
    if JSONStr[CurrentPos] = '[' then
    begin
     Inc(BracketDepth);
     Inc(CurrentPos);
    end
    else if JSONStr[CurrentPos] = ']' then
    begin
     Dec(BracketDepth);
     if BracketDepth = 1 then
     begin
       Inc(D2);
       D3 := 0;
     end
     else if BracketDepth = 0 then
     begin
       Inc(D1);
       D2 := 0;
       D3 := 0;
     end;
     Inc(CurrentPos);
    end
    else if JSONStr[CurrentPos] in ['0'..'9', '-', '.'] then
    begin
     NumPos := CurrentPos;
     while (NumPos <= Length(JSONStr)) and (JSONStr[NumPos] in ['0'..'9', '-', '.', 'e', 'E']) do
       Inc(NumPos);
     
     Value := Copy(JSONStr, CurrentPos, NumPos - CurrentPos);
     try
       if D1 >= Length(Arr) then SetLength(Arr, D1 + 1);
       if D2 >= Length(Arr[D1]) then SetLength(Arr[D1], D2 + 1);
       if D3 >= Length(Arr[D1][D2]) then SetLength(Arr[D1][D2], D3 + 1);
       
       Arr[D1][D2][D3] := StrToFloat(Value);
       Inc(D3);
     except
     end;
     
     CurrentPos := NumPos;
    end
    else
     Inc(CurrentPos);
    end;
    end;

    procedure LoadWeights1DFromJSONOutputLayer(const JSONStr: string; NeuronIndex: Integer; var Arr: Darray);
    var
    P, Count, ElementPos, FieldPos, ArrayStartPos, ArrayEndPos: Integer;
    CurrentPos, NumPos: Integer;
    Value: string;
    DataCount: Integer;
    begin
    P := Pos('"outputLayer"', JSONStr);
    if P = 0 then Exit;
    
    P := PosEx('"neurons"', JSONStr, P);
    if P = 0 then Exit;
    
    P := PosEx('[', JSONStr, P);
    if P = 0 then Exit;
    
    Count := 0;
    ElementPos := P + 1;
    while (Count < NeuronIndex) and (ElementPos <= Length(JSONStr)) do
    begin
    if JSONStr[ElementPos] = '{' then Inc(Count);
    Inc(ElementPos);
    end;
    
    if Count <> NeuronIndex then Exit;
    
    FieldPos := Pos('"weights"', Copy(JSONStr, ElementPos, Length(JSONStr)));
    if FieldPos = 0 then Exit;
    
    P := ElementPos + FieldPos - 1;
    ArrayStartPos := PosEx('[', JSONStr, P);
    if ArrayStartPos = 0 then Exit;
    
    Count := 1;
    ArrayEndPos := ArrayStartPos + 1;
    while (Count > 0) and (ArrayEndPos <= Length(JSONStr)) do
    begin
    if JSONStr[ArrayEndPos] = '[' then Inc(Count)
    else if JSONStr[ArrayEndPos] = ']' then Dec(Count);
    Inc(ArrayEndPos);
    end;
    
    SetLength(Arr, 0);
    CurrentPos := ArrayStartPos + 1;
    DataCount := 0;
    
    while (CurrentPos < ArrayEndPos) and (JSONStr[CurrentPos] <> ']') do
    begin
    if JSONStr[CurrentPos] in ['0'..'9', '-', '.'] then
    begin
     NumPos := CurrentPos;
     while (NumPos <= Length(JSONStr)) and (JSONStr[NumPos] in ['0'..'9', '-', '.', 'e', 'E']) do
       Inc(NumPos);
     
     Value := Copy(JSONStr, CurrentPos, NumPos - CurrentPos);
     SetLength(Arr, DataCount + 1);
     try
       Arr[DataCount] := StrToFloat(Value);
     except
       Arr[DataCount] := 0.0;
     end;
     Inc(DataCount);
     
     CurrentPos := NumPos;
    end
    else
     Inc(CurrentPos);
    end;
    end;

    procedure TCNNFacade.LoadModelFromJSON(const Filename: string);
var
  JSONContent: TStringList;
  JSONStr: string;
  i, j, f, c, InputW, InputH, InputC, OutputSize, NumConvLayers, NumFCLayers, NumFilters, NumNeurons: Integer;
  KernelSize, Stride, Padding, InputChannels, FilterCount, NumWeights: Integer;
  LR, DR: Double;
begin
  JSONContent := TStringList.Create;
  try
    { Load entire file }
    JSONContent.LoadFromFile(Filename);
    JSONStr := JSONContent.Text;
    
    { Parse configuration from JSON }
    InputW := ExtractIntFromJSON(JSONStr, 'inputWidth');
    InputH := ExtractIntFromJSON(JSONStr, 'inputHeight');
    InputC := ExtractIntFromJSON(JSONStr, 'inputChannels');
    OutputSize := ExtractIntFromJSON(JSONStr, 'outputSize');
    LR := ExtractDoubleFromJSON(JSONStr, 'learningRate');
    DR := ExtractDoubleFromJSON(JSONStr, 'dropoutRate');
    
    FInputWidth := InputW;
    FInputHeight := InputH;
    FInputChannels := InputC;
    LearningRate := LR;
    DropoutRate := DR;
    Beta1 := ExtractDoubleFromJSON(JSONStr, 'beta1');
    Beta2 := ExtractDoubleFromJSON(JSONStr, 'beta2');
    AdamT := ExtractIntFromJSON(JSONStr, 'adamT');
    
    { Parse convolutional layers }
    NumConvLayers := CountArrayElements(JSONStr, 'convLayers');
    SetLength(ConvLayers, NumConvLayers);
    
    for i := 0 to NumConvLayers - 1 do
    begin
      KernelSize := ExtractIntFromJSONArray(JSONStr, 'convLayers', i, 'kernelSize');
      Stride := ExtractIntFromJSONArray(JSONStr, 'convLayers', i, 'stride');
      Padding := ExtractIntFromJSONArray(JSONStr, 'convLayers', i, 'padding');
      InputChannels := ExtractIntFromJSONArray(JSONStr, 'convLayers', i, 'inputChannels');
      
      FilterCount := CountNestedArrayElements(JSONStr, 'convLayers', i, 'filters');
      SetLength(ConvLayers[i].Filters, FilterCount);
      ConvLayers[i].KernelSize := KernelSize;
      ConvLayers[i].Stride := Stride;
      ConvLayers[i].Padding := Padding;
      ConvLayers[i].InputChannels := InputChannels;
      
      for f := 0 to FilterCount - 1 do
      begin
        ConvLayers[i].Filters[f].Bias := ExtractDoubleFromNestedJSON(JSONStr, 'convLayers', i, 'filters', f, 'bias');
        ConvLayers[i].Filters[f].BiasM := ExtractDoubleFromNestedJSON(JSONStr, 'convLayers', i, 'filters', f, 'biasM');
        ConvLayers[i].Filters[f].BiasV := ExtractDoubleFromNestedJSON(JSONStr, 'convLayers', i, 'filters', f, 'biasV');
        ConvLayers[i].Filters[f].BiasGrad := ExtractDoubleFromNestedJSON(JSONStr, 'convLayers', i, 'filters', f, 'biasGrad');
        
        { Load 3D weights array for filters }
        LoadWeights3DFromJSON(JSONStr, 'convLayers', i, 'filters', f, 'weights', ConvLayers[i].Filters[f].Weights);
        LoadWeights3DFromJSON(JSONStr, 'convLayers', i, 'filters', f, 'weightsM', ConvLayers[i].Filters[f].WeightsM);
        LoadWeights3DFromJSON(JSONStr, 'convLayers', i, 'filters', f, 'weightsV', ConvLayers[i].Filters[f].WeightsV);
        LoadWeights3DFromJSON(JSONStr, 'convLayers', i, 'filters', f, 'weightGrads', ConvLayers[i].Filters[f].WeightGrads);
        
        { Initialize arrays if not loaded }
        if Length(ConvLayers[i].Filters[f].WeightGrads) = 0 then
        begin
          SetLength(ConvLayers[i].Filters[f].WeightGrads, Length(ConvLayers[i].Filters[f].Weights), 
                    Length(ConvLayers[i].Filters[f].Weights[0]), Length(ConvLayers[i].Filters[f].Weights[0][0]));
        end;
        if Length(ConvLayers[i].Filters[f].WeightsM) = 0 then
        begin
          SetLength(ConvLayers[i].Filters[f].WeightsM, Length(ConvLayers[i].Filters[f].Weights),
                    Length(ConvLayers[i].Filters[f].Weights[0]), Length(ConvLayers[i].Filters[f].Weights[0][0]));
        end;
        if Length(ConvLayers[i].Filters[f].WeightsV) = 0 then
        begin
          SetLength(ConvLayers[i].Filters[f].WeightsV, Length(ConvLayers[i].Filters[f].Weights),
                    Length(ConvLayers[i].Filters[f].Weights[0]), Length(ConvLayers[i].Filters[f].Weights[0][0]));
        end;
      end;
    end;
    
    { Parse fully connected layers }
    NumFCLayers := CountArrayElements(JSONStr, 'fcLayers');
    SetLength(FullyConnectedLayers, NumFCLayers);
    
    for i := 0 to NumFCLayers - 1 do
    begin
      NumNeurons := CountNestedArrayElements(JSONStr, 'fcLayers', i, 'neurons');
      SetLength(FullyConnectedLayers[i].Neurons, NumNeurons);
      
      for j := 0 to NumNeurons - 1 do
      begin
        FullyConnectedLayers[i].Neurons[j].Bias := ExtractDoubleFromNestedJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'bias');
        FullyConnectedLayers[i].Neurons[j].BiasM := ExtractDoubleFromNestedJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'biasM');
        FullyConnectedLayers[i].Neurons[j].BiasV := ExtractDoubleFromNestedJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'biasV');
        
        LoadWeights1DFromJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'weights', FullyConnectedLayers[i].Neurons[j].Weights);
        LoadWeights1DFromJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'weightsM', FullyConnectedLayers[i].Neurons[j].WeightsM);
        LoadWeights1DFromJSON(JSONStr, 'fcLayers', i, 'neurons', j, 'weightsV', FullyConnectedLayers[i].Neurons[j].WeightsV);
        
        { Initialize gradient arrays if not loaded }
        if Length(FullyConnectedLayers[i].Neurons[j].WeightsM) = 0 then
          SetLength(FullyConnectedLayers[i].Neurons[j].WeightsM, Length(FullyConnectedLayers[i].Neurons[j].Weights));
        if Length(FullyConnectedLayers[i].Neurons[j].WeightsV) = 0 then
          SetLength(FullyConnectedLayers[i].Neurons[j].WeightsV, Length(FullyConnectedLayers[i].Neurons[j].Weights));
      end;
    end;
    
    { Parse output layer }
    NumNeurons := CountNestedArrayElements(JSONStr, 'outputLayer', 0, 'neurons');
    SetLength(OutputLayer.Neurons, NumNeurons);
    
    for j := 0 to NumNeurons - 1 do
    begin
      OutputLayer.Neurons[j].Bias := ExtractDoubleFromNestedJSON(JSONStr, 'outputLayer', 0, 'neurons', j, 'bias');
      OutputLayer.Neurons[j].BiasM := ExtractDoubleFromNestedJSON(JSONStr, 'outputLayer', 0, 'neurons', j, 'biasM');
      OutputLayer.Neurons[j].BiasV := ExtractDoubleFromNestedJSON(JSONStr, 'outputLayer', 0, 'neurons', j, 'biasV');
      
      LoadWeights1DFromJSONOutputLayer(JSONStr, j, OutputLayer.Neurons[j].Weights);
      LoadWeights1DFromJSON(JSONStr, 'outputLayer', 0, 'neurons', j, 'weightsM', OutputLayer.Neurons[j].WeightsM);
      LoadWeights1DFromJSON(JSONStr, 'outputLayer', 0, 'neurons', j, 'weightsV', OutputLayer.Neurons[j].WeightsV);
      
      { Initialize gradient arrays if not loaded }
      if Length(OutputLayer.Neurons[j].WeightsM) = 0 then
        SetLength(OutputLayer.Neurons[j].WeightsM, Length(OutputLayer.Neurons[j].Weights));
      if Length(OutputLayer.Neurons[j].WeightsV) = 0 then
        SetLength(OutputLayer.Neurons[j].WeightsV, Length(OutputLayer.Neurons[j].Weights));
    end;
    
    WriteLn('Model loaded successfully from: ' + Filename);
    
  finally
    JSONContent.Free;
  end;
end;

{ Input dimension getters }
function TCNNFacade.GetInputWidth: Integer;
begin
  Result := FInputWidth;
end;

function TCNNFacade.GetInputHeight: Integer;
begin
  Result := FInputHeight;
end;

function TCNNFacade.GetInputChannels: Integer;
begin
  Result := FInputChannels;
end;

{ Original stub methods - kept for backward compatibility }
procedure TCNNFacade.SaveModel(const Filename: string);
begin
  SaveModelToJSON(Filename);
end;

procedure TCNNFacade.LoadModel(const Filename: string);
begin
  LoadModelFromJSON(Filename);
end;

procedure TCNNFacade.InitializeConvLayer(var Layer: TConvLayer; NumFilters, InputChannels, KernelSize, Stride, Padding: Integer);
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

procedure TCNNFacade.InitializePoolLayer(var Layer: TPoolingLayer; PoolSize, Stride: Integer);
begin
   Layer.PoolSize := PoolSize;
   Layer.Stride := Stride;
end;

procedure TCNNFacade.InitializeFCLayer(var Layer: TFullyConnectedLayer; NumNeurons, NumInputs: Integer);
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

procedure TCNNFacade.ConvForward(var Layer: TConvLayer; const Input: D3array; InputWidth, InputHeight: Integer);
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
      for h := 0 to OutputHeight - 1 do
         for w := 0 to OutputWidth - 1 do
         begin
            Sum := Layer.Filters[f].Bias;
            for c := 0 to Layer.InputChannels - 1 do
               for kh := 0 to Layer.KernelSize - 1 do
                  for kw := 0 to Layer.KernelSize - 1 do
                  begin
                     InH := h * Layer.Stride + kh;
                     InW := w * Layer.Stride + kw;
                     Sum := Sum + Layer.PaddedInput[c][InH][InW] * Layer.Filters[f].Weights[c][kh][kw];
                  end;
            Layer.PreActivation[f][h][w] := Sum;
            Layer.OutputMaps[f][h][w] := ReLU(Sum);
         end;
end;

procedure TCNNFacade.PoolForward(var Layer: TPoolingLayer; const Input: D3array; InputWidth, InputHeight: Integer);
var
   c, h, w, ph, pw: Integer;
   OutputWidth, OutputHeight: Integer;
   MaxVal, Val: Double;
   MaxX, MaxY: Integer;
begin
   Layer.InputCache := Input;
   OutputWidth := InputWidth div Layer.PoolSize;
   OutputHeight := InputHeight div Layer.PoolSize;
   
   SetLength(Layer.OutputMaps, Length(Input), OutputHeight, OutputWidth);
   SetLength(Layer.MaxIndices, Length(Input), OutputHeight, OutputWidth);
   
   for c := 0 to High(Input) do
      for h := 0 to OutputHeight - 1 do
         for w := 0 to OutputWidth - 1 do
         begin
            MaxVal := -1e308;
            MaxX := 0; MaxY := 0;
            for ph := 0 to Layer.PoolSize - 1 do
               for pw := 0 to Layer.PoolSize - 1 do
               begin
                  Val := Input[c][h * Layer.PoolSize + ph][w * Layer.PoolSize + pw];
                  if Val > MaxVal then
                  begin
                     MaxVal := Val;
                     MaxX := pw;
                     MaxY := ph;
                  end;
               end;
            Layer.OutputMaps[c][h][w] := MaxVal;
            Layer.MaxIndices[c][h][w].X := MaxX;
            Layer.MaxIndices[c][h][w].Y := MaxY;
         end;
end;

procedure TCNNFacade.FlattenFeatures(const Input: D3array; InputWidth, InputHeight, InputChannels: Integer);
var
   c, h, w, idx: Integer;
begin
   SetLength(FFlattenedFeatures, InputWidth * InputHeight * InputChannels);
   idx := 0;
   for c := 0 to InputChannels - 1 do
      for h := 0 to InputHeight - 1 do
         for w := 0 to InputWidth - 1 do
         begin
            FFlattenedFeatures[idx] := Input[c][h][w];
            Inc(idx);
         end;
end;

procedure TCNNFacade.FCForward(var Layer: TFullyConnectedLayer; const Input: Darray);
var
   i, j: Integer;
   Sum: Double;
begin
   Layer.InputCache := Input;
   for i := 0 to High(Layer.Neurons) do
   begin
      Sum := Layer.Neurons[i].Bias;
      for j := 0 to High(Input) do
         Sum := Sum + Input[j] * Layer.Neurons[i].Weights[j];
      Layer.Neurons[i].PreActivation := Sum;
      Layer.Neurons[i].Output := ReLU(Sum) * Layer.Neurons[i].DropoutMask;
   end;
end;

function TCNNFacade.ConvBackward(var Layer: TConvLayer; const Grad: D3array): D3array;
var
   f, c, h, w, kh, kw: Integer;
   GradWithReLU: D3array;
   GradSum, WGrad: Double;
   InH, InW: Integer;
begin
   SetLength(GradWithReLU, Length(Grad), Length(Grad[0]), Length(Grad[0][0]));
   for f := 0 to High(Grad) do
      for h := 0 to High(Grad[f]) do
         for w := 0 to High(Grad[f][h]) do
            GradWithReLU[f][h][w] := Grad[f][h][w] * ReLUDerivative(Layer.PreActivation[f][h][w]);
   
   for f := 0 to High(Layer.Filters) do
   begin
      GradSum := 0;
      for h := 0 to High(GradWithReLU[f]) do
         for w := 0 to High(GradWithReLU[f][h]) do
            GradSum := GradSum + GradWithReLU[f][h][w];
      Layer.Filters[f].BiasGrad := GradSum;
      
      for c := 0 to Layer.InputChannels - 1 do
         for kh := 0 to Layer.KernelSize - 1 do
            for kw := 0 to Layer.KernelSize - 1 do
            begin
               WGrad := 0;
               for h := 0 to High(GradWithReLU[f]) do
                  for w := 0 to High(GradWithReLU[f][h]) do
                  begin
                     InH := h * Layer.Stride + kh;
                     InW := w * Layer.Stride + kw;
                     WGrad := WGrad + GradWithReLU[f][h][w] * Layer.PaddedInput[c][InH][InW];
                  end;
               Layer.Filters[f].WeightGrads[c][kh][kw] := WGrad;
            end;
   end;
   
   SetLength(Result, Layer.InputChannels, Length(Layer.InputCache[0]), Length(Layer.InputCache[0][0]));
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;
   
   for f := 0 to High(Layer.Filters) do
      for h := 0 to High(GradWithReLU[f]) do
         for w := 0 to High(GradWithReLU[f][h]) do
            for c := 0 to Layer.InputChannels - 1 do
               for kh := 0 to Layer.KernelSize - 1 do
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

function TCNNFacade.PoolBackward(var Layer: TPoolingLayer; const Grad: D3array): D3array;
var
   c, h, w, SrcH, SrcW: Integer;
begin
   SetLength(Result, Length(Layer.InputCache), Length(Layer.InputCache[0]), Length(Layer.InputCache[0][0]));
   for c := 0 to High(Result) do
      for h := 0 to High(Result[c]) do
         for w := 0 to High(Result[c][h]) do
            Result[c][h][w] := 0;
   
   for c := 0 to High(Grad) do
      for h := 0 to High(Grad[c]) do
         for w := 0 to High(Grad[c][h]) do
         begin
            SrcH := h * Layer.PoolSize + Layer.MaxIndices[c][h][w].Y;
            SrcW := w * Layer.PoolSize + Layer.MaxIndices[c][h][w].X;
            Result[c][SrcH][SrcW] := Grad[c][h][w];
         end;
end;

function TCNNFacade.FCBackward(var Layer: TFullyConnectedLayer; const Grad: Darray; IsOutputLayer: Boolean): Darray;
var
   i, j: Integer;
   Delta: Double;
begin
   SetLength(Result, Length(Layer.InputCache));
   for i := 0 to High(Result) do Result[i] := 0;
   
   for i := 0 to High(Layer.Neurons) do
   begin
      if IsOutputLayer then
         Delta := Grad[i]
      else
         Delta := Grad[i] * ReLUDerivative(Layer.Neurons[i].PreActivation) * Layer.Neurons[i].DropoutMask;
      Layer.Neurons[i].Error := Delta;
      for j := 0 to High(Layer.Neurons[i].Weights) do
         Result[j] := Result[j] + Delta * Layer.Neurons[i].Weights[j];
   end;
end;

function TCNNFacade.UnflattenGradient(const Grad: Darray): D3array;
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

procedure TCNNFacade.UpdateWeights;
var
   i, j, c, kh, kw: Integer;
   Grad, MHat, VHat, Update, T: Double;
begin
   Inc(AdamT);
   T := AdamT;
   
   for i := 0 to High(ConvLayers) do
      for j := 0 to High(ConvLayers[i].Filters) do
      begin
         Grad := ClipGrad(ConvLayers[i].Filters[j].BiasGrad);
         ConvLayers[i].Filters[j].BiasM := Beta1 * ConvLayers[i].Filters[j].BiasM + (1 - Beta1) * Grad;
         ConvLayers[i].Filters[j].BiasV := Beta2 * ConvLayers[i].Filters[j].BiasV + (1 - Beta2) * Grad * Grad;
         MHat := ConvLayers[i].Filters[j].BiasM / (1 - Power(Beta1, T));
         VHat := ConvLayers[i].Filters[j].BiasV / (1 - Power(Beta2, T));
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);
         if IsFiniteNum(Update) then
            ConvLayers[i].Filters[j].Bias := ConvLayers[i].Filters[j].Bias - Update;
         
         for c := 0 to High(ConvLayers[i].Filters[j].Weights) do
            for kh := 0 to High(ConvLayers[i].Filters[j].Weights[c]) do
               for kw := 0 to High(ConvLayers[i].Filters[j].Weights[c][kh]) do
               begin
                  Grad := ClipGrad(ConvLayers[i].Filters[j].WeightGrads[c][kh][kw]);
                  ConvLayers[i].Filters[j].WeightsM[c][kh][kw] := Beta1 * ConvLayers[i].Filters[j].WeightsM[c][kh][kw] + (1 - Beta1) * Grad;
                  ConvLayers[i].Filters[j].WeightsV[c][kh][kw] := Beta2 * ConvLayers[i].Filters[j].WeightsV[c][kh][kw] + (1 - Beta2) * Grad * Grad;
                  MHat := ConvLayers[i].Filters[j].WeightsM[c][kh][kw] / (1 - Power(Beta1, T));
                  VHat := ConvLayers[i].Filters[j].WeightsV[c][kh][kw] / (1 - Power(Beta2, T));
                  Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);
                  if IsFiniteNum(Update) then
                     ConvLayers[i].Filters[j].Weights[c][kh][kw] := ConvLayers[i].Filters[j].Weights[c][kh][kw] - Update;
               end;
      end;
   
   for i := 0 to High(FullyConnectedLayers) do
      for j := 0 to High(FullyConnectedLayers[i].Neurons) do
      begin
         for c := 0 to High(FullyConnectedLayers[i].Neurons[j].Weights) do
         begin
            Grad := ClipGrad(FullyConnectedLayers[i].Neurons[j].Error * FullyConnectedLayers[i].InputCache[c]);
            FullyConnectedLayers[i].Neurons[j].WeightsM[c] := Beta1 * FullyConnectedLayers[i].Neurons[j].WeightsM[c] + (1 - Beta1) * Grad;
            FullyConnectedLayers[i].Neurons[j].WeightsV[c] := Beta2 * FullyConnectedLayers[i].Neurons[j].WeightsV[c] + (1 - Beta2) * Grad * Grad;
            MHat := FullyConnectedLayers[i].Neurons[j].WeightsM[c] / (1 - Power(Beta1, T));
            VHat := FullyConnectedLayers[i].Neurons[j].WeightsV[c] / (1 - Power(Beta2, T));
            Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);
            if IsFiniteNum(Update) then
               FullyConnectedLayers[i].Neurons[j].Weights[c] := FullyConnectedLayers[i].Neurons[j].Weights[c] - Update;
         end;
         Grad := ClipGrad(FullyConnectedLayers[i].Neurons[j].Error);
         FullyConnectedLayers[i].Neurons[j].BiasM := Beta1 * FullyConnectedLayers[i].Neurons[j].BiasM + (1 - Beta1) * Grad;
         FullyConnectedLayers[i].Neurons[j].BiasV := Beta2 * FullyConnectedLayers[i].Neurons[j].BiasV + (1 - Beta2) * Grad * Grad;
         MHat := FullyConnectedLayers[i].Neurons[j].BiasM / (1 - Power(Beta1, T));
         VHat := FullyConnectedLayers[i].Neurons[j].BiasV / (1 - Power(Beta2, T));
         Update := LearningRate * MHat / (Sqrt(VHat) + EPSILON);
         if IsFiniteNum(Update) then
            FullyConnectedLayers[i].Neurons[j].Bias := FullyConnectedLayers[i].Neurons[j].Bias - Update;
      end;
   
   for j := 0 to High(OutputLayer.Neurons) do
   begin
      for c := 0 to High(OutputLayer.Neurons[j].Weights) do
      begin
         Grad := ClipGrad(OutputLayer.Neurons[j].Error * OutputLayer.InputCache[c]);
         OutputLayer.Neurons[j].WeightsM[c] := Beta1 * OutputLayer.Neurons[j].WeightsM[c] + (1 - Beta1) * Grad;
         OutputLayer.Neurons[j].WeightsV[c] := Beta2 * OutputLayer.Neurons[j].WeightsV[c] + (1 - Beta2) * Grad * Grad;
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

procedure TCNNFacade.ApplyDropout(var Layer: TFullyConnectedLayer);
var
   i: Integer;
begin
   for i := 0 to High(Layer.Neurons) do
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

{ Stage 1: Feature Map Access }

function TCNNFacade.GetFeatureMap(LayerIdx, FilterIdx: Integer): D2array;
var h, w: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].OutputMaps)) then Exit;
   SetLength(Result, Length(ConvLayers[LayerIdx].OutputMaps[FilterIdx]),
             Length(ConvLayers[LayerIdx].OutputMaps[FilterIdx][0]));
   for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx]) do
      for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx][h]) do
         Result[h][w] := ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w];
end;

procedure TCNNFacade.SetFeatureMap(LayerIdx, FilterIdx: Integer; const Map: D2array);
var h, w: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].OutputMaps)) then Exit;
   for h := 0 to High(Map) do
      for w := 0 to High(Map[h]) do
         if (h <= High(ConvLayers[LayerIdx].OutputMaps[FilterIdx])) and
            (w <= High(ConvLayers[LayerIdx].OutputMaps[FilterIdx][h])) then
            ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w] := Map[h][w];
end;

function TCNNFacade.GetPreActivation(LayerIdx, FilterIdx: Integer): D2array;
var h, w: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].PreActivation)) then Exit;
   SetLength(Result, Length(ConvLayers[LayerIdx].PreActivation[FilterIdx]),
             Length(ConvLayers[LayerIdx].PreActivation[FilterIdx][0]));
   for h := 0 to High(ConvLayers[LayerIdx].PreActivation[FilterIdx]) do
      for w := 0 to High(ConvLayers[LayerIdx].PreActivation[FilterIdx][h]) do
         Result[h][w] := ConvLayers[LayerIdx].PreActivation[FilterIdx][h][w];
end;

procedure TCNNFacade.SetPreActivation(LayerIdx, FilterIdx: Integer; const Map: D2array);
var h, w: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].PreActivation)) then Exit;
   for h := 0 to High(Map) do
      for w := 0 to High(Map[h]) do
         if (h <= High(ConvLayers[LayerIdx].PreActivation[FilterIdx])) and
            (w <= High(ConvLayers[LayerIdx].PreActivation[FilterIdx][h])) then
            ConvLayers[LayerIdx].PreActivation[FilterIdx][h][w] := Map[h][w];
end;

{ Remaining stubs for compilation }

function TCNNFacade.GetKernel(LayerIdx, FilterIdx, ChannelIdx: Integer): D2array;
var
   kh, kw: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   if (ChannelIdx < 0) or (ChannelIdx > High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights)) then Exit;
   
   SetLength(Result, Length(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx]),
             Length(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx][0]));
   for kh := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx]) do
      for kw := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx][kh]) do
         Result[kh][kw] := ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx][kh][kw];
end;

procedure TCNNFacade.SetKernel(LayerIdx, FilterIdx, ChannelIdx: Integer; const KernelArray: D2array);
var
   kh, kw: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   if (ChannelIdx < 0) or (ChannelIdx > High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights)) then Exit;
   
   for kh := 0 to High(KernelArray) do
      for kw := 0 to High(KernelArray[kh]) do
         if (kh <= High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx])) and
            (kw <= High(ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx][kh])) then
            ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx][kh][kw] := KernelArray[kh][kw];
end;

function TCNNFacade.GetBias(LayerIdx, FilterIdx: Integer): Double;
begin
   Result := 0;
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   Result := ConvLayers[LayerIdx].Filters[FilterIdx].Bias;
end;

procedure TCNNFacade.SetBias(LayerIdx, FilterIdx: Integer; Value: Double);
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   ConvLayers[LayerIdx].Filters[FilterIdx].Bias := Value;
end;

function TCNNFacade.GetBatchActivations(LayerIdx: Integer): D4array;
begin
   if (LayerIdx >= 0) and (LayerIdx <= High(FBatchActivations)) then
      Result := FBatchActivations[LayerIdx]
   else
      SetLength(Result, 0);
end;

procedure TCNNFacade.SetBatchActivations(LayerIdx: Integer; const BatchTensor: D4array);
begin
   if LayerIdx >= Length(FBatchActivations) then
      SetLength(FBatchActivations, LayerIdx + 1);
   if (LayerIdx >= 0) then
      FBatchActivations[LayerIdx] := BatchTensor;
end;

function TCNNFacade.GetPoolingIndices(LayerIdx, FilterIdx: Integer): D2array;
var
   h, w: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(PoolLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(PoolLayers[LayerIdx].MaxIndices)) then Exit;
   
   SetLength(Result, Length(PoolLayers[LayerIdx].MaxIndices[FilterIdx]),
             Length(PoolLayers[LayerIdx].MaxIndices[FilterIdx][0]));
   for h := 0 to High(PoolLayers[LayerIdx].MaxIndices[FilterIdx]) do
      for w := 0 to High(PoolLayers[LayerIdx].MaxIndices[FilterIdx][h]) do
         Result[h][w] := PoolLayers[LayerIdx].MaxIndices[FilterIdx][h][w].Y * 1000 + 
                         PoolLayers[LayerIdx].MaxIndices[FilterIdx][h][w].X;
end;

function TCNNFacade.GetDropoutMask(LayerIdx: Integer): Darray;
var
   i: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(FullyConnectedLayers)) then Exit;
   
   SetLength(Result, Length(FullyConnectedLayers[LayerIdx].Neurons));
   for i := 0 to High(FullyConnectedLayers[LayerIdx].Neurons) do
      Result[i] := FullyConnectedLayers[LayerIdx].Neurons[i].DropoutMask;
end;

procedure TCNNFacade.SetDropoutMask(LayerIdx: Integer; const Mask: Darray);
var
   i: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(FullyConnectedLayers)) then Exit;
   
   for i := 0 to High(Mask) do
      if i <= High(FullyConnectedLayers[LayerIdx].Neurons) then
         FullyConnectedLayers[LayerIdx].Neurons[i].DropoutMask := Mask[i];
end;

function TCNNFacade.GetFilterGradient(LayerIdx, FilterIdx, ChannelIdx: Integer): D2array;
var
   kh, kw: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   if (ChannelIdx < 0) or (ChannelIdx > High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads)) then Exit;
   
   SetLength(Result, Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx]),
             Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx][0]));
   for kh := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx]) do
      for kw := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx][kh]) do
         Result[kh][kw] := ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx][kh][kw];
end;

function TCNNFacade.GetBiasGradient(LayerIdx, FilterIdx: Integer): Double;
begin
   Result := 0;
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   Result := ConvLayers[LayerIdx].Filters[FilterIdx].BiasGrad;
end;

function TCNNFacade.GetActivationGradient(LayerIdx, FilterIdx, Y, X: Integer): Double;
begin
   Result := 0;
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].PreActivation)) then Exit;
   if (Y < 0) or (Y > High(ConvLayers[LayerIdx].PreActivation[FilterIdx])) then Exit;
   if (X < 0) or (X > High(ConvLayers[LayerIdx].PreActivation[FilterIdx][Y])) then Exit;
   Result := ReLUDerivative(ConvLayers[LayerIdx].PreActivation[FilterIdx][Y][X]);
end;

function TCNNFacade.GetOptimizerState(LayerIdx, FilterIdx: Integer; const Param: string): D3array;
var
   c, kh, kw: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   
   if (Param = 'M') or (Param = 'm') then
   begin
      SetLength(Result, Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM),
                Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM[0]),
                Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM[0][0]));
      for c := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM) do
         for kh := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM[c]) do
            for kw := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM[c][kh]) do
               Result[c][kh][kw] := ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM[c][kh][kw];
   end
   else if (Param = 'V') or (Param = 'v') then
   begin
      SetLength(Result, Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV),
                Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV[0]),
                Length(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV[0][0]));
      for c := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV) do
         for kh := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV[c]) do
            for kw := 0 to High(ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV[c][kh]) do
               Result[c][kh][kw] := ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV[c][kh][kw];
   end;
end;

function TCNNFacade.GetFlattenedFeatures: Darray;
begin Result := FFlattenedFeatures; end;

procedure TCNNFacade.SetFlattenedFeatures(const Vector: Darray);
begin FFlattenedFeatures := Vector; end;

function TCNNFacade.GetLogits: Darray;
var
   i: Integer;
begin
   SetLength(Result, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
      Result[i] := OutputLayer.Neurons[i].PreActivation;
end;

function TCNNFacade.GetSoftmax: Darray;
var
   Logits: Darray;
   i: Integer;
begin
   SetLength(Logits, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
      Logits[i] := OutputLayer.Neurons[i].PreActivation;
   Result := Softmax(Logits);
end;

function TCNNFacade.GetLayerConfig(LayerIdx: Integer): TLayerConfig;
var
   FCIdx: Integer;
begin
   FillChar(Result, SizeOf(Result), 0);
   
   if (LayerIdx >= 0) and (LayerIdx <= High(ConvLayers)) then
   begin
      Result.LayerType := 'conv';
      Result.FilterCount := Length(ConvLayers[LayerIdx].Filters);
      Result.KernelSize := ConvLayers[LayerIdx].KernelSize;
      Result.Stride := ConvLayers[LayerIdx].Stride;
      Result.Padding := ConvLayers[LayerIdx].Padding;
      Result.InputChannels := ConvLayers[LayerIdx].InputChannels;
      if Length(ConvLayers[LayerIdx].OutputMaps) > 0 then
      begin
         Result.OutputHeight := Length(ConvLayers[LayerIdx].OutputMaps[0]);
         if Length(ConvLayers[LayerIdx].OutputMaps[0]) > 0 then
            Result.OutputWidth := Length(ConvLayers[LayerIdx].OutputMaps[0][0]);
      end;
   end
   else
   begin
      FCIdx := LayerIdx - Length(ConvLayers);
      if (FCIdx >= 0) and (FCIdx <= High(FullyConnectedLayers)) then
      begin
         Result.LayerType := 'fc';
         Result.NeuronCount := Length(FullyConnectedLayers[FCIdx].Neurons);
         if Length(FullyConnectedLayers[FCIdx].Neurons) > 0 then
            Result.InputSize := Length(FullyConnectedLayers[FCIdx].Neurons[0].Weights);
      end
      else if FCIdx = Length(FullyConnectedLayers) then
      begin
         Result.LayerType := 'output';
         Result.NeuronCount := Length(OutputLayer.Neurons);
         if Length(OutputLayer.Neurons) > 0 then
            Result.InputSize := Length(OutputLayer.Neurons[0].Weights);
      end;
   end;
end;

function TCNNFacade.GetNumLayers: Integer;
begin Result := Length(ConvLayers) + Length(FullyConnectedLayers) + 1; end;

function TCNNFacade.GetNumConvLayers: Integer;
begin Result := Length(ConvLayers); end;

function TCNNFacade.GetNumFCLayers: Integer;
begin Result := Length(FullyConnectedLayers) + 1; end;

function TCNNFacade.GetNumFilters(LayerIdx: Integer): Integer;
begin
   if (LayerIdx >= 0) and (LayerIdx <= High(ConvLayers)) then
      Result := Length(ConvLayers[LayerIdx].Filters)
   else Result := 0;
end;

function TCNNFacade.GetSaliencyMap(LayerIdx, FilterIdx, InputIdx: Integer): D2array;
var
   h, w: Integer;
   MaxVal, MinVal, Range: Double;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].OutputMaps)) then Exit;
   
   SetLength(Result, Length(ConvLayers[LayerIdx].OutputMaps[FilterIdx]),
             Length(ConvLayers[LayerIdx].OutputMaps[FilterIdx][0]));
   
   MaxVal := -1e308;
   MinVal := 1e308;
   for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx]) do
      for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx][h]) do
      begin
         if ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w] > MaxVal then
            MaxVal := ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w];
         if ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w] < MinVal then
            MinVal := ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w];
      end;
   
   Range := MaxVal - MinVal;
   if Range < EPSILON then Range := 1;
   
   for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx]) do
      for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[FilterIdx][h]) do
         Result[h][w] := (ConvLayers[LayerIdx].OutputMaps[FilterIdx][h][w] - MinVal) / Range;
end;

function TCNNFacade.GetDeconv(LayerIdx, FilterIdx: Integer; UpToInput: Boolean): D3array;
var
   i, c, h, w, kh, kw: Integer;
   Grad: D3array;
   InH, InW: Integer;
begin
   SetLength(Result, 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].OutputMaps)) then Exit;
   
   SetLength(Grad, Length(ConvLayers[LayerIdx].OutputMaps),
             Length(ConvLayers[LayerIdx].OutputMaps[0]),
             Length(ConvLayers[LayerIdx].OutputMaps[0][0]));
   
   for c := 0 to High(Grad) do
      for h := 0 to High(Grad[c]) do
         for w := 0 to High(Grad[c][h]) do
            if c = FilterIdx then
               Grad[c][h][w] := ConvLayers[LayerIdx].OutputMaps[c][h][w]
            else
               Grad[c][h][w] := 0;
   
   if UpToInput then
   begin
      for i := LayerIdx downto 0 do
      begin
         if i <= High(PoolLayers) then
         begin
            if Length(PoolLayers[i].InputCache) > 0 then
            begin
               SetLength(Result, Length(PoolLayers[i].InputCache),
                         Length(PoolLayers[i].InputCache[0]),
                         Length(PoolLayers[i].InputCache[0][0]));
               for c := 0 to High(Result) do
                  for h := 0 to High(Result[c]) do
                     for w := 0 to High(Result[c][h]) do
                        Result[c][h][w] := 0;
               
               for c := 0 to High(Grad) do
                  for h := 0 to High(Grad[c]) do
                     for w := 0 to High(Grad[c][h]) do
                        if (c < Length(PoolLayers[i].MaxIndices)) and
                           (h < Length(PoolLayers[i].MaxIndices[c])) and
                           (w < Length(PoolLayers[i].MaxIndices[c][h])) then
                        begin
                           InH := h * PoolLayers[i].PoolSize + PoolLayers[i].MaxIndices[c][h][w].Y;
                           InW := w * PoolLayers[i].PoolSize + PoolLayers[i].MaxIndices[c][h][w].X;
                           if (InH <= High(Result[c])) and (InW <= High(Result[c][InH])) then
                              Result[c][InH][InW] := Grad[c][h][w];
                        end;
               Grad := Result;
            end;
         end;
      end;
   end;
   
   Result := Grad;
end;

procedure TCNNFacade.AddFilter(LayerIdx: Integer; const Params: D3array);
var
   NewIdx, c, kh, kw: Integer;
   Scale: Double;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   
   NewIdx := Length(ConvLayers[LayerIdx].Filters);
   SetLength(ConvLayers[LayerIdx].Filters, NewIdx + 1);
   
   if Length(Params) > 0 then
   begin
      ConvLayers[LayerIdx].Filters[NewIdx].Weights := Params;
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightsM, Length(Params), Length(Params[0]), Length(Params[0][0]));
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightsV, Length(Params), Length(Params[0]), Length(Params[0][0]));
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightGrads, Length(Params), Length(Params[0]), Length(Params[0][0]));
   end
   else
   begin
      Scale := Sqrt(2.0 / (ConvLayers[LayerIdx].InputChannels * ConvLayers[LayerIdx].KernelSize * ConvLayers[LayerIdx].KernelSize));
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].Weights, ConvLayers[LayerIdx].InputChannels,
                ConvLayers[LayerIdx].KernelSize, ConvLayers[LayerIdx].KernelSize);
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightsM, ConvLayers[LayerIdx].InputChannels,
                ConvLayers[LayerIdx].KernelSize, ConvLayers[LayerIdx].KernelSize);
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightsV, ConvLayers[LayerIdx].InputChannels,
                ConvLayers[LayerIdx].KernelSize, ConvLayers[LayerIdx].KernelSize);
      SetLength(ConvLayers[LayerIdx].Filters[NewIdx].WeightGrads, ConvLayers[LayerIdx].InputChannels,
                ConvLayers[LayerIdx].KernelSize, ConvLayers[LayerIdx].KernelSize);
      
      for c := 0 to ConvLayers[LayerIdx].InputChannels - 1 do
         for kh := 0 to ConvLayers[LayerIdx].KernelSize - 1 do
            for kw := 0 to ConvLayers[LayerIdx].KernelSize - 1 do
               ConvLayers[LayerIdx].Filters[NewIdx].Weights[c][kh][kw] := (Random - 0.5) * Scale;
   end;
   
   ConvLayers[LayerIdx].Filters[NewIdx].Bias := 0;
   ConvLayers[LayerIdx].Filters[NewIdx].BiasM := 0;
   ConvLayers[LayerIdx].Filters[NewIdx].BiasV := 0;
   ConvLayers[LayerIdx].Filters[NewIdx].BiasGrad := 0;
end;

procedure TCNNFacade.RemoveFilter(LayerIdx, FilterIdx: Integer);
var
   i: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(ConvLayers[LayerIdx].Filters)) then Exit;
   if Length(ConvLayers[LayerIdx].Filters) <= 1 then Exit;
   
   for i := FilterIdx to High(ConvLayers[LayerIdx].Filters) - 1 do
      ConvLayers[LayerIdx].Filters[i] := ConvLayers[LayerIdx].Filters[i + 1];
   SetLength(ConvLayers[LayerIdx].Filters, Length(ConvLayers[LayerIdx].Filters) - 1);
end;

procedure TCNNFacade.AddConvLayer(Position: Integer; NumFilters, KernelSize, Stride, Padding: Integer);
var
   i: Integer;
   InputChannels: Integer;
begin
   if Position < 0 then Position := 0;
   if Position > Length(ConvLayers) then Position := Length(ConvLayers);
   
   SetLength(ConvLayers, Length(ConvLayers) + 1);
   
   for i := High(ConvLayers) downto Position + 1 do
      ConvLayers[i] := ConvLayers[i - 1];
   
   if Position > 0 then
      InputChannels := Length(ConvLayers[Position - 1].Filters)
   else
      InputChannels := 1;
   
   InitializeConvLayer(ConvLayers[Position], NumFilters, InputChannels, KernelSize, Stride, Padding);
   
   SetLength(PoolLayers, Length(PoolLayers) + 1);
   InitializePoolLayer(PoolLayers[High(PoolLayers)], 2, 2);
end;

procedure TCNNFacade.RemoveLayer(LayerIdx: Integer);
var
   i: Integer;
begin
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if Length(ConvLayers) <= 1 then Exit;
   
   for i := LayerIdx to High(ConvLayers) - 1 do
      ConvLayers[i] := ConvLayers[i + 1];
   SetLength(ConvLayers, Length(ConvLayers) - 1);
   
   if LayerIdx <= High(PoolLayers) then
   begin
      for i := LayerIdx to High(PoolLayers) - 1 do
         PoolLayers[i] := PoolLayers[i + 1];
      SetLength(PoolLayers, Length(PoolLayers) - 1);
   end;
end;

function TCNNFacade.GetLayerStats(LayerIdx: Integer): TLayerStats;
var
   f, h, w: Integer;
   Sum, SumSq, Val: Double;
begin
   FillChar(Result, SizeOf(Result), 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if Length(ConvLayers[LayerIdx].OutputMaps) = 0 then Exit;
   
   Result.Min := 1e308;
   Result.Max := -1e308;
   Sum := 0;
   SumSq := 0;
   Result.Count := 0;
   
   for f := 0 to High(ConvLayers[LayerIdx].OutputMaps) do
      for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[f]) do
         for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[f][h]) do
         begin
            Val := ConvLayers[LayerIdx].OutputMaps[f][h][w];
            if Val < Result.Min then Result.Min := Val;
            if Val > Result.Max then Result.Max := Val;
            Sum := Sum + Val;
            SumSq := SumSq + Val * Val;
            Inc(Result.Count);
         end;
   
   if Result.Count > 0 then
   begin
      Result.Mean := Sum / Result.Count;
      Result.StdDev := Sqrt((SumSq / Result.Count) - (Result.Mean * Result.Mean));
   end;
end;

function TCNNFacade.GetActivationHistogram(LayerIdx: Integer; NumBins: Integer): Darray;
var
   f, h, w, BinIdx: Integer;
   MinVal, MaxVal, Range, Val: Double;
begin
   SetLength(Result, NumBins);
   for BinIdx := 0 to NumBins - 1 do Result[BinIdx] := 0;
   
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if Length(ConvLayers[LayerIdx].OutputMaps) = 0 then Exit;
   
   MinVal := 1e308;
   MaxVal := -1e308;
   for f := 0 to High(ConvLayers[LayerIdx].OutputMaps) do
      for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[f]) do
         for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[f][h]) do
         begin
            Val := ConvLayers[LayerIdx].OutputMaps[f][h][w];
            if Val < MinVal then MinVal := Val;
            if Val > MaxVal then MaxVal := Val;
         end;
   
   Range := MaxVal - MinVal;
   if Range < EPSILON then Range := 1;
   
   for f := 0 to High(ConvLayers[LayerIdx].OutputMaps) do
      for h := 0 to High(ConvLayers[LayerIdx].OutputMaps[f]) do
         for w := 0 to High(ConvLayers[LayerIdx].OutputMaps[f][h]) do
         begin
            Val := ConvLayers[LayerIdx].OutputMaps[f][h][w];
            BinIdx := Trunc((Val - MinVal) / Range * (NumBins - 1));
            if BinIdx >= NumBins then BinIdx := NumBins - 1;
            if BinIdx < 0 then BinIdx := 0;
            Result[BinIdx] := Result[BinIdx] + 1;
         end;
end;

function TCNNFacade.GetWeightHistogram(LayerIdx: Integer; NumBins: Integer): Darray;
var
   f, c, kh, kw, BinIdx: Integer;
   MinVal, MaxVal, Range, Val: Double;
begin
   SetLength(Result, NumBins);
   for BinIdx := 0 to NumBins - 1 do Result[BinIdx] := 0;
   
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   if Length(ConvLayers[LayerIdx].Filters) = 0 then Exit;
   
   MinVal := 1e308;
   MaxVal := -1e308;
   for f := 0 to High(ConvLayers[LayerIdx].Filters) do
      for c := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights) do
         for kh := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights[c]) do
            for kw := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights[c][kh]) do
            begin
               Val := ConvLayers[LayerIdx].Filters[f].Weights[c][kh][kw];
               if Val < MinVal then MinVal := Val;
               if Val > MaxVal then MaxVal := Val;
            end;
   
   Range := MaxVal - MinVal;
   if Range < EPSILON then Range := 1;
   
   for f := 0 to High(ConvLayers[LayerIdx].Filters) do
      for c := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights) do
         for kh := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights[c]) do
            for kw := 0 to High(ConvLayers[LayerIdx].Filters[f].Weights[c][kh]) do
            begin
               Val := ConvLayers[LayerIdx].Filters[f].Weights[c][kh][kw];
               BinIdx := Trunc((Val - MinVal) / Range * (NumBins - 1));
               if BinIdx >= NumBins then BinIdx := NumBins - 1;
               if BinIdx < 0 then BinIdx := 0;
               Result[BinIdx] := Result[BinIdx] + 1;
            end;
end;

function TCNNFacade.GetReceptiveField(LayerIdx, FeatureIdx, Y, X: Integer): TReceptiveField;
var
   i, c: Integer;
   CurrentStartX, CurrentEndX, CurrentStartY, CurrentEndY: Integer;
   KernelSize, Stride, Padding, PoolSize: Integer;
begin
   FillChar(Result, SizeOf(Result), 0);
   if (LayerIdx < 0) or (LayerIdx > High(ConvLayers)) then Exit;
   
   CurrentStartX := X;
   CurrentEndX := X;
   CurrentStartY := Y;
   CurrentEndY := Y;
   
   for i := LayerIdx downto 0 do
   begin
      if i <= High(PoolLayers) then
      begin
         PoolSize := PoolLayers[i].PoolSize;
         CurrentStartX := CurrentStartX * PoolSize;
         CurrentEndX := CurrentEndX * PoolSize + PoolSize - 1;
         CurrentStartY := CurrentStartY * PoolSize;
         CurrentEndY := CurrentEndY * PoolSize + PoolSize - 1;
      end;
      
      KernelSize := ConvLayers[i].KernelSize;
      Stride := ConvLayers[i].Stride;
      Padding := ConvLayers[i].Padding;
      
      CurrentStartX := CurrentStartX * Stride - Padding;
      CurrentEndX := CurrentEndX * Stride + KernelSize - 1 - Padding;
      CurrentStartY := CurrentStartY * Stride - Padding;
      CurrentEndY := CurrentEndY * Stride + KernelSize - 1 - Padding;
   end;
   
   Result.StartX := CurrentStartX;
   Result.EndX := CurrentEndX;
   Result.StartY := CurrentStartY;
   Result.EndY := CurrentEndY;
   
   SetLength(Result.Channels, ConvLayers[0].InputChannels);
   for c := 0 to ConvLayers[0].InputChannels - 1 do
      Result.Channels[c] := c;
end;

function TCNNFacade.GetBatchNormParams(LayerIdx: Integer): TBatchNormParams;
begin
   FillChar(Result, SizeOf(Result), 0);
   if (LayerIdx >= 0) and (LayerIdx <= High(FBatchNormParams)) then
      Result := FBatchNormParams[LayerIdx];
end;

procedure TCNNFacade.SetBatchNormParams(LayerIdx: Integer; const Params: TBatchNormParams);
begin
   if LayerIdx >= Length(FBatchNormParams) then
      SetLength(FBatchNormParams, LayerIdx + 1);
   if LayerIdx >= 0 then
      FBatchNormParams[LayerIdx] := Params;
end;

procedure TCNNFacade.SetFilterAttribute(LayerIdx, FilterIdx: Integer; const Key, Value: string);
var
   i, NewIdx: Integer;
   Found: Boolean;
begin
   if (LayerIdx < 0) then Exit;
   
   if LayerIdx >= Length(FFilterAttributes) then
      SetLength(FFilterAttributes, LayerIdx + 1);
   if FilterIdx >= Length(FFilterAttributes[LayerIdx]) then
      SetLength(FFilterAttributes[LayerIdx], FilterIdx + 1);
   
   Found := False;
   for i := 0 to High(FFilterAttributes[LayerIdx][FilterIdx]) do
      if FFilterAttributes[LayerIdx][FilterIdx][i].Key = Key then
      begin
         FFilterAttributes[LayerIdx][FilterIdx][i].Value := Value;
         Found := True;
         Break;
      end;
   
   if not Found then
   begin
      NewIdx := Length(FFilterAttributes[LayerIdx][FilterIdx]);
      SetLength(FFilterAttributes[LayerIdx][FilterIdx], NewIdx + 1);
      FFilterAttributes[LayerIdx][FilterIdx][NewIdx].Key := Key;
      FFilterAttributes[LayerIdx][FilterIdx][NewIdx].Value := Value;
   end;
end;

function TCNNFacade.GetFilterAttribute(LayerIdx, FilterIdx: Integer; const Key: string): string;
var
   i: Integer;
begin
   Result := '';
   if (LayerIdx < 0) or (LayerIdx > High(FFilterAttributes)) then Exit;
   if (FilterIdx < 0) or (FilterIdx > High(FFilterAttributes[LayerIdx])) then Exit;
   
   for i := 0 to High(FFilterAttributes[LayerIdx][FilterIdx]) do
      if FFilterAttributes[LayerIdx][FilterIdx][i].Key = Key then
      begin
         Result := FFilterAttributes[LayerIdx][FilterIdx][i].Value;
         Exit;
      end;
end;

procedure TCNNFacade.SetTrainingMode(Training: Boolean);
begin IsTraining := Training; end;

function TCNNFacade.GetTrainingMode: Boolean;
begin Result := IsTraining; end;

{ Helper Functions }

function ActivationToStr(act: string): string;
begin
   Result := LowerCase(act);
end;

function ParseActivation(const s: string): string;
begin
   if (LowerCase(s) = 'relu') or (LowerCase(s) = 'tanh') or (LowerCase(s) = 'sigmoid') then
      Result := LowerCase(s)
   else
      Result := 'relu';
end;

procedure ParseIntArrayHelper(const s: string; out Result: IntArray);
var
   tokens: TStringList;
   i: Integer;
begin
   tokens := TStringList.Create;
   try
      tokens.Delimiter := ',';
      tokens.DelimitedText := s;
      SetLength(Result, tokens.Count);
      for i := 0 to tokens.Count - 1 do
         Result[i] := StrToInt(Trim(tokens[i]));
   finally
      tokens.Free;
   end;
end;

procedure PrintUsage;
begin
   WriteLn('Facaded CNN');
   WriteLn;
   WriteLn('Commands:');
   WriteLn('  create      Create a new CNN model and save to JSON');
   WriteLn('  train       Train on sample data from JSON model');
   WriteLn('  predict     Make predictions with JSON model');
   WriteLn('  introspect  Examine layer internals (activations, weights, gradients)');
   WriteLn('  stats       Display layer statistics and histograms');
   WriteLn('  modify      Add/remove filters or layers dynamically');
   WriteLn('  analyze     Get saliency maps, deconv, receptive fields');
   WriteLn('  info        Display complete model architecture from JSON');
   WriteLn('  help        Show this help message');
   WriteLn;
   WriteLn('Create Options:');
   WriteLn('  --input-w=N            Input width (required)');
   WriteLn('  --input-h=N            Input height (required)');
   WriteLn('  --input-c=N            Input channels (required)');
   WriteLn('  --conv=N,N,...         Conv filters (required)');
   WriteLn('  --kernels=N,N,...      Kernel sizes (required)');
   WriteLn('  --pools=N,N,...        Pool sizes (required)');
   WriteLn('  --fc=N,N,...           FC layer sizes (required)');
   WriteLn('  --output=N             Output layer size (required)');
   WriteLn('  --save=FILE.json       Save model to JSON file (required)');
   WriteLn('  --lr=VALUE             Learning rate (default: 0.001)');
   WriteLn('  --dropout=VALUE        Dropout rate (default: 0.25)');
   WriteLn;
   WriteLn('Train Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (required)');
   WriteLn('  --epochs=N             Number of epochs (required)');
   WriteLn('  --save=FILE.json       Save trained model to JSON (required)');
   WriteLn;
   WriteLn('Predict Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (required)');
   WriteLn('  --random-input         Use random input data');
   WriteLn('  --mode=WHAT            basic | all (default: basic)');
   WriteLn;
   WriteLn('Introspect Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (optional)');
   WriteLn('  --layer=N              Layer index (0-based)');
   WriteLn('  --filter=N             Filter/neuron index');
   WriteLn('  --mode=WHAT            feature_map | pre_activation | kernel | bias');
   WriteLn('  --channel=N            Channel index (for kernels)');
   WriteLn;
   WriteLn('Stats Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (optional)');
   WriteLn('  --layer=N              Layer index');
   WriteLn('  --histogram=N          Number of bins for histogram (default: 50)');
   WriteLn('  --type=WHAT            weights | activations | both');
   WriteLn;
   WriteLn('Modify Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (optional)');
   WriteLn('  --action=ACTION        add_filter | remove_filter | add_layer');
   WriteLn('  --layer=N              Layer index for action');
   WriteLn('  --filter=N             Filter index (for remove_filter)');
   WriteLn('  --filters=N            Number of filters (for add_layer)');
   WriteLn('  --kernel=N             Kernel size (for add_layer)');
   WriteLn('  --stride=N             Stride (for add_layer, default: 1)');
   WriteLn('  --padding=N            Padding (for add_layer, default: 0)');
   WriteLn('  --save=FILE.json       Save modified model to JSON');
   WriteLn;
   WriteLn('Analyze Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (optional)');
   WriteLn('  --layer=N              Layer index');
   WriteLn('  --filter=N             Filter index');
   WriteLn('  --type=WHAT            saliency | deconv | receptive_field');
   WriteLn('  --x=N --y=N            Position for saliency/receptive field');
   WriteLn;
   WriteLn('Info Options:');
   WriteLn('  --model=FILE.json      Load model from JSON file (required)');
   WriteLn;
   WriteLn('Examples:');
   WriteLn('  facaded_cnn create --input-w=28 --input-h=28 --input-c=1 --conv=8,16 --kernels=3,3 --pools=2,2 --fc=64 --output=10 --save=model.json');
   WriteLn('  facaded_cnn train --model=model.json --epochs=50 --save=model_trained.json');
   WriteLn('  facaded_cnn predict --model=model.json --random-input --mode=all');
   WriteLn('  facaded_cnn introspect --model=model.json --layer=0 --filter=0 --mode=feature_map');
   WriteLn('  facaded_cnn info --model=model.json');
   WriteLn('  facaded_cnn modify --model=model.json --action=add_filter --layer=0 --save=model_modified.json');
end;

{ Main Program }

type
   TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdIntrospect, cmdStats, 
               cmdModify, cmdAnalyze, cmdInfo, cmdHelp);

var
   Command: TCommand;
   CmdStr: string;
   i, inputW, inputH, inputC, outputSize: Integer;
   convFilters, kernelSizes, poolSizes, fcLayerSizes: array of Integer;
   learningRate, dropoutRate: Double;
   modelFile, saveFile: string;
   arg, key, value: string;
   eqPos: Integer;
   CNN: TCNNFacade;
   Image: TImageData;
   Target, Output: Darray;
   c, h, w: Integer;
   Loss: Double;
   
   { Command-specific variables }
   layerIdx, filterIdx, channelIdx, binCount, posX, posY: Integer;
   intrinMode, statsType, modifyAction, analyzeType: string;
   numFiltersToAdd, kernelSizeAdd, strideAdd, paddingAdd: Integer;
   histogram: Darray;
   stats: TLayerStats;
   config: TLayerConfig;
   rf: TReceptiveField;
   j, k, binIdx: Integer;
   featureMap: D2array;
   preAct: D2array;
   kernel: D2array;
   saliency: D2array;
   filterParams: D3array;
   randomInput: Boolean;
   predMode: string;

begin
   Randomize;

   if ParamCount < 1 then
   begin
      PrintUsage;
      Exit;
   end;

   CmdStr := ParamStr(1);
   Command := cmdNone;

   if CmdStr = 'create' then Command := cmdCreate
   else if CmdStr = 'train' then Command := cmdTrain
   else if CmdStr = 'predict' then Command := cmdPredict
   else if CmdStr = 'introspect' then Command := cmdIntrospect
   else if CmdStr = 'stats' then Command := cmdStats
   else if CmdStr = 'modify' then Command := cmdModify
   else if CmdStr = 'analyze' then Command := cmdAnalyze
   else if CmdStr = 'info' then Command := cmdInfo
   else if (CmdStr = 'help') or (CmdStr = '--help') or (CmdStr = '-h') then Command := cmdHelp
   else
   begin
      WriteLn('Unknown command: ', CmdStr);
      PrintUsage;
      Exit;
   end;

   if Command = cmdHelp then
   begin
      PrintUsage;
      Exit;
   end;

   { Initialize defaults }
   inputW := 0;
   inputH := 0;
   inputC := 0;
   outputSize := 0;
   SetLength(convFilters, 0);
   SetLength(kernelSizes, 0);
   SetLength(poolSizes, 0);
   SetLength(fcLayerSizes, 0);
   learningRate := 0.001;
   dropoutRate := 0.25;
   modelFile := '';
   saveFile := '';
   
   { Command-specific defaults }
   layerIdx := 0;
   filterIdx := 0;
   channelIdx := 0;
   binCount := 50;
   posX := 0;
   posY := 0;
   intrinMode := 'feature_map';
   statsType := 'weights';
   modifyAction := 'add_filter';
   analyzeType := 'saliency';
   numFiltersToAdd := 1;
   kernelSizeAdd := 3;
   strideAdd := 1;
   paddingAdd := 0;
   randomInput := False;
   predMode := '';

   { Parse arguments }
   for i := 2 to ParamCount do
   begin
      arg := ParamStr(i);
      eqPos := Pos('=', arg);
      
      { Handle flag-style arguments without value }
      if (arg = '--random-input') then
      begin
         randomInput := True;
         Continue;
      end;
      
      if eqPos = 0 then
      begin
         WriteLn('Unknown argument: ', arg);
         Continue;
      end;

      key := Copy(arg, 1, eqPos - 1);
      value := Copy(arg, eqPos + 1, Length(arg));

      if key = '--input-w' then
         inputW := StrToInt(value)
      else if key = '--input-h' then
         inputH := StrToInt(value)
      else if key = '--input-c' then
         inputC := StrToInt(value)
      else if key = '--output' then
         outputSize := StrToInt(value)
      else if key = '--conv' then
         ParseIntArrayHelper(value, convFilters)
      else if key = '--kernels' then
         ParseIntArrayHelper(value, kernelSizes)
      else if key = '--pools' then
         ParseIntArrayHelper(value, poolSizes)
      else if key = '--fc' then
         ParseIntArrayHelper(value, fcLayerSizes)
      else if key = '--save' then
         saveFile := value
      else if key = '--model' then
         modelFile := value
      else if key = '--lr' then
         learningRate := StrToFloat(value)
      else if key = '--dropout' then
         dropoutRate := StrToFloat(value)
      { Introspect/Stats options }
      else if key = '--layer' then
         layerIdx := StrToInt(value)
      else if key = '--filter' then
         filterIdx := StrToInt(value)
      else if key = '--channel' then
         channelIdx := StrToInt(value)
      else if key = '--histogram' then
         binCount := StrToInt(value)
      else if key = '--mode' then
         intrinMode := LowerCase(value)
      else if key = '--type' then
         statsType := LowerCase(value)
      else if key = '--action' then
         modifyAction := LowerCase(value)
      else if key = '--filters' then
         numFiltersToAdd := StrToInt(value)
      else if key = '--kernel' then
         kernelSizeAdd := StrToInt(value)
      else if key = '--stride' then
         strideAdd := StrToInt(value)
      else if key = '--padding' then
         paddingAdd := StrToInt(value)
      { Analyze options }
      else if key = '--x' then
         posX := StrToInt(value)
      else if key = '--y' then
         posY := StrToInt(value)
      { Predict options }
      else if key = '--random-input' then
         randomInput := True
      else if (key = '--mode') and (Command = cmdPredict) then
         predMode := LowerCase(value)
      else
         WriteLn('Unknown option: ', key);
   end;

   { Execute command }
   if Command = cmdCreate then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;
      if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;

      WriteLn('Creating CNN Facade model...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);

      WriteLn('Created CNN Facade model:');
      WriteLn('  Input: ', inputW, 'x', inputH, 'x', inputC);
      Write('  Conv filters: ');
      for i := 0 to High(convFilters) do
      begin
         if i > 0 then Write(',');
         Write(convFilters[i]);
      end;
      WriteLn;
      Write('  Kernel sizes: ');
      for i := 0 to High(kernelSizes) do
      begin
         if i > 0 then Write(',');
         Write(kernelSizes[i]);
      end;
      WriteLn;
      Write('  Pool sizes: ');
      for i := 0 to High(poolSizes) do
      begin
         if i > 0 then Write(',');
         Write(poolSizes[i]);
      end;
      WriteLn;
      Write('  FC layers: ');
      for i := 0 to High(fcLayerSizes) do
      begin
         if i > 0 then Write(',');
         Write(fcLayerSizes[i]);
      end;
      WriteLn;
      WriteLn('  Output size: ', outputSize);
      WriteLn('  Learning rate: ', learningRate:0:6);
      WriteLn('  Dropout rate: ', dropoutRate:0:6);

      { Save model to JSON }
      CNN.SaveModelToJSON(saveFile);

      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdTrain then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Training CNN Facade model for 20 steps...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      Image.Width := inputW;
      Image.Height := inputH;
      Image.Channels := inputC;
      SetLength(Image.Data, inputC, inputH, inputW);
      SetLength(Target, outputSize);
      
      for i := 1 to 20 do
      begin
         { Generate random input }
         for c := 0 to inputC - 1 do
            for h := 0 to inputH - 1 do
               for w := 0 to inputW - 1 do
                  Image.Data[c][h][w] := Random;
         
         { Generate random target }
         for k := 0 to outputSize - 1 do Target[k] := 0;
         Target[i mod outputSize] := 1.0;
         
         Loss := CNN.TrainStep(Image, Target);
         WriteLn('Step ', i, ' Loss: ', Loss:0:6);
      end;
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdPredict then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Making predictions...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      if randomInput then
      begin
         Image.Width := inputW;
         Image.Height := inputH;
         Image.Channels := inputC;
         SetLength(Image.Data, inputC, inputH, inputW);
         WriteLn('Generating random input (', inputW, 'x', inputH, 'x', inputC, ')');
         for c := 0 to inputC - 1 do
            for h := 0 to inputH - 1 do
               for w := 0 to inputW - 1 do
                  Image.Data[c][h][w] := Random;
      end;
      
      Output := CNN.Predict(Image);
      WriteLn('Prediction output (', Length(Output), ' classes):');
      Write('  ');
      for i := 0 to High(Output) do
         Write(Output[i]:0:6, ' ');
      WriteLn;
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdIntrospect then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Introspecting CNN layer internals...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      { Forward pass first }
      Image.Width := inputW;
      Image.Height := inputH;
      Image.Channels := inputC;
      SetLength(Image.Data, inputC, inputH, inputW);
      for c := 0 to inputC - 1 do
         for h := 0 to inputH - 1 do
            for w := 0 to inputW - 1 do
               Image.Data[c][h][w] := Random;
      Output := CNN.Predict(Image);
      
      WriteLn('Layer ', layerIdx, ' Filter/Neuron ', filterIdx);
      
      if intrinMode = 'feature_map' then
      begin
         WriteLn('Feature Map:');
         featureMap := CNN.GetFeatureMap(layerIdx, filterIdx);
         if Length(featureMap) > 0 then
         begin
            WriteLn('  Size: ', Length(featureMap), 'x', Length(featureMap[0]));
            WriteLn('  First 5x5 (or less):');
            for h := 0 to Min(4, High(featureMap)) do
            begin
               Write('    ');
               for w := 0 to Min(4, High(featureMap[h])) do
                  Write(featureMap[h][w]:0:3, ' ');
               WriteLn;
            end;
         end
         else
            WriteLn('  (empty or invalid)');
      end
      else if intrinMode = 'pre_activation' then
      begin
         WriteLn('Pre-Activation Map:');
         preAct := CNN.GetPreActivation(layerIdx, filterIdx);
         if Length(preAct) > 0 then
         begin
            WriteLn('  Size: ', Length(preAct), 'x', Length(preAct[0]));
            WriteLn('  First 5x5 (or less):');
            for h := 0 to Min(4, High(preAct)) do
            begin
               Write('    ');
               for w := 0 to Min(4, High(preAct[h])) do
                  Write(preAct[h][w]:0:3, ' ');
               WriteLn;
            end;
         end
         else
            WriteLn('  (empty or invalid)');
      end
      else if intrinMode = 'kernel' then
      begin
         WriteLn('Kernel (Channel ', channelIdx, '):');
         kernel := CNN.GetKernel(layerIdx, filterIdx, channelIdx);
         if Length(kernel) > 0 then
         begin
            WriteLn('  Size: ', Length(kernel), 'x', Length(kernel[0]));
            for h := 0 to High(kernel) do
            begin
               for w := 0 to High(kernel[h]) do
                  Write(kernel[h][w]:0:4, ' ');
               WriteLn;
            end;
         end
         else
            WriteLn('  (empty or invalid)');
      end
      else if intrinMode = 'bias' then
      begin
         WriteLn('Bias: ', CNN.GetBias(layerIdx, filterIdx):0:6);
      end;
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdStats then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Computing layer statistics...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      { Forward pass }
      Image.Width := inputW;
      Image.Height := inputH;
      Image.Channels := inputC;
      SetLength(Image.Data, inputC, inputH, inputW);
      for c := 0 to inputC - 1 do
         for h := 0 to inputH - 1 do
            for w := 0 to inputW - 1 do
               Image.Data[c][h][w] := Random;
      Output := CNN.Predict(Image);
      
      WriteLn('Layer ', layerIdx, ' Statistics:');
      
      if (statsType = 'activations') or (statsType = 'both') then
      begin
         stats := CNN.GetLayerStats(layerIdx);
         WriteLn('Activation Stats:');
         WriteLn('  Count: ', stats.Count);
         WriteLn('  Mean: ', stats.Mean:0:6);
         WriteLn('  StdDev: ', stats.StdDev:0:6);
         WriteLn('  Min: ', stats.Min:0:6);
         WriteLn('  Max: ', stats.Max:0:6);
         
         if binCount > 0 then
         begin
            histogram := CNN.GetActivationHistogram(layerIdx, binCount);
            WriteLn('Activation Histogram (', binCount, ' bins):');
            for binIdx := 0 to Min(9, High(histogram)) do
               WriteLn('  Bin ', binIdx, ': ', histogram[binIdx]:0:0);
            if binCount > 10 then
               WriteLn('  ... (', binCount - 10, ' more bins)');
         end;
      end;
      
      if (statsType = 'weights') or (statsType = 'both') then
      begin
         WriteLn('Weight Histogram (', binCount, ' bins):');
         histogram := CNN.GetWeightHistogram(layerIdx, binCount);
         for binIdx := 0 to Min(9, High(histogram)) do
            WriteLn('  Bin ', binIdx, ': ', histogram[binIdx]:0:0);
         if binCount > 10 then
            WriteLn('  ... (', binCount - 10, ' more bins)');
      end;
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdModify then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Modifying CNN architecture...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      if modifyAction = 'add_filter' then
      begin
         WriteLn('Adding ', numFiltersToAdd, ' filter(s) to layer ', layerIdx);
         for j := 1 to numFiltersToAdd do
         begin
            SetLength(filterParams, 0);
            CNN.AddFilter(layerIdx, filterParams);
         end;
         WriteLn('Now layer ', layerIdx, ' has ', CNN.GetNumFilters(layerIdx), ' filters');
      end
      else if modifyAction = 'remove_filter' then
      begin
         WriteLn('Removing filter ', filterIdx, ' from layer ', layerIdx);
         CNN.RemoveFilter(layerIdx, filterIdx);
         WriteLn('Now layer ', layerIdx, ' has ', CNN.GetNumFilters(layerIdx), ' filters');
      end
      else if modifyAction = 'add_layer' then
      begin
         WriteLn('Adding conv layer at position ', layerIdx);
         WriteLn('  Filters: ', numFiltersToAdd, ', Kernel: ', kernelSizeAdd, 
                 ', Stride: ', strideAdd, ', Padding: ', paddingAdd);
         CNN.AddConvLayer(layerIdx, numFiltersToAdd, kernelSizeAdd, strideAdd, paddingAdd);
         WriteLn('Now model has ', CNN.GetNumConvLayers, ' conv layers');
      end;
      
      WriteLn('Modified architecture:');
      WriteLn('  Total layers: ', CNN.GetNumLayers);
      WriteLn('  Conv layers: ', CNN.GetNumConvLayers);
      WriteLn('  FC layers: ', CNN.GetNumFCLayers);
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdAnalyze then
   begin
      if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
      if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
      if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
      if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
      if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
      if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
      if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;

      WriteLn('Analyzing CNN layer features...');
      CNN := TCNNFacade.Create(inputW, inputH, inputC, convFilters, kernelSizes,
                               poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);
      
      { Forward pass }
      Image.Width := inputW;
      Image.Height := inputH;
      Image.Channels := inputC;
      SetLength(Image.Data, inputC, inputH, inputW);
      for c := 0 to inputC - 1 do
         for h := 0 to inputH - 1 do
            for w := 0 to inputW - 1 do
               Image.Data[c][h][w] := Random;
      Output := CNN.Predict(Image);
      
      if analyzeType = 'saliency' then
      begin
         WriteLn('Saliency map for layer ', layerIdx, ' filter ', filterIdx);
         saliency := CNN.GetSaliencyMap(layerIdx, filterIdx, 0);
         if Length(saliency) > 0 then
         begin
            WriteLn('  Size: ', Length(saliency), 'x', Length(saliency[0]));
            WriteLn('  First 5x5 (or less):');
            for h := 0 to Min(4, High(saliency)) do
            begin
               Write('    ');
               for w := 0 to Min(4, High(saliency[h])) do
                  Write(saliency[h][w]:0:3, ' ');
               WriteLn;
            end;
         end
         else
            WriteLn('  (empty or invalid)');
      end
      else if analyzeType = 'receptive_field' then
      begin
         WriteLn('Receptive field for layer ', layerIdx, ' at position (', posX, ',', posY, ')');
         rf := CNN.GetReceptiveField(layerIdx, filterIdx, posY, posX);
         WriteLn('  Input region: X[', rf.StartX, '..', rf.EndX, '] Y[', rf.StartY, '..', rf.EndY, ']');
         WriteLn('  Channels: ', Length(rf.Channels));
      end
      else
         WriteLn('Unknown analyze type: ', analyzeType);
      
      CNN.Free;
      WriteLn('Done.');
   end
   else if Command = cmdInfo then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;

      WriteLn('Loading model from JSON: ' + modelFile);
      CNN := TCNNFacade.Create(0, 0, 0, [], [], [], [], 0, 0.001, 0.25);
      CNN.LoadModelFromJSON(modelFile);
      
      WriteLn('CNN Architecture Info');
      WriteLn;
      
      WriteLn('Total Layers: ', CNN.GetNumLayers);
      WriteLn('  Conv Layers: ', CNN.GetNumConvLayers);
      WriteLn('  FC Layers: ', CNN.GetNumFCLayers);
      WriteLn;
      
      WriteLn('Layer Details:');
      for i := 0 to CNN.GetNumLayers - 1 do
      begin
         config := CNN.GetLayerConfig(i);
         WriteLn('Layer ', i, ': ', config.LayerType);
         if config.LayerType = 'conv' then
         begin
            WriteLn('  Filters: ', config.FilterCount);
            WriteLn('  Kernel: ', config.KernelSize, 'x', config.KernelSize);
            WriteLn('  Stride: ', config.Stride);
            WriteLn('  Padding: ', config.Padding);
            WriteLn('  Input Channels: ', config.InputChannels);
            WriteLn('  Output: ', config.OutputWidth, 'x', config.OutputHeight);
         end
         else if config.LayerType = 'pool' then
         begin
            WriteLn('  Pool Size: ', config.PoolSize);
            WriteLn('  Output: ', config.OutputWidth, 'x', config.OutputHeight);
         end
         else if (config.LayerType = 'fc') or (config.LayerType = 'output') then
         begin
            WriteLn('  Neurons: ', config.NeuronCount);
            WriteLn('  Input Size: ', config.InputSize);
         end;
      end;
      
      WriteLn;
      WriteLn('Training Config:');
      WriteLn('  Learning Rate: ', learningRate:0:6);
      WriteLn('  Dropout: ', dropoutRate:0:6);
      
      CNN.Free;
      WriteLn('Done.');
   end;
end.

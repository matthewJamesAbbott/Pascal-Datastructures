{
  CNNFacade - Comprehensive facade for CNN introspection and manipulation
  Matthew Abbott 2025
}

{$mode objfpc}{$H+}

unit CNNFacade;

interface

uses Classes, Math, SysUtils;

const
   EPSILON = 1e-8;
   GRAD_CLIP = 1.0;

type
   Darray = array of Double;
   D2array = array of array of Double;
   D3array = array of array of array of Double;
   D4array = array of array of array of array of Double;
   IntArray = array of Integer;

   TPoolIndex = record
      X, Y: Integer;
   end;
   TPoolIndexArray = array of array of array of TPoolIndex;

   TLayerStats = record
      Mean: Double;
      StdDev: Double;
      Min: Double;
      Max: Double;
      Count: Integer;
   end;

   TLayerConfig = record
      LayerType: string;
      FilterCount: Integer;
      KernelSize: Integer;
      Stride: Integer;
      Padding: Integer;
      InputChannels: Integer;
      OutputWidth: Integer;
      OutputHeight: Integer;
      PoolSize: Integer;
      NeuronCount: Integer;
      InputSize: Integer;
   end;

   TReceptiveField = record
      StartX, EndX: Integer;
      StartY, EndY: Integer;
      Channels: IntArray;
   end;

   TAttributeEntry = record
      Key: string;
      Value: string;
   end;
   TAttributeArray = array of TAttributeEntry;

   TImageData = record
      Width: Integer;
      Height: Integer;
      Channels: Integer;
      Data: D3array;
   end;

   TConvFilter = record
      Weights: D3array;
      Bias: Double;
      WeightsM: D3array;
      WeightsV: D3array;
      WeightGrads: D3array;
      BiasGrad: Double;
      BiasM: Double;
      BiasV: Double;
   end;

   TConvLayer = record
      Filters: array of TConvFilter;
      OutputMaps: D3array;
      PreActivation: D3array;
      InputCache: D3array;
      PaddedInput: D3array;
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
      MaxIndices: TPoolIndexArray;
   end;

   TNeuron = record
      Weights: Darray;
      Bias: Double;
      Output: Double;
      PreActivation: Double;
      Error: Double;
      DropoutMask: Double;
      WeightsM: Darray;
      WeightsV: Darray;
      BiasM: Double;
      BiasV: Double;
   end;

   TFullyConnectedLayer = record
      Neurons: array of TNeuron;
      InputCache: Darray;
   end;

   TBatchNormParams = record
      Gamma: Darray;
      Beta: Darray;
      RunningMean: Darray;
      RunningVar: Darray;
      Enabled: Boolean;
   end;

   TCNNFacade = class
   private
      LearningRate: Double;
      DropoutRate: Double;
      Beta1, Beta2: Double;
      AdamT: Integer;
      IsTraining: Boolean;

      ConvLayers: array of TConvLayer;
      PoolLayers: array of TPoolingLayer;
      FullyConnectedLayers: array of TFullyConnectedLayer;
      OutputLayer: TFullyConnectedLayer;

      FlattenedSize: Integer;
      FFlattenedFeatures: Darray;
      LastConvHeight, LastConvWidth, LastConvChannels: Integer;

      FBatchActivations: array of D4array;
      FBatchNormParams: array of TBatchNormParams;
      FFilterAttributes: array of array of TAttributeArray;

      function ReLU(x: Double): Double;
      function ReLUDerivative(x: Double): Double;
      function Softmax(const Logits: Darray): Darray;
      function CrossEntropyLoss(const Predicted, Target: Darray): Double;
      function ClipGrad(x: Double): Double;
      function IsFiniteNum(x: Double): Boolean;
      function Clamp(x, MinVal, MaxVal: Double): Double;
      function Pad3D(const Input: D3array; Padding: Integer): D3array;
      function ValidateInput(const Image: TImageData): Boolean;

      procedure InitializeConvLayer(var Layer: TConvLayer; NumFilters, InputChannels, KernelSize, Stride, Padding: Integer);
      procedure InitializePoolLayer(var Layer: TPoolingLayer; PoolSize, Stride: Integer);
      procedure InitializeFCLayer(var Layer: TFullyConnectedLayer; NumNeurons, NumInputs: Integer);

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

   public
      constructor Create(InputWidth, InputHeight, InputChannels: Integer;
                        ConvFilters, KernelSizes, PoolSizes, FCLayerSizes: array of Integer;
                        OutputSize: Integer; ALearningRate: Double = 0.001; ADropoutRate: Double = 0.25);
      destructor Destroy; override;

      function Predict(var Image: TImageData): Darray;
      function TrainStep(var Image: TImageData; const Target: Darray): Double;
      procedure SaveModel(const Filename: string);
      procedure LoadModel(const Filename: string);

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

implementation

{ Helper functions }

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

procedure TCNNFacade.SaveModel(const Filename: string);
begin
end;

procedure TCNNFacade.LoadModel(const Filename: string);
begin
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

end.

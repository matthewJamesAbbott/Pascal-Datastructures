{
  MLPFacade - Facade for TMultiLayerPerceptron
  Provides comprehensive accessors for neurons, layers, weights, gradients,
  optimizer states, dropout masks, batch normalization, and more.
  
  Matthew Abbott 2025
}

{$mode objfpc}
{$H+}

unit MLPFacade;

interface

uses Classes, Math, SysUtils;

const
   EPSILON = 1e-15;
   HISTOGRAM_BINS = 20;

type
   TActivationType = (atSigmoid, atTanh, atReLU, atSoftmax);
   TOptimizerType = (otSGD, otAdam, otRMSProp);
   
   Darray = array of Double;
   
   TDataPoint = record
      Input: Darray;
      Target: Darray;
   end;
   TDataPointArray = array of TDataPoint;
   
   THistogramBin = record
      RangeMin: Double;
      RangeMax: Double;
      Count: Integer;
   end;
   THistogram = array of THistogramBin;
   
   TConnection = record
      FromLayerIdx: Integer;
      FromNeuronIdx: Integer;
      Weight: Double;
   end;
   TConnectionArray = array of TConnection;

   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
      PreActivation: Double;
      M: array of Double;
      V: array of Double;
      MBias: Double;
      VBias: Double;
      L2Lambda: Double;
      Attributes: TStringList;
   end;
   PNeuron = ^TNeuron;

   TLayer = record
      Neurons: array of TNeuron;
      ActivationType: TActivationType;
      DropoutMask: array of Boolean;
      LearningRate: Double;
      RunningMean: Darray;
      RunningVar: Darray;
   end;
   PLayer = ^TLayer;

   TMultiLayerPerceptron = class
   private
      FInputLayer: TLayer;
      FHiddenLayers: array of TLayer;
      FOutputLayer: TLayer;
      FHiddenSizes: array of Integer;
      FInputSize: Integer;
      FOutputSize: Integer;
      FIsTraining: Boolean;
      procedure InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer; ActType: TActivationType);
      procedure FeedForward;
      procedure BackPropagate(Target: Darray);
      procedure UpdateWeights;
      procedure UpdateNeuronWeightsSGD(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
      procedure UpdateNeuronWeightsAdam(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
      procedure UpdateNeuronWeightsRMSProp(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
      procedure ApplyDropout(var Layer: TLayer);
      function InitializeWeights(NumInputs, NumOutputs: Integer; ActType: TActivationType): Darray;
   public
      LearningRate: Double;
      MaxIterations: Integer;
      Optimizer: TOptimizerType;
      HiddenActivation: TActivationType;
      OutputActivation: TActivationType;
      DropoutRate: Double;
      L2Lambda: Double;
      Beta1: Double;
      Beta2: Double;
      Timestep: Integer;
      EnableLRDecay: Boolean;
      LRDecayRate: Double;
      LRDecayEpochs: Integer;
      EnableEarlyStopping: Boolean;
      EarlyStoppingPatience: Integer;
      
      constructor Create(InputSize: Integer; HiddenSizes: array of Integer; OutputSize: Integer;
                        HiddenAct: TActivationType = atSigmoid; OutputAct: TActivationType = atSigmoid);
      destructor Destroy; override;
      function Predict(Input: Darray): Darray;
      procedure Train(Input, Target: Darray);
      procedure TrainEpoch(var Data: TDataPointArray; BatchSize: Integer);
      function ComputeLoss(const Predicted, Target: Darray): Double;
      procedure SaveMLPModel(const Filename: string);
      
      property InputLayer: TLayer read FInputLayer;
      property OutputLayer: TLayer read FOutputLayer;
      property InputSize: Integer read FInputSize;
      property OutputSize: Integer read FOutputSize;
      function GetHiddenLayer(Index: Integer): TLayer;
      function GetHiddenLayerCount: Integer;
   end;
   
   { TMLPFacade - Provides comprehensive access to MLP internals }
   TMLPFacade = class
   private
      FMLP: TMultiLayerPerceptron;
      function GetLayerPtr(LayerIdx: Integer): PLayer;
      function GetNeuronPtr(LayerIdx, NeuronIdx: Integer): PNeuron;
      function GetPreviousLayerOutputs(LayerIdx: Integer): Darray;
   public
      constructor Create(MLP: TMultiLayerPerceptron);
      
      { Neuron output and error }
      function GetNeuronOutput(LayerIdx, NeuronIdx: Integer): Double;
      procedure SetNeuronOutput(LayerIdx, NeuronIdx: Integer; Value: Double);
      function GetNeuronError(LayerIdx, NeuronIdx: Integer): Double;
      procedure SetNeuronError(LayerIdx, NeuronIdx: Integer; Value: Double);
      
      { Neuron bias and individual weights }
      function GetNeuronBias(LayerIdx, NeuronIdx: Integer): Double;
      procedure SetNeuronBias(LayerIdx, NeuronIdx: Integer; Value: Double);
      function GetNeuronWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
      procedure SetNeuronWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
      function GetNeuronWeights(LayerIdx, NeuronIdx: Integer): Darray;
      procedure SetNeuronWeights(LayerIdx, NeuronIdx: Integer; const Weights: Darray);
      
      { Layer queries }
      function GetLayerSize(LayerIdx: Integer): Integer;
      function GetTotalLayers: Integer;
      function GetWeightsPerNeuron(LayerIdx, NeuronIdx: Integer): Integer;
      
      { Gradients }
      function GetNeuronWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
      function GetNeuronBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
      
      { Optimizer state (Adam m/v, RMSProp v) }
      function GetNeuronOptimizerState(LayerIdx, NeuronIdx: Integer; Param: string): Double;
      function GetNeuronOptimizerStateArray(LayerIdx, NeuronIdx: Integer; Param: string): Darray;
      
      { Full neuron/layer access }
      function GetNeuron(LayerIdx, NeuronIdx: Integer): PNeuron;
      function GetLayer(LayerIdx: Integer): PLayer;
      
      { Dropout mask }
      function GetNeuronDropoutMask(LayerIdx, NeuronIdx: Integer): Boolean;
      procedure SetNeuronDropoutMask(LayerIdx, NeuronIdx: Integer; Value: Boolean);
      
      { Pre-activation values }
      function GetNeuronPreActivation(LayerIdx, NeuronIdx: Integer): Double;
      procedure SetNeuronPreActivation(LayerIdx, NeuronIdx: Integer; Value: Double);
      
      { Batch normalization running stats }
      function GetLayerRunningMean(LayerIdx: Integer): Darray;
      function GetLayerRunningVar(LayerIdx: Integer): Darray;
      procedure SetLayerRunningMean(LayerIdx: Integer; const Values: Darray);
      procedure SetLayerRunningVar(LayerIdx: Integer; const Values: Darray);
      
      { Custom neuron attributes }
      procedure SetNeuronAttribute(LayerIdx, NeuronIdx: Integer; Key: string; Value: string);
      function GetNeuronAttribute(LayerIdx, NeuronIdx: Integer; Key: string): string;
      
      { Per-layer learning rate }
      function GetLayerLearningRate(LayerIdx: Integer): Double;
      procedure SetLayerLearningRate(LayerIdx: Integer; Value: Double);
      
      { Per-neuron L2 regularization }
      function GetNeuronL2Lambda(LayerIdx, NeuronIdx: Integer): Double;
      procedure SetNeuronL2Lambda(LayerIdx, NeuronIdx: Integer; Value: Double);
      
      { Connection queries }
      function GetNeuronIncomingConnections(LayerIdx, NeuronIdx: Integer): TConnectionArray;
      function GetNeuronOutgoingConnections(LayerIdx, NeuronIdx: Integer): TConnectionArray;
      
      { Dynamic topology modification }
      function AddNeuron(LayerIdx: Integer): Integer;
      procedure RemoveNeuron(LayerIdx, NeuronIdx: Integer);
      function AddLayer(Position: Integer; Size: Integer; ActType: TActivationType): Integer;
      procedure RemoveLayer(LayerIdx: Integer);
      
      { Histograms for visualization }
      function GetLayerActivationHistogram(LayerIdx: Integer): THistogram;
      function GetLayerGradientHistogram(LayerIdx: Integer): THistogram;
      
      { Utility }
      property MLP: TMultiLayerPerceptron read FMLP;
   end;

{ Standalone activation functions }
function Sigmoid(x: Double): Double;
function DSigmoid(x: Double): Double;
function TanhActivation(x: Double): Double;
function DTanh(x: Double): Double;
function ReLU(x: Double): Double;
function DReLU(x: Double): Double;
function Softmax(const x: Darray): Darray;
function ApplyActivation(x: Double; ActType: TActivationType): Double;
function ApplyActivationDerivative(x: Double; ActType: TActivationType): Double;

{ Helper functions }
function MaxIndex(const Arr: Darray): Integer;
function CloneDataArray(const Data: TDataPointArray): TDataPointArray;
procedure ShuffleData(var Data: TDataPointArray);
function NormalizeData(var Data: TDataPointArray): Boolean;
procedure CheckDataQuality(const Data: TDataPointArray);
function LoadMLPModel(const Filename: string): TMultiLayerPerceptron;
function KFoldCrossValidation(var Data: TDataPointArray; NumFolds: Integer; MLP: TMultiLayerPerceptron): Double;
function TrainWithEarlyStopping(MLP: TMultiLayerPerceptron; var Data: TDataPointArray; MaxEpochs, BatchSize: Integer): Integer;
function PrecisionScore(var Data: TDataPointArray; MLP: TMultiLayerPerceptron; ClassIndex: Integer): Double;
function RecallScore(var Data: TDataPointArray; MLP: TMultiLayerPerceptron; ClassIndex: Integer): Double;
function F1Score(Precision, Recall: Double): Double;

implementation

{ Activation Functions }

function Sigmoid(x: Double): Double;
begin
   if x < -500 then
      Result := 0
   else if x > 500 then
      Result := 1
   else
      Result := 1 / (1 + Exp(-x));
end;

function DSigmoid(x: Double): Double;
begin
   Result := x * (1 - x);
end;

function TanhActivation(x: Double): Double;
begin
   Result := Tanh(x);
end;

function DTanh(x: Double): Double;
begin
   Result := 1 - (x * x);
end;

function ReLU(x: Double): Double;
begin
   if x > 0 then
      Result := x
   else
      Result := 0;
end;

function DReLU(x: Double): Double;
begin
   if x > 0 then
      Result := 1
   else
      Result := 0;
end;

function Softmax(const x: Darray): Darray;
var
   i: Integer;
   MaxVal, Sum: Double;
   ExpValues: Darray;
begin
   SetLength(Result, Length(x));
   SetLength(ExpValues, Length(x));
   
   MaxVal := x[0];
   for i := 1 to High(x) do
      if x[i] > MaxVal then
         MaxVal := x[i];
   
   Sum := 0;
   for i := 0 to High(x) do
   begin
      ExpValues[i] := Exp(x[i] - MaxVal);
      Sum := Sum + ExpValues[i];
   end;
   
   for i := 0 to High(x) do
   begin
      Result[i] := ExpValues[i] / Sum;
      if Result[i] < EPSILON then
         Result[i] := EPSILON
      else if Result[i] > 1 - EPSILON then
         Result[i] := 1 - EPSILON;
   end;
end;

function ApplyActivation(x: Double; ActType: TActivationType): Double;
begin
   case ActType of
      atSigmoid: Result := Sigmoid(x);
      atTanh: Result := TanhActivation(x);
      atReLU: Result := ReLU(x);
      else Result := Sigmoid(x);
   end;
end;

function ApplyActivationDerivative(x: Double; ActType: TActivationType): Double;
begin
   case ActType of
      atSigmoid: Result := DSigmoid(x);
      atTanh: Result := DTanh(x);
      atReLU: Result := DReLU(x);
      else Result := DSigmoid(x);
   end;
end;

{ Helper Functions }

function MaxIndex(const Arr: Darray): Integer;
var
   i: Integer;
begin
   Result := 0;
   for i := 1 to High(Arr) do
      if Arr[i] > Arr[Result] then
         Result := i;
end;

function CloneDataArray(const Data: TDataPointArray): TDataPointArray;
var
   i, j: Integer;
begin
   SetLength(Result, Length(Data));
   for i := 0 to High(Data) do
   begin
      SetLength(Result[i].Input, Length(Data[i].Input));
      SetLength(Result[i].Target, Length(Data[i].Target));
      for j := 0 to High(Data[i].Input) do
         Result[i].Input[j] := Data[i].Input[j];
      for j := 0 to High(Data[i].Target) do
         Result[i].Target[j] := Data[i].Target[j];
   end;
end;

procedure ShuffleData(var Data: TDataPointArray);
var
   i, j: Integer;
   Temp: TDataPoint;
begin
   for i := High(Data) downto 1 do
   begin
      j := Random(i + 1);
      Temp := Data[i];
      Data[i] := Data[j];
      Data[j] := Temp;
   end;
end;

function NormalizeData(var Data: TDataPointArray): Boolean;
var
   i, j: Integer;
   InputSize: Integer;
   Mins, Maxs: Darray;
   Range: Double;
begin
   Result := False;
   if Length(Data) = 0 then Exit;
   
   InputSize := Length(Data[0].Input);
   SetLength(Mins, InputSize);
   SetLength(Maxs, InputSize);
   
   for j := 0 to InputSize - 1 do
   begin
      Mins[j] := Data[0].Input[j];
      Maxs[j] := Data[0].Input[j];
   end;
   
   for i := 0 to High(Data) do
      for j := 0 to InputSize - 1 do
      begin
         if Data[i].Input[j] < Mins[j] then
            Mins[j] := Data[i].Input[j];
         if Data[i].Input[j] > Maxs[j] then
            Maxs[j] := Data[i].Input[j];
      end;
   
   for i := 0 to High(Data) do
      for j := 0 to InputSize - 1 do
      begin
         Range := Maxs[j] - Mins[j];
         if Range > 0 then
            Data[i].Input[j] := (Data[i].Input[j] - Mins[j]) / Range
         else
            Data[i].Input[j] := 0.5;
      end;
   
   Result := True;
end;

procedure CheckDataQuality(const Data: TDataPointArray);
var
   i, j: Integer;
   InputSize: Integer;
   MinVal, MaxVal: Double;
begin
   if Length(Data) = 0 then Exit;
   
   InputSize := Length(Data[0].Input);
   
   for j := 0 to InputSize - 1 do
   begin
      MinVal := Data[0].Input[j];
      MaxVal := Data[0].Input[j];
      
      for i := 1 to High(Data) do
      begin
         if Data[i].Input[j] < MinVal then
            MinVal := Data[i].Input[j];
         if Data[i].Input[j] > MaxVal then
            MaxVal := Data[i].Input[j];
      end;
      
      if (MaxVal - MinVal) > 100 then
         WriteLn('Warning: Feature ', j, ' has large range (', MinVal:0:2, ' to ', MaxVal:0:2, '). Consider normalizing.');
      if (MinVal < -10) or (MaxVal > 10) then
         WriteLn('Warning: Feature ', j, ' has values outside [-10, 10]. Consider normalizing.');
   end;
end;

{ TMultiLayerPerceptron }

constructor TMultiLayerPerceptron.Create(InputSize: Integer; HiddenSizes: array of Integer; 
                                         OutputSize: Integer; HiddenAct: TActivationType = atSigmoid; 
                                         OutputAct: TActivationType = atSigmoid);
var
   i: Integer;
   NumInputs: Integer;
begin
   LearningRate := 0.1;
   MaxIterations := 100;
   Optimizer := otSGD;
   HiddenActivation := HiddenAct;
   OutputActivation := OutputAct;
   DropoutRate := 0;
   L2Lambda := 0;
   Beta1 := 0.9;
   Beta2 := 0.999;
   Timestep := 0;
   EnableLRDecay := False;
   LRDecayRate := 0.95;
   LRDecayEpochs := 10;
   EnableEarlyStopping := False;
   EarlyStoppingPatience := 10;
   FIsTraining := True;
   
   FInputSize := InputSize;
   FOutputSize := OutputSize;
   
   SetLength(FHiddenLayers, Length(HiddenSizes));
   SetLength(FHiddenSizes, Length(HiddenSizes));
   
   for i := 0 to High(HiddenSizes) do
      FHiddenSizes[i] := HiddenSizes[i];

   InitializeLayer(FInputLayer, InputSize + 1, InputSize, atSigmoid);
   
   NumInputs := InputSize;
   for i := 0 to High(HiddenSizes) do
   begin
      InitializeLayer(FHiddenLayers[i], HiddenSizes[i] + 1, NumInputs + 1, HiddenActivation);
      NumInputs := HiddenSizes[i];
   end;
   
   InitializeLayer(FOutputLayer, OutputSize, NumInputs + 1, OutputActivation);
end;

destructor TMultiLayerPerceptron.Destroy;
var
   i, j: Integer;
begin
   for i := 0 to High(FInputLayer.Neurons) do
      if Assigned(FInputLayer.Neurons[i].Attributes) then
         FInputLayer.Neurons[i].Attributes.Free;
   
   for i := 0 to High(FHiddenLayers) do
      for j := 0 to High(FHiddenLayers[i].Neurons) do
         if Assigned(FHiddenLayers[i].Neurons[j].Attributes) then
            FHiddenLayers[i].Neurons[j].Attributes.Free;
   
   for i := 0 to High(FOutputLayer.Neurons) do
      if Assigned(FOutputLayer.Neurons[i].Attributes) then
         FOutputLayer.Neurons[i].Attributes.Free;
   
   inherited Destroy;
end;

function TMultiLayerPerceptron.GetHiddenLayer(Index: Integer): TLayer;
begin
   Result := FHiddenLayers[Index];
end;

function TMultiLayerPerceptron.GetHiddenLayerCount: Integer;
begin
   Result := Length(FHiddenLayers);
end;

function TMultiLayerPerceptron.InitializeWeights(NumInputs, NumOutputs: Integer; ActType: TActivationType): Darray;
var
   i: Integer;
   Limit: Double;
begin
   SetLength(Result, NumInputs);
   
   if ActType = atReLU then
      Limit := Sqrt(2.0 / NumInputs)
   else
      Limit := Sqrt(6.0 / (NumInputs + NumOutputs));
   
   for i := 0 to NumInputs - 1 do
      Result[i] := (Random * 2 - 1) * Limit;
end;

procedure TMultiLayerPerceptron.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer; ActType: TActivationType);
var
   i, j: Integer;
begin
   Layer.ActivationType := ActType;
   Layer.LearningRate := -1;
   SetLength(Layer.Neurons, NumNeurons);
   SetLength(Layer.DropoutMask, NumNeurons);
   SetLength(Layer.RunningMean, NumNeurons);
   SetLength(Layer.RunningVar, NumNeurons);
   
   for i := 0 to NumNeurons - 1 do
   begin
      Layer.Neurons[i].Weights := InitializeWeights(NumInputs, NumNeurons, ActType);
      Layer.Neurons[i].Bias := 0;
      Layer.Neurons[i].PreActivation := 0;
      Layer.Neurons[i].L2Lambda := -1;
      Layer.Neurons[i].Attributes := nil;
      Layer.DropoutMask[i] := True;
      Layer.RunningMean[i] := 0;
      Layer.RunningVar[i] := 1;
      
      SetLength(Layer.Neurons[i].M, NumInputs);
      SetLength(Layer.Neurons[i].V, NumInputs);
      for j := 0 to NumInputs - 1 do
      begin
         Layer.Neurons[i].M[j] := 0;
         Layer.Neurons[i].V[j] := 0;
      end;
      Layer.Neurons[i].MBias := 0;
      Layer.Neurons[i].VBias := 0;
   end;
end;

procedure TMultiLayerPerceptron.ApplyDropout(var Layer: TLayer);
var
   i: Integer;
   Scale: Double;
begin
   if (not FIsTraining) or (DropoutRate <= 0) then
   begin
      for i := 0 to High(Layer.Neurons) do
         Layer.DropoutMask[i] := True;
      Exit;
   end;
   
   Scale := 1.0 / (1.0 - DropoutRate);
   for i := 0 to High(Layer.Neurons) do
   begin
      if Random > DropoutRate then
      begin
         Layer.DropoutMask[i] := True;
         Layer.Neurons[i].Output := Layer.Neurons[i].Output * Scale;
      end
      else
      begin
         Layer.DropoutMask[i] := False;
         Layer.Neurons[i].Output := 0;
      end;
   end;
end;

procedure TMultiLayerPerceptron.FeedForward;
var
   i, j, k: Integer;
   Sum: Double;
   OutputSums: Darray;
   SoftmaxOutputs: Darray;
begin
   for k := 0 to High(FHiddenLayers) do
   begin
      for i := 0 to High(FHiddenLayers[k].Neurons) do
      begin
         Sum := FHiddenLayers[k].Neurons[i].Bias;
         if k = 0 then
         begin
            for j := 0 to High(FInputLayer.Neurons) do
               Sum := Sum + FInputLayer.Neurons[j].Output * FHiddenLayers[k].Neurons[i].Weights[j];
         end
         else
         begin
            for j := 0 to High(FHiddenLayers[k-1].Neurons) do
               Sum := Sum + FHiddenLayers[k-1].Neurons[j].Output * FHiddenLayers[k].Neurons[i].Weights[j];
         end;
         FHiddenLayers[k].Neurons[i].PreActivation := Sum;
         FHiddenLayers[k].Neurons[i].Output := ApplyActivation(Sum, FHiddenLayers[k].ActivationType);
      end;
      ApplyDropout(FHiddenLayers[k]);
   end;

   if OutputActivation = atSoftmax then
   begin
      SetLength(OutputSums, Length(FOutputLayer.Neurons));
      for i := 0 to High(FOutputLayer.Neurons) do
      begin
         Sum := FOutputLayer.Neurons[i].Bias;
         for j := 0 to High(FHiddenLayers[High(FHiddenLayers)].Neurons) do
            Sum := Sum + FHiddenLayers[High(FHiddenLayers)].Neurons[j].Output * FOutputLayer.Neurons[i].Weights[j];
         OutputSums[i] := Sum;
         FOutputLayer.Neurons[i].PreActivation := Sum;
      end;
      SoftmaxOutputs := Softmax(OutputSums);
      for i := 0 to High(FOutputLayer.Neurons) do
         FOutputLayer.Neurons[i].Output := SoftmaxOutputs[i];
   end
   else
   begin
      for i := 0 to High(FOutputLayer.Neurons) do
      begin
         Sum := FOutputLayer.Neurons[i].Bias;
         for j := 0 to High(FHiddenLayers[High(FHiddenLayers)].Neurons) do
            Sum := Sum + FHiddenLayers[High(FHiddenLayers)].Neurons[j].Output * FOutputLayer.Neurons[i].Weights[j];
         FOutputLayer.Neurons[i].PreActivation := Sum;
         FOutputLayer.Neurons[i].Output := ApplyActivation(Sum, OutputActivation);
      end;
   end;
end;

function TMultiLayerPerceptron.ComputeLoss(const Predicted, Target: Darray): Double;
var
   i, j, k: Integer;
   p, L2Sum, EffectiveLambda: Double;
begin
   Result := 0;
   
   if OutputActivation = atSoftmax then
   begin
      for i := 0 to High(Target) do
      begin
         p := Predicted[i];
         if p < EPSILON then p := EPSILON;
         if p > 1 - EPSILON then p := 1 - EPSILON;
         Result := Result - Target[i] * Ln(p);
      end;
   end
   else
   begin
      for i := 0 to High(Target) do
         Result := Result + 0.5 * Sqr(Target[i] - Predicted[i]);
   end;
   
   if L2Lambda > 0 then
   begin
      L2Sum := 0;
      for k := 0 to High(FHiddenLayers) do
         for i := 0 to High(FHiddenLayers[k].Neurons) do
         begin
            if FHiddenLayers[k].Neurons[i].L2Lambda >= 0 then
               EffectiveLambda := FHiddenLayers[k].Neurons[i].L2Lambda
            else
               EffectiveLambda := L2Lambda;
            for j := 0 to High(FHiddenLayers[k].Neurons[i].Weights) do
               L2Sum := L2Sum + EffectiveLambda * Sqr(FHiddenLayers[k].Neurons[i].Weights[j]);
         end;
      
      for i := 0 to High(FOutputLayer.Neurons) do
      begin
         if FOutputLayer.Neurons[i].L2Lambda >= 0 then
            EffectiveLambda := FOutputLayer.Neurons[i].L2Lambda
         else
            EffectiveLambda := L2Lambda;
         for j := 0 to High(FOutputLayer.Neurons[i].Weights) do
            L2Sum := L2Sum + EffectiveLambda * Sqr(FOutputLayer.Neurons[i].Weights[j]);
      end;
      
      Result := Result + 0.5 * L2Sum;
   end;
end;

procedure TMultiLayerPerceptron.BackPropagate(Target: Darray);
var
   i, j, k: Integer;
   ErrorSum: Double;
begin
   if OutputActivation = atSoftmax then
   begin
      for i := 0 to High(FOutputLayer.Neurons) do
         FOutputLayer.Neurons[i].Error := Target[i] - FOutputLayer.Neurons[i].Output;
   end
   else
   begin
      for i := 0 to High(FOutputLayer.Neurons) do
         FOutputLayer.Neurons[i].Error := ApplyActivationDerivative(FOutputLayer.Neurons[i].Output, OutputActivation) * 
                                          (Target[i] - FOutputLayer.Neurons[i].Output);
   end;

   for k := High(FHiddenLayers) downto 0 do
   begin
      for i := 0 to High(FHiddenLayers[k].Neurons) do
      begin
         if not FHiddenLayers[k].DropoutMask[i] then
         begin
            FHiddenLayers[k].Neurons[i].Error := 0;
            Continue;
         end;
         
         if k = High(FHiddenLayers) then
         begin
            ErrorSum := 0;
            for j := 0 to High(FOutputLayer.Neurons) do
               ErrorSum := ErrorSum + FOutputLayer.Neurons[j].Error * FOutputLayer.Neurons[j].Weights[i];
         end
         else
         begin
            ErrorSum := 0;
            for j := 0 to High(FHiddenLayers[k+1].Neurons) do
               ErrorSum := ErrorSum + FHiddenLayers[k+1].Neurons[j].Error * FHiddenLayers[k+1].Neurons[j].Weights[i];
         end;
         FHiddenLayers[k].Neurons[i].Error := ApplyActivationDerivative(FHiddenLayers[k].Neurons[i].Output, 
                                                                         FHiddenLayers[k].ActivationType) * ErrorSum;
      end;
   end;
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsSGD(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
var
   j: Integer;
   Gradient, EffectiveLR, EffectiveL2: Double;
begin
   EffectiveLR := LayerLR;
   if EffectiveLR < 0 then EffectiveLR := LearningRate;
   EffectiveL2 := NeuronL2;
   if EffectiveL2 < 0 then EffectiveL2 := L2Lambda;
   
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := Neuron.Error * PrevOutputs[j];
      if EffectiveL2 > 0 then
         Gradient := Gradient - EffectiveL2 * Neuron.Weights[j];
      Neuron.Weights[j] := Neuron.Weights[j] + EffectiveLR * Gradient;
   end;
   Neuron.Bias := Neuron.Bias + EffectiveLR * Neuron.Error;
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsAdam(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
var
   j: Integer;
   Gradient, MHat, VHat, EffectiveLR, EffectiveL2: Double;
   Eps: Double;
begin
   Eps := 1e-8;
   EffectiveLR := LayerLR;
   if EffectiveLR < 0 then EffectiveLR := LearningRate;
   EffectiveL2 := NeuronL2;
   if EffectiveL2 < 0 then EffectiveL2 := L2Lambda;
   
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := -Neuron.Error * PrevOutputs[j];
      if EffectiveL2 > 0 then
         Gradient := Gradient + EffectiveL2 * Neuron.Weights[j];
      
      Neuron.M[j] := Beta1 * Neuron.M[j] + (1 - Beta1) * Gradient;
      Neuron.V[j] := Beta2 * Neuron.V[j] + (1 - Beta2) * Gradient * Gradient;
      
      MHat := Neuron.M[j] / (1 - Power(Beta1, Timestep));
      VHat := Neuron.V[j] / (1 - Power(Beta2, Timestep));
      
      Neuron.Weights[j] := Neuron.Weights[j] - EffectiveLR * MHat / (Sqrt(VHat) + Eps);
   end;
   
   Gradient := -Neuron.Error;
   Neuron.MBias := Beta1 * Neuron.MBias + (1 - Beta1) * Gradient;
   Neuron.VBias := Beta2 * Neuron.VBias + (1 - Beta2) * Gradient * Gradient;
   MHat := Neuron.MBias / (1 - Power(Beta1, Timestep));
   VHat := Neuron.VBias / (1 - Power(Beta2, Timestep));
   Neuron.Bias := Neuron.Bias - EffectiveLR * MHat / (Sqrt(VHat) + Eps);
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsRMSProp(var Neuron: TNeuron; const PrevOutputs: Darray; LayerLR, NeuronL2: Double);
var
   j: Integer;
   Gradient, EffectiveLR, EffectiveL2: Double;
   Eps, Decay: Double;
begin
   Eps := 1e-8;
   Decay := 0.9;
   EffectiveLR := LayerLR;
   if EffectiveLR < 0 then EffectiveLR := LearningRate;
   EffectiveL2 := NeuronL2;
   if EffectiveL2 < 0 then EffectiveL2 := L2Lambda;
   
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := -Neuron.Error * PrevOutputs[j];
      if EffectiveL2 > 0 then
         Gradient := Gradient + EffectiveL2 * Neuron.Weights[j];
      
      Neuron.V[j] := Decay * Neuron.V[j] + (1 - Decay) * Gradient * Gradient;
      Neuron.Weights[j] := Neuron.Weights[j] - EffectiveLR * Gradient / (Sqrt(Neuron.V[j]) + Eps);
   end;
   
   Gradient := -Neuron.Error;
   Neuron.VBias := Decay * Neuron.VBias + (1 - Decay) * Gradient * Gradient;
   Neuron.Bias := Neuron.Bias - EffectiveLR * Gradient / (Sqrt(Neuron.VBias) + Eps);
end;

procedure TMultiLayerPerceptron.UpdateWeights;
var
   i, j, k: Integer;
   PrevOutputs: Darray;
   LayerLR, NeuronL2: Double;
begin
   Inc(Timestep);
   
   SetLength(PrevOutputs, Length(FHiddenLayers[High(FHiddenLayers)].Neurons));
   for j := 0 to High(FHiddenLayers[High(FHiddenLayers)].Neurons) do
      PrevOutputs[j] := FHiddenLayers[High(FHiddenLayers)].Neurons[j].Output;
   
   LayerLR := FOutputLayer.LearningRate;
   for i := 0 to High(FOutputLayer.Neurons) do
   begin
      NeuronL2 := FOutputLayer.Neurons[i].L2Lambda;
      case Optimizer of
         otSGD: UpdateNeuronWeightsSGD(FOutputLayer.Neurons[i], PrevOutputs, LayerLR, NeuronL2);
         otAdam: UpdateNeuronWeightsAdam(FOutputLayer.Neurons[i], PrevOutputs, LayerLR, NeuronL2);
         otRMSProp: UpdateNeuronWeightsRMSProp(FOutputLayer.Neurons[i], PrevOutputs, LayerLR, NeuronL2);
      end;
   end;

   for k := High(FHiddenLayers) downto 0 do
   begin
      if k = 0 then
      begin
         SetLength(PrevOutputs, Length(FInputLayer.Neurons));
         for j := 0 to High(FInputLayer.Neurons) do
            PrevOutputs[j] := FInputLayer.Neurons[j].Output;
      end
      else
      begin
         SetLength(PrevOutputs, Length(FHiddenLayers[k-1].Neurons));
         for j := 0 to High(FHiddenLayers[k-1].Neurons) do
            PrevOutputs[j] := FHiddenLayers[k-1].Neurons[j].Output;
      end;
      
      LayerLR := FHiddenLayers[k].LearningRate;
      for i := 0 to High(FHiddenLayers[k].Neurons) do
      begin
         NeuronL2 := FHiddenLayers[k].Neurons[i].L2Lambda;
         case Optimizer of
            otSGD: UpdateNeuronWeightsSGD(FHiddenLayers[k].Neurons[i], PrevOutputs, LayerLR, NeuronL2);
            otAdam: UpdateNeuronWeightsAdam(FHiddenLayers[k].Neurons[i], PrevOutputs, LayerLR, NeuronL2);
            otRMSProp: UpdateNeuronWeightsRMSProp(FHiddenLayers[k].Neurons[i], PrevOutputs, LayerLR, NeuronL2);
         end;
      end;
   end;
end;

function TMultiLayerPerceptron.Predict(Input: Darray): Darray;
var
   i: Integer;
begin
   FIsTraining := False;
   
   for i := 0 to High(FInputLayer.Neurons) do
      FInputLayer.Neurons[i].Output := Input[i];

   FeedForward;

   SetLength(Result, FOutputSize);
   for i := 0 to High(FOutputLayer.Neurons) do
      Result[i] := FOutputLayer.Neurons[i].Output;
   
   FIsTraining := True;
end;

procedure TMultiLayerPerceptron.Train(Input, Target: Darray);
var
   i: Integer;
begin
   FIsTraining := True;
   
   for i := 0 to High(FInputLayer.Neurons) do
      FInputLayer.Neurons[i].Output := Input[i];

   FeedForward;
   BackPropagate(Target);
   UpdateWeights;
end;

procedure TMultiLayerPerceptron.TrainEpoch(var Data: TDataPointArray; BatchSize: Integer);
var
   i, j, ActualBatchSize, BatchEnd: Integer;
   ShuffledData: TDataPointArray;
begin
   ActualBatchSize := BatchSize;
   if ActualBatchSize > Length(Data) then
      ActualBatchSize := Length(Data);
   if ActualBatchSize < 1 then
      ActualBatchSize := 1;
   
   ShuffledData := CloneDataArray(Data);
   ShuffleData(ShuffledData);
   
   i := 0;
   while i < Length(ShuffledData) do
   begin
      BatchEnd := i + ActualBatchSize;
      if BatchEnd > Length(ShuffledData) then
         BatchEnd := Length(ShuffledData);
      
      for j := i to BatchEnd - 1 do
         Train(ShuffledData[j].Input, ShuffledData[j].Target);
      
      i := BatchEnd;
   end;
end;

procedure TMultiLayerPerceptron.SaveMLPModel(const Filename: string);
var
   F: File;
   NumInputs, LayerCount, i, j, k: Integer;
   OptimizerInt, HiddenActInt, OutputActInt: Integer;
begin
   AssignFile(F, Filename);
   Rewrite(F, 1);
   
   LayerCount := Length(FHiddenLayers);
   BlockWrite(F, LayerCount, SizeOf(Integer));
   BlockWrite(F, FInputSize, SizeOf(Integer));
   for i := 0 to High(FHiddenLayers) do
      BlockWrite(F, FHiddenSizes[i], SizeOf(Integer));
   BlockWrite(F, FOutputSize, SizeOf(Integer));
   BlockWrite(F, LearningRate, SizeOf(Double));
   
   OptimizerInt := Ord(Optimizer);
   BlockWrite(F, OptimizerInt, SizeOf(Integer));
   HiddenActInt := Ord(HiddenActivation);
   BlockWrite(F, HiddenActInt, SizeOf(Integer));
   OutputActInt := Ord(OutputActivation);
   BlockWrite(F, OutputActInt, SizeOf(Integer));
   BlockWrite(F, DropoutRate, SizeOf(Double));
   BlockWrite(F, L2Lambda, SizeOf(Double));
   BlockWrite(F, Timestep, SizeOf(Integer));

   for i := 0 to High(FInputLayer.Neurons) do
   begin
      NumInputs := Length(FInputLayer.Neurons[i].Weights);
      BlockWrite(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(FInputLayer.Neurons[i].Weights) do
         BlockWrite(F, FInputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockWrite(F, FInputLayer.Neurons[i].Bias, SizeOf(Double));
   end;

   for k := 0 to High(FHiddenLayers) do
   begin
      for i := 0 to High(FHiddenLayers[k].Neurons) do
      begin
         NumInputs := Length(FHiddenLayers[k].Neurons[i].Weights);
         BlockWrite(F, NumInputs, SizeOf(Integer));
         for j := 0 to High(FHiddenLayers[k].Neurons[i].Weights) do
            BlockWrite(F, FHiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         BlockWrite(F, FHiddenLayers[k].Neurons[i].Bias, SizeOf(Double));
         
         for j := 0 to High(FHiddenLayers[k].Neurons[i].M) do
         begin
            BlockWrite(F, FHiddenLayers[k].Neurons[i].M[j], SizeOf(Double));
            BlockWrite(F, FHiddenLayers[k].Neurons[i].V[j], SizeOf(Double));
         end;
         BlockWrite(F, FHiddenLayers[k].Neurons[i].MBias, SizeOf(Double));
         BlockWrite(F, FHiddenLayers[k].Neurons[i].VBias, SizeOf(Double));
      end;
   end;

   for i := 0 to High(FOutputLayer.Neurons) do
   begin
      NumInputs := Length(FOutputLayer.Neurons[i].Weights);
      BlockWrite(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(FOutputLayer.Neurons[i].Weights) do
         BlockWrite(F, FOutputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockWrite(F, FOutputLayer.Neurons[i].Bias, SizeOf(Double));
      
      for j := 0 to High(FOutputLayer.Neurons[i].M) do
      begin
         BlockWrite(F, FOutputLayer.Neurons[i].M[j], SizeOf(Double));
         BlockWrite(F, FOutputLayer.Neurons[i].V[j], SizeOf(Double));
      end;
      BlockWrite(F, FOutputLayer.Neurons[i].MBias, SizeOf(Double));
      BlockWrite(F, FOutputLayer.Neurons[i].VBias, SizeOf(Double));
   end;
   
   CloseFile(F);
end;

function LoadMLPModel(const Filename: string): TMultiLayerPerceptron;
var
   F: File;
   HiddenLayerSize: array of Integer;
   InputSize, NumHiddenLayers, NumInputs, OutputSize, i, j, l: Integer;
   OptimizerInt, HiddenActInt, OutputActInt: Integer;
   MLP: TMultiLayerPerceptron;
begin
   AssignFile(F, Filename);
   Reset(F, 1);
   
   BlockRead(F, NumHiddenLayers, SizeOf(Integer));
   SetLength(HiddenLayerSize, NumHiddenLayers);
   BlockRead(F, InputSize, SizeOf(Integer));
   for i := 0 to High(HiddenLayerSize) do
      BlockRead(F, HiddenLayerSize[i], SizeOf(Integer));
   BlockRead(F, OutputSize, SizeOf(Integer));
   
   MLP := TMultiLayerPerceptron.Create(InputSize, HiddenLayerSize, OutputSize);
   BlockRead(F, MLP.LearningRate, SizeOf(Double));
   
   BlockRead(F, OptimizerInt, SizeOf(Integer));
   MLP.Optimizer := TOptimizerType(OptimizerInt);
   BlockRead(F, HiddenActInt, SizeOf(Integer));
   MLP.HiddenActivation := TActivationType(HiddenActInt);
   BlockRead(F, OutputActInt, SizeOf(Integer));
   MLP.OutputActivation := TActivationType(OutputActInt);
   BlockRead(F, MLP.DropoutRate, SizeOf(Double));
   BlockRead(F, MLP.L2Lambda, SizeOf(Double));
   BlockRead(F, MLP.Timestep, SizeOf(Integer));

   for i := 0 to High(MLP.FInputLayer.Neurons) do
   begin
      BlockRead(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.FInputLayer.Neurons[i].Weights) do
         BlockRead(F, MLP.FInputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockRead(F, MLP.FInputLayer.Neurons[i].Bias, SizeOf(Double));
   end;

   for l := 0 to High(MLP.FHiddenLayers) do
   begin
      for i := 0 to High(MLP.FHiddenLayers[l].Neurons) do
      begin
         BlockRead(F, NumInputs, SizeOf(Integer));
         for j := 0 to High(MLP.FHiddenLayers[l].Neurons[i].Weights) do
            BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].Weights[j], SizeOf(Double));
         BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].Bias, SizeOf(Double));
         
         for j := 0 to High(MLP.FHiddenLayers[l].Neurons[i].M) do
         begin
            BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].M[j], SizeOf(Double));
            BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].V[j], SizeOf(Double));
         end;
         BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].MBias, SizeOf(Double));
         BlockRead(F, MLP.FHiddenLayers[l].Neurons[i].VBias, SizeOf(Double));
      end;
   end;

   for i := 0 to High(MLP.FOutputLayer.Neurons) do
   begin
      BlockRead(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.FOutputLayer.Neurons[i].Weights) do
         BlockRead(F, MLP.FOutputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockRead(F, MLP.FOutputLayer.Neurons[i].Bias, SizeOf(Double));
      
      for j := 0 to High(MLP.FOutputLayer.Neurons[i].M) do
      begin
         BlockRead(F, MLP.FOutputLayer.Neurons[i].M[j], SizeOf(Double));
         BlockRead(F, MLP.FOutputLayer.Neurons[i].V[j], SizeOf(Double));
      end;
      BlockRead(F, MLP.FOutputLayer.Neurons[i].MBias, SizeOf(Double));
      BlockRead(F, MLP.FOutputLayer.Neurons[i].VBias, SizeOf(Double));
   end;
   
   CloseFile(F);
   Result := MLP;
end;

function KFoldCrossValidation(var Data: TDataPointArray; NumFolds: Integer; MLP: TMultiLayerPerceptron): Double;
var
   FoldSize, NumSamples, i, j, k: Integer;
   SumAccuracy, CorrectCount: Double;
   TestSet, TrainSet: TDataPointArray;
   Prediction: Darray;
   PredClass, ActualClass: Integer;
begin
   NumSamples := Length(Data);
   if NumSamples < NumFolds then
   begin
      Result := 0;
      Exit;
   end;
   
   FoldSize := NumSamples div NumFolds;

   SumAccuracy := 0;
   for i := 0 to NumFolds - 1 do
   begin
      SetLength(TestSet, FoldSize);
      for j := 0 to FoldSize - 1 do
         TestSet[j] := Data[i * FoldSize + j];

      SetLength(TrainSet, NumSamples - FoldSize);
      k := 0;
      for j := 0 to NumSamples - 1 do
      begin
         if (j >= i * FoldSize) and (j < (i + 1) * FoldSize) then
            Continue;
         TrainSet[k] := Data[j];
         Inc(k);
      end;

      for j := 0 to MLP.MaxIterations - 1 do
         for k := 0 to High(TrainSet) do
            MLP.Train(TrainSet[k].Input, TrainSet[k].Target);

      CorrectCount := 0;
      for j := 0 to High(TestSet) do
      begin
         Prediction := MLP.Predict(TestSet[j].Input);
         PredClass := MaxIndex(Prediction);
         ActualClass := MaxIndex(TestSet[j].Target);
         if PredClass = ActualClass then
            CorrectCount := CorrectCount + 1;
      end;
      
      SumAccuracy := SumAccuracy + CorrectCount;
   end;

   Result := SumAccuracy / NumSamples;
end;

function TrainWithEarlyStopping(MLP: TMultiLayerPerceptron; var Data: TDataPointArray; 
                                 MaxEpochs, BatchSize: Integer): Integer;
var
   Epoch, i, ValSize: Integer;
   BestLoss, ValLoss, InitialLR: Double;
   EpochsWithoutImprovement: Integer;
   TrainData, ValData: TDataPointArray;
   ShuffledData: TDataPointArray;
   Pred: Darray;
begin
   ShuffledData := CloneDataArray(Data);
   ShuffleData(ShuffledData);
   
   ValSize := Length(ShuffledData) div 10;
   if ValSize < 1 then ValSize := 1;
   
   SetLength(ValData, ValSize);
   SetLength(TrainData, Length(ShuffledData) - ValSize);
   
   for i := 0 to ValSize - 1 do
      ValData[i] := ShuffledData[i];
   for i := ValSize to High(ShuffledData) do
      TrainData[i - ValSize] := ShuffledData[i];
   
   BestLoss := 1e30;
   EpochsWithoutImprovement := 0;
   InitialLR := MLP.LearningRate;
   
   for Epoch := 1 to MaxEpochs do
   begin
      if MLP.EnableLRDecay and (Epoch > 1) and (Epoch mod MLP.LRDecayEpochs = 0) then
         MLP.LearningRate := MLP.LearningRate * MLP.LRDecayRate;
      
      MLP.TrainEpoch(TrainData, BatchSize);
      
      if MLP.EnableEarlyStopping then
      begin
         ValLoss := 0;
         for i := 0 to High(ValData) do
         begin
            Pred := MLP.Predict(ValData[i].Input);
            ValLoss := ValLoss + MLP.ComputeLoss(Pred, ValData[i].Target);
         end;
         ValLoss := ValLoss / Length(ValData);
         
         if ValLoss < BestLoss - EPSILON then
         begin
            BestLoss := ValLoss;
            EpochsWithoutImprovement := 0;
         end
         else
            Inc(EpochsWithoutImprovement);
         
         if EpochsWithoutImprovement >= MLP.EarlyStoppingPatience then
         begin
            WriteLn('Early stopping at epoch ', Epoch, ' (validation loss: ', ValLoss:0:6, ')');
            MLP.LearningRate := InitialLR;
            Result := Epoch;
            Exit;
         end;
      end;
   end;
   
   MLP.LearningRate := InitialLR;
   Result := MaxEpochs;
end;

function PrecisionScore(var Data: TDataPointArray; MLP: TMultiLayerPerceptron; ClassIndex: Integer): Double;
var
   TP, FP: Integer;
   i: Integer;
   Prediction: Darray;
   PredClass, ActualClass: Integer;
begin
   TP := 0;
   FP := 0;
   
   for i := 0 to High(Data) do
   begin
      Prediction := MLP.Predict(Data[i].Input);
      PredClass := MaxIndex(Prediction);
      ActualClass := MaxIndex(Data[i].Target);
      
      if PredClass = ClassIndex then
      begin
         if ActualClass = ClassIndex then
            Inc(TP)
         else
            Inc(FP);
      end;
   end;
   
   if TP + FP = 0 then
      Result := 0
   else
      Result := TP / (TP + FP);
end;

function RecallScore(var Data: TDataPointArray; MLP: TMultiLayerPerceptron; ClassIndex: Integer): Double;
var
   TP, FN: Integer;
   i: Integer;
   Prediction: Darray;
   PredClass, ActualClass: Integer;
begin
   TP := 0;
   FN := 0;
   
   for i := 0 to High(Data) do
   begin
      Prediction := MLP.Predict(Data[i].Input);
      PredClass := MaxIndex(Prediction);
      ActualClass := MaxIndex(Data[i].Target);
      
      if ActualClass = ClassIndex then
      begin
         if PredClass = ClassIndex then
            Inc(TP)
         else
            Inc(FN);
      end;
   end;
   
   if TP + FN = 0 then
      Result := 0
   else
      Result := TP / (TP + FN);
end;

function F1Score(Precision, Recall: Double): Double;
begin
   if (Precision + Recall) = 0 then
      Result := 0
   else
      Result := 2 * (Precision * Recall) / (Precision + Recall);
end;

{ TMLPFacade }

constructor TMLPFacade.Create(MLP: TMultiLayerPerceptron);
begin
   inherited Create;
   FMLP := MLP;
end;

function TMLPFacade.GetLayerPtr(LayerIdx: Integer): PLayer;
begin
   if LayerIdx = 0 then
      Result := @FMLP.FInputLayer
   else if LayerIdx = FMLP.GetHiddenLayerCount + 1 then
      Result := @FMLP.FOutputLayer
   else
      Result := @FMLP.FHiddenLayers[LayerIdx - 1];
end;

function TMLPFacade.GetNeuronPtr(LayerIdx, NeuronIdx: Integer): PNeuron;
var
   Layer: PLayer;
begin
   Layer := GetLayerPtr(LayerIdx);
   Result := @Layer^.Neurons[NeuronIdx];
end;

function TMLPFacade.GetPreviousLayerOutputs(LayerIdx: Integer): Darray;
var
   i: Integer;
   PrevLayer: PLayer;
begin
   if LayerIdx <= 0 then
   begin
      SetLength(Result, 0);
      Exit;
   end;
   
   PrevLayer := GetLayerPtr(LayerIdx - 1);
   SetLength(Result, Length(PrevLayer^.Neurons));
   for i := 0 to High(PrevLayer^.Neurons) do
      Result[i] := PrevLayer^.Neurons[i].Output;
end;

{ Neuron output and error }

function TMLPFacade.GetNeuronOutput(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.Output;
end;

procedure TMLPFacade.SetNeuronOutput(LayerIdx, NeuronIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.Output := Value;
end;

function TMLPFacade.GetNeuronError(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.Error;
end;

procedure TMLPFacade.SetNeuronError(LayerIdx, NeuronIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.Error := Value;
end;

{ Neuron bias and weights }

function TMLPFacade.GetNeuronBias(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.Bias;
end;

procedure TMLPFacade.SetNeuronBias(LayerIdx, NeuronIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.Bias := Value;
end;

function TMLPFacade.GetNeuronWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.Weights[WeightIdx];
end;

procedure TMLPFacade.SetNeuronWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.Weights[WeightIdx] := Value;
end;

function TMLPFacade.GetNeuronWeights(LayerIdx, NeuronIdx: Integer): Darray;
var
   Neuron: PNeuron;
   i: Integer;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   SetLength(Result, Length(Neuron^.Weights));
   for i := 0 to High(Neuron^.Weights) do
      Result[i] := Neuron^.Weights[i];
end;

procedure TMLPFacade.SetNeuronWeights(LayerIdx, NeuronIdx: Integer; const Weights: Darray);
var
   Neuron: PNeuron;
   i: Integer;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   for i := 0 to Min(High(Weights), High(Neuron^.Weights)) do
      Neuron^.Weights[i] := Weights[i];
end;

{ Layer queries }

function TMLPFacade.GetLayerSize(LayerIdx: Integer): Integer;
begin
   Result := Length(GetLayerPtr(LayerIdx)^.Neurons);
end;

function TMLPFacade.GetTotalLayers: Integer;
begin
   Result := FMLP.GetHiddenLayerCount + 2;
end;

function TMLPFacade.GetWeightsPerNeuron(LayerIdx, NeuronIdx: Integer): Integer;
begin
   Result := Length(GetNeuronPtr(LayerIdx, NeuronIdx)^.Weights);
end;

{ Gradients }

function TMLPFacade.GetNeuronWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
var
   Neuron: PNeuron;
   PrevOutputs: Darray;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   PrevOutputs := GetPreviousLayerOutputs(LayerIdx);
   if (WeightIdx >= 0) and (WeightIdx < Length(PrevOutputs)) then
      Result := Neuron^.Error * PrevOutputs[WeightIdx]
   else
      Result := 0;
end;

function TMLPFacade.GetNeuronBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.Error;
end;

{ Optimizer state }

function TMLPFacade.GetNeuronOptimizerState(LayerIdx, NeuronIdx: Integer; Param: string): Double;
var
   Neuron: PNeuron;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   if Param = 'MBias' then
      Result := Neuron^.MBias
   else if Param = 'VBias' then
      Result := Neuron^.VBias
   else
      Result := 0;
end;

function TMLPFacade.GetNeuronOptimizerStateArray(LayerIdx, NeuronIdx: Integer; Param: string): Darray;
var
   Neuron: PNeuron;
   i: Integer;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   if Param = 'M' then
   begin
      SetLength(Result, Length(Neuron^.M));
      for i := 0 to High(Neuron^.M) do
         Result[i] := Neuron^.M[i];
   end
   else if Param = 'V' then
   begin
      SetLength(Result, Length(Neuron^.V));
      for i := 0 to High(Neuron^.V) do
         Result[i] := Neuron^.V[i];
   end
   else
      SetLength(Result, 0);
end;

{ Full neuron/layer access }

function TMLPFacade.GetNeuron(LayerIdx, NeuronIdx: Integer): PNeuron;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx);
end;

function TMLPFacade.GetLayer(LayerIdx: Integer): PLayer;
begin
   Result := GetLayerPtr(LayerIdx);
end;

{ Dropout mask }

function TMLPFacade.GetNeuronDropoutMask(LayerIdx, NeuronIdx: Integer): Boolean;
begin
   Result := GetLayerPtr(LayerIdx)^.DropoutMask[NeuronIdx];
end;

procedure TMLPFacade.SetNeuronDropoutMask(LayerIdx, NeuronIdx: Integer; Value: Boolean);
begin
   GetLayerPtr(LayerIdx)^.DropoutMask[NeuronIdx] := Value;
end;

{ Pre-activation values }

function TMLPFacade.GetNeuronPreActivation(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := GetNeuronPtr(LayerIdx, NeuronIdx)^.PreActivation;
end;

procedure TMLPFacade.SetNeuronPreActivation(LayerIdx, NeuronIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.PreActivation := Value;
end;

{ Batch normalization running stats }

function TMLPFacade.GetLayerRunningMean(LayerIdx: Integer): Darray;
var
   Layer: PLayer;
   i: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   SetLength(Result, Length(Layer^.RunningMean));
   for i := 0 to High(Layer^.RunningMean) do
      Result[i] := Layer^.RunningMean[i];
end;

function TMLPFacade.GetLayerRunningVar(LayerIdx: Integer): Darray;
var
   Layer: PLayer;
   i: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   SetLength(Result, Length(Layer^.RunningVar));
   for i := 0 to High(Layer^.RunningVar) do
      Result[i] := Layer^.RunningVar[i];
end;

procedure TMLPFacade.SetLayerRunningMean(LayerIdx: Integer; const Values: Darray);
var
   Layer: PLayer;
   i: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   for i := 0 to Min(High(Values), High(Layer^.RunningMean)) do
      Layer^.RunningMean[i] := Values[i];
end;

procedure TMLPFacade.SetLayerRunningVar(LayerIdx: Integer; const Values: Darray);
var
   Layer: PLayer;
   i: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   for i := 0 to Min(High(Values), High(Layer^.RunningVar)) do
      Layer^.RunningVar[i] := Values[i];
end;

{ Custom neuron attributes }

procedure TMLPFacade.SetNeuronAttribute(LayerIdx, NeuronIdx: Integer; Key: string; Value: string);
var
   Neuron: PNeuron;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   if not Assigned(Neuron^.Attributes) then
      Neuron^.Attributes := TStringList.Create;
   Neuron^.Attributes.Values[Key] := Value;
end;

function TMLPFacade.GetNeuronAttribute(LayerIdx, NeuronIdx: Integer; Key: string): string;
var
   Neuron: PNeuron;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   if Assigned(Neuron^.Attributes) then
      Result := Neuron^.Attributes.Values[Key]
   else
      Result := '';
end;

{ Per-layer learning rate }

function TMLPFacade.GetLayerLearningRate(LayerIdx: Integer): Double;
var
   Layer: PLayer;
begin
   Layer := GetLayerPtr(LayerIdx);
   if Layer^.LearningRate < 0 then
      Result := FMLP.LearningRate
   else
      Result := Layer^.LearningRate;
end;

procedure TMLPFacade.SetLayerLearningRate(LayerIdx: Integer; Value: Double);
begin
   GetLayerPtr(LayerIdx)^.LearningRate := Value;
end;

{ Per-neuron L2 regularization }

function TMLPFacade.GetNeuronL2Lambda(LayerIdx, NeuronIdx: Integer): Double;
var
   Neuron: PNeuron;
begin
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   if Neuron^.L2Lambda < 0 then
      Result := FMLP.L2Lambda
   else
      Result := Neuron^.L2Lambda;
end;

procedure TMLPFacade.SetNeuronL2Lambda(LayerIdx, NeuronIdx: Integer; Value: Double);
begin
   GetNeuronPtr(LayerIdx, NeuronIdx)^.L2Lambda := Value;
end;

{ Connection queries }

function TMLPFacade.GetNeuronIncomingConnections(LayerIdx, NeuronIdx: Integer): TConnectionArray;
var
   Neuron: PNeuron;
   i: Integer;
begin
   if LayerIdx = 0 then
   begin
      SetLength(Result, 0);
      Exit;
   end;
   
   Neuron := GetNeuronPtr(LayerIdx, NeuronIdx);
   SetLength(Result, Length(Neuron^.Weights));
   
   for i := 0 to High(Neuron^.Weights) do
   begin
      Result[i].FromLayerIdx := LayerIdx - 1;
      Result[i].FromNeuronIdx := i;
      Result[i].Weight := Neuron^.Weights[i];
   end;
end;

function TMLPFacade.GetNeuronOutgoingConnections(LayerIdx, NeuronIdx: Integer): TConnectionArray;
var
   NextLayer: PLayer;
   i, Count: Integer;
begin
   if LayerIdx >= GetTotalLayers - 1 then
   begin
      SetLength(Result, 0);
      Exit;
   end;
   
   NextLayer := GetLayerPtr(LayerIdx + 1);
   Count := 0;
   
   for i := 0 to High(NextLayer^.Neurons) do
   begin
      if NeuronIdx < Length(NextLayer^.Neurons[i].Weights) then
      begin
         Inc(Count);
         SetLength(Result, Count);
         Result[Count - 1].FromLayerIdx := LayerIdx + 1;
         Result[Count - 1].FromNeuronIdx := i;
         Result[Count - 1].Weight := NextLayer^.Neurons[i].Weights[NeuronIdx];
      end;
   end;
end;

{ Dynamic topology modification }

function TMLPFacade.AddNeuron(LayerIdx: Integer): Integer;
var
   Layer, NextLayer: PLayer;
   NewIdx, NumInputs, i, j: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   NewIdx := Length(Layer^.Neurons);
   SetLength(Layer^.Neurons, NewIdx + 1);
   SetLength(Layer^.DropoutMask, NewIdx + 1);
   SetLength(Layer^.RunningMean, NewIdx + 1);
   SetLength(Layer^.RunningVar, NewIdx + 1);
   
   if LayerIdx = 0 then
      NumInputs := FMLP.InputSize
   else
      NumInputs := Length(GetLayerPtr(LayerIdx - 1)^.Neurons);
   
   SetLength(Layer^.Neurons[NewIdx].Weights, NumInputs);
   SetLength(Layer^.Neurons[NewIdx].M, NumInputs);
   SetLength(Layer^.Neurons[NewIdx].V, NumInputs);
   
   for i := 0 to NumInputs - 1 do
   begin
      Layer^.Neurons[NewIdx].Weights[i] := (Random * 2 - 1) * 0.1;
      Layer^.Neurons[NewIdx].M[i] := 0;
      Layer^.Neurons[NewIdx].V[i] := 0;
   end;
   
   Layer^.Neurons[NewIdx].Bias := 0;
   Layer^.Neurons[NewIdx].Output := 0;
   Layer^.Neurons[NewIdx].Error := 0;
   Layer^.Neurons[NewIdx].PreActivation := 0;
   Layer^.Neurons[NewIdx].MBias := 0;
   Layer^.Neurons[NewIdx].VBias := 0;
   Layer^.Neurons[NewIdx].L2Lambda := -1;
   Layer^.Neurons[NewIdx].Attributes := nil;
   Layer^.DropoutMask[NewIdx] := True;
   Layer^.RunningMean[NewIdx] := 0;
   Layer^.RunningVar[NewIdx] := 1;
   
   if LayerIdx < GetTotalLayers - 1 then
   begin
      NextLayer := GetLayerPtr(LayerIdx + 1);
      for i := 0 to High(NextLayer^.Neurons) do
      begin
         j := Length(NextLayer^.Neurons[i].Weights);
         SetLength(NextLayer^.Neurons[i].Weights, j + 1);
         SetLength(NextLayer^.Neurons[i].M, j + 1);
         SetLength(NextLayer^.Neurons[i].V, j + 1);
         NextLayer^.Neurons[i].Weights[j] := (Random * 2 - 1) * 0.1;
         NextLayer^.Neurons[i].M[j] := 0;
         NextLayer^.Neurons[i].V[j] := 0;
      end;
   end;
   
   Result := NewIdx;
end;

procedure TMLPFacade.RemoveNeuron(LayerIdx, NeuronIdx: Integer);
var
   Layer, NextLayer: PLayer;
   i, j: Integer;
begin
   Layer := GetLayerPtr(LayerIdx);
   
   if Assigned(Layer^.Neurons[NeuronIdx].Attributes) then
      Layer^.Neurons[NeuronIdx].Attributes.Free;
   
   for i := NeuronIdx to High(Layer^.Neurons) - 1 do
   begin
      Layer^.Neurons[i] := Layer^.Neurons[i + 1];
      Layer^.DropoutMask[i] := Layer^.DropoutMask[i + 1];
      Layer^.RunningMean[i] := Layer^.RunningMean[i + 1];
      Layer^.RunningVar[i] := Layer^.RunningVar[i + 1];
   end;
   
   SetLength(Layer^.Neurons, Length(Layer^.Neurons) - 1);
   SetLength(Layer^.DropoutMask, Length(Layer^.DropoutMask) - 1);
   SetLength(Layer^.RunningMean, Length(Layer^.RunningMean) - 1);
   SetLength(Layer^.RunningVar, Length(Layer^.RunningVar) - 1);
   
   if LayerIdx < GetTotalLayers - 1 then
   begin
      NextLayer := GetLayerPtr(LayerIdx + 1);
      for i := 0 to High(NextLayer^.Neurons) do
      begin
         for j := NeuronIdx to High(NextLayer^.Neurons[i].Weights) - 1 do
         begin
            NextLayer^.Neurons[i].Weights[j] := NextLayer^.Neurons[i].Weights[j + 1];
            NextLayer^.Neurons[i].M[j] := NextLayer^.Neurons[i].M[j + 1];
            NextLayer^.Neurons[i].V[j] := NextLayer^.Neurons[i].V[j + 1];
         end;
         SetLength(NextLayer^.Neurons[i].Weights, Length(NextLayer^.Neurons[i].Weights) - 1);
         SetLength(NextLayer^.Neurons[i].M, Length(NextLayer^.Neurons[i].M) - 1);
         SetLength(NextLayer^.Neurons[i].V, Length(NextLayer^.Neurons[i].V) - 1);
      end;
   end;
end;

function TMLPFacade.AddLayer(Position: Integer; Size: Integer; ActType: TActivationType): Integer;
var
   i, j, NumInputs, NumOutputs: Integer;
   NewLayer: TLayer;
   NextLayer: PLayer;
begin
   if (Position < 1) or (Position > FMLP.GetHiddenLayerCount + 1) then
   begin
      Result := -1;
      Exit;
   end;
   
   if Position = 1 then
      NumInputs := Length(FMLP.FInputLayer.Neurons)
   else
      NumInputs := Length(FMLP.FHiddenLayers[Position - 2].Neurons);
   
   NewLayer.ActivationType := ActType;
   NewLayer.LearningRate := -1;
   SetLength(NewLayer.Neurons, Size);
   SetLength(NewLayer.DropoutMask, Size);
   SetLength(NewLayer.RunningMean, Size);
   SetLength(NewLayer.RunningVar, Size);
   
   for i := 0 to Size - 1 do
   begin
      SetLength(NewLayer.Neurons[i].Weights, NumInputs);
      SetLength(NewLayer.Neurons[i].M, NumInputs);
      SetLength(NewLayer.Neurons[i].V, NumInputs);
      for j := 0 to NumInputs - 1 do
      begin
         NewLayer.Neurons[i].Weights[j] := (Random * 2 - 1) * Sqrt(2.0 / NumInputs);
         NewLayer.Neurons[i].M[j] := 0;
         NewLayer.Neurons[i].V[j] := 0;
      end;
      NewLayer.Neurons[i].Bias := 0;
      NewLayer.Neurons[i].Output := 0;
      NewLayer.Neurons[i].Error := 0;
      NewLayer.Neurons[i].PreActivation := 0;
      NewLayer.Neurons[i].MBias := 0;
      NewLayer.Neurons[i].VBias := 0;
      NewLayer.Neurons[i].L2Lambda := -1;
      NewLayer.Neurons[i].Attributes := nil;
      NewLayer.DropoutMask[i] := True;
      NewLayer.RunningMean[i] := 0;
      NewLayer.RunningVar[i] := 1;
   end;
   
   SetLength(FMLP.FHiddenLayers, Length(FMLP.FHiddenLayers) + 1);
   SetLength(FMLP.FHiddenSizes, Length(FMLP.FHiddenSizes) + 1);
   
   for i := High(FMLP.FHiddenLayers) downto Position do
   begin
      FMLP.FHiddenLayers[i] := FMLP.FHiddenLayers[i - 1];
      FMLP.FHiddenSizes[i] := FMLP.FHiddenSizes[i - 1];
   end;
   
   FMLP.FHiddenLayers[Position - 1] := NewLayer;
   FMLP.FHiddenSizes[Position - 1] := Size;
   
   if Position <= FMLP.GetHiddenLayerCount then
      NextLayer := @FMLP.FHiddenLayers[Position]
   else
      NextLayer := @FMLP.FOutputLayer;
   
   NumOutputs := Size;
   for i := 0 to High(NextLayer^.Neurons) do
   begin
      SetLength(NextLayer^.Neurons[i].Weights, NumOutputs);
      SetLength(NextLayer^.Neurons[i].M, NumOutputs);
      SetLength(NextLayer^.Neurons[i].V, NumOutputs);
      for j := 0 to NumOutputs - 1 do
      begin
         NextLayer^.Neurons[i].Weights[j] := (Random * 2 - 1) * Sqrt(2.0 / NumOutputs);
         NextLayer^.Neurons[i].M[j] := 0;
         NextLayer^.Neurons[i].V[j] := 0;
      end;
   end;
   
   Result := Position;
end;

procedure TMLPFacade.RemoveLayer(LayerIdx: Integer);
var
   i, j, PrevSize: Integer;
   NextLayer, PrevLayer: PLayer;
begin
   if (LayerIdx < 1) or (LayerIdx > FMLP.GetHiddenLayerCount) then
      Exit;
   
   for i := 0 to High(FMLP.FHiddenLayers[LayerIdx - 1].Neurons) do
      if Assigned(FMLP.FHiddenLayers[LayerIdx - 1].Neurons[i].Attributes) then
         FMLP.FHiddenLayers[LayerIdx - 1].Neurons[i].Attributes.Free;
   
   for i := LayerIdx - 1 to High(FMLP.FHiddenLayers) - 1 do
   begin
      FMLP.FHiddenLayers[i] := FMLP.FHiddenLayers[i + 1];
      FMLP.FHiddenSizes[i] := FMLP.FHiddenSizes[i + 1];
   end;
   
   SetLength(FMLP.FHiddenLayers, Length(FMLP.FHiddenLayers) - 1);
   SetLength(FMLP.FHiddenSizes, Length(FMLP.FHiddenSizes) - 1);
   
   if LayerIdx - 1 = 0 then
      PrevLayer := @FMLP.FInputLayer
   else
      PrevLayer := @FMLP.FHiddenLayers[LayerIdx - 2];
   
   if LayerIdx - 1 < Length(FMLP.FHiddenLayers) then
      NextLayer := @FMLP.FHiddenLayers[LayerIdx - 1]
   else
      NextLayer := @FMLP.FOutputLayer;
   
   PrevSize := Length(PrevLayer^.Neurons);
   for i := 0 to High(NextLayer^.Neurons) do
   begin
      SetLength(NextLayer^.Neurons[i].Weights, PrevSize);
      SetLength(NextLayer^.Neurons[i].M, PrevSize);
      SetLength(NextLayer^.Neurons[i].V, PrevSize);
      for j := 0 to PrevSize - 1 do
      begin
         NextLayer^.Neurons[i].Weights[j] := (Random * 2 - 1) * Sqrt(2.0 / PrevSize);
         NextLayer^.Neurons[i].M[j] := 0;
         NextLayer^.Neurons[i].V[j] := 0;
      end;
   end;
end;

{ Histograms }

function TMLPFacade.GetLayerActivationHistogram(LayerIdx: Integer): THistogram;
var
   Layer: PLayer;
   i, BinIdx: Integer;
   MinVal, MaxVal, BinWidth, Val: Double;
begin
   Layer := GetLayerPtr(LayerIdx);
   SetLength(Result, HISTOGRAM_BINS);
   
   if Length(Layer^.Neurons) = 0 then Exit;
   
   MinVal := Layer^.Neurons[0].Output;
   MaxVal := Layer^.Neurons[0].Output;
   
   for i := 1 to High(Layer^.Neurons) do
   begin
      if Layer^.Neurons[i].Output < MinVal then
         MinVal := Layer^.Neurons[i].Output;
      if Layer^.Neurons[i].Output > MaxVal then
         MaxVal := Layer^.Neurons[i].Output;
   end;
   
   if MaxVal = MinVal then
      MaxVal := MinVal + 1;
   
   BinWidth := (MaxVal - MinVal) / HISTOGRAM_BINS;
   
   for i := 0 to HISTOGRAM_BINS - 1 do
   begin
      Result[i].RangeMin := MinVal + i * BinWidth;
      Result[i].RangeMax := MinVal + (i + 1) * BinWidth;
      Result[i].Count := 0;
   end;
   
   for i := 0 to High(Layer^.Neurons) do
   begin
      Val := Layer^.Neurons[i].Output;
      BinIdx := Trunc((Val - MinVal) / BinWidth);
      if BinIdx >= HISTOGRAM_BINS then
         BinIdx := HISTOGRAM_BINS - 1;
      if BinIdx < 0 then
         BinIdx := 0;
      Inc(Result[BinIdx].Count);
   end;
end;

function TMLPFacade.GetLayerGradientHistogram(LayerIdx: Integer): THistogram;
var
   Layer: PLayer;
   i, BinIdx: Integer;
   MinVal, MaxVal, BinWidth, Val: Double;
begin
   Layer := GetLayerPtr(LayerIdx);
   SetLength(Result, HISTOGRAM_BINS);
   
   if Length(Layer^.Neurons) = 0 then Exit;
   
   MinVal := Layer^.Neurons[0].Error;
   MaxVal := Layer^.Neurons[0].Error;
   
   for i := 1 to High(Layer^.Neurons) do
   begin
      if Layer^.Neurons[i].Error < MinVal then
         MinVal := Layer^.Neurons[i].Error;
      if Layer^.Neurons[i].Error > MaxVal then
         MaxVal := Layer^.Neurons[i].Error;
   end;
   
   if MaxVal = MinVal then
      MaxVal := MinVal + 1;
   
   BinWidth := (MaxVal - MinVal) / HISTOGRAM_BINS;
   
   for i := 0 to HISTOGRAM_BINS - 1 do
   begin
      Result[i].RangeMin := MinVal + i * BinWidth;
      Result[i].RangeMax := MinVal + (i + 1) * BinWidth;
      Result[i].Count := 0;
   end;
   
   for i := 0 to High(Layer^.Neurons) do
   begin
      Val := Layer^.Neurons[i].Error;
      BinIdx := Trunc((Val - MinVal) / BinWidth);
      if BinIdx >= HISTOGRAM_BINS then
         BinIdx := HISTOGRAM_BINS - 1;
      if BinIdx < 0 then
         BinIdx := 0;
      Inc(Result[BinIdx].Count);
   end;
end;

end.

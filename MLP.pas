//
// Matthew Abbott 2025
// MLP
//

{$mode objfpc}
{$M+}

program MLPtest;

uses Classes, Math, SysUtils, StrUtils;

const
   EPSILON = 1e-15;
   MODEL_MAGIC = 'MLPBKND01';

type
   TActivationType = (atSigmoid, atTanh, atReLU, atSoftmax);
   TOptimizerType = (otSGD, otAdam, otRMSProp);
   TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp);
   
   Darray = array of Double;
   TDoubleArray = array of Double;
   TIntArray = array of Integer;
   
   TDataPoint = record
      Input: Darray;
      Target: Darray;
   end;
   TDataPointArray = array of TDataPoint;

   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
      M: array of Double;      // First moment (Adam)
      V: array of Double;      // Second moment (Adam/RMSProp)
      MBias: Double;
      VBias: Double;
   end;

   TLayer = record
      Neurons: array of TNeuron;
      ActivationType: TActivationType;
      DropoutMask: array of Boolean;
   end;

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
      procedure UpdateNeuronWeightsSGD(var Neuron: TNeuron; const PrevOutputs: Darray);
      procedure UpdateNeuronWeightsAdam(var Neuron: TNeuron; const PrevOutputs: Darray);
      procedure UpdateNeuronWeightsRMSProp(var Neuron: TNeuron; const PrevOutputs: Darray);
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
      Beta1: Double;           // Adam parameter
      Beta2: Double;           // Adam parameter
      Timestep: Integer;
      EnableLRDecay: Boolean;
      LRDecayRate: Double;
      LRDecayEpochs: Integer;
      EnableEarlyStopping: Boolean;
      EarlyStoppingPatience: Integer;
      
      constructor Create(InputSize: Integer; HiddenSizes: array of Integer; OutputSize: Integer;
                        HiddenAct: TActivationType = atSigmoid; OutputAct: TActivationType = atSigmoid);
      function Predict(Input: Darray): Darray;
      procedure Train(Input, Target: Darray);
      procedure TrainEpoch(var Data: TDataPointArray; BatchSize: Integer);
      function ComputeLoss(const Predicted, Target: Darray): Double;
      procedure SaveMLPModel(const Filename: string);
      procedure Save(const Filename: string);
      
      { JSON serialization methods }
      procedure SaveModelToJSON(const Filename: string);
      procedure LoadModelFromJSON(const Filename: string);
      
      { JSON serialization helper functions }
      function Array1DToJSON(const Arr: Darray): string;
      
      property InputLayer: TLayer read FInputLayer;
      property OutputLayer: TLayer read FOutputLayer;
      function GetHiddenLayer(Index: Integer): TLayer;
      function GetHiddenLayerCount: Integer;
      function GetInputSize: Integer;
      function GetOutputSize: Integer;
      end;

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

function TMultiLayerPerceptron.GetHiddenLayer(Index: Integer): TLayer;
begin
   Result := FHiddenLayers[Index];
end;

function TMultiLayerPerceptron.GetHiddenLayerCount: Integer;
begin
   Result := Length(FHiddenLayers);
end;

function TMultiLayerPerceptron.GetInputSize: Integer;
begin
   Result := FInputSize;
end;

function TMultiLayerPerceptron.GetOutputSize: Integer;
begin
   Result := FOutputSize;
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
   SetLength(Layer.Neurons, NumNeurons);
   SetLength(Layer.DropoutMask, NumNeurons);
   
   for i := 0 to NumNeurons - 1 do
   begin
      Layer.Neurons[i].Weights := InitializeWeights(NumInputs, NumNeurons, ActType);
      Layer.Neurons[i].Bias := 0;
      Layer.DropoutMask[i] := True;
      
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
         FOutputLayer.Neurons[i].Output := ApplyActivation(Sum, OutputActivation);
      end;
   end;
end;

function TMultiLayerPerceptron.ComputeLoss(const Predicted, Target: Darray): Double;
var
   i, j, k: Integer;
   p, L2Sum: Double;
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
            for j := 0 to High(FHiddenLayers[k].Neurons[i].Weights) do
               L2Sum := L2Sum + Sqr(FHiddenLayers[k].Neurons[i].Weights[j]);
      
      for i := 0 to High(FOutputLayer.Neurons) do
         for j := 0 to High(FOutputLayer.Neurons[i].Weights) do
            L2Sum := L2Sum + Sqr(FOutputLayer.Neurons[i].Weights[j]);
      
      Result := Result + (L2Lambda / 2) * L2Sum;
   end;
end;

procedure TMultiLayerPerceptron.BackPropagate(Target: Darray);
var
   i, j, k: Integer;
   ErrorSum: Double;
begin
   for i := 0 to High(FOutputLayer.Neurons) do
   begin
      if OutputActivation = atSoftmax then
         FOutputLayer.Neurons[i].Error := Target[i] - FOutputLayer.Neurons[i].Output
      else
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
         
         ErrorSum := 0;
         if k = High(FHiddenLayers) then
         begin
            for j := 0 to High(FOutputLayer.Neurons) do
               ErrorSum := ErrorSum + FOutputLayer.Neurons[j].Error * FOutputLayer.Neurons[j].Weights[i];
         end
         else
         begin
            for j := 0 to High(FHiddenLayers[k+1].Neurons) do
               ErrorSum := ErrorSum + FHiddenLayers[k+1].Neurons[j].Error * FHiddenLayers[k+1].Neurons[j].Weights[i];
         end;
         
         FHiddenLayers[k].Neurons[i].Error := ApplyActivationDerivative(FHiddenLayers[k].Neurons[i].Output, FHiddenLayers[k].ActivationType) * ErrorSum;
      end;
   end;
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsSGD(var Neuron: TNeuron; const PrevOutputs: Darray);
var
   j: Integer;
   Gradient: Double;
begin
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := Neuron.Error * PrevOutputs[j];
      if L2Lambda > 0 then
         Gradient := Gradient - L2Lambda * Neuron.Weights[j];
      Neuron.Weights[j] := Neuron.Weights[j] + LearningRate * Gradient;
   end;
   Neuron.Bias := Neuron.Bias + LearningRate * Neuron.Error;
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsAdam(var Neuron: TNeuron; const PrevOutputs: Darray);
var
   j: Integer;
   Gradient, Beta1T, Beta2T, MHat, VHat: Double;
   Eps: Double;
begin
   Eps := 1e-8;
   Inc(Timestep);
   Beta1T := Power(Beta1, Timestep);
   Beta2T := Power(Beta2, Timestep);
   
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := -Neuron.Error * PrevOutputs[j];
      if L2Lambda > 0 then
         Gradient := Gradient + L2Lambda * Neuron.Weights[j];
      
      Neuron.M[j] := Beta1 * Neuron.M[j] + (1 - Beta1) * Gradient;
      Neuron.V[j] := Beta2 * Neuron.V[j] + (1 - Beta2) * Gradient * Gradient;
      
      MHat := Neuron.M[j] / (1 - Beta1T);
      VHat := Neuron.V[j] / (1 - Beta2T);
      
      Neuron.Weights[j] := Neuron.Weights[j] - LearningRate * MHat / (Sqrt(VHat) + Eps);
   end;
   
   Gradient := -Neuron.Error;
   Neuron.MBias := Beta1 * Neuron.MBias + (1 - Beta1) * Gradient;
   Neuron.VBias := Beta2 * Neuron.VBias + (1 - Beta2) * Gradient * Gradient;
   
   MHat := Neuron.MBias / (1 - Beta1T);
   VHat := Neuron.VBias / (1 - Beta2T);
   
   Neuron.Bias := Neuron.Bias - LearningRate * MHat / (Sqrt(VHat) + Eps);
end;

procedure TMultiLayerPerceptron.UpdateNeuronWeightsRMSProp(var Neuron: TNeuron; const PrevOutputs: Darray);
var
   j: Integer;
   Gradient: Double;
   Eps, Decay: Double;
begin
   Eps := 1e-8;
   Decay := 0.9;
   
   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := -Neuron.Error * PrevOutputs[j];
      if L2Lambda > 0 then
         Gradient := Gradient + L2Lambda * Neuron.Weights[j];
      
      Neuron.V[j] := Decay * Neuron.V[j] + (1 - Decay) * Gradient * Gradient;
      Neuron.Weights[j] := Neuron.Weights[j] - LearningRate * Gradient / (Sqrt(Neuron.V[j]) + Eps);
   end;
   
   Gradient := -Neuron.Error;
   Neuron.VBias := Decay * Neuron.VBias + (1 - Decay) * Gradient * Gradient;
   Neuron.Bias := Neuron.Bias - LearningRate * Gradient / (Sqrt(Neuron.VBias) + Eps);
end;

procedure TMultiLayerPerceptron.UpdateWeights;
var
   i, j, k: Integer;
   PrevOutputs: Darray;
begin
   for k := 0 to High(FHiddenLayers) do
   begin
      for i := 0 to High(FHiddenLayers[k].Neurons) do
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
         
         case Optimizer of
            otSGD: UpdateNeuronWeightsSGD(FHiddenLayers[k].Neurons[i], PrevOutputs);
            otAdam: UpdateNeuronWeightsAdam(FHiddenLayers[k].Neurons[i], PrevOutputs);
            otRMSProp: UpdateNeuronWeightsRMSProp(FHiddenLayers[k].Neurons[i], PrevOutputs);
         end;
      end;
   end;

   SetLength(PrevOutputs, Length(FHiddenLayers[High(FHiddenLayers)].Neurons));
   for j := 0 to High(FHiddenLayers[High(FHiddenLayers)].Neurons) do
      PrevOutputs[j] := FHiddenLayers[High(FHiddenLayers)].Neurons[j].Output;
   
   for i := 0 to High(FOutputLayer.Neurons) do
   begin
      case Optimizer of
         otSGD: UpdateNeuronWeightsSGD(FOutputLayer.Neurons[i], PrevOutputs);
         otAdam: UpdateNeuronWeightsAdam(FOutputLayer.Neurons[i], PrevOutputs);
         otRMSProp: UpdateNeuronWeightsRMSProp(FOutputLayer.Neurons[i], PrevOutputs);
      end;
   end;
end;

function TMultiLayerPerceptron.Predict(Input: Darray): Darray;
var
   i: Integer;
begin
   FIsTraining := False;
   
   for i := 0 to High(Input) do
      FInputLayer.Neurons[i].Output := Input[i];
   
   FeedForward;
   
   SetLength(Result, Length(FOutputLayer.Neurons));
   for i := 0 to High(FOutputLayer.Neurons) do
      Result[i] := FOutputLayer.Neurons[i].Output;
end;

procedure TMultiLayerPerceptron.Train(Input, Target: Darray);
var
   i: Integer;
begin
   FIsTraining := True;
   
   for i := 0 to High(Input) do
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
begin
   Save(Filename);
end;

procedure TMultiLayerPerceptron.Save(const Filename: string);
var
   F: File;
   NumInputs, LayerCount, i, j, k: Integer;
   OptimizerInt, HiddenActInt, OutputActInt: Integer;
   MagicStr: string;
begin
   AssignFile(F, Filename);
   Rewrite(F, 1);
   
   MagicStr := MODEL_MAGIC;
   BlockWrite(F, MagicStr[1], Length(MagicStr));
   
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
   BlockWrite(F, Beta1, SizeOf(Double));
   BlockWrite(F, Beta2, SizeOf(Double));
   BlockWrite(F, Timestep, SizeOf(Integer));
   BlockWrite(F, EnableLRDecay, SizeOf(Boolean));
   BlockWrite(F, LRDecayRate, SizeOf(Double));
   BlockWrite(F, LRDecayEpochs, SizeOf(Integer));
   BlockWrite(F, EnableEarlyStopping, SizeOf(Boolean));
   BlockWrite(F, EarlyStoppingPatience, SizeOf(Integer));

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
   MagicStr: string;
begin
   AssignFile(F, Filename);
   Reset(F, 1);
   
   SetLength(MagicStr, Length(MODEL_MAGIC));
   BlockRead(F, MagicStr[1], Length(MODEL_MAGIC));
   
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
   BlockRead(F, MLP.Beta1, SizeOf(Double));
   BlockRead(F, MLP.Beta2, SizeOf(Double));
   BlockRead(F, MLP.Timestep, SizeOf(Integer));
   BlockRead(F, MLP.EnableLRDecay, SizeOf(Boolean));
   BlockRead(F, MLP.LRDecayRate, SizeOf(Double));
   BlockRead(F, MLP.LRDecayEpochs, SizeOf(Integer));
   BlockRead(F, MLP.EnableEarlyStopping, SizeOf(Boolean));
   BlockRead(F, MLP.EarlyStoppingPatience, SizeOf(Integer));

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

// ========== Forward Declarations ==========
function ActivationToStr(act: TActivationType): string; forward;
function OptimizerToStr(opt: TOptimizerType): string; forward;
function ParseActivation(const s: string): TActivationType; forward;
function ParseOptimizer(const s: string): TOptimizerType; forward;

function ActivationToStr(act: TActivationType): string;
begin
   case act of
      atSigmoid: Result := 'sigmoid';
      atTanh: Result := 'tanh';
      atReLU: Result := 'relu';
      atSoftmax: Result := 'softmax';
   else
      Result := 'sigmoid';
   end;
end;

function OptimizerToStr(opt: TOptimizerType): string;
begin
   case opt of
      otSGD: Result := 'sgd';
      otAdam: Result := 'adam';
      otRMSProp: Result := 'rmsprop';
   else
      Result := 'sgd';
   end;
end;

function TMultiLayerPerceptron.Array1DToJSON(const Arr: Darray): string;
var
  I: Integer;
begin
  Result := '[';
  for I := 0 to High(Arr) do
  begin
    if I > 0 then Result := Result + ',';
    Result := Result + FloatToStr(Arr[I]);
  end;
  Result := Result + ']';
end;

function Array2DToJSON(const Arr: array of Darray): string;
var
  I: Integer;
  
  function Arr1DToJSON(const A: Darray): string;
  var
    J: Integer;
  begin
    Result := '[';
    for J := 0 to High(A) do
    begin
      if J > 0 then Result := Result + ',';
      Result := Result + FloatToStr(A[J]);
    end;
    Result := Result + ']';
  end;
  
begin
  Result := '[';
  for I := 0 to High(Arr) do
  begin
    if I > 0 then Result := Result + ',';
    Result := Result + Arr1DToJSON(Arr[I]);
  end;
  Result := Result + ']';
end;

procedure TMultiLayerPerceptron.SaveModelToJSON(const Filename: string);
var
  SL: TStringList;
  I, J, K: Integer;
  WeightsArr: array of Darray;
  BiasArr: Darray;
begin
  SL := TStringList.Create;
  try
    SL.Add('{');
    SL.Add('  "model_type": "MLP",');
    SL.Add('  "input_size": ' + IntToStr(FInputSize) + ',');
    SL.Add('  "output_size": ' + IntToStr(FOutputSize) + ',');
    
    { Hidden sizes array }
    SL.Add('  "hidden_sizes": [');
    for I := 0 to High(FHiddenSizes) do
    begin
      if I < High(FHiddenSizes) then
        SL.Add('    ' + IntToStr(FHiddenSizes[I]) + ',')
      else
        SL.Add('    ' + IntToStr(FHiddenSizes[I]));
    end;
    SL.Add('  ],');
    
    { Configuration }
    SL.Add('  "hidden_activation": "' + ActivationToStr(HiddenActivation) + '",');
    SL.Add('  "output_activation": "' + ActivationToStr(OutputActivation) + '",');
    SL.Add('  "optimizer": "' + OptimizerToStr(Optimizer) + '",');
    SL.Add('  "learning_rate": ' + FloatToStr(LearningRate) + ',');
    SL.Add('  "dropout_rate": ' + FloatToStr(DropoutRate) + ',');
    SL.Add('  "l2_lambda": ' + FloatToStr(L2Lambda) + ',');
    SL.Add('  "beta1": ' + FloatToStr(Beta1) + ',');
    SL.Add('  "beta2": ' + FloatToStr(Beta2) + ',');
    SL.Add('  "timestep": ' + IntToStr(Timestep) + ',');
    SL.Add('  "max_iterations": ' + IntToStr(MaxIterations) + ',');
    
    { Hidden layers weights and biases }
    SL.Add('  "hidden_layers": [');
    for K := 0 to High(FHiddenLayers) do
    begin
      SL.Add('    {');
      
      { Collect weights for this layer }
      SetLength(WeightsArr, Length(FHiddenLayers[K].Neurons));
      SetLength(BiasArr, Length(FHiddenLayers[K].Neurons));
      for I := 0 to High(FHiddenLayers[K].Neurons) do
      begin
        SetLength(WeightsArr[I], Length(FHiddenLayers[K].Neurons[I].Weights));
        for J := 0 to High(FHiddenLayers[K].Neurons[I].Weights) do
          WeightsArr[I][J] := FHiddenLayers[K].Neurons[I].Weights[J];
        BiasArr[I] := FHiddenLayers[K].Neurons[I].Bias;
      end;
      
      SL.Add('      "weights": ' + Array2DToJSON(WeightsArr) + ',');
      SL.Add('      "biases": ' + Array1DToJSON(BiasArr));
      
      if K < High(FHiddenLayers) then
        SL.Add('    },')
      else
        SL.Add('    }');
    end;
    SL.Add('  ],');
    
    { Output layer weights and biases }
    SetLength(WeightsArr, Length(FOutputLayer.Neurons));
    SetLength(BiasArr, Length(FOutputLayer.Neurons));
    for I := 0 to High(FOutputLayer.Neurons) do
    begin
      SetLength(WeightsArr[I], Length(FOutputLayer.Neurons[I].Weights));
      for J := 0 to High(FOutputLayer.Neurons[I].Weights) do
        WeightsArr[I][J] := FOutputLayer.Neurons[I].Weights[J];
      BiasArr[I] := FOutputLayer.Neurons[I].Bias;
    end;
    
    SL.Add('  "output_layer": {');
    SL.Add('    "weights": ' + Array2DToJSON(WeightsArr) + ',');
    SL.Add('    "biases": ' + Array1DToJSON(BiasArr));
    SL.Add('  }');
    
    SL.Add('}');
    
    SL.SaveToFile(Filename);
    WriteLn('Model saved to JSON: ', Filename);
  finally
    SL.Free;
  end;
end;

procedure TMultiLayerPerceptron.LoadModelFromJSON(const Filename: string);
var
  SL: TStringList;
  Content: string;
  ValueStr: string;
  TempInputSize, TempOutputSize: Integer;
  TempHiddenSizes: array of Integer;
  TempHiddenAct, TempOutputAct: TActivationType;
  TempOptimizer: TOptimizerType;
  I, J, K, NumInputs: Integer;
  ArrayStart, ArrayEnd, BracketCount, Pos1, Pos2: Integer;
  ArrayStr, NumStr: string;
  tokens: TStringList;
  LayerStart, LayerEnd, WeightsStart, WeightsEnd, BiasesStart, BiasesEnd: Integer;
  LayerStr, WeightsStr, BiasesStr: string;
  WeightRows: TStringList;
  WeightRow: string;
  RowStart, RowEnd: Integer;
  
  function ExtractJSONValue(const json: string; const key: string): string;
  var
    KeyPos, ColonPos, QuotePos1, QuotePos2, StartPos, EndPos: Integer;
  begin
    KeyPos := Pos('"' + key + '"', json);
    if KeyPos > 0 then
    begin
      ColonPos := PosEx(':', json, KeyPos);
      if ColonPos > 0 then
      begin
        StartPos := ColonPos + 1;
        while (StartPos <= Length(json)) and (json[StartPos] in [' ', #9, #10, #13]) do
          Inc(StartPos);
        
        if (StartPos <= Length(json)) and (json[StartPos] = '"') then
        begin
          QuotePos1 := StartPos;
          QuotePos2 := PosEx('"', json, QuotePos1 + 1);
          if QuotePos2 > 0 then
            Result := Copy(json, QuotePos1 + 1, QuotePos2 - QuotePos1 - 1)
          else
            Result := '';
        end
        else
        begin
          EndPos := PosEx(',', json, StartPos);
          if EndPos = 0 then
            EndPos := PosEx('}', json, StartPos);
          if EndPos = 0 then
            EndPos := PosEx(']', json, StartPos);
          Result := Trim(Copy(json, StartPos, EndPos - StartPos));
        end;
      end
      else
        Result := '';
    end
    else
      Result := '';
  end;
  
  function ExtractJSONArray(const json: string; const key: string): string;
  var
    KeyPos, ArrayStart, BracketCount, I: Integer;
  begin
    Result := '';
    KeyPos := Pos('"' + key + '"', json);
    if KeyPos > 0 then
    begin
      ArrayStart := PosEx('[', json, KeyPos);
      if ArrayStart > 0 then
      begin
        BracketCount := 1;
        I := ArrayStart + 1;
        while (I <= Length(json)) and (BracketCount > 0) do
        begin
          if json[I] = '[' then Inc(BracketCount)
          else if json[I] = ']' then Dec(BracketCount);
          Inc(I);
        end;
        Result := Copy(json, ArrayStart, I - ArrayStart);
      end;
    end;
  end;
  
  function ParseIntArray(const arrStr: string): TIntArray;
  var
    tokens: TStringList;
    cleanStr: string;
    I: Integer;
  begin
    cleanStr := StringReplace(arrStr, '[', '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, ']', '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, #10, '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, #13, '', [rfReplaceAll]);
    
    tokens := TStringList.Create;
    try
      tokens.Delimiter := ',';
      tokens.DelimitedText := cleanStr;
      SetLength(Result, 0);
      for I := 0 to tokens.Count - 1 do
      begin
        if Trim(tokens[I]) <> '' then
        begin
          SetLength(Result, Length(Result) + 1);
          Result[High(Result)] := StrToInt(Trim(tokens[I]));
        end;
      end;
    finally
      tokens.Free;
    end;
  end;
  
  function ParseDoubleArray(const arrStr: string): Darray;
  var
    tokens: TStringList;
    cleanStr: string;
    I: Integer;
  begin
    cleanStr := StringReplace(arrStr, '[', '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, ']', '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, #10, '', [rfReplaceAll]);
    cleanStr := StringReplace(cleanStr, #13, '', [rfReplaceAll]);
    
    tokens := TStringList.Create;
    try
      tokens.Delimiter := ',';
      tokens.DelimitedText := cleanStr;
      SetLength(Result, 0);
      for I := 0 to tokens.Count - 1 do
      begin
        if Trim(tokens[I]) <> '' then
        begin
          SetLength(Result, Length(Result) + 1);
          Result[High(Result)] := StrToFloat(Trim(tokens[I]));
        end;
      end;
    finally
      tokens.Free;
    end;
  end;
  
  function ExtractJSONObject(const json: string; const key: string): string;
  var
    KeyPos, ObjStart, BraceCount, I: Integer;
  begin
    Result := '';
    KeyPos := Pos('"' + key + '"', json);
    if KeyPos > 0 then
    begin
      ObjStart := PosEx('{', json, KeyPos);
      if ObjStart > 0 then
      begin
        BraceCount := 1;
        I := ObjStart + 1;
        while (I <= Length(json)) and (BraceCount > 0) do
        begin
          if json[I] = '{' then Inc(BraceCount)
          else if json[I] = '}' then Dec(BraceCount);
          Inc(I);
        end;
        Result := Copy(json, ObjStart, I - ObjStart);
      end;
    end;
  end;

begin
  SL := TStringList.Create;
  try
    SL.LoadFromFile(Filename);
    Content := SL.Text;
    
    { Load basic configuration }
    ValueStr := ExtractJSONValue(Content, 'input_size');
    if ValueStr <> '' then
      TempInputSize := StrToInt(ValueStr)
    else
      TempInputSize := FInputSize;
    
    ValueStr := ExtractJSONValue(Content, 'output_size');
    if ValueStr <> '' then
      TempOutputSize := StrToInt(ValueStr)
    else
      TempOutputSize := FOutputSize;
    
    { Load hidden sizes }
    ArrayStr := ExtractJSONArray(Content, 'hidden_sizes');
    if ArrayStr <> '' then
      TempHiddenSizes := ParseIntArray(ArrayStr)
    else
      TempHiddenSizes := FHiddenSizes;
    
    { Load activations }
    ValueStr := ExtractJSONValue(Content, 'hidden_activation');
    if ValueStr <> '' then
      TempHiddenAct := ParseActivation(ValueStr)
    else
      TempHiddenAct := HiddenActivation;
    
    ValueStr := ExtractJSONValue(Content, 'output_activation');
    if ValueStr <> '' then
      TempOutputAct := ParseActivation(ValueStr)
    else
      TempOutputAct := OutputActivation;
    
    { Load optimizer }
    ValueStr := ExtractJSONValue(Content, 'optimizer');
    if ValueStr <> '' then
      TempOptimizer := ParseOptimizer(ValueStr)
    else
      TempOptimizer := Optimizer;
    
    { Reinitialize the model with correct architecture }
    FInputSize := TempInputSize;
    FOutputSize := TempOutputSize;
    SetLength(FHiddenSizes, Length(TempHiddenSizes));
    for I := 0 to High(TempHiddenSizes) do
      FHiddenSizes[I] := TempHiddenSizes[I];
    HiddenActivation := TempHiddenAct;
    OutputActivation := TempOutputAct;
    Optimizer := TempOptimizer;
    
    { Reinitialize layers with correct sizes }
    SetLength(FHiddenLayers, Length(FHiddenSizes));
    InitializeLayer(FInputLayer, FInputSize + 1, FInputSize, atSigmoid);
    
    NumInputs := FInputSize;
    for I := 0 to High(FHiddenSizes) do
    begin
      InitializeLayer(FHiddenLayers[I], FHiddenSizes[I] + 1, NumInputs + 1, HiddenActivation);
      NumInputs := FHiddenSizes[I];
    end;
    InitializeLayer(FOutputLayer, FOutputSize, NumInputs + 1, OutputActivation);
    
    { Load other parameters }
    ValueStr := ExtractJSONValue(Content, 'learning_rate');
    if ValueStr <> '' then
      LearningRate := StrToFloat(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'dropout_rate');
    if ValueStr <> '' then
      DropoutRate := StrToFloat(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'l2_lambda');
    if ValueStr <> '' then
      L2Lambda := StrToFloat(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'beta1');
    if ValueStr <> '' then
      Beta1 := StrToFloat(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'beta2');
    if ValueStr <> '' then
      Beta2 := StrToFloat(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'timestep');
    if ValueStr <> '' then
      Timestep := StrToInt(ValueStr);
    
    ValueStr := ExtractJSONValue(Content, 'max_iterations');
    if ValueStr <> '' then
      MaxIterations := StrToInt(ValueStr);
    
    { Load hidden layer weights }
    ArrayStr := ExtractJSONArray(Content, 'hidden_layers');
    if ArrayStr <> '' then
    begin
      for K := 0 to High(FHiddenLayers) do
      begin
        { Find the K-th layer object }
        Pos1 := 1;
        for I := 0 to K do
        begin
          Pos1 := PosEx('{', ArrayStr, Pos1);
          if (I < K) and (Pos1 > 0) then
            Pos1 := Pos1 + 1;
        end;
        
        if Pos1 > 0 then
        begin
          BracketCount := 1;
          Pos2 := Pos1 + 1;
          while (Pos2 <= Length(ArrayStr)) and (BracketCount > 0) do
          begin
            if ArrayStr[Pos2] = '{' then Inc(BracketCount)
            else if ArrayStr[Pos2] = '}' then Dec(BracketCount);
            Inc(Pos2);
          end;
          LayerStr := Copy(ArrayStr, Pos1, Pos2 - Pos1);
          
          { Extract weights array }
          WeightsStr := ExtractJSONArray(LayerStr, 'weights');
          BiasesStr := ExtractJSONArray(LayerStr, 'biases');
          
          if (WeightsStr <> '') and (BiasesStr <> '') then
          begin
            { Parse biases }
            BiasArr := ParseDoubleArray(BiasesStr);
            for I := 0 to Min(High(FHiddenLayers[K].Neurons), High(BiasArr)) do
              FHiddenLayers[K].Neurons[I].Bias := BiasArr[I];
            
            { Parse weights - need to parse 2D array }
            { Find each row [....] inside the outer brackets }
            RowStart := 2;
            I := 0;
            while (RowStart < Length(WeightsStr)) and (I <= High(FHiddenLayers[K].Neurons)) do
            begin
              RowStart := PosEx('[', WeightsStr, RowStart);
              if RowStart = 0 then Break;
              
              BracketCount := 1;
              RowEnd := RowStart + 1;
              while (RowEnd <= Length(WeightsStr)) and (BracketCount > 0) do
              begin
                if WeightsStr[RowEnd] = '[' then Inc(BracketCount)
                else if WeightsStr[RowEnd] = ']' then Dec(BracketCount);
                Inc(RowEnd);
              end;
              
              WeightRow := Copy(WeightsStr, RowStart, RowEnd - RowStart);
              WeightsArr := ParseDoubleArray(WeightRow);
              
              for J := 0 to Min(High(FHiddenLayers[K].Neurons[I].Weights), High(WeightsArr)) do
                FHiddenLayers[K].Neurons[I].Weights[J] := WeightsArr[J];
              
              RowStart := RowEnd;
              Inc(I);
            end;
          end;
        end;
      end;
    end;
    
    { Load output layer weights }
    LayerStr := ExtractJSONObject(Content, 'output_layer');
    if LayerStr <> '' then
    begin
      WeightsStr := ExtractJSONArray(LayerStr, 'weights');
      BiasesStr := ExtractJSONArray(LayerStr, 'biases');
      
      if (WeightsStr <> '') and (BiasesStr <> '') then
      begin
        { Parse biases }
        BiasArr := ParseDoubleArray(BiasesStr);
        for I := 0 to Min(High(FOutputLayer.Neurons), High(BiasArr)) do
          FOutputLayer.Neurons[I].Bias := BiasArr[I];
        
        { Parse weights }
        RowStart := 2;
        I := 0;
        while (RowStart < Length(WeightsStr)) and (I <= High(FOutputLayer.Neurons)) do
        begin
          RowStart := PosEx('[', WeightsStr, RowStart);
          if RowStart = 0 then Break;
          
          BracketCount := 1;
          RowEnd := RowStart + 1;
          while (RowEnd <= Length(WeightsStr)) and (BracketCount > 0) do
          begin
            if WeightsStr[RowEnd] = '[' then Inc(BracketCount)
            else if WeightsStr[RowEnd] = ']' then Dec(BracketCount);
            Inc(RowEnd);
          end;
          
          WeightRow := Copy(WeightsStr, RowStart, RowEnd - RowStart);
          WeightsArr := ParseDoubleArray(WeightRow);
          
          for J := 0 to Min(High(FOutputLayer.Neurons[I].Weights), High(WeightsArr)) do
            FOutputLayer.Neurons[I].Weights[J] := WeightsArr[J];
          
          RowStart := RowEnd;
          Inc(I);
        end;
      end;
    end;
    
    WriteLn('Model loaded from JSON: ', Filename);
    WriteLn('  Input size: ', FInputSize);
    Write('  Hidden sizes: ');
    for I := 0 to High(FHiddenSizes) do
    begin
      if I > 0 then Write(',');
      Write(FHiddenSizes[I]);
    end;
    WriteLn;
    WriteLn('  Output size: ', FOutputSize);
    WriteLn('  Hidden activation: ', ActivationToStr(HiddenActivation));
    WriteLn('  Output activation: ', ActivationToStr(OutputActivation));
    WriteLn('  Optimizer: ', OptimizerToStr(Optimizer));
    WriteLn('  Learning rate: ', LearningRate:0:6);
  finally
    SL.Free;
  end;
end;

procedure PrintUsage;
begin
  WriteLn('MLP - Command-line Multi-Layer Perceptron');
  WriteLn;
  WriteLn('Commands:');
  WriteLn('  create   Create a new MLP model and save to JSON');
  WriteLn('  train    Train an existing model with data from JSON');
  WriteLn('  predict  Make predictions with a trained model from JSON');
  WriteLn('  info     Display model information from JSON');
  WriteLn('  help     Show this help message');
  WriteLn;
  WriteLn('Create Options:');
  WriteLn('  --input=N              Input layer size (required)');
  WriteLn('  --hidden=N,N,...       Hidden layer sizes (required)');
  WriteLn('  --output=N             Output layer size (required)');
  WriteLn('  --save=FILE.json       Save model to JSON file (required)');
  WriteLn('  --lr=VALUE             Learning rate (default: 0.1)');
  WriteLn('  --optimizer=TYPE       sgd|adam|rmsprop (default: sgd)');
  WriteLn('  --hidden-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)');
  WriteLn('  --output-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)');
  WriteLn('  --dropout=VALUE        Dropout rate 0-1 (default: 0)');
  WriteLn('  --l2=VALUE             L2 regularization (default: 0)');
  WriteLn('  --beta1=VALUE          Adam beta1 (default: 0.9)');
  WriteLn('  --beta2=VALUE          Adam beta2 (default: 0.999)');
  WriteLn;
  WriteLn('Train Options:');
  WriteLn('  --model=FILE.json      Load model from JSON file (required)');
  WriteLn('  --data=FILE.csv        Training data CSV file (required)');
  WriteLn('  --save=FILE.json       Save trained model to JSON (required)');
  WriteLn('  --epochs=N             Number of training epochs (default: 100)');
  WriteLn('  --batch=N              Batch size (default: 1)');
  WriteLn('  --lr=VALUE             Override learning rate');
  WriteLn('  --lr-decay             Enable learning rate decay');
  WriteLn('  --lr-decay-rate=VALUE  LR decay rate (default: 0.95)');
  WriteLn('  --lr-decay-epochs=N    Epochs between decay (default: 10)');
  WriteLn('  --early-stop           Enable early stopping');
  WriteLn('  --patience=N           Early stopping patience (default: 10)');
  WriteLn('  --normalize            Normalize input data');
  WriteLn('  --verbose              Print training progress');
  WriteLn;
  WriteLn('Predict Options:');
  WriteLn('  --model=FILE.json      Load model from JSON file (required)');
  WriteLn('  --input=v1,v2,...      Input values as CSV (required)');
  WriteLn;
  WriteLn('Info Options:');
  WriteLn('  --model=FILE.json      Load model from JSON file (required)');
  WriteLn;
  WriteLn('Examples:');
  WriteLn('  mlp create --input=2 --hidden=8 --output=1 --save=xor.json');
  WriteLn('  mlp train --model=xor.json --data=xor.csv --epochs=1000 --save=xor_trained.json');
  WriteLn('  mlp predict --model=xor_trained.json --input=1,0');
  WriteLn('  mlp info --model=xor_trained.json');
end;

procedure ParseIntArrayHelper(const s: string; out result: TIntArray);
var
   tokens: TStringList;
   i: Integer;
   temp: TIntArray;
begin
   tokens := TStringList.Create;
   try
      tokens.Delimiter := ',';
      tokens.DelimitedText := s;
      SetLength(temp, tokens.Count);
      for i := 0 to tokens.Count - 1 do
         temp[i] := StrToInt(Trim(tokens[i]));
      result := temp;
   finally
      tokens.Free;
   end;
end;

procedure ParseDoubleArrayHelper(const s: string; out result: TDoubleArray);
var
   tokens: TStringList;
   i: Integer;
   temp: TDoubleArray;
begin
   tokens := TStringList.Create;
   try
      tokens.Delimiter := ',';
      tokens.DelimitedText := s;
      SetLength(temp, tokens.Count);
      for i := 0 to tokens.Count - 1 do
         temp[i] := StrToFloat(Trim(tokens[i]));
      result := temp;
   finally
      tokens.Free;
   end;
end;

function ParseActivation(const s: string): TActivationType;
begin
   if LowerCase(s) = 'tanh' then
      Result := atTanh
   else if LowerCase(s) = 'relu' then
      Result := atReLU
   else if LowerCase(s) = 'softmax' then
      Result := atSoftmax
   else
      Result := atSigmoid;
end;

function ParseOptimizer(const s: string): TOptimizerType;
begin
   if LowerCase(s) = 'adam' then
      Result := otAdam
   else if LowerCase(s) = 'rmsprop' then
      Result := otRMSProp
   else
      Result := otSGD;
end;

procedure LoadDataCSV(const filename: string; inputSize, outputSize: Integer; var data: TDataPointArray);
var
   f: TextFile;
   line: string;
   values: array of Double;
   i: Integer;
   count: Integer;
begin
   SetLength(data, 0);
   count := 0;
   
   AssignFile(f, filename);
   try
      Reset(f);
      while not Eof(f) do
      begin
         ReadLn(f, line);
         if line = '' then Continue;
         
         ParseDoubleArrayHelper(line, values);
         if Length(values) < inputSize + outputSize then Continue;
         
         SetLength(data, count + 1);
         SetLength(data[count].Input, inputSize);
         SetLength(data[count].Target, outputSize);
         
         for i := 0 to inputSize - 1 do
            data[count].Input[i] := values[i];
         for i := 0 to outputSize - 1 do
            data[count].Target[i] := values[inputSize + i];
         
         Inc(count);
      end;
   finally
      CloseFile(f);
   end;
end;

var
   Command: TCommand;
   CmdStr: string;
   i: Integer;
   arg, key, value: string;
   eqPos: Integer;
   
   inputSize, outputSize, epochs, batchSize: Integer;
   hiddenSizes: array of Integer;
   learningRate, dropoutRate, l2Lambda, beta1, beta2: Double;
   lrDecayRate: Double;
   lrDecayEpochs, patience: Integer;
   hiddenAct, outputAct: TActivationType;
   optimizer: TOptimizerType;
   modelFile, saveFile, dataFile: string;
   inputValues: array of Double;
   lrDecay, earlyStop, normalize, verbose: Boolean;
   
   MLP: TMultiLayerPerceptron;
   data: TDataPointArray;
   output: Darray;
   j: Integer;
   loss: Double;
   epoch: Integer;
   maxIdx: Integer;
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
   
   // Initialize defaults
   inputSize := 0;
   outputSize := 0;
   SetLength(hiddenSizes, 0);
   learningRate := 0.1;
   dropoutRate := 0;
   l2Lambda := 0;
   beta1 := 0.9;
   beta2 := 0.999;
   epochs := 100;
   batchSize := 1;
   lrDecay := False;
   lrDecayRate := 0.95;
   lrDecayEpochs := 10;
   earlyStop := False;
   patience := 10;
   normalize := False;
   verbose := False;
   hiddenAct := atSigmoid;
   outputAct := atSigmoid;
   optimizer := otSGD;
   modelFile := '';
   saveFile := '';
   dataFile := '';
   SetLength(inputValues, 0);
   
   // Parse arguments
   for i := 2 to ParamCount do
   begin
      arg := ParamStr(i);
      
      if arg = '--lr-decay' then
         lrDecay := True
      else if arg = '--early-stop' then
         earlyStop := True
      else if arg = '--normalize' then
         normalize := True
      else if arg = '--verbose' then
         verbose := True
      else
      begin
         eqPos := Pos('=', arg);
         if eqPos = 0 then
         begin
            WriteLn('Invalid argument: ', arg);
            Continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--input' then
         begin
            if Command = cmdPredict then
               ParseDoubleArrayHelper(value, inputValues)
            else
               inputSize := StrToInt(value);
         end
         else if key = '--hidden' then
            ParseIntArrayHelper(value, hiddenSizes)
         else if key = '--output' then
            outputSize := StrToInt(value)
         else if key = '--save' then
            saveFile := value
         else if key = '--model' then
            modelFile := value
         else if key = '--data' then
            dataFile := value
         else if key = '--lr' then
            learningRate := StrToFloat(value)
         else if key = '--optimizer' then
            optimizer := ParseOptimizer(value)
         else if key = '--hidden-act' then
            hiddenAct := ParseActivation(value)
         else if key = '--output-act' then
            outputAct := ParseActivation(value)
         else if key = '--dropout' then
            dropoutRate := StrToFloat(value)
         else if key = '--l2' then
            l2Lambda := StrToFloat(value)
         else if key = '--beta1' then
            beta1 := StrToFloat(value)
         else if key = '--beta2' then
            beta2 := StrToFloat(value)
         else if key = '--epochs' then
            epochs := StrToInt(value)
         else if key = '--batch' then
            batchSize := StrToInt(value)
         else if key = '--lr-decay-rate' then
            lrDecayRate := StrToFloat(value)
         else if key = '--lr-decay-epochs' then
            lrDecayEpochs := StrToInt(value)
         else if key = '--patience' then
            patience := StrToInt(value)
         else
            WriteLn('Unknown option: ', key);
      end;
   end;
   
   // Execute command
   if Command = cmdCreate then
   begin
      if inputSize <= 0 then begin WriteLn('Error: --input is required'); Exit; end;
      if Length(hiddenSizes) = 0 then begin WriteLn('Error: --hidden is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;
      if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
      
      MLP := TMultiLayerPerceptron.Create(inputSize, hiddenSizes, outputSize, hiddenAct, outputAct);
      MLP.LearningRate := learningRate;
      MLP.Optimizer := optimizer;
      MLP.DropoutRate := dropoutRate;
      MLP.L2Lambda := l2Lambda;
      MLP.Beta1 := beta1;
      MLP.Beta2 := beta2;
      
      WriteLn('Created MLP model:');
      WriteLn('  Input size: ', inputSize);
      Write('  Hidden sizes: ');
      for i := 0 to High(hiddenSizes) do
      begin
         if i > 0 then Write(',');
         Write(hiddenSizes[i]);
      end;
      WriteLn;
      WriteLn('  Output size: ', outputSize);
      WriteLn('  Hidden activation: ', ActivationToStr(hiddenAct));
      WriteLn('  Output activation: ', ActivationToStr(outputAct));
      WriteLn('  Optimizer: ', OptimizerToStr(optimizer));
      WriteLn('  Learning rate: ', learningRate:0:6);
      WriteLn('  Dropout rate: ', dropoutRate:0:4);
      WriteLn('  L2 lambda: ', l2Lambda:0:6);
      
      { Save model to JSON }
      MLP.SaveModelToJSON(saveFile);

      MLP.Free;
   end
   else if Command = cmdTrain then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
      WriteLn('Loading model from JSON: ' + modelFile);
      MLP := TMultiLayerPerceptron.Create(1, [1], 1, atSigmoid, atSigmoid);
      MLP.LoadModelFromJSON(modelFile);
      WriteLn('Model loaded successfully. Training functionality not yet implemented.');
      MLP.Free;
   end
   else if Command = cmdPredict then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      WriteLn('Loading model from JSON: ' + modelFile);
      MLP := TMultiLayerPerceptron.Create(1, [1], 1, atSigmoid, atSigmoid);
      MLP.LoadModelFromJSON(modelFile);
      WriteLn('Model loaded successfully. Prediction functionality not yet implemented.');
      MLP.Free;
   end
   else if Command = cmdInfo then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      WriteLn('Loading model from JSON: ' + modelFile);
      MLP := TMultiLayerPerceptron.Create(1, [1], 1, atSigmoid, atSigmoid);
      MLP.LoadModelFromJSON(modelFile);
      WriteLn('Model information displayed above.');
      MLP.Free;
   end;
end.

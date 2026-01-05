(*
 * MIT License
 * 
 * Copyright (c) 2025 Matthew Abbott
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software. 
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *)

{$mode objfpc}
{$M+}

program MLPtest;

uses Classes, Math, SysUtils;

const
   EPSILON = 1e-15;

type
   TActivationType = (atSigmoid, atTanh, atReLU, atSoftmax);
   TOptimizerType = (otSGD, otAdam, otRMSProp);

   Darray = array of Double;

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

      property InputLayer: TLayer read FInputLayer;
      property OutputLayer: TLayer read FOutputLayer;
      function GetHiddenLayer(Index: Integer): TLayer;
      function GetHiddenLayerCount: Integer;
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

      Result := Result + 0.5 * L2Lambda * L2Sum;
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
   Gradient, MHat, VHat: Double;
   Eps: Double;
begin
   Eps := 1e-8;

   for j := 0 to High(Neuron.Weights) do
   begin
      Gradient := -Neuron.Error * PrevOutputs[j];
      if L2Lambda > 0 then
         Gradient := Gradient + L2Lambda * Neuron.Weights[j];

      Neuron.M[j] := Beta1 * Neuron.M[j] + (1 - Beta1) * Gradient;
      Neuron.V[j] := Beta2 * Neuron.V[j] + (1 - Beta2) * Gradient * Gradient;

      MHat := Neuron.M[j] / (1 - Power(Beta1, Timestep));
      VHat := Neuron.V[j] / (1 - Power(Beta2, Timestep));

      Neuron.Weights[j] := Neuron.Weights[j] - LearningRate * MHat / (Sqrt(VHat) + Eps);
   end;

   Gradient := -Neuron.Error;
   Neuron.MBias := Beta1 * Neuron.MBias + (1 - Beta1) * Gradient;
   Neuron.VBias := Beta2 * Neuron.VBias + (1 - Beta2) * Gradient * Gradient;
   MHat := Neuron.MBias / (1 - Power(Beta1, Timestep));
   VHat := Neuron.VBias / (1 - Power(Beta2, Timestep));
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
   Inc(Timestep);

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

      for i := 0 to High(FHiddenLayers[k].Neurons) do
      begin
         case Optimizer of
            otSGD: UpdateNeuronWeightsSGD(FHiddenLayers[k].Neurons[i], PrevOutputs);
            otAdam: UpdateNeuronWeightsAdam(FHiddenLayers[k].Neurons[i], PrevOutputs);
            otRMSProp: UpdateNeuronWeightsRMSProp(FHiddenLayers[k].Neurons[i], PrevOutputs);
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

var
   Data: TDataPointArray;
   NumFolds, i, c: Integer;
   MLP, MLP2: TMultiLayerPerceptron;
   Accuracy, AvgPrecision, AvgRecall, AvgF1: Double;
   ClassPrecision, ClassRecall, ClassF1: Double;
   InputSize: Integer = 4;
   OutputSize: Integer = 3;
begin
   Randomize;

   SetLength(Data, 7500);
   for i := 0 to 2499 do
   begin
      SetLength(Data[i].Input, 4);
      SetLength(Data[i].Target, 3);
      Data[i].Input[0] := Random * 0.5;
      Data[i].Input[1] := Random * 0.5;
      Data[i].Input[2] := Random * 0.5;
      Data[i].Input[3] := Random * 0.5;
      Data[i].Target[0] := 1;
      Data[i].Target[1] := 0;
      Data[i].Target[2] := 0;
   end;
   for i := 2500 to 4999 do
   begin
      SetLength(Data[i].Input, 4);
      SetLength(Data[i].Target, 3);
      Data[i].Input[0] := 0.5 + Random * 0.5;
      Data[i].Input[1] := 0.5 + Random * 0.5;
      Data[i].Input[2] := 0.5 + Random * 0.5;
      Data[i].Input[3] := 0.5 + Random * 0.5;
      Data[i].Target[0] := 0;
      Data[i].Target[1] := 1;
      Data[i].Target[2] := 0;
   end;
   for i := 5000 to 7499 do
   begin
      SetLength(Data[i].Input, 4);
      SetLength(Data[i].Target, 3);
      Data[i].Input[0] := Random * 0.5;
      Data[i].Input[1] := 0.5 + Random * 0.5;
      Data[i].Input[2] := Random * 0.5;
      Data[i].Input[3] := 0.5 + Random * 0.5;
      Data[i].Target[0] := 0;
      Data[i].Target[1] := 0;
      Data[i].Target[2] := 1;
   end;

   WriteLn('=== Enhanced MLP Test ===');
   WriteLn;

   CheckDataQuality(Data);
   WriteLn;

   MLP := TMultiLayerPerceptron.Create(InputSize, [8, 8, 8], OutputSize, atSigmoid, atSoftmax);
   MLP.MaxIterations := 30;
   MLP.Optimizer := otAdam;
   MLP.LearningRate := 0.001;
   MLP.DropoutRate := 0.1;
   MLP.L2Lambda := 0.0001;
   MLP.EnableLRDecay := True;
   MLP.LRDecayRate := 0.95;
   MLP.LRDecayEpochs := 10;
   MLP.EnableEarlyStopping := True;
   MLP.EarlyStoppingPatience := 5;

   WriteLn('Configuration:');
   WriteLn('  Optimizer: Adam');
   WriteLn('  Hidden Activation: Sigmoid');
   WriteLn('  Output Activation: Softmax');
   WriteLn('  Dropout Rate: ', MLP.DropoutRate:0:2);
   WriteLn('  L2 Lambda: ', MLP.L2Lambda:0:6);
   WriteLn('  Learning Rate: ', MLP.LearningRate:0:4);
   WriteLn('  LR Decay: ', MLP.LRDecayRate:0:2, ' every ', MLP.LRDecayEpochs, ' epochs');
   WriteLn;

   NumFolds := 10;

   WriteLn('Training with early stopping...');
   TrainWithEarlyStopping(MLP, Data, 100, 32);
   WriteLn;

   Accuracy := KFoldCrossValidation(Data, NumFolds, MLP);

   AvgPrecision := 0;
   AvgRecall := 0;
   AvgF1 := 0;

   WriteLn('Per-class metrics:');
   for c := 0 to OutputSize - 1 do
   begin
      ClassPrecision := PrecisionScore(Data, MLP, c);
      ClassRecall := RecallScore(Data, MLP, c);
      ClassF1 := F1Score(ClassPrecision, ClassRecall);
      WriteLn('  Class ', c, ': Precision=', ClassPrecision:0:3, ' Recall=', ClassRecall:0:3, ' F1=', ClassF1:0:3);
      AvgPrecision := AvgPrecision + ClassPrecision;
      AvgRecall := AvgRecall + ClassRecall;
      AvgF1 := AvgF1 + ClassF1;
   end;

   AvgPrecision := AvgPrecision / OutputSize;
   AvgRecall := AvgRecall / OutputSize;
   AvgF1 := AvgF1 / OutputSize;

   WriteLn;
   WriteLn('Overall Results:');
   WriteLn('  Accuracy: ', Accuracy:0:3);
   WriteLn('  Avg Precision: ', AvgPrecision:0:3);
   WriteLn('  Avg Recall: ', AvgRecall:0:3);
   WriteLn('  Avg F1 Score: ', AvgF1:0:3);

   WriteLn;
   WriteLn('Saving model...');
   MLP.SaveMLPModel('TestSaveMLP_Enhanced.bin');

   WriteLn('Loading model...');
   MLP2 := LoadMLPModel('TestSaveMLP_Enhanced.bin');
   MLP2.MaxIterations := 30;

   Accuracy := KFoldCrossValidation(Data, NumFolds, MLP2);
   WriteLn('Loaded model accuracy: ', Accuracy:0:3);

   MLP.Free;
   MLP2.Free;
end.

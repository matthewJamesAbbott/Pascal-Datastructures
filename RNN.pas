//
// Matthew Abbott  2025
//

{$mode objfpc}
{$M+}

program MLP_RNNtest;

uses Classes, Math, SysUtils;

type
   Darray = array of Double;
   TDarray2D = array of Darray;
   TDataPoint = record
      Input: Darray;
      Target: Darray;
   end;

   TNeuron = record
      Weights: array of Double;         // Input weights
      RecurrentWeight: Double;          // Weight for recurrent connection
      Bias: Double;
      Output: Double;
      PrevOutput: Double;               // Store previous output for recurrence
      Error: Double;
   end;

   TLayer = record
      Neurons: array of TNeuron;
   end;

   TMultiLayerRNN = class
   private
      LearningRate: Double;
      MaxIterations: integer;
      InputLayer: TLayer;
      HiddenLayers: array of TLayer;
      OutputLayer: TLayer;
      procedure InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer; Recurrent: Boolean = false);
      procedure FeedForwardStep(const Input: Darray);
      procedure BackPropagate(Target: Darray);
      procedure UpdateWeights;
      procedure StepHiddenState;
      procedure ResetHiddenStates;
   public
      constructor Create(InputSize: Integer;HiddenSizes: Array of Integer;  OutputSize: Integer);
      function PredictSequence(const Sequence: TDarray2D): TDarray2D;
      procedure TrainSequence(const Inputs, Targets: TDarray2D);
   end;

var
   InputSize: integer = 4;
   HiddenSizesG: array of Integer;
   OutputSize: integer = 3;

constructor TMultiLayerRNN.Create(InputSize: Integer; HiddenSizes: array of Integer; OutputSize: Integer);
var
   i: integer;
   NumInputs,NumNeurons: integer;
begin
   LearningRate := 0.1;
   setLength(HiddenLayers, length(HiddenSizes));
   setLength(HiddenSizesG, length(HiddenSizes));

   for i := 0 to High(HiddenSizes) do
      HiddenSizesG[i] := HiddenSizes[i];

   // Initialize input layer with bias neuron and input neurons
   InitializeLayer(InputLayer, InputSize, InputSize, False);
   NumInputs := InputSize;
   // Hidden layers have recurrent connections:
   for i := 0 to High(HiddenSizes)  do
   begin
      NumNeurons := HiddenSizes[i];
      InitializeLayer(HiddenLayers[i], NumNeurons, NumInputs, True);
      NumInputs := NumNeurons;
   end;
   InitializeLayer(OutputLayer, OutputSize, NumInputs, False);
end;

procedure TMultiLayerRNN.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer; Recurrent: Boolean = false);
var
   i, j: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   for i := 0 to NumNeurons - 1 do
   begin
     SetLength(Layer.Neurons[i].Weights, NumInputs);
     for j := 0 to NumInputs - 1 do
       Layer.Neurons[i].Weights[j] := Random - 0.5;
     if Recurrent then
       Layer.Neurons[i].RecurrentWeight := Random - 0.5
     else
       Layer.Neurons[i].RecurrentWeight := 0.0;
     Layer.Neurons[i].Bias := Random - 0.5;
     Layer.Neurons[i].PrevOutput := 0.0;
   end;
end;

procedure TMultiLayerRNN.StepHiddenState;
var
  k, i: Integer;
begin
  for k := 0 to High(HiddenLayers) do
    for i := 0 to High(HiddenLayers[k].Neurons) do
      HiddenLayers[k].Neurons[i].PrevOutput := HiddenLayers[k].Neurons[i].Output;
end;

procedure TMultiLayerRNN.ResetHiddenStates;
var
  k, i: Integer;
begin
  for k := 0 to High(HiddenLayers) do
    for i := 0 to High(HiddenLayers[k].Neurons) do
      HiddenLayers[k].Neurons[i].PrevOutput := 0.0;
end;

procedure TMultiLayerRNN.FeedForwardStep(const Input: Darray);
var
   i, j, k: Integer;
   Sum: Double;
begin
   // Set input layer outputs
   for i := 0 to High(InputLayer.Neurons) do
      InputLayer.Neurons[i].Output := Input[i];

   // Hidden layers with recurrence
   for k := 0 to High(HiddenLayers) do
   begin
      for i := 0 to High(HiddenLayers[k].Neurons) do
      begin
         Sum := HiddenLayers[k].Neurons[i].Bias;
         if k = 0 then
         begin
            for j := 0 to High(InputLayer.Neurons) do
              Sum := Sum + InputLayer.Neurons[j].Output * HiddenLayers[k].Neurons[i].Weights[j];
         end
         else
         begin
            for j := 0 to High(HiddenLayers[k-1].Neurons) do
              Sum := Sum + HiddenLayers[k-1].Neurons[j].Output * HiddenLayers[k].Neurons[i].Weights[j];
         end;
         // Recurrent connection: add previous output weighted
         Sum := Sum + HiddenLayers[k].Neurons[i].PrevOutput * HiddenLayers[k].Neurons[i].RecurrentWeight;
         HiddenLayers[k].Neurons[i].Output := 1 / (1 + Exp(-Sum));
      end;
   end;

   // Output layer: unchanged (no recurrence in classic RNNs)
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to High(HiddenLayers[High(HiddenLayers)].Neurons) do
         Sum := Sum + HiddenLayers[High(HiddenLayers)].Neurons[j].Output * OutputLayer.Neurons[i].Weights[j];
      OutputLayer.Neurons[i].Output := 1 / (1 + Exp(-Sum));
   end;

   // Advance hidden state
   StepHiddenState;
end;

procedure TMultiLayerRNN.BackPropagate(Target: Darray);
var
   i, j, k: Integer;
   ErrorSum: Double;
begin
   // Output error
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Error := OutputLayer.Neurons[i].Output * (1 - OutputLayer.Neurons[i].Output) * (Target[i] - OutputLayer.Neurons[i].Output);

   // Hidden errors (ignoring recurrence in BPTT for vanilla simplicity)
   for k := High(HiddenLayers) downto 0 do
   begin
      for i := 0 to High(HiddenLayers[k].Neurons) do
      begin
         if k = High(HiddenLayers) then
         begin
            ErrorSum := 0;
            for j := 0 to High(OutputLayer.Neurons) do
              ErrorSum := ErrorSum + OutputLayer.Neurons[j].Error * OutputLayer.Neurons[j].Weights[i];
         end
         else
         begin
            ErrorSum := 0;
            for j := 0 to High(HiddenLayers[k+1].Neurons) do
               ErrorSum := ErrorSum + HiddenLayers[k+1].Neurons[j].Error * HiddenLayers[k+1].Neurons[j].Weights[i];
         end;
         HiddenLayers[k].Neurons[i].Error := HiddenLayers[k].Neurons[i].Output * (1 - HiddenLayers[k].Neurons[i].Output) * ErrorSum;
      end;
   end;
end;

procedure TMultiLayerRNN.UpdateWeights;
var
   i, j, k: Integer;
begin
   // Output layer weights
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      for j := 0 to High(HiddenLayers[High(HiddenLayers)].Neurons) do
         OutputLayer.Neurons[i].Weights[j] := OutputLayer.Neurons[i].Weights[j] + LearningRate * OutputLayer.Neurons[i].Error * HiddenLayers[High(HiddenLayers)].Neurons[j].Output;
      OutputLayer.Neurons[i].Bias := OutputLayer.Neurons[i].Bias + LearningRate * OutputLayer.Neurons[i].Error;
   end;

   // Hidden layers (add recurrence weight update as well)
   for k := High(HiddenLayers) downto 0 do
   begin
      for i := 0 to High(HiddenLayers[k].Neurons) do
      begin
         if k = 0 then
         begin
            for j := 0 to High(InputLayer.Neurons) do
               HiddenLayers[k].Neurons[i].Weights[j] := HiddenLayers[k].Neurons[i].Weights[j] + LearningRate * HiddenLayers[k].Neurons[i].Error * InputLayer.Neurons[j].Output;
         end
         else
         begin
            for j := 0 to High(HiddenLayers[k-1].Neurons) do
               HiddenLayers[k].Neurons[i].Weights[j] := HiddenLayers[k].Neurons[i].Weights[j] + LearningRate * HiddenLayers[k].Neurons[i].Error * HiddenLayers[k-1].Neurons[j].Output;
         end;
         HiddenLayers[k].Neurons[i].RecurrentWeight := HiddenLayers[k].Neurons[i].RecurrentWeight + LearningRate * HiddenLayers[k].Neurons[i].Error * HiddenLayers[k].Neurons[i].PrevOutput;
         HiddenLayers[k].Neurons[i].Bias := HiddenLayers[k].Neurons[i].Bias + LearningRate * HiddenLayers[k].Neurons[i].Error;
      end;
   end;
end;

// Sequence prediction (returns all outputs for all timesteps)
function TMultiLayerRNN.PredictSequence(const Sequence: TDarray2D): TDarray2D;
var
   t, i: integer;
begin
   ResetHiddenStates;
   SetLength(Result, Length(Sequence));
   for t := 0 to High(Sequence) do
   begin
      FeedForwardStep(Sequence[t]);
      SetLength(Result[t], Length(OutputLayer.Neurons));
      for i := 0 to High(OutputLayer.Neurons) do
        Result[t][i] := OutputLayer.Neurons[i].Output;
   end;
end;

// Trains on the whole input/target sequence
procedure TMultiLayerRNN.TrainSequence(const Inputs, Targets: TDarray2D);
var
   t, i: integer;
begin
   ResetHiddenStates;
   for t := 0 to High(Inputs) do
   begin
      FeedForwardStep(Inputs[t]);
      BackPropagate(Targets[t]);
      UpdateWeights;
   end;
end;

var
  RNN: TMultiLayerRNN;
  SequenceLen, HiddenSize, t, epoch: Integer;
  Inputs, Targets, Predictions: TDarray2D;
begin
  Randomize;

  // Parameters
  SequenceLen := 5;
  InputSize := 2;
  HiddenSize := 4;
  OutputSize := 2;

  // Create dummy input and target sequences: "identity" task (output = input)
  SetLength(Inputs, SequenceLen);
  SetLength(Targets, SequenceLen);
  for t := 0 to SequenceLen - 1 do
  begin
    SetLength(Inputs[t], InputSize);
    SetLength(Targets[t], OutputSize);
    Inputs[t][0] := t / SequenceLen;           // Example values in [0, 1)
    Inputs[t][1] := (SequenceLen - t) / SequenceLen;
    Targets[t][0] := Inputs[t][0];
    Targets[t][1] := Inputs[t][1];
  end;

  // Create RNN: 1 hidden layer, HiddenSize units
  RNN := TMultiLayerRNN.Create(InputSize, [HiddenSize], OutputSize);
  RNN.MaxIterations := 200;

  // Train on the same sequence several times
  for epoch := 1 to RNN.MaxIterations do
    RNN.TrainSequence(Inputs, Targets);

  // Test and print results
  Predictions := RNN.PredictSequence(Inputs);
  WriteLn('t | Input0   Input1   | Pred0    Pred1   | Target0  Target1');
  WriteLn('------------------------------------------------------------');
  for t := 0 to SequenceLen - 1 do
    WriteLn(
      t:1, ' | ',
      Inputs[t][0]:7:4, ' ', Inputs[t][1]:7:4, ' | ',
      Predictions[t][0]:7:4, ' ', Predictions[t][1]:7:4, ' | ',
      Targets[t][0]:7:4, ' ', Targets[t][1]:7:4
    );

  RNN.Free;
end.

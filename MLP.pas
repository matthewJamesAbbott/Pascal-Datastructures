//
// Matthew Abbott 19/3/2023
//

{$mode objfpc}
{$M+}

program MLPtest;

uses Classes, Math, SysUtils;

type
   Darray = array of Double;
   TDataPoint = record
      Input: Darray;
      Target: Darray;
   end;

   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
   end;

   TLayer = record
      Neurons: array of TNeuron;
   end;

   TMultiLayerPerceptron = class
   private
      LearningRate: Double;
      MaxIterations: integer;
      InputLayer: TLayer;
      HiddenLayers: array of TLayer;
      OutputLayer: TLayer;
      procedure InitializeLayer(var Layer: TLayer; NumNeurons: Integer; NumInputs: Integer);
      procedure FeedForward;
      procedure BackPropagate(Target: Darray);
      procedure UpdateWeights;
   public
      constructor Create(InputSize: Integer;HiddenSizes: Array of Integer;  OutputSize: Integer);
      function Predict(Input: Darray): Darray;
      procedure Train(Input, Target: Darray);
      procedure SaveMLPModel(const Filename: string; MLP: TMultiLayerPerceptron);
   end;
var
   InputSize: integer = 4;
   HiddenSizesG: array of Integer;
   OutputSize: integer = 3;

constructor TMultiLayerPerceptron.Create(InputSize: Integer; HiddenSizes: array of Integer; OutputSize: Integer);

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
   InitializeLayer(InputLayer, InputSize + 1, InputSize);
   NumInputs := InputSize;
   for i := 0 to High(HiddenSizes)  do
   begin
      NumNeurons := HiddenSizes[i];
      InitializeLayer(HiddenLayers[i], NumNeurons + 1, NumInputs + 1);
      NumInputs := NumNeurons;
   end;
   InitializeLayer(OutputLayer, OutputSize, NumInputs + 1);
end;

procedure TMultiLayerPerceptron.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer);
var
   i, j: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   for i := 0 to NumNeurons - 1 do
   begin
     SetLength(Layer.Neurons[i].Weights, NumInputs);
     for j := 0 to NumInputs - 1 do
       Layer.Neurons[i].Weights[j] := Random - 0.5;
     Layer.Neurons[i].Bias := Random - 0.5;
   end;
end;

procedure TMultiLayerPerceptron.FeedForward;
var
   i, j, k: Integer;
   Sum: Double;
begin
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
         HiddenLayers[k].Neurons[i].Output := 1 / (1 + Exp(-Sum));
      end;
   end;

   // Calculate output layer outputs
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to High(HiddenLayers[High(HiddenLayers)].Neurons) do
         Sum := Sum + HiddenLayers[High(HiddenLayers)].Neurons[j].Output * OutputLayer.Neurons[i].Weights[j];
      OutputLayer.Neurons[i].Output := 1 / (1 + Exp(-Sum));
   end;
end;

procedure TMultiLayerPerceptron.BackPropagate(Target: Darray);
var
   i, j, k: Integer;
   ErrorSum: Double;
begin
   // Calculate output layer errors
   for i := 0 to High(OutputLayer.Neurons) do
      OutputLayer.Neurons[i].Error := OutputLayer.Neurons[i].Output * (1 - OutputLayer.Neurons[i].Output) * (Target[i] - OutputLayer.Neurons[i].Output);

   // Calculate hidden layer errors
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

procedure TMultiLayerPerceptron.UpdateWeights;
var
   i, j, k: Integer;
begin
	
   // Update weights of output layer neurons
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      for j := 0 to High(HiddenLayers[High(HiddenLayers)].Neurons) do
         OutputLayer.Neurons[i].Weights[j] := OutputLayer.Neurons[i].Weights[j] + LearningRate * OutputLayer.Neurons[i].Error * HiddenLayers[High(HiddenLayers)].Neurons[j].Output;
      OutputLayer.Neurons[i].Bias := OutputLayer.Neurons[i].Bias + LearningRate * OutputLayer.Neurons[i].Error;
   end;

   // Update weights of hidden layer neurons
   for k := High(HiddenLayers) downto 0 do
   begin
      for i := 0 to High(HiddenLayers[k].Neurons) do
      begin
         if k = 0 then
         begin
            for j := 0 to High(InputLayer.Neurons) do
               HiddenLayers[k].Neurons[i].Weights[j] := HiddenLayers[k].Neurons[i].Weights[j] + LearningRate * HiddenLayers[k].Neurons[i].Error * InputLayer.Neurons[j].Output;
            HiddenLayers[k].Neurons[i].Bias := HiddenLayers[k].Neurons[i].Bias + LearningRate * HiddenLayers[k].Neurons[i].Error;
         end
         else
         begin
            for j := 0 to High(HiddenLayers[k-1].Neurons) do
               HiddenLayers[k].Neurons[i].Weights[j] := HiddenLayers[k].Neurons[i].Weights[j] + LearningRate * HiddenLayers[k].Neurons[i].Error * HiddenLayers[k-1].Neurons[j].Output;
            HiddenLayers[k].Neurons[i].Bias := HiddenLayers[k].Neurons[i].Bias + LearningRate * HiddenLayers[k].Neurons[i].Error;
         end;
      end;
   end;
end;

function TMultiLayerPerceptron.Predict(Input: Darray): Darray;
var
   i: Integer;
begin

   // Set input layer outputs
   for i := 0 to High(InputLayer.Neurons) do
      InputLayer.Neurons[i].Output := Input[i];

   // Feedforward
   FeedForward;

   // Return output layer outputs
   SetLength(Result, OutputSize);
   for i := 0 to High(OutputLayer.Neurons) do
      Result[i] := OutputLayer.Neurons[i].Output;
end;

procedure TMultiLayerPerceptron.Train(Input, Target: Darray);
var
   i: integer;
begin

   // Set input layer outputs
   for i := 0 to High(InputLayer.Neurons) do
      InputLayer.Neurons[i].Output := Input[i];

   // Feedforward
   FeedForward;

   // Backpropagate
   BackPropagate(Target);

   // Update weights
   UpdateWeights;
end;

procedure TMultiLayerPerceptron.SaveMLPModel(const Filename: string; MLP: TMultiLayerPerceptron);
var
   F: File;
   NumInputs, LayerCount, i, j, k: Integer;
begin
   AssignFile(F, Filename);
   Reset(F, 1);
   LayerCount := Length(HiddenLayers);
   // Write hyperparameters and learning rate
   BlockWrite(F, LayerCount, SizeOf(Integer));
   BlockWrite(F, InputSize, SizeOf(Integer)); // InputSize
   for i := 0 to High(HiddenLayers) do
      BlockWrite(F, HiddenSizesG[i], SizeOf(Integer)); 
   BlockWrite(F, OutputSize, SizeOf(Integer)); // OutputSize
   BlockWrite(F, MLP.LearningRate, SizeOf(Double));

   // Write weights and biases for input layer
   for i := 0 to High(MLP.InputLayer.Neurons) do
   begin
      NumInputs := Length(MLP.InputLayer.Neurons[i].Weights);
      BlockWrite(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.InputLayer.Neurons[i].Weights) do
         BlockWrite(F, MLP.InputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockWrite(F, MLP.InputLayer.Neurons[i].Bias, SizeOf(Double));
   end;

   // Write weights and biases for hidden layers 
   for k := 0 to High(HiddenLayers) do
   begin
      for i := 0 to High(MLP.HiddenLayers[k].Neurons) do
      begin
         NumInputs := Length(MLP.HiddenLayers[k].Neurons[i].Weights);
         BlockWrite(F, NumInputs, SizeOf(Integer));
         for j := 0 to High(MLP.HiddenLayers[k].Neurons[i].Weights) do
            BlockWrite(F, MLP.HiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         BlockWrite(F, MLP.HiddenLayers[k].Neurons[i].Bias, SizeOf(Double));
      end;
   end;
   // Write weights and biases for output layer
   for i := 0 to High(MLP.OutputLayer.Neurons) do
   begin
      NumInputs := High(MLP.OutputLayer.Neurons[i].Weights);
      BlockWrite(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.OutputLayer.Neurons[i].Weights) do
         BlockWrite(F, MLP.OutputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockWrite(F, MLP.OutputLayer.Neurons[i].Bias, SizeOf(Double));
   end;
   CloseFile(F);
end;


function LoadMLPModel(const Filename: string): TMultiLayerPerceptron;
var
   F: File;
   HiddenLayerSize: Array of Integer;
   InputSize, NumHiddenLayers,NumInputs, OutputSize, i, j, l: Integer;
   MLP: TMultiLayerPerceptron;
begin
   AssignFile(F, Filename);
   Reset(F, 1);
   BlockRead(F, NumHiddenLayers, SizeOf(Integer));
   setLength(HiddenLayerSize, NumHiddenLayers);
   BlockRead(F, InputSize, SizeOf(Integer)); // InputSize
   for i := 0 to High(HiddenLayerSize) do
      BlockRead(F, HiddenLayerSize[i], SizeOf(Integer)); // HiddenSize1
   BlockRead(F, OutputSize, SizeOf(Integer)); // OutputSize
   MLP := TMultiLayerPerceptron.Create(InputSize, HiddenLayerSize, OutputSize);
   BlockRead(F, MLP.LearningRate, SizeOf(Double));

   // Write weights and biases for input layer
   for i := 0 to High(MLP.InputLayer.Neurons) do
   begin
      NumInputs := Length(MLP.InputLayer.Neurons[i].Weights);
      BlockRead(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.InputLayer.Neurons[i].Weights) do
         BlockRead(F, MLP.InputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockRead(F, MLP.InputLayer.Neurons[i].Bias, SizeOf(Double));
   end;

   // Write weights and biases for hidden layers
   for l := 0 to High(MLP.HiddenLayers) do
   begin
      for i := 0 to High(MLP.HiddenLayers[l].Neurons) do
      begin 
         NumInputs := Length(MLP.HiddenLayers[l].Neurons[i].Weights);
         BlockRead(F, NumInputs, SizeOf(Integer));
         for j := 0 to High(MLP.HiddenLayers[l].Neurons[i].Weights) do
            BlockRead(F, MLP.HiddenLayers[l].Neurons[i].Weights[j], SizeOf(Double));
         BlockRead(F, MLP.HiddenLayers[l].Neurons[i].Bias, SizeOf(Double));
      end;
   end;

   // Write weights and biases for output layer
   for i := 0 to High(MLP.OutputLayer.Neurons) do
   begin
      NumInputs := High(MLP.OutputLayer.Neurons[i].Weights)-1;
      BlockRead(F, NumInputs, SizeOf(Integer));
      for j := 0 to High(MLP.OutputLayer.Neurons[i].Weights) do
         BlockRead(F, MLP.OutputLayer.Neurons[i].Weights[j], SizeOf(Double));
      BlockRead(F, MLP.OutputLayer.Neurons[i].Bias, SizeOf(Double));
   end;
   CloseFile(F);
   Result := MLP;
end;


function KFoldCrossValidation(Data: array of TDataPoint; NumFolds: Integer; MLP: TMultiLayerPerceptron): Double;
var
   FoldSize, NumSamples, i, j, k: Integer;
   SumAccuracy: Double;
   TestSet, TrainSet: array of TDataPoint;
begin
   // Part 1: Split data into folds
   NumSamples := Length(Data);
   FoldSize := NumSamples div NumFolds;

   SumAccuracy := 0;
   for i := 0 to NumFolds - 1 do
   begin
      // Select test set for fold i
      SetLength(TestSet, FoldSize);
      for j := 0 to FoldSize - 1 do
         TestSet[j] := Data[i * FoldSize + j];

      // Select training set for fold i
      SetLength(TrainSet, NumSamples - FoldSize);
      k := 0;
      for j := 0 to NumSamples - 1 do
      begin
         if (j >= i * FoldSize) and (j < (i + 1) * FoldSize) then
            Continue;
            TrainSet[k] := Data[j];
            Inc(k);
      end;

      // Part 2: Train MLP on training set
      for j := 0 to MLP.MaxIterations - 1 do
      begin
         for k := 0 to High(TrainSet) do
            MLP.Train(TrainSet[k].Input, TrainSet[k].Target);
      end;

      // Part 3: Evaluate accuracy on test set
      for j := 0 to High(TestSet) do
      begin
         if Round(MLP.Predict(TestSet[j].Input)[0]) = Round(TestSet[j].Target[0]) then
            SumAccuracy := SumAccuracy + 1;
      end;
   end;

   Result := SumAccuracy / NumSamples;
end;

function PrecisionScore(Data: array of TDataPoint; MLP: TMultiLayerPerceptron; NumFolds: Integer): Double;
var
   TP, FP: Integer;
   i, j: Integer;
   Prediction: Double;
begin
   TP := 0;
   FP := 0;
   for i := 0 to NumFolds - 1 do
   begin
      for j := i * Length(Data) div NumFolds to (i + 1) * Length(Data) div NumFolds - 1 do
      begin
         Prediction := MLP.Predict(Data[j].Input)[0];
         if Round(Prediction) = 1 then
         begin
            if Round(Data[j].Target[0]) = 1 then
               Inc(TP)
            else
               Inc(FP);
         end;
      end;
   end;
   if TP + FP = 0 then
      Result := 0
   else
      Result := TP / (TP + FP);
end;

function RecallScore(Data: array of TDataPoint; MLP: TMultiLayerPerceptron; NumFolds: Integer): Double;
var
   FoldSize, NumSamples, i, j, k, TruePositives, FalseNegatives: Integer;
   TestSet, TrainSet: array of TDataPoint;
   Predictions: Darray;
begin

   // Check that Data is not empty
   if Length(Data) = 0 then
   begin
      Result := 0.0;
      Exit;
   end;

   // Part 1: Split data into folds
   NumSamples := Length(Data);
   FoldSize := NumSamples div NumFolds;
   TruePositives := 0;
   FalseNegatives := 0;
   for i := 0 to NumFolds - 1 do
   begin
      // Select test set for fold i
      SetLength(TestSet, FoldSize);
      for j := 0 to FoldSize - 1 do
         TestSet[j] := Data[i * FoldSize + j];

    // Select training set for fold i
    SetLength(TrainSet, NumSamples - FoldSize + 1);
    j := 0;
    for k := 0 to NumSamples - 1 do
    begin
      if (k >= i * FoldSize) and (k < (i + 1) * FoldSize) then
        Continue;
      TrainSet[j] := Data[k];
      Inc(j);
    end;

    // Part 2: Train MLP on training set
    for k := 0 to MLP.MaxIterations - 1 do
    begin
      for j := 0 to High(TrainSet)-1 do
        MLP.Train(TrainSet[j].Input, TrainSet[j].Target);
    end;
    // Part 3: Evaluate recall on test set
    for j := 0 to High(TestSet)-1 do
    begin
      Predictions := MLP.Predict(TestSet[j].Input);
      if (Round(Predictions[0]) = 1) and (Round(TestSet[j].Target[0]) = 1) then
        Inc(TruePositives)
      else if (Round(Predictions[0]) = 0) and (Round(TestSet[j].Target[0]) = 1) then
        Inc(FalseNegatives);
    end;
  end;

  Result := TruePositives / (TruePositives + FalseNegatives);
end;


function F1Score(Precision, Recall: Double): Double;
begin
   if (Precision + Recall) = 0 then
      Result := 0
   else
      Result := 2 * (Precision * Recall) / (Precision + Recall);
end;

var
   Data: array of TDataPoint;
   NumFolds, i, j, k: Integer;
   MLP, MLP2: TMultiLayerPerceptron;
   Accuracy, Precision, Recall, F1ScoreVar: Double;
begin

   // Create data inputs
   SetLength(Data, 7500);
   for i := 0 to 2499 do
   begin
      Data[i].Input := [Random * 0.5, Random * 0.5, Random * 0.5, Random * 0.5];
      Data[i].Target := [1, 0, 0];
   end;
   for i := 2500 to 5999 do
   begin
      Data[i].Input := [0.5 + Random * 0.5, 0.5 + Random * 0.5, 0.5 + Random * 0.5, 0.5 + Random * 0.5];
      Data[i].Target := [0, 1, 0];
   end;
   for i := 5000 to 7500 do
   begin
      Data[i].Input := [Random * 0.5, 0.5 + Random * 0.5, Random * 0.5, 0.5 + Random * 0.5];
      Data[i].Target := [0, 0, 1];
   end;

   // Test accuracy, precision, recall, and F1 score using 10-fold cross validation
   MLP := TMultiLayerPerceptron.Create(InputSize, [8, 8, 8], OutputSize);
   MLP.MaxIterations := 30;
   NumFolds := 10;
   Accuracy := KFoldCrossValidation(Data, NumFolds, MLP);
   Precision := PrecisionScore(Data, MLP, NumFolds);
   Recall := RecallScore(Data, MLP, NumFolds);
   F1ScoreVar := F1Score(Precision, Recall);

   WriteLn('Accuracy: ', Accuracy:0:3);
   WriteLn('Precision: ', Precision:0:3);
   WriteLn('Recall: ', Recall:0:3);
   WriteLn('F1 Score: ', F1ScoreVar:0:3);
   MLP.SaveMLPModel('TestSaveMLP.bin',MLP);
   MLP2 := LoadMLPModel('TestSaveMLP.bin');
   MLP2.MaxIterations := 30;
   
   for k := 1 to High(MLP.HiddenLayers) do
   begin
      for i := 0 to High(MLP.HiddenLayers[k-1].Neurons) do
      begin
         WriteLn(Format('Neuron %d weights: %d for original MLP', [i, Length(MLP.HiddenLayers[k].Neurons[i].Weights)]));
         for j := 0 to High(MLP.HiddenLayers[k].Neurons[i].Weights) do
            Writeln(MLP.HiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         Writeln('Bias');
         Writeln(MLP.HiddenLayers[k].Neurons[i].Bias, SizeOf(Double));
   
         WriteLn(Format('Neuron %d weights: %d for loaded MLP', [i, Length(MLP2.HiddenLayers[1].Neurons[i].Weights)]));
         for j := 0 to High(MLP2.HiddenLayers[k].Neurons[i].Weights) do
            Writeln(MLP2.HiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         Writeln('Bias');
         Writeln(MLP2.HiddenLayers[k].Neurons[i].Bias, SizeOf(Double));
      end;
   end;

   Accuracy := KFoldCrossValidation(Data, NumFolds, MLP2);
   Precision := PrecisionScore(Data, MLP2, NumFolds);
   Recall := RecallScore(Data, MLP2, NumFolds);
   F1ScoreVar := F1Score(Precision, Recall);

   WriteLn('Accuracy For Loaded MLP: ', Accuracy:0:3);
   WriteLn('Precision For Loaded MLP: ', Precision:0:3);
   WriteLn('Recall For Loaded MLP: ', Recall:0:3);
   WriteLn('F1 Score For Loaded MLP:', F1ScoreVar:0:3);
   for k := 1 to High(MLP.HiddenLayers) do
   begin
      for i := 0 to High(MLP.HiddenLayers[k-1].Neurons) do
      begin
         WriteLn(Format('Neuron %d weights: %d for original MLP', [i, Length(MLP.HiddenLayers[k].Neurons[i].Weights)]));
         for j := 0 to High(MLP.HiddenLayers[k].Neurons[i].Weights) do
            Writeln(MLP.HiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         Writeln('Bias');
         Writeln(MLP.HiddenLayers[k].Neurons[i].Bias, SizeOf(Double));

	 WriteLn(Format('Neuron %d weights: %d for loaded MLP', [i, Length(MLP2.HiddenLayers[1].Neurons[i].Weights)]));
         for j := 0 to High(MLP2.HiddenLayers[k].Neurons[i].Weights) do
            Writeln(MLP2.HiddenLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         Writeln('Bias');
         Writeln(MLP2.HiddenLayers[k].Neurons[i].Bias, SizeOf(Double));
      end;
   end;
end.

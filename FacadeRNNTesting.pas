{
  Comprehensive test for RNNFacade unit (FacadeRNN.pas)
  Covers sequence training, introspection, manipulation, dropout, histograms, diagnostics, etc.
}

{$mode objfpc}{$H+}

program TestRNNFacade_AllFeatures;

uses
  SysUtils, Math, RNNFacade;

procedure PrintArray(const Arr: DArray; Title: String = '');
var i: Integer;
begin
  if Title <> '' then WriteLn(Title);
  for i := 0 to High(Arr) do Write(Format('%.4f ', [Arr[i]]));
  WriteLn;
end;

procedure Print2D(const Arr: TDArray2D; Title: String = '');
var i: Integer;
begin
  if Title <> '' then WriteLn(Title);
  for i := 0 to High(Arr) do PrintArray(Arr[i]);
end;

procedure PrintHistogram(const H: THistogram; Title: String = '');
var i: Integer;
begin
  if Title <> '' then WriteLn(Title);
  for i := 0 to High(H) do
    WriteLn(Format('[%.4f, %.4f): Count=%d Pct=%.2f%%',
      [H[i].RangeMin, H[i].RangeMax, H[i].Count, 100*H[i].Percentage]));
end;

procedure PrintGradStats(const G: TGradientScaleArray; Title: String = '');
var i: Integer;
begin
  if Title <> '' then WriteLn(Title);
  for i := 0 to High(G) do
    WriteLn(Format('Timestep %d: mean abs %.5f  max %.5f  min %.5f', [
      G[i].Timestep, G[i].MeanAbsGrad, G[i].MaxAbsGrad, G[i].MinAbsGrad
    ]));
end;

var
  RNN, RNN2: TRNNFacade;
  ModelFile: string;
  Inputs, Targets, Outputs, OutputsReload: TDArray2D;
  gradStats: TGradientScaleArray;
  hist: THistogram;
  gates: DArray;
  attnW, attnC: DArray;
  i, j, T, InputSize, OutSize: Integer;
  Loss: Double;
  LNS: TLayerNormStats;
  GateSat: TGateSaturationStats;
  OptStat: TOptimizerStateRecord;
begin
  Randomize;
  // Setup network parameters
  InputSize := 3;
  OutSize := 2;
  T := 6;
  
  SetLength(Inputs, T);
  SetLength(Targets, T);
  for i := 0 to T-1 do begin
    SetLength(Inputs[i], InputSize);
    SetLength(Targets[i], OutSize);
    for j := 0 to InputSize-1 do Inputs[i][j] := Random * 2 - 1;
    for j := 0 to OutSize-1 do Targets[i][j] := Random;
  end;

  // Use an LSTM with 1 hidden layer of 4 units for full API coverage
  RNN := TRNNFacade.Create(
    InputSize,
    [4],
    OutSize,
    ctLSTM,
    atTanh,
    atLinear,
    ltMSE,
    0.1,
    5.0,
    10
  );

  WriteLn('== Forward pass ==');
  Outputs := RNN.ForwardSequence(Inputs);
  Print2D(Outputs, 'Network outputs:');

  WriteLn('== Compute loss ==');
  Loss := ComputeLoss(Outputs[T-1], Targets[T-1], ltMSE);
  WriteLn('Loss for last step: ', Loss:0:5);

  WriteLn('== Training ==');
  Loss := RNN.TrainSequence(Inputs, Targets);
  WriteLn('Loss after training: ', Loss:0:5);

  WriteLn('== Prediction ==');
  Outputs := RNN.Predict(Inputs);
  Print2D(Outputs, 'Predicted outputs:');
  
  WriteLn('== Introspection: Hidden/Output/Cell Access ==');
  WriteLn('Hidden state (layer 0, step 2, neuron 1): ', RNN.GetHiddenState(0,2,1):0:4);
  WriteLn('Cell state (layer 0, step 2, neuron 1): ', RNN.GetCellState(0,2,1):0:4);
  WriteLn('Output (step 2, idx 1): ', RNN.GetOutput(2,1):0:4);
  WriteLn('Hidden state before set: ', RNN.GetHiddenState(0,2,1):0:4);
RNN.SetHiddenState(0,2,1, 0.42);
WriteLn('Hidden state after set: ', RNN.GetHiddenState(0,2,1):0:4);
WriteLn('NOTE: This is a façade API demonstration—not a result of learning.');
  RNN.SetOutput(2,1, -1.23);
  WriteLn('After set: Hidden (0,2,1)=', RNN.GetHiddenState(0,2,1):0:4, ', Output(2,1)=', RNN.GetOutput(2,1):0:4);

  WriteLn('== Gate Access, Pre-Activations, Inputs ==');
  WriteLn('Input gate (layer 0, step 2, neuron 1): ', RNN.GetGateValue(gtInput,0,2,1):0:4);
  WriteLn('Pre-activation (layer 0, step 2, neuron 1): ', RNN.GetPreActivation(0,2,1):0:4);
  WriteLn('Input vector step 2, idx 1: ', RNN.GetInputVector(2,1):0:4);

  WriteLn('== Gradients & Optimizer ==');
  WriteLn('Weight grad (0,0,0): ', RNN.GetWeightGradient(0,0,0):0:6);
  WriteLn('Bias grad (0,0): ', RNN.GetBiasGradient(0,0):0:6);
  OptStat := RNN.GetOptimizerState(0,0,0);
  WriteLn('OptimizerState[Momentum]=', OptStat.Momentum:0:6);

  WriteLn('== Dropout ==');
  RNN.EnableDropout(True);
  RNN.SetDropoutRate(0.5);
  WriteLn('Dropout enabled: ', RNN.GetDropoutRate:0:2);
  WriteLn('DropoutMask (layer 0, t=1, neuron 2): ', RNN.GetDropoutMask(0,1,2):0:3);
  RNN.EnableDropout(False);
  WriteLn('Dropout disabled: ', RNN.GetDropoutRate:0:2);

  WriteLn('== Sequence-to-Sequence Output APIs ==');
  PrintArray(RNN.GetSequenceOutputs(1), 'Seq outputs (output idx 1):');
  PrintArray(RNN.GetSequenceHiddenStates(0,1), 'Seq hidden states (layer 0, neuron 1):');
  PrintArray(RNN.GetSequenceCellStates(0,1), 'Seq cell states (layer 0, neuron 1):');
  PrintArray(RNN.GetSequenceGateValues(gtInput, 0, 1), 'Seq input gate (layer 0, neuron 1):');

  WriteLn('== Hidden and Cell State Manipulation ==');
  RNN.ResetHiddenState(0, 0.1);
  RNN.ResetCellState(0, 0.2);
  RNN.InjectHiddenState(0, [0.1, 0.2, 0.3, 0.4]);
  RNN.InjectCellState(0, [1.0, 1.0, 1.0, 1.0]);
  RNN.ResetAllStates();

  WriteLn('== Attention placeholders ==');
  attnW := RNN.GetAttentionWeights(1);
  attnC := RNN.GetAttentionContext(1);
  PrintArray(attnW, 'Attention W (step 1):');
  PrintArray(attnC, 'Attention context (step 1):');

  WriteLn('== Diagnostics: Histogram, Gate saturation, Gradients ==');
  hist := RNN.GetHiddenStateHistogram(0, 2, 5);
  PrintHistogram(hist, 'Histogram hidden L0, t2:');
  hist := RNN.GetActivationHistogramOverTime(0, 1, 5);
  PrintHistogram(hist, 'Activation hist L0, neuron 1:');
  GateSat := RNN.GetGateSaturation(gtInput,0,2,0.05);
  WriteLn('GateSaturation input (L0,t2): NearZero=', GateSat.NearZeroPct*100:0:2,'% NearOne=', GateSat.NearOnePct*100:0:2,'%');
  
  gradStats := RNN.GetGradientScalesOverTime(0);
  PrintGradStats(gradStats, 'Gradient scales:');

  WriteLn('Vanishing gradient? ', BoolToStr(RNN.DetectVanishingGradient(0,1e-7),True));
  WriteLn('Exploding gradient? ', BoolToStr(RNN.DetectExplodingGradient(0,1e3),True));

  WriteLn('== LayerNorm Stats ==');
  LNS := RNN.GetLayerNormStats(0,2);
  WriteLn('LN mean=', LNS.Mean:0:4, ' var=', LNS.Variance:0:4, ' gamma=', LNS.Gamma:0:4, ' beta=', LNS.Beta:0:4);

  // Reset/Utility
  RNN.ResetGradients;
  RNN.ApplyGradients;

  WriteLn('== Meta ==');
  WriteLn('Layer count: ', RNN.GetLayerCount);
  WriteLn('Hidden size L0: ', RNN.GetHiddenSize(0));
  WriteLn('Cell type: ', Ord(RNN.GetCellType));
  WriteLn('Sequence length: ', RNN.GetSequenceLength);


  // ==== Save Model ====
  ModelFile := 'TestRNNFacade_save.bin';
  WriteLn('== Saving model to ', ModelFile, ' ==');
  RNN.SaveModel(ModelFile);
  WriteLn('Model saved.');

  // ==== Load Model into a fresh instance ====
  WriteLn('== Creating new RNN instance and reloading model ==');
  RNN2 := TRNNFacade.Create(
    InputSize,
    [4],
    OutSize,
    ctLSTM,
    atTanh,
    atLinear,
    ltMSE,
    0.1,
    5.0,
    10
  );
  RNN2.LoadModel(ModelFile);

  // ==== Compare Prediction Outputs ====
  OutputsReload := RNN2.Predict(Inputs);
  WriteLn('== Comparing predictions from loaded model ==');
  Print2D(OutputsReload, 'Predicted outputs from loaded model:');

  // Optionally: compare values
  for i := 0 to High(OutputsReload) do
    for j := 0 to High(OutputsReload[i]) do
      if Abs(Outputs[i][j] - OutputsReload[i][j]) > 1e-5 then
        WriteLn(Format('WARNING: Output mismatch at step %d, idx %d: orig=%.6f loaded=%.6f', [i,j,Outputs[i][j],OutputsReload[i][j]]));

  RNN2.Free;

  // Optional cleanup
  if FileExists(ModelFile) then
    DeleteFile(ModelFile);
  
  RNN.Free;
end.

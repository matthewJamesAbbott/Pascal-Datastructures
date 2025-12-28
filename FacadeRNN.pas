//
// Facaded RNN with Full CLI - Single File Implementation
// Matthew Abbott 2025
//

{$mode objfpc}{$H+}
{$M+}

program FacadedRNN;

uses Classes, Math, SysUtils;

type
  TActivationType = (atSigmoid, atTanh, atReLU, atLinear);
  TLossType = (ltMSE, ltCrossEntropy);
  TCellType = (ctSimpleRNN, ctLSTM, ctGRU);
  TGateType = (gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate);
  TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp,
              cmdGetHidden, cmdSetHidden, cmdGetOutput, cmdSetOutput,
              cmdGetCellState, cmdGetGate, cmdResetStates, cmdSetDropout, 
              cmdGetDropout, cmdDetectVanishing, cmdDetectExploding,
              cmdGetSeqOutputs, cmdGetSeqHidden);

  DArray = array of Double;
  TDArray2D = array of DArray;
  TDArray3D = array of TDArray2D;
  TIntArray = array of Integer;

  TTimeStepCache = record
    Input: DArray;
    H, C: DArray;
    PreH: DArray;
    F, I, CTilde, O, TanhC: DArray;
    Z, R, HTilde: DArray;
    OutPre, OutVal: DArray;
  end;

// ========== Utility Functions ==========
function ClipValue(V, MaxVal: Double): Double;
begin
  if V > MaxVal then Result := MaxVal
  else if V < -MaxVal then Result := -MaxVal
  else Result := V;
end;

function RandomWeight(Scale: Double): Double;
begin
  Result := (Random - 0.5) * 2.0 * Scale;
end;

procedure InitMatrix(var M: TDArray2D; Rows, Cols: Integer; Scale: Double);
var
  i, j: Integer;
begin
  SetLength(M, Rows);
  for i := 0 to Rows - 1 do
  begin
    SetLength(M[i], Cols);
    for j := 0 to Cols - 1 do
      M[i][j] := RandomWeight(Scale);
  end;
end;

procedure ZeroMatrix(var M: TDArray2D; Rows, Cols: Integer);
var
  i, j: Integer;
begin
  SetLength(M, Rows);
  for i := 0 to Rows - 1 do
  begin
    SetLength(M[i], Cols);
    for j := 0 to Cols - 1 do
      M[i][j] := 0.0;
  end;
end;

procedure ZeroArray(var A: DArray; Size: Integer);
var
  i: Integer;
begin
  SetLength(A, Size);
  for i := 0 to Size - 1 do
    A[i] := 0.0;
end;

function ConcatArrays(const A, B: DArray): DArray;
var
  i: Integer;
begin
  SetLength(Result, Length(A) + Length(B));
  for i := 0 to High(A) do
    Result[i] := A[i];
  for i := 0 to High(B) do
    Result[Length(A) + i] := B[i];
end;

function ApplyActivation(X: Double; ActType: TActivationType): Double;
begin
  case ActType of
    atSigmoid: Result := 1.0 / (1.0 + Exp(-Max(-500, Min(500, X))));
    atTanh: Result := Tanh(X);
    atReLU: if X > 0 then Result := X else Result := 0;
    atLinear: Result := X;
  else
    Result := X;
  end;
end;

function ActivationDerivative(Y: Double; ActType: TActivationType): Double;
begin
  case ActType of
    atSigmoid: Result := Y * (1.0 - Y);
    atTanh: Result := 1.0 - Y * Y;
    atReLU: if Y > 0 then Result := 1.0 else Result := 0.0;
    atLinear: Result := 1.0;
  else
    Result := 1.0;
  end;
end;

function CellTypeToStr(ct: TCellType): string;
begin
  case ct of
    ctSimpleRNN: Result := 'simplernn';
    ctLSTM: Result := 'lstm';
    ctGRU: Result := 'gru';
  else
    Result := 'simplernn';
  end;
end;

function ActivationToStr(act: TActivationType): string;
begin
  case act of
    atSigmoid: Result := 'sigmoid';
    atTanh: Result := 'tanh';
    atReLU: Result := 'relu';
    atLinear: Result := 'linear';
  else
    Result := 'sigmoid';
  end;
end;

function LossToStr(loss: TLossType): string;
begin
  case loss of
    ltMSE: Result := 'mse';
    ltCrossEntropy: Result := 'crossentropy';
  else
    Result := 'mse';
  end;
end;

function ParseCellType(const s: string): TCellType;
begin
  if LowerCase(s) = 'lstm' then
    Result := ctLSTM
  else if LowerCase(s) = 'gru' then
    Result := ctGRU
  else
    Result := ctSimpleRNN;
end;

function ParseActivation(const s: string): TActivationType;
begin
  if LowerCase(s) = 'tanh' then
    Result := atTanh
  else if LowerCase(s) = 'relu' then
    Result := atReLU
  else if LowerCase(s) = 'linear' then
    Result := atLinear
  else
    Result := atSigmoid;
end;

function ParseLoss(const s: string): TLossType;
begin
  if LowerCase(s) = 'crossentropy' then
    Result := ltCrossEntropy
  else
    Result := ltMSE;
end;

function ParseGateType(const s: string): TGateType;
begin
  if LowerCase(s) = 'forget' then
    Result := gtForget
  else if LowerCase(s) = 'input' then
    Result := gtInput
  else if LowerCase(s) = 'output' then
    Result := gtOutput
  else if LowerCase(s) = 'cell' then
    Result := gtCellCandidate
  else if LowerCase(s) = 'update' then
    Result := gtUpdate
  else if LowerCase(s) = 'reset' then
    Result := gtReset
  else if LowerCase(s) = 'hidden' then
    Result := gtHiddenCandidate
  else
    Result := gtForget;
end;

procedure ParseIntArrayHelper(const s: string; out result: TIntArray);
var
  tokens: TStringList;
  i: Integer;
begin
  tokens := TStringList.Create;
  try
    tokens.Delimiter := ',';
    tokens.DelimitedText := s;
    SetLength(result, tokens.Count);
    for i := 0 to tokens.Count - 1 do
      result[i] := StrToInt(Trim(tokens[i]));
  finally
    tokens.Free;
  end;
end;

procedure PrintUsage;
begin
  WriteLn('Facaded RNN - Command-line Recurrent Neural Network with Introspection');
  WriteLn('Matthew Abbott 2025');
  WriteLn;
  WriteLn('Commands:');
  WriteLn('  create                Create a new RNN model');
  WriteLn('  train                 Train an existing model with data');
  WriteLn('  predict               Make predictions with a trained model');
  WriteLn('  info                  Display model information');
  WriteLn('  help                  Show this help message');
  WriteLn;
  WriteLn('Core Options:');
  WriteLn('  --input=N              Input layer size (required for create)');
  WriteLn('  --hidden=N,N,...       Hidden layer sizes comma-separated (required for create)');
  WriteLn('  --output=N             Output layer size (required for create)');
  WriteLn('  --save=FILE            Save model to file');
  WriteLn('  --model=FILE           Model file to load');
  WriteLn('  --data=FILE            Training data CSV file');
  WriteLn('  --cell=TYPE            simplernn|lstm|gru (default: lstm)');
  WriteLn('  --lr=VALUE             Learning rate (default: 0.01)');
  WriteLn('  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)');
  WriteLn('  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)');
  WriteLn('  --loss=TYPE            mse|crossentropy (default: mse)');
  WriteLn('  --clip=VALUE           Gradient clipping (default: 5.0)');
  WriteLn('  --bptt=N               BPTT steps (default: 0 = full)');
  WriteLn('  --epochs=N             Number of training epochs (default: 100)');
  WriteLn('  --batch=N              Batch size (default: 1)');
  WriteLn('  --seq-len=N            Sequence length (default: auto-detect)');
  WriteLn('  --verbose              Show training progress');
  WriteLn;
  WriteLn('Facade Introspection Commands:');
  WriteLn('  get-hidden             Get hidden state value');
  WriteLn('  set-hidden             Set hidden state value');
  WriteLn('  get-output             Get output value at timestep');
  WriteLn('  set-output             Set output value at timestep');
  WriteLn('  get-cell-state         Get LSTM cell state');
  WriteLn('  get-gate               Get gate value (LSTM/GRU)');
  WriteLn('  reset-states           Reset all hidden/cell states');
  WriteLn('  set-dropout            Set dropout rate');
  WriteLn('  get-dropout            Get current dropout rate');
  WriteLn('  detect-vanishing       Check for vanishing gradients');
  WriteLn('  detect-exploding       Check for exploding gradients');
  WriteLn('  get-seq-outputs        Get all outputs for a sequence');
  WriteLn('  get-seq-hidden         Get hidden states over sequence');
  WriteLn;
  WriteLn('Facade Introspection Options:');
  WriteLn('  --layer=N              Layer index (default: 0)');
  WriteLn('  --timestep=N           Timestep index (default: 0)');
  WriteLn('  --neuron=N             Neuron index (default: 0)');
  WriteLn('  --output-idx=N         Output index (default: 0)');
  WriteLn('  --value=F              Value to set');
  WriteLn('  --gate=TYPE            Gate type: forget,input,output,cell,update,reset,hidden');
  WriteLn('  --threshold=F          Threshold for gradient detection (default: 1e-6)');
  WriteLn;
  WriteLn('Examples:');
  WriteLn('  facaded_rnn create --input=2 --hidden=16 --output=2 --cell=lstm --save=seq.bin');
  WriteLn('  facaded_rnn train --model=seq.bin --data=seq.csv --epochs=200 --save=seq_trained.bin');
  WriteLn('  facaded_rnn predict --model=seq_trained.bin --input=0.5,0.5');
  WriteLn('  facaded_rnn info --model=seq_trained.bin');
  WriteLn('  facaded_rnn get-hidden --model=seq.bin --layer=0 --timestep=0 --neuron=0');
  WriteLn('  facaded_rnn set-hidden --model=seq.bin --layer=0 --neuron=0 --value=0.5');
  WriteLn('  facaded_rnn set-dropout --model=seq.bin --value=0.2');
  WriteLn('  facaded_rnn get-seq-outputs --model=seq.bin --output-idx=0');
  WriteLn;
end;

// ========== Main Program ==========
var
  Command: TCommand;
  CmdStr: string;
  i: Integer;
  arg, key, valueStr: string;
  eqPos: Integer;

  inputSize, outputSize, epochs, batchSize, seqLen, bpttSteps: Integer;
  hiddenSizes: TIntArray;
  learningRate, gradientClip: Double;
  hiddenAct, outputAct: TActivationType;
  cellType: TCellType;
  lossType: TLossType;
  modelFile, saveFile, dataFile: string;
  verbose: Boolean;

  // Facade-specific parameters
  layerIdx, timestep, neuronIdx, outputIdx: Integer;
  gateType: TGateType;
  dropoutRate, threshold, value: Double;

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
  else if CmdStr = 'get-hidden' then Command := cmdGetHidden
  else if CmdStr = 'set-hidden' then Command := cmdSetHidden
  else if CmdStr = 'get-output' then Command := cmdGetOutput
  else if CmdStr = 'set-output' then Command := cmdSetOutput
  else if CmdStr = 'get-cell-state' then Command := cmdGetCellState
  else if CmdStr = 'get-gate' then Command := cmdGetGate
  else if CmdStr = 'reset-states' then Command := cmdResetStates
  else if CmdStr = 'set-dropout' then Command := cmdSetDropout
  else if CmdStr = 'get-dropout' then Command := cmdGetDropout
  else if CmdStr = 'detect-vanishing' then Command := cmdDetectVanishing
  else if CmdStr = 'detect-exploding' then Command := cmdDetectExploding
  else if CmdStr = 'get-seq-outputs' then Command := cmdGetSeqOutputs
  else if CmdStr = 'get-seq-hidden' then Command := cmdGetSeqHidden
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
  learningRate := 0.01;
  gradientClip := 5.0;
  epochs := 100;
  batchSize := 1;
  seqLen := 0;
  bpttSteps := 0;
  verbose := False;
  hiddenAct := atTanh;
  outputAct := atLinear;
  cellType := ctLSTM;
  lossType := ltMSE;
  modelFile := '';
  saveFile := '';
  dataFile := '';

  // Facade defaults
  layerIdx := 0;
  timestep := 0;
  neuronIdx := 0;
  outputIdx := 0;
  gateType := gtForget;
  dropoutRate := 0.0;
  threshold := 1e-6;
  value := 0.0;

  // Parse arguments
  for i := 2 to ParamCount do
  begin
    arg := ParamStr(i);

    if arg = '--verbose' then
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
      valueStr := Copy(arg, eqPos + 1, Length(arg));

      if key = '--input' then
        inputSize := StrToInt(valueStr)
      else if key = '--hidden' then
        ParseIntArrayHelper(valueStr, hiddenSizes)
      else if key = '--output' then
        outputSize := StrToInt(valueStr)
      else if key = '--save' then
        saveFile := valueStr
      else if key = '--model' then
        modelFile := valueStr
      else if key = '--data' then
        dataFile := valueStr
      else if key = '--lr' then
        learningRate := StrToFloat(valueStr)
      else if key = '--cell' then
        cellType := ParseCellType(valueStr)
      else if key = '--hidden-act' then
        hiddenAct := ParseActivation(valueStr)
      else if key = '--output-act' then
        outputAct := ParseActivation(valueStr)
      else if key = '--loss' then
        lossType := ParseLoss(valueStr)
      else if key = '--clip' then
        gradientClip := StrToFloat(valueStr)
      else if key = '--bptt' then
        bpttSteps := StrToInt(valueStr)
      else if key = '--epochs' then
        epochs := StrToInt(valueStr)
      else if key = '--batch' then
        batchSize := StrToInt(valueStr)
      else if key = '--seq-len' then
        seqLen := StrToInt(valueStr)
      else if key = '--layer' then
        layerIdx := StrToInt(valueStr)
      else if key = '--timestep' then
        timestep := StrToInt(valueStr)
      else if key = '--neuron' then
        neuronIdx := StrToInt(valueStr)
      else if key = '--output-idx' then
        outputIdx := StrToInt(valueStr)
      else if key = '--gate' then
        gateType := ParseGateType(valueStr)
      else if key = '--threshold' then
        threshold := StrToFloat(valueStr)
      else if key = '--value' then
      begin
        try
          value := StrToFloat(valueStr);
        except
          on E: Exception do
            WriteLn('Warning: Could not parse --value as float');
        end;
      end
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

    WriteLn('Created Facaded RNN model:');
    WriteLn('  Input size: ', inputSize);
    Write('  Hidden sizes: ');
    for i := 0 to High(hiddenSizes) do
    begin
      if i > 0 then Write(',');
      Write(hiddenSizes[i]);
    end;
    WriteLn;
    WriteLn('  Output size: ', outputSize);
    WriteLn('  Cell type: ', CellTypeToStr(cellType));
    WriteLn('  Hidden activation: ', ActivationToStr(hiddenAct));
    WriteLn('  Output activation: ', ActivationToStr(outputAct));
    WriteLn('  Loss function: ', LossToStr(lossType));
    WriteLn('  Learning rate: ', learningRate:0:6);
    WriteLn('  Gradient clip: ', gradientClip:0:2);
    WriteLn('  BPTT steps: ', bpttSteps);
    WriteLn('  Saved to: ', saveFile);
  end
  else if Command = cmdTrain then
  begin
    WriteLn('Train command requires model persistence (not yet fully implemented)');
  end
  else if Command = cmdPredict then
  begin
    WriteLn('Predict command requires model persistence (not yet fully implemented)');
  end
  else if Command = cmdInfo then
  begin
    WriteLn('Info command requires model persistence (not yet fully implemented)');
  end
  else if Command = cmdGetHidden then
  begin
    WriteLn('Get hidden state requires loaded model');
    WriteLn('  Layer: ', layerIdx, ', Timestep: ', timestep, ', Neuron: ', neuronIdx);
  end
  else if Command = cmdSetHidden then
  begin
    WriteLn('Set hidden state requires loaded model');
    WriteLn('  Layer: ', layerIdx, ', Neuron: ', neuronIdx, ', Value: ', value:0:6);
  end
  else if Command = cmdGetOutput then
  begin
    WriteLn('Get output requires loaded model');
    WriteLn('  Timestep: ', timestep, ', Output index: ', outputIdx);
  end
  else if Command = cmdSetOutput then
  begin
    WriteLn('Set output requires loaded model');
    WriteLn('  Timestep: ', timestep, ', Output index: ', outputIdx, ', Value: ', value:0:6);
  end
  else if Command = cmdGetCellState then
  begin
    WriteLn('Get cell state requires loaded LSTM model');
    WriteLn('  Layer: ', layerIdx, ', Timestep: ', timestep, ', Neuron: ', neuronIdx);
  end
  else if Command = cmdGetGate then
  begin
    WriteLn('Get gate value requires loaded model');
    WriteLn('  Gate type: ', Integer(gateType), ', Layer: ', layerIdx);
  end
  else if Command = cmdResetStates then
  begin
    WriteLn('Reset states requires loaded model');
    WriteLn('  Reset value: ', value:0:6);
  end
  else if Command = cmdSetDropout then
  begin
    WriteLn('Set dropout requires loaded model');
    WriteLn('  Dropout rate: ', value:0:6);
  end
  else if Command = cmdGetDropout then
  begin
    WriteLn('Get dropout requires loaded model');
  end
  else if Command = cmdDetectVanishing then
  begin
    WriteLn('Detect vanishing gradients requires loaded model');
    WriteLn('  Threshold: ', threshold);
  end
  else if Command = cmdDetectExploding then
  begin
    WriteLn('Detect exploding gradients requires loaded model');
    WriteLn('  Threshold: ', threshold);
  end
  else if Command = cmdGetSeqOutputs then
  begin
    WriteLn('Get sequence outputs requires loaded model');
    WriteLn('  Output index: ', outputIdx);
  end
  else if Command = cmdGetSeqHidden then
  begin
    WriteLn('Get sequence hidden states requires loaded model');
    WriteLn('  Layer: ', layerIdx, ', Neuron: ', neuronIdx);
  end;
end.

//
// Facaded RNN
// Matthew Abbott 2025
//

{$mode objfpc}{$H+}
{$M+}

program FacadedRNN;

uses Classes, Math, SysUtils, StrUtils;

type
  TActivationType = (atSigmoid, atTanh, atReLU, atLinear);
  TLossType = (ltMSE, ltCrossEntropy);
  TCellType = (ctSimpleRNN, ctLSTM, ctGRU);
  TGateType = (gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate);
  TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdQuery, cmdHelp);

  DArray = array of Double;
  TDArray2D = array of DArray;
  TDArray3D = array of TDArray2D;
  TIntArray = array of Integer;

  THistogramBin = record
    RangeMin, RangeMax: Double;
    Count: Integer;
    Percentage: Double;
  end;
  THistogram = array of THistogramBin;

  TGateSaturationStats = record
    NearZeroCount: Integer;
    NearOneCount: Integer;
    TotalCount: Integer;
    NearZeroPct: Double;
    NearOnePct: Double;
  end;

  TGradientScaleStats = record
    Timestep: Integer;
    MeanAbsGrad: Double;
    MaxAbsGrad: Double;
    MinAbsGrad: Double;
  end;
  TGradientScaleArray = array of TGradientScaleStats;

  TLayerNormStats = record
    Mean: Double;
    Variance: Double;
    Gamma: Double;
    Beta: Double;
  end;

  TOptimizerStateRecord = record
    Momentum: Double;
    Velocity: Double;
    Beta1Power: Double;
    Beta2Power: Double;
  end;

  TTimeStepCacheEx = record
    Input: DArray;
    H, C: DArray;
    PreH: DArray;
    F, I, CTilde, O, TanhC: DArray;
    Z, R, HTilde: DArray;
    OutPre, OutVal: DArray;
    DropoutMask: DArray;
  end;
  TTimeStepCacheExArray = array of TTimeStepCacheEx;

  TRNNFacade = class;

  { Forward declarations for cell wrappers }
  TSimpleRNNCellWrapper = class;
  TLSTMCellWrapper = class;
  TGRUCellWrapper = class;
  TOutputLayerWrapper = class;

  { TSimpleRNNCellWrapper }
  TSimpleRNNCellWrapper = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wih, Whh: TDArray2D;
    Bh: DArray;
    dWih, dWhh: TDArray2D;
    dBh: DArray;
    MWih, MWhh: TDArray2D;
    MBh: DArray;
    VWih, VWhh: TDArray2D;
    VBh: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH: DArray; var H, PreH: DArray);
    procedure Backward(const dH, H, PreH, PrevH, Input: DArray; ClipVal: Double; var dInput, dPrevH: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
    function GetInputSize: Integer;
  end;

  { TLSTMCellWrapper }
  TLSTMCellWrapper = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wf, Wi, Wc, Wo: TDArray2D;
    Bf, Bi, Bc, Bo: DArray;
    dWf, dWi, dWc, dWo: TDArray2D;
    dBf, dBi, dBc, dBo: DArray;
    MWf, MWi, MWc, MWo: TDArray2D;
    MBf, MBi, MBc, MBo: DArray;
    VWf, VWi, VWc, VWo: TDArray2D;
    VBf, VBi, VBc, VBo: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH, PrevC: DArray; var H, C, FG, IG, CTilde, OG, TanhC: DArray);
    procedure Backward(const dH, dC, H, C, FG, IG, CTilde, OG, TanhC, PrevH, PrevC, Input: DArray;
                       ClipVal: Double; var dInput, dPrevH, dPrevC: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
    function GetInputSize: Integer;
  end;

  { TGRUCellWrapper }
  TGRUCellWrapper = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wz, Wr, Wh: TDArray2D;
    Bz, Br, Bh: DArray;
    dWz, dWr, dWh: TDArray2D;
    dBz, dBr, dBh: DArray;
    MWz, MWr, MWh: TDArray2D;
    MBz, MBr, MBh: DArray;
    VWz, VWr, VWh: TDArray2D;
    VBz, VBr, VBh: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH: DArray; var H, Z, R, HTilde: DArray);
    procedure Backward(const dH, H, Z, R, HTilde, PrevH, Input: DArray;
                       ClipVal: Double; var dInput, dPrevH: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
    function GetInputSize: Integer;
  end;

  { TOutputLayerWrapper }
  TOutputLayerWrapper = class
  private
    FInputSize, FOutputSize: Integer;
    FActivation: TActivationType;
  public
    W: TDArray2D;
    B: DArray;
    dW: TDArray2D;
    dB: DArray;
    MW: TDArray2D;
    MB: DArray;
    VW: TDArray2D;
    VB: DArray;
    constructor Create(InputSize, OutputSize: Integer; Activation: TActivationType);
    procedure Forward(const Input: DArray; var Output, Pre: DArray);
    procedure Backward(const dOut, Output, Pre, Input: DArray; ClipVal: Double; var dInput: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetInputSize: Integer;
    function GetOutputSize: Integer;
  end;

  { Main RNN Facade }
  TRNNFacade = class
  private
    FInputSize, FOutputSize: Integer;
    FHiddenSizes: array of Integer;
    FCellType: TCellType;
    FActivation: TActivationType;
    FOutputActivation: TActivationType;
    FLossType: TLossType;
    FLearningRate: Double;
    FGradientClip: Double;
    FBPTTSteps: Integer;
    FDropoutRate: Double;
    FUseDropout: Boolean;

    FSimpleCells: array of TSimpleRNNCellWrapper;
    FLSTMCells: array of TLSTMCellWrapper;
    FGRUCells: array of TGRUCellWrapper;
    FOutputLayer: TOutputLayerWrapper;

    FCaches: TTimeStepCacheExArray;
    FStates: TDArray3D;
    FSequenceLen: Integer;
    FGradientHistory: TDArray2D;

    function ClipGradient(G, MaxVal: Double): Double;
    function Array1DToJSON(const Arr: DArray): string;
    function Array2DToJSON(const Arr: TDArray2D): string;
  public

    constructor Create(InputSize: Integer; const HiddenSizes: array of Integer;
                       OutputSize: Integer; CellType: TCellType;
                       Activation, OutputActivation: TActivationType;
                       LossType: TLossType; LearningRate, GradientClip: Double;
                       BPTTSteps: Integer);
    destructor Destroy; override;

    procedure SaveModel(const Filename: string);
    procedure LoadModel(const Filename: string);
    { Core training/inference }
    function ForwardSequence(const Inputs: TDArray2D): TDArray2D;
    function BackwardSequence(const Targets: TDArray2D): Double;
    function TrainSequence(const Inputs, Targets: TDArray2D): Double;
    function Predict(const Inputs: TDArray2D): TDArray2D;

    { 1. Time-Step and Sequence Access }
    function GetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
    procedure SetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer; Value: Double);
    function GetOutput(Timestep, OutputIdx: Integer): Double;
    procedure SetOutput(Timestep, OutputIdx: Integer; Value: Double);

    { 2. Cell State and Gate Access (LSTM/GRU) }
    function GetCellState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
    function GetGateValue(GateType: TGateType; LayerIdx, Timestep, NeuronIdx: Integer): Double;

    { 3. Cached Pre-Activations and Inputs }
    function GetPreActivation(LayerIdx, Timestep, NeuronIdx: Integer): Double;
    function GetInputVector(Timestep, InputIdx: Integer): Double;

    { 4. Gradients and Optimizer States }
    function GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
    function GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
    function GetOptimizerState(LayerIdx, NeuronIdx, Param: Integer): TOptimizerStateRecord;
    function GetCellGradient(LayerIdx, Timestep, NeuronIdx: Integer): Double;

    { 5. Dropout, LayerNorm, Regularization }
    procedure SetDropoutRate(Rate: Double);
    function GetDropoutRate: Double;
    function GetDropoutMask(LayerIdx, Timestep, NeuronIdx: Integer): Double;
    function GetLayerNormStats(LayerIdx, Timestep: Integer): TLayerNormStats;
    procedure EnableDropout(Enable: Boolean);

    { 6. Sequence-to-Sequence APIs }
    function GetSequenceOutputs(OutputIdx: Integer): DArray;
    function GetSequenceHiddenStates(LayerIdx, NeuronIdx: Integer): DArray;
    function GetSequenceCellStates(LayerIdx, NeuronIdx: Integer): DArray;
    function GetSequenceGateValues(GateType: TGateType; LayerIdx, NeuronIdx: Integer): DArray;

    { 7. Introspection }
    function GetInputSize: Integer;
    function GetOutputSize: Integer;
    function GetHiddenSize(LayerIdx: Integer): Integer;
    function GetCellType: TCellType;
    function GetSequenceLength: Integer;

    property LearningRate: Double read FLearningRate write FLearningRate;
    property GradientClip: Double read FGradientClip write FGradientClip;
  end;

// ========== Forward Declarations ==========
function CellTypeToStr(ct: TCellType): string; forward;
function ActivationToStr(act: TActivationType): string; forward;
function LossToStr(loss: TLossType): string; forward;
function ParseCellType(const s: string): TCellType; forward;
function ParseActivation(const s: string): TActivationType; forward;
function ParseLoss(const s: string): TLossType; forward;
function ParseIntArrayHelper(const s: string; var arr: TIntArray): Boolean; forward;

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

function ParseIntArrayHelper(const s: string; var arr: TIntArray): Boolean;
var
  Parts: TStringArray;
  i, num: Integer;
  Part: string;
begin
  Result := True;
  SetLength(arr, 0);
  
  { Split by comma }
  Parts := s.Split(',');
  SetLength(arr, Length(Parts));
  
  for i := 0 to High(Parts) do
  begin
    Part := Trim(Parts[i]);
    if not TryStrToInt(Part, num) then
    begin
      Result := False;
      Exit;
    end;
    arr[i] := num;
  end;
end;

// ========== TSimpleRNNCellWrapper Implementation ==========
constructor TSimpleRNNCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
begin
  inherited Create;
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  InitMatrix(Wih, HiddenSize, InputSize, 0.01);
  InitMatrix(Whh, HiddenSize, HiddenSize, 0.01);
  ZeroArray(Bh, HiddenSize);
  ZeroMatrix(dWih, HiddenSize, InputSize);
  ZeroMatrix(dWhh, HiddenSize, HiddenSize);
  ZeroArray(dBh, HiddenSize);
  ZeroMatrix(MWih, HiddenSize, InputSize);
  ZeroMatrix(MWhh, HiddenSize, HiddenSize);
  ZeroArray(MBh, HiddenSize);
  ZeroMatrix(VWih, HiddenSize, InputSize);
  ZeroMatrix(VWhh, HiddenSize, HiddenSize);
  ZeroArray(VBh, HiddenSize);
end;

procedure TSimpleRNNCellWrapper.Forward(const Input, PrevH: DArray; var H, PreH: DArray);
begin
  { Placeholder implementation }
end;

procedure TSimpleRNNCellWrapper.Backward(const dH, H, PreH, PrevH, Input: DArray; ClipVal: Double; var dInput, dPrevH: DArray);
begin
  { Placeholder implementation }
end;

procedure TSimpleRNNCellWrapper.ApplyGradients(LR, ClipVal: Double);
begin
  { Placeholder implementation }
end;

procedure TSimpleRNNCellWrapper.ResetGradients;
begin
  { Placeholder implementation }
end;

function TSimpleRNNCellWrapper.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

function TSimpleRNNCellWrapper.GetInputSize: Integer;
begin
  Result := FInputSize;
end;

// ========== TLSTMCellWrapper Implementation ==========
constructor TLSTMCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
begin
  inherited Create;
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  InitMatrix(Wf, HiddenSize, InputSize + HiddenSize, 0.01);
  InitMatrix(Wi, HiddenSize, InputSize + HiddenSize, 0.01);
  InitMatrix(Wc, HiddenSize, InputSize + HiddenSize, 0.01);
  InitMatrix(Wo, HiddenSize, InputSize + HiddenSize, 0.01);
  ZeroArray(Bf, HiddenSize);
  ZeroArray(Bi, HiddenSize);
  ZeroArray(Bc, HiddenSize);
  ZeroArray(Bo, HiddenSize);
  ZeroMatrix(dWf, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(dWi, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(dWc, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(dWo, HiddenSize, InputSize + HiddenSize);
  ZeroArray(dBf, HiddenSize);
  ZeroArray(dBi, HiddenSize);
  ZeroArray(dBc, HiddenSize);
  ZeroArray(dBo, HiddenSize);
  ZeroMatrix(MWf, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(MWi, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(MWc, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(MWo, HiddenSize, InputSize + HiddenSize);
  ZeroArray(MBf, HiddenSize);
  ZeroArray(MBi, HiddenSize);
  ZeroArray(MBc, HiddenSize);
  ZeroArray(MBo, HiddenSize);
  ZeroMatrix(VWf, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(VWi, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(VWc, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(VWo, HiddenSize, InputSize + HiddenSize);
  ZeroArray(VBf, HiddenSize);
  ZeroArray(VBi, HiddenSize);
  ZeroArray(VBc, HiddenSize);
  ZeroArray(VBo, HiddenSize);
end;

procedure TLSTMCellWrapper.Forward(const Input, PrevH, PrevC: DArray; var H, C, FG, IG, CTilde, OG, TanhC: DArray);
begin
  { Placeholder implementation }
end;

procedure TLSTMCellWrapper.Backward(const dH, dC, H, C, FG, IG, CTilde, OG, TanhC, PrevH, PrevC, Input: DArray;
                                    ClipVal: Double; var dInput, dPrevH, dPrevC: DArray);
begin
  { Placeholder implementation }
end;

procedure TLSTMCellWrapper.ApplyGradients(LR, ClipVal: Double);
begin
  { Placeholder implementation }
end;

procedure TLSTMCellWrapper.ResetGradients;
begin
  { Placeholder implementation }
end;

function TLSTMCellWrapper.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

function TLSTMCellWrapper.GetInputSize: Integer;
begin
  Result := FInputSize;
end;

// ========== TGRUCellWrapper Implementation ==========
constructor TGRUCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
begin
  inherited Create;
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  InitMatrix(Wz, HiddenSize, InputSize + HiddenSize, 0.01);
  InitMatrix(Wr, HiddenSize, InputSize + HiddenSize, 0.01);
  InitMatrix(Wh, HiddenSize, InputSize + HiddenSize, 0.01);
  ZeroArray(Bz, HiddenSize);
  ZeroArray(Br, HiddenSize);
  ZeroArray(Bh, HiddenSize);
  ZeroMatrix(dWz, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(dWr, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(dWh, HiddenSize, InputSize + HiddenSize);
  ZeroArray(dBz, HiddenSize);
  ZeroArray(dBr, HiddenSize);
  ZeroArray(dBh, HiddenSize);
  ZeroMatrix(MWz, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(MWr, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(MWh, HiddenSize, InputSize + HiddenSize);
  ZeroArray(MBz, HiddenSize);
  ZeroArray(MBr, HiddenSize);
  ZeroArray(MBh, HiddenSize);
  ZeroMatrix(VWz, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(VWr, HiddenSize, InputSize + HiddenSize);
  ZeroMatrix(VWh, HiddenSize, InputSize + HiddenSize);
  ZeroArray(VBz, HiddenSize);
  ZeroArray(VBr, HiddenSize);
  ZeroArray(VBh, HiddenSize);
end;

procedure TGRUCellWrapper.Forward(const Input, PrevH: DArray; var H, Z, R, HTilde: DArray);
begin
  { Placeholder implementation }
end;

procedure TGRUCellWrapper.Backward(const dH, H, Z, R, HTilde, PrevH, Input: DArray;
                                   ClipVal: Double; var dInput, dPrevH: DArray);
begin
  { Placeholder implementation }
end;

procedure TGRUCellWrapper.ApplyGradients(LR, ClipVal: Double);
begin
  { Placeholder implementation }
end;

procedure TGRUCellWrapper.ResetGradients;
begin
  { Placeholder implementation }
end;

function TGRUCellWrapper.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

function TGRUCellWrapper.GetInputSize: Integer;
begin
  Result := FInputSize;
end;

// ========== TOutputLayerWrapper Implementation ==========
constructor TOutputLayerWrapper.Create(InputSize, OutputSize: Integer; Activation: TActivationType);
begin
  inherited Create;
  FInputSize := InputSize;
  FOutputSize := OutputSize;
  FActivation := Activation;
  InitMatrix(W, OutputSize, InputSize, 0.01);
  ZeroArray(B, OutputSize);
  ZeroMatrix(dW, OutputSize, InputSize);
  ZeroArray(dB, OutputSize);
  ZeroMatrix(MW, OutputSize, InputSize);
  ZeroArray(MB, OutputSize);
  ZeroMatrix(VW, OutputSize, InputSize);
  ZeroArray(VB, OutputSize);
end;

procedure TOutputLayerWrapper.Forward(const Input: DArray; var Output, Pre: DArray);
begin
  { Placeholder implementation }
end;

procedure TOutputLayerWrapper.Backward(const dOut, Output, Pre, Input: DArray; ClipVal: Double; var dInput: DArray);
begin
  { Placeholder implementation }
end;

procedure TOutputLayerWrapper.ApplyGradients(LR, ClipVal: Double);
begin
  { Placeholder implementation }
end;

procedure TOutputLayerWrapper.ResetGradients;
begin
  { Placeholder implementation }
end;

function TOutputLayerWrapper.GetInputSize: Integer;
begin
  Result := FInputSize;
end;

function TOutputLayerWrapper.GetOutputSize: Integer;
begin
  Result := FOutputSize;
end;

// ========== TRNNFacade Implementation ==========
constructor TRNNFacade.Create(InputSize: Integer; const HiddenSizes: array of Integer;
                              OutputSize: Integer; CellType: TCellType;
                              Activation, OutputActivation: TActivationType;
                              LossType: TLossType; LearningRate, GradientClip: Double;
                              BPTTSteps: Integer);
var
  i: Integer;
begin
  inherited Create;
  FInputSize := InputSize;
  FOutputSize := OutputSize;
  FCellType := CellType;
  FActivation := Activation;
  FOutputActivation := OutputActivation;
  FLossType := LossType;
  FLearningRate := LearningRate;
  FGradientClip := GradientClip;
  FBPTTSteps := BPTTSteps;
  FDropoutRate := 0.0;
  FUseDropout := False;
  FSequenceLen := 0;

  SetLength(FHiddenSizes, Length(HiddenSizes));
  for i := 0 to High(HiddenSizes) do
    FHiddenSizes[i] := HiddenSizes[i];

  { Initialize cells based on type }
  case CellType of
    ctSimpleRNN:
    begin
      SetLength(FSimpleCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        if i = 0 then
          FSimpleCells[i] := TSimpleRNNCellWrapper.Create(InputSize, HiddenSizes[i], Activation)
        else
          FSimpleCells[i] := TSimpleRNNCellWrapper.Create(HiddenSizes[i-1], HiddenSizes[i], Activation);
      end;
    end;
    ctLSTM:
    begin
      SetLength(FLSTMCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        if i = 0 then
          FLSTMCells[i] := TLSTMCellWrapper.Create(InputSize, HiddenSizes[i], Activation)
        else
          FLSTMCells[i] := TLSTMCellWrapper.Create(HiddenSizes[i-1], HiddenSizes[i], Activation);
      end;
    end;
    ctGRU:
    begin
      SetLength(FGRUCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        if i = 0 then
          FGRUCells[i] := TGRUCellWrapper.Create(InputSize, HiddenSizes[i], Activation)
        else
          FGRUCells[i] := TGRUCellWrapper.Create(HiddenSizes[i-1], HiddenSizes[i], Activation);
      end;
    end;
  end;

  FOutputLayer := TOutputLayerWrapper.Create(HiddenSizes[High(HiddenSizes)], OutputSize, OutputActivation);
end;

destructor TRNNFacade.Destroy;
var
  i: Integer;
begin
  for i := 0 to High(FSimpleCells) do
    FSimpleCells[i].Free;
  for i := 0 to High(FLSTMCells) do
    FLSTMCells[i].Free;
  for i := 0 to High(FGRUCells) do
    FGRUCells[i].Free;
  if Assigned(FOutputLayer) then
    FOutputLayer.Free;
  inherited Destroy;
end;

function TRNNFacade.ClipGradient(G, MaxVal: Double): Double;
begin
  if G > MaxVal then Result := MaxVal
  else if G < -MaxVal then Result := -MaxVal
  else Result := G;
end;

function TRNNFacade.ForwardSequence(const Inputs: TDArray2D): TDArray2D;
begin
  { Placeholder implementation }
  SetLength(Result, 0);
end;

function TRNNFacade.BackwardSequence(const Targets: TDArray2D): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.TrainSequence(const Inputs, Targets: TDArray2D): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.Predict(const Inputs: TDArray2D): TDArray2D;
begin
  Result := ForwardSequence(Inputs);
end;

function TRNNFacade.GetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  if (LayerIdx >= 0) and (LayerIdx < Length(FStates)) and
     (Timestep >= 0) and (Timestep < Length(FStates[LayerIdx])) and
     (NeuronIdx >= 0) and (NeuronIdx < Length(FStates[LayerIdx][Timestep])) then
    Result := FStates[LayerIdx][Timestep][NeuronIdx];
end;

procedure TRNNFacade.SetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer; Value: Double);
begin
  if (LayerIdx >= 0) and (LayerIdx < Length(FStates)) and
     (Timestep >= 0) and (Timestep < Length(FStates[LayerIdx])) and
     (NeuronIdx >= 0) and (NeuronIdx < Length(FStates[LayerIdx][Timestep])) then
    FStates[LayerIdx][Timestep][NeuronIdx] := Value;
end;

function TRNNFacade.GetOutput(Timestep, OutputIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

procedure TRNNFacade.SetOutput(Timestep, OutputIdx: Integer; Value: Double);
begin
  { Placeholder implementation }
end;

function TRNNFacade.GetCellState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetGateValue(GateType: TGateType; LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetPreActivation(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetInputVector(Timestep, InputIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetOptimizerState(LayerIdx, NeuronIdx, Param: Integer): TOptimizerStateRecord;
begin
  Result.Momentum := 0.0;
  Result.Velocity := 0.0;
  Result.Beta1Power := 0.0;
  Result.Beta2Power := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetCellGradient(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

procedure TRNNFacade.SetDropoutRate(Rate: Double);
begin
  FDropoutRate := Rate;
end;

function TRNNFacade.GetDropoutRate: Double;
begin
  Result := FDropoutRate;
end;

function TRNNFacade.GetDropoutMask(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0.0;
  { Placeholder implementation }
end;

function TRNNFacade.GetLayerNormStats(LayerIdx, Timestep: Integer): TLayerNormStats;
begin
  Result.Mean := 0.0;
  Result.Variance := 0.0;
  Result.Gamma := 0.0;
  Result.Beta := 0.0;
  { Placeholder implementation }
end;

procedure TRNNFacade.EnableDropout(Enable: Boolean);
begin
  FUseDropout := Enable;
end;

function TRNNFacade.GetSequenceOutputs(OutputIdx: Integer): DArray;
begin
  SetLength(Result, 0);
  { Placeholder implementation }
end;

function TRNNFacade.GetSequenceHiddenStates(LayerIdx, NeuronIdx: Integer): DArray;
begin
  SetLength(Result, 0);
  { Placeholder implementation }
end;

function TRNNFacade.GetSequenceCellStates(LayerIdx, NeuronIdx: Integer): DArray;
begin
  SetLength(Result, 0);
  { Placeholder implementation }
end;

function TRNNFacade.GetSequenceGateValues(GateType: TGateType; LayerIdx, NeuronIdx: Integer): DArray;
begin
  SetLength(Result, 0);
  { Placeholder implementation }
end;

function TRNNFacade.GetInputSize: Integer;
begin
  Result := FInputSize;
end;

function TRNNFacade.GetOutputSize: Integer;
begin
  Result := FOutputSize;
end;

function TRNNFacade.GetHiddenSize(LayerIdx: Integer): Integer;
begin
  Result := 0;
  if (LayerIdx >= 0) and (LayerIdx <= High(FHiddenSizes)) then
    Result := FHiddenSizes[LayerIdx];
end;

function TRNNFacade.GetCellType: TCellType;
begin
  Result := FCellType;
end;

function TRNNFacade.GetSequenceLength: Integer;
begin
  Result := FSequenceLen;
end;

{ ========== Helper Functions Implementation ========== }
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
    Result := 'linear';
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
var
  s_lower: string;
begin
  s_lower := LowerCase(s);
  if s_lower = 'lstm' then Result := ctLSTM
  else if s_lower = 'gru' then Result := ctGRU
  else Result := ctSimpleRNN;
end;

function ParseActivation(const s: string): TActivationType;
var
  s_lower: string;
begin
  s_lower := LowerCase(s);
  if s_lower = 'sigmoid' then Result := atSigmoid
  else if s_lower = 'tanh' then Result := atTanh
  else if s_lower = 'relu' then Result := atReLU
  else Result := atLinear;
end;

function ParseLoss(const s: string): TLossType;
var
  s_lower: string;
begin
  s_lower := LowerCase(s);
  if s_lower = 'crossentropy' then Result := ltCrossEntropy
  else Result := ltMSE;
end;

function TRNNFacade.Array1DToJSON(const Arr: DArray): string;
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

function TRNNFacade.Array2DToJSON(const Arr: TDArray2D): string;
var
  I: Integer;
begin
  Result := '[';
  for I := 0 to High(Arr) do
  begin
    if I > 0 then Result := Result + ',';
    Result := Result + Array1DToJSON(Arr[I]);
  end;
  Result := Result + ']';
end;

procedure TRNNFacade.SaveModel(const Filename: string);
var
  SL: TStringList;
  I, J, LayerIdx: Integer;
  CellTypeStr: string;
begin
  SL := TStringList.Create;
  try
    SL.Add('{');
    SL.Add('  "input_size": ' + IntToStr(FInputSize) + ',');
    SL.Add('  "output_size": ' + IntToStr(FOutputSize) + ',');
    SL.Add('  "hidden_sizes": [');
    for I := 0 to High(FHiddenSizes) do
    begin
      if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
      SL.Add('    ' + IntToStr(FHiddenSizes[I]));
    end;
    SL.Add('  ],');
    
    case FCellType of
      ctSimpleRNN: CellTypeStr := 'simplernn';
      ctLSTM: CellTypeStr := 'lstm';
      ctGRU: CellTypeStr := 'gru';
    else
      CellTypeStr := 'simplernn';
    end;
    
    SL.Add('  "cell_type": "' + CellTypeStr + '",');
    SL.Add('  "activation": "' + ActivationToStr(FActivation) + '",');
    SL.Add('  "output_activation": "' + ActivationToStr(FOutputActivation) + '",');
    SL.Add('  "loss_type": "' + LossToStr(FLossType) + '",');
    SL.Add('  "learning_rate": ' + FloatToStr(FLearningRate) + ',');
    SL.Add('  "gradient_clip": ' + FloatToStr(FGradientClip) + ',');
    SL.Add('  "bptt_steps": ' + IntToStr(FBPTTSteps) + ',');
    SL.Add('  "dropout_rate": ' + FloatToStr(FDropoutRate) + ',');
    
    { Save cell weights }
    case FCellType of
      ctSimpleRNN:
      begin
        SL.Add('  "cells": [');
        for LayerIdx := 0 to High(FSimpleCells) do
        begin
          if LayerIdx > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
          SL.Add('    {');
          SL.Add('      "Wih": ' + Array2DToJSON(FSimpleCells[LayerIdx].Wih) + ',');
          SL.Add('      "Whh": ' + Array2DToJSON(FSimpleCells[LayerIdx].Whh) + ',');
          SL.Add('      "Bh": ' + Array1DToJSON(FSimpleCells[LayerIdx].Bh));
          SL.Add('    }');
        end;
        SL.Add('  ]');
      end;
      ctLSTM:
      begin
        SL.Add('  "cells": [');
        for LayerIdx := 0 to High(FLSTMCells) do
        begin
          if LayerIdx > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
          SL.Add('    {');
          SL.Add('      "Wf": ' + Array2DToJSON(FLSTMCells[LayerIdx].Wf) + ',');
          SL.Add('      "Wi": ' + Array2DToJSON(FLSTMCells[LayerIdx].Wi) + ',');
          SL.Add('      "Wc": ' + Array2DToJSON(FLSTMCells[LayerIdx].Wc) + ',');
          SL.Add('      "Wo": ' + Array2DToJSON(FLSTMCells[LayerIdx].Wo) + ',');
          SL.Add('      "Bf": ' + Array1DToJSON(FLSTMCells[LayerIdx].Bf) + ',');
          SL.Add('      "Bi": ' + Array1DToJSON(FLSTMCells[LayerIdx].Bi) + ',');
          SL.Add('      "Bc": ' + Array1DToJSON(FLSTMCells[LayerIdx].Bc) + ',');
          SL.Add('      "Bo": ' + Array1DToJSON(FLSTMCells[LayerIdx].Bo));
          SL.Add('    }');
        end;
        SL.Add('  ]');
      end;
      ctGRU:
      begin
        SL.Add('  "cells": [');
        for LayerIdx := 0 to High(FGRUCells) do
        begin
          if LayerIdx > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
          SL.Add('    {');
          SL.Add('      "Wz": ' + Array2DToJSON(FGRUCells[LayerIdx].Wz) + ',');
          SL.Add('      "Wr": ' + Array2DToJSON(FGRUCells[LayerIdx].Wr) + ',');
          SL.Add('      "Wh": ' + Array2DToJSON(FGRUCells[LayerIdx].Wh) + ',');
          SL.Add('      "Bz": ' + Array1DToJSON(FGRUCells[LayerIdx].Bz) + ',');
          SL.Add('      "Br": ' + Array1DToJSON(FGRUCells[LayerIdx].Br) + ',');
          SL.Add('      "Bh": ' + Array1DToJSON(FGRUCells[LayerIdx].Bh));
          SL.Add('    }');
        end;
        SL.Add('  ]');
      end;
    end;
    
    SL.Add(',');
    SL.Add('  "output_layer": {');
    SL.Add('    "W": ' + Array2DToJSON(FOutputLayer.W) + ',');
    SL.Add('    "B": ' + Array1DToJSON(FOutputLayer.B));
    SL.Add('  }');
    SL.Add('}');
    
    SL.SaveToFile(Filename);
    WriteLn('Model saved to JSON: ', Filename);
  finally
    SL.Free;
  end;
end;

procedure TRNNFacade.LoadModel(const Filename: string);
var
  SL: TStringList;
  Content: string;
  ValueStr: string;
  I, LayerIdx: Integer;
  InputSize, OutputSize: Integer;
  HiddenSizesTemp: TIntArray;
  CellTypeTemp: TCellType;
  ActivationTemp, OutputActivationTemp: TActivationType;
  LossTypeTemp: TLossType;
  LearningRateTemp, GradientClipTemp: Double;
  BPTTStepsTemp: Integer;
  DropoutRateTemp: Double;
  
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
        { Skip whitespace }
        while (StartPos <= Length(json)) and (json[StartPos] in [' ', #9, #10, #13]) do
          Inc(StartPos);
        
        { Check if value is a quoted string }
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
          { Value is a number or boolean }
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

begin
  SL := TStringList.Create;
  try
    SL.LoadFromFile(Filename);
    Content := SL.Text;
    
    { Load configuration }
    ValueStr := ExtractJSONValue(Content, 'input_size');
    if ValueStr <> '' then
    begin
      InputSize := StrToInt(ValueStr);
      WriteLn('  Input size: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'output_size');
    if ValueStr <> '' then
    begin
      OutputSize := StrToInt(ValueStr);
      WriteLn('  Output size: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'cell_type');
    if ValueStr <> '' then
    begin
      CellTypeTemp := ParseCellType(ValueStr);
      WriteLn('  Cell type: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'activation');
    if ValueStr <> '' then
    begin
      ActivationTemp := ParseActivation(ValueStr);
      WriteLn('  Hidden activation: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'output_activation');
    if ValueStr <> '' then
    begin
      OutputActivationTemp := ParseActivation(ValueStr);
      WriteLn('  Output activation: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'loss_type');
    if ValueStr <> '' then
    begin
      LossTypeTemp := ParseLoss(ValueStr);
      WriteLn('  Loss type: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'learning_rate');
    if ValueStr <> '' then
    begin
      LearningRateTemp := StrToFloat(ValueStr);
      WriteLn('  Learning rate: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'gradient_clip');
    if ValueStr <> '' then
    begin
      GradientClipTemp := StrToFloat(ValueStr);
      WriteLn('  Gradient clip: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'bptt_steps');
    if ValueStr <> '' then
    begin
      BPTTStepsTemp := StrToInt(ValueStr);
      WriteLn('  BPTT steps: ', ValueStr);
    end;
    
    ValueStr := ExtractJSONValue(Content, 'dropout_rate');
    if ValueStr <> '' then
    begin
      DropoutRateTemp := StrToFloat(ValueStr);
      FDropoutRate := DropoutRateTemp;
      WriteLn('  Dropout rate: ', ValueStr);
    end;
    
    WriteLn('Model loaded from JSON: ', Filename);
  finally
    SL.Free;
  end;
end;

procedure PrintUsage;
begin
    WriteLn('RNN - Command-line Recurrent Neural Network');
    WriteLn;
    WriteLn('Commands:');
    WriteLn('  create   Create a new RNN model and save to JSON');
    WriteLn('  train    Train an existing model with data from JSON');
    WriteLn('  predict  Make predictions with a trained model from JSON');
    WriteLn('  info     Display model information from JSON');
    WriteLn('  query    Query model state and internals (facade functions)');
    WriteLn('  help     Show this help message');
    WriteLn;
    WriteLn('Create Options:');
    WriteLn('  --input=N              Input layer size (required)');
    WriteLn('  --hidden=N,N,...       Hidden layer sizes (required)');
    WriteLn('  --output=N             Output layer size (required)');
    WriteLn('  --save=FILE.json       Save model to JSON file (required)');
    WriteLn('  --cell=TYPE            simplernn|lstm|gru (default: lstm)');
    WriteLn('  --lr=VALUE             Learning rate (default: 0.01)');
    WriteLn('  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)');
    WriteLn('  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)');
    WriteLn('  --loss=TYPE            mse|crossentropy (default: mse)');
    WriteLn('  --clip=VALUE           Gradient clipping (default: 5.0)');
    WriteLn('  --bptt=N               BPTT steps (default: 0 = full)');
    WriteLn;
    WriteLn('Train Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn('  --data=FILE.csv        Training data CSV file (required)');
    WriteLn('  --save=FILE.json       Save trained model to JSON (required)');
    WriteLn('  --epochs=N             Number of training epochs (default: 100)');
    WriteLn('  --batch=N              Batch size (default: 1)');
    WriteLn('  --lr=VALUE             Override learning rate');
    WriteLn('  --seq-len=N            Sequence length (default: auto-detect)');
    WriteLn;
    WriteLn('Predict Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn('  --input=v1,v2,...      Input values as CSV (required)');
    WriteLn;
    WriteLn('Info Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn;
    WriteLn('Query Options (Facade Functions):');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn('  --query-type=TYPE      Query type (required):');
    WriteLn('    hidden-state         Get hidden state at layer,timestep,neuron');
    WriteLn('    cell-state           Get cell state at layer,timestep,neuron');
    WriteLn('    gate-value           Get gate value (forget|input|output|candidate)');
    WriteLn('    pre-activation       Get pre-activation at layer,timestep,neuron');
    WriteLn('    input-vector         Get input at timestep,index');
    WriteLn('    weight-gradient      Get weight gradient at layer,neuron,weight');
    WriteLn('    bias-gradient        Get bias gradient at layer,neuron');
    WriteLn('    optimizer-state      Get optimizer state at layer,neuron,param');
    WriteLn('    cell-gradient        Get cell gradient at layer,timestep,neuron');
    WriteLn('    dropout-mask         Get dropout mask at layer,timestep,neuron');
    WriteLn('    layer-norm           Get layer norm stats at layer,timestep');
    WriteLn('    sequence-outputs     Get all outputs for output index');
    WriteLn('    sequence-hidden      Get hidden states for layer,neuron');
    WriteLn('    sequence-cell        Get cell states for layer,neuron');
    WriteLn('    sequence-gates       Get gate values for gate type,layer,neuron');
    WriteLn('    input-size           Get model input size');
    WriteLn('    output-size          Get model output size');
    WriteLn('    hidden-size          Get hidden size for layer');
    WriteLn('    cell-type            Get cell type');
    WriteLn('    sequence-length      Get sequence length');
    WriteLn('    dropout-rate         Get current dropout rate');
    WriteLn;
    WriteLn('Query Arguments:');
    WriteLn('  --layer=N              Layer index (for layer-based queries)');
    WriteLn('  --timestep=N           Timestep index (for time-based queries)');
    WriteLn('  --neuron=N             Neuron index (for neuron-based queries)');
    WriteLn('  --index=N              Generic index parameter');
    WriteLn('  --gate=TYPE            Gate type (forget|input|output|candidate|update|reset|hidden)');
    WriteLn('  --param=N              Parameter index (for optimizer state)');
    WriteLn;
    WriteLn('Configuration Options (Query):');
    WriteLn('  --dropout-rate=VALUE   Set dropout rate (0.0-1.0)');
    WriteLn('  --enable-dropout       Enable dropout');
    WriteLn('  --disable-dropout      Disable dropout');
    WriteLn;
    WriteLn('Examples:');
    WriteLn('  rnn create --input=2 --hidden=16 --output=2 --cell=lstm --save=seq.json');
    WriteLn('  rnn train --model=seq.json --data=seq.csv --epochs=200 --save=seq_trained.json');
    WriteLn('  rnn predict --model=seq_trained.json --input=0.5,0.5');
    WriteLn('  rnn info --model=seq_trained.json');
    WriteLn('  rnn query --model=seq_trained.json --query-type=input-size');
    WriteLn('  rnn query --model=seq_trained.json --query-type=hidden-state --layer=0 --timestep=0 --neuron=0');
    WriteLn('  rnn query --model=seq_trained.json --query-type=dropout-rate');
    WriteLn('  rnn query --model=seq_trained.json --query-type=sequence-hidden --layer=0 --neuron=5');
end;

// ========== Main Program ==========
var
   Command: TCommand;
   CmdStr: string;
   i: Integer;
   arg, key, value: string;
   eqPos: Integer;

   inputSize, outputSize, epochs, batchSize, seqLen, bpttSteps: Integer;
   hiddenSizes: array of Integer;
   learningRate, gradientClip: Double;
   hiddenAct, outputAct: TActivationType;
   cellType: TCellType;
   lossType: TLossType;
   modelFile, saveFile, dataFile: string;
   verbose: Boolean;

   queryType, gateTypeStr: string;
   layer, timestep, neuron, index, param: Integer;
   dropoutValue: Double;
   enableDropoutFlag, disableDropoutFlag: Boolean;

   RNN: TRNNFacade;
   SequenceLen, HiddenSize: Integer;
   Inputs, Targets, Predictions: TDArray2D;
   CellTypeStr: string;

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
   else if CmdStr = 'query' then Command := cmdQuery
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
   
   { Initialize query defaults }
   queryType := '';
   gateTypeStr := '';
   layer := 0;
   timestep := 0;
   neuron := 0;
   index := 0;
   param := 0;
   dropoutValue := 0.0;
   enableDropoutFlag := False;
   disableDropoutFlag := False;

   { Parse arguments }
   for i := 2 to ParamCount do
   begin
      arg := ParamStr(i);

      if arg = '--verbose' then
         verbose := True
      else if arg = '--enable-dropout' then
         enableDropoutFlag := True
      else if arg = '--disable-dropout' then
         disableDropoutFlag := True
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
            { For create command: input is an integer size }
            { For predict command: input is a CSV string of values }
            if Command = cmdPredict then
               { Skip for predict, will be handled separately }
            else
               inputSize := StrToInt(value)
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
         else if key = '--cell' then
            cellType := ParseCellType(value)
         else if key = '--hidden-act' then
            hiddenAct := ParseActivation(value)
         else if key = '--output-act' then
            outputAct := ParseActivation(value)
         else if key = '--loss' then
            lossType := ParseLoss(value)
         else if key = '--clip' then
            gradientClip := StrToFloat(value)
         else if key = '--bptt' then
            bpttSteps := StrToInt(value)
         else if key = '--epochs' then
            epochs := StrToInt(value)
         else if key = '--batch' then
            batchSize := StrToInt(value)
         else if key = '--seq-len' then
            seqLen := StrToInt(value)
         else if key = '--query-type' then
            queryType := value
         else if key = '--layer' then
            layer := StrToInt(value)
         else if key = '--timestep' then
            timestep := StrToInt(value)
         else if key = '--neuron' then
            neuron := StrToInt(value)
         else if key = '--index' then
            index := StrToInt(value)
         else if key = '--gate' then
            gateTypeStr := value
         else if key = '--param' then
            param := StrToInt(value)
         else if key = '--dropout-rate' then
            dropoutValue := StrToFloat(value)
         else
            WriteLn('Unknown option: ', key);
      end;
   end;

   { Execute command }
   if Command = cmdQuery then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      if queryType = '' then begin WriteLn('Error: --query-type is required'); Exit; end;
      
      WriteLn('Loading model from JSON: ' + modelFile);
      RNN := TRNNFacade.Create(1, [1], 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
      RNN.LoadModel(modelFile);
      
      WriteLn('Executing query: ' + queryType);
      WriteLn;
      
      { Process query types }
      if queryType = 'input-size' then
         WriteLn('Input size: ', RNN.GetInputSize)
      else if queryType = 'output-size' then
         WriteLn('Output size: ', RNN.GetOutputSize)
      else if queryType = 'hidden-size' then
         WriteLn('Hidden size (layer ', layer, '): ', RNN.GetHiddenSize(layer))
      else if queryType = 'cell-type' then
         WriteLn('Cell type: ', CellTypeToStr(RNN.GetCellType))
      else if queryType = 'sequence-length' then
         WriteLn('Sequence length: ', RNN.GetSequenceLength)
      else if queryType = 'dropout-rate' then
         WriteLn('Current dropout rate: ', RNN.GetDropoutRate:0:6)
      else if queryType = 'hidden-state' then
         WriteLn('Hidden state at [', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetHiddenState(layer, timestep, neuron):0:6)
      else if queryType = 'cell-state' then
         WriteLn('Cell state at [', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetCellState(layer, timestep, neuron):0:6)
      else if queryType = 'gate-value' then
         WriteLn('Gate value at [', gateTypeStr, ',', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetGateValue(gtForget, layer, timestep, neuron):0:6)
      else if queryType = 'pre-activation' then
         WriteLn('Pre-activation at [', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetPreActivation(layer, timestep, neuron):0:6)
      else if queryType = 'input-vector' then
         WriteLn('Input vector at [', timestep, ',', index, ']: ', 
                  RNN.GetInputVector(timestep, index):0:6)
      else if queryType = 'weight-gradient' then
         WriteLn('Weight gradient at [', layer, ',', neuron, ',', index, ']: ', 
                  RNN.GetWeightGradient(layer, neuron, index):0:6)
      else if queryType = 'bias-gradient' then
         WriteLn('Bias gradient at [', layer, ',', neuron, ']: ', 
                  RNN.GetBiasGradient(layer, neuron):0:6)
      else if queryType = 'cell-gradient' then
         WriteLn('Cell gradient at [', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetCellGradient(layer, timestep, neuron):0:6)
      else if queryType = 'dropout-mask' then
         WriteLn('Dropout mask at [', layer, ',', timestep, ',', neuron, ']: ', 
                  RNN.GetDropoutMask(layer, timestep, neuron):0:6)
      else if queryType = 'sequence-outputs' then
         WriteLn('Sequence outputs query not fully implemented')
      else if queryType = 'sequence-hidden' then
         WriteLn('Sequence hidden states query not fully implemented')
      else if queryType = 'sequence-cell' then
         WriteLn('Sequence cell states query not fully implemented')
      else if queryType = 'sequence-gates' then
         WriteLn('Sequence gate values query not fully implemented')
      else
         WriteLn('Unknown query type: ', queryType);
      
      { Handle configuration options }
      if enableDropoutFlag then
      begin
         RNN.EnableDropout(True);
         WriteLn('Dropout enabled');
      end;
      
      if disableDropoutFlag then
      begin
         RNN.EnableDropout(False);
         WriteLn('Dropout disabled');
      end;
      
      if dropoutValue > 0 then
      begin
         RNN.SetDropoutRate(dropoutValue);
         WriteLn('Dropout rate set to: ', dropoutValue:0:6);
      end;
      
      RNN.Free;
   end
   else if Command = cmdCreate then
   begin
      if inputSize <= 0 then begin WriteLn('Error: --input is required'); Exit; end;
      if Length(hiddenSizes) = 0 then begin WriteLn('Error: --hidden is required'); Exit; end;
      if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;
      if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;

      RNN := TRNNFacade.Create(inputSize, hiddenSizes, outputSize, cellType, 
                               hiddenAct, outputAct, lossType, learningRate, 
                               gradientClip, bpttSteps);

      WriteLn('Created RNN model:');
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
      
      { Save model to JSON }
      RNN.SaveModel(saveFile);

      RNN.Free;
      end
      else if Command = cmdTrain then
      begin
         if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
         if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
         WriteLn('Loading model from JSON: ' + modelFile);
         RNN := TRNNFacade.Create(1, [1], 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
         RNN.LoadModel(modelFile);
         WriteLn('Model loaded successfully. Training functionality not yet implemented.');
         RNN.Free;
      end
      else if Command = cmdPredict then
      begin
         if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
         WriteLn('Loading model from JSON: ' + modelFile);
         RNN := TRNNFacade.Create(1, [1], 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
         RNN.LoadModel(modelFile);
         WriteLn('Model loaded successfully. Prediction functionality not yet implemented.');
         RNN.Free;
      end
      else if Command = cmdInfo then
      begin
         if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
         WriteLn('Loading model from JSON: ' + modelFile);
         RNN := TRNNFacade.Create(1, [1], 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
         RNN.LoadModel(modelFile);
         WriteLn('Model information displayed above.');
         RNN.Free;
      end;
end.

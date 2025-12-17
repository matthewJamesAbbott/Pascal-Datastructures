{
  RNN Facade - Comprehensive introspection and manipulation API
  Matthew Abbott 2025
}

{$mode objfpc}{$H+}

unit RNNFacade;

interface

uses Classes, Math, SysUtils;

type
  TActivationType = (atSigmoid, atTanh, atReLU, atLinear);
  TLossType = (ltMSE, ltCrossEntropy);
  TCellType = (ctSimpleRNN, ctLSTM, ctGRU);
  TGateType = (gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate);

  DArray = array of Double;
  TDArray2D = array of DArray;
  TDArray3D = array of TDArray2D;

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
  public
    constructor Create(InputSize: Integer; const HiddenSizes: array of Integer;
                       OutputSize: Integer; CellType: TCellType;
                       Activation, OutputActivation: TActivationType;
                       LossType: TLossType; LearningRate, GradientClip: Double;
                       BPTTSteps: Integer);
    destructor Destroy; override;

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

    { 7. Reset and Manipulate Hidden States }
    procedure ResetHiddenState(LayerIdx: Integer; Value: Double = 0.0);
    procedure ResetCellState(LayerIdx: Integer; Value: Double = 0.0);
    procedure ResetAllStates(Value: Double = 0.0);
    procedure InjectHiddenState(LayerIdx: Integer; const ValuesArray: DArray);
    procedure InjectCellState(LayerIdx: Integer; const ValuesArray: DArray);

    { 8. Attention Introspection (placeholder for future) }
    function GetAttentionWeights(Timestep: Integer): DArray;
    function GetAttentionContext(Timestep: Integer): DArray;

    { 9. Time-Series Diagnostics }
    function GetHiddenStateHistogram(LayerIdx, Timestep: Integer; NumBins: Integer = 10): THistogram;
    function GetActivationHistogramOverTime(LayerIdx, NeuronIdx: Integer; NumBins: Integer = 10): THistogram;
    function GetGateSaturation(GateType: TGateType; LayerIdx, Timestep: Integer; Threshold: Double = 0.05): TGateSaturationStats;
    function GetGradientScalesOverTime(LayerIdx: Integer): TGradientScaleArray;
    function DetectVanishingGradient(LayerIdx: Integer; Threshold: Double = 1e-6): Boolean;
    function DetectExplodingGradient(LayerIdx: Integer; Threshold: Double = 1e6): Boolean;

    { Utility }
    procedure ResetGradients;
    procedure ApplyGradients;
    function InitHiddenStates: TDArray3D;
    function GetLayerCount: Integer;
    function GetHiddenSize(LayerIdx: Integer): Integer;
    function GetCellType: TCellType;
    function GetSequenceLength: Integer;

    property LearningRate: Double read FLearningRate write FLearningRate;
    property GradientClip: Double read FGradientClip write FGradientClip;
    property DropoutRate: Double read FDropoutRate write FDropoutRate;
  end;

implementation

{ Utility functions }
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
var i, j: Integer;
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
var i, j: Integer;
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
var i: Integer;
begin
  SetLength(A, Size);
  for i := 0 to Size - 1 do
    A[i] := 0.0;
end;

function ConcatArrays(const A, B: DArray): DArray;
var i: Integer;
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

{ TSimpleRNNCellWrapper }
constructor TSimpleRNNCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var Scale: Double;
begin
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  Scale := Sqrt(2.0 / (InputSize + HiddenSize));
  InitMatrix(Wih, HiddenSize, InputSize, Scale);
  InitMatrix(Whh, HiddenSize, HiddenSize, Scale);
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
var i, j: Integer; Sum: Double;
begin
  SetLength(H, FHiddenSize);
  SetLength(PreH, FHiddenSize);
  for i := 0 to FHiddenSize - 1 do
  begin
    Sum := Bh[i];
    for j := 0 to FInputSize - 1 do
      Sum := Sum + Wih[i][j] * Input[j];
    for j := 0 to FHiddenSize - 1 do
      Sum := Sum + Whh[i][j] * PrevH[j];
    PreH[i] := Sum;
    H[i] := ApplyActivation(Sum, FActivation);
  end;
end;

procedure TSimpleRNNCellWrapper.Backward(const dH, H, PreH, PrevH, Input: DArray; ClipVal: Double; var dInput, dPrevH: DArray);
var i, j: Integer; dHRaw: DArray;
begin
  SetLength(dHRaw, FHiddenSize);
  SetLength(dInput, FInputSize);
  SetLength(dPrevH, FHiddenSize);
  for i := 0 to FInputSize - 1 do dInput[i] := 0;
  for i := 0 to FHiddenSize - 1 do dPrevH[i] := 0;
  for i := 0 to FHiddenSize - 1 do
    dHRaw[i] := ClipValue(dH[i] * ActivationDerivative(H[i], FActivation), ClipVal);
  for i := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to FInputSize - 1 do
    begin
      dWih[i][j] := dWih[i][j] + dHRaw[i] * Input[j];
      dInput[j] := dInput[j] + Wih[i][j] * dHRaw[i];
    end;
    for j := 0 to FHiddenSize - 1 do
    begin
      dWhh[i][j] := dWhh[i][j] + dHRaw[i] * PrevH[j];
      dPrevH[j] := dPrevH[j] + Whh[i][j] * dHRaw[i];
    end;
    dBh[i] := dBh[i] + dHRaw[i];
  end;
end;

procedure TSimpleRNNCellWrapper.ApplyGradients(LR, ClipVal: Double);
var i, j: Integer;
begin
  for i := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to FInputSize - 1 do
    begin
      Wih[i][j] := Wih[i][j] - LR * ClipValue(dWih[i][j], ClipVal);
      dWih[i][j] := 0;
    end;
    for j := 0 to FHiddenSize - 1 do
    begin
      Whh[i][j] := Whh[i][j] - LR * ClipValue(dWhh[i][j], ClipVal);
      dWhh[i][j] := 0;
    end;
    Bh[i] := Bh[i] - LR * ClipValue(dBh[i], ClipVal);
    dBh[i] := 0;
  end;
end;

procedure TSimpleRNNCellWrapper.ResetGradients;
begin
  ZeroMatrix(dWih, FHiddenSize, FInputSize);
  ZeroMatrix(dWhh, FHiddenSize, FHiddenSize);
  ZeroArray(dBh, FHiddenSize);
end;

function TSimpleRNNCellWrapper.GetHiddenSize: Integer;
begin Result := FHiddenSize; end;

function TSimpleRNNCellWrapper.GetInputSize: Integer;
begin Result := FInputSize; end;

{ TLSTMCellWrapper }
constructor TLSTMCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var Scale: Double; ConcatSize, i: Integer;
begin
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  ConcatSize := InputSize + HiddenSize;
  Scale := Sqrt(2.0 / ConcatSize);
  InitMatrix(Wf, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wi, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wc, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wo, HiddenSize, ConcatSize, Scale);
  SetLength(Bf, HiddenSize); SetLength(Bi, HiddenSize);
  SetLength(Bc, HiddenSize); SetLength(Bo, HiddenSize);
  for i := 0 to HiddenSize - 1 do
  begin Bf[i] := 1.0; Bi[i] := 0; Bc[i] := 0; Bo[i] := 0; end;
  ZeroMatrix(dWf, HiddenSize, ConcatSize); ZeroMatrix(dWi, HiddenSize, ConcatSize);
  ZeroMatrix(dWc, HiddenSize, ConcatSize); ZeroMatrix(dWo, HiddenSize, ConcatSize);
  ZeroArray(dBf, HiddenSize); ZeroArray(dBi, HiddenSize);
  ZeroArray(dBc, HiddenSize); ZeroArray(dBo, HiddenSize);
  ZeroMatrix(MWf, HiddenSize, ConcatSize); ZeroMatrix(MWi, HiddenSize, ConcatSize);
  ZeroMatrix(MWc, HiddenSize, ConcatSize); ZeroMatrix(MWo, HiddenSize, ConcatSize);
  ZeroArray(MBf, HiddenSize); ZeroArray(MBi, HiddenSize);
  ZeroArray(MBc, HiddenSize); ZeroArray(MBo, HiddenSize);
  ZeroMatrix(VWf, HiddenSize, ConcatSize); ZeroMatrix(VWi, HiddenSize, ConcatSize);
  ZeroMatrix(VWc, HiddenSize, ConcatSize); ZeroMatrix(VWo, HiddenSize, ConcatSize);
  ZeroArray(VBf, HiddenSize); ZeroArray(VBi, HiddenSize);
  ZeroArray(VBc, HiddenSize); ZeroArray(VBo, HiddenSize);
end;

procedure TLSTMCellWrapper.Forward(const Input, PrevH, PrevC: DArray; var H, C, FG, IG, CTilde, OG, TanhC: DArray);
var k, j: Integer; Concat: DArray; SumF, SumI, SumC, SumO: Double;
begin
  Concat := ConcatArrays(Input, PrevH);
  SetLength(H, FHiddenSize); SetLength(C, FHiddenSize);
  SetLength(FG, FHiddenSize); SetLength(IG, FHiddenSize);
  SetLength(CTilde, FHiddenSize); SetLength(OG, FHiddenSize);
  SetLength(TanhC, FHiddenSize);
  for k := 0 to FHiddenSize - 1 do
  begin
    SumF := Bf[k]; SumI := Bi[k]; SumC := Bc[k]; SumO := Bo[k];
    for j := 0 to High(Concat) do
    begin
      SumF := SumF + Wf[k][j] * Concat[j];
      SumI := SumI + Wi[k][j] * Concat[j];
      SumC := SumC + Wc[k][j] * Concat[j];
      SumO := SumO + Wo[k][j] * Concat[j];
    end;
    FG[k] := ApplyActivation(SumF, atSigmoid);
    IG[k] := ApplyActivation(SumI, atSigmoid);
    CTilde[k] := ApplyActivation(SumC, atTanh);
    OG[k] := ApplyActivation(SumO, atSigmoid);
    C[k] := FG[k] * PrevC[k] + IG[k] * CTilde[k];
    TanhC[k] := Tanh(C[k]);
    H[k] := OG[k] * TanhC[k];
  end;
end;

procedure TLSTMCellWrapper.Backward(const dH, dC, H, C, FG, IG, CTilde, OG, TanhC, PrevH, PrevC, Input: DArray;
                                    ClipVal: Double; var dInput, dPrevH, dPrevC: DArray);
var k, j: Integer; Concat: DArray; dOG, dCTotal, dFG, dIG, dCTilde: DArray; ConcatSize: Integer;
begin
  Concat := ConcatArrays(Input, PrevH);
  ConcatSize := Length(Concat);
  SetLength(dOG, FHiddenSize); SetLength(dCTotal, FHiddenSize);
  SetLength(dFG, FHiddenSize); SetLength(dIG, FHiddenSize);
  SetLength(dCTilde, FHiddenSize);
  SetLength(dInput, FInputSize); SetLength(dPrevH, FHiddenSize); SetLength(dPrevC, FHiddenSize);
  for k := 0 to FInputSize - 1 do dInput[k] := 0;
  for k := 0 to FHiddenSize - 1 do begin dPrevH[k] := 0; dPrevC[k] := 0; end;
  for k := 0 to FHiddenSize - 1 do
  begin
    dOG[k] := ClipValue(dH[k] * TanhC[k] * ActivationDerivative(OG[k], atSigmoid), ClipVal);
    dCTotal[k] := ClipValue(dH[k] * OG[k] * (1 - TanhC[k] * TanhC[k]) + dC[k], ClipVal);
    dFG[k] := ClipValue(dCTotal[k] * PrevC[k] * ActivationDerivative(FG[k], atSigmoid), ClipVal);
    dIG[k] := ClipValue(dCTotal[k] * CTilde[k] * ActivationDerivative(IG[k], atSigmoid), ClipVal);
    dCTilde[k] := ClipValue(dCTotal[k] * IG[k] * ActivationDerivative(CTilde[k], atTanh), ClipVal);
    dPrevC[k] := dCTotal[k] * FG[k];
  end;
  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      dWf[k][j] := dWf[k][j] + dFG[k] * Concat[j];
      dWi[k][j] := dWi[k][j] + dIG[k] * Concat[j];
      dWc[k][j] := dWc[k][j] + dCTilde[k] * Concat[j];
      dWo[k][j] := dWo[k][j] + dOG[k] * Concat[j];
      if j < FInputSize then
        dInput[j] := dInput[j] + Wf[k][j]*dFG[k] + Wi[k][j]*dIG[k] + Wc[k][j]*dCTilde[k] + Wo[k][j]*dOG[k]
      else
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] + Wf[k][j]*dFG[k] + Wi[k][j]*dIG[k] + Wc[k][j]*dCTilde[k] + Wo[k][j]*dOG[k];
    end;
    dBf[k] := dBf[k] + dFG[k]; dBi[k] := dBi[k] + dIG[k];
    dBc[k] := dBc[k] + dCTilde[k]; dBo[k] := dBo[k] + dOG[k];
  end;
end;

procedure TLSTMCellWrapper.ApplyGradients(LR, ClipVal: Double);
var i, j, ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  for i := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      Wf[i][j] := Wf[i][j] - LR * ClipValue(dWf[i][j], ClipVal); dWf[i][j] := 0;
      Wi[i][j] := Wi[i][j] - LR * ClipValue(dWi[i][j], ClipVal); dWi[i][j] := 0;
      Wc[i][j] := Wc[i][j] - LR * ClipValue(dWc[i][j], ClipVal); dWc[i][j] := 0;
      Wo[i][j] := Wo[i][j] - LR * ClipValue(dWo[i][j], ClipVal); dWo[i][j] := 0;
    end;
    Bf[i] := Bf[i] - LR * ClipValue(dBf[i], ClipVal); dBf[i] := 0;
    Bi[i] := Bi[i] - LR * ClipValue(dBi[i], ClipVal); dBi[i] := 0;
    Bc[i] := Bc[i] - LR * ClipValue(dBc[i], ClipVal); dBc[i] := 0;
    Bo[i] := Bo[i] - LR * ClipValue(dBo[i], ClipVal); dBo[i] := 0;
  end;
end;

procedure TLSTMCellWrapper.ResetGradients;
var ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  ZeroMatrix(dWf, FHiddenSize, ConcatSize); ZeroMatrix(dWi, FHiddenSize, ConcatSize);
  ZeroMatrix(dWc, FHiddenSize, ConcatSize); ZeroMatrix(dWo, FHiddenSize, ConcatSize);
  ZeroArray(dBf, FHiddenSize); ZeroArray(dBi, FHiddenSize);
  ZeroArray(dBc, FHiddenSize); ZeroArray(dBo, FHiddenSize);
end;

function TLSTMCellWrapper.GetHiddenSize: Integer;
begin Result := FHiddenSize; end;

function TLSTMCellWrapper.GetInputSize: Integer;
begin Result := FInputSize; end;

{ TGRUCellWrapper }
constructor TGRUCellWrapper.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var Scale: Double; ConcatSize: Integer;
begin
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  ConcatSize := InputSize + HiddenSize;
  Scale := Sqrt(2.0 / ConcatSize);
  InitMatrix(Wz, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wr, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wh, HiddenSize, ConcatSize, Scale);
  ZeroArray(Bz, HiddenSize); ZeroArray(Br, HiddenSize); ZeroArray(Bh, HiddenSize);
  ZeroMatrix(dWz, HiddenSize, ConcatSize); ZeroMatrix(dWr, HiddenSize, ConcatSize);
  ZeroMatrix(dWh, HiddenSize, ConcatSize);
  ZeroArray(dBz, HiddenSize); ZeroArray(dBr, HiddenSize); ZeroArray(dBh, HiddenSize);
  ZeroMatrix(MWz, HiddenSize, ConcatSize); ZeroMatrix(MWr, HiddenSize, ConcatSize);
  ZeroMatrix(MWh, HiddenSize, ConcatSize);
  ZeroArray(MBz, HiddenSize); ZeroArray(MBr, HiddenSize); ZeroArray(MBh, HiddenSize);
  ZeroMatrix(VWz, HiddenSize, ConcatSize); ZeroMatrix(VWr, HiddenSize, ConcatSize);
  ZeroMatrix(VWh, HiddenSize, ConcatSize);
  ZeroArray(VBz, HiddenSize); ZeroArray(VBr, HiddenSize); ZeroArray(VBh, HiddenSize);
end;

procedure TGRUCellWrapper.Forward(const Input, PrevH: DArray; var H, Z, R, HTilde: DArray);
var k, j: Integer; Concat, ConcatR: DArray; SumZ, SumR, SumH: Double;
begin
  Concat := ConcatArrays(Input, PrevH);
  SetLength(H, FHiddenSize); SetLength(Z, FHiddenSize);
  SetLength(R, FHiddenSize); SetLength(HTilde, FHiddenSize);
  for k := 0 to FHiddenSize - 1 do
  begin
    SumZ := Bz[k]; SumR := Br[k];
    for j := 0 to High(Concat) do
    begin
      SumZ := SumZ + Wz[k][j] * Concat[j];
      SumR := SumR + Wr[k][j] * Concat[j];
    end;
    Z[k] := ApplyActivation(SumZ, atSigmoid);
    R[k] := ApplyActivation(SumR, atSigmoid);
  end;
  SetLength(ConcatR, FInputSize + FHiddenSize);
  for j := 0 to FInputSize - 1 do ConcatR[j] := Input[j];
  for j := 0 to FHiddenSize - 1 do ConcatR[FInputSize + j] := R[j] * PrevH[j];
  for k := 0 to FHiddenSize - 1 do
  begin
    SumH := Bh[k];
    for j := 0 to High(ConcatR) do
      SumH := SumH + Wh[k][j] * ConcatR[j];
    HTilde[k] := ApplyActivation(SumH, atTanh);
    H[k] := (1 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
  end;
end;

procedure TGRUCellWrapper.Backward(const dH, H, Z, R, HTilde, PrevH, Input: DArray;
                                   ClipVal: Double; var dInput, dPrevH: DArray);
var k, j: Integer; Concat, ConcatR: DArray; dZ, dR, dHTilde: DArray; ConcatSize: Integer;
begin
  Concat := ConcatArrays(Input, PrevH);
  ConcatSize := Length(Concat);
  SetLength(ConcatR, ConcatSize);
  for j := 0 to FInputSize - 1 do ConcatR[j] := Input[j];
  for j := 0 to FHiddenSize - 1 do ConcatR[FInputSize + j] := R[j] * PrevH[j];
  SetLength(dZ, FHiddenSize); SetLength(dR, FHiddenSize); SetLength(dHTilde, FHiddenSize);
  SetLength(dInput, FInputSize); SetLength(dPrevH, FHiddenSize);
  for k := 0 to FInputSize - 1 do dInput[k] := 0;
  for k := 0 to FHiddenSize - 1 do dPrevH[k] := 0;
  for k := 0 to FHiddenSize - 1 do
  begin
    dZ[k] := ClipValue(dH[k] * (HTilde[k] - PrevH[k]) * ActivationDerivative(Z[k], atSigmoid), ClipVal);
    dHTilde[k] := ClipValue(dH[k] * Z[k] * ActivationDerivative(HTilde[k], atTanh), ClipVal);
    dPrevH[k] := dPrevH[k] + dH[k] * (1 - Z[k]);
  end;
  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      dWz[k][j] := dWz[k][j] + dZ[k] * Concat[j];
      dWh[k][j] := dWh[k][j] + dHTilde[k] * ConcatR[j];
    end;
    dBz[k] := dBz[k] + dZ[k]; dBh[k] := dBh[k] + dHTilde[k];
  end;
  for k := 0 to FHiddenSize - 1 do
  begin
    dR[k] := 0;
    for j := 0 to FHiddenSize - 1 do
      dR[k] := dR[k] + dHTilde[j] * Wh[j][FInputSize + k] * PrevH[k];
    dR[k] := ClipValue(dR[k] * ActivationDerivative(R[k], atSigmoid), ClipVal);
  end;
  for k := 0 to FHiddenSize - 1 do
    for j := 0 to ConcatSize - 1 do
      dWr[k][j] := dWr[k][j] + dR[k] * Concat[j];
  for k := 0 to FHiddenSize - 1 do dBr[k] := dBr[k] + dR[k];
  for k := 0 to FHiddenSize - 1 do
    for j := 0 to ConcatSize - 1 do
    begin
      if j < FInputSize then
        dInput[j] := dInput[j] + Wz[k][j]*dZ[k] + Wr[k][j]*dR[k] + Wh[k][j]*dHTilde[k]
      else
      begin
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] + Wz[k][j]*dZ[k] + Wr[k][j]*dR[k];
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] + Wh[k][j]*dHTilde[k]*R[j - FInputSize];
      end;
    end;
end;

procedure TGRUCellWrapper.ApplyGradients(LR, ClipVal: Double);
var i, j, ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  for i := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      Wz[i][j] := Wz[i][j] - LR * ClipValue(dWz[i][j], ClipVal); dWz[i][j] := 0;
      Wr[i][j] := Wr[i][j] - LR * ClipValue(dWr[i][j], ClipVal); dWr[i][j] := 0;
      Wh[i][j] := Wh[i][j] - LR * ClipValue(dWh[i][j], ClipVal); dWh[i][j] := 0;
    end;
    Bz[i] := Bz[i] - LR * ClipValue(dBz[i], ClipVal); dBz[i] := 0;
    Br[i] := Br[i] - LR * ClipValue(dBr[i], ClipVal); dBr[i] := 0;
    Bh[i] := Bh[i] - LR * ClipValue(dBh[i], ClipVal); dBh[i] := 0;
  end;
end;

procedure TGRUCellWrapper.ResetGradients;
var ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  ZeroMatrix(dWz, FHiddenSize, ConcatSize); ZeroMatrix(dWr, FHiddenSize, ConcatSize);
  ZeroMatrix(dWh, FHiddenSize, ConcatSize);
  ZeroArray(dBz, FHiddenSize); ZeroArray(dBr, FHiddenSize); ZeroArray(dBh, FHiddenSize);
end;

function TGRUCellWrapper.GetHiddenSize: Integer;
begin Result := FHiddenSize; end;

function TGRUCellWrapper.GetInputSize: Integer;
begin Result := FInputSize; end;

{ TOutputLayerWrapper }
constructor TOutputLayerWrapper.Create(InputSize, OutputSize: Integer; Activation: TActivationType);
var Scale: Double;
begin
  FInputSize := InputSize;
  FOutputSize := OutputSize;
  FActivation := Activation;
  Scale := Sqrt(2.0 / InputSize);
  InitMatrix(W, OutputSize, InputSize, Scale);
  ZeroArray(B, OutputSize);
  ZeroMatrix(dW, OutputSize, InputSize);
  ZeroArray(dB, OutputSize);
  ZeroMatrix(MW, OutputSize, InputSize);
  ZeroArray(MB, OutputSize);
  ZeroMatrix(VW, OutputSize, InputSize);
  ZeroArray(VB, OutputSize);
end;

procedure TOutputLayerWrapper.Forward(const Input: DArray; var Output, Pre: DArray);
var i, j: Integer; Sum: Double;
begin
  SetLength(Output, FOutputSize);
  SetLength(Pre, FOutputSize);
  for i := 0 to FOutputSize - 1 do
  begin
    Sum := B[i];
    for j := 0 to FInputSize - 1 do
      Sum := Sum + W[i][j] * Input[j];
    Pre[i] := Sum;
    Output[i] := ApplyActivation(Sum, FActivation);
  end;
end;

procedure TOutputLayerWrapper.Backward(const dOut, Output, Pre, Input: DArray; ClipVal: Double; var dInput: DArray);
var i, j: Integer; dRaw: DArray;
begin
  SetLength(dRaw, FOutputSize);
  SetLength(dInput, FInputSize);
  for i := 0 to FInputSize - 1 do dInput[i] := 0;
  for i := 0 to FOutputSize - 1 do
    dRaw[i] := ClipValue(dOut[i] * ActivationDerivative(Output[i], FActivation), ClipVal);
  for i := 0 to FOutputSize - 1 do
  begin
    for j := 0 to FInputSize - 1 do
    begin
      dW[i][j] := dW[i][j] + dRaw[i] * Input[j];
      dInput[j] := dInput[j] + W[i][j] * dRaw[i];
    end;
    dB[i] := dB[i] + dRaw[i];
  end;
end;

procedure TOutputLayerWrapper.ApplyGradients(LR, ClipVal: Double);
var i, j: Integer;
begin
  for i := 0 to FOutputSize - 1 do
  begin
    for j := 0 to FInputSize - 1 do
    begin
      W[i][j] := W[i][j] - LR * ClipValue(dW[i][j], ClipVal);
      dW[i][j] := 0;
    end;
    B[i] := B[i] - LR * ClipValue(dB[i], ClipVal);
    dB[i] := 0;
  end;
end;

procedure TOutputLayerWrapper.ResetGradients;
begin
  ZeroMatrix(dW, FOutputSize, FInputSize);
  ZeroArray(dB, FOutputSize);
end;

function TOutputLayerWrapper.GetInputSize: Integer;
begin Result := FInputSize; end;

function TOutputLayerWrapper.GetOutputSize: Integer;
begin Result := FOutputSize; end;

{ TRNNFacade }
constructor TRNNFacade.Create(InputSize: Integer; const HiddenSizes: array of Integer;
                              OutputSize: Integer; CellType: TCellType;
                              Activation, OutputActivation: TActivationType;
                              LossType: TLossType; LearningRate, GradientClip: Double;
                              BPTTSteps: Integer);
var i, PrevSize: Integer;
begin
  FInputSize := InputSize;
  FOutputSize := OutputSize;
  SetLength(FHiddenSizes, Length(HiddenSizes));
  for i := 0 to High(HiddenSizes) do
    FHiddenSizes[i] := HiddenSizes[i];
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

  PrevSize := InputSize;
  case CellType of
    ctSimpleRNN:
    begin
      SetLength(FSimpleCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FSimpleCells[i] := TSimpleRNNCellWrapper.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
    ctLSTM:
    begin
      SetLength(FLSTMCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FLSTMCells[i] := TLSTMCellWrapper.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
    ctGRU:
    begin
      SetLength(FGRUCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FGRUCells[i] := TGRUCellWrapper.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
  end;
  FOutputLayer := TOutputLayerWrapper.Create(PrevSize, OutputSize, OutputActivation);
  FStates := InitHiddenStates;
end;

destructor TRNNFacade.Destroy;
var i: Integer;
begin
  case FCellType of
    ctSimpleRNN: for i := 0 to High(FSimpleCells) do FSimpleCells[i].Free;
    ctLSTM: for i := 0 to High(FLSTMCells) do FLSTMCells[i].Free;
    ctGRU: for i := 0 to High(FGRUCells) do FGRUCells[i].Free;
  end;
  FOutputLayer.Free;
  inherited;
end;

function TRNNFacade.ClipGradient(G, MaxVal: Double): Double;
begin
  Result := ClipValue(G, MaxVal);
end;

function TRNNFacade.InitHiddenStates: TDArray3D;
var i: Integer;
begin
  SetLength(Result, Length(FHiddenSizes));
  for i := 0 to High(FHiddenSizes) do
  begin
    SetLength(Result[i], 2);
    ZeroArray(Result[i][0], FHiddenSizes[i]);
    ZeroArray(Result[i][1], FHiddenSizes[i]);
  end;
end;

function TRNNFacade.ForwardSequence(const Inputs: TDArray2D): TDArray2D;
var t, layer, k: Integer;
    X, H, C, PreH, F, I, CTilde, O, TanhC, Z, R, HTilde: DArray;
    OutVal, OutPre: DArray;
    NewStates: TDArray3D;
    DropMask: DArray;
begin
  FSequenceLen := Length(Inputs);
  SetLength(FCaches, FSequenceLen);
  SetLength(Result, FSequenceLen);
  NewStates := InitHiddenStates;

  for t := 0 to High(Inputs) do
  begin
    X := Copy(Inputs[t]);
    FCaches[t].Input := Copy(X);

    for layer := 0 to High(FHiddenSizes) do
    begin
      case FCellType of
        ctSimpleRNN:
        begin
          FSimpleCells[layer].Forward(X, FStates[layer][0], H, PreH);
          NewStates[layer][0] := Copy(H);
          FCaches[t].H := Copy(H);
          FCaches[t].PreH := Copy(PreH);
        end;
        ctLSTM:
        begin
          FLSTMCells[layer].Forward(X, FStates[layer][0], FStates[layer][1], H, C, F, I, CTilde, O, TanhC);
          NewStates[layer][0] := Copy(H);
          NewStates[layer][1] := Copy(C);
          FCaches[t].H := Copy(H);
          FCaches[t].C := Copy(C);
          FCaches[t].F := Copy(F);
          FCaches[t].I := Copy(I);
          FCaches[t].CTilde := Copy(CTilde);
          FCaches[t].O := Copy(O);
          FCaches[t].TanhC := Copy(TanhC);
        end;
        ctGRU:
        begin
          FGRUCells[layer].Forward(X, FStates[layer][0], H, Z, R, HTilde);
          NewStates[layer][0] := Copy(H);
          FCaches[t].H := Copy(H);
          FCaches[t].Z := Copy(Z);
          FCaches[t].R := Copy(R);
          FCaches[t].HTilde := Copy(HTilde);
        end;
      end;

      if FUseDropout and (FDropoutRate > 0) then
      begin
        SetLength(DropMask, Length(H));
        for k := 0 to High(H) do
        begin
          if Random < FDropoutRate then
          begin DropMask[k] := 0; H[k] := 0; end
          else
          begin DropMask[k] := 1.0 / (1.0 - FDropoutRate); H[k] := H[k] * DropMask[k]; end;
        end;
        FCaches[t].DropoutMask := Copy(DropMask);
      end;

      X := Copy(H);
    end;

    FOutputLayer.Forward(X, OutVal, OutPre);
    FCaches[t].OutVal := Copy(OutVal);
    FCaches[t].OutPre := Copy(OutPre);
    Result[t] := Copy(OutVal);
    FStates := NewStates;
  end;
end;

function ComputeLoss(const Pred, Target: DArray; LossType: TLossType): Double;
var i: Integer; P: Double;
begin
  Result := 0;
  case LossType of
    ltMSE:
      for i := 0 to High(Pred) do
        Result := Result + Sqr(Pred[i] - Target[i]);
    ltCrossEntropy:
      for i := 0 to High(Pred) do
      begin
        P := Max(1e-15, Min(1 - 1e-15, Pred[i]));
        Result := Result - (Target[i] * Ln(P) + (1 - Target[i]) * Ln(1 - P));
      end;
  end;
  Result := Result / Length(Pred);
end;

procedure ComputeLossGradient(const Pred, Target: DArray; LossType: TLossType; var Grad: DArray);
var i: Integer; P: Double;
begin
  SetLength(Grad, Length(Pred));
  case LossType of
    ltMSE:
      for i := 0 to High(Pred) do
        Grad[i] := Pred[i] - Target[i];
    ltCrossEntropy:
      for i := 0 to High(Pred) do
      begin
        P := Max(1e-15, Min(1 - 1e-15, Pred[i]));
        Grad[i] := (P - Target[i]) / (P * (1 - P) + 1e-15);
      end;
  end;
end;

function TRNNFacade.BackwardSequence(const Targets: TDArray2D): Double;
var t, layer, k: Integer;
    T_len, BPTTLimit: Integer;
    dOut, dH, dC, dInput, dPrevH, dPrevC: DArray;
    Grad: DArray;
    dStatesH, dStatesC: TDArray2D;
    TotalLoss: Double;
    PrevH, PrevC: DArray;
    GradScale: Double;
begin
  T_len := Length(Targets);
  if FBPTTSteps > 0 then BPTTLimit := FBPTTSteps else BPTTLimit := T_len;
  TotalLoss := 0;

  SetLength(dStatesH, Length(FHiddenSizes));
  SetLength(dStatesC, Length(FHiddenSizes));
  SetLength(FGradientHistory, T_len);
  for layer := 0 to High(FHiddenSizes) do
  begin
    ZeroArray(dStatesH[layer], FHiddenSizes[layer]);
    ZeroArray(dStatesC[layer], FHiddenSizes[layer]);
  end;

  for t := T_len - 1 downto Max(0, T_len - BPTTLimit) do
  begin
    TotalLoss := TotalLoss + ComputeLoss(FCaches[t].OutVal, Targets[t], FLossType);
    ComputeLossGradient(FCaches[t].OutVal, Targets[t], FLossType, Grad);
    FOutputLayer.Backward(Grad, FCaches[t].OutVal, FCaches[t].OutPre, FCaches[t].H, FGradientClip, dH);

    GradScale := 0;
    for k := 0 to High(dH) do GradScale := GradScale + Abs(dH[k]);
    SetLength(FGradientHistory[t], 1);
    FGradientHistory[t][0] := GradScale / Length(dH);

    for layer := High(FHiddenSizes) downto 0 do
    begin
      SetLength(dOut, FHiddenSizes[layer]);
      for k := 0 to FHiddenSizes[layer] - 1 do
        dOut[k] := dH[k] + dStatesH[layer][k];

      if t > 0 then PrevH := FCaches[t-1].H
      else ZeroArray(PrevH, FHiddenSizes[layer]);

      case FCellType of
        ctSimpleRNN:
        begin
          FSimpleCells[layer].Backward(dOut, FCaches[t].H, FCaches[t].PreH, PrevH,
                                       FCaches[t].Input, FGradientClip, dInput, dPrevH);
          dStatesH[layer] := Copy(dPrevH);
        end;
        ctLSTM:
        begin
          if t > 0 then PrevC := FCaches[t-1].C
          else ZeroArray(PrevC, FHiddenSizes[layer]);
          SetLength(dC, FHiddenSizes[layer]);
          for k := 0 to FHiddenSizes[layer] - 1 do
            dC[k] := dStatesC[layer][k];
          FLSTMCells[layer].Backward(dOut, dC, FCaches[t].H, FCaches[t].C,
                                     FCaches[t].F, FCaches[t].I, FCaches[t].CTilde,
                                     FCaches[t].O, FCaches[t].TanhC,
                                     PrevH, PrevC, FCaches[t].Input,
                                     FGradientClip, dInput, dPrevH, dPrevC);
          dStatesH[layer] := Copy(dPrevH);
          dStatesC[layer] := Copy(dPrevC);
        end;
        ctGRU:
        begin
          FGRUCells[layer].Backward(dOut, FCaches[t].H, FCaches[t].Z, FCaches[t].R,
                                    FCaches[t].HTilde, PrevH, FCaches[t].Input,
                                    FGradientClip, dInput, dPrevH);
          dStatesH[layer] := Copy(dPrevH);
        end;
      end;
      dH := Copy(dInput);
    end;
  end;
  Result := TotalLoss / T_len;
end;

function TRNNFacade.TrainSequence(const Inputs, Targets: TDArray2D): Double;
begin
  ResetGradients;
  FStates := InitHiddenStates;
  ForwardSequence(Inputs);
  Result := BackwardSequence(Targets);
  ApplyGradients;
end;

function TRNNFacade.Predict(const Inputs: TDArray2D): TDArray2D;
begin
  FStates := InitHiddenStates;
  Result := ForwardSequence(Inputs);
end;

procedure TRNNFacade.ResetGradients;
var i: Integer;
begin
  case FCellType of
    ctSimpleRNN: for i := 0 to High(FSimpleCells) do FSimpleCells[i].ResetGradients;
    ctLSTM: for i := 0 to High(FLSTMCells) do FLSTMCells[i].ResetGradients;
    ctGRU: for i := 0 to High(FGRUCells) do FGRUCells[i].ResetGradients;
  end;
  FOutputLayer.ResetGradients;
end;

procedure TRNNFacade.ApplyGradients;
var i: Integer;
begin
  case FCellType of
    ctSimpleRNN: for i := 0 to High(FSimpleCells) do FSimpleCells[i].ApplyGradients(FLearningRate, FGradientClip);
    ctLSTM: for i := 0 to High(FLSTMCells) do FLSTMCells[i].ApplyGradients(FLearningRate, FGradientClip);
    ctGRU: for i := 0 to High(FGRUCells) do FGRUCells[i].ApplyGradients(FLearningRate, FGradientClip);
  end;
  FOutputLayer.ApplyGradients(FLearningRate, FGradientClip);
end;

{ 1. Time-Step and Sequence Access }
function TRNNFacade.GetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (NeuronIdx >= 0) then
    if NeuronIdx < Length(FCaches[Timestep].H) then
      Result := FCaches[Timestep].H[NeuronIdx];
end;

procedure TRNNFacade.SetHiddenState(LayerIdx, Timestep, NeuronIdx: Integer; Value: Double);
begin
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (NeuronIdx >= 0) then
    if NeuronIdx < Length(FCaches[Timestep].H) then
      FCaches[Timestep].H[NeuronIdx] := Value;
end;

function TRNNFacade.GetOutput(Timestep, OutputIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (OutputIdx >= 0) then
    if OutputIdx < Length(FCaches[Timestep].OutVal) then
      Result := FCaches[Timestep].OutVal[OutputIdx];
end;

procedure TRNNFacade.SetOutput(Timestep, OutputIdx: Integer; Value: Double);
begin
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (OutputIdx >= 0) then
    if OutputIdx < Length(FCaches[Timestep].OutVal) then
      FCaches[Timestep].OutVal[OutputIdx] := Value;
end;

{ 2. Cell State and Gate Access }
function TRNNFacade.GetCellState(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if FCellType <> ctLSTM then Exit;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (NeuronIdx >= 0) then
    if NeuronIdx < Length(FCaches[Timestep].C) then
      Result := FCaches[Timestep].C[NeuronIdx];
end;

function TRNNFacade.GetGateValue(GateType: TGateType; LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep < 0) or (Timestep >= FSequenceLen) or (NeuronIdx < 0) then Exit;
  case FCellType of
    ctLSTM:
      case GateType of
        gtForget: if NeuronIdx < Length(FCaches[Timestep].F) then Result := FCaches[Timestep].F[NeuronIdx];
        gtInput: if NeuronIdx < Length(FCaches[Timestep].I) then Result := FCaches[Timestep].I[NeuronIdx];
        gtOutput: if NeuronIdx < Length(FCaches[Timestep].O) then Result := FCaches[Timestep].O[NeuronIdx];
        gtCellCandidate: if NeuronIdx < Length(FCaches[Timestep].CTilde) then Result := FCaches[Timestep].CTilde[NeuronIdx];
      end;
    ctGRU:
      case GateType of
        gtUpdate: if NeuronIdx < Length(FCaches[Timestep].Z) then Result := FCaches[Timestep].Z[NeuronIdx];
        gtReset: if NeuronIdx < Length(FCaches[Timestep].R) then Result := FCaches[Timestep].R[NeuronIdx];
        gtHiddenCandidate: if NeuronIdx < Length(FCaches[Timestep].HTilde) then Result := FCaches[Timestep].HTilde[NeuronIdx];
      end;
  end;
end;

{ 3. Cached Pre-Activations and Inputs }
function TRNNFacade.GetPreActivation(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (NeuronIdx >= 0) then
    if NeuronIdx < Length(FCaches[Timestep].PreH) then
      Result := FCaches[Timestep].PreH[NeuronIdx];
end;

function TRNNFacade.GetInputVector(Timestep, InputIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (InputIdx >= 0) then
    if InputIdx < Length(FCaches[Timestep].Input) then
      Result := FCaches[Timestep].Input[InputIdx];
end;

{ 4. Gradients and Optimizer States }
function TRNNFacade.GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
  Result := 0;
  if (LayerIdx < 0) or (LayerIdx > High(FHiddenSizes)) then Exit;
  case FCellType of
    ctSimpleRNN:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (WeightIdx >= 0) and (WeightIdx < Length(FSimpleCells[LayerIdx].dWih[NeuronIdx])) then
          Result := FSimpleCells[LayerIdx].dWih[NeuronIdx][WeightIdx];
    ctLSTM:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (WeightIdx >= 0) and (WeightIdx < Length(FLSTMCells[LayerIdx].dWf[NeuronIdx])) then
          Result := FLSTMCells[LayerIdx].dWf[NeuronIdx][WeightIdx];
    ctGRU:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (WeightIdx >= 0) and (WeightIdx < Length(FGRUCells[LayerIdx].dWz[NeuronIdx])) then
          Result := FGRUCells[LayerIdx].dWz[NeuronIdx][WeightIdx];
  end;
end;

function TRNNFacade.GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if (LayerIdx < 0) or (LayerIdx > High(FHiddenSizes)) then Exit;
  case FCellType of
    ctSimpleRNN:
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FSimpleCells[LayerIdx].dBh)) then
        Result := FSimpleCells[LayerIdx].dBh[NeuronIdx];
    ctLSTM:
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FLSTMCells[LayerIdx].dBf)) then
        Result := FLSTMCells[LayerIdx].dBf[NeuronIdx];
    ctGRU:
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGRUCells[LayerIdx].dBz)) then
        Result := FGRUCells[LayerIdx].dBz[NeuronIdx];
  end;
end;

function TRNNFacade.GetOptimizerState(LayerIdx, NeuronIdx, Param: Integer): TOptimizerStateRecord;
begin
  Result.Momentum := 0; Result.Velocity := 0;
  Result.Beta1Power := 0; Result.Beta2Power := 0;
  if (LayerIdx < 0) or (LayerIdx > High(FHiddenSizes)) then Exit;
  case FCellType of
    ctSimpleRNN:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (Param >= 0) and (Param < Length(FSimpleCells[LayerIdx].MWih[NeuronIdx])) then
        begin
          Result.Momentum := FSimpleCells[LayerIdx].MWih[NeuronIdx][Param];
          Result.Velocity := FSimpleCells[LayerIdx].VWih[NeuronIdx][Param];
        end;
    ctLSTM:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (Param >= 0) and (Param < Length(FLSTMCells[LayerIdx].MWf[NeuronIdx])) then
        begin
          Result.Momentum := FLSTMCells[LayerIdx].MWf[NeuronIdx][Param];
          Result.Velocity := FLSTMCells[LayerIdx].VWf[NeuronIdx][Param];
        end;
    ctGRU:
      if (NeuronIdx >= 0) and (NeuronIdx < FHiddenSizes[LayerIdx]) then
        if (Param >= 0) and (Param < Length(FGRUCells[LayerIdx].MWz[NeuronIdx])) then
        begin
          Result.Momentum := FGRUCells[LayerIdx].MWz[NeuronIdx][Param];
          Result.Velocity := FGRUCells[LayerIdx].VWz[NeuronIdx][Param];
        end;
  end;
end;

function TRNNFacade.GetCellGradient(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 0;
  if (Timestep >= 0) and (Timestep < Length(FGradientHistory)) then
    if Length(FGradientHistory[Timestep]) > 0 then
      Result := FGradientHistory[Timestep][0];
end;

{ 5. Dropout, LayerNorm, Regularization }
procedure TRNNFacade.SetDropoutRate(Rate: Double);
begin
  FDropoutRate := Rate;
  FUseDropout := Rate > 0;
end;

function TRNNFacade.GetDropoutRate: Double;
begin
  Result := FDropoutRate;
end;

function TRNNFacade.GetDropoutMask(LayerIdx, Timestep, NeuronIdx: Integer): Double;
begin
  Result := 1.0;
  if (Timestep >= 0) and (Timestep < FSequenceLen) and (NeuronIdx >= 0) then
    if NeuronIdx < Length(FCaches[Timestep].DropoutMask) then
      Result := FCaches[Timestep].DropoutMask[NeuronIdx];
end;

function TRNNFacade.GetLayerNormStats(LayerIdx, Timestep: Integer): TLayerNormStats;
var i: Integer; Sum, SumSq, Mean, Variance: Double;
begin
  Result.Mean := 0; Result.Variance := 1; Result.Gamma := 1; Result.Beta := 0;
  if (Timestep < 0) or (Timestep >= FSequenceLen) then Exit;
  if Length(FCaches[Timestep].H) = 0 then Exit;
  Sum := 0; SumSq := 0;
  for i := 0 to High(FCaches[Timestep].H) do
  begin
    Sum := Sum + FCaches[Timestep].H[i];
    SumSq := SumSq + Sqr(FCaches[Timestep].H[i]);
  end;
  Mean := Sum / Length(FCaches[Timestep].H);
  Variance := SumSq / Length(FCaches[Timestep].H) - Sqr(Mean);
  Result.Mean := Mean;
  Result.Variance := Variance;
end;

procedure TRNNFacade.EnableDropout(Enable: Boolean);
begin
  FUseDropout := Enable;
end;

{ 6. Sequence-to-Sequence APIs }
function TRNNFacade.GetSequenceOutputs(OutputIdx: Integer): DArray;
var t: Integer;
begin
  SetLength(Result, FSequenceLen);
  for t := 0 to FSequenceLen - 1 do
    if (OutputIdx >= 0) and (OutputIdx < Length(FCaches[t].OutVal)) then
      Result[t] := FCaches[t].OutVal[OutputIdx]
    else
      Result[t] := 0;
end;

function TRNNFacade.GetSequenceHiddenStates(LayerIdx, NeuronIdx: Integer): DArray;
var t: Integer;
begin
  SetLength(Result, FSequenceLen);
  for t := 0 to FSequenceLen - 1 do
    if (NeuronIdx >= 0) and (NeuronIdx < Length(FCaches[t].H)) then
      Result[t] := FCaches[t].H[NeuronIdx]
    else
      Result[t] := 0;
end;

function TRNNFacade.GetSequenceCellStates(LayerIdx, NeuronIdx: Integer): DArray;
var t: Integer;
begin
  SetLength(Result, FSequenceLen);
  if FCellType <> ctLSTM then Exit;
  for t := 0 to FSequenceLen - 1 do
    if (NeuronIdx >= 0) and (NeuronIdx < Length(FCaches[t].C)) then
      Result[t] := FCaches[t].C[NeuronIdx]
    else
      Result[t] := 0;
end;

function TRNNFacade.GetSequenceGateValues(GateType: TGateType; LayerIdx, NeuronIdx: Integer): DArray;
var t: Integer;
begin
  SetLength(Result, FSequenceLen);
  for t := 0 to FSequenceLen - 1 do
    Result[t] := GetGateValue(GateType, LayerIdx, t, NeuronIdx);
end;

{ 7. Reset and Manipulate Hidden States }
procedure TRNNFacade.ResetHiddenState(LayerIdx: Integer; Value: Double);
var i: Integer;
begin
  if (LayerIdx >= 0) and (LayerIdx <= High(FHiddenSizes)) then
    for i := 0 to FHiddenSizes[LayerIdx] - 1 do
      FStates[LayerIdx][0][i] := Value;
end;

procedure TRNNFacade.ResetCellState(LayerIdx: Integer; Value: Double);
var i: Integer;
begin
  if FCellType <> ctLSTM then Exit;
  if (LayerIdx >= 0) and (LayerIdx <= High(FHiddenSizes)) then
    for i := 0 to FHiddenSizes[LayerIdx] - 1 do
      FStates[LayerIdx][1][i] := Value;
end;

procedure TRNNFacade.ResetAllStates(Value: Double);
var layer, i: Integer;
begin
  for layer := 0 to High(FHiddenSizes) do
  begin
    for i := 0 to FHiddenSizes[layer] - 1 do
    begin
      FStates[layer][0][i] := Value;
      if FCellType = ctLSTM then
        FStates[layer][1][i] := Value;
    end;
  end;
end;

procedure TRNNFacade.InjectHiddenState(LayerIdx: Integer; const ValuesArray: DArray);
var i: Integer;
begin
  if (LayerIdx < 0) or (LayerIdx > High(FHiddenSizes)) then Exit;
  for i := 0 to Min(High(ValuesArray), FHiddenSizes[LayerIdx] - 1) do
    FStates[LayerIdx][0][i] := ValuesArray[i];
end;

procedure TRNNFacade.InjectCellState(LayerIdx: Integer; const ValuesArray: DArray);
var i: Integer;
begin
  if FCellType <> ctLSTM then Exit;
  if (LayerIdx < 0) or (LayerIdx > High(FHiddenSizes)) then Exit;
  for i := 0 to Min(High(ValuesArray), FHiddenSizes[LayerIdx] - 1) do
    FStates[LayerIdx][1][i] := ValuesArray[i];
end;

{ 8. Attention Introspection (placeholder for RNN+Attention) }
function TRNNFacade.GetAttentionWeights(Timestep: Integer): DArray;
begin
  SetLength(Result, 0);
end;

function TRNNFacade.GetAttentionContext(Timestep: Integer): DArray;
begin
  SetLength(Result, 0);
end;

{ 9. Time-Series Diagnostics }
function TRNNFacade.GetHiddenStateHistogram(LayerIdx, Timestep: Integer; NumBins: Integer): THistogram;
var i, BinIdx: Integer;
    MinVal, MaxVal, BinWidth, Val: Double;
begin
  SetLength(Result, NumBins);
  if (Timestep < 0) or (Timestep >= FSequenceLen) then Exit;
  if Length(FCaches[Timestep].H) = 0 then Exit;

  MinVal := FCaches[Timestep].H[0];
  MaxVal := FCaches[Timestep].H[0];
  for i := 1 to High(FCaches[Timestep].H) do
  begin
    if FCaches[Timestep].H[i] < MinVal then MinVal := FCaches[Timestep].H[i];
    if FCaches[Timestep].H[i] > MaxVal then MaxVal := FCaches[Timestep].H[i];
  end;
  if MaxVal = MinVal then MaxVal := MinVal + 1;
  BinWidth := (MaxVal - MinVal) / NumBins;

  for i := 0 to NumBins - 1 do
  begin
    Result[i].RangeMin := MinVal + i * BinWidth;
    Result[i].RangeMax := MinVal + (i + 1) * BinWidth;
    Result[i].Count := 0;
    Result[i].Percentage := 0;
  end;

  for i := 0 to High(FCaches[Timestep].H) do
  begin
    Val := FCaches[Timestep].H[i];
    BinIdx := Trunc((Val - MinVal) / BinWidth);
    if BinIdx >= NumBins then BinIdx := NumBins - 1;
    if BinIdx < 0 then BinIdx := 0;
    Inc(Result[BinIdx].Count);
  end;

  for i := 0 to NumBins - 1 do
    Result[i].Percentage := Result[i].Count / Length(FCaches[Timestep].H) * 100;
end;

function TRNNFacade.GetActivationHistogramOverTime(LayerIdx, NeuronIdx: Integer; NumBins: Integer): THistogram;
var t, i, BinIdx: Integer;
    MinVal, MaxVal, BinWidth, Val: Double;
    Values: DArray;
begin
  SetLength(Result, NumBins);
  if FSequenceLen = 0 then Exit;

  SetLength(Values, FSequenceLen);
  for t := 0 to FSequenceLen - 1 do
    if (NeuronIdx >= 0) and (NeuronIdx < Length(FCaches[t].H)) then
      Values[t] := FCaches[t].H[NeuronIdx]
    else
      Values[t] := 0;

  MinVal := Values[0]; MaxVal := Values[0];
  for t := 1 to High(Values) do
  begin
    if Values[t] < MinVal then MinVal := Values[t];
    if Values[t] > MaxVal then MaxVal := Values[t];
  end;
  if MaxVal = MinVal then MaxVal := MinVal + 1;
  BinWidth := (MaxVal - MinVal) / NumBins;

  for i := 0 to NumBins - 1 do
  begin
    Result[i].RangeMin := MinVal + i * BinWidth;
    Result[i].RangeMax := MinVal + (i + 1) * BinWidth;
    Result[i].Count := 0;
  end;

  for t := 0 to High(Values) do
  begin
    Val := Values[t];
    BinIdx := Trunc((Val - MinVal) / BinWidth);
    if BinIdx >= NumBins then BinIdx := NumBins - 1;
    if BinIdx < 0 then BinIdx := 0;
    Inc(Result[BinIdx].Count);
  end;

  for i := 0 to NumBins - 1 do
    Result[i].Percentage := Result[i].Count / FSequenceLen * 100;
end;

function TRNNFacade.GetGateSaturation(GateType: TGateType; LayerIdx, Timestep: Integer; Threshold: Double): TGateSaturationStats;
var i: Integer; Val: Double; GateArr: DArray;
begin
  Result.NearZeroCount := 0; Result.NearOneCount := 0;
  Result.TotalCount := 0; Result.NearZeroPct := 0; Result.NearOnePct := 0;
  if (Timestep < 0) or (Timestep >= FSequenceLen) then Exit;

  case FCellType of
    ctLSTM:
      case GateType of
        gtForget: GateArr := FCaches[Timestep].F;
        gtInput: GateArr := FCaches[Timestep].I;
        gtOutput: GateArr := FCaches[Timestep].O;
      else Exit;
      end;
    ctGRU:
      case GateType of
        gtUpdate: GateArr := FCaches[Timestep].Z;
        gtReset: GateArr := FCaches[Timestep].R;
      else Exit;
      end;
  else Exit;
  end;

  Result.TotalCount := Length(GateArr);
  for i := 0 to High(GateArr) do
  begin
    Val := GateArr[i];
    if Val < Threshold then Inc(Result.NearZeroCount);
    if Val > (1 - Threshold) then Inc(Result.NearOneCount);
  end;
  if Result.TotalCount > 0 then
  begin
    Result.NearZeroPct := Result.NearZeroCount / Result.TotalCount * 100;
    Result.NearOnePct := Result.NearOneCount / Result.TotalCount * 100;
  end;
end;

function TRNNFacade.GetGradientScalesOverTime(LayerIdx: Integer): TGradientScaleArray;
var t: Integer;
begin
  SetLength(Result, FSequenceLen);
  for t := 0 to FSequenceLen - 1 do
  begin
    Result[t].Timestep := t;
    if (t < Length(FGradientHistory)) and (Length(FGradientHistory[t]) > 0) then
    begin
      Result[t].MeanAbsGrad := FGradientHistory[t][0];
      Result[t].MaxAbsGrad := FGradientHistory[t][0];
      Result[t].MinAbsGrad := FGradientHistory[t][0];
    end
    else
    begin
      Result[t].MeanAbsGrad := 0;
      Result[t].MaxAbsGrad := 0;
      Result[t].MinAbsGrad := 0;
    end;
  end;
end;

function TRNNFacade.DetectVanishingGradient(LayerIdx: Integer; Threshold: Double): Boolean;
var t: Integer;
begin
  Result := False;
  for t := 0 to High(FGradientHistory) do
    if (Length(FGradientHistory[t]) > 0) and (FGradientHistory[t][0] < Threshold) then
    begin
      Result := True;
      Exit;
    end;
end;

function TRNNFacade.DetectExplodingGradient(LayerIdx: Integer; Threshold: Double): Boolean;
var t: Integer;
begin
  Result := False;
  for t := 0 to High(FGradientHistory) do
    if (Length(FGradientHistory[t]) > 0) and (FGradientHistory[t][0] > Threshold) then
    begin
      Result := True;
      Exit;
    end;
end;

{ Utility }
function TRNNFacade.GetLayerCount: Integer;
begin
  Result := Length(FHiddenSizes);
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

end.

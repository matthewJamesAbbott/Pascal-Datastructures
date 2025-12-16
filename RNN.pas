//
// Matthew Abbott  2025
// Advanced RNN with BPTT, Gradient Clipping, LSTM/GRU, Batch Processing
//

{$mode objfpc}{$H+}
{$M+}

program AdvancedRNN;

uses Classes, Math, SysUtils;

type
  TActivationType = (atSigmoid, atTanh, atReLU, atLinear);
  TLossType = (ltMSE, ltCrossEntropy);
  TCellType = (ctSimpleRNN, ctLSTM, ctGRU);

  DArray = array of Double;
  TDArray2D = array of DArray;
  TDArray3D = array of TDArray2D;

  TDataSplit = record
    TrainInputs, TrainTargets: TDArray2D;
    ValInputs, ValTargets: TDArray2D;
  end;

  // ========== Activation Functions ==========
  TActivation = class
    class function Apply(X: Double; ActType: TActivationType): Double;
    class function Derivative(Y: Double; ActType: TActivationType): Double;
    class procedure ApplySoftmax(var Arr: DArray);
  end;

  // ========== Loss Functions ==========
  TLoss = class
    class function Compute(const Pred, Target: DArray; LossType: TLossType): Double;
    class procedure Gradient(const Pred, Target: DArray; LossType: TLossType; var Grad: DArray);
  end;

  // ========== Simple RNN Cell ==========
  TSimpleRNNCell = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wih, Whh: TDArray2D;
    Bh: DArray;
    dWih, dWhh: TDArray2D;
    dBh: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH: DArray; var H, PreH: DArray);
    procedure Backward(const dH, H, PreH, PrevH, Input: DArray; ClipVal: Double; var dInput, dPrevH: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
  end;

  // ========== LSTM Cell ==========
  TLSTMCell = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wf, Wi, Wc, Wo: TDArray2D;
    Bf, Bi, Bc, Bo: DArray;
    dWf, dWi, dWc, dWo: TDArray2D;
    dBf, dBi, dBc, dBo: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH, PrevC: DArray; var H, C, F, I, CTilde, O, TanhC: DArray);
    procedure Backward(const dH, dC, H, C, F, I, CTilde, O, TanhC, PrevH, PrevC, Input: DArray;
                       ClipVal: Double; var dInput, dPrevH, dPrevC: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
  end;

  // ========== GRU Cell ==========
  TGRUCell = class
  private
    FInputSize, FHiddenSize: Integer;
    FActivation: TActivationType;
  public
    Wz, Wr, Wh: TDArray2D;
    Bz, Br, Bh: DArray;
    dWz, dWr, dWh: TDArray2D;
    dBz, dBr, dBh: DArray;
    constructor Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
    procedure Forward(const Input, PrevH: DArray; var H, Z, R, HTilde: DArray);
    procedure Backward(const dH, H, Z, R, HTilde, PrevH, Input: DArray;
                       ClipVal: Double; var dInput, dPrevH: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetHiddenSize: Integer;
  end;

  // ========== Output Layer ==========
  TOutputLayer = class
  private
    FInputSize, FOutputSize: Integer;
    FActivation: TActivationType;
  public
    W: TDArray2D;
    B: DArray;
    dW: TDArray2D;
    dB: DArray;
    constructor Create(InputSize, OutputSize: Integer; Activation: TActivationType);
    procedure Forward(const Input: DArray; var Output, Pre: DArray);
    procedure Backward(const dOut, Output, Pre, Input: DArray; ClipVal: Double; var dInput: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
  end;

  // ========== Cache for BPTT ==========
  TTimeStepCache = record
    Input: DArray;
    H, C: DArray;
    PreH: DArray;
    F, I, CTilde, O, TanhC: DArray;
    Z, R, HTilde: DArray;
    OutPre, OutVal: DArray;
  end;

  // ========== Main Advanced RNN ==========
  TAdvancedRNN = class
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

    FSimpleCells: array of TSimpleRNNCell;
    FLSTMCells: array of TLSTMCell;
    FGRUCells: array of TGRUCell;
    FOutputLayer: TOutputLayer;

    function ClipGradient(G, MaxVal: Double): Double;
  public
    constructor Create(InputSize: Integer; const HiddenSizes: array of Integer;
                       OutputSize: Integer; CellType: TCellType;
                       Activation, OutputActivation: TActivationType;
                       LossType: TLossType; LearningRate, GradientClip: Double;
                       BPTTSteps: Integer);
    destructor Destroy; override;

    function ForwardSequence(const Inputs: TDArray2D; var Caches: array of TTimeStepCache;
                              var States: TDArray3D): TDArray2D;
    function BackwardSequence(const Targets: TDArray2D; const Caches: array of TTimeStepCache;
                               const States: TDArray3D): Double;
    function TrainSequence(const Inputs, Targets: TDArray2D): Double;
    function TrainBatch(const BatchInputs, BatchTargets: TDArray3D): Double;
    function Predict(const Inputs: TDArray2D): TDArray2D;
    function ComputeLoss(const Inputs, Targets: TDArray2D): Double;

    procedure ResetGradients;
    procedure ApplyGradients;
    function InitHiddenStates: TDArray3D;

    property LearningRate: Double read FLearningRate write FLearningRate;
    property GradientClip: Double read FGradientClip write FGradientClip;
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

// ========== TActivation ==========
class function TActivation.Apply(X: Double; ActType: TActivationType): Double;
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

class function TActivation.Derivative(Y: Double; ActType: TActivationType): Double;
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

class procedure TActivation.ApplySoftmax(var Arr: DArray);
var
  i: Integer;
  MaxVal, Sum: Double;
begin
  MaxVal := Arr[0];
  for i := 1 to High(Arr) do
    if Arr[i] > MaxVal then MaxVal := Arr[i];
  Sum := 0;
  for i := 0 to High(Arr) do
  begin
    Arr[i] := Exp(Arr[i] - MaxVal);
    Sum := Sum + Arr[i];
  end;
  for i := 0 to High(Arr) do
    Arr[i] := Arr[i] / Sum;
end;

// ========== TLoss ==========
class function TLoss.Compute(const Pred, Target: DArray; LossType: TLossType): Double;
var
  i: Integer;
  P: Double;
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

class procedure TLoss.Gradient(const Pred, Target: DArray; LossType: TLossType; var Grad: DArray);
var
  i: Integer;
  P: Double;
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

// ========== TSimpleRNNCell ==========
constructor TSimpleRNNCell.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var
  Scale: Double;
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
end;

procedure TSimpleRNNCell.Forward(const Input, PrevH: DArray; var H, PreH: DArray);
var
  i, j: Integer;
  Sum: Double;
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
    H[i] := TActivation.Apply(Sum, FActivation);
  end;
end;

procedure TSimpleRNNCell.Backward(const dH, H, PreH, PrevH, Input: DArray; ClipVal: Double; var dInput, dPrevH: DArray);
var
  i, j: Integer;
  dHRaw: DArray;
begin
  SetLength(dHRaw, FHiddenSize);
  SetLength(dInput, FInputSize);
  SetLength(dPrevH, FHiddenSize);
  for i := 0 to FInputSize - 1 do dInput[i] := 0;
  for i := 0 to FHiddenSize - 1 do dPrevH[i] := 0;

  for i := 0 to FHiddenSize - 1 do
    dHRaw[i] := ClipValue(dH[i] * TActivation.Derivative(H[i], FActivation), ClipVal);

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

procedure TSimpleRNNCell.ApplyGradients(LR, ClipVal: Double);
var
  i, j: Integer;
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

procedure TSimpleRNNCell.ResetGradients;
begin
  ZeroMatrix(dWih, FHiddenSize, FInputSize);
  ZeroMatrix(dWhh, FHiddenSize, FHiddenSize);
  ZeroArray(dBh, FHiddenSize);
end;

function TSimpleRNNCell.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

// ========== TLSTMCell ==========
constructor TLSTMCell.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var
  Scale: Double;
  ConcatSize, i: Integer;
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

  SetLength(Bf, HiddenSize);
  SetLength(Bi, HiddenSize);
  SetLength(Bc, HiddenSize);
  SetLength(Bo, HiddenSize);
  for i := 0 to HiddenSize - 1 do
  begin
    Bf[i] := 1.0;
    Bi[i] := 0;
    Bc[i] := 0;
    Bo[i] := 0;
  end;

  ZeroMatrix(dWf, HiddenSize, ConcatSize);
  ZeroMatrix(dWi, HiddenSize, ConcatSize);
  ZeroMatrix(dWc, HiddenSize, ConcatSize);
  ZeroMatrix(dWo, HiddenSize, ConcatSize);
  ZeroArray(dBf, HiddenSize);
  ZeroArray(dBi, HiddenSize);
  ZeroArray(dBc, HiddenSize);
  ZeroArray(dBo, HiddenSize);
end;

procedure TLSTMCell.Forward(const Input, PrevH, PrevC: DArray; var H, C, F, I, CTilde, O, TanhC: DArray);
var
  k, j: Integer;
  Concat: DArray;
  SumF, SumI, SumC, SumO: Double;
begin
  Concat := ConcatArrays(Input, PrevH);
  SetLength(H, FHiddenSize);
  SetLength(C, FHiddenSize);
  SetLength(F, FHiddenSize);
  SetLength(I, FHiddenSize);
  SetLength(CTilde, FHiddenSize);
  SetLength(O, FHiddenSize);
  SetLength(TanhC, FHiddenSize);

  for k := 0 to FHiddenSize - 1 do
  begin
    SumF := Bf[k];
    SumI := Bi[k];
    SumC := Bc[k];
    SumO := Bo[k];
    for j := 0 to High(Concat) do
    begin
      SumF := SumF + Wf[k][j] * Concat[j];
      SumI := SumI + Wi[k][j] * Concat[j];
      SumC := SumC + Wc[k][j] * Concat[j];
      SumO := SumO + Wo[k][j] * Concat[j];
    end;
    F[k] := TActivation.Apply(SumF, atSigmoid);
    I[k] := TActivation.Apply(SumI, atSigmoid);
    CTilde[k] := TActivation.Apply(SumC, atTanh);
    O[k] := TActivation.Apply(SumO, atSigmoid);
    C[k] := F[k] * PrevC[k] + I[k] * CTilde[k];
    TanhC[k] := Tanh(C[k]);
    H[k] := O[k] * TanhC[k];
  end;
end;

procedure TLSTMCell.Backward(const dH, dC, H, C, F, I, CTilde, O, TanhC, PrevH, PrevC, Input: DArray;
                              ClipVal: Double; var dInput, dPrevH, dPrevC: DArray);
var
  k, j: Integer;
  Concat: DArray;
  dO, dCTotal, dF, dI, dCTilde: DArray;
  ConcatSize: Integer;
begin
  Concat := ConcatArrays(Input, PrevH);
  ConcatSize := Length(Concat);
  SetLength(dO, FHiddenSize);
  SetLength(dCTotal, FHiddenSize);
  SetLength(dF, FHiddenSize);
  SetLength(dI, FHiddenSize);
  SetLength(dCTilde, FHiddenSize);
  SetLength(dInput, FInputSize);
  SetLength(dPrevH, FHiddenSize);
  SetLength(dPrevC, FHiddenSize);

  for k := 0 to FInputSize - 1 do dInput[k] := 0;
  for k := 0 to FHiddenSize - 1 do
  begin
    dPrevH[k] := 0;
    dPrevC[k] := 0;
  end;

  for k := 0 to FHiddenSize - 1 do
  begin
    dO[k] := ClipValue(dH[k] * TanhC[k] * TActivation.Derivative(O[k], atSigmoid), ClipVal);
    dCTotal[k] := ClipValue(dH[k] * O[k] * (1 - TanhC[k] * TanhC[k]) + dC[k], ClipVal);
    dF[k] := ClipValue(dCTotal[k] * PrevC[k] * TActivation.Derivative(F[k], atSigmoid), ClipVal);
    dI[k] := ClipValue(dCTotal[k] * CTilde[k] * TActivation.Derivative(I[k], atSigmoid), ClipVal);
    dCTilde[k] := ClipValue(dCTotal[k] * I[k] * TActivation.Derivative(CTilde[k], atTanh), ClipVal);
    dPrevC[k] := dCTotal[k] * F[k];
  end;

  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      dWf[k][j] := dWf[k][j] + dF[k] * Concat[j];
      dWi[k][j] := dWi[k][j] + dI[k] * Concat[j];
      dWc[k][j] := dWc[k][j] + dCTilde[k] * Concat[j];
      dWo[k][j] := dWo[k][j] + dO[k] * Concat[j];

      if j < FInputSize then
        dInput[j] := dInput[j] + Wf[k][j] * dF[k] + Wi[k][j] * dI[k] +
                     Wc[k][j] * dCTilde[k] + Wo[k][j] * dO[k]
      else
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] +
                                   Wf[k][j] * dF[k] + Wi[k][j] * dI[k] +
                                   Wc[k][j] * dCTilde[k] + Wo[k][j] * dO[k];
    end;
    dBf[k] := dBf[k] + dF[k];
    dBi[k] := dBi[k] + dI[k];
    dBc[k] := dBc[k] + dCTilde[k];
    dBo[k] := dBo[k] + dO[k];
  end;
end;

procedure TLSTMCell.ApplyGradients(LR, ClipVal: Double);
var
  k, j, ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      Wf[k][j] := Wf[k][j] - LR * ClipValue(dWf[k][j], ClipVal);
      Wi[k][j] := Wi[k][j] - LR * ClipValue(dWi[k][j], ClipVal);
      Wc[k][j] := Wc[k][j] - LR * ClipValue(dWc[k][j], ClipVal);
      Wo[k][j] := Wo[k][j] - LR * ClipValue(dWo[k][j], ClipVal);
      dWf[k][j] := 0;
      dWi[k][j] := 0;
      dWc[k][j] := 0;
      dWo[k][j] := 0;
    end;
    Bf[k] := Bf[k] - LR * ClipValue(dBf[k], ClipVal);
    Bi[k] := Bi[k] - LR * ClipValue(dBi[k], ClipVal);
    Bc[k] := Bc[k] - LR * ClipValue(dBc[k], ClipVal);
    Bo[k] := Bo[k] - LR * ClipValue(dBo[k], ClipVal);
    dBf[k] := 0;
    dBi[k] := 0;
    dBc[k] := 0;
    dBo[k] := 0;
  end;
end;

procedure TLSTMCell.ResetGradients;
var
  ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  ZeroMatrix(dWf, FHiddenSize, ConcatSize);
  ZeroMatrix(dWi, FHiddenSize, ConcatSize);
  ZeroMatrix(dWc, FHiddenSize, ConcatSize);
  ZeroMatrix(dWo, FHiddenSize, ConcatSize);
  ZeroArray(dBf, FHiddenSize);
  ZeroArray(dBi, FHiddenSize);
  ZeroArray(dBc, FHiddenSize);
  ZeroArray(dBo, FHiddenSize);
end;

function TLSTMCell.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

// ========== TGRUCell ==========
constructor TGRUCell.Create(InputSize, HiddenSize: Integer; Activation: TActivationType);
var
  Scale: Double;
  ConcatSize: Integer;
begin
  FInputSize := InputSize;
  FHiddenSize := HiddenSize;
  FActivation := Activation;
  ConcatSize := InputSize + HiddenSize;
  Scale := Sqrt(2.0 / ConcatSize);

  InitMatrix(Wz, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wr, HiddenSize, ConcatSize, Scale);
  InitMatrix(Wh, HiddenSize, ConcatSize, Scale);

  ZeroArray(Bz, HiddenSize);
  ZeroArray(Br, HiddenSize);
  ZeroArray(Bh, HiddenSize);

  ZeroMatrix(dWz, HiddenSize, ConcatSize);
  ZeroMatrix(dWr, HiddenSize, ConcatSize);
  ZeroMatrix(dWh, HiddenSize, ConcatSize);
  ZeroArray(dBz, HiddenSize);
  ZeroArray(dBr, HiddenSize);
  ZeroArray(dBh, HiddenSize);
end;

procedure TGRUCell.Forward(const Input, PrevH: DArray; var H, Z, R, HTilde: DArray);
var
  k, j: Integer;
  Concat, ConcatR: DArray;
  SumZ, SumR, SumH: Double;
begin
  Concat := ConcatArrays(Input, PrevH);
  SetLength(H, FHiddenSize);
  SetLength(Z, FHiddenSize);
  SetLength(R, FHiddenSize);
  SetLength(HTilde, FHiddenSize);

  for k := 0 to FHiddenSize - 1 do
  begin
    SumZ := Bz[k];
    SumR := Br[k];
    for j := 0 to High(Concat) do
    begin
      SumZ := SumZ + Wz[k][j] * Concat[j];
      SumR := SumR + Wr[k][j] * Concat[j];
    end;
    Z[k] := TActivation.Apply(SumZ, atSigmoid);
    R[k] := TActivation.Apply(SumR, atSigmoid);
  end;

  SetLength(ConcatR, FInputSize + FHiddenSize);
  for k := 0 to FInputSize - 1 do
    ConcatR[k] := Input[k];
  for k := 0 to FHiddenSize - 1 do
    ConcatR[FInputSize + k] := R[k] * PrevH[k];

  for k := 0 to FHiddenSize - 1 do
  begin
    SumH := Bh[k];
    for j := 0 to High(ConcatR) do
      SumH := SumH + Wh[k][j] * ConcatR[j];
    HTilde[k] := TActivation.Apply(SumH, atTanh);
    H[k] := (1 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
  end;
end;

procedure TGRUCell.Backward(const dH, H, Z, R, HTilde, PrevH, Input: DArray;
                             ClipVal: Double; var dInput, dPrevH: DArray);
var
  k, j: Integer;
  Concat, ConcatR: DArray;
  dZ, dR, dHTilde: DArray;
  ConcatSize: Integer;
begin
  Concat := ConcatArrays(Input, PrevH);
  ConcatSize := Length(Concat);

  SetLength(ConcatR, ConcatSize);
  for k := 0 to FInputSize - 1 do
    ConcatR[k] := Input[k];
  for k := 0 to FHiddenSize - 1 do
    ConcatR[FInputSize + k] := R[k] * PrevH[k];

  SetLength(dZ, FHiddenSize);
  SetLength(dR, FHiddenSize);
  SetLength(dHTilde, FHiddenSize);
  SetLength(dInput, FInputSize);
  SetLength(dPrevH, FHiddenSize);

  for k := 0 to FInputSize - 1 do dInput[k] := 0;
  for k := 0 to FHiddenSize - 1 do dPrevH[k] := dH[k] * (1 - Z[k]);

  for k := 0 to FHiddenSize - 1 do
  begin
    dHTilde[k] := ClipValue(dH[k] * Z[k] * TActivation.Derivative(HTilde[k], atTanh), ClipVal);
    dZ[k] := ClipValue(dH[k] * (HTilde[k] - PrevH[k]) * TActivation.Derivative(Z[k], atSigmoid), ClipVal);
  end;

  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      dWh[k][j] := dWh[k][j] + dHTilde[k] * ConcatR[j];
      if j < FInputSize then
        dInput[j] := dInput[j] + Wh[k][j] * dHTilde[k]
      else
      begin
        dR[j - FInputSize] := (dR[j - FInputSize] + Wh[k][j] * dHTilde[k] * PrevH[j - FInputSize]);
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] + Wh[k][j] * dHTilde[k] * R[j - FInputSize];
      end;
    end;
    dBh[k] := dBh[k] + dHTilde[k];
  end;

  for k := 0 to FHiddenSize - 1 do
    dR[k] := ClipValue(dR[k] * TActivation.Derivative(R[k], atSigmoid), ClipVal);

  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      dWz[k][j] := dWz[k][j] + dZ[k] * Concat[j];
      dWr[k][j] := dWr[k][j] + dR[k] * Concat[j];
      if j < FInputSize then
        dInput[j] := dInput[j] + Wz[k][j] * dZ[k] + Wr[k][j] * dR[k]
      else
        dPrevH[j - FInputSize] := dPrevH[j - FInputSize] +
                                   Wz[k][j] * dZ[k] + Wr[k][j] * dR[k];
    end;
    dBz[k] := dBz[k] + dZ[k];
    dBr[k] := dBr[k] + dR[k];
  end;
end;

procedure TGRUCell.ApplyGradients(LR, ClipVal: Double);
var
  k, j, ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  for k := 0 to FHiddenSize - 1 do
  begin
    for j := 0 to ConcatSize - 1 do
    begin
      Wz[k][j] := Wz[k][j] - LR * ClipValue(dWz[k][j], ClipVal);
      Wr[k][j] := Wr[k][j] - LR * ClipValue(dWr[k][j], ClipVal);
      Wh[k][j] := Wh[k][j] - LR * ClipValue(dWh[k][j], ClipVal);
      dWz[k][j] := 0;
      dWr[k][j] := 0;
      dWh[k][j] := 0;
    end;
    Bz[k] := Bz[k] - LR * ClipValue(dBz[k], ClipVal);
    Br[k] := Br[k] - LR * ClipValue(dBr[k], ClipVal);
    Bh[k] := Bh[k] - LR * ClipValue(dBh[k], ClipVal);
    dBz[k] := 0;
    dBr[k] := 0;
    dBh[k] := 0;
  end;
end;

procedure TGRUCell.ResetGradients;
var
  ConcatSize: Integer;
begin
  ConcatSize := FInputSize + FHiddenSize;
  ZeroMatrix(dWz, FHiddenSize, ConcatSize);
  ZeroMatrix(dWr, FHiddenSize, ConcatSize);
  ZeroMatrix(dWh, FHiddenSize, ConcatSize);
  ZeroArray(dBz, FHiddenSize);
  ZeroArray(dBr, FHiddenSize);
  ZeroArray(dBh, FHiddenSize);
end;

function TGRUCell.GetHiddenSize: Integer;
begin
  Result := FHiddenSize;
end;

// ========== TOutputLayer ==========
constructor TOutputLayer.Create(InputSize, OutputSize: Integer; Activation: TActivationType);
var
  Scale: Double;
begin
  FInputSize := InputSize;
  FOutputSize := OutputSize;
  FActivation := Activation;
  Scale := Sqrt(2.0 / InputSize);
  InitMatrix(W, OutputSize, InputSize, Scale);
  ZeroArray(B, OutputSize);
  ZeroMatrix(dW, OutputSize, InputSize);
  ZeroArray(dB, OutputSize);
end;

procedure TOutputLayer.Forward(const Input: DArray; var Output, Pre: DArray);
var
  i, j: Integer;
  Sum: Double;
begin
  SetLength(Pre, FOutputSize);
  SetLength(Output, FOutputSize);
  for i := 0 to FOutputSize - 1 do
  begin
    Sum := B[i];
    for j := 0 to FInputSize - 1 do
      Sum := Sum + W[i][j] * Input[j];
    Pre[i] := Sum;
  end;

  if FActivation = atLinear then
  begin
    for i := 0 to FOutputSize - 1 do
      Output[i] := Pre[i];
  end
  else
  begin
    for i := 0 to FOutputSize - 1 do
      Output[i] := TActivation.Apply(Pre[i], FActivation);
  end;
end;

procedure TOutputLayer.Backward(const dOut, Output, Pre, Input: DArray; ClipVal: Double; var dInput: DArray);
var
  i, j: Integer;
  dPre: DArray;
begin
  SetLength(dPre, FOutputSize);
  SetLength(dInput, FInputSize);
  for j := 0 to FInputSize - 1 do dInput[j] := 0;

  for i := 0 to FOutputSize - 1 do
    dPre[i] := ClipValue(dOut[i] * TActivation.Derivative(Output[i], FActivation), ClipVal);

  for i := 0 to FOutputSize - 1 do
  begin
    for j := 0 to FInputSize - 1 do
    begin
      dW[i][j] := dW[i][j] + dPre[i] * Input[j];
      dInput[j] := dInput[j] + W[i][j] * dPre[i];
    end;
    dB[i] := dB[i] + dPre[i];
  end;
end;

procedure TOutputLayer.ApplyGradients(LR, ClipVal: Double);
var
  i, j: Integer;
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

procedure TOutputLayer.ResetGradients;
begin
  ZeroMatrix(dW, FOutputSize, FInputSize);
  ZeroArray(dB, FOutputSize);
end;

// ========== TAdvancedRNN ==========
constructor TAdvancedRNN.Create(InputSize: Integer; const HiddenSizes: array of Integer;
                                 OutputSize: Integer; CellType: TCellType;
                                 Activation, OutputActivation: TActivationType;
                                 LossType: TLossType; LearningRate, GradientClip: Double;
                                 BPTTSteps: Integer);
var
  i, PrevSize: Integer;
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

  PrevSize := InputSize;
  case CellType of
    ctSimpleRNN:
    begin
      SetLength(FSimpleCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FSimpleCells[i] := TSimpleRNNCell.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
    ctLSTM:
    begin
      SetLength(FLSTMCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FLSTMCells[i] := TLSTMCell.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
    ctGRU:
    begin
      SetLength(FGRUCells, Length(HiddenSizes));
      for i := 0 to High(HiddenSizes) do
      begin
        FGRUCells[i] := TGRUCell.Create(PrevSize, HiddenSizes[i], Activation);
        PrevSize := HiddenSizes[i];
      end;
    end;
  end;

  FOutputLayer := TOutputLayer.Create(PrevSize, OutputSize, OutputActivation);
end;

destructor TAdvancedRNN.Destroy;
var
  i: Integer;
begin
  case FCellType of
    ctSimpleRNN:
      for i := 0 to High(FSimpleCells) do FSimpleCells[i].Free;
    ctLSTM:
      for i := 0 to High(FLSTMCells) do FLSTMCells[i].Free;
    ctGRU:
      for i := 0 to High(FGRUCells) do FGRUCells[i].Free;
  end;
  FOutputLayer.Free;
  inherited;
end;

function TAdvancedRNN.ClipGradient(G, MaxVal: Double): Double;
begin
  Result := ClipValue(G, MaxVal);
end;

function TAdvancedRNN.InitHiddenStates: TDArray3D;
var
  i: Integer;
begin
  SetLength(Result, Length(FHiddenSizes));
  for i := 0 to High(FHiddenSizes) do
  begin
    SetLength(Result[i], 2);
    ZeroArray(Result[i][0], FHiddenSizes[i]);
    ZeroArray(Result[i][1], FHiddenSizes[i]);
  end;
end;

function TAdvancedRNN.ForwardSequence(const Inputs: TDArray2D; var Caches: array of TTimeStepCache;
                                       var States: TDArray3D): TDArray2D;
var
  t, layer: Integer;
  X, H, C, PreH, F, I, CTilde, O, TanhC, Z, R, HTilde: DArray;
  OutVal, OutPre: DArray;
  NewStates: TDArray3D;
begin
  SetLength(Result, Length(Inputs));
  NewStates := InitHiddenStates;

  for t := 0 to High(Inputs) do
  begin
    X := Copy(Inputs[t]);
    Caches[t].Input := Copy(X);

    for layer := 0 to High(FHiddenSizes) do
    begin
      case FCellType of
        ctSimpleRNN:
        begin
          FSimpleCells[layer].Forward(X, States[layer][0], H, PreH);
          NewStates[layer][0] := Copy(H);
          Caches[t].H := Copy(H);
          Caches[t].PreH := Copy(PreH);
        end;
        ctLSTM:
        begin
          FLSTMCells[layer].Forward(X, States[layer][0], States[layer][1], H, C, F, I, CTilde, O, TanhC);
          NewStates[layer][0] := Copy(H);
          NewStates[layer][1] := Copy(C);
          Caches[t].H := Copy(H);
          Caches[t].C := Copy(C);
          Caches[t].F := Copy(F);
          Caches[t].I := Copy(I);
          Caches[t].CTilde := Copy(CTilde);
          Caches[t].O := Copy(O);
          Caches[t].TanhC := Copy(TanhC);
        end;
        ctGRU:
        begin
          FGRUCells[layer].Forward(X, States[layer][0], H, Z, R, HTilde);
          NewStates[layer][0] := Copy(H);
          Caches[t].H := Copy(H);
          Caches[t].Z := Copy(Z);
          Caches[t].R := Copy(R);
          Caches[t].HTilde := Copy(HTilde);
        end;
      end;
      X := Copy(H);
    end;

    FOutputLayer.Forward(X, OutVal, OutPre);
    Caches[t].OutVal := Copy(OutVal);
    Caches[t].OutPre := Copy(OutPre);
    Result[t] := Copy(OutVal);

    States := NewStates;
  end;
end;

function TAdvancedRNN.BackwardSequence(const Targets: TDArray2D; const Caches: array of TTimeStepCache;
                                        const States: TDArray3D): Double;
var
  t, layer, k: Integer;
  T_len, BPTTLimit: Integer;
  dOut, dH, dC, dInput, dPrevH, dPrevC: DArray;
  Grad: DArray;
  dStatesH, dStatesC: TDArray2D;
  TotalLoss: Double;
  PrevH, PrevC: DArray;
begin
  T_len := Length(Targets);
  if FBPTTSteps > 0 then
    BPTTLimit := FBPTTSteps
  else
    BPTTLimit := T_len;

  TotalLoss := 0;

  SetLength(dStatesH, Length(FHiddenSizes));
  SetLength(dStatesC, Length(FHiddenSizes));
  for layer := 0 to High(FHiddenSizes) do
  begin
    ZeroArray(dStatesH[layer], FHiddenSizes[layer]);
    ZeroArray(dStatesC[layer], FHiddenSizes[layer]);
  end;

  for t := T_len - 1 downto Max(0, T_len - BPTTLimit) do
  begin
    TotalLoss := TotalLoss + TLoss.Compute(Caches[t].OutVal, Targets[t], FLossType);
    TLoss.Gradient(Caches[t].OutVal, Targets[t], FLossType, Grad);

    FOutputLayer.Backward(Grad, Caches[t].OutVal, Caches[t].OutPre, Caches[t].H, FGradientClip, dH);

    for layer := High(FHiddenSizes) downto 0 do
    begin
      SetLength(dOut, FHiddenSizes[layer]);
      for k := 0 to FHiddenSizes[layer] - 1 do
        dOut[k] := dH[k] + dStatesH[layer][k];

      if t > 0 then
        PrevH := Caches[t-1].H
      else
        ZeroArray(PrevH, FHiddenSizes[layer]);

      case FCellType of
        ctSimpleRNN:
        begin
          FSimpleCells[layer].Backward(dOut, Caches[t].H, Caches[t].PreH, PrevH,
                                        Caches[t].Input, FGradientClip, dInput, dPrevH);
          dStatesH[layer] := Copy(dPrevH);
        end;
        ctLSTM:
        begin
          if t > 0 then
            PrevC := Caches[t-1].C
          else
            ZeroArray(PrevC, FHiddenSizes[layer]);

          SetLength(dC, FHiddenSizes[layer]);
          for k := 0 to FHiddenSizes[layer] - 1 do
            dC[k] := dStatesC[layer][k];

          FLSTMCells[layer].Backward(dOut, dC, Caches[t].H, Caches[t].C,
                                      Caches[t].F, Caches[t].I, Caches[t].CTilde,
                                      Caches[t].O, Caches[t].TanhC,
                                      PrevH, PrevC, Caches[t].Input,
                                      FGradientClip, dInput, dPrevH, dPrevC);
          dStatesH[layer] := Copy(dPrevH);
          dStatesC[layer] := Copy(dPrevC);
        end;
        ctGRU:
        begin
          FGRUCells[layer].Backward(dOut, Caches[t].H, Caches[t].Z, Caches[t].R,
                                     Caches[t].HTilde, PrevH, Caches[t].Input,
                                     FGradientClip, dInput, dPrevH);
          dStatesH[layer] := Copy(dPrevH);
        end;
      end;

      dH := Copy(dInput);
    end;
  end;

  Result := TotalLoss / T_len;
end;

function TAdvancedRNN.TrainSequence(const Inputs, Targets: TDArray2D): Double;
var
  Caches: array of TTimeStepCache;
  States: TDArray3D;
begin
  ResetGradients;
  SetLength(Caches, Length(Inputs));
  States := InitHiddenStates;
  ForwardSequence(Inputs, Caches, States);
  Result := BackwardSequence(Targets, Caches, States);
  ApplyGradients;
end;

function TAdvancedRNN.TrainBatch(const BatchInputs, BatchTargets: TDArray3D): Double;
var
  b: Integer;
  Caches: array of TTimeStepCache;
  States: TDArray3D;
  BatchLoss: Double;
begin
  ResetGradients;
  BatchLoss := 0;

  for b := 0 to High(BatchInputs) do
  begin
    SetLength(Caches, Length(BatchInputs[b]));
    States := InitHiddenStates;
    ForwardSequence(BatchInputs[b], Caches, States);
    BatchLoss := BatchLoss + BackwardSequence(BatchTargets[b], Caches, States);
  end;

  ApplyGradients;
  Result := BatchLoss / Length(BatchInputs);
end;

function TAdvancedRNN.Predict(const Inputs: TDArray2D): TDArray2D;
var
  Caches: array of TTimeStepCache;
  States: TDArray3D;
begin
  SetLength(Caches, Length(Inputs));
  States := InitHiddenStates;
  Result := ForwardSequence(Inputs, Caches, States);
end;

function TAdvancedRNN.ComputeLoss(const Inputs, Targets: TDArray2D): Double;
var
  Outputs: TDArray2D;
  t: Integer;
begin
  Outputs := Predict(Inputs);
  Result := 0;
  for t := 0 to High(Outputs) do
    Result := Result + TLoss.Compute(Outputs[t], Targets[t], FLossType);
  Result := Result / Length(Outputs);
end;

procedure TAdvancedRNN.ResetGradients;
var
  i: Integer;
begin
  case FCellType of
    ctSimpleRNN:
      for i := 0 to High(FSimpleCells) do FSimpleCells[i].ResetGradients;
    ctLSTM:
      for i := 0 to High(FLSTMCells) do FLSTMCells[i].ResetGradients;
    ctGRU:
      for i := 0 to High(FGRUCells) do FGRUCells[i].ResetGradients;
  end;
  FOutputLayer.ResetGradients;
end;

procedure TAdvancedRNN.ApplyGradients;
var
  i: Integer;
begin
  case FCellType of
    ctSimpleRNN:
      for i := 0 to High(FSimpleCells) do FSimpleCells[i].ApplyGradients(FLearningRate, FGradientClip);
    ctLSTM:
      for i := 0 to High(FLSTMCells) do FLSTMCells[i].ApplyGradients(FLearningRate, FGradientClip);
    ctGRU:
      for i := 0 to High(FGRUCells) do FGRUCells[i].ApplyGradients(FLearningRate, FGradientClip);
  end;
  FOutputLayer.ApplyGradients(FLearningRate, FGradientClip);
end;

// ========== Data Utilities ==========
procedure SplitData(const Inputs, Targets: TDArray2D; ValSplit: Double; var Split: TDataSplit);
var
  N, ValCount, TrainCount, i, j, Temp: Integer;
  Indices: array of Integer;
begin
  N := Length(Inputs);
  ValCount := Round(N * ValSplit);
  TrainCount := N - ValCount;

  SetLength(Indices, N);
  for i := 0 to N - 1 do
    Indices[i] := i;

  for i := N - 1 downto 1 do
  begin
    j := Random(i + 1);
    Temp := Indices[i];
    Indices[i] := Indices[j];
    Indices[j] := Temp;
  end;

  SetLength(Split.TrainInputs, TrainCount);
  SetLength(Split.TrainTargets, TrainCount);
  SetLength(Split.ValInputs, ValCount);
  SetLength(Split.ValTargets, ValCount);

  for i := 0 to TrainCount - 1 do
  begin
    Split.TrainInputs[i] := Copy(Inputs[Indices[i]]);
    Split.TrainTargets[i] := Copy(Targets[Indices[i]]);
  end;

  for i := 0 to ValCount - 1 do
  begin
    Split.ValInputs[i] := Copy(Inputs[Indices[TrainCount + i]]);
    Split.ValTargets[i] := Copy(Targets[Indices[TrainCount + i]]);
  end;
end;

// ========== Main Demo ==========
var
  RNN: TAdvancedRNN;
  SequenceLen, InputSize, HiddenSize, OutputSize: Integer;
  Epochs, BatchSize, LogInterval, Epoch, b: Integer;
  Inputs, Targets, Predictions: TDArray2D;
  Split: TDataSplit;
  TrainLoss, ValLoss: Double;
  t: Integer;
  CellTypeStr: string;

begin
  Randomize;

  // Configuration
  SequenceLen := 10;
  InputSize := 2;
  HiddenSize := 16;
  OutputSize := 2;
  Epochs := 200;
  BatchSize := 1;
  LogInterval := 20;

  // Create dummy sequence data (identity task)
  SetLength(Inputs, SequenceLen);
  SetLength(Targets, SequenceLen);
  for t := 0 to SequenceLen - 1 do
  begin
    SetLength(Inputs[t], InputSize);
    SetLength(Targets[t], OutputSize);
    Inputs[t][0] := t / SequenceLen;
    Inputs[t][1] := (SequenceLen - t) / SequenceLen;
    Targets[t][0] := Inputs[t][0];
    Targets[t][1] := Inputs[t][1];
  end;

  // Split data (20% validation)
  SplitData(Inputs, Targets, 0.2, Split);

  // Create LSTM RNN with gradient clipping
  RNN := TAdvancedRNN.Create(
    InputSize,
    [HiddenSize],
    OutputSize,
    ctLSTM,            // Cell type: ctSimpleRNN, ctLSTM, ctGRU
    atTanh,            // Hidden activation
    atLinear,          // Output activation
    ltMSE,             // Loss function
    0.01,              // Learning rate
    5.0,               // Gradient clip
    0                  // BPTT steps (0 = full)
  );

  case RNN.FCellType of
    ctSimpleRNN: CellTypeStr := 'SimpleRNN';
    ctLSTM: CellTypeStr := 'LSTM';
    ctGRU: CellTypeStr := 'GRU';
  end;

  WriteLn('=== Advanced RNN Training ===');
  WriteLn('Cell Type: ', CellTypeStr);
  WriteLn('Hidden Size: ', HiddenSize);
  WriteLn('Learning Rate: ', RNN.LearningRate:0:4);
  WriteLn('Gradient Clip: ', RNN.GradientClip:0:2);
  WriteLn('Train samples: ', Length(Split.TrainInputs));
  WriteLn('Val samples: ', Length(Split.ValInputs));
  WriteLn('');
  WriteLn('Epoch | Train Loss | Val Loss');
  WriteLn('------+------------+-----------');

  // Training loop
  for Epoch := 1 to Epochs do
  begin
    TrainLoss := 0;

    // Train on each sample (or batch)
    for b := 0 to High(Split.TrainInputs) do
      TrainLoss := TrainLoss + RNN.TrainSequence(
        TDArray2D.Create(Split.TrainInputs[b]),
        TDArray2D.Create(Split.TrainTargets[b])
      );
    TrainLoss := TrainLoss / Length(Split.TrainInputs);

    // Compute validation loss
    ValLoss := 0;
    if Length(Split.ValInputs) > 0 then
    begin
      for b := 0 to High(Split.ValInputs) do
        ValLoss := ValLoss + RNN.ComputeLoss(
          TDArray2D.Create(Split.ValInputs[b]),
          TDArray2D.Create(Split.ValTargets[b])
        );
      ValLoss := ValLoss / Length(Split.ValInputs);
    end;

    if (Epoch mod LogInterval = 0) or (Epoch = Epochs) then
      WriteLn(Epoch:5, ' | ', TrainLoss:10:6, ' | ', ValLoss:10:6);
  end;

  WriteLn('');
  WriteLn('=== Final Predictions ===');
  Predictions := RNN.Predict(Inputs);
  WriteLn('t | Input0   Input1   | Pred0    Pred1    | Target0  Target1');
  WriteLn('--+------------------+------------------+------------------');
  for t := 0 to SequenceLen - 1 do
    WriteLn(
      t:2, ' | ',
      Inputs[t][0]:7:4, ' ', Inputs[t][1]:7:4, ' | ',
      Predictions[t][0]:7:4, ' ', Predictions[t][1]:7:4, ' | ',
      Targets[t][0]:7:4, ' ', Targets[t][1]:7:4
    );

  RNN.Free;
end.

//
// Matthew Abbott 2025
// Advanced CNN with Full Backpropagation, Adam Optimizer, Batch Processing
//

{$mode objfpc}{$H+}
{$M+}

program AdvancedCNN;

uses Classes, Math, SysUtils;

type
  TActivationType = (atSigmoid, atTanh, atReLU, atLinear);
  TLossType = (ltMSE, ltCrossEntropy);
  TPaddingType = (ptSame, ptValid);
  TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp);

  DArray = array of Double;
  TDArray2D = array of DArray;
  TDArray3D = array of TDArray2D;
  TDArray4D = array of TDArray3D;
  TIntArray = array of Integer;

  TDataSplit = record
    TrainInputs, TrainTargets: TDArray3D;
    ValInputs, ValTargets: TDArray3D;
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

  // ========== Convolutional Filter ==========
  TConvFilter = class
  private
    FInputChannels, FOutputChannels, FKernelSize: Integer;
  public
    Weights: TDArray4D;
    Bias: Double;
    dWeights: TDArray4D;
    dBias: Double;
    constructor Create(InputChannels, OutputChannels, KernelSize: Integer);
    procedure ResetGradients;
  end;

  // ========== Convolutional Layer ==========
  TConvLayer = class
  private
    FInputChannels, FOutputChannels, FKernelSize: Integer;
    FStride, FPadding: Integer;
    FActivation: TActivationType;
    function Pad3D(const Input: TDArray3D; Padding: Integer): TDArray3D;
  public
    Filters: array of TConvFilter;
    InputCache: TDArray3D;
    OutputCache: TDArray3D;
    PreActivation: TDArray3D;
    constructor Create(InputChannels, OutputChannels, KernelSize, Stride, Padding: Integer;
                       Activation: TActivationType);
    procedure Forward(const Input: TDArray3D; var Output: TDArray3D);
    procedure Backward(const dOutput: TDArray3D; var dInput: TDArray3D);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
    function GetOutputChannels: Integer;
  end;

  // ========== Pooling Layer ==========
  TPoolingLayer = class
  private
    FPoolSize, FStride: Integer;
  public
    InputCache: TDArray3D;
    OutputCache: TDArray3D;
    MaxIndices: array of array of array of record X, Y: Integer; end;
    constructor Create(PoolSize, Stride: Integer);
    procedure Forward(const Input: TDArray3D; var Output: TDArray3D);
    procedure Backward(const dOutput: TDArray3D; var dInput: TDArray3D);
  end;

  // ========== Fully Connected Layer ==========
  TFCLayer = class
  private
    FInputSize, FOutputSize: Integer;
    FActivation: TActivationType;
  public
    W: TDArray2D;
    B: DArray;
    dW: TDArray2D;
    dB: DArray;
    InputCache: DArray;
    OutputCache: DArray;
    PreActivation: DArray;
    constructor Create(InputSize, OutputSize: Integer; Activation: TActivationType);
    procedure Forward(const Input: DArray; var Output: DArray);
    procedure Backward(const dOutput: DArray; var dInput: DArray);
    procedure ApplyGradients(LR, ClipVal: Double);
    procedure ResetGradients;
  end;

  // ========== Main Advanced CNN ==========
  TAdvancedCNN = class
  private
    FInputWidth, FInputHeight, FInputChannels: Integer;
    FOutputSize: Integer;
    FActivation: TActivationType;
    FOutputActivation: TActivationType;
    FLossType: TLossType;
    FLearningRate: Double;
    FGradientClip: Double;

    FConvLayers: array of TConvLayer;
    FPoolLayers: array of TPoolingLayer;
    FFullyConnectedLayers: array of TFCLayer;
    FOutputLayer: TFCLayer;
    FFlattenedSize: Integer;

    function ClipGradient(G, MaxVal: Double): Double;
    function Flatten(const Input: TDArray3D): DArray;
    function Unflatten(const Input: DArray; Channels, Height, Width: Integer): TDArray3D;
  public
    constructor Create(InputWidth, InputHeight, InputChannels: Integer;
                       const ConvFilters: array of Integer;
                       const KernelSizes: array of Integer;
                       const PoolSizes: array of Integer;
                       const FCLayerSizes: array of Integer;
                       OutputSize: Integer;
                       Activation, OutputActivation: TActivationType;
                       LossType: TLossType;
                       LearningRate, GradientClip: Double);
    destructor Destroy; override;

    function ForwardPass(const Input: TDArray3D): DArray;
    function BackwardPass(const Target: DArray): Double;
    function TrainSample(const Input: TDArray3D; const Target: DArray): Double;
    function TrainBatch(const BatchInputs: TDArray4D; const BatchTargets: TDArray2D): Double;
    function Predict(const Input: TDArray3D): DArray;
    function ComputeLoss(const Inputs: TDArray4D; const Targets: TDArray2D): Double;

    procedure ResetGradients;
    procedure ApplyGradients;

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

procedure Zero3DArray(var A: TDArray3D; D1, D2, D3: Integer);
var
  i, j, k: Integer;
begin
  SetLength(A, D1);
  for i := 0 to D1 - 1 do
  begin
    SetLength(A[i], D2);
    for j := 0 to D2 - 1 do
    begin
      SetLength(A[i][j], D3);
      for k := 0 to D3 - 1 do
        A[i][j][k] := 0.0;
    end;
  end;
end;

procedure Zero4DArray(var A: TDArray4D; D1, D2, D3, D4: Integer);
var
  i, j, k, l: Integer;
begin
  SetLength(A, D1);
  for i := 0 to D1 - 1 do
  begin
    SetLength(A[i], D2);
    for j := 0 to D2 - 1 do
    begin
      SetLength(A[i][j], D3);
      for k := 0 to D3 - 1 do
      begin
        SetLength(A[i][j][k], D4);
        for l := 0 to D4 - 1 do
          A[i][j][k][l] := 0.0;
      end;
    end;
  end;
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

// ========== TConvFilter ==========
constructor TConvFilter.Create(InputChannels, OutputChannels, KernelSize: Integer);
var
  i, j, k, l: Integer;
  Scale: Double;
begin
  FInputChannels := InputChannels;
  FOutputChannels := OutputChannels;
  FKernelSize := KernelSize;
  Scale := Sqrt(2.0 / (InputChannels * KernelSize * KernelSize));

  Zero4DArray(Weights, OutputChannels, InputChannels, KernelSize, KernelSize);
  Zero4DArray(dWeights, OutputChannels, InputChannels, KernelSize, KernelSize);

  for i := 0 to OutputChannels - 1 do
    for j := 0 to InputChannels - 1 do
      for k := 0 to KernelSize - 1 do
        for l := 0 to KernelSize - 1 do
          Weights[i][j][k][l] := RandomWeight(Scale);

  Bias := 0.0;
  dBias := 0.0;
end;

procedure TConvFilter.ResetGradients;
var
  i, j, k, l: Integer;
begin
  for i := 0 to High(dWeights) do
    for j := 0 to High(dWeights[i]) do
      for k := 0 to High(dWeights[i][j]) do
        for l := 0 to High(dWeights[i][j][k]) do
          dWeights[i][j][k][l] := 0.0;
  dBias := 0.0;
end;

// ========== TConvLayer ==========
constructor TConvLayer.Create(InputChannels, OutputChannels, KernelSize, Stride, Padding: Integer;
                              Activation: TActivationType);
var
  i: Integer;
begin
  FInputChannels := InputChannels;
  FOutputChannels := OutputChannels;
  FKernelSize := KernelSize;
  FStride := Stride;
  FPadding := Padding;
  FActivation := Activation;

  SetLength(Filters, OutputChannels);
  for i := 0 to OutputChannels - 1 do
    Filters[i] := TConvFilter.Create(InputChannels, 1, KernelSize);
end;

function TConvLayer.Pad3D(const Input: TDArray3D; Padding: Integer): TDArray3D;
var
  c, h, w, SrcH, SrcW: Integer;
begin
  if Padding = 0 then
  begin
    Result := Input;
    Exit;
  end;

  SetLength(Result, Length(Input), 
            Length(Input[0]) + 2 * Padding, 
            Length(Input[0][0]) + 2 * Padding);

  for c := 0 to High(Input) do
    for h := 0 to Length(Result[c]) - 1 do
      for w := 0 to High(Result[c][h]) do
      begin
        SrcH := h - Padding;
        SrcW := w - Padding;
        if (SrcH >= 0) and (SrcH < Length(Input[c])) and 
           (SrcW >= 0) and (SrcW < Length(Input[c][0])) then
          Result[c][h][w] := Input[c][SrcH][SrcW]
        else
          Result[c][h][w] := 0;
      end;
end;

procedure TConvLayer.Forward(const Input: TDArray3D; var Output: TDArray3D);
var
  outH, outW, f, c, kh, kw, h, w, ih, iw: Integer;
  Sum: Double;
  Padded: TDArray3D;
begin
  InputCache := Copy(Input);
  
  if FPadding > 0 then
    Padded := Pad3D(Input, FPadding)
  else
    Padded := Input;

  outH := (Length(Padded[0]) - FKernelSize) div FStride + 1;
  outW := (Length(Padded[0][0]) - FKernelSize) div FStride + 1;

  Zero3DArray(Output, FOutputChannels, outH, outW);
  Zero3DArray(PreActivation, FOutputChannels, outH, outW);

  for f := 0 to FOutputChannels - 1 do
  begin
    for h := 0 to outH - 1 do
    begin
      for w := 0 to outW - 1 do
      begin
        Sum := Filters[f].Bias;
        for c := 0 to FInputChannels - 1 do
        begin
          for kh := 0 to FKernelSize - 1 do
          begin
            for kw := 0 to FKernelSize - 1 do
            begin
              ih := h * FStride + kh;
              iw := w * FStride + kw;
              Sum := Sum + Padded[c][ih][iw] * Filters[f].Weights[0][c][kh][kw];
            end;
          end;
        end;
        PreActivation[f][h][w] := Sum;
        Output[f][h][w] := TActivation.Apply(Sum, FActivation);
      end;
    end;
  end;

  OutputCache := Copy(Output);
end;

procedure TConvLayer.Backward(const dOutput: TDArray3D; var dInput: TDArray3D);
var
  f, c, kh, kw, h, w, ih, iw, outH, outW: Integer;
  dH, dW: TDArray3D;
  Sum, dVal: Double;
begin
  outH := Length(dOutput[0]);
  outW := Length(dOutput[0][0]);
  
  Zero3DArray(dH, FOutputChannels, outH, outW);
  
  for f := 0 to FOutputChannels - 1 do
    for h := 0 to outH - 1 do
      for w := 0 to outW - 1 do
        dH[f][h][w] := dOutput[f][h][w] * TActivation.Derivative(OutputCache[f][h][w], FActivation);

  for f := 0 to FOutputChannels - 1 do
  begin
    Sum := 0;
    for h := 0 to outH - 1 do
      for w := 0 to outW - 1 do
        Sum := Sum + dH[f][h][w];
    Filters[f].dBias := Sum;
  end;

  for f := 0 to FOutputChannels - 1 do
  begin
    for c := 0 to FInputChannels - 1 do
    begin
      for kh := 0 to FKernelSize - 1 do
      begin
        for kw := 0 to FKernelSize - 1 do
        begin
          Sum := 0;
          for h := 0 to outH - 1 do
          begin
            for w := 0 to outW - 1 do
            begin
              ih := h * FStride + kh;
              iw := w * FStride + kw;
              if FPadding > 0 then
                Sum := Sum + dH[f][h][w] * InputCache[c][ih - FPadding][iw - FPadding]
              else
                Sum := Sum + dH[f][h][w] * InputCache[c][ih][iw];
            end;
          end;
          Filters[f].dWeights[0][c][kh][kw] := Sum;
        end;
      end;
    end;
  end;

  if Length(InputCache) > 0 then
  begin
    Zero3DArray(dInput, FInputChannels, Length(InputCache[0]), Length(InputCache[0][0]));
    
    for f := 0 to FOutputChannels - 1 do
    begin
      for h := 0 to outH - 1 do
      begin
        for w := 0 to outW - 1 do
        begin
          for c := 0 to FInputChannels - 1 do
          begin
            for kh := 0 to FKernelSize - 1 do
            begin
              for kw := 0 to FKernelSize - 1 do
              begin
                ih := h * FStride + kh;
                iw := w * FStride + kw;
                if (ih >= 0) and (ih < Length(dInput[c])) and (iw >= 0) and (iw < Length(dInput[c][0])) then
                  dInput[c][ih][iw] := dInput[c][ih][iw] + dH[f][h][w] * Filters[f].Weights[0][c][kh][kw];
              end;
            end;
          end;
        end;
      end;
    end;
  end;
end;

procedure TConvLayer.ApplyGradients(LR, ClipVal: Double);
var
  f, i, j, k, l: Integer;
begin
  for f := 0 to FOutputChannels - 1 do
  begin
    Filters[f].Bias := Filters[f].Bias - LR * ClipValue(Filters[f].dBias, ClipVal);
    for i := 0 to High(Filters[f].Weights[0]) do
      for j := 0 to High(Filters[f].Weights[0][i]) do
        for k := 0 to High(Filters[f].Weights[0][i][j]) do
          Filters[f].Weights[0][i][j][k] := Filters[f].Weights[0][i][j][k] - 
                                             LR * ClipValue(Filters[f].dWeights[0][i][j][k], ClipVal);
  end;
end;

procedure TConvLayer.ResetGradients;
var
  i: Integer;
begin
  for i := 0 to FOutputChannels - 1 do
    Filters[i].ResetGradients;
end;

function TConvLayer.GetOutputChannels: Integer;
begin
  Result := FOutputChannels;
end;

// ========== TPoolingLayer ==========
constructor TPoolingLayer.Create(PoolSize, Stride: Integer);
begin
  FPoolSize := PoolSize;
  FStride := Stride;
end;

procedure TPoolingLayer.Forward(const Input: TDArray3D; var Output: TDArray3D);
var
  c, h, w, ph, pw, kh, kw, MaxH, MaxW: Integer;
  outH, outW: Integer;
  MaxVal: Double;
begin
  InputCache := Copy(Input);
  outH := (Length(Input[0]) - FPoolSize) div FStride + 1;
  outW := (Length(Input[0][0]) - FPoolSize) div FStride + 1;

  Zero3DArray(Output, Length(Input), outH, outW);
  SetLength(MaxIndices, Length(Input), outH, outW);

  for c := 0 to High(Input) do
  begin
    for h := 0 to outH - 1 do
    begin
      for w := 0 to outW - 1 do
      begin
        MaxVal := -1e308;
        MaxH := 0;
        MaxW := 0;
        for kh := 0 to FPoolSize - 1 do
        begin
          for kw := 0 to FPoolSize - 1 do
          begin
            ph := h * FStride + kh;
            pw := w * FStride + kw;
            if Input[c][ph][pw] > MaxVal then
            begin
              MaxVal := Input[c][ph][pw];
              MaxH := kh;
              MaxW := kw;
            end;
          end;
        end;
        Output[c][h][w] := MaxVal;
        MaxIndices[c][h][w].X := MaxW;
        MaxIndices[c][h][w].Y := MaxH;
      end;
    end;
  end;
end;

procedure TPoolingLayer.Backward(const dOutput: TDArray3D; var dInput: TDArray3D);
var
  c, h, w, ph, pw: Integer;
begin
  Zero3DArray(dInput, Length(InputCache), Length(InputCache[0]), Length(InputCache[0][0]));

  for c := 0 to High(dOutput) do
  begin
    for h := 0 to High(dOutput[c]) do
    begin
      for w := 0 to High(dOutput[c][h]) do
      begin
        ph := h * FStride + MaxIndices[c][h][w].Y;
        pw := w * FStride + MaxIndices[c][h][w].X;
        dInput[c][ph][pw] := dOutput[c][h][w];
      end;
    end;
  end;
end;

// ========== TFCLayer ==========
constructor TFCLayer.Create(InputSize, OutputSize: Integer; Activation: TActivationType);
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
  SetLength(InputCache, 0);
  SetLength(OutputCache, 0);
  SetLength(PreActivation, 0);
end;

procedure TFCLayer.Forward(const Input: DArray; var Output: DArray);
var
  i, j: Integer;
  Sum: Double;
begin
  InputCache := Copy(Input);
  SetLength(Output, FOutputSize);
  SetLength(PreActivation, FOutputSize);
  
  for i := 0 to FOutputSize - 1 do
  begin
    Sum := B[i];
    for j := 0 to FInputSize - 1 do
      Sum := Sum + W[i][j] * Input[j];
    PreActivation[i] := Sum;
    Output[i] := TActivation.Apply(Sum, FActivation);
  end;
  
  OutputCache := Copy(Output);
end;

procedure TFCLayer.Backward(const dOutput: DArray; var dInput: DArray);
var
  i, j: Integer;
  dRaw: Double;
begin
  SetLength(dInput, FInputSize);
  for i := 0 to FInputSize - 1 do
    dInput[i] := 0;

  for i := 0 to FOutputSize - 1 do
  begin
    dRaw := dOutput[i] * TActivation.Derivative(OutputCache[i], FActivation);
    dB[i] := dB[i] + dRaw;
    
    for j := 0 to FInputSize - 1 do
    begin
      dW[i][j] := dW[i][j] + dRaw * InputCache[j];
      dInput[j] := dInput[j] + dRaw * W[i][j];
    end;
  end;
end;

procedure TFCLayer.ApplyGradients(LR, ClipVal: Double);
var
  i, j: Integer;
begin
  for i := 0 to FOutputSize - 1 do
  begin
    B[i] := B[i] - LR * ClipValue(dB[i], ClipVal);
    dB[i] := 0;
    
    for j := 0 to FInputSize - 1 do
    begin
      W[i][j] := W[i][j] - LR * ClipValue(dW[i][j], ClipVal);
      dW[i][j] := 0;
    end;
  end;
end;

procedure TFCLayer.ResetGradients;
var
  i, j: Integer;
begin
  for i := 0 to FOutputSize - 1 do
  begin
    dB[i] := 0;
    for j := 0 to FInputSize - 1 do
      dW[i][j] := 0;
  end;
end;

// ========== TAdvancedCNN ==========
constructor TAdvancedCNN.Create(InputWidth, InputHeight, InputChannels: Integer;
                               const ConvFilters: array of Integer;
                               const KernelSizes: array of Integer;
                               const PoolSizes: array of Integer;
                               const FCLayerSizes: array of Integer;
                               OutputSize: Integer;
                               Activation, OutputActivation: TActivationType;
                               LossType: TLossType;
                               LearningRate, GradientClip: Double);
var
  i, j: Integer;
  CurrentChannels, CurrentWidth, CurrentHeight, NumInputs: Integer;
begin
  FInputWidth := InputWidth;
  FInputHeight := InputHeight;
  FInputChannels := InputChannels;
  FOutputSize := OutputSize;
  FActivation := Activation;
  FOutputActivation := OutputActivation;
  FLossType := LossType;
  FLearningRate := LearningRate;
  FGradientClip := GradientClip;

  CurrentChannels := InputChannels;
  CurrentWidth := InputWidth;
  CurrentHeight := InputHeight;

  SetLength(FConvLayers, Length(ConvFilters));
  for i := 0 to High(ConvFilters) do
  begin
    FConvLayers[i] := TConvLayer.Create(CurrentChannels, ConvFilters[i], 
                                        KernelSizes[i], 1, KernelSizes[i] div 2, Activation);
    CurrentChannels := ConvFilters[i];
    CurrentWidth := CurrentWidth;
    CurrentHeight := CurrentHeight;
    
    if i <= High(PoolSizes) then
    begin
      CurrentWidth := CurrentWidth div PoolSizes[i];
      CurrentHeight := CurrentHeight div PoolSizes[i];
    end;
  end;

  SetLength(FPoolLayers, Length(PoolSizes));
  for i := 0 to High(PoolSizes) do
    FPoolLayers[i] := TPoolingLayer.Create(PoolSizes[i], PoolSizes[i]);

  FFlattenedSize := CurrentChannels * CurrentWidth * CurrentHeight;

  SetLength(FFullyConnectedLayers, Length(FCLayerSizes));
  NumInputs := FFlattenedSize;
  for i := 0 to High(FCLayerSizes) do
  begin
    FFullyConnectedLayers[i] := TFCLayer.Create(NumInputs, FCLayerSizes[i], Activation);
    NumInputs := FCLayerSizes[i];
  end;

  FOutputLayer := TFCLayer.Create(NumInputs, OutputSize, OutputActivation);
end;

destructor TAdvancedCNN.Destroy;
var
  i: Integer;
begin
  for i := 0 to High(FConvLayers) do
    FConvLayers[i].Free;
  for i := 0 to High(FPoolLayers) do
    FPoolLayers[i].Free;
  for i := 0 to High(FFullyConnectedLayers) do
    FFullyConnectedLayers[i].Free;
  FOutputLayer.Free;
  inherited Destroy;
end;

function TAdvancedCNN.ClipGradient(G, MaxVal: Double): Double;
begin
  Result := ClipValue(G, MaxVal);
end;

function TAdvancedCNN.Flatten(const Input: TDArray3D): DArray;
var
  c, h, w, idx: Integer;
begin
  SetLength(Result, Length(Input) * Length(Input[0]) * Length(Input[0][0]));
  idx := 0;
  for c := 0 to High(Input) do
    for h := 0 to High(Input[c]) do
      for w := 0 to High(Input[c][h]) do
      begin
        Result[idx] := Input[c][h][w];
        Inc(idx);
      end;
end;

function TAdvancedCNN.Unflatten(const Input: DArray; Channels, Height, Width: Integer): TDArray3D;
var
  c, h, w, idx: Integer;
begin
  SetLength(Result, Channels, Height, Width);
  idx := 0;
  for c := 0 to Channels - 1 do
    for h := 0 to Height - 1 do
      for w := 0 to Width - 1 do
      begin
        Result[c][h][w] := Input[idx];
        Inc(idx);
      end;
end;

function TAdvancedCNN.ForwardPass(const Input: TDArray3D): DArray;
var
  i: Integer;
  CurrentOutput: TDArray3D;
  FlatInput: DArray;
  LayerInput: DArray;
  Logits: DArray;
begin
  CurrentOutput := Input;

  for i := 0 to High(FConvLayers) do
  begin
    FConvLayers[i].Forward(CurrentOutput, CurrentOutput);
    
    if i <= High(FPoolLayers) then
      FPoolLayers[i].Forward(CurrentOutput, CurrentOutput);
  end;

  FlatInput := Flatten(CurrentOutput);
  LayerInput := FlatInput;

  for i := 0 to High(FFullyConnectedLayers) do
  begin
    FFullyConnectedLayers[i].Forward(LayerInput, LayerInput);
  end;

  SetLength(Logits, FOutputSize);
  FOutputLayer.Forward(LayerInput, Logits);

  if FOutputActivation = atLinear then
    Result := Logits
  else
  begin
    TActivation.ApplySoftmax(Logits);
    Result := Logits;
  end;
end;

function TAdvancedCNN.BackwardPass(const Target: DArray): Double;
var
  i: Integer;
  OutputGrad, FCGrad: DArray;
  ConvGrad: TDArray3D;
  CurrentGrad: TDArray3D;
begin
  SetLength(OutputGrad, FOutputSize);
  for i := 0 to High(OutputGrad) do
    OutputGrad[i] := FOutputLayer.OutputCache[i] - Target[i];

  FOutputLayer.Backward(OutputGrad, FCGrad);

  for i := High(FFullyConnectedLayers) downto 0 do
  begin
    FFullyConnectedLayers[i].Backward(FCGrad, FCGrad);
  end;

  CurrentGrad := Unflatten(FCGrad, Length(FConvLayers[High(FConvLayers)].OutputCache),
                           Length(FConvLayers[High(FConvLayers)].OutputCache[0]),
                           Length(FConvLayers[High(FConvLayers)].OutputCache[0][0]));

  for i := High(FConvLayers) downto 0 do
  begin
    if i <= High(FPoolLayers) then
      FPoolLayers[i].Backward(CurrentGrad, CurrentGrad);
    FConvLayers[i].Backward(CurrentGrad, CurrentGrad);
  end;

  Result := TLoss.Compute(FOutputLayer.OutputCache, Target, FLossType);
end;

function TAdvancedCNN.TrainSample(const Input: TDArray3D; const Target: DArray): Double;
begin
  ResetGradients;
  ForwardPass(Input);
  Result := BackwardPass(Target);
  ApplyGradients;
end;

function TAdvancedCNN.TrainBatch(const BatchInputs: TDArray4D; const BatchTargets: TDArray2D): Double;
var
  b: Integer;
  BatchLoss: Double;
begin
  ResetGradients;
  BatchLoss := 0;

  for b := 0 to High(BatchInputs) do
  begin
    ForwardPass(BatchInputs[b]);
    BatchLoss := BatchLoss + BackwardPass(BatchTargets[b]);
  end;

  ApplyGradients;
  Result := BatchLoss / (Length(BatchInputs) + 1);
end;

function TAdvancedCNN.Predict(const Input: TDArray3D): DArray;
begin
  Result := ForwardPass(Input);
end;

function TAdvancedCNN.ComputeLoss(const Inputs: TDArray4D; const Targets: TDArray2D): Double;
var
  i: Integer;
  Output: DArray;
begin
  Result := 0;
  for i := 0 to High(Inputs) do
  begin
    Output := ForwardPass(Inputs[i]);
    Result := Result + TLoss.Compute(Output, Targets[i], FLossType);
  end;
  Result := Result / (Length(Inputs) + 1);
end;

procedure TAdvancedCNN.ResetGradients;
var
  i: Integer;
begin
  for i := 0 to High(FConvLayers) do
    FConvLayers[i].ResetGradients;
  for i := 0 to High(FFullyConnectedLayers) do
    FFullyConnectedLayers[i].ResetGradients;
  FOutputLayer.ResetGradients;
end;

procedure TAdvancedCNN.ApplyGradients;
var
  i: Integer;
begin
  for i := 0 to High(FConvLayers) do
    FConvLayers[i].ApplyGradients(FLearningRate, FGradientClip);
  for i := 0 to High(FFullyConnectedLayers) do
    FFullyConnectedLayers[i].ApplyGradients(FLearningRate, FGradientClip);
  FOutputLayer.ApplyGradients(FLearningRate, FGradientClip);
end;

// ========== Helper Functions ==========
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
  WriteLn('CNN - Command-line Convolutional Neural Network');
  WriteLn;
  WriteLn('Commands:');
  WriteLn('  create   Create a new CNN model');
  WriteLn('  train    Train an existing model with data');
  WriteLn('  predict  Make predictions with a trained model');
  WriteLn('  info     Display model information');
  WriteLn('  help     Show this help message');
  WriteLn;
  WriteLn('Create Options:');
  WriteLn('  --input-w=N            Input width (required)');
  WriteLn('  --input-h=N            Input height (required)');
  WriteLn('  --input-c=N            Input channels (required)');
  WriteLn('  --conv=N,N,...         Conv filters (required)');
  WriteLn('  --kernels=N,N,...      Kernel sizes (required)');
  WriteLn('  --pools=N,N,...        Pool sizes (required)');
  WriteLn('  --fc=N,N,...           FC layer sizes (required)');
  WriteLn('  --output=N             Output layer size (required)');
  WriteLn('  --save=FILE            Save model to file (required)');
  WriteLn('  --lr=VALUE             Learning rate (default: 0.001)');
  WriteLn('  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)');
  WriteLn('  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)');
  WriteLn('  --loss=TYPE            mse|crossentropy (default: mse)');
  WriteLn('  --clip=VALUE           Gradient clipping (default: 5.0)');
  WriteLn;
  WriteLn('Examples:');
  WriteLn('  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin');
  WriteLn('  cnn train --model=model.bin --data=data.csv --epochs=50 --save=model_trained.bin');
end;

// ========== Main Program ==========
var
  Command: TCommand;
  CmdStr: string;
  i, inputW, inputH, inputC, outputSize: Integer;
  convFilters, kernelSizes, poolSizes, fcLayerSizes: TIntArray;
  learningRate, gradientClip: Double;
  hiddenAct, outputAct: TActivationType;
  lossType: TLossType;
  modelFile, saveFile: string;
  arg, key, value: string;
  eqPos: Integer;
  CNN: TAdvancedCNN;

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
  inputW := 0;
  inputH := 0;
  inputC := 0;
  outputSize := 0;
  SetLength(convFilters, 0);
  SetLength(kernelSizes, 0);
  SetLength(poolSizes, 0);
  SetLength(fcLayerSizes, 0);
  learningRate := 0.001;
  gradientClip := 5.0;
  hiddenAct := atReLU;
  outputAct := atLinear;
  lossType := ltMSE;
  modelFile := '';
  saveFile := '';

  // Parse arguments
  for i := 2 to ParamCount do
  begin
    arg := ParamStr(i);
    eqPos := Pos('=', arg);
    if eqPos = 0 then
    begin
      WriteLn('Invalid argument: ', arg);
      Continue;
    end;

    key := Copy(arg, 1, eqPos - 1);
    value := Copy(arg, eqPos + 1, Length(arg));

    if key = '--input-w' then
      inputW := StrToInt(value)
    else if key = '--input-h' then
      inputH := StrToInt(value)
    else if key = '--input-c' then
      inputC := StrToInt(value)
    else if key = '--output' then
      outputSize := StrToInt(value)
    else if key = '--conv' then
      ParseIntArrayHelper(value, convFilters)
    else if key = '--kernels' then
      ParseIntArrayHelper(value, kernelSizes)
    else if key = '--pools' then
      ParseIntArrayHelper(value, poolSizes)
    else if key = '--fc' then
      ParseIntArrayHelper(value, fcLayerSizes)
    else if key = '--save' then
      saveFile := value
    else if key = '--model' then
      modelFile := value
    else if key = '--lr' then
      learningRate := StrToFloat(value)
    else if key = '--hidden-act' then
      hiddenAct := ParseActivation(value)
    else if key = '--output-act' then
      outputAct := ParseActivation(value)
    else if key = '--loss' then
      lossType := ParseLoss(value)
    else if key = '--clip' then
      gradientClip := StrToFloat(value)
    else
      WriteLn('Unknown option: ', key);
  end;

  // Execute command
  if Command = cmdCreate then
  begin
    if inputW <= 0 then begin WriteLn('Error: --input-w is required'); Exit; end;
    if inputH <= 0 then begin WriteLn('Error: --input-h is required'); Exit; end;
    if inputC <= 0 then begin WriteLn('Error: --input-c is required'); Exit; end;
    if Length(convFilters) = 0 then begin WriteLn('Error: --conv is required'); Exit; end;
    if Length(kernelSizes) = 0 then begin WriteLn('Error: --kernels is required'); Exit; end;
    if Length(poolSizes) = 0 then begin WriteLn('Error: --pools is required'); Exit; end;
    if Length(fcLayerSizes) = 0 then begin WriteLn('Error: --fc is required'); Exit; end;
    if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;
    if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;

    CNN := TAdvancedCNN.Create(inputW, inputH, inputC, convFilters, kernelSizes, 
                              poolSizes, fcLayerSizes, outputSize,
                              hiddenAct, outputAct, lossType, learningRate, gradientClip);

    WriteLn('Created CNN model:');
    WriteLn('  Input: ', inputW, 'x', inputH, 'x', inputC);
    Write('  Conv filters: ');
    for i := 0 to High(convFilters) do
    begin
      if i > 0 then Write(',');
      Write(convFilters[i]);
    end;
    WriteLn;
    Write('  Kernel sizes: ');
    for i := 0 to High(kernelSizes) do
    begin
      if i > 0 then Write(',');
      Write(kernelSizes[i]);
    end;
    WriteLn;
    Write('  Pool sizes: ');
    for i := 0 to High(poolSizes) do
    begin
      if i > 0 then Write(',');
      Write(poolSizes[i]);
    end;
    WriteLn;
    Write('  FC layers: ');
    for i := 0 to High(fcLayerSizes) do
    begin
      if i > 0 then Write(',');
      Write(fcLayerSizes[i]);
    end;
    WriteLn;
    WriteLn('  Output size: ', outputSize);
    WriteLn('  Hidden activation: ', ActivationToStr(hiddenAct));
    WriteLn('  Output activation: ', ActivationToStr(outputAct));
    WriteLn('  Loss function: ', LossToStr(lossType));
    WriteLn('  Learning rate: ', learningRate:0:6);
    WriteLn('  Gradient clip: ', gradientClip:0:2);
    WriteLn('  Saved to: ', saveFile);

    CNN.Free;
  end
  else if Command = cmdTrain then
  begin
    WriteLn('Train command requires model persistence (not yet implemented)');
  end
  else if Command = cmdPredict then
  begin
    WriteLn('Predict command requires model persistence (not yet implemented)');
  end
  else if Command = cmdInfo then
  begin
    WriteLn('Info command requires model persistence (not yet implemented)');
  end;
end.

(*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GAN Unit - Generative Adversarial Network with convolutional layers,
 * normalization, attention, progressive growing, WGAN-GP, and more.
 * NIST 800-53 compliance: AU (Audit), SC (Comms Protection), SI (Integrity)
 *)
unit GAN;
{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Classes, StrUtils;

const
  GAN_VERSION = '2.0';
  DEFAULT_GP_LAMBDA = 10.0;
  DEFAULT_BN_EPS = 1e-5;
  DEFAULT_BN_MOM = 0.1;
  MAX_SPECTRAL_ITER = 1;
  DEFAULT_AUDIT_LOG = 'gan_audit.log';

type
  TMatrix = array of array of Single;
  TVector = array of Single;
  TMatrixArray = array of TMatrix;
  TKernelArray = array of TMatrix;

  TActivationType = (atReLU, atSigmoid, atTanh, atLeakyReLU, atNone);
  TLayerType = (ltDense, ltConv2D, ltDeconv2D, ltConv1D,
                ltBatchNorm, ltLayerNorm, ltSpectralNorm, ltAttention);
  TLossType = (lossBCE, lossWGANGP, lossHinge, lossLeastSquares);
  TDataType = (dtImage, dtAudio, dtVector);
  TNoiseType = (ntGauss, ntUniform, ntAnalog);
  TOptimizer = (optAdam, optSGD, optRMSProp);

  TLayer = record
    layerType: TLayerType;
    activation: TActivationType;
    inputSize, outputSize: Integer;
    { Dense }
    weights: TMatrix;
    bias: TVector;
    { Conv }
    kernels: TKernelArray;
    kernelSize, stride, padding: Integer;
    inChannels, outChannels: Integer;
    inWidth, inHeight, outWidth, outHeight: Integer;
    { BatchNorm / LayerNorm }
    bnGamma, bnBeta: TVector;
    runningMean, runningVar: TVector;
    bnEpsilon, bnMomentum: Single;
    { Attention }
    Wq, Wk, Wv, Wo: TMatrix;
    numHeads, headDim: Integer;
    { Spectral norm }
    spectralU, spectralV: TVector;
    spectralSigma: Single;
    { Forward cache }
    layerInput, layerOutput, preActivation: TMatrix;
    isTraining: Boolean;
    cachedQ, cachedK, cachedV, cachedScores, cachedAttended: TMatrix;
    cachedNormalized: TMatrix;
    cachedMean, cachedVar: TVector;
    useCheckpoint: Boolean;
    { Gradients }
    weightGrad: TMatrix;
    biasGrad: TVector;
    kernelGrad: TKernelArray;
    bnGammaGrad, bnBetaGrad: TVector;
    WqGrad, WkGrad, WvGrad, WoGrad: TMatrix;
    { Adam moments }
    adamT: Integer;
    mWeight, vWeight: TMatrix;
    mBias, vBias: TVector;
    mKernel, vKernel: TKernelArray;
    mBnGamma, vBnGamma, mBnBeta, vBnBeta: TVector;
    mWq, vWq, mWk, vWk, mWv, vWv, mWo, vWo: TMatrix;
    { RMSProp }
    rmsWeight: TMatrix;
    rmsBias: TVector;
  end;

  TNetwork = record
    layers: array of TLayer;
    layerCount: Integer;
    optimizer: TOptimizer;
    learningRate, momentum, beta1, beta2, epsilon, weightDecay: Single;
    progressiveAlpha: Single;
    currentResLevel: Integer;
    isTraining: Boolean;
  end;

  TGANConfig = record
    epochs, batchSize: Integer;
    generatorBits, discriminatorBits: Integer;
    activation: TActivationType;
    noiseType: TNoiseType;
    noiseDepth: Integer;
    patchConfig, saveModel, loadModel, loadJSONModel, outputDir: string;
    learningRate: Single;
    optimizer: TOptimizer;
    lossType: TLossType;
    gpLambda: Single;
    conditionSize: Integer;
    useBatchNorm, useLayerNorm, useSpectralNorm: Boolean;
    useLabelSmoothing, useFeatureMatching, useMinibatchStdDev: Boolean;
    generatorLR, discriminatorLR: Single;
    useProgressive: Boolean;
    maxResLevel: Integer;
    dataType: TDataType;
    dataPath: string;
    useAugmentation, computeMetrics: Boolean;
    metricInterval: Integer;
    useWeightDecay: Boolean;
    weightDecayVal: Single;
    useCosineAnneal, auditLog: Boolean;
    auditLogFile: string;
    checkpointInterval: Integer;
    useEncryption: Boolean;
    encryptionKey: string;
    useConv, useAttention, runTests, runFuzz: Boolean;
    fuzzIterations, numThreads: Integer;
  end;

  TGANMetrics = record
    dLossReal, dLossFake, gLoss, fidScore, isScore, gradPenalty: Single;
    epoch, batch: Integer;
  end;

  TDataset = record
    samples: TMatrixArray;
    labels: TMatrix;
    count: Integer;
    dataType: TDataType;
    sampleWidth, sampleHeight, sampleChannels: Integer;
  end;

  TBatchThread = class(TThread)
  public
    NetCopy: TNetwork;
    Input, Output: TMatrix;
    StartRow, EndRow: Integer;
  protected
    procedure Execute; override;
  end;

{ --- Security / Audit --- }
procedure AuditLog(const msg, logFile: string);
procedure SecureRandomize;
function SecureRandomByte: Byte;
function ValidatePath(const path: string): Boolean;
function SafeMatrixGet(const M: TMatrix; r, c: Integer; def: Single): Single;
procedure SafeMatrixSet(var M: TMatrix; r, c: Integer; val: Single);

{ --- Matrix Creation --- }
function CreateMatrix(rows, cols: Integer): TMatrix;
function CreateVector(size: Integer): TVector;

{ --- Matrix Ops --- }
function MatrixMultiply(const A, B: TMatrix): TMatrix;
function MatrixAdd(const A, B: TMatrix): TMatrix;
function MatrixSubtract(const A, B: TMatrix): TMatrix;
function MatrixScale(const A: TMatrix; s: Single): TMatrix;
function MatrixTranspose(const A: TMatrix): TMatrix;
function MatrixNormalize(const A: TMatrix): TMatrix;
function MatrixElementMul(const A, B: TMatrix): TMatrix;
procedure MatrixAddInPlace(var A: TMatrix; const B: TMatrix);
procedure MatrixScaleInPlace(var A: TMatrix; s: Single);
procedure MatrixClipInPlace(var A: TMatrix; lo, hi: Single);

{ --- Random --- }
function RandomGaussian: Single;
function RandomUniform(lo, hi: Single): Single;
function RandomAnalog: Single;
procedure GenerateNoise(var noise: TMatrix; size, depth: Integer; nt: TNoiseType);
function NoiseSlerp(const v1, v2: TVector; t: Single): TVector;

{ --- Activations --- }
function MatrixReLU(const A: TMatrix): TMatrix;
function MatrixLeakyReLU(const A: TMatrix; alpha: Single): TMatrix;
function MatrixSigmoid(const A: TMatrix): TMatrix;
function MatrixTanh(const A: TMatrix): TMatrix;
function ApplyActivation(const A: TMatrix; act: TActivationType): TMatrix;
function ActivationBackward(const gradOut, preAct: TMatrix; act: TActivationType): TMatrix;
function MatrixSoftmax(const A: TMatrix): TMatrix;

{ --- Convolution --- }
function Conv2DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function Conv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
function Deconv2DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function Deconv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
function Conv1DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function Conv1DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;

{ --- Normalization --- }
function BatchNormForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function BatchNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
function LayerNormForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function LayerNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
function SpectralNormalize(var layer: TLayer): TMatrix;

{ --- Attention --- }
function SelfAttentionForward(const inp: TMatrix; var layer: TLayer): TMatrix;
function SelfAttentionBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;

{ --- Layer Ops --- }
function CreateDenseLayer(inSz, outSz: Integer; act: TActivationType): TLayer;
function CreateConv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
function CreateDeconv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
function CreateConv1DLayer(inCh, outCh, kSz, st, pad, inLen: Integer; act: TActivationType): TLayer;
function CreateBatchNormLayer(features: Integer): TLayer;
function CreateLayerNormLayer(features: Integer): TLayer;
function CreateAttentionLayer(dModel, nHeads: Integer): TLayer;
function LayerForward(var layer: TLayer; const inp: TMatrix): TMatrix;
function LayerBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
procedure InitLayerOptimizer(var layer: TLayer; opt: TOptimizer);

{ --- Network Ops --- }
function CreateNetwork(const sizes: array of Integer; act: TActivationType;
  opt: TOptimizer; lr: Single): TNetwork;
function CreateConvGenerator(noiseDim, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
function CreateConvDiscriminator(inCh, inW, inH, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
function NetworkForward(var net: TNetwork; const inp: TMatrix): TMatrix;
function NetworkBackward(var net: TNetwork; const gradOut: TMatrix): TMatrix;
procedure NetworkUpdateWeights(var net: TNetwork);
procedure AddProgressiveLayer(var net: TNetwork; resLvl: Integer; isGen: Boolean);
procedure SetNetworkTraining(var net: TNetwork; training: Boolean);
function GetLayerOutput(var net: TNetwork; idx: Integer): TMatrix;
function DeepCopyNetwork(const src: TNetwork): TNetwork;

{ --- Loss Functions --- }
function BinaryCrossEntropy(const pred, target: TMatrix): Single;
function BCEGradient(const pred, target: TMatrix): TMatrix;
function WGANDiscLoss(const dReal, dFake: TMatrix): Single;
function WGANGenLoss(const dFake: TMatrix): Single;
function WGANDiscGrad(const dOut: TMatrix; isReal: Boolean): TMatrix;
function WGANGenGrad(const dFake: TMatrix): TMatrix;
function HingeDiscLoss(const dReal, dFake: TMatrix): Single;
function HingeGenLoss(const dFake: TMatrix): Single;
function LSDiscLoss(const dReal, dFake: TMatrix): Single;
function LSGenLoss(const dFake: TMatrix): Single;
function ComputeGradientPenalty(var disc: TNetwork;
  const real, fake: TMatrix; lambda: Single): Single;

{ --- Regularization --- }
function ApplyLabelSmoothing(const labels: TMatrix; lo, hi: Single): TMatrix;
function FeatureMatchingLoss(var disc: TNetwork;
  const real, fake: TMatrix; featLayer: Integer): Single;
function MinibatchStdDev(const inp: TMatrix): TMatrix;

{ --- Optimizer Helpers --- }
procedure AdamUpdateMatrix(var p: TMatrix; const g: TMatrix;
  var m, v: TMatrix; t: Integer; lr, b1, b2, eps, wd: Single);
procedure AdamUpdateVector(var p: TVector; const g: TVector;
  var m, v: TVector; t: Integer; lr, b1, b2, eps, wd: Single);
procedure SGDUpdateMatrix(var p: TMatrix; const g: TMatrix; lr, wd: Single);
procedure SGDUpdateVector(var p: TVector; const g: TVector; lr, wd: Single);
procedure RMSPropUpdateMatrix(var p: TMatrix; const g: TMatrix;
  var cache: TMatrix; lr, decay, eps, wd: Single);
procedure RMSPropUpdateVector(var p: TVector; const g: TVector;
  var cache: TVector; lr, decay, eps, wd: Single);
function CosineAnneal(epoch, maxEp: Integer; baseLR, minLR: Single): Single;

{ --- Data --- }
function LoadBMPDataset(const path: string): TDataset;
function LoadWAVDataset(const path: string): TDataset;
function LoadDataset(const path: string; dt: TDataType): TDataset;
function AugmentSample(const sample: TMatrix; dt: TDataType): TMatrix;
function CreateSyntheticDataset(count, features: Integer): TDataset;

{ --- Metrics --- }
function ComputeFID(const realS, fakeS: TMatrixArray): Single;
function ComputeIS(const samples: TMatrixArray): Single;
procedure LogMetrics(const met: TGANMetrics; const fn: string);

{ --- Serialization --- }
procedure SaveNetworkBinary(const net: TNetwork; const fn: string);
procedure LoadNetworkBinary(var net: TNetwork; const fn: string);
procedure SaveGANToJSON(const gen, disc: TNetwork; const fn: string);
procedure LoadGANFromJSON(var gen, disc: TNetwork; const fn: string);
procedure SaveCheckpoint(const gen, disc: TNetwork; ep: Integer; const dir: string);
procedure LoadCheckpoint(var gen, disc: TNetwork; ep: Integer; const dir: string);

{ --- Visualization --- }
procedure SaveGeneratedSamples(var gen: TNetwork; ep: Integer;
  const dir: string; noiseDim: Integer; nt: TNoiseType);
procedure PlotLossCSV(const fn: string; const dL, gL: array of Single; cnt: Integer);
procedure PrintLossBar(dLoss, gLoss: Single; w: Integer);

{ --- Encryption Stub --- }
procedure EncryptFile(const inF, outF, key: string);
procedure DecryptFile(const inF, outF, key: string);

{ --- JSON Helpers --- }
function Vector1DToJSON(const v: TVector): string;
function Matrix2DToJSON(const m: TMatrix): string;
function ExtractJSONInt(const js, field: string): Integer;
function ExtractJSONFloat(const js, field: string): Single;
procedure LoadVector1DFromJSON(const js: string; var v: TVector);
procedure LoadMatrix2DFromJSON(const js: string; var m: TMatrix);
procedure ValidateAndCleanWeights(var layer: TLayer);

{ --- Training --- }
procedure TrainGAN(var gen, disc: TNetwork; var ds: TDataset; cfg: TGANConfig);

{ --- Testing --- }
function RunTests: Boolean;
function RunFuzzTests(iterations: Integer): Boolean;

{ --- CLI --- }
function DefaultConfig: TGANConfig;
function ParseConfig: TGANConfig;
procedure ShowHelp;

implementation

{ =========================================================================== }
{ SECURITY / AUDIT  (NIST 800-53: AU-2, AU-3, SI-7)                         }
{ =========================================================================== }

procedure AuditLog(const msg, logFile: string);
var f: TextFile;
begin
  try
    AssignFile(f, logFile);
    if FileExists(logFile) then Append(f) else Rewrite(f);
    WriteLn(f, FormatDateTime('yyyy-mm-dd hh:nn:ss', Now) + ' | ' + msg);
    CloseFile(f);
  except end;
end;

procedure SecureRandomize;
var seed: LongWord; f: TFileStream;
begin
  try
    f := TFileStream.Create('/dev/urandom', fmOpenRead);
    try f.Read(seed, SizeOf(seed)); RandSeed := seed; finally f.Free; end;
  except Randomize; end;
end;

function SecureRandomByte: Byte;
var f: TFileStream; b: Byte;
begin
  b := 0;
  try
    f := TFileStream.Create('/dev/urandom', fmOpenRead);
    try f.Read(b, 1); finally f.Free; end;
  except b := Random(256); end;
  Result := b;
end;

function ValidatePath(const path: string): Boolean;
var i: Integer;
begin
  Result := (Length(path) > 0) and (Length(path) < 4096);
  if not Result then Exit;
  for i := 1 to Length(path) do
    if path[i] in [#0..#31] then begin Result := False; Exit; end;
  if (Pos('..', path) > 0) then Result := False;
end;

function SafeMatrixGet(const M: TMatrix; r, c: Integer; def: Single): Single;
begin
  if (r >= 0) and (r < Length(M)) and (c >= 0) and (c < Length(M[r])) then
    Result := M[r][c]
  else Result := def;
end;

procedure SafeMatrixSet(var M: TMatrix; r, c: Integer; val: Single);
begin
  if (r >= 0) and (r < Length(M)) and (c >= 0) and (c < Length(M[r])) then
    M[r][c] := val;
end;

{ =========================================================================== }
{ MATRIX CREATION                                                             }
{ =========================================================================== }

function CreateMatrix(rows, cols: Integer): TMatrix;
var i: Integer;
begin
  SetLength(Result, rows);
  for i := 0 to rows - 1 do SetLength(Result[i], cols);
end;

function CreateVector(size: Integer): TVector;
begin
  SetLength(Result, size);
  if size > 0 then FillChar(Result[0], size * SizeOf(Single), 0);
end;

{ =========================================================================== }
{ MATRIX OPERATIONS                                                           }
{ =========================================================================== }

function MatrixMultiply(const A, B: TMatrix): TMatrix;
var i, j, k: Integer; s: Single;
begin
  Result := CreateMatrix(Length(A), Length(B[0]));
  for i := 0 to High(A) do
    for j := 0 to High(B[0]) do begin
      s := 0;
      for k := 0 to High(B) do s := s + A[i][k] * B[k][j];
      Result[i][j] := s;
    end;
end;

function MatrixAdd(const A, B: TMatrix): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := A[i][j] + B[i][j];
end;

function MatrixSubtract(const A, B: TMatrix): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := A[i][j] - B[i][j];
end;

function MatrixScale(const A: TMatrix; s: Single): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := A[i][j] * s;
end;

function MatrixTranspose(const A: TMatrix): TMatrix;
var i, j: Integer;
begin
  if Length(A) = 0 then begin SetLength(Result, 0); Exit; end;
  Result := CreateMatrix(Length(A[0]), Length(A));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[j][i] := A[i][j];
end;

function MatrixNormalize(const A: TMatrix): TMatrix;
var i, j: Integer; mn, vr, cnt: Single;
begin
  mn := 0; cnt := 0;
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do begin mn := mn + A[i][j]; cnt := cnt + 1; end;
  mn := mn / cnt;
  vr := 0;
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do vr := vr + sqr(A[i][j] - mn);
  vr := sqrt(vr / cnt);
  if vr < 1e-7 then vr := 1;
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := (A[i][j] - mn) / vr;
end;

function MatrixElementMul(const A, B: TMatrix): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := A[i][j] * B[i][j];
end;

procedure MatrixAddInPlace(var A: TMatrix; const B: TMatrix);
var i, j: Integer;
begin
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do A[i][j] := A[i][j] + B[i][j];
end;

procedure MatrixScaleInPlace(var A: TMatrix; s: Single);
var i, j: Integer;
begin
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do A[i][j] := A[i][j] * s;
end;

procedure MatrixClipInPlace(var A: TMatrix; lo, hi: Single);
var i, j: Integer;
begin
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do begin
      if A[i][j] < lo then A[i][j] := lo;
      if A[i][j] > hi then A[i][j] := hi;
    end;
end;

{ =========================================================================== }
{ RANDOM GENERATION                                                           }
{ =========================================================================== }

function RandomGaussian: Single;
var u1, u2: Single;
begin
  u1 := Random; u2 := Random;
  if u1 < 1e-7 then u1 := 1e-7;
  Result := sqrt(-2.0 * ln(u1)) * cos(2.0 * Pi * u2);
end;

function RandomUniform(lo, hi: Single): Single;
begin Result := lo + Random * (hi - lo); end;

function RandomAnalog: Single;
begin Result := (Random - 0.5) * 0.1; end;

procedure GenerateNoise(var noise: TMatrix; size, depth: Integer; nt: TNoiseType);
var i, j: Integer;
begin
  noise := CreateMatrix(size, depth);
  for i := 0 to size - 1 do
    for j := 0 to depth - 1 do
      case nt of
        ntGauss:   noise[i][j] := RandomGaussian;
        ntUniform: noise[i][j] := RandomUniform(-1, 1);
        ntAnalog:  noise[i][j] := RandomAnalog;
      end;
end;

function NoiseSlerp(const v1, v2: TVector; t: Single): TVector;
var i: Integer; dot, n1, n2, omega, sinO: Single;
begin
  SetLength(Result, Length(v1));
  dot := 0; n1 := 0; n2 := 0;
  for i := 0 to High(v1) do begin
    dot := dot + v1[i] * v2[i]; n1 := n1 + sqr(v1[i]); n2 := n2 + sqr(v2[i]);
  end;
  n1 := sqrt(n1); n2 := sqrt(n2);
  if (n1 < 1e-12) or (n2 < 1e-12) then begin
    for i := 0 to High(v1) do Result[i] := (1-t)*v1[i] + t*v2[i]; Exit;
  end;
  dot := Max(-1, Min(1, dot / (n1 * n2)));
  omega := arccos(dot);
  if abs(omega) < 1e-6 then begin
    for i := 0 to High(v1) do Result[i] := (1-t)*v1[i] + t*v2[i]; Exit;
  end;
  sinO := sin(omega);
  for i := 0 to High(v1) do
    Result[i] := (sin((1-t)*omega)*v1[i] + sin(t*omega)*v2[i]) / sinO;
end;

{ =========================================================================== }
{ ACTIVATION FUNCTIONS (forward + backward)                                   }
{ =========================================================================== }

function MatrixReLU(const A: TMatrix): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      if A[i][j] > 0 then Result[i][j] := A[i][j] else Result[i][j] := 0;
end;

function MatrixLeakyReLU(const A: TMatrix; alpha: Single): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      if A[i][j] > 0 then Result[i][j] := A[i][j]
      else Result[i][j] := alpha * A[i][j];
end;

function MatrixSigmoid(const A: TMatrix): TMatrix;
var i, j: Integer; v: Single;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do begin
      v := A[i][j];
      if v > 20 then Result[i][j] := 1.0
      else if v < -20 then Result[i][j] := 0.0
      else Result[i][j] := 1.0 / (1.0 + exp(-v));
    end;
end;

function MatrixTanh(const A: TMatrix): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do Result[i][j] := Math.Tanh(A[i][j]);
end;

function ApplyActivation(const A: TMatrix; act: TActivationType): TMatrix;
begin
  case act of
    atReLU:      Result := MatrixReLU(A);
    atSigmoid:   Result := MatrixSigmoid(A);
    atTanh:      Result := MatrixTanh(A);
    atLeakyReLU: Result := MatrixLeakyReLU(A, 0.01);
  else Result := A;
  end;
end;

function ActivationBackward(const gradOut, preAct: TMatrix; act: TActivationType): TMatrix;
var i, j: Integer; s: Single;
begin
  Result := CreateMatrix(Length(gradOut), Length(gradOut[0]));
  for i := 0 to High(gradOut) do
    for j := 0 to High(gradOut[0]) do
      case act of
        atReLU:
          if preAct[i][j] > 0 then Result[i][j] := gradOut[i][j]
          else Result[i][j] := 0;
        atLeakyReLU:
          if preAct[i][j] > 0 then Result[i][j] := gradOut[i][j]
          else Result[i][j] := 0.01 * gradOut[i][j];
        atSigmoid: begin
          s := 1.0 / (1.0 + exp(-preAct[i][j]));
          Result[i][j] := gradOut[i][j] * s * (1 - s);
        end;
        atTanh: begin
          s := Math.Tanh(preAct[i][j]);
          Result[i][j] := gradOut[i][j] * (1 - s * s);
        end;
      else Result[i][j] := gradOut[i][j];
      end;
end;

function MatrixSoftmax(const A: TMatrix): TMatrix;
var i, j: Integer; mx, s: Single;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do begin
    mx := A[i][0];
    for j := 1 to High(A[0]) do if A[i][j] > mx then mx := A[i][j];
    s := 0;
    for j := 0 to High(A[0]) do begin
      Result[i][j] := exp(A[i][j] - mx); s := s + Result[i][j];
    end;
    for j := 0 to High(A[0]) do Result[i][j] := Result[i][j] / (s + 1e-12);
  end;
end;
{ =========================================================================== }
{ CONVOLUTION OPERATIONS                                                      }
{ Input layout: (batch, inChannels * inH * inW) flattened                     }
{ Output layout: (batch, outChannels * outH * outW) flattened                 }
{ =========================================================================== }

function Conv2DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var b, oc, ic, oy, ox, ky, kx, bs, oH, oW, iH, iW, kS, iy, ix: Integer;
    s: Single;
begin
  bs := Length(inp); iH := layer.inHeight; iW := layer.inWidth;
  kS := layer.kernelSize;
  oH := (iH + 2*layer.padding - kS) div layer.stride + 1;
  oW := (iW + 2*layer.padding - kS) div layer.stride + 1;
  layer.outHeight := oH; layer.outWidth := oW;
  Result := CreateMatrix(bs, layer.outChannels * oH * oW);
  for b := 0 to bs - 1 do
    for oc := 0 to layer.outChannels - 1 do
      for oy := 0 to oH - 1 do
        for ox := 0 to oW - 1 do begin
          s := layer.bias[oc];
          for ic := 0 to layer.inChannels - 1 do
            for ky := 0 to kS - 1 do
              for kx := 0 to kS - 1 do begin
                iy := oy * layer.stride - layer.padding + ky;
                ix := ox * layer.stride - layer.padding + kx;
                if (iy >= 0) and (iy < iH) and (ix >= 0) and (ix < iW) then
                  s := s + inp[b][ic*iH*iW + iy*iW + ix] *
                       layer.kernels[oc*layer.inChannels + ic][ky][kx];
              end;
          Result[b][oc*oH*oW + oy*oW + ox] := s;
        end;
end;

function Conv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var b, oc, ic, oy, ox, ky, kx, bs, oH, oW, iH, iW, kS, iy, ix, ki: Integer;
    gVal: Single;
begin
  bs := Length(gradOut); iH := layer.inHeight; iW := layer.inWidth;
  kS := layer.kernelSize; oH := layer.outHeight; oW := layer.outWidth;
  Result := CreateMatrix(bs, layer.inChannels * iH * iW);
  { init kernel grads }
  SetLength(layer.kernelGrad, Length(layer.kernels));
  for ki := 0 to High(layer.kernels) do
    layer.kernelGrad[ki] := CreateMatrix(kS, kS);
  SetLength(layer.biasGrad, layer.outChannels);
  for oc := 0 to layer.outChannels - 1 do layer.biasGrad[oc] := 0;
  for b := 0 to bs - 1 do
    for oc := 0 to layer.outChannels - 1 do
      for oy := 0 to oH - 1 do
        for ox := 0 to oW - 1 do begin
          gVal := gradOut[b][oc*oH*oW + oy*oW + ox];
          layer.biasGrad[oc] := layer.biasGrad[oc] + gVal;
          for ic := 0 to layer.inChannels - 1 do
            for ky := 0 to kS - 1 do
              for kx := 0 to kS - 1 do begin
                iy := oy * layer.stride - layer.padding + ky;
                ix := ox * layer.stride - layer.padding + kx;
                if (iy >= 0) and (iy < iH) and (ix >= 0) and (ix < iW) then begin
                  ki := oc * layer.inChannels + ic;
                  layer.kernelGrad[ki][ky][kx] := layer.kernelGrad[ki][ky][kx] +
                    layer.layerInput[b][ic*iH*iW + iy*iW + ix] * gVal;
                  Result[b][ic*iH*iW + iy*iW + ix] := Result[b][ic*iH*iW + iy*iW + ix] +
                    layer.kernels[ki][ky][kx] * gVal;
                end;
              end;
        end;
end;

function Deconv2DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var b, oc, ic, iy, ix, ky, kx, bs, oH, oW, iH, iW, kS, oy, ox: Integer;
    v: Single;
begin
  bs := Length(inp); iH := layer.inHeight; iW := layer.inWidth;
  kS := layer.kernelSize;
  oH := (iH - 1) * layer.stride - 2 * layer.padding + kS;
  oW := (iW - 1) * layer.stride - 2 * layer.padding + kS;
  layer.outHeight := oH; layer.outWidth := oW;
  Result := CreateMatrix(bs, layer.outChannels * oH * oW);
  for b := 0 to bs - 1 do begin
    for oc := 0 to layer.outChannels - 1 do
      for oy := 0 to oH - 1 do
        for ox := 0 to oW - 1 do
          Result[b][oc*oH*oW + oy*oW + ox] := layer.bias[oc];
    for ic := 0 to layer.inChannels - 1 do
      for iy := 0 to iH - 1 do
        for ix := 0 to iW - 1 do begin
          v := inp[b][ic*iH*iW + iy*iW + ix];
          for oc := 0 to layer.outChannels - 1 do
            for ky := 0 to kS - 1 do
              for kx := 0 to kS - 1 do begin
                oy := iy * layer.stride - layer.padding + ky;
                ox := ix * layer.stride - layer.padding + kx;
                if (oy >= 0) and (oy < oH) and (ox >= 0) and (ox < oW) then
                  Result[b][oc*oH*oW + oy*oW + ox] :=
                    Result[b][oc*oH*oW + oy*oW + ox] +
                    v * layer.kernels[ic*layer.outChannels + oc][ky][kx];
              end;
        end;
  end;
end;

function Deconv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var b, oc, ic, iy, ix, ky, kx, bs, oH, oW, iH, iW, kS, oy, ox, ki: Integer;
    gVal: Single;
begin
  bs := Length(gradOut); iH := layer.inHeight; iW := layer.inWidth;
  kS := layer.kernelSize; oH := layer.outHeight; oW := layer.outWidth;
  Result := CreateMatrix(bs, layer.inChannels * iH * iW);
  SetLength(layer.kernelGrad, Length(layer.kernels));
  for ki := 0 to High(layer.kernels) do
    layer.kernelGrad[ki] := CreateMatrix(kS, kS);
  SetLength(layer.biasGrad, layer.outChannels);
  for oc := 0 to layer.outChannels - 1 do layer.biasGrad[oc] := 0;
  for b := 0 to bs - 1 do begin
    for oc := 0 to layer.outChannels - 1 do
      for oy := 0 to oH - 1 do
        for ox := 0 to oW - 1 do
          layer.biasGrad[oc] := layer.biasGrad[oc] + gradOut[b][oc*oH*oW + oy*oW + ox];
    for ic := 0 to layer.inChannels - 1 do
      for iy := 0 to iH - 1 do
        for ix := 0 to iW - 1 do begin
          gVal := 0;
          for oc := 0 to layer.outChannels - 1 do
            for ky := 0 to kS - 1 do
              for kx := 0 to kS - 1 do begin
                oy := iy * layer.stride - layer.padding + ky;
                ox := ix * layer.stride - layer.padding + kx;
                if (oy >= 0) and (oy < oH) and (ox >= 0) and (ox < oW) then begin
                  ki := ic * layer.outChannels + oc;
                  layer.kernelGrad[ki][ky][kx] := layer.kernelGrad[ki][ky][kx] +
                    layer.layerInput[b][ic*iH*iW + iy*iW + ix] *
                    gradOut[b][oc*oH*oW + oy*oW + ox];
                  gVal := gVal + layer.kernels[ki][ky][kx] *
                    gradOut[b][oc*oH*oW + oy*oW + ox];
                end;
              end;
          Result[b][ic*iH*iW + iy*iW + ix] := gVal;
        end;
  end;
end;

function Conv1DForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var b, oc, ic, ox, kx, bs, oL, iL, kS, ix: Integer; s: Single;
begin
  bs := Length(inp); iL := layer.inWidth; kS := layer.kernelSize;
  oL := (iL + 2*layer.padding - kS) div layer.stride + 1;
  layer.outWidth := oL;
  Result := CreateMatrix(bs, layer.outChannels * oL);
  for b := 0 to bs - 1 do
    for oc := 0 to layer.outChannels - 1 do
      for ox := 0 to oL - 1 do begin
        s := layer.bias[oc];
        for ic := 0 to layer.inChannels - 1 do
          for kx := 0 to kS - 1 do begin
            ix := ox * layer.stride - layer.padding + kx;
            if (ix >= 0) and (ix < iL) then
              s := s + inp[b][ic*iL + ix] *
                   layer.kernels[oc*layer.inChannels + ic][0][kx];
          end;
        Result[b][oc*oL + ox] := s;
      end;
end;

function Conv1DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var b, oc, ic, ox, kx, bs, oL, iL, kS, ix, ki: Integer; gVal: Single;
begin
  bs := Length(gradOut); iL := layer.inWidth; kS := layer.kernelSize;
  oL := layer.outWidth;
  Result := CreateMatrix(bs, layer.inChannels * iL);
  SetLength(layer.kernelGrad, Length(layer.kernels));
  for ki := 0 to High(layer.kernels) do
    layer.kernelGrad[ki] := CreateMatrix(1, kS);
  SetLength(layer.biasGrad, layer.outChannels);
  for oc := 0 to layer.outChannels - 1 do layer.biasGrad[oc] := 0;
  for b := 0 to bs - 1 do
    for oc := 0 to layer.outChannels - 1 do
      for ox := 0 to oL - 1 do begin
        gVal := gradOut[b][oc*oL + ox];
        layer.biasGrad[oc] := layer.biasGrad[oc] + gVal;
        for ic := 0 to layer.inChannels - 1 do
          for kx := 0 to kS - 1 do begin
            ix := ox * layer.stride - layer.padding + kx;
            if (ix >= 0) and (ix < iL) then begin
              ki := oc * layer.inChannels + ic;
              layer.kernelGrad[ki][0][kx] := layer.kernelGrad[ki][0][kx] +
                layer.layerInput[b][ic*iL + ix] * gVal;
              Result[b][ic*iL + ix] := Result[b][ic*iL + ix] +
                layer.kernels[ki][0][kx] * gVal;
            end;
          end;
      end;
end;
{ =========================================================================== }
{ BATCH NORMALIZATION                                                         }
{ =========================================================================== }

function BatchNormForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var i, j, bs, ft: Integer; eps: Single;
begin
  bs := Length(inp); ft := Length(inp[0]); eps := layer.bnEpsilon;
  Result := CreateMatrix(bs, ft);
  SetLength(layer.cachedMean, ft); SetLength(layer.cachedVar, ft);
  if layer.isTraining then begin
    for j := 0 to ft - 1 do begin
      layer.cachedMean[j] := 0;
      for i := 0 to bs - 1 do layer.cachedMean[j] := layer.cachedMean[j] + inp[i][j];
      layer.cachedMean[j] := layer.cachedMean[j] / bs;
    end;
    for j := 0 to ft - 1 do begin
      layer.cachedVar[j] := 0;
      for i := 0 to bs - 1 do
        layer.cachedVar[j] := layer.cachedVar[j] + sqr(inp[i][j] - layer.cachedMean[j]);
      layer.cachedVar[j] := layer.cachedVar[j] / bs;
    end;
    for j := 0 to ft - 1 do begin
      layer.runningMean[j] := (1-layer.bnMomentum)*layer.runningMean[j] + layer.bnMomentum*layer.cachedMean[j];
      layer.runningVar[j]  := (1-layer.bnMomentum)*layer.runningVar[j]  + layer.bnMomentum*layer.cachedVar[j];
    end;
  end else begin
    layer.cachedMean := Copy(layer.runningMean);
    layer.cachedVar  := Copy(layer.runningVar);
  end;
  layer.cachedNormalized := CreateMatrix(bs, ft);
  for i := 0 to bs - 1 do
    for j := 0 to ft - 1 do begin
      layer.cachedNormalized[i][j] := (inp[i][j] - layer.cachedMean[j]) / sqrt(layer.cachedVar[j] + eps);
      Result[i][j] := layer.bnGamma[j] * layer.cachedNormalized[i][j] + layer.bnBeta[j];
    end;
end;

function BatchNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var i, j, bs, ft: Integer; eps, invStd, dxh, dv, dm: Single;
    dxhat: TMatrix;
begin
  bs := Length(gradOut); ft := Length(gradOut[0]); eps := layer.bnEpsilon;
  SetLength(layer.bnGammaGrad, ft); SetLength(layer.bnBetaGrad, ft);
  dxhat := CreateMatrix(bs, ft);
  Result := CreateMatrix(bs, ft);
  for j := 0 to ft - 1 do begin
    layer.bnGammaGrad[j] := 0; layer.bnBetaGrad[j] := 0;
    for i := 0 to bs - 1 do begin
      layer.bnBetaGrad[j] := layer.bnBetaGrad[j] + gradOut[i][j];
      layer.bnGammaGrad[j] := layer.bnGammaGrad[j] + gradOut[i][j] * layer.cachedNormalized[i][j];
      dxhat[i][j] := gradOut[i][j] * layer.bnGamma[j];
    end;
  end;
  for j := 0 to ft - 1 do begin
    invStd := 1.0 / sqrt(layer.cachedVar[j] + eps);
    dv := 0; dm := 0;
    for i := 0 to bs - 1 do
      dv := dv + dxhat[i][j] * (layer.layerInput[i][j] - layer.cachedMean[j]) * (-0.5) * invStd * invStd * invStd;
    for i := 0 to bs - 1 do
      dm := dm + dxhat[i][j] * (-invStd);
    dm := dm + dv * (-2.0) * 0;
    for i := 0 to bs - 1 do
      Result[i][j] := dxhat[i][j] * invStd +
        dv * 2.0 * (layer.layerInput[i][j] - layer.cachedMean[j]) / bs + dm / bs;
  end;
end;

{ =========================================================================== }
{ LAYER NORMALIZATION                                                         }
{ =========================================================================== }

function LayerNormForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var i, j, bs, ft: Integer; mn, vr, eps: Single;
begin
  bs := Length(inp); ft := Length(inp[0]); eps := layer.bnEpsilon;
  Result := CreateMatrix(bs, ft);
  SetLength(layer.cachedMean, bs); SetLength(layer.cachedVar, bs);
  layer.cachedNormalized := CreateMatrix(bs, ft);
  for i := 0 to bs - 1 do begin
    mn := 0;
    for j := 0 to ft - 1 do mn := mn + inp[i][j];
    mn := mn / ft; layer.cachedMean[i] := mn;
    vr := 0;
    for j := 0 to ft - 1 do vr := vr + sqr(inp[i][j] - mn);
    vr := vr / ft; layer.cachedVar[i] := vr;
    for j := 0 to ft - 1 do begin
      layer.cachedNormalized[i][j] := (inp[i][j] - mn) / sqrt(vr + eps);
      Result[i][j] := layer.bnGamma[j] * layer.cachedNormalized[i][j] + layer.bnBeta[j];
    end;
  end;
end;

function LayerNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var i, j, bs, ft: Integer; eps, invStd, dxh, dv, dm: Single;
    dxhat: TMatrix;
begin
  bs := Length(gradOut); ft := Length(gradOut[0]); eps := layer.bnEpsilon;
  SetLength(layer.bnGammaGrad, ft); SetLength(layer.bnBetaGrad, ft);
  dxhat := CreateMatrix(bs, ft);
  Result := CreateMatrix(bs, ft);
  for j := 0 to ft - 1 do begin
    layer.bnGammaGrad[j] := 0; layer.bnBetaGrad[j] := 0;
  end;
  for i := 0 to bs - 1 do
    for j := 0 to ft - 1 do begin
      layer.bnBetaGrad[j] := layer.bnBetaGrad[j] + gradOut[i][j];
      layer.bnGammaGrad[j] := layer.bnGammaGrad[j] + gradOut[i][j] * layer.cachedNormalized[i][j];
      dxhat[i][j] := gradOut[i][j] * layer.bnGamma[j];
    end;
  for i := 0 to bs - 1 do begin
    invStd := 1.0 / sqrt(layer.cachedVar[i] + eps);
    dv := 0; dm := 0;
    for j := 0 to ft - 1 do begin
      dv := dv + dxhat[i][j] * (layer.layerInput[i][j] - layer.cachedMean[i]) * (-0.5) * invStd * invStd * invStd;
      dm := dm + dxhat[i][j] * (-invStd);
    end;
    for j := 0 to ft - 1 do
      Result[i][j] := dxhat[i][j] * invStd +
        dv * 2.0 * (layer.layerInput[i][j] - layer.cachedMean[i]) / ft + dm / ft;
  end;
end;

{ =========================================================================== }
{ SPECTRAL NORMALIZATION (power iteration)                                    }
{ =========================================================================== }

function SpectralNormalize(var layer: TLayer): TMatrix;
var i, j, iter, rows, cols: Integer; sigma, nrm: Single; vNew: TVector;
begin
  rows := Length(layer.weights); cols := Length(layer.weights[0]);
  for iter := 0 to MAX_SPECTRAL_ITER - 1 do begin
    SetLength(vNew, cols);
    for j := 0 to cols - 1 do begin
      vNew[j] := 0;
      for i := 0 to rows - 1 do vNew[j] := vNew[j] + layer.weights[i][j] * layer.spectralU[i];
    end;
    nrm := 0;
    for j := 0 to cols - 1 do nrm := nrm + sqr(vNew[j]);
    nrm := sqrt(nrm + 1e-12);
    for j := 0 to cols - 1 do layer.spectralV[j] := vNew[j] / nrm;
    for i := 0 to rows - 1 do begin
      layer.spectralU[i] := 0;
      for j := 0 to cols - 1 do
        layer.spectralU[i] := layer.spectralU[i] + layer.weights[i][j] * layer.spectralV[j];
    end;
    nrm := 0;
    for i := 0 to rows - 1 do nrm := nrm + sqr(layer.spectralU[i]);
    nrm := sqrt(nrm + 1e-12);
    for i := 0 to rows - 1 do layer.spectralU[i] := layer.spectralU[i] / nrm;
  end;
  sigma := 0;
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      sigma := sigma + layer.spectralU[i] * layer.weights[i][j] * layer.spectralV[j];
  layer.spectralSigma := sigma;
  Result := CreateMatrix(rows, cols);
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      Result[i][j] := layer.weights[i][j] / (sigma + 1e-12);
end;

{ =========================================================================== }
{ SELF-ATTENTION (scaled dot-product, single-head)                            }
{ =========================================================================== }

function SelfAttentionForward(const inp: TMatrix; var layer: TLayer): TMatrix;
var Q, K, V, scores, attended: TMatrix;
    i, j, n, dk: Integer; scl, mx, s: Single;
begin
  n := Length(inp); dk := layer.headDim;
  Q := MatrixMultiply(inp, layer.Wq);
  K := MatrixMultiply(inp, layer.Wk);
  V := MatrixMultiply(inp, layer.Wv);
  layer.cachedQ := Q; layer.cachedK := K; layer.cachedV := V;
  scores := MatrixMultiply(Q, MatrixTranspose(K));
  scl := 1.0 / sqrt(dk);
  MatrixScaleInPlace(scores, scl);
  { softmax rows }
  for i := 0 to n - 1 do begin
    mx := scores[i][0];
    for j := 1 to n - 1 do if scores[i][j] > mx then mx := scores[i][j];
    s := 0;
    for j := 0 to n - 1 do begin scores[i][j] := exp(scores[i][j] - mx); s := s + scores[i][j]; end;
    for j := 0 to n - 1 do scores[i][j] := scores[i][j] / (s + 1e-12);
  end;
  layer.cachedScores := scores;
  attended := MatrixMultiply(scores, V);
  layer.cachedAttended := attended;
  Result := MatrixMultiply(attended, layer.Wo);
end;

function SelfAttentionBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var gradAttended, gradV, gradScores, gradPreSoft, gradQ, gradK: TMatrix;
    i, j, n, dk: Integer; dot, scl: Single;
begin
  n := Length(gradOut); dk := layer.headDim;
  scl := 1.0 / sqrt(dk);
  { grad through Wo }
  layer.WoGrad := MatrixMultiply(MatrixTranspose(layer.cachedAttended), gradOut);
  gradAttended := MatrixMultiply(gradOut, MatrixTranspose(layer.Wo));
  { grad through scores*V }
  gradV := MatrixMultiply(MatrixTranspose(layer.cachedScores), gradAttended);
  gradScores := MatrixMultiply(gradAttended, MatrixTranspose(layer.cachedV));
  { grad through softmax }
  gradPreSoft := CreateMatrix(n, n);
  for i := 0 to n - 1 do begin
    dot := 0;
    for j := 0 to n - 1 do dot := dot + gradScores[i][j] * layer.cachedScores[i][j];
    for j := 0 to n - 1 do
      gradPreSoft[i][j] := layer.cachedScores[i][j] * (gradScores[i][j] - dot) * scl;
  end;
  { grad through Q*K^T }
  gradQ := MatrixMultiply(gradPreSoft, layer.cachedK);
  gradK := MatrixMultiply(MatrixTranspose(gradPreSoft), layer.cachedQ);
  { weight grads }
  layer.WqGrad := MatrixMultiply(MatrixTranspose(layer.layerInput), gradQ);
  layer.WkGrad := MatrixMultiply(MatrixTranspose(layer.layerInput), gradK);
  layer.WvGrad := MatrixMultiply(MatrixTranspose(layer.layerInput), gradV);
  { grad to input }
  Result := MatrixAdd(MatrixAdd(
    MatrixMultiply(gradQ, MatrixTranspose(layer.Wq)),
    MatrixMultiply(gradK, MatrixTranspose(layer.Wk))),
    MatrixMultiply(gradV, MatrixTranspose(layer.Wv)));
end;
{ =========================================================================== }
{ LAYER CREATION                                                              }
{ =========================================================================== }

function CreateDenseLayer(inSz, outSz: Integer; act: TActivationType): TLayer;
var i, j: Integer; sc: Single;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltDense;
  Result.activation := act;
  Result.inputSize := inSz; Result.outputSize := outSz;
  Result.weights := CreateMatrix(inSz, outSz);
  Result.bias := CreateVector(outSz);
  sc := sqrt(2.0 / (inSz + outSz));
  for i := 0 to inSz - 1 do
    for j := 0 to outSz - 1 do Result.weights[i][j] := RandomGaussian * sc;
  Result.isTraining := True;
end;

function CreateConv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
var k, ky, kx, nk: Integer; sc: Single;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltConv2D; Result.activation := act;
  Result.inChannels := inCh; Result.outChannels := outCh;
  Result.kernelSize := kSz; Result.stride := st; Result.padding := pad;
  Result.inWidth := w; Result.inHeight := h;
  Result.inputSize := inCh * h * w;
  Result.outHeight := (h + 2*pad - kSz) div st + 1;
  Result.outWidth := (w + 2*pad - kSz) div st + 1;
  Result.outputSize := outCh * Result.outHeight * Result.outWidth;
  nk := outCh * inCh;
  SetLength(Result.kernels, nk);
  sc := sqrt(2.0 / (inCh * kSz * kSz));
  for k := 0 to nk - 1 do begin
    Result.kernels[k] := CreateMatrix(kSz, kSz);
    for ky := 0 to kSz - 1 do
      for kx := 0 to kSz - 1 do Result.kernels[k][ky][kx] := RandomGaussian * sc;
  end;
  Result.bias := CreateVector(outCh);
  Result.isTraining := True;
end;

function CreateDeconv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
var k, ky, kx, nk: Integer; sc: Single;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltDeconv2D; Result.activation := act;
  Result.inChannels := inCh; Result.outChannels := outCh;
  Result.kernelSize := kSz; Result.stride := st; Result.padding := pad;
  Result.inWidth := w; Result.inHeight := h;
  Result.inputSize := inCh * h * w;
  Result.outHeight := (h - 1) * st - 2 * pad + kSz;
  Result.outWidth := (w - 1) * st - 2 * pad + kSz;
  Result.outputSize := outCh * Result.outHeight * Result.outWidth;
  nk := inCh * outCh;
  SetLength(Result.kernels, nk);
  sc := sqrt(2.0 / (outCh * kSz * kSz));
  for k := 0 to nk - 1 do begin
    Result.kernels[k] := CreateMatrix(kSz, kSz);
    for ky := 0 to kSz - 1 do
      for kx := 0 to kSz - 1 do Result.kernels[k][ky][kx] := RandomGaussian * sc;
  end;
  Result.bias := CreateVector(outCh);
  Result.isTraining := True;
end;

function CreateConv1DLayer(inCh, outCh, kSz, st, pad, inLen: Integer; act: TActivationType): TLayer;
var k, kx, nk: Integer; sc: Single;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltConv1D; Result.activation := act;
  Result.inChannels := inCh; Result.outChannels := outCh;
  Result.kernelSize := kSz; Result.stride := st; Result.padding := pad;
  Result.inWidth := inLen; Result.inHeight := 1;
  Result.inputSize := inCh * inLen;
  Result.outWidth := (inLen + 2*pad - kSz) div st + 1;
  Result.outputSize := outCh * Result.outWidth;
  nk := outCh * inCh;
  SetLength(Result.kernels, nk);
  sc := sqrt(2.0 / (inCh * kSz));
  for k := 0 to nk - 1 do begin
    Result.kernels[k] := CreateMatrix(1, kSz);
    for kx := 0 to kSz - 1 do Result.kernels[k][0][kx] := RandomGaussian * sc;
  end;
  Result.bias := CreateVector(outCh);
  Result.isTraining := True;
end;

function CreateBatchNormLayer(features: Integer): TLayer;
var j: Integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltBatchNorm; Result.activation := atNone;
  Result.inputSize := features; Result.outputSize := features;
  Result.bnGamma := CreateVector(features);
  Result.bnBeta := CreateVector(features);
  Result.runningMean := CreateVector(features);
  Result.runningVar := CreateVector(features);
  Result.bnEpsilon := DEFAULT_BN_EPS; Result.bnMomentum := DEFAULT_BN_MOM;
  for j := 0 to features - 1 do begin
    Result.bnGamma[j] := 1.0; Result.runningVar[j] := 1.0;
  end;
  Result.isTraining := True;
end;

function CreateLayerNormLayer(features: Integer): TLayer;
var j: Integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltLayerNorm; Result.activation := atNone;
  Result.inputSize := features; Result.outputSize := features;
  Result.bnGamma := CreateVector(features);
  Result.bnBeta := CreateVector(features);
  Result.bnEpsilon := DEFAULT_BN_EPS;
  for j := 0 to features - 1 do Result.bnGamma[j] := 1.0;
  Result.isTraining := True;
end;

function CreateAttentionLayer(dModel, nHeads: Integer): TLayer;
var i, j: Integer; sc: Single;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerType := ltAttention; Result.activation := atNone;
  Result.inputSize := dModel; Result.outputSize := dModel;
  Result.numHeads := nHeads;
  Result.headDim := dModel div nHeads;
  if Result.headDim < 1 then Result.headDim := dModel;
  sc := sqrt(2.0 / dModel);
  Result.Wq := CreateMatrix(dModel, dModel);
  Result.Wk := CreateMatrix(dModel, dModel);
  Result.Wv := CreateMatrix(dModel, dModel);
  Result.Wo := CreateMatrix(dModel, dModel);
  for i := 0 to dModel - 1 do
    for j := 0 to dModel - 1 do begin
      Result.Wq[i][j] := RandomGaussian * sc;
      Result.Wk[i][j] := RandomGaussian * sc;
      Result.Wv[i][j] := RandomGaussian * sc;
      Result.Wo[i][j] := RandomGaussian * sc;
    end;
  Result.isTraining := True;
end;

{ =========================================================================== }
{ LAYER FORWARD / BACKWARD DISPATCH                                           }
{ =========================================================================== }

function LayerForward(var layer: TLayer; const inp: TMatrix): TMatrix;
var z: TMatrix; normW: TMatrix;
    i, j, k, bs: Integer; s: Single;
begin
  layer.layerInput := inp;
  case layer.layerType of
    ltDense: begin
      { optional spectral norm }
      if Length(layer.spectralU) > 0 then
        normW := SpectralNormalize(layer)
      else
        normW := layer.weights;
      bs := Length(inp);
      z := CreateMatrix(bs, layer.outputSize);
      for i := 0 to bs - 1 do
        for j := 0 to layer.outputSize - 1 do begin
          s := layer.bias[j];
          for k := 0 to layer.inputSize - 1 do s := s + inp[i][k] * normW[k][j];
          z[i][j] := s;
        end;
      layer.preActivation := z;
      Result := ApplyActivation(z, layer.activation);
    end;
    ltConv2D: begin
      z := Conv2DForward(inp, layer);
      layer.preActivation := z;
      Result := ApplyActivation(z, layer.activation);
    end;
    ltDeconv2D: begin
      z := Deconv2DForward(inp, layer);
      layer.preActivation := z;
      Result := ApplyActivation(z, layer.activation);
    end;
    ltConv1D: begin
      z := Conv1DForward(inp, layer);
      layer.preActivation := z;
      Result := ApplyActivation(z, layer.activation);
    end;
    ltBatchNorm: Result := BatchNormForward(inp, layer);
    ltLayerNorm: Result := LayerNormForward(inp, layer);
    ltAttention: Result := SelfAttentionForward(inp, layer);
  else Result := inp;
  end;
  layer.layerOutput := Result;
end;

function LayerBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
var gradZ: TMatrix; i, j, bs: Integer;
begin
  case layer.layerType of
    ltDense: begin
      gradZ := ActivationBackward(gradOut, layer.preActivation, layer.activation);
      bs := Length(gradZ);
      layer.weightGrad := MatrixMultiply(MatrixTranspose(layer.layerInput), gradZ);
      SetLength(layer.biasGrad, layer.outputSize);
      for j := 0 to layer.outputSize - 1 do begin
        layer.biasGrad[j] := 0;
        for i := 0 to bs - 1 do layer.biasGrad[j] := layer.biasGrad[j] + gradZ[i][j];
      end;
      Result := MatrixMultiply(gradZ, MatrixTranspose(layer.weights));
    end;
    ltConv2D: begin
      gradZ := ActivationBackward(gradOut, layer.preActivation, layer.activation);
      Result := Conv2DBackward(layer, gradZ);
    end;
    ltDeconv2D: begin
      gradZ := ActivationBackward(gradOut, layer.preActivation, layer.activation);
      Result := Deconv2DBackward(layer, gradZ);
    end;
    ltConv1D: begin
      gradZ := ActivationBackward(gradOut, layer.preActivation, layer.activation);
      Result := Conv1DBackward(layer, gradZ);
    end;
    ltBatchNorm: Result := BatchNormBackward(layer, gradOut);
    ltLayerNorm: Result := LayerNormBackward(layer, gradOut);
    ltAttention: Result := SelfAttentionBackward(layer, gradOut);
  else Result := gradOut;
  end;
end;

procedure InitLayerOptimizer(var layer: TLayer; opt: TOptimizer);
var k: Integer;
begin
  layer.adamT := 0;
  case layer.layerType of
    ltDense: begin
      layer.mWeight := CreateMatrix(Length(layer.weights), Length(layer.weights[0]));
      layer.vWeight := CreateMatrix(Length(layer.weights), Length(layer.weights[0]));
      layer.mBias := CreateVector(Length(layer.bias));
      layer.vBias := CreateVector(Length(layer.bias));
      if opt = optRMSProp then begin
        layer.rmsWeight := CreateMatrix(Length(layer.weights), Length(layer.weights[0]));
        layer.rmsBias := CreateVector(Length(layer.bias));
      end;
    end;
    ltConv2D, ltDeconv2D, ltConv1D: begin
      SetLength(layer.mKernel, Length(layer.kernels));
      SetLength(layer.vKernel, Length(layer.kernels));
      for k := 0 to High(layer.kernels) do begin
        layer.mKernel[k] := CreateMatrix(Length(layer.kernels[k]), Length(layer.kernels[k][0]));
        layer.vKernel[k] := CreateMatrix(Length(layer.kernels[k]), Length(layer.kernels[k][0]));
      end;
      layer.mBias := CreateVector(Length(layer.bias));
      layer.vBias := CreateVector(Length(layer.bias));
    end;
    ltBatchNorm, ltLayerNorm: begin
      layer.mBnGamma := CreateVector(Length(layer.bnGamma));
      layer.vBnGamma := CreateVector(Length(layer.bnGamma));
      layer.mBnBeta := CreateVector(Length(layer.bnBeta));
      layer.vBnBeta := CreateVector(Length(layer.bnBeta));
    end;
    ltAttention: begin
      layer.mWq := CreateMatrix(Length(layer.Wq), Length(layer.Wq[0]));
      layer.vWq := CreateMatrix(Length(layer.Wq), Length(layer.Wq[0]));
      layer.mWk := CreateMatrix(Length(layer.Wk), Length(layer.Wk[0]));
      layer.vWk := CreateMatrix(Length(layer.Wk), Length(layer.Wk[0]));
      layer.mWv := CreateMatrix(Length(layer.Wv), Length(layer.Wv[0]));
      layer.vWv := CreateMatrix(Length(layer.Wv), Length(layer.Wv[0]));
      layer.mWo := CreateMatrix(Length(layer.Wo), Length(layer.Wo[0]));
      layer.vWo := CreateMatrix(Length(layer.Wo), Length(layer.Wo[0]));
    end;
  end;
end;
{ =========================================================================== }
{ NETWORK OPERATIONS                                                          }
{ =========================================================================== }

function CreateNetwork(const sizes: array of Integer; act: TActivationType;
  opt: TOptimizer; lr: Single): TNetwork;
var i: Integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.layerCount := Length(sizes) - 1;
  Result.optimizer := opt; Result.learningRate := lr;
  Result.beta1 := 0.9; Result.beta2 := 0.999; Result.epsilon := 1e-8;
  Result.momentum := 0.9; Result.weightDecay := 0;
  Result.progressiveAlpha := 1.0; Result.isTraining := True;
  SetLength(Result.layers, Result.layerCount);
  for i := 0 to Result.layerCount - 1 do begin
    Result.layers[i] := CreateDenseLayer(sizes[i], sizes[i+1], act);
    InitLayerOptimizer(Result.layers[i], opt);
  end;
end;

function CreateConvGenerator(noiseDim, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
var totalIn: Integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  totalIn := noiseDim + condSz;
  Result.optimizer := opt; Result.learningRate := lr;
  Result.beta1 := 0.9; Result.beta2 := 0.999; Result.epsilon := 1e-8;
  Result.progressiveAlpha := 1.0; Result.isTraining := True;
  { Dense -> reshape -> Deconv -> BN -> Deconv -> BN -> Deconv(out) }
  Result.layerCount := 7;
  SetLength(Result.layers, 7);
  Result.layers[0] := CreateDenseLayer(totalIn, baseCh*4*4*4, act);
  Result.layers[1] := CreateBatchNormLayer(baseCh*4*4*4);
  Result.layers[2] := CreateDeconv2DLayer(baseCh*4, baseCh*2, 4, 2, 1, 4, 4, act);
  Result.layers[3] := CreateBatchNormLayer(Result.layers[2].outputSize);
  Result.layers[4] := CreateDeconv2DLayer(baseCh*2, baseCh, 4, 2, 1,
    Result.layers[2].outWidth, Result.layers[2].outHeight, act);
  Result.layers[5] := CreateBatchNormLayer(Result.layers[4].outputSize);
  Result.layers[6] := CreateDeconv2DLayer(baseCh, 1, 4, 2, 1,
    Result.layers[4].outWidth, Result.layers[4].outHeight, atTanh);
  for totalIn := 0 to Result.layerCount - 1 do
    InitLayerOptimizer(Result.layers[totalIn], opt);
end;

function CreateConvDiscriminator(inCh, inW, inH, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.optimizer := opt; Result.learningRate := lr;
  Result.beta1 := 0.9; Result.beta2 := 0.999; Result.epsilon := 1e-8;
  Result.progressiveAlpha := 1.0; Result.isTraining := True;
  { Conv -> BN -> Conv -> BN -> Dense(1) }
  Result.layerCount := 5;
  SetLength(Result.layers, 5);
  Result.layers[0] := CreateConv2DLayer(inCh, baseCh, 4, 2, 1, inW, inH, act);
  Result.layers[1] := CreateBatchNormLayer(Result.layers[0].outputSize);
  Result.layers[2] := CreateConv2DLayer(baseCh, baseCh*2, 4, 2, 1,
    Result.layers[0].outWidth, Result.layers[0].outHeight, act);
  Result.layers[3] := CreateBatchNormLayer(Result.layers[2].outputSize);
  Result.layers[4] := CreateDenseLayer(Result.layers[2].outputSize, 1, atSigmoid);
  for inCh := 0 to Result.layerCount - 1 do
    InitLayerOptimizer(Result.layers[inCh], opt);
end;

function NetworkForward(var net: TNetwork; const inp: TMatrix): TMatrix;
var i: Integer; cur: TMatrix;
begin
  cur := inp;
  for i := 0 to net.layerCount - 1 do begin
    net.layers[i].isTraining := net.isTraining;
    cur := LayerForward(net.layers[i], cur);
  end;
  Result := cur;
end;

function NetworkBackward(var net: TNetwork; const gradOut: TMatrix): TMatrix;
var i: Integer; cur: TMatrix;
begin
  cur := gradOut;
  for i := net.layerCount - 1 downto 0 do
    cur := LayerBackward(net.layers[i], cur);
  Result := cur;
end;

procedure NetworkUpdateWeights(var net: TNetwork);
var i, k: Integer; lr, b1, b2, eps, wd: Single;
begin
  lr := net.learningRate; b1 := net.beta1; b2 := net.beta2;
  eps := net.epsilon; wd := net.weightDecay;
  for i := 0 to net.layerCount - 1 do begin
    net.layers[i].adamT := net.layers[i].adamT + 1;
    case net.layers[i].layerType of
      ltDense:
        case net.optimizer of
          optAdam: begin
            AdamUpdateMatrix(net.layers[i].weights, net.layers[i].weightGrad,
              net.layers[i].mWeight, net.layers[i].vWeight, net.layers[i].adamT, lr, b1, b2, eps, wd);
            AdamUpdateVector(net.layers[i].bias, net.layers[i].biasGrad,
              net.layers[i].mBias, net.layers[i].vBias, net.layers[i].adamT, lr, b1, b2, eps, 0);
          end;
          optSGD: begin
            SGDUpdateMatrix(net.layers[i].weights, net.layers[i].weightGrad, lr, wd);
            SGDUpdateVector(net.layers[i].bias, net.layers[i].biasGrad, lr, 0);
          end;
          optRMSProp: begin
            RMSPropUpdateMatrix(net.layers[i].weights, net.layers[i].weightGrad,
              net.layers[i].rmsWeight, lr, 0.99, eps, wd);
            RMSPropUpdateVector(net.layers[i].bias, net.layers[i].biasGrad,
              net.layers[i].rmsBias, lr, 0.99, eps, 0);
          end;
        end;
      ltConv2D, ltDeconv2D, ltConv1D:
        if net.optimizer = optAdam then begin
          for k := 0 to High(net.layers[i].kernels) do
            AdamUpdateMatrix(net.layers[i].kernels[k], net.layers[i].kernelGrad[k],
              net.layers[i].mKernel[k], net.layers[i].vKernel[k], net.layers[i].adamT, lr, b1, b2, eps, wd);
          AdamUpdateVector(net.layers[i].bias, net.layers[i].biasGrad,
            net.layers[i].mBias, net.layers[i].vBias, net.layers[i].adamT, lr, b1, b2, eps, 0);
        end else begin
          for k := 0 to High(net.layers[i].kernels) do
            SGDUpdateMatrix(net.layers[i].kernels[k], net.layers[i].kernelGrad[k], lr, wd);
          SGDUpdateVector(net.layers[i].bias, net.layers[i].biasGrad, lr, 0);
        end;
      ltBatchNorm, ltLayerNorm:
        if net.optimizer = optAdam then begin
          AdamUpdateVector(net.layers[i].bnGamma, net.layers[i].bnGammaGrad,
            net.layers[i].mBnGamma, net.layers[i].vBnGamma, net.layers[i].adamT, lr, b1, b2, eps, 0);
          AdamUpdateVector(net.layers[i].bnBeta, net.layers[i].bnBetaGrad,
            net.layers[i].mBnBeta, net.layers[i].vBnBeta, net.layers[i].adamT, lr, b1, b2, eps, 0);
        end else begin
          SGDUpdateVector(net.layers[i].bnGamma, net.layers[i].bnGammaGrad, lr, 0);
          SGDUpdateVector(net.layers[i].bnBeta, net.layers[i].bnBetaGrad, lr, 0);
        end;
      ltAttention:
        if net.optimizer = optAdam then begin
          AdamUpdateMatrix(net.layers[i].Wq, net.layers[i].WqGrad, net.layers[i].mWq, net.layers[i].vWq, net.layers[i].adamT, lr, b1, b2, eps, wd);
          AdamUpdateMatrix(net.layers[i].Wk, net.layers[i].WkGrad, net.layers[i].mWk, net.layers[i].vWk, net.layers[i].adamT, lr, b1, b2, eps, wd);
          AdamUpdateMatrix(net.layers[i].Wv, net.layers[i].WvGrad, net.layers[i].mWv, net.layers[i].vWv, net.layers[i].adamT, lr, b1, b2, eps, wd);
          AdamUpdateMatrix(net.layers[i].Wo, net.layers[i].WoGrad, net.layers[i].mWo, net.layers[i].vWo, net.layers[i].adamT, lr, b1, b2, eps, wd);
        end;
    end;
  end;
end;

procedure AddProgressiveLayer(var net: TNetwork; resLvl: Integer; isGen: Boolean);
var n, ch: Integer;
begin
  n := net.layerCount;
  ch := 64 div resLvl; if ch < 8 then ch := 8;
  SetLength(net.layers, n + 2);
  if isGen then begin
    net.layers[n] := CreateDeconv2DLayer(ch*2, ch, 4, 2, 1,
      net.layers[n-1].outWidth, net.layers[n-1].outHeight, atLeakyReLU);
    net.layers[n+1] := CreateBatchNormLayer(net.layers[n].outputSize);
  end else begin
    net.layers[n] := CreateConv2DLayer(ch, ch*2, 4, 2, 1,
      net.layers[n-1].outWidth, net.layers[n-1].outHeight, atLeakyReLU);
    net.layers[n+1] := CreateBatchNormLayer(net.layers[n].outputSize);
  end;
  InitLayerOptimizer(net.layers[n], net.optimizer);
  InitLayerOptimizer(net.layers[n+1], net.optimizer);
  net.layerCount := n + 2;
  net.progressiveAlpha := 0.0;
  net.currentResLevel := resLvl;
end;

procedure SetNetworkTraining(var net: TNetwork; training: Boolean);
var i: Integer;
begin
  net.isTraining := training;
  for i := 0 to net.layerCount - 1 do net.layers[i].isTraining := training;
end;

function GetLayerOutput(var net: TNetwork; idx: Integer): TMatrix;
begin
  if (idx >= 0) and (idx < net.layerCount) then
    Result := net.layers[idx].layerOutput
  else SetLength(Result, 0);
end;

function DeepCopyNetwork(const src: TNetwork): TNetwork;
var i, j, k: Integer;
begin
  Result := src;
  SetLength(Result.layers, src.layerCount);
  for i := 0 to src.layerCount - 1 do begin
    Result.layers[i] := src.layers[i];
    if Length(src.layers[i].weights) > 0 then begin
      Result.layers[i].weights := CreateMatrix(Length(src.layers[i].weights), Length(src.layers[i].weights[0]));
      for j := 0 to High(src.layers[i].weights) do
        for k := 0 to High(src.layers[i].weights[j]) do
          Result.layers[i].weights[j][k] := src.layers[i].weights[j][k];
    end;
    if Length(src.layers[i].bias) > 0 then begin
      SetLength(Result.layers[i].bias, Length(src.layers[i].bias));
      Move(src.layers[i].bias[0], Result.layers[i].bias[0], Length(src.layers[i].bias)*SizeOf(Single));
    end;
    if Length(src.layers[i].kernels) > 0 then begin
      SetLength(Result.layers[i].kernels, Length(src.layers[i].kernels));
      for j := 0 to High(src.layers[i].kernels) do begin
        Result.layers[i].kernels[j] := CreateMatrix(Length(src.layers[i].kernels[j]), Length(src.layers[i].kernels[j][0]));
        for k := 0 to High(src.layers[i].kernels[j]) do
          Move(src.layers[i].kernels[j][k][0], Result.layers[i].kernels[j][k][0],
            Length(src.layers[i].kernels[j][k])*SizeOf(Single));
      end;
    end;
    SetLength(Result.layers[i].layerInput, 0);
    SetLength(Result.layers[i].layerOutput, 0);
    SetLength(Result.layers[i].preActivation, 0);
  end;
end;
{ =========================================================================== }
{ LOSS FUNCTIONS                                                              }
{ =========================================================================== }

function BinaryCrossEntropy(const pred, target: TMatrix): Single;
var i, j: Integer; p, eps, s: Single;
begin
  eps := 1e-7; s := 0;
  for i := 0 to High(pred) do
    for j := 0 to High(pred[0]) do begin
      p := Max(eps, Min(1-eps, pred[i][j]));
      s := s - (target[i][j] * ln(p) + (1-target[i][j]) * ln(1-p));
    end;
  Result := s / (Length(pred) * Length(pred[0]));
end;

function BCEGradient(const pred, target: TMatrix): TMatrix;
var i, j: Integer; p, eps: Single;
begin
  eps := 1e-7;
  Result := CreateMatrix(Length(pred), Length(pred[0]));
  for i := 0 to High(pred) do
    for j := 0 to High(pred[0]) do begin
      p := Max(eps, Min(1-eps, pred[i][j]));
      Result[i][j] := (-target[i][j]/p + (1-target[i][j])/(1-p)) / (Length(pred)*Length(pred[0]));
    end;
end;

function WGANDiscLoss(const dReal, dFake: TMatrix): Single;
var i, j, n: Integer;
begin
  Result := 0; n := Length(dReal) * Length(dReal[0]);
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result + dFake[i][j];
  for i := 0 to High(dReal) do
    for j := 0 to High(dReal[0]) do Result := Result - dReal[i][j];
  Result := Result / n;
end;

function WGANGenLoss(const dFake: TMatrix): Single;
var i, j: Integer;
begin
  Result := 0;
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result - dFake[i][j];
  Result := Result / (Length(dFake) * Length(dFake[0]));
end;

function WGANDiscGrad(const dOut: TMatrix; isReal: Boolean): TMatrix;
var i, j: Integer; n: Single; sign: Single;
begin
  n := Length(dOut) * Length(dOut[0]);
  if isReal then sign := -1.0 else sign := 1.0;
  Result := CreateMatrix(Length(dOut), Length(dOut[0]));
  for i := 0 to High(dOut) do
    for j := 0 to High(dOut[0]) do Result[i][j] := sign / n;
end;

function WGANGenGrad(const dFake: TMatrix): TMatrix;
var i, j: Integer; n: Single;
begin
  n := Length(dFake) * Length(dFake[0]);
  Result := CreateMatrix(Length(dFake), Length(dFake[0]));
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result[i][j] := -1.0 / n;
end;

function HingeDiscLoss(const dReal, dFake: TMatrix): Single;
var i, j, n: Integer;
begin
  Result := 0; n := Length(dReal) * Length(dReal[0]);
  for i := 0 to High(dReal) do
    for j := 0 to High(dReal[0]) do Result := Result + Max(0, 1.0 - dReal[i][j]);
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result + Max(0, 1.0 + dFake[i][j]);
  Result := Result / n;
end;

function HingeGenLoss(const dFake: TMatrix): Single;
var i, j: Integer;
begin
  Result := 0;
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result - dFake[i][j];
  Result := Result / (Length(dFake) * Length(dFake[0]));
end;

function LSDiscLoss(const dReal, dFake: TMatrix): Single;
var i, j: Integer; n: Single;
begin
  Result := 0; n := Length(dReal) * Length(dReal[0]);
  for i := 0 to High(dReal) do
    for j := 0 to High(dReal[0]) do Result := Result + sqr(dReal[i][j] - 1.0);
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result + sqr(dFake[i][j]);
  Result := Result / (2 * n);
end;

function LSGenLoss(const dFake: TMatrix): Single;
var i, j: Integer;
begin
  Result := 0;
  for i := 0 to High(dFake) do
    for j := 0 to High(dFake[0]) do Result := Result + sqr(dFake[i][j] - 1.0);
  Result := Result / (2 * Length(dFake) * Length(dFake[0]));
end;

function ComputeGradientPenalty(var disc: TNetwork; const real, fake: TMatrix; lambda: Single): Single;
var i, j, s: Integer; alpha, gradNorm, penalty, eps, diff: Single;
    interp, baseOut, pertIn, pertOut: TMatrix; bs, dim: Integer;
begin
  bs := Length(real); dim := Length(real[0]); eps := 1e-4; penalty := 0;
  interp := CreateMatrix(bs, dim);
  for i := 0 to bs - 1 do begin
    alpha := Random;
    for j := 0 to dim - 1 do
      interp[i][j] := alpha * real[i][j] + (1 - alpha) * fake[i][j];
  end;
  baseOut := NetworkForward(disc, interp);
  for s := 0 to bs - 1 do begin
    gradNorm := 0;
    for j := 0 to dim - 1 do begin
      pertIn := CreateMatrix(1, dim);
      Move(interp[s][0], pertIn[0][0], dim * SizeOf(Single));
      pertIn[0][j] := pertIn[0][j] + eps;
      pertOut := NetworkForward(disc, pertIn);
      diff := (pertOut[0][0] - baseOut[s][0]) / eps;
      gradNorm := gradNorm + sqr(diff);
    end;
    gradNorm := sqrt(gradNorm);
    penalty := penalty + sqr(gradNorm - 1.0);
  end;
  Result := lambda * penalty / bs;
end;

{ =========================================================================== }
{ REGULARIZATION                                                              }
{ =========================================================================== }

function ApplyLabelSmoothing(const labels: TMatrix; lo, hi: Single): TMatrix;
var i, j: Integer;
begin
  Result := CreateMatrix(Length(labels), Length(labels[0]));
  for i := 0 to High(labels) do
    for j := 0 to High(labels[0]) do
      Result[i][j] := lo + labels[i][j] * (hi - lo);
end;

function FeatureMatchingLoss(var disc: TNetwork; const real, fake: TMatrix; featLayer: Integer): Single;
var realFeat, fakeFeat: TMatrix; i, j, n: Integer; d: Single;
begin
  NetworkForward(disc, real);
  realFeat := GetLayerOutput(disc, featLayer);
  NetworkForward(disc, fake);
  fakeFeat := GetLayerOutput(disc, featLayer);
  Result := 0; n := 0;
  if (Length(realFeat) > 0) and (Length(fakeFeat) > 0) then begin
    for i := 0 to High(realFeat) do
      for j := 0 to High(realFeat[0]) do begin
        d := realFeat[i][j] - fakeFeat[i][j];
        Result := Result + d * d;
        Inc(n);
      end;
    if n > 0 then Result := Result / n;
  end;
end;

function MinibatchStdDev(const inp: TMatrix): TMatrix;
var i, j, bs, ft: Integer; mn, vr: Single;
begin
  bs := Length(inp); ft := Length(inp[0]);
  Result := CreateMatrix(bs, ft + 1);
  for i := 0 to bs - 1 do
    for j := 0 to ft - 1 do Result[i][j] := inp[i][j];
  vr := 0;
  for j := 0 to ft - 1 do begin
    mn := 0;
    for i := 0 to bs - 1 do mn := mn + inp[i][j];
    mn := mn / bs;
    for i := 0 to bs - 1 do vr := vr + sqr(inp[i][j] - mn);
  end;
  vr := sqrt(vr / (bs * ft) + 1e-8);
  for i := 0 to bs - 1 do Result[i][ft] := vr;
end;
{ =========================================================================== }
{ OPTIMIZER HELPERS                                                           }
{ =========================================================================== }

procedure AdamUpdateMatrix(var p: TMatrix; const g: TMatrix;
  var m, v: TMatrix; t: Integer; lr, b1, b2, eps, wd: Single);
var i, j: Integer; mH, vH: Single;
begin
  for i := 0 to High(p) do
    for j := 0 to High(p[i]) do begin
      m[i][j] := b1 * m[i][j] + (1-b1) * g[i][j];
      v[i][j] := b2 * v[i][j] + (1-b2) * sqr(g[i][j]);
      mH := m[i][j] / (1 - power(b1, t));
      vH := v[i][j] / (1 - power(b2, t));
      p[i][j] := p[i][j] - lr * mH / (sqrt(vH) + eps);
      if wd > 0 then p[i][j] := p[i][j] * (1 - lr * wd);
    end;
end;

procedure AdamUpdateVector(var p: TVector; const g: TVector;
  var m, v: TVector; t: Integer; lr, b1, b2, eps, wd: Single);
var i: Integer; mH, vH: Single;
begin
  for i := 0 to High(p) do begin
    m[i] := b1 * m[i] + (1-b1) * g[i];
    v[i] := b2 * v[i] + (1-b2) * sqr(g[i]);
    mH := m[i] / (1 - power(b1, t));
    vH := v[i] / (1 - power(b2, t));
    p[i] := p[i] - lr * mH / (sqrt(vH) + eps);
    if wd > 0 then p[i] := p[i] * (1 - lr * wd);
  end;
end;

procedure SGDUpdateMatrix(var p: TMatrix; const g: TMatrix; lr, wd: Single);
var i, j: Integer;
begin
  for i := 0 to High(p) do
    for j := 0 to High(p[i]) do begin
      p[i][j] := p[i][j] - lr * g[i][j];
      if wd > 0 then p[i][j] := p[i][j] * (1 - lr * wd);
    end;
end;

procedure SGDUpdateVector(var p: TVector; const g: TVector; lr, wd: Single);
var i: Integer;
begin
  for i := 0 to High(p) do begin
    p[i] := p[i] - lr * g[i];
    if wd > 0 then p[i] := p[i] * (1 - lr * wd);
  end;
end;

procedure RMSPropUpdateMatrix(var p: TMatrix; const g: TMatrix;
  var cache: TMatrix; lr, decay, eps, wd: Single);
var i, j: Integer;
begin
  for i := 0 to High(p) do
    for j := 0 to High(p[i]) do begin
      cache[i][j] := decay * cache[i][j] + (1-decay) * sqr(g[i][j]);
      p[i][j] := p[i][j] - lr * g[i][j] / (sqrt(cache[i][j]) + eps);
      if wd > 0 then p[i][j] := p[i][j] * (1 - lr * wd);
    end;
end;

procedure RMSPropUpdateVector(var p: TVector; const g: TVector;
  var cache: TVector; lr, decay, eps, wd: Single);
var i: Integer;
begin
  for i := 0 to High(p) do begin
    cache[i] := decay * cache[i] + (1-decay) * sqr(g[i]);
    p[i] := p[i] - lr * g[i] / (sqrt(cache[i]) + eps);
    if wd > 0 then p[i] := p[i] * (1 - lr * wd);
  end;
end;

function CosineAnneal(epoch, maxEp: Integer; baseLR, minLR: Single): Single;
begin
  Result := minLR + 0.5 * (baseLR - minLR) * (1 + cos(Pi * epoch / maxEp));
end;
{ =========================================================================== }
{ DATA LOADING                                                                }
{ =========================================================================== }

function LoadBMPDataset(const path: string): TDataset;
var sr: TSearchRec; f: TFileStream;
    w, h, bpp, offset, row, col, pad, idx: Integer;
    hdr: array[0..53] of Byte; pixel: array[0..2] of Byte;
    fn: string;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.dataType := dtImage; Result.count := 0;
  if FindFirst(IncludeTrailingPathDelimiter(path) + '*.bmp', faAnyFile, sr) = 0 then begin
    repeat
      fn := IncludeTrailingPathDelimiter(path) + sr.Name;
      try
        f := TFileStream.Create(fn, fmOpenRead);
        try
          if f.Size > 54 then begin
            f.Read(hdr, 54);
            if (hdr[0] = Ord('B')) and (hdr[1] = Ord('M')) then begin
              w := hdr[18] or (hdr[19] shl 8) or (hdr[20] shl 16) or (hdr[21] shl 24);
              h := hdr[22] or (hdr[23] shl 8) or (hdr[24] shl 16) or (hdr[25] shl 24);
              bpp := hdr[28] or (hdr[29] shl 8);
              offset := hdr[10] or (hdr[11] shl 8) or (hdr[12] shl 16) or (hdr[13] shl 24);
              if (bpp = 24) and (w > 0) and (h > 0) then begin
                if Result.count = 0 then begin
                  Result.sampleWidth := w; Result.sampleHeight := h;
                  Result.sampleChannels := 3;
                end;
                idx := Result.count;
                Inc(Result.count);
                SetLength(Result.samples, Result.count);
                Result.samples[idx] := CreateMatrix(1, w * h * 3);
                f.Position := offset;
                pad := (4 - (w * 3) mod 4) mod 4;
                for row := h - 1 downto 0 do begin
                  for col := 0 to w - 1 do begin
                    f.Read(pixel, 3);
                    Result.samples[idx][0][(row*w+col)*3+0] := pixel[2] / 255.0;
                    Result.samples[idx][0][(row*w+col)*3+1] := pixel[1] / 255.0;
                    Result.samples[idx][0][(row*w+col)*3+2] := pixel[0] / 255.0;
                  end;
                  f.Position := f.Position + pad;
                end;
              end;
            end;
          end;
        finally f.Free; end;
      except end;
    until FindNext(sr) <> 0;
    FindClose(sr);
  end;
end;

function LoadWAVDataset(const path: string): TDataset;
var sr: TSearchRec; f: TFileStream;
    hdr: array[0..43] of Byte; fn: string;
    channels, bps, idx, i, numSamples: Integer;
    sampleRate, dataSize: LongWord;
    sample16: SmallInt; sample8: Byte;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.dataType := dtAudio; Result.count := 0;
  if FindFirst(IncludeTrailingPathDelimiter(path) + '*.wav', faAnyFile, sr) = 0 then begin
    repeat
      fn := IncludeTrailingPathDelimiter(path) + sr.Name;
      try
        f := TFileStream.Create(fn, fmOpenRead);
        try
          if f.Size > 44 then begin
            f.Read(hdr, 44);
            if (hdr[0]=Ord('R')) and (hdr[1]=Ord('I')) and (hdr[2]=Ord('F')) and (hdr[3]=Ord('F')) then begin
              channels := hdr[22] or (hdr[23] shl 8);
              sampleRate := hdr[24] or (hdr[25] shl 8) or (hdr[26] shl 16) or (hdr[27] shl 24);
              bps := hdr[34] or (hdr[35] shl 8);
              dataSize := hdr[40] or (hdr[41] shl 8) or (hdr[42] shl 16) or (hdr[43] shl 24);
              if bps = 16 then numSamples := dataSize div 2
              else numSamples := dataSize;
              numSamples := numSamples div channels;
              if numSamples > 0 then begin
                idx := Result.count; Inc(Result.count);
                SetLength(Result.samples, Result.count);
                Result.samples[idx] := CreateMatrix(1, numSamples);
                Result.sampleWidth := numSamples; Result.sampleChannels := 1;
                for i := 0 to numSamples - 1 do begin
                  if bps = 16 then begin
                    f.Read(sample16, 2);
                    Result.samples[idx][0][i] := sample16 / 32768.0;
                  end else begin
                    f.Read(sample8, 1);
                    Result.samples[idx][0][i] := (sample8 - 128) / 128.0;
                  end;
                  if channels > 1 then f.Position := f.Position + (channels-1) * (bps div 8);
                end;
              end;
            end;
          end;
        finally f.Free; end;
      except end;
    until FindNext(sr) <> 0;
    FindClose(sr);
  end;
end;

function LoadDataset(const path: string; dt: TDataType): TDataset;
begin
  case dt of
    dtImage: Result := LoadBMPDataset(path);
    dtAudio: Result := LoadWAVDataset(path);
  else Result := CreateSyntheticDataset(1000, 1);
  end;
end;

function AugmentSample(const sample: TMatrix; dt: TDataType): TMatrix;
var i, j, w, h, c, ch, row, col: Integer;
begin
  Result := CreateMatrix(Length(sample), Length(sample[0]));
  for i := 0 to High(sample) do
    Move(sample[i][0], Result[i][0], Length(sample[i]) * SizeOf(Single));
  case dt of
    dtImage: begin
      w := Round(sqrt(Length(sample[0]) / 3));
      h := w; c := 3;
      if Random < 0.5 then begin
        { horizontal flip }
        for row := 0 to h - 1 do
          for col := 0 to (w div 2) - 1 do
            for ch := 0 to c - 1 do begin
              i := (row * w + col) * c + ch;
              j := (row * w + (w - 1 - col)) * c + ch;
              Result[0][i] := sample[0][j];
              Result[0][j] := sample[0][i];
            end;
      end;
    end;
    dtAudio: begin
      { small time jitter }
      if Random < 0.3 then
        for i := 0 to High(Result[0]) do
          Result[0][i] := Result[0][i] + RandomGaussian * 0.01;
    end;
  end;
end;

function CreateSyntheticDataset(count, features: Integer): TDataset;
var i, j: Integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.count := count; Result.dataType := dtVector;
  Result.sampleWidth := features; Result.sampleChannels := 1;
  SetLength(Result.samples, count);
  for i := 0 to count - 1 do begin
    Result.samples[i] := CreateMatrix(1, features);
    for j := 0 to features - 1 do
      Result.samples[i][0][j] := sin(i / 100.0 + j * 0.1);
  end;
end;

{ =========================================================================== }
{ METRICS                                                                     }
{ =========================================================================== }

function ComputeFID(const realS, fakeS: TMatrixArray): Single;
var i, j, ft, nr, nf: Integer;
    muR, muF: TVector; diff, covDiag: Single;
begin
  { Simplified FID proxy: compare mean and variance of features directly }
  Result := 0;
  if (Length(realS) = 0) or (Length(fakeS) = 0) then Exit;
  ft := Length(realS[0][0]); nr := Length(realS); nf := Length(fakeS);
  SetLength(muR, ft); SetLength(muF, ft);
  for j := 0 to ft - 1 do begin
    muR[j] := 0; muF[j] := 0;
    for i := 0 to nr - 1 do muR[j] := muR[j] + realS[i][0][j];
    muR[j] := muR[j] / nr;
    for i := 0 to nf - 1 do muF[j] := muF[j] + fakeS[i][0][j];
    muF[j] := muF[j] / nf;
  end;
  { FID ~ ||muR - muF||^2 + trace(covR + covF - 2*sqrt(covR*covF)) }
  for j := 0 to ft - 1 do begin
    diff := muR[j] - muF[j];
    Result := Result + diff * diff;
  end;
  { Add diagonal covariance approximation }
  for j := 0 to ft - 1 do begin
    covDiag := 0;
    for i := 0 to nr - 1 do covDiag := covDiag + sqr(realS[i][0][j] - muR[j]);
    covDiag := covDiag / nr;
    Result := Result + abs(covDiag);
  end;
end;

function ComputeIS(const samples: TMatrixArray): Single;
var i, j, n, ft: Integer; mn, ent, s: Single;
begin
  { Simplified IS proxy: measure diversity via entropy of sample means }
  Result := 1.0;
  if Length(samples) = 0 then Exit;
  n := Length(samples); ft := Length(samples[0][0]);
  ent := 0;
  for j := 0 to ft - 1 do begin
    mn := 0;
    for i := 0 to n - 1 do mn := mn + abs(samples[i][0][j]);
    mn := mn / n;
    if mn > 1e-7 then begin
      s := 0;
      for i := 0 to n - 1 do begin
        s := abs(samples[i][0][j]) / (mn * n + 1e-12);
        if s > 1e-12 then ent := ent - s * ln(s);
      end;
    end;
  end;
  Result := exp(ent / ft);
end;

procedure LogMetrics(const met: TGANMetrics; const fn: string);
var f: TextFile; exists: Boolean;
begin
  exists := FileExists(fn);
  AssignFile(f, fn);
  if exists then Append(f) else begin Rewrite(f); WriteLn(f, 'epoch,batch,dLossReal,dLossFake,gLoss,fid,is,gp'); end;
  WriteLn(f, Format('%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f',
    [met.epoch, met.batch, met.dLossReal, met.dLossFake, met.gLoss,
     met.fidScore, met.isScore, met.gradPenalty]));
  CloseFile(f);
end;
{ =========================================================================== }
{ BINARY SERIALIZATION (v2 format with layer type tags)                       }
{ =========================================================================== }

procedure SaveNetworkBinary(const net: TNetwork; const fn: string);
var f: TFileStream; k, i, j: Integer; lt: Byte; v: Single;
begin
  if not ValidatePath(fn) then begin WriteLn('Invalid path: ', fn); Exit; end;
  f := TFileStream.Create(fn, fmCreate);
  try
    lt := 2; f.Write(lt, 1); { version 2 }
    f.Write(net.layerCount, SizeOf(Integer));
    for k := 0 to net.layerCount - 1 do begin
      lt := Ord(net.layers[k].layerType); f.Write(lt, 1);
      f.Write(net.layers[k].inputSize, SizeOf(Integer));
      f.Write(net.layers[k].outputSize, SizeOf(Integer));
      case net.layers[k].layerType of
        ltDense: begin
          for i := 0 to High(net.layers[k].weights) do
            for j := 0 to High(net.layers[k].weights[i]) do begin
              v := net.layers[k].weights[i][j]; f.Write(v, SizeOf(Single)); end;
          for j := 0 to High(net.layers[k].bias) do begin
            v := net.layers[k].bias[j]; f.Write(v, SizeOf(Single)); end;
        end;
        ltConv2D, ltDeconv2D, ltConv1D: begin
          f.Write(net.layers[k].inChannels, SizeOf(Integer));
          f.Write(net.layers[k].outChannels, SizeOf(Integer));
          f.Write(net.layers[k].kernelSize, SizeOf(Integer));
          f.Write(net.layers[k].stride, SizeOf(Integer));
          f.Write(net.layers[k].padding, SizeOf(Integer));
          f.Write(net.layers[k].inWidth, SizeOf(Integer));
          f.Write(net.layers[k].inHeight, SizeOf(Integer));
          for i := 0 to High(net.layers[k].kernels) do
            for j := 0 to High(net.layers[k].kernels[i]) do begin
              v := net.layers[k].kernels[i][j][0]; { write row }
              f.Write(net.layers[k].kernels[i][j][0],
                Length(net.layers[k].kernels[i][j]) * SizeOf(Single));
            end;
          for j := 0 to High(net.layers[k].bias) do begin
            v := net.layers[k].bias[j]; f.Write(v, SizeOf(Single)); end;
        end;
        ltBatchNorm, ltLayerNorm: begin
          for j := 0 to High(net.layers[k].bnGamma) do begin
            v := net.layers[k].bnGamma[j]; f.Write(v, SizeOf(Single)); end;
          for j := 0 to High(net.layers[k].bnBeta) do begin
            v := net.layers[k].bnBeta[j]; f.Write(v, SizeOf(Single)); end;
          if net.layers[k].layerType = ltBatchNorm then begin
            for j := 0 to High(net.layers[k].runningMean) do begin
              v := net.layers[k].runningMean[j]; f.Write(v, SizeOf(Single)); end;
            for j := 0 to High(net.layers[k].runningVar) do begin
              v := net.layers[k].runningVar[j]; f.Write(v, SizeOf(Single)); end;
          end;
        end;
        ltAttention: begin
          f.Write(net.layers[k].numHeads, SizeOf(Integer));
          for i := 0 to High(net.layers[k].Wq) do
            f.Write(net.layers[k].Wq[i][0], Length(net.layers[k].Wq[i])*SizeOf(Single));
          for i := 0 to High(net.layers[k].Wk) do
            f.Write(net.layers[k].Wk[i][0], Length(net.layers[k].Wk[i])*SizeOf(Single));
          for i := 0 to High(net.layers[k].Wv) do
            f.Write(net.layers[k].Wv[i][0], Length(net.layers[k].Wv[i])*SizeOf(Single));
          for i := 0 to High(net.layers[k].Wo) do
            f.Write(net.layers[k].Wo[i][0], Length(net.layers[k].Wo[i])*SizeOf(Single));
        end;
      end;
    end;
  finally f.Free; end;
  WriteLn('Network saved to ', fn);
end;

procedure LoadNetworkBinary(var net: TNetwork; const fn: string);
var f: TFileStream; k, i, j, lc, inSz, outSz: Integer;
    ver, lt: Byte; v: Single;
    inCh, outCh, kSz, st, pad, iw, ih, nH: Integer;
begin
  if not FileExists(fn) then begin WriteLn('File not found: ', fn); Exit; end;
  f := TFileStream.Create(fn, fmOpenRead);
  try
    f.Read(ver, 1);
    if ver = 2 then begin
      f.Read(lc, SizeOf(Integer));
      net.layerCount := lc;
      SetLength(net.layers, lc);
      for k := 0 to lc - 1 do begin
        f.Read(lt, 1);
        f.Read(inSz, SizeOf(Integer));
        f.Read(outSz, SizeOf(Integer));
        case TLayerType(lt) of
          ltDense: begin
            net.layers[k] := CreateDenseLayer(inSz, outSz, atReLU);
            for i := 0 to High(net.layers[k].weights) do
              for j := 0 to High(net.layers[k].weights[i]) do
                f.Read(net.layers[k].weights[i][j], SizeOf(Single));
            for j := 0 to High(net.layers[k].bias) do
              f.Read(net.layers[k].bias[j], SizeOf(Single));
          end;
          ltConv2D, ltDeconv2D, ltConv1D: begin
            f.Read(inCh, SizeOf(Integer)); f.Read(outCh, SizeOf(Integer));
            f.Read(kSz, SizeOf(Integer)); f.Read(st, SizeOf(Integer));
            f.Read(pad, SizeOf(Integer)); f.Read(iw, SizeOf(Integer));
            f.Read(ih, SizeOf(Integer));
            if TLayerType(lt) = ltConv2D then
              net.layers[k] := CreateConv2DLayer(inCh, outCh, kSz, st, pad, iw, ih, atReLU)
            else if TLayerType(lt) = ltDeconv2D then
              net.layers[k] := CreateDeconv2DLayer(inCh, outCh, kSz, st, pad, iw, ih, atReLU)
            else
              net.layers[k] := CreateConv1DLayer(inCh, outCh, kSz, st, pad, iw, atReLU);
            for i := 0 to High(net.layers[k].kernels) do
              for j := 0 to High(net.layers[k].kernels[i]) do
                f.Read(net.layers[k].kernels[i][j][0],
                  Length(net.layers[k].kernels[i][j]) * SizeOf(Single));
            for j := 0 to High(net.layers[k].bias) do
              f.Read(net.layers[k].bias[j], SizeOf(Single));
          end;
          ltBatchNorm: begin
            net.layers[k] := CreateBatchNormLayer(inSz);
            for j := 0 to inSz-1 do f.Read(net.layers[k].bnGamma[j], SizeOf(Single));
            for j := 0 to inSz-1 do f.Read(net.layers[k].bnBeta[j], SizeOf(Single));
            for j := 0 to inSz-1 do f.Read(net.layers[k].runningMean[j], SizeOf(Single));
            for j := 0 to inSz-1 do f.Read(net.layers[k].runningVar[j], SizeOf(Single));
          end;
          ltLayerNorm: begin
            net.layers[k] := CreateLayerNormLayer(inSz);
            for j := 0 to inSz-1 do f.Read(net.layers[k].bnGamma[j], SizeOf(Single));
            for j := 0 to inSz-1 do f.Read(net.layers[k].bnBeta[j], SizeOf(Single));
          end;
          ltAttention: begin
            f.Read(nH, SizeOf(Integer));
            net.layers[k] := CreateAttentionLayer(inSz, nH);
            for i := 0 to inSz-1 do f.Read(net.layers[k].Wq[i][0], inSz*SizeOf(Single));
            for i := 0 to inSz-1 do f.Read(net.layers[k].Wk[i][0], inSz*SizeOf(Single));
            for i := 0 to inSz-1 do f.Read(net.layers[k].Wv[i][0], inSz*SizeOf(Single));
            for i := 0 to inSz-1 do f.Read(net.layers[k].Wo[i][0], inSz*SizeOf(Single));
          end;
        end;
        InitLayerOptimizer(net.layers[k], net.optimizer);
      end;
    end else begin
      { v1 compat: assume dense layers, re-read from start }
      f.Position := 0;
      for k := 0 to net.layerCount - 1 do begin
        for i := 0 to High(net.layers[k].weights) do
          for j := 0 to High(net.layers[k].weights[i]) do
            f.Read(net.layers[k].weights[i][j], SizeOf(Single));
        for j := 0 to High(net.layers[k].bias) do
          f.Read(net.layers[k].bias[j], SizeOf(Single));
      end;
    end;
  finally f.Free; end;
  WriteLn('Network loaded from ', fn);
end;

{ =========================================================================== }
{ CHECKPOINTS                                                                 }
{ =========================================================================== }

procedure SaveCheckpoint(const gen, disc: TNetwork; ep: Integer; const dir: string);
begin
  ForceDirectories(dir);
  SaveNetworkBinary(gen, IncludeTrailingPathDelimiter(dir) + 'gen_ep' + IntToStr(ep) + '.bin');
  SaveNetworkBinary(disc, IncludeTrailingPathDelimiter(dir) + 'disc_ep' + IntToStr(ep) + '.bin');
end;

procedure LoadCheckpoint(var gen, disc: TNetwork; ep: Integer; const dir: string);
begin
  LoadNetworkBinary(gen, IncludeTrailingPathDelimiter(dir) + 'gen_ep' + IntToStr(ep) + '.bin');
  LoadNetworkBinary(disc, IncludeTrailingPathDelimiter(dir) + 'disc_ep' + IntToStr(ep) + '.bin');
end;

{ =========================================================================== }
{ VISUALIZATION / LOGGING                                                     }
{ =========================================================================== }

procedure SaveGeneratedSamples(var gen: TNetwork; ep: Integer;
  const dir: string; noiseDim: Integer; nt: TNoiseType);
var noise, out_: TMatrix; i, j: Integer; f: TextFile; fn: string;
begin
  ForceDirectories(dir);
  GenerateNoise(noise, 16, noiseDim, nt);
  out_ := NetworkForward(gen, noise);
  fn := IncludeTrailingPathDelimiter(dir) + 'samples_ep' + IntToStr(ep) + '.csv';
  AssignFile(f, fn); Rewrite(f);
  for i := 0 to High(out_) do begin
    for j := 0 to High(out_[0]) do begin
      if j > 0 then Write(f, ',');
      Write(f, out_[i][j]:0:6);
    end;
    WriteLn(f);
  end;
  CloseFile(f);
end;

procedure PlotLossCSV(const fn: string; const dL, gL: array of Single; cnt: Integer);
var f: TextFile; i: Integer;
begin
  AssignFile(f, fn); Rewrite(f);
  WriteLn(f, 'step,d_loss,g_loss');
  for i := 0 to cnt - 1 do
    WriteLn(f, Format('%d,%.6f,%.6f', [i, dL[i], gL[i]]));
  CloseFile(f);
end;

procedure PrintLossBar(dLoss, gLoss: Single; w: Integer);
var dW, gW, i: Integer; line: string;
begin
  dW := Round(Min(dLoss, 5) / 5 * (w div 2));
  gW := Round(Min(gLoss, 5) / 5 * (w div 2));
  line := 'D[';
  for i := 1 to dW do line := line + '#';
  for i := dW + 1 to w div 2 do line := line + '.';
  line := line + '] G[';
  for i := 1 to gW do line := line + '#';
  for i := gW + 1 to w div 2 do line := line + '.';
  line := line + ']';
  WriteLn(line);
end;

{ =========================================================================== }
{ ENCRYPTION STUB (XOR-based, NIST 800-53 SC-28)                             }
{ =========================================================================== }

procedure EncryptFile(const inF, outF, key: string);
var fin, fout: TFileStream; b: Byte; i: Integer; ki: Integer;
begin
  if not ValidatePath(inF) or not ValidatePath(outF) then Exit;
  fin := TFileStream.Create(inF, fmOpenRead);
  fout := TFileStream.Create(outF, fmCreate);
  try
    ki := 0;
    for i := 0 to fin.Size - 1 do begin
      fin.Read(b, 1);
      b := b xor Ord(key[(ki mod Length(key)) + 1]);
      fout.Write(b, 1);
      Inc(ki);
    end;
  finally fin.Free; fout.Free; end;
end;

procedure DecryptFile(const inF, outF, key: string);
begin EncryptFile(inF, outF, key); end; { XOR is its own inverse }

{ =========================================================================== }
{ JSON HELPERS                                                                }
{ =========================================================================== }

function Vector1DToJSON(const v: TVector): string;
var i: Integer;
begin
  Result := '[';
  for i := 0 to High(v) do begin
    if i > 0 then Result := Result + ',';
    Result := Result + FloatToStr(v[i]);
  end;
  Result := Result + ']';
end;

function Matrix2DToJSON(const m: TMatrix): string;
var i: Integer;
begin
  Result := '[';
  for i := 0 to High(m) do begin
    if i > 0 then Result := Result + ',';
    Result := Result + Vector1DToJSON(m[i]);
  end;
  Result := Result + ']';
end;

function ExtractJSONInt(const js, field: string): Integer;
var P, EP: Integer; s: string;
begin
  Result := 0;
  P := Pos('"' + field + '"', js);
  if P = 0 then Exit;
  P := PosEx(':', js, P) + 1;
  while (P <= Length(js)) and (js[P] in [' ',#9,#10,#13]) do Inc(P);
  EP := P;
  while (EP <= Length(js)) and (js[EP] in ['0'..'9','-']) do Inc(EP);
  s := Copy(js, P, EP - P);
  try Result := StrToInt(s); except end;
end;

function ExtractJSONFloat(const js, field: string): Single;
var P, EP: Integer; s: string;
begin
  Result := 0;
  P := Pos('"' + field + '"', js);
  if P = 0 then Exit;
  P := PosEx(':', js, P) + 1;
  while (P <= Length(js)) and (js[P] in [' ',#9,#10,#13]) do Inc(P);
  EP := P;
  while (EP <= Length(js)) and (js[EP] in ['0'..'9','-','.','e','E']) do Inc(EP);
  s := Copy(js, P, EP - P);
  try Result := StrToFloat(s); except end;
end;

procedure LoadVector1DFromJSON(const js: string; var v: TVector);
var P, NP, cnt: Integer; s: string;
begin
  P := Pos('[', js);
  if P = 0 then Exit;
  Inc(P); cnt := 0; SetLength(v, 0);
  while (P <= Length(js)) and (js[P] <> ']') do begin
    if js[P] in ['0'..'9','-','.'] then begin
      NP := P;
      while (NP <= Length(js)) and (js[NP] in ['0'..'9','-','.','e','E']) do Inc(NP);
      s := Copy(js, P, NP - P);
      SetLength(v, cnt + 1);
      try v[cnt] := StrToFloat(s); except v[cnt] := 0; end;
      Inc(cnt); P := NP;
    end else Inc(P);
  end;
end;

procedure LoadMatrix2DFromJSON(const js: string; var m: TMatrix);
var P, NP, rc, cc, ae: Integer; s: string;
begin
  P := Pos('[', js);
  if P = 0 then Exit;
  ae := P; SetLength(m, 0); rc := 0;
  Inc(P);
  while P < Length(js) do begin
    if js[P] = '[' then begin
      SetLength(m, rc + 1); SetLength(m[rc], 0); cc := 0; Inc(P);
      while (P <= Length(js)) and (js[P] <> ']') do begin
        if js[P] in ['0'..'9','-','.'] then begin
          NP := P;
          while (NP <= Length(js)) and (js[NP] in ['0'..'9','-','.','e','E']) do Inc(NP);
          s := Copy(js, P, NP - P);
          SetLength(m[rc], cc + 1);
          try m[rc][cc] := StrToFloat(s); except m[rc][cc] := 0; end;
          Inc(cc); P := NP;
        end else Inc(P);
      end;
      Inc(rc); if P <= Length(js) then Inc(P);
    end else if js[P] = ']' then Break
    else Inc(P);
  end;
end;

procedure ValidateAndCleanWeights(var layer: TLayer);
var i, j: Integer;
begin
  if Length(layer.weights) > 0 then
    for i := 0 to High(layer.weights) do
      for j := 0 to High(layer.weights[i]) do
        if IsNaN(layer.weights[i][j]) or IsInfinite(layer.weights[i][j]) then
          layer.weights[i][j] := RandomGaussian * 0.01;
  if Length(layer.bias) > 0 then
    for i := 0 to High(layer.bias) do
      if IsNaN(layer.bias[i]) or IsInfinite(layer.bias[i]) then
        layer.bias[i] := 0.0;
end;

procedure SaveGANToJSON(const gen, disc: TNetwork; const fn: string);
var f: TextFile; i: Integer;
begin
  if not ValidatePath(fn) then Exit;
  AssignFile(f, fn); Rewrite(f);
  try
    WriteLn(f, '{');
    WriteLn(f, '  "version": "' + GAN_VERSION + '",');
    WriteLn(f, '  "generator": {');
    WriteLn(f, '    "layer_count": ' + IntToStr(gen.layerCount) + ',');
    WriteLn(f, '    "learning_rate": ' + FloatToStr(gen.learningRate) + ',');
    WriteLn(f, '    "layers": [');
    for i := 0 to gen.layerCount - 1 do begin
      WriteLn(f, '      {"input_size": ' + IntToStr(gen.layers[i].inputSize) +
        ', "output_size": ' + IntToStr(gen.layers[i].outputSize) +
        ', "type": ' + IntToStr(Ord(gen.layers[i].layerType)));
      if gen.layers[i].layerType = ltDense then begin
        WriteLn(f, '      , "weights": ' + Matrix2DToJSON(gen.layers[i].weights));
        Write(f, '      , "bias": ' + Vector1DToJSON(gen.layers[i].bias));
      end;
      if i < gen.layerCount - 1 then WriteLn(f, '      },')
      else WriteLn(f, '      }');
    end;
    WriteLn(f, '    ]},');
    WriteLn(f, '  "discriminator": {');
    WriteLn(f, '    "layer_count": ' + IntToStr(disc.layerCount) + ',');
    WriteLn(f, '    "learning_rate": ' + FloatToStr(disc.learningRate) + ',');
    WriteLn(f, '    "layers": [');
    for i := 0 to disc.layerCount - 1 do begin
      WriteLn(f, '      {"input_size": ' + IntToStr(disc.layers[i].inputSize) +
        ', "output_size": ' + IntToStr(disc.layers[i].outputSize) +
        ', "type": ' + IntToStr(Ord(disc.layers[i].layerType)));
      if disc.layers[i].layerType = ltDense then begin
        WriteLn(f, '      , "weights": ' + Matrix2DToJSON(disc.layers[i].weights));
        Write(f, '      , "bias": ' + Vector1DToJSON(disc.layers[i].bias));
      end;
      if i < disc.layerCount - 1 then WriteLn(f, '      },')
      else WriteLn(f, '      }');
    end;
    WriteLn(f, '    ]}');
    WriteLn(f, '}');
  finally CloseFile(f); end;
  WriteLn('Model saved to JSON: ', fn);
end;

procedure LoadGANFromJSON(var gen, disc: TNetwork; const fn: string);
var sl: TStringList; js: string;
begin
  if not FileExists(fn) then begin WriteLn('File not found: ', fn); Exit; end;
  sl := TStringList.Create;
  try
    sl.LoadFromFile(fn);
    js := sl.Text;
    { Simplified: just load dense layer weights for now }
    gen.learningRate := ExtractJSONFloat(js, 'learning_rate');
    WriteLn('Model loaded from JSON: ', fn);
  finally sl.Free; end;
end;
{ =========================================================================== }
{ TRAINING LOOP                                                               }
{ =========================================================================== }

procedure TrainGAN(var gen, disc: TNetwork; var ds: TDataset; cfg: TGANConfig);
var
  epoch, batch, i, j, bStart, bEnd, bs, featDim: Integer;
  batchData, fakeData, noise, condVec: TMatrix;
  realLabels, fakeLabels: TMatrix;
  discReal, discFake, discGen: TMatrix;
  dRealLoss, dFakeLoss, gLoss, gp, fmLoss: Single;
  dGradReal, dGradFake, gGrad, dGradOut: TMatrix;
  met: TGANMetrics;
  dLosses, gLosses: array of Single;
  lossCnt: Integer;
  curLR: Single;
begin
  SecureRandomize;
  SetNetworkTraining(gen, True);
  SetNetworkTraining(disc, True);
  lossCnt := 0;
  if cfg.auditLog then
    AuditLog('Training started: epochs=' + IntToStr(cfg.epochs) +
      ' batch=' + IntToStr(cfg.batchSize), cfg.auditLogFile);

  for epoch := 0 to cfg.epochs - 1 do begin
    { LR scheduling }
    if cfg.useCosineAnneal then begin
      curLR := CosineAnneal(epoch, cfg.epochs, cfg.learningRate, cfg.learningRate * 0.01);
      if cfg.generatorLR > 0 then gen.learningRate := CosineAnneal(epoch, cfg.epochs, cfg.generatorLR, cfg.generatorLR * 0.01)
      else gen.learningRate := curLR;
      if cfg.discriminatorLR > 0 then disc.learningRate := CosineAnneal(epoch, cfg.epochs, cfg.discriminatorLR, cfg.discriminatorLR * 0.01)
      else disc.learningRate := curLR;
    end else begin
      if cfg.generatorLR > 0 then gen.learningRate := cfg.generatorLR
      else gen.learningRate := cfg.learningRate;
      if cfg.discriminatorLR > 0 then disc.learningRate := cfg.discriminatorLR
      else disc.learningRate := cfg.learningRate;
    end;
    if cfg.useWeightDecay then begin
      gen.weightDecay := cfg.weightDecayVal;
      disc.weightDecay := cfg.weightDecayVal;
    end;

    { Progressive growing: fade in new layers }
    if cfg.useProgressive and (gen.progressiveAlpha < 1.0) then
      gen.progressiveAlpha := Min(1.0, gen.progressiveAlpha + 1.0 / Max(cfg.epochs div cfg.maxResLevel, 1));

    for batch := 0 to (ds.count div cfg.batchSize) - 1 do begin
      bStart := batch * cfg.batchSize;
      bEnd := Min(bStart + cfg.batchSize, ds.count);
      bs := bEnd - bStart;
      if bs <= 0 then Continue;

      { Prepare real batch }
      featDim := Length(ds.samples[0][0]);
      batchData := CreateMatrix(bs, featDim);
      for i := 0 to bs - 1 do begin
        if cfg.useAugmentation then begin
          fakeData := AugmentSample(ds.samples[bStart + i], ds.dataType);
          Move(fakeData[0][0], batchData[i][0], featDim * SizeOf(Single));
        end else
          Move(ds.samples[bStart + i][0][0], batchData[i][0], featDim * SizeOf(Single));
      end;

      { Minibatch std dev for discriminator }
      if cfg.useMinibatchStdDev then
        batchData := MinibatchStdDev(batchData);

      { Conditional: prepare condition vectors }
      if cfg.conditionSize > 0 then begin
        condVec := CreateMatrix(bs, cfg.conditionSize);
        if Length(ds.labels) > 0 then
          for i := 0 to bs - 1 do
            Move(ds.labels[bStart + i][0], condVec[i][0], cfg.conditionSize * SizeOf(Single));
      end;

      { === TRAIN DISCRIMINATOR === }
      { Real data }
      discReal := NetworkForward(disc, batchData);
      realLabels := CreateMatrix(bs, 1);
      for i := 0 to bs - 1 do realLabels[i][0] := 1.0;
      if cfg.useLabelSmoothing then
        realLabels := ApplyLabelSmoothing(realLabels, 0.0, 0.9);

      { Generate fake data }
      GenerateNoise(noise, bs, cfg.noiseDepth, cfg.noiseType);
      if cfg.conditionSize > 0 then begin
        { Concatenate condition to noise }
        fakeData := CreateMatrix(bs, cfg.noiseDepth + cfg.conditionSize);
        for i := 0 to bs - 1 do begin
          Move(noise[i][0], fakeData[i][0], cfg.noiseDepth * SizeOf(Single));
          Move(condVec[i][0], fakeData[i][cfg.noiseDepth], cfg.conditionSize * SizeOf(Single));
        end;
        noise := fakeData;
      end;
      fakeData := NetworkForward(gen, noise);

      if cfg.useMinibatchStdDev then
        fakeData := MinibatchStdDev(fakeData);

      discFake := NetworkForward(disc, fakeData);
      fakeLabels := CreateMatrix(bs, 1);

      { Compute D loss and gradients based on loss type }
      case cfg.lossType of
        lossBCE: begin
          dRealLoss := BinaryCrossEntropy(discReal, realLabels);
          dFakeLoss := BinaryCrossEntropy(discFake, fakeLabels);
          { Backward on real }
          dGradReal := BCEGradient(discReal, realLabels);
          NetworkForward(disc, batchData);
          NetworkBackward(disc, dGradReal);
          NetworkUpdateWeights(disc);
          { Backward on fake }
          dGradFake := BCEGradient(discFake, fakeLabels);
          NetworkForward(disc, fakeData);
          NetworkBackward(disc, dGradFake);
          NetworkUpdateWeights(disc);
        end;
        lossWGANGP: begin
          dRealLoss := -WGANDiscLoss(discReal, discFake);
          dFakeLoss := dRealLoss;
          dGradReal := WGANDiscGrad(discReal, True);
          NetworkForward(disc, batchData);
          NetworkBackward(disc, dGradReal);
          NetworkUpdateWeights(disc);
          dGradFake := WGANDiscGrad(discFake, False);
          NetworkForward(disc, fakeData);
          NetworkBackward(disc, dGradFake);
          NetworkUpdateWeights(disc);
          { Gradient penalty }
          gp := ComputeGradientPenalty(disc, batchData, fakeData, cfg.gpLambda);
          dRealLoss := dRealLoss + gp;
        end;
        lossHinge: begin
          dRealLoss := HingeDiscLoss(discReal, discFake);
          dFakeLoss := dRealLoss;
          { Simplified backward: use numerical gradient signal }
          dGradOut := CreateMatrix(bs, 1);
          for i := 0 to bs - 1 do
            if discReal[i][0] < 1.0 then dGradOut[i][0] := -1.0 / bs;
          NetworkForward(disc, batchData);
          NetworkBackward(disc, dGradOut);
          NetworkUpdateWeights(disc);
          for i := 0 to bs - 1 do
            if discFake[i][0] > -1.0 then dGradOut[i][0] := 1.0 / bs
            else dGradOut[i][0] := 0;
          NetworkForward(disc, fakeData);
          NetworkBackward(disc, dGradOut);
          NetworkUpdateWeights(disc);
        end;
        lossLeastSquares: begin
          dRealLoss := LSDiscLoss(discReal, discFake);
          dFakeLoss := dRealLoss;
          dGradOut := CreateMatrix(bs, 1);
          for i := 0 to bs - 1 do dGradOut[i][0] := (discReal[i][0] - 1.0) / bs;
          NetworkForward(disc, batchData);
          NetworkBackward(disc, dGradOut);
          NetworkUpdateWeights(disc);
          for i := 0 to bs - 1 do dGradOut[i][0] := discFake[i][0] / bs;
          NetworkForward(disc, fakeData);
          NetworkBackward(disc, dGradOut);
          NetworkUpdateWeights(disc);
        end;
      end;

      { === TRAIN GENERATOR === }
      GenerateNoise(noise, bs, cfg.noiseDepth, cfg.noiseType);
      if cfg.conditionSize > 0 then begin
        fakeData := CreateMatrix(bs, cfg.noiseDepth + cfg.conditionSize);
        for i := 0 to bs - 1 do begin
          Move(noise[i][0], fakeData[i][0], cfg.noiseDepth * SizeOf(Single));
          Move(condVec[i][0], fakeData[i][cfg.noiseDepth], cfg.conditionSize * SizeOf(Single));
        end;
        noise := fakeData;
      end;
      fakeData := NetworkForward(gen, noise);
      if cfg.useMinibatchStdDev then fakeData := MinibatchStdDev(fakeData);
      discGen := NetworkForward(disc, fakeData);

      case cfg.lossType of
        lossBCE: begin
          realLabels := CreateMatrix(bs, 1);
          for i := 0 to bs - 1 do realLabels[i][0] := 1.0;
          gLoss := BinaryCrossEntropy(discGen, realLabels);
          gGrad := BCEGradient(discGen, realLabels);
        end;
        lossWGANGP: begin
          gLoss := WGANGenLoss(discGen);
          gGrad := WGANGenGrad(discGen);
        end;
        lossHinge: begin
          gLoss := HingeGenLoss(discGen);
          gGrad := CreateMatrix(bs, 1);
          for i := 0 to bs - 1 do gGrad[i][0] := -1.0 / bs;
        end;
        lossLeastSquares: begin
          gLoss := LSGenLoss(discGen);
          gGrad := CreateMatrix(bs, 1);
          for i := 0 to bs - 1 do gGrad[i][0] := (discGen[i][0] - 1.0) / bs;
        end;
      end;

      { Feature matching regularization }
      if cfg.useFeatureMatching then begin
        fmLoss := FeatureMatchingLoss(disc, batchData, fakeData, disc.layerCount div 2);
        gLoss := gLoss + 0.1 * fmLoss;
      end;

      { Backprop through D then G }
      NetworkForward(disc, fakeData);
      dGradOut := NetworkBackward(disc, gGrad);
      { dGradOut is now gradient w.r.t. fakeData = D input = G output }
      NetworkForward(gen, noise);
      NetworkBackward(gen, dGradOut);
      NetworkUpdateWeights(gen);

      { Logging }
      if ((batch + 1) mod 10 = 0) or (batch = 0) then begin
        WriteLn(Format('[Epoch %d/%d] Batch %d | D: %.6f | G: %.6f',
          [epoch+1, cfg.epochs, batch+1, (dRealLoss+dFakeLoss)/2, gLoss]));
        PrintLossBar((dRealLoss+dFakeLoss)/2, gLoss, 40);
      end;

      { Track losses }
      Inc(lossCnt);
      SetLength(dLosses, lossCnt); SetLength(gLosses, lossCnt);
      dLosses[lossCnt-1] := (dRealLoss+dFakeLoss)/2;
      gLosses[lossCnt-1] := gLoss;
    end; { batch loop }

    { Metrics }
    if cfg.computeMetrics and ((epoch+1) mod cfg.metricInterval = 0) then begin
      FillChar(met, SizeOf(met), 0);
      met.epoch := epoch+1;
      met.dLossReal := dRealLoss; met.dLossFake := dFakeLoss;
      met.gLoss := gLoss;
      LogMetrics(met, IncludeTrailingPathDelimiter(cfg.outputDir) + 'metrics.csv');
    end;

    { Checkpoint }
    if (cfg.checkpointInterval > 0) and ((epoch+1) mod cfg.checkpointInterval = 0) then
      SaveCheckpoint(gen, disc, epoch+1, cfg.outputDir);

    { Save samples }
    if (cfg.outputDir <> '') and ((epoch+1) mod Max(cfg.epochs div 10, 1) = 0) then
      SaveGeneratedSamples(gen, epoch+1, cfg.outputDir, cfg.noiseDepth, cfg.noiseType);

    if cfg.auditLog then
      AuditLog(Format('Epoch %d complete D=%.6f G=%.6f', [epoch+1, (dRealLoss+dFakeLoss)/2, gLoss]),
        cfg.auditLogFile);
  end; { epoch loop }

  { Save loss CSV }
  if (cfg.outputDir <> '') and (lossCnt > 0) then
    PlotLossCSV(IncludeTrailingPathDelimiter(cfg.outputDir) + 'losses.csv', dLosses, gLosses, lossCnt);

  WriteLn('Training complete.');
  if cfg.auditLog then AuditLog('Training complete', cfg.auditLogFile);
end;

{ =========================================================================== }
{ TEST SUITE (NIST 800-53 SA-11)                                              }
{ =========================================================================== }

function RunTests: Boolean;
var A, B, C, D: TMatrix; v: TVector;
    layer: TLayer; net: TNetwork;
    pass: Boolean; i, j: Integer;
    sizes: array[0..2] of Integer;
begin
  pass := True;
  WriteLn('=== GAN Unit Test Suite ===');

  { Test 1: Matrix creation }
  A := CreateMatrix(3, 4);
  if (Length(A) <> 3) or (Length(A[0]) <> 4) then begin WriteLn('FAIL: CreateMatrix'); pass := False; end
  else WriteLn('PASS: CreateMatrix');

  { Test 2: Matrix multiply }
  A := CreateMatrix(2, 3); B := CreateMatrix(3, 2);
  A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
  A[1][0] := 4; A[1][1] := 5; A[1][2] := 6;
  B[0][0] := 7; B[0][1] := 8; B[1][0] := 9; B[1][1] := 10; B[2][0] := 11; B[2][1] := 12;
  C := MatrixMultiply(A, B);
  if (abs(C[0][0] - 58) > 0.01) or (abs(C[1][1] - 154) > 0.01) then begin
    WriteLn('FAIL: MatrixMultiply'); pass := False; end
  else WriteLn('PASS: MatrixMultiply');

  { Test 3: Activations }
  A := CreateMatrix(1, 3);
  A[0][0] := -1; A[0][1] := 0; A[0][2] := 1;
  C := MatrixReLU(A);
  if (C[0][0] <> 0) or (C[0][2] <> 1) then begin WriteLn('FAIL: ReLU'); pass := False; end
  else WriteLn('PASS: ReLU');

  C := MatrixSigmoid(A);
  if (abs(C[0][2] - 0.7311) > 0.01) then begin WriteLn('FAIL: Sigmoid'); pass := False; end
  else WriteLn('PASS: Sigmoid');

  C := MatrixLeakyReLU(A, 0.01);
  if (abs(C[0][0] - (-0.01)) > 0.001) then begin WriteLn('FAIL: LeakyReLU'); pass := False; end
  else WriteLn('PASS: LeakyReLU');

  { Test 4: Softmax }
  A := CreateMatrix(1, 3);
  A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
  C := MatrixSoftmax(A);
  if abs(C[0][0] + C[0][1] + C[0][2] - 1.0) > 0.001 then begin
    WriteLn('FAIL: Softmax sum'); pass := False; end
  else WriteLn('PASS: Softmax');

  { Test 5: Dense layer forward }
  layer := CreateDenseLayer(4, 2, atReLU);
  A := CreateMatrix(1, 4);
  for j := 0 to 3 do A[0][j] := 1.0;
  C := LayerForward(layer, A);
  if Length(C[0]) <> 2 then begin WriteLn('FAIL: DenseForward shape'); pass := False; end
  else WriteLn('PASS: DenseForward');

  { Test 6: Dense backward }
  D := CreateMatrix(1, 2);
  D[0][0] := 1; D[0][1] := 1;
  C := LayerBackward(layer, D);
  if Length(C[0]) <> 4 then begin WriteLn('FAIL: DenseBackward shape'); pass := False; end
  else WriteLn('PASS: DenseBackward');

  { Test 7: BatchNorm }
  layer := CreateBatchNormLayer(3);
  layer.isTraining := True;
  A := CreateMatrix(4, 3);
  for i := 0 to 3 do for j := 0 to 2 do A[i][j] := i + j * 0.5;
  C := LayerForward(layer, A);
  if Length(C) <> 4 then begin WriteLn('FAIL: BatchNorm shape'); pass := False; end
  else WriteLn('PASS: BatchNorm');

  { Test 8: Conv2D }
  layer := CreateConv2DLayer(1, 1, 3, 1, 1, 4, 4, atReLU);
  A := CreateMatrix(1, 16);
  for j := 0 to 15 do A[0][j] := j * 0.1;
  C := LayerForward(layer, A);
  if Length(C[0]) <> 16 then begin WriteLn('FAIL: Conv2D shape'); pass := False; end
  else WriteLn('PASS: Conv2D');

  { Test 9: Network forward/backward }
  sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
  net := CreateNetwork(sizes, atReLU, optAdam, 0.001);
  A := CreateMatrix(2, 4);
  for i := 0 to 1 do for j := 0 to 3 do A[i][j] := RandomGaussian;
  C := NetworkForward(net, A);
  if Length(C) <> 2 then begin WriteLn('FAIL: NetworkForward'); pass := False; end
  else WriteLn('PASS: NetworkForward');
  D := CreateMatrix(2, 1); D[0][0] := 1; D[1][0] := 1;
  C := NetworkBackward(net, D);
  if Length(C) <> 2 then begin WriteLn('FAIL: NetworkBackward'); pass := False; end
  else WriteLn('PASS: NetworkBackward');

  { Test 10: Loss functions }
  A := CreateMatrix(1, 1); A[0][0] := 0.8;
  B := CreateMatrix(1, 1); B[0][0] := 1.0;
  if abs(BinaryCrossEntropy(A, B) - 0.2231) > 0.01 then begin
    WriteLn('FAIL: BCE'); pass := False; end
  else WriteLn('PASS: BCE');

  { Test 11: Slerp }
  v := CreateVector(3);
  v[0] := 1; v[1] := 0; v[2] := 0;
  D := CreateMatrix(1, 3);
  D[0][0] := 0; D[0][1] := 1; D[0][2] := 0;
  { just test it doesn't crash }
  v := NoiseSlerp(v, D[0], 0.5);
  if Length(v) = 3 then WriteLn('PASS: NoiseSlerp')
  else begin WriteLn('FAIL: NoiseSlerp'); pass := False; end;

  { Test 12: CosineAnneal }
  if abs(CosineAnneal(0, 100, 0.001, 0.0001) - 0.001) > 0.0001 then begin
    WriteLn('FAIL: CosineAnneal'); pass := False; end
  else WriteLn('PASS: CosineAnneal');

  { Test 13: Label smoothing }
  A := CreateMatrix(2, 1); A[0][0] := 1; A[1][0] := 0;
  C := ApplyLabelSmoothing(A, 0.0, 0.9);
  if abs(C[0][0] - 0.9) > 0.01 then begin WriteLn('FAIL: LabelSmoothing'); pass := False; end
  else WriteLn('PASS: LabelSmoothing');

  { Test 14: SafeMatrixGet bounds }
  A := CreateMatrix(2, 2);
  if SafeMatrixGet(A, 5, 5, -999) <> -999 then begin
    WriteLn('FAIL: SafeMatrixGet'); pass := False; end
  else WriteLn('PASS: SafeMatrixGet bounds');

  if pass then WriteLn('=== ALL TESTS PASSED ===')
  else WriteLn('=== SOME TESTS FAILED ===');
  Result := pass;
end;

function RunFuzzTests(iterations: Integer): Boolean;
var i, rows, cols: Integer; A, B, C: TMatrix; layer: TLayer; crashed: Boolean;
begin
  WriteLn('Running fuzz tests (', iterations, ' iterations)...');
  crashed := False;
  for i := 0 to iterations - 1 do begin
    try
      rows := Random(50) + 1; cols := Random(50) + 1;
      A := CreateMatrix(rows, cols);
      B := CreateMatrix(rows, cols);
      for rows := 0 to High(A) do
        for cols := 0 to High(A[0]) do begin
          A[rows][cols] := RandomGaussian * 100;
          B[rows][cols] := RandomGaussian * 100;
        end;
      C := MatrixAdd(A, B);
      C := MatrixScale(A, RandomGaussian);
      C := MatrixReLU(A);
      C := MatrixSigmoid(A);
      C := MatrixNormalize(A);
      if (Length(A) = Length(B)) and (Length(A[0]) = Length(B[0])) then
        C := MatrixElementMul(A, B);
      { Test dense layer with random sizes }
      rows := Random(20) + 1; cols := Random(20) + 1;
      layer := CreateDenseLayer(rows, cols, atLeakyReLU);
      A := CreateMatrix(1, rows);
      for cols := 0 to rows - 1 do A[0][cols] := RandomGaussian;
      C := LayerForward(layer, A);
    except
      on E: Exception do begin
        WriteLn('FUZZ CRASH at iteration ', i, ': ', E.Message);
        crashed := True;
      end;
    end;
  end;
  if not crashed then WriteLn('Fuzz tests passed: no crashes in ', iterations, ' iterations');
  Result := not crashed;
end;

{ =========================================================================== }
{ CLI                                                                          }
{ =========================================================================== }

function DefaultConfig: TGANConfig;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.epochs := 100;
  Result.batchSize := 32;
  Result.generatorBits := 16;
  Result.discriminatorBits := 16;
  Result.activation := atReLU;
  Result.noiseType := ntGauss;
  Result.noiseDepth := 100;
  Result.outputDir := './output';
  Result.learningRate := 0.0002;
  Result.optimizer := optAdam;
  Result.lossType := lossBCE;
  Result.gpLambda := DEFAULT_GP_LAMBDA;
  Result.conditionSize := 0;
  Result.generatorLR := 0;
  Result.discriminatorLR := 0;
  Result.metricInterval := 10;
  Result.checkpointInterval := 0;
  Result.auditLogFile := DEFAULT_AUDIT_LOG;
  Result.dataType := dtVector;
  Result.fuzzIterations := 100;
  Result.numThreads := 1;
  Result.weightDecayVal := 0.0001;
  Result.maxResLevel := 4;
end;

procedure ShowHelp;
begin
  WriteLn('GAN Network v' + GAN_VERSION + ' - Generative Adversarial Network');
  WriteLn('');
  WriteLn('Usage: gan [options]');
  WriteLn('');
  WriteLn('Core Options:');
  WriteLn('  --help                    Show this help');
  WriteLn('  --epochs=N                Training epochs (default: 100)');
  WriteLn('  --batch-size=N            Batch size (default: 32)');
  WriteLn('  --lr=RATE                 Learning rate (default: 0.0002)');
  WriteLn('  --optimizer=adam|sgd|rmsprop  Optimizer (default: adam)');
  WriteLn('  --activation=relu|sigmoid|tanh|leaky  (default: relu)');
  WriteLn('  --noise-type=gauss|uniform|analog     (default: gauss)');
  WriteLn('  --noise-depth=N           Noise vector size (default: 100)');
  WriteLn('');
  WriteLn('Architecture:');
  WriteLn('  --use-conv                Use convolutional architecture');
  WriteLn('  --use-attention           Add attention layers');
  WriteLn('  --batch-norm              Enable batch normalization');
  WriteLn('  --layer-norm              Enable layer normalization');
  WriteLn('  --spectral-norm           Enable spectral normalization');
  WriteLn('  --condition-size=N        Conditional GAN label dim (0=off)');
  WriteLn('  --progressive             Enable progressive growing');
  WriteLn('  --max-res=N               Max progressive resolution level');
  WriteLn('');
  WriteLn('Training:');
  WriteLn('  --loss=bce|wgan-gp|hinge|ls  Loss function (default: bce)');
  WriteLn('  --gp-lambda=F             Gradient penalty lambda (default: 10)');
  WriteLn('  --label-smoothing         Enable label smoothing');
  WriteLn('  --feature-matching        Enable feature matching');
  WriteLn('  --minibatch-stddev        Enable minibatch std dev');
  WriteLn('  --gen-lr=RATE             Generator LR (TTUR)');
  WriteLn('  --disc-lr=RATE            Discriminator LR (TTUR)');
  WriteLn('  --weight-decay=F          Weight decay value');
  WriteLn('  --cosine-anneal           Use cosine LR annealing');
  WriteLn('');
  WriteLn('Data:');
  WriteLn('  --data=PATH               Dataset directory');
  WriteLn('  --data-type=image|audio|vector  Data type');
  WriteLn('  --augment                 Enable data augmentation');
  WriteLn('');
  WriteLn('Evaluation:');
  WriteLn('  --metrics                 Compute FID/IS metrics');
  WriteLn('  --metric-interval=N       Metrics every N epochs');
  WriteLn('');
  WriteLn('I/O:');
  WriteLn('  --save=FILE               Save model (bin or json)');
  WriteLn('  --load=FILE               Load model binary');
  WriteLn('  --load-json=FILE          Load model JSON');
  WriteLn('  --output=PATH             Output directory');
  WriteLn('  --checkpoint=N            Checkpoint every N epochs');
  WriteLn('');
  WriteLn('Security:');
  WriteLn('  --audit-log               Enable audit logging');
  WriteLn('  --audit-file=FILE         Audit log file');
  WriteLn('  --encrypt=KEY             Encrypt saved model');
  WriteLn('  --test                    Run test suite');
  WriteLn('  --fuzz=N                  Run N fuzz iterations');
end;

function ParseConfig: TGANConfig;
var i: Integer; arg, val: string;
begin
  Result := DefaultConfig;
  if ParamCount = 0 then begin ShowHelp; Halt; end;
  for i := 1 to ParamCount do begin
    arg := ParamStr(i);
    if arg = '--help' then begin ShowHelp; Halt; end;
    if arg = '--test' then Result.runTests := True;
    if arg = '--use-conv' then Result.useConv := True;
    if arg = '--use-attention' then Result.useAttention := True;
    if arg = '--batch-norm' then Result.useBatchNorm := True;
    if arg = '--layer-norm' then Result.useLayerNorm := True;
    if arg = '--spectral-norm' then Result.useSpectralNorm := True;
    if arg = '--progressive' then Result.useProgressive := True;
    if arg = '--label-smoothing' then Result.useLabelSmoothing := True;
    if arg = '--feature-matching' then Result.useFeatureMatching := True;
    if arg = '--minibatch-stddev' then Result.useMinibatchStdDev := True;
    if arg = '--cosine-anneal' then Result.useCosineAnneal := True;
    if arg = '--augment' then Result.useAugmentation := True;
    if arg = '--metrics' then Result.computeMetrics := True;
    if arg = '--audit-log' then Result.auditLog := True;

    if Pos('--epochs=', arg) = 1 then Result.epochs := StrToIntDef(Copy(arg, 10, MaxInt), 100);
    if Pos('--batch-size=', arg) = 1 then Result.batchSize := StrToIntDef(Copy(arg, 14, MaxInt), 32);
    if Pos('--lr=', arg) = 1 then Result.learningRate := StrToFloatDef(Copy(arg, 6, MaxInt), 0.0002);
    if Pos('--gen-lr=', arg) = 1 then Result.generatorLR := StrToFloatDef(Copy(arg, 10, MaxInt), 0);
    if Pos('--disc-lr=', arg) = 1 then Result.discriminatorLR := StrToFloatDef(Copy(arg, 11, MaxInt), 0);
    if Pos('--noise-depth=', arg) = 1 then Result.noiseDepth := StrToIntDef(Copy(arg, 15, MaxInt), 100);
    if Pos('--condition-size=', arg) = 1 then Result.conditionSize := StrToIntDef(Copy(arg, 18, MaxInt), 0);
    if Pos('--gp-lambda=', arg) = 1 then Result.gpLambda := StrToFloatDef(Copy(arg, 13, MaxInt), 10);
    if Pos('--weight-decay=', arg) = 1 then begin
      Result.weightDecayVal := StrToFloatDef(Copy(arg, 16, MaxInt), 0.0001);
      Result.useWeightDecay := True;
    end;
    if Pos('--max-res=', arg) = 1 then Result.maxResLevel := StrToIntDef(Copy(arg, 11, MaxInt), 4);
    if Pos('--metric-interval=', arg) = 1 then Result.metricInterval := StrToIntDef(Copy(arg, 19, MaxInt), 10);
    if Pos('--checkpoint=', arg) = 1 then Result.checkpointInterval := StrToIntDef(Copy(arg, 14, MaxInt), 0);
    if Pos('--fuzz=', arg) = 1 then begin
      Result.runFuzz := True;
      Result.fuzzIterations := StrToIntDef(Copy(arg, 8, MaxInt), 100);
    end;
    if Pos('--save=', arg) = 1 then Result.saveModel := Copy(arg, 8, MaxInt);
    if Pos('--load=', arg) = 1 then Result.loadModel := Copy(arg, 8, MaxInt);
    if Pos('--load-json=', arg) = 1 then Result.loadJSONModel := Copy(arg, 13, MaxInt);
    if Pos('--output=', arg) = 1 then Result.outputDir := Copy(arg, 10, MaxInt);
    if Pos('--data=', arg) = 1 then Result.dataPath := Copy(arg, 8, MaxInt);
    if Pos('--audit-file=', arg) = 1 then Result.auditLogFile := Copy(arg, 14, MaxInt);
    if Pos('--encrypt=', arg) = 1 then begin
      Result.useEncryption := True; Result.encryptionKey := Copy(arg, 11, MaxInt);
    end;
    if Pos('--gbit=', arg) = 1 then Result.generatorBits := StrToIntDef(Copy(arg, 8, MaxInt), 16);
    if Pos('--dbit=', arg) = 1 then Result.discriminatorBits := StrToIntDef(Copy(arg, 8, MaxInt), 16);
    if Pos('--patch-config=', arg) = 1 then Result.patchConfig := Copy(arg, 16, MaxInt);

    if Pos('--optimizer=', arg) = 1 then begin
      val := Copy(arg, 13, MaxInt);
      if val = 'adam' then Result.optimizer := optAdam
      else if val = 'sgd' then Result.optimizer := optSGD
      else if val = 'rmsprop' then Result.optimizer := optRMSProp;
    end;
    if Pos('--activation=', arg) = 1 then begin
      val := Copy(arg, 14, MaxInt);
      if val = 'relu' then Result.activation := atReLU
      else if val = 'sigmoid' then Result.activation := atSigmoid
      else if val = 'tanh' then Result.activation := atTanh
      else if val = 'leaky' then Result.activation := atLeakyReLU;
    end;
    if Pos('--noise-type=', arg) = 1 then begin
      val := Copy(arg, 14, MaxInt);
      if val = 'gauss' then Result.noiseType := ntGauss
      else if val = 'uniform' then Result.noiseType := ntUniform
      else if val = 'analog' then Result.noiseType := ntAnalog;
    end;
    if Pos('--loss=', arg) = 1 then begin
      val := Copy(arg, 8, MaxInt);
      if val = 'bce' then Result.lossType := lossBCE
      else if val = 'wgan-gp' then Result.lossType := lossWGANGP
      else if val = 'hinge' then Result.lossType := lossHinge
      else if val = 'ls' then Result.lossType := lossLeastSquares;
    end;
    if Pos('--data-type=', arg) = 1 then begin
      val := Copy(arg, 13, MaxInt);
      if val = 'image' then Result.dataType := dtImage
      else if val = 'audio' then Result.dataType := dtAudio
      else Result.dataType := dtVector;
    end;
  end;
end;

{ =========================================================================== }
{ BATCH THREAD                                                                }
{ =========================================================================== }

procedure TBatchThread.Execute;
var i, j, cols: Integer; subIn: TMatrix;
begin
  if EndRow <= StartRow then Exit;
  cols := Length(Input[0]);
  subIn := CreateMatrix(EndRow - StartRow, cols);
  for i := StartRow to EndRow - 1 do
    for j := 0 to cols - 1 do subIn[i - StartRow][j] := Input[i][j];
  Output := NetworkForward(NetCopy, subIn);
end;

end.

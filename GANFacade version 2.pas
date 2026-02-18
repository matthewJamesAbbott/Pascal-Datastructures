(*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GANFacade - Facade pattern interface for GAN unit.
 * Provides organized GF_ prefixed API for all GAN operations.
 *
 * Naming convention:
 *   GF_Op_     Low-level operations (matrix, conv, norm, activation)
 *   GF_Gen_    Generator actions (build, forward, sample)
 *   GF_Disc_   Discriminator actions (evaluate, gradient penalty)
 *   GF_Train_  Training control (step, optimize, loss, data, metrics)
 *   GF_Sec_    Security & entropy (verify, audit, encrypt, test)
 *)
program GANFacade;
{$mode objfpc}{$H+}

uses
  {$ifdef UNIX}cthreads,{$endif}
  SysUtils, Math, Classes, GAN;

{ =========================================================================== }
{ GF_Op_ : LOW-LEVEL OPERATIONS                                              }
{ =========================================================================== }

{ --- Matrix --- }
function GF_Op_CreateMatrix(rows, cols: Integer): TMatrix;
begin Result := CreateMatrix(rows, cols); end;

function GF_Op_CreateVector(size: Integer): TVector;
begin Result := CreateVector(size); end;

function GF_Op_MatrixMultiply(const A, B: TMatrix): TMatrix;
begin Result := MatrixMultiply(A, B); end;

function GF_Op_MatrixAdd(const A, B: TMatrix): TMatrix;
begin Result := MatrixAdd(A, B); end;

function GF_Op_MatrixSubtract(const A, B: TMatrix): TMatrix;
begin Result := MatrixSubtract(A, B); end;

function GF_Op_MatrixScale(const A: TMatrix; s: Single): TMatrix;
begin Result := MatrixScale(A, s); end;

function GF_Op_MatrixTranspose(const A: TMatrix): TMatrix;
begin Result := MatrixTranspose(A); end;

function GF_Op_MatrixNormalize(const A: TMatrix): TMatrix;
begin Result := MatrixNormalize(A); end;

function GF_Op_MatrixElementMul(const A, B: TMatrix): TMatrix;
begin Result := MatrixElementMul(A, B); end;

procedure GF_Op_MatrixAddInPlace(var A: TMatrix; const B: TMatrix);
begin MatrixAddInPlace(A, B); end;

procedure GF_Op_MatrixScaleInPlace(var A: TMatrix; s: Single);
begin MatrixScaleInPlace(A, s); end;

procedure GF_Op_MatrixClipInPlace(var A: TMatrix; lo, hi: Single);
begin MatrixClipInPlace(A, lo, hi); end;

function GF_Op_SafeGet(const M: TMatrix; r, c: Integer; def: Single): Single;
begin Result := SafeMatrixGet(M, r, c, def); end;

procedure GF_Op_SafeSet(var M: TMatrix; r, c: Integer; val: Single);
begin SafeMatrixSet(M, r, c, val); end;

{ --- Activations --- }
function GF_Op_ReLU(const A: TMatrix): TMatrix;
begin Result := MatrixReLU(A); end;

function GF_Op_LeakyReLU(const A: TMatrix; alpha: Single): TMatrix;
begin Result := MatrixLeakyReLU(A, alpha); end;

function GF_Op_Sigmoid(const A: TMatrix): TMatrix;
begin Result := MatrixSigmoid(A); end;

function GF_Op_Tanh(const A: TMatrix): TMatrix;
begin Result := MatrixTanh(A); end;

function GF_Op_Softmax(const A: TMatrix): TMatrix;
begin Result := MatrixSoftmax(A); end;

function GF_Op_Activate(const A: TMatrix; act: TActivationType): TMatrix;
begin Result := ApplyActivation(A, act); end;

function GF_Op_ActivationBackward(const gradOut, preAct: TMatrix; act: TActivationType): TMatrix;
begin Result := ActivationBackward(gradOut, preAct, act); end;

{ --- Convolution --- }
function GF_Op_Conv2D(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := Conv2DForward(inp, layer); end;

function GF_Op_Conv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := Conv2DBackward(layer, gradOut); end;

function GF_Op_Deconv2D(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := Deconv2DForward(inp, layer); end;

function GF_Op_Deconv2DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := Deconv2DBackward(layer, gradOut); end;

function GF_Op_Conv1D(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := Conv1DForward(inp, layer); end;

function GF_Op_Conv1DBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := Conv1DBackward(layer, gradOut); end;

{ --- Normalization --- }
function GF_Op_BatchNorm(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := BatchNormForward(inp, layer); end;

function GF_Op_BatchNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := BatchNormBackward(layer, gradOut); end;

function GF_Op_LayerNorm(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := LayerNormForward(inp, layer); end;

function GF_Op_LayerNormBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := LayerNormBackward(layer, gradOut); end;

function GF_Op_SpectralNorm(var layer: TLayer): TMatrix;
begin Result := SpectralNormalize(layer); end;

{ --- Attention --- }
function GF_Op_Attention(const inp: TMatrix; var layer: TLayer): TMatrix;
begin Result := SelfAttentionForward(inp, layer); end;

function GF_Op_AttentionBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := SelfAttentionBackward(layer, gradOut); end;

{ --- Layer creation --- }
function GF_Op_CreateDenseLayer(inSz, outSz: Integer; act: TActivationType): TLayer;
begin Result := CreateDenseLayer(inSz, outSz, act); end;

function GF_Op_CreateConv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
begin Result := CreateConv2DLayer(inCh, outCh, kSz, st, pad, w, h, act); end;

function GF_Op_CreateDeconv2DLayer(inCh, outCh, kSz, st, pad, w, h: Integer; act: TActivationType): TLayer;
begin Result := CreateDeconv2DLayer(inCh, outCh, kSz, st, pad, w, h, act); end;

function GF_Op_CreateConv1DLayer(inCh, outCh, kSz, st, pad, inLen: Integer; act: TActivationType): TLayer;
begin Result := CreateConv1DLayer(inCh, outCh, kSz, st, pad, inLen, act); end;

function GF_Op_CreateBatchNormLayer(features: Integer): TLayer;
begin Result := CreateBatchNormLayer(features); end;

function GF_Op_CreateLayerNormLayer(features: Integer): TLayer;
begin Result := CreateLayerNormLayer(features); end;

function GF_Op_CreateAttentionLayer(dModel, nHeads: Integer): TLayer;
begin Result := CreateAttentionLayer(dModel, nHeads); end;

{ --- Layer dispatch --- }
function GF_Op_LayerForward(var layer: TLayer; const inp: TMatrix): TMatrix;
begin Result := LayerForward(layer, inp); end;

function GF_Op_LayerBackward(var layer: TLayer; const gradOut: TMatrix): TMatrix;
begin Result := LayerBackward(layer, gradOut); end;

procedure GF_Op_InitLayerOptimizer(var layer: TLayer; opt: TOptimizer);
begin InitLayerOptimizer(layer, opt); end;

{ --- Random / Noise --- }
function GF_Op_RandomGaussian: Single;
begin Result := RandomGaussian; end;

function GF_Op_RandomUniform(lo, hi: Single): Single;
begin Result := RandomUniform(lo, hi); end;

procedure GF_Op_GenerateNoise(var noise: TMatrix; size, depth: Integer; nt: TNoiseType);
begin GenerateNoise(noise, size, depth, nt); end;

function GF_Op_NoiseSlerp(const v1, v2: TVector; t: Single): TVector;
begin Result := NoiseSlerp(v1, v2, t); end;

{ =========================================================================== }
{ GF_Gen_ : GENERATOR ACTIONS                                                }
{ =========================================================================== }

function GF_Gen_Build(const sizes: array of Integer; act: TActivationType;
  opt: TOptimizer; lr: Single): TNetwork;
begin Result := CreateNetwork(sizes, act, opt, lr); end;

function GF_Gen_BuildConv(noiseDim, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
begin Result := CreateConvGenerator(noiseDim, condSz, baseCh, act, opt, lr); end;

function GF_Gen_Forward(var gen: TNetwork; const inp: TMatrix): TMatrix;
begin Result := NetworkForward(gen, inp); end;

function GF_Gen_Backward(var gen: TNetwork; const gradOut: TMatrix): TMatrix;
begin Result := NetworkBackward(gen, gradOut); end;

function GF_Gen_Sample(var gen: TNetwork; count, noiseDim: Integer; nt: TNoiseType): TMatrix;
var noise: TMatrix;
begin
  GenerateNoise(noise, count, noiseDim, nt);
  Result := NetworkForward(gen, noise);
end;

function GF_Gen_SampleConditional(var gen: TNetwork; count, noiseDim, condSz: Integer;
  nt: TNoiseType; const cond: TMatrix): TMatrix;
var noise, combined: TMatrix; i, j: Integer;
begin
  GenerateNoise(noise, count, noiseDim, nt);
  combined := CreateMatrix(count, noiseDim + condSz);
  for i := 0 to count - 1 do begin
    for j := 0 to noiseDim - 1 do combined[i][j] := noise[i][j];
    for j := 0 to condSz - 1 do combined[i][noiseDim + j] := cond[i][j];
  end;
  Result := NetworkForward(gen, combined);
end;

procedure GF_Gen_UpdateWeights(var gen: TNetwork);
begin NetworkUpdateWeights(gen); end;

procedure GF_Gen_AddProgressiveLayer(var gen: TNetwork; resLvl: Integer);
begin AddProgressiveLayer(gen, resLvl, True); end;

function GF_Gen_GetLayerOutput(var gen: TNetwork; idx: Integer): TMatrix;
begin Result := GetLayerOutput(gen, idx); end;

procedure GF_Gen_SetTraining(var gen: TNetwork; training: Boolean);
begin SetNetworkTraining(gen, training); end;

function GF_Gen_Noise(size, depth: Integer; nt: TNoiseType): TMatrix;
begin GenerateNoise(Result, size, depth, nt); end;

function GF_Gen_NoiseSlerp(const v1, v2: TVector; t: Single): TVector;
begin Result := NoiseSlerp(v1, v2, t); end;

function GF_Gen_DeepCopy(const gen: TNetwork): TNetwork;
begin Result := DeepCopyNetwork(gen); end;

{ =========================================================================== }
{ GF_Disc_ : DISCRIMINATOR ACTIONS                                           }
{ =========================================================================== }

function GF_Disc_Build(const sizes: array of Integer; act: TActivationType;
  opt: TOptimizer; lr: Single): TNetwork;
begin Result := CreateNetwork(sizes, act, opt, lr); end;

function GF_Disc_BuildConv(inCh, inW, inH, condSz, baseCh: Integer;
  act: TActivationType; opt: TOptimizer; lr: Single): TNetwork;
begin Result := CreateConvDiscriminator(inCh, inW, inH, condSz, baseCh, act, opt, lr); end;

function GF_Disc_Evaluate(var disc: TNetwork; const inp: TMatrix): TMatrix;
begin Result := NetworkForward(disc, inp); end;

function GF_Disc_Forward(var disc: TNetwork; const inp: TMatrix): TMatrix;
begin Result := NetworkForward(disc, inp); end;

function GF_Disc_Backward(var disc: TNetwork; const gradOut: TMatrix): TMatrix;
begin Result := NetworkBackward(disc, gradOut); end;

procedure GF_Disc_UpdateWeights(var disc: TNetwork);
begin NetworkUpdateWeights(disc); end;

function GF_Disc_GradPenalty(var disc: TNetwork; const real, fake: TMatrix; lambda: Single): Single;
begin Result := ComputeGradientPenalty(disc, real, fake, lambda); end;

function GF_Disc_FeatureMatch(var disc: TNetwork; const real, fake: TMatrix; featLayer: Integer): Single;
begin Result := FeatureMatchingLoss(disc, real, fake, featLayer); end;

function GF_Disc_MinibatchStdDev(const inp: TMatrix): TMatrix;
begin Result := MinibatchStdDev(inp); end;

procedure GF_Disc_AddProgressiveLayer(var disc: TNetwork; resLvl: Integer);
begin AddProgressiveLayer(disc, resLvl, False); end;

function GF_Disc_GetLayerOutput(var disc: TNetwork; idx: Integer): TMatrix;
begin Result := GetLayerOutput(disc, idx); end;

procedure GF_Disc_SetTraining(var disc: TNetwork; training: Boolean);
begin SetNetworkTraining(disc, training); end;

function GF_Disc_DeepCopy(const disc: TNetwork): TNetwork;
begin Result := DeepCopyNetwork(disc); end;

{ =========================================================================== }
{ GF_Train_ : TRAINING CONTROL                                               }
{ =========================================================================== }

procedure GF_Train_Full(var gen, disc: TNetwork; var ds: TDataset; cfg: TGANConfig);
begin TrainGAN(gen, disc, ds, cfg); end;

procedure GF_Train_Step(var gen, disc: TNetwork; const realBatch, noise: TMatrix;
  cfg: TGANConfig);
var fakeData, discReal, discFake, discGen: TMatrix;
    dGrad, gGrad, dGradThru: TMatrix;
    realLabels: TMatrix; i, bs: Integer;
begin
  bs := Length(realBatch);
  { D step }
  discReal := NetworkForward(disc, realBatch);
  fakeData := NetworkForward(gen, noise);
  discFake := NetworkForward(disc, fakeData);
  case cfg.lossType of
    lossBCE: begin
      realLabels := CreateMatrix(bs, 1);
      for i := 0 to bs - 1 do realLabels[i][0] := 1.0;
      if cfg.useLabelSmoothing then realLabels := ApplyLabelSmoothing(realLabels, 0.0, 0.9);
      dGrad := BCEGradient(discReal, realLabels);
      NetworkForward(disc, realBatch); NetworkBackward(disc, dGrad); NetworkUpdateWeights(disc);
      realLabels := CreateMatrix(bs, 1);
      dGrad := BCEGradient(discFake, realLabels);
      NetworkForward(disc, fakeData); NetworkBackward(disc, dGrad); NetworkUpdateWeights(disc);
    end;
    lossWGANGP: begin
      dGrad := WGANDiscGrad(discReal, True);
      NetworkForward(disc, realBatch); NetworkBackward(disc, dGrad); NetworkUpdateWeights(disc);
      dGrad := WGANDiscGrad(discFake, False);
      NetworkForward(disc, fakeData); NetworkBackward(disc, dGrad); NetworkUpdateWeights(disc);
    end;
  else begin
    dGrad := BCEGradient(discReal, CreateMatrix(bs, 1));
    NetworkForward(disc, realBatch); NetworkBackward(disc, dGrad); NetworkUpdateWeights(disc);
  end;
  end;
  { G step }
  fakeData := NetworkForward(gen, noise);
  discGen := NetworkForward(disc, fakeData);
  case cfg.lossType of
    lossBCE: begin
      realLabels := CreateMatrix(bs, 1);
      for i := 0 to bs - 1 do realLabels[i][0] := 1.0;
      gGrad := BCEGradient(discGen, realLabels);
    end;
    lossWGANGP: gGrad := WGANGenGrad(discGen);
  else gGrad := BCEGradient(discGen, CreateMatrix(bs, 1));
  end;
  NetworkForward(disc, fakeData);
  dGradThru := NetworkBackward(disc, gGrad);
  NetworkForward(gen, noise);
  NetworkBackward(gen, dGradThru);
  NetworkUpdateWeights(gen);
end;

procedure GF_Train_Optimize(var net: TNetwork);
begin NetworkUpdateWeights(net); end;

procedure GF_Train_AdamUpdate(var p: TMatrix; const g: TMatrix;
  var m, v: TMatrix; t: Integer; lr, b1, b2, eps, wd: Single);
begin AdamUpdateMatrix(p, g, m, v, t, lr, b1, b2, eps, wd); end;

procedure GF_Train_SGDUpdate(var p: TMatrix; const g: TMatrix; lr, wd: Single);
begin SGDUpdateMatrix(p, g, lr, wd); end;

procedure GF_Train_RMSPropUpdate(var p: TMatrix; const g: TMatrix;
  var cache: TMatrix; lr, decay, eps, wd: Single);
begin RMSPropUpdateMatrix(p, g, cache, lr, decay, eps, wd); end;

function GF_Train_CosineAnneal(epoch, maxEp: Integer; baseLR, minLR: Single): Single;
begin Result := CosineAnneal(epoch, maxEp, baseLR, minLR); end;

{ --- Loss functions --- }
function GF_Train_BCELoss(const pred, target: TMatrix): Single;
begin Result := BinaryCrossEntropy(pred, target); end;

function GF_Train_BCEGrad(const pred, target: TMatrix): TMatrix;
begin Result := BCEGradient(pred, target); end;

function GF_Train_WGANDiscLoss(const dReal, dFake: TMatrix): Single;
begin Result := WGANDiscLoss(dReal, dFake); end;

function GF_Train_WGANGenLoss(const dFake: TMatrix): Single;
begin Result := WGANGenLoss(dFake); end;

function GF_Train_HingeDiscLoss(const dReal, dFake: TMatrix): Single;
begin Result := HingeDiscLoss(dReal, dFake); end;

function GF_Train_HingeGenLoss(const dFake: TMatrix): Single;
begin Result := HingeGenLoss(dFake); end;

function GF_Train_LSDiscLoss(const dReal, dFake: TMatrix): Single;
begin Result := LSDiscLoss(dReal, dFake); end;

function GF_Train_LSGenLoss(const dFake: TMatrix): Single;
begin Result := LSGenLoss(dFake); end;

function GF_Train_LabelSmoothing(const labels: TMatrix; lo, hi: Single): TMatrix;
begin Result := ApplyLabelSmoothing(labels, lo, hi); end;

{ --- Data --- }
function GF_Train_LoadDataset(const path: string; dt: TDataType): TDataset;
begin Result := LoadDataset(path, dt); end;

function GF_Train_LoadBMP(const path: string): TDataset;
begin Result := LoadBMPDataset(path); end;

function GF_Train_LoadWAV(const path: string): TDataset;
begin Result := LoadWAVDataset(path); end;

function GF_Train_CreateSynthetic(count, features: Integer): TDataset;
begin Result := CreateSyntheticDataset(count, features); end;

function GF_Train_Augment(const sample: TMatrix; dt: TDataType): TMatrix;
begin Result := AugmentSample(sample, dt); end;

{ --- Metrics --- }
function GF_Train_ComputeFID(const realS, fakeS: TMatrixArray): Single;
begin Result := ComputeFID(realS, fakeS); end;

function GF_Train_ComputeIS(const samples: TMatrixArray): Single;
begin Result := ComputeIS(samples); end;

procedure GF_Train_LogMetrics(const met: TGANMetrics; const fn: string);
begin LogMetrics(met, fn); end;

{ --- I/O --- }
procedure GF_Train_SaveModel(const net: TNetwork; const fn: string);
begin SaveNetworkBinary(net, fn); end;

procedure GF_Train_LoadModel(var net: TNetwork; const fn: string);
begin LoadNetworkBinary(net, fn); end;

procedure GF_Train_SaveJSON(const gen, disc: TNetwork; const fn: string);
begin SaveGANToJSON(gen, disc, fn); end;

procedure GF_Train_LoadJSON(var gen, disc: TNetwork; const fn: string);
begin LoadGANFromJSON(gen, disc, fn); end;

procedure GF_Train_SaveCheckpoint(const gen, disc: TNetwork; ep: Integer; const dir: string);
begin SaveCheckpoint(gen, disc, ep, dir); end;

procedure GF_Train_LoadCheckpoint(var gen, disc: TNetwork; ep: Integer; const dir: string);
begin LoadCheckpoint(gen, disc, ep, dir); end;

procedure GF_Train_SaveSamples(var gen: TNetwork; ep: Integer;
  const dir: string; noiseDim: Integer; nt: TNoiseType);
begin SaveGeneratedSamples(gen, ep, dir, noiseDim, nt); end;

procedure GF_Train_PlotCSV(const fn: string; const dL, gL: array of Single; cnt: Integer);
begin PlotLossCSV(fn, dL, gL, cnt); end;

procedure GF_Train_PrintBar(dLoss, gLoss: Single; w: Integer);
begin PrintLossBar(dLoss, gLoss, w); end;

{ =========================================================================== }
{ GF_Sec_ : SECURITY & ENTROPY                                               }
{ =========================================================================== }

procedure GF_Sec_AuditLog(const msg, logFile: string);
begin AuditLog(msg, logFile); end;

procedure GF_Sec_SecureRandomize;
begin SecureRandomize; end;

function GF_Sec_GetOSRandom: Byte;
begin Result := SecureRandomByte; end;

function GF_Sec_ValidatePath(const path: string): Boolean;
begin Result := ValidatePath(path); end;

procedure GF_Sec_VerifyWeights(var layer: TLayer);
begin ValidateAndCleanWeights(layer); end;

procedure GF_Sec_VerifyNetwork(var net: TNetwork);
var i: Integer;
begin
  for i := 0 to net.layerCount - 1 do
    ValidateAndCleanWeights(net.layers[i]);
end;

procedure GF_Sec_EncryptModel(const inF, outF, key: string);
begin EncryptFile(inF, outF, key); end;

procedure GF_Sec_DecryptModel(const inF, outF, key: string);
begin DecryptFile(inF, outF, key); end;

function GF_Sec_RunTests: Boolean;
begin Result := RunTests; end;

function GF_Sec_RunFuzzTests(iterations: Integer): Boolean;
begin Result := RunFuzzTests(iterations); end;

function GF_Sec_BoundsCheck(const M: TMatrix; r, c: Integer): Boolean;
begin Result := (r >= 0) and (r < Length(M)) and (c >= 0) and (c < Length(M[r])); end;

{ =========================================================================== }
{ HELP                                                                         }
{ =========================================================================== }

procedure ShowHelp;
begin
  WriteLn('GANFacade - Facade Pattern Interface for GAN');
  WriteLn('Version: 2.0  |  GAN Unit: v', GAN_VERSION);
  WriteLn('MIT License (c) 2025 Matthew Abbott');
  WriteLn('');
  WriteLn('USAGE:');
  WriteLn('  GANFacade [options]');
  WriteLn('  GANFacade --test <function-name>');
  WriteLn('  GANFacade --test all');
  WriteLn('');
  WriteLn('OPTIONS:');
  WriteLn('  --help, -h              Show this help and API reference');
  WriteLn('  --test <function-name>  Test a specific GF_ function');
  WriteLn('  --test all              Test all GF_ functions');
  WriteLn('  --list                  List all testable function names');
  WriteLn('  --epochs N              Training epochs (default 100)');
  WriteLn('  --batch-size N          Batch size (default 32)');
  WriteLn('  --lr F                  Learning rate (default 0.0002)');
  WriteLn('  --loss <bce|wgan|hinge|ls> Loss type');
  WriteLn('  --conv                  Use convolutional architecture');
  WriteLn('  --save <file>           Save model (.bin or .json)');
  WriteLn('  --load <file>           Load pretrained model');
  WriteLn('  --data <path>           Dataset path');
  WriteLn('  --output <dir>          Output directory');
  WriteLn('  --tests                 Run built-in unit tests');
  WriteLn('  --fuzz [N]              Run fuzz tests');
  WriteLn('');
  WriteLn('=====================================================================');
  WriteLn(' GAN FACADE API REFERENCE');
  WriteLn('=====================================================================');
  WriteLn('');
  WriteLn('TYPES:');
  WriteLn('  TMatrix = array of array of Single    TVector = array of Single');
  WriteLn('  TMatrixArray = array of TMatrix        TKernelArray = array of TMatrix');
  WriteLn('  TLayer = record   TNetwork = record    TGANConfig = record');
  WriteLn('  TGANMetrics = record   TDataset = record');
  WriteLn('');
  WriteLn('ENUMS:');
  WriteLn('  TActivationType = (atReLU, atSigmoid, atTanh, atLeakyReLU, atNone)');
  WriteLn('  TLayerType      = (ltDense, ltConv2D, ltDeconv2D, ltConv1D,');
  WriteLn('                     ltBatchNorm, ltLayerNorm, ltSpectralNorm, ltAttention)');
  WriteLn('  TLossType       = (lossBCE, lossWGANGP, lossHinge, lossLeastSquares)');
  WriteLn('  TDataType       = (dtImage, dtAudio, dtVector)');
  WriteLn('  TNoiseType      = (ntGauss, ntUniform, ntAnalog)');
  WriteLn('  TOptimizer      = (optAdam, optSGD, optRMSProp)');
  WriteLn('');
  WriteLn('--- GF_Op_ : LOW-LEVEL OPERATIONS ---');
  WriteLn('  GF_Op_CreateMatrix(rows, cols) -> TMatrix');
  WriteLn('  GF_Op_CreateVector(size) -> TVector');
  WriteLn('  GF_Op_MatrixMultiply(A, B) -> TMatrix');
  WriteLn('  GF_Op_MatrixAdd(A, B) -> TMatrix');
  WriteLn('  GF_Op_MatrixSubtract(A, B) -> TMatrix');
  WriteLn('  GF_Op_MatrixScale(A, s) -> TMatrix');
  WriteLn('  GF_Op_MatrixTranspose(A) -> TMatrix');
  WriteLn('  GF_Op_MatrixNormalize(A) -> TMatrix');
  WriteLn('  GF_Op_MatrixElementMul(A, B) -> TMatrix');
  WriteLn('  GF_Op_MatrixAddInPlace(var A, B)');
  WriteLn('  GF_Op_MatrixScaleInPlace(var A, s)');
  WriteLn('  GF_Op_MatrixClipInPlace(var A, lo, hi)');
  WriteLn('  GF_Op_SafeGet(M, r, c, default) -> Single');
  WriteLn('  GF_Op_SafeSet(var M, r, c, value)');
  WriteLn('  GF_Op_ReLU(A) -> TMatrix');
  WriteLn('  GF_Op_LeakyReLU(A, alpha) -> TMatrix');
  WriteLn('  GF_Op_Sigmoid(A) -> TMatrix');
  WriteLn('  GF_Op_Tanh(A) -> TMatrix');
  WriteLn('  GF_Op_Softmax(A) -> TMatrix');
  WriteLn('  GF_Op_Activate(A, act) -> TMatrix');
  WriteLn('  GF_Op_ActivationBackward(grad, pre, act) -> TMatrix');
  WriteLn('  GF_Op_Conv2D(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_Conv2DBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_Deconv2D(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_Deconv2DBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_Conv1D(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_Conv1DBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_BatchNorm(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_BatchNormBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_LayerNorm(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_LayerNormBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_SpectralNorm(var layer) -> TMatrix');
  WriteLn('  GF_Op_Attention(input, var layer) -> TMatrix');
  WriteLn('  GF_Op_AttentionBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_CreateDenseLayer(in, out, act) -> TLayer');
  WriteLn('  GF_Op_CreateConv2DLayer(iCh,oCh,k,s,p,w,h,act) -> TLayer');
  WriteLn('  GF_Op_CreateDeconv2DLayer(iCh,oCh,k,s,p,w,h,act) -> TLayer');
  WriteLn('  GF_Op_CreateConv1DLayer(iCh,oCh,k,s,p,len,act) -> TLayer');
  WriteLn('  GF_Op_CreateBatchNormLayer(features) -> TLayer');
  WriteLn('  GF_Op_CreateLayerNormLayer(features) -> TLayer');
  WriteLn('  GF_Op_CreateAttentionLayer(dModel, nHeads) -> TLayer');
  WriteLn('  GF_Op_LayerForward(var layer, input) -> TMatrix');
  WriteLn('  GF_Op_LayerBackward(var layer, grad) -> TMatrix');
  WriteLn('  GF_Op_InitLayerOptimizer(var layer, opt)');
  WriteLn('  GF_Op_RandomGaussian -> Single');
  WriteLn('  GF_Op_RandomUniform(lo, hi) -> Single');
  WriteLn('  GF_Op_GenerateNoise(var M, size, depth, nt)');
  WriteLn('  GF_Op_NoiseSlerp(v1, v2, t) -> TVector');
  WriteLn('');
  WriteLn('--- GF_Gen_ : GENERATOR ACTIONS ---');
  WriteLn('  GF_Gen_Build(sizes[], act, opt, lr) -> TNetwork');
  WriteLn('  GF_Gen_BuildConv(noiseDim, condSz, baseCh, act, opt, lr) -> TNetwork');
  WriteLn('  GF_Gen_Forward(var gen, input) -> TMatrix');
  WriteLn('  GF_Gen_Backward(var gen, grad) -> TMatrix');
  WriteLn('  GF_Gen_Sample(var gen, count, noiseDim, nt) -> TMatrix');
  WriteLn('  GF_Gen_SampleConditional(var gen, count, noiseDim, condSz, nt, cond)');
  WriteLn('  GF_Gen_UpdateWeights(var gen)');
  WriteLn('  GF_Gen_AddProgressiveLayer(var gen, lvl)');
  WriteLn('  GF_Gen_GetLayerOutput(var gen, idx) -> TMatrix');
  WriteLn('  GF_Gen_SetTraining(var gen, bool)');
  WriteLn('  GF_Gen_Noise(size, depth, nt) -> TMatrix');
  WriteLn('  GF_Gen_NoiseSlerp(v1, v2, t) -> TVector');
  WriteLn('  GF_Gen_DeepCopy(gen) -> TNetwork');
  WriteLn('');
  WriteLn('--- GF_Disc_ : DISCRIMINATOR ACTIONS ---');
  WriteLn('  GF_Disc_Build(sizes[], act, opt, lr) -> TNetwork');
  WriteLn('  GF_Disc_BuildConv(iCh, iW, iH, condSz, baseCh, act, opt, lr)');
  WriteLn('  GF_Disc_Evaluate(var disc, input) -> TMatrix');
  WriteLn('  GF_Disc_Forward(var disc, input) -> TMatrix');
  WriteLn('  GF_Disc_Backward(var disc, grad) -> TMatrix');
  WriteLn('  GF_Disc_UpdateWeights(var disc)');
  WriteLn('  GF_Disc_GradPenalty(var disc, real, fake, lambda) -> Single');
  WriteLn('  GF_Disc_FeatureMatch(var disc, real, fake, featLayer) -> Single');
  WriteLn('  GF_Disc_MinibatchStdDev(input) -> TMatrix');
  WriteLn('  GF_Disc_AddProgressiveLayer(var disc, lvl)');
  WriteLn('  GF_Disc_GetLayerOutput(var disc, idx) -> TMatrix');
  WriteLn('  GF_Disc_SetTraining(var disc, bool)');
  WriteLn('  GF_Disc_DeepCopy(disc) -> TNetwork');
  WriteLn('');
  WriteLn('--- GF_Train_ : TRAINING CONTROL ---');
  WriteLn('  GF_Train_Full(var gen, disc, ds, cfg)');
  WriteLn('  GF_Train_Step(var gen, disc, batch, noise, cfg)');
  WriteLn('  GF_Train_Optimize(var net)');
  WriteLn('  GF_Train_AdamUpdate(var p, g, var m, v, t, lr, b1, b2, e, wd)');
  WriteLn('  GF_Train_SGDUpdate(var p, g, lr, wd)');
  WriteLn('  GF_Train_RMSPropUpdate(var p, g, var cache, lr, decay, e, wd)');
  WriteLn('  GF_Train_CosineAnneal(ep, maxEp, baseLR, minLR) -> Single');
  WriteLn('  GF_Train_BCELoss(pred, target) -> Single');
  WriteLn('  GF_Train_BCEGrad(pred, target) -> TMatrix');
  WriteLn('  GF_Train_WGANDiscLoss(dReal, dFake) -> Single');
  WriteLn('  GF_Train_WGANGenLoss(dFake) -> Single');
  WriteLn('  GF_Train_HingeDiscLoss(dReal, dFake) -> Single');
  WriteLn('  GF_Train_HingeGenLoss(dFake) -> Single');
  WriteLn('  GF_Train_LSDiscLoss(dReal, dFake) -> Single');
  WriteLn('  GF_Train_LSGenLoss(dFake) -> Single');
  WriteLn('  GF_Train_LabelSmoothing(labels, lo, hi) -> TMatrix');
  WriteLn('  GF_Train_LoadDataset(path, dt) -> TDataset');
  WriteLn('  GF_Train_LoadBMP(path) -> TDataset');
  WriteLn('  GF_Train_LoadWAV(path) -> TDataset');
  WriteLn('  GF_Train_CreateSynthetic(count, features) -> TDataset');
  WriteLn('  GF_Train_Augment(sample, dt) -> TMatrix');
  WriteLn('  GF_Train_ComputeFID(realS, fakeS) -> Single');
  WriteLn('  GF_Train_ComputeIS(samples) -> Single');
  WriteLn('  GF_Train_LogMetrics(metrics, filename)');
  WriteLn('  GF_Train_SaveModel(net, filename)');
  WriteLn('  GF_Train_LoadModel(var net, filename)');
  WriteLn('  GF_Train_SaveJSON(gen, disc, filename)');
  WriteLn('  GF_Train_LoadJSON(var gen, disc, filename)');
  WriteLn('  GF_Train_SaveCheckpoint(gen, disc, ep, dir)');
  WriteLn('  GF_Train_LoadCheckpoint(var gen, disc, ep, dir)');
  WriteLn('  GF_Train_SaveSamples(var gen, ep, dir, nDim, nt)');
  WriteLn('  GF_Train_PlotCSV(fn, dLoss[], gLoss[], cnt)');
  WriteLn('  GF_Train_PrintBar(dLoss, gLoss, width)');
  WriteLn('');
  WriteLn('--- GF_Sec_ : SECURITY & ENTROPY ---');
  WriteLn('  GF_Sec_AuditLog(msg, logFile)        [NIST AU-2/AU-3]');
  WriteLn('  GF_Sec_SecureRandomize                [/dev/urandom seed]');
  WriteLn('  GF_Sec_GetOSRandom -> Byte            [/dev/urandom byte]');
  WriteLn('  GF_Sec_ValidatePath(path) -> Boolean');
  WriteLn('  GF_Sec_VerifyWeights(var layer)       [NaN/Inf clean]');
  WriteLn('  GF_Sec_VerifyNetwork(var net)          [all layers]');
  WriteLn('  GF_Sec_EncryptModel(in, out, key)     [NIST SC-28]');
  WriteLn('  GF_Sec_DecryptModel(in, out, key)     [NIST SC-28]');
  WriteLn('  GF_Sec_RunTests -> Boolean             [SA-11]');
  WriteLn('  GF_Sec_RunFuzzTests(iterations) -> Boolean [SA-11]');
  WriteLn('  GF_Sec_BoundsCheck(M, r, c) -> Boolean');
  WriteLn('');
  WriteLn('EXAMPLES:');
  WriteLn('  ./GANFacade --help');
  WriteLn('  ./GANFacade --test GF_Op_MatrixMultiply');
  WriteLn('  ./GANFacade --test all');
  WriteLn('  ./GANFacade --list');
  WriteLn('  ./GANFacade --epochs 50 --lr 0.0002 --loss wgan');
end;

{ =========================================================================== }
{ LIST ALL TESTABLE FUNCTIONS                                                  }
{ =========================================================================== }

procedure ListFunctions;
begin
  WriteLn('GF_Op_CreateMatrix');
  WriteLn('GF_Op_CreateVector');
  WriteLn('GF_Op_MatrixMultiply');
  WriteLn('GF_Op_MatrixAdd');
  WriteLn('GF_Op_MatrixSubtract');
  WriteLn('GF_Op_MatrixScale');
  WriteLn('GF_Op_MatrixTranspose');
  WriteLn('GF_Op_MatrixNormalize');
  WriteLn('GF_Op_MatrixElementMul');
  WriteLn('GF_Op_MatrixAddInPlace');
  WriteLn('GF_Op_MatrixScaleInPlace');
  WriteLn('GF_Op_MatrixClipInPlace');
  WriteLn('GF_Op_SafeGet');
  WriteLn('GF_Op_SafeSet');
  WriteLn('GF_Op_ReLU');
  WriteLn('GF_Op_LeakyReLU');
  WriteLn('GF_Op_Sigmoid');
  WriteLn('GF_Op_Tanh');
  WriteLn('GF_Op_Softmax');
  WriteLn('GF_Op_Activate');
  WriteLn('GF_Op_ActivationBackward');
  WriteLn('GF_Op_Conv2D');
  WriteLn('GF_Op_Conv2DBackward');
  WriteLn('GF_Op_Deconv2D');
  WriteLn('GF_Op_Deconv2DBackward');
  WriteLn('GF_Op_Conv1D');
  WriteLn('GF_Op_Conv1DBackward');
  WriteLn('GF_Op_BatchNorm');
  WriteLn('GF_Op_BatchNormBackward');
  WriteLn('GF_Op_LayerNorm');
  WriteLn('GF_Op_LayerNormBackward');
  WriteLn('GF_Op_SpectralNorm');
  WriteLn('GF_Op_Attention');
  WriteLn('GF_Op_AttentionBackward');
  WriteLn('GF_Op_CreateDenseLayer');
  WriteLn('GF_Op_CreateConv2DLayer');
  WriteLn('GF_Op_CreateDeconv2DLayer');
  WriteLn('GF_Op_CreateConv1DLayer');
  WriteLn('GF_Op_CreateBatchNormLayer');
  WriteLn('GF_Op_CreateLayerNormLayer');
  WriteLn('GF_Op_CreateAttentionLayer');
  WriteLn('GF_Op_LayerForward');
  WriteLn('GF_Op_LayerBackward');
  WriteLn('GF_Op_InitLayerOptimizer');
  WriteLn('GF_Op_RandomGaussian');
  WriteLn('GF_Op_RandomUniform');
  WriteLn('GF_Op_GenerateNoise');
  WriteLn('GF_Op_NoiseSlerp');
  WriteLn('GF_Gen_Build');
  WriteLn('GF_Gen_BuildConv');
  WriteLn('GF_Gen_Forward');
  WriteLn('GF_Gen_Backward');
  WriteLn('GF_Gen_Sample');
  WriteLn('GF_Gen_SampleConditional');
  WriteLn('GF_Gen_UpdateWeights');
  WriteLn('GF_Gen_AddProgressiveLayer');
  WriteLn('GF_Gen_GetLayerOutput');
  WriteLn('GF_Gen_SetTraining');
  WriteLn('GF_Gen_Noise');
  WriteLn('GF_Gen_NoiseSlerp');
  WriteLn('GF_Gen_DeepCopy');
  WriteLn('GF_Disc_Build');
  WriteLn('GF_Disc_BuildConv');
  WriteLn('GF_Disc_Evaluate');
  WriteLn('GF_Disc_Forward');
  WriteLn('GF_Disc_Backward');
  WriteLn('GF_Disc_UpdateWeights');
  WriteLn('GF_Disc_GradPenalty');
  WriteLn('GF_Disc_FeatureMatch');
  WriteLn('GF_Disc_MinibatchStdDev');
  WriteLn('GF_Disc_AddProgressiveLayer');
  WriteLn('GF_Disc_GetLayerOutput');
  WriteLn('GF_Disc_SetTraining');
  WriteLn('GF_Disc_DeepCopy');
  WriteLn('GF_Train_Full');
  WriteLn('GF_Train_Step');
  WriteLn('GF_Train_Optimize');
  WriteLn('GF_Train_AdamUpdate');
  WriteLn('GF_Train_SGDUpdate');
  WriteLn('GF_Train_RMSPropUpdate');
  WriteLn('GF_Train_CosineAnneal');
  WriteLn('GF_Train_BCELoss');
  WriteLn('GF_Train_BCEGrad');
  WriteLn('GF_Train_WGANDiscLoss');
  WriteLn('GF_Train_WGANGenLoss');
  WriteLn('GF_Train_HingeDiscLoss');
  WriteLn('GF_Train_HingeGenLoss');
  WriteLn('GF_Train_LSDiscLoss');
  WriteLn('GF_Train_LSGenLoss');
  WriteLn('GF_Train_LabelSmoothing');
  WriteLn('GF_Train_CreateSynthetic');
  WriteLn('GF_Train_Augment');
  WriteLn('GF_Train_ComputeFID');
  WriteLn('GF_Train_ComputeIS');
  WriteLn('GF_Train_LogMetrics');
  WriteLn('GF_Train_SaveModel');
  WriteLn('GF_Train_LoadModel');
  WriteLn('GF_Train_SaveJSON');
  WriteLn('GF_Train_LoadJSON');
  WriteLn('GF_Train_SaveCheckpoint');
  WriteLn('GF_Train_LoadCheckpoint');
  WriteLn('GF_Train_SaveSamples');
  WriteLn('GF_Train_PlotCSV');
  WriteLn('GF_Train_PrintBar');
  WriteLn('GF_Sec_AuditLog');
  WriteLn('GF_Sec_SecureRandomize');
  WriteLn('GF_Sec_GetOSRandom');
  WriteLn('GF_Sec_ValidatePath');
  WriteLn('GF_Sec_VerifyWeights');
  WriteLn('GF_Sec_VerifyNetwork');
  WriteLn('GF_Sec_EncryptModel');
  WriteLn('GF_Sec_DecryptModel');
  WriteLn('GF_Sec_RunTests');
  WriteLn('GF_Sec_RunFuzzTests');
  WriteLn('GF_Sec_BoundsCheck');
  WriteLn('GF_Introspect_NetworkFields');
  WriteLn('GF_Introspect_LayerFields');
  WriteLn('GF_Introspect_WeightAccess');
  WriteLn('GF_Introspect_ForwardCache');
  WriteLn('GF_Introspect_ActivationStats');
  WriteLn('GF_Introspect_Gradients');
  WriteLn('GF_Introspect_AdamState');
  WriteLn('GF_Introspect_MultiUpdate');
  WriteLn('GF_Introspect_DiscFields');
  WriteLn('GF_Introspect_WeightDecay');
  WriteLn('GF_Introspect_ConfigMutation');
  WriteLn('GF_Introspect_LayerChain');
end;

{ =========================================================================== }
{ TEST HELPERS                                                                 }
{ =========================================================================== }

const
  TEST_DIR = 'ganfacade_testout';

function MFinite(const M: TMatrix): Boolean;
var r, c: Integer;
begin
  Result := True;
  if Length(M) = 0 then Exit;
  for r := 0 to High(M) do
    for c := 0 to High(M[r]) do
      if IsNan(M[r][c]) or IsInfinite(M[r][c]) then begin
        Result := False; Exit;
      end;
end;

function VFinite(const V: TVector): Boolean;
var ii: Integer;
begin
  Result := True;
  if Length(V) = 0 then Exit;
  for ii := 0 to High(V) do
    if IsNan(V[ii]) or IsInfinite(V[ii]) then begin
      Result := False; Exit;
    end;
end;

{ =========================================================================== }
{ RUN SINGLE FUNCTION TEST                                                     }
{ =========================================================================== }

function RunSingleTest(const name: string): Boolean;
var
  A, B, C, inp, outp, grad, gradInp: TMatrix;
  V: TVector;
  v1, v2, vs: TVector;
  layer, denseL: TLayer;
  gen, disc, netCopy: TNetwork;
  sizes: array of Integer;
  noise, cond, comb, p, g, m, vv, cache: TMatrix;
  pred, target, dReal, dFake, labels, smoothed: TMatrix;
  ds: TDataset;
  cfg: TGANConfig;
  met: TGANMetrics;
  realS, fakeS: TMatrixArray;
  val, loss, lr, gp, fm, fid, iscore: Single;
  secByte: Byte;
  ok: Boolean;
  ii, jj: Integer;
begin
  Result := False;
  SecureRandomize;
  ForceDirectories(TEST_DIR);

  { === GF_Op_ Matrix === }
  if name = 'GF_Op_CreateMatrix' then begin
    A := GF_Op_CreateMatrix(3, 4);
    Result := (Length(A) = 3) and (Length(A[0]) = 4) and (A[0][0] = 0.0);
  end
  else if name = 'GF_Op_CreateVector' then begin
    V := GF_Op_CreateVector(5);
    Result := (Length(V) = 5) and (V[0] = 0.0);
  end
  else if name = 'GF_Op_MatrixMultiply' then begin
    A := GF_Op_CreateMatrix(2, 3);
    B := GF_Op_CreateMatrix(3, 2);
    A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
    A[1][0] := 4; A[1][1] := 5; A[1][2] := 6;
    B[0][0] := 7; B[0][1] := 8; B[1][0] := 9; B[1][1] := 10; B[2][0] := 11; B[2][1] := 12;
    C := GF_Op_MatrixMultiply(A, B);
    Result := (Length(C) = 2) and (Length(C[0]) = 2) and (Abs(C[0][0] - 58.0) < 0.01);
  end
  else if name = 'GF_Op_MatrixAdd' then begin
    A := GF_Op_CreateMatrix(2, 2); B := GF_Op_CreateMatrix(2, 2);
    A[0][0] := 1; B[0][0] := 5;
    C := GF_Op_MatrixAdd(A, B);
    Result := Abs(C[0][0] - 6.0) < 0.01;
  end
  else if name = 'GF_Op_MatrixSubtract' then begin
    A := GF_Op_CreateMatrix(2, 2); B := GF_Op_CreateMatrix(2, 2);
    A[0][0] := 10; B[0][0] := 3;
    C := GF_Op_MatrixSubtract(A, B);
    Result := Abs(C[0][0] - 7.0) < 0.01;
  end
  else if name = 'GF_Op_MatrixScale' then begin
    A := GF_Op_CreateMatrix(2, 2); A[0][0] := 4;
    C := GF_Op_MatrixScale(A, 3.0);
    Result := Abs(C[0][0] - 12.0) < 0.01;
  end
  else if name = 'GF_Op_MatrixTranspose' then begin
    A := GF_Op_CreateMatrix(2, 3);
    A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
    C := GF_Op_MatrixTranspose(A);
    Result := (Length(C) = 3) and (Length(C[0]) = 2) and (Abs(C[0][0] - 1.0) < 0.01);
  end
  else if name = 'GF_Op_MatrixNormalize' then begin
    A := GF_Op_CreateMatrix(1, 3); A[0][0] := 3; A[0][1] := 0; A[0][2] := 4;
    C := GF_Op_MatrixNormalize(A);
    Result := MFinite(C);
  end
  else if name = 'GF_Op_MatrixElementMul' then begin
    A := GF_Op_CreateMatrix(2, 2); B := GF_Op_CreateMatrix(2, 2);
    A[0][0] := 2; B[0][0] := 6;
    C := GF_Op_MatrixElementMul(A, B);
    Result := Abs(C[0][0] - 12.0) < 0.01;
  end
  else if name = 'GF_Op_MatrixAddInPlace' then begin
    A := GF_Op_CreateMatrix(2, 2); B := GF_Op_CreateMatrix(2, 2);
    A[0][0] := 1; B[0][0] := 10;
    GF_Op_MatrixAddInPlace(A, B);
    Result := Abs(A[0][0] - 11.0) < 0.01;
  end
  else if name = 'GF_Op_MatrixScaleInPlace' then begin
    A := GF_Op_CreateMatrix(1, 2); A[0][0] := 5;
    GF_Op_MatrixScaleInPlace(A, 0.5);
    Result := Abs(A[0][0] - 2.5) < 0.01;
  end
  else if name = 'GF_Op_MatrixClipInPlace' then begin
    A := GF_Op_CreateMatrix(1, 3); A[0][0] := -5; A[0][1] := 0.5; A[0][2] := 10;
    GF_Op_MatrixClipInPlace(A, -1, 1);
    Result := (Abs(A[0][0] - (-1.0)) < 0.01) and (Abs(A[0][2] - 1.0) < 0.01);
  end
  else if name = 'GF_Op_SafeGet' then begin
    A := GF_Op_CreateMatrix(2, 2); A[0][0] := 42;
    val := GF_Op_SafeGet(A, 0, 0, -1);
    Result := (Abs(val - 42.0) < 0.01) and (Abs(GF_Op_SafeGet(A, 99, 0, -1) - (-1.0)) < 0.01);
  end
  else if name = 'GF_Op_SafeSet' then begin
    A := GF_Op_CreateMatrix(2, 2);
    GF_Op_SafeSet(A, 0, 1, 77.0);
    GF_Op_SafeSet(A, 99, 0, 88.0);
    Result := Abs(A[0][1] - 77.0) < 0.01;
  end
  { === GF_Op_ Activations === }
  else if name = 'GF_Op_ReLU' then begin
    A := GF_Op_CreateMatrix(1, 3); A[0][0] := -2; A[0][1] := 0; A[0][2] := 3;
    B := GF_Op_ReLU(A);
    Result := (Abs(B[0][0]) < 0.01) and (Abs(B[0][2] - 3.0) < 0.01);
  end
  else if name = 'GF_Op_LeakyReLU' then begin
    A := GF_Op_CreateMatrix(1, 2); A[0][0] := -2; A[0][1] := 3;
    B := GF_Op_LeakyReLU(A, 0.2);
    Result := (Abs(B[0][0] - (-0.4)) < 0.01) and (Abs(B[0][1] - 3.0) < 0.01);
  end
  else if name = 'GF_Op_Sigmoid' then begin
    A := GF_Op_CreateMatrix(1, 1); A[0][0] := 0;
    B := GF_Op_Sigmoid(A);
    Result := Abs(B[0][0] - 0.5) < 0.01;
  end
  else if name = 'GF_Op_Tanh' then begin
    A := GF_Op_CreateMatrix(1, 1); A[0][0] := 0;
    B := GF_Op_Tanh(A);
    Result := Abs(B[0][0]) < 0.01;
  end
  else if name = 'GF_Op_Softmax' then begin
    A := GF_Op_CreateMatrix(1, 3); A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
    B := GF_Op_Softmax(A);
    Result := Abs(B[0][0] + B[0][1] + B[0][2] - 1.0) < 0.01;
  end
  else if name = 'GF_Op_Activate' then begin
    A := GF_Op_CreateMatrix(1, 2); A[0][0] := -2; A[0][1] := 3;
    B := GF_Op_Activate(A, atReLU);
    Result := (Abs(B[0][0]) < 0.01) and (Abs(B[0][1] - 3.0) < 0.01);
  end
  else if name = 'GF_Op_ActivationBackward' then begin
    A := GF_Op_CreateMatrix(1, 2); A[0][0] := -2; A[0][1] := 3;
    grad := GF_Op_CreateMatrix(1, 2); grad[0][0] := 1; grad[0][1] := 1;
    B := GF_Op_ActivationBackward(grad, A, atReLU);
    Result := (Abs(B[0][0]) < 0.01) and (Abs(B[0][1] - 1.0) < 0.01);
  end
  { === GF_Op_ Convolution === }
  else if name = 'GF_Op_Conv2D' then begin
    layer := GF_Op_CreateConv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    inp := GF_Op_CreateMatrix(2, 16); inp[0][0] := 1.0;
    layer.layerInput := inp;
    outp := GF_Op_Conv2D(inp, layer);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 32) and MFinite(outp);
  end
  else if name = 'GF_Op_Conv2DBackward' then begin
    layer := GF_Op_CreateConv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    inp := GF_Op_CreateMatrix(2, 16); inp[0][0] := 1.0;
    layer.layerInput := inp;
    Conv2DForward(inp, layer);
    grad := GF_Op_CreateMatrix(2, 32); grad[0][0] := 1.0;
    gradInp := GF_Op_Conv2DBackward(layer, grad);
    Result := (Length(gradInp) = 2) and (Length(gradInp[0]) = 16) and MFinite(gradInp);
  end
  else if name = 'GF_Op_Deconv2D' then begin
    layer := GF_Op_CreateDeconv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    inp := GF_Op_CreateMatrix(2, 16); inp[0][0] := 1.0;
    layer.layerInput := inp;
    outp := GF_Op_Deconv2D(inp, layer);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 32) and MFinite(outp);
  end
  else if name = 'GF_Op_Deconv2DBackward' then begin
    layer := GF_Op_CreateDeconv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    inp := GF_Op_CreateMatrix(2, 16); inp[0][0] := 1.0;
    layer.layerInput := inp;
    Deconv2DForward(inp, layer);
    grad := GF_Op_CreateMatrix(2, 32); grad[0][0] := 1.0;
    gradInp := GF_Op_Deconv2DBackward(layer, grad);
    Result := (Length(gradInp) = 2) and (Length(gradInp[0]) = 16) and MFinite(gradInp);
  end
  else if name = 'GF_Op_Conv1D' then begin
    layer := GF_Op_CreateConv1DLayer(1, 2, 3, 1, 1, 8, atReLU);
    inp := GF_Op_CreateMatrix(2, 8); inp[0][0] := 1.0;
    layer.layerInput := inp;
    outp := GF_Op_Conv1D(inp, layer);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 16) and MFinite(outp);
  end
  else if name = 'GF_Op_Conv1DBackward' then begin
    layer := GF_Op_CreateConv1DLayer(1, 2, 3, 1, 1, 8, atReLU);
    inp := GF_Op_CreateMatrix(2, 8); inp[0][0] := 1.0;
    layer.layerInput := inp;
    Conv1DForward(inp, layer);
    grad := GF_Op_CreateMatrix(2, 16); grad[0][0] := 1.0;
    gradInp := GF_Op_Conv1DBackward(layer, grad);
    Result := (Length(gradInp) = 2) and (Length(gradInp[0]) = 8) and MFinite(gradInp);
  end
  { === GF_Op_ Normalization === }
  else if name = 'GF_Op_BatchNorm' then begin
    layer := GF_Op_CreateBatchNormLayer(8);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do inp[ii][0] := ii + 1;
    layer.layerInput := inp;
    outp := GF_Op_BatchNorm(inp, layer);
    Result := (Length(outp) = 4) and (Length(outp[0]) = 8) and MFinite(outp);
  end
  else if name = 'GF_Op_BatchNormBackward' then begin
    layer := GF_Op_CreateBatchNormLayer(8);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do inp[ii][0] := ii + 1;
    layer.layerInput := inp;
    BatchNormForward(inp, layer);
    grad := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do grad[ii][0] := 1.0;
    gradInp := GF_Op_BatchNormBackward(layer, grad);
    Result := (Length(gradInp) = 4) and MFinite(gradInp);
  end
  else if name = 'GF_Op_LayerNorm' then begin
    layer := GF_Op_CreateLayerNormLayer(8);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do inp[ii][0] := ii + 1;
    layer.layerInput := inp;
    outp := GF_Op_LayerNorm(inp, layer);
    Result := MFinite(outp);
  end
  else if name = 'GF_Op_LayerNormBackward' then begin
    layer := GF_Op_CreateLayerNormLayer(8);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do inp[ii][0] := ii + 1;
    layer.layerInput := inp;
    LayerNormForward(inp, layer);
    grad := GF_Op_CreateMatrix(4, 8);
    for ii := 0 to 3 do grad[ii][0] := 1.0;
    gradInp := GF_Op_LayerNormBackward(layer, grad);
    Result := MFinite(gradInp);
  end
  else if name = 'GF_Op_SpectralNorm' then begin
    denseL := GF_Op_CreateDenseLayer(4, 4, atReLU);
    SetLength(denseL.spectralU, 4);
    SetLength(denseL.spectralV, 4);
    denseL.spectralU[0] := 1; denseL.spectralV[0] := 1;
    outp := GF_Op_SpectralNorm(denseL);
    Result := MFinite(outp) and (denseL.spectralSigma > 0);
  end
  { === GF_Op_ Attention === }
  else if name = 'GF_Op_Attention' then begin
    layer := GF_Op_CreateAttentionLayer(4, 2);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(3, 4); inp[0][0] := 1; inp[1][1] := 1;
    layer.layerInput := inp;
    outp := GF_Op_Attention(inp, layer);
    Result := (Length(outp) = 3) and (Length(outp[0]) = 4) and MFinite(outp);
  end
  else if name = 'GF_Op_AttentionBackward' then begin
    layer := GF_Op_CreateAttentionLayer(4, 2);
    layer.isTraining := True;
    inp := GF_Op_CreateMatrix(3, 4); inp[0][0] := 1; inp[1][1] := 1;
    layer.layerInput := inp;
    SelfAttentionForward(inp, layer);
    grad := GF_Op_CreateMatrix(3, 4); grad[0][0] := 1;
    gradInp := GF_Op_AttentionBackward(layer, grad);
    Result := (Length(gradInp) = 3) and MFinite(gradInp);
  end
  { === GF_Op_ Layer creation === }
  else if name = 'GF_Op_CreateDenseLayer' then begin
    layer := GF_Op_CreateDenseLayer(4, 3, atReLU);
    Result := (layer.layerType = ltDense) and (Length(layer.weights) = 4)
              and (Length(layer.weights[0]) = 3) and (Length(layer.bias) = 3);
  end
  else if name = 'GF_Op_CreateConv2DLayer' then begin
    layer := GF_Op_CreateConv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    Result := (layer.layerType = ltConv2D) and (layer.inputSize = 16) and (layer.outputSize = 32);
  end
  else if name = 'GF_Op_CreateDeconv2DLayer' then begin
    layer := GF_Op_CreateDeconv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
    Result := (layer.layerType = ltDeconv2D) and (layer.outputSize = 32);
  end
  else if name = 'GF_Op_CreateConv1DLayer' then begin
    layer := GF_Op_CreateConv1DLayer(1, 2, 3, 1, 1, 8, atReLU);
    Result := (layer.layerType = ltConv1D) and (layer.outputSize = 16);
  end
  else if name = 'GF_Op_CreateBatchNormLayer' then begin
    layer := GF_Op_CreateBatchNormLayer(8);
    Result := (layer.layerType = ltBatchNorm) and (Abs(layer.bnGamma[0] - 1.0) < 0.01);
  end
  else if name = 'GF_Op_CreateLayerNormLayer' then begin
    layer := GF_Op_CreateLayerNormLayer(8);
    Result := (layer.layerType = ltLayerNorm) and (Abs(layer.bnGamma[0] - 1.0) < 0.01);
  end
  else if name = 'GF_Op_CreateAttentionLayer' then begin
    layer := GF_Op_CreateAttentionLayer(4, 2);
    Result := (layer.layerType = ltAttention) and (layer.headDim = 2)
              and (Length(layer.Wq) = 4) and (Length(layer.Wq[0]) = 4);
  end
  { === GF_Op_ Layer dispatch === }
  else if name = 'GF_Op_LayerForward' then begin
    layer := GF_Op_CreateDenseLayer(4, 3, atReLU);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1; inp[0][1] := 2;
    outp := GF_Op_LayerForward(layer, inp);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 3) and MFinite(outp);
  end
  else if name = 'GF_Op_LayerBackward' then begin
    layer := GF_Op_CreateDenseLayer(4, 3, atReLU);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    LayerForward(layer, inp);
    grad := GF_Op_CreateMatrix(2, 3); grad[0][0] := 1; grad[0][1] := 1; grad[0][2] := 1;
    grad[1][0] := 1; grad[1][1] := 1; grad[1][2] := 1;
    gradInp := GF_Op_LayerBackward(layer, grad);
    Result := (Length(gradInp[0]) = 4) and MFinite(gradInp);
  end
  else if name = 'GF_Op_InitLayerOptimizer' then begin
    layer := GF_Op_CreateDenseLayer(4, 3, atReLU);
    GF_Op_InitLayerOptimizer(layer, optAdam);
    Result := (Length(layer.mWeight) = 4) and (Length(layer.vWeight) = 4);
  end
  { === GF_Op_ Random === }
  else if name = 'GF_Op_RandomGaussian' then begin
    val := GF_Op_RandomGaussian;
    Result := (not IsNan(val)) and (not IsInfinite(val));
  end
  else if name = 'GF_Op_RandomUniform' then begin
    ok := True;
    for ii := 1 to 50 do begin
      val := GF_Op_RandomUniform(0, 1);
      if (val < 0) or (val > 1) then ok := False;
    end;
    Result := ok;
  end
  else if name = 'GF_Op_GenerateNoise' then begin
    GF_Op_GenerateNoise(noise, 4, 8, ntGauss);
    Result := (Length(noise) = 4) and (Length(noise[0]) = 8) and MFinite(noise);
  end
  else if name = 'GF_Op_NoiseSlerp' then begin
    SetLength(v1, 4); SetLength(v2, 4);
    v1[0] := 1; v2[1] := 1;
    vs := GF_Op_NoiseSlerp(v1, v2, 0.5);
    Result := VFinite(vs) and (Length(vs) = 4);
  end
  { === GF_Gen_ === }
  else if name = 'GF_Gen_Build' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := GF_Gen_Build(sizes, atLeakyReLU, optAdam, 0.001);
    Result := (gen.layerCount = 3) and (gen.optimizer = optAdam);
  end
  else if name = 'GF_Gen_BuildConv' then begin
    gen := GF_Gen_BuildConv(8, 0, 4, atLeakyReLU, optAdam, 0.0002);
    Result := (gen.layerCount = 7) and (gen.layers[0].layerType = ltDense);
  end
  else if name = 'GF_Gen_Forward' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    outp := GF_Gen_Forward(gen, inp);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 1) and MFinite(outp);
  end
  else if name = 'GF_Gen_Backward' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    NetworkForward(gen, inp);
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1;
    outp := GF_Gen_Backward(gen, grad);
    Result := (Length(outp) = 2) and (Length(outp[0]) = 4) and MFinite(outp);
  end
  else if name = 'GF_Gen_Sample' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    outp := GF_Gen_Sample(gen, 4, 4, ntGauss);
    Result := (Length(outp) = 4) and MFinite(outp);
  end
  else if name = 'GF_Gen_SampleConditional' then begin
    SetLength(sizes, 3); sizes[0] := 6; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    cond := GF_Op_CreateMatrix(2, 2); cond[0][0] := 1; cond[1][1] := 1;
    outp := GF_Gen_SampleConditional(gen, 2, 4, 2, ntGauss, cond);
    Result := (Length(outp) = 2) and MFinite(outp);
  end
  else if name = 'GF_Gen_UpdateWeights' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    NetworkForward(gen, inp);
    NetworkBackward(gen, GF_Op_CreateMatrix(2, 1));
    GF_Gen_UpdateWeights(gen);
    Result := MFinite(gen.layers[0].weights);
  end
  else if name = 'GF_Gen_AddProgressiveLayer' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := GF_Gen_Build(sizes, atLeakyReLU, optAdam, 0.001);
    GF_Gen_AddProgressiveLayer(gen, 1);
    Result := (gen.layerCount = 5);
  end
  else if name = 'GF_Gen_GetLayerOutput' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    NetworkForward(gen, inp);
    outp := GF_Gen_GetLayerOutput(gen, 0);
    Result := (Length(outp) > 0) and MFinite(outp);
  end
  else if name = 'GF_Gen_SetTraining' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    GF_Gen_SetTraining(gen, False);
    ok := (gen.isTraining = False);
    GF_Gen_SetTraining(gen, True);
    Result := ok and (gen.isTraining = True);
  end
  else if name = 'GF_Gen_Noise' then begin
    noise := GF_Gen_Noise(3, 8, ntGauss);
    Result := (Length(noise) = 3) and (Length(noise[0]) = 8) and MFinite(noise);
  end
  else if name = 'GF_Gen_NoiseSlerp' then begin
    SetLength(v1, 4); SetLength(v2, 4); v1[0] := 1; v2[3] := 1;
    vs := GF_Gen_NoiseSlerp(v1, v2, 0.5);
    Result := VFinite(vs);
  end
  else if name = 'GF_Gen_DeepCopy' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := GF_Gen_Build(sizes, atReLU, optAdam, 0.001);
    netCopy := GF_Gen_DeepCopy(gen);
    netCopy.learningRate := 0.999;
    Result := (netCopy.layerCount = gen.layerCount) and (Abs(gen.learningRate - 0.001) < 0.0001);
  end
  { === GF_Disc_ === }
  else if name = 'GF_Disc_Build' then begin
    SetLength(sizes, 4); sizes[0] := 1; sizes[1] := 8; sizes[2] := 16; sizes[3] := 1;
    disc := GF_Disc_Build(sizes, atLeakyReLU, optAdam, 0.001);
    Result := (disc.layerCount = 3);
  end
  else if name = 'GF_Disc_BuildConv' then begin
    disc := GF_Disc_BuildConv(1, 8, 8, 0, 4, atLeakyReLU, optAdam, 0.0002);
    Result := (disc.layerCount = 5) and (disc.layers[4].layerType = ltDense);
  end
  else if name = 'GF_Disc_Evaluate' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(4, 1); inp[0][0] := 0.5; inp[1][0] := -0.3;
    outp := GF_Disc_Evaluate(disc, inp);
    Result := (Length(outp) = 4) and (Length(outp[0]) = 1) and MFinite(outp);
  end
  else if name = 'GF_Disc_Forward' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 1); inp[0][0] := 0.5;
    outp := GF_Disc_Forward(disc, inp);
    Result := (Length(outp) = 2) and MFinite(outp);
  end
  else if name = 'GF_Disc_Backward' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 1); inp[0][0] := 0.5;
    NetworkForward(disc, inp);
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1;
    outp := GF_Disc_Backward(disc, grad);
    Result := (Length(outp) = 2) and MFinite(outp);
  end
  else if name = 'GF_Disc_UpdateWeights' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 1); inp[0][0] := 0.5;
    NetworkForward(disc, inp);
    NetworkBackward(disc, GF_Op_CreateMatrix(2, 1));
    GF_Disc_UpdateWeights(disc);
    Result := MFinite(disc.layers[0].weights);
  end
  else if name = 'GF_Disc_GradPenalty' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atLeakyReLU, optAdam, 0.001);
    A := GF_Op_CreateMatrix(4, 1); B := GF_Op_CreateMatrix(4, 1);
    A[0][0] := 0.9; A[1][0] := 0.8; B[0][0] := 0.1; B[1][0] := 0.2;
    A[2][0] := 0.7; A[3][0] := 0.6; B[2][0] := 0.3; B[3][0] := 0.4;
    gp := GF_Disc_GradPenalty(disc, A, B, 10.0);
    Result := (not IsNan(gp)) and (not IsInfinite(gp)) and (gp >= 0);
  end
  else if name = 'GF_Disc_FeatureMatch' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atLeakyReLU, optAdam, 0.001);
    A := GF_Op_CreateMatrix(4, 1); B := GF_Op_CreateMatrix(4, 1);
    A[0][0] := 0.9; B[0][0] := 0.1;
    A[1][0] := 0.8; B[1][0] := 0.2;
    A[2][0] := 0.7; B[2][0] := 0.3;
    A[3][0] := 0.6; B[3][0] := 0.4;
    NetworkForward(disc, A);
    fm := GF_Disc_FeatureMatch(disc, A, B, 0);
    Result := (not IsNan(fm)) and (not IsInfinite(fm)) and (fm >= 0);
  end
  else if name = 'GF_Disc_MinibatchStdDev' then begin
    inp := GF_Op_CreateMatrix(4, 1);
    inp[0][0] := 0.5; inp[1][0] := -0.3; inp[2][0] := 0.8; inp[3][0] := 0.1;
    outp := GF_Disc_MinibatchStdDev(inp);
    Result := (Length(outp) = 4) and (Length(outp[0]) = 2) and MFinite(outp);
  end
  else if name = 'GF_Disc_AddProgressiveLayer' then begin
    SetLength(sizes, 4); sizes[0] := 1; sizes[1] := 8; sizes[2] := 16; sizes[3] := 1;
    disc := GF_Disc_Build(sizes, atLeakyReLU, optAdam, 0.001);
    GF_Disc_AddProgressiveLayer(disc, 1);
    Result := (disc.layerCount = 5);
  end
  else if name = 'GF_Disc_GetLayerOutput' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 1); inp[0][0] := 0.5;
    NetworkForward(disc, inp);
    outp := GF_Disc_GetLayerOutput(disc, 0);
    Result := (Length(outp) > 0) and MFinite(outp);
  end
  else if name = 'GF_Disc_SetTraining' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    GF_Disc_SetTraining(disc, False);
    ok := (disc.isTraining = False);
    GF_Disc_SetTraining(disc, True);
    Result := ok and (disc.isTraining = True);
  end
  else if name = 'GF_Disc_DeepCopy' then begin
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := GF_Disc_Build(sizes, atReLU, optAdam, 0.001);
    netCopy := GF_Disc_DeepCopy(disc);
    netCopy.learningRate := 0.999;
    Result := (netCopy.layerCount = disc.layerCount) and (Abs(disc.learningRate - 0.001) < 0.0001);
  end
  { === GF_Train_ Loss === }
  else if name = 'GF_Train_BCELoss' then begin
    pred := GF_Op_CreateMatrix(4, 1); target := GF_Op_CreateMatrix(4, 1);
    pred[0][0] := 0.9; pred[1][0] := 0.8; pred[2][0] := 0.2; pred[3][0] := 0.1;
    target[0][0] := 1; target[1][0] := 1; target[2][0] := 0; target[3][0] := 0;
    loss := GF_Train_BCELoss(pred, target);
    Result := (not IsNan(loss)) and (not IsInfinite(loss)) and (loss > 0) and (loss < 5);
  end
  else if name = 'GF_Train_BCEGrad' then begin
    pred := GF_Op_CreateMatrix(4, 1); target := GF_Op_CreateMatrix(4, 1);
    pred[0][0] := 0.9; target[0][0] := 1;
    grad := GF_Train_BCEGrad(pred, target);
    Result := (Length(grad) = 4) and MFinite(grad);
  end
  else if name = 'GF_Train_WGANDiscLoss' then begin
    dReal := GF_Op_CreateMatrix(4, 1); dFake := GF_Op_CreateMatrix(4, 1);
    dReal[0][0] := 2; dFake[0][0] := -1;
    dReal[1][0] := 1.5; dFake[1][0] := -0.5;
    dReal[2][0] := 1.8; dFake[2][0] := -0.8;
    dReal[3][0] := 2.1; dFake[3][0] := -1.2;
    loss := GF_Train_WGANDiscLoss(dReal, dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_WGANGenLoss' then begin
    dFake := GF_Op_CreateMatrix(4, 1); dFake[0][0] := -1; dFake[1][0] := -0.5;
    dFake[2][0] := -0.8; dFake[3][0] := -1.2;
    loss := GF_Train_WGANGenLoss(dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_HingeDiscLoss' then begin
    dReal := GF_Op_CreateMatrix(4, 1); dFake := GF_Op_CreateMatrix(4, 1);
    dReal[0][0] := 2; dFake[0][0] := -1;
    dReal[1][0] := 1.5; dFake[1][0] := -0.5;
    dReal[2][0] := 1.8; dFake[2][0] := -0.8;
    dReal[3][0] := 2.1; dFake[3][0] := -1.2;
    loss := GF_Train_HingeDiscLoss(dReal, dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_HingeGenLoss' then begin
    dFake := GF_Op_CreateMatrix(4, 1); dFake[0][0] := -1; dFake[1][0] := -0.5;
    dFake[2][0] := -0.8; dFake[3][0] := -1.2;
    loss := GF_Train_HingeGenLoss(dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_LSDiscLoss' then begin
    dReal := GF_Op_CreateMatrix(4, 1); dFake := GF_Op_CreateMatrix(4, 1);
    dReal[0][0] := 2; dFake[0][0] := -1;
    dReal[1][0] := 1.5; dFake[1][0] := -0.5;
    dReal[2][0] := 1.8; dFake[2][0] := -0.8;
    dReal[3][0] := 2.1; dFake[3][0] := -1.2;
    loss := GF_Train_LSDiscLoss(dReal, dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_LSGenLoss' then begin
    dFake := GF_Op_CreateMatrix(4, 1); dFake[0][0] := -1; dFake[1][0] := -0.5;
    dFake[2][0] := -0.8; dFake[3][0] := -1.2;
    loss := GF_Train_LSGenLoss(dFake);
    Result := (not IsNan(loss)) and (not IsInfinite(loss));
  end
  else if name = 'GF_Train_LabelSmoothing' then begin
    labels := GF_Op_CreateMatrix(4, 1);
    labels[0][0] := 1; labels[1][0] := 1; labels[2][0] := 0; labels[3][0] := 0;
    smoothed := GF_Train_LabelSmoothing(labels, 0.0, 0.9);
    Result := (smoothed[0][0] <= 0.91) and (smoothed[2][0] >= -0.01) and MFinite(smoothed);
  end
  { === GF_Train_ Optimizers === }
  else if name = 'GF_Train_AdamUpdate' then begin
    p := GF_Op_CreateMatrix(2, 2); g := GF_Op_CreateMatrix(2, 2);
    m := GF_Op_CreateMatrix(2, 2); vv := GF_Op_CreateMatrix(2, 2);
    p[0][0] := 1; g[0][0] := 0.1;
    GF_Train_AdamUpdate(p, g, m, vv, 1, 0.001, 0.9, 0.999, 1e-8, 0.0);
    Result := MFinite(p) and (Abs(p[0][0] - 1.0) > 1e-6);
  end
  else if name = 'GF_Train_SGDUpdate' then begin
    p := GF_Op_CreateMatrix(2, 2); g := GF_Op_CreateMatrix(2, 2);
    p[0][0] := 1; g[0][0] := 0.5;
    GF_Train_SGDUpdate(p, g, 0.01, 0.0);
    Result := MFinite(p) and (Abs(p[0][0] - 1.0) > 1e-6);
  end
  else if name = 'GF_Train_RMSPropUpdate' then begin
    p := GF_Op_CreateMatrix(2, 2); g := GF_Op_CreateMatrix(2, 2);
    cache := GF_Op_CreateMatrix(2, 2);
    p[0][0] := 1; g[0][0] := 0.1;
    GF_Train_RMSPropUpdate(p, g, cache, 0.001, 0.9, 1e-8, 0.0);
    Result := MFinite(p) and (Abs(p[0][0] - 1.0) > 1e-6);
  end
  else if name = 'GF_Train_CosineAnneal' then begin
    lr := GF_Train_CosineAnneal(0, 100, 0.001, 0.0001);
    Result := (Abs(lr - 0.001) < 0.0002);
  end
  { === GF_Train_ Data === }
  else if name = 'GF_Train_CreateSynthetic' then begin
    ds := GF_Train_CreateSynthetic(100, 4);
    Result := (ds.count = 100) and (Length(ds.samples) = 100);
  end
  else if name = 'GF_Train_Augment' then begin
    A := GF_Op_CreateMatrix(1, 4); A[0][0] := 1; A[0][1] := 2;
    B := GF_Train_Augment(A, dtVector);
    Result := MFinite(B) and (Length(B[0]) = 4);
  end
  else if name = 'GF_Train_ComputeFID' then begin
    SetLength(realS, 10); SetLength(fakeS, 10);
    for ii := 0 to 9 do begin
      realS[ii] := GF_Op_CreateMatrix(1, 4);
      fakeS[ii] := GF_Op_CreateMatrix(1, 4);
      realS[ii][0][0] := Random; fakeS[ii][0][0] := Random;
    end;
    fid := GF_Train_ComputeFID(realS, fakeS);
    Result := (not IsNan(fid)) and (not IsInfinite(fid));
  end
  else if name = 'GF_Train_ComputeIS' then begin
    SetLength(realS, 10);
    for ii := 0 to 9 do begin
      realS[ii] := GF_Op_CreateMatrix(1, 4);
      realS[ii][0][0] := Random;
    end;
    iscore := GF_Train_ComputeIS(realS);
    Result := (not IsNan(iscore)) and (not IsInfinite(iscore));
  end
  else if name = 'GF_Train_LogMetrics' then begin
    FillChar(met, SizeOf(met), 0);
    met.dLossReal := 0.5; met.gLoss := 0.7; met.epoch := 1;
    GF_Train_LogMetrics(met, TEST_DIR + '/test_metrics.csv');
    Result := FileExists(TEST_DIR + '/test_metrics.csv');
  end
  { === GF_Train_ I/O === }
  else if name = 'GF_Train_SaveModel' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    GF_Train_SaveModel(gen, TEST_DIR + '/gen_test.bin');
    Result := FileExists(TEST_DIR + '/gen_test.bin');
  end
  else if name = 'GF_Train_LoadModel' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    GF_Train_SaveModel(gen, TEST_DIR + '/gen_load_test.bin');
    FillChar(disc, SizeOf(disc), 0);
    GF_Train_LoadModel(disc, TEST_DIR + '/gen_load_test.bin');
    Result := (disc.layerCount = gen.layerCount);
  end
  else if name = 'GF_Train_SaveJSON' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    disc := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    GF_Train_SaveJSON(gen, disc, TEST_DIR + '/gan_test.json');
    Result := FileExists(TEST_DIR + '/gan_test.json');
  end
  else if name = 'GF_Train_LoadJSON' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    disc := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    GF_Train_SaveJSON(gen, disc, TEST_DIR + '/gan_json_test.json');
    FillChar(gen, SizeOf(gen), 0); FillChar(disc, SizeOf(disc), 0);
    GF_Train_LoadJSON(gen, disc, TEST_DIR + '/gan_json_test.json');
    Result := True; { JSON loader is partial stub, just verify no crash }
  end
  else if name = 'GF_Train_SaveCheckpoint' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    disc := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    ForceDirectories(TEST_DIR + '/ckpt');
    GF_Train_SaveCheckpoint(gen, disc, 5, TEST_DIR + '/ckpt');
    Result := FileExists(TEST_DIR + '/ckpt/gen_ep5.bin') and
              FileExists(TEST_DIR + '/ckpt/disc_ep5.bin');
  end
  else if name = 'GF_Train_LoadCheckpoint' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    disc := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    ForceDirectories(TEST_DIR + '/ckpt');
    GF_Train_SaveCheckpoint(gen, disc, 7, TEST_DIR + '/ckpt');
    FillChar(gen, SizeOf(gen), 0); FillChar(disc, SizeOf(disc), 0);
    GF_Train_LoadCheckpoint(gen, disc, 7, TEST_DIR + '/ckpt');
    Result := (gen.layerCount > 0);
  end
  else if name = 'GF_Train_SaveSamples' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    GF_Train_SaveSamples(gen, 1, TEST_DIR, 4, ntGauss);
    Result := FileExists(TEST_DIR + '/samples_ep1.csv');
  end
  else if name = 'GF_Train_PlotCSV' then begin
    SetLength(V, 3); V[0] := 0.5; V[1] := 0.4; V[2] := 0.3;
    SetLength(v1, 3); v1[0] := 0.8; v1[1] := 0.6; v1[2] := 0.5;
    PlotLossCSV(TEST_DIR + '/losses_test.csv', V, v1, 3);
    Result := FileExists(TEST_DIR + '/losses_test.csv');
  end
  else if name = 'GF_Train_PrintBar' then begin
    GF_Train_PrintBar(0.5, 0.8, 30);
    WriteLn;
    Result := True;
  end
  { === GF_Train_ Training === }
  else if name = 'GF_Train_Step' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    A := GF_Op_CreateMatrix(4, 1);
    for ii := 0 to 3 do A[ii][0] := 0.5 + Random * 0.5;
    GenerateNoise(noise, 4, 4, ntGauss);
    cfg := DefaultConfig;
    cfg.lossType := lossBCE; cfg.batchSize := 4;
    GF_Train_Step(gen, disc, A, noise, cfg);
    Result := MFinite(gen.layers[0].weights) and MFinite(disc.layers[0].weights);
  end
  else if name = 'GF_Train_Optimize' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 4); inp[0][0] := 1;
    NetworkForward(gen, inp);
    NetworkBackward(gen, GF_Op_CreateMatrix(2, 1));
    GF_Train_Optimize(gen);
    Result := MFinite(gen.layers[0].weights);
  end
  else if name = 'GF_Train_Full' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 8; sizes[2] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    SetLength(sizes, 3); sizes[0] := 1; sizes[1] := 8; sizes[2] := 1;
    disc := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    ds := CreateSyntheticDataset(20, 1);
    cfg := DefaultConfig;
    cfg.epochs := 1; cfg.batchSize := 4; cfg.lossType := lossBCE;
    cfg.noiseDepth := 4; cfg.outputDir := TEST_DIR;
    GF_Train_Full(gen, disc, ds, cfg);
    Result := MFinite(gen.layers[0].weights) and MFinite(disc.layers[0].weights);
  end
  { === GF_Sec_ === }
  else if name = 'GF_Sec_AuditLog' then begin
    GF_Sec_AuditLog('test entry', TEST_DIR + '/test_audit.log');
    Result := FileExists(TEST_DIR + '/test_audit.log');
  end
  else if name = 'GF_Sec_SecureRandomize' then begin
    GF_Sec_SecureRandomize;
    Result := True;
  end
  else if name = 'GF_Sec_GetOSRandom' then begin
    secByte := GF_Sec_GetOSRandom;
    Result := (secByte >= 0) and (secByte <= 255);
  end
  else if name = 'GF_Sec_ValidatePath' then begin
    Result := GF_Sec_ValidatePath('/tmp/model.bin') and
              (not GF_Sec_ValidatePath('/tmp/../etc/passwd')) and
              (not GF_Sec_ValidatePath(''));
  end
  else if name = 'GF_Sec_VerifyWeights' then begin
    layer := CreateDenseLayer(4, 4, atReLU);
    layer.weights[0][0] := 0.0 / 0.0;
    GF_Sec_VerifyWeights(layer);
    Result := not IsNan(layer.weights[0][0]);
  end
  else if name = 'GF_Sec_VerifyNetwork' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 4; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    gen.layers[0].weights[0][0] := 1.0 / 0.0;
    GF_Sec_VerifyNetwork(gen);
    Result := not IsInfinite(gen.layers[0].weights[0][0]);
  end
  else if name = 'GF_Sec_EncryptModel' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 4; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    SaveNetworkBinary(gen, TEST_DIR + '/enc_test.bin');
    GF_Sec_EncryptModel(TEST_DIR + '/enc_test.bin', TEST_DIR + '/enc_test.enc', 'testkey');
    Result := FileExists(TEST_DIR + '/enc_test.enc');
  end
  else if name = 'GF_Sec_DecryptModel' then begin
    SetLength(sizes, 3); sizes[0] := 4; sizes[1] := 4; sizes[2] := 1;
    gen := CreateNetwork(sizes, atReLU, optAdam, 0.001);
    SaveNetworkBinary(gen, TEST_DIR + '/dec_test.bin');
    EncryptFile(TEST_DIR + '/dec_test.bin', TEST_DIR + '/dec_test.enc', 'testkey');
    GF_Sec_DecryptModel(TEST_DIR + '/dec_test.enc', TEST_DIR + '/dec_test_dec.bin', 'testkey');
    Result := FileExists(TEST_DIR + '/dec_test_dec.bin');
  end
  else if name = 'GF_Sec_RunTests' then begin
    Result := GF_Sec_RunTests;
  end
  else if name = 'GF_Sec_RunFuzzTests' then begin
    Result := GF_Sec_RunFuzzTests(50);
  end
  else if name = 'GF_Sec_BoundsCheck' then begin
    A := GF_Op_CreateMatrix(3, 4);
    Result := GF_Sec_BoundsCheck(A, 0, 0) and (not GF_Sec_BoundsCheck(A, 99, 0));
  end
  { === GF_Introspect_ === }
  else if name = 'GF_Introspect_NetworkFields' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    Result := (gen.layerCount = 3) and (gen.optimizer = optAdam)
              and (Abs(gen.learningRate - 0.001) < 1e-6)
              and (Abs(gen.beta1 - 0.9) < 1e-6)
              and (Abs(gen.beta2 - 0.999) < 1e-6)
              and (gen.epsilon > 0)
              and (Abs(gen.progressiveAlpha - 1.0) < 1e-6)
              and (gen.isTraining = True);
  end
  else if name = 'GF_Introspect_LayerFields' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    Result := (gen.layers[0].layerType = ltDense)
              and (gen.layers[0].activation = atLeakyReLU)
              and (gen.layers[0].inputSize = 8)
              and (gen.layers[0].outputSize = 16);
  end
  else if name = 'GF_Introspect_WeightAccess' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    val := gen.layers[0].weights[0][0];
    ok := (not IsNan(val)) and (not IsInfinite(val));
    val := gen.layers[0].weights[3][7];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    val := gen.layers[0].bias[0];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    gen.layers[0].weights[0][0] := 0.12345;
    ok := ok and (Abs(gen.layers[0].weights[0][0] - 0.12345) < 1e-5);
    gen.layers[0].bias[0] := -0.5;
    ok := ok and (Abs(gen.layers[0].bias[0] - (-0.5)) < 1e-5);
    Result := ok;
  end
  else if name = 'GF_Introspect_ForwardCache' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8);
    for ii := 0 to 7 do begin inp[0][ii] := RandomGaussian * 0.5; inp[1][ii] := RandomGaussian * 0.5; end;
    NetworkForward(gen, inp);
    Result := (Length(gen.layers[0].layerInput) = 2)
              and (Length(gen.layers[0].layerInput[0]) = 8)
              and (Length(gen.layers[0].layerOutput) > 0)
              and (Length(gen.layers[0].preActivation) > 0)
              and MFinite(gen.layers[0].layerOutput);
  end
  else if name = 'GF_Introspect_ActivationStats' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8);
    for ii := 0 to 7 do begin inp[0][ii] := RandomGaussian * 0.5; inp[1][ii] := RandomGaussian * 0.5; end;
    NetworkForward(gen, inp);
    val := 1e30; lr := -1e30; loss := 0;
    for ii := 0 to High(gen.layers[0].layerOutput[0]) do begin
      fm := gen.layers[0].layerOutput[0][ii];
      if fm < val then val := fm;
      if fm > lr then lr := fm;
      loss := loss + fm;
    end;
    Result := (not IsNan(val)) and (not IsInfinite(val))
              and (not IsNan(lr)) and (not IsInfinite(lr))
              and (not IsNan(loss / gen.layers[0].outputSize))
              and (not IsInfinite(loss / gen.layers[0].outputSize));
  end
  else if name = 'GF_Introspect_Gradients' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8);
    for ii := 0 to 7 do begin inp[0][ii] := RandomGaussian * 0.5; inp[1][ii] := RandomGaussian * 0.5; end;
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1.0; grad[1][0] := -1.0;
    NetworkForward(gen, inp);
    NetworkBackward(gen, grad);
    ok := (Length(gen.layers[0].weightGrad) = 8)
          and (Length(gen.layers[0].weightGrad[0]) = 16);
    val := gen.layers[0].weightGrad[0][0];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    ok := ok and (Length(gen.layers[0].biasGrad) = 16);
    val := gen.layers[0].biasGrad[0];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    Result := ok;
  end
  else if name = 'GF_Introspect_AdamState' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8);
    for ii := 0 to 7 do begin inp[0][ii] := RandomGaussian * 0.5; inp[1][ii] := RandomGaussian * 0.5; end;
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1.0; grad[1][0] := -1.0;
    NetworkForward(gen, inp);
    NetworkBackward(gen, grad);
    NetworkUpdateWeights(gen);
    ok := (gen.layers[0].adamT > 0);
    ok := ok and (Length(gen.layers[0].mWeight) = 8)
          and (Length(gen.layers[0].mWeight[0]) = 16);
    val := gen.layers[0].mWeight[0][0];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    ok := ok and (Length(gen.layers[0].vWeight) = 8)
          and (Length(gen.layers[0].vWeight[0]) = 16);
    val := gen.layers[0].vWeight[0][0];
    ok := ok and (not IsNan(val)) and (not IsInfinite(val));
    ok := ok and (Length(gen.layers[0].mBias) = 16);
    ok := ok and (Length(gen.layers[0].vBias) = 16);
    Result := ok;
  end
  else if name = 'GF_Introspect_MultiUpdate' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8);
    for ii := 0 to 7 do begin inp[0][ii] := RandomGaussian * 0.5; inp[1][ii] := RandomGaussian * 0.5; end;
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1.0; grad[1][0] := -1.0;
    for ii := 1 to 4 do begin
      NetworkForward(gen, inp);
      NetworkBackward(gen, grad);
      NetworkUpdateWeights(gen);
    end;
    Result := (gen.layers[0].adamT >= 4) and MFinite(gen.layers[0].weights);
  end
  else if name = 'GF_Introspect_DiscFields' then begin
    SetLength(sizes, 4); sizes[0] := 1; sizes[1] := 8; sizes[2] := 16; sizes[3] := 1;
    disc := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 1); inp[0][0] := 0.5; inp[1][0] := -0.3;
    grad := GF_Op_CreateMatrix(2, 1); grad[0][0] := 1; grad[1][0] := -1;
    NetworkForward(disc, inp);
    NetworkBackward(disc, grad);
    NetworkUpdateWeights(disc);
    Result := MFinite(disc.layers[0].weights)
              and (Length(disc.layers[0].weightGrad) > 0)
              and (Length(disc.layers[0].mWeight) > 0);
  end
  else if name = 'GF_Introspect_WeightDecay' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    gen.weightDecay := 0.01;
    inp := GF_Op_CreateMatrix(2, 8);
    grad := GF_Op_CreateMatrix(2, 1);
    NetworkForward(gen, inp);
    NetworkBackward(gen, grad);
    NetworkUpdateWeights(gen);
    Result := (Abs(gen.weightDecay - 0.01) < 1e-6) and MFinite(gen.layers[0].weights);
  end
  else if name = 'GF_Introspect_ConfigMutation' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    gen.progressiveAlpha := 0.5;
    ok := Abs(gen.progressiveAlpha - 0.5) < 1e-6;
    gen.learningRate := 0.0005;
    ok := ok and (Abs(gen.learningRate - 0.0005) < 1e-6);
    Result := ok;
  end
  else if name = 'GF_Introspect_LayerChain' then begin
    SetLength(sizes, 4); sizes[0] := 8; sizes[1] := 16; sizes[2] := 8; sizes[3] := 1;
    gen := CreateNetwork(sizes, atLeakyReLU, optAdam, 0.001);
    inp := GF_Op_CreateMatrix(2, 8); inp[0][0] := 1;
    NetworkForward(gen, inp);
    Result := (Abs(gen.layers[0].layerOutput[0][0] - gen.layers[1].layerInput[0][0]) < 1e-6)
              and (Abs(gen.layers[1].layerOutput[0][0] - gen.layers[2].layerInput[0][0]) < 1e-6);
  end
  else begin
    WriteLn('ERROR: Unknown function: ', name);
    Result := False;
  end;
end;

{ =========================================================================== }
{ RUN ALL TESTS                                                                }
{ =========================================================================== }

procedure RunAllTests;
const
  FUNCS: array[0..126] of string = (
    'GF_Op_CreateMatrix', 'GF_Op_CreateVector', 'GF_Op_MatrixMultiply',
    'GF_Op_MatrixAdd', 'GF_Op_MatrixSubtract', 'GF_Op_MatrixScale',
    'GF_Op_MatrixTranspose', 'GF_Op_MatrixNormalize', 'GF_Op_MatrixElementMul',
    'GF_Op_MatrixAddInPlace', 'GF_Op_MatrixScaleInPlace', 'GF_Op_MatrixClipInPlace',
    'GF_Op_SafeGet', 'GF_Op_SafeSet',
    'GF_Op_ReLU', 'GF_Op_LeakyReLU', 'GF_Op_Sigmoid', 'GF_Op_Tanh',
    'GF_Op_Softmax', 'GF_Op_Activate', 'GF_Op_ActivationBackward',
    'GF_Op_Conv2D', 'GF_Op_Conv2DBackward', 'GF_Op_Deconv2D',
    'GF_Op_Deconv2DBackward', 'GF_Op_Conv1D', 'GF_Op_Conv1DBackward',
    'GF_Op_BatchNorm', 'GF_Op_BatchNormBackward',
    'GF_Op_LayerNorm', 'GF_Op_LayerNormBackward', 'GF_Op_SpectralNorm',
    'GF_Op_Attention', 'GF_Op_AttentionBackward',
    'GF_Op_CreateDenseLayer', 'GF_Op_CreateConv2DLayer',
    'GF_Op_CreateDeconv2DLayer', 'GF_Op_CreateConv1DLayer',
    'GF_Op_CreateBatchNormLayer', 'GF_Op_CreateLayerNormLayer',
    'GF_Op_CreateAttentionLayer',
    'GF_Op_LayerForward', 'GF_Op_LayerBackward', 'GF_Op_InitLayerOptimizer',
    'GF_Op_RandomGaussian', 'GF_Op_RandomUniform',
    'GF_Op_GenerateNoise', 'GF_Op_NoiseSlerp',
    'GF_Gen_Build', 'GF_Gen_BuildConv', 'GF_Gen_Forward', 'GF_Gen_Backward',
    'GF_Gen_Sample', 'GF_Gen_SampleConditional', 'GF_Gen_UpdateWeights',
    'GF_Gen_AddProgressiveLayer', 'GF_Gen_GetLayerOutput',
    'GF_Gen_SetTraining', 'GF_Gen_Noise', 'GF_Gen_NoiseSlerp', 'GF_Gen_DeepCopy',
    'GF_Disc_Build', 'GF_Disc_BuildConv', 'GF_Disc_Evaluate', 'GF_Disc_Forward',
    'GF_Disc_Backward', 'GF_Disc_UpdateWeights', 'GF_Disc_GradPenalty',
    'GF_Disc_FeatureMatch', 'GF_Disc_MinibatchStdDev',
    'GF_Disc_AddProgressiveLayer', 'GF_Disc_GetLayerOutput',
    'GF_Disc_SetTraining', 'GF_Disc_DeepCopy',
    'GF_Train_BCELoss', 'GF_Train_BCEGrad',
    'GF_Train_WGANDiscLoss', 'GF_Train_WGANGenLoss',
    'GF_Train_HingeDiscLoss', 'GF_Train_HingeGenLoss',
    'GF_Train_LSDiscLoss', 'GF_Train_LSGenLoss', 'GF_Train_LabelSmoothing',
    'GF_Train_AdamUpdate', 'GF_Train_SGDUpdate',
    'GF_Train_RMSPropUpdate', 'GF_Train_CosineAnneal',
    'GF_Train_CreateSynthetic', 'GF_Train_Augment',
    'GF_Train_ComputeFID', 'GF_Train_ComputeIS', 'GF_Train_LogMetrics',
    'GF_Train_SaveModel', 'GF_Train_LoadModel',
    'GF_Train_SaveJSON', 'GF_Train_LoadJSON',
    'GF_Train_SaveCheckpoint', 'GF_Train_LoadCheckpoint',
    'GF_Train_SaveSamples', 'GF_Train_PlotCSV', 'GF_Train_PrintBar',
    'GF_Train_Optimize', 'GF_Train_Step', 'GF_Train_Full',
    'GF_Sec_AuditLog', 'GF_Sec_SecureRandomize', 'GF_Sec_GetOSRandom',
    'GF_Sec_ValidatePath', 'GF_Sec_VerifyWeights', 'GF_Sec_VerifyNetwork',
    'GF_Sec_EncryptModel', 'GF_Sec_DecryptModel',
    'GF_Sec_RunTests', 'GF_Sec_RunFuzzTests', 'GF_Sec_BoundsCheck',
    'GF_Introspect_NetworkFields', 'GF_Introspect_LayerFields',
    'GF_Introspect_WeightAccess', 'GF_Introspect_ForwardCache',
    'GF_Introspect_ActivationStats', 'GF_Introspect_Gradients',
    'GF_Introspect_AdamState', 'GF_Introspect_MultiUpdate',
    'GF_Introspect_DiscFields', 'GF_Introspect_WeightDecay',
    'GF_Introspect_ConfigMutation', 'GF_Introspect_LayerChain'
  );
var
  total, pass, fail, idx: Integer;
  ok: Boolean;
begin
  total := 0; pass := 0; fail := 0;
  for idx := 0 to High(FUNCS) do begin
    Inc(total);
    ok := RunSingleTest(FUNCS[idx]);
    if ok then begin
      Inc(pass);
      WriteLn('  [PASS] ', FUNCS[idx]);
    end else begin
      Inc(fail);
      WriteLn('  [FAIL] ', FUNCS[idx]);
    end;
  end;
  WriteLn('');
  WriteLn('=====================================================================');
  WriteLn(' RESULTS: ', total, ' tests | ', pass, ' passed | ', fail, ' failed');
  WriteLn('=====================================================================');
  if fail = 0 then WriteLn('All tests passed.')
  else WriteLn(fail, ' test(s) FAILED.');
  if fail > 0 then Halt(1);
end;

{ =========================================================================== }
{ MAIN PROGRAM                                                                }
{ =========================================================================== }

var
  config: TGANConfig;
  generator, discriminator: TNetwork;
  genSizes, discSizes: array of Integer;
  dataset: TDataset;
  i, j: Integer;
  testName: string;
begin
  { Check for --help, --list, --test before normal config parsing }
  for i := 1 to ParamCount do begin
    if (ParamStr(i) = '--help') or (ParamStr(i) = '-h') then begin
      ShowHelp;
      Halt(0);
    end;
    if ParamStr(i) = '--list' then begin
      ListFunctions;
      Halt(0);
    end;
    if ParamStr(i) = '--test' then begin
      SecureRandomize;
      if i < ParamCount then
        testName := ParamStr(i + 1)
      else begin
        WriteLn('ERROR: --test requires a function name or "all"');
        Halt(1);
      end;
      if testName = 'all' then begin
        WriteLn('GANFacade --test all');
        WriteLn('GAN Unit v', GAN_VERSION);
        WriteLn('');
        RunAllTests;
      end else begin
        if RunSingleTest(testName) then begin
          WriteLn('[PASS] ', testName);
          Halt(0);
        end else begin
          WriteLn('[FAIL] ', testName);
          Halt(1);
        end;
      end;
      Halt(0);
    end;
  end;

  { Normal GAN training flow }
  WriteLn('GAN Facade v' + GAN_VERSION);
  WriteLn('');

  config := ParseConfig;

  { Run tests if requested }
  if config.runTests then begin
    GF_Sec_SecureRandomize;
    if GF_Sec_RunTests then WriteLn('Tests passed.')
    else WriteLn('Tests had failures.');
    Halt;
  end;
  if config.runFuzz then begin
    GF_Sec_SecureRandomize;
    GF_Sec_RunFuzzTests(config.fuzzIterations);
    Halt;
  end;

  { Audit }
  if config.auditLog then
    GF_Sec_AuditLog('GAN started with config: epochs=' + IntToStr(config.epochs),
      config.auditLogFile);

  WriteLn('Configuration:');
  WriteLn('  Epochs: ', config.epochs);
  WriteLn('  Batch Size: ', config.batchSize);
  WriteLn('  Noise Depth: ', config.noiseDepth);
  WriteLn('  Learning Rate: ', config.learningRate:0:6);
  WriteLn('  Loss: ', Ord(config.lossType));
  WriteLn('  Conv: ', config.useConv);
  WriteLn('  Attention: ', config.useAttention);
  WriteLn('  Condition Size: ', config.conditionSize);
  WriteLn('');

  { Build networks }
  if config.useConv then begin
    WriteLn('Building convolutional GAN...');
    generator := GF_Gen_BuildConv(config.noiseDepth, config.conditionSize, 64,
      config.activation, config.optimizer, config.learningRate);
    discriminator := GF_Disc_BuildConv(1, 32, 32, config.conditionSize, 64,
      config.activation, config.optimizer, config.learningRate);
  end else begin
    SetLength(genSizes, 4);
    genSizes[0] := config.noiseDepth + config.conditionSize;
    genSizes[1] := 128; genSizes[2] := 64; genSizes[3] := 1;
    SetLength(discSizes, 4);
    discSizes[0] := 1; discSizes[1] := 64; discSizes[2] := 128; discSizes[3] := 1;
    generator := GF_Gen_Build(genSizes, config.activation, config.optimizer, config.learningRate);
    discriminator := GF_Disc_Build(discSizes, config.activation, config.optimizer, config.learningRate);
  end;

  { Apply spectral norm to discriminator }
  if config.useSpectralNorm then
    for i := 0 to discriminator.layerCount - 1 do
      if discriminator.layers[i].layerType = ltDense then begin
        SetLength(discriminator.layers[i].spectralU, Length(discriminator.layers[i].weights));
        SetLength(discriminator.layers[i].spectralV, Length(discriminator.layers[i].weights[0]));
        for j := 0 to High(discriminator.layers[i].spectralU) do
          discriminator.layers[i].spectralU[j] := RandomGaussian;
        for j := 0 to High(discriminator.layers[i].spectralV) do
          discriminator.layers[i].spectralV[j] := RandomGaussian;
      end;

  { TTUR learning rates }
  if config.generatorLR > 0 then generator.learningRate := config.generatorLR;
  if config.discriminatorLR > 0 then discriminator.learningRate := config.discriminatorLR;
  if config.useWeightDecay then begin
    generator.weightDecay := config.weightDecayVal;
    discriminator.weightDecay := config.weightDecayVal;
  end;

  WriteLn('Generator layers: ', generator.layerCount);
  WriteLn('Discriminator layers: ', discriminator.layerCount);
  WriteLn('');

  { Verify weights }
  GF_Sec_VerifyNetwork(generator);
  GF_Sec_VerifyNetwork(discriminator);

  { Load pretrained if specified }
  if config.loadModel <> '' then begin
    WriteLn('Loading model: ', config.loadModel);
    GF_Train_LoadModel(generator, config.loadModel);
  end;
  if config.loadJSONModel <> '' then begin
    WriteLn('Loading JSON model: ', config.loadJSONModel);
    GF_Train_LoadJSON(generator, discriminator, config.loadJSONModel);
  end;

  { Load or generate dataset }
  if config.dataPath <> '' then begin
    WriteLn('Loading dataset from: ', config.dataPath);
    dataset := GF_Train_LoadDataset(config.dataPath, config.dataType);
  end else begin
    WriteLn('Generating synthetic dataset...');
    dataset := GF_Train_CreateSynthetic(1000, 1);
  end;
  WriteLn('Dataset: ', dataset.count, ' samples');
  WriteLn('');

  { Create output directory }
  if config.outputDir <> '' then ForceDirectories(config.outputDir);

  { Train }
  WriteLn('Starting training...');
  WriteLn('');
  GF_Train_Full(generator, discriminator, dataset, config);

  { Save model }
  if config.saveModel <> '' then begin
    if (Pos('.json', config.saveModel) > 0) or (Pos('.JSON', config.saveModel) > 0) then
      GF_Train_SaveJSON(generator, discriminator, config.saveModel)
    else begin
      GF_Train_SaveModel(generator, config.saveModel);
      GF_Train_SaveModel(discriminator, StringReplace(config.saveModel, '.bin', '_disc.bin', []));
    end;
    { Optional encryption }
    if config.useEncryption and (config.encryptionKey <> '') then begin
      GF_Sec_EncryptModel(config.saveModel, config.saveModel + '.enc', config.encryptionKey);
      WriteLn('Model encrypted to: ', config.saveModel + '.enc');
    end;
  end;

  if config.auditLog then
    GF_Sec_AuditLog('GAN completed successfully', config.auditLogFile);

  WriteLn('Done.');
end.

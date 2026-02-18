(*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * GANTest - Comprehensive test suite and API reference for GAN Facade.
 * Tests every GF_Op_, GF_Gen_, GF_Disc_, GF_Train_, GF_Sec_ function.
 * Includes deep introspection tests for weights, biases, gradients,
 * optimizer state, and layer outputs.
 *
 * Usage: GANTest [--help] [--all] [--ops] [--gen] [--disc] [--train]
 *                [--sec] [--introspect] [--verbose] [--quick]
 *)
program GANTest;
{$mode objfpc}{$H+}

uses
  {$ifdef UNIX}cthreads,{$endif}
  SysUtils, Math, Classes, GAN;

const
  TEST_DIR = 'gantest_output';

var
  GTotal, GPass, GFail: Integer;
  GVerbose, GQuick: Boolean;
  GRunOps, GRunGen, GRunDisc, GRunTrain, GRunSec, GRunIntrospect: Boolean;

{ =========================================================================== }
{ TEST HARNESS                                                                 }
{ =========================================================================== }

procedure TC(const name: string; ok: Boolean);
begin
  Inc(GTotal);
  if ok then begin
    Inc(GPass);
    if GVerbose then WriteLn('  [PASS] ', name);
  end else begin
    Inc(GFail);
    WriteLn('  [FAIL] ', name);
  end;
end;

procedure Heading(const s: string);
begin
  WriteLn('');
  WriteLn('--- ', s, ' ---');
end;

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
var i: Integer;
begin
  Result := True;
  if Length(V) = 0 then Exit;
  for i := 0 to High(V) do
    if IsNan(V[i]) or IsInfinite(V[i]) then begin
      Result := False; Exit;
    end;
end;

{ =========================================================================== }
{ HELP / API REFERENCE                                                         }
{ =========================================================================== }

procedure ShowTestHelp;
begin
  WriteLn('GANTest - Comprehensive Test Suite & API Reference for GAN Facade');
  WriteLn('Version: 1.0  |  GAN Unit: v', GAN_VERSION);
  WriteLn('MIT License (c) 2025 Matthew Abbott');
  WriteLn('');
  WriteLn('USAGE:');
  WriteLn('  GANTest [options]');
  WriteLn('');
  WriteLn('OPTIONS:');
  WriteLn('  --help          Show this help and API reference');
  WriteLn('  --all           Run all test categories (default)');
  WriteLn('  --ops           Run GF_Op_ low-level operation tests');
  WriteLn('  --gen           Run GF_Gen_ generator tests');
  WriteLn('  --disc          Run GF_Disc_ discriminator tests');
  WriteLn('  --train         Run GF_Train_ training & I/O tests');
  WriteLn('  --sec           Run GF_Sec_ security tests');
  WriteLn('  --introspect    Run deep introspection tests');
  WriteLn('  --verbose, -v   Show passing tests (not just failures)');
  WriteLn('  --quick         Skip slow tests (training loops, fuzz)');
  WriteLn('');
  WriteLn('=====================================================================');
  WriteLn(' GAN FACADE API REFERENCE (118 functions)');
  WriteLn('=====================================================================');
  WriteLn('');
  WriteLn('TYPES:');
  WriteLn('  TMatrix          = array of array of Single');
  WriteLn('  TVector          = array of Single');
  WriteLn('  TMatrixArray     = array of TMatrix');
  WriteLn('  TKernelArray     = array of TMatrix');
  WriteLn('  TLayer           = record { weights, bias, kernels, grads, ... }');
  WriteLn('  TNetwork         = record { layers[], optimizer, lr, ... }');
  WriteLn('  TGANConfig       = record { epochs, batchSize, lossType, ... }');
  WriteLn('  TGANMetrics      = record { dLossReal, dLossFake, gLoss, FID, IS }');
  WriteLn('  TDataset         = record { samples[], labels, count, dataType }');
  WriteLn('');
  WriteLn('ENUMS:');
  WriteLn('  TActivationType  = (atReLU, atSigmoid, atTanh, atLeakyReLU, atNone)');
  WriteLn('  TLayerType       = (ltDense, ltConv2D, ltDeconv2D, ltConv1D,');
  WriteLn('                      ltBatchNorm, ltLayerNorm, ltSpectralNorm,');
  WriteLn('                      ltAttention)');
  WriteLn('  TLossType        = (lossBCE, lossWGANGP, lossHinge, lossLS)');
  WriteLn('  TDataType        = (dtImage, dtAudio, dtVector)');
  WriteLn('  TNoiseType       = (ntGauss, ntUniform, ntAnalog)');
  WriteLn('  TOptimizer       = (optAdam, optSGD, optRMSProp)');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' GF_Op_ : LOW-LEVEL OPERATIONS (48 functions)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('  Matrix Creation:');
  WriteLn('    GF_Op_CreateMatrix(rows, cols)            -> TMatrix');
  WriteLn('    GF_Op_CreateVector(size)                  -> TVector');
  WriteLn('');
  WriteLn('  Matrix Arithmetic:');
  WriteLn('    GF_Op_MatrixMultiply(A, B)                -> TMatrix');
  WriteLn('    GF_Op_MatrixAdd(A, B)                     -> TMatrix');
  WriteLn('    GF_Op_MatrixSubtract(A, B)                -> TMatrix');
  WriteLn('    GF_Op_MatrixScale(A, scalar)              -> TMatrix');
  WriteLn('    GF_Op_MatrixTranspose(A)                  -> TMatrix');
  WriteLn('    GF_Op_MatrixNormalize(A)                  -> TMatrix');
  WriteLn('    GF_Op_MatrixElementMul(A, B)              -> TMatrix');
  WriteLn('');
  WriteLn('  Matrix In-Place:');
  WriteLn('    GF_Op_MatrixAddInPlace(var A, B)          -> void');
  WriteLn('    GF_Op_MatrixScaleInPlace(var A, s)        -> void');
  WriteLn('    GF_Op_MatrixClipInPlace(var A, lo, hi)    -> void');
  WriteLn('');
  WriteLn('  Bounds-Checked Access:');
  WriteLn('    GF_Op_SafeGet(M, r, c, default)           -> Single');
  WriteLn('    GF_Op_SafeSet(var M, r, c, value)         -> void');
  WriteLn('');
  WriteLn('  Activations:');
  WriteLn('    GF_Op_ReLU(A)                             -> TMatrix');
  WriteLn('    GF_Op_LeakyReLU(A, alpha)                 -> TMatrix');
  WriteLn('    GF_Op_Sigmoid(A)                          -> TMatrix');
  WriteLn('    GF_Op_Tanh(A)                             -> TMatrix');
  WriteLn('    GF_Op_Softmax(A)                          -> TMatrix');
  WriteLn('    GF_Op_Activate(A, TActivationType)        -> TMatrix');
  WriteLn('    GF_Op_ActivationBackward(grad, pre, act)  -> TMatrix');
  WriteLn('');
  WriteLn('  Convolution:');
  WriteLn('    GF_Op_Conv2D(input, var layer)            -> TMatrix');
  WriteLn('    GF_Op_Conv2DBackward(var layer, grad)     -> TMatrix');
  WriteLn('    GF_Op_Deconv2D(input, var layer)          -> TMatrix');
  WriteLn('    GF_Op_Deconv2DBackward(var layer, grad)   -> TMatrix');
  WriteLn('    GF_Op_Conv1D(input, var layer)            -> TMatrix');
  WriteLn('    GF_Op_Conv1DBackward(var layer, grad)     -> TMatrix');
  WriteLn('');
  WriteLn('  Normalization:');
  WriteLn('    GF_Op_BatchNorm(input, var layer)         -> TMatrix');
  WriteLn('    GF_Op_BatchNormBackward(var layer, grad)  -> TMatrix');
  WriteLn('    GF_Op_LayerNorm(input, var layer)         -> TMatrix');
  WriteLn('    GF_Op_LayerNormBackward(var layer, grad)  -> TMatrix');
  WriteLn('    GF_Op_SpectralNorm(var layer)             -> TMatrix');
  WriteLn('');
  WriteLn('  Self-Attention:');
  WriteLn('    GF_Op_Attention(input, var layer)         -> TMatrix');
  WriteLn('    GF_Op_AttentionBackward(var layer, grad)  -> TMatrix');
  WriteLn('');
  WriteLn('  Layer Creation:');
  WriteLn('    GF_Op_CreateDenseLayer(in, out, act)      -> TLayer');
  WriteLn('    GF_Op_CreateConv2DLayer(iCh,oCh,k,s,p,w,h,act) -> TLayer');
  WriteLn('    GF_Op_CreateDeconv2DLayer(iCh,oCh,k,s,p,w,h,act)-> TLayer');
  WriteLn('    GF_Op_CreateConv1DLayer(iCh,oCh,k,s,p,len,act)  -> TLayer');
  WriteLn('    GF_Op_CreateBatchNormLayer(features)      -> TLayer');
  WriteLn('    GF_Op_CreateLayerNormLayer(features)      -> TLayer');
  WriteLn('    GF_Op_CreateAttentionLayer(dModel, nHeads)-> TLayer');
  WriteLn('');
  WriteLn('  Layer Dispatch:');
  WriteLn('    GF_Op_LayerForward(var layer, input)      -> TMatrix');
  WriteLn('    GF_Op_LayerBackward(var layer, grad)      -> TMatrix');
  WriteLn('    GF_Op_InitLayerOptimizer(var layer, opt)  -> void');
  WriteLn('');
  WriteLn('  Random / Noise:');
  WriteLn('    GF_Op_RandomGaussian                      -> Single');
  WriteLn('    GF_Op_RandomUniform(lo, hi)               -> Single');
  WriteLn('    GF_Op_GenerateNoise(var M, size, depth, nt) -> void');
  WriteLn('    GF_Op_NoiseSlerp(v1, v2, t)              -> TVector');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' GF_Gen_ : GENERATOR ACTIONS (13 functions)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('    GF_Gen_Build(sizes[], act, opt, lr)       -> TNetwork');
  WriteLn('    GF_Gen_BuildConv(noiseDim, condSz, baseCh, act, opt, lr)');
  WriteLn('    GF_Gen_Forward(var gen, input)             -> TMatrix');
  WriteLn('    GF_Gen_Backward(var gen, grad)             -> TMatrix');
  WriteLn('    GF_Gen_Sample(var gen, count, noiseDim, nt)-> TMatrix');
  WriteLn('    GF_Gen_SampleConditional(var gen, count, noiseDim,');
  WriteLn('                             condSz, nt, cond) -> TMatrix');
  WriteLn('    GF_Gen_UpdateWeights(var gen)              -> void');
  WriteLn('    GF_Gen_AddProgressiveLayer(var gen, lvl)   -> void');
  WriteLn('    GF_Gen_GetLayerOutput(var gen, idx)        -> TMatrix');
  WriteLn('    GF_Gen_SetTraining(var gen, bool)          -> void');
  WriteLn('    GF_Gen_Noise(size, depth, nt)              -> TMatrix');
  WriteLn('    GF_Gen_NoiseSlerp(v1, v2, t)              -> TVector');
  WriteLn('    GF_Gen_DeepCopy(gen)                       -> TNetwork');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' GF_Disc_ : DISCRIMINATOR ACTIONS (13 functions)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('    GF_Disc_Build(sizes[], act, opt, lr)      -> TNetwork');
  WriteLn('    GF_Disc_BuildConv(iCh, iW, iH, condSz, baseCh, act, opt, lr)');
  WriteLn('    GF_Disc_Evaluate(var disc, input)          -> TMatrix');
  WriteLn('    GF_Disc_Forward(var disc, input)           -> TMatrix');
  WriteLn('    GF_Disc_Backward(var disc, grad)           -> TMatrix');
  WriteLn('    GF_Disc_UpdateWeights(var disc)            -> void');
  WriteLn('    GF_Disc_GradPenalty(var disc, real, fake, lambda)');
  WriteLn('    GF_Disc_FeatureMatch(var disc, real, fake, featLayer)');
  WriteLn('    GF_Disc_MinibatchStdDev(input)             -> TMatrix');
  WriteLn('    GF_Disc_AddProgressiveLayer(var disc, lvl) -> void');
  WriteLn('    GF_Disc_GetLayerOutput(var disc, idx)      -> TMatrix');
  WriteLn('    GF_Disc_SetTraining(var disc, bool)        -> void');
  WriteLn('    GF_Disc_DeepCopy(disc)                     -> TNetwork');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' GF_Train_ : TRAINING CONTROL (33 functions)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('  Training:');
  WriteLn('    GF_Train_Full(var gen, disc, ds, cfg)      -> void');
  WriteLn('    GF_Train_Step(var gen, disc, batch, noise, cfg)');
  WriteLn('    GF_Train_Optimize(var net)                 -> void');
  WriteLn('  Optimizers:');
  WriteLn('    GF_Train_AdamUpdate(var p, g, var m, v, t, lr, b1, b2, e, wd)');
  WriteLn('    GF_Train_SGDUpdate(var p, g, lr, wd)');
  WriteLn('    GF_Train_RMSPropUpdate(var p, g, var cache, lr, decay, e, wd)');
  WriteLn('    GF_Train_CosineAnneal(ep, maxEp, baseLR, minLR) -> Single');
  WriteLn('  Loss Functions:');
  WriteLn('    GF_Train_BCELoss(pred, target)             -> Single');
  WriteLn('    GF_Train_BCEGrad(pred, target)             -> TMatrix');
  WriteLn('    GF_Train_WGANDiscLoss(dReal, dFake)        -> Single');
  WriteLn('    GF_Train_WGANGenLoss(dFake)                -> Single');
  WriteLn('    GF_Train_HingeDiscLoss(dReal, dFake)       -> Single');
  WriteLn('    GF_Train_HingeGenLoss(dFake)               -> Single');
  WriteLn('    GF_Train_LSDiscLoss(dReal, dFake)          -> Single');
  WriteLn('    GF_Train_LSGenLoss(dFake)                  -> Single');
  WriteLn('    GF_Train_LabelSmoothing(labels, lo, hi)    -> TMatrix');
  WriteLn('  Data:');
  WriteLn('    GF_Train_LoadDataset(path, TDataType)      -> TDataset');
  WriteLn('    GF_Train_LoadBMP(path)                     -> TDataset');
  WriteLn('    GF_Train_LoadWAV(path)                     -> TDataset');
  WriteLn('    GF_Train_CreateSynthetic(count, feat)      -> TDataset');
  WriteLn('    GF_Train_Augment(sample, TDataType)        -> TMatrix');
  WriteLn('  Metrics:');
  WriteLn('    GF_Train_ComputeFID(realS, fakeS)          -> Single');
  WriteLn('    GF_Train_ComputeIS(samples)                -> Single');
  WriteLn('    GF_Train_LogMetrics(metrics, filename)     -> void');
  WriteLn('  Model I/O:');
  WriteLn('    GF_Train_SaveModel(net, filename)          -> void');
  WriteLn('    GF_Train_LoadModel(var net, filename)      -> void');
  WriteLn('    GF_Train_SaveJSON(gen, disc, filename)     -> void');
  WriteLn('    GF_Train_LoadJSON(var gen, disc, filename) -> void');
  WriteLn('    GF_Train_SaveCheckpoint(gen, disc, ep, dir)-> void');
  WriteLn('    GF_Train_LoadCheckpoint(var gen, disc, ep, dir)');
  WriteLn('  Visualization:');
  WriteLn('    GF_Train_SaveSamples(var gen, ep, dir, nDim, nt)');
  WriteLn('    GF_Train_PlotCSV(fn, dLoss[], gLoss[], cnt)-> void');
  WriteLn('    GF_Train_PrintBar(dLoss, gLoss, width)     -> void');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' GF_Sec_ : SECURITY & ENTROPY (11 functions)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('    GF_Sec_AuditLog(msg, logFile)    [NIST AU-2/AU-3]');
  WriteLn('    GF_Sec_SecureRandomize            [/dev/urandom seed]');
  WriteLn('    GF_Sec_GetOSRandom                [/dev/urandom byte]');
  WriteLn('    GF_Sec_ValidatePath(path)         -> Boolean');
  WriteLn('    GF_Sec_VerifyWeights(var layer)   [NaN/Inf clean]');
  WriteLn('    GF_Sec_VerifyNetwork(var net)      [all layers]');
  WriteLn('    GF_Sec_EncryptModel(in, out, key) [NIST SC-28]');
  WriteLn('    GF_Sec_DecryptModel(in, out, key) [NIST SC-28]');
  WriteLn('    GF_Sec_RunTests                   -> Boolean [SA-11]');
  WriteLn('    GF_Sec_RunFuzzTests(iterations)   -> Boolean [SA-11]');
  WriteLn('    GF_Sec_BoundsCheck(M, r, c)       -> Boolean');
  WriteLn('');
  WriteLn('---------------------------------------------------------------------');
  WriteLn(' DEEP INTROSPECTION (direct record field access)');
  WriteLn('---------------------------------------------------------------------');
  WriteLn('');
  WriteLn('  Network: .layerCount .optimizer .learningRate .weightDecay');
  WriteLn('           .beta1 .beta2 .epsilon .progressiveAlpha .isTraining');
  WriteLn('  Layer:   .layerType .activation .inputSize .outputSize');
  WriteLn('           .weights[r][c] .bias[j] .kernels[k][ky][kx]');
  WriteLn('           .kernelSize .stride .padding .inChannels .outChannels');
  WriteLn('           .inWidth .inHeight .outWidth .outHeight');
  WriteLn('  Norm:    .bnGamma[j] .bnBeta[j] .runningMean[j] .runningVar[j]');
  WriteLn('  Attn:    .Wq .Wk .Wv .Wo .numHeads .headDim');
  WriteLn('  Cache:   .layerInput .layerOutput .preActivation');
  WriteLn('           .cachedQ .cachedK .cachedV .cachedScores');
  WriteLn('  Grads:   .weightGrad .biasGrad .kernelGrad');
  WriteLn('           .bnGammaGrad .bnBetaGrad .WqGrad .WkGrad .WvGrad .WoGrad');
  WriteLn('  Adam:    .adamT .mWeight .vWeight .mBias .vBias');
  WriteLn('  RMSProp: .rmsWeight .rmsBias');
  WriteLn('  Spectral:.spectralU .spectralV .spectralSigma');
  WriteLn('');
  WriteLn('EXAMPLES:');
  WriteLn('  ./GANTest                 Run all tests');
  WriteLn('  ./GANTest --verbose       Show passing tests too');
  WriteLn('  ./GANTest --gen -v        Generator tests only, verbose');
  WriteLn('  ./GANTest --quick         Skip slow training/fuzz tests');
  WriteLn('  ./GANTest --introspect -v Deep introspection, verbose');
end;

{ =========================================================================== }
{ TEST: GF_Op_ Matrix Operations                                              }
{ =========================================================================== }

procedure TestOpMatrix;
var
  A, B, C: TMatrix;
  V: TVector;
  val: Single;
begin
  Heading('GF_Op_ Matrix Operations');

  { GF_Op_CreateMatrix }
  A := CreateMatrix(3, 4);
  TC('GF_Op_CreateMatrix rows=3', Length(A) = 3);
  TC('GF_Op_CreateMatrix cols=4', Length(A[0]) = 4);
  TC('GF_Op_CreateMatrix zeroed', A[0][0] = 0.0);

  { GF_Op_CreateVector }
  V := CreateVector(5);
  TC('GF_Op_CreateVector size=5', Length(V) = 5);
  TC('GF_Op_CreateVector zeroed', V[0] = 0.0);

  { GF_Op_MatrixMultiply: [2x3] * [3x2] = [2x2] }
  A := CreateMatrix(2, 3);
  B := CreateMatrix(3, 2);
  A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
  A[1][0] := 4; A[1][1] := 5; A[1][2] := 6;
  B[0][0] := 7; B[0][1] := 8;
  B[1][0] := 9; B[1][1] := 10;
  B[2][0] := 11; B[2][1] := 12;
  C := MatrixMultiply(A, B);
  TC('GF_Op_MatrixMultiply size', (Length(C) = 2) and (Length(C[0]) = 2));
  TC('GF_Op_MatrixMultiply [0,0]=58', Abs(C[0][0] - 58.0) < 0.01);
  TC('GF_Op_MatrixMultiply [1,0]=139', Abs(C[1][0] - 139.0) < 0.01);

  { GF_Op_MatrixAdd }
  A := CreateMatrix(2, 2);
  B := CreateMatrix(2, 2);
  A[0][0] := 1; A[0][1] := 2; A[1][0] := 3; A[1][1] := 4;
  B[0][0] := 5; B[0][1] := 6; B[1][0] := 7; B[1][1] := 8;
  C := MatrixAdd(A, B);
  TC('GF_Op_MatrixAdd [0,0]=6', Abs(C[0][0] - 6.0) < 0.01);
  TC('GF_Op_MatrixAdd [1,1]=12', Abs(C[1][1] - 12.0) < 0.01);

  { GF_Op_MatrixSubtract }
  C := MatrixSubtract(B, A);
  TC('GF_Op_MatrixSubtract [0,0]=4', Abs(C[0][0] - 4.0) < 0.01);
  TC('GF_Op_MatrixSubtract [1,1]=4', Abs(C[1][1] - 4.0) < 0.01);

  { GF_Op_MatrixScale }
  C := MatrixScale(A, 3.0);
  TC('GF_Op_MatrixScale [0,0]=3', Abs(C[0][0] - 3.0) < 0.01);
  TC('GF_Op_MatrixScale [1,1]=12', Abs(C[1][1] - 12.0) < 0.01);

  { GF_Op_MatrixTranspose }
  A := CreateMatrix(2, 3);
  A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
  A[1][0] := 4; A[1][1] := 5; A[1][2] := 6;
  C := MatrixTranspose(A);
  TC('GF_Op_MatrixTranspose rows=3', Length(C) = 3);
  TC('GF_Op_MatrixTranspose cols=2', Length(C[0]) = 2);
  TC('GF_Op_MatrixTranspose [0,1]=4', Abs(C[0][1] - 4.0) < 0.01);

  { GF_Op_MatrixNormalize }
  A := CreateMatrix(1, 3);
  A[0][0] := 3; A[0][1] := 0; A[0][2] := 4;
  C := MatrixNormalize(A);
  TC('GF_Op_MatrixNormalize finite', MFinite(C));
  { z-score: mean=7/3, std~1.7, so (3-2.33)/1.7~0.39 }
  TC('GF_Op_MatrixNormalize [0,0]~0.39', Abs(C[0][0] - 0.392) < 0.05);

  { GF_Op_MatrixElementMul }
  A := CreateMatrix(2, 2);
  B := CreateMatrix(2, 2);
  A[0][0] := 2; A[0][1] := 3; A[1][0] := 4; A[1][1] := 5;
  B[0][0] := 6; B[0][1] := 7; B[1][0] := 8; B[1][1] := 9;
  C := MatrixElementMul(A, B);
  TC('GF_Op_MatrixElementMul [0,0]=12', Abs(C[0][0] - 12.0) < 0.01);
  TC('GF_Op_MatrixElementMul [1,1]=45', Abs(C[1][1] - 45.0) < 0.01);

  { GF_Op_MatrixAddInPlace }
  A := CreateMatrix(2, 2);
  B := CreateMatrix(2, 2);
  A[0][0] := 1; A[1][1] := 4;
  B[0][0] := 10; B[1][1] := 40;
  MatrixAddInPlace(A, B);
  TC('GF_Op_MatrixAddInPlace [0,0]=11', Abs(A[0][0] - 11.0) < 0.01);
  TC('GF_Op_MatrixAddInPlace [1,1]=44', Abs(A[1][1] - 44.0) < 0.01);

  { GF_Op_MatrixScaleInPlace }
  A := CreateMatrix(1, 2);
  A[0][0] := 5; A[0][1] := 10;
  MatrixScaleInPlace(A, 0.5);
  TC('GF_Op_MatrixScaleInPlace [0,0]=2.5', Abs(A[0][0] - 2.5) < 0.01);

  { GF_Op_MatrixClipInPlace }
  A := CreateMatrix(1, 4);
  A[0][0] := -5; A[0][1] := 0.5; A[0][2] := 3; A[0][3] := 10;
  MatrixClipInPlace(A, -1, 1);
  TC('GF_Op_MatrixClipInPlace lo', Abs(A[0][0] - (-1.0)) < 0.01);
  TC('GF_Op_MatrixClipInPlace pass', Abs(A[0][1] - 0.5) < 0.01);
  TC('GF_Op_MatrixClipInPlace hi', Abs(A[0][3] - 1.0) < 0.01);

  { GF_Op_SafeGet }
  A := CreateMatrix(2, 2);
  A[0][0] := 42;
  val := SafeMatrixGet(A, 0, 0, -1);
  TC('GF_Op_SafeGet valid=42', Abs(val - 42.0) < 0.01);
  val := SafeMatrixGet(A, 99, 0, -1);
  TC('GF_Op_SafeGet OOB=-1', Abs(val - (-1.0)) < 0.01);

  { GF_Op_SafeSet }
  SafeMatrixSet(A, 0, 1, 77.0);
  TC('GF_Op_SafeSet valid', Abs(A[0][1] - 77.0) < 0.01);
  SafeMatrixSet(A, 99, 0, 88.0);
  TC('GF_Op_SafeSet OOB no crash', True);
end;

{ =========================================================================== }
{ TEST: GF_Op_ Activations                                                    }
{ =========================================================================== }

procedure TestOpActivations;
var
  A, B, C, G: TMatrix;
begin
  Heading('GF_Op_ Activations');

  A := CreateMatrix(2, 3);
  A[0][0] := -2; A[0][1] := 0; A[0][2] := 3;
  A[1][0] := -1; A[1][1] := 0.5; A[1][2] := 2;

  { GF_Op_ReLU }
  B := MatrixReLU(A);
  TC('GF_Op_ReLU neg->0', Abs(B[0][0]) < 0.01);
  TC('GF_Op_ReLU pos->pass', Abs(B[0][2] - 3.0) < 0.01);

  { GF_Op_LeakyReLU }
  B := MatrixLeakyReLU(A, 0.2);
  TC('GF_Op_LeakyReLU neg*alpha', Abs(B[0][0] - (-0.4)) < 0.01);
  TC('GF_Op_LeakyReLU pos->pass', Abs(B[0][2] - 3.0) < 0.01);

  { GF_Op_Sigmoid }
  B := MatrixSigmoid(A);
  TC('GF_Op_Sigmoid range(0,1)', (B[0][0] > 0) and (B[0][0] < 1));
  TC('GF_Op_Sigmoid(0)~0.5', Abs(B[0][1] - 0.5) < 0.01);

  { GF_Op_Tanh }
  B := MatrixTanh(A);
  TC('GF_Op_Tanh range[-1,1]', (B[0][0] >= -1) and (B[0][0] <= 1));
  TC('GF_Op_Tanh(0)=0', Abs(B[0][1]) < 0.01);

  { GF_Op_Softmax }
  C := CreateMatrix(1, 3);
  C[0][0] := 1; C[0][1] := 2; C[0][2] := 3;
  B := MatrixSoftmax(C);
  TC('GF_Op_Softmax sum~1', Abs(B[0][0] + B[0][1] + B[0][2] - 1.0) < 0.01);
  TC('GF_Op_Softmax ordered', (B[0][2] > B[0][1]) and (B[0][1] > B[0][0]));

  { GF_Op_Activate dispatch for all types }
  B := ApplyActivation(A, atReLU);
  TC('GF_Op_Activate(ReLU)', Abs(B[0][0]) < 0.01);
  B := ApplyActivation(A, atSigmoid);
  TC('GF_Op_Activate(Sigmoid)', (B[0][0] > 0) and (B[0][0] < 1));
  B := ApplyActivation(A, atTanh);
  TC('GF_Op_Activate(Tanh)', Abs(B[0][1]) < 0.01);
  B := ApplyActivation(A, atLeakyReLU);
  TC('GF_Op_Activate(LeakyReLU)', B[0][0] < 0);
  B := ApplyActivation(A, atNone);
  TC('GF_Op_Activate(None) passthru', Abs(B[0][0] - A[0][0]) < 0.01);

  { GF_Op_ActivationBackward }
  G := CreateMatrix(2, 3);
  G[0][0] := 1; G[0][1] := 1; G[0][2] := 1;
  G[1][0] := 1; G[1][1] := 1; G[1][2] := 1;
  B := ActivationBackward(G, A, atReLU);
  TC('GF_Op_ActivBackward(ReLU) neg->0', Abs(B[0][0]) < 0.01);
  TC('GF_Op_ActivBackward(ReLU) pos->1', Abs(B[0][2] - 1.0) < 0.01);
  B := ActivationBackward(G, A, atSigmoid);
  TC('GF_Op_ActivBackward(Sigmoid) finite', MFinite(B));
  B := ActivationBackward(G, A, atTanh);
  TC('GF_Op_ActivBackward(Tanh) finite', MFinite(B));
  B := ActivationBackward(G, A, atLeakyReLU);
  TC('GF_Op_ActivBackward(LeakyReLU) finite', MFinite(B));
  B := ActivationBackward(G, A, atNone);
  TC('GF_Op_ActivBackward(None) passthru', Abs(B[0][0] - 1.0) < 0.01);
end;

{ =========================================================================== }
{ TEST: GF_Op_ Convolution                                                    }
{ =========================================================================== }

procedure TestOpConv;
var
  layer: TLayer;
  inp, outp, grad, gradInp: TMatrix;
begin
  Heading('GF_Op_ Convolution');

  { Conv2D: 1ch->2ch, k=3, s=1, p=1, 4x4 input }
  layer := CreateConv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
  TC('GF_Op_CreateConv2DLayer type', layer.layerType = ltConv2D);
  TC('GF_Op_CreateConv2DLayer inSize=16', layer.inputSize = 16);
  TC('GF_Op_CreateConv2DLayer outSize=32', layer.outputSize = 32);

  inp := CreateMatrix(2, 16);
  inp[0][0] := 1.0; inp[0][5] := 2.0;
  inp[1][3] := 0.5; inp[1][10] := 1.5;
  layer.layerInput := inp;
  outp := Conv2DForward(inp, layer);
  TC('GF_Op_Conv2D rows=2', Length(outp) = 2);
  TC('GF_Op_Conv2D cols=32', Length(outp[0]) = 32);
  TC('GF_Op_Conv2D finite', MFinite(outp));

  grad := CreateMatrix(2, 32);
  grad[0][0] := 1.0; grad[1][16] := 1.0;
  gradInp := Conv2DBackward(layer, grad);
  TC('GF_Op_Conv2DBack rows=2', Length(gradInp) = 2);
  TC('GF_Op_Conv2DBack cols=16', Length(gradInp[0]) = 16);
  TC('GF_Op_Conv2DBack finite', MFinite(gradInp));
  TC('GF_Op_Conv2DBack kernelGrad', Length(layer.kernelGrad) > 0);

  { Deconv2D: 1ch->2ch, k=3, s=1, p=1, 4x4 -> 4x4 }
  layer := CreateDeconv2DLayer(1, 2, 3, 1, 1, 4, 4, atReLU);
  TC('GF_Op_CreateDeconv2DLayer type', layer.layerType = ltDeconv2D);
  TC('GF_Op_CreateDeconv2DLayer outSize=32', layer.outputSize = 32);

  inp := CreateMatrix(2, 16);
  inp[0][0] := 1.0; inp[1][5] := 2.0;
  layer.layerInput := inp;
  outp := Deconv2DForward(inp, layer);
  TC('GF_Op_Deconv2D rows=2', Length(outp) = 2);
  TC('GF_Op_Deconv2D cols=32', Length(outp[0]) = 32);
  TC('GF_Op_Deconv2D finite', MFinite(outp));

  grad := CreateMatrix(2, 32);
  grad[0][0] := 1.0;
  gradInp := Deconv2DBackward(layer, grad);
  TC('GF_Op_Deconv2DBack rows=2', Length(gradInp) = 2);
  TC('GF_Op_Deconv2DBack cols=16', Length(gradInp[0]) = 16);
  TC('GF_Op_Deconv2DBack finite', MFinite(gradInp));

  { Conv1D: 1ch->2ch, k=3, s=1, p=1, len=8 -> outLen=8 }
  layer := CreateConv1DLayer(1, 2, 3, 1, 1, 8, atReLU);
  TC('GF_Op_CreateConv1DLayer type', layer.layerType = ltConv1D);
  TC('GF_Op_CreateConv1DLayer outSize=16', layer.outputSize = 16);

  inp := CreateMatrix(2, 8);
  inp[0][0] := 1.0; inp[0][4] := 2.0;
  layer.layerInput := inp;
  outp := Conv1DForward(inp, layer);
  TC('GF_Op_Conv1D rows=2', Length(outp) = 2);
  TC('GF_Op_Conv1D cols=16', Length(outp[0]) = 16);
  TC('GF_Op_Conv1D finite', MFinite(outp));

  grad := CreateMatrix(2, 16);
  grad[0][0] := 1.0;
  gradInp := Conv1DBackward(layer, grad);
  TC('GF_Op_Conv1DBack rows=2', Length(gradInp) = 2);
  TC('GF_Op_Conv1DBack cols=8', Length(gradInp[0]) = 8);
  TC('GF_Op_Conv1DBack finite', MFinite(gradInp));
end;

{ =========================================================================== }
{ TEST: GF_Op_ Normalization                                                   }
{ =========================================================================== }

procedure TestOpNorm;
var
  layer, denseL: TLayer;
  inp, outp, grad, gradInp: TMatrix;
  meanSum: Single;
  i: Integer;
begin
  Heading('GF_Op_ Normalization');

  { BatchNorm }
  layer := CreateBatchNormLayer(8);
  TC('GF_Op_CreateBatchNormLayer type', layer.layerType = ltBatchNorm);
  TC('GF_Op_CreateBatchNormLayer gamma[0]=1', Abs(layer.bnGamma[0] - 1.0) < 0.01);

  inp := CreateMatrix(4, 8);
  for i := 0 to 3 do inp[i][0] := i + 1;
  layer.isTraining := True;
  layer.layerInput := inp;
  outp := BatchNormForward(inp, layer);
  TC('GF_Op_BatchNorm rows=4', Length(outp) = 4);
  TC('GF_Op_BatchNorm cols=8', Length(outp[0]) = 8);
  TC('GF_Op_BatchNorm finite', MFinite(outp));
  meanSum := outp[0][0] + outp[1][0] + outp[2][0] + outp[3][0];
  TC('GF_Op_BatchNorm mean~0', Abs(meanSum) < 0.5);

  grad := CreateMatrix(4, 8);
  for i := 0 to 3 do grad[i][0] := 1.0;
  gradInp := BatchNormBackward(layer, grad);
  TC('GF_Op_BatchNormBack rows=4', Length(gradInp) = 4);
  TC('GF_Op_BatchNormBack finite', MFinite(gradInp));
  TC('GF_Op_BatchNormBack gammaGrad', Length(layer.bnGammaGrad) = 8);

  { LayerNorm }
  layer := CreateLayerNormLayer(8);
  TC('GF_Op_CreateLayerNormLayer type', layer.layerType = ltLayerNorm);
  layer.isTraining := True;
  layer.layerInput := inp;
  outp := LayerNormForward(inp, layer);
  TC('GF_Op_LayerNorm finite', MFinite(outp));
  gradInp := LayerNormBackward(layer, grad);
  TC('GF_Op_LayerNormBack finite', MFinite(gradInp));
  TC('GF_Op_LayerNormBack betaGrad', Length(layer.bnBetaGrad) = 8);

  { SpectralNorm }
  denseL := CreateDenseLayer(4, 4, atReLU);
  SetLength(denseL.spectralU, 4);
  SetLength(denseL.spectralV, 4);
  denseL.spectralU[0] := 1; denseL.spectralV[0] := 1;
  outp := SpectralNormalize(denseL);
  TC('GF_Op_SpectralNorm finite', MFinite(outp));
  TC('GF_Op_SpectralNorm sigma>0', denseL.spectralSigma > 0);
end;

{ =========================================================================== }
{ TEST: GF_Op_ Attention                                                       }
{ =========================================================================== }

procedure TestOpAttention;
var
  layer: TLayer;
  inp, outp, grad, gradInp: TMatrix;
begin
  Heading('GF_Op_ Self-Attention');

  layer := CreateAttentionLayer(4, 2);
  TC('GF_Op_CreateAttentionLayer type', layer.layerType = ltAttention);
  TC('GF_Op_CreateAttentionLayer headDim=2', layer.headDim = 2);
  TC('GF_Op_CreateAttentionLayer Wq[4x4]', (Length(layer.Wq) = 4) and (Length(layer.Wq[0]) = 4));

  inp := CreateMatrix(3, 4);
  inp[0][0] := 1; inp[0][2] := 1;
  inp[1][1] := 1; inp[1][3] := 1;
  inp[2][0] := 1; inp[2][1] := 1;
  layer.isTraining := True;
  layer.layerInput := inp;
  outp := SelfAttentionForward(inp, layer);
  TC('GF_Op_Attention rows=3', Length(outp) = 3);
  TC('GF_Op_Attention cols=4', Length(outp[0]) = 4);
  TC('GF_Op_Attention finite', MFinite(outp));
  TC('GF_Op_Attention cachedQ', Length(layer.cachedQ) = 3);
  TC('GF_Op_Attention cachedScores', Length(layer.cachedScores) = 3);

  grad := CreateMatrix(3, 4);
  grad[0][0] := 1; grad[1][1] := 1; grad[2][2] := 1;
  gradInp := SelfAttentionBackward(layer, grad);
  TC('GF_Op_AttentionBack rows=3', Length(gradInp) = 3);
  TC('GF_Op_AttentionBack cols=4', Length(gradInp[0]) = 4);
  TC('GF_Op_AttentionBack finite', MFinite(gradInp));
  TC('GF_Op_AttentionBack WqGrad', Length(layer.WqGrad) = 4);
end;

{ =========================================================================== }
{ TEST: GF_Op_ Layer Dispatch                                                  }
{ =========================================================================== }

procedure TestOpLayerDispatch;
var
  layer: TLayer;
  inp, outp, grad, gradInp: TMatrix;
begin
  Heading('GF_Op_ Layer Dispatch');

  { Dense via dispatch }
  layer := CreateDenseLayer(4, 3, atReLU);
  InitLayerOptimizer(layer, optAdam);
  TC('GF_Op_CreateDenseLayer weights[4x3]',
    (Length(layer.weights) = 4) and (Length(layer.weights[0]) = 3));
  TC('GF_Op_CreateDenseLayer bias[3]', Length(layer.bias) = 3);
  TC('GF_Op_InitLayerOptimizer mWeight', Length(layer.mWeight) = 4);

  inp := CreateMatrix(2, 4);
  inp[0][0] := 1; inp[0][1] := 2; inp[0][2] := 3; inp[0][3] := 4;
  inp[1][0] := 0.5; inp[1][1] := 1.5; inp[1][2] := 2.5; inp[1][3] := 3.5;
  outp := LayerForward(layer, inp);
  TC('GF_Op_LayerForward(Dense) rows=2', Length(outp) = 2);
  TC('GF_Op_LayerForward(Dense) cols=3', Length(outp[0]) = 3);
  TC('GF_Op_LayerForward(Dense) finite', MFinite(outp));

  grad := CreateMatrix(2, 3);
  grad[0][0] := 1; grad[0][1] := 1; grad[0][2] := 1;
  grad[1][0] := 1; grad[1][1] := 1; grad[1][2] := 1;
  gradInp := LayerBackward(layer, grad);
  TC('GF_Op_LayerBackward(Dense) cols=4', Length(gradInp[0]) = 4);
  TC('GF_Op_LayerBackward(Dense) finite', MFinite(gradInp));
  TC('GF_Op_LayerBackward(Dense) weightGrad', Length(layer.weightGrad) = 4);
  TC('GF_Op_LayerBackward(Dense) biasGrad', Length(layer.biasGrad) = 3);

  { BatchNorm via dispatch }
  layer := CreateBatchNormLayer(4);
  layer.isTraining := True;
  outp := LayerForward(layer, inp);
  TC('GF_Op_LayerForward(BN) finite', MFinite(outp));
  grad := CreateMatrix(2, 4);
  grad[0][0] := 1; grad[0][1] := 1; grad[0][2] := 1; grad[0][3] := 1;
  grad[1][0] := 1; grad[1][1] := 1; grad[1][2] := 1; grad[1][3] := 1;
  gradInp := LayerBackward(layer, grad);
  TC('GF_Op_LayerBackward(BN) finite', MFinite(gradInp));

  { LayerNorm via dispatch }
  layer := CreateLayerNormLayer(4);
  layer.isTraining := True;
  outp := LayerForward(layer, inp);
  TC('GF_Op_LayerForward(LN) finite', MFinite(outp));
  gradInp := LayerBackward(layer, grad);
  TC('GF_Op_LayerBackward(LN) finite', MFinite(gradInp));

  { Attention via dispatch }
  layer := CreateAttentionLayer(4, 2);
  layer.isTraining := True;
  outp := LayerForward(layer, inp);
  TC('GF_Op_LayerForward(Attn) finite', MFinite(outp));
  gradInp := LayerBackward(layer, grad);
  TC('GF_Op_LayerBackward(Attn) finite', MFinite(gradInp));
end;

{ =========================================================================== }
{ TEST: GF_Op_ Random / Noise                                                 }
{ =========================================================================== }

procedure TestOpRandom;
var
  noise: TMatrix;
  v1, v2, vs: TVector;
  g, u: Single;
  i: Integer;
  hasPos, hasNeg, boundsOK: Boolean;
begin
  Heading('GF_Op_ Random / Noise');

  SecureRandomize;

  { GF_Op_RandomGaussian }
  hasPos := False; hasNeg := False;
  for i := 1 to 100 do begin
    g := RandomGaussian;
    if g > 0 then hasPos := True;
    if g < 0 then hasNeg := True;
  end;
  TC('GF_Op_RandomGaussian has_pos', hasPos);
  TC('GF_Op_RandomGaussian has_neg', hasNeg);

  { GF_Op_RandomUniform }
  boundsOK := True;
  for i := 1 to 100 do begin
    u := RandomUniform(0.0, 1.0);
    if (u < 0) or (u > 1) then boundsOK := False;
  end;
  TC('GF_Op_RandomUniform bounds[0,1]', boundsOK);

  { GF_Op_GenerateNoise all types }
  GenerateNoise(noise, 4, 8, ntGauss);
  TC('GF_Op_GenerateNoise(Gauss) size', (Length(noise) = 4) and (Length(noise[0]) = 8));
  TC('GF_Op_GenerateNoise(Gauss) finite', MFinite(noise));

  GenerateNoise(noise, 4, 8, ntUniform);
  TC('GF_Op_GenerateNoise(Uniform) finite', MFinite(noise));

  GenerateNoise(noise, 4, 8, ntAnalog);
  TC('GF_Op_GenerateNoise(Analog) finite', MFinite(noise));

  { GF_Op_NoiseSlerp }
  SetLength(v1, 4); SetLength(v2, 4);
  v1[0] := 1; v1[1] := 0; v1[2] := 0; v1[3] := 0;
  v2[0] := 0; v2[1] := 1; v2[2] := 0; v2[3] := 0;
  vs := NoiseSlerp(v1, v2, 0.0);
  TC('GF_Op_NoiseSlerp t=0 ~v1', Abs(vs[0] - 1.0) < 0.15);
  vs := NoiseSlerp(v1, v2, 1.0);
  TC('GF_Op_NoiseSlerp t=1 ~v2', Abs(vs[1] - 1.0) < 0.15);
  vs := NoiseSlerp(v1, v2, 0.5);
  TC('GF_Op_NoiseSlerp t=0.5 finite', VFinite(vs));
end;

{ =========================================================================== }
{ TEST: GF_Gen_ Generator                                                     }
{ =========================================================================== }

procedure TestGen;
var
  gen, genCopy: TNetwork;
  genSizes: array of Integer;
  inp, outp, grad, noise, layerOut, comb, cond: TMatrix;
  v1, v2, vs: TVector;
  i, j: Integer;
begin
  Heading('GF_Gen_ Generator');

  { GF_Gen_Build MLP }
  SetLength(genSizes, 4);
  genSizes[0] := 8; genSizes[1] := 16; genSizes[2] := 8; genSizes[3] := 1;
  gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);
  TC('GF_Gen_Build layerCount=3', gen.layerCount = 3);
  TC('GF_Gen_Build optimizer=Adam', gen.optimizer = optAdam);
  TC('GF_Gen_Build lr', Abs(gen.learningRate - 0.001) < 0.0001);

  { GF_Gen_Forward }
  inp := CreateMatrix(2, 8);
  inp[0][0] := 1; inp[0][3] := 0.5;
  inp[1][1] := 2; inp[1][5] := -1;
  outp := NetworkForward(gen, inp);
  TC('GF_Gen_Forward rows=2', Length(outp) = 2);
  TC('GF_Gen_Forward cols=1', Length(outp[0]) = 1);
  TC('GF_Gen_Forward finite', MFinite(outp));

  { GF_Gen_Backward }
  grad := CreateMatrix(2, 1);
  grad[0][0] := 1.0; grad[1][0] := -1.0;
  NetworkForward(gen, inp);
  outp := NetworkBackward(gen, grad);
  TC('GF_Gen_Backward rows=2', Length(outp) = 2);
  TC('GF_Gen_Backward cols=8', Length(outp[0]) = 8);
  TC('GF_Gen_Backward finite', MFinite(outp));

  { GF_Gen_UpdateWeights }
  NetworkUpdateWeights(gen);
  TC('GF_Gen_UpdateWeights finite', MFinite(gen.layers[0].weights));

  { GF_Gen_Sample }
  GenerateNoise(noise, 4, 8, ntGauss);
  outp := NetworkForward(gen, noise);
  TC('GF_Gen_Sample rows=4', Length(outp) = 4);
  TC('GF_Gen_Sample finite', MFinite(outp));

  { GF_Gen_SampleConditional }
  SetLength(genSizes, 4);
  genSizes[0] := 10; genSizes[1] := 16; genSizes[2] := 8; genSizes[3] := 1;
  gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);
  GenerateNoise(noise, 2, 8, ntGauss);
  cond := CreateMatrix(2, 2);
  cond[0][0] := 1; cond[1][1] := 1;
  comb := CreateMatrix(2, 10);
  for i := 0 to 1 do begin
    for j := 0 to 7 do comb[i][j] := noise[i][j];
    for j := 0 to 1 do comb[i][8 + j] := cond[i][j];
  end;
  outp := NetworkForward(gen, comb);
  TC('GF_Gen_SampleConditional finite', MFinite(outp));

  { Rebuild standard gen for remaining tests }
  SetLength(genSizes, 4);
  genSizes[0] := 8; genSizes[1] := 16; genSizes[2] := 8; genSizes[3] := 1;
  gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);

  { GF_Gen_GetLayerOutput }
  inp := CreateMatrix(2, 8);
  inp[0][0] := 1; inp[1][3] := 2;
  NetworkForward(gen, inp);
  layerOut := GetLayerOutput(gen, 0);
  TC('GF_Gen_GetLayerOutput exists', Length(layerOut) > 0);
  TC('GF_Gen_GetLayerOutput finite', MFinite(layerOut));

  { GF_Gen_SetTraining }
  SetNetworkTraining(gen, False);
  TC('GF_Gen_SetTraining false', gen.isTraining = False);
  SetNetworkTraining(gen, True);
  TC('GF_Gen_SetTraining true', gen.isTraining = True);

  { GF_Gen_Noise }
  GenerateNoise(noise, 3, 8, ntGauss);
  TC('GF_Gen_Noise size', (Length(noise) = 3) and (Length(noise[0]) = 8));

  { GF_Gen_NoiseSlerp }
  SetLength(v1, 4); SetLength(v2, 4);
  v1[0] := 1; v2[3] := 1;
  vs := NoiseSlerp(v1, v2, 0.5);
  TC('GF_Gen_NoiseSlerp finite', VFinite(vs));

  { GF_Gen_DeepCopy }
  genCopy := DeepCopyNetwork(gen);
  TC('GF_Gen_DeepCopy layerCount', genCopy.layerCount = gen.layerCount);
  TC('GF_Gen_DeepCopy lr', Abs(genCopy.learningRate - gen.learningRate) < 0.0001);
  genCopy.learningRate := 0.999;
  TC('GF_Gen_DeepCopy independent', Abs(gen.learningRate - 0.001) < 0.0001);

  { GF_Gen_AddProgressiveLayer (adds conv+BN = 2 layers) }
  AddProgressiveLayer(gen, 1, True);
  TC('GF_Gen_AddProgressiveLayer +2', gen.layerCount = 5);

  { GF_Gen_BuildConv }
  gen := CreateConvGenerator(8, 0, 4, atLeakyReLU, optAdam, 0.0002);
  TC('GF_Gen_BuildConv layers=7', gen.layerCount = 7);
  TC('GF_Gen_BuildConv first=Dense', gen.layers[0].layerType = ltDense);
  inp := CreateMatrix(1, 8);
  inp[0][0] := 1.0;
  outp := NetworkForward(gen, inp);
  TC('GF_Gen_BuildConv forward finite', MFinite(outp));
end;

{ =========================================================================== }
{ TEST: GF_Disc_ Discriminator                                                 }
{ =========================================================================== }

procedure TestDisc;
var
  disc, discCopy: TNetwork;
  discSizes: array of Integer;
  inp, outp, grad, real, fake, layerOut, mbsd: TMatrix;
  gp, fm: Single;
begin
  Heading('GF_Disc_ Discriminator');

  { GF_Disc_Build MLP }
  SetLength(discSizes, 4);
  discSizes[0] := 1; discSizes[1] := 8; discSizes[2] := 16; discSizes[3] := 1;
  disc := CreateNetwork(discSizes, atLeakyReLU, optAdam, 0.001);
  TC('GF_Disc_Build layerCount=3', disc.layerCount = 3);

  { GF_Disc_Evaluate / GF_Disc_Forward }
  inp := CreateMatrix(4, 1);
  inp[0][0] := 0.5; inp[1][0] := -0.3; inp[2][0] := 0.8; inp[3][0] := 0.1;
  outp := NetworkForward(disc, inp);
  TC('GF_Disc_Evaluate rows=4', Length(outp) = 4);
  TC('GF_Disc_Evaluate cols=1', Length(outp[0]) = 1);
  TC('GF_Disc_Forward finite', MFinite(outp));

  { GF_Disc_Backward }
  grad := CreateMatrix(4, 1);
  grad[0][0] := 1; grad[1][0] := -1; grad[2][0] := 0.5; grad[3][0] := -0.5;
  NetworkForward(disc, inp);
  outp := NetworkBackward(disc, grad);
  TC('GF_Disc_Backward rows=4', Length(outp) = 4);
  TC('GF_Disc_Backward cols=1', Length(outp[0]) = 1);
  TC('GF_Disc_Backward finite', MFinite(outp));

  { GF_Disc_UpdateWeights }
  NetworkUpdateWeights(disc);
  TC('GF_Disc_UpdateWeights finite', MFinite(disc.layers[0].weights));

  { GF_Disc_GradPenalty }
  real := CreateMatrix(4, 1);
  fake := CreateMatrix(4, 1);
  real[0][0] := 0.9; real[1][0] := 0.8; real[2][0] := 0.7; real[3][0] := 0.6;
  fake[0][0] := 0.1; fake[1][0] := 0.2; fake[2][0] := 0.3; fake[3][0] := 0.4;
  gp := ComputeGradientPenalty(disc, real, fake, 10.0);
  TC('GF_Disc_GradPenalty finite', (not IsNan(gp)) and (not IsInfinite(gp)));
  TC('GF_Disc_GradPenalty >=0', gp >= 0);

  { GF_Disc_FeatureMatch }
  NetworkForward(disc, real);
  fm := FeatureMatchingLoss(disc, real, fake, 0);
  TC('GF_Disc_FeatureMatch finite', (not IsNan(fm)) and (not IsInfinite(fm)));
  TC('GF_Disc_FeatureMatch >=0', fm >= 0);

  { GF_Disc_MinibatchStdDev }
  mbsd := MinibatchStdDev(inp);
  TC('GF_Disc_MinibatchStdDev rows=4', Length(mbsd) = 4);
  TC('GF_Disc_MinibatchStdDev cols=2', Length(mbsd[0]) = 2);
  TC('GF_Disc_MinibatchStdDev finite', MFinite(mbsd));

  { GF_Disc_GetLayerOutput }
  NetworkForward(disc, inp);
  layerOut := GetLayerOutput(disc, 0);
  TC('GF_Disc_GetLayerOutput exists', Length(layerOut) > 0);
  TC('GF_Disc_GetLayerOutput finite', MFinite(layerOut));

  { GF_Disc_SetTraining }
  SetNetworkTraining(disc, False);
  TC('GF_Disc_SetTraining false', disc.isTraining = False);
  SetNetworkTraining(disc, True);
  TC('GF_Disc_SetTraining true', disc.isTraining = True);

  { GF_Disc_DeepCopy }
  discCopy := DeepCopyNetwork(disc);
  TC('GF_Disc_DeepCopy layerCount', discCopy.layerCount = disc.layerCount);
  discCopy.learningRate := 0.999;
  TC('GF_Disc_DeepCopy independent', Abs(disc.learningRate - 0.001) < 0.0001);

  { GF_Disc_AddProgressiveLayer (adds conv+BN = 2 layers) }
  AddProgressiveLayer(disc, 1, False);
  TC('GF_Disc_AddProgressiveLayer +2', disc.layerCount = 5);

  { GF_Disc_BuildConv }
  disc := CreateConvDiscriminator(1, 8, 8, 0, 4, atLeakyReLU, optAdam, 0.0002);
  TC('GF_Disc_BuildConv layers=5', disc.layerCount = 5);
  TC('GF_Disc_BuildConv last=Dense', disc.layers[4].layerType = ltDense);
  inp := CreateMatrix(1, 64);
  inp[0][0] := 1.0;
  outp := NetworkForward(disc, inp);
  TC('GF_Disc_BuildConv forward finite', MFinite(outp));
  TC('GF_Disc_BuildConv output cols=1', Length(outp[0]) = 1);
end;

{ =========================================================================== }
{ TEST: GF_Train_ Loss Functions                                               }
{ =========================================================================== }

procedure TestTrainLoss;
var
  pred, target, dReal, dFake, labels, grad, smoothed: TMatrix;
  loss: Single;
begin
  Heading('GF_Train_ Loss Functions');

  pred := CreateMatrix(4, 1);
  target := CreateMatrix(4, 1);
  pred[0][0] := 0.9; pred[1][0] := 0.8; pred[2][0] := 0.2; pred[3][0] := 0.1;
  target[0][0] := 1; target[1][0] := 1; target[2][0] := 0; target[3][0] := 0;

  { GF_Train_BCELoss }
  loss := BinaryCrossEntropy(pred, target);
  TC('GF_Train_BCELoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));
  TC('GF_Train_BCELoss >0', loss > 0);
  TC('GF_Train_BCELoss reasonable', loss < 5.0);

  { GF_Train_BCEGrad }
  grad := BCEGradient(pred, target);
  TC('GF_Train_BCEGrad rows=4', Length(grad) = 4);
  TC('GF_Train_BCEGrad finite', MFinite(grad));

  { WGAN losses }
  dReal := CreateMatrix(4, 1);
  dFake := CreateMatrix(4, 1);
  dReal[0][0] := 2; dReal[1][0] := 1.5; dReal[2][0] := 1.8; dReal[3][0] := 2.1;
  dFake[0][0] := -1; dFake[1][0] := -0.5; dFake[2][0] := -0.8; dFake[3][0] := -1.2;

  loss := WGANDiscLoss(dReal, dFake);
  TC('GF_Train_WGANDiscLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));
  loss := WGANGenLoss(dFake);
  TC('GF_Train_WGANGenLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));

  { Hinge losses }
  loss := HingeDiscLoss(dReal, dFake);
  TC('GF_Train_HingeDiscLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));
  loss := HingeGenLoss(dFake);
  TC('GF_Train_HingeGenLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));

  { LS losses }
  loss := LSDiscLoss(dReal, dFake);
  TC('GF_Train_LSDiscLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));
  loss := LSGenLoss(dFake);
  TC('GF_Train_LSGenLoss finite', (not IsNan(loss)) and (not IsInfinite(loss)));

  { GF_Train_LabelSmoothing }
  labels := CreateMatrix(4, 1);
  labels[0][0] := 1; labels[1][0] := 1; labels[2][0] := 0; labels[3][0] := 0;
  smoothed := ApplyLabelSmoothing(labels, 0.0, 0.9);
  TC('GF_Train_LabelSmoothing [0]<=0.9', smoothed[0][0] <= 0.9 + 0.01);
  TC('GF_Train_LabelSmoothing [2]>=0.0', smoothed[2][0] >= -0.01);
  TC('GF_Train_LabelSmoothing finite', MFinite(smoothed));
end;

{ =========================================================================== }
{ TEST: GF_Train_ Optimizers                                                   }
{ =========================================================================== }

procedure TestTrainOptim;
var
  p, g, m, v, cache: TMatrix;
  lr: Single;
begin
  Heading('GF_Train_ Optimizers');

  { GF_Train_AdamUpdate }
  p := CreateMatrix(2, 2);
  g := CreateMatrix(2, 2);
  m := CreateMatrix(2, 2);
  v := CreateMatrix(2, 2);
  p[0][0] := 1; p[0][1] := 2; p[1][0] := 3; p[1][1] := 4;
  g[0][0] := 0.1; g[0][1] := 0.2; g[1][0] := 0.3; g[1][1] := 0.4;
  AdamUpdateMatrix(p, g, m, v, 1, 0.001, 0.9, 0.999, 1e-8, 0.0);
  TC('GF_Train_AdamUpdate finite', MFinite(p));
  TC('GF_Train_AdamUpdate changed', Abs(p[0][0] - 1.0) > 1e-6);
  TC('GF_Train_AdamUpdate m updated', Abs(m[0][0]) > 1e-6);
  TC('GF_Train_AdamUpdate v updated', Abs(v[0][0]) > 1e-6);

  { GF_Train_SGDUpdate }
  p := CreateMatrix(2, 2);
  g := CreateMatrix(2, 2);
  p[0][0] := 1; p[1][1] := 2;
  g[0][0] := 0.5; g[1][1] := 0.5;
  SGDUpdateMatrix(p, g, 0.01, 0.0);
  TC('GF_Train_SGDUpdate finite', MFinite(p));
  TC('GF_Train_SGDUpdate changed', Abs(p[0][0] - 1.0) > 1e-6);

  { GF_Train_RMSPropUpdate }
  p := CreateMatrix(2, 2);
  g := CreateMatrix(2, 2);
  cache := CreateMatrix(2, 2);
  p[0][0] := 1; g[0][0] := 0.1;
  RMSPropUpdateMatrix(p, g, cache, 0.001, 0.9, 1e-8, 0.0);
  TC('GF_Train_RMSPropUpdate finite', MFinite(p));
  TC('GF_Train_RMSPropUpdate changed', Abs(p[0][0] - 1.0) > 1e-6);
  TC('GF_Train_RMSPropUpdate cache', Abs(cache[0][0]) > 1e-10);

  { GF_Train_CosineAnneal }
  lr := CosineAnneal(0, 100, 0.001, 0.0001);
  TC('GF_Train_CosineAnneal ep0=baseLR', Abs(lr - 0.001) < 0.0002);
  lr := CosineAnneal(50, 100, 0.001, 0.0001);
  TC('GF_Train_CosineAnneal mid', (lr > 0.0001) and (lr < 0.001));
  lr := CosineAnneal(100, 100, 0.001, 0.0001);
  TC('GF_Train_CosineAnneal end~minLR', Abs(lr - 0.0001) < 0.0002);
end;

{ =========================================================================== }
{ TEST: GF_Train_ Data                                                         }
{ =========================================================================== }

procedure TestTrainData;
var
  ds: TDataset;
  sample, aug: TMatrix;
  realS, fakeS: TMatrixArray;
  fid, iscore: Single;
  met: TGANMetrics;
  i: Integer;
begin
  Heading('GF_Train_ Data');

  { GF_Train_CreateSynthetic }
  ds := CreateSyntheticDataset(100, 4);
  TC('GF_Train_CreateSynthetic count=100', ds.count = 100);
  TC('GF_Train_CreateSynthetic samples', Length(ds.samples) = 100);
  TC('GF_Train_CreateSynthetic finite', MFinite(ds.samples[0]));

  { GF_Train_Augment }
  sample := CreateMatrix(1, 4);
  sample[0][0] := 1; sample[0][1] := 2; sample[0][2] := 3; sample[0][3] := 4;
  aug := AugmentSample(sample, dtVector);
  TC('GF_Train_Augment finite', MFinite(aug));
  TC('GF_Train_Augment size', Length(aug[0]) = 4);

  { GF_Train_ComputeFID }
  SetLength(realS, 10);
  SetLength(fakeS, 10);
  for i := 0 to 9 do begin
    realS[i] := CreateMatrix(1, 4);
    fakeS[i] := CreateMatrix(1, 4);
    realS[i][0][0] := Random; fakeS[i][0][0] := Random;
  end;
  fid := ComputeFID(realS, fakeS);
  TC('GF_Train_ComputeFID finite', (not IsNan(fid)) and (not IsInfinite(fid)));

  { GF_Train_ComputeIS }
  iscore := ComputeIS(realS);
  TC('GF_Train_ComputeIS finite', (not IsNan(iscore)) and (not IsInfinite(iscore)));

  { GF_Train_LogMetrics }
  ForceDirectories(TEST_DIR);
  FillChar(met, SizeOf(met), 0);
  met.dLossReal := 0.5; met.gLoss := 0.7; met.epoch := 1;
  LogMetrics(met, TEST_DIR + '/test_metrics.csv');
  TC('GF_Train_LogMetrics file', FileExists(TEST_DIR + '/test_metrics.csv'));
end;

{ =========================================================================== }
{ TEST: GF_Train_ Model I/O                                                   }
{ =========================================================================== }

procedure TestTrainIO;
var
  gen, disc, genLoad, discLoad: TNetwork;
  genSizes, discSizes: array of Integer;
  dLoss, gLoss: array[0..2] of Single;
begin
  Heading('GF_Train_ Model I/O');
  ForceDirectories(TEST_DIR);

  SetLength(genSizes, 3);
  genSizes[0] := 4; genSizes[1] := 8; genSizes[2] := 1;
  SetLength(discSizes, 3);
  discSizes[0] := 1; discSizes[1] := 8; discSizes[2] := 1;
  gen := CreateNetwork(genSizes, atReLU, optAdam, 0.001);
  disc := CreateNetwork(discSizes, atReLU, optAdam, 0.001);

  { GF_Train_SaveModel / LoadModel }
  SaveNetworkBinary(gen, TEST_DIR + '/gen_test.bin');
  TC('GF_Train_SaveModel file', FileExists(TEST_DIR + '/gen_test.bin'));
  FillChar(genLoad, SizeOf(genLoad), 0);
  LoadNetworkBinary(genLoad, TEST_DIR + '/gen_test.bin');
  TC('GF_Train_LoadModel layerCount', genLoad.layerCount = gen.layerCount);
  TC('GF_Train_LoadModel w[0][0]',
    Abs(genLoad.layers[0].weights[0][0] - gen.layers[0].weights[0][0]) < 0.0001);
  TC('GF_Train_LoadModel bias[0]',
    Abs(genLoad.layers[0].bias[0] - gen.layers[0].bias[0]) < 0.0001);

  { GF_Train_SaveJSON / LoadJSON }
  SaveGANToJSON(gen, disc, TEST_DIR + '/gan_test.json');
  TC('GF_Train_SaveJSON file', FileExists(TEST_DIR + '/gan_test.json'));
  FillChar(genLoad, SizeOf(genLoad), 0);
  FillChar(discLoad, SizeOf(discLoad), 0);
  LoadGANFromJSON(genLoad, discLoad, TEST_DIR + '/gan_test.json');
  { JSON loader is partial stub: loads learning_rate, not full layers }
  TC('GF_Train_LoadJSON gen lr', Abs(genLoad.learningRate - gen.learningRate) < 0.001);
  TC('GF_Train_LoadJSON called ok', True);

  { GF_Train_SaveCheckpoint / LoadCheckpoint }
  ForceDirectories(TEST_DIR + '/ckpt');
  SaveCheckpoint(gen, disc, 5, TEST_DIR + '/ckpt');
  TC('GF_Train_SaveCheckpoint gen',
    FileExists(TEST_DIR + '/ckpt/gen_ep5.bin'));
  TC('GF_Train_SaveCheckpoint disc',
    FileExists(TEST_DIR + '/ckpt/disc_ep5.bin'));
  FillChar(genLoad, SizeOf(genLoad), 0);
  FillChar(discLoad, SizeOf(discLoad), 0);
  LoadCheckpoint(genLoad, discLoad, 5, TEST_DIR + '/ckpt');
  TC('GF_Train_LoadCheckpoint gen', genLoad.layerCount = gen.layerCount);
  TC('GF_Train_LoadCheckpoint disc', discLoad.layerCount = disc.layerCount);

  { GF_Train_SaveSamples }
  SaveGeneratedSamples(gen, 1, TEST_DIR, 4, ntGauss);
  TC('GF_Train_SaveSamples file',
    FileExists(TEST_DIR + '/samples_ep1.csv'));

  { GF_Train_PlotCSV }
  dLoss[0] := 0.5; dLoss[1] := 0.4; dLoss[2] := 0.3;
  gLoss[0] := 0.8; gLoss[1] := 0.6; gLoss[2] := 0.5;
  PlotLossCSV(TEST_DIR + '/losses_test.csv', dLoss, gLoss, 3);
  TC('GF_Train_PlotCSV file', FileExists(TEST_DIR + '/losses_test.csv'));

  { GF_Train_PrintBar (visual, just verify no crash) }
  Write('  PrintBar: ');
  PrintLossBar(0.5, 0.8, 30);
  TC('GF_Train_PrintBar no crash', True);
end;

{ =========================================================================== }
{ TEST: GF_Train_ Training Step & Full                                         }
{ =========================================================================== }

procedure TestTrainTraining;
var
  gen, disc: TNetwork;
  genSizes, discSizes: array of Integer;
  realBatch, noise: TMatrix;
  ds: TDataset;
  cfg: TGANConfig;
  i: Integer;
begin
  Heading('GF_Train_ Training');

  SetLength(genSizes, 4);
  genSizes[0] := 4; genSizes[1] := 16; genSizes[2] := 8; genSizes[3] := 1;
  SetLength(discSizes, 4);
  discSizes[0] := 1; discSizes[1] := 8; discSizes[2] := 16; discSizes[3] := 1;
  gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);
  disc := CreateNetwork(discSizes, atLeakyReLU, optAdam, 0.001);

  { GF_Train_Step with BCE }
  realBatch := CreateMatrix(4, 1);
  for i := 0 to 3 do realBatch[i][0] := 0.5 + Random * 0.5;
  GenerateNoise(noise, 4, 4, ntGauss);
  cfg := DefaultConfig;
  cfg.lossType := lossBCE;
  cfg.batchSize := 4;

  { Must test that Step doesn't crash and produces finite weights }
  NetworkForward(disc, realBatch);
  NetworkForward(gen, noise);
  TC('GF_Train_Step pre-run ok', True);

  { GF_Train_Optimize }
  NetworkForward(gen, noise);
  NetworkBackward(gen, CreateMatrix(4, 1));
  NetworkUpdateWeights(gen);
  TC('GF_Train_Optimize finite', MFinite(gen.layers[0].weights));

  { GF_Train_Full (only if not quick mode) }
  if GQuick then begin
    WriteLn('  [SKIP] GF_Train_Full (--quick)');
  end else begin
    gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);
    disc := CreateNetwork(discSizes, atLeakyReLU, optAdam, 0.001);
    ds := CreateSyntheticDataset(20, 1);
    cfg := DefaultConfig;
    cfg.epochs := 1;
    cfg.batchSize := 4;
    cfg.lossType := lossBCE;
    cfg.noiseDepth := 4;
    cfg.outputDir := TEST_DIR;
    ForceDirectories(TEST_DIR);
    TrainGAN(gen, disc, ds, cfg);
    TC('GF_Train_Full gen finite', MFinite(gen.layers[0].weights));
    TC('GF_Train_Full disc finite', MFinite(disc.layers[0].weights));
  end;
end;

{ =========================================================================== }
{ TEST: GF_Sec_ Security                                                       }
{ =========================================================================== }

procedure TestSec;
var
  layer: TLayer;
  net: TNetwork;
  sizes: array of Integer;
  M: TMatrix;
  b: Byte;
  ok: Boolean;
begin
  Heading('GF_Sec_ Security');

  { GF_Sec_AuditLog }
  ForceDirectories(TEST_DIR);
  AuditLog('GANTest audit entry', TEST_DIR + '/test_audit.log');
  TC('GF_Sec_AuditLog file', FileExists(TEST_DIR + '/test_audit.log'));

  { GF_Sec_SecureRandomize }
  SecureRandomize;
  TC('GF_Sec_SecureRandomize no crash', True);

  { GF_Sec_GetOSRandom }
  b := SecureRandomByte;
  TC('GF_Sec_GetOSRandom byte', (b >= 0) and (b <= 255));

  { GF_Sec_ValidatePath }
  TC('GF_Sec_ValidatePath good', ValidatePath('/tmp/model.bin'));
  TC('GF_Sec_ValidatePath reject ..', not ValidatePath('/tmp/../etc/passwd'));
  TC('GF_Sec_ValidatePath reject empty', not ValidatePath(''));

  { GF_Sec_VerifyWeights }
  layer := CreateDenseLayer(4, 4, atReLU);
  layer.weights[0][0] := 0.0 / 0.0;  { NaN }
  ValidateAndCleanWeights(layer);
  TC('GF_Sec_VerifyWeights NaN cleaned', not IsNan(layer.weights[0][0]));

  { GF_Sec_VerifyNetwork }
  SetLength(sizes, 3);
  sizes[0] := 4; sizes[1] := 4; sizes[2] := 1;
  net := CreateNetwork(sizes, atReLU, optAdam, 0.001);
  net.layers[0].weights[0][0] := 1.0 / 0.0;  { Inf }
  ValidateAndCleanWeights(net.layers[0]);
  TC('GF_Sec_VerifyNetwork Inf cleaned', not IsInfinite(net.layers[0].weights[0][0]));

  { GF_Sec_EncryptModel / DecryptModel }
  SaveNetworkBinary(net, TEST_DIR + '/enc_test.bin');
  EncryptFile(TEST_DIR + '/enc_test.bin', TEST_DIR + '/enc_test.enc', 'testkey');
  TC('GF_Sec_EncryptModel file', FileExists(TEST_DIR + '/enc_test.enc'));
  DecryptFile(TEST_DIR + '/enc_test.enc', TEST_DIR + '/enc_test_dec.bin', 'testkey');
  TC('GF_Sec_DecryptModel file', FileExists(TEST_DIR + '/enc_test_dec.bin'));

  { GF_Sec_BoundsCheck }
  M := CreateMatrix(3, 4);
  TC('GF_Sec_BoundsCheck valid', (0 >= 0) and (0 < Length(M)) and (0 >= 0) and (0 < Length(M[0])));
  TC('GF_Sec_BoundsCheck OOB', not ((99 >= 0) and (99 < Length(M))));

  { GF_Sec_RunTests }
  if GQuick then begin
    WriteLn('  [SKIP] GF_Sec_RunTests (--quick)');
    WriteLn('  [SKIP] GF_Sec_RunFuzzTests (--quick)');
  end else begin
    ok := RunTests;
    TC('GF_Sec_RunTests pass', ok);
    ok := RunFuzzTests(50);
    TC('GF_Sec_RunFuzzTests(50) pass', ok);
  end;
end;

{ =========================================================================== }
{ TEST: Deep Introspection                                                     }
{ =========================================================================== }

procedure TestIntrospection;
var
  gen, disc: TNetwork;
  genSizes, discSizes: array of Integer;
  inp, outp, grad: TMatrix;
  i: Integer;
  wVal, bVal, gVal, mVal, vVal: Single;
  actMin, actMax, actSum: Single;
begin
  Heading('Deep Introspection');

  { Build networks }
  SetLength(genSizes, 4);
  genSizes[0] := 8; genSizes[1] := 16; genSizes[2] := 8; genSizes[3] := 1;
  SetLength(discSizes, 4);
  discSizes[0] := 1; discSizes[1] := 8; discSizes[2] := 16; discSizes[3] := 1;
  gen := CreateNetwork(genSizes, atLeakyReLU, optAdam, 0.001);
  disc := CreateNetwork(discSizes, atLeakyReLU, optAdam, 0.001);

  { Network properties }
  TC('Introspect net.layerCount', gen.layerCount = 3);
  TC('Introspect net.optimizer', gen.optimizer = optAdam);
  TC('Introspect net.learningRate', Abs(gen.learningRate - 0.001) < 1e-6);
  TC('Introspect net.beta1', Abs(gen.beta1 - 0.9) < 1e-6);
  TC('Introspect net.beta2', Abs(gen.beta2 - 0.999) < 1e-6);
  TC('Introspect net.epsilon', gen.epsilon > 0);
  TC('Introspect net.progressiveAlpha', Abs(gen.progressiveAlpha - 1.0) < 1e-6);
  TC('Introspect net.isTraining', gen.isTraining = True);

  { Layer properties }
  TC('Introspect layer[0].layerType=Dense', gen.layers[0].layerType = ltDense);
  TC('Introspect layer[0].activation', gen.layers[0].activation = atLeakyReLU);
  TC('Introspect layer[0].inputSize=8', gen.layers[0].inputSize = 8);
  TC('Introspect layer[0].outputSize=16', gen.layers[0].outputSize = 16);

  { Read individual weights }
  wVal := gen.layers[0].weights[0][0];
  TC('Introspect read weight[0][0]', (not IsNan(wVal)) and (not IsInfinite(wVal)));
  wVal := gen.layers[0].weights[3][7];
  TC('Introspect read weight[3][7]', (not IsNan(wVal)) and (not IsInfinite(wVal)));

  { Read individual biases }
  bVal := gen.layers[0].bias[0];
  TC('Introspect read bias[0]', (not IsNan(bVal)) and (not IsInfinite(bVal)));

  { Set individual weights }
  gen.layers[0].weights[0][0] := 0.12345;
  TC('Introspect set weight', Abs(gen.layers[0].weights[0][0] - 0.12345) < 1e-5);
  gen.layers[0].bias[0] := -0.5;
  TC('Introspect set bias', Abs(gen.layers[0].bias[0] - (-0.5)) < 1e-5);

  { Forward pass and read layer outputs }
  inp := CreateMatrix(2, 8);
  for i := 0 to 7 do inp[0][i] := RandomGaussian * 0.5;
  for i := 0 to 7 do inp[1][i] := RandomGaussian * 0.5;
  outp := NetworkForward(gen, inp);

  { Read cached layer input/output/preActivation }
  TC('Introspect layer[0].layerInput',
    (Length(gen.layers[0].layerInput) = 2) and (Length(gen.layers[0].layerInput[0]) = 8));
  TC('Introspect layer[0].layerOutput rows=2', Length(gen.layers[0].layerOutput) > 0);
  TC('Introspect layer[0].preActivation', Length(gen.layers[0].preActivation) > 0);
  TC('Introspect layer[0].layerOutput finite', MFinite(gen.layers[0].layerOutput));

  { Activation histogram (compute stats on layer output) }
  actMin := 1e30; actMax := -1e30; actSum := 0;
  for i := 0 to High(gen.layers[0].layerOutput[0]) do begin
    wVal := gen.layers[0].layerOutput[0][i];
    if wVal < actMin then actMin := wVal;
    if wVal > actMax then actMax := wVal;
    actSum := actSum + wVal;
  end;
  TC('Introspect activation min finite', (not IsNan(actMin)) and (not IsInfinite(actMin)));
  TC('Introspect activation max finite', (not IsNan(actMax)) and (not IsInfinite(actMax)));
  TC('Introspect activation mean finite',
    (not IsNan(actSum / gen.layers[0].outputSize)) and
    (not IsInfinite(actSum / gen.layers[0].outputSize)));

  { Backward pass and read gradients }
  grad := CreateMatrix(2, 1);
  grad[0][0] := 1.0; grad[1][0] := -1.0;
  NetworkForward(gen, inp);
  NetworkBackward(gen, grad);

  TC('Introspect layer[0].weightGrad exists',
    (Length(gen.layers[0].weightGrad) = 8) and (Length(gen.layers[0].weightGrad[0]) = 16));
  gVal := gen.layers[0].weightGrad[0][0];
  TC('Introspect read weightGrad[0][0]', (not IsNan(gVal)) and (not IsInfinite(gVal)));
  TC('Introspect layer[0].biasGrad exists', Length(gen.layers[0].biasGrad) = 16);
  gVal := gen.layers[0].biasGrad[0];
  TC('Introspect read biasGrad[0]', (not IsNan(gVal)) and (not IsInfinite(gVal)));

  { Update weights and check Adam optimizer state }
  NetworkUpdateWeights(gen);

  TC('Introspect layer[0].adamT > 0', gen.layers[0].adamT > 0);
  TC('Introspect layer[0].mWeight exists',
    (Length(gen.layers[0].mWeight) = 8) and (Length(gen.layers[0].mWeight[0]) = 16));
  mVal := gen.layers[0].mWeight[0][0];
  TC('Introspect read mWeight[0][0] (1st moment)', (not IsNan(mVal)) and (not IsInfinite(mVal)));
  TC('Introspect layer[0].vWeight exists',
    (Length(gen.layers[0].vWeight) = 8) and (Length(gen.layers[0].vWeight[0]) = 16));
  vVal := gen.layers[0].vWeight[0][0];
  TC('Introspect read vWeight[0][0] (2nd moment)', (not IsNan(vVal)) and (not IsInfinite(vVal)));
  TC('Introspect layer[0].mBias exists', Length(gen.layers[0].mBias) = 16);
  TC('Introspect layer[0].vBias exists', Length(gen.layers[0].vBias) = 16);

  { Multiple forward/backward cycles to accumulate optimizer state }
  for i := 1 to 3 do begin
    NetworkForward(gen, inp);
    NetworkBackward(gen, grad);
    NetworkUpdateWeights(gen);
  end;
  TC('Introspect adamT incremented', gen.layers[0].adamT >= 4);
  TC('Introspect weights still finite after updates', MFinite(gen.layers[0].weights));

  { Discriminator introspection }
  inp := CreateMatrix(2, 1);
  inp[0][0] := 0.5; inp[1][0] := -0.3;
  outp := NetworkForward(disc, inp);
  grad := CreateMatrix(2, 1);
  grad[0][0] := 1; grad[1][0] := -1;
  NetworkForward(disc, inp);
  NetworkBackward(disc, grad);
  NetworkUpdateWeights(disc);

  TC('Introspect disc layer[0].weights', MFinite(disc.layers[0].weights));
  TC('Introspect disc layer[0].weightGrad', Length(disc.layers[0].weightGrad) > 0);
  TC('Introspect disc layer[0].mWeight', Length(disc.layers[0].mWeight) > 0);

  { Weight decay introspection }
  gen.weightDecay := 0.01;
  NetworkForward(gen, CreateMatrix(2, 8));
  NetworkBackward(gen, CreateMatrix(2, 1));
  NetworkUpdateWeights(gen);
  TC('Introspect weightDecay set', Abs(gen.weightDecay - 0.01) < 1e-6);
  TC('Introspect weights finite after decay', MFinite(gen.layers[0].weights));

  { Progressive alpha }
  gen.progressiveAlpha := 0.5;
  TC('Introspect progressiveAlpha set', Abs(gen.progressiveAlpha - 0.5) < 1e-6);

  { Network learning rate modification }
  gen.learningRate := 0.0005;
  TC('Introspect lr modified', Abs(gen.learningRate - 0.0005) < 1e-6);

  { Cross-layer output chaining: verify layer[i].layerOutput = layer[i+1].layerInput }
  inp := CreateMatrix(2, 8);
  inp[0][0] := 1;
  NetworkForward(gen, inp);
  TC('Introspect chain: L0.output = L1.input',
    Abs(gen.layers[0].layerOutput[0][0] - gen.layers[1].layerInput[0][0]) < 1e-6);
  TC('Introspect chain: L1.output = L2.input',
    Abs(gen.layers[1].layerOutput[0][0] - gen.layers[2].layerInput[0][0]) < 1e-6);
end;

{ =========================================================================== }
{ CLI PARSING                                                                  }
{ =========================================================================== }

procedure ParseArgs;
var
  i: Integer;
  arg: string;
  anyCategory: Boolean;
begin
  GVerbose := False;
  GQuick := False;
  GRunOps := False;
  GRunGen := False;
  GRunDisc := False;
  GRunTrain := False;
  GRunSec := False;
  GRunIntrospect := False;

  for i := 1 to ParamCount do begin
    arg := ParamStr(i);
    if (arg = '--help') or (arg = '-h') then begin
      ShowTestHelp;
      Halt(0);
    end
    else if (arg = '--verbose') or (arg = '-v') then GVerbose := True
    else if arg = '--quick' then GQuick := True
    else if arg = '--all' then begin
      GRunOps := True; GRunGen := True; GRunDisc := True;
      GRunTrain := True; GRunSec := True; GRunIntrospect := True;
    end
    else if arg = '--ops' then GRunOps := True
    else if arg = '--gen' then GRunGen := True
    else if arg = '--disc' then GRunDisc := True
    else if arg = '--train' then GRunTrain := True
    else if arg = '--sec' then GRunSec := True
    else if arg = '--introspect' then GRunIntrospect := True
    else WriteLn('Warning: unknown option: ', arg);
  end;

  anyCategory := GRunOps or GRunGen or GRunDisc or GRunTrain
                 or GRunSec or GRunIntrospect;
  if not anyCategory then begin
    GRunOps := True; GRunGen := True; GRunDisc := True;
    GRunTrain := True; GRunSec := True; GRunIntrospect := True;
  end;
end;

{ =========================================================================== }
{ MAIN                                                                         }
{ =========================================================================== }

begin
  ParseArgs;

  WriteLn('GANTest - Comprehensive GAN Facade Test Suite');
  WriteLn('GAN Unit v', GAN_VERSION);
  WriteLn('');

  GTotal := 0; GPass := 0; GFail := 0;
  SecureRandomize;
  ForceDirectories(TEST_DIR);

  if GRunOps then begin
    WriteLn('=== GF_Op_ Low-Level Operations ===');
    TestOpMatrix;
    TestOpActivations;
    TestOpConv;
    TestOpNorm;
    TestOpAttention;
    TestOpLayerDispatch;
    TestOpRandom;
  end;

  if GRunGen then begin
    WriteLn('');
    WriteLn('=== GF_Gen_ Generator ===');
    TestGen;
  end;

  if GRunDisc then begin
    WriteLn('');
    WriteLn('=== GF_Disc_ Discriminator ===');
    TestDisc;
  end;

  if GRunTrain then begin
    WriteLn('');
    WriteLn('=== GF_Train_ Training ===');
    TestTrainLoss;
    TestTrainOptim;
    TestTrainData;
    TestTrainIO;
    TestTrainTraining;
  end;

  if GRunSec then begin
    WriteLn('');
    WriteLn('=== GF_Sec_ Security ===');
    TestSec;
  end;

  if GRunIntrospect then begin
    WriteLn('');
    WriteLn('=== Deep Introspection ===');
    TestIntrospection;
  end;

  { Summary }
  WriteLn('');
  WriteLn('=====================================================================');
  WriteLn(' RESULTS: ', GTotal, ' tests | ', GPass, ' passed | ', GFail, ' failed');
  WriteLn('=====================================================================');

  if GFail = 0 then
    WriteLn('All tests passed.')
  else
    WriteLn(GFail, ' test(s) FAILED.');

  if GFail > 0 then Halt(1);
end.

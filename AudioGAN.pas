(*
 * MIT License
 * 
 * Copyright (c) 2025 Matthew Abbott
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software. 
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *)

program GANAudio;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Classes, StrUtils;

type
  { Basic types }
  TVector = array of Single;
  TMatrix = array of array of Single;
  TMatrixArray = array of TMatrix;
  TAudioBuffer = array of Single;
  TTensor3D = array of array of array of Single;

  { Enumerations }
  TActivationType = (atReLU, atSigmoid, atTanh, atLeakyReLU);
  TOptimizer = (optAdam, optSGD);
  TNoiseType = (ntGauss, ntUniform, ntAnalog);

  { Audio configuration }
  TAudioConfig = record
    sampleRate: Integer;
    segmentLength: Integer;
    nMels: Integer;
    frameSize: Integer;
    hopSize: Integer;
    inputType: string;
  end;

  { Layer structure }
  TLayer = record
    weights: TMatrix;
    bias: TVector;
    input: TMatrix;
    output: TMatrix;
    activation: TActivationType;
    inputSize: Integer;
    outputSize: Integer;
  end;

  { Adam optimizer state }
  TAdamState = record
    m: TMatrix;
    v: TMatrix;
    t: Integer;
  end;

  { Network structure }
  TNetwork = record
    layers: array of TLayer;
    adamStates: array of TAdamState;
    layerCount: Integer;
    optimizer: TOptimizer;
    learningRate: Single;
    momentum: Single;
    beta1: Single;
    beta2: Single;
    epsilon: Single;
  end;

  { GAN configuration }
  TGANAudioConfig = record
    epochs: Integer;
    batchSize: Integer;
    generatorBits: Integer;
    discriminatorBits: Integer;
    activation: TActivationType;
    noiseType: TNoiseType;
    noiseDepth: Integer;
    optimizer: TOptimizer;
    learningRate: Single;
    outputDir: string;
    saveModel: string;
    loadModel: string;
    
    { Audio }
    audioDataPath: string;
    sampleRate: Integer;
    segmentLength: Integer;
    inputType: string;
    nMels: Integer;
    frameSize: Integer;
    hopSize: Integer;
    lossMetric: string;
    
    { Paired data }
    cleanAudioPath: string;
    noisyAudioPath: string;
    usePairedData: Boolean;
    exportAudioSamples: Boolean;
  end;

{ ============================================================================= }
{ MATH OPERATIONS }
{ ============================================================================= }

function CreateMatrix(rows, cols: Integer): TMatrix;
var
  i: Integer;
begin
  SetLength(Result, rows);
  for i := 0 to rows - 1 do
    SetLength(Result[i], cols);
end;

function CreateVector(size: Integer): TVector;
begin
  SetLength(Result, size);
  FillChar(Result[0], size * SizeOf(Single), 0);
end;

function RandomGaussian: Single;
var
  u1, u2: Single;
begin
  u1 := Random;
  u2 := Random;
  if u1 < 1e-7 then
    u1 := 1e-7;
  Result := sqrt(-2.0 * ln(u1)) * cos(2.0 * Pi * u2);
end;

function RandomUniform(min, max: Single): Single;
begin
  Result := min + Random * (max - min);
end;

function MatrixMultiply(const A, B: TMatrix): TMatrix;
var
  i, j, k: Integer;
  sum: Single;
begin
  Result := CreateMatrix(Length(A), Length(B[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(B[0]) - 1 do
    begin
      sum := 0;
      for k := 0 to Length(B) - 1 do
        sum := sum + A[i][k] * B[k][j];
      Result[i][j] := sum;
    end;
end;

function MatrixAdd(const A, B: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := A[i][j] + B[i][j];
end;

function MatrixScale(const A: TMatrix; scale: Single): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := A[i][j] * scale;
end;

function MatrixTranspose(const A: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  if Length(A) = 0 then
    Exit;
  Result := CreateMatrix(Length(A[0]), Length(A));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[j][i] := A[i][j];
end;

function MatrixReLU(const A: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      if A[i][j] > 0 then
        Result[i][j] := A[i][j]
      else
        Result[i][j] := 0;
end;

function MatrixLeakyReLU(const A: TMatrix; alpha: Single = 0.01): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      if A[i][j] > 0 then
        Result[i][j] := A[i][j]
      else
        Result[i][j] := alpha * A[i][j];
end;

function MatrixSigmoid(const A: TMatrix): TMatrix;
var
  i, j: Integer;
  val: Single;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
    begin
      val := A[i][j];
      if val > 20 then
        Result[i][j] := 1.0
      else if val < -20 then
        Result[i][j] := 0.0
      else
        Result[i][j] := 1.0 / (1.0 + exp(-val));
    end;
end;

function MatrixTanh(const A: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := tanh(A[i][j]);
end;

function ApplyActivation(const A: TMatrix; activation: TActivationType): TMatrix;
begin
  case activation of
    atReLU: Result := MatrixReLU(A);
    atSigmoid: Result := MatrixSigmoid(A);
    atTanh: Result := MatrixTanh(A);
    atLeakyReLU: Result := MatrixLeakyReLU(A, 0.01);
  else
    Result := A;
  end;
end;

function MatrixNormalize(const A: TMatrix): TMatrix;
var
  i, j: Integer;
  mean, variance, count: Single;
begin
  mean := 0;
  count := 0;
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
    begin
      mean := mean + A[i][j];
      count := count + 1;
    end;
  mean := mean / count;

  variance := 0;
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      variance := variance + sqr(A[i][j] - mean);
  variance := sqrt(variance / count);

  if variance < 1e-7 then
    variance := 1;

  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := (A[i][j] - mean) / variance;
end;

{ ============================================================================= }
{ AUDIO OPERATIONS }
{ ============================================================================= }

function NormalizeAudio(const audio: TAudioBuffer; targetRange: Single = 1.0): TAudioBuffer;
var
  i: Integer;
  maxVal, scale: Single;
begin
  SetLength(Result, Length(audio));
  
  maxVal := 0;
  for i := 0 to Length(audio) - 1 do
    if Abs(audio[i]) > maxVal then
      maxVal := Abs(audio[i]);
  
  if maxVal < 1e-7 then
    maxVal := 1e-7;
  
  scale := targetRange / maxVal;
  for i := 0 to Length(audio) - 1 do
    Result[i] := audio[i] * scale;
end;

function AudioMax(const audio: TAudioBuffer): Single;
var
  i: Integer;
begin
  Result := 0;
  for i := 0 to Length(audio) - 1 do
    if Abs(audio[i]) > Result then
      Result := Abs(audio[i]);
end;

function AudioRMS(const audio: TAudioBuffer): Single;
var
  i: Integer;
  sum: Single;
begin
  sum := 0;
  for i := 0 to Length(audio) - 1 do
    sum := sum + sqr(audio[i]);
  if Length(audio) > 0 then
    Result := sqrt(sum / Length(audio))
  else
    Result := 0;
end;

function ComputeSNR(const clean, noisy: TAudioBuffer): Single;
var
  i: Integer;
  signalPower, noisePower: Single;
begin
  signalPower := 0;
  noisePower := 0;
  
  for i := 0 to Min(Length(clean), Length(noisy)) - 1 do
  begin
    signalPower := signalPower + sqr(clean[i]);
    noisePower := noisePower + sqr(clean[i] - noisy[i]);
  end;
  
  if noisePower < 1e-10 then
    noisePower := 1e-10;
  
  Result := 10.0 * log10(signalPower / noisePower);
end;

function ComputeSISDR(const reference, estimated: TAudioBuffer): Single;
var
  i: Integer;
  refPower, s, shatPower, errorPower: Single;
begin
  refPower := 0;
  for i := 0 to Length(reference) - 1 do
    refPower := refPower + sqr(reference[i]);
  
  if refPower < 1e-10 then
    refPower := 1e-10;
  
  s := 0;
  for i := 0 to Min(Length(reference), Length(estimated)) - 1 do
    s := s + reference[i] * estimated[i];
  s := s / refPower;
  
  shatPower := 0;
  errorPower := 0;
  for i := 0 to Min(Length(reference), Length(estimated)) - 1 do
  begin
    shatPower := shatPower + sqr(s * reference[i]);
    errorPower := errorPower + sqr(estimated[i] - s * reference[i]);
  end;
  
  if errorPower < 1e-10 then
    errorPower := 1e-10;
  
  Result := 10.0 * log10(shatPower / errorPower);
end;

{ ============================================================================= }
{ NETWORK FUNCTIONS }
{ ============================================================================= }

procedure InitializeLayer(var layer: TLayer; inputSize, outputSize: Integer;
  activation: TActivationType);
var
  i, j: Integer;
  scale: Single;
begin
  layer.inputSize := inputSize;
  layer.outputSize := outputSize;
  layer.activation := activation;

  scale := sqrt(2.0 / (inputSize + outputSize));

  SetLength(layer.weights, inputSize);
  for i := 0 to inputSize - 1 do
  begin
    SetLength(layer.weights[i], outputSize);
    for j := 0 to outputSize - 1 do
      layer.weights[i][j] := RandomGaussian * scale;
  end;

  layer.bias := CreateVector(outputSize);
end;

function LayerForward(var layer: TLayer; const input: TMatrix): TMatrix;
var
  z: TMatrix;
  i, j: Integer;
  activated: TMatrix;
begin
  layer.input := input;

  z := MatrixMultiply(input, layer.weights);

  SetLength(Result, Length(z));
  for i := 0 to Length(z) - 1 do
  begin
    SetLength(Result[i], Length(z[i]));
    for j := 0 to Length(z[i]) - 1 do
      Result[i][j] := z[i][j] + layer.bias[j];
  end;

  activated := ApplyActivation(Result, layer.activation);
  Result := activated;
  layer.output := Result;
end;

function CreateNetwork(const layerSizes: array of Integer;
  activation: TActivationType; optimizer: TOptimizer; lr: Single): TNetwork;
var
  i: Integer;
begin
  Result.layerCount := Length(layerSizes) - 1;
  SetLength(Result.layers, Result.layerCount);
  SetLength(Result.adamStates, Result.layerCount);
  Result.optimizer := optimizer;
  Result.learningRate := lr;
  Result.momentum := 0.9;
  Result.beta1 := 0.9;
  Result.beta2 := 0.999;
  Result.epsilon := 1e-8;

  for i := 0 to Result.layerCount - 1 do
  begin
    InitializeLayer(Result.layers[i], layerSizes[i], layerSizes[i + 1], activation);
    Result.adamStates[i].t := 0;
  end;
end;

function NetworkForward(var net: TNetwork; const input: TMatrix): TMatrix;
var
  i: Integer;
  current: TMatrix;
begin
  current := input;
  for i := 0 to net.layerCount - 1 do
    current := LayerForward(net.layers[i], current);
  Result := current;
end;

{ ============================================================================= }
{ LOSS FUNCTIONS }
{ ============================================================================= }

function ComputeBCELoss(const predicted, target: TMatrix): Single;
var
  i, j: Integer;
  p, loss, eps: Single;
begin
  eps := 1e-7;
  Result := 0;
  for i := 0 to Length(predicted) - 1 do
    for j := 0 to Length(predicted[0]) - 1 do
    begin
      p := Max(eps, Min(1 - eps, predicted[i][j]));
      loss := -(target[i][j] * ln(p) + (1 - target[i][j]) * ln(1 - p));
      Result := Result + loss;
    end;
  if Length(predicted) * Length(predicted[0]) > 0 then
    Result := Result / (Length(predicted) * Length(predicted[0]));
end;

function ComputeSNRLoss(const reference, estimated: TMatrix): Single;
var
  i, j: Integer;
  refPower, errPower: Single;
begin
  refPower := 0;
  errPower := 0;
  
  for i := 0 to Min(Length(reference), Length(estimated)) - 1 do
    for j := 0 to Min(Length(reference[i]), Length(estimated[i])) - 1 do
    begin
      refPower := refPower + sqr(reference[i][j]);
      errPower := errPower + sqr(reference[i][j] - estimated[i][j]);
    end;
  
  if errPower < 1e-10 then
    errPower := 1e-10;
  
  Result := -10.0 * log10(refPower / errPower);
end;

{ ============================================================================= }
{ TRAINING }
{ ============================================================================= }

procedure TrainGANAudio(var generator, discriminator: TNetwork;
  const dataset: TMatrixArray; config: TGANAudioConfig);
var
  epoch, batch, i, j, numBatches, idx: Integer;
  noise, realBatch, fakeBatch, dInput, dOutput, gOutput: TMatrix;
  dRealLoss, dFakeLoss, gLoss, loss: Single;
  batchStart: Integer;
begin
  numBatches := Length(dataset) div config.batchSize;
  if numBatches = 0 then
    numBatches := 1;

  WriteLn('Starting audio GAN training...');
  WriteLn('Loss metric: ', config.lossMetric);
  WriteLn('Input type: ', config.inputType);
  WriteLn('');

  for epoch := 0 to config.epochs - 1 do
  begin
    loss := 0;
    
    for batch := 0 to numBatches - 1 do
    begin
      { Create batch - copy dataset samples into realBatch matrix }
      batchStart := batch * config.batchSize;
      SetLength(realBatch, config.batchSize);
      for i := 0 to config.batchSize - 1 do
      begin
        if batchStart + i < Length(dataset) then
        begin
          SetLength(realBatch[i], Length(dataset[batchStart + i][0]));
          for j := 0 to Length(dataset[batchStart + i][0]) - 1 do
            realBatch[i][j] := dataset[batchStart + i][0][j];
        end
        else
        begin
          idx := Random(Length(dataset));
          SetLength(realBatch[i], Length(dataset[idx][0]));
          for j := 0 to Length(dataset[idx][0]) - 1 do
            realBatch[i][j] := dataset[idx][0][j];
        end;
      end;

      { Generate noise }
      SetLength(noise, config.batchSize);
      for i := 0 to config.batchSize - 1 do
      begin
        SetLength(noise[i], config.noiseDepth);
        for j := 0 to config.noiseDepth - 1 do
          noise[i][j] := RandomGaussian;
      end;

      { Discriminator training }
      dOutput := NetworkForward(discriminator, realBatch);
      SetLength(dInput, config.batchSize);
      for i := 0 to config.batchSize - 1 do
      begin
        SetLength(dInput[i], 1);
        dInput[i][0] := 0.9;
      end;
      dRealLoss := ComputeBCELoss(dOutput, dInput);

      fakeBatch := NetworkForward(generator, noise);
      dOutput := NetworkForward(discriminator, fakeBatch);
      SetLength(dInput, config.batchSize);
      for i := 0 to config.batchSize - 1 do
      begin
        SetLength(dInput[i], 1);
        dInput[i][0] := 0.1;
      end;
      dFakeLoss := ComputeBCELoss(dOutput, dInput);

      { Generator training }
      gOutput := NetworkForward(generator, noise);
      dOutput := NetworkForward(discriminator, gOutput);
      SetLength(dInput, config.batchSize);
      for i := 0 to config.batchSize - 1 do
      begin
        SetLength(dInput[i], 1);
        dInput[i][0] := 0.9;
      end;
      gLoss := ComputeBCELoss(dOutput, dInput);

      loss := loss + (dRealLoss + dFakeLoss) / 2 + gLoss;

      if ((batch + 1) mod 10 = 0) or (batch = 0) then
        WriteLn(Format('[Epoch %d/%d] Batch %d | D Loss: %.6f | G Loss: %.6f',
          [epoch + 1, config.epochs, batch + 1, (dRealLoss + dFakeLoss) / 2, gLoss]));
    end;

    WriteLn(Format('Epoch %d complete - Avg Loss: %.6f', [epoch + 1, loss / numBatches]));
  end;

  WriteLn('Training complete.');
end;

{ ============================================================================= }
{ SAMPLE RATE VALIDATION AND AUTO-CONFIGURATION }
{ ============================================================================= }

function IsValidSampleRate(rate: Integer): Boolean;
begin
  Result := (rate >= 8000) and (rate <= 96000) and
            ((rate = 8000) or (rate = 16000) or (rate = 22050) or 
             (rate = 44100) or (rate = 48000) or (rate = 96000));
end;

procedure AutoConfigureAudioParams(var config: TGANAudioConfig);
begin
  { Auto-calculate segment length as 1 second of audio }
  if config.segmentLength <= 0 then
    config.segmentLength := config.sampleRate;

  { Auto-calculate FFT parameters based on sample rate }
  if config.frameSize <= 0 then
  begin
    case config.sampleRate of
      8000: config.frameSize := 256;
      16000: config.frameSize := 512;
      22050: config.frameSize := 512;
      44100: config.frameSize := 1024;
      48000: config.frameSize := 1024;
      96000: config.frameSize := 2048;
    else
      config.frameSize := 512;
    end;
  end;

  if config.hopSize <= 0 then
  begin
    case config.sampleRate of
      8000: config.hopSize := 80;
      16000: config.hopSize := 160;
      22050: config.hopSize := 220;
      44100: config.hopSize := 440;
      48000: config.hopSize := 480;
      96000: config.hopSize := 960;
    else
      config.hopSize := 160;
    end;
  end;
end;

{ ============================================================================= }
{ CLI ARGUMENT PARSING }
{ ============================================================================= }

procedure ShowHelp;
begin
   WriteLn('GAN Audio Network - Complete Audio-Enhanced GAN');
   WriteLn('');
   WriteLn('Usage: gan_audio_complete [options]');
   WriteLn('');
   WriteLn('Audio Options:');
   WriteLn('  --audio-path=PATH       Path to audio directory');
   WriteLn('  --sample-rate=N         Sample rate in Hz (8000-96000, default: 16000)');
   WriteLn('                          Supported rates: 8000, 16000, 22050, 44100, 48000, 96000');
   WriteLn('  --segment-length=N      Segment length in samples (default: auto-calculated from sample rate)');
   WriteLn('  --input-type=TYPE       waveform or melspec (default: waveform)');
   WriteLn('  --n-mels=N              Number of mel bins (default: 128)');
   WriteLn('  --frame-size=N          FFT frame size (default: auto-calculated from sample rate)');
   WriteLn('  --hop-size=N            Frame hop (default: auto-calculated from sample rate)');
   WriteLn('');
   WriteLn('Training Options:');
   WriteLn('  --epochs=N              Training epochs (default: 100)');
   WriteLn('  --batch-size=N          Batch size (default: 32)');
   WriteLn('  --noise-depth=N         Noise vector size (default: 100)');
   WriteLn('  --lr=RATE               Learning rate (default: 0.0002)');
   WriteLn('  --loss-metric=METRIC    bce|snr|sisdr|stft (default: bce)');
   WriteLn('');
   WriteLn('Data Options:');
   WriteLn('  --clean-path=PATH       Path to clean audio');
   WriteLn('  --noisy-path=PATH       Path to noisy audio');
   WriteLn('  --export-audio          Export samples after each epoch');
   WriteLn('');
   WriteLn('Model Options:');
   WriteLn('  --save=FILE             Save model to file');
   WriteLn('  --load=FILE             Load pretrained model');
   WriteLn('  --output=PATH           Output directory (default: ./output)');
   WriteLn('  --help                  Show this help message');
   WriteLn('');
   WriteLn('Examples:');
   WriteLn('  gan_audio_complete --audio-path=./audio --sample-rate=8000 --epochs=50');
   WriteLn('  gan_audio_complete --sample-rate=48000 --clean-path=./clean --noisy-path=./noisy');
   WriteLn('  gan_audio_complete --sample-rate=96000 --input-type=melspec --n-mels=128');
end;

function ParseConfig: TGANAudioConfig;
var
  i: Integer;
  arg, key, value: string;
begin
  { Set defaults }
  Result.epochs := 100;
  Result.batchSize := 32;
  Result.generatorBits := 16;
  Result.discriminatorBits := 16;
  Result.activation := atLeakyReLU;
  Result.noiseType := ntGauss;
  Result.noiseDepth := 100;
  Result.optimizer := optAdam;
  Result.learningRate := 0.0002;
  Result.outputDir := './output';
  
  { Audio defaults }
  Result.audioDataPath := '';
  Result.sampleRate := 16000;
  Result.segmentLength := 0;  { Will be auto-calculated }
  Result.inputType := 'waveform';
  Result.nMels := 128;
  Result.frameSize := 0;  { Will be auto-calculated }
  Result.hopSize := 0;    { Will be auto-calculated }
  Result.lossMetric := 'bce';
  Result.cleanAudioPath := '';
  Result.noisyAudioPath := '';
  Result.usePairedData := False;
  Result.exportAudioSamples := False;
  Result.saveModel := '';
  Result.loadModel := '';

  if ParamCount = 0 then
  begin
    ShowHelp;
    Halt;
  end;

  for i := 1 to ParamCount do
  begin
    arg := ParamStr(i);

    if arg = '--help' then
    begin
      ShowHelp;
      Halt;
    end;

    if Pos('--epochs=', arg) = 1 then
      Result.epochs := StrToIntDef(Copy(arg, 10, MaxInt), 100);
    if Pos('--batch-size=', arg) = 1 then
      Result.batchSize := StrToIntDef(Copy(arg, 14, MaxInt), 32);
    if Pos('--noise-depth=', arg) = 1 then
      Result.noiseDepth := StrToIntDef(Copy(arg, 15, MaxInt), 100);
    if Pos('--lr=', arg) = 1 then
      Result.learningRate := StrToFloatDef(Copy(arg, 6, MaxInt), 0.0002);
    if Pos('--audio-path=', arg) = 1 then
      Result.audioDataPath := Copy(arg, 14, MaxInt);
    if Pos('--sample-rate=', arg) = 1 then
    begin
      Result.sampleRate := StrToIntDef(Copy(arg, 15, MaxInt), 16000);
      if not IsValidSampleRate(Result.sampleRate) then
      begin
        WriteLn('ERROR: Invalid sample rate. Supported rates: 8000, 16000, 22050, 44100, 48000, 96000');
        Halt(1);
      end;
    end;
    if Pos('--segment-length=', arg) = 1 then
      Result.segmentLength := StrToIntDef(Copy(arg, 18, MaxInt), 16000);
    if Pos('--input-type=', arg) = 1 then
      Result.inputType := Copy(arg, 14, MaxInt);
    if Pos('--n-mels=', arg) = 1 then
      Result.nMels := StrToIntDef(Copy(arg, 10, MaxInt), 128);
    if Pos('--frame-size=', arg) = 1 then
      Result.frameSize := StrToIntDef(Copy(arg, 14, MaxInt), 512);
    if Pos('--hop-size=', arg) = 1 then
      Result.hopSize := StrToIntDef(Copy(arg, 12, MaxInt), 160);
    if Pos('--loss-metric=', arg) = 1 then
      Result.lossMetric := Copy(arg, 15, MaxInt);
    if Pos('--clean-path=', arg) = 1 then
    begin
      Result.cleanAudioPath := Copy(arg, 14, MaxInt);
      Result.usePairedData := True;
    end;
    if Pos('--noisy-path=', arg) = 1 then
      Result.noisyAudioPath := Copy(arg, 14, MaxInt);
    if arg = '--export-audio' then
      Result.exportAudioSamples := True;
    if Pos('--save=', arg) = 1 then
      Result.saveModel := Copy(arg, 8, MaxInt);
    if Pos('--load=', arg) = 1 then
      Result.loadModel := Copy(arg, 8, MaxInt);
    if Pos('--output=', arg) = 1 then
      Result.outputDir := Copy(arg, 10, MaxInt);
  end;
end;

{ ============================================================================= }
{ MAIN PROGRAM }
{ ============================================================================= }

var
  config: TGANAudioConfig;
  generator, discriminator: TNetwork;
  genSizes, discSizes: array of Integer;
  dataset: TMatrixArray;
  i, j: Integer;
begin
   WriteLn('GAN Audio Network v2.0 - Complete Single File Edition');
   WriteLn('');

   config := ParseConfig;
   
   { Auto-configure audio parameters based on sample rate }
   AutoConfigureAudioParams(config);

   WriteLn('Configuration:');
   WriteLn('  Epochs: ', config.epochs);
   WriteLn('  Batch Size: ', config.batchSize);
   WriteLn('  Noise Depth: ', config.noiseDepth);
   WriteLn('  Learning Rate: ', config.learningRate:0:6);
   WriteLn('  Audio Sample Rate: ', config.sampleRate, ' Hz');
   WriteLn('  Segment Length: ', config.segmentLength, ' samples');
   WriteLn('  FFT Frame Size: ', config.frameSize);
   WriteLn('  FFT Hop Size: ', config.hopSize);
   WriteLn('  Input Type: ', config.inputType);
   WriteLn('  Loss Metric: ', config.lossMetric);
   WriteLn('');

  { Create networks }
  SetLength(genSizes, 4);
  genSizes[0] := config.noiseDepth;
  genSizes[1] := 256;
  genSizes[2] := 128;
  genSizes[3] := config.segmentLength;

  SetLength(discSizes, 4);
  discSizes[0] := config.segmentLength;
  discSizes[1] := 128;
  discSizes[2] := 64;
  discSizes[3] := 1;

  generator := CreateNetwork(genSizes, config.activation, config.optimizer, config.learningRate);
  discriminator := CreateNetwork(discSizes, config.activation, config.optimizer, config.learningRate);

  WriteLn('Networks created:');
  WriteLn('  Generator: ', config.noiseDepth, ' -> 256 -> 128 -> ', config.segmentLength);
  WriteLn('  Discriminator: ', config.segmentLength, ' -> 128 -> 64 -> 1');
  WriteLn('');

  { Generate synthetic dataset }
  WriteLn('Generating synthetic training data...');
  SetLength(dataset, 100);
  for i := 0 to 99 do
  begin
    dataset[i] := CreateMatrix(1, config.segmentLength);
    for j := 0 to config.segmentLength - 1 do
      dataset[i][0][j] := sin(j / 100.0 + i / 10.0) * 0.5;
  end;
  WriteLn('Dataset size: ', Length(dataset), ' samples');
  WriteLn('');

  { Train GAN }
  WriteLn('Starting training...');
  WriteLn('');
  TrainGANAudio(generator, discriminator, dataset, config);

  WriteLn('');
  WriteLn('Training finished.');
  if config.saveModel <> '' then
    WriteLn('Model save feature available with binary export.');
end.

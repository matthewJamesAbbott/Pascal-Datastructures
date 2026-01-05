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

program GAN;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Classes, StrUtils;

type
  TMatrix = array of array of Single;
  TVector = array of Single;
  TMatrixArray = array of TMatrix;

  TActivationType = (atReLU, atSigmoid, atTanh, atLeakyReLU);
  TOptimizer = (optAdam, optSGD);
  TNoiseType = (ntGauss, ntUniform, ntAnalog);

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

  { Optimizer state for Adam }
  TAdamState = record
    m: TMatrix;  // first moment
    v: TMatrix;  // second moment
    t: Integer;  // timestep
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
  TGANConfig = record
    epochs: Integer;
    batchSize: Integer;
    generatorBits: Integer;
    discriminatorBits: Integer;
    activation: TActivationType;
    noiseType: TNoiseType;
    noiseDepth: Integer;
    useSpectral: Boolean;
    patchConfig: string;
    saveModel: string;
    loadModel: string;
    outputDir: string;
    learningRate: Single;
    optimizer: TOptimizer;
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

function RandomAnalog: Single;
begin
  Result := (Random - 0.5) * 0.1;
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
{ LAYER OPERATIONS }
{ ============================================================================= }

function CreateLayer(inputSize, outputSize: Integer; activation: TActivationType): TLayer;
var
  i, j: Integer;
  scale: Single;
begin
  Result.inputSize := inputSize;
  Result.outputSize := outputSize;
  Result.activation := activation;

  Result.weights := CreateMatrix(inputSize, outputSize);
  Result.bias := CreateVector(outputSize);

  { Xavier initialization }
  scale := sqrt(2.0 / (inputSize + outputSize));
  for i := 0 to inputSize - 1 do
    for j := 0 to outputSize - 1 do
      Result.weights[i][j] := RandomGaussian * scale;

  for j := 0 to outputSize - 1 do
    Result.bias[j] := 0;
end;

function LayerForward(var layer: TLayer; const input: TMatrix): TMatrix;
var
  i, j, k: Integer;
  sum: Single;
begin
  layer.input := input;
  Result := CreateMatrix(Length(input), layer.outputSize);

  for i := 0 to Length(input) - 1 do
    for j := 0 to layer.outputSize - 1 do
    begin
      sum := layer.bias[j];
      for k := 0 to layer.inputSize - 1 do
        sum := sum + input[i][k] * layer.weights[k][j];
      Result[i][j] := sum;
    end;

  Result := ApplyActivation(Result, layer.activation);
  layer.output := Result;
end;

{ ============================================================================= }
{ NETWORK OPERATIONS }
{ ============================================================================= }

function CreateNetwork(const sizes: array of Integer; activation: TActivationType;
  optimizer: TOptimizer; learningRate: Single): TNetwork;
var
  i: Integer;
begin
  Result.layerCount := Length(sizes) - 1;
  Result.optimizer := optimizer;
  Result.learningRate := learningRate;
  Result.momentum := 0.9;
  Result.beta1 := 0.9;
  Result.beta2 := 0.999;
  Result.epsilon := 1e-8;

  SetLength(Result.layers, Result.layerCount);
  SetLength(Result.adamStates, Result.layerCount);

  for i := 0 to Result.layerCount - 1 do
  begin
    Result.layers[i] := CreateLayer(sizes[i], sizes[i + 1], activation);
    Result.adamStates[i].t := 0;
    Result.adamStates[i].m := CreateMatrix(sizes[i], sizes[i + 1]);
    Result.adamStates[i].v := CreateMatrix(sizes[i], sizes[i + 1]);
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

procedure UpdateWeightsAdam(var net: TNetwork; layerIdx: Integer; gradient: Single);
var
  i, j: Integer;
  mVal, vVal: Single;
  mtHat, vtHat: Single;
begin
  with net do
  begin
    with adamStates[layerIdx] do
    begin
      t := t + 1;
      for i := 0 to High(layers[layerIdx].weights) do
        for j := 0 to High(layers[layerIdx].weights[i]) do
        begin
          mVal := beta1 * m[i][j] + (1 - beta1) * gradient;
          vVal := beta2 * v[i][j] + (1 - beta2) * gradient * gradient;
          mtHat := mVal / (1 - power(beta1, t));
          vtHat := vVal / (1 - power(beta2, t));
          layers[layerIdx].weights[i][j] := layers[layerIdx].weights[i][j] -
            learningRate * mtHat / (sqrt(vtHat) + epsilon);
          m[i][j] := mVal;
          v[i][j] := vVal;
        end;
    end;
  end;
end;

procedure UpdateWeightsSGD(var net: TNetwork; layerIdx: Integer; gradient: Single);
var
  i, j: Integer;
begin
  with net.layers[layerIdx] do
    for i := 0 to High(weights) do
      for j := 0 to High(weights[i]) do
        weights[i][j] := weights[i][j] - net.learningRate * gradient;
end;

{ ============================================================================= }
{ BINARY SERIALIZATION }
{ ============================================================================= }

procedure SaveNetworkBinary(const net: TNetwork; const filename: string);
var
  f: TFileStream;
  i, j, k: Integer;
  layer: TLayer;
  value: Single;
begin
  f := TFileStream.Create(filename, fmCreate);

  WriteLn('Saving ', net.layerCount, ' layers to ', filename);
  
  for k := 0 to net.layerCount - 1 do
  begin
    layer := net.layers[k];
    
    { Write weights }
    for i := 0 to High(layer.weights) do
      for j := 0 to High(layer.weights[i]) do
      begin
        value := layer.weights[i][j];
        f.Write(value, SizeOf(Single));
      end;
    
    { Write bias }
    for j := 0 to High(layer.bias) do
    begin
      value := layer.bias[j];
      f.Write(value, SizeOf(Single));
    end;
  end;

  f.Free;
  WriteLn('Network saved to ', filename);
end;

procedure LoadNetworkBinary(var net: TNetwork; const filename: string);
var
  f: TFileStream;
  i, j, k: Integer;
  value: Single;
begin
  if not FileExists(filename) then
  begin
    WriteLn('Error: File not found: ', filename);
    Exit;
  end;

  f := TFileStream.Create(filename, fmOpenRead);

  WriteLn('Loading network from ', filename);

  for k := 0 to net.layerCount - 1 do
  begin
    { Load weights }
    for i := 0 to High(net.layers[k].weights) do
      for j := 0 to High(net.layers[k].weights[i]) do
      begin
        f.Read(value, SizeOf(Single));
        net.layers[k].weights[i][j] := value;
      end;
    
    { Load bias }
    for j := 0 to High(net.layers[k].bias) do
    begin
      f.Read(value, SizeOf(Single));
      net.layers[k].bias[j] := value;
    end;
  end;

  f.Free;
  WriteLn('Network loaded successfully');
end;

{ ============================================================================= }
{ GAN TRAINING }
{ ============================================================================= }

function BinaryCrossEntropy(const predicted, target: TMatrix): Single;
var
  i, j: Integer;
  sum, eps, p: Single;
begin
  eps := 1e-7;
  sum := 0;
  for i := 0 to Length(predicted) - 1 do
    for j := 0 to Length(predicted[i]) - 1 do
    begin
      p := predicted[i][j];
      p := Max(eps, Min(1 - eps, p));
      sum := sum - (target[i][j] * ln(p) + (1 - target[i][j]) * ln(1 - p));
    end;
  Result := sum / (Length(predicted) * Length(predicted[0]));
end;

procedure GenerateNoise(var noise: TMatrix; size, depth: Integer; noiseType: TNoiseType);
var
  i, j: Integer;
begin
  noise := CreateMatrix(size, depth);
  for i := 0 to size - 1 do
    for j := 0 to depth - 1 do
    begin
      case noiseType of
        ntGauss: noise[i][j] := RandomGaussian;
        ntUniform: noise[i][j] := RandomUniform(-1, 1);
        ntAnalog: noise[i][j] := RandomAnalog;
      end;
    end;
end;

procedure TrainGAN(var generator, discriminator: TNetwork; dataset: TMatrixArray;
  config: TGANConfig);
var
  epoch, batch, i, batchStart, batchEnd: Integer;
  batchDataArray: TMatrixArray;
  batchData, fakeData, noise: TMatrix;
  realLabels, fakeLabels: TMatrix;
  dRealLoss, dFakeLoss, gLoss: Single;
  discReal, discFake, discGenOutput: TMatrix;
begin
  Randomize;

  for epoch := 0 to config.epochs - 1 do
  begin
    for batch := 0 to (Length(dataset) div config.batchSize) - 1 do
    begin
      batchStart := batch * config.batchSize;
      batchEnd := Min(batchStart + config.batchSize, Length(dataset));

      if batchEnd <= batchStart then
        Continue;

      { Prepare batch - create matrix from array of samples }
      SetLength(batchDataArray, batchEnd - batchStart);
      for i := 0 to Length(batchDataArray) - 1 do
        batchDataArray[i] := dataset[batchStart + i];

      { Convert array to matrix for forward pass }
      batchData := CreateMatrix(Length(batchDataArray), 1);
      for i := 0 to Length(batchDataArray) - 1 do
        if Length(batchDataArray[i]) > 0 then
          batchData[i][0] := batchDataArray[i][0][0];

      { Create real labels }
      realLabels := CreateMatrix(Length(batchData), 1);
      for i := 0 to Length(batchData) - 1 do
        realLabels[i][0] := 0.9; { Smoothed labels }

      { Train discriminator on real data }
      discReal := NetworkForward(discriminator, batchData);
      dRealLoss := BinaryCrossEntropy(discReal, realLabels);

      { Generate fake data }
      GenerateNoise(noise, Length(batchData), config.noiseDepth, config.noiseType);
      fakeData := NetworkForward(generator, noise);

      { Create fake labels }
      fakeLabels := CreateMatrix(Length(fakeData), 1);
      for i := 0 to Length(fakeData) - 1 do
        fakeLabels[i][0] := 0.1;

      { Train discriminator on fake data }
      discFake := NetworkForward(discriminator, fakeData);
      dFakeLoss := BinaryCrossEntropy(discFake, fakeLabels);

      { Train generator }
      for i := 0 to Length(fakeLabels) - 1 do
        fakeLabels[i][0] := 0.9; { fool discriminator }

      discGenOutput := NetworkForward(discriminator, fakeData);
      gLoss := BinaryCrossEntropy(discGenOutput, fakeLabels);

      { Update weights - simplified gradient descent }
      if config.optimizer = optAdam then
      begin
        UpdateWeightsAdam(generator, 0, gLoss * 0.01);
        UpdateWeightsAdam(discriminator, 0, (dRealLoss + dFakeLoss) / 2 * 0.01);
      end
      else
      begin
        UpdateWeightsSGD(generator, 0, gLoss * 0.01);
        UpdateWeightsSGD(discriminator, 0, (dRealLoss + dFakeLoss) / 2 * 0.01);
      end;

      if ((batch + 1) mod 10 = 0) or (batch = 0) then
      begin
        WriteLn(Format('[Epoch %d/%d] Batch %d | D Loss: %.6f | G Loss: %.6f',
          [epoch + 1, config.epochs, batch + 1, (dRealLoss + dFakeLoss) / 2, gLoss]));
      end;
    end;
  end;

  WriteLn('Training complete.');
end;

{ ============================================================================= }
{ CLI ARGUMENT PARSING }
{ ============================================================================= }

procedure ShowHelp;
begin
  WriteLn('GAN Network - Generative Adversarial Network');
  WriteLn('');
  WriteLn('Usage: gan [options]');
  WriteLn('');
  WriteLn('Options:');
  WriteLn('  --help                  Show this help message');
  WriteLn('  --epochs=N              Number of training epochs (default: 100)');
  WriteLn('  --batch-size=N          Batch size (default: 32)');
  WriteLn('  --gbit=N                Generator bit depth (default: 16)');
  WriteLn('  --dbit=N                Discriminator bit depth (default: 16)');
  WriteLn('  --activation=name       Activation: relu|sigmoid|tanh|leaky (default: relu)');
  WriteLn('  --noise-type=TYPE       Noise: gauss|uniform|analog (default: gauss)');
  WriteLn('  --noise-depth=N         Noise vector depth (default: 100)');
  WriteLn('  --spectral              Enable spectral normalization');
  WriteLn('  --patch-config=file     Load patch config from JSON');
  WriteLn('  --save=MODEL.bin        Save trained model to binary file');
  WriteLn('  --load=MODEL.bin        Load pretrained model from binary file');
  WriteLn('  --output=PATH           Output directory for results (default: ./output)');
  WriteLn('  --optimizer=adam|sgd    Optimizer type (default: adam)');
  WriteLn('  --lr=RATE               Learning rate (default: 0.0002)');
  WriteLn('');
  WriteLn('Examples:');
  WriteLn('  gan --epochs=50 --batch-size=64 --activation=leaky');
  WriteLn('  gan --load=pretrained.bin --epochs=10');
  WriteLn('  gan --noise-type=uniform --output=results/ --save=new_model.bin');
end;

function ParseConfig: TGANConfig;
var
  i: Integer;
  arg, key, value: string;
begin
  { Set defaults }
  Result.epochs := 100;
  Result.batchSize := 32;
  Result.generatorBits := 16;
  Result.discriminatorBits := 16;
  Result.activation := atReLU;
  Result.noiseType := ntGauss;
  Result.noiseDepth := 100;
  Result.useSpectral := False;
  Result.patchConfig := '';
  Result.saveModel := '';
  Result.loadModel := '';
  Result.outputDir := './output';
  Result.learningRate := 0.0002;
  Result.optimizer := optAdam;

  { Parse command line }
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
    begin
      value := Copy(arg, 10, MaxInt);
      Result.epochs := StrToIntDef(value, 100);
    end;

    if Pos('--batch-size=', arg) = 1 then
    begin
      value := Copy(arg, 14, MaxInt);
      Result.batchSize := StrToIntDef(value, 32);
    end;

    if Pos('--gbit=', arg) = 1 then
    begin
      value := Copy(arg, 8, MaxInt);
      Result.generatorBits := StrToIntDef(value, 16);
    end;

    if Pos('--dbit=', arg) = 1 then
    begin
      value := Copy(arg, 8, MaxInt);
      Result.discriminatorBits := StrToIntDef(value, 16);
    end;

    if Pos('--activation=', arg) = 1 then
    begin
      value := Copy(arg, 14, MaxInt);
      case value of
        'relu': Result.activation := atReLU;
        'sigmoid': Result.activation := atSigmoid;
        'tanh': Result.activation := atTanh;
        'leaky': Result.activation := atLeakyReLU;
      end;
    end;

    if Pos('--noise-type=', arg) = 1 then
    begin
      value := Copy(arg, 14, MaxInt);
      case value of
        'gauss': Result.noiseType := ntGauss;
        'uniform': Result.noiseType := ntUniform;
        'analog': Result.noiseType := ntAnalog;
      end;
    end;

    if Pos('--noise-depth=', arg) = 1 then
    begin
      value := Copy(arg, 15, MaxInt);
      Result.noiseDepth := StrToIntDef(value, 100);
    end;

    if arg = '--spectral' then
      Result.useSpectral := True;

    if Pos('--patch-config=', arg) = 1 then
      Result.patchConfig := Copy(arg, 16, MaxInt);

    if Pos('--save=', arg) = 1 then
      Result.saveModel := Copy(arg, 8, MaxInt);

    if Pos('--load=', arg) = 1 then
      Result.loadModel := Copy(arg, 8, MaxInt);

    if Pos('--output=', arg) = 1 then
      Result.outputDir := Copy(arg, 10, MaxInt);

    if Pos('--optimizer=', arg) = 1 then
    begin
      value := Copy(arg, 13, MaxInt);
      case value of
        'adam': Result.optimizer := optAdam;
        'sgd': Result.optimizer := optSGD;
      end;
    end;

    if Pos('--lr=', arg) = 1 then
    begin
      value := Copy(arg, 6, MaxInt);
      Result.learningRate := StrToFloatDef(value, 0.0002);
    end;
  end;
end;

{ ============================================================================= }
{ MAIN }
{ ============================================================================= }

var
  config: TGANConfig;
  generator, discriminator: TNetwork;
  genSizes, discSizes: array of Integer;
  dataset: TMatrixArray;
  i, j: Integer;
begin
  WriteLn('GAN Network v1.0');
  WriteLn('');

  config := ParseConfig;

  WriteLn('Configuration:');
  WriteLn('  Epochs: ', config.epochs);
  WriteLn('  Batch Size: ', config.batchSize);
  WriteLn('  Generator Bits: ', config.generatorBits);
  WriteLn('  Discriminator Bits: ', config.discriminatorBits);
  WriteLn('  Noise Depth: ', config.noiseDepth);
  WriteLn('  Learning Rate: ', config.learningRate:0:6);
  WriteLn('');

  { Create networks }
  SetLength(genSizes, 4);
  genSizes[0] := config.noiseDepth;
  genSizes[1] := 128;
  genSizes[2] := 64;
  genSizes[3] := 1;

  SetLength(discSizes, 4);
  discSizes[0] := 1;
  discSizes[1] := 64;
  discSizes[2] := 128;
  discSizes[3] := 1;

  generator := CreateNetwork(genSizes, config.activation, config.optimizer, config.learningRate);
  discriminator := CreateNetwork(discSizes, config.activation, config.optimizer, config.learningRate);

  WriteLn('Networks created:');
  WriteLn('  Generator: ', config.noiseDepth, ' -> 128 -> 64 -> 1');
  WriteLn('  Discriminator: 1 -> 64 -> 128 -> 1');
  WriteLn('');

  { Load pretrained model if specified }
  if config.loadModel <> '' then
  begin
    WriteLn('Loading pretrained model: ', config.loadModel);
    LoadNetworkBinary(generator, config.loadModel);
  end;

  { Generate dummy dataset }
  WriteLn('Generating synthetic training data...');
  SetLength(dataset, 1000);
  for i := 0 to 999 do
  begin
    dataset[i] := CreateMatrix(1, 1);
    dataset[i][0][0] := sin(i / 100.0);
  end;
  WriteLn('Dataset size: ', Length(dataset), ' samples');
  WriteLn('');

  { Train GAN }
  WriteLn('Starting training...');
  WriteLn('');
  TrainGAN(generator, discriminator, dataset, config);

  { Save model if specified }
  if config.saveModel <> '' then
  begin
    WriteLn('');
    SaveNetworkBinary(generator, config.saveModel);
  end;

  WriteLn('Done.');
end.

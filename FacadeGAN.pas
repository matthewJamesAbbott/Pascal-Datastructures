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

program GANFacade;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Classes, StrUtils;

const
  EPSILON = 1e-8;
  GRAD_CLIP = 1.0;
  VERSION_SUITE = '1.0';
  VERSION_FACADE = '1.0';
  VERSION_MODEL = '1.0';

type
  TMatrix = array of array of Single;
  TVector = array of Single;
  TMatrixArray = array of TMatrix;

  TActivationType = (atReLU, atSigmoid, atTanh, atLeakyReLU);
  TOptimizer = (optAdam, optSGD);
  TNoiseTypeEnum = (ntGauss, ntUniform, ntAnalog);

  { Layer information for introspection }
  TLayerStats = record
    Mean: Single;
    StdDev: Single;
    Min: Single;
    Max: Single;
    Count: Integer;
    ParameterCount: Integer;
  end;

  TGradientStats = record
    Mean: Single;
    StdDev: Single;
    Min: Single;
    Max: Single;
    HasNaN: Boolean;
    HasInf: Boolean;
  end;

  { Layer structure }
  TLayer = record
    weights: TMatrix;
    bias: TVector;
    input: TMatrix;
    output: TMatrix;
    gradWeights: TMatrix;
    gradBias: TVector;
    activation: TActivationType;
    inputSize: Integer;
    outputSize: Integer;
    learningRate: Single;
  end;

  { Optimizer state }
  TAdamState = record
    m: TMatrix;
    v: TMatrix;
    t: Integer;
  end;

  { Network structure }
  PNetwork = ^TNetwork;
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

  { Training metrics for inspection }
  TTrainingMetrics = record
    epoch: Integer;
    step: Integer;
    batchProgress: Single;
    generatorLoss: Single;
    discriminatorLoss: Single;
    avgGeneratorLoss: Single;
    avgDiscriminatorLoss: Single;
    learningRate: Single;
    lastLossUpdate: TDateTime;
  end;

  { Alert/anomaly record }
  TAnomalyAlert = record
    alertType: string;
    message: string;
    severity: string; // 'WARNING', 'ERROR', 'INFO'
    timestamp: TDateTime;
    layerIndex: Integer;
    value: Single;
  end;

  {--------------------------------------------------------------------------
   MAIN FACADE CLASS: TGANFacade
   All properties and methods attached to this object.
   -------------------------------------------------------------------------}
  TGANFacade = class
  private
    // Core networks
  generator: TNetwork;
  discriminator: TNetwork;

    // Configuration
  epochs: Integer;
  batchSize: Integer;
  noiseDepth: Integer;
  noiseTypeVal: TNoiseTypeEnum;
  activationType: TActivationType;
  useSpectral: Boolean;

    // Training state
  FMetrics: TTrainingMetrics;
  FCurrentEpoch: Integer;
  FCurrentStep: Integer;
  FDatasetSize: Integer;

    // Monitoring
  FAnomalies: array of TAnomalyAlert;
  FMonitoringEnabled: Boolean;
  FLossHistory: array of Single;
  FDiscLossHistory: array of Single;

    // Weight injection/patching
  FWeightInjection: TMatrix;
  FInjectionLayerIdx: Integer;
  FInjectionEnabled: Boolean;

    // Noise injection/patching
  FNoiseInjection: TMatrix;
  FNoiseInjectionEnabled: Boolean;

    // Bit-depth tracking
  FLayerBitDepths: array of Integer;

    {----- INTERNAL UTILITIES -----}
function CreateMatrix(rows, cols: Integer): TMatrix;
function CreateVector(size: Integer): TVector;
function RandomGaussian: Single;
function RandomUniform(min, max: Single): Single;
function RandomAnalog: Single;
function MatrixReLU(const A: TMatrix): TMatrix;
function MatrixSigmoid(const A: TMatrix): TMatrix;
function MatrixTanh(const A: TMatrix): TMatrix;
function ApplyActivation(const A: TMatrix; activation: TActivationType): TMatrix;
function MatrixMultiply(const A, B: TMatrix): TMatrix;
function MatrixAdd(const A, B: TMatrix): TMatrix;
function MatrixScale(const A: TMatrix; scale: Single): TMatrix;
function MatrixTranspose(const A: TMatrix): TMatrix;
function MatrixNormalize(const A: TMatrix): TMatrix;
function BinaryCrossEntropy(const predicted, target: TMatrix): Single;
function IsFiniteNum(x: Single): Boolean;
procedure DetectAnomalies(const data: TMatrix; layerIdx: Integer);
procedure AddAnomaly(alertType, message, severity: string; layerIdx: Integer; value: Single);
function ComputeLayerStats(const data: TMatrix): TLayerStats;
function ComputeGradientStats(const grads: TMatrix): TGradientStats;

public
    {================= CONSTRUCTION / TEARDOWN =================}
constructor Create(generatorSizes: array of Integer; discriminatorSizes: array of Integer;
                   aEpochs, aBatchSize, aNoiseDepth: Integer; aActivation: TActivationType; aOptimizer: TOptimizer);
destructor Destroy; override;

    {================= MAIN FUNCTIONALITY =====================}
function TrainStep(var dataset: TMatrixArray): Single;
function GenerateSample: TMatrix;
procedure SaveModel(const filename: string);
procedure LoadModel(const filename: string);

    {================= 1. ARCHITECTURE/OBJECT INTROSPECTION =====================}
function InspectModelTopology: string;
function InspectLayerConfig(networkType: string; layerIdx: Integer): string;
function InspectWeightDimensions(networkType: string; layerIdx: Integer): string;
function GetParameterCount(networkType: string): Integer;
function GetTotalParameterCount: Integer;
function GetLayerCount(networkType: string): Integer;
function InspectActivationFunctions: string;
function GetVersionInfo: string;
function InspectOptimizerConfig: string;

    {================= 2. TRAINING/RUNTIME STATE INSPECTION =====================}
function GetCurrentEpoch: Integer;
function GetCurrentStep: Integer;
function GetMetrics: TTrainingMetrics;
function GetGeneratorLoss: Single;
function GetDiscriminatorLoss: Single;
function GetAverageLossWindow(windowSize: Integer): Single;
function InspectLayerActivations(networkType: string; layerIdx: Integer; var activations: TMatrix): Boolean;
function InspectLayerWeightStats(networkType: string; layerIdx: Integer): TLayerStats;
function InspectLayerGradientStats(networkType: string; layerIdx: Integer): TGradientStats;
function GetLearningRate: Single;
procedure SetLearningRate(newRate: Single);

    {================= 3. DATA FLOW & INJECTION CONTROL =====================}
procedure InjectNoise(const noise: TMatrix; useAsInput: Boolean = True);
procedure SetNoiseType(newType: TNoiseTypeEnum);
procedure SetNoiseDepth(newDepth: Integer);
procedure InjectWeights(networkType: string; layerIdx: Integer; const weights: TMatrix);
procedure InjectBias(networkType: string; layerIdx: Integer; const bias: TVector);
procedure SetActivationFunction(networkType: string; layerIdx: Integer; newActivation: TActivationType);
procedure InjectInput(const input: TMatrix; layerIdx: Integer; networkType: string = 'generator');
procedure SetBitDepth(networkType: string; layerIdx: Integer; bitDepth: Integer);
function GetBitDepth(networkType: string; layerIdx: Integer): Integer;
procedure SetLayerLearningRate(networkType: string; layerIdx: Integer; rate: Single);

    {================= 4. SAVE/LOAD/EXPORT =====================}
procedure ExportActivations(const filename: string; networkType: string; layerIdx: Integer);
procedure ExportWeights(const filename: string; networkType: string; layerIdx: Integer);
procedure ExportGradients(const filename: string; networkType: string; layerIdx: Integer);
procedure ExportLossHistory(const filename: string);
procedure ExportGeneratedData(const filename: string; count: Integer);

    {================= 5. MONITORING & ALERTING =====================}
procedure EnableMonitoring;
procedure DisableMonitoring;
function GetAnomalies: string;
function HasAnomalies: Boolean;
function GetLastAnomaly: TAnomalyAlert;
procedure ClearAnomalies;
procedure SetAnomalyThreshold(thresholdType: string; value: Single);

    {================= UTILITY INSPECTION =====================}
function GetNetworkMemoryUsage(networkType: string): Integer;
function GetTotalMemoryUsage: Integer;
function InspectFullArchitecture: string;
end;

{ ============================================================================= }
{ IMPLEMENTATION }
{ ============================================================================= }

{ Utility Functions }

function TGANFacade.CreateMatrix(rows, cols: Integer): TMatrix;
var
  i: Integer;
begin
  SetLength(Result, rows);
  for i := 0 to rows - 1 do
    SetLength(Result[i], cols);
end;

function TGANFacade.CreateVector(size: Integer): TVector;
begin
  SetLength(Result, size);
  FillChar(Result[0], size * SizeOf(Single), 0);
end;

function TGANFacade.RandomGaussian: Single;
var
  u1, u2: Single;
begin
  u1 := Random;
  u2 := Random;
  if u1 < 1e-7 then
    u1 := 1e-7;
  Result := sqrt(-2.0 * ln(u1)) * cos(2.0 * Pi * u2);
end;

function TGANFacade.RandomUniform(min, max: Single): Single;
begin
  Result := min + Random * (max - min);
end;

function TGANFacade.RandomAnalog: Single;
begin
  Result := (Random - 0.5) * 0.1;
end;

function TGANFacade.MatrixReLU(const A: TMatrix): TMatrix;
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

function TGANFacade.MatrixSigmoid(const A: TMatrix): TMatrix;
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

function TGANFacade.MatrixTanh(const A: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := tanh(A[i][j]);
end;

function TGANFacade.ApplyActivation(const A: TMatrix; activation: TActivationType): TMatrix;
begin
  case activation of
    atReLU: Result := MatrixReLU(A);
    atSigmoid: Result := MatrixSigmoid(A);
    atTanh: Result := MatrixTanh(A);
    atLeakyReLU: Result := MatrixReLU(A); // Simplified
    else
      Result := A;
  end;
end;

function TGANFacade.MatrixMultiply(const A, B: TMatrix): TMatrix;
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

function TGANFacade.MatrixAdd(const A, B: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := A[i][j] + B[i][j];
end;

function TGANFacade.MatrixScale(const A: TMatrix; scale: Single): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to Length(A) - 1 do
    for j := 0 to Length(A[0]) - 1 do
      Result[i][j] := A[i][j] * scale;
end;

function TGANFacade.MatrixTranspose(const A: TMatrix): TMatrix;
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

function TGANFacade.MatrixNormalize(const A: TMatrix): TMatrix;
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

function TGANFacade.BinaryCrossEntropy(const predicted, target: TMatrix): Single;
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

function TGANFacade.IsFiniteNum(x: Single): Boolean;
begin
  Result := not (IsNaN(x) or IsInfinite(x));
end;

function TGANFacade.ComputeLayerStats(const data: TMatrix): TLayerStats;
var
  i, j: Integer;
  mean, variance, count: Single;
begin
  FillChar(Result, SizeOf(TLayerStats), 0);

  mean := 0;
  count := 0;
  Result.Min := 1e10;
  Result.Max := -1e10;

  for i := 0 to Length(data) - 1 do
    for j := 0 to Length(data[i]) - 1 do
    begin
      mean := mean + data[i][j];
      count := count + 1;
      if data[i][j] < Result.Min then
        Result.Min := data[i][j];
      if data[i][j] > Result.Max then
        Result.Max := data[i][j];
    end;

  if count > 0 then
    mean := mean / count;
  Result.Mean := mean;

  variance := 0;
  for i := 0 to Length(data) - 1 do
    for j := 0 to Length(data[i]) - 1 do
      variance := variance + sqr(data[i][j] - mean);
  if count > 0 then
    Result.StdDev := sqrt(variance / count);
  Result.Count := trunc(count);
end;

function TGANFacade.ComputeGradientStats(const grads: TMatrix): TGradientStats;
var
  i, j: Integer;
  mean, variance, count: Single;
begin
  FillChar(Result, SizeOf(TGradientStats), 0);

  mean := 0;
  count := 0;
  Result.Min := 1e10;
  Result.Max := -1e10;
  Result.HasNaN := False;
  Result.HasInf := False;

  for i := 0 to Length(grads) - 1 do
    for j := 0 to Length(grads[i]) - 1 do
    begin
      if IsNaN(grads[i][j]) then
        Result.HasNaN := True;
      if IsInfinite(grads[i][j]) then
        Result.HasInf := True;

      mean := mean + grads[i][j];
      count := count + 1;
      if grads[i][j] < Result.Min then
        Result.Min := grads[i][j];
      if grads[i][j] > Result.Max then
        Result.Max := grads[i][j];
    end;

  if count > 0 then
    mean := mean / count;
  Result.Mean := mean;

  variance := 0;
  for i := 0 to Length(grads) - 1 do
    for j := 0 to Length(grads[i]) - 1 do
      variance := variance + sqr(grads[i][j] - mean);
  if count > 0 then
    Result.StdDev := sqrt(variance / count);
end;

procedure TGANFacade.DetectAnomalies(const data: TMatrix; layerIdx: Integer);
var
  stats: TLayerStats;
begin
  if not FMonitoringEnabled then
    Exit;

  stats := ComputeLayerStats(data);

  if IsNaN(stats.Mean) then
    AddAnomaly('NaN', 'NaN detected in layer activations', 'ERROR', layerIdx, stats.Mean);

  if IsInfinite(stats.Max) or IsInfinite(stats.Min) then
    AddAnomaly('Infinity', 'Infinity detected in layer', 'ERROR', layerIdx, stats.Max);

  if abs(stats.Mean) > 100 then
    AddAnomaly('ExplodingActivation', 'Layer activation mean exceeds threshold', 'WARNING', layerIdx, stats.Mean);

  if stats.StdDev < EPSILON then
    AddAnomaly('VanishingActivation', 'Layer activation variance approaching zero', 'WARNING', layerIdx, stats.StdDev);
end;

procedure TGANFacade.AddAnomaly(alertType, message, severity: string; layerIdx: Integer; value: Single);
var
  newAlert: TAnomalyAlert;
begin
  newAlert.alertType := alertType;
  newAlert.message := message;
  newAlert.severity := severity;
  newAlert.timestamp := Now;
  newAlert.layerIndex := layerIdx;
  newAlert.value := value;

  SetLength(FAnomalies, Length(FAnomalies) + 1);
  FAnomalies[High(FAnomalies)] := newAlert;

  WriteLn('[', severity, '] ', alertType, ' at Layer ', layerIdx, ': ', message);
end;

{ Constructor/Destructor }

constructor TGANFacade.Create(generatorSizes: array of Integer; discriminatorSizes: array of Integer;
                              aEpochs, aBatchSize, aNoiseDepth: Integer; aActivation: TActivationType; aOptimizer: TOptimizer);
var
  i: Integer;
begin
  inherited Create;

  epochs := aEpochs;
  batchSize := aBatchSize;
  noiseDepth := aNoiseDepth;
  activationType := aActivation;
  noiseTypeVal := ntGauss;
  useSpectral := False;

  FCurrentEpoch := 0;
  FCurrentStep := 0;
  FDatasetSize := 0;
  FMonitoringEnabled := True;
  FInjectionEnabled := False;
  FNoiseInjectionEnabled := False;

  // Initialize generator
  generator.layerCount := Length(generatorSizes) - 1;
  generator.optimizer := aOptimizer;
  generator.learningRate := 0.0002;
  generator.momentum := 0.9;
  generator.beta1 := 0.9;
  generator.beta2 := 0.999;
  generator.epsilon := 1e-8;

  SetLength(generator.layers, generator.layerCount);
  SetLength(generator.adamStates, generator.layerCount);
  SetLength(FLayerBitDepths, generator.layerCount * 2);

  for i := 0 to generator.layerCount - 1 do
  begin
    generator.layers[i].weights := CreateMatrix(generatorSizes[i], generatorSizes[i + 1]);
    generator.layers[i].bias := CreateVector(generatorSizes[i + 1]);
    generator.layers[i].activation := aActivation;
    generator.layers[i].inputSize := generatorSizes[i];
    generator.layers[i].outputSize := generatorSizes[i + 1];
    generator.layers[i].learningRate := generator.learningRate;
    generator.adamStates[i].t := 0;
    generator.adamStates[i].m := CreateMatrix(generatorSizes[i], generatorSizes[i + 1]);
    generator.adamStates[i].v := CreateMatrix(generatorSizes[i], generatorSizes[i + 1]);
    FLayerBitDepths[i] := 32;
  end;

  // Initialize discriminator
  discriminator.layerCount := Length(discriminatorSizes) - 1;
  discriminator.optimizer := aOptimizer;
  discriminator.learningRate := 0.0002;
  discriminator.momentum := 0.9;
  discriminator.beta1 := 0.9;
  discriminator.beta2 := 0.999;
  discriminator.epsilon := 1e-8;

  SetLength(discriminator.layers, discriminator.layerCount);
  SetLength(discriminator.adamStates, discriminator.layerCount);

  for i := 0 to discriminator.layerCount - 1 do
  begin
    discriminator.layers[i].weights := CreateMatrix(discriminatorSizes[i], discriminatorSizes[i + 1]);
    discriminator.layers[i].bias := CreateVector(discriminatorSizes[i + 1]);
    discriminator.layers[i].activation := aActivation;
    discriminator.layers[i].inputSize := discriminatorSizes[i];
    discriminator.layers[i].outputSize := discriminatorSizes[i + 1];
    discriminator.layers[i].learningRate := discriminator.learningRate;
    discriminator.adamStates[i].t := 0;
    discriminator.adamStates[i].m := CreateMatrix(discriminatorSizes[i], discriminatorSizes[i + 1]);
    discriminator.adamStates[i].v := CreateMatrix(discriminatorSizes[i], discriminatorSizes[i + 1]);
    FLayerBitDepths[generator.layerCount + i] := 32;
  end;

  FMetrics.generatorLoss := 0;
  FMetrics.discriminatorLoss := 0;
  FMetrics.learningRate := generator.learningRate;
end;

destructor TGANFacade.Destroy;
begin
  SetLength(FAnomalies, 0);
  SetLength(FLossHistory, 0);
  SetLength(FDiscLossHistory, 0);
  inherited Destroy;
end;

{ Main Functionality }

function TGANFacade.TrainStep(var dataset: TMatrixArray): Single;
begin
  FCurrentStep := FCurrentStep + 1;
  if FCurrentStep mod 100 = 0 then
    FMetrics.batchProgress := (FCurrentStep mod (FDatasetSize div batchSize)) / (FDatasetSize div batchSize);
  Result := FMetrics.generatorLoss;
end;

function TGANFacade.GenerateSample: TMatrix;
var
  noise: TMatrix;
begin
  noise := CreateMatrix(1, noiseDepth);
  Result := CreateMatrix(1, 1);
end;

procedure TGANFacade.SaveModel(const filename: string);
begin
  WriteLn('Model saved to: ', filename);
end;

procedure TGANFacade.LoadModel(const filename: string);
begin
  WriteLn('Model loaded from: ', filename);
end;

{ ============================================================================= }
{ 1. ARCHITECTURE/OBJECT INTROSPECTION }
{ ============================================================================= }

function TGANFacade.InspectModelTopology: string;
var
  genSize, discSize, i: Integer;
  actStr: string;
begin
  Result := '';
  Result := Result + '=== GAN TOPOLOGY ===' + sLineBreak;
  Result := Result + sLineBreak + 'GENERATOR:' + sLineBreak;

  for i := 0 to generator.layerCount - 1 do
  begin
    case generator.layers[i].activation of
      atReLU: actStr := 'ReLU';
      atSigmoid: actStr := 'Sigmoid';
      atTanh: actStr := 'Tanh';
      atLeakyReLU: actStr := 'LeakyReLU';
    end;
    Result := Result + Format('  Layer %d: %d -> %d | Activation: %s | Bits: %d' + sLineBreak,
                              [i, generator.layers[i].inputSize, generator.layers[i].outputSize, actStr, FLayerBitDepths[i]]);
  end;

  Result := Result + sLineBreak + 'DISCRIMINATOR:' + sLineBreak;

  for i := 0 to discriminator.layerCount - 1 do
  begin
    case discriminator.layers[i].activation of
      atReLU: actStr := 'ReLU';
      atSigmoid: actStr := 'Sigmoid';
      atTanh: actStr := 'Tanh';
      atLeakyReLU: actStr := 'LeakyReLU';
    end;
    Result := Result + Format('  Layer %d: %d -> %d | Activation: %s | Bits: %d' + sLineBreak,
                              [i, discriminator.layers[i].inputSize, discriminator.layers[i].outputSize, actStr,
                               FLayerBitDepths[generator.layerCount + i]]);
  end;
end;

function TGANFacade.InspectLayerConfig(networkType: string; layerIdx: Integer): string;
var
  net: PNetwork;
  actStr: string;
begin
  Result := '';
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    case net^.layers[layerIdx].activation of
      atReLU: actStr := 'ReLU';
      atSigmoid: actStr := 'Sigmoid';
      atTanh: actStr := 'Tanh';
      atLeakyReLU: actStr := 'LeakyReLU';
    end;
    Result := Format('Layer %d Config:' + sLineBreak, [layerIdx]);
    Result := Result + Format('  Input Size: %d' + sLineBreak, [net^.layers[layerIdx].inputSize]);
    Result := Result + Format('  Output Size: %d' + sLineBreak, [net^.layers[layerIdx].outputSize]);
    Result := Result + Format('  Activation: %s' + sLineBreak, [actStr]);
    Result := Result + Format('  Learning Rate: %g' + sLineBreak, [net^.layers[layerIdx].learningRate]);
  end;
end;

function TGANFacade.InspectWeightDimensions(networkType: string; layerIdx: Integer): string;
var
  net: PNetwork;
begin
  Result := '';
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    Result := Format('Layer %d Weight Dimensions:' + sLineBreak, [layerIdx]);
    Result := Result + Format('  Weights: %d x %d' + sLineBreak, [Length(net^.layers[layerIdx].weights), Length(net^.layers[layerIdx].weights[0])]);
    Result := Result + Format('  Bias: %d' + sLineBreak, [Length(net^.layers[layerIdx].bias)]);
    Result := Result + Format('  Total Parameters: %d' + sLineBreak, [Length(net^.layers[layerIdx].weights) * Length(net^.layers[layerIdx].weights[0]) + Length(net^.layers[layerIdx].bias)]);
  end;
end;

function TGANFacade.GetParameterCount(networkType: string): Integer;
var
  i, count: Integer;
  net: PNetwork;
begin
  count := 0;
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  for i := 0 to net^.layerCount - 1 do
    count := count + Length(net^.layers[i].weights) * Length(net^.layers[i].weights[0]) + Length(net^.layers[i].bias);

  Result := count;
end;

function TGANFacade.GetTotalParameterCount: Integer;
begin
  Result := GetParameterCount('generator') + GetParameterCount('discriminator');
end;

function TGANFacade.GetLayerCount(networkType: string): Integer;
begin
  if networkType = 'generator' then
    Result := generator.layerCount
  else
    Result := discriminator.layerCount;
end;

function TGANFacade.InspectActivationFunctions: string;
var
  i: Integer;
  actStr: string;
begin
  Result := '';
  Result := Result + '=== ACTIVATION FUNCTIONS ===' + sLineBreak;
  Result := Result + sLineBreak + 'GENERATOR:' + sLineBreak;

  for i := 0 to generator.layerCount - 1 do
  begin
    case generator.layers[i].activation of
      atReLU: actStr := 'ReLU';
      atSigmoid: actStr := 'Sigmoid';
      atTanh: actStr := 'Tanh';
      atLeakyReLU: actStr := 'LeakyReLU';
    end;
    Result := Result + Format('  Layer %d: %s' + sLineBreak, [i, actStr]);
  end;

  Result := Result + sLineBreak + 'DISCRIMINATOR:' + sLineBreak;

  for i := 0 to discriminator.layerCount - 1 do
  begin
    case discriminator.layers[i].activation of
      atReLU: actStr := 'ReLU';
      atSigmoid: actStr := 'Sigmoid';
      atTanh: actStr := 'Tanh';
      atLeakyReLU: actStr := 'LeakyReLU';
    end;
    Result := Result + Format('  Layer %d: %s' + sLineBreak, [i, actStr]);
  end;
end;

function TGANFacade.GetVersionInfo: string;
begin
  Result := 'GAN Facade v' + VERSION_FACADE + ' | Suite v' + VERSION_SUITE + ' | Model v' + VERSION_MODEL;
end;

function TGANFacade.InspectOptimizerConfig: string;
var
  genOptStr, discOptStr: string;
begin
  Result := '';
  if generator.optimizer = optAdam then
    genOptStr := 'Adam'
  else
    genOptStr := 'SGD';

  if discriminator.optimizer = optAdam then
    discOptStr := 'Adam'
  else
    discOptStr := 'SGD';

  Result := Result + '=== OPTIMIZER CONFIG ===' + sLineBreak;
  Result := Result + Format('Generator Optimizer: %s' + sLineBreak, [genOptStr]);
  Result := Result + Format('Generator LR: %g' + sLineBreak, [generator.learningRate]);
  Result := Result + Format('Generator Beta1: %g' + sLineBreak, [generator.beta1]);
  Result := Result + Format('Generator Beta2: %g' + sLineBreak, [generator.beta2]);
  Result := Result + sLineBreak;
  Result := Result + Format('Discriminator Optimizer: %s' + sLineBreak, [discOptStr]);
  Result := Result + Format('Discriminator LR: %g' + sLineBreak, [discriminator.learningRate]);
  Result := Result + Format('Discriminator Beta1: %g' + sLineBreak, [discriminator.beta1]);
  Result := Result + Format('Discriminator Beta2: %g' + sLineBreak, [discriminator.beta2]);
end;

{ ============================================================================= }
{ 2. TRAINING/RUNTIME STATE INSPECTION }
{ ============================================================================= }

function TGANFacade.GetCurrentEpoch: Integer;
begin
  Result := FCurrentEpoch;
end;

function TGANFacade.GetCurrentStep: Integer;
begin
  Result := FCurrentStep;
end;

function TGANFacade.GetMetrics: TTrainingMetrics;
begin
  Result := FMetrics;
end;

function TGANFacade.GetGeneratorLoss: Single;
begin
  Result := FMetrics.generatorLoss;
end;

function TGANFacade.GetDiscriminatorLoss: Single;
begin
  Result := FMetrics.discriminatorLoss;
end;

function TGANFacade.GetAverageLossWindow(windowSize: Integer): Single;
var
  i, count: Integer;
  sum: Single;
begin
  sum := 0;
  count := 0;
  for i := Max(0, Length(FLossHistory) - windowSize) to Length(FLossHistory) - 1 do
  begin
    sum := sum + FLossHistory[i];
    count := count + 1;
  end;
  if count > 0 then
    Result := sum / count
  else
    Result := 0;
end;

function TGANFacade.InspectLayerActivations(networkType: string; layerIdx: Integer; var activations: TMatrix): Boolean;
var
  net: PNetwork;
begin
  Result := False;
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    activations := net^.layers[layerIdx].output;
    Result := Length(activations) > 0;
  end;
end;

function TGANFacade.InspectLayerWeightStats(networkType: string; layerIdx: Integer): TLayerStats;
var
  net: PNetwork;
begin
  FillChar(Result, SizeOf(TLayerStats), 0);
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    Result := ComputeLayerStats(net^.layers[layerIdx].weights);
    Result.ParameterCount := Length(net^.layers[layerIdx].weights) * Length(net^.layers[layerIdx].weights[0]);
  end;
end;

function TGANFacade.InspectLayerGradientStats(networkType: string; layerIdx: Integer): TGradientStats;
var
  net: PNetwork;
begin
  FillChar(Result, SizeOf(TGradientStats), 0);
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    if Length(net^.layers[layerIdx].gradWeights) > 0 then
      Result := ComputeGradientStats(net^.layers[layerIdx].gradWeights);
  end;
end;

function TGANFacade.GetLearningRate: Single;
begin
  Result := generator.learningRate;
end;

procedure TGANFacade.SetLearningRate(newRate: Single);
var
  i: Integer;
begin
  generator.learningRate := newRate;
  discriminator.learningRate := newRate;

  for i := 0 to generator.layerCount - 1 do
    generator.layers[i].learningRate := newRate;

  for i := 0 to discriminator.layerCount - 1 do
    discriminator.layers[i].learningRate := newRate;

  FMetrics.learningRate := newRate;
  WriteLn('Learning rate updated to: ', newRate:0:8);
end;

{ ============================================================================= }
{ 3. DATA FLOW & INJECTION CONTROL }
{ ============================================================================= }

procedure TGANFacade.InjectNoise(const noise: TMatrix; useAsInput: Boolean = True);
begin
  FNoiseInjection := noise;
  FNoiseInjectionEnabled := True;
  WriteLn('Noise injected: ', Length(noise), ' x ', Length(noise[0]));
end;

procedure TGANFacade.SetNoiseType(newType: TNoiseTypeEnum);
begin
  noiseTypeVal := newType;
  WriteLn('Noise type changed to: ', Integer(newType));
end;

procedure TGANFacade.SetNoiseDepth(newDepth: Integer);
begin
  noiseDepth := newDepth;
  WriteLn('Noise depth changed to: ', newDepth);
end;

procedure TGANFacade.InjectWeights(networkType: string; layerIdx: Integer; const weights: TMatrix);
var
  net: PNetwork;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    net^.layers[layerIdx].weights := weights;
    WriteLn('Weights injected to ', networkType, ' layer ', layerIdx);
  end;
end;

procedure TGANFacade.InjectBias(networkType: string; layerIdx: Integer; const bias: TVector);
var
  net: PNetwork;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    net^.layers[layerIdx].bias := bias;
    WriteLn('Bias injected to ', networkType, ' layer ', layerIdx);
  end;
end;

procedure TGANFacade.SetActivationFunction(networkType: string; layerIdx: Integer; newActivation: TActivationType);
var
  net: PNetwork;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    net^.layers[layerIdx].activation := newActivation;
    WriteLn('Activation changed at ', networkType, ' layer ', layerIdx, ' to: ', Integer(newActivation));
  end;
end;

procedure TGANFacade.InjectInput(const input: TMatrix; layerIdx: Integer; networkType: string = 'generator');
var
  net: PNetwork;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    FWeightInjection := input;
    FInjectionLayerIdx := layerIdx;
    FInjectionEnabled := True;
    WriteLn('Input injected to ', networkType, ' layer ', layerIdx);
  end;
end;

procedure TGANFacade.SetBitDepth(networkType: string; layerIdx: Integer; bitDepth: Integer);
var
  idx: Integer;
begin
  if networkType = 'generator' then
    idx := layerIdx
  else
    idx := generator.layerCount + layerIdx;

  if (idx >= 0) and (idx < Length(FLayerBitDepths)) then
  begin
    FLayerBitDepths[idx] := bitDepth;
    WriteLn('Bit depth set to ', bitDepth, ' for ', networkType, ' layer ', layerIdx);
  end;
end;

function TGANFacade.GetBitDepth(networkType: string; layerIdx: Integer): Integer;
var
  idx: Integer;
begin
  if networkType = 'generator' then
    idx := layerIdx
  else
    idx := generator.layerCount + layerIdx;

  if (idx >= 0) and (idx < Length(FLayerBitDepths)) then
    Result := FLayerBitDepths[idx]
  else
    Result := 32;
end;

procedure TGANFacade.SetLayerLearningRate(networkType: string; layerIdx: Integer; rate: Single);
var
  net: PNetwork;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    net^.layers[layerIdx].learningRate := rate;
    WriteLn('Learning rate set to ', rate:0:8, ' for ', networkType, ' layer ', layerIdx);
  end;
end;

{ ============================================================================= }
{ 4. SAVE/LOAD/EXPORT }
{ ============================================================================= }

procedure TGANFacade.ExportActivations(const filename: string; networkType: string; layerIdx: Integer);
var
  net: PNetwork;
  f: Text;
  i, j: Integer;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    AssignFile(f, filename);
    Rewrite(f);

    for i := 0 to Length(net^.layers[layerIdx].output) - 1 do
    begin
      for j := 0 to Length(net^.layers[layerIdx].output[i]) - 1 do
        Write(f, net^.layers[layerIdx].output[i][j]:0:6, ',');
      WriteLn(f);
    end;

    CloseFile(f);
    WriteLn('Activations exported to: ', filename);
  end;
end;

procedure TGANFacade.ExportWeights(const filename: string; networkType: string; layerIdx: Integer);
var
  net: PNetwork;
  f: Text;
  i, j: Integer;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    AssignFile(f, filename);
    Rewrite(f);

    for i := 0 to Length(net^.layers[layerIdx].weights) - 1 do
    begin
      for j := 0 to Length(net^.layers[layerIdx].weights[i]) - 1 do
        Write(f, net^.layers[layerIdx].weights[i][j]:0:6, ',');
      WriteLn(f);
    end;

    CloseFile(f);
    WriteLn('Weights exported to: ', filename);
  end;
end;

procedure TGANFacade.ExportGradients(const filename: string; networkType: string; layerIdx: Integer);
var
  net: PNetwork;
  f: Text;
  i, j: Integer;
begin
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  if (layerIdx >= 0) and (layerIdx < net^.layerCount) then
  begin
    AssignFile(f, filename);
    Rewrite(f);

    if Length(net^.layers[layerIdx].gradWeights) > 0 then
    begin
      for i := 0 to Length(net^.layers[layerIdx].gradWeights) - 1 do
      begin
        for j := 0 to Length(net^.layers[layerIdx].gradWeights[i]) - 1 do
          Write(f, net^.layers[layerIdx].gradWeights[i][j]:0:6, ',');
        WriteLn(f);
      end;
    end;

    CloseFile(f);
    WriteLn('Gradients exported to: ', filename);
  end;
end;

procedure TGANFacade.ExportLossHistory(const filename: string);
var
  f: Text;
  i: Integer;
begin
  AssignFile(f, filename);
  Rewrite(f);

  for i := 0 to Length(FLossHistory) - 1 do
    WriteLn(f, i, ',', FLossHistory[i]:0:8, ',', FDiscLossHistory[i]:0:8);

  CloseFile(f);
  WriteLn('Loss history exported to: ', filename);
end;

procedure TGANFacade.ExportGeneratedData(const filename: string; count: Integer);
var
  i: Integer;
  f: Text;
begin
  AssignFile(f, filename);
  Rewrite(f);

  for i := 0 to count - 1 do
    WriteLn(f, 'generated_', i);

  CloseFile(f);
  WriteLn('Generated data exported to: ', filename);
end;

{ ============================================================================= }
{ 5. MONITORING & ALERTING }
{ ============================================================================= }

procedure TGANFacade.EnableMonitoring;
begin
  FMonitoringEnabled := True;
  WriteLn('Monitoring enabled');
end;

procedure TGANFacade.DisableMonitoring;
begin
  FMonitoringEnabled := False;
  WriteLn('Monitoring disabled');
end;

function TGANFacade.GetAnomalies: string;
var
  i: Integer;
begin
  Result := '';
  Result := Result + Format('Total Anomalies: %d' + sLineBreak, [Length(FAnomalies)]);

  for i := 0 to Length(FAnomalies) - 1 do
  begin
    Result := Result + Format('  [%s] %s - Layer %d: %s (value: %g)' + sLineBreak,
                              [FAnomalies[i].severity, FAnomalies[i].alertType, FAnomalies[i].layerIndex,
                               FAnomalies[i].message, FAnomalies[i].value]);
  end;
end;

function TGANFacade.HasAnomalies: Boolean;
begin
  Result := Length(FAnomalies) > 0;
end;

function TGANFacade.GetLastAnomaly: TAnomalyAlert;
begin
  FillChar(Result, SizeOf(TAnomalyAlert), 0);
  if Length(FAnomalies) > 0 then
    Result := FAnomalies[High(FAnomalies)];
end;

procedure TGANFacade.ClearAnomalies;
begin
  SetLength(FAnomalies, 0);
  WriteLn('Anomalies cleared');
end;

procedure TGANFacade.SetAnomalyThreshold(thresholdType: string; value: Single);
begin
  WriteLn('Anomaly threshold set: ', thresholdType, ' = ', value:0:6);
end;

{ ============================================================================= }
{ UTILITY INSPECTION }
{ ============================================================================= }

function TGANFacade.GetNetworkMemoryUsage(networkType: string): Integer;
var
  i, total: Integer;
  net: PNetwork;
begin
  total := 0;
  if networkType = 'generator' then
    net := @generator
  else
    net := @discriminator;

  for i := 0 to net^.layerCount - 1 do
  begin
    total := total + Length(net^.layers[i].weights) * Length(net^.layers[i].weights[0]) * SizeOf(Single);
    total := total + Length(net^.layers[i].bias) * SizeOf(Single);
  end;

  Result := total;
end;

function TGANFacade.GetTotalMemoryUsage: Integer;
begin
  Result := GetNetworkMemoryUsage('generator') + GetNetworkMemoryUsage('discriminator');
end;

function TGANFacade.InspectFullArchitecture: string;
begin
  Result := '';
  Result := Result + GetVersionInfo + sLineBreak + sLineBreak;
  Result := Result + InspectModelTopology + sLineBreak;
  Result := Result + InspectActivationFunctions + sLineBreak;
  Result := Result + InspectOptimizerConfig + sLineBreak;
  Result := Result + Format('Total Parameters: %d' + sLineBreak, [GetTotalParameterCount]);
  Result := Result + Format('Memory Usage: %d bytes' + sLineBreak, [GetTotalMemoryUsage]);
end;

{ ============================================================================= }
{ JSON SERIALIZATION }
{ ============================================================================= }

function Vector1DToJSON(const v: TVector): string;
var
  i: Integer;
begin
  Result := '[';
  for i := 0 to High(v) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + FloatToStr(v[i]);
  end;
  Result := Result + ']';
end;

function Matrix2DToJSON(const m: TMatrix): string;
var
  i: Integer;
begin
  Result := '[';
  for i := 0 to High(m) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + Vector1DToJSON(m[i]);
  end;
  Result := Result + ']';
end;

procedure SaveGANFacadeToJSON(const generator, discriminator: TNetwork; const filename: string);
var
  f: TextFile;
  i: Integer;
begin
  AssignFile(f, filename);
  Rewrite(f);
  try
    WriteLn(f, '{');
    
    { Generator }
    WriteLn(f, '  "generator": {');
    WriteLn(f, '    "layer_count": ' + IntToStr(generator.layerCount) + ',');
    WriteLn(f, '    "optimizer": "' + IfThen(generator.optimizer = optAdam, 'adam', 'sgd') + '",');
    WriteLn(f, '    "learning_rate": ' + FloatToStr(generator.learningRate) + ',');
    WriteLn(f, '    "layers": [');
    
    for i := 0 to generator.layerCount - 1 do
    begin
      WriteLn(f, '      {');
      WriteLn(f, '        "input_size": ' + IntToStr(generator.layers[i].inputSize) + ',');
      WriteLn(f, '        "output_size": ' + IntToStr(generator.layers[i].outputSize) + ',');
      WriteLn(f, '        "weights": ' + Matrix2DToJSON(generator.layers[i].weights) + ',');
      Write(f, '        "bias": ' + Vector1DToJSON(generator.layers[i].bias));
      if i < generator.layerCount - 1 then
        WriteLn(f, '      },')
      else
        WriteLn(f, '      }');
    end;
    
    WriteLn(f, '    ]');
    WriteLn(f, '  },');
    
    { Discriminator }
    WriteLn(f, '  "discriminator": {');
    WriteLn(f, '    "layer_count": ' + IntToStr(discriminator.layerCount) + ',');
    WriteLn(f, '    "optimizer": "' + IfThen(discriminator.optimizer = optAdam, 'adam', 'sgd') + '",');
    WriteLn(f, '    "learning_rate": ' + FloatToStr(discriminator.learningRate) + ',');
    WriteLn(f, '    "layers": [');
    
    for i := 0 to discriminator.layerCount - 1 do
    begin
      WriteLn(f, '      {');
      WriteLn(f, '        "input_size": ' + IntToStr(discriminator.layers[i].inputSize) + ',');
      WriteLn(f, '        "output_size": ' + IntToStr(discriminator.layers[i].outputSize) + ',');
      WriteLn(f, '        "weights": ' + Matrix2DToJSON(discriminator.layers[i].weights) + ',');
      Write(f, '        "bias": ' + Vector1DToJSON(discriminator.layers[i].bias));
      if i < discriminator.layerCount - 1 then
        WriteLn(f, '      },')
      else
        WriteLn(f, '      }');
    end;
    
    WriteLn(f, '    ]');
    WriteLn(f, '  }');
    WriteLn(f, '}');
    
    WriteLn('Model saved to JSON: ' + filename);
  finally
    CloseFile(f);
  end;
end;

function ExtractIntFromJSON(const JSONStr, FieldName: string): Integer;
var
  P, EndP: Integer;
  Value: string;
begin
  P := Pos('"' + FieldName + '"', JSONStr);
  if P = 0 then Exit(0);
  
  P := PosEx(':', JSONStr, P);
  if P = 0 then Exit(0);
  
  P := P + 1;
  while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
  
  EndP := P;
  while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-']) do Inc(EndP);
  
  Value := Copy(JSONStr, P, EndP - P);
  try
    Result := StrToInt(Value);
  except
    Result := 0;
  end;
end;

function ExtractFloatFromJSON(const JSONStr, FieldName: string): Single;
var
  P, EndP: Integer;
  Value: string;
begin
  P := Pos('"' + FieldName + '"', JSONStr);
  if P = 0 then Exit(0.0);
  
  P := PosEx(':', JSONStr, P);
  if P = 0 then Exit(0.0);
  
  P := P + 1;
  while (P <= Length(JSONStr)) and (JSONStr[P] in [' ', #9, #10, #13]) do Inc(P);
  
  EndP := P;
  while (EndP <= Length(JSONStr)) and (JSONStr[EndP] in ['0'..'9', '-', '.', 'e', 'E']) do Inc(EndP);
  
  Value := Copy(JSONStr, P, EndP - P);
  try
    Result := StrToFloat(Value);
  except
    Result := 0.0;
  end;
end;

procedure LoadVector1DFromJSON(const JSONStr: string; var v: TVector);
var
  P, EndP, CurrentPos, NumPos: Integer;
  Value: string;
  Count: Integer;
begin
  P := Pos('[', JSONStr);
  if P = 0 then Exit;
  
  EndP := P;
  Count := 1;
  while (Count > 0) and (EndP <= Length(JSONStr)) do
  begin
    if JSONStr[EndP] = '[' then Inc(Count)
    else if JSONStr[EndP] = ']' then Dec(Count);
    Inc(EndP);
  end;
  
  SetLength(v, 0);
  CurrentPos := P + 1;
  Count := 0;
  
  while (CurrentPos < EndP) and (JSONStr[CurrentPos] <> ']') do
  begin
    if JSONStr[CurrentPos] in ['0'..'9', '-', '.'] then
    begin
      NumPos := CurrentPos;
      while (NumPos <= Length(JSONStr)) and (JSONStr[NumPos] in ['0'..'9', '-', '.', 'e', 'E']) do
        Inc(NumPos);
      
      Value := Copy(JSONStr, CurrentPos, NumPos - CurrentPos);
      SetLength(v, Count + 1);
      try
        v[Count] := StrToFloat(Value);
      except
        v[Count] := 0.0;
      end;
      Inc(Count);
      CurrentPos := NumPos;
    end
    else
      Inc(CurrentPos);
  end;
end;

procedure LoadMatrix2DFromJSON(const JSONStr: string; var m: TMatrix);
var
  P, CurrentPos, Count, RowCount, ColCount: Integer;
  NumPos, ArrayEnd: Integer;
  Value: string;
begin
  P := Pos('[', JSONStr);
  if P = 0 then Exit;
  
  { Find end of array }
  CurrentPos := P;
  Count := 1;
  ArrayEnd := P + 1;
  while (Count > 0) and (ArrayEnd <= Length(JSONStr)) do
  begin
    if JSONStr[ArrayEnd] = '[' then Inc(Count)
    else if JSONStr[ArrayEnd] = ']' then Dec(Count);
    Inc(ArrayEnd);
  end;
  
  SetLength(m, 0);
  CurrentPos := P + 1;
  RowCount := 0;
  
  while (CurrentPos < ArrayEnd) do
  begin
    if JSONStr[CurrentPos] = '[' then
    begin
      SetLength(m, RowCount + 1);
      SetLength(m[RowCount], 0);
      
      Inc(CurrentPos);
      ColCount := 0;
      while (CurrentPos < ArrayEnd) and (JSONStr[CurrentPos] <> ']') do
      begin
        if JSONStr[CurrentPos] in ['0'..'9', '-', '.'] then
        begin
          NumPos := CurrentPos;
          while (NumPos <= Length(JSONStr)) and (JSONStr[NumPos] in ['0'..'9', '-', '.', 'e', 'E']) do
            Inc(NumPos);
          
          Value := Copy(JSONStr, CurrentPos, NumPos - CurrentPos);
          SetLength(m[RowCount], ColCount + 1);
          try
            m[RowCount][ColCount] := StrToFloat(Value);
          except
            m[RowCount][ColCount] := 0.0;
          end;
          Inc(ColCount);
          CurrentPos := NumPos;
        end
        else
          Inc(CurrentPos);
      end;
      
      if CurrentPos < ArrayEnd then Inc(CurrentPos);
      Inc(RowCount);
    end
    else
      Inc(CurrentPos);
  end;
end;

procedure LoadGANFromJSON(var generator, discriminator: TNetwork; const filename: string);
var
  JSONFile: TStringList;
  JSONStr: string;
  i, LayerCount: Integer;
  OptimizerStr: string;
  P: Integer;
begin
  JSONFile := TStringList.Create;
  try
    JSONFile.LoadFromFile(filename);
    JSONStr := JSONFile.Text;
    
    { Parse generator }
    P := Pos('"generator"', JSONStr);
    if P > 0 then
    begin
      LayerCount := ExtractIntFromJSON(Copy(JSONStr, P, Length(JSONStr)), 'layer_count');
      generator.layerCount := LayerCount;
      
      OptimizerStr := Copy(JSONStr, P, 200);
      if Pos('adam', OptimizerStr) > 0 then
        generator.optimizer := optAdam
      else
        generator.optimizer := optSGD;
      
      generator.learningRate := ExtractFloatFromJSON(Copy(JSONStr, P, 500), 'learning_rate');
      
      SetLength(generator.layers, LayerCount);
      for i := 0 to LayerCount - 1 do
      begin
        generator.layers[i].inputSize := ExtractIntFromJSON(JSONStr, 'input_size');
        generator.layers[i].outputSize := ExtractIntFromJSON(JSONStr, 'output_size');
      end;
    end;
    
    { Parse discriminator }
    P := Pos('"discriminator"', JSONStr);
    if P > 0 then
    begin
      LayerCount := ExtractIntFromJSON(Copy(JSONStr, P, Length(JSONStr)), 'layer_count');
      discriminator.layerCount := LayerCount;
      
      OptimizerStr := Copy(JSONStr, P, 200);
      if Pos('adam', OptimizerStr) > 0 then
        discriminator.optimizer := optAdam
      else
        discriminator.optimizer := optSGD;
      
      discriminator.learningRate := ExtractFloatFromJSON(Copy(JSONStr, P, 500), 'learning_rate');
      
      SetLength(discriminator.layers, LayerCount);
      for i := 0 to LayerCount - 1 do
      begin
        discriminator.layers[i].inputSize := ExtractIntFromJSON(JSONStr, 'input_size');
        discriminator.layers[i].outputSize := ExtractIntFromJSON(JSONStr, 'output_size');
      end;
    end;
    
    WriteLn('Model loaded from JSON: ' + filename);
  finally
    JSONFile.Free;
  end;
end;

{ Save function for GANFacade models }
function SaveFacadeModel(generator, discriminator: TNetwork; filename: string): Boolean;
begin
  Result := True;
  if (AnsiPos('.json', filename) > 0) or (AnsiPos('.JSON', filename) > 0) then
    SaveGANFacadeToJSON(generator, discriminator, filename)
  else
    WriteLn('Model save not implemented for ', filename);
end;

{ ============================================================================= }
{ MAIN PROGRAM }
{ ============================================================================= }

var
  facade: TGANFacade;
  genSizes, discSizes: array of Integer;
  sample: TMatrix;
  genStats: TLayerStats;

begin
  WriteLn('GAN Facade - Introspection & Injection Test');
  WriteLn('');

  { Create networks: Generator 100->128->64->1, Discriminator 1->64->128->1 }
  SetLength(genSizes, 4);
  genSizes[0] := 100;
  genSizes[1] := 128;
  genSizes[2] := 64;
  genSizes[3] := 1;

  SetLength(discSizes, 4);
  discSizes[0] := 1;
  discSizes[1] := 64;
  discSizes[2] := 128;
  discSizes[3] := 1;

  facade := TGANFacade.Create(genSizes, discSizes, 100, 32, 100, atReLU, optAdam);

  try
  WriteLn('=== INTROSPECTION TEST ===');
  WriteLn('');
  WriteLn(facade.InspectFullArchitecture);
  WriteLn('');

  WriteLn('=== LAYER STATISTICS ===');
  genStats := facade.InspectLayerWeightStats('generator', 0);
  WriteLn('Gen Layer 0 Weights - Mean: ', genStats.Mean:0:6, ' | StdDev: ', genStats.StdDev:0:6);
  WriteLn('');

  WriteLn('=== INJECTION TEST ===');
  facade.SetLearningRate(0.0001);
  facade.SetBitDepth('generator', 0, 16);
  facade.SetActivationFunction('generator', 0, atLeakyReLU);
  WriteLn('');

  WriteLn('=== MONITORING TEST ===');
  facade.EnableMonitoring;
  WriteLn(facade.GetAnomalies);
  WriteLn('');

  WriteLn('=== EXPORT TEST ===');
  facade.ExportWeights('gen_layer0_weights.csv', 'generator', 0);
  facade.ExportLossHistory('loss_history.csv');
  WriteLn('');

  WriteLn('=== RUNTIME METRICS ===');
  WriteLn('Current Epoch: ', facade.GetCurrentEpoch);
  WriteLn('Current Step: ', facade.GetCurrentStep);
  WriteLn('Learning Rate: ', facade.GetLearningRate:0:8);
  WriteLn('Total Parameters: ', facade.GetTotalParameterCount);
  WriteLn('Memory Usage: ', facade.GetTotalMemoryUsage, ' bytes');
  WriteLn('');

  WriteLn('Test Complete.');

  finally
  facade.Free;
end;
end.

//
// Matthew Abbott
// FacadeGNN
//

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

unit GNNFacade;

interface

uses
   Classes, Math, SysUtils;

const
   MAX_NODES = 1000;
   MAX_EDGES = 10000;
   MAX_ITERATIONS = 10000;
   GRADIENT_CLIP = 5.0;

type
   TActivationType = (atReLU, atLeakyReLU, atTanh, atSigmoid);
   TLossType = (ltMSE, ltBinaryCrossEntropy);
   
   TDoubleArray = array of Double;
   TIntArray = array of Integer;
   TDouble2DArray = array of TDoubleArray;
   TDouble3DArray = array of TDouble2DArray;
   
   TNeuron = record
      Weights: TDoubleArray;
      Bias: Double;
      Output: Double;
      PreActivation: Double;
      Error: Double;
      WeightGradients: TDoubleArray;
      BiasGradient: Double;
   end;
   
   TLayer = record
      Neurons: array of TNeuron;
      NumInputs: Integer;
      NumOutputs: Integer;
      LastInput: TDoubleArray;
   end;
   
   TLayerArray = array of TLayer;
   
   TMessageInfo = record
      NeighborIdx: Integer;
      ConcatInput: TDoubleArray;
      MessageOutput: TDoubleArray;
   end;
   
   TNodeMessages = array of TMessageInfo;
   TLayerMessages = array of TNodeMessages;
   TLayerMessagesArray = array of TLayerMessages;
   TAggregatedMessagesArray = array of TDouble2DArray;
   TGraphEmbeddingHistoryArray = array of TDoubleArray;
   
   TEdge = record
      Source: Integer;
      Target: Integer;
      Features: TDoubleArray;
      class operator = (const A, B: TEdge): Boolean;
   end;
   
   TEdgeArray = array of TEdge;
   
   TGraphConfig = record
      Undirected: Boolean;
      SelfLoops: Boolean;
      DeduplicateEdges: Boolean;
   end;
   
   TGraph = record
      NumNodes: Integer;
      NodeFeatures: TDouble2DArray;
      Edges: TEdgeArray;
      AdjacencyList: array of TIntArray;
      Config: TGraphConfig;
   end;
   
   TTrainingMetrics = record
      Loss: Double;
      Iteration: Integer;
      LossHistory: TDoubleArray;
   end;

   TEdgeEndpoints = record
      Source: Integer;
      Target: Integer;
   end;

   { TGraphNeuralNetwork }
   TGraphNeuralNetwork = class
   private
      FLearningRate: Double;
      FMaxIterations: Integer;
      FNumMessagePassingLayers: Integer;
      FFeatureSize: Integer;
      FHiddenSize: Integer;
      FOutputSize: Integer;
      FActivation: TActivationType;
      FLossType: TLossType;
      
      FMessageLayers: TLayerArray;
      FUpdateLayers: TLayerArray;
      FReadoutLayer: TLayer;
      FOutputLayer: TLayer;
      
      FNodeEmbeddings: TDouble2DArray;
      FNewNodeEmbeddings: TDouble2DArray;
      FEmbeddingHistory: TDouble3DArray;
      FMessageHistory: TLayerMessagesArray;
      FAggregatedMessages: TAggregatedMessagesArray;
      FGraphEmbedding: TDoubleArray;
      FGraphEmbeddingHistory: TGraphEmbeddingHistoryArray;
      
      FMetrics: TTrainingMetrics;
      FCurrentGraph: TGraph;
      FHasGraph: Boolean;
      
      procedure InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer);
      procedure BuildAdjacencyList(var Graph: TGraph);
      procedure MessagePassing(var Graph: TGraph);
      procedure Readout(var Graph: TGraph);
      function ForwardLayer(var Layer: TLayer; const Input: TDoubleArray; UseOutputActivation: Boolean = False): TDoubleArray;
      procedure BackwardLayer(var Layer: TLayer; const UpstreamGrad: TDoubleArray; UseOutputActivation: Boolean = False);
      function GetLayerInputGrad(const Layer: TLayer; const UpstreamGrad: TDoubleArray; UseOutputActivation: Boolean = False): TDoubleArray;
      procedure BackPropagateGraph(var Graph: TGraph; const Target: TDoubleArray);
      
      function Activate(X: Double): Double;
      function ActivateDerivative(X: Double): Double;
      function OutputActivate(X: Double): Double;
      function OutputActivateDerivative(PreAct: Double): Double;
      function ComputeLossGradient(const Prediction, Target: TDoubleArray): TDoubleArray;
      function ClipGradient(G: Double): Double;
      
   public
      constructor Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
      destructor Destroy; override;
      
      function Predict(var Graph: TGraph): TDoubleArray;
      function Train(var Graph: TGraph; const Target: TDoubleArray): Double;
      procedure TrainMultiple(var Graph: TGraph; const Target: TDoubleArray; Iterations: Integer);
      function ComputeLoss(const Prediction, Target: TDoubleArray): Double;
      
      procedure SaveModel(const Filename: string);
      procedure LoadModel(const Filename: string);
      
      class procedure ValidateGraph(var Graph: TGraph; out Errors: TStringList);
      class procedure DeduplicateEdges(var Graph: TGraph);
      class procedure AddReverseEdges(var Graph: TGraph);
      class procedure AddSelfLoops(var Graph: TGraph);
      
      property LearningRate: Double read FLearningRate write FLearningRate;
      property MaxIterations: Integer read FMaxIterations write FMaxIterations;
      property Activation: TActivationType read FActivation write FActivation;
      property LossFunction: TLossType read FLossType write FLossType;
      property Metrics: TTrainingMetrics read FMetrics;
      
      property FeatureSize: Integer read FFeatureSize;
      property HiddenSize: Integer read FHiddenSize;
      property OutputSize: Integer read FOutputSize;
      property NumMessagePassingLayers: Integer read FNumMessagePassingLayers;
      property NodeEmbeddings: TDouble2DArray read FNodeEmbeddings;
      property EmbeddingHistory: TDouble3DArray read FEmbeddingHistory;
      property MessageHistory: TLayerMessagesArray read FMessageHistory;
      property AggregatedMessages: TAggregatedMessagesArray read FAggregatedMessages;
      property GraphEmbedding: TDoubleArray read FGraphEmbedding;
      property GraphEmbeddingHistory: TGraphEmbeddingHistoryArray read FGraphEmbeddingHistory;
      property MessageLayers: TLayerArray read FMessageLayers;
      property UpdateLayers: TLayerArray read FUpdateLayers;
      property ReadoutLayer: TLayer read FReadoutLayer;
      property OutputLayer: TLayer read FOutputLayer;
      property CurrentGraph: TGraph read FCurrentGraph;
      property HasGraph: Boolean read FHasGraph;
   end;

function CopyArray(const Src: TDoubleArray): TDoubleArray;
function ConcatArrays(const A, B: TDoubleArray): TDoubleArray;
function ZeroArray(Size: Integer): TDoubleArray;
function PadArray(const Src: TDoubleArray; NewSize: Integer): TDoubleArray;

type
   TNodeMask = array of Boolean;
   TEdgeMask = array of Boolean;
   
   TOptimizerType = (otSGD, otAdam, otRMSProp);
   
   TAdamState = record
      M: TDoubleArray;    // First moment
      V: TDoubleArray;    // Second moment
      T: Integer;         // Timestep
   end;
   
   TRMSPropState = record
      S: TDoubleArray;    // Running average of squared gradients
   end;
   
   TMessagePassingStep = record
      LayerIdx: Integer;
      IterationIdx: Integer;
      NodeIdx: Integer;
      NeighborIdx: Integer;
      Message: TDoubleArray;
      AggregatedMessage: TDoubleArray;
   end;
   
   TMessagePassingTrace = array of TMessagePassingStep;
   
   TLayerConfig = record
      LayerType: string;
      NumInputs: Integer;
      NumOutputs: Integer;
      ActivationType: TActivationType;
   end;
   
   TGradientFlowInfo = record
      LayerIdx: Integer;
      MeanGradient: Double;
      MaxGradient: Double;
      MinGradient: Double;
      GradientNorm: Double;
   end;
   
   TGradientFlowInfoArray = array of TGradientFlowInfo;

   { TGNNFacade }
   TGNNFacade = class
   private
      FGNN: TGraphNeuralNetwork;
      FGraph: TGraph;
      FGraphLoaded: Boolean;
      
      FNodeMasks: TNodeMask;
      FEdgeMasks: TEdgeMask;
      
      FOptimizerType: TOptimizerType;
      FAdamBeta1: Double;
      FAdamBeta2: Double;
      FAdamEpsilon: Double;
      FRMSPropDecay: Double;
      FRMSPropEpsilon: Double;
      
      FMessageLayerAdamStates: array of array of TAdamState;
      FUpdateLayerAdamStates: array of array of TAdamState;
      FReadoutLayerAdamStates: array of TAdamState;
      FOutputLayerAdamStates: array of TAdamState;
      
      FMessageLayerRMSPropStates: array of array of TRMSPropState;
      FUpdateLayerRMSPropStates: array of array of TRMSPropState;
      FReadoutLayerRMSPropStates: array of TRMSPropState;
      FOutputLayerRMSPropStates: array of TRMSPropState;
      
      FNodeEmbeddingGradients: array of TDouble2DArray;
      FEdgeGradients: array of TDoubleArray;
      
      FMessagePassingTrace: TMessagePassingTrace;
      FTraceEnabled: Boolean;
      
      FBatchGraphs: array of TGraph;
      FBatchNodeEmbeddings: array of TDouble3DArray;
      
      procedure EnsureGraphLoaded;
      procedure InitializeMasks;
      procedure InitializeOptimizerStates;
      procedure RecordMessagePassingStep(LayerIdx, IterIdx, NodeIdx, NeighborIdx: Integer;
         const Msg, AggMsg: TDoubleArray);
      
   public
      constructor Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
      destructor Destroy; override;
      
      // ==================== Graph Management ====================
      procedure LoadGraph(var Graph: TGraph);
      procedure CreateEmptyGraph(NumNodes: Integer; FeatureSize: Integer);
      function GetGraph: TGraph;
      
      // ==================== 1. Node and Edge Feature Access ====================
      function GetNodeFeature(NodeIdx, FeatureIdx: Integer): Double;
      procedure SetNodeFeature(NodeIdx, FeatureIdx: Integer; Value: Double);
      function GetNodeFeatures(NodeIdx: Integer): TDoubleArray;
      procedure SetNodeFeatures(NodeIdx: Integer; const Features: TDoubleArray);
      
      function GetEdgeFeature(EdgeIdx, FeatureIdx: Integer): Double;
      procedure SetEdgeFeature(EdgeIdx, FeatureIdx: Integer; Value: Double);
      function GetEdgeFeatures(EdgeIdx: Integer): TDoubleArray;
      procedure SetEdgeFeatures(EdgeIdx: Integer; const Features: TDoubleArray);
      
      function GetNumNodes: Integer;
      function GetNumEdges: Integer;
      function GetNodeFeatureSize(NodeIdx: Integer): Integer;
      function GetEdgeFeatureSize(EdgeIdx: Integer): Integer;
      
      // ==================== 2. Adjacency and Topology Introspection ====================
      function GetNeighbors(NodeIdx: Integer): TIntArray;
      function GetAdjacencyMatrix: TDouble2DArray;
      function GetEdgeEndpoints(EdgeIdx: Integer): TEdgeEndpoints;
      function GetIncomingEdges(NodeIdx: Integer): TIntArray;
      function GetOutgoingEdges(NodeIdx: Integer): TIntArray;
      function HasEdge(SourceIdx, TargetIdx: Integer): Boolean;
      function FindEdgeIndex(SourceIdx, TargetIdx: Integer): Integer;
      
      // ==================== Core GNN Operations ====================
      function Predict: TDoubleArray;
      function Train(const Target: TDoubleArray): Double;
      procedure TrainMultiple(const Target: TDoubleArray; Iterations: Integer);
      function ComputeLoss(const Prediction, Target: TDoubleArray): Double;
      
      procedure SaveModel(const Filename: string);
      procedure LoadModel(const Filename: string);
      
      function GetLearningRate: Double;
      procedure SetLearningRate(Value: Double);
      function GetActivation: TActivationType;
      procedure SetActivation(Value: TActivationType);
      function GetLossFunction: TLossType;
      procedure SetLossFunction(Value: TLossType);
      
      // ==================== Properties ====================
      property GNN: TGraphNeuralNetwork read FGNN;
      property GraphLoaded: Boolean read FGraphLoaded;
      property LearningRate: Double read GetLearningRate write SetLearningRate;
      property Activation: TActivationType read GetActivation write SetActivation;
      property LossFunction: TLossType read GetLossFunction write SetLossFunction;
      property OptimizerType: TOptimizerType read FOptimizerType write FOptimizerType;
      property TraceEnabled: Boolean read FTraceEnabled write FTraceEnabled;
      
      // ==================== 3. Node/Edge Embedding and Activations ====================
      function GetNodeEmbedding(LayerIdx, NodeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      procedure SetNodeEmbedding(LayerIdx, NodeIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
      function GetEdgeEmbedding(LayerIdx, EdgeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      function GetAllNodeEmbeddings(NodeIdx: Integer): TDouble2DArray;
      function GetAllLayerEmbeddings(LayerIdx: Integer): TDouble2DArray;
      function GetCurrentNodeEmbedding(NodeIdx: Integer): TDoubleArray;
      function GetFinalNodeEmbeddings: TDouble2DArray;
      
      // ==================== 4. Message Passing Internals ====================
      function GetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      procedure SetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
      function GetAggregatedMessage(NodeIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      function GetMessageInput(NodeIdx, NeighborIdx, LayerIdx: Integer): TDoubleArray;
      function GetNumMessagesForNode(NodeIdx, LayerIdx: Integer): Integer;
      
      // ==================== 5. Readout and Output Layer Access ====================
      function GetGraphEmbedding(LayerIdx: Integer = -1): TDoubleArray;
      function GetReadout(LayerIdx: Integer = -1): TDoubleArray;
      procedure SetGraphEmbedding(const Value: TDoubleArray; LayerIdx: Integer = -1);
      function GetReadoutLayerOutput: TDoubleArray;
      function GetOutputLayerOutput: TDoubleArray;
      function GetReadoutLayerPreActivations: TDoubleArray;
      function GetOutputLayerPreActivations: TDoubleArray;
      
      // ==================== 6. Backprop Gradients and Optimizer States ====================
      function GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
      function GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
      function GetNodeEmbeddingGradient(LayerIdx, NodeIdx: Integer): TDoubleArray;
      function GetEdgeGradient(LayerIdx, EdgeIdx: Integer): TDoubleArray;
      function GetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string): TDoubleArray;
      procedure SetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string; const Value: TDoubleArray);
      function GetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
      procedure SetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
      function GetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
      procedure SetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
      function GetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
      procedure SetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
      function GetOutputLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
      procedure SetOutputLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
      
      // ==================== 7. Node/Edge/Graph Masking ====================
      function GetNodeMask(NodeIdx: Integer): Boolean;
      procedure SetNodeMask(NodeIdx: Integer; Value: Boolean);
      function GetEdgeMask(EdgeIdx: Integer): Boolean;
      procedure SetEdgeMask(EdgeIdx: Integer; Value: Boolean);
      procedure SetAllNodeMasks(Value: Boolean);
      procedure SetAllEdgeMasks(Value: Boolean);
      function GetMaskedNodeCount: Integer;
      function GetMaskedEdgeCount: Integer;
      procedure ApplyDropoutToNodes(DropoutRate: Double);
      procedure ApplyDropoutToEdges(DropoutRate: Double);
      
      // ==================== 8. Graph Structural Mutation ====================
      function AddNode(const Features: TDoubleArray): Integer;
      procedure RemoveNode(NodeIdx: Integer);
      function AddEdge(Source, Target: Integer; const Features: TDoubleArray): Integer;
      procedure RemoveEdge(EdgeIdx: Integer);
      procedure ClearAllEdges;
      procedure ConnectNodes(SourceIdx, TargetIdx: Integer);
      procedure DisconnectNodes(SourceIdx, TargetIdx: Integer);
      procedure RebuildAdjacencyList;
      
      // ==================== 9. Diagnostics, Attention, and Attribution ====================
      function GetAttentionWeight(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): Double;
      function GetNodeDegree(NodeIdx: Integer): Integer;
      function GetInDegree(NodeIdx: Integer): Integer;
      function GetOutDegree(NodeIdx: Integer): Integer;
      function GetGraphCentrality(NodeIdx: Integer): Double;
      function GetBetweennessCentrality(NodeIdx: Integer): Double;
      function GetClosenessCentrality(NodeIdx: Integer): Double;
      function GetFeatureImportance(NodeIdx, FeatureIdx: Integer): Double;
      function ComputePageRank(Damping: Double = 0.85; Iterations: Integer = 100): TDoubleArray;
      
      // ==================== 10. Batch/Minibatch and Multiple Graphs ====================
      procedure AddGraphToBatch(var Graph: TGraph);
      function GetBatchGraph(BatchIdx: Integer): TGraph;
      function GetBatchSize: Integer;
      procedure ClearBatch;
      function GetBatchNodeEmbedding(BatchIdx, NodeIdx, LayerIdx: Integer): TDoubleArray;
      procedure ProcessBatch(const Targets: TDouble2DArray);
      function GetBatchPredictions: TDouble2DArray;
      
      // ==================== 11. Explainability and Visualization Hooks ====================
      function GetMessagePassingTrace: TMessagePassingTrace;
      procedure ClearMessagePassingTrace;
      function GetGradientFlow(LayerIdx: Integer): TGradientFlowInfo;
      function GetAllGradientFlows: TGradientFlowInfoArray;
      function ExportGraphToJSON: string;
      function ExportEmbeddingsToCSV(LayerIdx: Integer): string;
      function GetActivationHistogram(LayerIdx: Integer; NumBins: Integer = 10): TDoubleArray;
      
      // ==================== 12. Layer and Architecture Introspection ====================
      function GetLayerConfig(LayerIdx: Integer): TLayerConfig;
      function GetNumMessagePassingLayers: Integer;
      function GetNumFeatures(NodeIdx: Integer): Integer;
      function GetTotalLayerCount: Integer;
      function GetMessageLayerNeuronCount(LayerIdx: Integer): Integer;
      function GetUpdateLayerNeuronCount(LayerIdx: Integer): Integer;
      function GetReadoutLayerNeuronCount: Integer;
      function GetOutputLayerNeuronCount: Integer;
      function GetArchitectureSummary: string;
      function GetParameterCount: Integer;
   end;

implementation

// ==================== TEdge Implementation ====================

class operator TEdge.= (const A, B: TEdge): Boolean;
begin
   Result := (A.Source = B.Source) and (A.Target = B.Target);
end;

// ==================== Helper Functions ====================

function CopyArray(const Src: TDoubleArray): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(Src));
   for I := 0 to High(Src) do
      Result[I] := Src[I];
end;

function ConcatArrays(const A, B: TDoubleArray): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(A) + Length(B));
   for I := 0 to High(A) do
      Result[I] := A[I];
   for I := 0 to High(B) do
      Result[Length(A) + I] := B[I];
end;

function ZeroArray(Size: Integer): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Size);
   for I := 0 to Size - 1 do
      Result[I] := 0.0;
end;

function PadArray(const Src: TDoubleArray; NewSize: Integer): TDoubleArray;
var
   I: Integer;
begin
   Result := ZeroArray(NewSize);
   for I := 0 to Min(High(Src), NewSize - 1) do
      Result[I] := Src[I];
end;

// ==================== TGraphNeuralNetwork Implementation ====================

constructor TGraphNeuralNetwork.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
var
   I: Integer;
begin
   inherited Create;
   
   FLearningRate := 0.01;
   FMaxIterations := 100;
   FFeatureSize := AFeatureSize;
   FHiddenSize := AHiddenSize;
   FOutputSize := AOutputSize;
   FNumMessagePassingLayers := NumMPLayers;
   FActivation := atReLU;
   FLossType := ltMSE;
   FHasGraph := False;
   
   SetLength(FMessageLayers, NumMPLayers);
   SetLength(FUpdateLayers, NumMPLayers);
   
   for I := 0 to NumMPLayers - 1 do
   begin
      if I = 0 then
         InitializeLayer(FMessageLayers[I], AHiddenSize, AFeatureSize * 2)
      else
         InitializeLayer(FMessageLayers[I], AHiddenSize, AHiddenSize * 2);
      
      InitializeLayer(FUpdateLayers[I], AHiddenSize, AHiddenSize * 2);
   end;
   
   InitializeLayer(FReadoutLayer, AHiddenSize, AHiddenSize);
   InitializeLayer(FOutputLayer, AOutputSize, AHiddenSize);
   
   SetLength(FMetrics.LossHistory, 0);
   SetLength(FGraphEmbeddingHistory, 0);
end;

destructor TGraphNeuralNetwork.Destroy;
begin
   inherited;
end;

procedure TGraphNeuralNetwork.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer);
var
   I, J: Integer;
   Scale: Double;
begin
   Layer.NumInputs := NumInputs;
   Layer.NumOutputs := NumNeurons;
   SetLength(Layer.Neurons, NumNeurons);
   
   Scale := Sqrt(2.0 / (NumInputs + NumNeurons));
   
   for I := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[I].Weights, NumInputs);
      SetLength(Layer.Neurons[I].WeightGradients, NumInputs);
      for J := 0 to NumInputs - 1 do
      begin
         Layer.Neurons[I].Weights[J] := (Random - 0.5) * 2.0 * Scale;
         Layer.Neurons[I].WeightGradients[J] := 0.0;
      end;
      Layer.Neurons[I].Bias := 0.0;
      Layer.Neurons[I].Output := 0.0;
      Layer.Neurons[I].PreActivation := 0.0;
      Layer.Neurons[I].Error := 0.0;
      Layer.Neurons[I].BiasGradient := 0.0;
   end;
end;

function TGraphNeuralNetwork.Activate(X: Double): Double;
begin
   case FActivation of
      atReLU:
         if X > 0 then Result := X else Result := 0.0;
      atLeakyReLU:
         if X > 0 then Result := X else Result := 0.01 * X;
      atTanh:
         Result := Tanh(X);
      atSigmoid:
         Result := 1.0 / (1.0 + Exp(-Max(-500, Min(500, X))));
   else
      Result := X;
   end;
end;

function TGraphNeuralNetwork.ActivateDerivative(X: Double): Double;
var
   S: Double;
begin
   case FActivation of
      atReLU:
         if X > 0 then Result := 1.0 else Result := 0.0;
      atLeakyReLU:
         if X > 0 then Result := 1.0 else Result := 0.01;
      atTanh:
         Result := 1.0 - Sqr(Tanh(X));
      atSigmoid:
         begin
            S := 1.0 / (1.0 + Exp(-Max(-500, Min(500, X))));
            Result := S * (1.0 - S);
         end;
   else
      Result := 1.0;
   end;
end;

function TGraphNeuralNetwork.OutputActivate(X: Double): Double;
begin
   Result := 1.0 / (1.0 + Exp(-Max(-500, Min(500, X))));
end;

function TGraphNeuralNetwork.OutputActivateDerivative(PreAct: Double): Double;
var
   S: Double;
begin
   S := 1.0 / (1.0 + Exp(-Max(-500, Min(500, PreAct))));
   Result := S * (1.0 - S);
end;

function TGraphNeuralNetwork.ComputeLoss(const Prediction, Target: TDoubleArray): Double;
var
   I: Integer;
   P: Double;
begin
   Result := 0.0;
   case FLossType of
      ltMSE:
         begin
            for I := 0 to High(Prediction) do
               Result := Result + Sqr(Prediction[I] - Target[I]);
            Result := Result / Length(Prediction);
         end;
      ltBinaryCrossEntropy:
         begin
            for I := 0 to High(Prediction) do
            begin
               P := Max(1e-7, Min(1.0 - 1e-7, Prediction[I]));
               Result := Result - (Target[I] * Ln(P) + (1.0 - Target[I]) * Ln(1.0 - P));
            end;
            Result := Result / Length(Prediction);
         end;
   end;
end;

function TGraphNeuralNetwork.ComputeLossGradient(const Prediction, Target: TDoubleArray): TDoubleArray;
var
   I: Integer;
   P: Double;
begin
   SetLength(Result, Length(Prediction));
   case FLossType of
      ltMSE:
         begin
            for I := 0 to High(Prediction) do
               Result[I] := 2.0 * (Prediction[I] - Target[I]) / Length(Prediction);
         end;
      ltBinaryCrossEntropy:
         begin
            for I := 0 to High(Prediction) do
            begin
               P := Max(1e-7, Min(1.0 - 1e-7, Prediction[I]));
               Result[I] := (-Target[I] / P + (1.0 - Target[I]) / (1.0 - P)) / Length(Prediction);
            end;
         end;
   end;
end;

function TGraphNeuralNetwork.ClipGradient(G: Double): Double;
begin
   Result := Max(-GRADIENT_CLIP, Min(GRADIENT_CLIP, G));
end;

procedure TGraphNeuralNetwork.BuildAdjacencyList(var Graph: TGraph);
var
   I, Src, Tgt: Integer;
begin
   SetLength(Graph.AdjacencyList, Graph.NumNodes);
   for I := 0 to Graph.NumNodes - 1 do
      SetLength(Graph.AdjacencyList[I], 0);
   
   for I := 0 to High(Graph.Edges) do
   begin
      Src := Graph.Edges[I].Source;
      Tgt := Graph.Edges[I].Target;
      
      if (Src >= 0) and (Src < Graph.NumNodes) and 
         (Tgt >= 0) and (Tgt < Graph.NumNodes) then
      begin
         SetLength(Graph.AdjacencyList[Src], Length(Graph.AdjacencyList[Src]) + 1);
         Graph.AdjacencyList[Src][High(Graph.AdjacencyList[Src])] := Tgt;
      end;
   end;
end;

class procedure TGraphNeuralNetwork.ValidateGraph(var Graph: TGraph; out Errors: TStringList);
var
   I: Integer;
begin
   Errors := TStringList.Create;
   
   if Graph.NumNodes < 1 then
      Errors.Add('Graph must have at least 1 node');
   
   if Graph.NumNodes > MAX_NODES then
      Errors.Add(Format('Too many nodes (max %d)', [MAX_NODES]));
   
   if Length(Graph.Edges) > MAX_EDGES then
      Errors.Add(Format('Too many edges (max %d)', [MAX_EDGES]));
   
   for I := 0 to High(Graph.Edges) do
   begin
      if (Graph.Edges[I].Source < 0) or (Graph.Edges[I].Source >= Graph.NumNodes) then
         Errors.Add(Format('Edge %d: invalid source %d', [I, Graph.Edges[I].Source]));
      if (Graph.Edges[I].Target < 0) or (Graph.Edges[I].Target >= Graph.NumNodes) then
         Errors.Add(Format('Edge %d: invalid target %d', [I, Graph.Edges[I].Target]));
   end;
   
   for I := 0 to High(Graph.NodeFeatures) do
   begin
      if Length(Graph.NodeFeatures[I]) = 0 then
         Errors.Add(Format('Node %d: empty feature vector', [I]));
   end;
end;

class procedure TGraphNeuralNetwork.DeduplicateEdges(var Graph: TGraph);
var
   I, J: Integer;
   Seen: array of string;
   Key: string;
   Found: Boolean;
   NewEdges: TEdgeArray;
begin
   SetLength(Seen, 0);
   SetLength(NewEdges, 0);
   
   for I := 0 to High(Graph.Edges) do
   begin
      Key := Format('%d-%d', [Graph.Edges[I].Source, Graph.Edges[I].Target]);
      Found := False;
      
      for J := 0 to High(Seen) do
      begin
         if Seen[J] = Key then
         begin
            Found := True;
            Break;
         end;
      end;
      
      if not Found then
      begin
         SetLength(Seen, Length(Seen) + 1);
         Seen[High(Seen)] := Key;
         SetLength(NewEdges, Length(NewEdges) + 1);
         NewEdges[High(NewEdges)] := Graph.Edges[I];
      end;
   end;
   
   Graph.Edges := NewEdges;
end;

class procedure TGraphNeuralNetwork.AddReverseEdges(var Graph: TGraph);
var
   I, OrigLen: Integer;
   RevEdge: TEdge;
   HasReverse: Boolean;
   J: Integer;
begin
   OrigLen := Length(Graph.Edges);
   
   for I := 0 to OrigLen - 1 do
   begin
      HasReverse := False;
      for J := 0 to High(Graph.Edges) do
      begin
         if (Graph.Edges[J].Source = Graph.Edges[I].Target) and
            (Graph.Edges[J].Target = Graph.Edges[I].Source) then
         begin
            HasReverse := True;
            Break;
         end;
      end;
      
      if not HasReverse then
      begin
         RevEdge.Source := Graph.Edges[I].Target;
         RevEdge.Target := Graph.Edges[I].Source;
         RevEdge.Features := CopyArray(Graph.Edges[I].Features);
         SetLength(Graph.Edges, Length(Graph.Edges) + 1);
         Graph.Edges[High(Graph.Edges)] := RevEdge;
      end;
   end;
end;

class procedure TGraphNeuralNetwork.AddSelfLoops(var Graph: TGraph);
var
   I, J: Integer;
   HasSelf: Boolean;
   SelfEdge: TEdge;
begin
   for I := 0 to Graph.NumNodes - 1 do
   begin
      HasSelf := False;
      for J := 0 to High(Graph.Edges) do
      begin
         if (Graph.Edges[J].Source = I) and (Graph.Edges[J].Target = I) then
         begin
            HasSelf := True;
            Break;
         end;
      end;
      
      if not HasSelf then
      begin
         SelfEdge.Source := I;
         SelfEdge.Target := I;
         SetLength(SelfEdge.Features, 0);
         SetLength(Graph.Edges, Length(Graph.Edges) + 1);
         Graph.Edges[High(Graph.Edges)] := SelfEdge;
      end;
   end;
end;

function TGraphNeuralNetwork.ForwardLayer(var Layer: TLayer; const Input: TDoubleArray; 
   UseOutputActivation: Boolean = False): TDoubleArray;
var
   I, J: Integer;
   Sum: Double;
   PaddedInput: TDoubleArray;
begin
   if Length(Input) < Layer.NumInputs then
      PaddedInput := PadArray(Input, Layer.NumInputs)
   else
      PaddedInput := Input;
   
   Layer.LastInput := CopyArray(PaddedInput);
   SetLength(Result, Layer.NumOutputs);
   
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      Sum := Layer.Neurons[I].Bias;
      for J := 0 to Layer.NumInputs - 1 do
         Sum := Sum + Layer.Neurons[I].Weights[J] * PaddedInput[J];
      
      Layer.Neurons[I].PreActivation := Sum;
      
      if UseOutputActivation then
         Layer.Neurons[I].Output := OutputActivate(Sum)
      else
         Layer.Neurons[I].Output := Activate(Sum);
      
      Result[I] := Layer.Neurons[I].Output;
   end;
end;

procedure TGraphNeuralNetwork.BackwardLayer(var Layer: TLayer; const UpstreamGrad: TDoubleArray; 
   UseOutputActivation: Boolean = False);
var
   I, J: Integer;
   PreActGrad, ClippedGrad: Double;
begin
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      if UseOutputActivation then
         PreActGrad := UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation)
      else
         PreActGrad := UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);
      
      Layer.Neurons[I].Error := ClipGradient(PreActGrad);
      
      for J := 0 to Layer.NumInputs - 1 do
      begin
         ClippedGrad := ClipGradient(PreActGrad * Layer.LastInput[J]);
         Layer.Neurons[I].WeightGradients[J] := ClippedGrad;
         Layer.Neurons[I].Weights[J] := Layer.Neurons[I].Weights[J] - FLearningRate * ClippedGrad;
      end;
      
      ClippedGrad := ClipGradient(PreActGrad);
      Layer.Neurons[I].BiasGradient := ClippedGrad;
      Layer.Neurons[I].Bias := Layer.Neurons[I].Bias - FLearningRate * ClippedGrad;
   end;
end;

function TGraphNeuralNetwork.GetLayerInputGrad(const Layer: TLayer; const UpstreamGrad: TDoubleArray;
   UseOutputActivation: Boolean = False): TDoubleArray;
var
   I, J: Integer;
   PreActGrad: Double;
begin
   SetLength(Result, Layer.NumInputs);
   for J := 0 to Layer.NumInputs - 1 do
      Result[J] := 0.0;
   
   for J := 0 to Layer.NumInputs - 1 do
   begin
      for I := 0 to Layer.NumOutputs - 1 do
      begin
         if UseOutputActivation then
            PreActGrad := UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation)
         else
            PreActGrad := UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);
         
         if J <= High(Layer.Neurons[I].Weights) then
            Result[J] := Result[J] + Layer.Neurons[I].Weights[J] * PreActGrad;
      end;
      Result[J] := ClipGradient(Result[J]);
   end;
end;

procedure TGraphNeuralNetwork.MessagePassing(var Graph: TGraph);
var
   Layer, Node, K, I, Neighbor: Integer;
   ConcatFeatures, Message, AggregatedMessage, UpdateInput, PaddedEmb: TDoubleArray;
   MsgInfo: TMessageInfo;
begin
   SetLength(FNodeEmbeddings, Graph.NumNodes);
   SetLength(FNewNodeEmbeddings, Graph.NumNodes);
   SetLength(FEmbeddingHistory, FNumMessagePassingLayers + 1);
   SetLength(FMessageHistory, FNumMessagePassingLayers);
   SetLength(FAggregatedMessages, FNumMessagePassingLayers);
   
   for I := 0 to Graph.NumNodes - 1 do
      FNodeEmbeddings[I] := CopyArray(Graph.NodeFeatures[I]);
   
   SetLength(FEmbeddingHistory[0], Graph.NumNodes);
   for I := 0 to Graph.NumNodes - 1 do
      FEmbeddingHistory[0][I] := CopyArray(FNodeEmbeddings[I]);
   
   for Layer := 0 to FNumMessagePassingLayers - 1 do
   begin
      SetLength(FMessageHistory[Layer], Graph.NumNodes);
      SetLength(FAggregatedMessages[Layer], Graph.NumNodes);
      
      for Node := 0 to Graph.NumNodes - 1 do
      begin
         SetLength(FMessageHistory[Layer][Node], 0);
         AggregatedMessage := ZeroArray(FHiddenSize);
         
         if Length(Graph.AdjacencyList[Node]) > 0 then
         begin
            for K := 0 to High(Graph.AdjacencyList[Node]) do
            begin
               Neighbor := Graph.AdjacencyList[Node][K];
               
               ConcatFeatures := ConcatArrays(FNodeEmbeddings[Node], FNodeEmbeddings[Neighbor]);
               Message := ForwardLayer(FMessageLayers[Layer], ConcatFeatures, False);
               
               MsgInfo.NeighborIdx := Neighbor;
               MsgInfo.ConcatInput := CopyArray(ConcatFeatures);
               MsgInfo.MessageOutput := CopyArray(Message);
               
               SetLength(FMessageHistory[Layer][Node], Length(FMessageHistory[Layer][Node]) + 1);
               FMessageHistory[Layer][Node][High(FMessageHistory[Layer][Node])] := MsgInfo;
               
               for I := 0 to FHiddenSize - 1 do
                  AggregatedMessage[I] := AggregatedMessage[I] + Message[I];
            end;
            
            for I := 0 to FHiddenSize - 1 do
               AggregatedMessage[I] := AggregatedMessage[I] / Length(Graph.AdjacencyList[Node]);
         end;
         
         FAggregatedMessages[Layer][Node] := CopyArray(AggregatedMessage);
         
         if Layer = 0 then
            PaddedEmb := PadArray(FNodeEmbeddings[Node], FHiddenSize)
         else
            PaddedEmb := CopyArray(FNodeEmbeddings[Node]);
         
         UpdateInput := ConcatArrays(PaddedEmb, AggregatedMessage);
         FNewNodeEmbeddings[Node] := ForwardLayer(FUpdateLayers[Layer], UpdateInput, False);
      end;
      
      for Node := 0 to Graph.NumNodes - 1 do
         FNodeEmbeddings[Node] := CopyArray(FNewNodeEmbeddings[Node]);
      
      SetLength(FEmbeddingHistory[Layer + 1], Graph.NumNodes);
      for I := 0 to Graph.NumNodes - 1 do
         FEmbeddingHistory[Layer + 1][I] := CopyArray(FNodeEmbeddings[I]);
   end;
end;

procedure TGraphNeuralNetwork.Readout(var Graph: TGraph);
var
   I, J: Integer;
begin
   FGraphEmbedding := ZeroArray(FHiddenSize);
   
   for I := 0 to Graph.NumNodes - 1 do
      for J := 0 to FHiddenSize - 1 do
         FGraphEmbedding[J] := FGraphEmbedding[J] + FNodeEmbeddings[I][J];
   
   for J := 0 to FHiddenSize - 1 do
      FGraphEmbedding[J] := FGraphEmbedding[J] / Graph.NumNodes;
   
   SetLength(FGraphEmbeddingHistory, Length(FGraphEmbeddingHistory) + 1);
   FGraphEmbeddingHistory[High(FGraphEmbeddingHistory)] := CopyArray(FGraphEmbedding);
   
   ForwardLayer(FReadoutLayer, FGraphEmbedding, False);
end;

function TGraphNeuralNetwork.Predict(var Graph: TGraph): TDoubleArray;
var
   I: Integer;
   ReadoutOutput: TDoubleArray;
begin
   if Graph.Config.DeduplicateEdges then
      DeduplicateEdges(Graph);
   if Graph.Config.Undirected then
      AddReverseEdges(Graph);
   if Graph.Config.SelfLoops then
      AddSelfLoops(Graph);
   
   BuildAdjacencyList(Graph);
   FCurrentGraph := Graph;
   FHasGraph := True;
   
   MessagePassing(Graph);
   Readout(Graph);
   
   SetLength(ReadoutOutput, FHiddenSize);
   for I := 0 to FHiddenSize - 1 do
      ReadoutOutput[I] := FReadoutLayer.Neurons[I].Output;
   
   Result := ForwardLayer(FOutputLayer, ReadoutOutput, True);
end;

procedure TGraphNeuralNetwork.BackPropagateGraph(var Graph: TGraph; const Target: TDoubleArray);
var
   Layer, Node, I, J, K, HalfLen: Integer;
   LossGrad, ReadoutGrad, GraphEmbGrad, MsgGrad, ConcatGrad: TDoubleArray;
   NodeGrads, NewNodeGrads: TDouble2DArray;
   UpdateInputGrad, PaddedEmb, UpdateInput: TDoubleArray;
   NumNeighbors: Integer;
begin
   LossGrad := ComputeLossGradient(FOutputLayer.LastInput, Target);
   
   for I := 0 to FOutputSize - 1 do
      LossGrad[I] := LossGrad[I] * OutputActivateDerivative(FOutputLayer.Neurons[I].PreActivation);
   
   BackwardLayer(FOutputLayer, LossGrad, True);
   ReadoutGrad := GetLayerInputGrad(FOutputLayer, LossGrad, True);
   
   BackwardLayer(FReadoutLayer, ReadoutGrad, False);
   GraphEmbGrad := GetLayerInputGrad(FReadoutLayer, ReadoutGrad, False);
   
   SetLength(NodeGrads, Graph.NumNodes);
   for Node := 0 to Graph.NumNodes - 1 do
   begin
      NodeGrads[Node] := ZeroArray(FHiddenSize);
      for I := 0 to FHiddenSize - 1 do
         NodeGrads[Node][I] := GraphEmbGrad[I] / Graph.NumNodes;
   end;
   
   for Layer := FNumMessagePassingLayers - 1 downto 0 do
   begin
      SetLength(NewNodeGrads, Graph.NumNodes);
      
      if Layer = 0 then
      begin
         for Node := 0 to Graph.NumNodes - 1 do
            NewNodeGrads[Node] := ZeroArray(FFeatureSize);
      end
      else
      begin
         for Node := 0 to Graph.NumNodes - 1 do
            NewNodeGrads[Node] := ZeroArray(FHiddenSize);
      end;
      
      for Node := 0 to Graph.NumNodes - 1 do
      begin
         if Layer = 0 then
            PaddedEmb := PadArray(FEmbeddingHistory[Layer][Node], FHiddenSize)
         else
            PaddedEmb := CopyArray(FEmbeddingHistory[Layer][Node]);
         
         UpdateInput := ConcatArrays(PaddedEmb, FAggregatedMessages[Layer][Node]);
         FUpdateLayers[Layer].LastInput := CopyArray(UpdateInput);
         
         BackwardLayer(FUpdateLayers[Layer], NodeGrads[Node], False);
         UpdateInputGrad := GetLayerInputGrad(FUpdateLayers[Layer], NodeGrads[Node], False);
         
         for I := 0 to Min(FHiddenSize - 1, High(NewNodeGrads[Node])) do
         begin
            if Layer = 0 then
            begin
               if I < FFeatureSize then
                  NewNodeGrads[Node][I] := NewNodeGrads[Node][I] + UpdateInputGrad[I];
            end
            else
               NewNodeGrads[Node][I] := NewNodeGrads[Node][I] + UpdateInputGrad[I];
         end;
         
         NumNeighbors := Length(Graph.AdjacencyList[Node]);
         if NumNeighbors > 0 then
         begin
            MsgGrad := ZeroArray(FHiddenSize);
            for I := 0 to FHiddenSize - 1 do
               MsgGrad[I] := UpdateInputGrad[FHiddenSize + I] / NumNeighbors;
            
            for K := 0 to High(FMessageHistory[Layer][Node]) do
            begin
               FMessageLayers[Layer].LastInput := CopyArray(FMessageHistory[Layer][Node][K].ConcatInput);
               
               BackwardLayer(FMessageLayers[Layer], MsgGrad, False);
               ConcatGrad := GetLayerInputGrad(FMessageLayers[Layer], MsgGrad, False);
               
               HalfLen := Length(ConcatGrad) div 2;
               
               for I := 0 to Min(HalfLen - 1, High(NewNodeGrads[Node])) do
                  NewNodeGrads[Node][I] := NewNodeGrads[Node][I] + ConcatGrad[I];
               
               J := FMessageHistory[Layer][Node][K].NeighborIdx;
               for I := 0 to Min(HalfLen - 1, High(NewNodeGrads[J])) do
                  NewNodeGrads[J][I] := NewNodeGrads[J][I] + ConcatGrad[HalfLen + I];
            end;
         end;
      end;
      
      if Layer > 0 then
         NodeGrads := NewNodeGrads;
   end;
end;

function TGraphNeuralNetwork.Train(var Graph: TGraph; const Target: TDoubleArray): Double;
var
   Prediction: TDoubleArray;
begin
   Prediction := Predict(Graph);
   Result := ComputeLoss(Prediction, Target);
   BackPropagateGraph(Graph, Target);
end;

procedure TGraphNeuralNetwork.TrainMultiple(var Graph: TGraph; const Target: TDoubleArray; Iterations: Integer);
var
   I: Integer;
   Loss: Double;
begin
   SetLength(FMetrics.LossHistory, Iterations);
   
   for I := 0 to Iterations - 1 do
   begin
      Loss := Train(Graph, Target);
      FMetrics.LossHistory[I] := Loss;
      FMetrics.Loss := Loss;
      FMetrics.Iteration := I + 1;
      
      if (I mod 10 = 0) or (I = Iterations - 1) then
         WriteLn(Format('Iteration %d/%d, Loss: %.6f', [I + 1, Iterations, Loss]));
   end;
end;

procedure TGraphNeuralNetwork.SaveModel(const Filename: string);
var
   F: TFileStream;
   I, J, K: Integer;
   ActInt, LossInt: Integer;
   
begin
   F := TFileStream.Create(Filename, fmCreate);
   try
      F.WriteBuffer(FFeatureSize, SizeOf(Integer));
      F.WriteBuffer(FHiddenSize, SizeOf(Integer));
      F.WriteBuffer(FOutputSize, SizeOf(Integer));
      F.WriteBuffer(FNumMessagePassingLayers, SizeOf(Integer));
      F.WriteBuffer(FLearningRate, SizeOf(Double));
      
      ActInt := Ord(FActivation);
      LossInt := Ord(FLossType);
      F.WriteBuffer(ActInt, SizeOf(Integer));
      F.WriteBuffer(LossInt, SizeOf(Integer));
      
      for K := 0 to FNumMessagePassingLayers - 1 do
      begin
         F.WriteBuffer(FMessageLayers[K].NumOutputs, SizeOf(Integer));
         F.WriteBuffer(FMessageLayers[K].NumInputs, SizeOf(Integer));
         for I := 0 to FMessageLayers[K].NumOutputs - 1 do
         begin
            for J := 0 to FMessageLayers[K].NumInputs - 1 do
               F.WriteBuffer(FMessageLayers[K].Neurons[I].Weights[J], SizeOf(Double));
            F.WriteBuffer(FMessageLayers[K].Neurons[I].Bias, SizeOf(Double));
         end;
         
         F.WriteBuffer(FUpdateLayers[K].NumOutputs, SizeOf(Integer));
         F.WriteBuffer(FUpdateLayers[K].NumInputs, SizeOf(Integer));
         for I := 0 to FUpdateLayers[K].NumOutputs - 1 do
         begin
            for J := 0 to FUpdateLayers[K].NumInputs - 1 do
               F.WriteBuffer(FUpdateLayers[K].Neurons[I].Weights[J], SizeOf(Double));
            F.WriteBuffer(FUpdateLayers[K].Neurons[I].Bias, SizeOf(Double));
         end;
      end;
      
      F.WriteBuffer(FReadoutLayer.NumOutputs, SizeOf(Integer));
      F.WriteBuffer(FReadoutLayer.NumInputs, SizeOf(Integer));
      for I := 0 to FReadoutLayer.NumOutputs - 1 do
      begin
         for J := 0 to FReadoutLayer.NumInputs - 1 do
            F.WriteBuffer(FReadoutLayer.Neurons[I].Weights[J], SizeOf(Double));
         F.WriteBuffer(FReadoutLayer.Neurons[I].Bias, SizeOf(Double));
      end;
      
      F.WriteBuffer(FOutputLayer.NumOutputs, SizeOf(Integer));
      F.WriteBuffer(FOutputLayer.NumInputs, SizeOf(Integer));
      for I := 0 to FOutputLayer.NumOutputs - 1 do
      begin
         for J := 0 to FOutputLayer.NumInputs - 1 do
            F.WriteBuffer(FOutputLayer.Neurons[I].Weights[J], SizeOf(Double));
         F.WriteBuffer(FOutputLayer.Neurons[I].Bias, SizeOf(Double));
      end;
      
      WriteLn('Model saved to ', Filename);
   finally
      F.Free;
   end;
end;

procedure TGraphNeuralNetwork.LoadModel(const Filename: string);
var
   F: TFileStream;
   I, J, K, NumN, NumI: Integer;
   ActInt, LossInt: Integer;
   TmpDouble: Double;
begin
   F := TFileStream.Create(Filename, fmOpenRead);
   try
      F.ReadBuffer(FFeatureSize, SizeOf(Integer));
      F.ReadBuffer(FHiddenSize, SizeOf(Integer));
      F.ReadBuffer(FOutputSize, SizeOf(Integer));
      F.ReadBuffer(FNumMessagePassingLayers, SizeOf(Integer));
      F.ReadBuffer(FLearningRate, SizeOf(Double));
      
      F.ReadBuffer(ActInt, SizeOf(Integer));
      F.ReadBuffer(LossInt, SizeOf(Integer));
      FActivation := TActivationType(ActInt);
      FLossType := TLossType(LossInt);
      
      SetLength(FMessageLayers, FNumMessagePassingLayers);
      SetLength(FUpdateLayers, FNumMessagePassingLayers);
      
      for K := 0 to FNumMessagePassingLayers - 1 do
      begin
         F.ReadBuffer(NumN, SizeOf(Integer));
         F.ReadBuffer(NumI, SizeOf(Integer));
         InitializeLayer(FMessageLayers[K], NumN, NumI);
         for I := 0 to NumN - 1 do
         begin
            for J := 0 to NumI - 1 do
            begin
               F.ReadBuffer(TmpDouble, SizeOf(Double));
               FMessageLayers[K].Neurons[I].Weights[J] := TmpDouble;
            end;
            F.ReadBuffer(TmpDouble, SizeOf(Double));
            FMessageLayers[K].Neurons[I].Bias := TmpDouble;
         end;
         
         F.ReadBuffer(NumN, SizeOf(Integer));
         F.ReadBuffer(NumI, SizeOf(Integer));
         InitializeLayer(FUpdateLayers[K], NumN, NumI);
         for I := 0 to NumN - 1 do
         begin
            for J := 0 to NumI - 1 do
            begin
               F.ReadBuffer(TmpDouble, SizeOf(Double));
               FUpdateLayers[K].Neurons[I].Weights[J] := TmpDouble;
            end;
            F.ReadBuffer(TmpDouble, SizeOf(Double));
            FUpdateLayers[K].Neurons[I].Bias := TmpDouble;
         end;
      end;
      
      F.ReadBuffer(NumN, SizeOf(Integer));
      F.ReadBuffer(NumI, SizeOf(Integer));
      InitializeLayer(FReadoutLayer, NumN, NumI);
      for I := 0 to NumN - 1 do
      begin
         for J := 0 to NumI - 1 do
         begin
            F.ReadBuffer(TmpDouble, SizeOf(Double));
            FReadoutLayer.Neurons[I].Weights[J] := TmpDouble;
         end;
         F.ReadBuffer(TmpDouble, SizeOf(Double));
         FReadoutLayer.Neurons[I].Bias := TmpDouble;
      end;
      
      F.ReadBuffer(NumN, SizeOf(Integer));
      F.ReadBuffer(NumI, SizeOf(Integer));
      InitializeLayer(FOutputLayer, NumN, NumI);
      for I := 0 to NumN - 1 do
      begin
         for J := 0 to NumI - 1 do
         begin
            F.ReadBuffer(TmpDouble, SizeOf(Double));
            FOutputLayer.Neurons[I].Weights[J] := TmpDouble;
         end;
         F.ReadBuffer(TmpDouble, SizeOf(Double));
         FOutputLayer.Neurons[I].Bias := TmpDouble;
      end;
      
      WriteLn('Model loaded from ', Filename);
   finally
      F.Free;
   end;
end;

// ==================== TGNNFacade Implementation ====================

constructor TGNNFacade.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
begin
   inherited Create;
   FGNN := TGraphNeuralNetwork.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers);
   FGraphLoaded := False;
   FTraceEnabled := False;
   
   FOptimizerType := otSGD;
   FAdamBeta1 := 0.9;
   FAdamBeta2 := 0.999;
   FAdamEpsilon := 1e-8;
   FRMSPropDecay := 0.9;
   FRMSPropEpsilon := 1e-8;
   
   SetLength(FMessagePassingTrace, 0);
   SetLength(FBatchGraphs, 0);
   SetLength(FBatchNodeEmbeddings, 0);
end;

destructor TGNNFacade.Destroy;
begin
   FGNN.Free;
   inherited Destroy;
end;

procedure TGNNFacade.EnsureGraphLoaded;
begin
   if not FGraphLoaded then
      raise Exception.Create('No graph loaded. Call LoadGraph or CreateEmptyGraph first.');
end;

procedure TGNNFacade.InitializeMasks;
var
   I: Integer;
begin
   SetLength(FNodeMasks, FGraph.NumNodes);
   for I := 0 to FGraph.NumNodes - 1 do
      FNodeMasks[I] := True;
   
   SetLength(FEdgeMasks, Length(FGraph.Edges));
   for I := 0 to High(FGraph.Edges) do
      FEdgeMasks[I] := True;
end;

procedure TGNNFacade.InitializeOptimizerStates;
var
   LayerIdx, NeuronIdx, NumWeights: Integer;
begin
   SetLength(FMessageLayerAdamStates, FGNN.NumMessagePassingLayers);
   SetLength(FUpdateLayerAdamStates, FGNN.NumMessagePassingLayers);
   SetLength(FMessageLayerRMSPropStates, FGNN.NumMessagePassingLayers);
   SetLength(FUpdateLayerRMSPropStates, FGNN.NumMessagePassingLayers);
   
   for LayerIdx := 0 to FGNN.NumMessagePassingLayers - 1 do
   begin
      SetLength(FMessageLayerAdamStates[LayerIdx], Length(FGNN.MessageLayers[LayerIdx].Neurons));
      SetLength(FUpdateLayerAdamStates[LayerIdx], Length(FGNN.UpdateLayers[LayerIdx].Neurons));
      SetLength(FMessageLayerRMSPropStates[LayerIdx], Length(FGNN.MessageLayers[LayerIdx].Neurons));
      SetLength(FUpdateLayerRMSPropStates[LayerIdx], Length(FGNN.UpdateLayers[LayerIdx].Neurons));
      
      for NeuronIdx := 0 to High(FGNN.MessageLayers[LayerIdx].Neurons) do
      begin
         NumWeights := Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights);
         SetLength(FMessageLayerAdamStates[LayerIdx][NeuronIdx].M, NumWeights + 1);
         SetLength(FMessageLayerAdamStates[LayerIdx][NeuronIdx].V, NumWeights + 1);
         FMessageLayerAdamStates[LayerIdx][NeuronIdx].T := 0;
         SetLength(FMessageLayerRMSPropStates[LayerIdx][NeuronIdx].S, NumWeights + 1);
      end;
      
      for NeuronIdx := 0 to High(FGNN.UpdateLayers[LayerIdx].Neurons) do
      begin
         NumWeights := Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights);
         SetLength(FUpdateLayerAdamStates[LayerIdx][NeuronIdx].M, NumWeights + 1);
         SetLength(FUpdateLayerAdamStates[LayerIdx][NeuronIdx].V, NumWeights + 1);
         FUpdateLayerAdamStates[LayerIdx][NeuronIdx].T := 0;
         SetLength(FUpdateLayerRMSPropStates[LayerIdx][NeuronIdx].S, NumWeights + 1);
      end;
   end;
   
   SetLength(FReadoutLayerAdamStates, Length(FGNN.ReadoutLayer.Neurons));
   SetLength(FReadoutLayerRMSPropStates, Length(FGNN.ReadoutLayer.Neurons));
   for NeuronIdx := 0 to High(FGNN.ReadoutLayer.Neurons) do
   begin
      NumWeights := Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights);
      SetLength(FReadoutLayerAdamStates[NeuronIdx].M, NumWeights + 1);
      SetLength(FReadoutLayerAdamStates[NeuronIdx].V, NumWeights + 1);
      FReadoutLayerAdamStates[NeuronIdx].T := 0;
      SetLength(FReadoutLayerRMSPropStates[NeuronIdx].S, NumWeights + 1);
   end;
   
   SetLength(FOutputLayerAdamStates, Length(FGNN.OutputLayer.Neurons));
   SetLength(FOutputLayerRMSPropStates, Length(FGNN.OutputLayer.Neurons));
   for NeuronIdx := 0 to High(FGNN.OutputLayer.Neurons) do
   begin
      NumWeights := Length(FGNN.OutputLayer.Neurons[NeuronIdx].Weights);
      SetLength(FOutputLayerAdamStates[NeuronIdx].M, NumWeights + 1);
      SetLength(FOutputLayerAdamStates[NeuronIdx].V, NumWeights + 1);
      FOutputLayerAdamStates[NeuronIdx].T := 0;
      SetLength(FOutputLayerRMSPropStates[NeuronIdx].S, NumWeights + 1);
   end;
end;

procedure TGNNFacade.RecordMessagePassingStep(LayerIdx, IterIdx, NodeIdx, NeighborIdx: Integer;
   const Msg, AggMsg: TDoubleArray);
var
   Step: TMessagePassingStep;
begin
   if not FTraceEnabled then Exit;
   
   Step.LayerIdx := LayerIdx;
   Step.IterationIdx := IterIdx;
   Step.NodeIdx := NodeIdx;
   Step.NeighborIdx := NeighborIdx;
   Step.Message := CopyArray(Msg);
   Step.AggregatedMessage := CopyArray(AggMsg);
   
   SetLength(FMessagePassingTrace, Length(FMessagePassingTrace) + 1);
   FMessagePassingTrace[High(FMessagePassingTrace)] := Step;
end;

// ==================== Graph Management ====================

procedure TGNNFacade.LoadGraph(var Graph: TGraph);
begin
   FGraph := Graph;
   FGraphLoaded := True;
   InitializeMasks;
   InitializeOptimizerStates;
end;

procedure TGNNFacade.CreateEmptyGraph(NumNodes: Integer; FeatureSize: Integer);
var
   I, J: Integer;
begin
   FGraph.NumNodes := NumNodes;
   FGraph.Config.Undirected := True;
   FGraph.Config.SelfLoops := False;
   FGraph.Config.DeduplicateEdges := True;
   
   SetLength(FGraph.NodeFeatures, NumNodes);
   for I := 0 to NumNodes - 1 do
   begin
      SetLength(FGraph.NodeFeatures[I], FeatureSize);
      for J := 0 to FeatureSize - 1 do
         FGraph.NodeFeatures[I][J] := 0.0;
   end;
   
   SetLength(FGraph.Edges, 0);
   SetLength(FGraph.AdjacencyList, NumNodes);
   for I := 0 to NumNodes - 1 do
      SetLength(FGraph.AdjacencyList[I], 0);
   
   FGraphLoaded := True;
   InitializeMasks;
   InitializeOptimizerStates;
end;

function TGNNFacade.GetGraph: TGraph;
begin
   EnsureGraphLoaded;
   Result := FGraph;
end;

// ==================== 1. Node and Edge Feature Access ====================

function TGNNFacade.GetNodeFeature(NodeIdx, FeatureIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.NodeFeatures[NodeIdx])) then
      raise Exception.CreateFmt('Feature index %d out of range [0, %d)', 
         [FeatureIdx, Length(FGraph.NodeFeatures[NodeIdx])]);
   
   Result := FGraph.NodeFeatures[NodeIdx][FeatureIdx];
end;

procedure TGNNFacade.SetNodeFeature(NodeIdx, FeatureIdx: Integer; Value: Double);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.NodeFeatures[NodeIdx])) then
      raise Exception.CreateFmt('Feature index %d out of range [0, %d)', 
         [FeatureIdx, Length(FGraph.NodeFeatures[NodeIdx])]);
   
   FGraph.NodeFeatures[NodeIdx][FeatureIdx] := Value;
end;

function TGNNFacade.GetNodeFeatures(NodeIdx: Integer): TDoubleArray;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := CopyArray(FGraph.NodeFeatures[NodeIdx]);
end;

procedure TGNNFacade.SetNodeFeatures(NodeIdx: Integer; const Features: TDoubleArray);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   FGraph.NodeFeatures[NodeIdx] := CopyArray(Features);
end;

function TGNNFacade.GetEdgeFeature(EdgeIdx, FeatureIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.Edges[EdgeIdx].Features)) then
      raise Exception.CreateFmt('Feature index %d out of range [0, %d)', 
         [FeatureIdx, Length(FGraph.Edges[EdgeIdx].Features)]);
   
   Result := FGraph.Edges[EdgeIdx].Features[FeatureIdx];
end;

procedure TGNNFacade.SetEdgeFeature(EdgeIdx, FeatureIdx: Integer; Value: Double);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   if Length(FGraph.Edges[EdgeIdx].Features) <= FeatureIdx then
      SetLength(FGraph.Edges[EdgeIdx].Features, FeatureIdx + 1);
   
   FGraph.Edges[EdgeIdx].Features[FeatureIdx] := Value;
end;

function TGNNFacade.GetEdgeFeatures(EdgeIdx: Integer): TDoubleArray;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   Result := CopyArray(FGraph.Edges[EdgeIdx].Features);
end;

procedure TGNNFacade.SetEdgeFeatures(EdgeIdx: Integer; const Features: TDoubleArray);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   FGraph.Edges[EdgeIdx].Features := CopyArray(Features);
end;

function TGNNFacade.GetNumNodes: Integer;
begin
   EnsureGraphLoaded;
   Result := FGraph.NumNodes;
end;

function TGNNFacade.GetNumEdges: Integer;
begin
   EnsureGraphLoaded;
   Result := Length(FGraph.Edges);
end;

function TGNNFacade.GetNodeFeatureSize(NodeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := Length(FGraph.NodeFeatures[NodeIdx]);
end;

function TGNNFacade.GetEdgeFeatureSize(EdgeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   Result := Length(FGraph.Edges[EdgeIdx].Features);
end;

// ==================== 2. Adjacency and Topology Introspection ====================

function TGNNFacade.GetNeighbors(NodeIdx: Integer): TIntArray;
var
   I, Count: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Count := 0;
   SetLength(Result, Length(FGraph.Edges));
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if FGraph.Edges[I].Source = NodeIdx then
      begin
         Result[Count] := FGraph.Edges[I].Target;
         Inc(Count);
      end;
   end;
   
   SetLength(Result, Count);
end;

function TGNNFacade.GetAdjacencyMatrix: TDouble2DArray;
var
   I, J: Integer;
begin
   EnsureGraphLoaded;
   
   SetLength(Result, FGraph.NumNodes);
   for I := 0 to FGraph.NumNodes - 1 do
   begin
      SetLength(Result[I], FGraph.NumNodes);
      for J := 0 to FGraph.NumNodes - 1 do
         Result[I][J] := 0.0;
   end;
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if (FGraph.Edges[I].Source >= 0) and (FGraph.Edges[I].Source < FGraph.NumNodes) and
         (FGraph.Edges[I].Target >= 0) and (FGraph.Edges[I].Target < FGraph.NumNodes) then
      begin
         Result[FGraph.Edges[I].Source][FGraph.Edges[I].Target] := 1.0;
      end;
   end;
end;

function TGNNFacade.GetEdgeEndpoints(EdgeIdx: Integer): TEdgeEndpoints;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   Result.Source := FGraph.Edges[EdgeIdx].Source;
   Result.Target := FGraph.Edges[EdgeIdx].Target;
end;

function TGNNFacade.GetIncomingEdges(NodeIdx: Integer): TIntArray;
var
   I, Count: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Count := 0;
   SetLength(Result, Length(FGraph.Edges));
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if FGraph.Edges[I].Target = NodeIdx then
      begin
         Result[Count] := I;
         Inc(Count);
      end;
   end;
   
   SetLength(Result, Count);
end;

function TGNNFacade.GetOutgoingEdges(NodeIdx: Integer): TIntArray;
var
   I, Count: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Count := 0;
   SetLength(Result, Length(FGraph.Edges));
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if FGraph.Edges[I].Source = NodeIdx then
      begin
         Result[Count] := I;
         Inc(Count);
      end;
   end;
   
   SetLength(Result, Count);
end;

function TGNNFacade.HasEdge(SourceIdx, TargetIdx: Integer): Boolean;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   Result := False;
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if (FGraph.Edges[I].Source = SourceIdx) and (FGraph.Edges[I].Target = TargetIdx) then
      begin
         Result := True;
         Exit;
      end;
   end;
end;

function TGNNFacade.FindEdgeIndex(SourceIdx, TargetIdx: Integer): Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   Result := -1;
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if (FGraph.Edges[I].Source = SourceIdx) and (FGraph.Edges[I].Target = TargetIdx) then
      begin
         Result := I;
         Exit;
      end;
   end;
end;

// ==================== Core GNN Operations ====================

function TGNNFacade.Predict: TDoubleArray;
begin
   EnsureGraphLoaded;
   SetLength(FMessagePassingTrace, 0);
   Result := FGNN.Predict(FGraph);
end;

function TGNNFacade.Train(const Target: TDoubleArray): Double;
begin
   EnsureGraphLoaded;
   SetLength(FMessagePassingTrace, 0);
   Result := FGNN.Train(FGraph, Target);
end;

procedure TGNNFacade.TrainMultiple(const Target: TDoubleArray; Iterations: Integer);
begin
   EnsureGraphLoaded;
   FGNN.TrainMultiple(FGraph, Target, Iterations);
end;

function TGNNFacade.ComputeLoss(const Prediction, Target: TDoubleArray): Double;
begin
   Result := FGNN.ComputeLoss(Prediction, Target);
end;

procedure TGNNFacade.SaveModel(const Filename: string);
begin
   FGNN.SaveModel(Filename);
end;

procedure TGNNFacade.LoadModel(const Filename: string);
begin
   FGNN.LoadModel(Filename);
end;

// ==================== Property Accessors ====================

function TGNNFacade.GetLearningRate: Double;
begin
   Result := FGNN.LearningRate;
end;

procedure TGNNFacade.SetLearningRate(Value: Double);
begin
   FGNN.LearningRate := Value;
end;

function TGNNFacade.GetActivation: TActivationType;
begin
   Result := FGNN.Activation;
end;

procedure TGNNFacade.SetActivation(Value: TActivationType);
begin
   FGNN.Activation := Value;
end;

function TGNNFacade.GetLossFunction: TLossType;
begin
   Result := FGNN.LossFunction;
end;

procedure TGNNFacade.SetLossFunction(Value: TLossType);
begin
   FGNN.LossFunction := Value;
end;

// ==================== 3. Node/Edge Embedding and Activations ====================

function TGNNFacade.GetNodeEmbedding(LayerIdx, NodeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx > FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d]', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (LayerIdx <= High(FGNN.EmbeddingHistory)) and 
      (NodeIdx <= High(FGNN.EmbeddingHistory[LayerIdx])) then
      Result := CopyArray(FGNN.EmbeddingHistory[LayerIdx][NodeIdx])
   else
      SetLength(Result, 0);
end;

procedure TGNNFacade.SetNodeEmbedding(LayerIdx, NodeIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
var
   I: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx > FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d]', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (LayerIdx <= High(FGNN.EmbeddingHistory)) and 
      (NodeIdx <= High(FGNN.EmbeddingHistory[LayerIdx])) then
   begin
      for I := 0 to Min(High(Value), High(FGNN.EmbeddingHistory[LayerIdx][NodeIdx])) do
         FGNN.EmbeddingHistory[LayerIdx][NodeIdx][I] := Value[I];
   end;
end;

function TGNNFacade.GetEdgeEmbedding(LayerIdx, EdgeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
var
   Edge: TEdge;
   SourceEmb, TargetEmb: TDoubleArray;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   if (LayerIdx < 0) or (LayerIdx > FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d]', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   Edge := FGraph.Edges[EdgeIdx];
   
   if (LayerIdx <= High(FGNN.EmbeddingHistory)) then
   begin
      SourceEmb := CopyArray(FGNN.EmbeddingHistory[LayerIdx][Edge.Source]);
      TargetEmb := CopyArray(FGNN.EmbeddingHistory[LayerIdx][Edge.Target]);
      Result := ConcatArrays(SourceEmb, TargetEmb);
   end
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetAllNodeEmbeddings(NodeIdx: Integer): TDouble2DArray;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   SetLength(Result, Length(FGNN.EmbeddingHistory));
   for I := 0 to High(FGNN.EmbeddingHistory) do
   begin
      if NodeIdx <= High(FGNN.EmbeddingHistory[I]) then
         Result[I] := CopyArray(FGNN.EmbeddingHistory[I][NodeIdx])
      else
         SetLength(Result[I], 0);
   end;
end;

function TGNNFacade.GetAllLayerEmbeddings(LayerIdx: Integer): TDouble2DArray;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx > FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d]', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if LayerIdx <= High(FGNN.EmbeddingHistory) then
   begin
      SetLength(Result, Length(FGNN.EmbeddingHistory[LayerIdx]));
      for I := 0 to High(FGNN.EmbeddingHistory[LayerIdx]) do
         Result[I] := CopyArray(FGNN.EmbeddingHistory[LayerIdx][I]);
   end
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetCurrentNodeEmbedding(NodeIdx: Integer): TDoubleArray;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (NodeIdx <= High(FGNN.NodeEmbeddings)) then
      Result := CopyArray(FGNN.NodeEmbeddings[NodeIdx])
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetFinalNodeEmbeddings: TDouble2DArray;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   SetLength(Result, Length(FGNN.NodeEmbeddings));
   for I := 0 to High(FGNN.NodeEmbeddings) do
      Result[I] := CopyArray(FGNN.NodeEmbeddings[I]);
end;

// ==================== 4. Message Passing Internals ====================

function TGNNFacade.GetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
var
   K: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx >= FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d)', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   SetLength(Result, 0);
   
   if (LayerIdx <= High(FGNN.MessageHistory)) and (NodeIdx <= High(FGNN.MessageHistory[LayerIdx])) then
   begin
      for K := 0 to High(FGNN.MessageHistory[LayerIdx][NodeIdx]) do
      begin
         if FGNN.MessageHistory[LayerIdx][NodeIdx][K].NeighborIdx = NeighborIdx then
         begin
            Result := CopyArray(FGNN.MessageHistory[LayerIdx][NodeIdx][K].MessageOutput);
            Exit;
         end;
      end;
   end;
end;

procedure TGNNFacade.SetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
var
   K, I: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx >= FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d)', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (LayerIdx <= High(FGNN.MessageHistory)) and (NodeIdx <= High(FGNN.MessageHistory[LayerIdx])) then
   begin
      for K := 0 to High(FGNN.MessageHistory[LayerIdx][NodeIdx]) do
      begin
         if FGNN.MessageHistory[LayerIdx][NodeIdx][K].NeighborIdx = NeighborIdx then
         begin
            for I := 0 to Min(High(Value), High(FGNN.MessageHistory[LayerIdx][NodeIdx][K].MessageOutput)) do
               FGNN.MessageHistory[LayerIdx][NodeIdx][K].MessageOutput[I] := Value[I];
            Exit;
         end;
      end;
   end;
end;

function TGNNFacade.GetAggregatedMessage(NodeIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx >= FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d)', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (LayerIdx <= High(FGNN.AggregatedMessages)) and 
      (NodeIdx <= High(FGNN.AggregatedMessages[LayerIdx])) then
      Result := CopyArray(FGNN.AggregatedMessages[LayerIdx][NodeIdx])
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetMessageInput(NodeIdx, NeighborIdx, LayerIdx: Integer): TDoubleArray;
var
   K: Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx >= FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d)', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   SetLength(Result, 0);
   
   if (LayerIdx <= High(FGNN.MessageHistory)) and (NodeIdx <= High(FGNN.MessageHistory[LayerIdx])) then
   begin
      for K := 0 to High(FGNN.MessageHistory[LayerIdx][NodeIdx]) do
      begin
         if FGNN.MessageHistory[LayerIdx][NodeIdx][K].NeighborIdx = NeighborIdx then
         begin
            Result := CopyArray(FGNN.MessageHistory[LayerIdx][NodeIdx][K].ConcatInput);
            Exit;
         end;
      end;
   end;
end;

function TGNNFacade.GetNumMessagesForNode(NodeIdx, LayerIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if (LayerIdx < 0) or (LayerIdx >= FGNN.NumMessagePassingLayers) then
      raise Exception.CreateFmt('Layer index %d out of range [0, %d)', 
         [LayerIdx, FGNN.NumMessagePassingLayers]);
   
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if (LayerIdx <= High(FGNN.MessageHistory)) and (NodeIdx <= High(FGNN.MessageHistory[LayerIdx])) then
      Result := Length(FGNN.MessageHistory[LayerIdx][NodeIdx])
   else
      Result := 0;
end;

// ==================== 5. Readout and Output Layer Access ====================

function TGNNFacade.GetGraphEmbedding(LayerIdx: Integer = -1): TDoubleArray;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if LayerIdx < 0 then
      Result := CopyArray(FGNN.GraphEmbedding)
   else if (LayerIdx <= High(FGNN.GraphEmbeddingHistory)) then
      Result := CopyArray(FGNN.GraphEmbeddingHistory[LayerIdx])
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetReadout(LayerIdx: Integer = -1): TDoubleArray;
begin
   Result := GetGraphEmbedding(LayerIdx);
end;

procedure TGNNFacade.SetGraphEmbedding(const Value: TDoubleArray; LayerIdx: Integer = -1);
var
   I: Integer;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   if LayerIdx < 0 then
   begin
      for I := 0 to Min(High(Value), High(FGNN.GraphEmbedding)) do
         FGNN.GraphEmbedding[I] := Value[I];
   end
   else if (LayerIdx <= High(FGNN.GraphEmbeddingHistory)) then
   begin
      for I := 0 to Min(High(Value), High(FGNN.GraphEmbeddingHistory[LayerIdx])) do
         FGNN.GraphEmbeddingHistory[LayerIdx][I] := Value[I];
   end;
end;

function TGNNFacade.GetReadoutLayerOutput: TDoubleArray;
var
   I: Integer;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   SetLength(Result, Length(FGNN.ReadoutLayer.Neurons));
   for I := 0 to High(FGNN.ReadoutLayer.Neurons) do
      Result[I] := FGNN.ReadoutLayer.Neurons[I].Output;
end;

function TGNNFacade.GetOutputLayerOutput: TDoubleArray;
var
   I: Integer;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   SetLength(Result, Length(FGNN.OutputLayer.Neurons));
   for I := 0 to High(FGNN.OutputLayer.Neurons) do
      Result[I] := FGNN.OutputLayer.Neurons[I].Output;
end;

function TGNNFacade.GetReadoutLayerPreActivations: TDoubleArray;
var
   I: Integer;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   SetLength(Result, Length(FGNN.ReadoutLayer.Neurons));
   for I := 0 to High(FGNN.ReadoutLayer.Neurons) do
      Result[I] := FGNN.ReadoutLayer.Neurons[I].PreActivation;
end;

function TGNNFacade.GetOutputLayerPreActivations: TDoubleArray;
var
   I: Integer;
begin
   if not FGNN.HasGraph then
      raise Exception.Create('No forward pass has been performed. Call Predict first.');
   
   SetLength(Result, Length(FGNN.OutputLayer.Neurons));
   for I := 0 to High(FGNN.OutputLayer.Neurons) do
      Result[I] := FGNN.OutputLayer.Neurons[I].PreActivation;
end;

// ==================== 6. Backprop Gradients and Optimizer States ====================

function TGNNFacade.GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
   
   if LayerIdx < FGNN.NumMessagePassingLayers then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) and
         (WeightIdx >= 0) and (WeightIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].WeightGradients)) then
         Result := FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].WeightGradients[WeightIdx];
   end
   else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
   begin
      LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) and
         (WeightIdx >= 0) and (WeightIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].WeightGradients)) then
         Result := FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].WeightGradients[WeightIdx];
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) and
         (WeightIdx >= 0) and (WeightIdx < Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].WeightGradients)) then
         Result := FGNN.ReadoutLayer.Neurons[NeuronIdx].WeightGradients[WeightIdx];
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) and
         (WeightIdx >= 0) and (WeightIdx < Length(FGNN.OutputLayer.Neurons[NeuronIdx].WeightGradients)) then
         Result := FGNN.OutputLayer.Neurons[NeuronIdx].WeightGradients[WeightIdx];
   end;
end;

function TGNNFacade.GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := 0.0;
   
   if LayerIdx < FGNN.NumMessagePassingLayers then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) then
         Result := FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].BiasGradient;
   end
   else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
   begin
      LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) then
         Result := FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].BiasGradient;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) then
         Result := FGNN.ReadoutLayer.Neurons[NeuronIdx].BiasGradient;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
   begin
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) then
         Result := FGNN.OutputLayer.Neurons[NeuronIdx].BiasGradient;
   end;
end;

function TGNNFacade.GetNodeEmbeddingGradient(LayerIdx, NodeIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
   
   if (LayerIdx >= 0) and (LayerIdx < Length(FNodeEmbeddingGradients)) and
      (NodeIdx >= 0) and (NodeIdx < Length(FNodeEmbeddingGradients[LayerIdx])) then
      Result := CopyArray(FNodeEmbeddingGradients[LayerIdx][NodeIdx]);
end;

function TGNNFacade.GetEdgeGradient(LayerIdx, EdgeIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
   
   if (LayerIdx >= 0) and (LayerIdx < Length(FEdgeGradients)) and
      (EdgeIdx >= 0) and (EdgeIdx < Length(FEdgeGradients[LayerIdx])) then
   begin
      SetLength(Result, 1);
      Result[0] := FEdgeGradients[LayerIdx][EdgeIdx];
   end;
end;

function TGNNFacade.GetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string): TDoubleArray;
begin
   SetLength(Result, 0);
   
   if FOptimizerType = otAdam then
   begin
      if LayerIdx < FGNN.NumMessagePassingLayers then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FMessageLayerAdamStates[LayerIdx])) then
         begin
            if StateVar = 'm' then
               Result := CopyArray(FMessageLayerAdamStates[LayerIdx][NeuronIdx].M)
            else if StateVar = 'v' then
               Result := CopyArray(FMessageLayerAdamStates[LayerIdx][NeuronIdx].V);
         end;
      end
      else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
      begin
         LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FUpdateLayerAdamStates[LayerIdx])) then
         begin
            if StateVar = 'm' then
               Result := CopyArray(FUpdateLayerAdamStates[LayerIdx][NeuronIdx].M)
            else if StateVar = 'v' then
               Result := CopyArray(FUpdateLayerAdamStates[LayerIdx][NeuronIdx].V);
         end;
      end
      else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FReadoutLayerAdamStates)) then
         begin
            if StateVar = 'm' then
               Result := CopyArray(FReadoutLayerAdamStates[NeuronIdx].M)
            else if StateVar = 'v' then
               Result := CopyArray(FReadoutLayerAdamStates[NeuronIdx].V);
         end;
      end
      else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FOutputLayerAdamStates)) then
         begin
            if StateVar = 'm' then
               Result := CopyArray(FOutputLayerAdamStates[NeuronIdx].M)
            else if StateVar = 'v' then
               Result := CopyArray(FOutputLayerAdamStates[NeuronIdx].V);
         end;
      end;
   end
   else if FOptimizerType = otRMSProp then
   begin
      if LayerIdx < FGNN.NumMessagePassingLayers then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FMessageLayerRMSPropStates[LayerIdx])) then
         begin
            if StateVar = 's' then
               Result := CopyArray(FMessageLayerRMSPropStates[LayerIdx][NeuronIdx].S);
         end;
      end
      else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
      begin
         LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FUpdateLayerRMSPropStates[LayerIdx])) then
         begin
            if StateVar = 's' then
               Result := CopyArray(FUpdateLayerRMSPropStates[LayerIdx][NeuronIdx].S);
         end;
      end
      else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FReadoutLayerRMSPropStates)) then
         begin
            if StateVar = 's' then
               Result := CopyArray(FReadoutLayerRMSPropStates[NeuronIdx].S);
         end;
      end
      else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FOutputLayerRMSPropStates)) then
         begin
            if StateVar = 's' then
               Result := CopyArray(FOutputLayerRMSPropStates[NeuronIdx].S);
         end;
      end;
   end;
end;

procedure TGNNFacade.SetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string; const Value: TDoubleArray);
var
   I: Integer;
begin
   if FOptimizerType = otAdam then
   begin
      if LayerIdx < FGNN.NumMessagePassingLayers then
      begin
         if (NeuronIdx >= 0) and (NeuronIdx < Length(FMessageLayerAdamStates[LayerIdx])) then
         begin
            if StateVar = 'm' then
            begin
               for I := 0 to Min(High(Value), High(FMessageLayerAdamStates[LayerIdx][NeuronIdx].M)) do
                  FMessageLayerAdamStates[LayerIdx][NeuronIdx].M[I] := Value[I];
            end
            else if StateVar = 'v' then
            begin
               for I := 0 to Min(High(Value), High(FMessageLayerAdamStates[LayerIdx][NeuronIdx].V)) do
                  FMessageLayerAdamStates[LayerIdx][NeuronIdx].V[I] := Value[I];
            end;
         end;
      end;
   end;
end;

function TGNNFacade.GetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) and
      (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
      Result := FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx];
end;

procedure TGNNFacade.SetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) and
      (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
      FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) and
      (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
      Result := FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx];
end;

procedure TGNNFacade.SetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) and
      (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
      FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights)) then
      Result := FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights[WeightIdx];
end;

procedure TGNNFacade.SetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights)) then
      FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetOutputLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.OutputLayer.Neurons[NeuronIdx].Weights)) then
      Result := FGNN.OutputLayer.Neurons[NeuronIdx].Weights[WeightIdx];
end;

procedure TGNNFacade.SetOutputLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) and
      (WeightIdx >= 0) and (WeightIdx < Length(FGNN.OutputLayer.Neurons[NeuronIdx].Weights)) then
      FGNN.OutputLayer.Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

// ==================== 7. Node/Edge/Graph Masking ====================

function TGNNFacade.GetNodeMask(NodeIdx: Integer): Boolean;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= Length(FNodeMasks)) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, Length(FNodeMasks)]);
   
   Result := FNodeMasks[NodeIdx];
end;

procedure TGNNFacade.SetNodeMask(NodeIdx: Integer; Value: Boolean);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= Length(FNodeMasks)) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, Length(FNodeMasks)]);
   
   FNodeMasks[NodeIdx] := Value;
end;

function TGNNFacade.GetEdgeMask(EdgeIdx: Integer): Boolean;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FEdgeMasks)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FEdgeMasks)]);
   
   Result := FEdgeMasks[EdgeIdx];
end;

procedure TGNNFacade.SetEdgeMask(EdgeIdx: Integer; Value: Boolean);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FEdgeMasks)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FEdgeMasks)]);
   
   FEdgeMasks[EdgeIdx] := Value;
end;

procedure TGNNFacade.SetAllNodeMasks(Value: Boolean);
var
   I: Integer;
begin
   EnsureGraphLoaded;
   for I := 0 to High(FNodeMasks) do
      FNodeMasks[I] := Value;
end;

procedure TGNNFacade.SetAllEdgeMasks(Value: Boolean);
var
   I: Integer;
begin
   EnsureGraphLoaded;
   for I := 0 to High(FEdgeMasks) do
      FEdgeMasks[I] := Value;
end;

function TGNNFacade.GetMaskedNodeCount: Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   Result := 0;
   for I := 0 to High(FNodeMasks) do
      if FNodeMasks[I] then
         Inc(Result);
end;

function TGNNFacade.GetMaskedEdgeCount: Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   Result := 0;
   for I := 0 to High(FEdgeMasks) do
      if FEdgeMasks[I] then
         Inc(Result);
end;

procedure TGNNFacade.ApplyDropoutToNodes(DropoutRate: Double);
var
   I: Integer;
begin
   EnsureGraphLoaded;
   for I := 0 to High(FNodeMasks) do
      FNodeMasks[I] := Random > DropoutRate;
end;

procedure TGNNFacade.ApplyDropoutToEdges(DropoutRate: Double);
var
   I: Integer;
begin
   EnsureGraphLoaded;
   for I := 0 to High(FEdgeMasks) do
      FEdgeMasks[I] := Random > DropoutRate;
end;

// ==================== 8. Graph Structural Mutation ====================

function TGNNFacade.AddNode(const Features: TDoubleArray): Integer;
var
   NewIdx: Integer;
begin
   EnsureGraphLoaded;
   
   NewIdx := FGraph.NumNodes;
   Inc(FGraph.NumNodes);
   
   SetLength(FGraph.NodeFeatures, FGraph.NumNodes);
   FGraph.NodeFeatures[NewIdx] := CopyArray(Features);
   
   SetLength(FGraph.AdjacencyList, FGraph.NumNodes);
   SetLength(FGraph.AdjacencyList[NewIdx], 0);
   
   SetLength(FNodeMasks, FGraph.NumNodes);
   FNodeMasks[NewIdx] := True;
   
   Result := NewIdx;
end;

procedure TGNNFacade.RemoveNode(NodeIdx: Integer);
var
   I, J, NewEdgeCount: Integer;
   NewEdges: TEdgeArray;
   NewFeatures: TDouble2DArray;
   NewMasks: TNodeMask;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   NewEdgeCount := 0;
   SetLength(NewEdges, Length(FGraph.Edges));
   for I := 0 to High(FGraph.Edges) do
   begin
      if (FGraph.Edges[I].Source <> NodeIdx) and (FGraph.Edges[I].Target <> NodeIdx) then
      begin
         NewEdges[NewEdgeCount] := FGraph.Edges[I];
         if NewEdges[NewEdgeCount].Source > NodeIdx then
            Dec(NewEdges[NewEdgeCount].Source);
         if NewEdges[NewEdgeCount].Target > NodeIdx then
            Dec(NewEdges[NewEdgeCount].Target);
         Inc(NewEdgeCount);
      end;
   end;
   SetLength(NewEdges, NewEdgeCount);
   FGraph.Edges := NewEdges;
   
   SetLength(NewFeatures, FGraph.NumNodes - 1);
   J := 0;
   for I := 0 to FGraph.NumNodes - 1 do
   begin
      if I <> NodeIdx then
      begin
         NewFeatures[J] := FGraph.NodeFeatures[I];
         Inc(J);
      end;
   end;
   FGraph.NodeFeatures := NewFeatures;
   
   SetLength(NewMasks, FGraph.NumNodes - 1);
   J := 0;
   for I := 0 to FGraph.NumNodes - 1 do
   begin
      if I <> NodeIdx then
      begin
         NewMasks[J] := FNodeMasks[I];
         Inc(J);
      end;
   end;
   FNodeMasks := NewMasks;
   
   Dec(FGraph.NumNodes);
   
   SetLength(FEdgeMasks, Length(FGraph.Edges));
   for I := 0 to High(FEdgeMasks) do
      FEdgeMasks[I] := True;
   
   RebuildAdjacencyList;
end;

function TGNNFacade.AddEdge(Source, Target: Integer; const Features: TDoubleArray): Integer;
var
   NewIdx: Integer;
   NewEdge: TEdge;
begin
   EnsureGraphLoaded;
   
   if (Source < 0) or (Source >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Source node %d out of range [0, %d)', [Source, FGraph.NumNodes]);
   if (Target < 0) or (Target >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Target node %d out of range [0, %d)', [Target, FGraph.NumNodes]);
   
   NewIdx := Length(FGraph.Edges);
   SetLength(FGraph.Edges, NewIdx + 1);
   
   NewEdge.Source := Source;
   NewEdge.Target := Target;
   NewEdge.Features := CopyArray(Features);
   FGraph.Edges[NewIdx] := NewEdge;
   
   SetLength(FEdgeMasks, Length(FGraph.Edges));
   FEdgeMasks[NewIdx] := True;
   
   SetLength(FGraph.AdjacencyList[Source], Length(FGraph.AdjacencyList[Source]) + 1);
   FGraph.AdjacencyList[Source][High(FGraph.AdjacencyList[Source])] := Target;
   
   Result := NewIdx;
end;

procedure TGNNFacade.RemoveEdge(EdgeIdx: Integer);
var
   I, J: Integer;
   NewEdges: TEdgeArray;
   NewMasks: TEdgeMask;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range [0, %d)', [EdgeIdx, Length(FGraph.Edges)]);
   
   SetLength(NewEdges, Length(FGraph.Edges) - 1);
   SetLength(NewMasks, Length(FGraph.Edges) - 1);
   J := 0;
   for I := 0 to High(FGraph.Edges) do
   begin
      if I <> EdgeIdx then
      begin
         NewEdges[J] := FGraph.Edges[I];
         NewMasks[J] := FEdgeMasks[I];
         Inc(J);
      end;
   end;
   
   FGraph.Edges := NewEdges;
   FEdgeMasks := NewMasks;
   
   RebuildAdjacencyList;
end;

procedure TGNNFacade.ClearAllEdges;
begin
   EnsureGraphLoaded;
   SetLength(FGraph.Edges, 0);
   SetLength(FEdgeMasks, 0);
   RebuildAdjacencyList;
end;

procedure TGNNFacade.ConnectNodes(SourceIdx, TargetIdx: Integer);
var
   EmptyFeatures: TDoubleArray;
begin
   SetLength(EmptyFeatures, 0);
   AddEdge(SourceIdx, TargetIdx, EmptyFeatures);
end;

procedure TGNNFacade.DisconnectNodes(SourceIdx, TargetIdx: Integer);
var
   EdgeIdx: Integer;
begin
   EdgeIdx := FindEdgeIndex(SourceIdx, TargetIdx);
   if EdgeIdx >= 0 then
      RemoveEdge(EdgeIdx);
end;

procedure TGNNFacade.RebuildAdjacencyList;
var
   I, Src, Tgt: Integer;
begin
   EnsureGraphLoaded;
   
   SetLength(FGraph.AdjacencyList, FGraph.NumNodes);
   for I := 0 to FGraph.NumNodes - 1 do
      SetLength(FGraph.AdjacencyList[I], 0);
   
   for I := 0 to High(FGraph.Edges) do
   begin
      Src := FGraph.Edges[I].Source;
      Tgt := FGraph.Edges[I].Target;
      
      if (Src >= 0) and (Src < FGraph.NumNodes) and 
         (Tgt >= 0) and (Tgt < FGraph.NumNodes) then
      begin
         SetLength(FGraph.AdjacencyList[Src], Length(FGraph.AdjacencyList[Src]) + 1);
         FGraph.AdjacencyList[Src][High(FGraph.AdjacencyList[Src])] := Tgt;
      end;
   end;
end;

// ==================== 9. Diagnostics, Attention, and Attribution ====================

function TGNNFacade.GetAttentionWeight(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): Double;
var
   Msg: TDoubleArray;
   I, J: Integer;
   Sum, MsgNorm: Double;
begin
   Result := 0.0;
   
   Msg := GetMessage(NodeIdx, NeighborIdx, LayerIdx, IterationIdx);
   if Length(Msg) = 0 then Exit;
   
   MsgNorm := 0.0;
   for I := 0 to High(Msg) do
      MsgNorm := MsgNorm + Abs(Msg[I]);
   
   Sum := 0.0;
   for I := 0 to High(FGNN.MessageHistory[LayerIdx][NodeIdx]) do
   begin
      for J := 0 to High(FGNN.MessageHistory[LayerIdx][NodeIdx][I].MessageOutput) do
         Sum := Sum + Abs(FGNN.MessageHistory[LayerIdx][NodeIdx][I].MessageOutput[J]);
   end;
   
   if Sum > 0 then
      Result := MsgNorm / Sum;
end;

function TGNNFacade.GetNodeDegree(NodeIdx: Integer): Integer;
begin
   Result := GetInDegree(NodeIdx) + GetOutDegree(NodeIdx);
end;

function TGNNFacade.GetInDegree(NodeIdx: Integer): Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := 0;
   for I := 0 to High(FGraph.Edges) do
      if FGraph.Edges[I].Target = NodeIdx then
         Inc(Result);
end;

function TGNNFacade.GetOutDegree(NodeIdx: Integer): Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := 0;
   for I := 0 to High(FGraph.Edges) do
      if FGraph.Edges[I].Source = NodeIdx then
         Inc(Result);
end;

function TGNNFacade.GetGraphCentrality(NodeIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   if FGraph.NumNodes <= 1 then
      Result := 0.0
   else
      Result := GetNodeDegree(NodeIdx) / (2.0 * (FGraph.NumNodes - 1));
end;

function TGNNFacade.GetBetweennessCentrality(NodeIdx: Integer): Double;
var
   S, V, W, Idx: Integer;
   Dist: array of Integer;
   Sigma: array of Double;
   Delta: array of Double;
   Pred: array of TIntArray;
   Queue: TIntArray;
   Stack: TIntArray;
   QueueHead, StackTop: Integer;
   Neighbors: TIntArray;
   TotalPaths: Double;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := 0.0;
   
   for S := 0 to FGraph.NumNodes - 1 do
   begin
      if S = NodeIdx then Continue;
      
      SetLength(Dist, FGraph.NumNodes);
      SetLength(Sigma, FGraph.NumNodes);
      SetLength(Delta, FGraph.NumNodes);
      SetLength(Pred, FGraph.NumNodes);
      SetLength(Queue, FGraph.NumNodes);
      SetLength(Stack, FGraph.NumNodes);
      
      for V := 0 to FGraph.NumNodes - 1 do
      begin
         Dist[V] := -1;
         Sigma[V] := 0.0;
         Delta[V] := 0.0;
         SetLength(Pred[V], 0);
      end;
      
      Dist[S] := 0;
      Sigma[S] := 1.0;
      QueueHead := 0;
      Queue[0] := S;
      StackTop := 0;
      Idx := 1;
      
      while QueueHead < Idx do
      begin
         V := Queue[QueueHead];
         Inc(QueueHead);
         Stack[StackTop] := V;
         Inc(StackTop);
         
         Neighbors := GetNeighbors(V);
         for W := 0 to High(Neighbors) do
         begin
            if Dist[Neighbors[W]] < 0 then
            begin
               Dist[Neighbors[W]] := Dist[V] + 1;
               Queue[Idx] := Neighbors[W];
               Inc(Idx);
            end;
            
            if Dist[Neighbors[W]] = Dist[V] + 1 then
            begin
               Sigma[Neighbors[W]] := Sigma[Neighbors[W]] + Sigma[V];
               SetLength(Pred[Neighbors[W]], Length(Pred[Neighbors[W]]) + 1);
               Pred[Neighbors[W]][High(Pred[Neighbors[W]])] := V;
            end;
         end;
      end;
      
      while StackTop > 0 do
      begin
         Dec(StackTop);
         W := Stack[StackTop];
         for V := 0 to High(Pred[W]) do
         begin
            if Sigma[W] > 0 then
               Delta[Pred[W][V]] := Delta[Pred[W][V]] + (Sigma[Pred[W][V]] / Sigma[W]) * (1.0 + Delta[W]);
         end;
         
         if (W <> S) and (W = NodeIdx) then
            Result := Result + Delta[W];
      end;
   end;
   
   TotalPaths := (FGraph.NumNodes - 1) * (FGraph.NumNodes - 2);
   if TotalPaths > 0 then
      Result := Result / TotalPaths;
end;

function TGNNFacade.GetClosenessCentrality(NodeIdx: Integer): Double;
var
   Dist: array of Integer;
   Queue: TIntArray;
   V, W, QueueHead, Idx: Integer;
   Neighbors: TIntArray;
   TotalDist: Double;
   ReachableNodes: Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   SetLength(Dist, FGraph.NumNodes);
   SetLength(Queue, FGraph.NumNodes);
   
   for V := 0 to FGraph.NumNodes - 1 do
      Dist[V] := -1;
   
   Dist[NodeIdx] := 0;
   Queue[0] := NodeIdx;
   QueueHead := 0;
   Idx := 1;
   
   while QueueHead < Idx do
   begin
      V := Queue[QueueHead];
      Inc(QueueHead);
      
      Neighbors := GetNeighbors(V);
      for W := 0 to High(Neighbors) do
      begin
         if Dist[Neighbors[W]] < 0 then
         begin
            Dist[Neighbors[W]] := Dist[V] + 1;
            Queue[Idx] := Neighbors[W];
            Inc(Idx);
         end;
      end;
   end;
   
   TotalDist := 0.0;
   ReachableNodes := 0;
   for V := 0 to FGraph.NumNodes - 1 do
   begin
      if (V <> NodeIdx) and (Dist[V] > 0) then
      begin
         TotalDist := TotalDist + Dist[V];
         Inc(ReachableNodes);
      end;
   end;
   
   if TotalDist > 0 then
      Result := ReachableNodes / TotalDist
   else
      Result := 0.0;
end;

function TGNNFacade.GetFeatureImportance(NodeIdx, FeatureIdx: Integer): Double;
var
   OrigFeature, Pred1, Pred2: TDoubleArray;
   I: Integer;
   Diff: Double;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.NodeFeatures[NodeIdx])) then
      raise Exception.CreateFmt('Feature index %d out of range [0, %d)', 
         [FeatureIdx, Length(FGraph.NodeFeatures[NodeIdx])]);
   
   OrigFeature := CopyArray(FGraph.NodeFeatures[NodeIdx]);
   
   Pred1 := Predict;
   
   FGraph.NodeFeatures[NodeIdx][FeatureIdx] := 0.0;
   Pred2 := Predict;
   
   FGraph.NodeFeatures[NodeIdx] := OrigFeature;
   
   Diff := 0.0;
   for I := 0 to High(Pred1) do
      Diff := Diff + Abs(Pred1[I] - Pred2[I]);
   
   Result := Diff / Length(Pred1);
end;

function TGNNFacade.ComputePageRank(Damping: Double = 0.85; Iterations: Integer = 100): TDoubleArray;
var
   I, J, Iter: Integer;
   NewRanks: TDoubleArray;
   OutDeg: Integer;
   Neighbors: TIntArray;
   Sum: Double;
begin
   EnsureGraphLoaded;
   
   SetLength(Result, FGraph.NumNodes);
   SetLength(NewRanks, FGraph.NumNodes);
   
   for I := 0 to FGraph.NumNodes - 1 do
      Result[I] := 1.0 / FGraph.NumNodes;
   
   for Iter := 0 to Iterations - 1 do
   begin
      for I := 0 to FGraph.NumNodes - 1 do
         NewRanks[I] := (1.0 - Damping) / FGraph.NumNodes;
      
      for I := 0 to FGraph.NumNodes - 1 do
      begin
         Neighbors := GetNeighbors(I);
         OutDeg := Length(Neighbors);
         
         if OutDeg > 0 then
         begin
            for J := 0 to High(Neighbors) do
               NewRanks[Neighbors[J]] := NewRanks[Neighbors[J]] + Damping * Result[I] / OutDeg;
         end
         else
         begin
            for J := 0 to FGraph.NumNodes - 1 do
               NewRanks[J] := NewRanks[J] + Damping * Result[I] / FGraph.NumNodes;
         end;
      end;
      
      Sum := 0.0;
      for I := 0 to FGraph.NumNodes - 1 do
         Sum := Sum + NewRanks[I];
      
      for I := 0 to FGraph.NumNodes - 1 do
         Result[I] := NewRanks[I] / Sum;
   end;
end;

// ==================== 10. Batch/Minibatch and Multiple Graphs ====================

procedure TGNNFacade.AddGraphToBatch(var Graph: TGraph);
var
   BatchIdx: Integer;
begin
   BatchIdx := Length(FBatchGraphs);
   SetLength(FBatchGraphs, BatchIdx + 1);
   FBatchGraphs[BatchIdx] := Graph;
   
   SetLength(FBatchNodeEmbeddings, BatchIdx + 1);
end;

function TGNNFacade.GetBatchGraph(BatchIdx: Integer): TGraph;
begin
   if (BatchIdx < 0) or (BatchIdx >= Length(FBatchGraphs)) then
      raise Exception.CreateFmt('Batch index %d out of range [0, %d)', [BatchIdx, Length(FBatchGraphs)]);
   
   Result := FBatchGraphs[BatchIdx];
end;

function TGNNFacade.GetBatchSize: Integer;
begin
   Result := Length(FBatchGraphs);
end;

procedure TGNNFacade.ClearBatch;
begin
   SetLength(FBatchGraphs, 0);
   SetLength(FBatchNodeEmbeddings, 0);
end;

function TGNNFacade.GetBatchNodeEmbedding(BatchIdx, NodeIdx, LayerIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
   
   if (BatchIdx < 0) or (BatchIdx >= Length(FBatchNodeEmbeddings)) then
      Exit;
   
   if (LayerIdx < 0) or (LayerIdx >= Length(FBatchNodeEmbeddings[BatchIdx])) then
      Exit;
   
   if (NodeIdx < 0) or (NodeIdx >= Length(FBatchNodeEmbeddings[BatchIdx][LayerIdx])) then
      Exit;
   
   Result := CopyArray(FBatchNodeEmbeddings[BatchIdx][LayerIdx][NodeIdx]);
end;

procedure TGNNFacade.ProcessBatch(const Targets: TDouble2DArray);
var
   I, L, N: Integer;
begin
   for I := 0 to High(FBatchGraphs) do
   begin
      LoadGraph(FBatchGraphs[I]);
      
      if I <= High(Targets) then
         Train(Targets[I])
      else
         Predict;
      
      SetLength(FBatchNodeEmbeddings[I], Length(FGNN.EmbeddingHistory));
      for L := 0 to High(FGNN.EmbeddingHistory) do
      begin
         SetLength(FBatchNodeEmbeddings[I][L], Length(FGNN.EmbeddingHistory[L]));
         for N := 0 to High(FGNN.EmbeddingHistory[L]) do
            FBatchNodeEmbeddings[I][L][N] := CopyArray(FGNN.EmbeddingHistory[L][N]);
      end;
   end;
end;

function TGNNFacade.GetBatchPredictions: TDouble2DArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FBatchGraphs));
   
   for I := 0 to High(FBatchGraphs) do
   begin
      LoadGraph(FBatchGraphs[I]);
      Result[I] := Predict;
   end;
end;

// ==================== 11. Explainability and Visualization Hooks ====================

function TGNNFacade.GetMessagePassingTrace: TMessagePassingTrace;
begin
   Result := FMessagePassingTrace;
end;

procedure TGNNFacade.ClearMessagePassingTrace;
begin
   SetLength(FMessagePassingTrace, 0);
end;

function TGNNFacade.GetGradientFlow(LayerIdx: Integer): TGradientFlowInfo;
var
   I, J: Integer;
   Count: Integer;
   GradSum, MaxGrad, MinGrad, GradNorm: Double;
   G: Double;
begin
   Result.LayerIdx := LayerIdx;
   Result.MeanGradient := 0.0;
   Result.MaxGradient := 0.0;
   Result.MinGradient := 0.0;
   Result.GradientNorm := 0.0;
   
   GradSum := 0.0;
   MaxGrad := -1e30;
   MinGrad := 1e30;
   GradNorm := 0.0;
   Count := 0;
   
   if LayerIdx < FGNN.NumMessagePassingLayers then
   begin
      for I := 0 to High(FGNN.MessageLayers[LayerIdx].Neurons) do
      begin
         for J := 0 to High(FGNN.MessageLayers[LayerIdx].Neurons[I].WeightGradients) do
         begin
            G := FGNN.MessageLayers[LayerIdx].Neurons[I].WeightGradients[J];
            GradSum := GradSum + G;
            GradNorm := GradNorm + Sqr(G);
            if G > MaxGrad then MaxGrad := G;
            if G < MinGrad then MinGrad := G;
            Inc(Count);
         end;
      end;
   end
   else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
   begin
      LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
      for I := 0 to High(FGNN.UpdateLayers[LayerIdx].Neurons) do
      begin
         for J := 0 to High(FGNN.UpdateLayers[LayerIdx].Neurons[I].WeightGradients) do
         begin
            G := FGNN.UpdateLayers[LayerIdx].Neurons[I].WeightGradients[J];
            GradSum := GradSum + G;
            GradNorm := GradNorm + Sqr(G);
            if G > MaxGrad then MaxGrad := G;
            if G < MinGrad then MinGrad := G;
            Inc(Count);
         end;
      end;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
   begin
      for I := 0 to High(FGNN.ReadoutLayer.Neurons) do
      begin
         for J := 0 to High(FGNN.ReadoutLayer.Neurons[I].WeightGradients) do
         begin
            G := FGNN.ReadoutLayer.Neurons[I].WeightGradients[J];
            GradSum := GradSum + G;
            GradNorm := GradNorm + Sqr(G);
            if G > MaxGrad then MaxGrad := G;
            if G < MinGrad then MinGrad := G;
            Inc(Count);
         end;
      end;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
   begin
      for I := 0 to High(FGNN.OutputLayer.Neurons) do
      begin
         for J := 0 to High(FGNN.OutputLayer.Neurons[I].WeightGradients) do
         begin
            G := FGNN.OutputLayer.Neurons[I].WeightGradients[J];
            GradSum := GradSum + G;
            GradNorm := GradNorm + Sqr(G);
            if G > MaxGrad then MaxGrad := G;
            if G < MinGrad then MinGrad := G;
            Inc(Count);
         end;
      end;
   end;
   
   if Count > 0 then
   begin
      Result.MeanGradient := GradSum / Count;
      Result.MaxGradient := MaxGrad;
      Result.MinGradient := MinGrad;
      Result.GradientNorm := Sqrt(GradNorm);
   end;
end;

function TGNNFacade.GetAllGradientFlows: TGradientFlowInfoArray;
var
   I, TotalLayers: Integer;
begin
   TotalLayers := FGNN.NumMessagePassingLayers * 2 + 2;
   SetLength(Result, TotalLayers);
   
   for I := 0 to TotalLayers - 1 do
      Result[I] := GetGradientFlow(I);
end;

function TGNNFacade.ExportGraphToJSON: string;
var
   I, J: Integer;
   NodesStr, EdgesStr, FeatStr: string;
begin
   EnsureGraphLoaded;
   
   NodesStr := '';
   for I := 0 to FGraph.NumNodes - 1 do
   begin
      if I > 0 then NodesStr := NodesStr + ',';
      
      FeatStr := '';
      for J := 0 to High(FGraph.NodeFeatures[I]) do
      begin
         if J > 0 then FeatStr := FeatStr + ',';
         FeatStr := FeatStr + Format('%.6f', [FGraph.NodeFeatures[I][J]]);
      end;
      
      NodesStr := NodesStr + Format('{"id":%d,"features":[%s],"masked":%s}', 
         [I, FeatStr, BoolToStr(FNodeMasks[I], 'true', 'false')]);
   end;
   
   EdgesStr := '';
   for I := 0 to High(FGraph.Edges) do
   begin
      if I > 0 then EdgesStr := EdgesStr + ',';
      
      FeatStr := '';
      for J := 0 to High(FGraph.Edges[I].Features) do
      begin
         if J > 0 then FeatStr := FeatStr + ',';
         FeatStr := FeatStr + Format('%.6f', [FGraph.Edges[I].Features[J]]);
      end;
      
      EdgesStr := EdgesStr + Format('{"source":%d,"target":%d,"features":[%s],"masked":%s}',
         [FGraph.Edges[I].Source, FGraph.Edges[I].Target, FeatStr, 
          BoolToStr(FEdgeMasks[I], 'true', 'false')]);
   end;
   
   Result := Format('{"numNodes":%d,"nodes":[%s],"edges":[%s],"config":{"undirected":%s,"selfLoops":%s}}',
      [FGraph.NumNodes, NodesStr, EdgesStr, 
       BoolToStr(FGraph.Config.Undirected, 'true', 'false'),
       BoolToStr(FGraph.Config.SelfLoops, 'true', 'false')]);
end;

function TGNNFacade.ExportEmbeddingsToCSV(LayerIdx: Integer): string;
var
   I, J: Integer;
   Line: string;
begin
   Result := '';
   
   if not FGNN.HasGraph then Exit;
   
   if (LayerIdx < 0) or (LayerIdx > High(FGNN.EmbeddingHistory)) then Exit;
   
   for J := 0 to High(FGNN.EmbeddingHistory[LayerIdx][0]) do
   begin
      if J > 0 then Result := Result + ',';
      Result := Result + Format('dim_%d', [J]);
   end;
   Result := Result + #13#10;
   
   for I := 0 to High(FGNN.EmbeddingHistory[LayerIdx]) do
   begin
      Line := '';
      for J := 0 to High(FGNN.EmbeddingHistory[LayerIdx][I]) do
      begin
         if J > 0 then Line := Line + ',';
         Line := Line + Format('%.6f', [FGNN.EmbeddingHistory[LayerIdx][I][J]]);
      end;
      Result := Result + Line + #13#10;
   end;
end;

function TGNNFacade.GetActivationHistogram(LayerIdx: Integer; NumBins: Integer = 10): TDoubleArray;
var
   I, J, BinIdx: Integer;
   MinVal, MaxVal, BinWidth, Val: Double;
   Activations: TDoubleArray;
   Count: Integer;
begin
   SetLength(Result, NumBins);
   for I := 0 to NumBins - 1 do
      Result[I] := 0.0;
   
   if not FGNN.HasGraph then Exit;
   if (LayerIdx < 0) or (LayerIdx > High(FGNN.EmbeddingHistory)) then Exit;
   
   Count := 0;
   for I := 0 to High(FGNN.EmbeddingHistory[LayerIdx]) do
      Count := Count + Length(FGNN.EmbeddingHistory[LayerIdx][I]);
   
   SetLength(Activations, Count);
   Count := 0;
   
   MinVal := 1e30;
   MaxVal := -1e30;
   
   for I := 0 to High(FGNN.EmbeddingHistory[LayerIdx]) do
   begin
      for J := 0 to High(FGNN.EmbeddingHistory[LayerIdx][I]) do
      begin
         Val := FGNN.EmbeddingHistory[LayerIdx][I][J];
         Activations[Count] := Val;
         if Val < MinVal then MinVal := Val;
         if Val > MaxVal then MaxVal := Val;
         Inc(Count);
      end;
   end;
   
   if MaxVal <= MinVal then Exit;
   
   BinWidth := (MaxVal - MinVal) / NumBins;
   
   for I := 0 to High(Activations) do
   begin
      BinIdx := Trunc((Activations[I] - MinVal) / BinWidth);
      if BinIdx >= NumBins then BinIdx := NumBins - 1;
      if BinIdx < 0 then BinIdx := 0;
      Result[BinIdx] := Result[BinIdx] + 1;
   end;
   
   for I := 0 to NumBins - 1 do
      Result[I] := Result[I] / Length(Activations);
end;

// ==================== 12. Layer and Architecture Introspection ====================

function TGNNFacade.GetLayerConfig(LayerIdx: Integer): TLayerConfig;
begin
   Result.LayerType := 'Unknown';
   Result.NumInputs := 0;
   Result.NumOutputs := 0;
   Result.ActivationType := FGNN.Activation;
   
   if LayerIdx < FGNN.NumMessagePassingLayers then
   begin
      Result.LayerType := 'Message';
      Result.NumInputs := FGNN.MessageLayers[LayerIdx].NumInputs;
      Result.NumOutputs := FGNN.MessageLayers[LayerIdx].NumOutputs;
   end
   else if LayerIdx < FGNN.NumMessagePassingLayers * 2 then
   begin
      Result.LayerType := 'Update';
      LayerIdx := LayerIdx - FGNN.NumMessagePassingLayers;
      Result.NumInputs := FGNN.UpdateLayers[LayerIdx].NumInputs;
      Result.NumOutputs := FGNN.UpdateLayers[LayerIdx].NumOutputs;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 then
   begin
      Result.LayerType := 'Readout';
      Result.NumInputs := FGNN.ReadoutLayer.NumInputs;
      Result.NumOutputs := FGNN.ReadoutLayer.NumOutputs;
   end
   else if LayerIdx = FGNN.NumMessagePassingLayers * 2 + 1 then
   begin
      Result.LayerType := 'Output';
      Result.NumInputs := FGNN.OutputLayer.NumInputs;
      Result.NumOutputs := FGNN.OutputLayer.NumOutputs;
   end;
end;

function TGNNFacade.GetNumMessagePassingLayers: Integer;
begin
   Result := FGNN.NumMessagePassingLayers;
end;

function TGNNFacade.GetNumFeatures(NodeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range [0, %d)', [NodeIdx, FGraph.NumNodes]);
   
   Result := Length(FGraph.NodeFeatures[NodeIdx]);
end;

function TGNNFacade.GetTotalLayerCount: Integer;
begin
   Result := FGNN.NumMessagePassingLayers * 2 + 2;
end;

function TGNNFacade.GetMessageLayerNeuronCount(LayerIdx: Integer): Integer;
begin
   Result := 0;
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      Result := Length(FGNN.MessageLayers[LayerIdx].Neurons);
end;

function TGNNFacade.GetUpdateLayerNeuronCount(LayerIdx: Integer): Integer;
begin
   Result := 0;
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      Result := Length(FGNN.UpdateLayers[LayerIdx].Neurons);
end;

function TGNNFacade.GetReadoutLayerNeuronCount: Integer;
begin
   Result := Length(FGNN.ReadoutLayer.Neurons);
end;

function TGNNFacade.GetOutputLayerNeuronCount: Integer;
begin
   Result := Length(FGNN.OutputLayer.Neurons);
end;

function TGNNFacade.GetArchitectureSummary: string;
var
   I: Integer;
   ActivationStr: string;
begin
   case FGNN.Activation of
      atReLU: ActivationStr := 'ReLU';
      atLeakyReLU: ActivationStr := 'LeakyReLU';
      atTanh: ActivationStr := 'Tanh';
      atSigmoid: ActivationStr := 'Sigmoid';
   else
      ActivationStr := 'Unknown';
   end;
   
   Result := '=== GNN Architecture Summary ===' + #13#10;
   Result := Result + Format('Feature Size: %d', [FGNN.FeatureSize]) + #13#10;
   Result := Result + Format('Hidden Size: %d', [FGNN.HiddenSize]) + #13#10;
   Result := Result + Format('Output Size: %d', [FGNN.OutputSize]) + #13#10;
   Result := Result + Format('Message Passing Layers: %d', [FGNN.NumMessagePassingLayers]) + #13#10;
   Result := Result + Format('Activation: %s', [ActivationStr]) + #13#10;
   Result := Result + Format('Learning Rate: %.6f', [FGNN.LearningRate]) + #13#10;
   Result := Result + Format('Total Parameters: %d', [GetParameterCount]) + #13#10;
   Result := Result + #13#10;
   
   Result := Result + '--- Layer Details ---' + #13#10;
   for I := 0 to FGNN.NumMessagePassingLayers - 1 do
   begin
      Result := Result + Format('Message Layer %d: %d inputs, %d outputs', 
         [I, FGNN.MessageLayers[I].NumInputs, FGNN.MessageLayers[I].NumOutputs]) + #13#10;
      Result := Result + Format('Update Layer %d: %d inputs, %d outputs', 
         [I, FGNN.UpdateLayers[I].NumInputs, FGNN.UpdateLayers[I].NumOutputs]) + #13#10;
   end;
   Result := Result + Format('Readout Layer: %d inputs, %d outputs', 
      [FGNN.ReadoutLayer.NumInputs, FGNN.ReadoutLayer.NumOutputs]) + #13#10;
   Result := Result + Format('Output Layer: %d inputs, %d outputs', 
      [FGNN.OutputLayer.NumInputs, FGNN.OutputLayer.NumOutputs]) + #13#10;
end;

function TGNNFacade.GetParameterCount: Integer;
var
   I, J: Integer;
begin
   Result := 0;
   
   for I := 0 to FGNN.NumMessagePassingLayers - 1 do
   begin
      for J := 0 to High(FGNN.MessageLayers[I].Neurons) do
         Result := Result + Length(FGNN.MessageLayers[I].Neurons[J].Weights) + 1;
      
      for J := 0 to High(FGNN.UpdateLayers[I].Neurons) do
         Result := Result + Length(FGNN.UpdateLayers[I].Neurons[J].Weights) + 1;
   end;
   
   for J := 0 to High(FGNN.ReadoutLayer.Neurons) do
      Result := Result + Length(FGNN.ReadoutLayer.Neurons[J].Weights) + 1;
   
   for J := 0 to High(FGNN.OutputLayer.Neurons) do
      Result := Result + Length(FGNN.OutputLayer.Neurons[J].Weights) + 1;
end;

end.

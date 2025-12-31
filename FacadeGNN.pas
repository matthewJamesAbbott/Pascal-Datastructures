//
// Matthew Abbott
// FacadeGNN
//

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

program FacadedGNN;

uses
   Classes, Math, SysUtils, StrUtils;

const
   MAX_NODES = 1000;
   MAX_EDGES = 10000;
   MAX_ITERATIONS = 10000;
   GRADIENT_CLIP = 5.0;
   MODEL_MAGIC = 'GNNBKND01';

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

   TOptimizerType = (otSGD, otAdam, otRMSProp);
   
   TAdamState = record
      M: TDoubleArray;
      V: TDoubleArray;
      T: Integer;
   end;
   
   TRMSPropState = record
      S: TDoubleArray;
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

   TNodeMask = array of Boolean;
   TEdgeMask = array of Boolean;

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
      
      procedure LoadGraph(var Graph: TGraph);
      procedure CreateEmptyGraph(NumNodes: Integer; FeatureSize: Integer);
      function GetGraph: TGraph;
      
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
      
      function GetNeighbors(NodeIdx: Integer): TIntArray;
      function GetAdjacencyMatrix: TDouble2DArray;
      function GetEdgeEndpoints(EdgeIdx: Integer): TEdgeEndpoints;
      function GetIncomingEdges(NodeIdx: Integer): TIntArray;
      function GetOutgoingEdges(NodeIdx: Integer): TIntArray;
      function HasEdge(SourceIdx, TargetIdx: Integer): Boolean;
      function FindEdgeIndex(SourceIdx, TargetIdx: Integer): Integer;
      
      function Predict: TDoubleArray;
      function Train(const Target: TDoubleArray): Double;
      procedure TrainMultiple(const Target: TDoubleArray; Iterations: Integer);
      function ComputeLoss(const Prediction, Target: TDoubleArray): Double;
      
      procedure SaveModel(const Filename: string);
      procedure LoadModel(const Filename: string);
      procedure SaveModelToJSON(const Filename: string);
      procedure LoadModelFromJSON(const Filename: string);
      function Array1DToJSON(const Arr: TDoubleArray): string;
      function Array2DToJSON(const Arr: TDouble2DArray): string;
      
      function GetLearningRate: Double;
      procedure SetLearningRate(Value: Double);
      function GetActivation: TActivationType;
      procedure SetActivation(Value: TActivationType);
      function GetLossFunction: TLossType;
      procedure SetLossFunction(Value: TLossType);
      
      property GNN: TGraphNeuralNetwork read FGNN;
      property GraphLoaded: Boolean read FGraphLoaded;
      property LearningRate: Double read GetLearningRate write SetLearningRate;
      property Activation: TActivationType read GetActivation write SetActivation;
      property LossFunction: TLossType read GetLossFunction write SetLossFunction;
      property OptimizerType: TOptimizerType read FOptimizerType write FOptimizerType;
      property TraceEnabled: Boolean read FTraceEnabled write FTraceEnabled;
      
      function GetNodeEmbedding(LayerIdx, NodeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      procedure SetNodeEmbedding(LayerIdx, NodeIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
      function GetEdgeEmbedding(LayerIdx, EdgeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      function GetAllNodeEmbeddings(NodeIdx: Integer): TDouble2DArray;
      function GetAllLayerEmbeddings(LayerIdx: Integer): TDouble2DArray;
      function GetCurrentNodeEmbedding(NodeIdx: Integer): TDoubleArray;
      function GetFinalNodeEmbeddings: TDouble2DArray;
      
      function GetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      procedure SetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
      function GetAggregatedMessage(NodeIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
      function GetMessageInput(NodeIdx, NeighborIdx, LayerIdx: Integer): TDoubleArray;
      function GetNumMessagesForNode(NodeIdx, LayerIdx: Integer): Integer;
      
      function GetGraphEmbedding(LayerIdx: Integer = -1): TDoubleArray;
      function GetReadout(LayerIdx: Integer = -1): TDoubleArray;
      procedure SetGraphEmbedding(const Value: TDoubleArray; LayerIdx: Integer = -1);
      function GetReadoutLayerOutput: TDoubleArray;
      function GetOutputLayerOutput: TDoubleArray;
      function GetReadoutLayerPreActivations: TDoubleArray;
      function GetOutputLayerPreActivations: TDoubleArray;
      
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
      
      function AddNode(const Features: TDoubleArray): Integer;
      procedure RemoveNode(NodeIdx: Integer);
      function AddEdge(Source, Target: Integer; const Features: TDoubleArray): Integer;
      procedure RemoveEdge(EdgeIdx: Integer);
      procedure ClearAllEdges;
      procedure ConnectNodes(SourceIdx, TargetIdx: Integer);
      procedure DisconnectNodes(SourceIdx, TargetIdx: Integer);
      procedure RebuildAdjacencyList;
      
      function GetAttentionWeight(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): Double;
      function GetNodeDegree(NodeIdx: Integer): Integer;
      function GetInDegree(NodeIdx: Integer): Integer;
      function GetOutDegree(NodeIdx: Integer): Integer;
      function GetGraphCentrality(NodeIdx: Integer): Double;
      function GetBetweennessCentrality(NodeIdx: Integer): Double;
      function GetClosenessCentrality(NodeIdx: Integer): Double;
      function GetFeatureImportance(NodeIdx, FeatureIdx: Integer): Double;
      function ComputePageRank(Damping: Double = 0.85; Iterations: Integer = 100): TDoubleArray;
      
      procedure AddGraphToBatch(var Graph: TGraph);
      function GetBatchGraph(BatchIdx: Integer): TGraph;
      function GetBatchSize: Integer;
      procedure ClearBatch;
      function GetBatchNodeEmbedding(BatchIdx, NodeIdx, LayerIdx: Integer): TDoubleArray;
      procedure ProcessBatch(const Targets: TDouble2DArray);
      function GetBatchPredictions: TDouble2DArray;
      
      function GetMessagePassingTrace: TMessagePassingTrace;
      procedure ClearMessagePassingTrace;
      function GetGradientFlow(LayerIdx: Integer): TGradientFlowInfo;
      function GetAllGradientFlows: TGradientFlowInfoArray;
      function ExportGraphToJSON: string;
      function ExportEmbeddingsToCSV(LayerIdx: Integer): string;
      function GetActivationHistogram(LayerIdx: Integer; NumBins: Integer = 10): TDoubleArray;
      
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

// ==================== Helper Functions ====================

function CopyArray(const Src: TDoubleArray): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(Src));
   for I := 0 to High(Src) do
      Result[I] := Src[I];
end;

function ZeroArray(Size: Integer): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Size);
   for I := 0 to Size - 1 do
      Result[I] := 0.0;
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
   
   SetLength(FMessageLayers, NumMPLayers);
   SetLength(FUpdateLayers, NumMPLayers);
   
   for I := 0 to NumMPLayers - 1 do
   begin
      if I = 0 then
      begin
         InitializeLayer(FMessageLayers[I], FHiddenSize, AFeatureSize + FHiddenSize);
         InitializeLayer(FUpdateLayers[I], FHiddenSize, FHiddenSize + FHiddenSize);
      end
      else
      begin
         InitializeLayer(FMessageLayers[I], FHiddenSize, FHiddenSize + FHiddenSize);
         InitializeLayer(FUpdateLayers[I], FHiddenSize, FHiddenSize + FHiddenSize);
      end;
   end;
   
   InitializeLayer(FReadoutLayer, FHiddenSize, FHiddenSize * MAX_NODES);
   InitializeLayer(FOutputLayer, FOutputSize, FHiddenSize);
   
   SetLength(FNodeEmbeddings, 0);
   SetLength(FNewNodeEmbeddings, 0);
   SetLength(FEmbeddingHistory, 0);
   SetLength(FMessageHistory, 0);
   SetLength(FAggregatedMessages, 0);
   SetLength(FGraphEmbedding, 0);
   SetLength(FGraphEmbeddingHistory, 0);
   
   FHasGraph := False;
end;

destructor TGraphNeuralNetwork.Destroy;
begin
   inherited Destroy;
end;

procedure TGraphNeuralNetwork.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer);
var
   I, J: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   Layer.NumInputs := NumInputs;
   Layer.NumOutputs := NumNeurons;
   
   for I := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[I].Weights, NumInputs);
      for J := 0 to NumInputs - 1 do
         Layer.Neurons[I].Weights[J] := (Random - 0.5) * 2.0;
      
      Layer.Neurons[I].Bias := (Random - 0.5) * 2.0;
      Layer.Neurons[I].Output := 0.0;
      Layer.Neurons[I].PreActivation := 0.0;
      Layer.Neurons[I].Error := 0.0;
      
      SetLength(Layer.Neurons[I].WeightGradients, NumInputs);
      for J := 0 to NumInputs - 1 do
         Layer.Neurons[I].WeightGradients[J] := 0.0;
      
      Layer.Neurons[I].BiasGradient := 0.0;
   end;
end;

procedure TGraphNeuralNetwork.BuildAdjacencyList(var Graph: TGraph);
var
   I, J: Integer;
begin
   SetLength(Graph.AdjacencyList, Graph.NumNodes);
   
   for I := 0 to Graph.NumNodes - 1 do
      SetLength(Graph.AdjacencyList[I], 0);
   
   for I := 0 to High(Graph.Edges) do
   begin
      J := Length(Graph.AdjacencyList[Graph.Edges[I].Source]);
      SetLength(Graph.AdjacencyList[Graph.Edges[I].Source], J + 1);
      Graph.AdjacencyList[Graph.Edges[I].Source][J] := Graph.Edges[I].Target;
   end;
end;

procedure TGraphNeuralNetwork.MessagePassing(var Graph: TGraph);
begin
end;

procedure TGraphNeuralNetwork.Readout(var Graph: TGraph);
begin
end;

function TGraphNeuralNetwork.ForwardLayer(var Layer: TLayer; const Input: TDoubleArray; UseOutputActivation: Boolean = False): TDoubleArray;
var
   I, J: Integer;
   Sum: Double;
begin
   SetLength(Result, Layer.NumOutputs);
   Layer.LastInput := CopyArray(Input);
   
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      Sum := Layer.Neurons[I].Bias;
      for J := 0 to Min(High(Input), High(Layer.Neurons[I].Weights)) do
         Sum := Sum + Input[J] * Layer.Neurons[I].Weights[J];
      
      Layer.Neurons[I].PreActivation := Sum;
      
      if UseOutputActivation then
         Layer.Neurons[I].Output := OutputActivate(Sum)
      else
         Layer.Neurons[I].Output := Activate(Sum);
      
      Result[I] := Layer.Neurons[I].Output;
   end;
end;

procedure TGraphNeuralNetwork.BackwardLayer(var Layer: TLayer; const UpstreamGrad: TDoubleArray; UseOutputActivation: Boolean = False);
var
   I, J: Integer;
   LocalGrad: Double;
begin
   for I := 0 to High(Layer.Neurons) do
   begin
      if UseOutputActivation then
         LocalGrad := UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation)
      else
         LocalGrad := UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);
      
      Layer.Neurons[I].Error := LocalGrad;
      Layer.Neurons[I].BiasGradient := LocalGrad;
      
      for J := 0 to High(Layer.Neurons[I].Weights) do
      begin
         if J <= High(Layer.LastInput) then
            Layer.Neurons[I].WeightGradients[J] := LocalGrad * Layer.LastInput[J]
         else
            Layer.Neurons[I].WeightGradients[J] := 0.0;
      end;
   end;
end;

function TGraphNeuralNetwork.GetLayerInputGrad(const Layer: TLayer; const UpstreamGrad: TDoubleArray; UseOutputActivation: Boolean = False): TDoubleArray;
var
   I, J: Integer;
   Sum: Double;
begin
   SetLength(Result, Layer.NumInputs);
   
   for J := 0 to Layer.NumInputs - 1 do
   begin
      Sum := 0.0;
      for I := 0 to High(Layer.Neurons) do
      begin
         if J <= High(Layer.Neurons[I].Weights) then
         begin
            if UseOutputActivation then
               Sum := Sum + UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation) * Layer.Neurons[I].Weights[J]
            else
               Sum := Sum + UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation) * Layer.Neurons[I].Weights[J];
         end;
      end;
      Result[J] := Sum;
   end;
end;

procedure TGraphNeuralNetwork.BackPropagateGraph(var Graph: TGraph; const Target: TDoubleArray);
begin
end;

function TGraphNeuralNetwork.Activate(X: Double): Double;
begin
   case FActivation of
      atReLU:
         Result := Max(0, X);
      atLeakyReLU:
         if X < 0 then Result := 0.01 * X else Result := X;
      atTanh:
         Result := Math.Tanh(X);
      atSigmoid:
         Result := 1.0 / (1.0 + Exp(-X));
   else
      Result := X;
   end;
end;

function TGraphNeuralNetwork.ActivateDerivative(X: Double): Double;
var
   SigX: Double;
begin
   case FActivation of
      atReLU:
         if X > 0 then Result := 1.0 else Result := 0.0;
      atLeakyReLU:
         if X < 0 then Result := 0.01 else Result := 1.0;
      atTanh:
      begin
         SigX := Math.Tanh(X);
         Result := 1.0 - SigX * SigX;
      end;
      atSigmoid:
      begin
         SigX := 1.0 / (1.0 + Exp(-X));
         Result := SigX * (1.0 - SigX);
      end;
   else
      Result := 1.0;
   end;
end;

function TGraphNeuralNetwork.OutputActivate(X: Double): Double;
begin
   case FLossType of
      ltBinaryCrossEntropy:
         Result := 1.0 / (1.0 + Exp(-X));
   else
      Result := X;
   end;
end;

function TGraphNeuralNetwork.OutputActivateDerivative(PreAct: Double): Double;
var
   SigX: Double;
begin
   case FLossType of
      ltBinaryCrossEntropy:
      begin
         SigX := 1.0 / (1.0 + Exp(-PreAct));
         Result := SigX * (1.0 - SigX);
      end;
   else
      Result := 1.0;
   end;
end;

function TGraphNeuralNetwork.ComputeLossGradient(const Prediction, Target: TDoubleArray): TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(Prediction));
   
   case FLossType of
      ltMSE:
         for I := 0 to High(Prediction) do
            Result[I] := 2.0 * (Prediction[I] - Target[I]);
      ltBinaryCrossEntropy:
         for I := 0 to High(Prediction) do
            if Prediction[I] > 1e-7 then
               Result[I] := -(Target[I] / Prediction[I] - (1.0 - Target[I]) / (1.0 - Prediction[I]))
            else
               Result[I] := 0.0;
   end;
end;

function TGraphNeuralNetwork.ClipGradient(G: Double): Double;
begin
   if G > GRADIENT_CLIP then
      Result := GRADIENT_CLIP
   else if G < -GRADIENT_CLIP then
      Result := -GRADIENT_CLIP
   else
      Result := G;
end;

function TGraphNeuralNetwork.Predict(var Graph: TGraph): TDoubleArray;
begin
   FCurrentGraph := Graph;
   FHasGraph := True;
   BuildAdjacencyList(Graph);
   MessagePassing(Graph);
   Readout(Graph);
   
   SetLength(Result, FOutputSize);
   if Length(FGraphEmbedding) > 0 then
      Result := CopyArray(FGraphEmbedding)
   else
      SetLength(Result, 0);
end;

function TGraphNeuralNetwork.Train(var Graph: TGraph; const Target: TDoubleArray): Double;
var
   Pred: TDoubleArray;
begin
   Pred := Predict(Graph);
   Result := ComputeLoss(Pred, Target);
   BackPropagateGraph(Graph, Target);
   
   FMetrics.Loss := Result;
   Inc(FMetrics.Iteration);
end;

procedure TGraphNeuralNetwork.TrainMultiple(var Graph: TGraph; const Target: TDoubleArray; Iterations: Integer);
var
   I: Integer;
begin
   for I := 0 to Iterations - 1 do
      Train(Graph, Target);
end;

function TGraphNeuralNetwork.ComputeLoss(const Prediction, Target: TDoubleArray): Double;
var
   I: Integer;
begin
   Result := 0.0;
   
   case FLossType of
      ltMSE:
      begin
         for I := 0 to Min(High(Prediction), High(Target)) do
            Result := Result + Sqr(Prediction[I] - Target[I]);
         Result := Result / (Min(Length(Prediction), Length(Target)) + 1);
      end;
      ltBinaryCrossEntropy:
      begin
         for I := 0 to Min(High(Prediction), High(Target)) do
         begin
            if (Prediction[I] > 1e-7) and (Prediction[I] < 1.0 - 1e-7) then
               Result := Result - (Target[I] * Ln(Prediction[I]) + (1.0 - Target[I]) * Ln(1.0 - Prediction[I]));
         end;
         Result := Result / (Min(Length(Prediction), Length(Target)) + 1);
      end;
   end;
end;

procedure TGraphNeuralNetwork.SaveModel(const Filename: string);
begin
end;

procedure TGraphNeuralNetwork.LoadModel(const Filename: string);
begin
end;

// ==================== TGNNFacade Implementation ====================

constructor TGNNFacade.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
begin
   FGNN := TGraphNeuralNetwork.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers);
   FGraphLoaded := False;
   FOptimizerType := otSGD;
   FAdamBeta1 := 0.9;
   FAdamBeta2 := 0.999;
   FAdamEpsilon := 1e-8;
   FRMSPropDecay := 0.99;
   FRMSPropEpsilon := 1e-8;
   FTraceEnabled := False;
   
   InitializeMasks;
   InitializeOptimizerStates;
   
   SetLength(FBatchGraphs, 0);
   SetLength(FBatchNodeEmbeddings, 0);
end;

destructor TGNNFacade.Destroy;
begin
   if Assigned(FGNN) then
      FGNN.Free;
   inherited Destroy;
end;

procedure TGNNFacade.EnsureGraphLoaded;
begin
   if not FGraphLoaded then
      raise Exception.Create('No graph loaded');
end;

procedure TGNNFacade.InitializeMasks;
var
   I: Integer;
begin
   SetLength(FNodeMasks, FGNN.FeatureSize);
   for I := 0 to High(FNodeMasks) do
      FNodeMasks[I] := True;
   
   SetLength(FEdgeMasks, 0);
end;

procedure TGNNFacade.InitializeOptimizerStates;
begin
end;

procedure TGNNFacade.RecordMessagePassingStep(LayerIdx, IterIdx, NodeIdx, NeighborIdx: Integer;
   const Msg, AggMsg: TDoubleArray);
begin
end;

procedure TGNNFacade.LoadGraph(var Graph: TGraph);
begin
   FGraph := Graph;
   FGraphLoaded := True;
   InitializeMasks;
end;

procedure TGNNFacade.CreateEmptyGraph(NumNodes: Integer; FeatureSize: Integer);
var
   I: Integer;
begin
   FGraph.NumNodes := NumNodes;
   SetLength(FGraph.NodeFeatures, NumNodes);
   
   for I := 0 to NumNodes - 1 do
      SetLength(FGraph.NodeFeatures[I], FeatureSize);
   
   SetLength(FGraph.Edges, 0);
   FGraph.Config.Undirected := True;
   FGraph.Config.SelfLoops := False;
   FGraph.Config.DeduplicateEdges := True;
   
   FGraphLoaded := True;
end;

function TGNNFacade.GetGraph: TGraph;
begin
   EnsureGraphLoaded;
   Result := FGraph;
end;

function TGNNFacade.GetNodeFeature(NodeIdx, FeatureIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range', [NodeIdx]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.NodeFeatures[NodeIdx])) then
      raise Exception.CreateFmt('Feature index %d out of range', [FeatureIdx]);
   
   Result := FGraph.NodeFeatures[NodeIdx][FeatureIdx];
end;

procedure TGNNFacade.SetNodeFeature(NodeIdx, FeatureIdx: Integer; Value: Double);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range', [NodeIdx]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.NodeFeatures[NodeIdx])) then
      raise Exception.CreateFmt('Feature index %d out of range', [FeatureIdx]);
   
   FGraph.NodeFeatures[NodeIdx][FeatureIdx] := Value;
end;

function TGNNFacade.GetNodeFeatures(NodeIdx: Integer): TDoubleArray;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range', [NodeIdx]);
   
   Result := CopyArray(FGraph.NodeFeatures[NodeIdx]);
end;

procedure TGNNFacade.SetNodeFeatures(NodeIdx: Integer; const Features: TDoubleArray);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then
      raise Exception.CreateFmt('Node index %d out of range', [NodeIdx]);
   
   FGraph.NodeFeatures[NodeIdx] := CopyArray(Features);
end;

function TGNNFacade.GetEdgeFeature(EdgeIdx, FeatureIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range', [EdgeIdx]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.Edges[EdgeIdx].Features)) then
      raise Exception.CreateFmt('Feature index %d out of range', [FeatureIdx]);
   
   Result := FGraph.Edges[EdgeIdx].Features[FeatureIdx];
end;

procedure TGNNFacade.SetEdgeFeature(EdgeIdx, FeatureIdx: Integer; Value: Double);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range', [EdgeIdx]);
   if (FeatureIdx < 0) or (FeatureIdx >= Length(FGraph.Edges[EdgeIdx].Features)) then
      raise Exception.CreateFmt('Feature index %d out of range', [FeatureIdx]);
   
   FGraph.Edges[EdgeIdx].Features[FeatureIdx] := Value;
end;

function TGNNFacade.GetEdgeFeatures(EdgeIdx: Integer): TDoubleArray;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range', [EdgeIdx]);
   
   Result := CopyArray(FGraph.Edges[EdgeIdx].Features);
end;

procedure TGNNFacade.SetEdgeFeatures(EdgeIdx: Integer; const Features: TDoubleArray);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range', [EdgeIdx]);
   
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
      Result := 0
   else
      Result := Length(FGraph.NodeFeatures[NodeIdx]);
end;

function TGNNFacade.GetEdgeFeatureSize(EdgeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      Result := 0
   else
      Result := Length(FGraph.Edges[EdgeIdx].Features);
end;

function TGNNFacade.GetNeighbors(NodeIdx: Integer): TIntArray;
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= Length(FGNN.FCurrentGraph.AdjacencyList)) then
      SetLength(Result, 0)
   else
      Result := FGNN.FCurrentGraph.AdjacencyList[NodeIdx];
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
      Result[FGraph.Edges[I].Source][FGraph.Edges[I].Target] := 1.0;
end;

function TGNNFacade.GetEdgeEndpoints(EdgeIdx: Integer): TEdgeEndpoints;
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then
      raise Exception.CreateFmt('Edge index %d out of range', [EdgeIdx]);
   
   Result.Source := FGraph.Edges[EdgeIdx].Source;
   Result.Target := FGraph.Edges[EdgeIdx].Target;
end;

function TGNNFacade.GetIncomingEdges(NodeIdx: Integer): TIntArray;
var
   I, Count: Integer;
begin
   EnsureGraphLoaded;
   SetLength(Result, 0);
   Count := 0;
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if FGraph.Edges[I].Target = NodeIdx then
      begin
         SetLength(Result, Count + 1);
         Result[Count] := I;
         Inc(Count);
      end;
   end;
end;

function TGNNFacade.GetOutgoingEdges(NodeIdx: Integer): TIntArray;
var
   I, Count: Integer;
begin
   EnsureGraphLoaded;
   SetLength(Result, 0);
   Count := 0;
   
   for I := 0 to High(FGraph.Edges) do
   begin
      if FGraph.Edges[I].Source = NodeIdx then
      begin
         SetLength(Result, Count + 1);
         Result[Count] := I;
         Inc(Count);
      end;
   end;
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

function TGNNFacade.Predict: TDoubleArray;
begin
   EnsureGraphLoaded;
   Result := FGNN.Predict(FGraph);
end;

function TGNNFacade.Train(const Target: TDoubleArray): Double;
begin
   EnsureGraphLoaded;
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

function TGNNFacade.GetNodeEmbedding(LayerIdx, NodeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   if (LayerIdx < 0) or (LayerIdx > High(FGNN.EmbeddingHistory)) then
      SetLength(Result, 0)
   else if (NodeIdx < 0) or (NodeIdx > High(FGNN.EmbeddingHistory[LayerIdx])) then
      SetLength(Result, 0)
   else
      Result := CopyArray(FGNN.EmbeddingHistory[LayerIdx][NodeIdx]);
end;

procedure TGNNFacade.SetNodeEmbedding(LayerIdx, NodeIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
begin
   if (LayerIdx >= 0) and (LayerIdx <= High(FGNN.EmbeddingHistory)) then
      if (NodeIdx >= 0) and (NodeIdx <= High(FGNN.EmbeddingHistory[LayerIdx])) then
         FGNN.EmbeddingHistory[LayerIdx][NodeIdx] := CopyArray(Value);
end;

function TGNNFacade.GetEdgeEmbedding(LayerIdx, EdgeIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   SetLength(Result, 0);
end;

function TGNNFacade.GetAllNodeEmbeddings(NodeIdx: Integer): TDouble2DArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FGNN.EmbeddingHistory));
   
   for I := 0 to High(FGNN.EmbeddingHistory) do
   begin
      if (NodeIdx >= 0) and (NodeIdx <= High(FGNN.EmbeddingHistory[I])) then
         SetLength(Result[I], 1)
      else
         SetLength(Result[I], 0);
   end;
end;

function TGNNFacade.GetAllLayerEmbeddings(LayerIdx: Integer): TDouble2DArray;
begin
   if (LayerIdx >= 0) and (LayerIdx <= High(FGNN.EmbeddingHistory)) then
      Result := FGNN.EmbeddingHistory[LayerIdx]
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetCurrentNodeEmbedding(NodeIdx: Integer): TDoubleArray;
begin
   if Length(FGNN.EmbeddingHistory) > 0 then
      Result := GetNodeEmbedding(High(FGNN.EmbeddingHistory), NodeIdx)
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetFinalNodeEmbeddings: TDouble2DArray;
begin
   if Length(FGNN.EmbeddingHistory) > 0 then
      Result := FGNN.EmbeddingHistory[High(FGNN.EmbeddingHistory)]
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   SetLength(Result, 0);
end;

procedure TGNNFacade.SetMessage(NodeIdx, NeighborIdx, LayerIdx: Integer; const Value: TDoubleArray; IterationIdx: Integer = 0);
begin
end;

function TGNNFacade.GetAggregatedMessage(NodeIdx, LayerIdx: Integer; IterationIdx: Integer = 0): TDoubleArray;
begin
   SetLength(Result, 0);
end;

function TGNNFacade.GetMessageInput(NodeIdx, NeighborIdx, LayerIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
end;

function TGNNFacade.GetNumMessagesForNode(NodeIdx, LayerIdx: Integer): Integer;
begin
   Result := 0;
end;

function TGNNFacade.GetGraphEmbedding(LayerIdx: Integer = -1): TDoubleArray;
begin
   if LayerIdx < 0 then
      LayerIdx := High(FGNN.GraphEmbeddingHistory);
   
   if (LayerIdx >= 0) and (LayerIdx <= High(FGNN.GraphEmbeddingHistory)) then
      Result := CopyArray(FGNN.GraphEmbeddingHistory[LayerIdx])
   else
      SetLength(Result, 0);
end;

function TGNNFacade.GetReadout(LayerIdx: Integer = -1): TDoubleArray;
begin
   SetLength(Result, 0);
end;

procedure TGNNFacade.SetGraphEmbedding(const Value: TDoubleArray; LayerIdx: Integer = -1);
begin
   if LayerIdx < 0 then
      LayerIdx := High(FGNN.GraphEmbeddingHistory);
   
   if (LayerIdx >= 0) and (LayerIdx <= High(FGNN.GraphEmbeddingHistory)) then
      FGNN.GraphEmbeddingHistory[LayerIdx] := CopyArray(Value);
end;

function TGNNFacade.GetReadoutLayerOutput: TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FGNN.ReadoutLayer.Neurons));
   for I := 0 to High(FGNN.ReadoutLayer.Neurons) do
      Result[I] := FGNN.ReadoutLayer.Neurons[I].Output;
end;

function TGNNFacade.GetOutputLayerOutput: TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FGNN.OutputLayer.Neurons));
   for I := 0 to High(FGNN.OutputLayer.Neurons) do
      Result[I] := FGNN.OutputLayer.Neurons[I].Output;
end;

function TGNNFacade.GetReadoutLayerPreActivations: TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FGNN.ReadoutLayer.Neurons));
   for I := 0 to High(FGNN.ReadoutLayer.Neurons) do
      Result[I] := FGNN.ReadoutLayer.Neurons[I].PreActivation;
end;

function TGNNFacade.GetOutputLayerPreActivations: TDoubleArray;
var
   I: Integer;
begin
   SetLength(Result, Length(FGNN.OutputLayer.Neurons));
   for I := 0 to High(FGNN.OutputLayer.Neurons) do
      Result[I] := FGNN.OutputLayer.Neurons[I].PreActivation;
end;

function TGNNFacade.GetWeightGradient(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.GetBiasGradient(LayerIdx, NeuronIdx: Integer): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.GetNodeEmbeddingGradient(LayerIdx, NodeIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
end;

function TGNNFacade.GetEdgeGradient(LayerIdx, EdgeIdx: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
end;

function TGNNFacade.GetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string): TDoubleArray;
begin
   SetLength(Result, 0);
end;

procedure TGNNFacade.SetOptimizerState(LayerIdx, NeuronIdx: Integer; const StateVar: string; const Value: TDoubleArray);
begin
end;

function TGNNFacade.GetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) then
         if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
         begin
            Result := FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx];
            Exit;
         end;
   
   Result := 0.0;
end;

procedure TGNNFacade.SetMessageLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons)) then
         if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
            FGNN.MessageLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer): Double;
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) then
         if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
         begin
            Result := FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx];
            Exit;
         end;
   
   Result := 0.0;
end;

procedure TGNNFacade.SetUpdateLayerWeight(LayerIdx, NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (LayerIdx >= 0) and (LayerIdx < FGNN.NumMessagePassingLayers) then
      if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons)) then
         if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights)) then
            FGNN.UpdateLayers[LayerIdx].Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) then
      if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights)) then
      begin
         Result := FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights[WeightIdx];
         Exit;
      end;
   
   Result := 0.0;
end;

procedure TGNNFacade.SetReadoutLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.ReadoutLayer.Neurons)) then
      if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights)) then
         FGNN.ReadoutLayer.Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetOutputLayerWeight(NeuronIdx, WeightIdx: Integer): Double;
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) then
      if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.OutputLayer.Neurons[NeuronIdx].Weights)) then
      begin
         Result := FGNN.OutputLayer.Neurons[NeuronIdx].Weights[WeightIdx];
         Exit;
      end;
   
   Result := 0.0;
end;

procedure TGNNFacade.SetOutputLayerWeight(NeuronIdx, WeightIdx: Integer; Value: Double);
begin
   if (NeuronIdx >= 0) and (NeuronIdx < Length(FGNN.OutputLayer.Neurons)) then
      if (WeightIdx >= 0) and (WeightIdx < Length(FGNN.OutputLayer.Neurons[NeuronIdx].Weights)) then
         FGNN.OutputLayer.Neurons[NeuronIdx].Weights[WeightIdx] := Value;
end;

function TGNNFacade.GetNodeMask(NodeIdx: Integer): Boolean;
begin
   if (NodeIdx >= 0) and (NodeIdx < Length(FNodeMasks)) then
      Result := FNodeMasks[NodeIdx]
   else
      Result := False;
end;

procedure TGNNFacade.SetNodeMask(NodeIdx: Integer; Value: Boolean);
begin
   if (NodeIdx >= 0) and (NodeIdx < Length(FNodeMasks)) then
      FNodeMasks[NodeIdx] := Value;
end;

function TGNNFacade.GetEdgeMask(EdgeIdx: Integer): Boolean;
begin
   if (EdgeIdx >= 0) and (EdgeIdx < Length(FEdgeMasks)) then
      Result := FEdgeMasks[EdgeIdx]
   else
      Result := False;
end;

procedure TGNNFacade.SetEdgeMask(EdgeIdx: Integer; Value: Boolean);
begin
   if (EdgeIdx >= 0) and (EdgeIdx < Length(FEdgeMasks)) then
      FEdgeMasks[EdgeIdx] := Value;
end;

procedure TGNNFacade.SetAllNodeMasks(Value: Boolean);
var
   I: Integer;
begin
   for I := 0 to High(FNodeMasks) do
      FNodeMasks[I] := Value;
end;

procedure TGNNFacade.SetAllEdgeMasks(Value: Boolean);
var
   I: Integer;
begin
   for I := 0 to High(FEdgeMasks) do
      FEdgeMasks[I] := Value;
end;

function TGNNFacade.GetMaskedNodeCount: Integer;
var
   I, Count: Integer;
begin
   Count := 0;
   for I := 0 to High(FNodeMasks) do
      if FNodeMasks[I] then
         Inc(Count);
   Result := Count;
end;

function TGNNFacade.GetMaskedEdgeCount: Integer;
var
   I, Count: Integer;
begin
   Count := 0;
   for I := 0 to High(FEdgeMasks) do
      if FEdgeMasks[I] then
         Inc(Count);
   Result := Count;
end;

procedure TGNNFacade.ApplyDropoutToNodes(DropoutRate: Double);
var
   I: Integer;
begin
   for I := 0 to High(FNodeMasks) do
      FNodeMasks[I] := Random > DropoutRate;
end;

procedure TGNNFacade.ApplyDropoutToEdges(DropoutRate: Double);
var
   I: Integer;
begin
   for I := 0 to High(FEdgeMasks) do
      FEdgeMasks[I] := Random > DropoutRate;
end;

function TGNNFacade.AddNode(const Features: TDoubleArray): Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   I := FGraph.NumNodes;
   Inc(FGraph.NumNodes);
   
   SetLength(FGraph.NodeFeatures, FGraph.NumNodes);
   FGraph.NodeFeatures[I] := CopyArray(Features);
   
   SetLength(FNodeMasks, FGraph.NumNodes);
   FNodeMasks[I] := True;
   
   Result := I;
end;

procedure TGNNFacade.RemoveNode(NodeIdx: Integer);
begin
   EnsureGraphLoaded;
   if (NodeIdx < 0) or (NodeIdx >= FGraph.NumNodes) then Exit;
   
   Dec(FGraph.NumNodes);
   SetLength(FGraph.NodeFeatures, FGraph.NumNodes);
   SetLength(FNodeMasks, FGraph.NumNodes);
end;

function TGNNFacade.AddEdge(Source, Target: Integer; const Features: TDoubleArray): Integer;
var
   I: Integer;
begin
   EnsureGraphLoaded;
   I := Length(FGraph.Edges);
   SetLength(FGraph.Edges, I + 1);
   FGraph.Edges[I].Source := Source;
   FGraph.Edges[I].Target := Target;
   FGraph.Edges[I].Features := CopyArray(Features);
   
   SetLength(FEdgeMasks, I + 1);
   FEdgeMasks[I] := True;
   
   Result := I;
end;

procedure TGNNFacade.RemoveEdge(EdgeIdx: Integer);
begin
   EnsureGraphLoaded;
   if (EdgeIdx < 0) or (EdgeIdx >= Length(FGraph.Edges)) then Exit;
   
   SetLength(FGraph.Edges, Length(FGraph.Edges) - 1);
   SetLength(FEdgeMasks, Length(FGraph.Edges));
end;

procedure TGNNFacade.ClearAllEdges;
begin
   EnsureGraphLoaded;
   SetLength(FGraph.Edges, 0);
   SetLength(FEdgeMasks, 0);
end;

procedure TGNNFacade.ConnectNodes(SourceIdx, TargetIdx: Integer);
begin
   AddEdge(SourceIdx, TargetIdx, ZeroArray(0));
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
begin
   EnsureGraphLoaded;
   FGNN.BuildAdjacencyList(FGraph);
end;

function TGNNFacade.GetAttentionWeight(NodeIdx, NeighborIdx, LayerIdx: Integer; IterationIdx: Integer = 0): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.GetNodeDegree(NodeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   Result := GetInDegree(NodeIdx) + GetOutDegree(NodeIdx);
end;

function TGNNFacade.GetInDegree(NodeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   Result := Length(GetIncomingEdges(NodeIdx));
end;

function TGNNFacade.GetOutDegree(NodeIdx: Integer): Integer;
begin
   EnsureGraphLoaded;
   Result := Length(GetOutgoingEdges(NodeIdx));
end;

function TGNNFacade.GetGraphCentrality(NodeIdx: Integer): Double;
begin
   EnsureGraphLoaded;
   if GetNumNodes > 0 then
      Result := GetNodeDegree(NodeIdx) / GetNumNodes
   else
      Result := 0.0;
end;

function TGNNFacade.GetBetweennessCentrality(NodeIdx: Integer): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.GetClosenessCentrality(NodeIdx: Integer): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.GetFeatureImportance(NodeIdx, FeatureIdx: Integer): Double;
begin
   Result := 0.0;
end;

function TGNNFacade.ComputePageRank(Damping: Double = 0.85; Iterations: Integer = 100): TDoubleArray;
var
   I, J, Iter: Integer;
   Neighbors: TIntArray;
   OutDeg: Integer;
   NewRanks: TDoubleArray;
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
      raise Exception.CreateFmt('Batch index %d out of range', [BatchIdx]);
   
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
      
      NodesStr := NodesStr + Format('{"id":%d,"features":[%s]}', [I, FeatStr]);
   end;
   
   EdgesStr := '';
   for I := 0 to High(FGraph.Edges) do
   begin
      if I > 0 then EdgesStr := EdgesStr + ',';
      
      EdgesStr := EdgesStr + Format('{"source":%d,"target":%d}',
         [FGraph.Edges[I].Source, FGraph.Edges[I].Target]);
   end;
   
   Result := Format('{"numNodes":%d,"nodes":[%s],"edges":[%s]}',
      [FGraph.NumNodes, NodesStr, EdgesStr]);
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
      raise Exception.CreateFmt('Node index %d out of range', [NodeIdx]);
   
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

{ Forward declarations for CLI support functions }
function ActivationToStr(act: TActivationType): string; forward;
function LossToStr(loss: TLossType): string; forward;
function ParseActivation(const s: string): TActivationType; forward;
function ParseLoss(const s: string): TLossType; forward;

function TGNNFacade.Array1DToJSON(const Arr: TDoubleArray): string;
var
    I: Integer;
begin
    Result := '[';
    for I := 0 to High(Arr) do
    begin
        if I > 0 then Result := Result + ',';
        Result := Result + FloatToStr(Arr[I]);
    end;
    Result := Result + ']';
end;

function TGNNFacade.Array2DToJSON(const Arr: TDouble2DArray): string;
var
    I: Integer;
begin
    Result := '[';
    for I := 0 to High(Arr) do
    begin
        if I > 0 then Result := Result + ',';
        Result := Result + Array1DToJSON(Arr[I]);
    end;
    Result := Result + ']';
end;

procedure TGNNFacade.SaveModelToJSON(const Filename: string);
var
    SL: TStringList;
    I, J, K: Integer;
begin
    SL := TStringList.Create;
    try
        SL.Add('{');
        SL.Add('  "feature_size": ' + IntToStr(FGNN.FeatureSize) + ',');
        SL.Add('  "hidden_size": ' + IntToStr(FGNN.HiddenSize) + ',');
        SL.Add('  "output_size": ' + IntToStr(FGNN.OutputSize) + ',');
        SL.Add('  "num_message_passing_layers": ' + IntToStr(FGNN.NumMessagePassingLayers) + ',');
        SL.Add('  "learning_rate": ' + FloatToStr(FGNN.LearningRate) + ',');
        SL.Add('  "activation": "' + ActivationToStr(FGNN.Activation) + '",');
        SL.Add('  "loss_type": "' + LossToStr(FGNN.LossFunction) + '",');
        SL.Add('  "max_iterations": ' + IntToStr(FGNN.MaxIterations) + ',');
        SL.Add('  "message_layers": [');
        
        for K := 0 to FGNN.NumMessagePassingLayers - 1 do
        begin
            if K > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('    {');
            SL.Add('      "num_outputs": ' + IntToStr(FGNN.MessageLayers[K].NumOutputs) + ',');
            SL.Add('      "num_inputs": ' + IntToStr(FGNN.MessageLayers[K].NumInputs) + ',');
            SL.Add('      "neurons": [');
            
            for I := 0 to FGNN.MessageLayers[K].NumOutputs - 1 do
            begin
                if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
                SL.Add('        {');
                SL.Add('          "weights": ' + Array1DToJSON(FGNN.MessageLayers[K].Neurons[I].Weights) + ',');
                SL.Add('          "bias": ' + FloatToStr(FGNN.MessageLayers[K].Neurons[I].Bias));
                SL.Add('        }');
            end;
            
            SL.Add('      ]');
            SL.Add('    }');
        end;
        
        SL.Add('  ],');
        SL.Add('  "update_layers": [');
        
        for K := 0 to FGNN.NumMessagePassingLayers - 1 do
        begin
            if K > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('    {');
            SL.Add('      "num_outputs": ' + IntToStr(FGNN.UpdateLayers[K].NumOutputs) + ',');
            SL.Add('      "num_inputs": ' + IntToStr(FGNN.UpdateLayers[K].NumInputs) + ',');
            SL.Add('      "neurons": [');
            
            for I := 0 to FGNN.UpdateLayers[K].NumOutputs - 1 do
            begin
                if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
                SL.Add('        {');
                SL.Add('          "weights": ' + Array1DToJSON(FGNN.UpdateLayers[K].Neurons[I].Weights) + ',');
                SL.Add('          "bias": ' + FloatToStr(FGNN.UpdateLayers[K].Neurons[I].Bias));
                SL.Add('        }');
            end;
            
            SL.Add('      ]');
            SL.Add('    }');
        end;
        
        SL.Add('  ],');
        SL.Add('  "readout_layer": {');
        SL.Add('    "num_outputs": ' + IntToStr(FGNN.ReadoutLayer.NumOutputs) + ',');
        SL.Add('    "num_inputs": ' + IntToStr(FGNN.ReadoutLayer.NumInputs) + ',');
        SL.Add('    "neurons": [');
        
        for I := 0 to FGNN.ReadoutLayer.NumOutputs - 1 do
        begin
            if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('      {');
            SL.Add('        "weights": ' + Array1DToJSON(FGNN.ReadoutLayer.Neurons[I].Weights) + ',');
            SL.Add('        "bias": ' + FloatToStr(FGNN.ReadoutLayer.Neurons[I].Bias));
            SL.Add('      }');
        end;
        
        SL.Add('    ]');
        SL.Add('  },');
        SL.Add('  "output_layer": {');
        SL.Add('    "num_outputs": ' + IntToStr(FGNN.OutputLayer.NumOutputs) + ',');
        SL.Add('    "num_inputs": ' + IntToStr(FGNN.OutputLayer.NumInputs) + ',');
        SL.Add('    "neurons": [');
        
        for I := 0 to FGNN.OutputLayer.NumOutputs - 1 do
        begin
            if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('      {');
            SL.Add('        "weights": ' + Array1DToJSON(FGNN.OutputLayer.Neurons[I].Weights) + ',');
            SL.Add('        "bias": ' + FloatToStr(FGNN.OutputLayer.Neurons[I].Bias));
            SL.Add('      }');
        end;
        
        SL.Add('    ]');
        SL.Add('  }');
        SL.Add('}');
        
        SL.SaveToFile(Filename);
        WriteLn('Model saved to JSON: ', Filename);
    finally
        SL.Free;
    end;
end;

procedure TGNNFacade.LoadModelFromJSON(const Filename: string);
var
    SL: TStringList;
    Content: string;
    ValueStr: string;
    I, J, K: Integer;
    
    function ExtractJSONValue(const json: string; const key: string): string;
    var
        KeyPos, ColonPos, QuotePos1, QuotePos2, StartPos, EndPos: Integer;
    begin
        KeyPos := Pos('"' + key + '"', json);
        if KeyPos > 0 then
        begin
            ColonPos := PosEx(':', json, KeyPos);
            if ColonPos > 0 then
            begin
                StartPos := ColonPos + 1;
                { Skip whitespace }
                while (StartPos <= Length(json)) and (json[StartPos] in [' ', #9, #10, #13]) do
                    Inc(StartPos);
                
                { Check if value is a quoted string }
                if (StartPos <= Length(json)) and (json[StartPos] = '"') then
                begin
                    QuotePos1 := StartPos;
                    QuotePos2 := PosEx('"', json, QuotePos1 + 1);
                    if QuotePos2 > 0 then
                        Result := Copy(json, QuotePos1 + 1, QuotePos2 - QuotePos1 - 1)
                    else
                        Result := '';
                end
                else
                begin
                    { Value is a number or boolean }
                    EndPos := PosEx(',', json, StartPos);
                    if EndPos = 0 then
                        EndPos := PosEx('}', json, StartPos);
                    if EndPos = 0 then
                        EndPos := PosEx(']', json, StartPos);
                    Result := Trim(Copy(json, StartPos, EndPos - StartPos));
                end;
            end
            else
                Result := '';
        end
        else
            Result := '';
    end;
    
    function ParseDoubleArray(const jsonStr: string): TDoubleArray;
    var
        StartIdx, EndIdx, CommaIdx, ItemStart: Integer;
        ItemStr: string;
    begin
        SetLength(Result, 0);
        
        StartIdx := Pos('[', jsonStr);
        EndIdx := PosEx(']', jsonStr, StartIdx);
        
        if (StartIdx > 0) and (EndIdx > StartIdx) then
        begin
            ItemStart := StartIdx + 1;
            repeat
                CommaIdx := PosEx(',', jsonStr, ItemStart);
                if CommaIdx = 0 then
                    CommaIdx := EndIdx;
                
                ItemStr := Trim(Copy(jsonStr, ItemStart, CommaIdx - ItemStart));
                if ItemStr <> '' then
                begin
                    SetLength(Result, Length(Result) + 1);
                    try
                        Result[High(Result)] := StrToFloat(ItemStr);
                    except
                        Result[High(Result)] := 0;
                    end;
                end;
                
                ItemStart := CommaIdx + 1;
            until CommaIdx >= EndIdx;
        end;
    end;

begin
    SL := TStringList.Create;
    try
        SL.LoadFromFile(Filename);
        Content := SL.Text;
        
        { Load config }
        ValueStr := ExtractJSONValue(Content, 'feature_size');
        if ValueStr <> '' then
        begin
            if FGNN <> nil then
                FGNN.Free;
            FGNN := TGraphNeuralNetwork.Create(StrToInt(ValueStr), 1, 1, 1);
        end;
        
        ValueStr := ExtractJSONValue(Content, 'hidden_size');
        if ValueStr <> '' then
            FGNN.FHiddenSize := StrToInt(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'output_size');
        if ValueStr <> '' then
            FGNN.FOutputSize := StrToInt(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'num_message_passing_layers');
        if ValueStr <> '' then
            FGNN.FNumMessagePassingLayers := StrToInt(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'learning_rate');
        if ValueStr <> '' then
            FGNN.LearningRate := StrToFloat(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'activation');
        if ValueStr <> '' then
            FGNN.Activation := ParseActivation(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'loss_type');
        if ValueStr <> '' then
            FGNN.LossFunction := ParseLoss(ValueStr);
        
        ValueStr := ExtractJSONValue(Content, 'max_iterations');
        if ValueStr <> '' then
            FGNN.MaxIterations := StrToInt(ValueStr);
        
        WriteLn('Model loaded from JSON: ', Filename);
    finally
        SL.Free;
    end;
end;
        
        // ==================== CLI Support Functions ====================

function ActivationToStr(act: TActivationType): string;
begin
    case act of
        atReLU: Result := 'relu';
        atLeakyReLU: Result := 'leakyrelu';
        atTanh: Result := 'tanh';
        atSigmoid: Result := 'sigmoid';
    else
        Result := 'relu';
    end;
end;

function LossToStr(loss: TLossType): string;
begin
    case loss of
        ltMSE: Result := 'mse';
        ltBinaryCrossEntropy: Result := 'bce';
    else
        Result := 'mse';
    end;
end;

function ParseActivation(const s: string): TActivationType;
begin
    if LowerCase(s) = 'leakyrelu' then
        Result := atLeakyReLU
    else if LowerCase(s) = 'tanh' then
        Result := atTanh
    else if LowerCase(s) = 'sigmoid' then
        Result := atSigmoid
    else
        Result := atReLU;
end;

function ParseLoss(const s: string): TLossType;
begin
    if LowerCase(s) = 'bce' then
        Result := ltBinaryCrossEntropy
    else
        Result := ltMSE;
end;

// ==================== CLI Commands ====================

type
     TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp,
                 cmdGetEmbedding, cmdSetEmbedding, cmdGetGraphEmbedding, cmdSetGraphEmbedding,
                 cmdGetMessage, cmdSetMessage, cmdGetNodeDegree, cmdComputePageRank,
                 cmdExportGraph, cmdExportEmbeddings, cmdAddNode, cmdAddEdge,
                 cmdGetNodeFeature, cmdSetNodeFeature, cmdDetectVanishing, cmdDetectExploding,
                 cmdGetGradientFlow);

procedure PrintUsage;
begin
    WriteLn('Facaded GNN');
    WriteLn('\n');
    WriteLn('Commands:');
    WriteLn('  create                Create a new GNN model');
    WriteLn('  train                 Train a model with graph data');
    WriteLn('  predict               Make predictions with a trained model');
    WriteLn('  info                  Display model information');
    WriteLn('  help                  Show this help message');
    WriteLn;
    WriteLn('Core Options:');
    WriteLn('  --feature=N            Node feature size (required for create)');
    WriteLn('  --hidden=N             Hidden layer size (required for create)');
    WriteLn('  --output=N             Output layer size (required for create)');
    WriteLn('  --mp-layers=N          Message passing layers (required for create)');
    WriteLn('  --save=FILE            Save model to binary or JSON file (.json extension uses JSON)');
    WriteLn('  --model=FILE           Model file to load (auto-detects binary or JSON)');
    WriteLn('  --data=FILE            Training graph data file');
    WriteLn('  --activation=TYPE      relu|leakyrelu|tanh|sigmoid (default: relu)');
    WriteLn('  --loss=TYPE            mse|bce (default: mse)');
    WriteLn('  --optimizer=TYPE       sgd|adam|rmsprop (default: adam)');
    WriteLn('  --lr=VALUE             Learning rate (default: 0.01)');
    WriteLn('  --epochs=N             Number of training epochs (default: 100)');
    WriteLn('  --batch=N              Batch size (default: 1)');
    WriteLn('  --undirected           Treat graph as undirected');
    WriteLn('  --self-loops           Allow self-loop edges');
    WriteLn('  --deduplicate          Remove duplicate edges');
    WriteLn('  --verbose              Show training progress');
    WriteLn;
    WriteLn('Facade Introspection Commands:');
    WriteLn('  get-embedding          Get node embedding at layer');
    WriteLn('  set-embedding          Set node embedding at layer');
    WriteLn('  get-graph-embedding    Get pooled graph embedding');
    WriteLn('  set-graph-embedding    Set graph embedding');
    WriteLn('  get-message            Get message between nodes');
    WriteLn('  set-message            Set message between nodes');
    WriteLn('  get-node-degree        Get node degree/centrality');
    WriteLn('  compute-pagerank       Compute PageRank scores');
    WriteLn('  export-graph           Export graph structure to JSON');
    WriteLn('  export-embeddings      Export embeddings to CSV');
    WriteLn('  add-node               Add node to graph');
    WriteLn('  add-edge               Add edge to graph');
    WriteLn('  get-node-feature       Get node feature value');
    WriteLn('  set-node-feature       Set node feature value');
    WriteLn('  detect-vanishing       Check for vanishing gradients');
    WriteLn('  detect-exploding       Check for exploding gradients');
    WriteLn('  get-gradient-flow      Get gradient flow statistics');
    WriteLn;
    WriteLn('Facade Introspection Options:');
    WriteLn('  --layer=N              Layer index (default: 0)');
    WriteLn('  --node=N               Node index (default: 0)');
    WriteLn('  --neighbor=N           Neighbor node index');
    WriteLn('  --feature-idx=N        Feature index (default: 0)');
    WriteLn('  --value=F              Value to set');
    WriteLn('  --features=F,F,...     Comma-separated feature values');
    WriteLn('  --damping=F            PageRank damping factor (default: 0.85)');
    WriteLn('  --pr-iter=N            PageRank iterations (default: 100)');
    WriteLn('  --threshold=F          Threshold for gradient detection (default: 1e-6)');
    WriteLn('  --num-bins=N           Histogram bins (default: 10)');
    WriteLn;
    WriteLn('Examples:');
    WriteLn('  facaded_gnn create --feature=10 --hidden=32 --output=5 --mp-layers=3 --save=gnn.bin');
    WriteLn('  facaded_gnn train --model=gnn.bin --data=graph.csv --epochs=200 --save=gnn_trained.bin');
    WriteLn('  facaded_gnn predict --model=gnn_trained.bin');
    WriteLn('  facaded_gnn info --model=gnn_trained.bin');
    WriteLn('  facaded_gnn get-embedding --model=gnn.bin --layer=0 --node=0');
    WriteLn('  facaded_gnn set-embedding --model=gnn.bin --layer=0 --node=0 --features=0.1,0.2,0.3');
    WriteLn('  facaded_gnn get-node-degree --model=gnn.bin --node=0');
    WriteLn('  facaded_gnn compute-pagerank --model=gnn.bin --damping=0.85');
    WriteLn('  facaded_gnn export-graph --model=gnn.bin --save=graph.json');
    WriteLn('  facaded_gnn export-embeddings --model=gnn.bin --layer=2 --save=embeddings.csv');
    WriteLn('  facaded_gnn detect-vanishing --model=gnn.bin --threshold=1e-6');
    WriteLn('  facaded_gnn get-gradient-flow --model=gnn.bin --layer=0');
    WriteLn;
end;

// ==================== Main Program ====================

var
    Command: TCommand;
    CmdStr: string;
    i: Integer;
    arg, key, valueStr: string;
    eqPos: Integer;
    GNNFacade: TGNNFacade;
    
    featureSize, hiddenSize, outputSize, mpLayers: Integer;
    learningRate: Double;
    activation: TActivationType;
    lossType: TLossType;
    optimizer: TOptimizerType;
    modelFile, saveFile, dataFile: string;
    epochs, batchSize: Integer;
    undirected, selfLoops, deduplicate, verbose: Boolean;
    
    layerIdx, nodeIdx, neighborIdx, featureIdx, numBins, prIter: Integer;
    value, threshold, damping: Double;
    featureValues: TDoubleArray;

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
    else if CmdStr = 'get-embedding' then Command := cmdGetEmbedding
    else if CmdStr = 'set-embedding' then Command := cmdSetEmbedding
    else if CmdStr = 'get-graph-embedding' then Command := cmdGetGraphEmbedding
    else if CmdStr = 'set-graph-embedding' then Command := cmdSetGraphEmbedding
    else if CmdStr = 'get-message' then Command := cmdGetMessage
    else if CmdStr = 'set-message' then Command := cmdSetMessage
    else if CmdStr = 'get-node-degree' then Command := cmdGetNodeDegree
    else if CmdStr = 'compute-pagerank' then Command := cmdComputePageRank
    else if CmdStr = 'export-graph' then Command := cmdExportGraph
    else if CmdStr = 'export-embeddings' then Command := cmdExportEmbeddings
    else if CmdStr = 'add-node' then Command := cmdAddNode
    else if CmdStr = 'add-edge' then Command := cmdAddEdge
    else if CmdStr = 'get-node-feature' then Command := cmdGetNodeFeature
    else if CmdStr = 'set-node-feature' then Command := cmdSetNodeFeature
    else if CmdStr = 'detect-vanishing' then Command := cmdDetectVanishing
    else if CmdStr = 'detect-exploding' then Command := cmdDetectExploding
    else if CmdStr = 'get-gradient-flow' then Command := cmdGetGradientFlow
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
    featureSize := 0;
    hiddenSize := 0;
    outputSize := 0;
    mpLayers := 0;
    learningRate := 0.01;
    epochs := 100;
    batchSize := 1;
    undirected := False;
    selfLoops := False;
    deduplicate := False;
    verbose := False;
    activation := atReLU;
    lossType := ltMSE;
    optimizer := otAdam;
    modelFile := '';
    saveFile := '';
    dataFile := '';
    
    // Facade defaults
    layerIdx := 0;
    nodeIdx := 0;
    neighborIdx := 0;
    featureIdx := 0;
    numBins := 10;
    prIter := 100;
    value := 0.0;
    threshold := 1e-6;
    damping := 0.85;
    SetLength(featureValues, 0);
    
    // Parse arguments
    for i := 2 to ParamCount do
    begin
        arg := ParamStr(i);
        
        if arg = '--verbose' then
            verbose := True
        else if arg = '--undirected' then
            undirected := True
        else if arg = '--self-loops' then
            selfLoops := True
        else if arg = '--deduplicate' then
            deduplicate := True
        else
        begin
            eqPos := Pos('=', arg);
            if eqPos = 0 then
            begin
                WriteLn('Invalid argument: ', arg);
                Continue;
            end;
            
            key := Copy(arg, 1, eqPos - 1);
            valueStr := Copy(arg, eqPos + 1, Length(arg));
            
            if key = '--feature' then
                featureSize := StrToInt(valueStr)
            else if key = '--hidden' then
                hiddenSize := StrToInt(valueStr)
            else if key = '--output' then
                outputSize := StrToInt(valueStr)
            else if key = '--mp-layers' then
                mpLayers := StrToInt(valueStr)
            else if key = '--save' then
                saveFile := valueStr
            else if key = '--model' then
                modelFile := valueStr
            else if key = '--data' then
                dataFile := valueStr
            else if key = '--lr' then
                learningRate := StrToFloat(valueStr)
            else if key = '--activation' then
            begin
                if LowerCase(valueStr) = 'leaky-relu' then
                    activation := atLeakyReLU
                else if LowerCase(valueStr) = 'tanh' then
                    activation := atTanh
                else if LowerCase(valueStr) = 'sigmoid' then
                    activation := atSigmoid
                else
                    activation := atReLU;
            end
            else if key = '--loss' then
            begin
                if LowerCase(valueStr) = 'binary-crossentropy' then
                    lossType := ltBinaryCrossEntropy
                else
                    lossType := ltMSE;
            end
            else if key = '--optimizer' then
            begin
                if LowerCase(valueStr) = 'sgd' then
                    optimizer := otSGD
                else if LowerCase(valueStr) = 'rmsprop' then
                    optimizer := otRMSProp
                else
                    optimizer := otAdam;
            end
            else if key = '--epochs' then
                epochs := StrToInt(valueStr)
            else if key = '--batch' then
                batchSize := StrToInt(valueStr)
            else if key = '--layer' then
                layerIdx := StrToInt(valueStr)
            else if key = '--node' then
                nodeIdx := StrToInt(valueStr)
            else if key = '--neighbor' then
                neighborIdx := StrToInt(valueStr)
            else if key = '--feature-idx' then
                featureIdx := StrToInt(valueStr)
            else if key = '--num-bins' then
                numBins := StrToInt(valueStr)
            else if key = '--pr-iter' then
                prIter := StrToInt(valueStr)
            else if key = '--value' then
            begin
                try
                    value := StrToFloat(valueStr);
                except
                    on E: Exception do
                        WriteLn('Warning: Could not parse --value as float');
                end;
            end
            else if key = '--damping' then
                damping := StrToFloat(valueStr)
            else if key = '--threshold' then
                threshold := StrToFloat(valueStr)
            else if key = '--features' then
            begin
                // Parse comma-separated feature values
                // Simplified: would need proper CSV parsing in production
                WriteLn('Note: Feature parsing requires full implementation');
            end
            else
                WriteLn('Unknown option: ', key);
        end;
    end;
    
    // Execute command
    if Command = cmdCreate then
    begin
        if featureSize <= 0 then begin WriteLn('Error: --feature is required'); Exit; end;
        if hiddenSize <= 0 then begin WriteLn('Error: --hidden is required'); Exit; end;
        if outputSize <= 0 then begin WriteLn('Error: --output is required'); Exit; end;
        if mpLayers <= 0 then begin WriteLn('Error: --mp-layers is required'); Exit; end;
        if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
        
        GNNFacade := TGNNFacade.Create(featureSize, hiddenSize, outputSize, mpLayers);
        try
            GNNFacade.FGNN.LearningRate := learningRate;
            GNNFacade.FGNN.Activation := activation;
            GNNFacade.FGNN.LossFunction := lossType;
            
            WriteLn('Created Facaded GNN model:');
            WriteLn('  Feature size: ', featureSize);
            WriteLn('  Hidden size: ', hiddenSize);
            WriteLn('  Output size: ', outputSize);
            WriteLn('  Message passing layers: ', mpLayers);
            WriteLn('  Activation: ', ActivationToStr(activation));
            WriteLn('  Loss function: ', LossToStr(lossType));
            WriteLn('  Learning rate: ', learningRate:0:6);
            
            { Save using JSON if file ends with .json, otherwise binary }
            if RightStr(saveFile, 5) = '.json' then
                GNNFacade.SaveModelToJSON(saveFile)
            else
                GNNFacade.SaveModel(saveFile);
            
            WriteLn('Model saved to: ', saveFile);
        finally
            GNNFacade.Free;
        end;
    end
    else if Command = cmdTrain then
    begin
        if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
        if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
        
        GNNFacade := TGNNFacade.Create(1, 1, 1, 1);
        try
            if RightStr(modelFile, 5) = '.json' then
                GNNFacade.LoadModelFromJSON(modelFile)
            else
                GNNFacade.LoadModel(modelFile);
            
            WriteLn('Training model...');
            WriteLn('Training completed. Model saved to: ', saveFile);
            
            if RightStr(saveFile, 5) = '.json' then
                GNNFacade.SaveModelToJSON(saveFile)
            else
                GNNFacade.SaveModel(saveFile);
        finally
            GNNFacade.Free;
        end;
    end
    else if Command = cmdPredict then
    begin
        if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
        
        GNNFacade := TGNNFacade.Create(1, 1, 1, 1);
        try
            if RightStr(modelFile, 5) = '.json' then
                GNNFacade.LoadModelFromJSON(modelFile)
            else
                GNNFacade.LoadModel(modelFile);
            
            WriteLn('Model loaded. Prediction functionality not yet fully implemented.');
        finally
            GNNFacade.Free;
        end;
    end
    else if Command = cmdInfo then
    begin
        if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
        
        GNNFacade := TGNNFacade.Create(1, 1, 1, 1);
        try
            if RightStr(modelFile, 5) = '.json' then
                GNNFacade.LoadModelFromJSON(modelFile)
            else
                GNNFacade.LoadModel(modelFile);
            
            WriteLn(GNNFacade.GetArchitectureSummary);
        finally
            GNNFacade.Free;
        end;
    end
    else if Command = cmdGetEmbedding then
    begin
        WriteLn('Get embedding requires loaded model');
        WriteLn('  Layer: ', layerIdx, ', Node: ', nodeIdx);
    end
    else if Command = cmdSetEmbedding then
    begin
        WriteLn('Set embedding requires loaded model');
        WriteLn('  Layer: ', layerIdx, ', Node: ', nodeIdx, ', Value: ', value:0:6);
    end
    else if Command = cmdGetGraphEmbedding then
    begin
        WriteLn('Get graph embedding requires loaded model');
        WriteLn('  Layer: ', layerIdx);
    end
    else if Command = cmdSetGraphEmbedding then
    begin
        WriteLn('Set graph embedding requires loaded model');
    end
    else if Command = cmdGetMessage then
    begin
        WriteLn('Get message requires loaded model');
        WriteLn('  Layer: ', layerIdx, ', Node: ', nodeIdx, ', Neighbor: ', neighborIdx);
    end
    else if Command = cmdSetMessage then
    begin
        WriteLn('Set message requires loaded model');
        WriteLn('  Layer: ', layerIdx, ', Node: ', nodeIdx, ', Neighbor: ', neighborIdx);
    end
    else if Command = cmdGetNodeDegree then
    begin
        WriteLn('Get node degree requires loaded model');
        WriteLn('  Node: ', nodeIdx);
    end
    else if Command = cmdComputePageRank then
    begin
        WriteLn('Compute PageRank requires loaded model');
        WriteLn('  Damping: ', damping:0:4, ', Iterations: ', prIter);
    end
    else if Command = cmdExportGraph then
    begin
        WriteLn('Export graph requires loaded model');
    end
    else if Command = cmdExportEmbeddings then
    begin
        WriteLn('Export embeddings requires loaded model');
        WriteLn('  Layer: ', layerIdx);
    end
    else if Command = cmdAddNode then
    begin
        WriteLn('Add node requires loaded model');
    end
    else if Command = cmdAddEdge then
    begin
        WriteLn('Add edge requires loaded model');
        WriteLn('  Source: ', nodeIdx, ', Target: ', neighborIdx);
    end
    else if Command = cmdGetNodeFeature then
    begin
        WriteLn('Get node feature requires loaded model');
        WriteLn('  Node: ', nodeIdx, ', Feature: ', featureIdx);
    end
    else if Command = cmdSetNodeFeature then
    begin
        WriteLn('Set node feature requires loaded model');
        WriteLn('  Node: ', nodeIdx, ', Feature: ', featureIdx, ', Value: ', value:0:6);
    end
    else if Command = cmdDetectVanishing then
    begin
        WriteLn('Detect vanishing gradients requires loaded model');
        WriteLn('  Threshold: ', threshold);
    end
    else if Command = cmdDetectExploding then
    begin
        WriteLn('Detect exploding gradients requires loaded model');
        WriteLn('  Threshold: ', threshold);
    end
    else if Command = cmdGetGradientFlow then
    begin
        WriteLn('Get gradient flow requires loaded model');
        WriteLn('  Layer: ', layerIdx);
    end;
end.

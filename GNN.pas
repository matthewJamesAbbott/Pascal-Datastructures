//
// Matthew Abbott
// Graph Neural Network - CLI Support
//

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

program GNNtest;

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
   TCommand = (cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp);
   
   TDoubleArray = array of Double;
   TIntArray = array of Integer;
   TDouble2DArray = array of TDoubleArray;
   
   TEdge = record
      Source: Integer;
      Target: Integer;
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
   
   TNeuron = record
      Weights: TDoubleArray;
      Bias: Double;
      Output: Double;
      PreActivation: Double;
      Error: Double;
   end;
   
   TLayer = record
      Neurons: array of TNeuron;
      NumInputs: Integer;
      NumOutputs: Integer;
      LastInput: TDoubleArray;
   end;
   
   TMessageInfo = record
      NeighborIdx: Integer;
      ConcatInput: TDoubleArray;
      MessageOutput: TDoubleArray;
   end;
   
   TNodeMessages = array of TMessageInfo;
   TLayerMessages = array of TNodeMessages;
   
   TTrainingMetrics = record
      Loss: Double;
      Iteration: Integer;
      LossHistory: TDoubleArray;
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
      
      FMessageLayers: array of TLayer;
      FUpdateLayers: array of TLayer;
      FReadoutLayer: TLayer;
      FOutputLayer: TLayer;
      
      FNodeEmbeddings: TDouble2DArray;
      FNewNodeEmbeddings: TDouble2DArray;
      FEmbeddingHistory: array of TDouble2DArray;
      FMessageHistory: array of TLayerMessages;
      FAggregatedMessages: array of TDouble2DArray;
      FGraphEmbedding: TDoubleArray;
      
      FMetrics: TTrainingMetrics;
      
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
      function ComputeLoss(const Prediction, Target: TDoubleArray): Double;
      function ComputeLossGradient(const Prediction, Target: TDoubleArray): TDoubleArray;
      function ClipGradient(G: Double): Double;
      
   public
      constructor Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
      destructor Destroy; override;
      
      function Predict(var Graph: TGraph): TDoubleArray;
      function Train(var Graph: TGraph; const Target: TDoubleArray): Double;
      procedure TrainMultiple(var Graph: TGraph; const Target: TDoubleArray; Iterations: Integer);
      
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
      function GetFeatureSize: Integer;
      function GetHiddenSize: Integer;
      function GetOutputSize: Integer;
      
      { JSON serialization methods }
      procedure SaveModelToJSON(const Filename: string);
      procedure LoadModelFromJSON(const Filename: string);
      
      { JSON serialization helper functions }
      function Array1DToJSON(const Arr: TDoubleArray): string;
      function Array2DToJSON(const Arr: TDouble2DArray): string;
      end;

// ==================== Forward Declarations ====================

function ActivationToStr(act: TActivationType): string; forward;
function LossToStr(loss: TLossType): string; forward;
function ParseActivation(const s: string): TActivationType; forward;
function ParseLoss(const s: string): TLossType; forward;

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
end;

destructor TGraphNeuralNetwork.Destroy;
begin
   inherited;
end;

function TGraphNeuralNetwork.GetFeatureSize: Integer;
begin
   Result := FFeatureSize;
end;

function TGraphNeuralNetwork.GetHiddenSize: Integer;
begin
   Result := FHiddenSize;
end;

function TGraphNeuralNetwork.GetOutputSize: Integer;
begin
   Result := FOutputSize;
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
      for J := 0 to NumInputs - 1 do
         Layer.Neurons[I].Weights[J] := (Random - 0.5) * 2.0 * Scale;
      Layer.Neurons[I].Bias := 0.0;
      Layer.Neurons[I].Output := 0.0;
      Layer.Neurons[I].PreActivation := 0.0;
      Layer.Neurons[I].Error := 0.0;
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
begin
   OrigLen := Length(Graph.Edges);
   SetLength(Graph.Edges, OrigLen * 2);
   
   for I := 0 to OrigLen - 1 do
   begin
      if Graph.Edges[I].Source <> Graph.Edges[I].Target then
      begin
         RevEdge.Source := Graph.Edges[I].Target;
         RevEdge.Target := Graph.Edges[I].Source;
         Graph.Edges[OrigLen + I] := RevEdge;
      end
      else
         Graph.Edges[OrigLen + I] := Graph.Edges[I];
   end;
   
   DeduplicateEdges(Graph);
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
begin
   Layer.LastInput := CopyArray(Input);
   SetLength(Result, Layer.NumOutputs);
   
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      Sum := Layer.Neurons[I].Bias;
      for J := 0 to Layer.NumInputs - 1 do
      begin
         if J <= High(Input) then
            Sum := Sum + Layer.Neurons[I].Weights[J] * Input[J];
      end;
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
   PreActGrad, DeltaW: Double;
begin
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      if UseOutputActivation then
         PreActGrad := UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation)
      else
         PreActGrad := UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);
      
      Layer.Neurons[I].Error := PreActGrad;
      
      for J := 0 to Layer.NumInputs - 1 do
      begin
         if J <= High(Layer.LastInput) then
         begin
            DeltaW := FLearningRate * PreActGrad * Layer.LastInput[J];
            Layer.Neurons[I].Weights[J] := Layer.Neurons[I].Weights[J] - DeltaW;
         end;
      end;
      
      Layer.Neurons[I].Bias := Layer.Neurons[I].Bias - (FLearningRate * PreActGrad);
   end;
end;

function TGraphNeuralNetwork.GetLayerInputGrad(const Layer: TLayer; const UpstreamGrad: TDoubleArray; 
                                               UseOutputActivation: Boolean = False): TDoubleArray;
var
   I, J: Integer;
   PreActGrad: Double;
begin
   SetLength(Result, Layer.NumInputs);
   for I := 0 to Layer.NumInputs - 1 do
      Result[I] := 0.0;
   
   for I := 0 to Layer.NumOutputs - 1 do
   begin
      if UseOutputActivation then
         PreActGrad := UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation)
      else
         PreActGrad := UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);
      
      for J := 0 to Layer.NumInputs - 1 do
      begin
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

      function TGraphNeuralNetwork.Array1DToJSON(const Arr: TDoubleArray): string;
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

      function TGraphNeuralNetwork.Array2DToJSON(const Arr: TDouble2DArray): string;
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

      procedure TGraphNeuralNetwork.SaveModelToJSON(const Filename: string);
      var
      SL: TStringList;
      I, J, K: Integer;
      begin
      SL := TStringList.Create;
      try
      SL.Add('{');
      SL.Add('  "feature_size": ' + IntToStr(FFeatureSize) + ',');
      SL.Add('  "hidden_size": ' + IntToStr(FHiddenSize) + ',');
      SL.Add('  "output_size": ' + IntToStr(FOutputSize) + ',');
      SL.Add('  "num_message_passing_layers": ' + IntToStr(FNumMessagePassingLayers) + ',');
      SL.Add('  "learning_rate": ' + FloatToStr(FLearningRate) + ',');
      SL.Add('  "activation": "' + ActivationToStr(FActivation) + '",');
      SL.Add('  "loss_type": "' + LossToStr(FLossType) + '",');
      SL.Add('  "max_iterations": ' + IntToStr(FMaxIterations) + ',');
      SL.Add('  "message_layers": [');
      
      for K := 0 to FNumMessagePassingLayers - 1 do
      begin
         if K > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
         SL.Add('    {');
         SL.Add('      "num_outputs": ' + IntToStr(FMessageLayers[K].NumOutputs) + ',');
         SL.Add('      "num_inputs": ' + IntToStr(FMessageLayers[K].NumInputs) + ',');
         SL.Add('      "neurons": [');
         
         for I := 0 to FMessageLayers[K].NumOutputs - 1 do
         begin
            if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('        {');
            SL.Add('          "weights": ' + Array1DToJSON(FMessageLayers[K].Neurons[I].Weights) + ',');
            SL.Add('          "bias": ' + FloatToStr(FMessageLayers[K].Neurons[I].Bias));
            SL.Add('        }');
         end;
         
         SL.Add('      ]');
         SL.Add('    }');
      end;
      
      SL.Add('  ],');
      SL.Add('  "update_layers": [');
      
      for K := 0 to FNumMessagePassingLayers - 1 do
      begin
         if K > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
         SL.Add('    {');
         SL.Add('      "num_outputs": ' + IntToStr(FUpdateLayers[K].NumOutputs) + ',');
         SL.Add('      "num_inputs": ' + IntToStr(FUpdateLayers[K].NumInputs) + ',');
         SL.Add('      "neurons": [');
         
         for I := 0 to FUpdateLayers[K].NumOutputs - 1 do
         begin
            if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
            SL.Add('        {');
            SL.Add('          "weights": ' + Array1DToJSON(FUpdateLayers[K].Neurons[I].Weights) + ',');
            SL.Add('          "bias": ' + FloatToStr(FUpdateLayers[K].Neurons[I].Bias));
            SL.Add('        }');
         end;
         
         SL.Add('      ]');
         SL.Add('    }');
      end;
      
      SL.Add('  ],');
      SL.Add('  "readout_layer": {');
      SL.Add('    "num_outputs": ' + IntToStr(FReadoutLayer.NumOutputs) + ',');
      SL.Add('    "num_inputs": ' + IntToStr(FReadoutLayer.NumInputs) + ',');
      SL.Add('    "neurons": [');
      
      for I := 0 to FReadoutLayer.NumOutputs - 1 do
      begin
         if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
         SL.Add('      {');
         SL.Add('        "weights": ' + Array1DToJSON(FReadoutLayer.Neurons[I].Weights) + ',');
         SL.Add('        "bias": ' + FloatToStr(FReadoutLayer.Neurons[I].Bias));
         SL.Add('      }');
      end;
      
      SL.Add('    ]');
      SL.Add('  },');
      SL.Add('  "output_layer": {');
      SL.Add('    "num_outputs": ' + IntToStr(FOutputLayer.NumOutputs) + ',');
      SL.Add('    "num_inputs": ' + IntToStr(FOutputLayer.NumInputs) + ',');
      SL.Add('    "neurons": [');
      
      for I := 0 to FOutputLayer.NumOutputs - 1 do
      begin
         if I > 0 then SL[SL.Count - 1] := SL[SL.Count - 1] + ',';
         SL.Add('      {');
         SL.Add('        "weights": ' + Array1DToJSON(FOutputLayer.Neurons[I].Weights) + ',');
         SL.Add('        "bias": ' + FloatToStr(FOutputLayer.Neurons[I].Bias));
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

      procedure TGraphNeuralNetwork.LoadModelFromJSON(const Filename: string);
      var
      SL: TStringList;
      Content: string;
      JSONPos, StartPos, EndPos: Integer;
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
      
      begin
      SL := TStringList.Create;
      try
      SL.LoadFromFile(Filename);
      Content := SL.Text;
      
      ValueStr := ExtractJSONValue(Content, 'feature_size');
      if ValueStr <> '' then FFeatureSize := StrToInt(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'hidden_size');
      if ValueStr <> '' then FHiddenSize := StrToInt(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'output_size');
      if ValueStr <> '' then FOutputSize := StrToInt(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'num_message_passing_layers');
      if ValueStr <> '' then FNumMessagePassingLayers := StrToInt(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'learning_rate');
      if ValueStr <> '' then FLearningRate := StrToFloat(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'activation');
      if ValueStr <> '' then FActivation := ParseActivation(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'loss_type');
      if ValueStr <> '' then FLossType := ParseLoss(ValueStr);
      
      ValueStr := ExtractJSONValue(Content, 'max_iterations');
      if ValueStr <> '' then FMaxIterations := StrToInt(ValueStr);
      
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

procedure PrintUsage;
begin
    WriteLn('GNN');
    WriteLn;
    WriteLn('Commands:');
    WriteLn('  create   Create a new GNN model and save to JSON');
    WriteLn('  train    Train an existing model with graph data from JSON');
    WriteLn('  predict  Make predictions with a trained model from JSON');
    WriteLn('  info     Display model information from JSON');
    WriteLn('  help     Show this help message');
    WriteLn;
    WriteLn('Create Options:');
    WriteLn('  --feature=N            Input feature size (required)');
    WriteLn('  --hidden=N             Hidden layer size (required)');
    WriteLn('  --output=N             Output size (required)');
    WriteLn('  --mp-layers=N          Message passing layers (required)');
    WriteLn('  --save=FILE.json       Save model to JSON file (required)');
    WriteLn('  --lr=VALUE             Learning rate (default: 0.01)');
    WriteLn('  --activation=TYPE      relu|leakyrelu|tanh|sigmoid (default: relu)');
    WriteLn('  --loss=TYPE            mse|bce (default: mse)');
    WriteLn;
    WriteLn('Train Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn('  --graph=FILE.json      Graph file (JSON format) (required)');
    WriteLn('  --save=FILE.json       Save trained model to JSON (required)');
    WriteLn('  --epochs=N             Number of training epochs (default: 100)');
    WriteLn('  --lr=VALUE             Override learning rate');
    WriteLn;
    WriteLn('Predict Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn('  --graph=FILE.json      Graph file (required)');
    WriteLn;
    WriteLn('Info Options:');
    WriteLn('  --model=FILE.json      Load model from JSON file (required)');
    WriteLn;
    WriteLn('Examples:');
    WriteLn('  gnn create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=model.json');
    WriteLn('  gnn train --model=model.json --graph=data.json --epochs=500 --save=trained.json');
    WriteLn('  gnn predict --model=trained.json --graph=data.json');
    WriteLn('  gnn info --model=trained.json');
 end;

procedure ParseIntArrayHelper(const s: string; out result: TIntArray);
var
   tokens: TStringList;
   i: Integer;
   temp: TIntArray;
begin
   tokens := TStringList.Create;
   try
      tokens.Delimiter := ',';
      tokens.DelimitedText := s;
      SetLength(temp, tokens.Count);
      for i := 0 to tokens.Count - 1 do
         temp[i] := StrToInt(Trim(tokens[i]));
      result := temp;
   finally
      tokens.Free;
   end;
end;

procedure ParseDoubleArrayHelper(const s: string; out result: TDoubleArray);
var
    tokens: TStringList;
    i: Integer;
    temp: TDoubleArray;
begin
    tokens := TStringList.Create;
    try
       tokens.Delimiter := ',';
       tokens.DelimitedText := s;
       SetLength(temp, tokens.Count);
       for i := 0 to tokens.Count - 1 do
          temp[i] := StrToFloat(Trim(tokens[i]));
       result := temp;
    finally
       tokens.Free;
    end;
 end;

var
   Command: TCommand;
   CmdStr: string;
   i: Integer;
   arg, key, value: string;
   eqPos: Integer;
   
   featureSize, hiddenSize, outputSize, mpLayers, epochs: Integer;
   learningRate: Double;
   activation: TActivationType;
   loss: TLossType;
   modelFile, saveFile, graphFile: string;
   verbose: Boolean;
   
   GNN: TGraphNeuralNetwork;
   Graph: TGraph;
   target: TDoubleArray;
   j: Integer;
   prediction: TDoubleArray;

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
   featureSize := 0;
   hiddenSize := 0;
   outputSize := 0;
   mpLayers := 0;
   learningRate := 0.01;
   epochs := 100;
   verbose := False;
   activation := atReLU;
   loss := ltMSE;
   modelFile := '';
   saveFile := '';
   graphFile := '';
   
   // Parse arguments
   for i := 2 to ParamCount do
   begin
      arg := ParamStr(i);
      
      if arg = '--verbose' then
         verbose := True
      else
      begin
         eqPos := Pos('=', arg);
         if eqPos = 0 then
         begin
            WriteLn('Invalid argument: ', arg);
            Continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--feature' then
            featureSize := StrToInt(value)
         else if key = '--hidden' then
            hiddenSize := StrToInt(value)
         else if key = '--output' then
            outputSize := StrToInt(value)
         else if key = '--mp-layers' then
            mpLayers := StrToInt(value)
         else if key = '--save' then
            saveFile := value
         else if key = '--model' then
            modelFile := value
         else if key = '--graph' then
            graphFile := value
         else if key = '--lr' then
            learningRate := StrToFloat(value)
         else if key = '--activation' then
            activation := ParseActivation(value)
         else if key = '--loss' then
            loss := ParseLoss(value)
         else if key = '--epochs' then
            epochs := StrToInt(value)
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
      
      GNN := TGraphNeuralNetwork.Create(featureSize, hiddenSize, outputSize, mpLayers);
      GNN.LearningRate := learningRate;
      GNN.Activation := activation;
      GNN.LossFunction := loss;
      
      WriteLn('Created GNN model:');
      WriteLn('  Feature size: ', featureSize);
      WriteLn('  Hidden size: ', hiddenSize);
      WriteLn('  Output size: ', outputSize);
      WriteLn('  Message passing layers: ', mpLayers);
      WriteLn('  Activation: ', ActivationToStr(activation));
      WriteLn('  Loss function: ', LossToStr(loss));
      WriteLn('  Learning rate: ', learningRate:0:6);
      
      { Save model to JSON }
      GNN.SaveModelToJSON(saveFile);

      GNN.Free;
   end
   else if Command = cmdTrain then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      if graphFile = '' then begin WriteLn('Error: --graph is required'); Exit; end;
      if saveFile = '' then begin WriteLn('Error: --save is required'); Exit; end;
      
      WriteLn('Loading model from JSON: ' + modelFile);
      GNN := TGraphNeuralNetwork.Create(1, 1, 1, 1);
      GNN.LoadModelFromJSON(modelFile);
      WriteLn('Model loaded successfully. Training functionality not yet implemented.');
      GNN.Free;
   end
   else if Command = cmdPredict then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      if graphFile = '' then begin WriteLn('Error: --graph is required'); Exit; end;
      
      WriteLn('Loading model from JSON: ' + modelFile);
      GNN := TGraphNeuralNetwork.Create(1, 1, 1, 1);
      GNN.LoadModelFromJSON(modelFile);
      WriteLn('Model loaded successfully. Prediction functionality not yet implemented.');
      GNN.Free;
   end
   else if Command = cmdInfo then
   begin
      if modelFile = '' then begin WriteLn('Error: --model is required'); Exit; end;
      
      WriteLn('Loading model from JSON: ' + modelFile);
      GNN := TGraphNeuralNetwork.Create(1, 1, 1, 1);
      GNN.LoadModelFromJSON(modelFile);
      WriteLn('Model information displayed above.');
      GNN.Free;
   end;
end.


//
// Graph Neural Network Implementation
// Based on Message Passing Neural Network architecture
//

{$mode objfpc}
{$M+}

program GNNtest;

uses Classes, Math, SysUtils;

type
   Darray = array of Double;
   Iarray = array of Integer;
   
   TEdge = record
      Source: Integer;
      Target: Integer;
   end;
   
   TGraph = record
      NumNodes: Integer;
      NodeFeatures: array of Darray;  // Node feature vectors
      Edges: array of TEdge;
      AdjacencyList: array of Iarray; // For efficient neighbor lookup
   end;
   
   TGraphDataPoint = record
      Graph: TGraph;
      Target: Darray;  // Graph-level or node-level targets
   end;

   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
   end;

   TLayer = record
      Neurons: array of TNeuron;
   end;

   TGraphNeuralNetwork = class
   private
      LearningRate: Double;
      MaxIterations: Integer;
      NumMessagePassingLayers: Integer;
      FeatureSize: Integer; 
      HiddenSize: Integer;
      OutputSize: Integer;
      // Message passing layers
      MessageLayers: array of TLayer;
      UpdateLayers: array of TLayer;
      
      // Readout/aggregation layer
      ReadoutLayer: TLayer;
      OutputLayer: TLayer;
      
      // Temporary storage for node embeddings
      NodeEmbeddings: array of Darray;
      NewNodeEmbeddings: array of Darray;
      
      procedure InitializeLayer(var Layer: TLayer; NumNeurons: Integer; NumInputs: Integer);
      procedure BuildAdjacencyList(var Graph: TGraph);
      procedure MessagePassing(var Graph: TGraph);
      procedure Readout(var Graph: TGraph; var GraphEmbedding: Darray);
      procedure BackPropagateGraph(var Graph: TGraph; Target: Darray);
      procedure UpdateWeights;
      function Sigmoid(x: Double): Double;
      function ReLU(x: Double): Double;
      
   public
      constructor Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
      function Predict(var Graph: TGraph): Darray;
      procedure Train(var Graph: TGraph; Target: Darray);
      procedure SaveGNNModel(const Filename: string);
      destructor Destroy; override;
   end;

constructor TGraphNeuralNetwork.Create(AFeatureSize, AHiddenSize, AOutputSize, NumMPLayers: Integer);
var
   i: Integer;
begin
   LearningRate := 0.01;
   Self.FeatureSize := AFeatureSize;
   Self.HiddenSize := AHiddenSize;
   Self.OutputSize := AOutputSize;
   Self.NumMessagePassingLayers := NumMPLayers;
   
   // Initialize message passing layers
   SetLength(MessageLayers, NumMPLayers);
   SetLength(UpdateLayers, NumMPLayers);
   
   for i := 0 to NumMPLayers - 1 do
   begin
      if i = 0 then
         InitializeLayer(MessageLayers[i], HiddenSize, FeatureSize * 2)  // Concatenate node + neighbor
      else
         InitializeLayer(MessageLayers[i], HiddenSize, HiddenSize * 2);
         
      InitializeLayer(UpdateLayers[i], HiddenSize, HiddenSize * 2);  // Current + aggregated message
   end;
   
   // Initialize readout and output layers
   InitializeLayer(ReadoutLayer, HiddenSize, HiddenSize);
   InitializeLayer(OutputLayer, OutputSize, HiddenSize);
end;

procedure TGraphNeuralNetwork.InitializeLayer(var Layer: TLayer; NumNeurons, NumInputs: Integer);
var
   i, j: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   for i := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[i].Weights, NumInputs);
      for j := 0 to NumInputs - 1 do
         Layer.Neurons[i].Weights[j] := (Random - 0.5) * 0.1;  // Xavier-like initialization
      Layer.Neurons[i].Bias := 0.0;
   end;
end;

function TGraphNeuralNetwork.Sigmoid(x: Double): Double;
begin
   Result := 1.0 / (1.0 + Exp(-x));
end;

function TGraphNeuralNetwork.ReLU(x: Double): Double;
begin
   if x > 0 then
      Result := x
   else
      Result := 0.0;
end;

procedure TGraphNeuralNetwork.BuildAdjacencyList(var Graph: TGraph);
var
   i, src, tgt: Integer;
begin
   SetLength(Graph.AdjacencyList, Graph.NumNodes);
   
   // Initialize empty lists
   for i := 0 to Graph.NumNodes - 1 do
      SetLength(Graph.AdjacencyList[i], 0);
   
   // Build adjacency list from edges
   for i := 0 to High(Graph.Edges) do
   begin
      src := Graph.Edges[i].Source;
      tgt := Graph.Edges[i].Target;
      
      SetLength(Graph.AdjacencyList[src], Length(Graph.AdjacencyList[src]) + 1);
      Graph.AdjacencyList[src][High(Graph.AdjacencyList[src])] := tgt;
   end;
end;

procedure TGraphNeuralNetwork.MessagePassing(var Graph: TGraph);
var
   layer, node, neighbor, i, j, k: Integer;
   ConcatFeatures: Darray;
   Message, AggregatedMessage, UpdateInput: Darray;
   Sum: Double;
begin
   // Initialize node embeddings with input features
   SetLength(NodeEmbeddings, Graph.NumNodes);
   SetLength(NewNodeEmbeddings, Graph.NumNodes);
   
   for i := 0 to Graph.NumNodes - 1 do
   begin
      SetLength(NodeEmbeddings[i], FeatureSize);
      for j := 0 to FeatureSize - 1 do
         NodeEmbeddings[i][j] := Graph.NodeFeatures[i][j];
   end;
   
   // Perform message passing for each layer
   for layer := 0 to NumMessagePassingLayers - 1 do
   begin
      // For each node, aggregate messages from neighbors
      for node := 0 to Graph.NumNodes - 1 do
      begin
         SetLength(AggregatedMessage, HiddenSize);
         for i := 0 to HiddenSize - 1 do
            AggregatedMessage[i] := 0.0;
         
         // Aggregate messages from neighbors
         for k := 0 to High(Graph.AdjacencyList[node]) do
         begin
            neighbor := Graph.AdjacencyList[node][k];
            
            // Concatenate node and neighbor features
            SetLength(ConcatFeatures, Length(NodeEmbeddings[node]) + Length(NodeEmbeddings[neighbor]));
            for i := 0 to High(NodeEmbeddings[node]) do
               ConcatFeatures[i] := NodeEmbeddings[node][i];
            for i := 0 to High(NodeEmbeddings[neighbor]) do
               ConcatFeatures[Length(NodeEmbeddings[node]) + i] := NodeEmbeddings[neighbor][i];
            
            // Compute message using message layer
            SetLength(Message, HiddenSize);
            for i := 0 to HiddenSize - 1 do
            begin
               Sum := MessageLayers[layer].Neurons[i].Bias;
               for j := 0 to High(ConcatFeatures) do
                  Sum := Sum + ConcatFeatures[j] * MessageLayers[layer].Neurons[i].Weights[j];
               Message[i] := ReLU(Sum);
            end;
            
            // Aggregate (sum)
            for i := 0 to HiddenSize - 1 do
               AggregatedMessage[i] := AggregatedMessage[i] + Message[i];
         end;
         
         // Average the aggregated messages
         if Length(Graph.AdjacencyList[node]) > 0 then
         begin
            for i := 0 to HiddenSize - 1 do
               AggregatedMessage[i] := AggregatedMessage[i] / Length(Graph.AdjacencyList[node]);
         end;
         
         // Update node embedding
         SetLength(UpdateInput, Length(NodeEmbeddings[node]) + HiddenSize);
         for i := 0 to High(NodeEmbeddings[node]) do
            UpdateInput[i] := NodeEmbeddings[node][i];
         for i := 0 to HiddenSize - 1 do
            UpdateInput[Length(NodeEmbeddings[node]) + i] := AggregatedMessage[i];
         
         SetLength(NewNodeEmbeddings[node], HiddenSize);
         for i := 0 to HiddenSize - 1 do
         begin
            Sum := UpdateLayers[layer].Neurons[i].Bias;
            for j := 0 to High(UpdateInput) do
               Sum := Sum + UpdateInput[j] * UpdateLayers[layer].Neurons[i].Weights[j];
            NewNodeEmbeddings[node][i] := ReLU(Sum);
         end;
      end;
      
      // Copy new embeddings to current embeddings
      for node := 0 to Graph.NumNodes - 1 do
      begin
         SetLength(NodeEmbeddings[node], HiddenSize);
         for i := 0 to HiddenSize - 1 do
            NodeEmbeddings[node][i] := NewNodeEmbeddings[node][i];
      end;
   end;
end;

procedure TGraphNeuralNetwork.Readout(var Graph: TGraph; var GraphEmbedding: Darray);
var
   i, j: Integer;
   Sum, NodeSum: Double;
begin
   // Simple global mean pooling for readout
   SetLength(GraphEmbedding, HiddenSize);
   for i := 0 to HiddenSize - 1 do
      GraphEmbedding[i] := 0.0;
   
   // Sum all node embeddings
   for i := 0 to Graph.NumNodes - 1 do
   begin
      for j := 0 to HiddenSize - 1 do
         GraphEmbedding[j] := GraphEmbedding[j] + NodeEmbeddings[i][j];
   end;
   
   // Average
   for i := 0 to HiddenSize - 1 do
      GraphEmbedding[i] := GraphEmbedding[i] / Graph.NumNodes;
   
   // Pass through readout layer
   for i := 0 to HiddenSize - 1 do
   begin
      Sum := ReadoutLayer.Neurons[i].Bias;
      for j := 0 to HiddenSize - 1 do
         Sum := Sum + GraphEmbedding[j] * ReadoutLayer.Neurons[i].Weights[j];
      ReadoutLayer.Neurons[i].Output := ReLU(Sum);
   end;
end;

function TGraphNeuralNetwork.Predict(var Graph: TGraph): Darray;
var
   GraphEmbedding: Darray;
   i, j: Integer;
   Sum: Double;
begin
   BuildAdjacencyList(Graph);
   MessagePassing(Graph);
   Readout(Graph, GraphEmbedding);
   
   // Output layer
   SetLength(Result, OutputSize);
   for i := 0 to OutputSize - 1 do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to HiddenSize - 1 do
         Sum := Sum + ReadoutLayer.Neurons[j].Output * OutputLayer.Neurons[i].Weights[j];
      Result[i] := Sigmoid(Sum);
      OutputLayer.Neurons[i].Output := Result[i];
   end;
end;

procedure TGraphNeuralNetwork.Train(var Graph: TGraph; Target: Darray);
var
   Prediction: Darray;
begin
   Prediction := Predict(Graph);
   BackPropagateGraph(Graph, Target);
   UpdateWeights;
end;

procedure TGraphNeuralNetwork.BackPropagateGraph(var Graph: TGraph; Target: Darray);
var
   i, j: Integer;
begin
   // Calculate output layer errors (simplified - full backprop through graph omitted for brevity)
   for i := 0 to OutputSize - 1 do
      OutputLayer.Neurons[i].Error := OutputLayer.Neurons[i].Output * 
         (1 - OutputLayer.Neurons[i].Output) * (Target[i] - OutputLayer.Neurons[i].Output);
   
   // Backpropagate to readout layer
   for i := 0 to HiddenSize - 1 do
   begin
      ReadoutLayer.Neurons[i].Error := 0.0;
      for j := 0 to OutputSize - 1 do
         ReadoutLayer.Neurons[i].Error := ReadoutLayer.Neurons[i].Error + 
            OutputLayer.Neurons[j].Error * OutputLayer.Neurons[j].Weights[i];
      
      if ReadoutLayer.Neurons[i].Output > 0 then  // ReLU derivative
         ReadoutLayer.Neurons[i].Error := ReadoutLayer.Neurons[i].Error * 1.0;
   end;
end;

procedure TGraphNeuralNetwork.UpdateWeights;
var
   i, j: Integer;
begin
   // Update output layer weights
   for i := 0 to OutputSize - 1 do
   begin
      for j := 0 to HiddenSize - 1 do
         OutputLayer.Neurons[i].Weights[j] := OutputLayer.Neurons[i].Weights[j] + 
            LearningRate * OutputLayer.Neurons[i].Error * ReadoutLayer.Neurons[j].Output;
      OutputLayer.Neurons[i].Bias := OutputLayer.Neurons[i].Bias + 
         LearningRate * OutputLayer.Neurons[i].Error;
   end;
   
   // Update readout layer weights (simplified)
   for i := 0 to HiddenSize - 1 do
   begin
      ReadoutLayer.Neurons[i].Bias := ReadoutLayer.Neurons[i].Bias + 
         LearningRate * ReadoutLayer.Neurons[i].Error;
   end;
end;

procedure TGraphNeuralNetwork.SaveGNNModel(const Filename: string);
var
   F: File;
   i, j, k, NumWeights: Integer;
begin
   AssignFile(F, Filename);
   Rewrite(F, 1);
   
   // Write hyperparameters
   BlockWrite(F, FeatureSize, SizeOf(Integer));
   BlockWrite(F, HiddenSize, SizeOf(Integer));
   BlockWrite(F, OutputSize, SizeOf(Integer));
   BlockWrite(F, NumMessagePassingLayers, SizeOf(Integer));
   BlockWrite(F, LearningRate, SizeOf(Double));
   
   // Write message and update layer weights
   for k := 0 to NumMessagePassingLayers - 1 do
   begin
      for i := 0 to High(MessageLayers[k].Neurons) do
      begin
         NumWeights := Length(MessageLayers[k].Neurons[i].Weights);
         BlockWrite(F, NumWeights, SizeOf(Integer));
         for j := 0 to High(MessageLayers[k].Neurons[i].Weights) do
            BlockWrite(F, MessageLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         BlockWrite(F, MessageLayers[k].Neurons[i].Bias, SizeOf(Double));
      end;
      
      for i := 0 to High(UpdateLayers[k].Neurons) do
      begin
         NumWeights := Length(UpdateLayers[k].Neurons[i].Weights);
         BlockWrite(F, NumWeights, SizeOf(Integer));
         for j := 0 to High(UpdateLayers[k].Neurons[i].Weights) do
            BlockWrite(F, UpdateLayers[k].Neurons[i].Weights[j], SizeOf(Double));
         BlockWrite(F, UpdateLayers[k].Neurons[i].Bias, SizeOf(Double));
      end;
   end;
   
   CloseFile(F);
   WriteLn('GNN model saved to ', Filename);
end;

destructor TGraphNeuralNetwork.Destroy;
begin
   inherited;
end;

// Example usage
var
   GNN: TGraphNeuralNetwork;
   Graph: TGraph;
   Prediction: Darray;
   i: Integer;
begin
   Randomize;
   
   // Create a simple graph
   Graph.NumNodes := 5;
   SetLength(Graph.NodeFeatures, 5);
   SetLength(Graph.Edges, 6);
   
   // Initialize node features (3 features per node)
   for i := 0 to 4 do
   begin
      SetLength(Graph.NodeFeatures[i], 3);
      Graph.NodeFeatures[i][0] := Random;
      Graph.NodeFeatures[i][1] := Random;
      Graph.NodeFeatures[i][2] := Random;
   end;
   
   // Define edges (creating a simple connected graph)
   Graph.Edges[0].Source := 0; Graph.Edges[0].Target := 1;
   Graph.Edges[1].Source := 1; Graph.Edges[1].Target := 2;
   Graph.Edges[2].Source := 2; Graph.Edges[2].Target := 3;
   Graph.Edges[3].Source := 3; Graph.Edges[3].Target := 4;
   Graph.Edges[4].Source := 4; Graph.Edges[4].Target := 0;
   Graph.Edges[5].Source := 1; Graph.Edges[5].Target := 3;
   
   // Create GNN: 3 input features, 16 hidden units, 2 output classes, 2 message passing layers
   GNN := TGraphNeuralNetwork.Create(3, 16, 2, 2);
   GNN.MaxIterations := 100;
   
   WriteLn('Training GNN on sample graph...');
   
   // Train on the graph
   for i := 0 to 99 do
      GNN.Train(Graph, [1.0, 0.0]);
   
   // Make prediction
   Prediction := GNN.Predict(Graph);
   WriteLn('Prediction: [', Prediction[0]:0:4, ', ', Prediction[1]:0:4, ']');
   
   // Save model
   GNN.SaveGNNModel('TestGNN.bin');
   
   GNN.Free;
   WriteLn('Done!');
end.

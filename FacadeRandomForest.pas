//
// Created by Matthew Abbott 2/3/2023
// Random Forest Facade - Interactive Dashboard & Experimental API
//

{$mode objfpc}
{$M+}

unit RandomForestFacade;

interface

uses
   RandomForest, Math, SysUtils;

const
   MAX_NODE_INFO = 1000;
   MAX_FEATURE_STATS = 100;
   MAX_SAMPLE_TRACK = 1000;

type
   { Node information for inspection }
   TNodeInfo = record
      nodeId: integer;
      depth: integer;
      isLeaf: boolean;
      featureIndex: integer;
      threshold: double;
      prediction: double;
      classLabel: integer;
      impurity: double;
      numSamples: integer;
      leftChildId: integer;
      rightChildId: integer;
   end;
   TNodeInfoArray = array[0..MAX_NODE_INFO-1] of TNodeInfo;

   { Tree structure overview }
   TTreeInfo = record
      treeId: integer;
      numNodes: integer;
      maxDepth: integer;
      numLeaves: integer;
      featuresUsed: array[0..MAX_FEATURES-1] of boolean;
      numFeaturesUsed: integer;
      oobError: double;
      nodes: TNodeInfoArray;
   end;

   { Feature usage statistics }
   TFeatureStats = record
      featureIndex: integer;
      timesUsed: integer;
      treesUsedIn: integer;
      avgImportance: double;
      totalImportance: double;
   end;
   TFeatureStatsArray = array[0..MAX_FEATURE_STATS-1] of TFeatureStats;

   { Sample tracking info }
   TSampleTrackInfo = record
      sampleIndex: integer;
      treesInfluenced: array[0..MAX_TREES-1] of boolean;
      numTreesInfluenced: integer;
      oobTrees: array[0..MAX_TREES-1] of boolean;
      numOobTrees: integer;
      predictions: array[0..MAX_TREES-1] of double;
   end;

   { OOB summary per tree }
   TOOBTreeInfo = record
      treeId: integer;
      numOobSamples: integer;
      oobError: double;
      oobAccuracy: double;
   end;
   TOOBTreeInfoArray = array[0..MAX_TREES-1] of TOOBTreeInfo;

   { Forest comparison result }
   TForestComparison = record
      numDifferentPredictions: integer;
      avgPredictionDiff: double;
      forestAAccuracy: double;
      forestBAccuracy: double;
      forestAMSE: double;
      forestBMSE: double;
   end;

   { Aggregation methods }
   TAggregationMethod = (MajorityVote, WeightedVote, Mean, WeightedMean);

   TRandomForestFacade = object

   private
      forest: TRandomForest;
      forestInitialized: boolean;
      currentAggregation: TAggregationMethod;
      treeWeights: array[0..MAX_TREES-1] of double;
      featureEnabled: array[0..MAX_FEATURES-1] of boolean;
      
      { Internal helpers }
      function collectNodeInfo(node: TreeNode; depth: integer; 
                               var nodes: TNodeInfoArray; var count: integer): integer;
      function calculateTreeDepth(node: TreeNode): integer;
      function countLeaves(node: TreeNode): integer;
      procedure collectFeaturesUsed(node: TreeNode; var used: array of boolean);
      function findNodeById(node: TreeNode; targetId: integer; var currentId: integer): TreeNode;
      procedure freeSubtree(node: TreeNode);

   public
      constructor create();
      
      { ===== Initialization ===== }
      procedure initForest();
      function getForest(): TRandomForest;
      
      { ===== Hyperparameter Control ===== }
      procedure setHyperparameter(paramName: string; value: integer);
      procedure setHyperparameterFloat(paramName: string; value: double);
      function getHyperparameter(paramName: string): integer;
      procedure setTaskType(t: TaskType);
      procedure setCriterion(c: SplitCriterion);
      procedure printHyperparameters();
      
      { ===== Data Handling ===== }
      procedure loadData(var inputData: TDataMatrix; var inputTargets: TTargetArray;
                        nSamples, nFeatures: integer);
      procedure trainForest();
      
      { ===== Tree-Level Inspection ===== }
      function inspectTree(treeId: integer): TTreeInfo;
      procedure printTreeStructure(treeId: integer);
      procedure printNodeDetails(treeId: integer; nodeId: integer);
      function getTreeDepth(treeId: integer): integer;
      function getTreeNumNodes(treeId: integer): integer;
      function getTreeNumLeaves(treeId: integer): integer;
      
      { ===== Tree-Level Manipulation ===== }
      procedure pruneTree(treeId: integer; nodeId: integer);
      procedure modifySplit(treeId: integer; nodeId: integer; newThreshold: double);
      procedure modifyLeafValue(treeId: integer; nodeId: integer; newValue: double);
      procedure convertToLeaf(treeId: integer; nodeId: integer; leafValue: double);
      
      { ===== Forest-Level Controls ===== }
      procedure addTree();
      procedure removeTree(treeId: integer);
      procedure replaceTree(treeId: integer);
      procedure retrainTree(treeId: integer);
      function getNumTrees(): integer;
      
      { ===== Feature Controls ===== }
      procedure enableFeature(featureIndex: integer);
      procedure disableFeature(featureIndex: integer);
      procedure setFeatureEnabled(featureIndex: integer; enabled: boolean);
      function isFeatureEnabled(featureIndex: integer): boolean;
      procedure resetFeatureFilters();
      function featureUsageSummary(): TFeatureStatsArray;
      procedure printFeatureUsageSummary();
      function getFeatureImportance(featureIndex: integer): double;
      procedure printFeatureImportances();
      
      { ===== Aggregation Control ===== }
      procedure setAggregationMethod(method: TAggregationMethod);
      function getAggregationMethod(): TAggregationMethod;
      procedure setTreeWeight(treeId: integer; weight: double);
      function getTreeWeight(treeId: integer): double;
      procedure resetTreeWeights();
      function aggregatePredictions(var sample: TDataRow): double;
      
      { ===== Prediction ===== }
      function predict(var sample: TDataRow): double;
      function predictClass(var sample: TDataRow): integer;
      function predictWithTree(treeId: integer; var sample: TDataRow): double;
      procedure predictBatch(var samples: TDataMatrix; nSamples: integer;
                            var predictions: TTargetArray);
      
      { ===== Sample Tracking ===== }
      function trackSample(sampleIndex: integer): TSampleTrackInfo;
      procedure printSampleTracking(sampleIndex: integer);
      
      { ===== OOB Analysis ===== }
      function oobErrorSummary(): TOOBTreeInfoArray;
      procedure printOOBSummary();
      function getGlobalOOBError(): double;
      procedure markProblematicTrees(errorThreshold: double);
      
      { ===== Diagnostics & Metrics ===== }
      function accuracy(var predictions, actual: TTargetArray; nSamples: integer): double;
      function meanSquaredError(var predictions, actual: TTargetArray; nSamples: integer): double;
      function rSquared(var predictions, actual: TTargetArray; nSamples: integer): double;
      function precision(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      function recall(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      function f1Score(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      procedure printMetrics(var predictions, actual: TTargetArray; nSamples: integer);
      
      { ===== Error Analysis ===== }
      procedure highlightMisclassified(var predictions, actual: TTargetArray; nSamples: integer);
      procedure highlightHighResidual(var predictions, actual: TTargetArray; 
                                      nSamples: integer; threshold: double);
      procedure findWorstTrees(var actual: TTargetArray; nSamples: integer; topN: integer);
      
      { ===== Visualization ===== }
      procedure visualizeTree(treeId: integer);
      procedure visualizeSplitDistribution(treeId: integer; nodeId: integer);
      procedure printForestOverview();
      procedure printFeatureHeatmap();
      
      { ===== Advanced / Experimental ===== }
      procedure swapCriterion(newCriterion: SplitCriterion);
      function compareForests(var otherForest: TRandomForest; 
                             var testData: TDataMatrix; var testTargets: TTargetArray;
                             nSamples: integer): TForestComparison;
      procedure printComparison(var comparison: TForestComparison);
      
      { ===== Cleanup ===== }
      procedure freeForest();

   end;

implementation

{ ============================================================================ }
{ Constructor }
{ ============================================================================ }

constructor TRandomForestFacade.create();
var
   i: integer;
begin
   forest.create();
   forestInitialized := false;
   currentAggregation := MajorityVote;
   
   for i := 0 to MAX_TREES - 1 do
      treeWeights[i] := 1.0;
   
   for i := 0 to MAX_FEATURES - 1 do
      featureEnabled[i] := true;
end;

{ ============================================================================ }
{ Internal Helpers }
{ ============================================================================ }

function TRandomForestFacade.collectNodeInfo(node: TreeNode; depth: integer;
                                              var nodes: TNodeInfoArray; 
                                              var count: integer): integer;
var
   currentId: integer;
begin
   if node = nil then
   begin
      collectNodeInfo := -1;
      exit;
   end;
   
   if count >= MAX_NODE_INFO then
   begin
      collectNodeInfo := -1;
      exit;
   end;
   
   currentId := count;
   nodes[count].nodeId := count;
   nodes[count].depth := depth;
   nodes[count].isLeaf := node^.isLeaf;
   nodes[count].featureIndex := node^.featureIndex;
   nodes[count].threshold := node^.threshold;
   nodes[count].prediction := node^.prediction;
   nodes[count].classLabel := node^.classLabel;
   nodes[count].impurity := node^.impurity;
   nodes[count].numSamples := node^.numSamples;
   nodes[count].leftChildId := -1;
   nodes[count].rightChildId := -1;
   inc(count);
   
   if not node^.isLeaf then
   begin
      nodes[currentId].leftChildId := collectNodeInfo(node^.left, depth + 1, nodes, count);
      nodes[currentId].rightChildId := collectNodeInfo(node^.right, depth + 1, nodes, count);
   end;
   
   collectNodeInfo := currentId;
end;

function TRandomForestFacade.calculateTreeDepth(node: TreeNode): integer;
var
   leftDepth, rightDepth: integer;
begin
   if node = nil then
   begin
      calculateTreeDepth := 0;
      exit;
   end;
   
   if node^.isLeaf then
   begin
      calculateTreeDepth := 1;
      exit;
   end;
   
   leftDepth := calculateTreeDepth(node^.left);
   rightDepth := calculateTreeDepth(node^.right);
   
   if leftDepth > rightDepth then
      calculateTreeDepth := leftDepth + 1
   else
      calculateTreeDepth := rightDepth + 1;
end;

function TRandomForestFacade.countLeaves(node: TreeNode): integer;
begin
   if node = nil then
   begin
      countLeaves := 0;
      exit;
   end;
   
   if node^.isLeaf then
   begin
      countLeaves := 1;
      exit;
   end;
   
   countLeaves := countLeaves(node^.left) + countLeaves(node^.right);
end;

procedure TRandomForestFacade.collectFeaturesUsed(node: TreeNode; var used: array of boolean);
begin
   if node = nil then
      exit;
   
   if not node^.isLeaf then
   begin
      if (node^.featureIndex >= 0) and (node^.featureIndex < MAX_FEATURES) then
         used[node^.featureIndex] := true;
      collectFeaturesUsed(node^.left, used);
      collectFeaturesUsed(node^.right, used);
   end;
end;

function TRandomForestFacade.findNodeById(node: TreeNode; targetId: integer; var currentId: integer): TreeNode;
var
   foundNode: TreeNode;
begin
   if node = nil then
   begin
      findNodeById := nil;
      exit;
   end;
   
   if currentId = targetId then
   begin
      findNodeById := node;
      exit;
   end;
   
   inc(currentId);
   
   if not node^.isLeaf then
   begin
      foundNode := findNodeById(node^.left, targetId, currentId);
      if foundNode <> nil then
      begin
         findNodeById := foundNode;
         exit;
      end;
      
      foundNode := findNodeById(node^.right, targetId, currentId);
      if foundNode <> nil then
      begin
         findNodeById := foundNode;
         exit;
      end;
   end;
   
   findNodeById := nil;
end;

procedure TRandomForestFacade.freeSubtree(node: TreeNode);
begin
   if node = nil then
      exit;
   
   freeSubtree(node^.left);
   freeSubtree(node^.right);
   dispose(node);
end;

{ ============================================================================ }
{ Initialization }
{ ============================================================================ }

procedure TRandomForestFacade.initForest();
begin
   forest.create();
   forestInitialized := true;
end;

function TRandomForestFacade.getForest(): TRandomForest;
begin
   getForest := forest;
end;

{ ============================================================================ }
{ Hyperparameter Control }
{ ============================================================================ }

procedure TRandomForestFacade.setHyperparameter(paramName: string; value: integer);
begin
   if paramName = 'n_estimators' then
      forest.setNumTrees(value)
   else if paramName = 'max_depth' then
      forest.setMaxDepth(value)
   else if paramName = 'min_samples_leaf' then
      forest.setMinSamplesLeaf(value)
   else if paramName = 'min_samples_split' then
      forest.setMinSamplesSplit(value)
   else if paramName = 'max_features' then
      forest.setMaxFeatures(value)
   else if paramName = 'random_seed' then
      forest.setRandomSeed(value)
   else
      writeln('Unknown hyperparameter: ', paramName);
end;

procedure TRandomForestFacade.setHyperparameterFloat(paramName: string; value: double);
begin
   writeln('Float hyperparameters not yet implemented: ', paramName);
end;

function TRandomForestFacade.getHyperparameter(paramName: string): integer;
begin
   getHyperparameter := 0;
end;

procedure TRandomForestFacade.setTaskType(t: TaskType);
begin
   forest.setTaskType(t);
end;

procedure TRandomForestFacade.setCriterion(c: SplitCriterion);
begin
   forest.setCriterion(c);
end;

procedure TRandomForestFacade.printHyperparameters();
begin
   forest.printForestInfo();
end;

{ ============================================================================ }
{ Data Handling }
{ ============================================================================ }

procedure TRandomForestFacade.loadData(var inputData: TDataMatrix; var inputTargets: TTargetArray;
                                       nSamples, nFeatures: integer);
begin
   forest.loadData(inputData, inputTargets, nSamples, nFeatures);
end;

procedure TRandomForestFacade.trainForest();
begin
   forest.fit();
   forestInitialized := true;
end;

{ ============================================================================ }
{ Tree-Level Inspection }
{ ============================================================================ }

function TRandomForestFacade.inspectTree(treeId: integer): TTreeInfo;
var
   info: TTreeInfo;
   i, count: integer;
begin
   info.treeId := treeId;
   info.numNodes := 0;
   info.maxDepth := 0;
   info.numLeaves := 0;
   info.numFeaturesUsed := 0;
   info.oobError := 0.0;
   
   for i := 0 to MAX_FEATURES - 1 do
      info.featuresUsed[i] := false;
   
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      inspectTree := info;
      exit;
   end;
   
   count := 0;
   collectNodeInfo(forest.getTree(treeId)^.root, 0, info.nodes, count);
   info.numNodes := count;
   info.maxDepth := calculateTreeDepth(forest.getTree(treeId)^.root);
   info.numLeaves := countLeaves(forest.getTree(treeId)^.root);
   
   collectFeaturesUsed(forest.getTree(treeId)^.root, info.featuresUsed);
   for i := 0 to MAX_FEATURES - 1 do
      if info.featuresUsed[i] then
         inc(info.numFeaturesUsed);
   
   inspectTree := info;
end;

procedure TRandomForestFacade.printTreeStructure(treeId: integer);
var
   info: TTreeInfo;
   i: integer;
begin
   info := inspectTree(treeId);
   
   writeln('=== Tree ', treeId, ' Structure ===');
   writeln('Nodes: ', info.numNodes);
   writeln('Max Depth: ', info.maxDepth);
   writeln('Leaves: ', info.numLeaves);
   writeln('Features Used: ', info.numFeaturesUsed);
   write('Feature Indices: ');
   for i := 0 to MAX_FEATURES - 1 do
      if info.featuresUsed[i] then
         write(i, ' ');
   writeln;
   writeln;
   
   writeln('Node Details:');
   writeln('ID    Depth  Leaf   Feature  Threshold      Prediction  Samples  Impurity');
   writeln('----------------------------------------------------------------------');
   for i := 0 to info.numNodes - 1 do
   begin
      write(info.nodes[i].nodeId:4, '  ');
      write(info.nodes[i].depth:4, '   ');
      if info.nodes[i].isLeaf then
         write('Yes    ')
      else
         write('No     ');
      write(info.nodes[i].featureIndex:6, '  ');
      write(info.nodes[i].threshold:12:4, '  ');
      write(info.nodes[i].prediction:10:4, '  ');
      write(info.nodes[i].numSamples:6, '  ');
      writeln(info.nodes[i].impurity:8:4);
   end;
end;

procedure TRandomForestFacade.printNodeDetails(treeId: integer; nodeId: integer);
var
   info: TTreeInfo;
begin
   info := inspectTree(treeId);
   
   if (nodeId < 0) or (nodeId >= info.numNodes) then
   begin
      writeln('Invalid node ID: ', nodeId);
      exit;
   end;
   
   writeln('=== Node ', nodeId, ' in Tree ', treeId, ' ===');
   writeln('Depth: ', info.nodes[nodeId].depth);
   writeln('Is Leaf: ', info.nodes[nodeId].isLeaf);
   if not info.nodes[nodeId].isLeaf then
   begin
      writeln('Split Feature: ', info.nodes[nodeId].featureIndex);
      writeln('Threshold: ', info.nodes[nodeId].threshold:0:4);
      writeln('Left Child: ', info.nodes[nodeId].leftChildId);
      writeln('Right Child: ', info.nodes[nodeId].rightChildId);
   end;
   writeln('Prediction: ', info.nodes[nodeId].prediction:0:4);
   writeln('Class Label: ', info.nodes[nodeId].classLabel);
   writeln('Samples: ', info.nodes[nodeId].numSamples);
   writeln('Impurity: ', info.nodes[nodeId].impurity:0:4);
end;

function TRandomForestFacade.getTreeDepth(treeId: integer): integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      getTreeDepth := 0;
      exit;
   end;
   getTreeDepth := calculateTreeDepth(forest.getTree(treeId)^.root);
end;

function TRandomForestFacade.getTreeNumNodes(treeId: integer): integer;
var
   info: TTreeInfo;
begin
   info := inspectTree(treeId);
   getTreeNumNodes := info.numNodes;
end;

function TRandomForestFacade.getTreeNumLeaves(treeId: integer): integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      getTreeNumLeaves := 0;
      exit;
   end;
   getTreeNumLeaves := countLeaves(forest.getTree(treeId)^.root);
end;

{ ============================================================================ }
{ Tree-Level Manipulation }
{ ============================================================================ }

procedure TRandomForestFacade.pruneTree(treeId: integer; nodeId: integer);
var
   tree: TDecisionTree;
   node: TreeNode;
   searchId: integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   tree := forest.getTree(treeId);
   if tree = nil then
   begin
      writeln('Tree not found: ', treeId);
      exit;
   end;
   
   searchId := 0;
   node := findNodeById(tree^.root, nodeId, searchId);
   
   if node = nil then
   begin
      writeln('Node not found: ', nodeId);
      exit;
   end;
   
   if node^.isLeaf then
   begin
      writeln('Cannot prune a leaf node');
      exit;
   end;
   
   freeSubtree(node^.left);
   freeSubtree(node^.right);
   node^.left := nil;
   node^.right := nil;
   node^.isLeaf := true;
   
   writeln('Pruned node ', nodeId, ' in tree ', treeId);
end;

procedure TRandomForestFacade.modifySplit(treeId: integer; nodeId: integer; newThreshold: double);
var
   tree: TDecisionTree;
   node: TreeNode;
   searchId: integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   tree := forest.getTree(treeId);
   if tree = nil then
   begin
      writeln('Tree not found: ', treeId);
      exit;
   end;
   
   searchId := 0;
   node := findNodeById(tree^.root, nodeId, searchId);
   
   if node = nil then
   begin
      writeln('Node not found: ', nodeId);
      exit;
   end;
   
   if node^.isLeaf then
   begin
      writeln('Cannot modify split on a leaf node');
      exit;
   end;
   
   writeln('Modified threshold from ', node^.threshold:0:4, ' to ', newThreshold:0:4);
   node^.threshold := newThreshold;
end;

procedure TRandomForestFacade.modifyLeafValue(treeId: integer; nodeId: integer; newValue: double);
var
   tree: TDecisionTree;
   node: TreeNode;
   searchId: integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   tree := forest.getTree(treeId);
   if tree = nil then
   begin
      writeln('Tree not found: ', treeId);
      exit;
   end;
   
   searchId := 0;
   node := findNodeById(tree^.root, nodeId, searchId);
   
   if node = nil then
   begin
      writeln('Node not found: ', nodeId);
      exit;
   end;
   
   if not node^.isLeaf then
   begin
      writeln('Node is not a leaf');
      exit;
   end;
   
   writeln('Modified leaf value from ', node^.prediction:0:4, ' to ', newValue:0:4);
   node^.prediction := newValue;
   node^.classLabel := round(newValue);
end;

procedure TRandomForestFacade.convertToLeaf(treeId: integer; nodeId: integer; leafValue: double);
var
   tree: TDecisionTree;
   node: TreeNode;
   searchId: integer;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   tree := forest.getTree(treeId);
   if tree = nil then
   begin
      writeln('Tree not found: ', treeId);
      exit;
   end;
   
   searchId := 0;
   node := findNodeById(tree^.root, nodeId, searchId);
   
   if node = nil then
   begin
      writeln('Node not found: ', nodeId);
      exit;
   end;
   
   if node^.isLeaf then
   begin
      writeln('Node is already a leaf');
      exit;
   end;
   
   freeSubtree(node^.left);
   freeSubtree(node^.right);
   node^.left := nil;
   node^.right := nil;
   node^.isLeaf := true;
   node^.prediction := leafValue;
   node^.classLabel := round(leafValue);
   node^.featureIndex := -1;
   node^.threshold := 0;
   
   writeln('Converted node ', nodeId, ' to leaf with value ', leafValue:0:4);
end;

{ ============================================================================ }
{ Forest-Level Controls }
{ ============================================================================ }

procedure TRandomForestFacade.addTree();
var
   oldCount: integer;
begin
   oldCount := getNumTrees();
   forest.addNewTree();
   if getNumTrees() > oldCount then
      writeln('Added new tree. Total trees: ', getNumTrees())
   else
      writeln('Failed to add tree');
end;

procedure TRandomForestFacade.removeTree(treeId: integer);
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   forest.removeTreeAt(treeId);
   writeln('Removed tree ', treeId, '. Total trees: ', getNumTrees());
end;

procedure TRandomForestFacade.replaceTree(treeId: integer);
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   forest.retrainTreeAt(treeId);
   writeln('Replaced tree ', treeId, ' with new bootstrap sample');
end;

procedure TRandomForestFacade.retrainTree(treeId: integer);
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   forest.retrainTreeAt(treeId);
   writeln('Retrained tree ', treeId);
end;

function TRandomForestFacade.getNumTrees(): integer;
begin
   getNumTrees := forest.getNumTrees();
end;

{ ============================================================================ }
{ Feature Controls }
{ ============================================================================ }

procedure TRandomForestFacade.enableFeature(featureIndex: integer);
begin
   if (featureIndex >= 0) and (featureIndex < MAX_FEATURES) then
      featureEnabled[featureIndex] := true;
end;

procedure TRandomForestFacade.disableFeature(featureIndex: integer);
begin
   if (featureIndex >= 0) and (featureIndex < MAX_FEATURES) then
      featureEnabled[featureIndex] := false;
end;

procedure TRandomForestFacade.setFeatureEnabled(featureIndex: integer; enabled: boolean);
begin
   if (featureIndex >= 0) and (featureIndex < MAX_FEATURES) then
      featureEnabled[featureIndex] := enabled;
end;

function TRandomForestFacade.isFeatureEnabled(featureIndex: integer): boolean;
begin
   if (featureIndex >= 0) and (featureIndex < MAX_FEATURES) then
      isFeatureEnabled := featureEnabled[featureIndex]
   else
      isFeatureEnabled := false;
end;

procedure TRandomForestFacade.resetFeatureFilters();
var
   i: integer;
begin
   for i := 0 to MAX_FEATURES - 1 do
      featureEnabled[i] := true;
end;

function TRandomForestFacade.featureUsageSummary(): TFeatureStatsArray;
var
   stats: TFeatureStatsArray;
   i, t: integer;
   treeInfo: TTreeInfo;
begin
   for i := 0 to MAX_FEATURE_STATS - 1 do
   begin
      stats[i].featureIndex := i;
      stats[i].timesUsed := 0;
      stats[i].treesUsedIn := 0;
      stats[i].avgImportance := 0.0;
      stats[i].totalImportance := 0.0;
   end;
   
   for t := 0 to getNumTrees() - 1 do
   begin
      treeInfo := inspectTree(t);
      for i := 0 to MAX_FEATURES - 1 do
      begin
         if treeInfo.featuresUsed[i] then
            inc(stats[i].treesUsedIn);
      end;
   end;
   
   for i := 0 to forest.getNumFeatures() - 1 do
   begin
      stats[i].totalImportance := forest.getFeatureImportance(i);
      if getNumTrees() > 0 then
         stats[i].avgImportance := stats[i].totalImportance;
   end;
   
   featureUsageSummary := stats;
end;

procedure TRandomForestFacade.printFeatureUsageSummary();
var
   stats: TFeatureStatsArray;
   i: integer;
begin
   stats := featureUsageSummary();
   
   writeln('=== Feature Usage Summary ===');
   writeln('Feature  Trees Used In  Importance');
   writeln('----------------------------------');
   for i := 0 to forest.getNumFeatures() - 1 do
   begin
      write(i:6, '  ');
      write(stats[i].treesUsedIn:12, '  ');
      writeln(stats[i].totalImportance:10:4);
   end;
end;

function TRandomForestFacade.getFeatureImportance(featureIndex: integer): double;
begin
   getFeatureImportance := forest.getFeatureImportance(featureIndex);
end;

procedure TRandomForestFacade.printFeatureImportances();
begin
   forest.printFeatureImportances();
end;

{ ============================================================================ }
{ Aggregation Control }
{ ============================================================================ }

procedure TRandomForestFacade.setAggregationMethod(method: TAggregationMethod);
begin
   currentAggregation := method;
end;

function TRandomForestFacade.getAggregationMethod(): TAggregationMethod;
begin
   getAggregationMethod := currentAggregation;
end;

procedure TRandomForestFacade.setTreeWeight(treeId: integer; weight: double);
begin
   if (treeId >= 0) and (treeId < MAX_TREES) then
      treeWeights[treeId] := weight;
end;

function TRandomForestFacade.getTreeWeight(treeId: integer): double;
begin
   if (treeId >= 0) and (treeId < MAX_TREES) then
      getTreeWeight := treeWeights[treeId]
   else
      getTreeWeight := 1.0;
end;

procedure TRandomForestFacade.resetTreeWeights();
var
   i: integer;
begin
   for i := 0 to MAX_TREES - 1 do
      treeWeights[i] := 1.0;
end;

function TRandomForestFacade.aggregatePredictions(var sample: TDataRow): double;
var
   i: integer;
   sum, weightSum, pred: double;
   votes: array[0..99] of double;
   maxVotes: double;
   maxClass, classLabel: integer;
begin
   case currentAggregation of
      MajorityVote:
      begin
         for i := 0 to 99 do
            votes[i] := 0;
         
         for i := 0 to getNumTrees() - 1 do
         begin
            classLabel := round(forest.predictTree(forest.getTree(i)^.root, sample));
            if (classLabel >= 0) and (classLabel <= 99) then
               votes[classLabel] := votes[classLabel] + 1;
         end;
         
         maxVotes := 0;
         maxClass := 0;
         for i := 0 to 99 do
         begin
            if votes[i] > maxVotes then
            begin
               maxVotes := votes[i];
               maxClass := i;
            end;
         end;
         aggregatePredictions := maxClass;
      end;
      
      WeightedVote:
      begin
         for i := 0 to 99 do
            votes[i] := 0;
         
         for i := 0 to getNumTrees() - 1 do
         begin
            classLabel := round(forest.predictTree(forest.getTree(i)^.root, sample));
            if (classLabel >= 0) and (classLabel <= 99) then
               votes[classLabel] := votes[classLabel] + treeWeights[i];
         end;
         
         maxVotes := 0;
         maxClass := 0;
         for i := 0 to 99 do
         begin
            if votes[i] > maxVotes then
            begin
               maxVotes := votes[i];
               maxClass := i;
            end;
         end;
         aggregatePredictions := maxClass;
      end;
      
      Mean:
      begin
         sum := 0;
         for i := 0 to getNumTrees() - 1 do
            sum := sum + forest.predictTree(forest.getTree(i)^.root, sample);
         aggregatePredictions := sum / getNumTrees();
      end;
      
      WeightedMean:
      begin
         sum := 0;
         weightSum := 0;
         for i := 0 to getNumTrees() - 1 do
         begin
            pred := forest.predictTree(forest.getTree(i)^.root, sample);
            sum := sum + pred * treeWeights[i];
            weightSum := weightSum + treeWeights[i];
         end;
         if weightSum > 0 then
            aggregatePredictions := sum / weightSum
         else
            aggregatePredictions := 0;
      end;
   end;
end;

{ ============================================================================ }
{ Prediction }
{ ============================================================================ }

function TRandomForestFacade.predict(var sample: TDataRow): double;
begin
   predict := aggregatePredictions(sample);
end;

function TRandomForestFacade.predictClass(var sample: TDataRow): integer;
begin
   predictClass := round(predict(sample));
end;

function TRandomForestFacade.predictWithTree(treeId: integer; var sample: TDataRow): double;
begin
   if (treeId < 0) or (treeId >= getNumTrees()) then
   begin
      predictWithTree := 0;
      exit;
   end;
   predictWithTree := forest.predictTree(forest.getTree(treeId)^.root, sample);
end;

procedure TRandomForestFacade.predictBatch(var samples: TDataMatrix; nSamples: integer;
                                           var predictions: TTargetArray);
var
   i: integer;
begin
   for i := 0 to nSamples - 1 do
      predictions[i] := predict(samples[i]);
end;

{ ============================================================================ }
{ Sample Tracking }
{ ============================================================================ }

function TRandomForestFacade.trackSample(sampleIndex: integer): TSampleTrackInfo;
var
   info: TSampleTrackInfo;
   t: integer;
   tree: TDecisionTree;
   sampleRow: TDataRow;
   j: integer;
begin
   info.sampleIndex := sampleIndex;
   info.numTreesInfluenced := 0;
   info.numOobTrees := 0;
   
   for t := 0 to MAX_TREES - 1 do
   begin
      info.treesInfluenced[t] := false;
      info.oobTrees[t] := false;
      info.predictions[t] := 0;
   end;
   
   if (sampleIndex < 0) or (sampleIndex >= forest.getNumSamples()) then
   begin
      trackSample := info;
      exit;
   end;
   
   for j := 0 to forest.getNumFeatures() - 1 do
      sampleRow[j] := forest.getData(sampleIndex, j);
   
   for t := 0 to getNumTrees() - 1 do
   begin
      tree := forest.getTree(t);
      if tree <> nil then
      begin
         if tree^.oobIndices[sampleIndex] then
         begin
            info.oobTrees[t] := true;
            inc(info.numOobTrees);
         end
         else
         begin
            info.treesInfluenced[t] := true;
            inc(info.numTreesInfluenced);
         end;
         
         info.predictions[t] := forest.predictTree(tree^.root, sampleRow);
      end;
   end;
   
   trackSample := info;
end;

procedure TRandomForestFacade.printSampleTracking(sampleIndex: integer);
var
   info: TSampleTrackInfo;
   t: integer;
begin
   info := trackSample(sampleIndex);
   
   writeln('=== Sample ', sampleIndex, ' Tracking ===');
   writeln('Trees Influenced (in bootstrap): ', info.numTreesInfluenced);
   writeln('OOB Trees (not in bootstrap): ', info.numOobTrees);
   writeln;
   
   writeln('Tree  Influenced  OOB  Prediction');
   writeln('----------------------------------');
   for t := 0 to getNumTrees() - 1 do
   begin
      write(t:4, '  ');
      if info.treesInfluenced[t] then
         write('Yes        ')
      else
         write('No         ');
      if info.oobTrees[t] then
         write('Yes  ')
      else
         write('No   ');
      writeln(info.predictions[t]:0:4);
   end;
end;

{ ============================================================================ }
{ OOB Analysis }
{ ============================================================================ }

function TRandomForestFacade.oobErrorSummary(): TOOBTreeInfoArray;
var
   summary: TOOBTreeInfoArray;
   t, s: integer;
   tree: TDecisionTree;
   sampleRow: TDataRow;
   j: integer;
   pred: double;
   errors, correct: integer;
begin
   for t := 0 to MAX_TREES - 1 do
   begin
      summary[t].treeId := t;
      summary[t].numOobSamples := 0;
      summary[t].oobError := 0.0;
      summary[t].oobAccuracy := 0.0;
   end;
   
   for t := 0 to getNumTrees() - 1 do
   begin
      tree := forest.getTree(t);
      if tree = nil then
         continue;
      
      errors := 0;
      correct := 0;
      
      for s := 0 to forest.getNumSamples() - 1 do
      begin
         if tree^.oobIndices[s] then
         begin
            inc(summary[t].numOobSamples);
            
            for j := 0 to forest.getNumFeatures() - 1 do
               sampleRow[j] := forest.getData(s, j);
            
            pred := forest.predictTree(tree^.root, sampleRow);
            
            if forest.getTaskType() = Classification then
            begin
               if round(pred) = round(forest.getTarget(s)) then
                  inc(correct)
               else
                  inc(errors);
            end
            else
            begin
               summary[t].oobError := summary[t].oobError + 
                  sqr(pred - forest.getTarget(s));
            end;
         end;
      end;
      
      if summary[t].numOobSamples > 0 then
      begin
         if forest.getTaskType() = Classification then
         begin
            summary[t].oobError := errors / summary[t].numOobSamples;
            summary[t].oobAccuracy := correct / summary[t].numOobSamples;
         end
         else
         begin
            summary[t].oobError := summary[t].oobError / summary[t].numOobSamples;
            summary[t].oobAccuracy := 1.0 - summary[t].oobError;
         end;
      end;
   end;
   
   oobErrorSummary := summary;
end;

procedure TRandomForestFacade.printOOBSummary();
var
   summary: TOOBTreeInfoArray;
   t: integer;
   totalError: double;
   count: integer;
begin
   summary := oobErrorSummary();
   
   writeln('=== OOB Error Summary ===');
   writeln('Tree  OOB Samples  OOB Error  OOB Accuracy');
   writeln('-------------------------------------------');
   
   totalError := 0;
   count := 0;
   
   for t := 0 to getNumTrees() - 1 do
   begin
      write(t:4, '  ');
      write(summary[t].numOobSamples:10, '  ');
      write(summary[t].oobError:9:4, '  ');
      writeln(summary[t].oobAccuracy:11:4);
      
      if summary[t].numOobSamples > 0 then
      begin
         totalError := totalError + summary[t].oobError;
         inc(count);
      end;
   end;
   
   writeln('-------------------------------------------');
   if count > 0 then
      writeln('Average OOB Error: ', (totalError / count):0:4)
   else
      writeln('Average OOB Error: N/A');
   writeln('Global OOB Error: ', getGlobalOOBError():0:4);
end;

function TRandomForestFacade.getGlobalOOBError(): double;
begin
   getGlobalOOBError := forest.calculateOOBError();
end;

procedure TRandomForestFacade.markProblematicTrees(errorThreshold: double);
var
   summary: TOOBTreeInfoArray;
   t: integer;
   problemCount: integer;
begin
   summary := oobErrorSummary();
   problemCount := 0;
   
   writeln('=== Problematic Trees (Error > ', errorThreshold:0:4, ') ===');
   
   for t := 0 to getNumTrees() - 1 do
   begin
      if (summary[t].numOobSamples > 0) and (summary[t].oobError > errorThreshold) then
      begin
         writeln('Tree ', t, ': OOB Error = ', summary[t].oobError:0:4, 
                 ' (', summary[t].numOobSamples, ' OOB samples)');
         inc(problemCount);
      end;
   end;
   
   if problemCount = 0 then
      writeln('No problematic trees found.')
   else
      writeln('Total problematic trees: ', problemCount);
end;

{ ============================================================================ }
{ Diagnostics & Metrics }
{ ============================================================================ }

function TRandomForestFacade.accuracy(var predictions, actual: TTargetArray; nSamples: integer): double;
begin
   accuracy := forest.accuracy(predictions, actual, nSamples);
end;

function TRandomForestFacade.meanSquaredError(var predictions, actual: TTargetArray; nSamples: integer): double;
begin
   meanSquaredError := forest.meanSquaredError(predictions, actual, nSamples);
end;

function TRandomForestFacade.rSquared(var predictions, actual: TTargetArray; nSamples: integer): double;
begin
   rSquared := forest.rSquared(predictions, actual, nSamples);
end;

function TRandomForestFacade.precision(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
begin
   precision := forest.precision(predictions, actual, nSamples, posClass);
end;

function TRandomForestFacade.recall(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
begin
   recall := forest.recall(predictions, actual, nSamples, posClass);
end;

function TRandomForestFacade.f1Score(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
begin
   f1Score := forest.f1Score(predictions, actual, nSamples, posClass);
end;

procedure TRandomForestFacade.printMetrics(var predictions, actual: TTargetArray; nSamples: integer);
begin
   writeln('=== Performance Metrics ===');
   writeln('Accuracy: ', accuracy(predictions, actual, nSamples):0:4);
   writeln('MSE: ', meanSquaredError(predictions, actual, nSamples):0:4);
   writeln('R-Squared: ', rSquared(predictions, actual, nSamples):0:4);
end;

{ ============================================================================ }
{ Error Analysis }
{ ============================================================================ }

procedure TRandomForestFacade.highlightMisclassified(var predictions, actual: TTargetArray; nSamples: integer);
var
   i: integer;
   count: integer;
begin
   writeln('=== Misclassified Samples ===');
   writeln('Index  Predicted  Actual');
   writeln('-------------------------');
   
   count := 0;
   for i := 0 to nSamples - 1 do
   begin
      if round(predictions[i]) <> round(actual[i]) then
      begin
         writeln(i:5, '  ', round(predictions[i]):9, '  ', round(actual[i]):6);
         inc(count);
      end;
   end;
   
   writeln('-------------------------');
   writeln('Total misclassified: ', count, ' / ', nSamples, 
           ' (', (count / nSamples * 100):0:2, '%)');
end;

procedure TRandomForestFacade.highlightHighResidual(var predictions, actual: TTargetArray;
                                                    nSamples: integer; threshold: double);
var
   i: integer;
   count: integer;
   residual: double;
begin
   writeln('=== High Residual Samples (> ', threshold:0:4, ') ===');
   writeln('Index  Predicted  Actual   Residual');
   writeln('-------------------------------------');
   
   count := 0;
   for i := 0 to nSamples - 1 do
   begin
      residual := abs(predictions[i] - actual[i]);
      if residual > threshold then
      begin
         writeln(i:5, '  ', predictions[i]:9:4, '  ', actual[i]:6:4, '  ', residual:8:4);
         inc(count);
      end;
   end;
   
   writeln('-------------------------------------');
   writeln('Total high-residual samples: ', count, ' / ', nSamples);
end;

procedure TRandomForestFacade.findWorstTrees(var actual: TTargetArray; nSamples: integer; topN: integer);
var
   summary: TOOBTreeInfoArray;
   treeErrors: array[0..MAX_TREES-1] of record
      treeId: integer;
      error: double;
   end;
   t, i, j: integer;
   tempId: integer;
   tempError: double;
begin
   summary := oobErrorSummary();
   
   for t := 0 to getNumTrees() - 1 do
   begin
      treeErrors[t].treeId := t;
      treeErrors[t].error := summary[t].oobError;
   end;
   
   for i := 0 to getNumTrees() - 2 do
   begin
      for j := i + 1 to getNumTrees() - 1 do
      begin
         if treeErrors[j].error > treeErrors[i].error then
         begin
            tempId := treeErrors[i].treeId;
            tempError := treeErrors[i].error;
            treeErrors[i].treeId := treeErrors[j].treeId;
            treeErrors[i].error := treeErrors[j].error;
            treeErrors[j].treeId := tempId;
            treeErrors[j].error := tempError;
         end;
      end;
   end;
   
   writeln('=== Top ', topN, ' Worst Trees ===');
   writeln('Rank  Tree  OOB Error');
   writeln('----------------------');
   
   for i := 0 to topN - 1 do
   begin
      if i >= getNumTrees() then
         break;
      writeln(i + 1:4, '  ', treeErrors[i].treeId:4, '  ', treeErrors[i].error:9:4);
   end;
end;

{ ============================================================================ }
{ Visualization }
{ ============================================================================ }

procedure TRandomForestFacade.visualizeTree(treeId: integer);
var
   info: TTreeInfo;
   i, indent, j: integer;
begin
   info := inspectTree(treeId);
   
   writeln('=== Tree ', treeId, ' Visualization ===');
   writeln;
   
   for i := 0 to info.numNodes - 1 do
   begin
      indent := info.nodes[i].depth * 2;
      for j := 1 to indent do
         write(' ');
      
      if info.nodes[i].isLeaf then
         writeln('[Leaf] -> ', info.nodes[i].prediction:0:2, 
                 ' (n=', info.nodes[i].numSamples, ')')
      else
         writeln('[Split] Feature ', info.nodes[i].featureIndex, 
                 ' <= ', info.nodes[i].threshold:0:4, 
                 ' (n=', info.nodes[i].numSamples, ', imp=', info.nodes[i].impurity:0:4, ')');
   end;
end;

procedure TRandomForestFacade.visualizeSplitDistribution(treeId: integer; nodeId: integer);
var
   info: TTreeInfo;
begin
   info := inspectTree(treeId);
   
   if (nodeId < 0) or (nodeId >= info.numNodes) then
   begin
      writeln('Invalid node ID: ', nodeId);
      exit;
   end;
   
   writeln('=== Split Distribution for Tree ', treeId, ', Node ', nodeId, ' ===');
   writeln('Feature Index: ', info.nodes[nodeId].featureIndex);
   writeln('Threshold: ', info.nodes[nodeId].threshold:0:4);
   writeln('Impurity: ', info.nodes[nodeId].impurity:0:4);
   writeln('Samples at Node: ', info.nodes[nodeId].numSamples);
   writeln('Is Leaf: ', info.nodes[nodeId].isLeaf);
   
   if not info.nodes[nodeId].isLeaf then
   begin
      writeln('Left Child ID: ', info.nodes[nodeId].leftChildId);
      writeln('Right Child ID: ', info.nodes[nodeId].rightChildId);
   end;
end;

procedure TRandomForestFacade.printForestOverview();
var
   i: integer;
   totalNodes, totalLeaves: integer;
   avgDepth: double;
begin
   writeln('=== Forest Overview ===');
   forest.printForestInfo();
   writeln;
   
   totalNodes := 0;
   totalLeaves := 0;
   avgDepth := 0;
   
   for i := 0 to getNumTrees() - 1 do
   begin
      totalNodes := totalNodes + getTreeNumNodes(i);
      totalLeaves := totalLeaves + getTreeNumLeaves(i);
      avgDepth := avgDepth + getTreeDepth(i);
   end;
   
   if getNumTrees() > 0 then
      avgDepth := avgDepth / getNumTrees();
   
   writeln('Forest Statistics:');
   writeln('  Total Nodes: ', totalNodes);
   writeln('  Total Leaves: ', totalLeaves);
   writeln('  Average Tree Depth: ', avgDepth:0:2);
   writeln;
   
   writeln('Tree Summary:');
   writeln('Tree  Depth  Nodes  Leaves  Weight');
   writeln('------------------------------------');
   for i := 0 to getNumTrees() - 1 do
   begin
      write(i:4, '  ');
      write(getTreeDepth(i):5, '  ');
      write(getTreeNumNodes(i):5, '  ');
      write(getTreeNumLeaves(i):6, '  ');
      writeln(treeWeights[i]:6:2);
   end;
end;

procedure TRandomForestFacade.printFeatureHeatmap();
var
   t, f: integer;
   info: TTreeInfo;
   maxUsage: integer;
   usage: array[0..MAX_FEATURES-1, 0..MAX_TREES-1] of boolean;
begin
   writeln('=== Feature Usage Heatmap ===');
   writeln;
   
   for f := 0 to MAX_FEATURES - 1 do
      for t := 0 to MAX_TREES - 1 do
         usage[f][t] := false;
   
   for t := 0 to getNumTrees() - 1 do
   begin
      info := inspectTree(t);
      for f := 0 to MAX_FEATURES - 1 do
         usage[f][t] := info.featuresUsed[f];
   end;
   
   write('Feat  ');
   for t := 0 to getNumTrees() - 1 do
   begin
      if t < 10 then
         write(t, ' ')
      else
         write(t mod 10, ' ');
   end;
   writeln('  Total');
   
   writeln(StringOfChar('-', 8 + getNumTrees() * 2 + 8));
   
   for f := 0 to forest.getNumFeatures() - 1 do
   begin
      write(f:4, '  ');
      maxUsage := 0;
      for t := 0 to getNumTrees() - 1 do
      begin
         if usage[f][t] then
         begin
            write('X ');
            inc(maxUsage);
         end
         else
            write('. ');
      end;
      writeln('  ', maxUsage:4);
   end;
end;

{ ============================================================================ }
{ Advanced / Experimental }
{ ============================================================================ }

procedure TRandomForestFacade.swapCriterion(newCriterion: SplitCriterion);
begin
   forest.setCriterion(newCriterion);
   writeln('Criterion changed. Retrain forest to apply.');
end;

function TRandomForestFacade.compareForests(var otherForest: TRandomForest;
                                            var testData: TDataMatrix; var testTargets: TTargetArray;
                                            nSamples: integer): TForestComparison;
var
   comparison: TForestComparison;
   predsA, predsB: TTargetArray;
   i: integer;
   diff: double;
begin
   comparison.numDifferentPredictions := 0;
   comparison.avgPredictionDiff := 0;
   
   for i := 0 to nSamples - 1 do
   begin
      predsA[i] := predict(testData[i]);
      predsB[i] := otherForest.predict(testData[i]);
      
      diff := abs(predsA[i] - predsB[i]);
      comparison.avgPredictionDiff := comparison.avgPredictionDiff + diff;
      
      if round(predsA[i]) <> round(predsB[i]) then
         inc(comparison.numDifferentPredictions);
   end;
   
   comparison.avgPredictionDiff := comparison.avgPredictionDiff / nSamples;
   comparison.forestAAccuracy := accuracy(predsA, testTargets, nSamples);
   comparison.forestBAccuracy := accuracy(predsB, testTargets, nSamples);
   comparison.forestAMSE := meanSquaredError(predsA, testTargets, nSamples);
   comparison.forestBMSE := meanSquaredError(predsB, testTargets, nSamples);
   
   compareForests := comparison;
end;

procedure TRandomForestFacade.printComparison(var comparison: TForestComparison);
begin
   writeln('=== Forest Comparison ===');
   writeln('Different Predictions: ', comparison.numDifferentPredictions);
   writeln('Avg Prediction Diff: ', comparison.avgPredictionDiff:0:4);
   writeln('Forest A Accuracy: ', comparison.forestAAccuracy:0:4);
   writeln('Forest B Accuracy: ', comparison.forestBAccuracy:0:4);
   writeln('Forest A MSE: ', comparison.forestAMSE:0:4);
   writeln('Forest B MSE: ', comparison.forestBMSE:0:4);
end;

{ ============================================================================ }
{ Cleanup }
{ ============================================================================ }

procedure TRandomForestFacade.freeForest();
begin
   forest.freeForest();
   forestInitialized := false;
end;

end.

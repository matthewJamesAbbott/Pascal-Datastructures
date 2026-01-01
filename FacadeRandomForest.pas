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
 
{$mode objfpc}
{$H+}

program RandomForest;

uses
   Math, SysUtils, Classes, StrUtils;

const
   MAX_FEATURES = 100;
   MAX_SAMPLES = 2000;
   MAX_TREES = 500;
   MAX_DEPTH_DEFAULT = 10;
   MIN_SAMPLES_LEAF_DEFAULT = 1;
   MIN_SAMPLES_SPLIT_DEFAULT = 2;
   MAX_NODE_INFO = 1000;
   MAX_FEATURE_STATS = 100;
   MAX_SAMPLE_TRACK = 1000;

type
   TaskType = (Classification, Regression);
   SplitCriterion = (Gini, Entropy, MSE, VarianceReduction);

   TDataRow = array[0..MAX_FEATURES-1] of double;
   TTargetArray = array[0..MAX_SAMPLES-1] of double;
   TDataMatrix = array[0..MAX_SAMPLES-1] of TDataRow;
   TIndexArray = array[0..MAX_SAMPLES-1] of integer;
   TFeatureArray = array[0..MAX_FEATURES-1] of integer;
   TBoolArray = array[0..MAX_SAMPLES-1] of boolean;
   TDoubleArray = array[0..MAX_FEATURES-1] of double;

   TreeNode = ^TreeNodeRec;
   TreeNodeRec = record
      isLeaf: boolean;
      featureIndex: integer;
      threshold: double;
      prediction: double;
      classLabel: integer;
      impurity: double;
      numSamples: integer;
      left, right: TreeNode;
   end;

   TDecisionTree = ^TDecisionTreeRec;
   TDecisionTreeRec = record
      root: TreeNode;
      maxDepth: integer;
      minSamplesLeaf: integer;
      minSamplesSplit: integer;
      maxFeatures: integer;
      taskType: TaskType;
      criterion: SplitCriterion;
      oobIndices: TBoolArray;
      numOobIndices: integer;
   end;

   TRandomForest = object

   private
      trees: array[0..MAX_TREES-1] of TDecisionTree;
      numTrees: integer;
      maxDepth: integer;
      minSamplesLeaf: integer;
      minSamplesSplit: integer;
      maxFeatures: integer;
      numFeatures: integer;
      numSamples: integer;
      taskType: TaskType;
      criterion: SplitCriterion;
      featureImportances: TDoubleArray;
      randomSeed: longint;

      data: TDataMatrix;
      targets: TTargetArray;

   public
      constructor create();

      { Hyperparameter Handling }
      procedure setNumTrees(n: integer);
      procedure setMaxDepth(d: integer);
      procedure setMinSamplesLeaf(m: integer);
      procedure setMinSamplesSplit(m: integer);
      procedure setMaxFeatures(m: integer);
      procedure setTaskType(t: TaskType);
      procedure setCriterion(c: SplitCriterion);
      procedure setRandomSeed(seed: longint);

      { Random Number Generator }
      function randomInt(maxVal: integer): integer;
      function randomDouble(): double;

      { Data Handling Functions }
      procedure loadData(var inputData: TDataMatrix; var inputTargets: TTargetArray;
                        nSamples, nFeatures: integer);
      procedure trainTestSplit(var trainIndices, testIndices: TIndexArray;
                              var numTrain, numTest: integer; testRatio: double);
      procedure bootstrap(var sampleIndices: TIndexArray; var numBootstrap: integer;
                         var oobMask: TBoolArray);
      procedure selectFeatureSubset(var featureIndices: TFeatureArray;
                                   var numSelected: integer);

      { Decision Tree Functions }
      function calculateGini(var indices: TIndexArray; numIndices: integer): double;
      function calculateEntropy(var indices: TIndexArray; numIndices: integer): double;
      function calculateMSE(var indices: TIndexArray; numIndices: integer): double;
      function calculateVariance(var indices: TIndexArray; numIndices: integer): double;
      function calculateImpurity(var indices: TIndexArray; numIndices: integer): double;
      function findBestSplit(var indices: TIndexArray; numIndices: integer;
                            var featureIndices: TFeatureArray; nFeatures: integer;
                            var bestFeature: integer; var bestThreshold: double;
                            var bestGain: double): boolean;
      function getMajorityClass(var indices: TIndexArray; numIndices: integer): integer;
      function getMeanTarget(var indices: TIndexArray; numIndices: integer): double;
      function createLeafNode(var indices: TIndexArray; numIndices: integer): TreeNode;
      function shouldStop(depth, numIndices: integer; impurity: double): boolean;
      function buildTree(var indices: TIndexArray; numIndices: integer;
                        depth: integer; tree: TDecisionTree): TreeNode;
      function predictTree(node: TreeNode; var sample: TDataRow): double;
      procedure freeTreeNode(node: TreeNode);
      procedure freeTree(tree: TDecisionTree);

      { Random Forest Training }
      procedure fit();
      procedure fitTree(treeIndex: integer);

      { Random Forest Prediction }
      function predict(var sample: TDataRow): double;
      function predictClass(var sample: TDataRow): integer;
      procedure predictBatch(var samples: TDataMatrix; nSamples: integer;
                            var predictions: TTargetArray);

      { Out-of-Bag Error }
      function calculateOOBError(): double;

      { Feature Importance }
      procedure calculateFeatureImportance();
      function getFeatureImportance(featureIndex: integer): double;
      procedure printFeatureImportances();

      { Performance Metrics }
      function accuracy(var predictions: TTargetArray; var actual: TTargetArray;
                       nSamples: integer): double;
      function precision(var predictions: TTargetArray; var actual: TTargetArray;
                        nSamples: integer; positiveClass: integer): double;
      function recall(var predictions: TTargetArray; var actual: TTargetArray;
                     nSamples: integer; positiveClass: integer): double;
      function f1Score(var predictions: TTargetArray; var actual: TTargetArray;
                      nSamples: integer; positiveClass: integer): double;
      function meanSquaredError(var predictions: TTargetArray; var actual: TTargetArray;
                               nSamples: integer): double;
      function rSquared(var predictions: TTargetArray; var actual: TTargetArray;
                       nSamples: integer): double;

      { Utility }
      procedure printForestInfo();
      procedure freeForest();

      { Accessors for Facade }
      function getNumTrees(): integer;
      function getNumFeatures(): integer;
      function getNumSamples(): integer;
      function getMaxDepth(): integer;
      function getMinSamplesLeaf(): integer;
      function getMinSamplesSplit(): integer;
      function getMaxFeatures(): integer;
      function getRandomSeed(): integer;
      function getTree(treeId: integer): TDecisionTree;
      function getData(sampleIdx, featureIdx: integer): double;
      function getTarget(sampleIdx: integer): double;
      function getTaskType(): TaskType;
      function getCriterion(): SplitCriterion;
      
      { Tree Management for Facade }
      procedure addNewTree();
      procedure removeTreeAt(treeId: integer);
      procedure retrainTreeAt(treeId: integer);
      procedure setTree(treeId: integer; tree: TDecisionTree);

   end;

   { Facade Types }
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

   TFeatureStats = record
      featureIndex: integer;
      timesUsed: integer;
      treesUsedIn: integer;
      avgImportance: double;
      totalImportance: double;
   end;
   TFeatureStatsArray = array[0..MAX_FEATURE_STATS-1] of TFeatureStats;

   TSampleTrackInfo = record
      sampleIndex: integer;
      treesInfluenced: array[0..MAX_TREES-1] of boolean;
      numTreesInfluenced: integer;
      oobTrees: array[0..MAX_TREES-1] of boolean;
      numOobTrees: integer;
      predictions: array[0..MAX_TREES-1] of double;
   end;

   TOOBTreeInfo = record
      treeId: integer;
      numOobSamples: integer;
      oobError: double;
      oobAccuracy: double;
   end;
   TOOBTreeInfoArray = array[0..MAX_TREES-1] of TOOBTreeInfo;

   TForestComparison = record
      numDifferentPredictions: integer;
      avgPredictionDiff: double;
      forestAAccuracy: double;
      forestBAccuracy: double;
      forestAMSE: double;
      forestBMSE: double;
   end;

   TAggregationMethod = (MajorityVote, WeightedVote, Mean, WeightedMean);

   TRandomForestFacade = object
   private
      forest: TRandomForest;
      forestInitialized: boolean;
      currentAggregation: TAggregationMethod;
      treeWeights: array[0..MAX_TREES-1] of double;
      featureEnabled: array[0..MAX_FEATURES-1] of boolean;
      
      function collectNodeInfo(node: TreeNode; depth: integer; 
                               var nodes: TNodeInfoArray; var count: integer): integer;
      function calculateTreeDepth(node: TreeNode): integer;
      function countLeaves(node: TreeNode): integer;
      procedure collectFeaturesUsed(node: TreeNode; var used: array of boolean);
      function findNodeById(node: TreeNode; targetId: integer; var currentId: integer): TreeNode;
      procedure freeSubtree(node: TreeNode);

   public
      constructor create();
      
      procedure initForest();
      function getForest(): TRandomForest;
      
      procedure setHyperparameter(paramName: string; value: integer);
      procedure setHyperparameterFloat(paramName: string; value: double);
      function getHyperparameter(paramName: string): integer;
      procedure setTaskType(t: TaskType);
      procedure setCriterion(c: SplitCriterion);
      procedure printHyperparameters();
      
      procedure loadData(var inputData: TDataMatrix; var inputTargets: TTargetArray;
                        nSamples, nFeatures: integer);
      procedure trainForest();
      
      function inspectTree(treeId: integer): TTreeInfo;
      procedure printTreeStructure(treeId: integer);
      procedure printNodeDetails(treeId: integer; nodeId: integer);
      function getTreeDepth(treeId: integer): integer;
      function getTreeNumNodes(treeId: integer): integer;
      function getTreeNumLeaves(treeId: integer): integer;
      
      procedure pruneTree(treeId: integer; nodeId: integer);
      procedure modifySplit(treeId: integer; nodeId: integer; newThreshold: double);
      procedure modifyLeafValue(treeId: integer; nodeId: integer; newValue: double);
      procedure convertToLeaf(treeId: integer; nodeId: integer; leafValue: double);
      
      procedure addTree();
      procedure removeTree(treeId: integer);
      procedure replaceTree(treeId: integer);
      procedure retrainTree(treeId: integer);
      function getNumTrees(): integer;
      
      procedure enableFeature(featureIndex: integer);
      procedure disableFeature(featureIndex: integer);
      procedure setFeatureEnabled(featureIndex: integer; enabled: boolean);
      function isFeatureEnabled(featureIndex: integer): boolean;
      procedure resetFeatureFilters();
      function featureUsageSummary(): TFeatureStatsArray;
      procedure printFeatureUsageSummary();
      function getFeatureImportance(featureIndex: integer): double;
      procedure printFeatureImportances();
      
      procedure setAggregationMethod(method: TAggregationMethod);
      function getAggregationMethod(): TAggregationMethod;
      procedure setTreeWeight(treeId: integer; weight: double);
      function getTreeWeight(treeId: integer): double;
      procedure resetTreeWeights();
      function aggregatePredictions(var sample: TDataRow): double;
      
      function predict(var sample: TDataRow): double;
      function predictClass(var sample: TDataRow): integer;
      function predictWithTree(treeId: integer; var sample: TDataRow): double;
      procedure predictBatch(var samples: TDataMatrix; nSamples: integer;
                            var predictions: TTargetArray);
      
      function trackSample(sampleIndex: integer): TSampleTrackInfo;
      procedure printSampleTracking(sampleIndex: integer);
      
      function oobErrorSummary(): TOOBTreeInfoArray;
      procedure printOOBSummary();
      function getGlobalOOBError(): double;
      procedure markProblematicTrees(errorThreshold: double);
      
      function accuracy(var predictions, actual: TTargetArray; nSamples: integer): double;
      function meanSquaredError(var predictions, actual: TTargetArray; nSamples: integer): double;
      function rSquared(var predictions, actual: TTargetArray; nSamples: integer): double;
      function precision(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      function recall(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      function f1Score(var predictions, actual: TTargetArray; nSamples, posClass: integer): double;
      procedure printMetrics(var predictions, actual: TTargetArray; nSamples: integer);
      
      procedure highlightMisclassified(var predictions, actual: TTargetArray; nSamples: integer);
      procedure highlightHighResidual(var predictions, actual: TTargetArray; 
                                      nSamples: integer; threshold: double);
      procedure findWorstTrees(var actual: TTargetArray; nSamples: integer; topN: integer);
      
      procedure visualizeTree(treeId: integer);
      procedure visualizeSplitDistribution(treeId: integer; nodeId: integer);
      procedure printForestOverview();
      procedure printFeatureHeatmap();
      
      procedure swapCriterion(newCriterion: SplitCriterion);
      function compareForests(var otherForest: TRandomForest; 
                             var testData: TDataMatrix; var testTargets: TTargetArray;
                             nSamples: integer): TForestComparison;
      procedure printComparison(var comparison: TForestComparison);
      
      procedure saveModel(filename: string);
      function loadModel(filename: string): boolean;
      
      { JSON serialization methods }
      procedure saveModelToJSON(filename: string);
      function loadModelFromJSON(filename: string): boolean;
      
      { JSON helper functions }
      function taskTypeToStr(t: TaskType): string;
      function criterionToStr(c: SplitCriterion): string;
      function parseTaskType(const s: string): TaskType;
      function parseCriterion(const s: string): SplitCriterion;
      function treeNodeToJSON(node: TreeNode): string;
      function JSONToTreeNode(const json: string): TreeNode;
      
      procedure freeForest();
      end;

{ ============================================================================ }
{ Constructor }
{ ============================================================================ }

constructor TRandomForest.create();
var
   i: integer;
begin
   numTrees := 100;
   maxDepth := MAX_DEPTH_DEFAULT;
   minSamplesLeaf := MIN_SAMPLES_LEAF_DEFAULT;
   minSamplesSplit := MIN_SAMPLES_SPLIT_DEFAULT;
   maxFeatures := 0;
   numFeatures := 0;
   numSamples := 0;
   taskType := Classification;
   criterion := Gini;
   randomSeed := 42;

   for i := 0 to MAX_TREES - 1 do
      trees[i] := nil;

   for i := 0 to MAX_FEATURES - 1 do
      featureImportances[i] := 0.0;

   randomize;
end;

{ ============================================================================ }
{ Hyperparameter Handling }
{ ============================================================================ }

procedure TRandomForest.setNumTrees(n: integer);
begin
   if n > MAX_TREES then
      numTrees := MAX_TREES
   else if n < 1 then
      numTrees := 1
   else
      numTrees := n;
end;

procedure TRandomForest.setMaxDepth(d: integer);
begin
   if d < 1 then
      maxDepth := 1
   else
      maxDepth := d;
end;

procedure TRandomForest.setMinSamplesLeaf(m: integer);
begin
   if m < 1 then
      minSamplesLeaf := 1
   else
      minSamplesLeaf := m;
end;

procedure TRandomForest.setMinSamplesSplit(m: integer);
begin
   if m < 2 then
      minSamplesSplit := 2
   else
      minSamplesSplit := m;
end;

procedure TRandomForest.setMaxFeatures(m: integer);
begin
   maxFeatures := m;
end;

procedure TRandomForest.setTaskType(t: TaskType);
begin
   taskType := t;
   if t = Classification then
      criterion := Gini
   else
      criterion := MSE;
end;

procedure TRandomForest.setCriterion(c: SplitCriterion);
begin
   criterion := c;
end;

procedure TRandomForest.setRandomSeed(seed: longint);
begin
   randomSeed := seed;
   randseed := seed;
end;

{ ============================================================================ }
{ Random Number Generator }
{ ============================================================================ }

function TRandomForest.randomInt(maxVal: integer): integer;
begin
   randomInt := random(maxVal);
end;

function TRandomForest.randomDouble(): double;
begin
   randomDouble := random;
end;

{ ============================================================================ }
{ Data Handling Functions }
{ ============================================================================ }

procedure TRandomForest.loadData(var inputData: TDataMatrix; var inputTargets: TTargetArray;
                                 nSamples, nFeatures: integer);
var
   i, j: integer;
begin
   numSamples := nSamples;
   numFeatures := nFeatures;

   if maxFeatures = 0 then
   begin
      if taskType = Classification then
         maxFeatures := round(sqrt(nFeatures))
      else
         maxFeatures := nFeatures div 3;
      if maxFeatures < 1 then
         maxFeatures := 1;
   end;

   for i := 0 to nSamples - 1 do
   begin
      for j := 0 to nFeatures - 1 do
         data[i][j] := inputData[i][j];
      targets[i] := inputTargets[i];
   end;
end;

procedure TRandomForest.trainTestSplit(var trainIndices, testIndices: TIndexArray;
                                       var numTrain, numTest: integer; testRatio: double);
var
   i, j, temp: integer;
   shuffled: TIndexArray;
begin
   for i := 0 to numSamples - 1 do
      shuffled[i] := i;

   for i := numSamples - 1 downto 1 do
   begin
      j := randomInt(i + 1);
      temp := shuffled[i];
      shuffled[i] := shuffled[j];
      shuffled[j] := temp;
   end;

   numTest := round(numSamples * testRatio);
   numTrain := numSamples - numTest;

   for i := 0 to numTrain - 1 do
      trainIndices[i] := shuffled[i];

   for i := 0 to numTest - 1 do
      testIndices[i] := shuffled[numTrain + i];
end;

procedure TRandomForest.bootstrap(var sampleIndices: TIndexArray; var numBootstrap: integer;
                                  var oobMask: TBoolArray);
var
   i, idx: integer;
begin
   numBootstrap := numSamples;

   for i := 0 to numSamples - 1 do
      oobMask[i] := true;

   for i := 0 to numBootstrap - 1 do
   begin
      idx := randomInt(numSamples);
      sampleIndices[i] := idx;
      oobMask[idx] := false;
   end;
end;

procedure TRandomForest.selectFeatureSubset(var featureIndices: TFeatureArray;
                                            var numSelected: integer);
var
   i, j, temp: integer;
   available: TFeatureArray;
begin
   for i := 0 to numFeatures - 1 do
      available[i] := i;

   for i := numFeatures - 1 downto 1 do
   begin
      j := randomInt(i + 1);
      temp := available[i];
      available[i] := available[j];
      available[j] := temp;
   end;

   numSelected := maxFeatures;
   if numSelected > numFeatures then
      numSelected := numFeatures;

   for i := 0 to numSelected - 1 do
      featureIndices[i] := available[i];
end;

{ ============================================================================ }
{ Decision Tree - Impurity Functions }
{ ============================================================================ }

function TRandomForest.calculateGini(var indices: TIndexArray; numIndices: integer): double;
var
   i: integer;
   classCount: array[0..99] of integer;
   numClasses, classLabel: integer;
   prob, gini: double;
begin
   if numIndices = 0 then
   begin
      calculateGini := 0.0;
      exit;
   end;

   for i := 0 to 99 do
      classCount[i] := 0;

   numClasses := 0;
   for i := 0 to numIndices - 1 do
   begin
      classLabel := round(targets[indices[i]]);
      if classLabel > numClasses then
         numClasses := classLabel;
      inc(classCount[classLabel]);
   end;

   gini := 1.0;
   for i := 0 to numClasses do
   begin
      prob := classCount[i] / numIndices;
      gini := gini - (prob * prob);
   end;

   calculateGini := gini;
end;

function TRandomForest.calculateEntropy(var indices: TIndexArray; numIndices: integer): double;
var
   i: integer;
   classCount: array[0..99] of integer;
   numClasses, classLabel: integer;
   prob, entropy: double;
begin
   if numIndices = 0 then
   begin
      calculateEntropy := 0.0;
      exit;
   end;

   for i := 0 to 99 do
      classCount[i] := 0;

   numClasses := 0;
   for i := 0 to numIndices - 1 do
   begin
      classLabel := round(targets[indices[i]]);
      if classLabel > numClasses then
         numClasses := classLabel;
      inc(classCount[classLabel]);
   end;

   entropy := 0.0;
   for i := 0 to numClasses do
   begin
      if classCount[i] > 0 then
      begin
         prob := classCount[i] / numIndices;
         entropy := entropy - (prob * ln(prob) / ln(2));
      end;
   end;

   calculateEntropy := entropy;
end;

function TRandomForest.calculateMSE(var indices: TIndexArray; numIndices: integer): double;
var
   i: integer;
   mean, mse, diff: double;
begin
   if numIndices = 0 then
   begin
      calculateMSE := 0.0;
      exit;
   end;

   mean := 0.0;
   for i := 0 to numIndices - 1 do
      mean := mean + targets[indices[i]];
   mean := mean / numIndices;

   mse := 0.0;
   for i := 0 to numIndices - 1 do
   begin
      diff := targets[indices[i]] - mean;
      mse := mse + (diff * diff);
   end;

   calculateMSE := mse / numIndices;
end;

function TRandomForest.calculateVariance(var indices: TIndexArray; numIndices: integer): double;
begin
   calculateVariance := calculateMSE(indices, numIndices);
end;

function TRandomForest.calculateImpurity(var indices: TIndexArray; numIndices: integer): double;
begin
   case criterion of
      Gini: calculateImpurity := calculateGini(indices, numIndices);
      Entropy: calculateImpurity := calculateEntropy(indices, numIndices);
      MSE: calculateImpurity := calculateMSE(indices, numIndices);
      VarianceReduction: calculateImpurity := calculateVariance(indices, numIndices);
   else
      calculateImpurity := calculateGini(indices, numIndices);
   end;
end;

{ ============================================================================ }
{ Decision Tree - Split Functions }
{ ============================================================================ }

function TRandomForest.findBestSplit(var indices: TIndexArray; numIndices: integer;
                                     var featureIndices: TFeatureArray; nFeatures: integer;
                                     var bestFeature: integer; var bestThreshold: double;
                                     var bestGain: double): boolean;
var
   f, i, j, feat: integer;
   leftIndices, rightIndices: TIndexArray;
   numLeft, numRight: integer;
   threshold, gain, parentImpurity: double;
   leftImpurity, rightImpurity: double;
   values: array[0..MAX_SAMPLES-1] of double;
   sortedIndices: TIndexArray;
   temp: integer;
begin
   findBestSplit := false;
   bestGain := 0.0;
   bestFeature := -1;
   bestThreshold := 0.0;

   if numIndices < minSamplesSplit then
      exit;

   parentImpurity := calculateImpurity(indices, numIndices);

   for f := 0 to nFeatures - 1 do
   begin
      feat := featureIndices[f];

      for i := 0 to numIndices - 1 do
      begin
         values[i] := data[indices[i]][feat];
         sortedIndices[i] := i;
      end;

      for i := 0 to numIndices - 2 do
      begin
         for j := i + 1 to numIndices - 1 do
         begin
            if values[sortedIndices[j]] < values[sortedIndices[i]] then
            begin
               temp := sortedIndices[i];
               sortedIndices[i] := sortedIndices[j];
               sortedIndices[j] := temp;
            end;
         end;
      end;

      for i := 0 to numIndices - 2 do
      begin
         if values[sortedIndices[i]] = values[sortedIndices[i + 1]] then
            continue;

         threshold := (values[sortedIndices[i]] + values[sortedIndices[i + 1]]) / 2;

         numLeft := 0;
         numRight := 0;
         for j := 0 to numIndices - 1 do
         begin
            if data[indices[j]][feat] <= threshold then
            begin
               leftIndices[numLeft] := indices[j];
               inc(numLeft);
            end
            else
            begin
               rightIndices[numRight] := indices[j];
               inc(numRight);
            end;
         end;

         if (numLeft < minSamplesLeaf) or (numRight < minSamplesLeaf) then
            continue;

         leftImpurity := calculateImpurity(leftIndices, numLeft);
         rightImpurity := calculateImpurity(rightIndices, numRight);

         gain := parentImpurity -
                 (numLeft / numIndices) * leftImpurity -
                 (numRight / numIndices) * rightImpurity;

         if gain > bestGain then
         begin
            bestGain := gain;
            bestFeature := feat;
            bestThreshold := threshold;
            findBestSplit := true;
         end;
      end;
   end;
end;

{ ============================================================================ }
{ Decision Tree - Leaf Functions }
{ ============================================================================ }

function TRandomForest.getMajorityClass(var indices: TIndexArray; numIndices: integer): integer;
var
   i: integer;
   classCount: array[0..99] of integer;
   maxCount, maxClass, classLabel: integer;
begin
   for i := 0 to 99 do
      classCount[i] := 0;

   for i := 0 to numIndices - 1 do
   begin
      classLabel := round(targets[indices[i]]);
      inc(classCount[classLabel]);
   end;

   maxCount := 0;
   maxClass := 0;
   for i := 0 to 99 do
   begin
      if classCount[i] > maxCount then
      begin
         maxCount := classCount[i];
         maxClass := i;
      end;
   end;

   getMajorityClass := maxClass;
end;

function TRandomForest.getMeanTarget(var indices: TIndexArray; numIndices: integer): double;
var
   i: integer;
   sum: double;
begin
   if numIndices = 0 then
   begin
      getMeanTarget := 0.0;
      exit;
   end;

   sum := 0.0;
   for i := 0 to numIndices - 1 do
      sum := sum + targets[indices[i]];

   getMeanTarget := sum / numIndices;
end;

function TRandomForest.createLeafNode(var indices: TIndexArray; numIndices: integer): TreeNode;
var
   node: TreeNode;
begin
   new(node);
   node^.isLeaf := true;
   node^.featureIndex := -1;
   node^.threshold := 0.0;
   node^.numSamples := numIndices;
   node^.impurity := calculateImpurity(indices, numIndices);
   node^.left := nil;
   node^.right := nil;

   if taskType = Classification then
   begin
      node^.classLabel := getMajorityClass(indices, numIndices);
      node^.prediction := node^.classLabel;
   end
   else
   begin
      node^.prediction := getMeanTarget(indices, numIndices);
      node^.classLabel := round(node^.prediction);
   end;

   createLeafNode := node;
end;

{ ============================================================================ }
{ Decision Tree - Stopping Conditions }
{ ============================================================================ }

function TRandomForest.shouldStop(depth, numIndices: integer; impurity: double): boolean;
begin
   shouldStop := false;

   if depth >= maxDepth then
      shouldStop := true
   else if numIndices < minSamplesSplit then
      shouldStop := true
   else if numIndices <= minSamplesLeaf then
      shouldStop := true
   else if impurity < 1e-10 then
      shouldStop := true;
end;

{ ============================================================================ }
{ Decision Tree - Tree Building }
{ ============================================================================ }

function TRandomForest.buildTree(var indices: TIndexArray; numIndices: integer;
                                 depth: integer; tree: TDecisionTree): TreeNode;
var
   node: TreeNode;
   featureIndices: TFeatureArray;
   numSelectedFeatures: integer;
   bestFeature: integer;
   bestThreshold, bestGain: double;
   leftIndices, rightIndices: TIndexArray;
   numLeft, numRight, i: integer;
   currentImpurity: double;
begin
   currentImpurity := calculateImpurity(indices, numIndices);

   if shouldStop(depth, numIndices, currentImpurity) then
   begin
      buildTree := createLeafNode(indices, numIndices);
      exit;
   end;

   selectFeatureSubset(featureIndices, numSelectedFeatures);

   if not findBestSplit(indices, numIndices, featureIndices, numSelectedFeatures,
                        bestFeature, bestThreshold, bestGain) then
   begin
      buildTree := createLeafNode(indices, numIndices);
      exit;
   end;

   new(node);
   node^.isLeaf := false;
   node^.featureIndex := bestFeature;
   node^.threshold := bestThreshold;
   node^.numSamples := numIndices;
   node^.impurity := currentImpurity;

   if taskType = Classification then
      node^.classLabel := getMajorityClass(indices, numIndices)
   else
      node^.prediction := getMeanTarget(indices, numIndices);

   numLeft := 0;
   numRight := 0;
   for i := 0 to numIndices - 1 do
   begin
      if data[indices[i]][bestFeature] <= bestThreshold then
      begin
         leftIndices[numLeft] := indices[i];
         inc(numLeft);
      end
      else
      begin
         rightIndices[numRight] := indices[i];
         inc(numRight);
      end;
   end;

   featureImportances[bestFeature] := featureImportances[bestFeature] +
      (numIndices * currentImpurity -
       numLeft * calculateImpurity(leftIndices, numLeft) -
       numRight * calculateImpurity(rightIndices, numRight));

   node^.left := buildTree(leftIndices, numLeft, depth + 1, tree);
   node^.right := buildTree(rightIndices, numRight, depth + 1, tree);

   buildTree := node;
end;

{ ============================================================================ }
{ Decision Tree - Prediction }
{ ============================================================================ }

function TRandomForest.predictTree(node: TreeNode; var sample: TDataRow): double;
begin
   if node = nil then
   begin
      predictTree := 0.0;
      exit;
   end;

   if node^.isLeaf then
   begin
      predictTree := node^.prediction;
      exit;
   end;

   if sample[node^.featureIndex] <= node^.threshold then
      predictTree := predictTree(node^.left, sample)
   else
      predictTree := predictTree(node^.right, sample);
end;

{ ============================================================================ }
{ Decision Tree - Memory Management }
{ ============================================================================ }

procedure TRandomForest.freeTreeNode(node: TreeNode);
begin
   if node = nil then
      exit;

   freeTreeNode(node^.left);
   freeTreeNode(node^.right);
   dispose(node);
end;

procedure TRandomForest.freeTree(tree: TDecisionTree);
begin
   if tree = nil then
      exit;

   freeTreeNode(tree^.root);
   dispose(tree);
end;

{ ============================================================================ }
{ Random Forest - Training }
{ ============================================================================ }

procedure TRandomForest.fit();
var
   i: integer;
begin
   for i := 0 to MAX_FEATURES - 1 do
      featureImportances[i] := 0.0;

   for i := 0 to numTrees - 1 do
      fitTree(i);

   calculateFeatureImportance();
end;

procedure TRandomForest.fitTree(treeIndex: integer);
var
   tree: TDecisionTree;
   sampleIndices: TIndexArray;
   numBootstrap, i: integer;
   oobMask: TBoolArray;
begin
   new(tree);
   tree^.maxDepth := maxDepth;
   tree^.minSamplesLeaf := minSamplesLeaf;
   tree^.minSamplesSplit := minSamplesSplit;
   tree^.maxFeatures := maxFeatures;
   tree^.taskType := taskType;
   tree^.criterion := criterion;

   bootstrap(sampleIndices, numBootstrap, oobMask);

   for i := 0 to numSamples - 1 do
      tree^.oobIndices[i] := oobMask[i];

   tree^.numOobIndices := 0;
   for i := 0 to numSamples - 1 do
      if oobMask[i] then
         inc(tree^.numOobIndices);

   tree^.root := buildTree(sampleIndices, numBootstrap, 0, tree);

   trees[treeIndex] := tree;
end;

{ ============================================================================ }
{ Random Forest - Prediction }
{ ============================================================================ }

function TRandomForest.predict(var sample: TDataRow): double;
var
   i: integer;
   sum: double;
   votes: array[0..99] of integer;
   maxVotes, maxClass, classLabel: integer;
begin
   if taskType = Regression then
   begin
      sum := 0.0;
      for i := 0 to numTrees - 1 do
         sum := sum + predictTree(trees[i]^.root, sample);
      predict := sum / numTrees;
   end
   else
   begin
      for i := 0 to 99 do
         votes[i] := 0;

      for i := 0 to numTrees - 1 do
      begin
         classLabel := round(predictTree(trees[i]^.root, sample));
         if (classLabel >= 0) and (classLabel <= 99) then
            inc(votes[classLabel]);
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

      predict := maxClass;
   end;
end;

function TRandomForest.predictClass(var sample: TDataRow): integer;
begin
   predictClass := round(predict(sample));
end;

procedure TRandomForest.predictBatch(var samples: TDataMatrix; nSamples: integer;
                                     var predictions: TTargetArray);
var
   i: integer;
begin
   for i := 0 to nSamples - 1 do
      predictions[i] := predict(samples[i]);
end;

{ ============================================================================ }
{ Out-of-Bag Error }
{ ============================================================================ }

function TRandomForest.calculateOOBError(): double;
var
   i, t, count: integer;
   predictions: array[0..MAX_SAMPLES-1] of double;
   predCounts: array[0..MAX_SAMPLES-1] of integer;
   pred: double;
   error, diff: double;
   votes: array[0..MAX_SAMPLES-1, 0..99] of integer;
   maxVotes, maxClass, j: integer;
begin
   for i := 0 to numSamples - 1 do
   begin
      predictions[i] := 0.0;
      predCounts[i] := 0;
      for j := 0 to 99 do
         votes[i][j] := 0;
   end;

   for t := 0 to numTrees - 1 do
   begin
      for i := 0 to numSamples - 1 do
      begin
         if trees[t]^.oobIndices[i] then
         begin
            pred := predictTree(trees[t]^.root, data[i]);
            if taskType = Regression then
               predictions[i] := predictions[i] + pred
            else
            begin
               j := round(pred);
               if (j >= 0) and (j <= 99) then
                  inc(votes[i][j]);
            end;
            inc(predCounts[i]);
         end;
      end;
   end;

   error := 0.0;
   count := 0;

   for i := 0 to numSamples - 1 do
   begin
      if predCounts[i] > 0 then
      begin
         if taskType = Regression then
         begin
            pred := predictions[i] / predCounts[i];
            diff := pred - targets[i];
            error := error + (diff * diff);
         end
         else
         begin
            maxVotes := 0;
            maxClass := 0;
            for j := 0 to 99 do
            begin
               if votes[i][j] > maxVotes then
               begin
                  maxVotes := votes[i][j];
                  maxClass := j;
               end;
            end;
            if maxClass <> round(targets[i]) then
               error := error + 1.0;
         end;
         inc(count);
      end;
   end;

   if count > 0 then
      calculateOOBError := error / count
   else
      calculateOOBError := 0.0;
end;

{ ============================================================================ }
{ Feature Importance }
{ ============================================================================ }

procedure TRandomForest.calculateFeatureImportance();
var
   i: integer;
   total: double;
begin
   total := 0.0;
   for i := 0 to numFeatures - 1 do
      total := total + featureImportances[i];

   if total > 0 then
   begin
      for i := 0 to numFeatures - 1 do
         featureImportances[i] := featureImportances[i] / total;
   end;
end;

function TRandomForest.getFeatureImportance(featureIndex: integer): double;
begin
   if (featureIndex >= 0) and (featureIndex < numFeatures) then
      getFeatureImportance := featureImportances[featureIndex]
   else
      getFeatureImportance := 0.0;
end;

procedure TRandomForest.printFeatureImportances();
var
   i: integer;
begin
   writeln('Feature Importances:');
   for i := 0 to numFeatures - 1 do
      writeln('  Feature ', i, ': ', featureImportances[i]:0:4);
end;

{ ============================================================================ }
{ Performance Metrics }
{ ============================================================================ }

function TRandomForest.accuracy(var predictions: TTargetArray; var actual: TTargetArray;
                                nSamples: integer): double;
var
   i, correct: integer;
begin
   correct := 0;
   for i := 0 to nSamples - 1 do
   begin
      if round(predictions[i]) = round(actual[i]) then
         inc(correct);
   end;
   accuracy := correct / nSamples;
end;

function TRandomForest.precision(var predictions: TTargetArray; var actual: TTargetArray;
                                 nSamples: integer; positiveClass: integer): double;
var
   i, tp, fp: integer;
begin
   tp := 0;
   fp := 0;
   for i := 0 to nSamples - 1 do
   begin
      if round(predictions[i]) = positiveClass then
      begin
         if round(actual[i]) = positiveClass then
            inc(tp)
         else
            inc(fp);
      end;
   end;

   if (tp + fp) > 0 then
      precision := tp / (tp + fp)
   else
      precision := 0.0;
end;

function TRandomForest.recall(var predictions: TTargetArray; var actual: TTargetArray;
                              nSamples: integer; positiveClass: integer): double;
var
   i, tp, fn: integer;
begin
   tp := 0;
   fn := 0;
   for i := 0 to nSamples - 1 do
   begin
      if round(actual[i]) = positiveClass then
      begin
         if round(predictions[i]) = positiveClass then
            inc(tp)
         else
            inc(fn);
      end;
   end;

   if (tp + fn) > 0 then
      recall := tp / (tp + fn)
   else
      recall := 0.0;
end;

function TRandomForest.f1Score(var predictions: TTargetArray; var actual: TTargetArray;
                               nSamples: integer; positiveClass: integer): double;
var
   p, r: double;
begin
   p := precision(predictions, actual, nSamples, positiveClass);
   r := recall(predictions, actual, nSamples, positiveClass);

   if (p + r) > 0 then
      f1Score := 2 * p * r / (p + r)
   else
      f1Score := 0.0;
end;

function TRandomForest.meanSquaredError(var predictions: TTargetArray; var actual: TTargetArray;
                                        nSamples: integer): double;
var
   i: integer;
   mse, diff: double;
begin
   mse := 0.0;
   for i := 0 to nSamples - 1 do
   begin
      diff := predictions[i] - actual[i];
      mse := mse + (diff * diff);
   end;
   meanSquaredError := mse / nSamples;
end;

function TRandomForest.rSquared(var predictions: TTargetArray; var actual: TTargetArray;
                                nSamples: integer): double;
var
   i: integer;
   mean, ssRes, ssTot, diff: double;
begin
   mean := 0.0;
   for i := 0 to nSamples - 1 do
      mean := mean + actual[i];
   mean := mean / nSamples;

   ssRes := 0.0;
   ssTot := 0.0;
   for i := 0 to nSamples - 1 do
   begin
      diff := predictions[i] - actual[i];
      ssRes := ssRes + (diff * diff);
      diff := actual[i] - mean;
      ssTot := ssTot + (diff * diff);
   end;

   if ssTot > 0 then
      rSquared := 1.0 - (ssRes / ssTot)
   else
      rSquared := 0.0;
end;

{ ============================================================================ }
{ Utility Functions }
{ ============================================================================ }

procedure TRandomForest.printForestInfo();
begin
   writeln('Random Forest Configuration:');
   writeln('  Number of Trees: ', numTrees);
   writeln('  Max Depth: ', maxDepth);
   writeln('  Min Samples Leaf: ', minSamplesLeaf);
   writeln('  Min Samples Split: ', minSamplesSplit);
   writeln('  Max Features: ', maxFeatures);
   writeln('  Number of Features: ', numFeatures);
   writeln('  Number of Samples: ', numSamples);
   if taskType = Classification then
      writeln('  Task Type: Classification')
   else
      writeln('  Task Type: Regression');
   case criterion of
      Gini: writeln('  Criterion: Gini');
      Entropy: writeln('  Criterion: Entropy');
      MSE: writeln('  Criterion: MSE');
      VarianceReduction: writeln('  Criterion: Variance Reduction');
   end;
end;

procedure TRandomForest.freeForest();
var
   i: integer;
begin
   for i := 0 to numTrees - 1 do
   begin
      if trees[i] <> nil then
      begin
         freeTree(trees[i]);
         trees[i] := nil;
      end;
   end;
end;

{ ============================================================================ }
{ Accessors for Facade }
{ ============================================================================ }

function TRandomForest.getNumTrees(): integer;
begin
   getNumTrees := numTrees;
end;

function TRandomForest.getNumFeatures(): integer;
begin
   getNumFeatures := numFeatures;
end;

function TRandomForest.getNumSamples(): integer;
begin
   getNumSamples := numSamples;
end;

function TRandomForest.getMaxDepth(): integer;
begin
   getMaxDepth := maxDepth;
end;

function TRandomForest.getTree(treeId: integer): TDecisionTree;
begin
   if (treeId >= 0) and (treeId < numTrees) then
      getTree := trees[treeId]
   else
      getTree := nil;
end;

function TRandomForest.getData(sampleIdx, featureIdx: integer): double;
begin
   if (sampleIdx >= 0) and (sampleIdx < numSamples) and
      (featureIdx >= 0) and (featureIdx < numFeatures) then
      getData := data[sampleIdx][featureIdx]
   else
      getData := 0.0;
end;

function TRandomForest.getTarget(sampleIdx: integer): double;
begin
   if (sampleIdx >= 0) and (sampleIdx < numSamples) then
      getTarget := targets[sampleIdx]
   else
      getTarget := 0.0;
end;

function TRandomForest.getTaskType(): TaskType;
begin
   getTaskType := taskType;
end;

function TRandomForest.getCriterion(): SplitCriterion;
begin
   getCriterion := criterion;
end;

function TRandomForest.getMinSamplesLeaf(): integer;
begin
   getMinSamplesLeaf := minSamplesLeaf;
end;

function TRandomForest.getMinSamplesSplit(): integer;
begin
   getMinSamplesSplit := minSamplesSplit;
end;

function TRandomForest.getMaxFeatures(): integer;
begin
   getMaxFeatures := maxFeatures;
end;

function TRandomForest.getRandomSeed(): integer;
begin
   getRandomSeed := randomSeed;
end;

{ ============================================================================ }
{ Tree Management for Facade }
{ ============================================================================ }

procedure TRandomForest.addNewTree();
begin
   if numTrees >= MAX_TREES then
   begin
      writeln('Maximum number of trees reached');
      exit;
   end;
   
   fitTree(numTrees);
   inc(numTrees);
end;

procedure TRandomForest.removeTreeAt(treeId: integer);
var
   i: integer;
begin
   if (treeId < 0) or (treeId >= numTrees) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   freeTree(trees[treeId]);
   
   for i := treeId to numTrees - 2 do
      trees[i] := trees[i + 1];
   
   trees[numTrees - 1] := nil;
   dec(numTrees);
end;

procedure TRandomForest.retrainTreeAt(treeId: integer);
begin
   if (treeId < 0) or (treeId >= numTrees) then
   begin
      writeln('Invalid tree ID: ', treeId);
      exit;
   end;
   
   freeTree(trees[treeId]);
   trees[treeId] := nil;
   fitTree(treeId);
end;

procedure TRandomForest.setTree(treeId: integer; tree: TDecisionTree);
begin
   if (treeId >= 0) and (treeId < MAX_TREES) then
      trees[treeId] := tree;
end;

{ ============================================================================ }
{ TRandomForestFacade - Constructor }
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
{ TRandomForestFacade - Internal Helpers }
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
{ TRandomForestFacade - Initialization }
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
{ TRandomForestFacade - Hyperparameter Control }
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
{ TRandomForestFacade - Data Handling }
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
{ TRandomForestFacade - Tree-Level Inspection }
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
{ TRandomForestFacade - Tree-Level Manipulation }
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
{ TRandomForestFacade - Forest-Level Controls }
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
{ TRandomForestFacade - Feature Controls }
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
{ TRandomForestFacade - Aggregation Control }
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
            if (forest.getTree(i) <> nil) and (forest.getTree(i)^.root <> nil) then
            begin
               classLabel := round(forest.predictTree(forest.getTree(i)^.root, sample));
               if (classLabel >= 0) and (classLabel <= 99) then
                  votes[classLabel] := votes[classLabel] + 1;
            end;
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
            if (forest.getTree(i) <> nil) and (forest.getTree(i)^.root <> nil) then
            begin
               classLabel := round(forest.predictTree(forest.getTree(i)^.root, sample));
               if (classLabel >= 0) and (classLabel <= 99) then
                  votes[classLabel] := votes[classLabel] + treeWeights[i];
            end;
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
         begin
            if (forest.getTree(i) <> nil) and (forest.getTree(i)^.root <> nil) then
               sum := sum + forest.predictTree(forest.getTree(i)^.root, sample);
         end;
         if getNumTrees() > 0 then
            aggregatePredictions := sum / getNumTrees()
         else
            aggregatePredictions := 0;
      end;
      
      WeightedMean:
      begin
         sum := 0;
         weightSum := 0;
         for i := 0 to getNumTrees() - 1 do
         begin
            if (forest.getTree(i) <> nil) and (forest.getTree(i)^.root <> nil) then
            begin
               pred := forest.predictTree(forest.getTree(i)^.root, sample);
               sum := sum + pred * treeWeights[i];
               weightSum := weightSum + treeWeights[i];
            end;
         end;
         if weightSum > 0 then
            aggregatePredictions := sum / weightSum
         else
            aggregatePredictions := 0;
      end;
   end;
end;

{ ============================================================================ }
{ TRandomForestFacade - Prediction }
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
{ TRandomForestFacade - Sample Tracking }
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
{ TRandomForestFacade - OOB Analysis }
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
{ TRandomForestFacade - Diagnostics & Metrics }
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
{ TRandomForestFacade - Error Analysis }
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
{ TRandomForestFacade - Visualization }
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
{ TRandomForestFacade - Advanced / Experimental }
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
{ TRandomForestFacade - Save/Load Model }
{ ============================================================================ }

procedure SaveTreeNodeRec(var f: file; node: TreeNode);
var
   hasLeft, hasRight: byte;
begin
   if node = nil then exit;
   
   blockwrite(f, node^.isLeaf, sizeof(boolean));
   blockwrite(f, node^.featureIndex, sizeof(integer));
   blockwrite(f, node^.threshold, sizeof(double));
   blockwrite(f, node^.prediction, sizeof(double));
   blockwrite(f, node^.classLabel, sizeof(integer));
   blockwrite(f, node^.impurity, sizeof(double));
   blockwrite(f, node^.numSamples, sizeof(integer));
   
   if node^.left <> nil then hasLeft := 1 else hasLeft := 0;
   if node^.right <> nil then hasRight := 1 else hasRight := 0;
   blockwrite(f, hasLeft, 1);
   blockwrite(f, hasRight, 1);
   
   if node^.left <> nil then SaveTreeNodeRec(f, node^.left);
   if node^.right <> nil then SaveTreeNodeRec(f, node^.right);
end;

function LoadTreeNodeRec(var f: file): TreeNode;
var
   node: TreeNode;
   hasLeft, hasRight: byte;
begin
   new(node);
   blockread(f, node^.isLeaf, sizeof(boolean));
   blockread(f, node^.featureIndex, sizeof(integer));
   blockread(f, node^.threshold, sizeof(double));
   blockread(f, node^.prediction, sizeof(double));
   blockread(f, node^.classLabel, sizeof(integer));
   blockread(f, node^.impurity, sizeof(double));
   blockread(f, node^.numSamples, sizeof(integer));
   
   blockread(f, hasLeft, 1);
   blockread(f, hasRight, 1);
   
   if hasLeft = 1 then
      node^.left := LoadTreeNodeRec(f)
   else
      node^.left := nil;
   
   if hasRight = 1 then
      node^.right := LoadTreeNodeRec(f)
   else
      node^.right := nil;
   
   LoadTreeNodeRec := node;
end;

procedure TRandomForestFacade.saveModel(filename: string);
var
   f: file;
   i: integer;
   numT, numF, numS, maxD: integer;
   tt: TaskType;
   sc: SplitCriterion;
   tree: TDecisionTree;
   magic: longint;
   ext: string;
begin
   { Check if filename ends with .json - use JSON format }
   ext := LowerCase(ExtractFileExt(filename));
   if ext = '.json' then
   begin
      saveModelToJSON(filename);
      exit;
   end;
   
   assign(f, filename);
   rewrite(f, 1);
   
   magic := $52464D44;
   blockwrite(f, magic, sizeof(longint));
   
   numT := forest.getNumTrees();
   numF := forest.getNumFeatures();
   numS := forest.getNumSamples();
   maxD := forest.getMaxDepth();
   tt := forest.getTaskType();
   sc := forest.getCriterion();
   
   blockwrite(f, numT, sizeof(integer));
   blockwrite(f, numF, sizeof(integer));
   blockwrite(f, numS, sizeof(integer));
   blockwrite(f, maxD, sizeof(integer));
   blockwrite(f, tt, sizeof(TaskType));
   blockwrite(f, sc, sizeof(SplitCriterion));
   
   for i := 0 to numT - 1 do
   begin
      tree := forest.getTree(i);
      if tree <> nil then
         SaveTreeNodeRec(f, tree^.root);
   end;
   
   close(f);
   writeln('Model saved to ', filename, ' (', numT, ' trees)');
end;

function TRandomForestFacade.loadModel(filename: string): boolean;
var
   f: file;
   i: integer;
   numT, numF, numS, maxD: integer;
   tt: TaskType;
   sc: SplitCriterion;
   tree: TDecisionTree;
   magic: longint;
begin
   loadModel := false;
   
   { Try JSON first }
   if loadModelFromJSON(filename) then
   begin
      loadModel := true;
      exit;
   end;
   
   { Fall back to binary format }
   assign(f, filename);
   {$I-}
   reset(f, 1);
   {$I+}
   if IOResult <> 0 then
   begin
      writeln('Error: Cannot open model file: ', filename);
      exit;
   end;
   
   blockread(f, magic, sizeof(longint));
   if magic <> $52464D44 then
   begin
      writeln('Error: Invalid model file format');
      close(f);
      exit;
   end;
   
   blockread(f, numT, sizeof(integer));
   blockread(f, numF, sizeof(integer));
   blockread(f, numS, sizeof(integer));
   blockread(f, maxD, sizeof(integer));
   blockread(f, tt, sizeof(TaskType));
   blockread(f, sc, sizeof(SplitCriterion));
   
   forest.create();
   forest.setNumTrees(numT);
   forest.setMaxDepth(maxD);
   forest.setTaskType(tt);
   forest.setCriterion(sc);
   
   for i := 0 to numT - 1 do
   begin
      new(tree);
      tree^.maxDepth := maxD;
      tree^.taskType := tt;
      tree^.criterion := sc;
      tree^.root := LoadTreeNodeRec(f);
      forest.setTree(i, tree);
   end;
   
   forestInitialized := true;
   close(f);
   loadModel := true;
   writeln('Model loaded from ', filename, ' (', numT, ' trees)');
end;

{ ============================================================================ }
{ TRandomForestFacade - JSON Serialization }
{ ============================================================================ }

function TRandomForestFacade.taskTypeToStr(t: TaskType): string;
begin
   if t = Classification then
      taskTypeToStr := 'classification'
   else
      taskTypeToStr := 'regression';
end;

function TRandomForestFacade.criterionToStr(c: SplitCriterion): string;
begin
   case c of
      Gini: criterionToStr := 'gini';
      Entropy: criterionToStr := 'entropy';
      MSE: criterionToStr := 'mse';
      VarianceReduction: criterionToStr := 'variance';
   else
      criterionToStr := 'gini';
   end;
end;

function TRandomForestFacade.parseTaskType(const s: string): TaskType;
begin
   if lowercase(s) = 'regression' then
      parseTaskType := Regression
   else
      parseTaskType := Classification;
end;

function TRandomForestFacade.parseCriterion(const s: string): SplitCriterion;
begin
    if lowercase(s) = 'gini' then
       parseCriterion := Gini
    else if lowercase(s) = 'entropy' then
       parseCriterion := Entropy
    else if lowercase(s) = 'mse' then
       parseCriterion := MSE
    else if lowercase(s) = 'variance' then
       parseCriterion := VarianceReduction
    else
       parseCriterion := Gini;
end;

function TRandomForestFacade.treeNodeToJSON(node: TreeNode): string;
var
    leftJSON, rightJSON, impurity_str, threshold_str: string;
begin
    if node = nil then
    begin
        Result := 'null';
        Exit;
    end;
    
    Result := '{';
    Result := Result + '"isLeaf":' + BoolToStr(node^.isLeaf, 'true', 'false');
    
    if node^.isLeaf then
    begin
        Result := Result + ',"prediction":' + FloatToStr(node^.prediction);
        Result := Result + ',"classLabel":' + IntToStr(node^.classLabel);
    end
    else
    begin
        Result := Result + ',"featureIndex":' + IntToStr(node^.featureIndex);
        
        threshold_str := FloatToStr(node^.threshold);
        Result := Result + ',"threshold":' + threshold_str;
        
        impurity_str := FloatToStr(node^.impurity);
        Result := Result + ',"impurity":' + impurity_str;
        
        Result := Result + ',"numSamples":' + IntToStr(node^.numSamples);
        
        leftJSON := treeNodeToJSON(node^.left);
        Result := Result + ',"left":' + leftJSON;
        
        rightJSON := treeNodeToJSON(node^.right);
        Result := Result + ',"right":' + rightJSON;
    end;
    
    Result := Result + '}';
end;

function TRandomForestFacade.JSONToTreeNode(const json: string): TreeNode;
var
    node: TreeNode;
    isLeaf: boolean;
    s: string;
    
    function ExtractJSONValue(const json: string; const key: string): string;
    var
       KeyPos, ColonPos, QuotePos1, QuotePos2, StartPos, EndPos: integer;
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
    
    function ExtractJSONSubObject(const json: string; const key: string): string;
    var
        KeyPos, ColonPos, StartPos, BraceCount, i: integer;
    begin
        KeyPos := Pos('"' + key + '"', json);
        if KeyPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        ColonPos := PosEx(':', json, KeyPos);
        if ColonPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        StartPos := ColonPos + 1;
        while (StartPos <= Length(json)) and (json[StartPos] in [' ', #9, #10, #13]) do
            Inc(StartPos);
        
        if (StartPos > Length(json)) or (json[StartPos] <> '{') then
        begin
            Result := '';
            Exit;
        end;
        
        BraceCount := 0;
        i := StartPos;
        while i <= Length(json) do
        begin
            if json[i] = '{' then
                Inc(BraceCount)
            else if json[i] = '}' then
            begin
                Dec(BraceCount);
                if BraceCount = 0 then
                begin
                    Result := Copy(json, StartPos, i - StartPos + 1);
                    Exit;
                end;
            end;
            Inc(i);
        end;
        Result := '';
    end;

begin
    if Trim(json) = 'null' then
    begin
        Result := nil;
        Exit;
    end;
    
    New(node);
    
    s := ExtractJSONValue(json, 'isLeaf');
    isLeaf := (s = 'true');
    node^.isLeaf := isLeaf;
    
    if isLeaf then
    begin
        s := ExtractJSONValue(json, 'prediction');
        if s <> '' then
            node^.prediction := StrToFloat(s);
        
        s := ExtractJSONValue(json, 'classLabel');
        if s <> '' then
            node^.classLabel := StrToInt(s);
        
        node^.left := nil;
        node^.right := nil;
    end
    else
    begin
        s := ExtractJSONValue(json, 'featureIndex');
        if s <> '' then
            node^.featureIndex := StrToInt(s);
        
        s := ExtractJSONValue(json, 'threshold');
        if s <> '' then
            node^.threshold := StrToFloat(s);
        
        s := ExtractJSONValue(json, 'impurity');
        if s <> '' then
            node^.impurity := StrToFloat(s);
        
        s := ExtractJSONValue(json, 'numSamples');
        if s <> '' then
            node^.numSamples := StrToInt(s);
        
        s := ExtractJSONSubObject(json, 'left');
        if s <> '' then
            node^.left := JSONToTreeNode(s)
        else
            node^.left := nil;
        
        s := ExtractJSONSubObject(json, 'right');
        if s <> '' then
            node^.right := JSONToTreeNode(s)
        else
            node^.right := nil;
    end;
    
    Result := node;
end;

procedure TRandomForestFacade.saveModelToJSON(filename: string);
var
     SL: TStringList;
     i: integer;
     tree: TDecisionTree;
     treeJSON: string;
begin
     SL := TStringList.Create;
     try
        SL.Add('{');
        SL.Add('  "num_trees": ' + IntToStr(forest.getNumTrees()) + ',');
        SL.Add('  "max_depth": ' + IntToStr(forest.getMaxDepth()) + ',');
        SL.Add('  "min_samples_leaf": ' + IntToStr(forest.getMinSamplesLeaf()) + ',');
        SL.Add('  "min_samples_split": ' + IntToStr(forest.getMinSamplesSplit()) + ',');
        SL.Add('  "max_features": ' + IntToStr(forest.getMaxFeatures()) + ',');
        SL.Add('  "num_features": ' + IntToStr(forest.getNumFeatures()) + ',');
        SL.Add('  "num_samples": ' + IntToStr(forest.getNumSamples()) + ',');
        SL.Add('  "task_type": "' + taskTypeToStr(forest.getTaskType()) + '",');
        SL.Add('  "criterion": "' + criterionToStr(forest.getCriterion()) + '",');
        SL.Add('  "random_seed": ' + IntToStr(forest.getRandomSeed()) + ',');
        
        { Serialize trees }
        SL.Add('  "trees": [');
        for i := 0 to forest.getNumTrees() - 1 do
        begin
            tree := forest.getTree(i);
            if tree <> nil then
            begin
                treeJSON := treeNodeToJSON(tree^.root);
                if i < forest.getNumTrees() - 1 then
                    SL.Add('    ' + treeJSON + ',')
                else
                    SL.Add('    ' + treeJSON);
            end
            else
            begin
                if i < forest.getNumTrees() - 1 then
                    SL.Add('    null,')
                else
                    SL.Add('    null');
            end;
        end;
        SL.Add('  ]');
        SL.Add('}');
        
        SL.SaveToFile(filename);
        WriteLn('Model saved to JSON: ', filename);
     finally
        SL.Free;
     end;
end;

function TRandomForestFacade.loadModelFromJSON(filename: string): boolean;
var
    SL: TStringList;
    Content: string;
    ValueStr: string;
    numT, maxD, minL, minS, maxF: integer;
    tt: TaskType;
    sc: SplitCriterion;
    i, numTrees: integer;
    treeJSON: string;
    tree: TDecisionTree;
    
    function ExtractJSONValueFromContent(const content: string; const key: string): string;
    var
       KeyPos, ColonPos, QuotePos1, QuotePos2, StartPos, EndPos: integer;
    begin
       KeyPos := Pos('"' + key + '"', content);
       if KeyPos > 0 then
       begin
          ColonPos := PosEx(':', content, KeyPos);
          if ColonPos > 0 then
          begin
             StartPos := ColonPos + 1;
             { Skip whitespace }
             while (StartPos <= Length(content)) and (content[StartPos] in [' ', #9, #10, #13]) do
                Inc(StartPos);
             
             { Check if value is a quoted string }
             if (StartPos <= Length(content)) and (content[StartPos] = '"') then
             begin
                QuotePos1 := StartPos;
                QuotePos2 := PosEx('"', content, QuotePos1 + 1);
                if QuotePos2 > 0 then
                   Result := Copy(content, QuotePos1 + 1, QuotePos2 - QuotePos1 - 1)
                else
                   Result := '';
             end
             else
             begin
                { Value is a number or boolean }
                EndPos := PosEx(',', content, StartPos);
                if EndPos = 0 then
                   EndPos := PosEx('}', content, StartPos);
                if EndPos = 0 then
                   EndPos := PosEx(']', content, StartPos);
                Result := Trim(Copy(content, StartPos, EndPos - StartPos));
             end;
          end
          else
             Result := '';
       end
       else
          Result := '';
    end;
    
    function ExtractJSONArray(const content: string; const key: string): string;
    var
        KeyPos, BracketPos, StartPos, BracketCount, j: integer;
    begin
        KeyPos := Pos('"' + key + '"', content);
        if KeyPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        BracketPos := PosEx('[', content, KeyPos);
        if BracketPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        StartPos := BracketPos;
        BracketCount := 0;
        j := StartPos;
        
        while j <= Length(content) do
        begin
            if content[j] = '[' then
                Inc(BracketCount)
            else if content[j] = ']' then
            begin
                Dec(BracketCount);
                if BracketCount = 0 then
                begin
                    Result := Copy(content, StartPos, j - StartPos + 1);
                    Exit;
                end;
            end;
            Inc(j);
        end;
        Result := '';
    end;
    
    function ExtractArrayElement(const arrayJSON: string; index: integer): string;
    var
        StartPos, BraceCount, ElementCount, j: integer;
        inString: boolean;
    begin
        StartPos := Pos('[', arrayJSON);
        if StartPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        StartPos := StartPos + 1;
        ElementCount := 0;
        BraceCount := 0;
        inString := false;
        j := StartPos;
        
        while j <= Length(arrayJSON) do
        begin
            if (arrayJSON[j] = '"') and ((j = 1) or (arrayJSON[j-1] <> '\')) then
                inString := not inString;
            
            if not inString then
            begin
                if arrayJSON[j] = '{' then
                begin
                    if ElementCount = index then
                        StartPos := j;
                    Inc(BraceCount);
                end
                else if arrayJSON[j] = '}' then
                begin
                    Dec(BraceCount);
                    if (BraceCount = 0) and (ElementCount = index) then
                    begin
                        Result := Copy(arrayJSON, StartPos, j - StartPos + 1);
                        Exit;
                    end;
                end
                else if (arrayJSON[j] = ',') and (BraceCount = 0) then
                begin
                    if ElementCount = index then
                        Exit;
                    Inc(ElementCount);
                end;
            end;
            Inc(j);
        end;
        Result := '';
    end;

begin
    loadModelFromJSON := false;
    
    SL := TStringList.Create;
    try
       try
          SL.LoadFromFile(filename);
          except
             WriteLn('Error: Cannot open model file: ', filename);
             exit;
          end;
          
          Content := SL.Text;
          
          { Initialize with defaults }
          numT := 100;
          maxD := MAX_DEPTH_DEFAULT;
          minL := MIN_SAMPLES_LEAF_DEFAULT;
          minS := MIN_SAMPLES_SPLIT_DEFAULT;
          maxF := 0;
          tt := Classification;
          sc := Gini;
          
          { Load hyperparameters }
          ValueStr := ExtractJSONValueFromContent(Content, 'num_trees');
          if ValueStr <> '' then
             numT := StrToInt(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'max_depth');
          if ValueStr <> '' then
             maxD := StrToInt(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'min_samples_leaf');
          if ValueStr <> '' then
             minL := StrToInt(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'min_samples_split');
          if ValueStr <> '' then
             minS := StrToInt(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'max_features');
          if ValueStr <> '' then
             maxF := StrToInt(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'task_type');
          if ValueStr <> '' then
             tt := parseTaskType(ValueStr);
          
          ValueStr := ExtractJSONValueFromContent(Content, 'criterion');
          if ValueStr <> '' then
             sc := parseCriterion(ValueStr);
       
       { Configure forest }
       forest.create();
       forest.setNumTrees(numT);
       forest.setMaxDepth(maxD);
       forest.setMinSamplesLeaf(minL);
       forest.setMinSamplesSplit(minS);
       forest.setMaxFeatures(maxF);
       forest.setTaskType(tt);
       forest.setCriterion(sc);
       
       { Load trees if present }
       ValueStr := ExtractJSONArray(Content, 'trees');
       if ValueStr <> '' then
       begin
           for i := 0 to numT - 1 do
           begin
               treeJSON := ExtractArrayElement(ValueStr, i);
               if (treeJSON <> '') and (treeJSON <> 'null') then
               begin
                   New(tree);
                   tree^.root := JSONToTreeNode(treeJSON);
                   tree^.maxDepth := maxD;
                   tree^.minSamplesLeaf := minL;
                   tree^.minSamplesSplit := minS;
                   tree^.maxFeatures := maxF;
                   tree^.taskType := tt;
                   tree^.criterion := sc;
                   forest.setTree(i, tree);
               end;
           end;
       end;
       
       forestInitialized := true;
       loadModelFromJSON := true;
       WriteLn('Model configuration and trees loaded from JSON: ', filename);
    finally
       SL.Free;
    end;
end;

{ ============================================================================ }
{ TRandomForestFacade - Cleanup }
{ ============================================================================ }

procedure TRandomForestFacade.freeForest();
begin
   forest.freeForest();
   forestInitialized := false;
end;

{ ============================================================================ }
{ CLI - CSV Loading }
{ ============================================================================ }

function LoadCSV(filename: string; var data: TDataMatrix; var targets: TTargetArray;
                 var nSamples, nFeatures: integer; targetCol: integer): boolean;
var
   f: text;
   line: string;
   i, j, col, commaPos: integer;
   values: array[0..MAX_FEATURES] of double;
   valStr: string;
   numCols: integer;
begin
   LoadCSV := false;
   nSamples := 0;
   nFeatures := 0;
   
   assign(f, filename);
   {$I-}
   reset(f);
   {$I+}
   if IOResult <> 0 then
   begin
      writeln('Error: Cannot open file: ', filename);
      exit;
   end;
   
   while not eof(f) and (nSamples < MAX_SAMPLES) do
   begin
      readln(f, line);
      if length(line) = 0 then
         continue;
      
      col := 0;
      valStr := '';
      for i := 1 to length(line) do
      begin
         if (line[i] = ',') or (i = length(line)) then
         begin
            if i = length(line) then
               valStr := valStr + line[i];
            if length(valStr) > 0 then
            begin
               val(valStr, values[col], j);
               if j <> 0 then
                  values[col] := 0;
               inc(col);
            end;
            valStr := '';
         end
         else
            valStr := valStr + line[i];
      end;
      
      numCols := col;
      if nSamples = 0 then
      begin
         if targetCol < 0 then
            nFeatures := numCols - 1
         else
            nFeatures := numCols - 1;
      end;
      
      j := 0;
      for i := 0 to numCols - 1 do
      begin
         if targetCol < 0 then
         begin
            if i = numCols - 1 then
               targets[nSamples] := values[i]
            else
            begin
               data[nSamples][j] := values[i];
               inc(j);
            end;
         end
         else
         begin
            if i = targetCol then
               targets[nSamples] := values[i]
            else
            begin
               if j < MAX_FEATURES then
               begin
                  data[nSamples][j] := values[i];
                  inc(j);
               end;
            end;
         end;
      end;
      
      inc(nSamples);
   end;
   
   close(f);
   LoadCSV := true;
   writeln('Loaded ', nSamples, ' samples with ', nFeatures, ' features from ', filename);
end;

procedure SavePredictionsCSV(filename: string; var predictions: TTargetArray; nSamples: integer);
var
   f: text;
   i: integer;
begin
   assign(f, filename);
   rewrite(f);
   writeln(f, 'prediction');
   for i := 0 to nSamples - 1 do
      writeln(f, predictions[i]:0:6);
   close(f);
   writeln('Saved ', nSamples, ' predictions to ', filename);
end;

{ ============================================================================ }
{ CLI - Model Save/Load }
{ ============================================================================ }

{ ============================================================================ }
{ CLI - Help & Usage }
{ ============================================================================ }

procedure PrintHelp();
begin
   writeln('Facaded Random Forest');
   writeln;
   writeln('Usage: facaded_forest <command> [options]');
   writeln;
   writeln('=== Core Commands ===');
   writeln('  create         Create a new empty forest model and save to JSON');
   writeln('  train          Train a new random forest model');
   writeln('  predict        Make predictions using a trained model');
   writeln('  evaluate       Evaluate model on test data');
   writeln('  info           Show model information from JSON');
   writeln('  inspect        Inspect tree structure');
   writeln('  help           Show this help message');
   writeln;
   writeln('=== Tree Management (Facade) ===');
   writeln('  add-tree       Add a new tree to the forest');
   writeln('  remove-tree    Remove a tree from the forest');
   writeln('  retrain-tree   Retrain a specific tree');
   writeln('  prune          Prune a tree at a specific node');
   writeln('  modify-split   Modify split threshold at a node');
   writeln('  modify-leaf    Modify leaf prediction value');
   writeln('  convert-leaf   Convert internal node to leaf');
   writeln;
   writeln('=== Aggregation Control ===');
   writeln('  set-aggregation  Set prediction aggregation method');
   writeln('  set-weight       Set weight for a specific tree');
   writeln('  reset-weights    Reset all tree weights to 1.0');
   writeln;
   writeln('=== Feature Analysis ===');
   writeln('  feature-usage    Show feature usage summary');
   writeln('  feature-heatmap  Show feature usage heatmap across trees');
   writeln('  importance       Show feature importances');
   writeln;
   writeln('=== OOB Analysis ===');
   writeln('  oob-summary      Show OOB error summary per tree');
   writeln('  problematic      Find trees with high OOB error');
   writeln('  worst-trees      Find N worst performing trees');
   writeln;
   writeln('=== Diagnostics ===');
   writeln('  misclassified    Show misclassified samples');
   writeln('  high-residual    Show samples with high residual');
   writeln('  track-sample     Track a sample through the forest');
   writeln;
   writeln('=== Visualization ===');
   writeln('  visualize        Visualize tree structure');
   writeln('  node-details     Show details of a specific node');
   writeln('  split-dist       Show split distribution at a node');
   writeln;
   writeln('=== Options ===');
   writeln;
   writeln('Create Options:');
   writeln('  --trees <n>            Number of trees (default: 100)');
   writeln('  --depth <n>            Max tree depth (default: 10)');
   writeln('  --min-leaf <n>         Min samples per leaf (default: 1)');
   writeln('  --min-split <n>        Min samples to split (default: 2)');
   writeln('  --max-features <n>     Max features per split (default: sqrt)');
   writeln('  --task <class|reg>     Task type (default: class)');
   writeln('  --criterion <g|e|m>    Split criterion: gini/entropy/mse (default: gini)');
   writeln('  --seed <n>             Random seed (default: 42)');
   writeln('  --save <file.json>     Save model config to JSON file (required)');
   writeln;
   writeln('Training Options:');
   writeln('  --model <file.json>    Load model from JSON file (required)');
   writeln('  --data <file.csv>      Training data (required)');
   writeln('  --save <file.json>     Save trained model to JSON (required)');
   writeln('  --target-col <n>       Target column index (default: last)');
   writeln;
   writeln('Prediction Options:');
   writeln('  --model <file.json>    Load model from JSON file (required)');
   writeln('  --data <file.csv>      Input data (required)');
   writeln('  --output <file.csv>    Output predictions file');
   writeln('  --aggregation <method> Aggregation: majority/weighted/mean/wmean');
   writeln;
   writeln('Evaluation Options:');
   writeln('  --model <file.json>    Load model from JSON file (required)');
   writeln('  --data <file.csv>      Test data with labels (required)');
   writeln('  --target-col <n>       Target column index (default: last)');
   writeln('  --positive-class <n>   Positive class for precision/recall (default: 1)');
   writeln;
   writeln('Tree Management Options:');
   writeln('  --model <file.json>    Model file (required)');
   writeln('  --tree <n>             Tree index');
   writeln('  --node <n>             Node index');
   writeln('  --threshold <f>        New threshold value');
   writeln('  --value <f>            New leaf value');
   writeln('  --weight <f>           Tree weight');
   writeln;
   writeln('Analysis Options:');
   writeln('  --threshold <f>        Error/residual threshold');
   writeln('  --top <n>              Number of top results (default: 5)');
   writeln('  --sample <n>           Sample index to track');
   writeln;
   writeln('=== Examples ===');
   writeln;
   writeln('Creating:');
   writeln('  forest create --trees 50 --depth 8 --min-leaf 2 --criterion gini --task class --save config.json');
   writeln;
   writeln('Training:');
   writeln('  forest train --model config.json --data train.csv --save trained.json');
   writeln;
   writeln('Prediction:');
   writeln('  forest predict --model trained.json --data test.csv --output preds.csv');
   writeln('  forest predict --model trained.json --data test.csv --aggregation weighted');
   writeln;
   writeln('Evaluation:');
   writeln('  forest evaluate --model trained.json --data test.csv');
   writeln;
   writeln('Tree Management:');
   writeln('  forest add-tree --model trained.json --data train.csv');
   writeln('  forest remove-tree --model trained.json --tree 5');
   writeln('  forest retrain-tree --model trained.json --tree 3 --data train.csv');
   writeln('  forest prune --model trained.json --tree 0 --node 5');
   writeln('  forest modify-split --model trained.json --tree 0 --node 3 --threshold 2.5');
   writeln('  forest modify-leaf --model trained.json --tree 0 --node 10 --value 1.0');
   writeln('  forest convert-leaf --model trained.json --tree 0 --node 5 --value 0.0');
   writeln;
   writeln('Aggregation:');
   writeln('  forest set-aggregation --model trained.json --method weighted');
   writeln('  forest set-weight --model trained.json --tree 0 --weight 2.0');
   writeln;
   writeln('Analysis:');
   writeln('  forest oob-summary --model trained.json');
   writeln('  forest problematic --model trained.json --threshold 0.3');
   writeln('  forest worst-trees --model trained.json --top 10');
   writeln('  forest misclassified --model trained.json --data test.csv');
   writeln('  forest high-residual --model trained.json --data test.csv --threshold 1.0');
   writeln('  forest track-sample --model trained.json --data train.csv --sample 0');
   writeln;
   writeln('Visualization:');
   writeln('  forest visualize --model trained.json --tree 0');
   writeln('  forest node-details --model trained.json --tree 0 --node 5');
   writeln('  forest feature-heatmap --model trained.json');
   writeln;
   writeln('JSON Format:');
   writeln('  Models are saved to/loaded from JSON format containing:');
   writeln('    - metadata: version, model type, creation date');
   writeln('    - hyperparameters: trees, depth, leaf samples, split samples, features, task, criterion, seed');
   writeln('    - training_data_info: sample count, feature count');
   end;

function GetArg(name: string): string;
var
   i: integer;
   arg: string;
   eqPos: integer;
   key, value: string;
begin
   GetArg := '';
   for i := 1 to ParamCount do
   begin
      arg := ParamStr(i);
      
      { Try --key=value format first }
      eqPos := Pos('=', arg);
      if eqPos > 0 then
      begin
         key := Copy(arg, 1, eqPos - 1);
         if key = name then
         begin
            GetArg := Copy(arg, eqPos + 1, Length(arg));
            exit;
         end;
      end
      else if arg = name then
      begin
         { Try --key value format }
         if i < ParamCount then
         begin
            GetArg := ParamStr(i + 1);
            exit;
         end;
      end;
   end;
end;

function HasArg(name: string): boolean;
var
   i: integer;
begin
   HasArg := false;
   for i := 1 to ParamCount do
   begin
      if ParamStr(i) = name then
      begin
         HasArg := true;
         exit;
      end;
   end;
end;

function GetArgInt(name: string; default: integer): integer;
var
   s: string;
   v, code: integer;
begin
   s := GetArg(name);
   if s = '' then
      GetArgInt := default
   else
   begin
      val(s, v, code);
      if code = 0 then
         GetArgInt := v
      else
         GetArgInt := default;
   end;
end;

function GetArgFloat(name: string; default: double): double;
var
   s: string;
   v: double;
   code: integer;
begin
   s := GetArg(name);
   if s = '' then
      GetArgFloat := default
   else
   begin
      val(s, v, code);
      if code = 0 then
         GetArgFloat := v
      else
         GetArgFloat := default;
   end;
end;

{ ============================================================================ }
{ CLI - Commands }
{ ============================================================================ }

procedure CmdCreate();
var
   facade: TRandomForestFacade;
   numTrees, maxDepth, minLeaf, minSplit, maxFeatures: integer;
   taskStr, critStr, modelFile: string;
   task: TaskType;
   crit: SplitCriterion;
begin
   numTrees := GetArgInt('--trees', 100);
   maxDepth := GetArgInt('--depth', MAX_DEPTH_DEFAULT);
   minLeaf := GetArgInt('--min-leaf', MIN_SAMPLES_LEAF_DEFAULT);
   minSplit := GetArgInt('--min-split', MIN_SAMPLES_SPLIT_DEFAULT);
   maxFeatures := GetArgInt('--max-features', 0);
   taskStr := GetArg('--task');
   critStr := GetArg('--criterion');
   modelFile := GetArg('--save');
   
   if modelFile = '' then
      modelFile := 'forest.json';
   
   { Parse task type }
   taskStr := lowercase(taskStr);
   if (taskStr = 'reg') or (taskStr = 'regression') then
      task := Regression
   else
      task := Classification;
   
   { Parse criterion }
   critStr := lowercase(critStr);
   if critStr = 'e' then
      crit := Entropy
   else if critStr = 'entropy' then
      crit := Entropy
   else if critStr = 'm' then
      crit := MSE
   else if critStr = 'mse' then
      crit := MSE
   else if critStr = 'variance' then
      crit := VarianceReduction
   else
      crit := Gini;
   
   facade.create();
   facade.setHyperparameter('n_estimators', numTrees);
   facade.setHyperparameter('max_depth', maxDepth);
   facade.setHyperparameter('min_samples_leaf', minLeaf);
   facade.setHyperparameter('min_samples_split', minSplit);
   facade.setHyperparameter('max_features', maxFeatures);
   facade.setTaskType(task);
   facade.setCriterion(crit);
   
   writeln('Created Random Forest model:');
   writeln('  Number of trees: ', numTrees);
   writeln('  Max depth: ', maxDepth);
   writeln('  Min samples leaf: ', minLeaf);
   writeln('  Min samples split: ', minSplit);
   writeln('  Max features: ', maxFeatures);
   case crit of
      Gini: writeln('  Criterion: Gini');
      Entropy: writeln('  Criterion: Entropy');
      MSE: writeln('  Criterion: MSE');
      VarianceReduction: writeln('  Criterion: Variance Reduction');
   end;
   if task = Classification then
      writeln('  Task: Classification')
   else
      writeln('  Task: Regression');
   
   facade.saveModelToJSON(modelFile);
   
   facade.freeForest();
end;

procedure CmdTrain();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   predictions: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile, taskStr, critStr: string;
   numTrees, maxDepth, minLeaf, minSplit, maxFeat, seed, targetCol: integer;
   acc, oobErr: double;
begin
   dataFile := GetArg('--data');
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   
   modelFile := GetArg('--model');
   if modelFile = '' then modelFile := 'model.bin';
   
   numTrees := GetArgInt('--trees', 100);
   maxDepth := GetArgInt('--depth', 10);
   minLeaf := GetArgInt('--min-leaf', 1);
   minSplit := GetArgInt('--min-split', 2);
   maxFeat := GetArgInt('--max-features', 0);
   seed := GetArgInt('--seed', 42);
   targetCol := GetArgInt('--target-col', -1);
   taskStr := GetArg('--task');
   critStr := GetArg('--criterion');
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.create();
   facade.setHyperparameter('n_estimators', numTrees);
   facade.setHyperparameter('max_depth', maxDepth);
   facade.setHyperparameter('min_samples_leaf', minLeaf);
   facade.setHyperparameter('min_samples_split', minSplit);
   if maxFeat > 0 then
      facade.setHyperparameter('max_features', maxFeat);
   facade.setHyperparameter('random_seed', seed);
   
   if (taskStr = 'reg') or (taskStr = 'regression') then
      facade.setTaskType(Regression)
   else
      facade.setTaskType(Classification);
   
   if critStr = 'e' then
      facade.setCriterion(Entropy)
   else if critStr = 'm' then
      facade.setCriterion(MSE)
   else
      facade.setCriterion(Gini);
   
   writeln;
   writeln('Training Random Forest...');
   writeln('  Trees: ', numTrees);
   writeln('  Max Depth: ', maxDepth);
   writeln('  Samples: ', nSamples);
   writeln('  Features: ', nFeatures);
   
   facade.loadData(data, targets, nSamples, nFeatures);
   facade.trainForest();
   
   facade.predictBatch(data, nSamples, predictions);
   acc := facade.accuracy(predictions, targets, nSamples);
   oobErr := facade.getGlobalOOBError();
   
   writeln;
   writeln('Training Complete!');
   writeln('  Training Accuracy: ', acc:0:4);
   writeln('  OOB Error: ', oobErr:0:4);
   
   facade.saveModel(modelFile);
   
   facade.freeForest();
end;

procedure CmdPredict();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   predictions: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile, outputFile: string;
   i: integer;
begin
   dataFile := GetArg('--data');
   modelFile := GetArg('--model');
   outputFile := GetArg('--output');
   
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, -1) then
      exit;
   
   writeln('Making predictions...');
   facade.predictBatch(data, nSamples, predictions);
   
   if outputFile <> '' then
      SavePredictionsCSV(outputFile, predictions, nSamples)
   else
   begin
      writeln;
      writeln('Predictions:');
      for i := 0 to nSamples - 1 do
         writeln('  Sample ', i, ': ', predictions[i]:0:4);
   end;
   
   facade.freeForest();
end;

procedure CmdEvaluate();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   predictions: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile: string;
   targetCol: integer;
   acc, prec, rec, f1Sc, mseV, r2V: double;
begin
   dataFile := GetArg('--data');
   modelFile := GetArg('--model');
   targetCol := GetArgInt('--target-col', -1);
   
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   writeln('Evaluating model...');
   facade.predictBatch(data, nSamples, predictions);
   
   acc := facade.accuracy(predictions, targets, nSamples);
   prec := facade.precision(predictions, targets, nSamples, 1);
   rec := facade.recall(predictions, targets, nSamples, 1);
   f1Sc := facade.f1Score(predictions, targets, nSamples, 1);
   mseV := facade.meanSquaredError(predictions, targets, nSamples);
   r2V := facade.rSquared(predictions, targets, nSamples);
   
   writeln;
   writeln('=== Evaluation Results ===');
   writeln('Samples: ', nSamples);
   writeln('Accuracy: ', acc:0:4);
   writeln('Precision: ', prec:0:4);
   writeln('Recall: ', rec:0:4);
   writeln('F1 Score: ', f1Sc:0:4);
   writeln('MSE: ', mseV:0:4);
   writeln('R-Squared: ', r2V:0:4);
   
   facade.freeForest();
end;

procedure CmdInfo();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printForestOverview();
   facade.printFeatureImportances();
   
   facade.freeForest();
end;

procedure CmdInspect();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId: integer;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', 0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.visualizeTree(treeId);
   writeln;
   facade.printTreeStructure(treeId);
   
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - Tree Management Commands }
{ ============================================================================ }

procedure CmdAddTree();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile, saveFile: string;
   targetCol: integer;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   saveFile := GetArg('--save');
   targetCol := GetArgInt('--target-col', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if dataFile = '' then
   begin
      writeln('Error: --data is required (for bootstrap sampling)');
      exit;
   end;
   if saveFile = '' then
      saveFile := modelFile;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.loadData(data, targets, nSamples, nFeatures);
   facade.addTree();
   writeln('Added new tree. Total trees: ', facade.getNumTrees());
   facade.saveModel(saveFile);
   facade.freeForest();
end;

procedure CmdRemoveTree();
var
   facade: TRandomForestFacade;
   modelFile, saveFile: string;
   treeId: integer;
begin
   modelFile := GetArg('--model');
   saveFile := GetArg('--save');
   treeId := GetArgInt('--tree', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if saveFile = '' then
      saveFile := modelFile;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.removeTree(treeId);
   writeln('Removed tree ', treeId, '. Total trees: ', facade.getNumTrees());
   facade.saveModel(saveFile);
   facade.freeForest();
end;

procedure CmdRetrainTree();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile, saveFile: string;
   treeId, targetCol: integer;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   saveFile := GetArg('--save');
   treeId := GetArgInt('--tree', -1);
   targetCol := GetArgInt('--target-col', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if saveFile = '' then
      saveFile := modelFile;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.loadData(data, targets, nSamples, nFeatures);
   facade.retrainTree(treeId);
   writeln('Retrained tree ', treeId);
   facade.saveModel(saveFile);
   facade.freeForest();
end;

procedure CmdPrune();
var
   facade: TRandomForestFacade;
   modelFile, saveFile: string;
   treeId, nodeId: integer;
begin
   modelFile := GetArg('--model');
   saveFile := GetArg('--save');
   treeId := GetArgInt('--tree', -1);
   nodeId := GetArgInt('--node', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if nodeId < 0 then
   begin
      writeln('Error: --node is required');
      exit;
   end;
   if saveFile = '' then
      saveFile := modelFile;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.pruneTree(treeId, nodeId);
   writeln('Pruned tree ', treeId, ' at node ', nodeId);
   facade.saveModel(saveFile);
   facade.freeForest();
end;

procedure CmdModifySplit();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId, nodeId: integer;
   threshold: double;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', -1);
   nodeId := GetArgInt('--node', -1);
   threshold := GetArgFloat('--threshold', -999999.0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if nodeId < 0 then
   begin
      writeln('Error: --node is required');
      exit;
   end;
   if threshold = -999999.0 then
   begin
      writeln('Error: --threshold is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.modifySplit(treeId, nodeId, threshold);
   facade.saveModel(modelFile);
   facade.freeForest();
end;

procedure CmdModifyLeaf();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId, nodeId: integer;
   value: double;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', -1);
   nodeId := GetArgInt('--node', -1);
   value := GetArgFloat('--value', -999999.0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if nodeId < 0 then
   begin
      writeln('Error: --node is required');
      exit;
   end;
   if value = -999999.0 then
   begin
      writeln('Error: --value is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.modifyLeafValue(treeId, nodeId, value);
   facade.saveModel(modelFile);
   facade.freeForest();
end;

procedure CmdConvertLeaf();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId, nodeId: integer;
   value: double;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', -1);
   nodeId := GetArgInt('--node', -1);
   value := GetArgFloat('--value', 0.0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   if nodeId < 0 then
   begin
      writeln('Error: --node is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.convertToLeaf(treeId, nodeId, value);
   facade.saveModel(modelFile);
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - Aggregation Commands }
{ ============================================================================ }

procedure CmdSetAggregation();
var
   facade: TRandomForestFacade;
   modelFile, methodStr: string;
begin
   modelFile := GetArg('--model');
   methodStr := GetArg('--method');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if methodStr = '' then
   begin
      writeln('Error: --method is required (majority/weighted/mean/wmean)');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if (methodStr = 'majority') or (methodStr = 'vote') then
      facade.setAggregationMethod(MajorityVote)
   else if (methodStr = 'weighted') or (methodStr = 'wvote') then
      facade.setAggregationMethod(WeightedVote)
   else if methodStr = 'mean' then
      facade.setAggregationMethod(Mean)
   else if (methodStr = 'wmean') or (methodStr = 'weightedmean') then
      facade.setAggregationMethod(WeightedMean)
   else
      writeln('Unknown method: ', methodStr);
   
   writeln('Aggregation method set to: ', methodStr);
   facade.freeForest();
end;

procedure CmdSetWeight();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId: integer;
   weight: double;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', -1);
   weight := GetArgFloat('--weight', 1.0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if treeId < 0 then
   begin
      writeln('Error: --tree is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.setTreeWeight(treeId, weight);
   writeln('Set tree ', treeId, ' weight to ', weight:0:4);
   facade.freeForest();
end;

procedure CmdResetWeights();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   facade.resetTreeWeights();
   writeln('All tree weights reset to 1.0');
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - Feature Analysis Commands }
{ ============================================================================ }

procedure CmdFeatureUsage();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printFeatureUsageSummary();
   facade.freeForest();
end;

procedure CmdFeatureHeatmap();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printFeatureHeatmap();
   facade.freeForest();
end;

procedure CmdImportance();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printFeatureImportances();
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - OOB Analysis Commands }
{ ============================================================================ }

procedure CmdOOBSummary();
var
   facade: TRandomForestFacade;
   modelFile: string;
begin
   modelFile := GetArg('--model');
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printOOBSummary();
   facade.freeForest();
end;

procedure CmdProblematic();
var
   facade: TRandomForestFacade;
   modelFile: string;
   threshold: double;
begin
   modelFile := GetArg('--model');
   threshold := GetArgFloat('--threshold', 0.3);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.markProblematicTrees(threshold);
   facade.freeForest();
end;

procedure CmdWorstTrees();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile: string;
   topN, targetCol: integer;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   topN := GetArgInt('--top', 5);
   targetCol := GetArgInt('--target-col', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if dataFile <> '' then
   begin
      if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
         exit;
   end
   else
      nSamples := 0;
   
   writeln;
   facade.findWorstTrees(targets, nSamples, topN);
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - Diagnostic Commands }
{ ============================================================================ }

procedure CmdMisclassified();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   predictions: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile: string;
   targetCol: integer;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   targetCol := GetArgInt('--target-col', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.predictBatch(data, nSamples, predictions);
   writeln;
   facade.highlightMisclassified(predictions, targets, nSamples);
   facade.freeForest();
end;

procedure CmdHighResidual();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   predictions: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile: string;
   targetCol: integer;
   threshold: double;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   targetCol := GetArgInt('--target-col', -1);
   threshold := GetArgFloat('--threshold', 1.0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.predictBatch(data, nSamples, predictions);
   writeln;
   facade.highlightHighResidual(predictions, targets, nSamples, threshold);
   facade.freeForest();
end;

procedure CmdTrackSample();
var
   facade: TRandomForestFacade;
   data: TDataMatrix;
   targets: TTargetArray;
   nSamples, nFeatures: integer;
   dataFile, modelFile: string;
   sampleIdx, targetCol: integer;
begin
   modelFile := GetArg('--model');
   dataFile := GetArg('--data');
   sampleIdx := GetArgInt('--sample', 0);
   targetCol := GetArgInt('--target-col', -1);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   if dataFile = '' then
   begin
      writeln('Error: --data is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   if not LoadCSV(dataFile, data, targets, nSamples, nFeatures, targetCol) then
      exit;
   
   facade.loadData(data, targets, nSamples, nFeatures);
   writeln;
   facade.printSampleTracking(sampleIdx);
   facade.freeForest();
end;

{ ============================================================================ }
{ CLI - Visualization Commands }
{ ============================================================================ }

procedure CmdVisualize();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId: integer;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', 0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.visualizeTree(treeId);
   facade.freeForest();
end;

procedure CmdNodeDetails();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId, nodeId: integer;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', 0);
   nodeId := GetArgInt('--node', 0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.printNodeDetails(treeId, nodeId);
   facade.freeForest();
end;

procedure CmdSplitDist();
var
   facade: TRandomForestFacade;
   modelFile: string;
   treeId, nodeId: integer;
begin
   modelFile := GetArg('--model');
   treeId := GetArgInt('--tree', 0);
   nodeId := GetArgInt('--node', 0);
   
   if modelFile = '' then
   begin
      writeln('Error: --model is required');
      exit;
   end;
   
   facade.create();
   if not facade.loadModel(modelFile) then
      exit;
   
   writeln;
   facade.visualizeSplitDistribution(treeId, nodeId);
   facade.freeForest();
end;

{ ============================================================================ }
{ Main Program }
{ ============================================================================ }

var
   cmd: string;

begin
   if ParamCount < 1 then
   begin
      PrintHelp();
      exit;
   end;
   
   cmd := ParamStr(1);
   
   if (cmd = 'help') or (cmd = '--help') or (cmd = '-h') then
      PrintHelp()

   { Core Commands }
   else if cmd = 'create' then
      CmdCreate()
   else if cmd = 'train' then
      CmdTrain()
   else if cmd = 'predict' then
      CmdPredict()
   else if cmd = 'evaluate' then
      CmdEvaluate()
   else if cmd = 'info' then
      CmdInfo()
   else if cmd = 'inspect' then
      CmdInspect()
   
   { Tree Management Commands }
   else if cmd = 'add-tree' then
      CmdAddTree()
   else if cmd = 'remove-tree' then
      CmdRemoveTree()
   else if cmd = 'retrain-tree' then
      CmdRetrainTree()
   else if cmd = 'prune' then
      CmdPrune()
   else if cmd = 'modify-split' then
      CmdModifySplit()
   else if cmd = 'modify-leaf' then
      CmdModifyLeaf()
   else if cmd = 'convert-leaf' then
      CmdConvertLeaf()
   
   { Aggregation Commands }
   else if cmd = 'set-aggregation' then
      CmdSetAggregation()
   else if cmd = 'set-weight' then
      CmdSetWeight()
   else if cmd = 'reset-weights' then
      CmdResetWeights()
   
   { Feature Analysis Commands }
   else if cmd = 'feature-usage' then
      CmdFeatureUsage()
   else if cmd = 'feature-heatmap' then
      CmdFeatureHeatmap()
   else if cmd = 'importance' then
      CmdImportance()
   
   { OOB Analysis Commands }
   else if cmd = 'oob-summary' then
      CmdOOBSummary()
   else if cmd = 'problematic' then
      CmdProblematic()
   else if cmd = 'worst-trees' then
      CmdWorstTrees()
   
   { Diagnostic Commands }
   else if cmd = 'misclassified' then
      CmdMisclassified()
   else if cmd = 'high-residual' then
      CmdHighResidual()
   else if cmd = 'track-sample' then
      CmdTrackSample()
   
   { Visualization Commands }
   else if cmd = 'visualize' then
      CmdVisualize()
   else if cmd = 'node-details' then
      CmdNodeDetails()
   else if cmd = 'split-dist' then
      CmdSplitDist()
   
   else
   begin
      writeln('Unknown command: ', cmd);
      writeln('Use "forest help" for usage information.');
   end;
end.

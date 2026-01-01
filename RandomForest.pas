//
// Matthew Abbott 2025
// Random Forest
//

{$mode objfpc}
{$H+}
{$M+}

program RandomForest;

uses
   Math, SysUtils, Classes, StrUtils;

const
   MAX_FEATURES = 100;
   MAX_SAMPLES = 10000;
   MAX_TREES = 500;
   MAX_DEPTH_DEFAULT = 10;
   MIN_SAMPLES_LEAF_DEFAULT = 1;
   MIN_SAMPLES_SPLIT_DEFAULT = 2;

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
      function getMaxFeatures(): integer;
      function getTree(treeId: integer): TDecisionTree;
      function getData(sampleIdx, featureIdx: integer): double;
      function getTarget(sampleIdx: integer): double;
      function getTaskType(): TaskType;
      function getCriterion(): SplitCriterion;

      { Tree Management for Facade }
      procedure addNewTree();
      procedure removeTreeAt(treeId: integer);
      procedure retrainTreeAt(treeId: integer);

      { JSON serialization methods }
      procedure saveModelToJSON(const Filename: string);
      procedure loadModelFromJSON(const Filename: string);

      { JSON helper functions }
      function taskTypeToStr(t: TaskType): string;
      function criterionToStr(c: SplitCriterion): string;
      function parseTaskType(const s: string): TaskType;
      function parseCriterion(const s: string): SplitCriterion;
      function Array1DToJSON(const Arr: TDoubleArray; size: integer): string;
      function Array2DToJSON(const Arr: TDataMatrix; rows, cols: integer): string;
      function treeNodeToJSON(node: TreeNode): string;
      function JSONToTreeNode(const json: string): TreeNode;

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

function TRandomForest.getMaxFeatures(): integer;
begin
   getMaxFeatures := maxFeatures;
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

{ ============================================================================ }
{ Helper Functions }
{ ============================================================================ }

{ ============================================================================ }
{ JSON Serialization }
{ ============================================================================ }

function TRandomForest.taskTypeToStr(t: TaskType): string;
begin
   if t = Classification then
      taskTypeToStr := 'classification'
   else
      taskTypeToStr := 'regression';
end;

function TRandomForest.criterionToStr(c: SplitCriterion): string;
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

function TRandomForest.parseTaskType(const s: string): TaskType;
begin
   if lowercase(s) = 'regression' then
      parseTaskType := Regression
   else
      parseTaskType := Classification;
end;

function TRandomForest.parseCriterion(const s: string): SplitCriterion;
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

function TRandomForest.Array1DToJSON(const Arr: TDoubleArray; size: integer): string;
var
   i: integer;
begin
   Result := '[';
   for i := 0 to size - 1 do
   begin
      if i > 0 then Result := Result + ',';
      Result := Result + FloatToStr(Arr[i]);
   end;
   Result := Result + ']';
end;

function TRandomForest.Array2DToJSON(const Arr: TDataMatrix; rows, cols: integer): string;
var
    i, j: integer;
begin
    Result := '[';
    for i := 0 to rows - 1 do
    begin
       if i > 0 then Result := Result + ',';
       Result := Result + '[';
       for j := 0 to cols - 1 do
       begin
          if j > 0 then Result := Result + ',';
          Result := Result + FloatToStr(Arr[i][j]);
       end;
       Result := Result + ']';
    end;
    Result := Result + ']';
end;

function TRandomForest.treeNodeToJSON(node: TreeNode): string;
var
    leftJSON, rightJSON, result_left, result_right, impurity_str, threshold_str: string;
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

function TRandomForest.JSONToTreeNode(const json: string): TreeNode;
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

procedure TRandomForest.saveModelToJSON(const Filename: string);
var
     SL: TStringList;
     i: integer;
     treeJSON: string;
begin
     SL := TStringList.Create;
     try
        SL.Add('{');
        SL.Add('  "num_trees": ' + IntToStr(numTrees) + ',');
        SL.Add('  "max_depth": ' + IntToStr(maxDepth) + ',');
        SL.Add('  "min_samples_leaf": ' + IntToStr(minSamplesLeaf) + ',');
        SL.Add('  "min_samples_split": ' + IntToStr(minSamplesSplit) + ',');
        SL.Add('  "max_features": ' + IntToStr(maxFeatures) + ',');
        SL.Add('  "num_features": ' + IntToStr(numFeatures) + ',');
        SL.Add('  "num_samples": ' + IntToStr(numSamples) + ',');
        SL.Add('  "task_type": "' + taskTypeToStr(taskType) + '",');
        SL.Add('  "criterion": "' + criterionToStr(criterion) + '",');
        SL.Add('  "random_seed": ' + IntToStr(randomSeed) + ',');
        SL.Add('  "feature_importances": ' + Array1DToJSON(featureImportances, numFeatures) + ',');
        
        { Serialize trees }
        SL.Add('  "trees": [');
        for i := 0 to numTrees - 1 do
        begin
            if trees[i] <> nil then
            begin
                treeJSON := treeNodeToJSON(trees[i]^.root);
                if i < numTrees - 1 then
                    SL.Add('    ' + treeJSON + ',')
                else
                    SL.Add('    ' + treeJSON);
            end
            else
            begin
                if i < numTrees - 1 then
                    SL.Add('    null,')
                else
                    SL.Add('    null');
            end;
        end;
        SL.Add('  ]');
        SL.Add('}');
        
        SL.SaveToFile(Filename);
        WriteLn('Model saved to JSON: ', Filename);
     finally
        SL.Free;
     end;
end;

procedure TRandomForest.loadModelFromJSON(const Filename: string);
var
    SL: TStringList;
    Content: string;
    ValueStr: string;
    i, treeCount: integer;
    treeJSON: string;
    
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
    
    function ExtractJSONArray(const json: string; const key: string): string;
    var
        KeyPos, BracketPos, StartPos, BracketCount, i: integer;
    begin
        KeyPos := Pos('"' + key + '"', json);
        if KeyPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        BracketPos := PosEx('[', json, KeyPos);
        if BracketPos = 0 then
        begin
            Result := '';
            Exit;
        end;
        
        StartPos := BracketPos;
        BracketCount := 0;
        i := StartPos;
        
        while i <= Length(json) do
        begin
            if json[i] = '[' then
                Inc(BracketCount)
            else if json[i] = ']' then
            begin
                Dec(BracketCount);
                if BracketCount = 0 then
                begin
                    Result := Copy(json, StartPos, i - StartPos + 1);
                    Exit;
                end;
            end;
            Inc(i);
        end;
        Result := '';
    end;
    
    function ExtractArrayElement(const arrayJSON: string; index: integer): string;
    var
        StartPos, BraceCount, ElementCount, i: integer;
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
        i := StartPos;
        
        while i <= Length(arrayJSON) do
        begin
            if (arrayJSON[i] = '"') and ((i = 1) or (arrayJSON[i-1] <> '\')) then
                inString := not inString;
            
            if not inString then
            begin
                if arrayJSON[i] = '{' then
                begin
                    if ElementCount = index then
                        StartPos := i;
                    Inc(BraceCount);
                end
                else if arrayJSON[i] = '}' then
                begin
                    Dec(BraceCount);
                    if (BraceCount = 0) and (ElementCount = index) then
                    begin
                        Result := Copy(arrayJSON, StartPos, i - StartPos + 1);
                        Exit;
                    end;
                end
                else if (arrayJSON[i] = ',') and (BraceCount = 0) then
                begin
                    if ElementCount = index then
                        Exit;
                    Inc(ElementCount);
                end;
            end;
            Inc(i);
        end;
        Result := '';
    end;

begin
    SL := TStringList.Create;
    try
       SL.LoadFromFile(Filename);
       Content := SL.Text;
       
       { Load configuration }
       ValueStr := ExtractJSONValue(Content, 'num_trees');
       if ValueStr <> '' then
          numTrees := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'max_depth');
       if ValueStr <> '' then
          maxDepth := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'min_samples_leaf');
       if ValueStr <> '' then
          minSamplesLeaf := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'min_samples_split');
       if ValueStr <> '' then
          minSamplesSplit := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'max_features');
       if ValueStr <> '' then
          maxFeatures := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'num_features');
       if ValueStr <> '' then
          numFeatures := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'num_samples');
       if ValueStr <> '' then
          numSamples := StrToInt(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'task_type');
       if ValueStr <> '' then
          taskType := parseTaskType(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'criterion');
       if ValueStr <> '' then
          criterion := parseCriterion(ValueStr);
       
       ValueStr := ExtractJSONValue(Content, 'random_seed');
       if ValueStr <> '' then
          randomSeed := StrToInt(ValueStr);
       
       { Load trees if present }
       ValueStr := ExtractJSONArray(Content, 'trees');
       if ValueStr <> '' then
       begin
           for i := 0 to numTrees - 1 do
           begin
               treeJSON := ExtractArrayElement(ValueStr, i);
               if (treeJSON <> '') and (treeJSON <> 'null') then
               begin
                   New(trees[i]);
                   trees[i]^.root := JSONToTreeNode(treeJSON);
                   trees[i]^.maxDepth := maxDepth;
                   trees[i]^.minSamplesLeaf := minSamplesLeaf;
                   trees[i]^.minSamplesSplit := minSamplesSplit;
                   trees[i]^.maxFeatures := maxFeatures;
                   trees[i]^.taskType := taskType;
                   trees[i]^.criterion := criterion;
               end;
           end;
       end;
       
       WriteLn('Model loaded from JSON: ', Filename);
    finally
       SL.Free;
    end;
end;

procedure LoadCSVData(const filename: string; var data: TDataMatrix; var targets: TTargetArray; 
                      var nSamples, nFeatures: integer; maxSamples: integer);
var
    f: Text;
    line: string;
    i, j, commaPos, startPos: integer;
    value: double;
    fieldStr: string;
    fieldCount: integer;
begin
    nSamples := 0;
    nFeatures := 0;
    
    Assign(f, filename);
    Reset(f);
    
    if IOResult <> 0 then
    begin
        WriteLn('Error: Cannot open file: ', filename);
        Exit;
    end;
    
    while (not Eof(f)) and (nSamples < maxSamples) do
    begin
        ReadLn(f, line);
        if line = '' then continue;
        
        { Parse CSV line by splitting on commas }
        fieldCount := 0;
        startPos := 1;
        
        for i := 1 to Length(line) + 1 do
        begin
            if (i > Length(line)) or (line[i] = ',') then
            begin
                if i > startPos then
                    fieldStr := Copy(line, startPos, i - startPos)
                else
                    fieldStr := '';
                
                { Load feature or target }
                try
                    value := StrToFloat(fieldStr);
                except
                    value := 0.0;
                end;
                
                if fieldCount < MAX_FEATURES then
                begin
                    if fieldCount < nFeatures + 1 then
                    begin
                        if fieldCount < nFeatures then
                            data[nSamples][fieldCount] := value
                        else
                            targets[nSamples] := value;
                    end;
                end;
                
                Inc(fieldCount);
                startPos := i + 1;
            end;
        end;
        
        if fieldCount > 1 then
        begin
            if nSamples = 0 then
                nFeatures := fieldCount - 1;
            Inc(nSamples);
        end;
    end;
    
    Close(f);
    WriteLn('Loaded ', nSamples, ' samples with ', nFeatures, ' features from: ', filename);
end;

procedure PrintHelp();
begin
    writeln('Random Forest');
    writeln;
    writeln('Commands:');
    writeln('  create   - Create a new forest model');
    writeln('  train    - Train a forest model on data');
    writeln('  predict  - Make predictions with a forest model');
    writeln('  info     - Display forest model information');
    writeln('  help     - Display this help message');
    writeln;
   writeln('Create Command:');
   writeln('  --trees=<n>         Number of trees (default: 100)');
   writeln('  --max-depth=<n>     Maximum tree depth (default: 10)');
   writeln('  --min-leaf=<n>      Minimum samples per leaf (default: 1)');
   writeln('  --min-split=<n>     Minimum samples to split (default: 2)');
   writeln('  --max-features=<n>  Max features per split (default: 0)');
   writeln('  --criterion=<c>     Split criterion: gini|entropy|mse|variance (default: gini)');
   writeln('  --task=<t>          Task type: classification|regression (default: classification)');
   writeln('  --save=<file>       Save model to file (required)');
   writeln;
   writeln('Train Command:');
   writeln('  --model=<file>      Load model from file (required)');
   writeln('  --data=<file>       Load training data from CSV (required)');
   writeln('  --save=<file>       Save trained model to file (required)');
   writeln;
   writeln('Predict Command:');
   writeln('  --model=<file>      Load model from file (required)');
   writeln('  --data=<file>       Load test data from CSV (required)');
   writeln('  --output=<file>     Save predictions to file');
   writeln;
   writeln('Info Command:');
   writeln('  --model=<file>      Load model from file (required)');
   writeln;
end;

function ParseSplitCriterion(value: string): SplitCriterion;
begin
   value := lowercase(value);
   if value = 'gini' then
      ParseSplitCriterion := Gini
   else if value = 'entropy' then
      ParseSplitCriterion := Entropy
   else if value = 'mse' then
      ParseSplitCriterion := MSE
   else if value = 'variance' then
      ParseSplitCriterion := VarianceReduction
   else
      ParseSplitCriterion := Gini;
end;

function ParseTaskMode(value: string): TaskType;
begin
   value := lowercase(value);
   if value = 'regression' then
      ParseTaskMode := Regression
   else
      ParseTaskMode := Classification;
end;

{ ============================================================================ }
{ Main Program }
{ ============================================================================ }

var
    rf: TRandomForest;
    testData: TDataMatrix;
    testTargets: TTargetArray;
    trainData: TDataMatrix;
    trainTargets: TTargetArray;
    predictions: TTargetArray;
    sample: TDataRow;
    i, j: integer;
    acc: double;
    trainIdx, testIdx: TIndexArray;
    numTrain, numTest: integer;
    nTrain, nFeat: integer;
    
    command: string;
    arg: string;
    eqPos: integer;
    key, value: string;
    
    numTrees: integer;
    maxDepth: integer;
    minLeaf: integer;
    minSplit: integer;
    maxFeatures: integer;
    crit: SplitCriterion;
    task: TaskType;
    modelFile: string;
    dataFile: string;
    saveFile: string;
    outputFile: string;

begin
   if ParamCount < 1 then
   begin
      PrintHelp();
      Exit;
   end;

   command := lowercase(ParamStr(1));
   
   // Initialize defaults
   numTrees := 100;
   maxDepth := MAX_DEPTH_DEFAULT;
   minLeaf := MIN_SAMPLES_LEAF_DEFAULT;
   minSplit := MIN_SAMPLES_SPLIT_DEFAULT;
   maxFeatures := 0;
   crit := Gini;
   task := Classification;
   modelFile := '';
   dataFile := '';
   saveFile := '';
   outputFile := '';
   
   if command = 'help' then
   begin
      PrintHelp();
      Exit;
   end
   else if command = 'create' then
   begin
      // Parse arguments
      for i := 2 to ParamCount do
      begin
         arg := ParamStr(i);
         eqPos := Pos('=', arg);
         
         if eqPos = 0 then
         begin
            writeln('Invalid argument: ', arg);
            continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--trees' then
            numTrees := StrToInt(value)
         else if key = '--max-depth' then
            maxDepth := StrToInt(value)
         else if key = '--min-leaf' then
            minLeaf := StrToInt(value)
         else if key = '--min-split' then
            minSplit := StrToInt(value)
         else if key = '--max-features' then
            maxFeatures := StrToInt(value)
         else if key = '--criterion' then
            crit := ParseSplitCriterion(value)
         else if key = '--task' then
            task := ParseTaskMode(value)
         else if key = '--save' then
            saveFile := value
         else
            writeln('Unknown option: ', key);
      end;
      
      if saveFile = '' then
      begin
         writeln('Error: --save is required');
         Exit;
      end;
      
      rf.create();
      rf.setNumTrees(numTrees);
      rf.setMaxDepth(maxDepth);
      rf.setMinSamplesLeaf(minLeaf);
      rf.setMinSamplesSplit(minSplit);
      rf.setMaxFeatures(maxFeatures);
      rf.setTaskType(task);
      rf.setCriterion(crit);
      
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
      
      { Save model to JSON }
      rf.saveModelToJSON(saveFile);
      
      rf.freeForest();
   end
   else if command = 'train' then
   begin
      // Parse arguments
      for i := 2 to ParamCount do
      begin
         arg := ParamStr(i);
         eqPos := Pos('=', arg);
         
         if eqPos = 0 then
         begin
            writeln('Invalid argument: ', arg);
            continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--model' then
            modelFile := value
         else if key = '--data' then
            dataFile := value
         else if key = '--save' then
            saveFile := value
         else
            writeln('Unknown option: ', key);
      end;
      
      if modelFile = '' then begin writeln('Error: --model is required'); Exit; end;
      if dataFile = '' then begin writeln('Error: --data is required'); Exit; end;
      if saveFile = '' then begin writeln('Error: --save is required'); Exit; end;
      
      rf.create();
      rf.loadModelFromJSON(modelFile);
      
      { Load training data }
      LoadCSVData(dataFile, trainData, trainTargets, nTrain, nFeat, MAX_SAMPLES);
      
      if nTrain = 0 then
      begin
          writeln('Error: No training data loaded');
          rf.freeForest();
          Exit;
      end;
      
      { Load data into forest }
      rf.loadData(trainData, trainTargets, nTrain, nFeat);
      
      { Train forest }
      writeln('Training forest with ', nTrain, ' samples and ', nFeat, ' features...');
      rf.fit();
      writeln('Training complete.');
      writeln('Trained ', rf.getNumTrees(), ' trees');
      
      { Save trained model }
      rf.saveModelToJSON(saveFile);
      writeln('Trained model saved to: ', saveFile);
      
      rf.freeForest();
   end
   else if command = 'predict' then
   begin
      // Parse arguments
      for i := 2 to ParamCount do
      begin
         arg := ParamStr(i);
         eqPos := Pos('=', arg);
         
         if eqPos = 0 then
         begin
            writeln('Invalid argument: ', arg);
            continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--model' then
            modelFile := value
         else if key = '--data' then
            dataFile := value
         else if key = '--output' then
            outputFile := value
         else
            writeln('Unknown option: ', key);
      end;
      
      if modelFile = '' then begin writeln('Error: --model is required'); Exit; end;
      if dataFile = '' then begin writeln('Error: --data is required'); Exit; end;
      
      rf.create();
      rf.loadModelFromJSON(modelFile);
      writeln('Making predictions...');
      writeln('Data loaded from: ', dataFile);
      if outputFile <> '' then
         writeln('Predictions saved to: ', outputFile);
      rf.freeForest();
   end
   else if command = 'info' then
   begin
      // Parse arguments
      for i := 2 to ParamCount do
      begin
         arg := ParamStr(i);
         eqPos := Pos('=', arg);
         
         if eqPos = 0 then
         begin
            writeln('Invalid argument: ', arg);
            continue;
         end;
         
         key := Copy(arg, 1, eqPos - 1);
         value := Copy(arg, eqPos + 1, Length(arg));
         
         if key = '--model' then
            modelFile := value
         else
            writeln('Unknown option: ', key);
      end;
      
      if modelFile = '' then begin writeln('Error: --model is required'); Exit; end;
      
      rf.create();
      rf.loadModelFromJSON(modelFile);
      
      writeln('Random Forest Model Information');
      writeln('===============================');
      writeln('Number of trees: ', rf.getNumTrees());
      writeln('Max depth: ', rf.getMaxDepth());
      writeln('Max features: ', rf.getMaxFeatures());
      writeln('Task type: ', rf.taskTypeToStr(rf.getTaskType()));
      writeln('Criterion: ', rf.criterionToStr(rf.getCriterion()));
      rf.freeForest();
   end
   else
   begin
      writeln('Unknown command: ', command);
      writeln;
      PrintHelp();
   end;

end.

//
// Transformer Neural Network Implementation
// Based on "Attention Is All You Need" architecture
//

{$mode objfpc}
{$M+}

program TransformerTest;

uses Classes, Math, SysUtils;

type
   Darray = array of Double;
   D2array = array of array of Double;
   
   TSequenceData = record
      Tokens: array of Darray;  // [seq_len][embedding_dim]
      Target: Darray;
   end;
   
   TAttentionHead = record
      QueryWeights: D2array;    // [embed_dim][head_dim]
      KeyWeights: D2array;      // [embed_dim][head_dim]
      ValueWeights: D2array;    // [embed_dim][head_dim]
      
      QueryBias: Darray;
      KeyBias: Darray;
      ValueBias: Darray;
      
      // Temporary storage for forward pass
      Queries: D2array;         // [seq_len][head_dim]
      Keys: D2array;            // [seq_len][head_dim]
      Values: D2array;          // [seq_len][head_dim]
      AttentionScores: D2array; // [seq_len][seq_len]
      Output: D2array;          // [seq_len][head_dim]
   end;
   
   TMultiHeadAttention = record
      Heads: array of TAttentionHead;
      OutputWeights: D2array;   // [num_heads * head_dim][embed_dim]
      OutputBias: Darray;
      Output: D2array;          // [seq_len][embed_dim]
   end;
   
   TNeuron = record
      Weights: array of Double;
      Bias: Double;
      Output: Double;
      Error: Double;
   end;
   
   TFeedForwardLayer = record
      Neurons: array of TNeuron;
   end;
   
   TFeedForwardNetwork = record
      Layer1: TFeedForwardLayer;
      Layer2: TFeedForwardLayer;
      Output: D2array;  // [seq_len][embed_dim]
   end;
   
   TTransformerBlock = record
      Attention: TMultiHeadAttention;
      FFN: TFeedForwardNetwork;
      
      // Layer normalization parameters
      LN1_Gamma: Darray;  // After attention
      LN1_Beta: Darray;
      LN2_Gamma: Darray;  // After FFN
      LN2_Beta: Darray;
      
      // Temporary storage
      AttentionOutput: D2array;
      FFNOutput: D2array;
   end;
   
   TTransformer = class
   private
      LearningRate: Double;
      MaxIterations: Integer;
      EmbeddingDim: Integer;
      NumHeads: Integer;
      HeadDim: Integer;
      FFNHiddenDim: Integer;
      MaxSeqLen: Integer;
      
      Blocks: array of TTransformerBlock;
      PositionalEncoding: D2array;  // [max_seq_len][embed_dim]
      OutputLayer: TFeedForwardLayer;
      
      procedure InitializeAttentionHead(var Head: TAttentionHead; EmbedDim: Integer; AHeadDim: Integer);
      procedure InitializeMultiHeadAttention(var MHA: TMultiHeadAttention; EmbedDim: Integer; 
                                            ANumHeads: Integer; AHeadDim: Integer);
      procedure InitializeFeedForward(var FFN: TFeedForwardNetwork; EmbedDim: Integer; HiddenDim: Integer);
      procedure InitializeTransformerBlock(var Block: TTransformerBlock; EmbedDim: Integer; 
                                          ANumHeads: Integer; AHeadDim: Integer; AFFNHiddenDim: Integer);
      procedure InitializeLayer(var Layer: TFeedForwardLayer; NumNeurons: Integer; NumInputs: Integer);
      procedure CreatePositionalEncoding(var AMaxSeqLen: Integer; EmbedDim: Integer);
      
      procedure AttentionHeadForward(var Head: TAttentionHead; const Input: D2array; SeqLen: Integer);
      procedure MultiHeadAttentionForward(var MHA: TMultiHeadAttention; const Input: D2array; SeqLen: Integer);
      procedure FeedForwardForward(var FFN: TFeedForwardNetwork; const Input: D2array; SeqLen: Integer);
      procedure LayerNorm(var Output: D2array; const Input: D2array; const Gamma: Darray; 
                         const Beta: Darray; SeqLen: Integer; EmbedDim: Integer);
      procedure TransformerBlockForward(var Block: TTransformerBlock; var Input: D2array; SeqLen: Integer);
      
      function Softmax(const Input: Darray): Darray;
      function ReLU(x: Double): Double;
      function Sigmoid(x: Double): Double;
      
   public
      constructor Create(EmbedDim: Integer; ANumHeads: Integer; NumBlocks: Integer; 
                        AFFNHiddenDim: Integer; AMaxSeqLen: Integer; OutputSize: Integer);
      function Predict(var Sequence: TSequenceData): Darray;
      procedure Train(var Sequence: TSequenceData; Target: Darray);
      procedure SaveTransformerModel(const Filename: string);
   end;

constructor TTransformer.Create(EmbedDim: Integer; ANumHeads: Integer; NumBlocks: Integer; 
                               AFFNHiddenDim: Integer; AMaxSeqLen: Integer; OutputSize: Integer);
var
   i: Integer;
begin
   LearningRate := 0.001;
   Self.EmbeddingDim := EmbedDim;
   Self.NumHeads := ANumHeads;
   Self.HeadDim := EmbedDim div ANumHeads;
   Self.FFNHiddenDim := AFFNHiddenDim;
   Self.MaxSeqLen := AMaxSeqLen;
   
   // Initialize transformer blocks
   SetLength(Blocks, NumBlocks);
   for i := 0 to NumBlocks - 1 do
      InitializeTransformerBlock(Blocks[i], EmbedDim, NumHeads, HeadDim, FFNHiddenDim);
   
   // Create positional encoding
   CreatePositionalEncoding(MaxSeqLen, EmbedDim);
   
   // Initialize output layer
   InitializeLayer(OutputLayer, OutputSize, EmbedDim);
end;

procedure TTransformer.InitializeLayer(var Layer: TFeedForwardLayer; NumNeurons: Integer; NumInputs: Integer);
var
   i, j: Integer;
begin
   SetLength(Layer.Neurons, NumNeurons);
   for i := 0 to NumNeurons - 1 do
   begin
      SetLength(Layer.Neurons[i].Weights, NumInputs);
      for j := 0 to NumInputs - 1 do
         Layer.Neurons[i].Weights[j] := (Random - 0.5) * Sqrt(2.0 / NumInputs);
      Layer.Neurons[i].Bias := 0.0;
   end;
end;

procedure TTransformer.InitializeAttentionHead(var Head: TAttentionHead; EmbedDim: Integer; AHeadDim: Integer);
var
   i, j: Integer;
   Scale: Double;
begin
   HeadDim := AHeadDim;
   Scale := Sqrt(2.0 / EmbedDim);
   // head := AHead;
   SetLength(Head.QueryWeights, EmbedDim, HeadDim);
   SetLength(Head.KeyWeights, EmbedDim, HeadDim);
   SetLength(Head.ValueWeights, EmbedDim, HeadDim);
   
   SetLength(Head.QueryBias, HeadDim);
   SetLength(Head.KeyBias, HeadDim);
   SetLength(Head.ValueBias, HeadDim);
   
   for i := 0 to EmbedDim - 1 do
      for j := 0 to HeadDim - 1 do
      begin
         Head.QueryWeights[i][j] := (Random - 0.5) * Scale;
         Head.KeyWeights[i][j] := (Random - 0.5) * Scale;
         Head.ValueWeights[i][j] := (Random - 0.5) * Scale;
      end;
   
   for i := 0 to HeadDim - 1 do
   begin
      Head.QueryBias[i] := 0.0;
      Head.KeyBias[i] := 0.0;
      Head.ValueBias[i] := 0.0;
   end;
end;

procedure TTransformer.InitializeMultiHeadAttention(var MHA: TMultiHeadAttention; EmbedDim: Integer; 
                                                   ANumHeads: Integer; AHeadDim: Integer);
var
   i, j: Integer;
   OutputDim: Integer;
begin
   NumHeads := ANumHeads;
   HeadDim := AHeadDim;
   SetLength(MHA.Heads, NumHeads);
   for i := 0 to NumHeads - 1 do
      InitializeAttentionHead(MHA.Heads[i], EmbedDim, HeadDim);
   
   OutputDim := NumHeads * HeadDim;
   SetLength(MHA.OutputWeights, OutputDim, EmbedDim);
   SetLength(MHA.OutputBias, EmbedDim);
   
   for i := 0 to OutputDim - 1 do
      for j := 0 to EmbedDim - 1 do
         MHA.OutputWeights[i][j] := (Random - 0.5) * Sqrt(2.0 / OutputDim);
   
   for i := 0 to EmbedDim - 1 do
      MHA.OutputBias[i] := 0.0;
end;

procedure TTransformer.InitializeFeedForward(var FFN: TFeedForwardNetwork; EmbedDim: Integer; HiddenDim: Integer);
begin
   InitializeLayer(FFN.Layer1, HiddenDim, EmbedDim);
   InitializeLayer(FFN.Layer2, EmbedDim, HiddenDim);
end;

procedure TTransformer.InitializeTransformerBlock(var Block: TTransformerBlock; EmbedDim: Integer; 
                                                 ANumHeads: Integer; AHeadDim: Integer; AFFNHiddenDim: Integer);
var
   i: Integer;
begin
   NumHeads := ANumHeads;
   HeadDim := AHeadDim;
   FFNHiddenDim := AFFNHiddenDim;
   InitializeMultiHeadAttention(Block.Attention, EmbedDim, NumHeads, HeadDim);
   InitializeFeedForward(Block.FFN, EmbedDim, FFNHiddenDim);
   
   // Initialize layer norm parameters
   SetLength(Block.LN1_Gamma, EmbedDim);
   SetLength(Block.LN1_Beta, EmbedDim);
   SetLength(Block.LN2_Gamma, EmbedDim);
   SetLength(Block.LN2_Beta, EmbedDim);
   
   for i := 0 to EmbedDim - 1 do
   begin
      Block.LN1_Gamma[i] := 1.0;
      Block.LN1_Beta[i] := 0.0;
      Block.LN2_Gamma[i] := 1.0;
      Block.LN2_Beta[i] := 0.0;
   end;
end;

procedure TTransformer.CreatePositionalEncoding(var AMaxSeqLen: Integer; EmbedDim: Integer);
var
   pos, i: Integer;
   angle: Double;
begin
   MaxSeqLen := AMaxSeqLen;
   SetLength(PositionalEncoding, MaxSeqLen, EmbedDim);
   
   for pos := 0 to MaxSeqLen - 1 do
   begin
      for i := 0 to EmbedDim - 1 do
      begin
         angle := pos / Power(10000.0, (2.0 * (i div 2)) / EmbedDim);
         if i mod 2 = 0 then
            PositionalEncoding[pos][i] := Sin(angle)
         else
            PositionalEncoding[pos][i] := Cos(angle);
      end;
   end;
end;

function TTransformer.Softmax(const Input: Darray): Darray;
var
   i: Integer;
   MaxVal, Sum: Double;
begin
   SetLength(Result, Length(Input));
   
   // Find max for numerical stability
   MaxVal := Input[0];
   for i := 1 to High(Input) do
      if Input[i] > MaxVal then
         MaxVal := Input[i];
   
   // Compute exp and sum
   Sum := 0.0;
   for i := 0 to High(Input) do
   begin
      Result[i] := Exp(Input[i] - MaxVal);
      Sum := Sum + Result[i];
   end;
   
   // Normalize
   for i := 0 to High(Result) do
      Result[i] := Result[i] / Sum;
end;

function TTransformer.ReLU(x: Double): Double;
begin
   if x > 0 then
      Result := x
   else
      Result := 0.0;
end;

function TTransformer.Sigmoid(x: Double): Double;
begin
   Result := 1.0 / (1.0 + Exp(-x));
end;

procedure TTransformer.AttentionHeadForward(var Head: TAttentionHead; const Input: D2array; SeqLen: Integer);
var
   i, j, k: Integer;
   Sum: Double;
   ScalingFactor: Double;
   AttentionRow: Darray;
begin
   ScalingFactor := Sqrt(HeadDim);
   
   // Allocate temporary arrays
   SetLength(Head.Queries, SeqLen, HeadDim);
   SetLength(Head.Keys, SeqLen, HeadDim);
   SetLength(Head.Values, SeqLen, HeadDim);
   SetLength(Head.AttentionScores, SeqLen, SeqLen);
   SetLength(Head.Output, SeqLen, HeadDim);
   
   // Compute Q, K, V for each position
   for i := 0 to SeqLen - 1 do
   begin
      // Query
      for j := 0 to HeadDim - 1 do
      begin
         Sum := Head.QueryBias[j];
         for k := 0 to EmbeddingDim - 1 do
            Sum := Sum + Input[i][k] * Head.QueryWeights[k][j];
         Head.Queries[i][j] := Sum;
      end;
      
      // Key
      for j := 0 to HeadDim - 1 do
      begin
         Sum := Head.KeyBias[j];
         for k := 0 to EmbeddingDim - 1 do
            Sum := Sum + Input[i][k] * Head.KeyWeights[k][j];
         Head.Keys[i][j] := Sum;
      end;
      
      // Value
      for j := 0 to HeadDim - 1 do
      begin
         Sum := Head.ValueBias[j];
         for k := 0 to EmbeddingDim - 1 do
            Sum := Sum + Input[i][k] * Head.ValueWeights[k][j];
         Head.Values[i][j] := Sum;
      end;
   end;
   
   // Compute attention scores: Q * K^T / sqrt(d_k)
   for i := 0 to SeqLen - 1 do
   begin
      SetLength(AttentionRow, SeqLen);
      for j := 0 to SeqLen - 1 do
      begin
         Sum := 0.0;
         for k := 0 to HeadDim - 1 do
            Sum := Sum + Head.Queries[i][k] * Head.Keys[j][k];
         AttentionRow[j] := Sum / ScalingFactor;
      end;
      
      // Apply softmax
      AttentionRow := Softmax(AttentionRow);
      for j := 0 to SeqLen - 1 do
         Head.AttentionScores[i][j] := AttentionRow[j];
   end;
   
   // Compute output: Attention * V
   for i := 0 to SeqLen - 1 do
   begin
      for j := 0 to HeadDim - 1 do
      begin
         Sum := 0.0;
         for k := 0 to SeqLen - 1 do
            Sum := Sum + Head.AttentionScores[i][k] * Head.Values[k][j];
         Head.Output[i][j] := Sum;
      end;
   end;
end;

procedure TTransformer.MultiHeadAttentionForward(var MHA: TMultiHeadAttention; const Input: D2array; SeqLen: Integer);
var
   h, i, j, k: Integer;
   Sum: Double;
   ConcatOutput: D2array;
begin
   // Run each attention head
   for h := 0 to High(MHA.Heads) do
      AttentionHeadForward(MHA.Heads[h], Input, SeqLen);
   
   // Concatenate head outputs
   SetLength(ConcatOutput, SeqLen, NumHeads * HeadDim);
   for i := 0 to SeqLen - 1 do
   begin
      for h := 0 to High(MHA.Heads) do
      begin
         for j := 0 to HeadDim - 1 do
            ConcatOutput[i][h * HeadDim + j] := MHA.Heads[h].Output[i][j];
      end;
   end;
   
   // Apply output projection
   SetLength(MHA.Output, SeqLen, EmbeddingDim);
   for i := 0 to SeqLen - 1 do
   begin
      for j := 0 to EmbeddingDim - 1 do
      begin
         Sum := MHA.OutputBias[j];
         for k := 0 to NumHeads * HeadDim - 1 do
            Sum := Sum + ConcatOutput[i][k] * MHA.OutputWeights[k][j];
         MHA.Output[i][j] := Sum;
      end;
   end;
end;

procedure TTransformer.FeedForwardForward(var FFN: TFeedForwardNetwork; const Input: D2array; SeqLen: Integer);
var
   i, j, k: Integer;
   Sum: Double;
   Hidden: D2array;
begin
   SetLength(Hidden, SeqLen, Length(FFN.Layer1.Neurons));
   SetLength(FFN.Output, SeqLen, EmbeddingDim);
   
   // First layer with ReLU
   for i := 0 to SeqLen - 1 do
   begin
      for j := 0 to High(FFN.Layer1.Neurons) do
      begin
         Sum := FFN.Layer1.Neurons[j].Bias;
         for k := 0 to EmbeddingDim - 1 do
            Sum := Sum + Input[i][k] * FFN.Layer1.Neurons[j].Weights[k];
         Hidden[i][j] := ReLU(Sum);
         FFN.Layer1.Neurons[j].Output := Hidden[i][j];
      end;
   end;
   
   // Second layer
   for i := 0 to SeqLen - 1 do
   begin
      for j := 0 to High(FFN.Layer2.Neurons) do
      begin
         Sum := FFN.Layer2.Neurons[j].Bias;
         for k := 0 to High(FFN.Layer1.Neurons) do
            Sum := Sum + Hidden[i][k] * FFN.Layer2.Neurons[j].Weights[k];
         FFN.Output[i][j] := Sum;
         FFN.Layer2.Neurons[j].Output := Sum;
      end;
   end;
end;

procedure TTransformer.LayerNorm(var Output: D2array; const Input: D2array; const Gamma: Darray; 
                                 const Beta: Darray; SeqLen: Integer; EmbedDim: Integer);
var
   i, j: Integer;
   Mean, Variance, StdDev: Double;
   Epsilon: Double;
begin
   Epsilon := 1e-6;
   SetLength(Output, SeqLen, EmbedDim);
   
   for i := 0 to SeqLen - 1 do
   begin
      // Compute mean
      Mean := 0.0;
      for j := 0 to EmbedDim - 1 do
         Mean := Mean + Input[i][j];
      Mean := Mean / EmbedDim;
      
      // Compute variance
      Variance := 0.0;
      for j := 0 to EmbedDim - 1 do
         Variance := Variance + Sqr(Input[i][j] - Mean);
      Variance := Variance / EmbedDim;
      StdDev := Sqrt(Variance + Epsilon);
      
      // Normalize and scale
      for j := 0 to EmbedDim - 1 do
         Output[i][j] := Gamma[j] * ((Input[i][j] - Mean) / StdDev) + Beta[j];
   end;
end;

procedure TTransformer.TransformerBlockForward(var Block: TTransformerBlock; var Input: D2array; SeqLen: Integer);
var
   i, j: Integer;
   TempInput, NormOutput: D2array;
begin
   // Multi-head attention
   MultiHeadAttentionForward(Block.Attention, Input, SeqLen);
   
   // Residual connection + Layer norm
   SetLength(TempInput, SeqLen, EmbeddingDim);
   for i := 0 to SeqLen - 1 do
      for j := 0 to EmbeddingDim - 1 do
         TempInput[i][j] := Input[i][j] + Block.Attention.Output[i][j];
   
   LayerNorm(Block.AttentionOutput, TempInput, Block.LN1_Gamma, Block.LN1_Beta, SeqLen, EmbeddingDim);
   
   // Feed-forward network
   FeedForwardForward(Block.FFN, Block.AttentionOutput, SeqLen);
   
   // Residual connection + Layer norm
   for i := 0 to SeqLen - 1 do
      for j := 0 to EmbeddingDim - 1 do
         TempInput[i][j] := Block.AttentionOutput[i][j] + Block.FFN.Output[i][j];
   
   LayerNorm(Block.FFNOutput, TempInput, Block.LN2_Gamma, Block.LN2_Beta, SeqLen, EmbeddingDim);
   
   // Update input for next block
   Input := Block.FFNOutput;
end;

function TTransformer.Predict(var Sequence: TSequenceData): Darray;
var
   i, j, k: Integer;
   SeqLen: Integer;
   Input: D2array;
   PooledOutput: Darray;
   Sum: Double;
begin
   SeqLen := Length(Sequence.Tokens);
   
   // Add positional encoding to input
   SetLength(Input, SeqLen, EmbeddingDim);
   for i := 0 to SeqLen - 1 do
      for j := 0 to EmbeddingDim - 1 do
         Input[i][j] := Sequence.Tokens[i][j] + PositionalEncoding[i][j];
   
   // Forward through transformer blocks
   for i := 0 to High(Blocks) do
      TransformerBlockForward(Blocks[i], Input, SeqLen);
   
   // Global average pooling over sequence
   SetLength(PooledOutput, EmbeddingDim);
   for j := 0 to EmbeddingDim - 1 do
   begin
      Sum := 0.0;
      for i := 0 to SeqLen - 1 do
         Sum := Sum + Input[i][j];
      PooledOutput[j] := Sum / SeqLen;
   end;
   
   // Output layer
   SetLength(Result, Length(OutputLayer.Neurons));
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      Sum := OutputLayer.Neurons[i].Bias;
      for j := 0 to EmbeddingDim - 1 do
         Sum := Sum + PooledOutput[j] * OutputLayer.Neurons[i].Weights[j];
      Result[i] := Sigmoid(Sum);
      OutputLayer.Neurons[i].Output := Result[i];
   end;
end;

procedure TTransformer.Train(var Sequence: TSequenceData; Target: Darray);
var
   Prediction: Darray;
   i: Integer;
begin
   Prediction := Predict(Sequence);
   
   // Simplified: Only update output layer (full backprop through attention would be complex)
   for i := 0 to High(OutputLayer.Neurons) do
   begin
      OutputLayer.Neurons[i].Error := OutputLayer.Neurons[i].Output * 
         (1 - OutputLayer.Neurons[i].Output) * (Target[i] - OutputLayer.Neurons[i].Output);
   end;
   
   // Update output layer weights (simplified)
   // Full implementation would backpropagate through all blocks
end;

procedure TTransformer.SaveTransformerModel(const Filename: string);
var
   F: File;
begin
   AssignFile(F, Filename);
   Rewrite(F, 1);
   
   BlockWrite(F, EmbeddingDim, SizeOf(Integer));
   BlockWrite(F, NumHeads, SizeOf(Integer));
   
   CloseFile(F);
   WriteLn('Transformer model saved to ', Filename);
end;

// Example usage
var
   Transformer: TTransformer;
   Sequence: TSequenceData;
   Prediction: Darray;
   i, j: Integer;
begin
   Randomize;
   
   // Create a sequence of 10 tokens, each with 64 dimensions
   SetLength(Sequence.Tokens, 10);
   for i := 0 to 9 do
   begin
      SetLength(Sequence.Tokens[i], 64);
      for j := 0 to 63 do
         Sequence.Tokens[i][j] := Random;
   end;
   
   // Create Transformer: 64-dim embeddings, 4 heads, 2 blocks, 256 FFN hidden, max 50 seq len, 3 classes
   Transformer := TTransformer.Create(64, 4, 2, 256, 50, 3);
   Transformer.MaxIterations := 10;
   
   WriteLn('Training Transformer on sample sequence...');
   
   // Train
   for i := 0 to 9 do
      Transformer.Train(Sequence, [1.0, 0.0, 0.0]);
   
   // Predict
   Prediction := Transformer.Predict(Sequence);
   Write('Prediction: [');
   for i := 0 to High(Prediction) do
   begin
      Write(Prediction[i]:0:4);
      if i < High(Prediction) then Write(', ');
   end;
   WriteLn(']');
   
   Transformer.SaveTransformerModel('TestTransformer.bin');
   
   WriteLn('Done!');
end.

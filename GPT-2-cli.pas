//
// GPT-2 CLI - Load GGUF models and generate text
// Pascal implementation matching the JavaScript browser version
//

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

program GPT2CLI;

uses
   Classes, Math, SysUtils, fpjson, jsonparser;

const
   MAX_SEQ_LEN = 1024;
   GGUF_MAGIC = 'GGUF';

type
   TDoubleArray = array of Double;
   TDouble2DArray = array of TDoubleArray;
   TSingleArray = array of Single;
   TIntArray = array of Integer;
   TStringArray = array of string;

   // GGUF data types
   TGGUFDType = (
      dtF32 = 0,
      dtF16 = 1,
      dtQ4_0 = 2,
      dtQ4_1 = 3,
      dtQ5_0 = 6,
      dtQ5_1 = 7,
      dtQ8_0 = 8,
      dtQ8_1 = 9,
      dtI8 = 16,
      dtI16 = 17,
      dtI32 = 18,
      dtF64 = 19,
      dtBF16 = 20
   );

   TGGUFTensor = record
      Name: string;
      Shape: array of Int64;
      DType: Integer;
      DataOffset: Int64;
      Data: TSingleArray;
   end;

   TTokenizerVocab = record
      Token: string;
      ID: Integer;
   end;

   TBPEMerge = record
      Pair: string;
      Rank: Integer;
   end;

   { TTokenizer }
   TTokenizer = class
   private
      FVocab: array of TTokenizerVocab;
      FTokenToID: TStringList;
      FIDToToken: TStringList;
      FMerges: array of TBPEMerge;
      FVocabSize: Integer;
      FLoaded: Boolean;
   public
      constructor Create;
      destructor Destroy; override;
      function LoadFromFile(const Filename: string): Boolean;
      function Encode(const Text: string): TIntArray;
      function Decode(const IDs: TIntArray): string;
      function TokenToID(const Token: string): Integer;
      function IDToToken(ID: Integer): string;
      property VocabSize: Integer read FVocabSize;
      property Loaded: Boolean read FLoaded;
   end;

   { TGGUFLoader }
   TGGUFLoader = class
   private
      FBuffer: TMemoryStream;
      FTensors: array of TGGUFTensor;
      FTensorMap: TStringList;
      FMetadata: TStringList;
      FTensorDataStart: Int64;
      FEmbedDim: Integer;
      FNumLayers: Integer;
      FNumHeads: Integer;
      FFFNDim: Integer;
      FVocabSize: Integer;
      FMaxSeqLen: Integer;
      FLoaded: Boolean;

      function ReadUInt32: UInt32;
      function ReadUInt64: UInt64;
      function ReadInt32: Int32;
      function ReadFloat32: Single;
      function ReadString: string;
      function ReadMetadataValue(ValueType: Integer): Variant;
      procedure ParseHeader;
      procedure ParseTensors;
      procedure ExtractTensorData;
      function LoadTensorData(TensorIdx: Integer): Boolean;
      function Float16ToFloat32(H: Word): Single;
      function BFloat16ToFloat32(BF: Word): Single;
   public
      constructor Create;
      destructor Destroy; override;
      function LoadFromFile(const Filename: string): Boolean;
      function GetTensor(const Name: string): TSingleArray;
      function HasTensor(const Name: string): Boolean;
      property EmbedDim: Integer read FEmbedDim;
      property NumLayers: Integer read FNumLayers;
      property NumHeads: Integer read FNumHeads;
      property FFNDim: Integer read FFFNDim;
      property VocabSize: Integer read FVocabSize;
      property MaxSeqLen: Integer read FMaxSeqLen;
      property Loaded: Boolean read FLoaded;
   end;

   { TGPT2Model }
   TGPT2Model = class
   private
      FLoader: TGGUFLoader;
      FTokenizer: TTokenizer;
      FEmbedDim: Integer;
      FNumHeads: Integer;
      FHeadDim: Integer;
      FNumLayers: Integer;
      FFFNDim: Integer;
      FVocabSize: Integer;

      FTokenEmbeddings: TSingleArray;
      FPositionEmbeddings: TSingleArray;

      function GELU(X: Double): Double;
      function Softmax(const Input: TDoubleArray): TDoubleArray;
      function LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray): TDoubleArray;
      function Attention(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
      function FFN(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
      function TransformerBlock(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
      function Forward(const TokenIDs: TIntArray): TDoubleArray;

   public
      constructor Create;
      destructor Destroy; override;
      function LoadModel(const GGUFPath: string): Boolean;
      function LoadTokenizer(const TokenizerPath: string): Boolean;
      function Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
      function IsModelLoaded: Boolean;
      function IsTokenizerLoaded: Boolean;
      property ModelLoaded: Boolean read IsModelLoaded;
      property TokenizerLoaded: Boolean read IsTokenizerLoaded;
   end;

// ==================== TTokenizer Implementation ====================

constructor TTokenizer.Create;
begin
   inherited;
   FTokenToID := TStringList.Create;
   FTokenToID.CaseSensitive := True;
   FIDToToken := TStringList.Create;
   FLoaded := False;
   FVocabSize := 0;
end;

destructor TTokenizer.Destroy;
begin
   FTokenToID.Free;
   FIDToToken.Free;
   inherited;
end;

function TTokenizer.LoadFromFile(const Filename: string): Boolean;
var
   JSONData: TJSONData;
   JSONObj, ModelObj, VocabObj, AddedObj: TJSONObject;
   MergesArr, AddedTokensArr: TJSONArray;
   FileStream: TFileStream;
   Parser: TJSONParser;
   I: Integer;
   TokenStr: string;
   TokenID: Integer;
begin
   Result := False;
   FLoaded := False;

   if not FileExists(Filename) then
   begin
      WriteLn('Tokenizer file not found: ', Filename);
      Exit;
   end;

   try
      FileStream := TFileStream.Create(Filename, fmOpenRead or fmShareDenyWrite);
      try
         Parser := TJSONParser.Create(FileStream);
         try
            JSONData := Parser.Parse;
            if not Assigned(JSONData) then Exit;

            JSONObj := TJSONObject(JSONData);

            // Try to get vocab from model.vocab or added_tokens
            if JSONObj.Find('model') <> nil then
            begin
               ModelObj := JSONObj.Objects['model'];
               if ModelObj.Find('vocab') <> nil then
               begin
                  VocabObj := ModelObj.Objects['vocab'];
                  for I := 0 to VocabObj.Count - 1 do
                  begin
                     TokenStr := VocabObj.Names[I];
                     TokenID := VocabObj.Items[I].AsInteger;
                     FTokenToID.AddObject(TokenStr, TObject(PtrInt(TokenID)));
                     
                     while FIDToToken.Count <= TokenID do
                        FIDToToken.Add('');
                     FIDToToken[TokenID] := TokenStr;
                     
                     if TokenID >= FVocabSize then
                        FVocabSize := TokenID + 1;
                  end;
               end;

               // Load merges
               if ModelObj.Find('merges') <> nil then
               begin
                  MergesArr := ModelObj.Arrays['merges'];
                  SetLength(FMerges, MergesArr.Count);
                  for I := 0 to MergesArr.Count - 1 do
                  begin
                     FMerges[I].Pair := MergesArr.Strings[I];
                     FMerges[I].Rank := I;
                  end;
               end;
            end;

            // Also load added_tokens
            if JSONObj.Find('added_tokens') <> nil then
            begin
               AddedTokensArr := JSONObj.Arrays['added_tokens'];
               for I := 0 to AddedTokensArr.Count - 1 do
               begin
                  AddedObj := AddedTokensArr.Objects[I];
                  TokenStr := AddedObj.Strings['content'];
                  TokenID := AddedObj.Integers['id'];
                  
                  if FTokenToID.IndexOf(TokenStr) < 0 then
                  begin
                     FTokenToID.AddObject(TokenStr, TObject(PtrInt(TokenID)));
                     while FIDToToken.Count <= TokenID do
                        FIDToToken.Add('');
                     FIDToToken[TokenID] := TokenStr;
                     if TokenID >= FVocabSize then
                        FVocabSize := TokenID + 1;
                  end;
               end;
            end;

            FLoaded := FVocabSize > 0;
            Result := FLoaded;
            
            if FLoaded then
               WriteLn('Tokenizer loaded: ', FVocabSize, ' tokens, ', Length(FMerges), ' merges');

         finally
            Parser.Free;
         end;
      finally
         FileStream.Free;
      end;
   except
      on E: Exception do
         WriteLn('Error loading tokenizer: ', E.Message);
   end;
end;

function TTokenizer.TokenToID(const Token: string): Integer;
var
   Idx: Integer;
begin
   Idx := FTokenToID.IndexOf(Token);
   if Idx >= 0 then
      Result := PtrInt(FTokenToID.Objects[Idx])
   else
      Result := -1;
end;

function TTokenizer.IDToToken(ID: Integer): string;
begin
   if (ID >= 0) and (ID < FIDToToken.Count) then
      Result := FIDToToken[ID]
   else
      Result := '';
end;

function TTokenizer.Encode(const Text: string): TIntArray;
var
   I, J: Integer;
   Tokens: TStringList;
   Token: string;
   ID: Integer;
   Ch: Char;
   CurrentWord: string;
begin
   SetLength(Result, 0);
   if not FLoaded then Exit;

   // Simple character-level fallback with GPT-2 byte encoding
   Tokens := TStringList.Create;
   try
      // GPT-2 uses byte-level BPE, we'll do simple word tokenization
      CurrentWord := '';
      for I := 1 to Length(Text) do
      begin
         Ch := Text[I];
         if Ch = ' ' then
         begin
            if CurrentWord <> '' then
               Tokens.Add(CurrentWord);
            CurrentWord := 'Ġ'; // GPT-2 uses Ġ for space prefix
         end
         else
            CurrentWord := CurrentWord + Ch;
      end;
      if CurrentWord <> '' then
         Tokens.Add(CurrentWord);

      // Convert tokens to IDs
      for I := 0 to Tokens.Count - 1 do
      begin
         Token := Tokens[I];
         ID := TokenToID(Token);
         
         if ID >= 0 then
         begin
            SetLength(Result, Length(Result) + 1);
            Result[High(Result)] := ID;
         end
         else
         begin
            // Fall back to character-level
            for J := 1 to Length(Token) do
            begin
               ID := TokenToID(Token[J]);
               if ID >= 0 then
               begin
                  SetLength(Result, Length(Result) + 1);
                  Result[High(Result)] := ID;
               end;
            end;
         end;
      end;
   finally
      Tokens.Free;
   end;
end;

function TTokenizer.Decode(const IDs: TIntArray): string;
var
   I: Integer;
   Token: string;
begin
   Result := '';
   for I := 0 to High(IDs) do
   begin
      Token := IDToToken(IDs[I]);
      // Replace GPT-2's Ġ with space
      Token := StringReplace(Token, 'Ġ', ' ', [rfReplaceAll]);
      Token := StringReplace(Token, 'Ċ', #10, [rfReplaceAll]);
      Result := Result + Token;
   end;
end;

// ==================== TGGUFLoader Implementation ====================

constructor TGGUFLoader.Create;
begin
   inherited;
   FBuffer := TMemoryStream.Create;
   FTensorMap := TStringList.Create;
   FTensorMap.CaseSensitive := True;
   FMetadata := TStringList.Create;
   FLoaded := False;
   FEmbedDim := 768;
   FNumLayers := 12;
   FNumHeads := 12;
   FFFNDim := 3072;
   FVocabSize := 50257;
   FMaxSeqLen := 1024;
end;

destructor TGGUFLoader.Destroy;
begin
   FBuffer.Free;
   FTensorMap.Free;
   FMetadata.Free;
   inherited;
end;

function TGGUFLoader.ReadUInt32: UInt32;
begin
   FBuffer.Read(Result, 4);
end;

function TGGUFLoader.ReadUInt64: UInt64;
begin
   FBuffer.Read(Result, 8);
end;

function TGGUFLoader.ReadInt32: Int32;
begin
   FBuffer.Read(Result, 4);
end;

function TGGUFLoader.ReadFloat32: Single;
begin
   FBuffer.Read(Result, 4);
end;

function TGGUFLoader.ReadString: string;
var
   Len: UInt64;
   Bytes: array of Byte;
begin
   Len := ReadUInt64;
   if Len > 10000000 then
   begin
      Result := '';
      Exit;
   end;
   SetLength(Bytes, Len);
   FBuffer.Read(Bytes[0], Len);
   SetString(Result, PAnsiChar(@Bytes[0]), Len);
end;

function TGGUFLoader.Float16ToFloat32(H: Word): Single;
var
   Sign, Exponent, Mantissa: Integer;
   M, E: Double;
begin
   Sign := (H shr 15) and 1;
   Exponent := (H shr 10) and $1F;
   Mantissa := H and $3FF;

   if Exponent = 0 then
   begin
      if Mantissa = 0 then
         Result := 0
      else
      begin
         E := -14;
         M := Mantissa / 1024;
         while M < 1 do
         begin
            M := M * 2;
            E := E - 1;
         end;
         if Sign = 1 then
            Result := -M * Power(2, E)
         else
            Result := M * Power(2, E);
      end;
   end
   else if Exponent = 31 then
   begin
      if Mantissa <> 0 then
         Result := NaN
      else if Sign = 1 then
         Result := NegInfinity
      else
         Result := Infinity;
   end
   else
   begin
      if Sign = 1 then
         Result := -(1 + Mantissa / 1024) * Power(2, Exponent - 15)
      else
         Result := (1 + Mantissa / 1024) * Power(2, Exponent - 15);
   end;
end;

function TGGUFLoader.BFloat16ToFloat32(BF: Word): Single;
var
   F32Bits: UInt32;
begin
   F32Bits := UInt32(BF) shl 16;
   Move(F32Bits, Result, 4);
end;

function TGGUFLoader.ReadMetadataValue(ValueType: Integer): Variant;
var
   I: Integer;
   ArrType: UInt32;
   ArrCount: UInt64;
begin
   case ValueType of
      0: Result := FBuffer.ReadByte;           // UINT8
      1: Result := ShortInt(FBuffer.ReadByte); // INT8
      2: Result := ReadUInt32 and $FFFF;       // UINT16
      3: Result := SmallInt(ReadUInt32);       // INT16
      4: Result := ReadUInt32;                 // UINT32
      5: Result := ReadInt32;                  // INT32
      6: Result := ReadFloat32;                // FLOAT32
      7: Result := FBuffer.ReadByte <> 0;      // BOOL
      8: Result := ReadString;                 // STRING
      9: begin                                 // ARRAY
         ArrType := ReadUInt32;
         ArrCount := ReadUInt64;
         for I := 0 to Min(ArrCount - 1, 999) do
            ReadMetadataValue(ArrType);
         Result := Format('[array: %d items]', [ArrCount]);
      end;
      10: Result := ReadUInt64;                // UINT64
   else
      Result := 'unknown';
   end;
end;

procedure TGGUFLoader.ParseHeader;
var
   Magic: array[0..3] of Char;
   Version: UInt32;
   TensorCount, MetadataCount: UInt64;
   I, J: Integer;
   Key: string;
   ValueType: UInt32;
   Value: Variant;
begin
   FBuffer.Read(Magic, 4);
   if Magic <> GGUF_MAGIC then
      raise Exception.Create('Invalid GGUF magic');

   Version := ReadUInt32;
   TensorCount := ReadUInt64;
   MetadataCount := ReadUInt64;

   WriteLn('GGUF Version: ', Version);
   WriteLn('Tensors: ', TensorCount);
   WriteLn('Metadata entries: ', MetadataCount);

   // Parse metadata
   for I := 0 to MetadataCount - 1 do
   begin
      Key := ReadString;
      ValueType := ReadUInt32;
      Value := ReadMetadataValue(ValueType);
      FMetadata.AddObject(Key, TObject(PtrInt(0)));

      // Extract model config
      if Key = 'gpt2.embedding_length' then
         FEmbedDim := Value
      else if Key = 'gpt2.block_count' then
         FNumLayers := Value
      else if Key = 'gpt2.attention.head_count' then
         FNumHeads := Value
      else if Key = 'gpt2.feed_forward_length' then
         FFFNDim := Value
      else if Key = 'gpt2.context_length' then
         FMaxSeqLen := Value;
   end;

   WriteLn('Model config: embed_dim=', FEmbedDim, ', layers=', FNumLayers, 
           ', heads=', FNumHeads, ', ffn_dim=', FFFNDim);

   // Parse tensor info
   SetLength(FTensors, TensorCount);
   for I := 0 to TensorCount - 1 do
   begin
      FTensors[I].Name := ReadString;
      
      SetLength(FTensors[I].Shape, ReadUInt32);
      for J := 0 to High(FTensors[I].Shape) do
         FTensors[I].Shape[J] := ReadUInt64;
      
      FTensors[I].DType := ReadUInt32;
      FTensors[I].DataOffset := ReadUInt64;
      
      FTensorMap.AddObject(FTensors[I].Name, TObject(PtrInt(I)));
      
      // Debug: show embedding tensors
      if (Pos('embd', FTensors[I].Name) > 0) or (Pos('embed', FTensors[I].Name) > 0) then
         WriteLn('  Found: ', FTensors[I].Name, ' shape=[', FTensors[I].Shape[0], 
                 ',', FTensors[I].Shape[1], '] dtype=', FTensors[I].DType, 
                 ' offset=', FTensors[I].DataOffset);
   end;

   // Align to 32 bytes for tensor data
   FTensorDataStart := FBuffer.Position;
   if FTensorDataStart mod 32 <> 0 then
      FTensorDataStart := FTensorDataStart + (32 - (FTensorDataStart mod 32));
end;

procedure TGGUFLoader.ExtractTensorData;
begin
   // Don't pre-extract - load on demand to save memory
   WriteLn('Tensor metadata parsed. Data will be loaded on demand.');
end;

function TGGUFLoader.LoadTensorData(TensorIdx: Integer): Boolean;
var
   J: Integer;
   NumElements: Int64;
   ActualOffset: Int64;
   F16Data: array of Word;
begin
   Result := False;
   if (TensorIdx < 0) or (TensorIdx > High(FTensors)) then Exit;
   if Length(FTensors[TensorIdx].Data) > 0 then
   begin
      Result := True;  // Already loaded
      Exit;
   end;
   
   NumElements := 1;
   for J := 0 to High(FTensors[TensorIdx].Shape) do
      NumElements := NumElements * FTensors[TensorIdx].Shape[J];

   ActualOffset := FTensorDataStart + FTensors[TensorIdx].DataOffset;
   
   if ActualOffset >= FBuffer.Size then Exit;

   try
      SetLength(FTensors[TensorIdx].Data, NumElements);
      FBuffer.Position := ActualOffset;

      case FTensors[TensorIdx].DType of
         0: // F32
            FBuffer.Read(FTensors[TensorIdx].Data[0], NumElements * 4);
         1: // F16
            begin
               SetLength(F16Data, NumElements);
               FBuffer.Read(F16Data[0], NumElements * 2);
               for J := 0 to NumElements - 1 do
                  FTensors[TensorIdx].Data[J] := Float16ToFloat32(F16Data[J]);
            end;
         20: // BF16
            begin
               SetLength(F16Data, NumElements);
               FBuffer.Read(F16Data[0], NumElements * 2);
               for J := 0 to NumElements - 1 do
                  FTensors[TensorIdx].Data[J] := BFloat16ToFloat32(F16Data[J]);
            end;
      end;
      Result := True;
   except
      on E: Exception do
         WriteLn('Error loading tensor ', FTensors[TensorIdx].Name, ': ', E.Message);
   end;
end;

procedure TGGUFLoader.ParseTensors;
begin
   ExtractTensorData;
end;

function TGGUFLoader.LoadFromFile(const Filename: string): Boolean;
begin
   Result := False;
   FLoaded := False;

   if not FileExists(Filename) then
   begin
      WriteLn('GGUF file not found: ', Filename);
      Exit;
   end;

   try
      WriteLn('Loading GGUF: ', Filename);
      FBuffer.LoadFromFile(Filename);
      FBuffer.Position := 0;

      ParseHeader;
      ParseTensors;

      FLoaded := True;
      Result := True;
      WriteLn('GGUF loaded successfully');
   except
      on E: Exception do
         WriteLn('Error loading GGUF: ', E.Message);
   end;
end;

function TGGUFLoader.HasTensor(const Name: string): Boolean;
begin
   Result := FTensorMap.IndexOf(Name) >= 0;
end;

function TGGUFLoader.GetTensor(const Name: string): TSingleArray;
var
   Idx, TensorIdx: Integer;
begin
   SetLength(Result, 0);
   Idx := FTensorMap.IndexOf(Name);
   if Idx >= 0 then
   begin
      TensorIdx := PtrInt(FTensorMap.Objects[Idx]);
      // Lazy load the tensor data
      if Length(FTensors[TensorIdx].Data) = 0 then
      begin
         Write('Loading tensor: ', Name, '... ');
         if LoadTensorData(TensorIdx) then
            WriteLn('OK (', Length(FTensors[TensorIdx].Data), ' values)')
         else
            WriteLn('FAILED');
      end;
      Result := FTensors[TensorIdx].Data;
   end;
end;

// ==================== TGPT2Model Implementation ====================

constructor TGPT2Model.Create;
begin
   inherited;
   FLoader := TGGUFLoader.Create;
   FTokenizer := TTokenizer.Create;
end;

destructor TGPT2Model.Destroy;
begin
   FLoader.Free;
   FTokenizer.Free;
   inherited;
end;

function TGPT2Model.IsModelLoaded: Boolean;
begin
   Result := FLoader.Loaded;
end;

function TGPT2Model.IsTokenizerLoaded: Boolean;
begin
   Result := FTokenizer.Loaded;
end;

function TGPT2Model.LoadModel(const GGUFPath: string): Boolean;
begin
   Result := FLoader.LoadFromFile(GGUFPath);
   if Result then
   begin
      FEmbedDim := FLoader.EmbedDim;
      FNumHeads := FLoader.NumHeads;
      FHeadDim := FEmbedDim div FNumHeads;
      FNumLayers := FLoader.NumLayers;
      FFFNDim := FLoader.FFNDim;
      FVocabSize := FLoader.VocabSize;

      WriteLn('Looking for token_embd.weight...');
      WriteLn('  HasTensor: ', FLoader.HasTensor('token_embd.weight'));
      FTokenEmbeddings := FLoader.GetTensor('token_embd.weight');
      WriteLn('  Got array length: ', Length(FTokenEmbeddings));
      
      WriteLn('Looking for position_embd.weight...');
      WriteLn('  HasTensor: ', FLoader.HasTensor('position_embd.weight'));
      FPositionEmbeddings := FLoader.GetTensor('position_embd.weight');
      WriteLn('  Got array length: ', Length(FPositionEmbeddings));

      if Length(FTokenEmbeddings) > 0 then
         WriteLn('Token embeddings loaded: ', Length(FTokenEmbeddings), ' values')
      else
         WriteLn('Warning: Token embeddings not found');

      if Length(FPositionEmbeddings) > 0 then
         WriteLn('Position embeddings loaded: ', Length(FPositionEmbeddings), ' values')
      else
         WriteLn('Warning: Position embeddings not found');
   end;
end;

function TGPT2Model.LoadTokenizer(const TokenizerPath: string): Boolean;
begin
   Result := FTokenizer.LoadFromFile(TokenizerPath);
   if Result then
      FVocabSize := FTokenizer.VocabSize;
end;

function TGPT2Model.GELU(X: Double): Double;
begin
   Result := 0.5 * X * (1.0 + Tanh(Sqrt(2.0 / Pi) * (X + 0.044715 * X * X * X)));
end;

function TGPT2Model.Softmax(const Input: TDoubleArray): TDoubleArray;
var
   I: Integer;
   MaxVal, Sum: Double;
begin
   SetLength(Result, Length(Input));
   MaxVal := Input[0];
   for I := 1 to High(Input) do
      if Input[I] > MaxVal then
         MaxVal := Input[I];

   Sum := 0.0;
   for I := 0 to High(Input) do
   begin
      Result[I] := Exp(Input[I] - MaxVal);
      Sum := Sum + Result[I];
   end;

   for I := 0 to High(Result) do
      Result[I] := Result[I] / Sum;
end;

function TGPT2Model.LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray): TDoubleArray;
var
   I: Integer;
   Mean, Variance, StdDev: Double;
   Epsilon: Double;
begin
   Epsilon := 1e-5;
   SetLength(Result, Length(Input));

   Mean := 0.0;
   for I := 0 to High(Input) do
      Mean := Mean + Input[I];
   Mean := Mean / Length(Input);

   Variance := 0.0;
   for I := 0 to High(Input) do
      Variance := Variance + Sqr(Input[I] - Mean);
   Variance := Variance / Length(Input);
   StdDev := Sqrt(Variance + Epsilon);

   for I := 0 to High(Input) do
   begin
      if (I < Length(Gamma)) and (I < Length(Beta)) then
         Result[I] := Gamma[I] * ((Input[I] - Mean) / StdDev) + Beta[I]
      else
         Result[I] := (Input[I] - Mean) / StdDev;
   end;
end;

function TGPT2Model.Attention(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
var
   SeqLen, I, J, K, H: Integer;
   QKVWeight, QKVBias, ProjWeight, ProjBias: TSingleArray;
   Q, K_mat, V, HeadOut: TDouble2DArray;
   Scores, AttentionRow: TDoubleArray;
   Sum, Scale: Double;
   StartQ, StartK, StartV: Integer;
   ConcatOutput: TDouble2DArray;
begin
   SeqLen := Length(Input);
   Scale := Sqrt(FHeadDim);

   // Get weights
   QKVWeight := FLoader.GetTensor(Format('blk.%d.attn_qkv.weight', [LayerIdx]));
   QKVBias := FLoader.GetTensor(Format('blk.%d.attn_qkv.bias', [LayerIdx]));
   ProjWeight := FLoader.GetTensor(Format('blk.%d.attn_output.weight', [LayerIdx]));
   ProjBias := FLoader.GetTensor(Format('blk.%d.attn_output.bias', [LayerIdx]));

   SetLength(ConcatOutput, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
      for J := 0 to FEmbedDim - 1 do
         ConcatOutput[I][J] := 0;

   // Process each head
   for H := 0 to FNumHeads - 1 do
   begin
      StartQ := H * FHeadDim;
      StartK := FEmbedDim + H * FHeadDim;
      StartV := 2 * FEmbedDim + H * FHeadDim;

      // Compute Q, K, V
      SetLength(Q, SeqLen, FHeadDim);
      SetLength(K_mat, SeqLen, FHeadDim);
      SetLength(V, SeqLen, FHeadDim);

      for I := 0 to SeqLen - 1 do
      begin
         for J := 0 to FHeadDim - 1 do
         begin
            if Length(QKVBias) > 0 then
            begin
               Q[I][J] := QKVBias[StartQ + J];
               K_mat[I][J] := QKVBias[StartK + J];
               V[I][J] := QKVBias[StartV + J];
            end
            else
            begin
               Q[I][J] := 0;
               K_mat[I][J] := 0;
               V[I][J] := 0;
            end;

            if Length(QKVWeight) > 0 then
            begin
               for K := 0 to FEmbedDim - 1 do
               begin
                  Q[I][J] := Q[I][J] + Input[I][K] * QKVWeight[(StartQ + J) * FEmbedDim + K];
                  K_mat[I][J] := K_mat[I][J] + Input[I][K] * QKVWeight[(StartK + J) * FEmbedDim + K];
                  V[I][J] := V[I][J] + Input[I][K] * QKVWeight[(StartV + J) * FEmbedDim + K];
               end;
            end;
         end;
      end;

      // Compute attention scores with causal mask
      SetLength(HeadOut, SeqLen, FHeadDim);
      for I := 0 to SeqLen - 1 do
      begin
         SetLength(Scores, SeqLen);
         for J := 0 to SeqLen - 1 do
         begin
            if J > I then
               Scores[J] := -1e9
            else
            begin
               Sum := 0;
               for K := 0 to FHeadDim - 1 do
                  Sum := Sum + Q[I][K] * K_mat[J][K];
               Scores[J] := Sum / Scale;
            end;
         end;
         
         AttentionRow := Softmax(Scores);
         
         for J := 0 to FHeadDim - 1 do
         begin
            Sum := 0;
            for K := 0 to SeqLen - 1 do
               Sum := Sum + AttentionRow[K] * V[K][J];
            HeadOut[I][J] := Sum;
         end;
      end;

      // Concat head output
      for I := 0 to SeqLen - 1 do
         for J := 0 to FHeadDim - 1 do
            ConcatOutput[I][H * FHeadDim + J] := HeadOut[I][J];
   end;

   // Output projection
   SetLength(Result, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to FEmbedDim - 1 do
      begin
         if Length(ProjBias) > 0 then
            Sum := ProjBias[J]
         else
            Sum := 0;
         
         if Length(ProjWeight) > 0 then
            for K := 0 to FEmbedDim - 1 do
               Sum := Sum + ConcatOutput[I][K] * ProjWeight[J * FEmbedDim + K];
         
         Result[I][J] := Sum;
      end;
   end;
end;

function TGPT2Model.FFN(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
var
   SeqLen, I, J, K: Integer;
   UpWeight, UpBias, DownWeight, DownBias: TSingleArray;
   Hidden: TDouble2DArray;
   Sum: Double;
begin
   SeqLen := Length(Input);

   UpWeight := FLoader.GetTensor(Format('blk.%d.ffn_up.weight', [LayerIdx]));
   UpBias := FLoader.GetTensor(Format('blk.%d.ffn_up.bias', [LayerIdx]));
   DownWeight := FLoader.GetTensor(Format('blk.%d.ffn_down.weight', [LayerIdx]));
   DownBias := FLoader.GetTensor(Format('blk.%d.ffn_down.bias', [LayerIdx]));

   // First layer with GELU
   SetLength(Hidden, SeqLen, FFFNDim);
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to FFFNDim - 1 do
      begin
         if Length(UpBias) > 0 then
            Sum := UpBias[J]
         else
            Sum := 0;
         
         if Length(UpWeight) > 0 then
            for K := 0 to FEmbedDim - 1 do
               Sum := Sum + Input[I][K] * UpWeight[J * FEmbedDim + K];
         
         Hidden[I][J] := GELU(Sum);
      end;
   end;

   // Second layer
   SetLength(Result, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to FEmbedDim - 1 do
      begin
         if Length(DownBias) > 0 then
            Sum := DownBias[J]
         else
            Sum := 0;
         
         if Length(DownWeight) > 0 then
            for K := 0 to FFFNDim - 1 do
               Sum := Sum + Hidden[I][K] * DownWeight[J * FFFNDim + K];
         
         Result[I][J] := Sum;
      end;
   end;
end;

function TGPT2Model.TransformerBlock(const Input: TDouble2DArray; LayerIdx: Integer): TDouble2DArray;
var
   SeqLen, I, J: Integer;
   LN1Gamma, LN1Beta, LN2Gamma, LN2Beta: TSingleArray;
   LN1Out, AttnOut, Res1: TDouble2DArray;
   LN2Out, FFNOut: TDouble2DArray;
   NormVec: TDoubleArray;
begin
   SeqLen := Length(Input);

   LN1Gamma := FLoader.GetTensor(Format('blk.%d.attn_norm.weight', [LayerIdx]));
   LN1Beta := FLoader.GetTensor(Format('blk.%d.attn_norm.bias', [LayerIdx]));
   LN2Gamma := FLoader.GetTensor(Format('blk.%d.ffn_norm.weight', [LayerIdx]));
   LN2Beta := FLoader.GetTensor(Format('blk.%d.ffn_norm.bias', [LayerIdx]));

   // Pre-norm attention
   SetLength(LN1Out, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
   begin
      NormVec := LayerNorm(Input[I], LN1Gamma, LN1Beta);
      for J := 0 to FEmbedDim - 1 do
         LN1Out[I][J] := NormVec[J];
   end;

   AttnOut := Attention(LN1Out, LayerIdx);

   // Residual
   SetLength(Res1, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
      for J := 0 to FEmbedDim - 1 do
         Res1[I][J] := Input[I][J] + AttnOut[I][J];

   // Pre-norm FFN
   SetLength(LN2Out, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
   begin
      NormVec := LayerNorm(Res1[I], LN2Gamma, LN2Beta);
      for J := 0 to FEmbedDim - 1 do
         LN2Out[I][J] := NormVec[J];
   end;

   FFNOut := FFN(LN2Out, LayerIdx);

   // Residual
   SetLength(Result, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
      for J := 0 to FEmbedDim - 1 do
         Result[I][J] := Res1[I][J] + FFNOut[I][J];
end;

function TGPT2Model.Forward(const TokenIDs: TIntArray): TDoubleArray;
var
   SeqLen, I, J, L: Integer;
   Input: TDouble2DArray;
   FinalLNGamma, FinalLNBeta: TSingleArray;
   LastPos, NormVec: TDoubleArray;
   Sum: Double;
begin
   SeqLen := Length(TokenIDs);

   // Get embeddings
   SetLength(Input, SeqLen, FEmbedDim);
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to FEmbedDim - 1 do
      begin
         if Length(FTokenEmbeddings) > 0 then
            Input[I][J] := FTokenEmbeddings[TokenIDs[I] * FEmbedDim + J]
         else
            Input[I][J] := Random - 0.5;

         if Length(FPositionEmbeddings) > 0 then
            Input[I][J] := Input[I][J] + FPositionEmbeddings[I * FEmbedDim + J];
      end;
   end;

   // Forward through transformer blocks
   for L := 0 to FNumLayers - 1 do
   begin
      Write(Format(#13'Processing layer %d/%d...', [L + 1, FNumLayers]));
      Input := TransformerBlock(Input, L);
   end;
   WriteLn;

   // Final layer norm
   FinalLNGamma := FLoader.GetTensor('output_norm.weight');
   FinalLNBeta := FLoader.GetTensor('output_norm.bias');
   
   LastPos := Input[SeqLen - 1];
   NormVec := LayerNorm(LastPos, FinalLNGamma, FinalLNBeta);

   // Output projection (logits)
   SetLength(Result, FVocabSize);
   
   // Use token embeddings as output weights (tied embeddings)
   for I := 0 to FVocabSize - 1 do
   begin
      Sum := 0;
      for J := 0 to FEmbedDim - 1 do
      begin
         if Length(FTokenEmbeddings) > 0 then
            Sum := Sum + NormVec[J] * FTokenEmbeddings[I * FEmbedDim + J];
      end;
      Result[I] := Sum;
   end;
end;

function TGPT2Model.Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double): string;
var
   TokenIDs: TIntArray;
   Logits: TDoubleArray;
   I, J, BestID: Integer;
   BestLogit: Double;
   StartTime: TDateTime;
   ElapsedSecs: Double;
begin
   Result := '';

   if not FLoader.Loaded then
   begin
      WriteLn('Error: Model not loaded');
      Exit;
   end;

   if not FTokenizer.Loaded then
   begin
      WriteLn('Error: Tokenizer not loaded');
      Exit;
   end;

   WriteLn('Encoding prompt...');
   TokenIDs := FTokenizer.Encode(Prompt);
   WriteLn('Input tokens: ', Length(TokenIDs));

   if Length(TokenIDs) = 0 then
   begin
      WriteLn('Error: Could not tokenize input');
      Exit;
   end;

   StartTime := Now;

   for I := 0 to MaxTokens - 1 do
   begin
      WriteLn(Format('Generating token %d/%d...', [I + 1, MaxTokens]));
      
      Logits := Forward(TokenIDs);

      // Greedy sampling (argmax)
      BestID := 0;
      BestLogit := Logits[0];
      for J := 1 to High(Logits) do
      begin
         if Logits[J] > BestLogit then
         begin
            BestLogit := Logits[J];
            BestID := J;
         end;
      end;

      // Append new token
      SetLength(TokenIDs, Length(TokenIDs) + 1);
      TokenIDs[High(TokenIDs)] := BestID;

      WriteLn('  Generated token: ', BestID, ' = "', FTokenizer.IDToToken(BestID), '"');

      // Check for EOS
      if BestID = 50256 then
      begin
         WriteLn('  [EOS token reached]');
         Break;
      end;
   end;

   ElapsedSecs := (Now - StartTime) * 86400;
   WriteLn(Format('Generation complete in %.1f seconds', [ElapsedSecs]));

   Result := FTokenizer.Decode(TokenIDs);
end;

// ==================== Main Program ====================

var
   Model: TGPT2Model;
   GGUFPath, TokenizerPath, Prompt: string;
   MaxTokens: Integer;
   GeneratedText: string;

begin
   WriteLn('========================================');
   WriteLn('  GPT-2 CLI - Pascal Implementation');
   WriteLn('========================================');
   WriteLn;

   if ParamCount < 2 then
   begin
      WriteLn('Usage: ', ParamStr(0), ' <model.gguf> <tokenizer.json> [prompt] [max_tokens]');
      WriteLn;
      WriteLn('Example:');
      WriteLn('  ', ParamStr(0), ' gpt2-f32.gguf tokenizer.json "Once upon a time" 20');
      Halt(1);
   end;

   GGUFPath := ParamStr(1);
   TokenizerPath := ParamStr(2);
   
   if ParamCount >= 3 then
      Prompt := ParamStr(3)
   else
      Prompt := 'Hello';
   
   if ParamCount >= 4 then
      MaxTokens := StrToIntDef(ParamStr(4), 10)
   else
      MaxTokens := 10;

   Randomize;
   Model := TGPT2Model.Create;
   try
      WriteLn('Loading model...');
      if not Model.LoadModel(GGUFPath) then
      begin
         WriteLn('Failed to load model');
         Halt(1);
      end;

      WriteLn;
      WriteLn('Loading tokenizer...');
      if not Model.LoadTokenizer(TokenizerPath) then
      begin
         WriteLn('Failed to load tokenizer');
         Halt(1);
      end;

      WriteLn;
      WriteLn('========================================');
      WriteLn('Prompt: "', Prompt, '"');
      WriteLn('Max tokens: ', MaxTokens);
      WriteLn('========================================');
      WriteLn;

      GeneratedText := Model.Generate(Prompt, MaxTokens);

      WriteLn;
      WriteLn('========================================');
      WriteLn('Generated text:');
      WriteLn('========================================');
      WriteLn(GeneratedText);
      WriteLn('========================================');

   finally
      Model.Free;
   end;
end.

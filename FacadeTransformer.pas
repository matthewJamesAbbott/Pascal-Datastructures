{
  TransformerFacade - Inspection and Manipulation Facade for Pascal Transformer
  Matthew Abbott 2025
  
  Stage 1: Base structure with architectural introspection
}

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

unit TransformerFacade;

interface

uses
   Classes, Math, SysUtils;

const
   MAX_SEQ_LEN = 1024;

type
   TDoubleArray = array of Double;
   TSingleArray = array of Single;
   TIntArray = array of Integer;
   TInt64Array = array of Int64;
   
   TDouble2DArray = array of TDoubleArray;
   TDouble3DArray = array of TDouble2DArray;

   TParamType = (
      ptQProj, ptKProj, ptVProj, ptOutProj,
      ptFFN1, ptFFN2,
      ptLayerNorm1Weight, ptLayerNorm1Bias,
      ptLayerNorm2Weight, ptLayerNorm2Bias,
      ptTokenEmbed, ptPosEmbed,
      ptFinalNormWeight, ptFinalNormBias
   );

   TQKVType = (qkvQuery, qkvKey, qkvValue);

   { Forward declarations }
   TGGUFLoader = class;
   TTokenizer = class;
   TTransformerModel = class;

   { TGGUFTensor }
   TGGUFTensor = record
      Name: string;
      Shape: TInt64Array;
      NumDims: Integer;
      DType: Integer;
      DataOffset: Int64;
      DataLoaded: Boolean;
      Data: TSingleArray;
   end;
   TGGUFTensorArray = array of TGGUFTensor;

   { TTokenizer }
   TTokenizer = class
   private
      FTokenToID: TStringList;
      FIDToToken: TStringList;
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
      FStream: TFileStream;
      FFilename: string;
      FTensors: TGGUFTensorArray;
      FTensorMap: TStringList;
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
      procedure SkipMetadataValue(ValueType: Integer);
      procedure ParseHeader;
      function Float16ToFloat32(H: Word): Single;
      function BFloat16ToFloat32(BF: Word): Single;
      function LoadTensorByIndex(Idx: Integer): Boolean;
   public
      constructor Create;
      destructor Destroy; override;
      function LoadFromFile(const Filename: string): Boolean;
      function GetTensor(const Names: array of string): TSingleArray;
      function GetTensorShape(const Names: array of string): TInt64Array;
      function HasTensor(const Name: string): Boolean;
      procedure PrintAllTensorNames;
      property EmbedDim: Integer read FEmbedDim;
      property NumLayers: Integer read FNumLayers;
      property NumHeads: Integer read FNumHeads;
      property FFNDim: Integer read FFFNDim;
      property VocabSize: Integer read FVocabSize;
      property MaxSeqLen: Integer read FMaxSeqLen;
      property Loaded: Boolean read FLoaded;
      property Tensors: TGGUFTensorArray read FTensors;
   end;

   { TTransformerModel - Core model with exposed internals for facade }
   TTransformerModel = class
   private
      FLoader: TGGUFLoader;
      FTokenizer: TTokenizer;
      FEmbedDim: Integer;
      FNumHeads: Integer;
      FHeadDim: Integer;
      FNumLayers: Integer;
      FFFNDim: Integer;
      FVocabSize: Integer;
      
      FLastHiddenStates: TDouble2DArray;
      FLastAttentionWeights: TDouble2DArray;
      FLastAttentionLogits: TDouble2DArray;
      FLastQVectors: TDouble2DArray;
      FLastKVectors: TDouble2DArray;
      FLastVVectors: TDouble2DArray;
      FLastLayerNormOutputs: TDouble2DArray;
      FLastFFNOutputs: TDouble2DArray;
      FLastLogits: TDoubleArray;
      FLastResidualInputs: TDouble2DArray;
      FLastResidualOutputs: TDouble2DArray;
      FLastSeqLen: Integer;

      function GELU(X: Double): Double;
      function Softmax(const Input: TDoubleArray): TDoubleArray;
      function LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray; Dim: Integer): TDoubleArray;
      function EmbedTokens(const TokenIDs: TIntArray): TDoubleArray;
      function AttentionBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function FFNBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function TransformerBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function ComputeLogits(const Input: TDoubleArray; SeqLen: Integer): TDoubleArray;

   public
      constructor Create;
      destructor Destroy; override;
      function LoadModel(const GGUFPath: string): Boolean;
      function LoadTokenizer(const TokenizerPath: string): Boolean;
      function Forward(const TokenIDs: TIntArray): TDoubleArray;
      function Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
      function IsModelLoaded: Boolean;
      function IsTokenizerLoaded: Boolean;
      
      property Loader: TGGUFLoader read FLoader;
      property Tokenizer: TTokenizer read FTokenizer;
      property EmbedDim: Integer read FEmbedDim;
      property NumHeads: Integer read FNumHeads;
      property HeadDim: Integer read FHeadDim;
      property NumLayers: Integer read FNumLayers;
      property FFNDim: Integer read FFFNDim;
      property VocabSize: Integer read FVocabSize;
      property LastSeqLen: Integer read FLastSeqLen;
      property LastHiddenStates: TDouble2DArray read FLastHiddenStates write FLastHiddenStates;
      property LastAttentionWeights: TDouble2DArray read FLastAttentionWeights;
      property LastAttentionLogits: TDouble2DArray read FLastAttentionLogits;
      property LastQVectors: TDouble2DArray read FLastQVectors;
      property LastKVectors: TDouble2DArray read FLastKVectors;
      property LastVVectors: TDouble2DArray read FLastVVectors;
      property LastLayerNormOutputs: TDouble2DArray read FLastLayerNormOutputs write FLastLayerNormOutputs;
      property LastFFNOutputs: TDouble2DArray read FLastFFNOutputs write FLastFFNOutputs;
      property LastLogits: TDoubleArray read FLastLogits;
      property LastResidualInputs: TDouble2DArray read FLastResidualInputs;
      property LastResidualOutputs: TDouble2DArray read FLastResidualOutputs;
   end;

   { TTransformerFacade - Main inspection/manipulation facade }
   TTransformerFacade = class
   private
      FModel: TTransformerModel;
      FOwnsModel: Boolean;
      
      FKVCacheK: TDouble3DArray;
      FKVCacheV: TDouble3DArray;
      FActivationTrace: TDouble2DArray;
      
   public
      constructor Create; overload;
      constructor Create(AModel: TTransformerModel; AOwnsModel: Boolean = False); overload;
      destructor Destroy; override;
      
      function LoadModel(const GGUFPath: string): Boolean;
      function LoadTokenizer(const TokenizerPath: string): Boolean;
      function RunForward(const TokenIDs: TIntArray): TDoubleArray;
      function Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
      
      { === 7. Structural and Architectural Introspection === }
      function GetNumLayers: Integer;
      function GetNumHeads(LayerIdx: Integer = 0): Integer;
      function GetHiddenSize(LayerIdx: Integer = 0): Integer;
      function GetHeadDim: Integer;
      function GetFFNDim: Integer;
      function GetVocabSize: Integer;
      function GetMaxSeqLen: Integer;
      function GetLastSeqLen: Integer;
      function IsModelLoaded: Boolean;
      function IsTokenizerLoaded: Boolean;
      
      { === 2. Input/Output Embedding and Positional Encoding === }
      function GetTokenEmbedding(TokenId: Integer): TDoubleArray;
      procedure SetTokenEmbedding(TokenId: Integer; const Vector: TDoubleArray);
      function GetPositionalEncoding(Position: Integer): TDoubleArray;
      
      { === 1. Attention Mechanism Inspection === }
      function GetAttentionWeights(LayerIdx, HeadIdx, FromPos, ToPos: Integer): Double;
      function GetAttentionLogits(LayerIdx, HeadIdx, FromPos, ToPos: Integer): Double;
      function GetAllAttentionWeights(LayerIdx: Integer): TDouble3DArray;
      
      { === 3. Per-Layer and Per-Head Activations === }
      function GetHiddenState(LayerIdx, Position: Integer): TDoubleArray;
      procedure SetHiddenState(LayerIdx, Position: Integer; const Vector: TDoubleArray);
      function GetQKV(LayerIdx, HeadIdx: Integer; QKVType: TQKVType; Position: Integer): TDoubleArray;
      function GetLayerNormOutput(LayerIdx, Position: Integer): TDoubleArray;
      procedure SetLayerNormOutput(LayerIdx, Position: Integer; const Vector: TDoubleArray);
      function GetFFNOutput(LayerIdx, Position: Integer): TDoubleArray;
      procedure SetFFNOutput(LayerIdx, Position: Integer; const Vector: TDoubleArray);
      
      { === 4. Output/Logits Inspection === }
      function GetLogits(Position: Integer): TDoubleArray;
      function GetSoftmaxOutput(Position: Integer): TDoubleArray;
      
      { === 5. Parameter and Weight Introspection === }
      function GetWeight(LayerIdx: Integer; ParamType: TParamType): TSingleArray;
      procedure SetWeight(LayerIdx: Integer; ParamType: TParamType; const Data: TSingleArray);
      function GetWeightShape(LayerIdx: Integer; ParamType: TParamType): TInt64Array;
      
      { === 6. Sequence and Memory State Access === }
      function GetKeyValueCache(LayerIdx, HeadIdx: Integer; IsKey: Boolean; Position: Integer): TDoubleArray;
      procedure ResetKeyValueCache;
      function GetActivationTrace(TokenIdx, LayerIdx: Integer): TDoubleArray;
      
      { === 8. Batch Norm / Layer Norm Stats === }
      procedure GetLayerNormStats(LayerIdx: Integer; out Mean, StdDev: Double);
      
      { === 9. Gradient and Backprop Access === }
      function GetParameterGradient(LayerIdx: Integer; ParamType: TParamType): TDoubleArray;
      
      { === 10. Structural Mutation / Dynamic Changes === }
      function AddLayer(Position: Integer): Boolean;
      function RemoveLayer(LayerIdx: Integer): Boolean;
      function AddHead(LayerIdx: Integer): Boolean;
      function RemoveHead(LayerIdx, HeadIdx: Integer): Boolean;
      
      { === 11. Explainability and Attribution Tools === }
      function GetSaliencyMap(TokenIdx, LayerIdx: Integer): TDoubleArray;
      function GetIntegratedGradients(const InputTokens: TIntArray; Steps: Integer = 50): TDouble2DArray;
      
      { === 12. Residual Connection Access === }
      function GetResidualInput(LayerIdx, Position: Integer): TDoubleArray;
      function GetResidualOutput(LayerIdx, Position: Integer): TDoubleArray;
      
      { === 13. Visualization/Diagnostics === }
      function GetActivationHistogram(LayerIdx, HeadIdx: Integer; NumBins: Integer = 50): TDoubleArray;
      function GetAttentionEntropy(LayerIdx, HeadIdx: Integer): Double;
      
      { === 5b. Optimizer State Access === }
      function GetOptimizerState(LayerIdx: Integer; ParamType: TParamType; StateType: Integer): TDoubleArray;
      procedure SetOptimizerState(LayerIdx: Integer; ParamType: TParamType; StateType: Integer; const Value: TDoubleArray);
      
      property Model: TTransformerModel read FModel;
   end;

implementation

uses fpjson, jsonparser;

{ ==================== TTokenizer Implementation ==================== }

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
   AddedTokensArr: TJSONArray;
   FileStream: TFileStream;
   Parser: TJSONParser;
   I: Integer;
   TokenStr: string;
   TokenID: Integer;
begin
   Result := False;
   FLoaded := False;

   if not FileExists(Filename) then Exit;

   try
      FileStream := TFileStream.Create(Filename, fmOpenRead or fmShareDenyWrite);
      try
         Parser := TJSONParser.Create(FileStream);
         try
            JSONData := Parser.Parse;
            if not Assigned(JSONData) then Exit;

            JSONObj := TJSONObject(JSONData);

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
            end;

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

         finally
            Parser.Free;
         end;
      finally
         FileStream.Free;
      end;
   except
      on E: Exception do;
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

   Tokens := TStringList.Create;
   try
      CurrentWord := '';
      for I := 1 to Length(Text) do
      begin
         Ch := Text[I];
         if Ch = ' ' then
         begin
            if CurrentWord <> '' then
               Tokens.Add(CurrentWord);
            CurrentWord := 'Ġ';
         end
         else
            CurrentWord := CurrentWord + Ch;
      end;
      if CurrentWord <> '' then
         Tokens.Add(CurrentWord);

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
      Token := StringReplace(Token, 'Ġ', ' ', [rfReplaceAll]);
      Token := StringReplace(Token, 'Ċ', #10, [rfReplaceAll]);
      Result := Result + Token;
   end;
end;

{ ==================== TGGUFLoader Implementation ==================== }

constructor TGGUFLoader.Create;
begin
   inherited;
   FStream := nil;
   FTensorMap := TStringList.Create;
   FTensorMap.CaseSensitive := True;
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
   if Assigned(FStream) then
      FStream.Free;
   FTensorMap.Free;
   inherited;
end;

function TGGUFLoader.ReadUInt32: UInt32;
begin
   FStream.Read(Result, 4);
end;

function TGGUFLoader.ReadUInt64: UInt64;
begin
   FStream.Read(Result, 8);
end;

function TGGUFLoader.ReadInt32: Int32;
begin
   FStream.Read(Result, 4);
end;

function TGGUFLoader.ReadFloat32: Single;
begin
   FStream.Read(Result, 4);
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
   if Len > 0 then
      FStream.Read(Bytes[0], Len);
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
         while M < 1 do begin M := M * 2; E := E - 1; end;
         if Sign = 1 then Result := -M * Power(2, E) else Result := M * Power(2, E);
      end;
   end
   else if Exponent = 31 then
   begin
      if Mantissa <> 0 then Result := NaN
      else if Sign = 1 then Result := NegInfinity
      else Result := Infinity;
   end
   else
   begin
      if Sign = 1 then Result := -(1 + Mantissa / 1024) * Power(2, Exponent - 15)
      else Result := (1 + Mantissa / 1024) * Power(2, Exponent - 15);
   end;
end;

function TGGUFLoader.BFloat16ToFloat32(BF: Word): Single;
var
   F32Bits: UInt32;
begin
   F32Bits := UInt32(BF) shl 16;
   Move(F32Bits, Result, 4);
end;

procedure TGGUFLoader.SkipMetadataValue(ValueType: Integer);
var
   I: Integer;
   ArrType: UInt32;
   ArrCount: UInt64;
   StrLen: UInt64;
begin
   case ValueType of
      0, 1: FStream.Seek(1, soCurrent);
      2, 3: FStream.Seek(2, soCurrent);
      4, 5, 6: FStream.Seek(4, soCurrent);
      7: FStream.Seek(1, soCurrent);
      8: begin
         StrLen := ReadUInt64;
         FStream.Seek(StrLen, soCurrent);
      end;
      9: begin
         ArrType := ReadUInt32;
         ArrCount := ReadUInt64;
         for I := 0 to Min(Int64(ArrCount) - 1, 999999) do
            SkipMetadataValue(ArrType);
      end;
      10, 11, 12: FStream.Seek(8, soCurrent);
   end;
end;

procedure TGGUFLoader.ParseHeader;
const
   GGUF_MAGIC = 'GGUF';
var
   Magic: array[0..3] of Char;
   Version: UInt32;
   TensorCount, MetadataCount: UInt64;
   I: Integer;
   Key: string;
   ValueType: UInt32;
   IntVal: Int64;
begin
   FStream.Read(Magic, 4);
   if Magic <> GGUF_MAGIC then
      raise Exception.Create('Invalid GGUF magic: ' + Magic);

   Version := ReadUInt32;
   TensorCount := ReadUInt64;
   MetadataCount := ReadUInt64;

   for I := 0 to MetadataCount - 1 do
   begin
      Key := ReadString;
      ValueType := ReadUInt32;

      if (Key = 'gpt2.embedding_length') and (ValueType in [4, 5, 10]) then
      begin
         if ValueType = 10 then IntVal := ReadUInt64
         else IntVal := ReadUInt32;
         FEmbedDim := IntVal;
      end
      else if (Key = 'gpt2.block_count') and (ValueType in [4, 5, 10]) then
      begin
         if ValueType = 10 then IntVal := ReadUInt64
         else IntVal := ReadUInt32;
         FNumLayers := IntVal;
      end
      else if (Key = 'gpt2.attention.head_count') and (ValueType in [4, 5, 10]) then
      begin
         if ValueType = 10 then IntVal := ReadUInt64
         else IntVal := ReadUInt32;
         FNumHeads := IntVal;
      end
      else if (Key = 'gpt2.feed_forward_length') and (ValueType in [4, 5, 10]) then
      begin
         if ValueType = 10 then IntVal := ReadUInt64
         else IntVal := ReadUInt32;
         FFFNDim := IntVal;
      end
      else if (Key = 'gpt2.context_length') and (ValueType in [4, 5, 10]) then
      begin
         if ValueType = 10 then IntVal := ReadUInt64
         else IntVal := ReadUInt32;
         FMaxSeqLen := IntVal;
      end
      else
         SkipMetadataValue(ValueType);
   end;

   SetLength(FTensors, TensorCount);
   FTensorMap.Clear;

   for I := 0 to TensorCount - 1 do
   begin
      FTensors[I].Name := ReadString;
      FTensors[I].NumDims := ReadUInt32;
      SetLength(FTensors[I].Shape, FTensors[I].NumDims);
      if FTensors[I].NumDims > 0 then
         FStream.Read(FTensors[I].Shape[0], FTensors[I].NumDims * 8);
      FTensors[I].DType := ReadUInt32;
      FTensors[I].DataOffset := ReadUInt64;
      FTensors[I].DataLoaded := False;
      FTensorMap.AddObject(FTensors[I].Name, TObject(PtrInt(I)));
   end;

   FTensorDataStart := FStream.Position;
   while (FTensorDataStart mod 32) <> 0 do
      Inc(FTensorDataStart);
end;

function TGGUFLoader.LoadTensorByIndex(Idx: Integer): Boolean;
var
   TotalElements: Int64;
   D: Integer;
   ByteData: array of Byte;
   WordData: array of Word;
   I: Integer;
begin
   Result := False;
   if (Idx < 0) or (Idx >= Length(FTensors)) then Exit;
   if FTensors[Idx].DataLoaded then
   begin
      Result := True;
      Exit;
   end;

   TotalElements := 1;
   for D := 0 to FTensors[Idx].NumDims - 1 do
      TotalElements := TotalElements * FTensors[Idx].Shape[D];

   SetLength(FTensors[Idx].Data, TotalElements);

   FStream.Seek(FTensorDataStart + FTensors[Idx].DataOffset, soBeginning);

   case FTensors[Idx].DType of
      0: begin
         SetLength(FTensors[Idx].Data, TotalElements);
         FStream.Read(FTensors[Idx].Data[0], TotalElements * 4);
      end;
      1: begin
         SetLength(WordData, TotalElements);
         FStream.Read(WordData[0], TotalElements * 2);
         for I := 0 to TotalElements - 1 do
            FTensors[Idx].Data[I] := Float16ToFloat32(WordData[I]);
      end;
      else
         Exit;
   end;

   FTensors[Idx].DataLoaded := True;
   Result := True;
end;

function TGGUFLoader.LoadFromFile(const Filename: string): Boolean;
begin
   Result := False;
   FLoaded := False;
   FFilename := Filename;

   if not FileExists(Filename) then Exit;

   try
      FStream := TFileStream.Create(Filename, fmOpenRead or fmShareDenyWrite);
      ParseHeader;
      FLoaded := True;
      Result := True;
   except
      on E: Exception do;
   end;
end;

function TGGUFLoader.GetTensor(const Names: array of string): TSingleArray;
var
   I, Idx: Integer;
begin
   SetLength(Result, 0);
   for I := 0 to High(Names) do
   begin
      Idx := FTensorMap.IndexOf(Names[I]);
      if Idx >= 0 then
      begin
         Idx := PtrInt(FTensorMap.Objects[Idx]);
         if LoadTensorByIndex(Idx) then
         begin
            Result := FTensors[Idx].Data;
            Exit;
         end;
      end;
   end;
end;

function TGGUFLoader.GetTensorShape(const Names: array of string): TInt64Array;
var
   I, Idx: Integer;
begin
   SetLength(Result, 0);
   for I := 0 to High(Names) do
   begin
      Idx := FTensorMap.IndexOf(Names[I]);
      if Idx >= 0 then
      begin
         Idx := PtrInt(FTensorMap.Objects[Idx]);
         Result := FTensors[Idx].Shape;
         Exit;
      end;
   end;
end;

function TGGUFLoader.HasTensor(const Name: string): Boolean;
begin
   Result := FTensorMap.IndexOf(Name) >= 0;
end;

procedure TGGUFLoader.PrintAllTensorNames;
var
   I: Integer;
begin
   for I := 0 to High(FTensors) do
      WriteLn(FTensors[I].Name);
end;

{ ==================== TTransformerModel Implementation ==================== }

constructor TTransformerModel.Create;
begin
   inherited;
   FLoader := TGGUFLoader.Create;
   FTokenizer := TTokenizer.Create;
   FLastSeqLen := 0;
end;

destructor TTransformerModel.Destroy;
begin
   FLoader.Free;
   FTokenizer.Free;
   inherited;
end;

function TTransformerModel.LoadModel(const GGUFPath: string): Boolean;
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
      
      SetLength(FLastHiddenStates, FNumLayers + 1);
      SetLength(FLastAttentionWeights, FNumLayers);
      SetLength(FLastAttentionLogits, FNumLayers);
      SetLength(FLastQVectors, FNumLayers);
      SetLength(FLastKVectors, FNumLayers);
      SetLength(FLastVVectors, FNumLayers);
      SetLength(FLastLayerNormOutputs, FNumLayers);
      SetLength(FLastFFNOutputs, FNumLayers);
      SetLength(FLastResidualInputs, FNumLayers);
      SetLength(FLastResidualOutputs, FNumLayers);
   end;
end;

function TTransformerModel.LoadTokenizer(const TokenizerPath: string): Boolean;
begin
   Result := FTokenizer.LoadFromFile(TokenizerPath);
end;

function TTransformerModel.IsModelLoaded: Boolean;
begin
   Result := FLoader.Loaded;
end;

function TTransformerModel.IsTokenizerLoaded: Boolean;
begin
   Result := FTokenizer.Loaded;
end;

function TTransformerModel.GELU(X: Double): Double;
begin
   Result := 0.5 * X * (1.0 + Tanh(Sqrt(2.0 / Pi) * (X + 0.044715 * X * X * X)));
end;

function TTransformerModel.Softmax(const Input: TDoubleArray): TDoubleArray;
var
   MaxVal, Sum: Double;
   I: Integer;
begin
   SetLength(Result, Length(Input));
   MaxVal := Input[0];
   for I := 1 to High(Input) do
      if Input[I] > MaxVal then MaxVal := Input[I];

   Sum := 0;
   for I := 0 to High(Input) do
   begin
      Result[I] := Exp(Input[I] - MaxVal);
      Sum := Sum + Result[I];
   end;

   if Sum > 0 then
      for I := 0 to High(Result) do
         Result[I] := Result[I] / Sum;
end;

function TTransformerModel.LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray; Dim: Integer): TDoubleArray;
var
   Mean, Variance, StdDev: Double;
   I: Integer;
   Eps: Double;
begin
   Eps := 1e-5;
   SetLength(Result, Dim);

   Mean := 0;
   for I := 0 to Dim - 1 do
      Mean := Mean + Input[I];
   Mean := Mean / Dim;

   Variance := 0;
   for I := 0 to Dim - 1 do
      Variance := Variance + Sqr(Input[I] - Mean);
   Variance := Variance / Dim;

   StdDev := Sqrt(Variance + Eps);

   for I := 0 to Dim - 1 do
   begin
      Result[I] := (Input[I] - Mean) / StdDev;
      if Length(Gamma) > I then
         Result[I] := Result[I] * Gamma[I];
      if Length(Beta) > I then
         Result[I] := Result[I] + Beta[I];
   end;
end;

function TTransformerModel.EmbedTokens(const TokenIDs: TIntArray): TDoubleArray;
var
   TokenEmb, PosEmb: TSingleArray;
   SeqLen, I, J, Idx: Integer;
begin
   SeqLen := Length(TokenIDs);
   SetLength(Result, SeqLen * FEmbedDim);

   TokenEmb := FLoader.GetTensor(['token_embd.weight', 'wte.weight']);
   PosEmb := FLoader.GetTensor(['position_embd.weight', 'wpe.weight']);

   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to FEmbedDim - 1 do
      begin
         Idx := I * FEmbedDim + J;
         Result[Idx] := TokenEmb[TokenIDs[I] * FEmbedDim + J];
         if Length(PosEmb) > I * FEmbedDim + J then
            Result[Idx] := Result[Idx] + PosEmb[I * FEmbedDim + J];
      end;
   end;
end;

function TTransformerModel.AttentionBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
var
   LN1G, LN1B, QKVWeight, QKVBias, ProjWeight, ProjBias: TSingleArray;
   NormInput: TDoubleArray;
   Q, K, V, AttnOut: TDoubleArray;
   Scores, AttnWeights: TDoubleArray;
   Scale, Sum: Double;
   Pos, I, J, H, SrcPos, HeadStart, QIdx, KIdx, VIdx: Integer;
begin
   SetLength(Result, SeqLen * FEmbedDim);
   SetLength(Q, SeqLen * FEmbedDim);
   SetLength(K, SeqLen * FEmbedDim);
   SetLength(V, SeqLen * FEmbedDim);
   SetLength(AttnOut, SeqLen * FEmbedDim);

   LN1G := FLoader.GetTensor([Format('blk.%d.attn_norm.weight', [LayerIdx])]);
   LN1B := FLoader.GetTensor([Format('blk.%d.attn_norm.bias', [LayerIdx])]);
   QKVWeight := FLoader.GetTensor([Format('blk.%d.attn_qkv.weight', [LayerIdx])]);
   QKVBias := FLoader.GetTensor([Format('blk.%d.attn_qkv.bias', [LayerIdx])]);
   ProjWeight := FLoader.GetTensor([Format('blk.%d.attn_output.weight', [LayerIdx])]);
   ProjBias := FLoader.GetTensor([Format('blk.%d.attn_output.bias', [LayerIdx])]);

   Scale := Sqrt(FHeadDim);
   SetLength(NormInput, FEmbedDim);

   for Pos := 0 to SeqLen - 1 do
   begin
      for I := 0 to FEmbedDim - 1 do
         NormInput[I] := Input[Pos * FEmbedDim + I];

      if Length(LN1G) >= FEmbedDim then
         NormInput := LayerNorm(NormInput, LN1G, LN1B, FEmbedDim);

      for I := 0 to FEmbedDim - 1 do
      begin
         if Length(QKVBias) > I then Sum := QKVBias[I] else Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[I * FEmbedDim + J];
         Q[Pos * FEmbedDim + I] := Sum;

         if Length(QKVBias) > FEmbedDim + I then Sum := QKVBias[FEmbedDim + I] else Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[(FEmbedDim + I) * FEmbedDim + J];
         K[Pos * FEmbedDim + I] := Sum;

         if Length(QKVBias) > 2 * FEmbedDim + I then Sum := QKVBias[2 * FEmbedDim + I] else Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[(2 * FEmbedDim + I) * FEmbedDim + J];
         V[Pos * FEmbedDim + I] := Sum;
      end;
   end;

   FLastQVectors[LayerIdx] := Q;
   FLastKVectors[LayerIdx] := K;
   FLastVVectors[LayerIdx] := V;

   SetLength(Scores, SeqLen);
   SetLength(AttnWeights, SeqLen);
   SetLength(FLastAttentionWeights[LayerIdx], FNumHeads * SeqLen * SeqLen);
   SetLength(FLastAttentionLogits[LayerIdx], FNumHeads * SeqLen * SeqLen);

   for H := 0 to FNumHeads - 1 do
   begin
      HeadStart := H * FHeadDim;

      for Pos := 0 to SeqLen - 1 do
      begin
         for SrcPos := 0 to SeqLen - 1 do
         begin
            if SrcPos > Pos then
               Scores[SrcPos] := -1e9
            else
            begin
               Sum := 0;
               for I := 0 to FHeadDim - 1 do
               begin
                  QIdx := Pos * FEmbedDim + HeadStart + I;
                  KIdx := SrcPos * FEmbedDim + HeadStart + I;
                  Sum := Sum + Q[QIdx] * K[KIdx];
               end;
               Scores[SrcPos] := Sum / Scale;
            end;
            FLastAttentionLogits[LayerIdx][H * SeqLen * SeqLen + Pos * SeqLen + SrcPos] := Scores[SrcPos];
         end;

         AttnWeights := Softmax(Scores);
         
         for SrcPos := 0 to SeqLen - 1 do
            FLastAttentionWeights[LayerIdx][H * SeqLen * SeqLen + Pos * SeqLen + SrcPos] := AttnWeights[SrcPos];

         for I := 0 to FHeadDim - 1 do
         begin
            Sum := 0;
            for SrcPos := 0 to SeqLen - 1 do
            begin
               VIdx := SrcPos * FEmbedDim + HeadStart + I;
               Sum := Sum + AttnWeights[SrcPos] * V[VIdx];
            end;
            AttnOut[Pos * FEmbedDim + HeadStart + I] := Sum;
         end;
      end;
   end;

   for Pos := 0 to SeqLen - 1 do
   begin
      for I := 0 to FEmbedDim - 1 do
      begin
         if Length(ProjBias) > I then Sum := ProjBias[I] else Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + AttnOut[Pos * FEmbedDim + J] * ProjWeight[I * FEmbedDim + J];
         Result[Pos * FEmbedDim + I] := Input[Pos * FEmbedDim + I] + Sum;
      end;
   end;
end;

function TTransformerModel.FFNBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
var
   LN2G, LN2B, UpWeight, UpBias, DownWeight, DownBias: TSingleArray;
   NormInput, Hidden: TDoubleArray;
   Pos, I, J: Integer;
   Sum: Double;
begin
   SetLength(Result, SeqLen * FEmbedDim);

   LN2G := FLoader.GetTensor([Format('blk.%d.ffn_norm.weight', [LayerIdx])]);
   LN2B := FLoader.GetTensor([Format('blk.%d.ffn_norm.bias', [LayerIdx])]);
   UpWeight := FLoader.GetTensor([Format('blk.%d.ffn_up.weight', [LayerIdx])]);
   UpBias := FLoader.GetTensor([Format('blk.%d.ffn_up.bias', [LayerIdx])]);
   DownWeight := FLoader.GetTensor([Format('blk.%d.ffn_down.weight', [LayerIdx])]);
   DownBias := FLoader.GetTensor([Format('blk.%d.ffn_down.bias', [LayerIdx])]);

   if (Length(UpWeight) = 0) or (Length(DownWeight) = 0) then
   begin
      Move(Input[0], Result[0], SeqLen * FEmbedDim * SizeOf(Double));
      Exit;
   end;

   SetLength(NormInput, FEmbedDim);
   SetLength(Hidden, FFFNDim);
   SetLength(FLastLayerNormOutputs[LayerIdx], SeqLen * FEmbedDim);

   for Pos := 0 to SeqLen - 1 do
   begin
      for I := 0 to FEmbedDim - 1 do
         NormInput[I] := Input[Pos * FEmbedDim + I];

      if Length(LN2G) >= FEmbedDim then
         NormInput := LayerNorm(NormInput, LN2G, LN2B, FEmbedDim);

      for I := 0 to FEmbedDim - 1 do
         FLastLayerNormOutputs[LayerIdx][Pos * FEmbedDim + I] := NormInput[I];

      for I := 0 to FFFNDim - 1 do
      begin
         if Length(UpBias) > I then Sum := UpBias[I] else Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * UpWeight[I * FEmbedDim + J];
         Hidden[I] := GELU(Sum);
      end;

      for I := 0 to FEmbedDim - 1 do
      begin
         if Length(DownBias) > I then Sum := DownBias[I] else Sum := 0;
         for J := 0 to FFFNDim - 1 do
            Sum := Sum + Hidden[J] * DownWeight[I * FFFNDim + J];
         Result[Pos * FEmbedDim + I] := Input[Pos * FEmbedDim + I] + Sum;
      end;
   end;

   FLastFFNOutputs[LayerIdx] := Copy(Result);
end;

function TTransformerModel.TransformerBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
var
   AfterAttn: TDoubleArray;
   I, NaNCount: Integer;
begin
   FLastResidualInputs[LayerIdx] := Copy(Input);
   
   AfterAttn := AttentionBlock(Input, SeqLen, LayerIdx);

   NaNCount := 0;
   for I := 0 to High(AfterAttn) do
      if IsNan(AfterAttn[I]) or IsInfinite(AfterAttn[I]) then
      begin
         Inc(NaNCount);
         AfterAttn[I] := 0;
      end;

   Result := FFNBlock(AfterAttn, SeqLen, LayerIdx);

   NaNCount := 0;
   for I := 0 to High(Result) do
      if IsNan(Result[I]) or IsInfinite(Result[I]) then
      begin
         Inc(NaNCount);
         Result[I] := 0;
      end;

   FLastResidualOutputs[LayerIdx] := Copy(Result);
end;

function TTransformerModel.ComputeLogits(const Input: TDoubleArray; SeqLen: Integer): TDoubleArray;
var
   FinalLNG, FinalLNB, TokenEmb: TSingleArray;
   LastPos, NormedPos: TDoubleArray;
   I, J: Integer;
   Sum, Val: Double;
begin
   SetLength(Result, FVocabSize);

   FinalLNG := FLoader.GetTensor(['output_norm.weight', 'ln_f.weight']);
   FinalLNB := FLoader.GetTensor(['output_norm.bias', 'ln_f.bias']);

   SetLength(LastPos, FEmbedDim);
   for I := 0 to FEmbedDim - 1 do
   begin
      Val := Input[(SeqLen - 1) * FEmbedDim + I];
      if IsNan(Val) or IsInfinite(Val) then Val := 0;
      LastPos[I] := Val;
   end;

   if Length(FinalLNG) >= FEmbedDim then
      NormedPos := LayerNorm(LastPos, FinalLNG, FinalLNB, FEmbedDim)
   else
      NormedPos := LastPos;

   TokenEmb := FLoader.GetTensor(['token_embd.weight', 'wte.weight']);

   for I := 0 to FVocabSize - 1 do
   begin
      Sum := 0;
      for J := 0 to FEmbedDim - 1 do
      begin
         Val := NormedPos[J] * TokenEmb[I * FEmbedDim + J];
         if not (IsNan(Val) or IsInfinite(Val)) then
            Sum := Sum + Val;
      end;
      Result[I] := Sum;
   end;

   FLastLogits := Copy(Result);
end;

function TTransformerModel.Forward(const TokenIDs: TIntArray): TDoubleArray;
var
   Hidden: TDoubleArray;
   SeqLen, L: Integer;
begin
   SeqLen := Length(TokenIDs);
   FLastSeqLen := SeqLen;

   Hidden := EmbedTokens(TokenIDs);
   FLastHiddenStates[0] := Copy(Hidden);
   
   if Length(Hidden) <> SeqLen * FEmbedDim then
   begin
      SetLength(Result, 0);
      Exit;
   end;

   for L := 0 to FNumLayers - 1 do
   begin
      Hidden := TransformerBlock(Hidden, SeqLen, L);
      FLastHiddenStates[L + 1] := Copy(Hidden);
   end;

   Result := ComputeLogits(Hidden, SeqLen);
end;

function TTransformerModel.Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
var
   TokenIDs: TIntArray;
   Logits, Probs: TDoubleArray;
   I, J, BestID, SelectedID: Integer;
   BestLogit, R, CumulativeProb: Double;
begin
   Result := '';
   if not FLoader.Loaded or not FTokenizer.Loaded then Exit;

   TokenIDs := FTokenizer.Encode(Prompt);
   if Length(TokenIDs) = 0 then Exit;

   for I := 0 to MaxTokens - 1 do
   begin
      Logits := Forward(TokenIDs);
      if Length(Logits) = 0 then Break;

      BestID := 0;
      BestLogit := Logits[0];
      for J := 1 to High(Logits) do
         if Logits[J] > BestLogit then
         begin
            BestLogit := Logits[J];
            BestID := J;
         end;

      if Temperature <= 0.01 then
         SelectedID := BestID
      else
      begin
         for J := 0 to High(Logits) do
            Logits[J] := Logits[J] / Temperature;
         Probs := Softmax(Logits);
         R := Random;
         CumulativeProb := 0.0;
         SelectedID := 0;
         for J := 0 to High(Probs) do
         begin
            CumulativeProb := CumulativeProb + Probs[J];
            if R <= CumulativeProb then
            begin
               SelectedID := J;
               Break;
            end;
         end;
      end;

      SetLength(TokenIDs, Length(TokenIDs) + 1);
      TokenIDs[High(TokenIDs)] := SelectedID;

      if SelectedID = 50256 then Break;
   end;

   Result := FTokenizer.Decode(TokenIDs);
end;

{ ==================== TTransformerFacade Implementation ==================== }

constructor TTransformerFacade.Create;
begin
   inherited;
   FModel := TTransformerModel.Create;
   FOwnsModel := True;
end;

constructor TTransformerFacade.Create(AModel: TTransformerModel; AOwnsModel: Boolean);
begin
   inherited Create;
   FModel := AModel;
   FOwnsModel := AOwnsModel;
end;

destructor TTransformerFacade.Destroy;
begin
   if FOwnsModel and Assigned(FModel) then
      FModel.Free;
   inherited;
end;

function TTransformerFacade.LoadModel(const GGUFPath: string): Boolean;
begin
   Result := FModel.LoadModel(GGUFPath);
end;

function TTransformerFacade.LoadTokenizer(const TokenizerPath: string): Boolean;
begin
   Result := FModel.LoadTokenizer(TokenizerPath);
end;

function TTransformerFacade.RunForward(const TokenIDs: TIntArray): TDoubleArray;
begin
   Result := FModel.Forward(TokenIDs);
end;

function TTransformerFacade.Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double): string;
begin
   Result := FModel.Generate(Prompt, MaxTokens, Temperature);
end;

{ === 7. Structural and Architectural Introspection === }

function TTransformerFacade.GetNumLayers: Integer;
begin
   Result := FModel.NumLayers;
end;

function TTransformerFacade.GetNumHeads(LayerIdx: Integer): Integer;
begin
   Result := FModel.NumHeads;
end;

function TTransformerFacade.GetHiddenSize(LayerIdx: Integer): Integer;
begin
   Result := FModel.EmbedDim;
end;

function TTransformerFacade.GetHeadDim: Integer;
begin
   Result := FModel.HeadDim;
end;

function TTransformerFacade.GetFFNDim: Integer;
begin
   Result := FModel.FFNDim;
end;

function TTransformerFacade.GetVocabSize: Integer;
begin
   Result := FModel.VocabSize;
end;

function TTransformerFacade.GetMaxSeqLen: Integer;
begin
   Result := FModel.Loader.MaxSeqLen;
end;

function TTransformerFacade.GetLastSeqLen: Integer;
begin
   Result := FModel.LastSeqLen;
end;

function TTransformerFacade.IsModelLoaded: Boolean;
begin
   Result := FModel.IsModelLoaded;
end;

function TTransformerFacade.IsTokenizerLoaded: Boolean;
begin
   Result := FModel.IsTokenizerLoaded;
end;

{ === 2. Input/Output Embedding and Positional Encoding === }

function TTransformerFacade.GetTokenEmbedding(TokenId: Integer): TDoubleArray;
var
   TokenEmb: TSingleArray;
   I, EmbedDim: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   
   EmbedDim := FModel.EmbedDim;
   TokenEmb := FModel.Loader.GetTensor(['token_embd.weight', 'wte.weight']);
   
   if (TokenId < 0) or (TokenId >= FModel.VocabSize) then Exit;
   if Length(TokenEmb) < (TokenId + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := TokenEmb[TokenId * EmbedDim + I];
end;

procedure TTransformerFacade.SetTokenEmbedding(TokenId: Integer; const Vector: TDoubleArray);
var
   TensorIdx, I, EmbedDim: Integer;
begin
   if not IsModelLoaded then Exit;
   
   EmbedDim := FModel.EmbedDim;
   if Length(Vector) <> EmbedDim then Exit;
   if (TokenId < 0) or (TokenId >= FModel.VocabSize) then Exit;
   
   TensorIdx := FModel.Loader.FTensorMap.IndexOf('token_embd.weight');
   if TensorIdx < 0 then
      TensorIdx := FModel.Loader.FTensorMap.IndexOf('wte.weight');
   if TensorIdx < 0 then Exit;
   
   TensorIdx := PtrInt(FModel.Loader.FTensorMap.Objects[TensorIdx]);
   
   for I := 0 to EmbedDim - 1 do
      FModel.Loader.FTensors[TensorIdx].Data[TokenId * EmbedDim + I] := Vector[I];
end;

function TTransformerFacade.GetPositionalEncoding(Position: Integer): TDoubleArray;
var
   PosEmb: TSingleArray;
   I, EmbedDim: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   
   EmbedDim := FModel.EmbedDim;
   PosEmb := FModel.Loader.GetTensor(['position_embd.weight', 'wpe.weight']);
   
   if (Position < 0) or (Position >= FModel.Loader.MaxSeqLen) then Exit;
   if Length(PosEmb) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := PosEmb[Position * EmbedDim + I];
end;

{ === 1. Attention Mechanism Inspection === }

function TTransformerFacade.GetAttentionWeights(LayerIdx, HeadIdx, FromPos, ToPos: Integer): Double;
var
   SeqLen, Idx: Integer;
begin
   Result := 0;
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if SeqLen = 0 then Exit;
   if (FromPos < 0) or (FromPos >= SeqLen) then Exit;
   if (ToPos < 0) or (ToPos >= SeqLen) then Exit;
   
   if Length(FModel.LastAttentionWeights) <= LayerIdx then Exit;
   if Length(FModel.LastAttentionWeights[LayerIdx]) = 0 then Exit;
   
   Idx := HeadIdx * SeqLen * SeqLen + FromPos * SeqLen + ToPos;
   if Idx < Length(FModel.LastAttentionWeights[LayerIdx]) then
      Result := FModel.LastAttentionWeights[LayerIdx][Idx];
end;

function TTransformerFacade.GetAttentionLogits(LayerIdx, HeadIdx, FromPos, ToPos: Integer): Double;
var
   SeqLen, Idx: Integer;
begin
   Result := 0;
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if SeqLen = 0 then Exit;
   if (FromPos < 0) or (FromPos >= SeqLen) then Exit;
   if (ToPos < 0) or (ToPos >= SeqLen) then Exit;
   
   if Length(FModel.LastAttentionLogits) <= LayerIdx then Exit;
   if Length(FModel.LastAttentionLogits[LayerIdx]) = 0 then Exit;
   
   Idx := HeadIdx * SeqLen * SeqLen + FromPos * SeqLen + ToPos;
   if Idx < Length(FModel.LastAttentionLogits[LayerIdx]) then
      Result := FModel.LastAttentionLogits[LayerIdx][Idx];
end;

function TTransformerFacade.GetAllAttentionWeights(LayerIdx: Integer): TDouble3DArray;
var
   SeqLen, H, I, J, Idx: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if SeqLen = 0 then Exit;
   if Length(FModel.LastAttentionWeights) <= LayerIdx then Exit;
   if Length(FModel.LastAttentionWeights[LayerIdx]) = 0 then Exit;
   
   SetLength(Result, FModel.NumHeads);
   for H := 0 to FModel.NumHeads - 1 do
   begin
      SetLength(Result[H], SeqLen);
      for I := 0 to SeqLen - 1 do
      begin
         SetLength(Result[H][I], SeqLen);
         for J := 0 to SeqLen - 1 do
         begin
            Idx := H * SeqLen * SeqLen + I * SeqLen + J;
            Result[H][I][J] := FModel.LastAttentionWeights[LayerIdx][Idx];
         end;
      end;
   end;
end;

{ === 3. Per-Layer and Per-Head Activations === }

function TTransformerFacade.GetHiddenState(LayerIdx, Position: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx > FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastHiddenStates) <= LayerIdx then Exit;
   if Length(FModel.LastHiddenStates[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastHiddenStates[LayerIdx][Position * EmbedDim + I];
end;

procedure TTransformerFacade.SetHiddenState(LayerIdx, Position: Integer; const Vector: TDoubleArray);
var
   I, EmbedDim, SeqLen: Integer;
begin
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx > FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if Length(Vector) <> EmbedDim then Exit;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastHiddenStates) <= LayerIdx then Exit;
   if Length(FModel.LastHiddenStates[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   for I := 0 to EmbedDim - 1 do
      FModel.LastHiddenStates[LayerIdx][Position * EmbedDim + I] := Vector[I];
end;

function TTransformerFacade.GetQKV(LayerIdx, HeadIdx: Integer; QKVType: TQKVType; Position: Integer): TDoubleArray;
var
   I, HeadDim, EmbedDim, SeqLen, HeadStart: Integer;
   SourceArray: TDoubleArray;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   
   case QKVType of
      qkvQuery: SourceArray := FModel.LastQVectors[LayerIdx];
      qkvKey: SourceArray := FModel.LastKVectors[LayerIdx];
      qkvValue: SourceArray := FModel.LastVVectors[LayerIdx];
   end;
   
   if Length(SourceArray) = 0 then Exit;
   
   EmbedDim := FModel.EmbedDim;
   HeadDim := FModel.HeadDim;
   HeadStart := HeadIdx * HeadDim;
   
   SetLength(Result, HeadDim);
   for I := 0 to HeadDim - 1 do
      Result[I] := SourceArray[Position * EmbedDim + HeadStart + I];
end;

function TTransformerFacade.GetLayerNormOutput(LayerIdx, Position: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastLayerNormOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastLayerNormOutputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastLayerNormOutputs[LayerIdx][Position * EmbedDim + I];
end;

procedure TTransformerFacade.SetLayerNormOutput(LayerIdx, Position: Integer; const Vector: TDoubleArray);
var
   I, EmbedDim, SeqLen: Integer;
begin
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if Length(Vector) <> EmbedDim then Exit;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastLayerNormOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastLayerNormOutputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   for I := 0 to EmbedDim - 1 do
      FModel.LastLayerNormOutputs[LayerIdx][Position * EmbedDim + I] := Vector[I];
end;

function TTransformerFacade.GetFFNOutput(LayerIdx, Position: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastFFNOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastFFNOutputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastFFNOutputs[LayerIdx][Position * EmbedDim + I];
end;

procedure TTransformerFacade.SetFFNOutput(LayerIdx, Position: Integer; const Vector: TDoubleArray);
var
   I, EmbedDim, SeqLen: Integer;
begin
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if Length(Vector) <> EmbedDim then Exit;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastFFNOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastFFNOutputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   for I := 0 to EmbedDim - 1 do
      FModel.LastFFNOutputs[LayerIdx][Position * EmbedDim + I] := Vector[I];
end;

{ === 4. Output/Logits Inspection === }

function TTransformerFacade.GetLogits(Position: Integer): TDoubleArray;
begin
   if Position = -1 then
      Result := Copy(FModel.LastLogits)
   else
   begin
      SetLength(Result, 0);
   end;
end;

function TTransformerFacade.GetSoftmaxOutput(Position: Integer): TDoubleArray;
var
   Logits: TDoubleArray;
   MaxVal, Sum: Double;
   I: Integer;
begin
   Logits := GetLogits(Position);
   if Length(Logits) = 0 then
   begin
      SetLength(Result, 0);
      Exit;
   end;
   
   SetLength(Result, Length(Logits));
   
   MaxVal := Logits[0];
   for I := 1 to High(Logits) do
      if Logits[I] > MaxVal then MaxVal := Logits[I];
   
   Sum := 0;
   for I := 0 to High(Logits) do
   begin
      Result[I] := Exp(Logits[I] - MaxVal);
      Sum := Sum + Result[I];
   end;
   
   if Sum > 0 then
      for I := 0 to High(Result) do
         Result[I] := Result[I] / Sum;
end;

{ === 5. Parameter and Weight Introspection === }

function GetTensorNameForParam(LayerIdx: Integer; ParamType: TParamType): string;
begin
   case ParamType of
      ptQProj, ptKProj, ptVProj:
         Result := Format('blk.%d.attn_qkv.weight', [LayerIdx]);
      ptOutProj:
         Result := Format('blk.%d.attn_output.weight', [LayerIdx]);
      ptFFN1:
         Result := Format('blk.%d.ffn_up.weight', [LayerIdx]);
      ptFFN2:
         Result := Format('blk.%d.ffn_down.weight', [LayerIdx]);
      ptLayerNorm1Weight:
         Result := Format('blk.%d.attn_norm.weight', [LayerIdx]);
      ptLayerNorm1Bias:
         Result := Format('blk.%d.attn_norm.bias', [LayerIdx]);
      ptLayerNorm2Weight:
         Result := Format('blk.%d.ffn_norm.weight', [LayerIdx]);
      ptLayerNorm2Bias:
         Result := Format('blk.%d.ffn_norm.bias', [LayerIdx]);
      ptTokenEmbed:
         Result := 'token_embd.weight';
      ptPosEmbed:
         Result := 'position_embd.weight';
      ptFinalNormWeight:
         Result := 'output_norm.weight';
      ptFinalNormBias:
         Result := 'output_norm.bias';
   else
      Result := '';
   end;
end;

function TTransformerFacade.GetWeight(LayerIdx: Integer; ParamType: TParamType): TSingleArray;
var
   TensorName: string;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   
   TensorName := GetTensorNameForParam(LayerIdx, ParamType);
   if TensorName = '' then Exit;
   
   Result := FModel.Loader.GetTensor([TensorName]);
end;

procedure TTransformerFacade.SetWeight(LayerIdx: Integer; ParamType: TParamType; const Data: TSingleArray);
var
   TensorName: string;
   TensorIdx, I: Integer;
begin
   if not IsModelLoaded then Exit;
   
   TensorName := GetTensorNameForParam(LayerIdx, ParamType);
   if TensorName = '' then Exit;
   
   TensorIdx := FModel.Loader.FTensorMap.IndexOf(TensorName);
   if TensorIdx < 0 then Exit;
   
   TensorIdx := PtrInt(FModel.Loader.FTensorMap.Objects[TensorIdx]);
   
   if Length(Data) <> Length(FModel.Loader.FTensors[TensorIdx].Data) then Exit;
   
   for I := 0 to High(Data) do
      FModel.Loader.FTensors[TensorIdx].Data[I] := Data[I];
end;

function TTransformerFacade.GetWeightShape(LayerIdx: Integer; ParamType: TParamType): TInt64Array;
var
   TensorName: string;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   
   TensorName := GetTensorNameForParam(LayerIdx, ParamType);
   if TensorName = '' then Exit;
   
   Result := FModel.Loader.GetTensorShape([TensorName]);
end;

{ === 6. Sequence and Memory State Access === }

function TTransformerFacade.GetKeyValueCache(LayerIdx, HeadIdx: Integer; IsKey: Boolean; Position: Integer): TDoubleArray;
var
   I, HeadDim, EmbedDim, SeqLen, HeadStart: Integer;
   SourceArray: TDoubleArray;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   
   if IsKey then
      SourceArray := FModel.LastKVectors[LayerIdx]
   else
      SourceArray := FModel.LastVVectors[LayerIdx];
   
   if Length(SourceArray) = 0 then Exit;
   
   EmbedDim := FModel.EmbedDim;
   HeadDim := FModel.HeadDim;
   HeadStart := HeadIdx * HeadDim;
   
   SetLength(Result, HeadDim);
   for I := 0 to HeadDim - 1 do
      Result[I] := SourceArray[Position * EmbedDim + HeadStart + I];
end;

procedure TTransformerFacade.ResetKeyValueCache;
var
   L: Integer;
begin
   SetLength(FKVCacheK, 0);
   SetLength(FKVCacheV, 0);
   
   if IsModelLoaded then
   begin
      for L := 0 to FModel.NumLayers - 1 do
      begin
         SetLength(FModel.LastKVectors[L], 0);
         SetLength(FModel.LastVVectors[L], 0);
      end;
   end;
end;

function TTransformerFacade.GetActivationTrace(TokenIdx, LayerIdx: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx > FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (TokenIdx < 0) or (TokenIdx >= SeqLen) then Exit;
   if Length(FModel.LastHiddenStates) <= LayerIdx then Exit;
   if Length(FModel.LastHiddenStates[LayerIdx]) < (TokenIdx + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastHiddenStates[LayerIdx][TokenIdx * EmbedDim + I];
end;

{ === 8. Batch Norm / Layer Norm Stats === }

procedure TTransformerFacade.GetLayerNormStats(LayerIdx: Integer; out Mean, StdDev: Double);
var
   I, Count, EmbedDim, SeqLen: Integer;
   Sum, SumSq, Val, Variance: Double;
begin
   Mean := 0;
   StdDev := 0;
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if Length(FModel.LastLayerNormOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastLayerNormOutputs[LayerIdx]) = 0 then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   Count := SeqLen * EmbedDim;
   
   Sum := 0;
   for I := 0 to Count - 1 do
      Sum := Sum + FModel.LastLayerNormOutputs[LayerIdx][I];
   Mean := Sum / Count;
   
   SumSq := 0;
   for I := 0 to Count - 1 do
   begin
      Val := FModel.LastLayerNormOutputs[LayerIdx][I] - Mean;
      SumSq := SumSq + Val * Val;
   end;
   Variance := SumSq / Count;
   StdDev := Sqrt(Variance);
end;

{ === 9. Gradient and Backprop Access === }

function TTransformerFacade.GetParameterGradient(LayerIdx: Integer; ParamType: TParamType): TDoubleArray;
begin
   SetLength(Result, 0);
end;

{ === 10. Structural Mutation / Dynamic Changes === }

function TTransformerFacade.AddLayer(Position: Integer): Boolean;
begin
   Result := False;
end;

function TTransformerFacade.RemoveLayer(LayerIdx: Integer): Boolean;
begin
   Result := False;
end;

function TTransformerFacade.AddHead(LayerIdx: Integer): Boolean;
begin
   Result := False;
end;

function TTransformerFacade.RemoveHead(LayerIdx, HeadIdx: Integer): Boolean;
begin
   Result := False;
end;

{ === 11. Explainability and Attribution Tools === }

function TTransformerFacade.GetSaliencyMap(TokenIdx, LayerIdx: Integer): TDoubleArray;
var
   HiddenState: TDoubleArray;
   I, EmbedDim: Integer;
   MaxAbs: Double;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   
   HiddenState := GetHiddenState(LayerIdx, TokenIdx);
   if Length(HiddenState) = 0 then Exit;
   
   EmbedDim := FModel.EmbedDim;
   SetLength(Result, EmbedDim);
   
   MaxAbs := 0;
   for I := 0 to EmbedDim - 1 do
      if Abs(HiddenState[I]) > MaxAbs then
         MaxAbs := Abs(HiddenState[I]);
   
   if MaxAbs > 0 then
      for I := 0 to EmbedDim - 1 do
         Result[I] := Abs(HiddenState[I]) / MaxAbs
   else
      for I := 0 to EmbedDim - 1 do
         Result[I] := 0;
end;

function TTransformerFacade.GetIntegratedGradients(const InputTokens: TIntArray; Steps: Integer): TDouble2DArray;
var
   SeqLen, I, S: Integer;
   BaselineEmb, InputEmb, InterpolatedEmb: TDoubleArray;
   Alpha: Double;
   Logits1, Logits2: TDoubleArray;
   EmbedDim, J: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if Steps < 1 then Steps := 50;
   
   SeqLen := Length(InputTokens);
   if SeqLen = 0 then Exit;
   
   EmbedDim := FModel.EmbedDim;
   SetLength(Result, SeqLen);
   for I := 0 to SeqLen - 1 do
      SetLength(Result[I], EmbedDim);
   
   SetLength(BaselineEmb, EmbedDim);
   for J := 0 to EmbedDim - 1 do
      BaselineEmb[J] := 0;
   
   for I := 0 to SeqLen - 1 do
   begin
      InputEmb := GetTokenEmbedding(InputTokens[I]);
      if Length(InputEmb) = 0 then Continue;
      
      for J := 0 to EmbedDim - 1 do
         Result[I][J] := 0;
      
      for S := 0 to Steps - 1 do
      begin
         Alpha := S / Steps;
         SetLength(InterpolatedEmb, EmbedDim);
         for J := 0 to EmbedDim - 1 do
            InterpolatedEmb[J] := BaselineEmb[J] + Alpha * (InputEmb[J] - BaselineEmb[J]);
         
         for J := 0 to EmbedDim - 1 do
            Result[I][J] := Result[I][J] + (InputEmb[J] - BaselineEmb[J]) / Steps;
      end;
   end;
end;

{ === 12. Residual Connection Access === }

function TTransformerFacade.GetResidualInput(LayerIdx, Position: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastResidualInputs) <= LayerIdx then Exit;
   if Length(FModel.LastResidualInputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastResidualInputs[LayerIdx][Position * EmbedDim + I];
end;

function TTransformerFacade.GetResidualOutput(LayerIdx, Position: Integer): TDoubleArray;
var
   I, EmbedDim, SeqLen: Integer;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   EmbedDim := FModel.EmbedDim;
   if (Position < 0) or (Position >= SeqLen) then Exit;
   if Length(FModel.LastResidualOutputs) <= LayerIdx then Exit;
   if Length(FModel.LastResidualOutputs[LayerIdx]) < (Position + 1) * EmbedDim then Exit;
   
   SetLength(Result, EmbedDim);
   for I := 0 to EmbedDim - 1 do
      Result[I] := FModel.LastResidualOutputs[LayerIdx][Position * EmbedDim + I];
end;

{ === 13. Visualization/Diagnostics === }

function TTransformerFacade.GetActivationHistogram(LayerIdx, HeadIdx: Integer; NumBins: Integer): TDoubleArray;
var
   I, J, BinIdx, SeqLen, HeadDim, EmbedDim, HeadStart, Count: Integer;
   MinVal, MaxVal, Range, Val: Double;
   SourceArray: TDoubleArray;
begin
   SetLength(Result, 0);
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   if NumBins < 1 then NumBins := 50;
   
   SeqLen := FModel.LastSeqLen;
   if SeqLen = 0 then Exit;
   
   EmbedDim := FModel.EmbedDim;
   HeadDim := FModel.HeadDim;
   HeadStart := HeadIdx * HeadDim;
   
   if Length(FModel.LastHiddenStates) <= LayerIdx then Exit;
   if Length(FModel.LastHiddenStates[LayerIdx]) = 0 then Exit;
   
   SourceArray := FModel.LastHiddenStates[LayerIdx];
   
   MinVal := SourceArray[HeadStart];
   MaxVal := MinVal;
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to HeadDim - 1 do
      begin
         Val := SourceArray[I * EmbedDim + HeadStart + J];
         if Val < MinVal then MinVal := Val;
         if Val > MaxVal then MaxVal := Val;
      end;
   end;
   
   Range := MaxVal - MinVal;
   if Range = 0 then Range := 1;
   
   SetLength(Result, NumBins);
   for I := 0 to NumBins - 1 do
      Result[I] := 0;
   
   Count := 0;
   for I := 0 to SeqLen - 1 do
   begin
      for J := 0 to HeadDim - 1 do
      begin
         Val := SourceArray[I * EmbedDim + HeadStart + J];
         BinIdx := Trunc((Val - MinVal) / Range * (NumBins - 1));
         if BinIdx >= NumBins then BinIdx := NumBins - 1;
         if BinIdx < 0 then BinIdx := 0;
         Result[BinIdx] := Result[BinIdx] + 1;
         Inc(Count);
      end;
   end;
   
   if Count > 0 then
      for I := 0 to NumBins - 1 do
         Result[I] := Result[I] / Count;
end;

function TTransformerFacade.GetAttentionEntropy(LayerIdx, HeadIdx: Integer): Double;
var
   SeqLen, Pos, SrcPos, Idx: Integer;
   Weight, Sum: Double;
begin
   Result := 0;
   if not IsModelLoaded then Exit;
   if (LayerIdx < 0) or (LayerIdx >= FModel.NumLayers) then Exit;
   if (HeadIdx < 0) or (HeadIdx >= FModel.NumHeads) then Exit;
   
   SeqLen := FModel.LastSeqLen;
   if SeqLen = 0 then Exit;
   if Length(FModel.LastAttentionWeights) <= LayerIdx then Exit;
   if Length(FModel.LastAttentionWeights[LayerIdx]) = 0 then Exit;
   
   Sum := 0;
   for Pos := 0 to SeqLen - 1 do
   begin
      for SrcPos := 0 to SeqLen - 1 do
      begin
         Idx := HeadIdx * SeqLen * SeqLen + Pos * SeqLen + SrcPos;
         Weight := FModel.LastAttentionWeights[LayerIdx][Idx];
         if Weight > 1e-10 then
            Sum := Sum - Weight * Ln(Weight);
      end;
   end;
   
   Result := Sum / SeqLen;
end;

{ === 5b. Optimizer State Access === }

function TTransformerFacade.GetOptimizerState(LayerIdx: Integer; ParamType: TParamType; StateType: Integer): TDoubleArray;
begin
   SetLength(Result, 0);
end;

procedure TTransformerFacade.SetOptimizerState(LayerIdx: Integer; ParamType: TParamType; StateType: Integer; const Value: TDoubleArray);
begin
end;

end.

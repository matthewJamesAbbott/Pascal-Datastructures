//
// GGUF f32 CLI Transformer
// Matthew Abbott 2025
//

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

program Transformer;

uses
   Classes, Math, SysUtils, fpjson, jsonparser;

const
   MAX_SEQ_LEN = 1024;
   GGUF_MAGIC = 'GGUF';

type
   TDoubleArray = array of Double;
   TSingleArray = array of Single;
   TIntArray = array of Integer;
   TInt64Array = array of Int64;

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
   end;

   { TTransformerModel }
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

      function GELU(X: Double): Double;
      function Softmax(const Input: TDoubleArray): TDoubleArray;
      function LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray; Dim: Integer): TDoubleArray;
      
      // All use flat arrays with explicit indexing
      function EmbedTokens(const TokenIDs: TIntArray): TDoubleArray; // Returns [seq_len * embed_dim]
      function AttentionBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function FFNBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function TransformerBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
      function ComputeLogits(const Input: TDoubleArray; SeqLen: Integer): TDoubleArray;
      function Forward(const TokenIDs: TIntArray): TDoubleArray;

   public
      constructor Create;
      destructor Destroy; override;
      function LoadModel(const GGUFPath: string): Boolean;
      function LoadTokenizer(const TokenizerPath: string): Boolean;
      function Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
      function IsModelLoaded: Boolean;
      function IsTokenizerLoaded: Boolean;
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
   AddedTokensArr: TJSONArray;
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

            // Get vocab from model.vocab
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
               WriteLn('Tokenizer loaded: ', FVocabSize, ' tokens');

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

// ==================== TGGUFLoader Implementation ====================

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
      0, 1: FStream.Seek(1, soCurrent);        // UINT8, INT8
      2, 3: FStream.Seek(2, soCurrent);        // UINT16, INT16
      4, 5, 6: FStream.Seek(4, soCurrent);     // UINT32, INT32, FLOAT32
      7: FStream.Seek(1, soCurrent);           // BOOL
      8: begin                                  // STRING
         StrLen := ReadUInt64;
         FStream.Seek(StrLen, soCurrent);
      end;
      9: begin                                  // ARRAY
         ArrType := ReadUInt32;
         ArrCount := ReadUInt64;
         for I := 0 to Min(Int64(ArrCount) - 1, 999999) do
            SkipMetadataValue(ArrType);
      end;
      10, 11, 12: FStream.Seek(8, soCurrent);  // UINT64, INT64, FLOAT64
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
   IntVal: Int64;
begin
   FStream.Read(Magic, 4);
   if Magic <> GGUF_MAGIC then
      raise Exception.Create('Invalid GGUF magic: ' + Magic);

   Version := ReadUInt32;
   TensorCount := ReadUInt64;
   MetadataCount := ReadUInt64;

   WriteLn('GGUF Version: ', Version);
   WriteLn('Tensors: ', TensorCount);
   WriteLn('Metadata entries: ', MetadataCount);

   // Parse metadata - skip values but extract config
   for I := 0 to MetadataCount - 1 do
   begin
      Key := ReadString;
      ValueType := ReadUInt32;
      
      // Extract config values we care about
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

   WriteLn('Config: embed=', FEmbedDim, ' layers=', FNumLayers, 
           ' heads=', FNumHeads, ' ffn=', FFFNDim);

   // Parse tensor info (metadata only, no data loading)
   SetLength(FTensors, TensorCount);
   for I := 0 to TensorCount - 1 do
   begin
      FTensors[I].Name := ReadString;
      FTensors[I].NumDims := ReadUInt32;
      SetLength(FTensors[I].Shape, FTensors[I].NumDims);
      for J := 0 to FTensors[I].NumDims - 1 do
         FTensors[I].Shape[J] := ReadUInt64;
      FTensors[I].DType := ReadUInt32;
      FTensors[I].DataOffset := ReadUInt64;
      FTensors[I].DataLoaded := False;
      SetLength(FTensors[I].Data, 0);
      
      FTensorMap.AddObject(FTensors[I].Name, TObject(PtrInt(I)));
   end;

   // Align to 32 bytes for tensor data
   FTensorDataStart := FStream.Position;
   if FTensorDataStart mod 32 <> 0 then
      FTensorDataStart := FTensorDataStart + (32 - (FTensorDataStart mod 32));
   
   WriteLn('Tensor data starts at offset: ', FTensorDataStart);
end;

function TGGUFLoader.LoadTensorByIndex(Idx: Integer): Boolean;
var
   J: Integer;
   NumElements: Int64;
   ActualOffset: Int64;
   F16Data: array of Word;
begin
   Result := False;
   if (Idx < 0) or (Idx > High(FTensors)) then Exit;
   if FTensors[Idx].DataLoaded then
   begin
      Result := True;
      Exit;
   end;
   
   NumElements := 1;
   for J := 0 to High(FTensors[Idx].Shape) do
      NumElements := NumElements * FTensors[Idx].Shape[J];

   ActualOffset := FTensorDataStart + FTensors[Idx].DataOffset;
   
   if ActualOffset >= FStream.Size then
   begin
      WriteLn('  ERROR: Offset ', ActualOffset, ' beyond file size ', FStream.Size);
      Exit;
   end;

   try
      SetLength(FTensors[Idx].Data, NumElements);
      FStream.Position := ActualOffset;

      case FTensors[Idx].DType of
         0: // F32
            FStream.Read(FTensors[Idx].Data[0], NumElements * 4);
         1: // F16
            begin
               SetLength(F16Data, NumElements);
               FStream.Read(F16Data[0], NumElements * 2);
               for J := 0 to NumElements - 1 do
                  FTensors[Idx].Data[J] := Float16ToFloat32(F16Data[J]);
            end;
         20: // BF16
            begin
               SetLength(F16Data, NumElements);
               FStream.Read(F16Data[0], NumElements * 2);
               for J := 0 to NumElements - 1 do
                  FTensors[Idx].Data[J] := BFloat16ToFloat32(F16Data[J]);
            end;
      else
         WriteLn('  WARNING: Unsupported dtype ', FTensors[Idx].DType);
         Exit;
      end;
      
      FTensors[Idx].DataLoaded := True;
      Result := True;
   except
      on E: Exception do
      begin
         WriteLn('  ERROR loading tensor: ', E.Message);
         SetLength(FTensors[Idx].Data, 0);
      end;
   end;
end;

function TGGUFLoader.LoadFromFile(const Filename: string): Boolean;
begin
   Result := False;
   FLoaded := False;
   FFilename := Filename;

   if not FileExists(Filename) then
   begin
      WriteLn('GGUF file not found: ', Filename);
      Exit;
   end;

   try
      WriteLn('Loading GGUF: ', Filename);
      FStream := TFileStream.Create(Filename, fmOpenRead or fmShareDenyWrite);
      ParseHeader;
      FLoaded := True;
      Result := True;
      WriteLn('GGUF parsed successfully (', Length(FTensors), ' tensors)');
   except
      on E: Exception do
         WriteLn('Error loading GGUF: ', E.Message);
   end;
end;

procedure TGGUFLoader.PrintAllTensorNames;
var
   I: Integer;
   ShapeStr: string;
   J: Integer;
begin
   WriteLn('=== All Tensor Names ===');
   for I := 0 to High(FTensors) do
   begin
      ShapeStr := '[';
      for J := 0 to High(FTensors[I].Shape) do
      begin
         if J > 0 then ShapeStr := ShapeStr + ',';
         ShapeStr := ShapeStr + IntToStr(FTensors[I].Shape[J]);
      end;
      ShapeStr := ShapeStr + ']';
      WriteLn(Format('  %3d: %s %s dtype=%d', [I, FTensors[I].Name, ShapeStr, FTensors[I].DType]));
   end;
   WriteLn('========================');
end;

function TGGUFLoader.HasTensor(const Name: string): Boolean;
begin
   Result := FTensorMap.IndexOf(Name) >= 0;
end;

function TGGUFLoader.GetTensor(const Names: array of string): TSingleArray;
var
   I, MapIdx, TensorIdx: Integer;
   FoundName: string;
begin
   SetLength(Result, 0);
   
   // Try each name until we find one
   for I := 0 to High(Names) do
   begin
      MapIdx := FTensorMap.IndexOf(Names[I]);
      if MapIdx >= 0 then
      begin
         FoundName := Names[I];
         TensorIdx := PtrInt(FTensorMap.Objects[MapIdx]);
         
         if not FTensors[TensorIdx].DataLoaded then
         begin
            Write('  Loading: ', FoundName, ' ... ');
            if LoadTensorByIndex(TensorIdx) then
               WriteLn('OK (', Length(FTensors[TensorIdx].Data), ' floats)')
            else
            begin
               WriteLn('FAILED');
               Exit;
            end;
         end;
         
         Result := FTensors[TensorIdx].Data;
         Exit;
      end;
   end;
   
   // Not found
   Write('  WARNING: Tensor not found, tried: ');
   for I := 0 to High(Names) do
   begin
      if I > 0 then Write(', ');
      Write(Names[I]);
   end;
   WriteLn;
end;

function TGGUFLoader.GetTensorShape(const Names: array of string): TInt64Array;
var
   I, MapIdx, TensorIdx: Integer;
begin
   SetLength(Result, 0);
   for I := 0 to High(Names) do
   begin
      MapIdx := FTensorMap.IndexOf(Names[I]);
      if MapIdx >= 0 then
      begin
         TensorIdx := PtrInt(FTensorMap.Objects[MapIdx]);
         Result := FTensors[TensorIdx].Shape;
         Exit;
      end;
   end;
end;

// ==================== TTransformerModel Implementation ====================

constructor TTransformerModel.Create;
begin
   inherited;
   FLoader := TGGUFLoader.Create;
   FTokenizer := TTokenizer.Create;
end;

destructor TTransformerModel.Destroy;
begin
   FLoader.Free;
   FTokenizer.Free;
   inherited;
end;

function TTransformerModel.IsModelLoaded: Boolean;
begin
   Result := FLoader.Loaded;
end;

function TTransformerModel.IsTokenizerLoaded: Boolean;
begin
   Result := FTokenizer.Loaded;
end;

function TTransformerModel.LoadModel(const GGUFPath: string): Boolean;
var
   TokenEmb, PosEmb: TSingleArray;
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

      WriteLn;
      WriteLn('Model config:');
      WriteLn('  embed_dim = ', FEmbedDim);
      WriteLn('  num_heads = ', FNumHeads);
      WriteLn('  head_dim  = ', FHeadDim);
      WriteLn('  num_layers = ', FNumLayers);
      WriteLn('  ffn_dim   = ', FFFNDim);
      WriteLn('  vocab_size = ', FVocabSize);
      
      // Test loading key tensors
      WriteLn;
      WriteLn('Verifying key tensors...');
      
      TokenEmb := FLoader.GetTensor(['token_embd.weight', 'wte.weight', 'model.wte.weight']);
      if Length(TokenEmb) > 0 then
      begin
         WriteLn('  Token embeddings: ', Length(TokenEmb), ' values');
         WriteLn('    Sample [0..4]: ', TokenEmb[0]:0:6, ', ', TokenEmb[1]:0:6, ', ', 
                 TokenEmb[2]:0:6, ', ', TokenEmb[3]:0:6, ', ', TokenEmb[4]:0:6);
      end
      else
         WriteLn('  ERROR: Token embeddings not found!');
      
      PosEmb := FLoader.GetTensor(['position_embd.weight', 'wpe.weight', 'model.wpe.weight']);
      if Length(PosEmb) > 0 then
      begin
         WriteLn('  Position embeddings: ', Length(PosEmb), ' values');
         WriteLn('    Sample [0..4]: ', PosEmb[0]:0:6, ', ', PosEmb[1]:0:6, ', ', 
                 PosEmb[2]:0:6, ', ', PosEmb[3]:0:6, ', ', PosEmb[4]:0:6);
      end
      else
         WriteLn('  ERROR: Position embeddings not found!');
   end;
end;

function TTransformerModel.LoadTokenizer(const TokenizerPath: string): Boolean;
begin
   Result := FTokenizer.LoadFromFile(TokenizerPath);
   if Result then
      FVocabSize := FTokenizer.VocabSize;
end;

function TTransformerModel.GELU(X: Double): Double;
var
   Inner: Double;
begin
   if IsNan(X) or IsInfinite(X) then
   begin
      if X > 0 then Result := X else Result := 0;
      Exit;
   end;
   if X > 10 then begin Result := X; Exit; end;
   if X < -10 then begin Result := 0; Exit; end;
   Inner := Sqrt(2.0 / Pi) * (X + 0.044715 * X * X * X);
   if Inner > 20 then Inner := 20;
   if Inner < -20 then Inner := -20;
   Result := 0.5 * X * (1.0 + Tanh(Inner));
end;

function TTransformerModel.Softmax(const Input: TDoubleArray): TDoubleArray;
var
   I, J, ValidCount: Integer;
   MaxVal, Sum, Val, ExpVal: Double;
begin
   SetLength(Result, Length(Input));
   if Length(Input) = 0 then Exit;
   
   MaxVal := -1e30;
   ValidCount := 0;
   for I := 0 to High(Input) do
   begin
      Val := Input[I];
      if IsNan(Val) then Continue;
      if IsInfinite(Val) and (Val < 0) then Continue;
      if IsInfinite(Val) and (Val > 0) then
      begin
         for J := 0 to High(Result) do Result[J] := 0;
         Result[I] := 1.0;
         Exit;
      end;
      Inc(ValidCount);
      if Val > MaxVal then MaxVal := Val;
   end;
   
   if ValidCount = 0 then
   begin
      for I := 0 to High(Result) do
         Result[I] := 1.0 / Length(Result);
      Exit;
   end;
   
   if MaxVal < -1e20 then MaxVal := 0;

   Sum := 0.0;
   for I := 0 to High(Input) do
   begin
      Val := Input[I];
      if IsNan(Val) or IsInfinite(Val) then
      begin
         Result[I] := 0;
         Continue;
      end;
      Val := Val - MaxVal;
      if Val < -88 then
         ExpVal := 0
      else
         ExpVal := Exp(Val);
      Result[I] := ExpVal;
      Sum := Sum + ExpVal;
   end;

   if Sum > 1e-30 then
      for I := 0 to High(Result) do
         Result[I] := Result[I] / Sum
   else
      for I := 0 to High(Result) do
         Result[I] := 1.0 / Length(Result);
end;

function TTransformerModel.LayerNorm(const Input: TDoubleArray; const Gamma, Beta: TSingleArray; Dim: Integer): TDoubleArray;
var
   I: Integer;
   Mean, Variance, StdDev, Val, Normalized, BetaVal: Double;
   Epsilon: Double;
begin
   Epsilon := 1e-5;
   SetLength(Result, Dim);
   
   if Length(Input) < Dim then
   begin
      for I := 0 to Dim - 1 do Result[I] := 0;
      Exit;
   end;
   if Length(Gamma) < Dim then
   begin
      Move(Input[0], Result[0], Dim * SizeOf(Double));
      Exit;
   end;

   Mean := 0.0;
   for I := 0 to Dim - 1 do
   begin
      Val := Input[I];
      if IsNan(Val) or IsInfinite(Val) then Val := 0;
      Mean := Mean + Val;
   end;
   Mean := Mean / Dim;

   Variance := 0.0;
   for I := 0 to Dim - 1 do
   begin
      Val := Input[I];
      if IsNan(Val) or IsInfinite(Val) then Val := 0;
      Variance := Variance + Sqr(Val - Mean);
   end;
   Variance := Variance / Dim;
   
   if Variance < 0 then Variance := 0;
   StdDev := Sqrt(Variance + Epsilon);
   if StdDev < Epsilon then StdDev := Epsilon;

   for I := 0 to Dim - 1 do
   begin
      Val := Input[I];
      if IsNan(Val) or IsInfinite(Val) then Val := 0;
      Normalized := (Val - Mean) / StdDev;
      if Normalized > 100 then Normalized := 100;
      if Normalized < -100 then Normalized := -100;
      if (Length(Beta) > I) then
         BetaVal := Beta[I]
      else
         BetaVal := 0;
      Result[I] := Gamma[I] * Normalized + BetaVal;
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
   
   if (Length(TokenEmb) = 0) or (Length(PosEmb) = 0) then
   begin
      WriteLn('ERROR: Missing embeddings');
      Exit;
   end;
   
   // Result[i * embed_dim + j] = token_emb + pos_emb
   for I := 0 to SeqLen - 1 do
   begin
      Idx := TokenIDs[I];
      if (Idx < 0) or (Idx >= FVocabSize) then
      begin
         WriteLn('WARNING: Token ID ', Idx, ' out of range');
         Idx := 0;
      end;
      
      for J := 0 to FEmbedDim - 1 do
      begin
         // Token embedding: [vocab_size, embed_dim] row-major
         // Position embedding: [max_seq_len, embed_dim] row-major
         Result[I * FEmbedDim + J] := 
            TokenEmb[Idx * FEmbedDim + J] + PosEmb[I * FEmbedDim + J];
      end;
   end;
end;

function TTransformerModel.AttentionBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
var
   LN1G, LN1B, QKVWeight, QKVBias, ProjWeight, ProjBias: TSingleArray;
   NormInput: TDoubleArray;
   Q, K, V, AttnOut: TDoubleArray;  // Flat arrays
   I, J, K_idx, H, Pos, SrcPos: Integer;
   Sum, Scale: Double;
   Scores, AttnWeights: TDoubleArray;
   HeadStart, QIdx, KIdx, VIdx: Integer;
begin
   SetLength(Result, SeqLen * FEmbedDim);
   
   // Load weights
   LN1G := FLoader.GetTensor([Format('blk.%d.attn_norm.weight', [LayerIdx])]);
   LN1B := FLoader.GetTensor([Format('blk.%d.attn_norm.bias', [LayerIdx])]);
   QKVWeight := FLoader.GetTensor([Format('blk.%d.attn_qkv.weight', [LayerIdx])]);
   QKVBias := FLoader.GetTensor([Format('blk.%d.attn_qkv.bias', [LayerIdx])]);
   ProjWeight := FLoader.GetTensor([Format('blk.%d.attn_output.weight', [LayerIdx])]);
   ProjBias := FLoader.GetTensor([Format('blk.%d.attn_output.bias', [LayerIdx])]);
   
   if (Length(QKVWeight) = 0) or (Length(ProjWeight) = 0) then
   begin
      WriteLn('ERROR: Missing attention weights for layer ', LayerIdx);
      Move(Input[0], Result[0], SeqLen * FEmbedDim * SizeOf(Double));
      Exit;
   end;
   
   Scale := Sqrt(FHeadDim);
   
   // Allocate Q, K, V: [seq_len, embed_dim] flat
   SetLength(Q, SeqLen * FEmbedDim);
   SetLength(K, SeqLen * FEmbedDim);
   SetLength(V, SeqLen * FEmbedDim);
   SetLength(AttnOut, SeqLen * FEmbedDim);
   SetLength(NormInput, FEmbedDim);
   
   // For each position: LayerNorm -> compute Q, K, V
   for Pos := 0 to SeqLen - 1 do
   begin
      // Extract this position's input
      for I := 0 to FEmbedDim - 1 do
         NormInput[I] := Input[Pos * FEmbedDim + I];
      
      // Apply layer norm
      if Length(LN1G) >= FEmbedDim then
         NormInput := LayerNorm(NormInput, LN1G, LN1B, FEmbedDim);
      
      // Compute Q, K, V for this position
      // GGUF stores QKV as [3*embed_dim, embed_dim] (output_dim, input_dim)
      // Q = rows 0..embed_dim-1, K = rows embed_dim..2*embed_dim-1, V = rows 2*embed_dim..3*embed_dim-1
      for I := 0 to FEmbedDim - 1 do
      begin
         // Q[pos, i] = sum_j(input[j] * W[i, j]) where W is [embed_dim, embed_dim]
         if Length(QKVBias) > I then
            Sum := QKVBias[I]
         else
            Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[I * FEmbedDim + J];
         Q[Pos * FEmbedDim + I] := Sum;
         
         // K[pos, i]
         if Length(QKVBias) > FEmbedDim + I then
            Sum := QKVBias[FEmbedDim + I]
         else
            Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[(FEmbedDim + I) * FEmbedDim + J];
         K[Pos * FEmbedDim + I] := Sum;
         
         // V[pos, i]
         if Length(QKVBias) > 2 * FEmbedDim + I then
            Sum := QKVBias[2 * FEmbedDim + I]
         else
            Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * QKVWeight[(2 * FEmbedDim + I) * FEmbedDim + J];
         V[Pos * FEmbedDim + I] := Sum;
      end;
   end;
   
   // Multi-head attention with causal mask
   SetLength(Scores, SeqLen);
   SetLength(AttnWeights, SeqLen);
   
   for H := 0 to FNumHeads - 1 do
   begin
      HeadStart := H * FHeadDim;
      
      for Pos := 0 to SeqLen - 1 do
      begin
         // Compute attention scores for this head at this position
         for SrcPos := 0 to SeqLen - 1 do
         begin
            if SrcPos > Pos then
               Scores[SrcPos] := -1e9  // Causal mask
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
         end;
         
         // Softmax
         AttnWeights := Softmax(Scores);
         
         // Weighted sum of V
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
   
   // Output projection and residual
   // ProjWeight is [embed_dim, embed_dim] stored as [out, in]
   for Pos := 0 to SeqLen - 1 do
   begin
      for I := 0 to FEmbedDim - 1 do
      begin
         if Length(ProjBias) > I then
            Sum := ProjBias[I]
         else
            Sum := 0;
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
   
   // Load weights
   LN2G := FLoader.GetTensor([Format('blk.%d.ffn_norm.weight', [LayerIdx])]);
   LN2B := FLoader.GetTensor([Format('blk.%d.ffn_norm.bias', [LayerIdx])]);
   UpWeight := FLoader.GetTensor([Format('blk.%d.ffn_up.weight', [LayerIdx])]);
   UpBias := FLoader.GetTensor([Format('blk.%d.ffn_up.bias', [LayerIdx])]);
   DownWeight := FLoader.GetTensor([Format('blk.%d.ffn_down.weight', [LayerIdx])]);
   DownBias := FLoader.GetTensor([Format('blk.%d.ffn_down.bias', [LayerIdx])]);
   
   if (Length(UpWeight) = 0) or (Length(DownWeight) = 0) then
   begin
      WriteLn('ERROR: Missing FFN weights for layer ', LayerIdx);
      Move(Input[0], Result[0], SeqLen * FEmbedDim * SizeOf(Double));
      Exit;
   end;
   
   SetLength(NormInput, FEmbedDim);
   SetLength(Hidden, FFFNDim);
   
   for Pos := 0 to SeqLen - 1 do
   begin
      // Extract and normalize
      for I := 0 to FEmbedDim - 1 do
         NormInput[I] := Input[Pos * FEmbedDim + I];
      
      if Length(LN2G) >= FEmbedDim then
         NormInput := LayerNorm(NormInput, LN2G, LN2B, FEmbedDim);
      
      // Up projection with GELU
      // UpWeight shape [768,3072] stored row-major: W[out_idx, in_idx] = W[out_idx * 768 + in_idx]
      for I := 0 to FFFNDim - 1 do
      begin
         if Length(UpBias) > I then
            Sum := UpBias[I]
         else
            Sum := 0;
         for J := 0 to FEmbedDim - 1 do
            Sum := Sum + NormInput[J] * UpWeight[I * FEmbedDim + J];
         Hidden[I] := GELU(Sum);
      end;
      
      // Down projection with residual
      // DownWeight shape [3072,768] stored row-major: W[out_idx, in_idx] = W[out_idx * 3072 + in_idx]
      for I := 0 to FEmbedDim - 1 do
      begin
         if Length(DownBias) > I then
            Sum := DownBias[I]
         else
            Sum := 0;
         for J := 0 to FFFNDim - 1 do
            Sum := Sum + Hidden[J] * DownWeight[I * FFFNDim + J];
         
         Result[Pos * FEmbedDim + I] := Input[Pos * FEmbedDim + I] + Sum;
      end;
   end;
end;

function TTransformerModel.TransformerBlock(const Input: TDoubleArray; SeqLen, LayerIdx: Integer): TDoubleArray;
var
   AfterAttn: TDoubleArray;
   I, NaNCount: Integer;
begin
   AfterAttn := AttentionBlock(Input, SeqLen, LayerIdx);
   
   NaNCount := 0;
   for I := 0 to High(AfterAttn) do
      if IsNan(AfterAttn[I]) or IsInfinite(AfterAttn[I]) then
      begin
         Inc(NaNCount);
         AfterAttn[I] := 0;
      end;
   if NaNCount > 0 then
      WriteLn('  Layer ', LayerIdx, ' attention: ', NaNCount, ' NaN/Inf fixed');
   
   Result := FFNBlock(AfterAttn, SeqLen, LayerIdx);
   
   NaNCount := 0;
   for I := 0 to High(Result) do
      if IsNan(Result[I]) or IsInfinite(Result[I]) then
      begin
         Inc(NaNCount);
         Result[I] := 0;
      end;
   if NaNCount > 0 then
      WriteLn('  Layer ', LayerIdx, ' FFN: ', NaNCount, ' NaN/Inf fixed');
end;

function TTransformerModel.ComputeLogits(const Input: TDoubleArray; SeqLen: Integer): TDoubleArray;
var
   FinalLNG, FinalLNB, TokenEmb: TSingleArray;
   LastPos, NormedPos: TDoubleArray;
   I, J, NaNCount: Integer;
   Sum, Val: Double;
begin
   SetLength(Result, FVocabSize);
   
   FinalLNG := FLoader.GetTensor(['output_norm.weight', 'ln_f.weight']);
   FinalLNB := FLoader.GetTensor(['output_norm.bias', 'ln_f.bias']);
   
   SetLength(LastPos, FEmbedDim);
   NaNCount := 0;
   for I := 0 to FEmbedDim - 1 do
   begin
      Val := Input[(SeqLen - 1) * FEmbedDim + I];
      if IsNan(Val) or IsInfinite(Val) then
      begin
         Inc(NaNCount);
         Val := 0;
      end;
      LastPos[I] := Val;
   end;
   if NaNCount > 0 then
      WriteLn('WARNING: ', NaNCount, ' NaN/Inf values in final hidden state');
   
   if Length(FinalLNG) >= FEmbedDim then
      NormedPos := LayerNorm(LastPos, FinalLNG, FinalLNB, FEmbedDim)
   else
      NormedPos := LastPos;
   
   TokenEmb := FLoader.GetTensor(['token_embd.weight', 'wte.weight']);
   
   if Length(TokenEmb) < FVocabSize * FEmbedDim then
   begin
      WriteLn('ERROR: Token embeddings too small for logits');
      Exit;
   end;
   
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
end;

function TTransformerModel.Forward(const TokenIDs: TIntArray): TDoubleArray;
var
   Hidden: TDoubleArray;
   SeqLen, L: Integer;
begin
   SeqLen := Length(TokenIDs);
   
   // Embed tokens
   Hidden := EmbedTokens(TokenIDs);
   if Length(Hidden) <> SeqLen * FEmbedDim then
   begin
      WriteLn('ERROR: Embedding failed');
      SetLength(Result, 0);
      Exit;
   end;
   
   // Forward through transformer blocks
   for L := 0 to FNumLayers - 1 do
   begin
      Write(#13, 'Layer ', L + 1, '/', FNumLayers, '...');
      Hidden := TransformerBlock(Hidden, SeqLen, L);
   end;
   WriteLn(' done');
   
   // Compute logits
   Result := ComputeLogits(Hidden, SeqLen);
end;

function TTransformerModel.Generate(const Prompt: string; MaxTokens: Integer; Temperature: Double = 1.0): string;
var
   TokenIDs: TIntArray;
   Logits, Probs: TDoubleArray;
   I, J, BestID, SelectedID: Integer;
   BestLogit, R, CumulativeProb: Double;
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
   WriteLn('Temperature: ', Temperature:0:2);
   
   if Length(TokenIDs) = 0 then
   begin
      WriteLn('Error: Could not tokenize input');
      Exit;
   end;
   
   Write('Token IDs: ');
   for I := 0 to Min(High(TokenIDs), 9) do
      Write(TokenIDs[I], ' ');
   if Length(TokenIDs) > 10 then Write('...');
   WriteLn;

   StartTime := Now;

   for I := 0 to MaxTokens - 1 do
   begin
      WriteLn;
      WriteLn('=== Generating token ', I + 1, '/', MaxTokens, ' ===');
      
      Logits := Forward(TokenIDs);
      
      if Length(Logits) = 0 then
      begin
         WriteLn('ERROR: Forward pass failed');
         Break;
      end;

      // Find best for display
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

      // Temperature sampling
      if Temperature <= 0.01 then
      begin
         // Near-zero temperature: greedy
         SelectedID := BestID;
      end
      else
      begin
         // Apply temperature scaling to logits
         for J := 0 to High(Logits) do
            Logits[J] := Logits[J] / Temperature;
         
         // Convert to probabilities via softmax
         Probs := Softmax(Logits);
         
         // Sample from distribution
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

      WriteLn('Generated token: ', SelectedID, ' = "', FTokenizer.IDToToken(SelectedID), '" (best was: ', BestID, ' logit: ', BestLogit:0:4, ')');
      
      // Show top 5 logits
      Write('Top logits: ');
      for J := 0 to 4 do
         Write(Logits[J]:0:2, ' ');
      WriteLn('...');

      // Append new token
      SetLength(TokenIDs, Length(TokenIDs) + 1);
      TokenIDs[High(TokenIDs)] := SelectedID;

      // Check for EOS
      if SelectedID = 50256 then
      begin
         WriteLn('[EOS token reached]');
         Break;
      end;
   end;

   ElapsedSecs := (Now - StartTime) * 86400;
   WriteLn;
   WriteLn(Format('Generation complete in %.1f seconds', [ElapsedSecs]));

   Result := FTokenizer.Decode(TokenIDs);
end;

// ==================== Main Program ====================

var
   Model: TTransformerModel;
   GGUFPath, TokenizerPath, Prompt: string;
   MaxTokens: Integer;
   Temperature: Double;
   GeneratedText: string;
   ShowTensors: Boolean;

begin
   WriteLn('========================================');
   WriteLn('  GPT-2 CLI - Pascal Implementation');
   WriteLn('========================================');
   WriteLn;

   if ParamCount < 2 then
   begin
      WriteLn('Usage: ', ParamStr(0), ' <model.gguf> <tokenizer.json> [prompt] [max_tokens] [temperature] [--list-tensors]');
      WriteLn;
      WriteLn('Example:');
      WriteLn('  ', ParamStr(0), ' gpt2-f32.gguf tokenizer.json "Hello world" 10 0.8');
      WriteLn('  ', ParamStr(0), ' gpt2-f32.gguf tokenizer.json --list-tensors');
      Halt(1);
   end;

   GGUFPath := ParamStr(1);
   TokenizerPath := ParamStr(2);
   ShowTensors := False;
   
   if ParamCount >= 3 then
   begin
      if ParamStr(3) = '--list-tensors' then
         ShowTensors := True
      else
         Prompt := ParamStr(3);
   end
   else
      Prompt := 'Hello';
   
   if ParamCount >= 4 then
   begin
      if ParamStr(4) = '--list-tensors' then
         ShowTensors := True
      else
         MaxTokens := StrToIntDef(ParamStr(4), 5);
   end
   else
      MaxTokens := 5;

   Temperature := 1.0;
   if ParamCount >= 5 then
   begin
      if ParamStr(5) = '--list-tensors' then
         ShowTensors := True
      else
         Temperature := StrToFloatDef(ParamStr(5), 1.0);
   end;

   Randomize;
   Model := TTransformerModel.Create;
   try
      WriteLn('Loading model...');
      if not Model.LoadModel(GGUFPath) then
      begin
         WriteLn('Failed to load model');
         Halt(1);
      end;

      if ShowTensors then
      begin
         Model.FLoader.PrintAllTensorNames;
         Halt(0);
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
      WriteLn('Temperature: ', Temperature:0:2);
      WriteLn('========================================');

      GeneratedText := Model.Generate(Prompt, MaxTokens, Temperature);

      WriteLn;
      WriteLn('========================================');
      WriteLn('GENERATED TEXT:');
      WriteLn('========================================');
      WriteLn(GeneratedText);
      WriteLn('========================================');

   finally
      Model.Free;
   end;
end.

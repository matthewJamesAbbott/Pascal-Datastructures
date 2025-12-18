//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit Trie;

interface

const
   ALPHABET_SIZE = 26;

type

   trieNode = ^TrieNodeRec;
   TrieNodeRec = record
      children: array[0..ALPHABET_SIZE-1] of trieNode;
      isEndOfWord: boolean;
      parent: trieNode;
      character: char;
   end;

   TTrie = object

   public
      constructor create();
      function charToIndex(c: char): integer;
      function indexToChar(i: integer): char;
      function createNode(c: char; parentNode: trieNode): trieNode;
      function search(root: trieNode; word: string): boolean;
      function startsWith(root: trieNode; prefix: string): boolean;
      function isEmpty(node: trieNode): boolean;
      procedure insert(var root: trieNode; word: string);
      procedure delete(var root: trieNode; word: string);
      procedure printAllWords(node: trieNode; currentWord: string);

   end;

   var
      root: trieNode;

implementation

constructor TTrie.create();
begin
   root := nil;
end;

function TTrie.charToIndex(c: char): integer;
begin
   charToIndex := ord(c) - ord('a');
end;

function TTrie.indexToChar(i: integer): char;
begin
   indexToChar := chr(i + ord('a'));
end;

function TTrie.createNode(c: char; parentNode: trieNode): trieNode;
var
   node: trieNode;
   i: integer;
begin
   node := new(trieNode);
   node^.isEndOfWord := false;
   node^.parent := parentNode;
   node^.character := c;
   for i := 0 to ALPHABET_SIZE - 1 do
      node^.children[i] := nil;
   createNode := node;
end;

function TTrie.isEmpty(node: trieNode): boolean;
var
   i: integer;
begin
   isEmpty := true;
   for i := 0 to ALPHABET_SIZE - 1 do
   begin
      if node^.children[i] <> nil then
      begin
         isEmpty := false;
         exit;
      end;
   end;
end;

procedure TTrie.insert(var root: trieNode; word: string);
var
   current: trieNode;
   i, index: integer;
   c: char;
begin
   if root = nil then
      root := createNode(#0, nil);

   current := root;
   for i := 1 to length(word) do
   begin
      c := word[i];
      index := charToIndex(c);

      if (index < 0) or (index >= ALPHABET_SIZE) then
         continue;

      if current^.children[index] = nil then
         current^.children[index] := createNode(c, current);

      current := current^.children[index];
   end;

   current^.isEndOfWord := true;
end;

function TTrie.search(root: trieNode; word: string): boolean;
var
   current: trieNode;
   i, index: integer;
begin
   if root = nil then
   begin
      search := false;
      exit;
   end;

   current := root;
   for i := 1 to length(word) do
   begin
      index := charToIndex(word[i]);

      if (index < 0) or (index >= ALPHABET_SIZE) then
      begin
         search := false;
         exit;
      end;

      if current^.children[index] = nil then
      begin
         search := false;
         exit;
      end;

      current := current^.children[index];
   end;

   search := current^.isEndOfWord;
end;

function TTrie.startsWith(root: trieNode; prefix: string): boolean;
var
   current: trieNode;
   i, index: integer;
begin
   if root = nil then
   begin
      startsWith := false;
      exit;
   end;

   current := root;
   for i := 1 to length(prefix) do
   begin
      index := charToIndex(prefix[i]);

      if (index < 0) or (index >= ALPHABET_SIZE) then
      begin
         startsWith := false;
         exit;
      end;

      if current^.children[index] = nil then
      begin
         startsWith := false;
         exit;
      end;

      current := current^.children[index];
   end;

   startsWith := true;
end;

procedure TTrie.delete(var root: trieNode; word: string);
var
   current, temp: trieNode;
   i, index: integer;
begin
   if root = nil then
      exit;

   current := root;
   for i := 1 to length(word) do
   begin
      index := charToIndex(word[i]);

      if (index < 0) or (index >= ALPHABET_SIZE) then
         exit;

      if current^.children[index] = nil then
         exit;

      current := current^.children[index];
   end;

   if not current^.isEndOfWord then
      exit;

   current^.isEndOfWord := false;

   while (current <> root) and (isEmpty(current)) and (not current^.isEndOfWord) do
   begin
      temp := current;
      current := current^.parent;
      index := charToIndex(temp^.character);
      current^.children[index] := nil;
      dispose(temp);
   end;
end;

procedure TTrie.printAllWords(node: trieNode; currentWord: string);
var
   i: integer;
begin
   if node = nil then
      exit;

   if node^.isEndOfWord then
      writeln(currentWord);

   for i := 0 to ALPHABET_SIZE - 1 do
   begin
      if node^.children[i] <> nil then
         printAllWords(node^.children[i], currentWord + indexToChar(i));
   end;
end;

end.

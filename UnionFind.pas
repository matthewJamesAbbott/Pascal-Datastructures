//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit UnionFind;

interface

const
   MAX_ELEMENTS = 1000;

type

   setNode = ^SetNodeRec;
   SetNodeRec = record
      data: integer;
      rank: integer;
      parent: setNode;
   end;

   TUnionFind = object

   private
      nodes: array[0..MAX_ELEMENTS-1] of setNode;
      numElements: integer;

   public
      constructor create();
      function createNode(data: integer): setNode;
      function makeSet(data: integer): setNode;
      function find(node: setNode): setNode;
      function findByData(data: integer): setNode;
      function connected(nodeA, nodeB: setNode): boolean;
      function getSetCount(): integer;
      procedure union(nodeA, nodeB: setNode);
      procedure unionByData(dataA, dataB: integer);
      procedure printSets();

   end;

implementation

constructor TUnionFind.create();
var
   i: integer;
begin
   numElements := 0;
   for i := 0 to MAX_ELEMENTS - 1 do
      nodes[i] := nil;
end;

function TUnionFind.createNode(data: integer): setNode;
var
   node: setNode;
begin
   node := new(setNode);
   node^.data := data;
   node^.rank := 0;
   node^.parent := node;
   createNode := node;
end;

function TUnionFind.makeSet(data: integer): setNode;
var
   node: setNode;
begin
   if numElements >= MAX_ELEMENTS then
   begin
      makeSet := nil;
      exit;
   end;

   node := createNode(data);
   nodes[numElements] := node;
   inc(numElements);
   makeSet := node;
end;

function TUnionFind.find(node: setNode): setNode;
begin
   if node = nil then
   begin
      find := nil;
      exit;
   end;

   if node^.parent <> node then
      node^.parent := find(node^.parent);

   find := node^.parent;
end;

function TUnionFind.findByData(data: integer): setNode;
var
   i: integer;
begin
   findByData := nil;
   for i := 0 to numElements - 1 do
   begin
      if nodes[i]^.data = data then
      begin
         findByData := find(nodes[i]);
         exit;
      end;
   end;
end;

function TUnionFind.connected(nodeA, nodeB: setNode): boolean;
begin
   connected := find(nodeA) = find(nodeB);
end;

procedure TUnionFind.union(nodeA, nodeB: setNode);
var
   rootA, rootB: setNode;
begin
   rootA := find(nodeA);
   rootB := find(nodeB);

   if rootA = rootB then
      exit;

   if rootA^.rank < rootB^.rank then
      rootA^.parent := rootB
   else if rootA^.rank > rootB^.rank then
      rootB^.parent := rootA
   else
   begin
      rootB^.parent := rootA;
      inc(rootA^.rank);
   end;
end;

procedure TUnionFind.unionByData(dataA, dataB: integer);
var
   nodeA, nodeB: setNode;
   i: integer;
begin
   nodeA := nil;
   nodeB := nil;

   for i := 0 to numElements - 1 do
   begin
      if nodes[i]^.data = dataA then
         nodeA := nodes[i];
      if nodes[i]^.data = dataB then
         nodeB := nodes[i];
   end;

   if (nodeA <> nil) and (nodeB <> nil) then
      union(nodeA, nodeB);
end;

function TUnionFind.getSetCount(): integer;
var
   i, count: integer;
begin
   count := 0;
   for i := 0 to numElements - 1 do
   begin
      if nodes[i]^.parent = nodes[i] then
         inc(count);
   end;
   getSetCount := count;
end;

procedure TUnionFind.printSets();
var
   i: integer;
   root: setNode;
begin
   writeln('Elements and their sets:');
   for i := 0 to numElements - 1 do
   begin
      root := find(nodes[i]);
      writeln('  ', nodes[i]^.data, ' -> Set representative: ', root^.data);
   end;
   writeln('Total sets: ', getSetCount());
end;

end.

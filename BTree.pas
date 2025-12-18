//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit BTree;

interface

const
   MIN_DEGREE = 3;
   MAX_KEYS = 2 * MIN_DEGREE - 1;
   MAX_CHILDREN = 2 * MIN_DEGREE;

type

   treeNode = ^TreeNodeRec;
   TreeNodeRec = record
      keys: array[0..MAX_KEYS-1] of integer;
      children: array[0..MAX_CHILDREN-1] of treeNode;
      numKeys: integer;
      isLeaf: boolean;
      parent: treeNode;
   end;

   TBTree = object

   public
      constructor create();
      function search(node: treeNode; key: integer): treeNode;
      function findInsertIndex(node: treeNode; key: integer): integer;
      procedure splitChild(parent: treeNode; index: integer; child: treeNode);
      procedure insertNonFull(node: treeNode; key: integer);
      procedure insert(var root: treeNode; key: integer);
      procedure inorderTraversal(node: treeNode);

   end;

   var
      root: treeNode;

implementation

constructor TBTree.create();
begin
   root := nil;
end;

function TBTree.search(node: treeNode; key: integer): treeNode;
var
   i: integer;
begin
   if node = nil then
   begin
      search := nil;
      exit;
   end;

   i := 0;
   while (i < node^.numKeys) and (key > node^.keys[i]) do
      inc(i);

   if (i < node^.numKeys) and (key = node^.keys[i]) then
      search := node
   else if node^.isLeaf then
      search := nil
   else
      search := search(node^.children[i], key);
end;

function TBTree.findInsertIndex(node: treeNode; key: integer): integer;
var
   i: integer;
begin
   i := node^.numKeys - 1;
   while (i >= 0) and (key < node^.keys[i]) do
      dec(i);
   findInsertIndex := i + 1;
end;

procedure TBTree.splitChild(parent: treeNode; index: integer; child: treeNode);
var
   newNode: treeNode;
   i, mid: integer;
begin
   mid := MIN_DEGREE - 1;

   newNode := new(treeNode);
   newNode^.isLeaf := child^.isLeaf;
   newNode^.numKeys := MIN_DEGREE - 1;
   newNode^.parent := parent;

   for i := 0 to MIN_DEGREE - 2 do
      newNode^.keys[i] := child^.keys[i + MIN_DEGREE];

   if not child^.isLeaf then
   begin
      for i := 0 to MIN_DEGREE - 1 do
      begin
         newNode^.children[i] := child^.children[i + MIN_DEGREE];
         if newNode^.children[i] <> nil then
            newNode^.children[i]^.parent := newNode;
      end;
   end;

   child^.numKeys := MIN_DEGREE - 1;

   for i := parent^.numKeys downto index + 1 do
      parent^.children[i + 1] := parent^.children[i];

   parent^.children[index + 1] := newNode;

   for i := parent^.numKeys - 1 downto index do
      parent^.keys[i + 1] := parent^.keys[i];

   parent^.keys[index] := child^.keys[mid];
   inc(parent^.numKeys);
end;

procedure TBTree.insertNonFull(node: treeNode; key: integer);
var
   i: integer;
begin
   i := node^.numKeys - 1;

   if node^.isLeaf then
   begin
      while (i >= 0) and (key < node^.keys[i]) do
      begin
         node^.keys[i + 1] := node^.keys[i];
         dec(i);
      end;
      node^.keys[i + 1] := key;
      inc(node^.numKeys);
   end
   else
   begin
      while (i >= 0) and (key < node^.keys[i]) do
         dec(i);
      inc(i);

      if node^.children[i]^.numKeys = MAX_KEYS then
      begin
         splitChild(node, i, node^.children[i]);
         if key > node^.keys[i] then
            inc(i);
      end;
      insertNonFull(node^.children[i], key);
   end;
end;

procedure TBTree.insert(var root: treeNode; key: integer);
var
   newRoot, oldRoot: treeNode;
   i: integer;
begin
   if root = nil then
   begin
      root := new(treeNode);
      root^.isLeaf := true;
      root^.numKeys := 1;
      root^.keys[0] := key;
      root^.parent := nil;
      for i := 0 to MAX_CHILDREN - 1 do
         root^.children[i] := nil;
      exit;
   end;

   if root^.numKeys = MAX_KEYS then
   begin
      newRoot := new(treeNode);
      newRoot^.isLeaf := false;
      newRoot^.numKeys := 0;
      newRoot^.parent := nil;
      for i := 0 to MAX_CHILDREN - 1 do
         newRoot^.children[i] := nil;

      oldRoot := root;
      root := newRoot;
      newRoot^.children[0] := oldRoot;
      oldRoot^.parent := newRoot;

      splitChild(newRoot, 0, oldRoot);
      insertNonFull(newRoot, key);
   end
   else
      insertNonFull(root, key);
end;

procedure TBTree.inorderTraversal(node: treeNode);
var
   i: integer;
begin
   if node = nil then
      exit;

   for i := 0 to node^.numKeys - 1 do
   begin
      if not node^.isLeaf then
         inorderTraversal(node^.children[i]);
      write(node^.keys[i], ' ');
   end;

   if not node^.isLeaf then
      inorderTraversal(node^.children[node^.numKeys]);
end;

end.

//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit AVLTree;

interface

type

   treeNode = ^TreeNodeRec;
   TreeNodeRec = record
      data: integer;
      height: integer;
      left, right, parent: treeNode;
   end;

   TAVLTree = object

   public
      constructor create();
      function getHeight(node: treeNode): integer;
      function getBalance(node: treeNode): integer;
      function maxVal(a, b: integer): integer;
      procedure updateHeight(node: treeNode);
      procedure rotateLeft(var root: treeNode; node: treeNode);
      procedure rotateRight(var root: treeNode; node: treeNode);
      procedure insertFixup(var root: treeNode; node: treeNode);
      procedure insert(var root: treeNode; data: integer);
      procedure inorderTraversal(node: treeNode);

   end;

   var
      root: treeNode;

implementation

constructor TAVLTree.create();
begin
   root := nil;
end;

function TAVLTree.getHeight(node: treeNode): integer;
begin
   if node = nil then
      getHeight := 0
   else
      getHeight := node^.height;
end;

function TAVLTree.getBalance(node: treeNode): integer;
begin
   if node = nil then
      getBalance := 0
   else
      getBalance := getHeight(node^.left) - getHeight(node^.right);
end;

function TAVLTree.maxVal(a, b: integer): integer;
begin
   if a > b then
      maxVal := a
   else
      maxVal := b;
end;

procedure TAVLTree.updateHeight(node: treeNode);
begin
   if node <> nil then
      node^.height := 1 + maxVal(getHeight(node^.left), getHeight(node^.right));
end;

procedure TAVLTree.rotateLeft(var root: treeNode; node: treeNode);
var
   pivot: treeNode;
begin
   pivot := node^.right;

   if pivot <> nil then
   begin
      node^.right := pivot^.left;
      if pivot^.left <> nil then
         pivot^.left^.parent := node;
      pivot^.parent := node^.parent;
   end;

   if node^.parent = nil then
      root := pivot
   else if node = node^.parent^.left then
      node^.parent^.left := pivot
   else
      node^.parent^.right := pivot;

   if pivot <> nil then
      pivot^.left := node;
   node^.parent := pivot;

   updateHeight(node);
   updateHeight(pivot);
end;

procedure TAVLTree.rotateRight(var root: treeNode; node: treeNode);
var
   pivot: treeNode;
begin
   pivot := node^.left;

   if pivot <> nil then
   begin
      node^.left := pivot^.right;
      if pivot^.right <> nil then
         pivot^.right^.parent := node;
      pivot^.parent := node^.parent;
   end;

   if node^.parent = nil then
      root := pivot
   else if node = node^.parent^.right then
      node^.parent^.right := pivot
   else
      node^.parent^.left := pivot;

   if pivot <> nil then
      pivot^.right := node;
   node^.parent := pivot;

   updateHeight(node);
   updateHeight(pivot);
end;

procedure TAVLTree.insertFixup(var root: treeNode; node: treeNode);
var
   current: treeNode;
   balance: integer;
begin
   current := node^.parent;

   while current <> nil do
   begin
      updateHeight(current);
      balance := getBalance(current);

      if (balance > 1) and (node^.data < current^.left^.data) then
      begin
         rotateRight(root, current);
      end
      else if (balance < -1) and (node^.data > current^.right^.data) then
      begin
         rotateLeft(root, current);
      end
      else if (balance > 1) and (node^.data > current^.left^.data) then
      begin
         rotateLeft(root, current^.left);
         rotateRight(root, current);
      end
      else if (balance < -1) and (node^.data < current^.right^.data) then
      begin
         rotateRight(root, current^.right);
         rotateLeft(root, current);
      end;

      current := current^.parent;
   end;
end;

procedure TAVLTree.insert(var root: treeNode; data: integer);
var
   node, parent: treeNode;
begin
   node := new(treeNode);
   node^.data := data;
   node^.left := nil;
   node^.right := nil;
   node^.height := 1;

   parent := nil;
   while root <> nil do
   begin
      parent := root;
      if node^.data < root^.data then
         root := root^.left
      else
         root := root^.right;
   end;

   node^.parent := parent;
   if parent = nil then
      root := node
   else if node^.data < parent^.data then
      parent^.left := node
   else
      parent^.right := node;

   insertFixup(root, node);
end;

procedure TAVLTree.inorderTraversal(node: treeNode);
begin
   if node = nil then
      exit;
   inorderTraversal(node^.left);
   writeln(node^.data, ' (h=', node^.height, ', bal=', getBalance(node), ')');
   inorderTraversal(node^.right);
end;

end.

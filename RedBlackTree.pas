//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit RedBlackTree;

interface

type

   Color = (Red, Black);
   treeNode = ^TreeNodeRec;
   TreeNodeRec = record
      data: integer;
      color: Color;
      left, right, parent: treeNode;
   end;

   TRedBlackTree = object

   public
      constructor create();
      function grandparent(node: treeNode): treenode;
      function sibling(node: treeNode): treeNode;
      function uncle(node: treeNode): treeNode;
      procedure rotateLeft(var root: treeNode; node: treeNode);
      procedure rotateRight(var root: treeNode; node: treeNode);
      procedure insertFixup(var root: treeNode; node: treeNode);
      procedure insert(var root: treeNode; data: integer);
      procedure inorderTraversal(node: treeNode);

   end;

   var
      root: treeNode;

implementation

constructor TRedBlackTree.create();
begin
   root := nil;
end;

function TRedBlackTree.grandparent(node: treeNode): treeNode;
begin
   if (node <> nil) and (node^.parent <> nil) then
      grandparent := node^.parent^.parent
   else
      grandparent := nil;
end;

function TRedBlackTree.sibling(node: treeNode): treeNode;
begin
   if (node = nil) or (node^.parent = nil) then
      sibling := nil
   else if node = node^.parent^.left then
      sibling := node^.parent^.right
   else
      sibling := node^.parent^.left;
end;

function TRedBlackTree.uncle(node: treeNode): treeNode;
begin
   if grandparent(node) = nil then
      uncle := nil
   else if node^.parent = grandparent(node)^.left then
      uncle := grandparent(node)^.right
   else
      uncle := grandparent(node)^.left;
end;

procedure TRedBlackTree.rotateLeft(var root: treeNode; node: treeNode);
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
end;

procedure TRedBlackTree.rotateRight(var root: treeNode; node: treeNode);
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
end;

procedure TRedBlackTree.insertFixup(var root: treeNode; node: treeNode);
var
   u, g: treeNode;
begin
   while (node^.parent <> nil) and (node^.parent^.color = Red) do
   begin
      g := grandparent(node);

      if node^.parent = g^.left then
      begin
         u := g^.right;
         if (u <> nil) and (u^.color = Red) then
         begin
            node^.parent^.color := Black;
            u^.color := Black;
            g^.color := Red;
            node := g;
         end
         else
         begin
            if node = node^.parent^.right then
            begin
               node := node^.parent;
               rotateLeft(root, node);
            end;

            node^.parent^.color := Black;
            g^.color := Red;
            rotateRight(root, g);
         end;
      end
      else
      begin
         u := grandparent(node);
         if (u <> nil) and (u^.color = Red) then
         begin
            node^.parent^.color := Black;
            u^.color := Black;
            g^.color := Red;
            node := g;
         end
         else
         begin
            if node = node^.parent^.left then
            begin
               node := node^.parent;
               rotateRight(root, node);
            end;

            node^.parent^.color := Black;
            g^.color := Red;
            rotateLeft(root, g);
         end;
      end;
   end;

   root^.color := Black;
end;

procedure TRedBlackTree.insert(var root: treeNode; data: integer);
var
   node, parent: treeNode;
begin
   node := new(treeNode);
   node^.data := data;
   node^.left := nil;
   node^.right := nil;
   node^.color := Red;

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

procedure TRedBlackTree.inorderTraversal(node: treeNode);
begin
   if node = nil then
      exit;
   inorderTraversal(node^.left);
   writeln(node^.data, ' (', node^.color, ')');
   inorderTraversal(node^.right);
end;

end.


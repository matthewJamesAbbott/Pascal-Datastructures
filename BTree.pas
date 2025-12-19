//
// Matthew Abbott 2025
//

{$mode objfpc}
{$M+}

program BTree;

//
// What is a B-Tree?
// -----------------
// A B-Tree is a "balanced" search tree used in file systems and databases,
// designed for keeping data sorted and allowing fast insert, find, and traverse
// operations even when the data is so large it won't fit in memory. Each node can
// hold between t-1 and 2t-1 keys (where t is "minimum degree").
//
// This program implements a B-Tree with integer keys and demonstrates the basic
// operations.
//

// ---------------
// B-Tree SETTINGS
// ---------------
const
   MIN_DEGREE = 3;                   // "t": Min. degree. Each node has at least t-1 keys, at most 2t-1 keys.
   MAX_KEYS = 2 * MIN_DEGREE - 1;    // Greatest number of keys any node may have (e.g., 5 if t=3).
   MAX_CHILDREN = 2 * MIN_DEGREE;    // Greatest number of children (subtrees) any node may have (6 if t=3).

// ----------
// DATA TYPES
// ----------
type
   treeNode = ^TreeNodeRec;      // Define a pointer type for B-Tree nodes

   // Each node contains:
   TreeNodeRec = record
      keys: array[0..MAX_KEYS-1] of integer;          // Up to MAX_KEYS sorted keys
      children: array[0..MAX_CHILDREN-1] of treeNode; // Child pointers (subtrees)
      numKeys: integer;                               // Actual number of keys used in keys[]
      isLeaf: boolean;                                // True if node is a "leaf" (has no children)
      parent: treeNode;                               // Pointer to parent node (not used by algorithm, but helps with debugging)
   end;

// Holds methods for operating on the B-Tree (constructor, insert/search/traverse)
type
   TBTree = object

      // Constructor: makes the tree empty by setting root to nil
      constructor create();

      // Search for a key in this subtree. Return pointer to node containing it, or nil if missing.
      function search(node: treeNode; key: integer): treeNode;

      // Returns the place in a node to insert a new key (to keep keys sorted).
      function findInsertIndex(node: treeNode; key: integer): integer;

      // Split a child of parent into two nodes, promoting the middle key.
      procedure splitChild(parent: treeNode; index: integer; child: treeNode);

      // For a node that isn't full, insert key into it (may be leaf or internal).
      procedure insertNonFull(node: treeNode; key: integer);

      // Insert a key into the tree, creating new root as needed.
      procedure insert(var root: treeNode; key: integer);

      // Print all keys in the tree in ascending order.
      procedure inorderTraversal(node: treeNode);
   end;

// "root" points to the top node of the whole tree
var
   root: treeNode;


//===============================
//    B-TREE IMPLEMENTATION
//===============================

// Create an empty tree by initializing root to nil
constructor TBTree.create();
begin
   root := nil;
end;

// Search recursively for given key in (sub)tree. 
// Returns pointer to node if found, nil if missing.
function TBTree.search(node: treeNode; key: integer): treeNode;
var
   i: integer;
begin
   if node = nil then
   begin
      search := nil; // base case: key not found
      exit;
   end;

   // Walk through keys in this node from left to right
   i := 0;
   while (i < node^.numKeys) and (key > node^.keys[i]) do
      inc(i);

   if (i < node^.numKeys) and (key = node^.keys[i]) then
      search := node                 // Found in this node!
   else if node^.isLeaf then
      search := nil                  // Can't go deeper - missing
   else
      search := search(node^.children[i], key); // Try appropriate child
end;

// Find index where new key belongs in a given node (keeps keys in order)
function TBTree.findInsertIndex(node: treeNode; key: integer): integer;
var
   i: integer;
begin
   i := node^.numKeys - 1;
   while (i >= 0) and (key < node^.keys[i]) do
      dec(i);
   findInsertIndex := i + 1; // Return the position after the last lower key
end;

// Split a full child: child at parent^.children[index] 
procedure TBTree.splitChild(parent: treeNode; index: integer; child: treeNode);
var
   newNode: treeNode;
   i, mid: integer;
begin
   mid := MIN_DEGREE - 1; // The median key of child

   newNode := new(treeNode);     // Allocate right half
   newNode^.isLeaf := child^.isLeaf;
   newNode^.numKeys := MIN_DEGREE - 1;
   newNode^.parent := parent;

   // Copy second half of keys into new node
   for i := 0 to MIN_DEGREE - 2 do
      newNode^.keys[i] := child^.keys[i + MIN_DEGREE];

   // If not leaf, split off the matching children
   if not child^.isLeaf then
   begin
      for i := 0 to MIN_DEGREE - 1 do
      begin
         newNode^.children[i] := child^.children[i + MIN_DEGREE];
         if newNode^.children[i] <> nil then
            newNode^.children[i]^.parent := newNode;
      end;
   end;

   child^.numKeys := MIN_DEGREE - 1;  // Shrink left node

   // Move parent child pointers right to make room
   for i := parent^.numKeys downto index + 1 do
      parent^.children[i + 1] := parent^.children[i];

   parent^.children[index + 1] := newNode;

   // Move parent's keys right to make room, then insert median
   for i := parent^.numKeys - 1 downto index do
      parent^.keys[i + 1] := parent^.keys[i];

   parent^.keys[index] := child^.keys[mid];

   inc(parent^.numKeys);
end;


// Insert a new key into node and its subtree (assumes node is not full)
procedure TBTree.insertNonFull(node: treeNode; key: integer);
var
   i: integer;
begin
   i := node^.numKeys - 1; // Start from right-most key

   if node^.isLeaf then
   begin
      // Shift higher keys right to make space
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
      // Find correct child to insert into
      while (i >= 0) and (key < node^.keys[i]) do
         dec(i);
      inc(i);

      // If chosen child is full, split before inserting
      if node^.children[i]^.numKeys = MAX_KEYS then
      begin
         splitChild(node, i, node^.children[i]);
         if key > node^.keys[i] then
            inc(i); // Adjust if new key is greater than split median
      end;

      insertNonFull(node^.children[i], key);
   end;
end;

// Main insert routine: handles splitting the root if needed
procedure TBTree.insert(var root: treeNode; key: integer);
var
   newRoot, oldRoot: treeNode;
   i: integer;
begin
   if root = nil then
   begin
      // Start a new root
      root := new(treeNode);
      root^.isLeaf := true;
      root^.numKeys := 1;
      root^.keys[0] := key;
      root^.parent := nil;
      for i := 0 to MAX_CHILDREN - 1 do
         root^.children[i] := nil;
      exit;
   end;

   // If root is full, split it and grow the tree
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

// Print the tree (all keys in increasing order). This recursively prints children and keys.
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


//===============================
//           MAIN
//===============================

var
   Tree: TBTree;    // The BTree object instance
   i, key: integer; // Loop counter and input variable

begin
   // Initialize the B-Tree to empty (root := nil)
   Tree.create();

   writeln('B-Tree demonstration!');
   writeln('---------------------');
   writeln('Inserting the numbers: 12, 7, 25, 15, 11, 21, 2, 8');

   // Insert a set of test keys (hardcoded for demo)
   Tree.insert(root, 12);
   Tree.insert(root, 7);
   Tree.insert(root, 25);
   Tree.insert(root, 15);
   Tree.insert(root, 11);
   Tree.insert(root, 21);
   Tree.insert(root, 2);
   Tree.insert(root, 8);

   writeln('All numbers inserted.');
   writeln;

   // Print all keys in tree in increasing order
   writeln('B-Tree contents, in order:');
   Tree.inorderTraversal(root);
   writeln; // Move to next line

   writeln;
   writeln('Type a value to search for (e.g. 15), or type 0 to exit.');
   repeat
     write('Enter value to search: ');
     readln(key); // Read search value from user
     if key = 0 then
       break;
     if Tree.search(root, key) <> nil then
       writeln('Found!')
     else
       writeln('Not found.');
   until key = 0;

   writeln('Done!');
end.

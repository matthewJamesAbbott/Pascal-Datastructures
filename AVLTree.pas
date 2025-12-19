//
// Matthew Abbott, 2025
//

program AVLTreeTest;

{$mode objfpc}
{$M+}

{
  Simple, highly commented AVL tree implementation in Pascal,
  with a test driver as the main program.
  Every step is explained so that a complete beginner can follow!
}

type
  // Define a type alias `treeNode` that is a pointer to a `TreeNodeRec` record.
  treeNode = ^TreeNodeRec;
  
  // Each TreeNodeRec represents a single node in the AVL tree.
  TreeNodeRec = record
    data: integer;          // The value (payload) stored at this tree node
    height: integer;        // The height of this node's subtree (distance to the farthest descendant, or 1 for leaf)
    left, right, parent: treeNode; // Pointers to the left child, right child, and parent nodes
  end;

  // TAVLTree object defines all logic for managing the AVL tree.
  // It uses the global 'root' variable (defined below) as the tree's entry point.
  TAVLTree = object
  public
    constructor create();                          // Initializes (makes) an empty AVL tree
    function getHeight(node: treeNode): integer;   // Returns height of 'node' or 0 if none
    function getBalance(node: treeNode): integer;  // Returns difference in left/right subtree heights
    function maxVal(a, b: integer): integer;       // Gets the bigger of two numbers
    procedure updateHeight(node: treeNode);        // Updates and fixes the height field of 'node'
    procedure rotateLeft(var root: treeNode; node: treeNode);  // Applies a left tree rotation to 'node'
    procedure rotateRight(var root: treeNode; node: treeNode); // Applies a right tree rotation to 'node'
    procedure insertFixup(var root: treeNode; node: treeNode); // Restores AVL property up the tree after insertion
    procedure insert(var root: treeNode; value: integer);      // Inserts a new value into tree (keeps balanced)
    procedure inorderTraversal(node: treeNode);                // Prints all tree nodes in sorted order, with stats
  end;

// This variable always points to the top/root of the AVL tree.
// Most operations will start from here.
var
  root: treeNode;

{ --------- METHODS --------- }

// Initializes (creates) an empty AVL tree by setting root to nil (meaning: tree is empty)
constructor TAVLTree.create();
begin
  root := nil;
end;

// Returns the height of 'node'. If node is nil, height is 0 (empty subtree).
function TAVLTree.getHeight(node: treeNode): integer;
begin
  if node = nil then
    getHeight := 0
  else
    getHeight := node^.height;
end;

// Computes the AVL balance factor for 'node': height(left subtree) - height(right subtree).
// - Balanced node: -1, 0, or +1
// - If this is not true, rebalancing (rotations) is needed.
function TAVLTree.getBalance(node: treeNode): integer;
begin
  if node = nil then
    getBalance := 0
  else
    getBalance := getHeight(node^.left) - getHeight(node^.right);
end;

// Returns the larger of a and b, for height calculations.
function TAVLTree.maxVal(a, b: integer): integer;
begin
  if a > b then
    maxVal := a
  else
    maxVal := b;
end;

// Recalculates and sets the height field of 'node', based on its children’s heights.
procedure TAVLTree.updateHeight(node: treeNode);
begin
  if node <> nil then
    node^.height := 1 + maxVal(getHeight(node^.left), getHeight(node^.right));
end;

// rotateLeft restores balance by rotating the subtree rooted at 'node' to the left.
// Used when the right subtree is too tall (balance = -2)
// Diagrams are very helpful here: see "AVL left rotation" on Google.
procedure TAVLTree.rotateLeft(var root: treeNode; node: treeNode);
var
  pivot: treeNode;
begin
  // The right child moves up; node moves down/left
  pivot := node^.right;
  if pivot <> nil then
  begin
    node^.right := pivot^.left;
    if pivot^.left <> nil then
      pivot^.left^.parent := node;
    pivot^.parent := node^.parent;
  end;

  // Change parent to point to pivot, or set new root if this was the top
  if node^.parent = nil then
    root := pivot
  else if node = node^.parent^.left then
    node^.parent^.left := pivot
  else
    node^.parent^.right := pivot;

  if pivot <> nil then
    pivot^.left := node;
  node^.parent := pivot;

  // Update heights, since subtree shape changed
  updateHeight(node);
  updateHeight(pivot);
end;

// rotateRight restores balance by rotating the subtree rooted at 'node' to the right.
// Used when the left subtree is too tall (balance = +2)
procedure TAVLTree.rotateRight(var root: treeNode; node: treeNode);
var
  pivot: treeNode;
begin
  // The left child moves up; node moves down/right
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

// insertFixup walks upwards from 'node', fixing AVL imbalances
// This only handles the simplest "single rotation" fixups for clarity.
// More robust AVL implementations add double rotation cases too.
procedure TAVLTree.insertFixup(var root: treeNode; node: treeNode);
var
  current: treeNode;
  balance: integer;
begin
  current := node;
  while current <> nil do
  begin
    // Update the height for each ancestor
    updateHeight(current);
    // Get current balance at this node
    balance := getBalance(current);

    // Check for left-heavy and fix with a single right rotation
    if (balance > 1) and (getBalance(current^.left) >= 0) then
      rotateRight(root, current)
    // Check for right-heavy and fix with a single left rotation
    else if (balance < -1) and (getBalance(current^.right) <= 0) then
      rotateLeft(root, current);

    // Move up to parent and continue
    current := current^.parent;
  end;
end;

// insert adds 'value' into tree with BST logic, then walk back up to rebalance.
// (This is a very basic single-rotation AVL. Production implementations have more cases!)
procedure TAVLTree.insert(var root: treeNode; value: integer);
var
  newNode, current, parent: treeNode;
begin
  // Allocate and fill a new node
  new(newNode);
  newNode^.data := value;
  newNode^.height := 1;      // New leaf node starts at height 1
  newNode^.left := nil;
  newNode^.right := nil;
  newNode^.parent := nil;

  // Special case: Tree is empty, so new node is root
  if root = nil then
  begin
    root := newNode;
    exit;
  end;

  // Binary search: Find the correct parent node to attach new value
  current := root;
  parent := nil;
  while current <> nil do
  begin
    parent := current;
    if value < current^.data then
      current := current^.left
    else
      current := current^.right;
  end;

  // Attach the new node to its parent (left or right side)
  newNode^.parent := parent;
  if value < parent^.data then
    parent^.left := newNode
  else
    parent^.right := newNode;

  // Restore the AVL balance property starting at the parent
  insertFixup(root, parent);
end;

// inorderTraversal: Recursively visit nodes in increasing order (left, node, right)
// For each node, it prints data, height, and balance factor.
procedure TAVLTree.inorderTraversal(node: treeNode);
begin
  if node = nil then exit;
  inorderTraversal(node^.left);
  Write('Node: ', node^.data, ', ');
  Write('Height: ', node^.height, ', ');
  Write('Balance: ', getBalance(node));
  Writeln;
  inorderTraversal(node^.right);
end;

{ ---------------- MAIN TEST PROGRAM ----------------- }

// The test program builds an AVL tree, adds values, and prints them.
// All steps are explained for absolute clarity.

var
  tree: TAVLTree;  // Our tree "manager" object
  i: integer;      // Loop counter
  // A set of values to insert into the tree, NOT sorted, to force balancing
  values: array[1..7] of integer = (10, 5, 13, 2, 7, 11, 15);

begin
  // Step 1: Create (initialize) the AVL tree.
  tree.create;

  // Step 2: Insert values (one by one) into the AVL tree
  Writeln('Inserting values:');
  for i := 1 to 7 do
  begin
    Write(values[i], ' ');
    tree.insert(root, values[i]);
  end;
  Writeln;

  // Step 3: Print the resulting tree in "sorted" order,
  // showing node value, height, and balance at each position.
  Writeln('In-order traversal (should be sorted):');
  tree.inorderTraversal(root);

  // Done! Study the output to understand how it self-balances and keeps track of the stats.
end.

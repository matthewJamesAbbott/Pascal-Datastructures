//
// Created by Matthew Abbott 25/8/22
//

{$mode objfpc}
{$M+}

unit StackBinaryTree;

interface

type

    treeNode = ^TreeNodeRec;
    TreeNodeRec = record
        data, nodeNumber: integer;
        leftChild, rightChild: treeNode;
    end;
	
   stackArray = array of treeNode;

    TStackBinaryTree = object

    public
        constructor create();
        procedure insertData(inputData: integer);
        function deleteNode(key: integer): boolean;
        function countNodes(): integer;
        function findNodeNumber(key: integer): integer;
    
    private
        procedure preOrder(localRoot: treeNode);
        procedure inOrder(localRoot: treeNode);
        procedure postOrder(localRoot: treeNode);
        function getSuccessor(delNode: treeNode): treeNode;
    end;
        
            TStack = object

    public
        constructor create(maxSizeInput: integer);
        procedure push(inputNode: treeNode);
        function pop(): treeNode;
        function isEmpty(): boolean;

    end;

    var
        top, maxSize: integer;
        stackArrayVar: stackArray;
        root: treeNode;
        counter, nodeCount: integer;

implementation

constructor TStackBinaryTree.create();
begin
    root := nil;
end;

procedure TStackBinaryTree.insertData(inputData: integer);

var
    newTreeNode, current, parent: treeNode;

begin
    new(newTreeNode);
    newTreeNode^.data := inputData;

    if root = nil then
        begin
            root := newTreeNode;
            nodeCount := 1;
            root^.nodeNumber := 1
        end
        else
            begin
                inc(nodeCount);
                current := root;
                while 1 <> 0 do
                    begin
                        parent := current;
                        if inputData <= current^.data then
                            begin
                                current := current^.leftChild;
                                if current = nil then
                                    begin
                                        parent^.leftChild := newTreeNode;
                                        parent^.leftChild^.nodeNumber := nodeCount;
                                        break;
                                    end;
                            end;
                        if inputData > current^.data then
                            begin
                                current := current^.rightChild;
                                if current = nil then
                                    begin
                                         parent^.rightChild := newTreeNode;
                                         parent^.rightChild^.nodeNumber := nodeCount;
                                         break;
                                    end;
                            end;
                    end;
            end;

end;

function TStackBinaryTree.deleteNode(key: integer): boolean;

var
    current, parent, successor: treeNode;
    isLeftChild: boolean;
    
begin
    current := root;
    parent := root;
    isLeftChild := true;
    while current^.data <> key do
        begin
            parent := current;
            if key < current^.data then
                begin
                    isLeftChild := true;
                    current := current^.leftChild
                end
                else
                    begin
                        isLeftChild := false;
                        current := current^.rightChild;
                    end;
            if current = nil then
                begin
                    deleteNode := false;
                end;
            if current^.leftChild = nil then
                begin
                    if current^.rightChild = nil then
                        begin
                            if current = root then 
                                root := nil
                            else if isLeftChild = true then
                                parent^.leftChild := nil
                            else
                                parent^.rightChild := nil;
                        end;
                end
                else if current^.rightChild = nil then
                    begin
                        if current = root then
                            root := current^.leftChild
                        else if isLeftChild = true then
                            parent^.leftChild := current^.rightChild
                        else
                            parent^.rightChild := current^.rightChild;
                    end
                else if current^.leftChild = nil then
                    begin
                        if current = root then
                            root := current^.rightChild
                        else if isLeftChild = true then
                            parent^.leftChild := current^.rightChild
                        else
                            parent^.rightChild := current^.rightChild;
                    end
                else
                    begin
                        successor := getSuccessor(current);
                        if current = root then
                            root := successor
                            else if isLeftChild = true then
                                parent^.leftChild := successor
                            else
                                parent^.rightChild := successor;
                        successor^.leftChild := current^.leftChild;
                    end;
            deleteNode := true;

        end;

end;

function TStackBinaryTree.getSuccessor(delNode: treeNode): treeNode;

var
    successorParent, successor, current: treeNode;

begin
    successorParent := delNode;
    successor := delNode;
    current := delNode^.rightChild;
    while current <> nil do
        begin
            successorParent := successor;
            successor := current;
            current := current^.leftChild;
        end;
    if successor <> delNode^.rightChild then
        begin
            successorParent^.leftChild := successor^.rightChild;
            successor^.rightChild := delNode^.rightChild;
        end;
    getSuccessor := successor;
end;

function TStackBinaryTree.countNodes(): integer;
begin
    counter := 1;
    preOrder(root);
    countNodes := counter;
end;

procedure TStackBinaryTree.preOrder(localRoot: treeNode);

begin
    if localRoot <> nil then
        begin
            inc(counter); {replace this line with code for pre order execution on the tree}
            preOrder(localRoot^.leftChild);
            preOrder(localRoot^.rightChild);
        end;
        
end;

procedure TStackBinaryTree.inOrder(localRoot: treeNode);

begin
    inOrder(localRoot^.leftChild);
    {insert code here for inorder execution on the tree}
    inOrder(localRoot^.rightChild);
end;

procedure TStackBinaryTree.postOrder(localRoot: treeNode);
begin
    postOrder(localRoot^.leftChild);
    postOrder(localRoot^.rightChild);
    {insert code here for postorder execution on the tree}
end;

function TStackBinaryTree.findNodeNumber(key: integer): integer;

var
    current: treeNode;

begin
    current := root;
    while current^.data <> key do
        begin
            if key < current^.data then
                current := current^.leftChild
            else
                current := current^.rightChild;
        if current = nil then
            findNodeNumber := 0;
        end;
    findNodeNumber := current^.nodeNumber;
    
end;

constructor TStack.create(maxSizeInput: integer);
begin
    maxSize := maxSizeInput;
    setLength(stackArrayVar, maxSizeInput);
    top := 0;
end;

procedure TStack.push(inputNode: treeNode);

begin
    inc(top);
    stackArrayVar[top] := inputNode;
end;

function TStack.pop(): treeNode;
begin
    dec(top);
    pop := stackArrayVar[top + 1];
end;

function TStack.isEmpty(): boolean;
begin
    if top = -1 then
        isEmpty := true
        else
            isEmpty := false;
end;

end.

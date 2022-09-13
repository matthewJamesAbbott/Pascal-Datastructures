//
// Created by Matthew Abbott 31/8/22 two days till lord of the rings rings of power. edit on 13/9/22 it really sucked what a let down.
//

{$mode objfpc}
{$M+}

unit HeapBinaryTree;

interface

uses HeapBinaryTreeNode in './HeapBinaryTreeNode.pas';

type

    THeapBinaryTree = class

    public
        constructor create();
        procedure insertData(inputData: integer);
        procedure printTree();
        function deleteNode(key: integer): boolean;
        function countNodes(): integer;
        function findNodeNumber(key: integer): integer;
    
    private
        procedure printPreOrder(p: THeapBinaryTreeNode; indent: integer);
        procedure preOrder(localRoot: THeapBinaryTreeNode);
        procedure inOrder(localRoot: THeapBinaryTreeNode);
        procedure postOrder(localRoot: THeapBinaryTreeNode);
        function getSuccessor(delNode: THeapBinaryTreeNode): THeapBinaryTreeNode;
    end;
        
    var
        root: THeapBinaryTreeNode;
        counter, nodeCount: integer;

implementation

constructor THeapBinaryTree.create();
begin
    root := nil;
end;

procedure THeapBinaryTree.insertData(inputData: integer);

var
    newTreeNode, current, parent, temp: THeapBinaryTreeNode;

begin
    newTreeNode := THeapBinaryTreeNode.create;
    newTreeNode.setData(inputData);

    if root = nil then
        begin
            root := newTreeNode;
            nodeCount := 1;
            root.setNodeNumber(1);
        end
        else
            begin
                inc(nodeCount);
                current := root;
                while 1 <> 0 do
                    begin
                        parent := current;
                        if inputData <= current.getData then
                            begin
                                current := current.getLeftChild;
                                if current = nil then
                                    begin
                                        parent.setLeftChild(newTreeNode);
                                        temp := parent.getLeftChild;
                                        temp.setNodeNumber(nodeCount);
                                        break;
                                    end;
                            end;
                        if inputData > current.getData then
                            begin
                                current := current.getRightChild;
                                if current = nil then
                                    begin
                                         parent.setRightChild(newTreeNode);
                                         temp := parent.getRightChild;
                                         temp.setNodeNumber(nodeCount);
                                         break;
                                    end;
                            end;
                    end;
            end;

end;

function THeapBinaryTree.deleteNode(key: integer): boolean;

var
    current, parent, successor: THeapBinaryTreeNode;
    isLeftChild: boolean;
    
begin
    current := root;
    parent := root;
    isLeftChild := true;
    while current.getData <> key do
        begin
            writeln(current.getData);
            parent := current;
            if key < current.getData then
                begin
                    isLeftChild := true;
                    current := current.getLeftChild
                end
                else
                    begin
                        isLeftChild := false;
                        current := current.getRightChild;
                    end;
            if current = nil then
                begin
                    deleteNode := false;
                end;
        end;
            if current.getLeftChild = nil then
                begin
                    if current.getRightChild = nil then
                        begin
                            if current = root then 
                                root := nil
                            else if isLeftChild = true then
                                parent.setLeftChild(nil)
                            else
                                parent.setRightChild(nil);
                        end;
                end
                else if current.getRightChild = nil then
                    begin
                        if current = root then
                            root := current.getLeftChild
                        else if isLeftChild = true then
                            parent.setLeftChild(current.getLeftChild)
                        else
                            parent.setRightChild(current.getLeftChild);
                    end
                else if current.getLeftChild = nil then
                    begin
                        if current = root then
                            root := current.getRightChild
                        else if isLeftChild = true then
                            parent.setLeftChild(current.getRightChild)
                        else
                            parent.setRightChild(current.getRightChild);
                    end
                else
                    begin
                        successor := getSuccessor(current);
                        if current = root then
                            root := successor
                            else if isLeftChild = true then
                                parent.setLeftChild(successor)
                            else
                                parent.setRightChild(successor);
                            successor.setLeftChild(current.getLeftChild);
                    end;
            deleteNode := true;

        

end;

function THeapBinaryTree.getSuccessor(delNode: THeapBinaryTreeNode): THeapBinaryTreeNode;

var
    successorParent, successor, current: THeapBinaryTreeNode;

begin
    successorParent := delNode;
    successor := delNode;
    current := delNode.getRightChild;
    while current <> nil do
        begin
            successorParent := successor;
            successor := current;
            current := current.getLeftChild;
        end;
    if successor <> delNode.getRightChild then
        begin
            successorParent.setLeftChild(successor.getRightChild);
            successor.setRightChild(delNode.getRightChild);
        end;
    getSuccessor := successor;
end;

function THeapBinaryTree.countNodes(): integer;
begin
    counter := 0;
    preOrder(root);
    countNodes := counter;
end;

procedure THeapBinaryTree.preOrder(localRoot: THeapBinaryTreeNode);

begin
    if localRoot <> nil then
        begin
            inc(counter); {replace this line with code for pre order execution on the tree}
            preOrder(localRoot.getLeftChild);
            preOrder(localRoot.getRightChild);
        end;
        
end;

procedure THeapBinaryTree.inOrder(localRoot: THeapBinaryTreeNode);

begin
    inOrder(localRoot.getLeftChild);
    {insert code here for inorder execution on the tree}
    inOrder(localRoot.getRightChild);
end;

procedure THeapBinaryTree.postOrder(localRoot: THeapBinaryTreeNode);
begin
    postOrder(localRoot.getLeftChild);
    postOrder(localRoot.getRightChild);
    {insert code here for postorder execution on the tree}
end;

function THeapBinaryTree.findNodeNumber(key: integer): integer;

var
    current: THeapBinaryTreeNode;

begin
    current := root;
    while current.getData <> key do
        begin
            if key < current.getData then
                current := current.getLeftChild
            else
                current := current.getRightChild;
        if current = nil then
            findNodeNumber := 0;
        end;
    findNodeNumber := current.getNodeNumber;
    
end;

procedure THeapBinaryTree.printTree();
begin
    printPreOrder(root, 0);
end;

procedure THeapBinaryTree.printPreOrder(p: THeapBinaryTreeNode; indent: integer);

var
    i: integer;

begin
    if indent <> 0 then
        begin
            for i := 0 to indent do
                begin
                    write(' ');
                end;
        end;
 
    writeln(p.getData);


    if p.getLeftChild <> nil then
        begin
            printPreOrder(p.getLeftChild, indent + 4);
        end;
    if p.getRightChild <> nil then
        begin
            printPreOrder(p.getRightChild, indent + 4);
        end;

         
end;
end.

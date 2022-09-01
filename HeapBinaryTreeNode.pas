//
// Created by Matthew Abbott 31/8/22
//

{$mode objfpc}
{$M+}

unit HeapBinaryTreeNode;

interface

type

    THeapBinaryTreeNode = class
    private
        data, nodeNumber: integer;
        leftChild, rightChild: THeapBinaryTreeNode;

    public
        constructor create();
        procedure setLeftChild(inputNode: THeapBinaryTreeNode);
        function getLeftChild: THeapBinaryTreeNode;
        procedure setRightChild(inputNode: THeapBinaryTreeNode);
        function getRightChild: THeapBinaryTreeNode;
        procedure setData(inputData: integer);
        function getData: integer;
        procedure setNodeNumber(inputNumber: integer);
        function getNodeNumber: integer;
    end;

implementation

constructor THeapBinaryTreeNode.create();
begin
    leftChild := nil;
    rightChild := nil;
end;

procedure THeapBinaryTreeNode.setLeftChild(inputNode: THeapBinaryTreeNode);
begin
    leftChild := inputNode;
end;

procedure THeapBinaryTreeNode.setRightChild(inputNode: THeapBinaryTreeNode);
begin
    rightChild := inputNode;
end;

function THeapBinaryTreeNode.getLeftChild: THeapBinaryTreeNode;
begin
    getLeftChild := leftChild;
end;

function THeapBinaryTreeNode.getRightChild: THeapBinaryTreeNode;
begin
    getRightChild := rightChild;
end;

procedure THeapBinaryTreeNode.setData(inputData: integer);
begin
    data := inputData;
end;

function THeapBinaryTreeNode.getData: integer;
begin
    getData := data;
end;

function THeapBinaryTreeNode.getNodeNumber: integer;
begin
    getNodeNumber := nodeNumber;
end;

procedure THeapBinaryTreeNode.setNodeNumber(inputNumber: integer);
begin
    nodeNumber := inputNumber;
end;
end.



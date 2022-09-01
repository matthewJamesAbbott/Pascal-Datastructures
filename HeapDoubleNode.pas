//
// Created by Matthew Abbott 29/8/22
//

{$mode objfpc}
{$M+}

unit HeapDoubleNode;

interface

type

    THeapDoubleNode = class
    private
        data: integer;

    public
        next, previous: THeapDoubleNode;
        constructor create();
        procedure setNext(inputNode: THeapDoubleNode);
        procedure setPrevious(inputNode: THeapDoubleNode);
        procedure setNextPrevious(inputNode: THeapDoubleNode);
        function getNext: THeapDoubleNode;
        function getPrevious: THeapDoubleNode;
        procedure setData(inputData: integer);
        function getData: integer;
    end;

implementation

constructor THeapDoubleNode.create();
begin
    next := nil;
end;

procedure THeapDoubleNode.setNext(inputNode: THeapDoubleNode);
begin
    next := inputNode;
end;

procedure THeapDoubleNode.setPrevious(inputNode: THeapDoubleNode);
begin
    previous := inputNode;
end;

procedure THeapDoubleNode.setNextPrevious(inputNode: THeapDoubleNode);
begin
    next.setPrevious(inputNode);
end;

function THeapDoubleNode.getPrevious: THeapDoubleNode;
begin
    getPrevious := previous;
end;

function THeapDoubleNode.getNext: THeapDoubleNode;
begin
    getNext := next;
end;

procedure THeapDoubleNode.setData(inputData: integer);
begin
    data := inputData;
end;

function THeapDoubleNode.getData: integer;
begin
    getData := data;
end;
end.


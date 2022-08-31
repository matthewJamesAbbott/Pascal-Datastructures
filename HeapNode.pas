//
// Created by Matthew Abbott 28/8/22
//

{$mode objfpc}
{$M+}

unit HeapNode;

interface

type
    
    THeapNode = class
    private
        data: integer;
        next: THeapNode;

    public
        constructor create();
        procedure setNext(inputNode: THeapNode);
        function getNext: THeapNode;
        procedure setData(inputData: integer);
        function getData: integer;
    end;

implementation

constructor THeapNode.create();
begin
    next := nil;
end;


procedure THeapNode.setNext(inputNode: THeapNode);
begin
    next := inputNode;
end;

function THeapNode.getNext: THeapNode;
begin
    getNext := next;
end;

procedure THeapNode.setData(inputData: integer);
begin
    data := inputData;
end;

function THeapNode.getData: integer;
begin
    getData := data;
end;
end.

//
// Created by Matthew Abbott 28/8/22
//

{$mode objfpc}
{$M+}

unit HeapLinkedList;

interface

uses
    HeapNode in './HeapNode.pas';

type

    THeapLinkedList = class

    public
        constructor create();
        procedure destroyNodes();
        procedure addNode(inputData: integer);
        procedure deleteFirstNode();
        procedure deleteLastNode();
        procedure deleteSpecificNode(nodeNumber: integer);
        function countNodes(): integer;
        function returnSpecificNodesData(nodeNumber: integer): integer;
        function returnHeadsData(): integer;
        function returnTailsData(): integer;
        function returnNodeNumberOfFirstInstanceOfData(inputData: integer): integer;
    end;
var 
    head: THeapNode;

implementation

constructor THeapLinkedList.create();
begin
end;

procedure THeapLinkedList.destroyNodes();

var
    iterator: integer;

begin
    for iterator := 1 to countNodes() -1 do
        begin
            deleteLastNode();
        end;
end;
procedure THeapLinkedList.addNode(inputData: integer);
var
    temp, newNode: THeapNode;
begin

    newNode := THeapNode.create();
        newNode.setData(inputData);
        if head = nil then
        head := newNode
    else
        begin
            temp := head;
            while temp.getNext <> nil do
                begin
                    temp := temp.getNext;
                end;
            temp.setNext(newNode);
        end;
end;

procedure THeapLinkedList.deleteFirstNode();

var
    temp: THeapNode;
begin
    temp := head;
    head := head.getNext;
    temp.destroy;
end;

procedure THeapLinkedList.deleteLastNode();
var
    temp1,temp2: THeapNode;
begin
    temp1 := head;
    while temp1.getNext <> nil do
        begin
            if temp1.getNext <> nil then
                begin
                    temp2 := temp1;
                end;
            temp1 := temp1.getNext;
        end;
        temp2.setNext(nil);
        temp2.destroy;
end;

procedure THeapLinkedList.deleteSpecificNode(nodeNumber: integer);
var
    temp1,temp2: THeapNode;
    index: integer;
begin
    if nodeNumber = 1 then
        deleteFirstNode()
    else
        begin
            temp1 := head;
            for index := 2 to nodeNumber do
                begin
                    temp2 := temp1;
                    temp1 := temp1.getNext
                end;
            temp2.setNext(temp1.getNext);
        end;
end;

function THeapLinkedList.countNodes(): integer;
var
    index: integer;
    temp: THeapNode;
begin
    if head = nil then
        countNodes := 0
    else
        begin
            index := 1;
            temp := head;
            while temp.getNext <> nil do
                begin
                    temp := temp.getNext;
                    inc(index);
                end;
            countNodes := index;
        end;
end;

function THeapLinkedList.returnSpecificNodesData(nodeNumber: integer): integer;
var
    temp: THeapNode;
    index: integer;

begin
    temp := head;
    if nodeNumber = 1 then
        returnSpecificNodesData := temp.getData
    else
        begin
            for index := 2 to nodeNumber do
                begin
                    temp := temp.getNext;
                end;
        end;
    
    returnSpecificNodesData := temp.getData;
end;

function THeapLinkedList.returnHeadsData(): integer;
begin
    returnHeadsData := head.getData;
end;

function THeapLinkedList.returnTailsData(): integer;
var
    temp: THeapNode;
begin
    temp := head;
    while temp.getNext <> nil do
        begin
            temp := temp.getNext;
        end;
    returnTailsData := temp.getData;
end;

function THeapLinkedList.returnNodeNumberOfFirstInstanceOfData(inputData: integer): integer;
var
    temp: THeapNode;
    index: integer;
begin
    temp := head;
    if temp.getData = inputData then
        returnNodeNumberOfFirstInstanceOfData := 1
    else
        begin
            index := 1;
            while temp.getNext <> nil do
                begin
                    inc(index);
                    temp := temp.getNext;
                    if temp.getData = inputData then
                        begin
                            returnNodeNumberOfFirstInstanceOfData := index; 
                        end;
                end;
        end;
end;
end.

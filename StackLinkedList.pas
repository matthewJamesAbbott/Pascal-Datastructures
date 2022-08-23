//
// Created by Matthew Abbott 23/8/22
//

{$mode objfpc}
{$M+}

unit StackLinkedList;

interface

type

    node = ^NodeRec;
    NodeRec = record
        data: integer;
        next: node;
    end;

    TStackLinkedList = object

    public
        constructor create();
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
    head : node;

implementation

constructor TStackLinkedList.create();
begin
end;

procedure TStackLinkedList.addNode(inputData: integer);
var
    temp, newNode: node;
begin
    new(newNode);
    newNode^.data := inputData;
    newNode^.next := nil;
    if head = nil then
        head := newNode
    else
        begin
            temp := head;
            while temp^.next <> nil do
                begin
                    temp := temp^.next
                end;
            temp^.next := newNode;
        end;
end;

procedure TStackLinkedList.deleteFirstNode();
begin
    head := head^.next;
end;

procedure TStackLinkedList.deleteLastNode();
var
    temp1,temp2: node;
begin
    temp1 := head;
    while temp1^.next <> nil do
        begin
            temp2 := temp1;
            temp1 := temp1^.next;
        end;
        temp1 := temp2;

end;

procedure TStackLinkedList.deleteSpecificNode(nodeNumber: integer);
var
    temp1,temp2: node;
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
                    temp1 := temp1^.next
                end;
            temp2^.next := temp1^.next;
        end;
end;

function TStackLinkedList.countNodes(): integer;
var
    index: integer;
    temp: node;
begin
    if head = nil then
        countNodes := 0
    else
        begin
            index := 1;
            temp := head;
            while temp^.next <> nil do
                begin
                    temp := temp^.next;
                    inc(index);
                end;
            countNodes := index;
        end;
end;

function TStackLinkedList.returnSpecificNodesData(nodeNumber: integer): integer;
var
    temp: node;
    index: integer;

begin
    if nodeNumber = 1 then
        temp := head
    else
        begin
            for index := 2 to nodeNumber do
                begin
                    temp := temp^.next;
                end;
        end;
    returnSpecificNodesData := temp^.data;
end;

function TStackLinkedList.returnHeadsData(): integer;
begin
    returnHeadsData := head^.data;
end;

function TStackLinkedList.returnTailsData(): integer;
var
    temp: node;
begin
    temp := head;
    while temp^.next <> nil do
        begin
            temp := temp^.next;
        end;
    returnTailsData := temp^.data;
end;

function TStackLinkedList.returnNodeNumberOfFirstInstanceOfData(inputData: integer): integer;
var
    temp: node;
    index: integer;
begin
    temp := head;
    if temp^.data = inputData then
        returnNodeNumberOfFirstInstanceOfData := 1
    else
        begin
            index := 1;
            while temp^.next <> nil do
                begin
                    inc(index);
                    temp := temp^.next;
                    if temp^.data = inputData then
                        begin
                            returnNodeNumberOfFirstInstanceOfData := index; 
                        end;
                end;
        end;
end;
end.

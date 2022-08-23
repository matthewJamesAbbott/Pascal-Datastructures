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
        function countNodes(): integer;
        function returnSpecificNodesData(nodeNumber: integer): integer;
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

function TStackLinkedList.countNodes(): integer;
var
    index, indexAdd: integer;
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
                    indexAdd := index;
                    index := indexAdd + 1;
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
end.

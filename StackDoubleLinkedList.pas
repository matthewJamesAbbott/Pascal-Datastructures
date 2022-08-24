//
// Created by Matthew Abbott 24/8/22
//

{$mode objfpc}
{$M+}

unit StackDoubleLinkedList;

interface

type

    doubleNode = ^DoubleNodeRec;
    DoubleNodeRec = record
        data: integer;
        next, previous: doubleNode;
    end;

    TStackDoubleLinkedList = object

    public 
        constructor create();
        procedure insertFirst(inputData: integer);
        procedure insertLast(inputData: integer);
        procedure deleteFirst();
        procedure deleteLast();
        function deleteNodeForFirstInstanceOfData(key: integer): boolean;
        function insertAfter(key, inputData: integer): boolean;
        function returnSpecificNodesData(nodeNumber: integer): integer;
        function countNodes(): integer;
    end;



var
    head, tail: doubleNode;

implementation

constructor TStackDoubleLinkedList.create();
begin
end;

procedure TStackDoubleLinkedList.insertFirst(inputData: integer);

var
    newDoubleNode: doubleNode;
begin
    new(newDoubleNode);
    newDoubleNode^.data := inputData;
    if head = nil then
            tail := newDoubleNode
        else
            begin
                head^.previous := newDoubleNode;
            end;
    newDoubleNode^.next := head;
    head := newDoubleNode;
end;

procedure TStackDoubleLinkedList.insertLast(inputData: integer);

var
    newDoubleNode: doubleNode;
begin
    new(newDoubleNode);
    newDoubleNode^.data := inputData;
    if head = nil then
            head := newDoubleNode
        else
            begin
                tail^.next := newDoubleNode;
                newDoubleNode^.previous := tail;
            end;
            tail := newDoubleNode;
end;

procedure TStackDoubleLinkedList.deleteFirst();

begin
    if head^.next = nil then
            tail := nil
        else
            begin
                head^.next^.previous := nil;
            end;
    head := head^.next;
end;

procedure TStackDoubleLinkedList.deleteLast();
begin
    if head^.next = nil then
            head := nil
        else
            begin
                tail^.previous^.next := nil;
            end;
    tail := tail^.previous;
end;

function TStackDoubleLinkedList.deleteNodeForFirstInstanceOfData(key: integer): boolean;

var
    current: doubleNode;

begin
    current := head;
    while current^.data <> key do
        begin
            current := current^.next;
            if current = nil then
                begin
                    deleteNodeForFirstInstanceOfData := false;
                end;
            if current = head then
                begin
                    head := current^.next;
                    deleteNodeForFirstInstanceOfData := true
                end
                else
                    begin
                        current^.previous^.next := current^.next;
                        deleteNodeForFirstInstanceOfData := true;
                    end;
            if current = tail then
                begin
                    tail := current^.previous;
                    deleteNodeForFirstInstanceOfData := true
                end

                else
                    begin
                        current^.next^.previous := current^.previous;
                        deleteNodeForFirstInstanceOfData := true;
                    end;
        end;

end;

function TStackDoubleLinkedList.insertAfter(key, inputData: integer): boolean;

var
    current, newDoubleNode: doubleNode;

begin
    current := head;
    while current^.data <> key do
        begin
            current := current^.next;
            if current = nil then
                begin
                    insertAfter := false;
                end;
        end;
    new(newDoubleNode);
    newDoubleNode^.data := inputData;
    if current = tail then
        begin
            newDoubleNode^.next := nil;
            tail := newDoubleNode
        end

        else
            begin
                newDoubleNode^.next := current^.next;
                current^.next^.previous := newDoubleNode;
            end;
    newDoubleNode^.previous := current;
    current^.next := newDoubleNode;
    insertAfter := true;
end;

function TStackDoubleLinkedList.returnSpecificNodesData(nodeNumber: integer): integer;

var
    temp: doubleNode;
    index: integer;

begin
    temp := head;
    if nodeNumber = 1 then
        returnSpecificNodesData := temp^.data
        else
            begin
                for index := 2 to nodeNumber do
                    begin
                        temp := temp^.next;
                    end;
            end;
    returnSpecificNodesData := temp^.data;
end;

function TStackDoubleLinkedList.countNodes(): integer;

var
    index: integer;
    temp: doubleNode;

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
end.

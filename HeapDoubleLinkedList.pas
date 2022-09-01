//
// Created by Matthew Abbott 24/8/22
//

{$mode objfpc}
{$M+}

unit HeapDoubleLinkedList;

interface

uses
    HeapDoubleNode in './HeapDoubleNode.pas';
type


    THeapDoubleLinkedList = class

    public 
        constructor create();
        procedure destroyNodes();
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
    head, tail: THeapDoubleNode;

implementation

constructor THeapDoubleLinkedList.create();
begin
end;

procedure THeapDoubleLinkedList.destroyNodes();
 
var
    iterator: integer;
 
begin
    for iterator := 1 to countNodes() -1 do
        begin
            deleteLast();
        end;
end;

    
procedure THeapDoubleLinkedList.insertFirst(inputData: integer);

var
    newDoubleNode: THeapDoubleNode;
begin
    newDoubleNode := THeapDoubleNode.create();
    newDoubleNode.setData(inputData);
    if head = nil then
            tail := newDoubleNode
        else
            begin
                head.setPrevious(newDoubleNode);
            end;
    newDoubleNode.setNext(head);
    head := newDoubleNode;
end;

procedure THeapDoubleLinkedList.insertLast(inputData: integer);

var
    newDoubleNode: THeapDoubleNode;
begin
    newDoubleNode := THeapDoubleNode.create();
    newDoubleNode.setData(inputData);
    if head = nil then
            head := newDoubleNode
        else
            begin
                tail.setNext(newDoubleNode);
                newDoubleNode.setPrevious(tail);
            end;
            tail := newDoubleNode;
end;

procedure THeapDoubleLinkedList.deleteFirst();

var
    temp: THeapDoubleNode;

begin
    if head.getNext() = nil then
            tail := nil
        else
            begin
                temp := head.getNext();
                temp.setPrevious(nil);
            end;
    head := head.getNext();
end;

procedure THeapDoubleLinkedList.deleteLast();

var
    temp: THeapDoubleNode;

begin
    if head.getNext() = nil then
            head := nil
        else
            begin
                temp := tail.getPrevious;
                temp.setNext(nil);
            end;
    tail := tail.getPrevious();
end;

function THeapDoubleLinkedList.deleteNodeForFirstInstanceOfData(key: integer): boolean;

var
    current, temp: THeapDoubleNode;

begin
    current := head;
    while current.getData() <> key do
        begin
            current := current.getNext();
            if current = nil then
                begin
                    deleteNodeForFirstInstanceOfData := false;
                end;
            if current = head then
                begin
                    head := current.getNext();
                    deleteNodeForFirstInstanceOfData := true
                end
                else
                    begin
                        temp := current.getPrevious();
                        temp.setNext(current.getNext());
                        deleteNodeForFirstInstanceOfData := true;
                    end;
            if current = tail then
                begin
                    tail := current.getPrevious();
                    deleteNodeForFirstInstanceOfData := true
                end

                else
                    begin
                        temp := current.getNext();
                        temp.setPrevious(current.getPrevious());
                        deleteNodeForFirstInstanceOfData := true;
                    end;
        end;

end;

function THeapDoubleLinkedList.insertAfter(key, inputData: integer): boolean;

var
    current, newDoubleNode, temp: THeapDoubleNode;

begin
    current := head;
    while current.getData() <> key do
        begin
            current := current.next;
            if current = nil then
                begin
                    insertAfter := false;
                end;
        end;
    newDoubleNode := THeapDoubleNode.create();
    newDoubleNode.setData(inputData);
    if current = tail then
        begin
            tail := newDoubleNode
        end

        else
            begin
                writeln('attempting insert');
                newDoubleNode.setNext(current.getNext());
                current.setNextPrevious(newDoubleNode);
                temp := current.getNext();
                temp.setPrevious(newDoubleNode);

            end;
    newDoubleNode.setPrevious(current);
    current.setNext(newDoubleNode);
    insertAfter := true;
end;

function THeapDoubleLinkedList.returnSpecificNodesData(nodeNumber: integer): integer;

var
    temp: THeapDoubleNode;
    index: integer;

begin
    temp := head;
    if nodeNumber = 1 then
        returnSpecificNodesData := temp.getData()
        else
            begin
                for index := 2 to nodeNumber do
                    begin
                        temp := temp.getNext();
                    end;
            end;
    returnSpecificNodesData := temp.getData();
end;

function THeapDoubleLinkedList.countNodes(): integer;

var
    index: integer;
    temp: THeapDoubleNode;

begin
    if head = nil then
            countNodes := 0
        else
            begin
                index := 1;
                temp := head;
                while temp.getNext() <> nil do
                    begin
                        temp := temp.getNext();
                        inc(index);
                    end;
                countNodes := index;
            end;
end;
end.

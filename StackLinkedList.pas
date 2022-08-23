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

end.

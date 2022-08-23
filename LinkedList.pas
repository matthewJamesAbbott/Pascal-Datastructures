//
// Created by Matthew Abbott 23/8/22
//

{$mode objfpc}
{$M+}

unit LinkedList;

interface

type

    node = ^NodeRec;
    NodeRec = record
        data: integer;
        next: node;
    end;

    TLinkedList = class

    public
        constructor create();
        procedure addNode(inputData: integer);
    end;
var 
    head : node;

implementation

constructor TLinkedList.create();
begin
end;

procedure TLinkedList.addNode(inputData: integer);
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

end.

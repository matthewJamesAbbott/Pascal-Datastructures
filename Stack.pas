//
// Created by Matthew Abbott 24/8/22
//

{$mode objfpc}
{$M+}

unit Stack;

interface

type
    stackArray = array of integer;

    TStack = object

    public
        constructor create(maxSizeInput: integer);
        procedure push(inputNumber: integer);
        function pop(): integer;
        function peek(): integer;
        function isEmpty(): boolean;
        function isFull(): boolean;

    end;

var
    top, maxSize: integer;
    stackArrayVar: stackArray;

implementation

constructor TStack.create(maxSizeInput: integer);
begin
    maxSize := maxSizeInput;
    setLength(stackArrayVar, maxSizeInput);
    top := 0;
end;

procedure TStack.push(inputNumber: integer);

begin
    inc(top);
    stackArrayVar[top] := inputNumber;
end;

function TStack.pop(): integer;
begin
    dec(top);
    pop := stackArrayVar[top + 1];
end;

function TStack.peek(): integer;
begin
    peek := stackArrayVar[top];
end;

function TStack.isEmpty(): boolean;
begin
    if top = -1 then
        isEmpty := true
        else
            isEmpty := false;
end;

function TStack.isFull(): boolean;
begin
    if top = maxSize - 1 then
        isFull := true
        else
            isFull := false;
end;

end.

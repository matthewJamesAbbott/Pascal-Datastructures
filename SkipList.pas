//
// Created by Matthew Abbott 2/3/2023
//

{$mode objfpc}
{$M+}

unit SkipList;

interface

const
   MAX_LEVEL = 16;
   PROBABILITY = 0.5;

type

   skipNode = ^SkipNodeRec;
   ForwardArray = array[0..MAX_LEVEL-1] of skipNode;
   SkipNodeRec = record
      data: integer;
      level: integer;
      forward: ForwardArray;
   end;

   TSkipList = object

   private
      currentLevel: integer;
      header: skipNode;

   public
      constructor create();
      function randomLevel(): integer;
      function createNode(data: integer; level: integer): skipNode;
      function search(data: integer): skipNode;
      procedure insert(data: integer);
      procedure delete(data: integer);
      procedure printList();
      procedure printAllLevels();

   end;

implementation

constructor TSkipList.create();
var
   i: integer;
begin
   randomize;
   currentLevel := 0;
   header := createNode(-MaxInt, MAX_LEVEL);
   for i := 0 to MAX_LEVEL - 1 do
      header^.forward[i] := nil;
end;

function TSkipList.randomLevel(): integer;
var
   lvl: integer;
begin
   lvl := 0;
   while (random < PROBABILITY) and (lvl < MAX_LEVEL - 1) do
      inc(lvl);
   randomLevel := lvl;
end;

function TSkipList.createNode(data: integer; level: integer): skipNode;
var
   node: skipNode;
   i: integer;
begin
   node := new(skipNode);
   node^.data := data;
   node^.level := level;
   for i := 0 to MAX_LEVEL - 1 do
      node^.forward[i] := nil;
   createNode := node;
end;

function TSkipList.search(data: integer): skipNode;
var
   current: skipNode;
   i: integer;
begin
   current := header;

   for i := currentLevel downto 0 do
   begin
      while (current^.forward[i] <> nil) and (current^.forward[i]^.data < data) do
         current := current^.forward[i];
   end;

   current := current^.forward[0];

   if (current <> nil) and (current^.data = data) then
      search := current
   else
      search := nil;
end;

procedure TSkipList.insert(data: integer);
var
   current: skipNode;
   update: ForwardArray;
   newNode: skipNode;
   lvl, i: integer;
begin
   current := header;

   for i := 0 to MAX_LEVEL - 1 do
      update[i] := nil;

   for i := currentLevel downto 0 do
   begin
      while (current^.forward[i] <> nil) and (current^.forward[i]^.data < data) do
         current := current^.forward[i];
      update[i] := current;
   end;

   current := current^.forward[0];

   if (current = nil) or (current^.data <> data) then
   begin
      lvl := randomLevel();

      if lvl > currentLevel then
      begin
         for i := currentLevel + 1 to lvl do
            update[i] := header;
         currentLevel := lvl;
      end;

      newNode := createNode(data, lvl);

      for i := 0 to lvl do
      begin
         newNode^.forward[i] := update[i]^.forward[i];
         update[i]^.forward[i] := newNode;
      end;
   end;
end;

procedure TSkipList.delete(data: integer);
var
   current: skipNode;
   update: ForwardArray;
   i: integer;
begin
   current := header;

   for i := 0 to MAX_LEVEL - 1 do
      update[i] := nil;

   for i := currentLevel downto 0 do
   begin
      while (current^.forward[i] <> nil) and (current^.forward[i]^.data < data) do
         current := current^.forward[i];
      update[i] := current;
   end;

   current := current^.forward[0];

   if (current <> nil) and (current^.data = data) then
   begin
      for i := 0 to currentLevel do
      begin
         if update[i]^.forward[i] <> current then
            break;
         update[i]^.forward[i] := current^.forward[i];
      end;

      dispose(current);

      while (currentLevel > 0) and (header^.forward[currentLevel] = nil) do
         dec(currentLevel);
   end;
end;

procedure TSkipList.printList();
var
   current: skipNode;
begin
   current := header^.forward[0];
   write('List: ');
   while current <> nil do
   begin
      write(current^.data, ' ');
      current := current^.forward[0];
   end;
   writeln;
end;

procedure TSkipList.printAllLevels();
var
   current: skipNode;
   i: integer;
begin
   for i := currentLevel downto 0 do
   begin
      write('Level ', i, ': ');
      current := header^.forward[i];
      while current <> nil do
      begin
         write(current^.data, ' ');
         current := current^.forward[i];
      end;
      writeln;
   end;
end;

end.

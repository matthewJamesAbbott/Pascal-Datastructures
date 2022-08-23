{$mode objfpc}
{$M+}

program DataStructureTest;


uses
    StackLinkedList in './StackLinkedList.pas';

    
var
    stackLinkedListVar: TStackLinkedList;
    index: integer;

begin

    stackLinkedListVar.create();
    writeln('created List object');
    stackLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    stackLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    stackLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    stackLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    stackLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    stackLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    stackLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    stackLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    stackLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    stackLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    stackLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    stackLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    stackLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    stackLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    stackLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    stackLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    stackLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    stackLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    stackLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    stackLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    stackLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    stackLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    stackLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    stackLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    stackLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');    
    writeln('there are ' , stackLinkedListVar.countNodes() , ' items in the list');
    
    for index := 1 to stackLinkedListVar.countNodes() do
        begin
            writeln('extracting data from node ', index , ' : ' , stackLinkedListVar.returnSpecificNodesData(index));
        end;
        

end.

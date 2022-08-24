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
    stackLinkedListVar.deleteFirstNode();
    writeln('deleted first node of list');
    writeln('there are ' , stackLinkedListVar.countNodes() , ' items in the list');

    for index := 1 to stackLinkedListVar.countNodes() do
        begin
            writeln('extracting data from node ', index , ' : ' , stackLinkedListVar.returnSpecificNodesData(index));
        end;
    stackLinkedListVar.deleteLastNode();
    writeln('deleted last node of list');
    writeln('there are ' , stackLinkedListVar.countNodes() , ' items in the list');

    for index := 1 to stackLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , stackLinkedListVar.returnSpecificNodesData(index));
         end;
    stackLinkedListVar.deleteSpecificNode(13);
    writeln('deleted the 13th node of list');
    writeln('there are ' , stackLinkedListVar.countNodes() , ' items in the list');

    for index := 1 to stackLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , stackLinkedListVar.returnSpecificNodesData(index));
         end;
    writeln('extracting data from node 13 : ', stackLinkedListVar.returnSpecificNodesData(13));
    writeln('extracting data from head node : ', stackLinkedListVar.returnHeadsData());
    writeln('extracting data from tail node', stackLinkedListVar.returnTailsData());
    writeln('the number 5 first instance found at node : ', stackLinkedListVar.returnNodeNumberOfFirstInstanceOfData(5));


end.

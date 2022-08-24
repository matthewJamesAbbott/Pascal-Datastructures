{$mode objfpc}
{$M+}

program DataStructureTest;


uses
    StackLinkedList in './StackLinkedList.pas',
    StackDoubleLinkedList in './StackDoubleLinkedList.pat';

    
var
    stackLinkedListVar: TStackLinkedList;
    stackDoubleLinkedListVar: TStackDoubleLinkedList;
    index: integer;

begin

    { -----START PROCESS TESTS FOR StackLinkedList.pas----- }

    writeln('{ -----START PROCESS TESTS FOR StackLinkedList.pas----- }');
    writeln();
    writeln();
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
    writeln('extracting data from tail node : ', stackLinkedListVar.returnTailsData());
    writeln('the number 5 first instance found at node : ', stackLinkedListVar.returnNodeNumberOfFirstInstanceOfData(5));

    { -----END PROCESS TESTS FOR StackLinkedList.pas----- }

    { -----START PROCESS TEST FOR StackDoubleLinkList.pas----- }

    writeln();
    writeln();
    writeln('{ -----START PROCESS TEST FOR StackDoubleLinkList.pas----- }');
    writeln();
    writeln();

    stackDoubleLinkedListVar.create();
    writeln('created list object');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into list');
    stackDoubleLinkedListVar.insertFirst(2);
    writeln('inserted the number 2 into list');
    stackDoubleLinkedListVar.insertFirst(7);
    writeln('inserted the number 7 into list');
    stackDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into list');
    stackDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into list');
    stackDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into list');
    stackDoubleLinkedListVar.insertFirst(9);
    writeln('inserted the number 9 into list');
    stackDoubleLinkedListVar.insertFirst(4);
    writeln('inserted the number 4 into list');
    stackDoubleLinkedListVar.insertFirst(8);
    writeln('inserted the number 8 into list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into list');
    stackDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into list');
    writeln('there are ', stackDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
         end;

    { -----END PROCSS TESTS FOR StackDoubleLinkedList.pas----- }

end.

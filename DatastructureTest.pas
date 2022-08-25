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
    writeln('inserted the number 3 into First position of list');
    stackDoubleLinkedListVar.insertFirst(2);
    writeln('inserted the number 2 into First position of list');
    stackDoubleLinkedListVar.insertFirst(7);
    writeln('inserted the number 7 into First position of list');
    stackDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into First position of list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    stackDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into First position of list');
    stackDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into First position of list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    stackDoubleLinkedListVar.insertFirst(9);
    writeln('inserted the number 9 into First position of list');
    stackDoubleLinkedListVar.insertFirst(4);
    writeln('inserted the number 4 into First position of list');
    stackDoubleLinkedListVar.insertFirst(8);
    writeln('inserted the number 8 into First position of list');
    stackDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    stackDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into First position of list');
    writeln('there are ', stackDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
         end;

    stackDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    stackDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    stackDoubleLinkedListVar.insertLast(8);
    writeln('inserted the number 8 into last position of the list');
    stackDoubleLinkedListVar.insertLast(9);
    writeln('inserted the number 9 into last position of the list');
    stackDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    stackDoubleLinkedListVar.insertLast(0);
    writeln('inserted the number 0 into last position of the list');
    stackDoubleLinkedListVar.insertLast(2);
    writeln('inserted the number 2 into last position of the list');
    stackDoubleLinkedListVar.insertLast(7);
    writeln('inserted the number 7 into last position of the list');
    stackDoubleLinkedListVar.insertLast(5);
    writeln('inserted the number 5 into last position of the list');
    stackDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    stackDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    stackDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    stackDoubleLinkedListVar.insertLast(8);
    writeln('inserted the number 8 into last position of the list');
    stackDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    stackDoubleLinkedListVar.insertLast(4);
    writeln('inserted the number 4 into last position of the list');
    stackDoubleLinkedListVar.insertLast(0);
    writeln('inserted the number 0 into last position of the list');
    stackDoubleLinkedListVar.insertLast(4);
    writeln('inserted the number 4 into last position of the list');
    stackDoubleLinkedListVar.insertLast(1);
    writeln('inserted the number 1 into last position of the list');
    writeln('there are ', stackDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
          begin
              writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
          end;
    stackDoubleLinkedListVar.deleteFirst();
    writeln('the first node has been deleted');
    writeln('there are ', stackDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
           begin
               writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
           end;
    stackDoubleLinkedListVar.insertAfter(8,8);
    writeln('inserted 8 after the first instance of 8');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
           begin
               writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
           end;
    writeln('node 23 contains : ', stackDoubleLinkedListVar.returnSpecificNodesData(23));
    stackDoubleLinkedListVar.deleteLast();
    writeln('the last node of the list has been deleted');
    writeln('there are ', stackDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to stackDoubleLinkedListVar.countNodes() do
           begin
                writeln('extracting data from node ', index , ' : ' , stackDoubleLinkedListVar.returnSpecificNodesData(index));
           end;

    writeln();
    writeln();
    writeln('{ -----END PROCESS TESTS FOR StackDoubleLinkedList.pas----- }');
    writeln();
    writeln();

           
    { -----END PROCESS TESTS FOR StackDoubleLinkedList.pas----- }
end.

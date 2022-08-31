{$mode objfpc}
{$M+}

program DataStructureTest;


uses
    StackLinkedList in './StackLinkedList.pas',
    StackDoubleLinkedList in './StackDoubleLinkedList.pas',
    StackBinaryTree in './StackBinaryTree.pas',
    HeapLinkedList in './HeapLinkedList.pas',
    HeapDoubleLinkedList in './HeapDoubleLinkedList.pas';

var
    stackLinkedListVar: TStackLinkedList;
    stackDoubleLinkedListVar: TStackDoubleLinkedList;
    stackBinaryTreeVar: TStackBinaryTree;
    heapLinkedListVar: THeapLinkedList;
    heapDoubleLinkedListVar: THeapDoubleLinkedList;
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

    writeln();
    writeln();
    writeln('{ -----END PROCESS TESTS FOR StackLinkedList.pas----- }');
    writeln();
    writeln();


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

    { -----START PROCESS TEST FOR StackBinaryTree.pas----- }

    writeln();
    writeln();
    writeln('{ -----START PROCESS TESTS FOR StackBinaryTree.pas----- }');
    writeln();
    writeln();

    stackBinaryTreeVar.create();
    writeln('created tree object');
    stackBinaryTreeVar.insertData(4);
    writeln('inserted the number 4 into tree');
    stackBinaryTreeVar.insertData(6);
    writeln('inserted the number 6 into tree');
    stackBinaryTreeVar.insertData(3);
    writeln('inserted the number 3 into tree');
    stackBinaryTreeVar.insertData(5);
    writeln('inserted the number 5 into tree');
    stackBinaryTreeVar.insertData(8);
    writeln('inserted the number 8 into tree');
    stackBinaryTreeVar.insertData(7);
    writeln('inserted the number 7 into tree');
    stackBinaryTreeVar.insertData(9);
    writeln('inserted the number 0 into tree');
    stackBinaryTreeVar.insertData(2);
    writeln('inserted the number 2 into tree');
    stackBinaryTreeVar.insertData(3);
    writeln('inserted the number 3 into tree');
    stackBinaryTreeVar.insertData(1);
    writeln('inserted the number 1 into tree');
    stackBinaryTreeVar.insertData(5);
    writeln('inserted the number 5 into tree');
    stackBinaryTreeVar.insertData(9);
    writeln('inserted the number 9 into tree');
    stackBinaryTreeVar.insertData(5);
    writeln('inserted the number 5 into tree');
    stackBinaryTreeVar.insertData(4);
    writeln('inserted the number 4 into tree');
    writeln('there are ', stackBinaryTreeVar.countNodes(), ' nodes on the tree');
    writeln('the number 4 was found at node number ', stackBinaryTreeVar.findNodeNumber(4));
    writeln('the number 8 was found at node number ', stackBinaryTreeVar.findNodeNumber(8));
    stackBinaryTreeVar.printTree();
    writeln(stackBinaryTreeVar.deleteNode(8));
    writeln('the node for number 8 has been deleted');
    writeln('there are ', stackBinaryTreeVar.countNodes(), ' nodes on the tree');
    stackBinaryTreeVar.printTree();
    writeln('tree printed to terminal');    

    writeln();
    writeln();
    writeln('{ -----END OF PROCESS TESTS FOR StackBinaryTree.pas----- }');
    writeln();
    writeln();


    { -----END OF PROCESS TESTS FOR StackBinaryTree.pas----- }


    { -----START PROCESS TESTS FOR HeapLinkedList.pas----- }

    writeln('{ -----START PROCESS TESTS FOR HeapLinkedList.pas----- }');
    writeln();
    writeln();
    
    
    heapLinkedListVar := THeapLinkedList.create();
    writeln('created List object');
    heapLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    heapLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    heapLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    heapLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    heapLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    heapLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    heapLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    heapLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    heapLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    heapLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    heapLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    heapLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    heapLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    heapLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    heapLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    heapLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    heapLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
  heapLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    heapLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    heapLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');
    heapLinkedListVar.addNode(5);
    writeln('inserted the number 5 into list');
    heapLinkedListVar.addNode(2);
    writeln('inserted the number 2 into list');
    heapLinkedListVar.addNode(3);
    writeln('inserted the number 3 into list');
    heapLinkedListVar.addNode(7);
    writeln('inserted the number 7 into list');
    heapLinkedListVar.addNode(1);
    writeln('inserted the number 1 into list');    
    writeln('there are ' , heapLinkedListVar.countNodes() , ' items in the list');
    for index := 1 to heapLinkedListVar.countNodes() do
        begin
            writeln('extracting data from node ', index , ' : ' , heapLinkedListVar.returnSpecificNodesData(index));
        end;
    heapLinkedListVar.deleteFirstNode();
    writeln('deleted first node of list');
    writeln('there are ' , heapLinkedListVar.countNodes() , ' items in the list');
    for index := 1 to heapLinkedListVar.countNodes() do
        begin
            writeln('extracting data from node ', index , ' : ' , heapLinkedListVar.returnSpecificNodesData(index));
        end;
    heapLinkedListVar.deleteLastNode();
    writeln('deleted last node of list');
    writeln('there are ' , heapLinkedListVar.countNodes() , ' items in the list');
    for index := 1 to heapLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , heapLinkedListVar.returnSpecificNodesData(index));
         end;
    heapLinkedListVar.deleteSpecificNode(13);
    writeln('deleted the 13th node of list');
    writeln('there are ' , heapLinkedListVar.countNodes() , ' items in the list');
    for index := 1 to heapLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , heapLinkedListVar.returnSpecificNodesData(index));
         end;
    writeln('extracting data from node 13 : ', heapLinkedListVar.returnSpecificNodesData(13));
    writeln('extracting data from head node : ', heapLinkedListVar.returnHeadsData());
    writeln('extracting data from tail node : ', heapLinkedListVar.returnTailsData());
    writeln('the number 5 first instance found at node : ', heapLinkedListVar.returnNodeNumberOfFirstInstanceOfData(5));
    heapLinkedListVar.destroyNodes();
    writeln('linked list has been destroyed and all of its nodes taken out of heap memory');
    heapLinkedListVar.destroy();
    writeln();
    writeln();
    writeln('{ -----END PROCESS TESTS FOR heapLinkedList.pas----- }');
    writeln();
    writeln();
    
 
        { -----START PROCESS TEST FOR HeapDoubleLinkedList.pas----- }

    writeln();
    writeln();
    writeln('{ -----START PROCESS TEST FOR HeapDoubleLinkedList.pas----- }');
    writeln();
    writeln();

    heapDoubleLinkedListVar := THeapDoubleLinkedList.create();
    writeln('created list object');
    heapDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    heapDoubleLinkedListVar.insertFirst(2);
    writeln('inserted the number 2 into First position of list');
    heapDoubleLinkedListVar.insertFirst(7);
    writeln('inserted the number 7 into First position of list');
    heapDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into First position of list');
    heapDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    heapDoubleLinkedListVar.insertFirst(5);
    writeln('inserted the number 5 into First position of list');
    heapDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into First position of list');
    heapDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    heapDoubleLinkedListVar.insertFirst(9);
    writeln('inserted the number 9 into First position of list');
    heapDoubleLinkedListVar.insertFirst(4);
    writeln('inserted the number 4 into First position of list');
    heapDoubleLinkedListVar.insertFirst(8);
    writeln('inserted the number 8 into First position of list');
    heapDoubleLinkedListVar.insertFirst(3);
    writeln('inserted the number 3 into First position of list');
    heapDoubleLinkedListVar.insertFirst(1);
    writeln('inserted the number 1 into First position of list');
    writeln('there are ', heapDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to heapDoubleLinkedListVar.countNodes() do
         begin
             writeln('extracting data from node ', index , ' : ' , heapDoubleLinkedListVar.returnSpecificNodesData(index));
         end;

    heapDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    heapDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    heapDoubleLinkedListVar.insertLast(8);
    writeln('inserted the number 8 into last position of the list');
    heapDoubleLinkedListVar.insertLast(9);
    writeln('inserted the number 9 into last position of the list');
    heapDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    heapDoubleLinkedListVar.insertLast(0);
    writeln('inserted the number 0 into last position of the list');
    heapDoubleLinkedListVar.insertLast(2);
    writeln('inserted the number 2 into last position of the list');
    heapDoubleLinkedListVar.insertLast(7);
    writeln('inserted the number 7 into last position of the list');
    heapDoubleLinkedListVar.insertLast(5);
    writeln('inserted the number 5 into last position of the list');
    heapDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    heapDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    heapDoubleLinkedListVar.insertLast(3);
    writeln('inserted the number 3 into last position of the list');
    heapDoubleLinkedListVar.insertLast(8);
    writeln('inserted the number 8 into last position of the list');
    heapDoubleLinkedListVar.insertLast(6);
    writeln('inserted the number 6 into last position of the list');
    heapDoubleLinkedListVar.insertLast(4);
    writeln('inserted the number 4 into last position of the list');
    heapDoubleLinkedListVar.insertLast(0);
    writeln('inserted the number 0 into last position of the list');
    heapDoubleLinkedListVar.insertLast(4);
    writeln('inserted the number 4 into last position of the list');
    heapDoubleLinkedListVar.insertLast(1);
    writeln('inserted the number 1 into last position of the list');
    writeln('there are ', heapDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to heapDoubleLinkedListVar.countNodes() do
          begin
              writeln('extracting data from node ', index , ' : ' , heapDoubleLinkedListVar.returnSpecificNodesData(index));
          end;
    heapDoubleLinkedListVar.deleteFirst();
    writeln('the first node has been deleted');
    writeln('there are ', heapDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to heapDoubleLinkedListVar.countNodes() do
           begin
               writeln('extracting data from node ', index , ' : ' , heapDoubleLinkedListVar.returnSpecificNodesData(index));
           end;
    heapDoubleLinkedListVar.insertAfter(8,8);
    writeln('inserted 8 after the first instance of 8');
    writeln('notice the disconnect on the previous node hence there being no nodes before 8 and no insertion');
    for index := 1 to heapDoubleLinkedListVar.countNodes() do
           begin
               writeln('extracting data from node ', index , ' : ' , heapDoubleLinkedListVar.returnSpecificNodesData(index));
           end;
    writeln('node 23 contains : ', heapDoubleLinkedListVar.returnSpecificNodesData(23));
    heapDoubleLinkedListVar.deleteLast();
    writeln('the last node of the list has been deleted');
    writeln('there are ', heapDoubleLinkedListVar.countNodes(), ' nodes in list');
    for index := 1 to heapDoubleLinkedListVar.countNodes() do
           begin
                writeln('extracting data from node ', index , ' : ' , heapDoubleLinkedListVar.returnSpecificNodesData(index));
           end;
    heapDoubleLinkedListVar.destroyNodes();
    heapDoubleLinkedListVar.destroy();
    writeln('double linked list has been destroyed and all of its nodes taken out of heap memory');

    writeln();
    writeln();
    writeln('{ -----END PROCESS TESTS FOR HeapDoubleLinkedList.pas----- }');
    writeln();
    writeln();

           
    { -----END PROCESS TESTS FOR HeapDoubleLinkedList.pas----- }

    

end.

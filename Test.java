import java.util.*;

public class Test {

  public void treeTest() {
    Practice2 P =new Practice2();
    Practice2.TreeNode n1 = P.new TreeNode(5);
    Practice2.TreeNode n2 = P.new TreeNode(3);
    Practice2.TreeNode n3 = P.new TreeNode(6);
    Practice2.TreeNode n4 = P.new TreeNode(2);
    Practice2.TreeNode n5 = P.new TreeNode(4);
    Practice2.TreeNode n6 = P.new TreeNode(7);

    n1.left=n2;
    n1.right=n3;
    n2.left=n4;
    n2.right=n5;
    n3.right=n6;

    P.deleteNode2(n1,3);
  }

  public void linkedListTest(){
    Practice2 P = new Practice2();
    Practice2.ListNode l1 = P.new ListNode(2);
    Practice2.ListNode l2 = P.new ListNode(1);
    Practice2.ListNode l3 = P.new ListNode(5);
    Practice2.ListNode l4 = P.new ListNode(4);
    Practice2.ListNode l5 = P.new ListNode(5);
    l1.next=l2;
    l2.next=l3;
//    l3.next=l4;
//    l4.next=l5;
    P.nextLargerNodes(l1);
  }

  public static void main(String[] args){
     Test t = new Test();
     t.treeTest();
  }
}

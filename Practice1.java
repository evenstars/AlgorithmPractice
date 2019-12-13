import java.util.*;

import javafx.util.Pair;
class Practice1 {
  public class Node {
    public int val;
    public List<Node> children;

    public Node() {
    }

    public Node(int _val, List<Node> _children) {
      val = _val;
      children = _children;
    }
  }

  public class BiListNode {
    int val;
    BiListNode next;
    BiListNode last;
  }

  public class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
      val = x;
    }
  }

  public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
      val = x;
    }
  }

  public void treeTest() {
    TreeNode n1 = new TreeNode(1);
    TreeNode n2 = new TreeNode(2);
    TreeNode n3 = new TreeNode(3);
    TreeNode n4 = new TreeNode(-5);
    TreeNode n5 = new TreeNode(4);
    TreeNode n6 = new TreeNode(0);
    TreeNode n7 = new TreeNode(8);
    TreeNode n8 = new TreeNode(7);
    TreeNode n9 = new TreeNode(4);
    TreeNode n10 = new TreeNode(2);

    n1.left = n3;
    n3.right = n2;

    recoverTree2(n1);
    treeIte(n1);
  }

  public void treeIte(TreeNode root) {
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      int n = q.size();
      String sent = "";
      for (int i = 0; i < n; i++) {
        TreeNode temp = q.poll();
        sent = sent + temp.val + " ";
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      System.out.println(sent);
    }
  }

  public int reverse(int x) {
    int INTMAX = Integer.MAX_VALUE;
    int INTMIN = Integer.MIN_VALUE;
    int ans = 0;

    while (x != 0) {
      int pop = x % 10;
      x /= 10;
      if ((ans > INTMAX / 10) || (ans == INTMAX / 10 && pop >= 7))
        return 0;
      if ((ans < INTMIN / 10) || (ans == INTMIN / 10 && pop <= -8))
        return 0;
      ans = ans * 10 + pop;
    }
    return ans;
  }

  public boolean isPalindrome(int x) {
    if (x < 0)
      return false;
    else if (x >= 0 && x < 10)
      return true;

    int temp = 0;
    int mem = x;
    while (mem > 0) {
      int digit = mem % 10;
      temp = temp * 10 + digit;
      mem /= 10;
    }
    if (temp == x)
      return true;
    else
      return false;
  }

  public int romanToInt(String s) {
//        I             1
//        V             5
//        X             10
//        L             50
//        C             100
//        D             500
//        M             1000
    HashMap<Character, Integer> convertor = new HashMap<>();
    convertor.put('I', 1);
    convertor.put('V', 5);
    convertor.put('X', 10);
    convertor.put('L', 50);
    convertor.put('C', 100);
    convertor.put('D', 500);
    convertor.put('M', 1000);
    int ans = 0;
    int general = 0;
    for (int i = s.length() - 1; i >= 0; i--) {
      int temp = convertor.get(s.charAt(i));

      if (temp >= general)
        ans += temp;
      else
        ans -= temp;
      general = temp;
    }
    return ans;
  }

  public boolean isValid(String s) {
    int len = s.length();
    Character[] charStack = new Character[len / 2];
    int index = -1;
    char pop = '0';
    for (int i = 0; i < len; i++) {
      switch (s.charAt(i)) {
        case '(':
          if (++index >= len / 2)
            return false;
          charStack[index] = '(';
          break;
        case '{':
          if (++index >= len / 2)
            return false;
          charStack[index] = '{';
          break;
        case '[':
          if (++index >= len / 2)
            return false;
          charStack[index] = '[';
          break;
        case ')':
          if (index < 0)
            return false;
          pop = charStack[index];
          if (pop != '(')
            return false;
          charStack[index] = null;
          index--;
          break;
        case ']':
          if (index < 0)
            return false;
          pop = charStack[index];
          if (pop != '[')
            return false;
          charStack[index] = null;
          index--;
          break;
        case '}':
          if (index < 0)
            return false;
          pop = charStack[index];
          if (pop != '{')
            return false;
          charStack[index] = null;
          index--;
          break;
        default:
          return false;
      }
    }
    if (index == -1)
      return true;
    else
      return false;
  }

  public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0)
      return "";
    String prefix = strs[0];
    for (int i = 1; i < strs.length; i++) {
      String temp = strs[i];
      while (temp.indexOf(prefix) != 0) {
        prefix = prefix.substring(0, prefix.length() - 1);
      }
    }
    return prefix;
  }

  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null)
      return l2;
    else if (l2 == null)
      return l1;

    ListNode ans = new ListNode(-1);
    ListNode cursor = ans;
    while (l1 != null || l2 != null) {
      int n1 = l1.val;
      int n2 = l2.val;
      if (n1 >= n2) {
        cursor.val = n2;
        l2 = l2.next;
        if (l2 == null) {
          cursor.next = l1;
          break;
        } else {
          ListNode next = new ListNode(-1);
          cursor.next = next;
          cursor = cursor.next;
        }
      } else {
        cursor.val = n1;
        l1 = l1.next;
        if (l1 == null) {
          cursor.next = l2;
          break;
        } else {
          ListNode next = new ListNode(-1);
          cursor.next = next;
          cursor = cursor.next;
        }
      }
    }
    return ans;
  }

  public void LinkedListtest() {
    ListNode n1 = new ListNode(1);
    ListNode n2 = new ListNode(4);
    ListNode n3 = new ListNode(5);
    ListNode n4 = new ListNode(1);
    ListNode n5 = new ListNode(3);
    ListNode n6 = new ListNode(4);
    ListNode n7 = new ListNode(2);
    ListNode n8 = new ListNode(6);
    n1.next = n2;
    n2.next = n3;

    n4.next=n5;
    n5.next = n6;
    n7.next = n8;
    ListNode ans = mergeKLists4(new ListNode[]{n1,n4,n7});
//    System.out.println(ans);
    while(ans!=null){
      System.out.println(ans.val);
      ans=ans.next;}

  }

  public boolean hasGroupsSizeX(int[] deck) {
    int len = deck.length;
    if (len <= 1)
      return false;
    Map<Integer, Integer> cal = new HashMap<>();
    for (int i = 0; i < len; i++) {
      if (cal.containsKey(deck[i]))
        cal.put(deck[i], cal.get(deck[i]) + 1);
      else
        cal.put(deck[i], 1);
    }
    int x = 2;
    boolean sign = true;
    Iterator ite;
    while (true) {
      ite = cal.entrySet().iterator();
      while (ite.hasNext()) {
        Map.Entry entry = (Map.Entry) ite.next();
        if ((int) entry.getValue() < x)
          return false;
        else if ((int) entry.getValue() % x != 0) {
          sign = false;
          break;
        }
      }
      if (sign == true) {
        System.out.println(x);
        return true;
      } else {
        sign = true;
        x++;
      }
    }
  }

  public boolean wordPattern(String pattern, String str) {
    String[] words = str.split(" ");
    int len = pattern.length();
    if (words.length != len)
      return false;
    Map<String, Character> correspond = new HashMap();
    for (int i = 0; i < len; i++) {
      String mod = words[i];
      if (!correspond.containsKey(mod)) {
        if (correspond.containsValue(pattern.charAt(i)))
          return false;
        correspond.put(mod, pattern.charAt(i));
      } else {
        Character oriWord = correspond.get(mod);
        if (!oriWord.equals(pattern.charAt(i)))
          return false;
      }
    }
    return true;
  }

  public int numUniqueEmails(String[] emails) {
    Set<String> ans = new HashSet<String>();
    StringBuilder sb;
    for (String s : emails) {
      sb = new StringBuilder(s);
      int pind = s.indexOf("+");
      int ati = s.indexOf("@");
      int fore;
      if (pind >= 0) {
        sb.delete(pind, ati);
        fore = pind;
      } else {
        fore = ati;
      }
      for (int i = fore - 1; i >= 0; i--) {
        if (s.charAt(i) == '.') {
          sb.deleteCharAt(i);
        }
      }
      ans.add(sb.toString());
    }
    return ans.size();
  }

  public int repeatedNTimes(int[] A) {
    for (int i = 0; i < A.length; i += 2) {
      if (A[i] == A[i + 1]) {
        return A[i];
      }
    }
    for (int i = 0; i < 2; i++)
      for (int j = 2; j < 4; j++) {
        if (A[i] == A[j])
          return A[i];
      }
    return -1;
  }

  public String toLowerCase(String str) {
    StringBuilder sb = new StringBuilder();
    for (char t : str.toCharArray()) {
      char n = t <= 90 && t >= 65 ? (char) (t + 32) : t;
      sb.append(n);
    }
    return sb.toString();
  }

  public int rangeSumBST(TreeNode root, int L, int R) {
    if (root == null)
      return 0;
    if (root.val >= L && root.val <= R)
      return rangeSumBST(root.left, L, R) + root.val + rangeSumBST(root.right, L, R);
    else if (root.val > R)
      return rangeSumBST(root.left, L, R) + root.val;
    else
      return root.val + rangeSumBST(root.right, L, R);
  }

  public int[] deckRevealedIncreasing(int[] deck) {
    int len = deck.length;
    Queue<Integer> que = new LinkedList<Integer>();
    for (int i = 0; i < len; i++)
      que.offer(i);
    Arrays.sort(deck);
    int[] ans = new int[len];
    for (int i = 0; i < len; i++) {
      ans[que.poll()] = deck[i];
      if (!que.isEmpty()) {
        que.offer(que.poll());
      }
    }
    return ans;
  }

  public int[] sortArrayByParity(int[] A) {
//        Deque<Integer> num = new LinkedList<Integer>();
//        for (int a:A){
//           if (a%2==0)
//               num.offerFirst(a);
//           else
//               num.offerLast(a);
//        }
//        int len = num.size();
//        int[] ans = new int[len];
//        for(int i=0;i<len;i++){
//            ans[i] = num.pollFirst();
//        }
//        return ans;

//        int[] ans = new int[A.length];
//        int startIn=0,lastIn = A.length-1;
//        for (int a:A){
//            if (a%2==0)
//                ans[startIn++] = a;
//            else
//                ans[lastIn--] = a;
//        }
//        return ans;

    int beg = 0, end = A.length - 1;
    while (beg < end) {
      while (beg < end && A[beg] % 2 == 0)
        beg++;
      while (beg < end && A[end] % 2 != 0)
        end--;
      int temp = A[beg];
      A[beg] = A[end];
      A[end] = temp;
    }
    return A;
  }

  public int minDeletionSize(String[] A) {
    int ans = 0;
    boolean isNoD = true;
    int ALen = A.length, strLen = A[0].length();
    for (int i = 0; i < strLen; i++) {
      for (int j = 0; j < ALen - 1; j++) {
        if (A[j].charAt(i) > A[j + 1].charAt(i)) {
          isNoD = false;
          break;
        }
      }
      if (isNoD == false)
        ans++;
      isNoD = true;
    }
    return ans;
  }

  public int[] diStringMatch(String S) {
    int[] ans = new int[S.length() + 1];
    int beg = 0, end = S.length();
    char[] temp = S.toCharArray();
    for (int i = 0; i < S.length(); i++) {
      ans[i] = temp[i] == 'I' ? beg++ : end--;
    }
    ans[ans.length - 1] = beg;
    return ans;
  }

  public TreeNode insertIntoBST(TreeNode root, int val) {
    insertTree(root, val);
    return root;
  }

  public void insertTree(TreeNode root, int val) {
    if (val > root.val) {
      if (root.right == null)
        root.right = new TreeNode(val);
      else
        insertTree(root.right, val);
    } else {
      if (root.left == null)
        root.left = new TreeNode(val);
      else
        insertTree(root.left, val);
    }
  }

  public List<String> findAndReplacePattern(String[] words, String pattern) {
    List<String> ans = new LinkedList<String>();
    Map<Character, Character> cor;
    for (String word : words) {
      if (isCorrespond(word, pattern))
        ans.add(word);
    }
    return ans;
  }

  public boolean isCorrespond(String word, String pattern) {
    if (word.length() != pattern.length())
      return false;
    Map<Character, Character> cor = new HashMap<Character, Character>();
    for (int i = 0; i < pattern.length(); i++) {
      char w = word.charAt(i);
      char p = pattern.charAt(i);
      if (!cor.containsKey(p)) {
        if (cor.containsValue(w))
          return false;
        cor.put(p, w);
      } else if (w != cor.get(p))
        return false;
    }
    return true;
  }

  public int peakIndexInMountainArray(int[] A) {
    int ans = -1;
    for (int i = 0; i < A.length - 1; i++) {
      if (A[i] > A[i + 1]) {
        ans = i;
        break;
      }
    }
    return ans;
  }

  public int arrayPairSum(int[] nums) {
//        int[] sorted = quickSort(nums);
//        int ans=0;
//        for (int i=0;i<nums.length;i+=2){
//            ans += sorted[i];
//        }
//        return ans;
    int[] exist = new int[20001];
    for (int i = 0; i < nums.length; i++) {
      exist[nums[i] + 10000]++;
    }
    int sum = 0;
    boolean odd = true;
    for (int i = 0; i < exist.length; i++) {
      while (exist[i] > 0) {
        if (odd) {
          sum += i - 10000;
        }
        odd = !odd;
        exist[i]--;
      }
    }
    return sum;
  }

  public int[] quickSort(int[] nums) {
    quickSort(nums, 0, nums.length - 1);
    return nums;
  }

  public void quickSort(int[] nums, int start, int end) {
    if (start >= end)
      return;
    int partition = start, beg = start, last = end + 1;
    while (true) {
      while (nums[++beg] <= nums[partition])
        if (beg == end)
          break;

      while (nums[--last] >= nums[partition])
        if (last == start)
          break;

      if (beg >= last)
        break;
      int temp = nums[beg];
      nums[beg] = nums[last];
      nums[last] = temp;
    }
    int tempP = nums[partition];
    nums[partition] = nums[last];
    nums[last] = tempP;
    partition = last;
    quickSort(nums, start, partition - 1);
    quickSort(nums, partition + 1, end);
  }

  private Queue<Integer> recP;

  public void RecentCounter() {
    this.recP = new LinkedList<Integer>();
  }

  public int ping(int t) {
    recP.offer(t);
    int ans = 0;
    int size = recP.size();
    for (int i = 0; i < size; i++) {
      if ((t - recP.peek()) > 3000)
        recP.poll();
      else
        ans++;
    }
    return ans;
  }

  public int minAddToMakeValid(String S) {
//        LinkedList<Character> st = new LinkedList<Character>();
//        char[] chars = S.toCharArray();
//        for (char c:chars){
//            if (c==')' && st.size()>0 &&st.getFirst()=='(')
//                st.removeFirst();
//            else
//                st.addFirst(c);
//        }
//        return st.size();

    if (S.length() == 0) return 0;
    int count = 0, stack = 0;
    for (char c : S.toCharArray()) {
      if (c == '(') stack++;
      else if (stack <= 0) count++;
      else stack--;
    }
    return count + stack;
  }

  public int[] sortArrayByParityII(int[] A) {
    int len = A.length, evSt = 0, odSt = 1;
    int[] ans = new int[len];
    for (int i = 0; i < len; i++) {
      if (A[i] % 2 == 0) {
        ans[evSt] = A[i];
        evSt += 2;
      } else {
        ans[odSt] = A[i];
        odSt += 2;
      }
    }
    return ans;
  }

  public List<Integer> preorder(Node root) {
    List<Integer> ans = new LinkedList<Integer>();
    preorder(root, ans);
    return ans;
  }

  public void preorder(Node node, List l) {
    if (node == null)
      return;
    l.add(node.val);
    for (Node n : node.children) {
      preorder(n, l);
    }
  }

  public List<Integer> postorder(Node root) {
    List<Integer> ans = new LinkedList<Integer>();
    postorder(root, ans);
    return ans;
  }

  public void postorder(Node node, List list) {
    if (node == null)
      return;
    for (Node n : node.children)
      postorder(n, list);
    list.add(node.val);
  }

  public String customSortString(String S, String T) {
    int[] count = new int[26];
    StringBuilder st = new StringBuilder();
    for (char c : T.toCharArray())
      count[c - 'a']++;
    for (char c : S.toCharArray())
      while (count[c - 'a']-- > 0)
        st.append(c);
    for (int i = 0; i < 26; i++)
      while (count[i]-- > 0)
        st.append((char) ('a' + i));
    return st.toString();
  }

  public TreeNode searchBST(TreeNode root, int val) {
    if (root == null)
      return null;
    if (val > root.val)
      return searchBST(root.right, val);
    else if (val < root.val)
      return searchBST(root.left, val);
    else
      return root;
  }

  public int projectionArea(int[][] grid) {
    int x = grid.length, y = grid[0].length, ans = 0, big = 0;
    for (int i = 0; i < x; i++) {
      big = 0;
      for (int j = 0; j < y; j++) {
        if (grid[i][j] > big)
          big = grid[i][j];
        if (grid[i][j] != 0)
          ans++;
      }
      ans += big;
    }

    for (int i = 0; i < y; i++) {
      big = 0;
      for (int j = 0; j < x; j++)
        if (grid[j][i] > big)
          big = grid[j][i];
      ans += big;
    }
    return ans;
  }

  public int smallestRangeI(int[] A, int K) {
    int min = 10000, max = 0;
    for (int a : A) {
      if (a < min)
        min = a;
      if (a > max)
        max = a;
    }
    if ((max - min) > 2 * K)
      return (max - min) - 2 * K;
    else
      return 0;
  }

  public boolean flipEquiv(TreeNode root1, TreeNode root2) {
    if (root1 == null && root2 == null)
      return true;
    else if (root1 == null || root2 == null || root1.val != root2.val)
      return false;
    else
      return ((flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) || (flipEquiv(root1.left, root2.right) && (flipEquiv(root1.right, root2.left))));
  }

  public int[][] transpose(int[][] A) {
    int row = A.length, col = A[0].length;
    int[][] ans = new int[col][row];
    for (int i = 0; i < row; i++)
      for (int j = 0; j < col; j++)
        ans[j][i] = A[i][j];
    return ans;
  }

  public int maxDepth(Node root) {
    if (root == null)
      return 0;
    else {
      int maxD = 0;
      for (Node c : root.children) {
        int depth = maxDepth(c);
        if (depth > maxD)
          maxD = depth;
      }
      return maxD + 1;
    }
  }

  public ListNode middleNode(ListNode head) {
    int index = 1;
    ListNode ans = head;
    while (head.next != null) {
      head = head.next;
      index++;
      if (index % 2 == 0)
        ans = ans.next;
    }
    return ans;
  }

  public List<TreeNode> allPossibleFBT(int N) {
    List<TreeNode> ans = new LinkedList<TreeNode>();
    if (N % 2 == 0)
      return ans;
    ans.add(new TreeNode(0));
    if (N == 1) {
      return ans;
    }
    Map<Integer, List<TreeNode>> dp = new HashMap<Integer, List<TreeNode>>();
    dp.put(1, ans);
    for (int i = 3; i <= N; i += 2) {
      List<TreeNode> ls = new LinkedList<TreeNode>();
      for (int j = 1; j <= i - 2; j += 2) {
        List<TreeNode> left = dp.get(j);
        List<TreeNode> right = dp.get(i - 1 - j);
        for (int k = 0; k < left.size(); k++)
          for (int l = 0; l < right.size(); l++) {
            TreeNode rt = new TreeNode(0);
            rt.left = left.get(k);
            rt.right = right.get(l);
            ls.add(rt);
          }
      }
      dp.put(i, ls);
    }
    return dp.get(N);
  }

  public int[][] spiralMatrixIII(int R, int C, int r0, int c0) {
    int all = R * C;
    int[][] ans = new int[all][2];
    int[][] dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int index = 1, distant = 1, sp = 1;
    int[] loc = {r0, c0};
    ans[0][0] = r0;
    ans[0][1] = c0;
    while (index < all) {
      int[] move = dir[(sp - 1) % 4];
      int dx = move[0] * distant;
      int dy = move[1] * distant;

      if (dx != 0 && loc[1] >= 0 && loc[1] < C) {
        for (int i = 1; i <= distant; i++) {
          int loct = loc[0] + move[0] * i;
          if (loct >= 0 && loct < R) {
            ans[index][0] = loct;
            ans[index++][1] = loc[1];
          }
        }
      } else if (dy != 0 && loc[0] >= 0 && loc[0] < R) {
        for (int i = 1; i <= distant; i++) {
          int loct = loc[1] + move[1] * i;
          if (loct >= 0 && loct < C) {
            ans[index][0] = loc[0];
            ans[index++][1] = loct;
          }
        }
      }

      loc[0] += dx;
      loc[1] += dy;
      if ((sp++) % 2 == 0)
        distant++;
    }
    return ans;
  }

  public int matrixScore(int[][] A) {
//        int N = A[0].length, M = A.length;
//        for(int[] a : A){
//            if(a[0] == 0){
//                for(int i = 0 ; i < N ; i++){
//                    a[i] ^= 1;
//                }
//            }
//        }
//        int res = 0;
//        for(int j = 1 ; j < N ; j++){
//            int count = 0;
//            for(int i = 0 ; i < M ; i++){
//                count += A[i][j];
//            }
//            res += Math.max(count, M - count) * (1 << (N - j - 1));
//        }
//        return res + (1 << N - 1) * M;
    int R = A.length, C = A[0].length;
    for (int k = 0; k < R; k++)
      if (A[k][0] == 0)
        toggleRow(A, k);

    for (int i = 1; i < C; i++) {
      int zeros = 0, ones = 0;
      for (int j = 0; j < R; j++) {
        if (A[j][i] == 0)
          zeros++;
        else
          ones++;
      }
      if (zeros > ones)
        toggleCol(A, i);
    }

    int ans = 0;
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        ans += A[i][j] * (1 << (C - j - 1));
      }
    }
    return ans;
  }

  public void toggleCol(int[][] a, int col) {
    for (int i = 0; i < a.length; i++)
      a[i][col] ^= 1;
  }

  public void toggleRow(int[][] a, int row) {
    for (int i = 0; i < a[0].length; i++)
      a[row][i] ^= 1;
  }

  public List<String> subdomainVisits(String[] cpdomains) {
    List<String> ans = new LinkedList<String>();
    Map<String, Integer> mp = new HashMap<String, Integer>();
    for (String cpd : cpdomains) {
      String[] pieces = cpd.split("\\s+");
      int count = Integer.parseInt(pieces[0]);
      String[] frags = pieces[1].split("\\.");
      StringBuilder sb = new StringBuilder();
      for (int i = frags.length - 1; i >= 0; i--) {
        sb.insert(0, frags[i]);
        String temp = sb.toString();
        mp.put(temp, count + mp.getOrDefault(temp, 0));
        sb.insert(0, '.');
      }
    }
    for (String dm : mp.keySet()) {
      int c = mp.get(dm);
      ans.add(c + " " + dm);
    }
    return ans;
  }

  public String reverseWords(String s) {
    String[] frags = s.split("\\s");
    StringBuilder sb = new StringBuilder();
    char temp;
    for (int j = 0; j < frags.length; j++) {
      String word = frags[j];
      char[] cs = word.toCharArray();
      for (int i = 0; i < cs.length / 2; i++) {
        temp = cs[i];
        cs[i] = cs[cs.length - i - 1];
        cs[cs.length - i - 1] = temp;
      }
      sb.append(cs);
      sb.append(' ');
    }
    return sb.toString().trim();
  }

  public String reverseString(String s) {
    char[] cs = s.toCharArray();
    int start = 0, end = cs.length - 1;
    char temp;
    while (start < end) {
      temp = cs[start];
      cs[start] = cs[end];
      cs[end] = temp;
      start++;
      end--;
    }
    return String.valueOf(cs);
  }

  public int[] shortestToChar(String S, char C) {
    char[] cs = S.toCharArray();
    int len = cs.length;
    int[] ans = new int[len];
    int prev = Integer.MIN_VALUE / 2;
    for (int i = 0; i < len; i++) {
      if (cs[i] == C)
        prev = i;
      else
        ans[i] = i - prev;
    }
    prev = Integer.MAX_VALUE;
    for (int j = len - 1; j >= 0; j--) {
      if (cs[j] == C)
        prev = j;
      else {
        int distant = prev - j;
        ans[j] = distant > ans[j] ? ans[j] : distant;
      }
    }
    return ans;
  }

  public int[] numberOfLines(int[] widths, String S) {
    int lines = S.length() > 0 ? 1 : 0;
    int lastLine = 0;
    char[] cs = S.toCharArray();
    for (char c : cs) {
      int units = widths[c - 'a'];
      if (lastLine + units > 100) {
        lastLine = units;
        lines++;
      } else
        lastLine += units;
    }
    return new int[]{lines, lastLine};
  }

  public int[] id;
  public int count;

  public int regionsBySlashes(String[] grid) {
    int len = grid.length;
    count = 4 * len * len;
    id = new int[4 * len * len];
    for (int i = 0; i < len * len * 4; i++)
      id[i] = i;
    for (int i = 0; i < len; i++)
      for (int j = 0; j < len; j++) {
        if (i > 0)
          union(4 * i * len + j * 4, 4 * (i - 1) * len + j * 4 + 2);
        if (j > 0)
          union(4 * i * len + j * 4 + 3, 4 * i * len + (j - 1) * 4 + 1);
        unionUnit(grid, i, j);
      }
    return count;
  }

  public void unionUnit(String[] grid, int row, int col) {
    int prevNum = 4 * row * grid.length + col * 4;
    if (grid[row].charAt(col) == '/') {
      union(prevNum, prevNum + 3);
      union(prevNum + 1, prevNum + 2);
    } else if (grid[row].charAt(col) == '\\') {
      union(prevNum, prevNum + 1);
      union(prevNum + 2, prevNum + 3);
    } else
      for (int i = 1; i < 4; i++)
        union(prevNum + i, prevNum + i - 1);
  }

  public int find(int p) {
    return id[p];
  }

  public void union(int p, int q) {
    int idP = id[p];
    int idQ = id[q];
    if (idP == idQ)
      return;
    for (int i = 0; i < id.length; i++)
      if (id[i] == idQ)
        id[i] = idP;
    count--;
  }

  public boolean isUnivalTree(TreeNode root) {
    int val = root.val;
    if (isUnivalTree(root.left, val) == false)
      return false;
    return isUnivalTree(root.right, val);
  }

  public boolean isUnivalTree(TreeNode root, int val) {
    if (root == null)
      return true;
    if (root.val != val)
      return false;
    if (isUnivalTree(root.left, val) == false)
      return false;
    return isUnivalTree(root.right, val);
  }

  public boolean leafSimilar(TreeNode root1, TreeNode root2) {
    List<Integer> seq1 = new LinkedList<Integer>();
    List<Integer> seq2 = new LinkedList<Integer>();
    getLeafSequence(root1, seq1);
    getLeafSequence(root2, seq2);
    if (seq1.size() != seq2.size())
      return false;
    else {
      for (int i = 0; i < seq1.size(); i++) {
        if (seq1.get(i) != seq2.get(i))
          return false;
      }
      return true;
    }
  }

  public void getLeafSequence(TreeNode root, List<Integer> seq) {
    if (root == null)
      return;
    getLeafSequence(root.left, seq);
    if (root.left == null && root.right == null)
      seq.add(root.val);
    getLeafSequence(root.right, seq);
  }

  public int findComplement(int num) {
    int len = 0;
    int num1 = num;
    while (num != 0) {
      num /= 2;
      len++;
    }
    int temp = Integer.MAX_VALUE;
    return temp >> (31 - len) ^ num1;
  }

  public String[] findWords(String[] words) {
    int[] alToLine = new int[]{1, 2, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 2};
    List<String> ans = new LinkedList<String>();
    for (String w : words) {
      char[] cs = w.toLowerCase().toCharArray();
      int firstA = alToLine[cs[0] - 'a'];
      boolean isOneL = true;
      for (int i = 1; i < cs.length; i++)
        if (alToLine[cs[i] - 'a'] != firstA) {
          isOneL = false;
          break;
        }
      if (isOneL)
        ans.add(w);
    }
    return ans.toArray(new String[ans.size()]);
  }

  public int numSpecialEquivGroups(String[] A) {
    Set<String> set = new HashSet<String>();
    for (String s : A) {
      char[] cs = s.toCharArray();
      int[][] alp = new int[2][26];
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < cs.length; i++)
        alp[i % 2][cs[i] - 'a']++;

      for (int[] line : alp)
        for (int j = 0; j < line.length; j++)
          while ((line[j]--) > 0)
            sb.append((char) (j + 'a'));

      set.add(sb.toString());
    }
    return set.size();
  }

  public TreeNode ans = new TreeNode(0);

  public TreeNode increasingBST(TreeNode root) {
    //TreeNode ans = new TreeNode(0);
    TreeNode temp = ans;
    increasingBST1(root, ans);

    return temp;
  }

  public void increasingBST1(TreeNode root, TreeNode ans) {
    if (root == null)
      return;
    increasingBST1(root.left, ans);
    ans.right = root;
    ans = ans.right;
    ans.left = null;
    increasingBST1(root.right, ans);
  }

  //dfs
  boolean[] markedDFS;
  int countDFS;

  public int removeStones1(int[][] stones) {
    int stonesNum = stones.length;
    markedDFS = new boolean[stonesNum];
    countDFS = 0;
    for (int i = 0; i < stonesNum; i++)
      if (!markedDFS[i]) {
        rsDFS(stones, stones[i], i);
        countDFS++;
      }
    return stonesNum - countDFS;
  }

  public void rsDFS(int[][] stones, int[] stone, int i) {
    markedDFS[i] = true;
    for (int j = 0; j < markedDFS.length; j++)
      if (!markedDFS[j] && isConnected(stones[j], stone))
        rsDFS(stones, stones[j], j);
  }

  public boolean isConnected(int[] st1, int[] st2) {
    if (st1[0] == st2[0] || st1[1] == st2[1])
      return true;
    return false;
  }

  int countUF;
  int[] idUF;
  int[] sizeUF;

  public int removeStones(int[][] stones) {
    int len = stones.length;
    countUF = len;
    sizeUF = new int[len];
    idUF = new int[len];
    for (int i = 0; i < len; i++) {
      idUF[i] = i;
      sizeUF[i] = 1;
    }

    for (int i = 0; i < len; i++)
      for (int j = i + 1; j < len; j++)
        if (isConnected(stones[i], stones[j]))
          unionUF(i, j);

    return len - countUF;
  }

  public int findUF(int i) {
    int temp = i;
    while (idUF[i] != i)
      i = idUF[i];

    while (temp != i) {
      int tp = idUF[temp];
      idUF[temp] = i;
      temp = tp;
    }

    return i;
  }

  public void unionUF(int p, int q) {
    int idP = findUF(p);
    int idQ = findUF(q);
    if (idP == idQ)
      return;
    if (sizeUF[idP] >= sizeUF[idQ]) {
      idUF[idQ] = idP;
      sizeUF[idP] += sizeUF[idQ];
    } else {
      idUF[idP] = idQ;
      sizeUF[idQ] += sizeUF[idP];
    }
    countUF--;
  }

  public boolean isToeplitzMatrix(int[][] matrix) {
    int[] temp = new int[matrix[0].length - 1];
    for (int t = 0; t < temp.length; t++)
      temp[t] = matrix[0][t];
    for (int i = 1; i < matrix.length; i++) {
      for (int j = 1; j < matrix[0].length; j++) {
        if (temp[j - 1] != matrix[i][j])
          return false;
        else
          temp[j - 1] = matrix[i][j - 1];
      }
    }
    return true;
  }

  public String[] uncommonFromSentences(String A, String B) {
    Map<String, Integer> wordCount = new HashMap<String, Integer>();
    String[] Aw = A.split(" ");
    String[] Bw = B.split(" ");
    for (String aw : Aw)
      wordCount.put(aw, wordCount.getOrDefault(aw, 0) + 1);
    for (String bw : Bw)
      wordCount.put(bw, wordCount.getOrDefault(bw, 0) + 1);
    List<String> ls = new LinkedList<String>();
    Iterator<Map.Entry<String, Integer>> it = wordCount.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<String, Integer> entry = it.next();
      if (entry.getValue() == 1)
        ls.add(entry.getKey());
    }
    return ls.toArray(new String[0]);
  }

  public List<Integer> pancakeSort(int[] A) {
    List<Integer> ans = new LinkedList<Integer>();
    for (int i = A.length - 1; i >= 0; i--) {
      int max = 0;
      for (int j = 0; j <= i; j++) {
        if (A[j] > A[max])
          max = j;
      }
      if (max != i) {
        ans.add(max + 1);
        reverse(A, max + 1);
        ans.add(i + 1);
        reverse(A, i + 1);
      }
    }
    return ans;
  }

  public void reverse(int[] A, int k) {
    int beg = 0, end = k - 1;
    while (beg < end) {
      int temp = A[beg];
      A[beg] = A[end];
      A[end] = temp;
      beg++;
      end--;
    }
  }

  public boolean stoneGame(int[] piles) {
    int n = piles.length;
    int[][] dp = new int[n][n];
    for (int i = 0; i < n; i++)
      dp[i][i] = piles[i];
    for (int d = 1; d < n; d++)
      for (int i = 0; i < n - d; i++)
        dp[i][i + d] = Math.max(piles[i] - dp[i + 1][i + d], piles[i + d] - dp[i][i + d - 1]);
    return dp[0][n - 1] > 0;
  }

  public int calPoints(String[] ops) {
    Stack<Integer> ans = new Stack<Integer>();
    for (String op : ops) {
      if (op.equals("C"))
        ans.pop();
      else if (op.equals("D"))
        ans.push(ans.peek() * 2);
      else if (op.equals("+")) {
        int temp = ans.pop();
        int add = temp + ans.peek();
        ans.push(temp);
        ans.push(add);
      } else
        ans.push(Integer.valueOf(op));
    }

    Iterator<Integer> res = ans.iterator();
    int a = 0;
    while (res.hasNext()) {
      a += res.next();
    }
    return a;
  }

  public int islandPerimeter(int[][] grid) {
    int nodeNum = 0, edgeNum = 0;
    for (int i = 0; i < grid.length; i++)
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] == 0)
          continue;
        nodeNum++;

        edgeNum = i - 1 >= 0 && grid[i - 1][j] == 1 ? edgeNum + 1 : edgeNum;
        edgeNum = i + 1 < grid.length && grid[i + 1][j] == 1 ? edgeNum + 1 : edgeNum;
        edgeNum = j - 1 >= 0 && grid[i][j - 1] == 1 ? edgeNum + 1 : edgeNum;
        edgeNum = j + 1 < grid[0].length && grid[i][j + 1] == 1 ? edgeNum + 1 : edgeNum;
      }
    return 4 * nodeNum - edgeNum;
  }

  public TreeNode trimBST(TreeNode root, int L, int R) {
    if (root == null)
      return null;
    if (root.val < L)
      return trimBST(root.right, L, R);
    else if (root.val > R)
      return trimBST(root.left, L, R);
    else {
      root.left = trimBST(root.left, L, R);
      root.right = trimBST(root.right, L, R);
      return root;
    }
  }

  public TreeNode min(TreeNode root) {
    if (root.left == null)
      return root;
    return min(root.left);
  }

  public TreeNode deleteMin(TreeNode root) {
    if (root.left == null)
      return root.right;
    root.left = deleteMin(root.left);
    return root;
  }

  public List<Integer> findDuplicates(int[] nums) {
    List<Integer> ans = new LinkedList<Integer>();
    for (int i = 0; i < nums.length; i++) {
      int index = Math.abs(nums[i]) - 1;
      if (nums[index] < 0)
        ans.add(index + 1);
      else
        nums[index] *= -1;
    }
    return ans;
  }

  public int fib(int N) {
//    if (N == 0)
//      return 0;
//    if (N ==1)
//      return 1;
//    return fib(N-1)+fib(N-2);
    if (N <= 1)
      return N;
    int[] res = new int[N + 1];
    res[1] = 1;
    for (int i = 2; i <= N; i++) {
      res[i] = res[i - 1] + res[i - 2];
    }
    return res[N];
  }

  public int binaryGap(int N) {
    char[] bi = Integer.toBinaryString(N).toCharArray();
    int max = 0, start = -1, end;
    for (int i = 0; i < bi.length; i++) {
      if (bi[i] == '0')
        continue;
      if (start == -1) {
        start = i;
        continue;
      }
      end = i;
      max = end - start > max ? end - start : max;
      start = i;
    }
    return max;
//    int res = 0;
//    int last = -1;
//    for (int i = 0; i < 32; i++) {
//      if (((N >> i) & 1) > 0) {
//        if (last >= 0) {
//          res = Math.max(i-last, res);
//        }
//        last = i;
//      }
//    }
//    return res;
  }

  boolean[] CVRMarked;
  int CVRCount;

  public boolean canVisitAllRooms1(List<List<Integer>> rooms) {
    CVRCount = 0;
    int n = rooms.size();
    CVRMarked = new boolean[n];
    for (int i = 0; i < n; i++)
      if (!CVRMarked[i]) {
        CVRdfs(rooms, i);
        CVRCount++;
      }
    return CVRCount == 1;
  }

  public void CVRdfs(List<List<Integer>> rooms, int i) {
    CVRMarked[i] = true;
    for (int room : rooms.get(i))
      if (!CVRMarked[room])
        CVRdfs(rooms, room);
  }

  int[] CVARid;
  int[] CVARsize;
  int CVARcount;

  public boolean canVisitAllRooms(List<List<Integer>> rooms) {
    int n = rooms.size();
    CVARid = new int[n];
    CVARsize = new int[n];
    CVARcount = n;
    for (int i = 0; i < n; i++) {
      CVARsize[i] = 1;
      CVARid[i] = i;
    }

    for (int i = 0; i < n; i++)
      for (int room : rooms.get(i))
        CVARunion(i, room);
    return CVARcount == 1;
  }

  public int CVARfind(int i) {
    int temp = i;
    if (CVARid[i] != i)
      i = CVARid[i];

    while (temp != i) {
      int next = CVARid[temp];
      CVARid[temp] = i;
      temp = next;
    }
    return i;
  }

  public void CVARunion(int p, int q) {
    int idP = CVARfind(p);
    int idQ = CVARfind(q);
    if (idP == idQ)
      return;
    if (CVARsize[idP] >= CVARsize[idQ]) {
      CVARid[idQ] = idP;
      CVARsize[idP] += CVARsize[idQ];
    } else {
      CVARid[idP] = idQ;
      CVARsize[idQ] += CVARsize[idP];
    }
    CVARcount--;
  }

  public int distributeCandies(int[] candies) {
//    Map<Integer,Integer> kinds = new HashMap<Integer, Integer>();
//    int n = candies.length;
//    for (int i=0;i<n;i++)
//      kinds.put(candies[i],kinds.getOrDefault(candies[i],0)+1);
    int kinds = 0, n = candies.length;
    int[] ks = new int[n];
    for (int i = 0; i < n; i++)
      ks[candies[i]] += 1;
    for (int j = 0; j < n; j++)
      if (ks[j] != 0)
        kinds++;
    return Math.min(n / 2, kinds);
  }

  public List<String> fizzBuzz(int n) {
    List<String> ans = new LinkedList<String>();
    StringBuilder sb = new StringBuilder();
    for (int i = 1; i <= n; i++) {
      sb.delete(0, sb.length());
      if (i % 3 != 0 && i % 5 != 0)
        sb.append(i);
      if (i % 3 == 0)
        sb.append("Fizz");
      if (i % 5 == 0)
        sb.append("Buzz");
      ans.add(sb.toString());
    }
    return ans;
  }

  public int maxDepth(TreeNode root) {
    return root == null ? 0 : Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
  }

  public int singleNumber(int[] nums) {
    int ans = 0;
    for (int i : nums)
      ans ^= i;
    return ans;
  }

  public int[][] reconstructQueue(int[][] people) {
    if (people.length == 0)
      return people;
    List<int[]> res = new LinkedList<int[]>();
    Arrays.sort(people, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        if (o1[0] != o2[0])
          return o1[0] - o2[0];
        else
          return o2[1] - o1[1];
      }
    });

    for (int n = people.length - 1; n >= 0; n--)
      res.add(people[n][1], people[n]);

    return res.toArray(new int[people.length][people[0].length]);
  }

  public boolean validateStackSequences(int[] pushed, int[] popped) {
    Stack<Integer> ans = new Stack<Integer>();
    int i = 0;
    for (int pu : pushed) {
      ans.push(pu);
      while (ans.size() != 0 && i <= popped.length && ans.peek() == popped[i]) {
        ans.pop();
        i++;
      }
    }
    return ans.size() == 0;
  }

  public int[][] kClosest(int[][] points, int K) {
    Map<Double, int[]> res = new TreeMap<Double, int[]>();
    for (int[] p : points)
      res.put(Math.pow(p[0], 2) + Math.pow(p[1], 2), p);
    int[][] ans = new int[K][2];
    Iterator<Map.Entry<Double, int[]>> kv = res.entrySet().iterator();
    for (int i = 0; i < K; i++)
      ans[i] = kv.next().getValue();
    return ans;
  }

  public int[][] matrixReshape(int[][] nums, int r, int c) {
//    if (r*c>nums.length*nums[0].length)
//      return nums;
//    int[][] ans = new int[r][c];
//    int[] temp = new int[nums.length*nums[0].length];
//    int index = 0;
//    for (int i=0;i<nums.length;i++)
//      for (int j=0;j<nums[0].length;j++)
//        temp[index++] = nums[i][j];
//    index = 0;
//    for (int i=0;i<r;i++)
//      for (int j=0;j<c;j++)
//        ans[i][j] = temp[index++];
//    return ans;
    int n = nums.length, m = nums[0].length;
    if (r * c > n * m)
      return nums;
    int len = n * m;
    int[][] ans = new int[r][c];
    for (int i = 0; i < len; i++)
      ans[i / c][i % c] = nums[i / m][i % m];
    return ans;
  }

  public int findBottomLeftValue(TreeNode root) {
    Map<Integer, TreeNode> leefs = new HashMap<>();
    recordDepth(leefs, root, 1);
    Iterator<Map.Entry<Integer, TreeNode>> et = leefs.entrySet().iterator();
    int maxD = Integer.MIN_VALUE, value = -1;
    while (et.hasNext()) {
      Map.Entry<Integer, TreeNode> temp = et.next();
      if (temp.getKey() >= maxD) {
        maxD = temp.getKey();
        value = temp.getValue().val;
      }
    }
    return value;
  }

  public void recordDepth(Map<Integer, TreeNode> leefs, TreeNode root, int depth) {
    if (root == null)
      return;
    recordDepth(leefs, root.left, depth + 1);
    if (root.right == null && root.left == null)
      leefs.putIfAbsent(depth, root);
    recordDepth(leefs, root.right, depth + 1);
  }

  public int[] dailyTemperatures(int[] T) {
//    int n=T.length;
//    int[] ans = new int[n];
//    for (int i=0;i<n;i++)
//      for (int j=i+1;j<n;j++)
//        if(T[j]>T[i]){
//          ans[i] = j-i;
//          break;
//        }
//
//    return ans;
    int n = T.length;
    Stack<Integer> st = new Stack<>();
    int[] ans = new int[n];
    for (int i = 0; i < n; i++) {
      while (!st.isEmpty() && T[i] > T[st.peek()]) {
        int ind = st.pop();
        ans[ind] = i - ind;
      }
      st.push(i);
    }
    return ans;
  }

  public int[] nextGreaterElement(int[] nums1, int[] nums2) {
    Map<Integer, Integer> nb = new HashMap<>();
    //Stack<Integer> st = new Stack<>();
    int[] stack = new int[1000];
    int index = 0;
    int n1 = nums1.length, n2 = nums2.length;
    for (int i = 0; i < n2; i++) {
      while (index > 0 && stack[index - 1] < nums2[i]) {
        int num = stack[(index--) - 1];
        nb.putIfAbsent(num, nums2[i]);
      }
      stack[(index++)] = nums2[i];
    }
    int[] ans = new int[n1];
    for (int i = 0; i < n1; i++)
      ans[i] = nb.getOrDefault(nums1[i], -1);
    return ans;
  }

  public List<List<Integer>> levelOrder(Node root) {
    List<List<Integer>> ans = new LinkedList<>();
    if (root != null)
      loIte(ans, root, 1);
    return ans;
  }

  public void loIte(List<List<Integer>> ls, Node root, int depth) {
    if (ls.size() < depth)
      ls.add(new LinkedList<>());
    ls.get(depth - 1).add(root.val);
    for (Node n : root.children)
      loIte(ls, n, depth + 1);
  }

  public String[] reorderLogFiles(String[] logs) {
    Arrays.sort(logs, new Comparator<String>() {
      @Override
      public int compare(String o1, String o2) {
        String[] o11 = o1.split(" ", 2);
        String[] o22 = o2.split(" ", 2);
        char id1 = o11[1].charAt(0);
        char id2 = o22[1].charAt(0);
        if (id1 >= 48 && id1 <= 57 && id2 >= 48 && id2 <= 57)
          return 0;
        else if (id1 >= 97 && id1 <= 122 && id2 >= 97 && id2 <= 122)
          return o11[1].compareTo(o22[1]);
        else if (id1 >= 48 && id1 <= 57)
          return 1;
        else
          return -1;
      }
    });
    return logs;
  }

  public List<Double> averageOfLevels1(TreeNode root) {
    List<Double[]> res = new LinkedList<>();
    AOLIte(res, root, 1);
    List<Double> ans = new LinkedList<>();
    Iterator<Double[]> ite = res.iterator();
    while (ite.hasNext()) {
      Double[] temp = ite.next();
      ans.add(temp[0] / temp[1]);
    }
    return ans;
  }

  public void AOLIte(List<Double[]> ls, TreeNode root, int depth) {
    if (root == null)
      return;
    if (ls.size() < depth) {
      Double[] temp = new Double[2];
      temp[0] = 0.0;
      temp[1] = 0.0;
      ls.add(temp);
    }
    ls.get(depth - 1)[0] += root.val;
    ls.get(depth - 1)[1]++;
    AOLIte(ls, root.left, depth + 1);
    AOLIte(ls, root.right, depth + 1);
  }

  public List<Double> averageOfLevels(TreeNode root) {
    List<Double> ls = new LinkedList<>();
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      double sum = 0, count = 0;
      int n = q.size();
      for (int i = 0; i < n; i++) {
        TreeNode temp = q.poll();
        sum += temp.val;
        count++;
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      ls.add(sum / count);
    }
    return ls;
  }

  public boolean hasAlternatingBits(int n) {
//    char[] res = Integer.toBinaryString(n).toCharArray();
//    for (int i=1;i<res.length;i++)
//      if (res[i]==res[i-1])
//        return false;
//    return true;
    int last = n & 1, remain = n >> 1;
    while (remain > 0) {
      int nextL = remain & 1;
      if (last == nextL)
        return false;
      last = nextL;
      remain >>= 1;
    }
    return true;
  }

  public int countPrimeSetBits(int L, int R) {
    int count = 0;
    for (int i = L; i <= R; i++)
      if (isPrime32(countSetBits(i)))
        count++;
    return count;
  }

  public int countSetBits(int n) {
    int last, remain = n, count = 0;
    while (remain > 0) {
      last = remain & 1;
      remain >>= 1;
      if (last == 1)
        count++;
    }
    return count;
  }

  public boolean isPrime32(int n) {
    return (n == 2 || n == 3 || n == 5 || n == 7 || n == 11 || n == 13 || n == 17 || n == 19 || n == 23 || n == 29 || n == 31);
  }

  public boolean isPrime0(int n) {
    if (n <= 1)
      return false;
    for (int i = 2; i < n; i++)
      if (n % i == 0)
        return false;
    return true;
  }

  public int[] getPrimeArrays(int len) {
    int[] ans = new int[len];
    ans[0] = -1;
    ans[1] = -1;
    for (int i = 2; i < len; i++)
      if (ans[i] == 0) {
        ans[i] = 1;
        for (int j = 2; j * i < len; j++)
          ans[i * j] = -1;
      }
    return ans;
  }

  public TreeNode constructFromPrePost1(int[] pre, int[] post) {
    return CFPP(pre, post, 0, pre.length, 0, post.length);
  }

  public TreeNode CFPP(int[] pre, int[] post, int preS, int preE, int postS, int postE) {
    if (preE <= preS)
      return null;
    TreeNode root = new TreeNode(pre[preS]);
    if (preE - preS == 1)
      return root;
    int divider = 0;
    for (int i = preS + 1; i < preE; i++)
      if (pre[i] == post[postE - 2]) {
        divider = i;
        break;
      }

    int leftL = divider - preS - 1;
    int rightL = preE - divider;
    root.left = CFPP(pre, post, preS + 1, preS + leftL + 1, postS, postS + leftL);
    root.right = CFPP(pre, post, preS + leftL + 1, preE, postS + leftL, postE - 1);
    return root;
  }

  public List<Integer> largestValues1(TreeNode root) {
    List<Integer> ans = new LinkedList<>();
    LVdfs(ans, root, 1);
    return ans;
  }

  public void LVdfs(List<Integer> ls, TreeNode root, int depth) {
    if (root == null)
      return;
    if (ls.size() < depth)
      ls.add(Integer.MIN_VALUE);
    if (ls.get(depth - 1) < root.val)
      ls.set(depth - 1, root.val);
    LVdfs(ls, root.left, depth + 1);
    LVdfs(ls, root.right, depth + 1);
  }

  public List<Integer> largestValues(TreeNode root) {
    List<Integer> ans = new LinkedList<>();
    if (root == null)
      return ans;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      int n = q.size(), max = Integer.MIN_VALUE;
      for (int i = 0; i < n; i++) {
        TreeNode temp = q.poll();
        if (temp.val > max)
          max = temp.val;
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      ans.add(max);
    }
    return ans;
  }

  public int singleNonDuplicate(int[] nums) {
    int a = 0;
    for (int i : nums)
      a ^= i;
    return a;
  }

  public TreeNode constructFromPrePost(int[] pre, int[] post) {
    Deque<TreeNode> s = new ArrayDeque<>();
    s.offer(new TreeNode(pre[0]));
    for (int i = 1, j = 0; i < pre.length; ++i) {
      TreeNode node = new TreeNode(pre[i]);
      while (s.getLast().val == post[j]) {
        s.pollLast();
        j++;
      }
      if (s.getLast().left == null) s.getLast().left = node;
      else s.getLast().right = node;
      s.offer(node);
    }
    return s.getFirst();
  }

  public List<Integer> JZOfindUnique(int[] array) {
//    List<Integer> ans = new LinkedList<>();
//    Map<Integer,Integer> mp = new HashMap<>();
//    for (int a:array)
//      mp.put(a,mp.getOrDefault(a,0)+1);
//    Iterator<Map.Entry<Integer,Integer>> it = mp.entrySet().iterator();
//    while(it.hasNext()){
//      Map.Entry<Integer,Integer> temp = it.next();
//      if (temp.getValue()==2)
//        ans.add(temp.getKey());
//    }
//    return ans;

//    Set<Integer> ans = new HashSet<>();
//    for (int i=0;i<array.length;i++){
//      array[Math.abs(array[i])] *= -1;
//    }
//    Set<Integer> st = new HashSet<>();
//    for (int i=0;i<array.length;i++)
//      if (array[i]<0)
//        st.add(i);
//    for (int i:array)
//      if (!st.contains(Math.abs(i)))
//        ans.add(Math.abs(i));
//    List<Integer> rs = new LinkedList<>();
//    rs.addAll(ans);
//    return rs;

    List<Integer> ls = new LinkedList<>();
    for (int i = 0; i < array.length; ) {
      if (array[i] == i) {
        i++;
        continue;
      }
      int changed = array[array[i]];
      if (changed == array[i]) {
        ls.add(array[i]);
        i++;
        continue;
      } else {
        array[array[i]] = array[i];
        array[i] = changed;
      }
    }
    return ls;
  }

  public boolean JZOfindIn2DMatrix(int[][] a, int k) {
    int r = a.length, c = a[0].length;
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++) {
        int cur = a[i][j];
        if (cur == k)
          return true;
        else if (cur > k)
          c = i;
      }
    return false;
  }

  public String JZOreplaceSpace(String a, String re) {
    StringBuilder sb = new StringBuilder();
    char[] cs = a.toCharArray();
    for (int i = 0; i < cs.length; i++)
      if (cs[i] == ' ')
        sb.append(re);
      else
        sb.append(cs[i]);
    return sb.toString();
  }

  public TreeNode JZOreconstructTreeByPreMid(int[] pre, int[] mid) {
    TreeNode root = JZOpmhelper(pre, mid, 0, pre.length - 1, 0, mid.length - 1);
    return root;
  }

  public TreeNode JZOpmhelper(int[] pre, int[] mid, int preS, int preE, int midS, int midE) {
    if (preE <= preS)
      return null;
    if (preE - preS == 1)
      return new TreeNode(pre[preS]);
    TreeNode root = new TreeNode(pre[preS]);
    int rootIndexInMid = 0;
    for (int i = midS; i <= midE; i++)
      if (mid[i] == root.val) {
        rootIndexInMid = i;
        break;
      }
    int leftLen = rootIndexInMid - midS;
    int rightLen = midE - rootIndexInMid;
    root.left = JZOpmhelper(pre, mid, preS + 1, preS + leftLen, midS, midS + leftLen - 1);
    root.right = JZOpmhelper(pre, mid, preS + leftLen + 1, preE, midS + leftLen + 1, midE);
    return root;
  }

  boolean[][] PIMmarked;
  int PIMindex;
  int sLen;
  String S;

  public boolean JZOpathInMatrix(char[][] c, String s) {
    PIMmarked = new boolean[c.length][c[0].length];
    sLen = s.length();
    S = s;
    for (int i = 0; i < c.length; i++)
      for (int j = 0; j < c[0].length; j++)
        if (!PIMmarked[i][j] && c[i][j] == s.charAt(PIMindex)) {
          PIMdfs(c, i, j);
          if (PIMindex == sLen - 1)
            return true;
          PIMindex = 0;
        } else
          PIMmarked[i][j] = true;
    return false;
  }

  public void PIMdfs(char[][] C, int r, int c) {
    if (PIMindex == sLen - 1)
      return;
    PIMmarked[r][c] = true;
    PIMindex++;
    if (r - 1 >= 0 && !PIMmarked[r - 1][c] && C[r - 1][c] == S.charAt(PIMindex))
      PIMdfs(C, r - 1, c);
    if (r + 1 < C.length && !PIMmarked[r + 1][c] && C[r + 1][c] == S.charAt(PIMindex))
      PIMdfs(C, r + 1, c);
    if (c - 1 >= 0 && !PIMmarked[r][c - 1] && C[r][c - 1] == S.charAt(PIMindex))
      PIMdfs(C, r, c - 1);
    if (c + 1 < C.length && !PIMmarked[r][c + 1] && C[r][c + 1] == S.charAt(PIMindex))
      PIMdfs(C, r, c + 1);
  }

  public int JZOgetBiZeroNum(int n) {
//    int last,remain = n,ans=0;
//    while(remain>0){
//      last = remain&1;
//      remain>>>=1;
//      ans += last;
//    }
//    return ans;

//    int detector = 1,count = 0;
//    while(detector>0){
//      if ((detector & n)>0)
//        count++;
//      detector<<=1;
//    }
//    return count;
    int count = 0;
    while (n > 0) {
      count++;
      n = (n - 1) & n;
    }
    return count;
  }

  public int minFallingPathSum(int[][] A) {
    int r = A.length, c = A[0].length;
    int[][] res = new int[r][c];
    for (int i = 0; i < c; i++)
      res[r - 1][i] = A[r - 1][i];

    for (int i = r - 2; i >= 0; i--)
      for (int j = c - 1; j >= 0; j--) {
        int nextS = j - 1 >= 0 ? j - 1 : j;
        int nextE = j + 1 == c ? j : j + 1;
        int min = Integer.MAX_VALUE;
        for (int x = nextS; x <= nextE; x++)
          if (res[i + 1][x] < min)
            min = res[i + 1][x];
        res[i][j] = A[i][j] + min;
      }
    int ans = Integer.MAX_VALUE;
    for (int i = 0; i < c; i++)
      if (res[0][i] < ans)
        ans = res[0][i];
    return ans;
  }

  public int[] singleNumber1(int[] nums) {
    int dif = 0;
    for (int n : nums)
      dif ^= n;
    dif &= -dif;
    int[] ans = new int[2];
    for (int n : nums)
      if ((n & dif) == 0)
        ans[0] ^= n;
      else
        ans[1] ^= n;
    return ans;
  }

  int MARIarea;
  boolean[][] MARImarked;
  int MARIcurArea;

  public int maxAreaOfIsland(int[][] grid) {
    MARIarea = 0;
    MARIcurArea = 0;
    int r = grid.length, c = grid[0].length;
    MARImarked = new boolean[r][c];
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        if (!MARImarked[i][j]) {
          if (grid[i][j] == 1) {
            MARIdfs(grid, i, j);
            if (MARIcurArea > MARIarea)
              MARIarea = MARIcurArea;
            MARIcurArea = 0;
          } else
            MARImarked[i][j] = true;
        }
    return MARIarea;
  }

  public void MARIdfs(int[][] grid, int r, int c) {
    MARIcurArea++;
    MARImarked[r][c] = true;
    if (r - 1 >= 0 && !MARImarked[r - 1][c] && grid[r - 1][c] == 1)
      MARIdfs(grid, r - 1, c);
    if (r + 1 < grid.length && !MARImarked[r + 1][c] && grid[r + 1][c] == 1)
      MARIdfs(grid, r + 1, c);
    if (c - 1 >= 0 && !MARImarked[r][c - 1] && grid[r][c - 1] == 1)
      MARIdfs(grid, r, c - 1);
    if (c + 1 < grid[0].length && !MARImarked[r][c + 1] && grid[r][c + 1] == 1)
      MARIdfs(grid, r, c + 1);
  }

  public int[] JZOadjustArrayOrder(int[] a) {
    int len = a.length;
    int[] ans = new int[len];
    if (len == 0)
      return ans;

    int s = 0, e = len - 1;
    while (s < e) {
      while ((a[s] & 1) == 1 && s < e)
        s++;
      while ((a[e] & 1) == 0 && s < e)
        e--;
      if (s >= e)
        break;
      int temp = a[s];
      a[s] = a[e];
      a[e] = temp;
    }
    return a;
  }

  public ListNode JZOreverseLinkedNode(ListNode l) {
    if (l == null)
      return null;
    if (l.next == null)
      return l;
    ListNode cur = l, pre = null, next = l.next;
    while (cur != null) {
      cur.next = pre;
      pre = cur;
      cur = next;
      next = next == null ? null : next.next;
    }
    return pre;
  }

  public ListNode JZOmergeLinkedNode(ListNode l1, ListNode l2) {
    if (l1 == null)
      return l2;
    else if (l2 == null)
      return l1;
    ListNode ans = null;
    if (l1.val <= l2.val) {
      ans = l1;
      ans.next = JZOmergeLinkedNode(l1.next, l2);
    } else {
      ans = l2;
      ans.next = JZOmergeLinkedNode(l1, l2.next);
    }
    return ans;
  }

  public boolean JZOisSubTree(TreeNode root, TreeNode sb) {
    if (sb == null)
      return true;
    if (root.val == sb.val)
      if (JZOhasSubTree(root, sb))
        return true;
    return JZOhasSubTree(root.left, sb) || JZOhasSubTree(root.right, sb);
  }

  public boolean JZOhasSubTree(TreeNode root, TreeNode sb) {
    if (sb == null)
      return true;
    if (root == null && sb != null)
      return false;
    if (root.val != sb.val)
      return false;
    return JZOhasSubTree(root.left, sb.left) && JZOhasSubTree(root.right, sb.right);
  }

  public TreeNode JZOmirrorTree(TreeNode root) {
    if (root == null)
      return null;
    TreeNode temp = root.left;
    root.left = root.right;
    root.right = temp;
    JZOmirrorTree(root.left);
    JZOmirrorTree(root.right);
    return root;
  }

  List<Integer> ISTvalues;

  public boolean JZOisSymmetricalTree(TreeNode root) {
    ISTvalues = new LinkedList<>();
    JZOstIte(root);
    Integer[] ans = ISTvalues.toArray(new Integer[0]);
    int s = 0, e = ans.length - 1;
    while (s < e) {
      if (!JZOstEqual(ans[s], ans[e]))
        return false;
      s++;
      e--;
    }
    return true;
  }

  public boolean JZOstEqual(Integer a, Integer b) {
    if (a != null && b != null && a.equals(b))
      return true;
    else if ((a != null && b == null) || (a == null && b != null))
      return false;
    else if (a == null && b == null)
      return true;
    else
      return false;
  }

  public void JZOstIte(TreeNode root) {
    if (root == null) {
      ISTvalues.add(null);
      return;
    }
    JZOstIte(root.left);
    ISTvalues.add(root.val);
    JZOstIte(root.right);
  }

  public void JZOprintMatrix(int[][] M) {
    int re = M.length - 1, ce = M[0].length - 1, rs = 0, cs = 0, dir = 0;

    while (rs <= re && cs <= ce) {
      if (dir % 4 == 0) {
        for (int i = cs; i <= ce; i++)
          System.out.println(M[rs][i]);
        rs++;
        dir++;
      } else if (dir % 4 == 1) {
        for (int i = rs; i <= re; i++)
          System.out.println(M[i][ce]);
        ce--;
        dir++;
      } else if (dir % 4 == 2) {
        for (int i = ce; i >= cs; i--)
          System.out.println(M[re][i]);
        re--;
        dir++;
      } else {
        for (int i = re; i >= rs; i--)
          System.out.println(M[i][cs]);
        cs++;
        dir++;
      }
    }
  }

  class JZOStack {
    private int[] values;
    private int index;
    private int[] min;

    public JZOStack(int size) {
      index = -1;
      values = new int[size];
      min = new int[size];
    }

    public int min() {
      return min[index];
    }

    public void push(int a) throws Exception {
      if (index == values.length - 1)
        throw new Exception();
      index++;
      if (index - 1 >= 0)
        min[index] = a < min[index - 1] ? a : min[index - 1];
      else
        min[index] = a;
      values[index] = a;
    }

    public int pop() throws Exception {
      if (index < 0)
        throw new Exception();
      int ans = values[index];
      index--;
      return ans;
    }
  }

  public boolean JZOisStackOutOrder(int[] stack, int[] out) {
    int sIndex = 0, oIndex = 0, n = stack.length;
    int[] temp = new int[n];
    for (int i = 0; i < n; i++) {
      temp[sIndex] = stack[i];
      while (oIndex < n && sIndex >= 0 && out[oIndex] == temp[sIndex]) {
        sIndex--;
        oIndex++;
      }
      sIndex++;
    }
    if (sIndex == 0)
      return true;
    return false;
  }

  public void JZOprintTreeAccordingLayers(TreeNode root) {
    if (root == null)
      throw new IllegalArgumentException();
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      TreeNode temp = q.poll();
      System.out.print(temp.val + " ");
      if (temp.left != null)
        q.offer(temp.left);
      if (temp.right != null)
        q.offer(temp.right);
    }
  }

  public void JZOprintTreeAccordingLayers1(TreeNode root) {
    if (root == null)
      throw new IllegalArgumentException();
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      int n = q.size();
      String ss = "";
      for (int i = 0; i < n; i++) {
        TreeNode temp = q.poll();
        ss = ss + temp.val + " ";
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      System.out.println(ss);
    }
  }

  public void JZOprintTreeAccordingLayers2(TreeNode root) {
    if (root == null)
      throw new IllegalArgumentException();
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    int in = 0;
    while (!q.isEmpty()) {
      int n = q.size();
      int[] print = new int[n];
      for (int i = 0; i < n; i++) {
        TreeNode temp = q.poll();
        print[i] = temp.val;
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      String s = "";
      if ((in & 1) == 0)
        for (int j = 0; j < n; j++)
          s = s + print[j] + " ";
      else
        for (int j = n - 1; j >= 0; j--)
          s = s + print[j] + " ";
      System.out.println(s);
      in++;
    }
  }

  public boolean JZOisPostOrder(TreeNode root, int[] post) {
    List<Integer> res = new LinkedList<>();
    JZOrecordPost(res, root);
    Iterator<Integer> it = res.iterator();
    if (post.length != res.size())
      return false;
    int index = 0;
    while (it.hasNext()) {
      if (post[index] != it.next())
        return false;
      index++;
    }
    return true;
  }

  public void JZOrecordPost(List<Integer> ls, TreeNode root) {
    if (root == null)
      return;
    JZOrecordPost(ls, root.left);
    JZOrecordPost(ls, root.right);
    ls.add(root.val);
  }

  public boolean JZOisATree(int[] order, int s, int e) {
    if (order == null)
      throw new IllegalArgumentException();
    if (e <= s)
      return true;
    int root = order[e], partition = s;
    while (order[partition] < root) {
      partition++;
    }
    for (int i = partition; i < e; i++)
      if (order[i] <= root)
        return false;
    return JZOisATree(order, s, partition - 1) && JZOisATree(order, partition, e - 1);
  }

  Stack<TreeNode> JZOstack;
  List<List<TreeNode>> JZOgtAns;

  public List<List<TreeNode>> JZOgetTraces(TreeNode root, int sum) {
    if (root == null)
      return null;
    JZOstack = new Stack<>();
    JZOgtAns = new LinkedList<>();
    JZOgetTracesHelper(root, sum);
    return JZOgtAns;
  }

  public void JZOgetTracesHelper(TreeNode root, int sum) {
    if (root == null)
      return;
    if (root.val == sum && root.left == null && root.right == null) {
      JZOstack.push(root);
      List<TreeNode> ls = new LinkedList<>();
      for (int i = 0; i < JZOstack.size(); i++)
        ls.add(JZOstack.get(i));
      JZOgtAns.add(ls);
      JZOstack.pop();
      return;
    }

    JZOstack.push(root);
    JZOgetTracesHelper(root.left, sum - root.val);
    JZOgetTracesHelper(root.right, sum - root.val);
    JZOstack.pop();
  }

  class ComplexListNode {
    int val;
    ComplexListNode next;
    ComplexListNode sibling;
  }

  public ComplexListNode clone(ComplexListNode head) {
    if (head == null)
      throw new IllegalArgumentException();
    Map<Integer, ComplexListNode> cor = new HashMap<>();
    ComplexListNode ans = new ComplexListNode();
    ans.val = head.val;
    cor.put(ans.val, ans);
    ComplexListNode temp = ans, store = head;
    while (head.next != null) {
      ComplexListNode next = new ComplexListNode();
      next.val = head.next.val;
      temp.next = next;
      cor.put(temp.val, temp);
      temp = temp.next;
      head = head.next;
    }
    ComplexListNode h = ans;
    while (store != null) {
      h.next = cor.get(store.sibling.val);
      h = h.next;
      store = store.next;
    }
    return ans;
  }

  TreeNode JZOTTBLast;

  public TreeNode JZOTreeToBiList(TreeNode root) {
    JZOTTBLast = null;
    JZOTTBHelper(root);
    TreeNode ans = root;
    while (ans.left != null)
      ans = ans.left;
    return ans;
  }

  public void JZOTTBHelper(TreeNode root) {
    if (root == null)
      return;
    if (root.left != null)
      JZOTTBHelper(root.left);
    root.left = JZOTTBLast;
    if (JZOTTBLast != null)
      JZOTTBLast.right = root;
    JZOTTBLast = root;
    if (root.right != null)
      JZOTTBHelper(root.right);
  }

  public String JZOSerializeTree(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    STHelper(root, sb);
    return sb.toString();
  }

  public void STHelper(TreeNode root, StringBuilder sb) {
    if (root == null) {
      sb.append("$;");
      return;
    }
    sb.append(root.val + ";");
    STHelper(root.left, sb);
    STHelper(root.right, sb);
  }

  private int DSTIndex;

  public TreeNode JZODeserializeTree(String s) {
    String[] pieces = s.split(";");
    DSTIndex = 0;
    TreeNode ans = DSTHelper(pieces);
    return ans;
  }

  private TreeNode DSTHelper(String[] pieces) {
    if (DSTIndex == pieces.length)
      return null;
    TreeNode ans;
    ans = pieces[DSTIndex].equals("$") ? null : new TreeNode(Integer.valueOf(pieces[DSTIndex]));
    DSTIndex++;
    if (ans != null) {
      ans.left = DSTHelper(pieces);
      ans.right = DSTHelper(pieces);
    }
    return ans;
  }

  public List<String> JZOpermutation(String s) {
    List<String> ans = new LinkedList<>();
    pHelper(s.toCharArray(), 0, ans);
    return ans;
  }

  private void pHelper(char[] cs, int begin, List<String> ls) {
    if (begin == cs.length - 1) {
      String temp = new String(cs);
      ls.add(temp);
    } else {
      for (int i = begin; i < cs.length; i++) {
        pExchange(cs, begin, i);
        pHelper(cs, begin + 1, ls);
        pExchange(cs, i, begin);
      }
    }
  }

  private void pExchange(char[] cs, int i, int j) {
    char temp = cs[i];
    cs[i] = cs[j];
    cs[j] = temp;
  }

  public int JZOfindNumOccureMoreThanHalf(int[] nums) {
    int n = nums.length;
    for (int i = 0; i < n; i += 2)
      if (nums[i] == nums[i + 1])
        return nums[i];
    return nums[n - 1];
  }

  public int[] JZOgetLeastKNumsByPartition(int[] nums, int k) {
    if (k <= 0 || nums == null || nums.length < k)
      throw new IllegalArgumentException();
    int start = 0, end = nums.length - 1;
    int index = GLKPartition(nums, start, end);
    while (index != k - 1)
      if (index < k - 1)
        index = GLKPartition(nums, index + 1, end);
      else
        index = GLKPartition(nums, start, index - 1);

    int[] ans = new int[k];
    for (int i = 0; i < k; i++)
      ans[i] = nums[i];
    return ans;
  }

  public int GLKPartition(int[] nums, int start, int end) {
    if (start < 0 || end >= nums.length)
      throw new IllegalArgumentException();
    int partition = nums[start], i = start + 1, j = end;
    while (i < j) {
      while (i <= end && nums[i] < partition)
        i++;
      while (j > start && nums[j] > partition)
        j--;
      if (i > j)
        break;
      exchangeArrayElement(nums, i, j);
    }
    exchangeArrayElement(nums, start, j);
    return j;
  }

  private void exchangeArrayElement(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  public int[] JZOgetLeastKNumsByHeap(int[] nums, int k) {
    if (k <= 0 || k >= nums.length)
      throw new IllegalArgumentException();
    PriorityQueue<Integer> pq = new PriorityQueue<>(k, new Comparator<Integer>() {
      @Override
      public int compare(Integer o1, Integer o2) {
        if (o1.equals(o2))
          return 0;
        else if (o1.compareTo(o2) > 0)
          return -1;
        else
          return 1;
      }
    });

    for (int a : nums)
      if (pq.size() < k)
        pq.offer(a);
      else if (a < pq.peek()) {
        pq.poll();
        pq.offer(a);
      }
    int[] ans = new int[k];
    for (int i = 0; i < k; i++)
      ans[i] = pq.poll();
    return ans;
  }

  public double JZOgetMediumNun(double[] nums) {
    PriorityQueue<Double> input = new PriorityQueue<>();
    PriorityQueue<Double> output = new PriorityQueue<>(new Comparator<Double>() {
      @Override
      public int compare(Double o1, Double o2) {
        return o2.compareTo(o1);
      }
    });
    int index = 0;
    while (index < nums.length) {
      input.offer(nums[index]);
      if ((index & 1) == 0)
        output.offer(input.poll());
      index++;
    }

    return (index & 1) == 1 ? output.poll() : (output.poll() + input.poll()) / 2;
  }

  public int JZOmaxSubArray1(int[] a) {
    int max = 0, sum = 0;
    for (int num : a) {
      if (num < 0)
        max = sum;
      sum += num;
      if (sum < num) {
        sum = num;
        max = 0;
      }
    }
    return Math.max(max, sum);
  }

  public int JZOmaxSubArray2(int[] nums) {
    if (nums.length < 0)
      throw new IllegalArgumentException();
    if (nums.length == 0)
      return nums[0];
    int len = nums.length, max = 0;
    int[] res = new int[len];
    res[0] = nums[0];
    for (int i = 1; i < len; i++) {
      if (res[i - 1] < 0)
        res[i] = nums[i];
      else
        res[i] = nums[i] + res[i - 1];
      if (res[i] > max)
        max = res[i];
    }
    return max;
  }

  public int JZOoneNums(int n) {
    if (n < 1)
      throw new IllegalArgumentException();
    if (n == 1)
      return 1;
    int l = getLongtitude(n);
    int ans = ONHelper(n, l);
    return ans;
  }

  private int ONHelper(int n, int digits) {
    if (digits == 0)
      return 0;
    int ans = 0, firstD = (int) (n / Math.pow(10, digits - 1));
    for (int i = 0; i <= firstD; i++) {
      if (i == firstD) {
        if (i == 1)
          ans += n - firstD * Math.pow(10, digits - 1) + 1;
        ans += ONHelper((int) (n - firstD * Math.pow(10, digits - 1)), digits - 1);
      } else if (i == 1) {
        ans += (int) Math.pow(10, digits - 1);
        ans += ONHelper((int) (Math.pow(10, digits - 1) - 1), digits - 1);
      } else
        ans += ONHelper((int) (Math.pow(10, digits - 1) - 1), digits - 1);
    }
    return ans;
  }

  private int getLongtitude(int n) {
    int i = 0;
    while (n > 0) {
      n = n / 10;
      i++;
    }
    return i;
  }

  public int JZOgetNNums(int n) {
    if (n < 0)
      throw new IllegalArgumentException();
    if (n < 10)
      return n;
    int longtitude = 1, base = 10;
    while (n > base) {
      longtitude++;
      n -= base;
      base = longtitude * ((int) (Math.pow(10, longtitude) - Math.pow(10, longtitude - 1)));
    }

    int beg = n / longtitude, ind = longtitude - n % longtitude, res = 0;
    int ans = (int) Math.pow(10, longtitude - 1) + beg;
    for (int i = 0; i <= ind; i++) {
      res = ans % 10;
    }
    return res;
  }

  public int JZOgetMinNum(Integer[] nums) {
    if (nums.length <= 0)
      throw new IllegalArgumentException();

    Arrays.sort(nums, new Comparator<Integer>() {
      public int compare(Integer o1, Integer o2) {
        return mergeTwoNum(o1, o2) - mergeTwoNum(o2, o1);
      }
    });
    int ans = 0;
    for (int i = 0; i < nums.length; i++)
      ans = mergeTwoNum(ans, nums[i]);
    return ans;
  }

  private int mergeTwoNum(int o1, int o2) {
    int o1Long = getDigits(o1), o2Long = getDigits(o2);
    return (int) Math.pow(10, o2Long) * o1 + o2;
  }

  private int getDigits(int num) {
    int ans = 0;
    while (num > 0) {
      num /= 10;
      ans++;
    }
    return ans;
  }

  public int JZOgetTranslationNums(int a) {
    if (a < 0)
      throw new IllegalArgumentException();
    if (a == 0)
      return 1;
    Integer[] digits = GTNhelper(a);
    int len = digits.length;
    int[] res = new int[len + 1];
    res[len - 1] = 1;
    res[len] = 1;
    for (int i = len - 2; i >= 0; i--) {
      res[i] = res[i + 1] + res[i + 2] * GTNisLegal(digits[i] * 10 + digits[i + 1]);
    }
    return res[0];
  }

  private int GTNisLegal(int a) {
    if (a >= 0 && a <= 25)
      return 1;
    else
      return 0;
  }

  private Integer[] GTNhelper(int a) {
    Stack<Integer> ans = new Stack<>();
    int remain = a, last = 0;
    while (remain > 0) {
      last = remain % 10;
      ans.push(last);
      remain /= 10;
    }
    Integer[] res = new Integer[ans.size()];
    int index = 0;
    while (!ans.isEmpty())
      res[index++] = ans.pop();

    return res;
  }

  public int JZOmostValuablePath(int[][] a) {
    if (a == null || a.length == 0 || a[0].length == 0)
      throw new IllegalArgumentException();
    if (a.length == 1 || a[0].length == 1) {
      int ans = 0;
      for (int i = 0; i < a.length; i++)
        for (int j = 0; j < a[0].length; j++)
          ans += a[i][j];
      return ans;
    }
    int r = a.length, c = a[0].length;
    int[][] ans = new int[r][c];
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++) {
        int fromLeft = j - 1 < 0 ? 0 : ans[i][j - 1];
        int fromUp = i - 1 < 0 ? 0 : ans[i - 1][j];
        ans[i][j] = a[i][j] + Math.max(fromLeft, fromUp);
      }
    return ans[r - 1][c - 1];
  }

  public int JZOgetLongestSubString1(String s) {
    if (s == null || s.length() == 0)
      throw new IllegalArgumentException();
    if (s.length() == 1)
      return s.length();
    int max = 0, beg = 0, storedMax = 0;
    Map<Character, Integer> mp = new HashMap<>();
    char[] cs = s.toCharArray();
    for (int i = 0; i < cs.length; i++) {
      if (mp.containsKey(cs[i]) && (mp.get(cs[i]) >= beg)) {
        beg = mp.get(cs[i]) + 1;
        max = i - beg + 1;
        storedMax = storedMax > max ? storedMax : max;
      } else
        max++;
      mp.put(cs[i], i);
    }
    return Math.max(max, storedMax);
  }

  public int JZOgetNUglyNum(int n) {
    if (n <= 0)
      throw new IllegalArgumentException();
    if (n == 1)
      return 1;
    int maxIndex = 0, m2 = 0, m3 = 0, m5 = 0;
    int[] res = new int[n];
    res[0] = 1;
    while (maxIndex < n - 1) {
      int m2Res = 2 * res[m2], m3Res = 3 * res[m3], m5Res = 5 * res[m5];
      int min = maxAmongThreeNums(m2Res, m3Res, m5Res);
      if (min == m2Res)
        m2++;
      else if (min == m3Res)
        m3++;
      else
        m5++;
      if (min != res[maxIndex])
        res[++maxIndex] = min;
    }
    return res[n - 1];
  }

  private int maxAmongThreeNums(int a, int b, int c) {
    int temp = Math.min(a, b);
    return Math.min(temp, c);
  }

  public char JZOfirstOnceNum(String s) {
    if (s == null || s.length() == 0)
      throw new IllegalArgumentException();
    if (s.length() == 1)
      return s.charAt(0);
    Map<Character, Integer> mp = new HashMap<>();
    char[] cs = s.toCharArray();
    for (char c : cs)
      mp.put(c, 1 + mp.getOrDefault(c, 0));
    Iterator<Map.Entry<Character, Integer>> it = mp.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Character, Integer> temp = it.next();
      if (temp.getValue() == 1)
        return temp.getKey();
    }
    return ' ';
  }

  int GRPcount;
  int[] GRPaux;

  public int JZOgetReversePair(int[] nums) {
    if (nums == null || nums.length == 1)
      throw new IllegalArgumentException();
    if (nums.length == 2)
      if (nums[0] > nums[1])
        return 1;
      else
        return 0;
    GRPcount = 0;
    GRPaux = new int[nums.length];
//    mergeSortByRecursion(nums, 0, nums.length - 1);
    mergeSortByIteration(nums);
    return GRPcount;
  }

  private void mergeSortByRecursion(int[] a, int lo, int hi) {
    if (lo >= hi)
      return;
    int mid = (lo + hi) / 2;
    mergeSortByRecursion(a, lo, mid);
    mergeSortByRecursion(a, mid + 1, hi);
    JZORPmerge(a, lo, mid, hi);
  }

  private void mergeSortByIteration(int[] a) {
    int len = a.length;
    for (int sz = 1; sz < len; sz *= 2)
      for (int lo = 0; lo < len - sz; lo += (2 * sz))
        JZORPmerge(a, lo, lo + sz - 1, Math.min(lo + sz * 2 - 1, len - 1));
  }

  private void JZORPmerge(int[] a, int start, int medium, int high) {
    int i = medium, j = high, len = high - medium;
    for (int k = start; k <= high; k++)
      GRPaux[k] = a[k];
    for (int k = high; k >= start; k--)
      if (i < start)
        a[k] = GRPaux[j--];
      else if (j <= medium)
        a[k] = GRPaux[i--];
      else if (GRPaux[i] <= GRPaux[j]) {
        len--;
        a[k] = GRPaux[j--];
      } else {
        GRPcount += len;
        a[k] = GRPaux[i--];
      }
  }

  public ListNode JZOgetFirstCommonNode(ListNode l1, ListNode l2) {
    if (l1 == null || l2 == null)
      throw new IllegalArgumentException();
    int len1 = 0, len2 = 0;
    ListNode t1 = l1, t2 = l2;
    while (t1.next != null) {
      t1 = t1.next;
      len1++;
    }
    while (t2.next != null) {
      t2 = t2.next;
      len2++;
    }
    if (len1 > len2) {
      int gap = len1 - len2;
      for (int i = 0; i < gap; i++)
        l1 = l1.next;
    } else
      for (int i = 0; i < len2 - len1; i++)
        l2 = l2.next;
    while (l1.next != null) {
      if (l1.next == l2.next)
        return l1.next;
      l1 = l1.next;
      l2 = l2.next;
    }
    return null;
  }

  public int JZOgetOccuranceNum(int[] nums, int a) {
    if (nums==null)
      throw new IllegalArgumentException();
    int start = BinarySearchForFirst(nums,a,0,nums.length-1);
    int end = BinarySearchForLast(nums,a,0,nums.length-1);
    if (start==-1||end==-1)
      return -1;
    else
      return end-start+1;
  }

  private int BinarySearchForFirst(int[] nums,int a,int start,int end){
    if (start>end)
      return -1;
    int medium = (start+end)/2;
    if (nums[medium]==a&&(medium==start||nums[medium-1]!=a))
      return medium;
    if (nums[medium]>a||(nums[medium]==a&&nums[medium-1]==a))
      return BinarySearchForFirst(nums,a,start,medium);
    else
      return BinarySearchForFirst(nums,a,medium+1,end);
  }

  private int BinarySearchForLast(int[] nums,int a,int start,int end){
    if (start>end)
      return -1;
    int medium = (start+end)/2;
    if (nums[medium]==a&&(medium==end||nums[medium+1]!=a))
      return medium;
    if (nums[medium]<a||(nums[medium]==a&&nums[medium+1]==a))
      return BinarySearchForLast(nums,a,medium,end);
    else
      return BinarySearchForLast(nums,a,start,medium);
  }

  public int JZOlackedNumber(int[] nums){
    if (nums==null)
      throw new IllegalArgumentException();
    int ans = JZOLNBinarySearch(nums,0,nums.length-1);
    return ans;
  }

  private int JZOLNBinarySearch(int[] nums,int start,int end){
    if (start>end)
      return nums.length;
    int medium = (start+end)>>>1;
    if (nums[medium]!=medium&&(medium==start||nums[medium-1]==(medium-1)))
      return medium;
    if (nums[medium]==medium)
      return JZOLNBinarySearch(nums,medium+1,end);
    else
      return JZOLNBinarySearch(nums,start,medium);
  }

  List<TreeNode> GKNList;
  public int JZOGetKNodeInTree(TreeNode root,int k){
      GKNList = new LinkedList<>();
      GKNIteration(root,k);
      return GKNList.get(k-1).val;
  }

  private void GKNIteration(TreeNode root,int k){
    if (root==null)
      return;
    GKNIteration(root.left,k);
    GKNList.add(root);
    if (GKNList.size()==k)
      return;
    GKNIteration(root.right,k);
  }

  public int JZOgetTreeDepth(TreeNode root){
    return root==null?0:Math.max(JZOgetTreeDepth(root.left),JZOgetTreeDepth(root.right))+1;
  }

  Map<TreeNode,Integer> IAVLmap;
  public boolean JZOisAVL(TreeNode root) {
    if (root==null)
      return true;
    IAVLmap = new HashMap<>();
    IAVLgetDepth(root);
    return IAVLHelper(root);
  }

  private boolean IAVLHelper(TreeNode root){
    if (root==null)
      return true;
    boolean isLeftAVL = IAVLHelper(root.left),isRightAVL = IAVLHelper(root.right);
    if (isLeftAVL&&isRightAVL){
      int leftH = IAVLmap.getOrDefault(root.left,0);
      int rightH =IAVLmap.getOrDefault(root.right,0);
      if (Math.abs(leftH-rightH)>1)
        return false;
      else
        return true;
    }
    else
      return false;
  }

  private void IAVLgetDepth(TreeNode root){
    if (root==null)
      return;
    IAVLgetDepth(root.left);
    IAVLgetDepth(root.right);
    if (root.left==null&&root.right==null)
      IAVLmap.put(root,1);
    else{
      if (root.left==null)
        IAVLmap.put(root,IAVLmap.get(root.right)+1);
      else if(root.right==null)
        IAVLmap.put(root,IAVLmap.get(root.left)+1);
      else
        IAVLmap.put(root,Math.max(IAVLmap.get(root.left),IAVLmap.get(root.right))+1);
    }
  }

  public int[] JZOfindTwoSingleNums(int[] nums){
    if (nums==null||nums.length<=2)
      throw new IllegalArgumentException();
    int mix = 0;
    for (int i:nums)
      mix ^= i;
    mix &= -mix;
    int[] ans = new int[2];
    for (int i:nums)
      if ((i&mix)==0)
        ans[0] ^=i;
      else
        ans[1]^=i;
    return ans;
  }

  public int JZOfindSingleAmongThree(int[] nums){
    int[] res = new int[32];
    for (int i:nums){
      int remain = i,last=0,index=0;
      while(remain>0){
        last = remain&1;
        remain>>=1;
        res[index++] +=last;
      }
    }
    int ans=0;
    for (int i=0;i<32;i++){
      int bit = res[i]%3;
      ans += (bit<<i);
    }
    return ans;
  }

  public int[] JZOgetTwoSumIsK(int[] nums,int k){
    if (nums==null||nums.length<2)
      throw new IllegalArgumentException();
    int beg = 0,end = nums.length-1;
    while(beg<end){
      if (nums[beg]+nums[end]==k)
        return new int[]{nums[beg],nums[end]};
      else if (nums[beg]+nums[end]<k)
        beg++;
      else
        end--;
    }
    return null;
  }

  public List<List<Integer>> JZOgetAllContinuedNums(int[] nums,int k){
    if (nums==null||nums.length<2)
      throw new IllegalArgumentException();
    int small=0,big=1,tempSum=nums[0],dir=1;
    List<List<Integer>> ans = new LinkedList<>();
    while(big>=nums.length||small>nums.length/2){
      if (dir==1)
        tempSum+=nums[big];
      else
        tempSum-=nums[small];
      if (tempSum==k){
        List<Integer> ls = new LinkedList<>();
        for (int i=small;i<=big;i++)
          ls.add(nums[i]);
        ans.add(ls);
        dir=1;
        big++;
      }
      else if (tempSum<k){
        big++;
        dir=1;
      }
      else{
        small++;
        dir=-1;
      }
    }
    return ans;
  }

  public String JZOreverseString(String s){
    if (s==null||s.length()==0)
      throw new IllegalArgumentException();
    char[] cs = s.toCharArray();
    int start = 0,end = cs.length-1;
    while(start<end){
      char temp = cs[start];
      cs[start] = cs[end];
      cs[end] = temp;
      start++;
      end--;
    }
    return String.valueOf(cs);
  }

  public String JZOreverseWords(String s){
    if (s==null||s.length()==0)
      throw new IllegalArgumentException();
    String[] words = s.split(" ");
    StringBuilder sb = new StringBuilder();
    for (String word:words){
      String temp = JZOreverseString(word);
      sb.append(temp+" ");
    }
    return sb.toString().trim();
  }

  public String JZOLeftRotate(String s,int k){
    if (s==null||s.length()==0)
      throw new IllegalArgumentException();
    StringBuilder sb = new StringBuilder(s);
    String left = sb.substring(0,k);
    sb.delete(0,k);
    sb.append(left);
    return sb.toString();
  }

  public Integer[] JZOgetMaxValueInSlideWindows(int[] nums,int k){
    if (nums==null||k<=0)
      throw new IllegalArgumentException();
    List<Integer> ans = new LinkedList<>();
    Queue<Integer> cache = new LinkedList<>();
    for (int i:nums){
      cache.offer(i);
      Integer max = JZOSLmax(cache,k);
      if (max!=null)
        ans.add(max);
    }
    return ans.toArray(new Integer[0]);
  }

  private Integer JZOSLmax(Queue<Integer> q,int k){
    if (q.size()<k)
      return null;
    if (q.size()>k){
      int n = q.size()-k;
      for (int i=0;i<n;i++)
        q.poll();
    }
    int max=Integer.MIN_VALUE;
    Iterator<Integer> it = q.iterator();
    while (it.hasNext()){
      Integer temp = it.next();
      if (temp>max)
        max=temp;
    }
    return max;
  }

  public Double[] JZOgetProbOfNSZ(int n){
    if (n<0)
      throw new IllegalArgumentException();
    if(n==0)
      return new Double[0];
    int sum=0;
    int[] last = new int[6*n+1];
    int[] ans = new int[6*n+1];
    for (int i=1;i<=6;i++)
      last[i]= 1;

    for (int i=2;i<=n;i++){
      for (int j=i;j<=6*i;j++)
        for (int k=1;k<=6;k++){
          int remain = j-k;
          if (remain>=(i-1)&&remain<=6*(i-1))
            ans[j] += last[remain];
        }
      for (int x=0;x<=6*n;x++){
        last[x] += ans[x];
        ans[x]=0;
      }
    }

    List<Double> a = new LinkedList<>();
    for (int i=0;i<=6*n;i++)
      sum += last[i];
    for(int j=0;j<=6*n;j++)
      if (last[j]>0)
        a.add((double)last[j]/sum);
    return a.toArray(new Double[0]);
  }

  public boolean JZOisSortedPork(int[] porks){
    if (porks==null||porks.length!=5)
      throw new IllegalArgumentException();
    insertSort(porks);
    int[] res = new int[14];
    int continued=0;
    for (int i:porks)
      res[i]++;
    for (int i=0;i<res.length;i++)
      if ((i>1&&res[i]>0&&res[i-1]>0)||(i<res.length-1&&res[i]>0&&res[i+1]>0))
        continued++;
    if (continued==5)
      return true;
    else if (continued==4&&res[0]>0)
      return true;
    else if (continued==3&&res[0]==2)
      return true;
    else
      return false;
  }

  public int JZOJosephuseCircle1(ListNode head,int m){
    if (head==null||head.next==null||m<=0)
      throw new IllegalArgumentException();
    ListNode cur = head;
    while(cur.next!=cur){
      for (int i=0;i<m-1;i++)
        cur = cur.next;
      ListNode temp = cur.next;
      cur.next = temp.next;
      temp=null;
    }
    return cur.val;
  }

  public int JZOJosephuseCircle2(int n,int m){
    if (n<=0||m<=0)
      throw new IllegalArgumentException();
    if (n==1)
      return 0;
    else
      return (JZOJosephuseCircle2(n-1,m)+m)%n;
  }

  private void insertSort(int[] nums){
    if (nums==null)
      throw new IllegalArgumentException();
    if (nums.length==0||nums.length==1)
      return;
    for (int i=0;i<nums.length;i++)
      for (int j=i;j>0&&(nums[j]<nums[j-1]);j--)
        exchange(nums,j,j-1);
  }

  private void exchange(int[] nums,int i,int j){
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  public int JZOmaxProfit(int[] prices){
    if (prices==null||prices.length<=1)
      throw new IllegalArgumentException();

    int min = prices[0],maxProfit = Integer.MIN_VALUE;
    for (int i=1;i<prices.length;i++)
      if (prices[i]<=min)
        min = prices[i];
      else
        maxProfit=prices[i]-min>maxProfit?prices[i]-min:maxProfit;
    return maxProfit;
  }

  public int JZOgetSumWithoutIfAndLoop(int n){
    int sum = n;
//    boolean b = n>1&&(sum+=JZOgetSumWithoutIfAndLoop(n-1))>0;
    boolean c = n==1||(sum+=JZOgetSumWithoutIfAndLoop(n-1))>0;
    return sum;
  }

  public int JZOSumWithourOperation(int a,int b){
    int sum=0,carry=0;
    sum = a^b;
    carry = (a&b)<<1;
    while(carry>0){
      int s = sum^carry;
      int c = (sum&carry)<<1;
      sum = s;
      carry=c;
    }
    return sum;
  }

  public void exchange1(int[] nums, int i,int j){
    nums[i] = nums[i]+nums[j];
    nums[j] = nums[i]-nums[j];
    nums[i] =nums[i]-nums[j];
  }

  public void exchange2(int[] nums,int i,int j){
    nums[i] = nums[i]^nums[j];
    nums[j] = nums[i]^nums[j];
    nums[i] = nums[i]^nums[j];
  }

  public int[] JZOArrayConvert(int[] nums){
    if (nums==null||nums.length<1)
      throw new IllegalArgumentException();
    int n = nums.length,smaller=1,bigger=1;
    int[] ans = new int[n];
    for (int i=0;i<n;i++)
      ans[i]=1;
    for (int i=1;i<n;i++){
      smaller*=nums[i-1];
      ans[i] *= smaller;
    }
    for (int j=n-2;j>=0;j--){
      bigger*=nums[j+1];
      ans[j]*=bigger;
    }
    return ans;
  }

  public TreeNode invertTree(TreeNode root) {
    if(root==null)
      return null;
    TreeNode temp = root.left;
    root.left = invertTree(root.right);
    root.right = invertTree(temp);
    return root;
  }

  public String toGoatLatin(String S) {
    if (S==null||S.length()==0)
      return S;
    String[] words = S.split("\\s+");
    StringBuilder sb = new StringBuilder();
    for (int i=1;i<=words.length;i++)
      sb.append(convertToGL(words[i-1],i));
    return sb.toString().trim();
  }

  private char[] convertToGL(String s,int index){
    char[] cs = s.toCharArray();
    int n=cs.length;
    char[] ans = new char[n+2+index+1];

    if (cs[0]=='a'||cs[0]=='e'||cs[0]=='i'||cs[0]=='o'||cs[0]=='u'||cs[0]=='A'||cs[0]=='E'||cs[0]=='I'||cs[0]=='O'||cs[0]=='U')
      for (int i=0;i<n;i++)
        ans[i]=cs[i];
    else{
      for (int i=1;i<n;i++)
        ans[i-1]=cs[i];
      ans[n-1] = cs[0];
    }
    ans[n]='m';
    for (int i=n+1;i<ans.length-1;i++)
      ans[i]='a';
    ans[ans.length-1]=' ';
    return ans;
  }

  public boolean isAlienSorted(String[] words, String order) {
    for (int i=0;i<words.length-1;i++)
      if (compare(words[i],words[i+1],order)>0)
        return false;
    return true;
  }

  private int compare(String s1,String s2,String order){
    int index=0,n1=s1.length(),n2=s2.length(),n=n1>n2?n2:n1;
    while(index<n){
      int o1 = order.indexOf(s1.charAt(index));
      int o2 = order.indexOf(s2.charAt(index));
      if (o1==o2){
        index++;
        continue;
      }
      else
        return o1-o2;
    }
    if(index==n1&&index==n2)
      return 0;
    else if (index==n1)
      return -1;
    else
      return 1;
  }

  public int countSubstrings1(String s) {
    if (s.length()==0||s.length()==1)
      return 1;

    char[] cs = s.toCharArray();
    int n=s.length(),ans=0;
    for (int sz=1;sz<=n;sz++)
      for (int i=0;i<n;i++)
        if (i+sz<=n&&isPalindromic(s.substring(i,i+sz)))
          ans++;
    return ans;
  }

  private boolean isPalindromic(String s){
    if (s.length()==1)
      return true;
    if (s.charAt(0)!=s.charAt(s.length()-1))
      return false;
    int start=0,end=s.length()-1;
    while(start<end){
      if (s.charAt(start)!=s.charAt(end))
        return false;
      start++;
      end--;
    }
    return true;
  }


  public int countSubstrings(String s) {
    if (s.length()==0||s.length()==1)
      return s.length();
    int CSScount=0,n=s.length();
    for (int i=0;i<n;i++){
      CSScount+=CSShelper(s,i,i);
      CSScount+=CSShelper(s,i,i+1);
    }
    return CSScount;
  }

  private int CSShelper(String s,int beg,int end){
    int count=0;
    while (beg>=0&&end<s.length()&&s.charAt(beg)==s.charAt(end)){
      count++;
      beg--;
      end++;
    }
    return count;
  }

  public int largestPerimeter(int[] A) {
    Arrays.sort(A);
    for (int edgeOne=A.length-1;edgeOne>=2;edgeOne--)
      if (A[edgeOne-1]+A[edgeOne-2]>A[edgeOne])
        return A[edgeOne]+A[edgeOne-1]+A[edgeOne-2];
    return 0;
  }

  public int findLUSlength(String a, String b) {
    return a.equals(b)?-1:Math.max(a.length(),b.length());
  }

  public int mincostTickets(int[] days, int[] costs) {
    int n= days.length;
    int[] spend = new int[n+1];
    spend[n-1]=costs[0];
    spend[n]=0;
    for (int i=n-2;i>=0;i--){
      int cost1 = costs[0]+spend[findNextDay(days,days[i]+1)];
      int cost2= costs[1]+spend[findNextDay(days,days[i]+7)];
      int cost3=costs[2]+spend[findNextDay(days,days[i]+30)];
      cost1 = cost1<cost2?cost1:cost2;
      cost1=cost1<cost3?cost1:cost3;
      spend[i]=cost1;
    }
    return spend[0];
  }

  private int findNextDay(int[] days,int from){
    int ans =days.length;
    for (int i=days.length-1;i>=0;i--)
      if (days[i]>=from)
        ans=i;
      else
        break;
    return ans;
  }

  public int numberOfArithmeticSlices(int[] A) {
    if (A==null||A.length<3)
      return 0;
    int count=0;
    for (int i=1;i<A.length-1;i++){
      count+=ArithmeticNum(A,i,i);
      count+=ArithmeticNum(A,i,i+1);
    }
    return count;
  }

  private int ArithmeticNum(int[] A,int start,int end){
    int count=0,gap;
    if (start==end)
      gap=start>0?A[start]-A[start-1]:A[end+1]-A[end];
    else
      gap=A[end]-A[start];

    while(start>0&&end<A.length-1&&A[start]-A[start-1]==gap&&A[end+1]-A[end]==gap){
      count++;
      start--;
      end++;
    }
    return count;
  }

  boolean[][] UPmarked;
  int UPcountPath;
  int UPZeroNums;
  public int uniquePathsIII(int[][] grid) {
    int n=grid.length,m=grid[0].length,startR=0,startC=0;
    UPcountPath=0;
    for (int i=0;i<n;i++)
      for (int j=0;j<m;j++)
        if (grid[i][j]==0)
          UPZeroNums++;
        else if (grid[i][j]==1){
          startR=i;
          startC=j;
        }
    UPmarked=new boolean[n][m];
    UPdfs(grid,startR,startC);
    return UPcountPath;
  }

  private void UPdfs(int[][] grid,int r,int c){
    if (grid[r][c]==0)
      UPmarked[r][c]=true;
    else if (grid[r][c]==2){
      if (UPisOverAllZeros())
        UPcountPath++;
      return;
    }

    if (r>0&&((grid[r-1][c]==0&&UPmarked[r-1][c]==false)||grid[r-1][c]==2))
      UPdfs(grid,r-1,c);
    if (r<grid.length-1&&((grid[r+1][c]==0&&UPmarked[r+1][c]==false)||grid[r+1][c]==2))
      UPdfs(grid,r+1,c);
    if (c>0&&((grid[r][c-1]==0&&UPmarked[r][c-1]==false)||grid[r][c-1]==2))
      UPdfs(grid,r,c-1);
    if (c<grid[0].length-1&&((grid[r][c+1]==0&&UPmarked[r][c+1]==false)||grid[r][c+1]==2))
      UPdfs(grid,r,c+1);

    UPmarked[r][c]=false;
  }

  private boolean UPisOverAllZeros() {
    int count = 0;
    int n = UPmarked.length,m=UPmarked[0].length;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
        if (UPmarked[i][j] == true)
          count++;
    return count==UPZeroNums;
  }

  public String reverseOnlyLetters(String S) {
    if (S.length()<2)
      return S;
    char[] cs = S.toCharArray();
    int start=0,end=cs.length-1;
    while(start<end){
      while(start<cs.length&&!isAlphabet(cs[start]))
        start++;
      while(end>=0&&!isAlphabet(cs[end]))
        end--;
      if (start>=end)
        break;
      exchangeChar(cs,start++,end--);
    }
    return new String(cs);
  }

  private boolean isAlphabet(char c){
    if ((c>=65&&c<=90)||(c>=97&&c<=122))
      return true;
    else
      return false;
  }

  private void exchangeChar(char[] cs,int i,int j){
    char temp = cs[i];
    cs[i]=cs[j];
    cs[j]=temp;
  }

  public boolean canWinNim(int n) {
    if (n%4==0)
      return false;
    else
      return true;
//    if (n<=0)
//      throw new IllegalArgumentException();
//    if (n<=3)
//      return true;
//    if (n==4)
//      return false;
//
//    boolean[] dp = new boolean[3];
//    boolean ans=false;
//    int index=0;
//    dp[0]=true;
//    dp[1]=true;
//    dp[2]=false;
//
//    for (int i=5;i<=n;i++)
//      dp[index++%3]=!(dp[1]&&dp[2]&&dp[0]);
//    return dp[(index-1)%3];
  }


  public String optimalDivision(int[] nums) {
    if (nums.length==1)
      return nums[0]+"";
    else if (nums.length==2)
      return nums[0]+"/"+nums[1];
    else{
      StringBuilder sb = new StringBuilder();
      sb.append(nums[0]);
      sb.append("/(");
      for (int i=1;i<nums.length;i++){
        sb.append(nums[i]);
        sb.append('/');
      }
      sb.deleteCharAt(sb.length()-1);
      sb.append(')');
      return sb.toString();
    }
  }

  private String ODconstructor(int[] nums,int start,int end,int status){
    if (end==start)
      return nums[start]+"";
    else if (end-start==1)
      return nums[start]+"/"+nums[end];
    else
      if (status==1)
        return nums[start]+"/("+ODconstructor(nums,start+1,end,-1)+")";
      else
        return ODconstructor(nums,start,end-1,-1)+"/"+nums[end];
  }

  public int scoreOfParentheses(String S) {
    Stack<Integer> st = new Stack<>();
    st.push(0);
    for (char c:S.toCharArray())
      if (c=='(')
        st.push(0);
      else{
        int v=st.pop();
        int w=st.pop();
        st.push(w+Math.max(v*2,1));
      }
    return st.peek();
//    if (S.length()==2)
//      return 1;
//    char[] cs = S.toCharArray();
//    return (int)SPgetScore(cs,0,cs.length-1);
  }

  private double SPgetScore(char[] cs,int start,int end){
    if (end-start==1)
      return 1;
    else if (start-end==1)
      return 0.5;
    double ans=0.0;
    int bal=0,pre=start;
    List<Integer> divide = new LinkedList<>();
    for (int i=start;i<=end;i++)
      if (cs[i]=='(')
        bal++;
      else{
        bal--;
        if (bal==0)
          divide.add(i);
      }
    for (int index:divide){
      int s = pre;
      int e = index;
      pre = e+1;
      ans +=2* SPgetScore(cs,s+1,e-1);
    }
    return ans;
  }

  public int surfaceArea(int[][] grid) {
    int topAspect=0,leftAspect=0,frontAspect=0,n=grid.length;
    for (int row=0;row<n;row++)
      for (int col=0;col<n;col++){
        if (grid[row][col]!=0)
          topAspect+=2;
        frontAspect+=getDirectedSurface(grid,row,col,row-1,col)+getDirectedSurface(grid,row,col,row+1,col);
        leftAspect += getDirectedSurface(grid,row,col,row,col-1)+getDirectedSurface(grid,row,col,row,col+1);
      }
    return topAspect+leftAspect+frontAspect;
  }

  private int getDirectedSurface(int[][] grid,int r,int c,int rn,int rc){
     if (rn<0||rc<0||rn>=grid.length||rc>=grid[0].length)
       return grid[r][c];
     else
       return Math.max(grid[r][c]-grid[rn][rc],0);
  }

  public int[] fairCandySwap(int[] A, int[] B) {
    int ASum=0,BSum=0,diffAtoB=0;
    for (int a:A)
      ASum+=a;
    for (int b:B)
      BSum+=b;
    diffAtoB = (ASum-BSum)/2;
//    for (int i=0;i<A.length;i++)
//      for (int j=0;j<B.length;j++)
//        if (A[i]-B[j]==diffAtoB)
//          return new int[]{A[i],B[j]};
    Set<Integer> Bset = new HashSet<>();
    for (int b:B)
      Bset.add(b);
    for (int a:A)
      if (Bset.contains(a-diffAtoB))
        return new int[]{a,a-diffAtoB};
    return null;
  }

  class CBTInserter {
    private TreeNode root;
    private TreeNode cur;
    private Queue<TreeNode> q;

    public CBTInserter(TreeNode root) {
      this.root=root;
      q = new LinkedList<>();
      q.offer(root);
      while(!q.isEmpty()){
        TreeNode temp = q.peek();
        if (temp.left==null||temp.right==null){
          cur=temp;
          break;
        }
        else{
          q.poll();
          q.offer(temp.left);
          q.offer(temp.right);
        }
      }
    }

    public int insert(int v) {
      TreeNode node = new TreeNode(v);
      int par;
      if (cur.left==null){
        cur.left=node;
        par = cur.val;
      }
      else{
        par = cur.val;
        cur.right=node;
        q.offer(cur.left);
        q.offer(cur.right);
        q.poll();
        cur=q.peek();
      }
      return par;
    }

    public TreeNode get_root() {
      return root;
    }
  }

  public String frequencySort(String s) {
    Map<Character,Integer> mp = new HashMap<>();
    List<Character>[] cs= new List[s.length()];
    for (int i=0;i<cs.length;i++)
      cs[i]=new LinkedList<>();
    for (char c:s.toCharArray())
      mp.put(c,mp.getOrDefault(c,0)+1);
    Iterator<Map.Entry<Character,Integer>> it = mp.entrySet().iterator();
    while(it.hasNext()){
      Map.Entry<Character,Integer> temp = it.next();
      cs[temp.getValue()-1].add(temp.getKey());
    }
    StringBuilder sb = new StringBuilder();
    for (int i=cs.length-1;i>=0;i--)
      for (Character c:cs[i])
        for (int j=0;j<=i;j++)
          sb.append(c);
    return sb.toString();
  }

  class MyCalendarThree1 {
    SortedMap<Integer,Integer> time;
    public MyCalendarThree1() {
      time = new TreeMap<>();
    }

    public int book(int start, int end) {
      time.put(start,time.getOrDefault(start,0)+1);
      time.put(end,time.getOrDefault(end,0)-1);
      int active=0,ans=0;
      for (int tv:time.values()){
        active += tv;
        if (active>ans)
          ans = active;
      }
      return ans;
    }
  }


  class MyCalendarThree {
    private SegmentTreeNode root;
    private int maxIntersectionNum;
    public MyCalendarThree() {
      maxIntersectionNum = 0;
      root=new SegmentTreeNode(Integer.MIN_VALUE,Integer.MAX_VALUE,0);
    }

    public int book(int start, int end) {
      root.add(start,end,1);
      return maxIntersectionNum;
    }

    class SegmentTreeNode{
      int left,right,val,mid=-1;
      SegmentTreeNode leftSubTree,rightSubTree;
      public SegmentTreeNode(int l,int r,int count){
        left=l;
        right=r;
        this.val = count;
      }

      public void add(int start,int end,int count){
        if (this.mid !=-1)
          if (start>=mid){
            rightSubTree.add(start,end,count);
            return;
          }
          else if(end <=mid){
            leftSubTree.add(start,end,count);
            return;
          }
          else{
            leftSubTree.add(start,mid,count);
            rightSubTree.add(mid,end,count);
            return;
          }

        if (start==this.left&&end==this.right){
          val+=count;
          maxIntersectionNum= maxIntersectionNum<val?val:maxIntersectionNum;
        }
        else if (start==left&&end!=right){
          constructSubTree(left,end,right);
          leftSubTree.add(start,end,count);
        }
        else if (start!=left&&end==right){
          constructSubTree(left,start,end);
          rightSubTree.add(start,end,count);
        }
        else{
          constructSubTree(left,end,right);
          leftSubTree.add(start,end,count);
        }
      }

      private void constructSubTree(int l,int m,int r){
        mid = m;
        leftSubTree = new SegmentTreeNode(l,m,val);
        rightSubTree = new SegmentTreeNode(m,r,val);
      }
    }
  }

  public double largestTriangleArea(int[][] points) {
    double ans = 0;
    for (int i = 0; i < points.length; i++)
      for (int j = i + 1; j < points.length; j++)
        for (int k = j + 1; k < points.length; k++) {
          double area = shoelaceTiangleArea(points[i], points[j], points[k]);
          ans = ans > area ? ans : area;
        }
    return ans;
  }

  private double shoelaceTiangleArea(int[] a1,int[] a2,int[] a3){
    return Math.abs(0.5*(double)(a1[0]*a2[1]+a2[0]*a3[1]+a3[0]*a1[1]-a1[0]*a3[1]-a2[0]*a1[1]-a3[0]*a2[1]));
  }

  public List<String> letterCasePermutation1(String S) {
    Queue<String> ans = new LinkedList<>();
    ans.offer(S);
    for (int i = 0; i < S.length(); i++) {
      char temp = S.charAt(i);
      if (Character.isDigit(temp))
        continue;
      char toggle = (char) (temp <= 90 ? temp + 32 : temp - 32);
      int iteNum = ans.size();
      for (int j = 0; j < iteNum; j++) {
        String first = ans.poll();
        char[] cs = first.toCharArray();
        ans.offer(first);
        cs[i] = toggle;
        ans.offer(String.valueOf(cs));
      }
    }
    return new LinkedList<>(ans);
  }

  public List<String> letterCasePermutation(String S) {
     List<String> ls = new LinkedList<>();
     Set<String> set = new HashSet<>();
     if (S==null)
       return ls;
     LCPprocessor(S.toCharArray(),0,set);
     ls.addAll(set);
     return ls;
  }

  private void LCPprocessor(char[] cs,int pos,Set<String> ls){
    if (pos==cs.length){
      ls.add(String.valueOf(cs));
      return;
    }

    if (Character.isDigit(cs[pos]))
      LCPprocessor(cs,pos+1,ls);
    cs[pos] = Character.toLowerCase(cs[pos]);
    LCPprocessor(cs,pos+1,ls);
    cs[pos] = Character.toUpperCase(cs[pos]);
    LCPprocessor(cs,pos+1,ls);
  }

  public boolean isMonotonic1(int[] A) {
    if (A.length<=2)
      return true;
    int dir=0;
    for (int i=1;i<A.length;i++){
      int diff = A[i]-A[i-1];
      if (diff==0)
        continue;
      if (dir==0)
        dir = diff;
      else
        if (dir*diff<0)
          return false;
    }
    return true;
  }
  public boolean isMonotonic(int[] A) {
    boolean inc=true,dec=true;
    for (int i=1;i<A.length;i++){
      inc &= A[i]-A[i-1]>=0;
      dec &= A[i]-A[i-1]<=0;
    }
    return inc||dec;
  }

  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> ans = new LinkedList<>();
    Stack<TreeNode> st = new Stack<>();
    TreeNode cur = root;
    while(cur!=null||!st.isEmpty()){
      while(cur!=null){
        st.push(cur);
        cur = cur.left;
      }
      TreeNode temp = st.pop();
      ans.add(temp.val);
      cur = temp.right;
    }
    return ans;
  }

  private void inorderByRecursion(TreeNode root, List<Integer> ls){
    if (root==null)
      return;
    inorderByRecursion(root.left,ls);
    ls.add(root.val);
    inorderByRecursion(root.right,ls);
  }

  public boolean escapeGhosts(int[][] ghosts, int[] target) {
    int maxStep = Math.abs(target[0])+Math.abs(target[1]);
    for (int[] g:ghosts){
      int gStep = Math.abs(g[0]-target[0])+Math.abs(g[1]-target[1]);
      if (gStep<=maxStep)
        return false;
    }
    return true;
  }

  public int minDeletionSize3(String[] A) {
    if (A[0].length()==1)
      return 0;
    int N = A.length,len = A[0].length(),maxOrder=0;
    int[] dp = new int[len];
    Arrays.fill(dp,1);
    for (int i=1;i<len;i++){
      for (int j=0;j<i;j++)
        if (isOrdered(A,j,i))
          dp[i] = Math.max(dp[i],dp[j]+1);
      maxOrder = Math.max(dp[i],maxOrder);
    }
    return len-maxOrder;
  }

  private boolean isOrdered(String[] A,int l,int r){
    for (String S:A)
      if (S.charAt(l)>S.charAt(r))
        return false;
    return true;
  }

  public int findMaxConsecutiveOnes(int[] nums) {
    int max=0,curLen = 0;
    for (int i:nums)
      if ((i&1)==1)
        curLen++;
      else
        if (curLen!=0){
          max = Math.max(max,curLen);
          curLen=0;
        }
    return Math.max(max,curLen);
  }

  class QuadNode {
    public boolean val;
    public boolean isLeaf;
    public QuadNode topLeft;
    public QuadNode topRight;
    public QuadNode bottomLeft;
    public QuadNode bottomRight;

    public QuadNode() {}

    public QuadNode(boolean _val,boolean _isLeaf,QuadNode _topLeft,QuadNode _topRight,QuadNode _bottomLeft,QuadNode _bottomRight) {
      val = _val;
      isLeaf = _isLeaf;
      topLeft = _topLeft;
      topRight = _topRight;
      bottomLeft = _bottomLeft;
      bottomRight = _bottomRight;
    }
  }

  public QuadNode construct(int[][] grid) {
    return constructQuadTree(grid,0,grid[0].length-1,0,grid.length-1);
  }

  private QuadNode constructQuadTree(int[][] grid,int l,int r,int t,int b){
    if (l==r&&t==b)
      return new QuadNode(grid[t][l]==1?true:false,true,null,null,null,null);
    QuadNode temp;
    if (isAllSame(grid,l,r,t,b))
      temp = new QuadNode(grid[t][l]==1?true:false,true,null,null,null,null);
    else{
      int tbMid = (t+b)/2,lrMid = (r+l)/2;
      QuadNode topLeft = constructQuadTree(grid,l,lrMid,t,tbMid);
      QuadNode topRight = constructQuadTree(grid,lrMid+1,r,t,tbMid);
      QuadNode bottomLeft = constructQuadTree(grid,l,lrMid,tbMid+1,b);
      QuadNode bottomRight = constructQuadTree(grid,lrMid+1,r,tbMid+1,b);
      temp = new QuadNode(false,false,topLeft,topRight,bottomLeft,bottomRight);
    }
    return temp;
  }

  private boolean isAllSame(int[][] grid,int l,int r,int t,int b){
    int val = grid[t][l];
    for (int i=t;i<=b;i++)
      for (int j=l;j<=r;j++)
        if (grid[i][j]!=val)
          return false;
    return true;
  }

  List<List<TreeNode>> deepestPaths;
  int maxDepth;
  public TreeNode subtreeWithAllDeepest1(TreeNode root) {
    if(root==null)
      return null;
    if (root.left==null&&root.right==null)
      return root;
    maxDepth=0;
    deepestPaths = new LinkedList<>();
    LinkedList<TreeNode> singlePath = new LinkedList<>();
    getDeepestPaths(root,singlePath);
    int depth = deepestPaths.get(0).size();
    for (int i=0;i<deepestPaths.get(0).size();i++)
      if (isCommonParent(deepestPaths,i))
        return deepestPaths.get(0).get(i);
    return null;
  }

  private boolean isCommonParent(List<List<TreeNode>> ls,int pos){
    int val = ls.get(0).get(pos).val;
    for (int i=0;i<ls.size();i++)
      if (ls.get(i).get(pos).val!=val)
        return false;
    return true;
  }

  private void getDeepestPaths(TreeNode root, LinkedList<TreeNode> path) {
    if (root.left == null && root.right == null) {
      path.addFirst(root);
      int curDepth = path.size();
      if (curDepth == maxDepth) {
        deepestPaths.add(new LinkedList<>(path));
        path.removeFirst();
        return;
      } else if (curDepth > maxDepth) {
        maxDepth = curDepth;
        deepestPaths.clear();
        deepestPaths.add(new LinkedList<>(path));
        path.removeFirst();
        return;
      } else{
        path.removeFirst();
        return;
      }

    }
    path.addFirst(root);
    if (root.left != null)
      getDeepestPaths(root.left, path);
    if (root.right != null)
      getDeepestPaths(root.right, path);
    path.removeFirst();
  }


  public TreeNode subtreeWithAllDeepest(TreeNode root) {
    return getDepthMaps(root).getValue();
  }

  private Pair<Integer,TreeNode> getDepthMaps(TreeNode root){
    if (root==null)
      return new Pair<>(0,null);
    Pair<Integer,TreeNode> left = getDepthMaps(root.left),right = getDepthMaps(root.right);
    int ld = left.getKey(),rd = right.getKey();
    return new Pair<>(Math.max(ld,rd)+1,ld==rd?root:ld>rd?left.getValue():right.getValue());
  }
  public void moveZeroes(int[] nums) {
    Queue<Integer> zeros = new LinkedList<>();
    int len = nums.length;
    for (int i=0;i<len;i++)
      if (nums[i]==0)
        zeros.offer(i);
      else if (!zeros.isEmpty()){
        int zeroPos = zeros.poll();
        exchange(nums,zeroPos,i);
        zeros.offer(i);
      }
  }

  int[] MSaux;
  public void mergeSordByRec(int[] nums){
    MSaux = new int[nums.length];
    mergeSortByRec(nums,0,nums.length-1);
  }

  private void mergeSortByRec(int[] nums,int lo,int hi){
    if (hi<=lo)
      return;
    int mid = (hi - lo) / 2+lo;
    mergeSortByRec(nums, lo, mid);
    mergeSortByRec(nums, mid + 1, hi);
    mergeInMS(nums,lo,mid,hi);
  }

  private void mergeInMS(int[] nums,int lo,int mid,int hi){
    int i = lo,j = mid+1;
    for (int k=lo;k<=hi;k++)
      MSaux[k] = nums[k];

    for (int k=lo;k<=hi;k++)
      if (i>mid)
        nums[k] = MSaux[j++];
      else if (j>hi)
        nums[k] = MSaux[i++];
      else if (MSaux[j]!=0&&MSaux[i]==0)
        nums[k] = MSaux[j++];
      else
        nums[k] = MSaux[i++];
  }

  class MyHashMap {

    class ListNode {
      int key;
      int value;
      ListNode next;

      ListNode(int key, int value) {
        this.key = key;
        this.value = value;
      }
    }

    /** Initialize your data structure here. */
    ListNode[] map;
    int MAX_NODES = 10000;
    public MyHashMap() {
      map = new ListNode[MAX_NODES];
    }

    /** value will always be non-negative. */
    public void put(int key, int value) {
      int index = getIndex(key);
      ListNode node = map[index];
      if (node == null) {
        node = new ListNode(-1, -1);
        map[index] = node;
      }

      node = findNode(node, key);
      if (node.next == null) {
        node.next = new ListNode(key, value);
      } else {
        node.next.value = value;
      }
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
      int index = getIndex(key);
      ListNode node = map[index];
      if (node == null) {
        return -1;
      }

      node = findNode(node, key);
      return node.next == null ? -1 : node.next.value;
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
      int index = getIndex(key);
      ListNode node = map[index];
      if (node == null) {
        return;
      }

      node = findNode(node, key);
      if (node.next != null) {
        node.next = node.next.next;
      }
    }

    private int getIndex(int key) {
      return key % MAX_NODES;
    }

    private ListNode findNode(ListNode node, int key) {
      ListNode prev = null;
      while(node != null && node.key != key) {
        prev = node;
        node = node.next;
      }

      return prev;
    }
  }

  public int addDigits(int num) {
//    int temp;
//    while(num>=10){
//      temp=0;
//      do{
//        temp += (num%10);
//        num/=10;
//      }while(num>0);
//      num=temp;
//    }
//    return num;
    return 1+(num-1)%9;
  }

  public String shortestCompletingWord1(String licensePlate, String[] words) {
    String ans=null;
    List<Character> lp = new ArrayList<>();
    for (char c:licensePlate.toCharArray()){
      if (Character.isAlphabetic(c))
        lp.add(Character.toLowerCase(c));
    }
    for (String temp:words)
      if (isMatch(lp,temp))
        ans = (ans==null||ans.length()>temp.length())?ans = temp:ans;
    return ans;
  }

  private boolean isMatch(List<Character> ls,String temp){
    Map<Character,Integer> mp = new HashMap<>();
    for (char c:temp.toLowerCase().toCharArray())
      mp.put(c,mp.getOrDefault(c,0)+1);
    for (Character C:ls)
      if (mp.getOrDefault(C,0)<=0)
        return false;
      else
        mp.put(C,mp.get(C)-1);
    return true;
  }

  public List<List<String>> findDuplicate(String[] paths) {
    Map<String,List<String>> mp=new HashMap<>();
    for (String path:paths){
      String[] col = path.split(" ");
      for (int i=1;i<col.length;i++){
        String content=null,file=null;
        String[] res = col[i].split("\\(");
        content = res[1].substring(0,res[1].length()-1);
        file=res[0];
        if (!mp.containsKey(content))
          mp.put(content,new LinkedList<>());
        mp.get(content).add(col[0]+"/"+file);
      }
    }

    List<List<String>> ans = new LinkedList<>();
    Iterator<Map.Entry<String,List<String>>> it = mp.entrySet().iterator();
    while(it.hasNext()){
      Map.Entry<String,List<String>> temp = it.next();
      if (temp.getValue().size()>1)
        ans.add(temp.getValue());
    }
    return ans;
  }

  Map<Integer,Integer> RDnumsClassifier;
  public int rotatedDigits(int N) {
    int ans=0;
    RDnumsClassifier = new HashMap<>();
    RDnumsClassifier.put(2,1);
    RDnumsClassifier.put(5,1);
    RDnumsClassifier.put(6,1);
    RDnumsClassifier.put(9,1);
    RDnumsClassifier.put(0,2);
    RDnumsClassifier.put(1,2);
    RDnumsClassifier.put(8,2);
    RDnumsClassifier.put(3,3);
    RDnumsClassifier.put(4,3);
    RDnumsClassifier.put(7,3);
    for (int i=1;i<=N;i++)
      if (isGoodNum(i))
        ans++;
    return ans;
  }

  private boolean isGoodNum(int n){
    int goodNum=0,last,remain=n,type;
    do{
      last = remain%10;
      remain/=10;
      type = RDnumsClassifier.get(last);
      switch (type){
        case 1:
          goodNum++;
          break;
        case 3:
          return false;
      }
    }while (remain>0);

    return goodNum>0?true:false;
  }

  class Employee {
    // It's the unique id of each node;
    // unique id of this employee
    public int id;
    // the importance value of this employee
    public int importance;
    // the id of direct subordinates
    public List<Integer> subordinates;
  }

  public int getImportance(List<Employee> employees, int id) {
    int ans=0;
    Map<Integer,Employee> mp = new HashMap<>();
    for (Employee e:employees)
      mp.put(e.id,e);
    Queue<Employee> q = new LinkedList<>();
    q.offer(mp.get(id));
    while(!q.isEmpty()){
      Employee temp = q.poll();
      ans += temp.importance;
      for (int subId:temp.subordinates)
        q.offer(mp.get(subId));
    }
    return ans;
  }

  public int countArrangement(int N) {
    int ans =0;
    int[] arr = new int[N+1];
    for (int i=1;i<=N;i++)
      arr[i] = i;
    ans = countArrangement(arr,1);
    return ans;
  }

  private int countArrangement(int[] arr,int i){
    if (i==arr.length){
      return 1;
    }
    int count=0;
    for (int k=i;k<=arr.length-1;k++){
      exchange(arr,i,k);
      if (isBeautiful(arr[i],i))
        count+=countArrangement(arr,i+1);
      exchange(arr,i,k);
    }
    return count;
  }

  private boolean isBeautiful(int i,int loc){
    return Math.max(i,loc)%Math.min(i,loc)==0;
  }

  Map<Integer,Integer> fftsMp;
  int fftsMaxConSum;
  public int[] findFrequentTreeSum(TreeNode root) {
    fftsMp = new HashMap<>();
    fftsMaxConSum = 0;
    ffts(root);
    List<Integer> res = new ArrayList<>();
    Iterator<Map.Entry<Integer,Integer>> it = fftsMp.entrySet().iterator();
    while(it.hasNext()){
      Map.Entry<Integer,Integer> et = it.next();
      if (et.getValue()==fftsMaxConSum)
        res.add(et.getKey());
    }
    int[] ans = new int[res.size()];
    for (int i=0;i<res.size();i++)
      ans[i] = res.get(i);
    return ans;
  }

  private int ffts(TreeNode root){
    if (root==null)
      return 0;
    int sum = 0;
    sum += ffts(root.left);
    sum += ffts(root.right);
    sum += root.val;
    fftsMp.put(sum,fftsMp.getOrDefault(sum,0)+1);
    int con=fftsMp.getOrDefault(sum,0);
    if (con>fftsMaxConSum)
      fftsMaxConSum = con;
    return sum;
  }

  public char findTheDifference(String s, String t) {
    int sum = 0;
    for (char c:t.toCharArray())
      sum ^= c;
    for (char sc:s.toCharArray())
      sum ^=sc;
    return (char)sum;
  }

  public int[] intersection(int[] nums1, int[] nums2) {
    Map<Integer,Integer> mp = new HashMap<>();
    Set<Integer> st = new HashSet<>();
    for (int i=0;i<nums1.length;i++)
      mp.putIfAbsent(nums1[i],i);
    for (int n:nums2)
      if (mp.containsKey(n))
        st.add(n);
    Iterator<Integer> it = st.iterator();
    int[] ans = new int[st.size()];
    for (int i=0;i<st.size();i++)
      ans[i] = it.next();
    return ans;
  }


  public int numComponents(ListNode head, int[] G) {
    Map<Integer,Integer> gmp=new HashMap<>();
    int len = G.length,ans=len;
    boolean last = false,cur;
    for (int j=0;j<len;j++)
      gmp.put(G[j],j);
    while(head!=null){
      cur = gmp.containsKey(head.val);
      if (last&&cur)
        ans--;
      last = cur;
      head=head.next;
    }
    return ans;
  }

  public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] ans = new int[n];
    int left=1,right=1;
    ans[0] =left;
    for (int i=1;i<n;i++){
      left *=nums[i-1];
      ans[i]= left;
    }
    for (int j=n-2;j>=0;j--){
      right = right*nums[j+1];
      ans[j] *=  right;
    }
    return ans;
  }

  //  class FreqStack {
//    class element{
//      public int id;
//      public int value;
//      public element(int _value,int _id){
//        this.value = _value;
//        this.id = _id;
//      }
//    }
//
//    private List<element> elementData;
//    private Map<Integer,Integer> elementFrequency;
//    private int compare(element o1, element o2) {
//      int o1Freq = elementFrequency.get(o1.value),o2Freq = elementFrequency.get(o2.value);
//      return o1Freq!=o2Freq?o2Freq-o1Freq:o2.id-o1.id;
//    }
//    private int elementNum=0;

  //    public FreqStack() {
//      elementFrequency = new HashMap<>();elementFrequency.computeIfAbsent()
//      elementData = new ArrayList<>();
//    }
//
//    public void push(int x) {
//      element temp = new element(x,elementNum++);
//      elementFrequency.put(temp.value,elementFrequency.getOrDefault(temp.value,0)+1);
//      elementData.add(temp);
//    }
//
//    public int pop() {
//      if (elementData.size() == 0)
//        return -1;
//      int i=0;
//      element temp = elementData.get(0);
//      for (int k=1;k<elementData.size();k++)
//        if (compare(elementData.get(k),temp)<0){
//          temp = elementData.get(k);
//          i=k;
//        }
//      elementData.remove(i);
//      elementFrequency.put(temp.value,elementFrequency.get(temp.value)-1);
//      return temp.value;
//    }
//  }
  class FreqStack {
    Map<Integer, Integer> freq;
    Map<Integer, Stack<Integer>> group;
    int maxfreq;

    public FreqStack() {
      freq = new HashMap();
      group = new HashMap();
      maxfreq = 0;
    }

    public void push(int x) {
      int f = freq.getOrDefault(x, 0) + 1;
      freq.put(x, f);
      if (f > maxfreq)
        maxfreq = f;

      group.computeIfAbsent(f, z-> new Stack()).push(x);
    }

    public int pop() {
      int x = group.get(maxfreq).pop();
      freq.put(x, freq.get(x) - 1);
      if (group.get(maxfreq).size() == 0)
        maxfreq--;
      return x;
    }
  }

  public List<Integer> findDisappearedNumbers(int[] nums) {
    if (nums==null||nums.length==0)
      return new ArrayList<>();
    for (int i=0;i<nums.length;i++)
      if (nums[Math.abs(nums[i])-1]>0)
        nums[Math.abs(nums[i])-1]*=-1;
    List<Integer> ans = new LinkedList<>();
    for (int i=0;i<nums.length;i++)
      if (nums[i]>0)
        ans.add(i+1);
    return  ans;
  }

  public boolean isCousins(TreeNode root, int x, int y) {
    Queue<TreeNode> q = new LinkedList<>();
    q.add(root);
    int findNum;
    while(!q.isEmpty()){
      findNum=0;
      int layerSize = q.size();
      for (int i=0;i<layerSize;i++){
        TreeNode temp = q.poll();
        if (temp.left!=null&&temp.right!=null &&( (temp.left.val==x&&temp.right.val==y)||(temp.left.val==y&&temp.right.val==x)))
          return false;
        if (temp.val==x||temp.val==y)
          findNum++;
        if (findNum==2)
          return true;
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
      }
      if (findNum==1)
        return false;
    }
    return false;
  }

  public List<Integer> topKFrequent(int[] nums, int k) {
//    Map<Integer,Integer> mp = new HashMap<>();
//    Map<Integer,Stack<Integer>> group = new HashMap<>();
//    Set<Integer> res = new HashSet<>();
//    int maxFreq=0;
//    for (int i:nums){
//      int freq = mp.getOrDefault(i,0)+1;
//      maxFreq = Math.max(maxFreq,freq);
//      mp.put(i,freq);
//      group.computeIfAbsent(freq,f->new Stack<>()).push(i);
//    }
//    for (int f = maxFreq;f>0;f--)
//      for (int i:group.get(f)){
//        res.add(i);
//        if (res.size()==k)
//          return new ArrayList<>(res);
//      }
//    return null;
    Map<Integer,Integer> vTof = new HashMap<>();
    List<Integer>[] fTov = new List[nums.length+1];
    for (int i:nums)
      vTof.put(i,vTof.getOrDefault(i,0)+1);
    Iterator<Map.Entry<Integer,Integer>> it = vTof.entrySet().iterator();
    while(it.hasNext()){
      Map.Entry<Integer,Integer> temp= it.next();
      int freq = temp.getValue(),val = temp.getKey();
      if (fTov[freq]==null)
        fTov[freq]=new LinkedList<>();
      fTov[freq].add(val);
    }

    List<Integer> ans = new LinkedList<>();
    int index=0;
    for (int i=nums.length;i>0;i--)
      if (fTov[i]!=null)
        for (int j:fTov[i]){
          ans.add(j);
          if (++index>=k)
            return ans;
        }
    return ans;
  }

  public int minimumDeleteSum(String s1, String s2) {
    Map<Character, List<Integer>> mp = new HashMap<>();
    int s1Asc = 0, s2Asc = 0;
    for (int i = 0; i < s2.length(); i++) {
      mp.computeIfAbsent(s2.charAt(i), c -> new LinkedList<>()).add(i);
      s2Asc += s2.charAt(i);
    }
    List<List<Integer>> res = new LinkedList<>();
    for (char c : s1.toCharArray()) {
      s1Asc += c;
      if (mp.containsKey(c))
        res.add(mp.get(c));
    }
    int maxAscii = getMaxSubAsc(res,s2);
    return s1Asc + s2Asc - 2 * maxAscii;
  }

  private int getMaxSubAsc( List<List<Integer>> res,String s){
    int n = s.length(),max=0,last;
    int[] ans = new int[n];
    for (int i=res.size()-1;i>=0;i--)
      for (int j:res.get(i)){
        last=0;
        for (int k=j+1;k<n;k++)
          last = Math.max(last,ans[k]);
        ans[j] = Math.max(s.charAt(j)+last,ans[j]);
      }
    for (int i=0;i<n;i++)
      max = Math.max(max,ans[i]);
    return max;
  }

  public ListNode reverseList1(ListNode head) {
    if (head==null||head.next==null)
      return head;
    ListNode last=null,cur = head,next;
    while(cur!=null){
      next = cur.next;
      cur.next=last;
      last = cur;
      cur = next;
    }
    return last;
  }

  public ListNode reverseList(ListNode head) {
    if (head==null||head.next==null)
      return head;
    return reverseList(null,head);
  }

  private ListNode reverseList(ListNode last,ListNode cur){
    if (cur==null)
      return last;
    ListNode next = cur.next;
    cur.next = last;
    return reverseList(cur,next);
  }

  public int countBinarySubstrings(String s) {
    if (s.length()<2)
      return 0;
    int count=0;
    char[] cs = s.toCharArray();
    for (int i=0;i<cs.length-1;i++)
      if ((cs[i]=='0'&&cs[i+1]=='1')||(cs[i]=='1'&&cs[i+1]=='0')){
        count++;
        int expend=1;
        while(i-expend>=0 && i+1+expend<cs.length&& cs[i-expend]==cs[i] && cs[i+1+expend] == cs[i+1]){
          count++;
          expend++;
        }
      }
    return count;
  }

  List<String> gpLs;
  public List<String> generateParenthesis(int n) {
    gpLs = new LinkedList<>();
    if (n<=0)
      return gpLs;
    generateParenthesis("",0,n,n);
    return gpLs;
  }

  private void generateParenthesis(String cur,int sum,int lp,int rp){
    if (lp==0||rp==0){
      char temp = lp!=0?'(':')';
      int times= Math.abs(lp-rp);
      for (;times>0;times--)
        cur+=temp;
      gpLs.add(cur);
      return;
    }
    if (sum==0)
      generateParenthesis(cur+'(',-1,lp-1,rp);
    else{
      generateParenthesis(cur+'(',sum-1,lp-1,rp);
      generateParenthesis(cur+')',sum+1,lp,rp-1);
    }
  }

  public int lengthOfLIS(int[] nums) {
//    if (nums==null||nums.length==0)
//      return 0;
//    if (nums.length==1)
//      return 1;
//    int n=nums.length,length,ans=0;
//    int[] dp = new int[n];
//    for (int i=0;i<n;i++){
//      length=1;
//      for (int j=0;j<i;j++){
//        int len = nums[j]<nums[i]?dp[j]+1:1;
//        length = Math.max(len,length);
//      }
//      dp[i]= length;
//      ans = Math.max(length,ans);
//    }
//    return ans;
    int[] dp = new int[nums.length];
    int len = 0;
    for (int num : nums) {
      int i = Arrays.binarySearch(dp, 0, len, num);
      if (i < 0) {
        i = -(i + 1);
      }
      dp[i] = num;
      if (i == len) {
        len++;
      }
    }
    return len;
  }

  public int countTriplets(int[] A) {
    int n=A.length,ans=0;
    for (int i=0;i<n;i++)
      for (int j=i;j<n;j++){
        int temp = A[i]&A[j];
        for (int k=j;k<n;k++)
          if ((temp&A[k])==0)
            if (i==j&&j==k)
              ans++;
            else if (i==j||j==k||i==k)
              ans+=3;
            else
              ans+=6;
      }
    return ans;
  }

  public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next = node.next.next;
  }

  public boolean detectCapitalUse(String word) {
    boolean allUpper=true;
    char[] cs = word.toCharArray();
    for (int i=0;i<cs.length;i++){
      if (allUpper&&Character.isLowerCase(cs[i])){
        if (i>1)
          return false;
        allUpper=false;
      }
      if (!allUpper&&Character.isUpperCase(cs[i]))
        return false;
    }
    return true;
  }

  public String smallestFromLeaf(TreeNode root) {
   if (root.left==null&&root.right==null)
     return (char)(root.val+'a')+"";
   String leftSmallest = root.left==null?null:smallestFromLeaf(root.left);
   String rightSmallest = root.right==null?null:smallestFromLeaf(root.right);
   char cur = (char)('a'+root.val);
   if (leftSmallest==null)
     return rightSmallest+cur;
   else if (rightSmallest==null)
     return leftSmallest+cur;
   else return (stringComparator(leftSmallest,rightSmallest)<=0?leftSmallest:rightSmallest)+cur;
  }

  private int stringComparator(String a,String b){
    int al = a.length(),bl=b.length(),minL = Math.min(al,bl);
    for (int i=0;i<minL;i++)
      if (a.charAt(i)<b.charAt(i))
        return -1;
      else if (a.charAt(i)>b.charAt(i))
        return 1;
    return Integer.compare(al,bl);
  }

  boolean[] fcnMarked;
  int fcnCount;
  public int findCircleNum1(int[][] M) {
    if (M==null)
      throw new IllegalArgumentException();
    int N = M.length;
    fcnMarked = new boolean[N];
    fcnCount=0;
    for (int i=0;i<N;i++)
      if (!fcnMarked[i]){
        fcnDFS(M,i);
        fcnCount++;
      }
    return fcnCount;
  }

  private void fcnDFS(int[][] M,int i){
    fcnMarked[i]=true;
    for (int j=0;j<M.length;j++)
      if (!fcnMarked[j]&&M[i][j]==1)
        fcnDFS(M,j);
  }

  int[] fcnId;
  int[] fcnWeight;
  int fcnFUCount;
  public int findCircleNum(int[][] M) {
    if (M==null)
      throw new IllegalArgumentException();
    int N = M.length;
    fcnFUCount=N;
    fcnId = new int[N];
    fcnWeight = new int[N];
    for (int i=0;i<N;i++){
      fcnId[i]=i;
      fcnWeight[i]=1;
    }
    for (int i=0;i<N;i++)
      for (int j=i+1;j<N;j++)
        if (M[i][j]==1)
          fcnUnion(i,j);
    return fcnFUCount;
  }

  private int fcnFind(int i){
    if (fcnId[i]==i)
      return i;
    int last = i;
    while(fcnId[i]!=i)
      i=fcnId[i];

    while(fcnId[last]!=i){
      int temp=fcnId[last];
      fcnId[last]=i;
      last = temp;
    }
    return i;
  }

  private void fcnUnion(int i,int j){
    int idI = fcnFind(i);
    int idJ = fcnFind(j);
    if (idI==idJ)
      return;
    if (fcnWeight[idI]<=fcnWeight[idJ]){
      fcnId[idI] = idJ;
      fcnWeight[idJ] += fcnWeight[idI];
    }
    else{
      fcnId[idJ] = idI;
      fcnWeight[idI] += fcnWeight[idJ];
    }
    fcnFUCount--;
  }

  public int minKBitFlips1(int[] A, int K) {
    if (K<=0||A==null||A.length==0)
      throw new IllegalArgumentException();
    int N = A.length,count=0;
    for (int i=0;i<N;i++)
      if (A[i]==0)
        if (N-i<K)
          return -1;
        else{
          for (int j=0;j<K;j++)
            A[i+j] ^=1;
          count++;
        }
    return count;
  }

  public int minKBitFlips(int[] A, int K) {
    int N = A.length;
    int[] hint = new int[N];
    int ans = 0, flip = 0;

    // When we flip a subarray like A[i], A[i+1], ..., A[i+K-1]
    // we can instead flip our current writing state, and put a hint at
    // position i+K to flip back our writing state.
    for (int i = 0; i < N; ++i) {
      flip ^= hint[i];
      if (A[i] == flip) {  // If we must flip the subarray starting here...
        ans++;  // We're flipping the subarray from A[i] to A[i+K-1]
        if (i + K > N) return -1;  //If we can't flip the entire subarray, its impossible
        flip ^= 1;
        if (i + K < N) hint[i + K] ^= 1;
      }
    }

    return ans;
  }

  public boolean findTarget(TreeNode root, int k) {
    List<Integer> st = new LinkedList<>();
    getTree(root,st);
    Integer[] res = st.toArray(new Integer[0]);
    int beg=0,end=res.length-1;
    while(beg<end){
      int val = res[beg]+res[end];
      if (val==k)
        return true;
      else if (val>k)
        end--;
      else
        beg++;
    }
    return false;
  }

  private void getTree(TreeNode root,List<Integer> st){
    if (root==null)
      return;
    getTree(root.left,st);
    st.add(root.val);
    getTree(root.right,st);
  }

  public int majorityElement(int[] nums) {
    Map<Integer,Integer> mp = new HashMap<>();
    int max=0,mn=0;
    for (int n:nums){
      int times = mp.getOrDefault(n,0)+1;
      mp.put(n,times);
      if (times>max){
        max=times;
        mn = n;
      }
    }
    return mn;
  }

  public int findPoisonedDuration(int[] timeSeries, int duration) {
    if (timeSeries==null||timeSeries.length==0)
      return 0;
    if (timeSeries.length==1)
      return duration;
    int endp=timeSeries[0]+duration,ans=0;
    for (int i=1;i<timeSeries.length;i++){
      if (endp<=timeSeries[i])
        ans+=duration;
      else
        ans+=(timeSeries[i]-timeSeries[i-1]);
      endp = timeSeries[i]+duration;
    }
    ans+=duration;
    return ans;
  }

  public int minMoves2(int[] nums) {
    if (nums==null||nums.length==0||nums.length==1)
      return 0;
    int N = nums.length,mid,ans=0;
    Arrays.sort(nums);
    mid = nums[N>>1];
    for (int i:nums)
      ans += Math.abs(i-mid);
    return ans;
  }

  public int minSwapsCouples(int[] row) {
    int N = row.length,count=0;
    for (int i=0;i<N;i+=2){
      int spouse = (row[i]&1)==1?row[i]-1:row[i]+1;
      if (row[i+1]==spouse)
        continue;
      for (int j=i+2;j<N;j++)
        if (row[j]==spouse){
          exchange(row,i+1,j);
          count++;
          break;
        }
    }
    return count;
  }

  public int getSum1(int a, int b) {
    int add =0,remainA=a,remainB = b,lastA,lastB,ans=0,shift=0;
    while(shift<=31&&(add!=0||remainA!=0||remainB!=0)){
      lastA = remainA&1;
      lastB = remainB&1;
      ans ^= (lastA^lastB^add)<<shift;
      if((lastA==1&&lastB==1)||((lastA==1||lastB==1)&&add==1))
        add=1;
      else
        add=0;
      shift = addOne(shift);
      remainA>>=1;
      remainB>>=1;
    }
    return ans;
  }

  private int addOne(int a){
    if ((a&1)==0)
      return a^1;
    else
      return addOne(a>>1)<<1;
  }

  public int getSum(int a, int b) {
    return b==0?a:getSum(a^b,(a&b)<<1);
  }

  public String tree2str(TreeNode t) {
    if (t==null)
      return "";
    String ans = String.valueOf(t.val);
    String left = tree2str(t.left);
    String right = tree2str(t.right);
    if (left==""&&right=="")
      return ans;
    else if (left!=""&&right=="")
      return ans+"("+left+")";
    else if (left==""&&right!="")
      return ans+"()"+"("+right+")";
    else
      return ans+"("+left+")"+"("+right+")";
  }

  public char[][] updateBoard(char[][] board, int[] click) {
    if (board[click[0]][click[1]]=='M'){
      board[click[0]][click[1]] = 'X';
      return board;
    }
    exploreE(board,click);
    return board;
  }

  private void exploreE(char[][] board,int[] click){
    List<int[]> blanks=new LinkedList<>();
    char BNum='0';
    for (int i=click[0]-1;i<=click[0]+1;i++)
      for (int j=click[1]-1;j<=click[1]+1;j++){
        if ((i==click[0]&&j==click[1])||i<0||i>=board.length||j<0||j>=board[0].length)
          continue;
        char temp=board[i][j];
        if (temp=='M')
          BNum++;
        else if (temp=='E')
          blanks.add(new int[]{i,j});
      }
    if (BNum=='0'){
      board[click[0]][click[1]]='B';
      for (int[] loc:blanks)
        exploreE(board,loc);
    }
    else
      board[click[0]][click[1]] = BNum;
  }


  public int arrayNesting(int[] nums) {
    int N = nums.length,max=0;
    boolean[] marked = new boolean[N];
    for (int i=0;i<N;i++)
      if (!marked[i])
        max = Math.max(getNumDepth(nums,marked,i),max);
    return max;
  }

  private int getNumDepth(int[] nums,boolean[] marked,int i){
    marked[i]=true;
    int depth=1;
    if (!marked[nums[i]])
      depth+=getNumDepth(nums,marked,nums[i]);
    return depth;
  }

  public boolean isAnagram(String s, String t) {
    int[] res = new int[26];
    for (char a:s.toCharArray())
      res[a-'a']++;
    for(char b:t.toCharArray())
      res[b-'a']--;
    for (int c:res)
      if (c!=0)
        return false;
    return true;
  }

  public int maxProfit(int[] prices) {
    if (prices==null||prices.length==1)
      return 0;
    int ans=0,j;
    for (int i=0;i<prices.length-1;i++){
      for (j=i+1;j<prices.length&&prices[j]>prices[j-1];j++)
        continue;
      if (j!=i+1){
        ans += prices[j-1]-prices[i];
        i = j-1;
      }
    }
    return ans;
  }

  public int mirrorReflection(int p, int q) {
    int g =gcd1(p,q);
    p=p/g&1;
    q = q/g&1;
    return p==0?2:q==0?0:1;
  }

  private int gcd1(int a,int b){
    return a==0?b:gcd1(b%a,a);
  }


  public int[] constructArray(int n, int k) {
    int[] res = new int[n];
    if (k==1){
      for (int i=0;i<n;i++)
        res[i] = i+1;
      return res;
    }
    int big = n,sma =1,index=0;
    for (int i=0;i<k;i++){
      int temp = (i&1)==0?big--:sma++;
      res[index++]=temp;
    }
    if ((k&1)==1)
      for (;index<n;index++)
        res[index] = big--;
    else
      for (;index<n;index++)
        res[index]=sma++;
    return res;
  }

  public int titleToNumber(String s) {
    if (s==null||s.length()==0)
      return 0;
    int ans=0;
    for (char c:s.toCharArray())
      if (Character.isUpperCase(c))
        ans =ans*26+ c-'A'+1;
    return ans;
  }

  public boolean containsDuplicate(int[] nums) {
    if (nums==null||nums.length<=1)
      return false;
    Set<Integer> st = new HashSet<>();
    for (int i:nums)
      if (st.contains(i))
        return true;
      else
        st.add(i);
    return false;
  }

  int[] aux3;
  public void mergeSort3(int[] nums){
    aux3 = new int[nums.length];
    mergeSort3(nums,0,nums.length);
  }

  private void mergeSort3(int[] nums,int beg,int end){
    int mid = (end+beg)/2;
    if (end-beg>1){
      mergeSort3(nums,beg,mid);
      mergeSort3(nums,mid+1,end);
    }
    merge3(nums,beg,mid,end);
  }

  private void merge3(int[] nums,int beg,int mid,int end){
    int i=beg,j=mid+1;
    for (int k=beg;k<=end;k++)
      aux3[k]=nums[k];
    for (int k=beg;k<=end;k++)
      if (i>mid)
        nums[k] = aux3[j++];
      else if (j>end)
        nums[k]=aux3[i++];
      else if (aux3[i]<=aux3[j])
        nums[k]=aux3[i++];
      else
        nums[k]=aux3[j++];
  }

  class MapSum {
    private int R=256;
    private DictNode root;
    private class DictNode{
      public Integer val;
      public DictNode[] nexts= new DictNode[R];
    }

    /** Initialize your data structure here. */
    public MapSum() {
      root = new DictNode();
    }

    public void insert(String key, int val) {
      insert(key,val,0,root);
    }

    private DictNode insert(String key,int val,int d,DictNode root){
      if (root==null)
        root = new DictNode();
      if (d==key.length()){
        root.val = val;
        return root;
      }
      char c= key.charAt(d);
      root.nexts[c] = insert(key,val,d+1,root.nexts[c]);
      return root;
    }

    public int sum(String prefix) {
      DictNode temp = get(prefix,0,root);
      if (temp==null)
        return 0;
      int ans=0;
      Queue<DictNode> q = new LinkedList<>();
      q.offer(temp);
      while(!q.isEmpty()){
        DictNode t = q.poll();
        if (t.val!=null)
          ans += t.val;
        for (int i=0;i<R;i++)
          if (t.nexts[i]!=null)
            q.offer(t.nexts[i]);
      }
      return ans;
    }

    public Integer get(String pre){
      DictNode temp = get(pre,0,root);
      if (temp==null)
        return null;
      return temp.val;
    }

    private DictNode get(String pre,int d,DictNode root){
      if (root==null)
        return null;
      if (d==pre.length())
        return root;
      char c = pre.charAt(d);
      return get(pre,d+1,root.nexts[c]);
    }
  }

  public int[] beautifulArray(int N) {
    ArrayList<Integer> res = new ArrayList<>();
    res.add(1);
    while(res.size()<N){
      ArrayList<Integer> temp = new ArrayList<>();
      for (int i:res)
        if (2*i-1<=N)
          temp.add(2*i-1);
      for (int i:res)
        if (2*i<=N)
          temp.add(2*i);
      res = temp;
    }
    return res.stream().mapToInt(i->i).toArray();
  }

  class MyHashSet {

    int[] table;
    /** Initialize your data structure here. */
    public MyHashSet() {
      table = new int[1000000];
    }

    public void add(int key) {
      if (table[key]==0)
        table[key]=1;
    }

    public void remove(int key) {
      if (table[key]==1)
        table[key]=0;
    }

    /** Returns true if this set contains the specified element */
    public boolean contains(int key) {
      return table[key]==1;
    }
  }

  public boolean lemonadeChange(int[] bills) {
    int[] deposit = new int[2];
    for (int b:bills)
      if (b==5)
        deposit[0]++;
      else if (b==10)
        if (deposit[0]==0)
          return false;
        else{
          deposit[1]++;
          deposit[0]--;
        }
      else
        if (deposit[0]>=1&&deposit[1]>=1){
          deposit[0]--;
          deposit[1]--;
        }
        else if (deposit[0]>=3)
          deposit[0]-=3;
        else
          return false;
    return true;
  }

  public List<List<String>> printTree(TreeNode root) {
    int depth = PTgetDepth(root);
    int PTWidth = (1<<depth)-1;
    List<List<String>> PTRes = new ArrayList<>(depth);
    for (int i=0;i<depth;i++){
      List<String> temp=new ArrayList<>(PTWidth);
      for (int j=0;j<PTWidth;j++)
        temp.add("");
      PTRes.add(temp);
    }
    printTreeHelper(PTRes,root,0,PTWidth-1,0);
    return PTRes;
  }

  private void printTreeHelper(List<List<String>> res, TreeNode root,int l,int r,int d){
    if (root==null)
      return;
    int mid = (l+r)>>1;
    res.get(d).set(mid,String.valueOf(root.val));
    printTreeHelper(res,root.left,l,mid-1,d+1);
    printTreeHelper(res,root.right,mid+1,r,d+1);
  }

  private int PTgetDepth(TreeNode root){
    return root==null?0:Math.max(PTgetDepth(root.left),PTgetDepth(root.right))+1;
  }

  int CBSTcount;
  public TreeNode convertBST(TreeNode root) {
    CBSTcount=0;
    postTraverseBST(root);
    return root;
  }

  private void postTraverseBST(TreeNode root){
    if (root==null)
      return;
    postTraverseBST(root.right);
    int temp=root.val;
    root.val+=CBSTcount;
    CBSTcount += temp;
    postTraverseBST(root.left);
  }

  int mdTemp=Integer.MAX_VALUE,mdMin=Integer.MAX_VALUE;
  public int minDiffInBST(TreeNode root) {
    if (root==null)
      return 0;
    minDiffInBST(root.right);
    mdMin = Math.min(mdMin,mdTemp-root.val);
    mdTemp = root.val;
    minDiffInBST(root.left);
    return mdMin;
  }

  public boolean isValid1(String S) {
    Stack<Character> st = new Stack<>();
    char[] cs = S.toCharArray(),template = new char[]{'a','b','c'};
    int csL= cs.length,tpL = template.length;
    for (char c:cs){
      if (c==template[tpL-1]&& st.size()>=tpL-1){
        for (int i=tpL-2;i>=0;i--)
          if (st.pop()!=template[i])
            return false;
      }
      else
        st.push(c);
    }
    return st.isEmpty();
  }

  public int oddEvenJumps(int[] A) {
    int N = A.length,res=1;
    boolean[] higher=new boolean[N],lower = new boolean[N];
    higher[N-1]=lower[N-1]=true;
    TreeMap<Integer,Integer> mp = new TreeMap<>();
    mp.put(A[N-1],N-1);
    for (int i=N-2;i>=0;i--){
      Map.Entry lo = mp.floorEntry(A[i]),hi=mp.ceilingEntry(A[i]);
      if (hi!=null)
        higher[i]= lower[(int)hi.getValue()];
      if (lo!=null)
        lower[i] = higher[(int)lo.getValue()];
      if (higher[i])
        res++;
      mp.put(A[i],i);
    }
    return res;
  }

  boolean[][] FLMarked;
  public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
    int R=image.length,C = image[0].length;
    FLMarked=new boolean[R][C];
    floodFillHelper(image,sr,sc,newColor);
    return image;
  }

  private void floodFillHelper(int[][] image,int r,int c,int color){
    FLMarked[r][c]=true;
    int oriColor=image[r][c];
    image[r][c]=color;
    if (r-1>=0 && !FLMarked[r-1][c] && image[r-1][c]==oriColor)
      floodFillHelper(image,r-1,c,color);
    if (r+1<image.length && !FLMarked[r+1][c]&& image[r+1][c]==oriColor)
      floodFillHelper(image,r+1,c,color);
    if (c-1>=0 && !FLMarked[r][c-1]&& image[r][c-1]==oriColor)
      floodFillHelper(image,r,c-1,color);
    if (c+1<image[0].length && !FLMarked[r][c+1]&& image[r][c+1]==oriColor)
      floodFillHelper(image,r,c+1,color);
  }

  int GMDmin = Integer.MAX_VALUE,GMDlast = Integer.MAX_VALUE;
  public int getMinimumDifference(TreeNode root) {
    if (root==null)
      return 0;
    getMinimumDifference(root.right);
    GMDmin = Math.min(GMDmin,GMDlast-root.val);
    GMDlast = root.val;
    getMinimumDifference(root.left);
    return GMDmin;
  }

  public int maxChunksToSorted(int[] arr) {
    int res=0,N=arr.length,chunkBeg=-1,chunkEnd=-1;
    int[] valToIndex = new int[N];
    for (int i=0;i<N;i++)
      valToIndex[arr[i]]=i;
    for (int j=0;j<N;j++)
      if (valToIndex[j]==j&&chunkBeg==-1){
        res++;
        continue;
      }
      else if (chunkBeg==-1){
        chunkBeg=j;
        chunkEnd = valToIndex[j];
      }
      else if (valToIndex[j]>chunkEnd)
        chunkEnd = valToIndex[j];
      else if (j==chunkEnd){
        res++;
        chunkBeg=-1;
        chunkEnd=-1;
      }
    return res;
  }

  List<List<Integer>> SSres;
  public List<List<Integer>> subsets1(int[] nums) {
    SSres = new LinkedList<>();
    SSres.add(new LinkedList<>());
    for (int i=1;i<=nums.length;i++)
      for (int j=0;j<=nums.length-i;j++){
        List<Integer> temp = new LinkedList<>();
        subsets1(temp,nums,j,i);
      }
    return SSres;
  }

  private void subsets1(List<Integer> res,int[] nums,int beg,int remain){
    res.add(nums[beg]);
    if (remain==1){
      SSres.add(res);
      return;
    }

    for (int i=beg+1;i<=nums.length-remain+1;i++){
      List<Integer> temp = new LinkedList<>(res);
      subsets1(temp,nums,i,remain-1);
    }
  }

  public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new LinkedList<>();
    subsets(res,new Stack<>(),nums,0);
    return res;
   }

  private void subsets(List<List<Integer>> res, Stack<Integer> path, int[] nums, int beg) {
    res.add(new ArrayList<>(path));
    for (int i=beg;i<nums.length;i++){
      path.push(nums[i]);
      subsets(res,path,nums,i+1);
      path.pop();
    }
  }

  int queens=0;
  public int totalNQueens(int n) {
    boolean[] cols = new boolean[n],d1 = new boolean[n*2],d2 = new boolean[n*2];
    queenBackTrack(0,n,cols,d1,d2);
    return queens;
  }

  private void queenBackTrack(int row,int n,boolean[] cols,boolean[] d1,boolean[] d2){
    if (row==n){
      queens++;
      return;
    }

    for (int c=0;c<n;c++){
      int d1Index = row+c;
      int d2Index = c-row+n;
      if (cols[c]||d1[d1Index]||d2[d2Index])
        continue;
      cols[c]=true;
      d1[d1Index]=true;
      d2[d2Index]=true;
      queenBackTrack(row+1,n,cols,d1,d2);
      cols[c]=false;
      d1[d1Index]=false;
      d2[d2Index]=false;
    }
  }

  public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p==null&&q==null)
      return true;
    else if (p==null||q==null)
      return false;
    return p.val==q.val && isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);
  }

  public int[] twoSum(int[] numbers, int target) {
    int beg=0,end=numbers.length-1;
    while(beg<end){
      int sum=numbers[beg]+numbers[end];
      if (sum==target)
        return new int[]{beg+1,end+1};
      else if (sum<target)
        beg++;
      else
        end--;
    }
    return null;
  }

  public int longestOnes(int[] A, int K) {
    int beg=0,end;
    for (end=0;end<A.length;end++){
      if (A[end]==0)
        K--;
      if (K<0 && A[beg++]==0)
        K++;
    }
    return end-beg;
  }

  class MagicDictionary {
    private class Node{
      boolean isWord;
      Node[] children=new Node[26];
    }

    private Node root;
    /** Initialize your data structure here. */
    public MagicDictionary() {
      root = new Node();
    }

    private void buildTrie(String S){
       buildTrie(S,0,root);
    }

    private void buildTrie(String S,int d,Node root){
      if (d==S.length()){
        root.isWord=true;
        return;
      }
      int index=S.charAt(d)-'a';
      if (root.children[index]==null)
        root.children[index] = new Node();
      buildTrie(S,d+1,root.children[index]);
    }

    /** Build a dictionary through a list of words */
    public void buildDict(String[] dict) {
      for (String S:dict)
        buildTrie(S);
    }

    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    public boolean search(String word) {
      return search(word,0,root,false);
    }

    private boolean search(String word,int d,Node root,boolean isExchanged){
      if (d==word.length())
        return root.isWord&&isExchanged;
      int index = word.charAt(d)-'a';
      if (root.children[index]!=null)
        if (search(word,d+1,root.children[index],isExchanged))
          return true;
      if (!isExchanged)
        for (int i=0;i<26;i++)
          if (root.children[i]!=null && i!=index &&search(word,d+1,root.children[i],true))
            return true;
      return false;
    }
  }

  public int slidingPuzzle(int[][] board) {
    String target = "123450";
    String cur = "";
    int stepsCount=0;
    for (int i=0;i<board.length;i++)
      for (int j=0;j<board[0].length;j++)
        cur += board[i][j];
    Queue<String> q = new LinkedList<>();
    Set<String> isVisited = new HashSet<>();
    int[][] dirs = new int[][]{{1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4}};
    q.offer(cur);
    isVisited.add(cur);
    while(!q.isEmpty()){
      int size = q.size();
      for (int i=0;i<size;i++){
        String temp = q.poll();
        if (temp.equals(target))
          return stepsCount;
        int indexOfZero = temp.indexOf('0');
        for (int dir:dirs[indexOfZero]){
          String changed = swapPuzzle(temp,indexOfZero,dir);
          if (isVisited.add(changed))
            q.offer(changed);
        }
      }
      stepsCount++;
    }
    return -1;
  }

  private String swapPuzzle(String temp,int zero,int des){
    StringBuilder sb = new StringBuilder(temp);
    sb.setCharAt(zero,temp.charAt(des));
    sb.setCharAt(des,temp.charAt(zero));
    return sb.toString();
  }


  public List<String> commonChars(String[] A) {
    int N=A.length;
    int[] res=new int[26];
    Arrays.fill(res,Integer.MAX_VALUE);
    for (String S:A){
      int[] temp = new int[26];
      for (char c:S.toCharArray())
        temp[c-'a']++;
      for (int i=0;i<26;i++)
        res[i]=Math.min(res[i],temp[i]);
    }
    List<String> ans = new LinkedList<>();
    for (int i=0;i<26;i++)
      while(res[i]-->0)
        ans.add(String.valueOf((char)(i+'a')));
    return ans;
  }

  public TreeNode sortedArrayToBST(int[] nums) {
    return constructBST(nums,0,nums.length-1);
  }

  private TreeNode constructBST(int[] nums,int beg,int end){
    if (beg>end)
      return null;
    if (beg==end)
      return new TreeNode(nums[beg]);
    int mid = (end+beg)/2;
    TreeNode temp = new TreeNode(nums[mid]);
    temp.left = constructBST(nums,beg,mid-1);
    temp.right=constructBST(nums,mid+1,end);
    return temp;
  }

  class RWTrieNode{
    boolean isRoot;
    RWTrieNode[] children=new RWTrieNode[26];
  }

  RWTrieNode RWroot;
  public String replaceWords(List<String> dict, String sentence) {
    RWroot = new RWTrieNode();
    for (String s:dict)
      constructRWTrie(RWroot,s,0);
    String[] words = sentence.split("\\s");
    for (int i=0;i<words.length;i++)
      words[i] = searchRoot(RWroot,words[i],0);
    StringBuilder sb = new StringBuilder();
    for (String S:words){
      sb.append(S);
      sb.append(" ");
    }
    return sb.toString().trim();
  }

  private String searchRoot(RWTrieNode root,String key,int d){
    int index;
    if (root.isRoot)
      return key.substring(0,d);
    else if (d==key.length()||root.children[(index=key.charAt(d)-'a')]==null)
      return key;
    else
      return searchRoot(root.children[index],key,d+1);
  }

  private void constructRWTrie(RWTrieNode root,String key,int d){
    if (d==key.length()){
      root.isRoot=true;
      return;
    }
    int index = key.charAt(d)-'a';
    if (root.children[index]==null)
      root.children[index] = new RWTrieNode();
    constructRWTrie(root.children[index],key,d+1);
  }

  public int numRabbits(int[] answers) {
    if (answers == null || answers.length == 0)
      return 0;
    Map<Integer, int[]> res = new HashMap<>();
    for (int a : answers)
      if (!res.containsKey(a + 1))
        res.put(a + 1, new int[]{1, 1});
      else {
        int[] inf = res.get(a + 1);
        if (inf[1] == a + 1) {
          inf[0]++;
          inf[1] = 1;
        } else
          inf[1]++;
        res.put(a + 1, inf);
      }
    int ans=0;
    for (Map.Entry entry:res.entrySet()){
      int[] inf = (int[])entry.getValue();
      ans+=inf[0]*(int)entry.getKey();
    }
    return ans;
  }

  public String makeLargestSpecial(String S) {
    int count =0, start=0;
    List<String> res = new LinkedList<>();
    for (int i=0;i<S.length();i++){
      if (S.charAt(i)=='1')
        count++;
      else
        count--;
      if (count==0){
        res.add("1"+makeLargestSpecial( S.substring(start+1,i))+"0");
        start = i+1;
      }
    }
    Collections.sort(res,Collections.reverseOrder());
    StringBuilder sb = new StringBuilder();
    for (String s:res)
      sb.append(s);
    return sb.toString();
  }

  public boolean canConstruct(String ransomNote, String magazine) {
    int[] res = new int[26];
    for (char c:magazine.toCharArray())
      res[c-'a']++;
    for (char r:ransomNote.toCharArray())
      if (res[r-'a']==0)
        return false;
      else
        res[r-'a']--;
    return true;
  }

  public int findShortestSubArray(int[] nums) {
    Map<Integer,Integer> left=new HashMap<>(),right=new HashMap<>(),count=new HashMap<>();
    for (int i=0;i<nums.length;i++){
      left.putIfAbsent(nums[i],i);
      right.put(nums[i],i);
      count.put(nums[i],count.getOrDefault(nums[i],0)+1);
    }
    int max = Collections.max(count.values()),ans=nums.length;
    for(Map.Entry entry:count.entrySet())
      if ((int)entry.getValue()==max)
        ans =Math.min(ans,right.get(entry.getKey())-left.get(entry.getKey())+1);
    return ans;
  }

  public List<List<Integer>> combinationSum3(int k, int n) {
    List<List<Integer>> res = new LinkedList<>();
    combinationSum3(res,new Stack<>(),n,k,1);
    return res;
  }

  private void combinationSum3(List<List<Integer>> res,Stack<Integer> path,int n,int k,int start){
    if (n==0){
      if (k==0)
        res.add(new LinkedList<>(path));
      return;
    }
    for (int i=start;i<=9;i++)
      if (n-i>=0){
        path.push(i);
        combinationSum3(res,path,n-i,k-1,i+1);
        path.pop();
      }
  }

  public int largestOverlap1(int[][] A, int[][] B) {
    int maxOverlap=0, N = A.length;
    for (int i=1-N;i<=N-1;i++)
      for (int j=1-N;j<=N-1;j++)
        maxOverlap = Math.max(maxOverlap,getOverlap(A,B,i,j));
    return maxOverlap;
  }

  private int getOverlap(int[][] A,int[][] B,int rowMove,int colMove){
    int N=A.length,ans=0;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        if (A[i][j]==1&&i+rowMove>=0&&i+rowMove<N&&j+colMove>=0&&j+colMove<N&&B[i+rowMove][j+colMove]==1)
          ans++;
    return ans;
  }

  public int largestOverlap(int[][] A, int[][] B) {
    List<int[]> la=new LinkedList<>();
    List<int[]> lb = new LinkedList<>();
    Map<String,Integer> count = new HashMap<>();
    int N = A.length,max=0;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++){
        if (A[i][j]==1)
          la.add(new int[]{i,j});
        if (B[i][j]==1)
          lb.add(new int[]{i,j});
      }
    for (int[] a:la)
      for (int[] b:lb){
        String key = (a[0]-b[0])+" "+(a[1]-b[1]);
        count.put(key,count.getOrDefault(key,0)+1);
      }
    for (Map.Entry e:count.entrySet())
      max = Math.max((int)e.getValue(),max);
    return max;
  }

  Map<Integer,Integer> NQcount;
  Map<Integer,List<Integer>> squares;
  public int numSquarefulPerms(int[] A) {
    NQcount = new HashMap<>();
    squares = new HashMap<>();
    int N = A.length;
    for (int i:A)
      NQcount.put(i,NQcount.getOrDefault(i,0)+1);
    for (int i:NQcount.keySet())
      squares.put(i,new LinkedList<>());
    for (int i:NQcount.keySet())
      for (int j:NQcount.keySet())
        if ((i+j)==Math.pow((int)Math.sqrt(i+j),2))
          squares.get(i).add(j);
    int ans=0;
    for (int i:squares.keySet())
      ans += btSquare(i,N-1);
    return ans;
  }

  private int btSquare(int i,int remain){
    if (remain==0)
      return 1;
    int ans=0;
    NQcount.put(i,NQcount.get(i)-1);
    for (int y:squares.get(i))
      if (NQcount.get(y)>0)
        ans+=btSquare(y,remain-1);
    NQcount.put(i,NQcount.get(i)+1);
    return ans;
  }

  public int numberOfBoomerangs(int[][] points) {
    int ans=0,N=points.length;
    Map<Integer,Integer> mp=new HashMap<>();
    for (int i=0;i<N;i++){
      mp.clear();
      for (int j=0;j<N;j++){
        if(j==i)
          continue;
        int dist= (int) (Math.pow(points[j][0]-points[i][0],2)+Math.pow(points[j][1]-points[i][1],2));
        int times=mp.getOrDefault(dist,0);
        mp.put(dist,times+1);
        if (times!=0)
          ans+=times*2;
      }
    }
    return ans;
  }

  public int firstUniqChar(String s) {
    if (s==null||s.equals(""))
      return -1;
    int[] count = new int[26];
    List<Integer> order = new ArrayList<>();
    for (char c:s.toCharArray()){
      int index=c-'a';
      count[index]++;
      order.add(index);
    }

    Integer[] res = order.toArray(new Integer[1]);
    for (int i=0;i<res.length;i++)
      if (count[res[i]]==1)
        return i;
    return -1;
  }

  public boolean pyramidTransition(String bottom, List<String> allowed) {
    Map<String,List<Character>> mp = new HashMap<>();
    String[] res=allowed.toArray(new String[0]);
    for (int i=0;i<res.length;i++)
      mp.computeIfAbsent(res[i].substring(0,2),c->new ArrayList<>()).add(res[i].charAt(2));
    return pyramidTransition(bottom,mp);
  }

  private boolean pyramidTransition(String bottom,Map<String,List<Character>> mp){
    if (bottom.length()==1)
      return true;
    List<String> upper = new LinkedList<>();
    collectUpper(upper,"",0,bottom,mp);
    for (String s:upper){
      if (s.length()!=bottom.length()-1)
        return false;
      if (pyramidTransition(s,mp))
        return true;
    }
    return false;
  }

  private void collectUpper(List<String> upper, String temp, int i, String bottom, Map<String, List<Character>> mp){
    if (i==bottom.length()-1){
      upper.add(temp);
      return;
    }
    String bot=bottom.substring(i,i+2);
    if (!mp.containsKey(bot))
      return;
    for (char c:mp.get(bot))
      collectUpper(upper,temp+String.valueOf(c),i+1,bottom,mp);
  }

  public List<Integer> postorderTraversal1(TreeNode root) {
    List<Integer> ans= new LinkedList<>();
    postorderTraversal(ans,root);
    return ans;
  }

  private void postorderTraversal(List<Integer> ls,TreeNode root){
    if (root==null)
      return;
    postorderTraversal(ls,root.left);
    postorderTraversal(ls,root.right);
    ls.add(root.val);
  }

  class BinaryTrie{
    int val;
    boolean isEnd;
    BinaryTrie one;
    BinaryTrie zero;
  }
  BinaryTrie MXroot;
  private void BinaryTrieInsert(int num){
    int mask=1<<30;
    BinaryTrie root=MXroot;
    for (int i=31;i>0;i--){
      int index=(mask&num)==0?0:1;
      if (index==1&&root.one==null)
        root.one=new BinaryTrie();
      else if (index==0&&root.zero==null)
        root.zero=new BinaryTrie();
      root=index==1?root.one:root.zero;
      mask>>=1;
    }
    root.isEnd=true;
    root.val=num;
  }

  public int findMaximumXOR(int[] nums) {
    if (nums.length==1)
      return 0;
    else if (nums.length==2)
      return nums[0]^nums[1];
    MXroot = new BinaryTrie();
    BinaryTrie root=MXroot;
    for (int n:nums)
      BinaryTrieInsert(n);
    while(root.one==null||root.zero==null)
      root=root.one!=null?root.one:root.zero;
    return getMaxXOR(root.one,root.zero);
  }

  private int getMaxXOR(BinaryTrie r1,BinaryTrie r2){
    if (r1.isEnd&&r2.isEnd)
      return r1.val^r2.val;
    if (r1.one==null)
      return getMaxXOR(r1.zero,r2.one!=null?r2.one:r2.zero);
    else if (r1.zero==null)
      return getMaxXOR(r1.one,r2.zero!=null?r2.zero:r2.one);
    else if(r2.one==null)
      return getMaxXOR(r1.one,r2.zero);
    else if (r2.zero==null)
      return getMaxXOR(r1.zero,r2.one);
    else
      return Math.max(getMaxXOR(r1.one,r2.zero),getMaxXOR(r1.zero,r2.one));
  }

  public List<Integer> postorderTraversal(TreeNode root) {
    if (root==null)
      return new LinkedList<>();
    Stack<TreeNode> st = new Stack<>();
    Set<TreeNode> hasVisited=new HashSet<>();
    List<Integer> ans =new LinkedList<>();
    TreeNode cur = root;
    st.push(root);
    while(!st.isEmpty()){
      if (hasVisited.contains(cur)){
        TreeNode temp = st.peek();
        if (temp.right!=null&& !hasVisited.contains(temp.right)){
          st.push(temp.right);
          cur= temp.right;
        }
        else {
          temp=st.pop();
          ans.add(temp.val);
          cur=temp;
        }
        continue;
      }
      hasVisited.add(cur);
      while(cur.left!=null){
        st.push(cur.left);
        hasVisited.add(cur.left);
        cur=cur.left;
      }
    }
    return ans;
  }

  public int minMoves(int[] nums) {
    int steps=0,min=Integer.MAX_VALUE;
    for (int n:nums)
      min = Math.min(n,min);
    for (int n:nums)
      steps+=n-min;
    return steps;
  }

  public boolean isOneBitCharacter(int[] bits) {
    int N = bits.length,i=0,label=2;
    while(i<N)
      if (bits[i]==1){
        i+=2;
        label=2;
      }
      else{
        i++;
        label=1;
      }
    return label==1;
  }

  public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> ans = new LinkedList<>();
    Stack<TreeNode> st = new Stack<>();
    TreeNode cur = root;
    while(!st.isEmpty()||cur!=null)
      if (cur!=null){
        ans.add(cur.val);
        st.push(cur);
        cur=cur.left;
      }
      else{
        TreeNode temp = st.pop();
        cur=temp.right;
      }
    return ans;
  }

  public List<Integer> preorderTraversalMorris(TreeNode root) {
    List<Integer> ans = new LinkedList<>();
    TreeNode cur = root;
    while(cur!=null){
      if (cur.left==null){
        ans.add(cur.val);
        cur=cur.right;
        continue;
      }
      TreeNode pre = cur.left;
      while(pre.right!=null&&pre.right!=cur)
        pre=pre.right;
      if (pre.right==null){
        pre.right=cur;
        ans.add(cur.val);
        cur=cur.left;
      }
      else{
        pre.right=null;
        cur=cur.right;
      }
    }
    return ans;
  }

  int[] FRid;
  int[] FRweight;
  public int[] findRedundantConnection(int[][] edges) {
    int N = edges.length;
    FRid = new int[N];
    FRweight = new int[N];
    for (int i=0;i<N;i++){
      FRid[i]=i;
      FRweight[i]=1;
    }
    for (int[] e:edges)
      if (FRunion(e[0]-1,e[1]-1))
        return e;
    return null;
  }

  private int FRfind(int id){
    int tempId=id;
    while (FRid[id]!=id)
      id=FRid[id];

    while(tempId!=id){
      int temp=FRid[tempId];
      FRid[tempId]=id;
      tempId=temp;
    }
    return id;
  }

  private boolean FRunion(int i1,int i2){
    int id1=FRfind(i1);
    int id2=FRfind(i2);
    if (id1==id2)
      return true;
    if (FRweight[id1]>=FRweight[id2]){
      FRid[id2]=id1;
      FRweight[id1]+=FRweight[id2];
    }
    else{
      FRid[id1]=id2;
      FRweight[id2]+=FRweight[id1];
    }
    return false;
  }

  Set<Integer> hasVisited;
  public int swimInWater1(int[][] grid) {
    int N=grid.length,t=0;
    hasVisited=new HashSet<>();
    hasVisited.add(0);
    while(true){
      for (int loc:hasVisited.toArray(new Integer[0]))
        if (grid[0][0]<=t&&SWexplore(grid,loc,t))
          return t;
      t++;
    }
  }

  private boolean SWexplore(int[][] grid,int loc,int t){
    int r=loc>>6,c=loc&63,N=grid.length;
    if (r==N-1&&c==N-1)
      return true;
    if (!hasVisited.contains(loc))
      hasVisited.add(loc);
    if (r-1>=0 &&grid[r-1][c]<=t && !hasVisited.contains((r-1)<<6 ^c))
      if(SWexplore(grid,(r-1)<<6 ^c,t))
        return true;
    if (r+1<N &&grid[r+1][c]<=t && !hasVisited.contains((r+1)<<6 ^c))
      if(SWexplore(grid,(r+1)<<6 ^c,t))
        return true;
    if (c-1>=0 &&grid[r][c-1]<=t && !hasVisited.contains(r<<6 ^(c-1)))
      if(SWexplore(grid,r<<6 ^(c-1),t))
        return true;
    if (c+1<N &&grid[r][c+1]<=t && !hasVisited.contains(r<<6 ^ (c+1)))
      if(SWexplore(grid,r<<6 ^(c+1),t))
        return true;
    return false;
  }

  boolean[][] SWmarked;
  public int swimInWater(int[][] grid) {
    int N=grid.length,start=grid[0][0],end=N*N-1;
    while(start<end){
      SWmarked=new boolean[N][N];
      int mid=start+(end-start)/2;
      if (SWreachable(grid,0,0,mid))
        end=mid;
      else
        start=mid+1;
    }
    return start;
  }

  private boolean SWreachable(int[][] grid,int r,int c,int t){
    int N=grid.length;
    if (r<0||r>=N||c<0||c>=N||grid[r][c]>t||SWmarked[r][c])
      return false;
    if (r==N-1&&c==N-1)
      return true;
    SWmarked[r][c]=true;
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int[] d:dirs)
      if (SWreachable(grid,r+d[0],c+d[1],t))
        return true;
    return false;
  }

  public int largestSumAfterKNegations(int[] A, int K) {
    PriorityQueue<Integer> pq=new PriorityQueue<>();
    for (int a:A)
      pq.offer(a);
    for (int i=0;i<K;i++)
      pq.offer(-pq.poll());
    Integer[] res=pq.toArray(new Integer[0]);
    int ans=0;
    for (int r:res)
      ans+=r;
    return ans;
  }

  public boolean rotateString(String A, String B) {
//    if (A==null||B==null||A.length()!=B.length())
//      return false;
//    else if (A.length()==1)
//      return A.charAt(0)==B.charAt(0);
//    else if (A.length()==0)
//      return true;
//    char[] a=A.toCharArray(),b=B.toCharArray();
//    int h=a[0],N=a.length;
//    for (int i=0;i<N;i++)
//      if (b[i]==h &&  A.substring(0,N-i).equals(B.substring(i)) && A.substring(N-i).equals(B.substring(0,i)))
//        return true;
//    return false;
    return A.length()==B.length()&&(A+A).contains(B);
  }

  public int[] nextGreaterElements(int[] nums) {
//    int N = nums.length;
//    int[] ans = new int[N];
//    Arrays.fill(ans,-1);
//    for (int i=0;i<N;i++)
//      for (int j=0;j<N;j++)
//        if (nums[(i+j)%N]>nums[i]){
//          ans[i]=nums[(i+j)%N];
//          break;
//        }
//    return ans;
    int N=nums.length;
    int[] ans = new int[N];
    Stack<Integer> st = new Stack<>();
    Arrays.fill(ans,-1);
    for (int i=0;i<2*N;i++){
      int curVal=nums[i%N];
      while(!st.isEmpty()&&nums[st.peek()]<curVal)
        ans[st.pop()]=curVal;
      if (i<N)
        st.push(i);
    }
    return ans;
  }

  public boolean reorderedPowerOf2(int N) {
    int len=getDigitsLength(N);
    return isPowerOf2(N,len,len-1);
  }

  private boolean isPowerOf2(int N,int len,int index){
    if (index==0)
      return (N&(N-1))==0;
    for (int i=index;i>=0;i--){
      if (index==len-1&& (int)(N/Math.pow(10,i))%10==0)
        continue;
      int temp=exchangeDigits(N,index,i);
      if(isPowerOf2(temp,len,index-1))
        return true;
    }
    return false;
  }

  private int getDigitsLength(int N){
    int ans=0;
    while(N>0){
      ans++;
      N/=10;
    }
    return ans;
  }

  private int exchangeDigits(int N,int i,int j){
    int iDig=(int)(Math.pow(10,i)),jDig=(int)(Math.pow(10,j));
    int iNum=N/iDig%10,jNum=N/jDig%10;
    N = N-iNum*iDig+iNum*jDig-jNum*jDig+jNum*iDig;
    return N;
  }

  public int sumOfLeftLeaves(TreeNode root) {
    if (root==null)
      return 0;
    int ans=0;
    if (root.left!=null&&root.left.left==null&&root.left.right==null)
      ans += root.left.val;
    ans += sumOfLeftLeaves(root.left);
    ans += sumOfLeftLeaves(root.right);
    return ans;
  }

  public int sumOfLeftLeaves1(TreeNode root) {
    Stack<TreeNode> st= new Stack<>();
    TreeNode cur = root;
    int ans=0;
    while(!st.isEmpty() || cur!=null)
      if (cur!=null){
        st.push(cur);
        if (cur.left!=null&&cur.left.left==null&&cur.left.right==null)
          ans += cur.left.val;
        cur=cur.left;
      }
      else{
        TreeNode temp=st.pop();
        cur=temp.right;
      }
    return ans;
  }

  public int sumOfLeftLeaves2(TreeNode root) {
    TreeNode cur = root;
    int ans=0;
    while(cur!=null)
      if (cur.left==null)
        cur=cur.right;
      else{
        TreeNode pre=cur.left;
        while(pre.right!=null&&pre.right!=cur)
          pre=pre.right;
        if (pre.right!=cur){
          if (cur.left!=null&&cur.left.left==null&&cur.left.right==null)
            ans += cur.left.val;
          pre.right=cur;
          cur=cur.left;
        }
        else{
          pre.right=null;
          cur=cur.right;
        }
      }
    return ans;
  }

  public int maxCount(int m, int n, int[][] ops) {
    int mr=m,mc=n;
    for (int[] o:ops){
      mr=Math.min(mr,o[0]);
      mc=Math.min(mc,o[1]);
    }
    return mr*mc;
  }

  public int kthSmallest(TreeNode root, int k) {
    TreeNode cur=root;
    while(cur!=null)
      if (cur.left==null){
        if (--k==0)
          return cur.val;
        cur=cur.right;
      }
      else{
        TreeNode pre=cur.left;
        while(pre.right!=null&&pre.right!=cur)
          pre=pre.right;
        if (pre.right!=cur){
          pre.right=cur;
          cur=cur.left;
        }
        else{
          if (--k==0)
            return cur.val;
          pre.right=null;
          cur=cur.right;
        }
      }
    return 0;
  }

  public int minAreaRect(int[][] points) {
    Map<Integer,Set<Integer>> cols=new TreeMap<>();
    int ans=Integer.MAX_VALUE;
    for (int[] p:points)
      cols.computeIfAbsent(p[0],i->new HashSet<>()).add(p[1]);
    for (int[] p1:points)
      for (int[] p2:points)
        if (p1[0]==p2[0]||p1[1]==p2[1])
          continue;
        else if (cols.get(p1[0]).contains(p2[1]) && cols.get(p2[0]).contains(p1[1]))
          ans = Math.min(ans,Math.abs((p1[0]-p2[0])*(p1[1]-p2[1])));
    return ans<Integer.MAX_VALUE?ans:0;
  }

  public int[] constructRectangle(int area) {
    if (area==0||area==1)
      return new int[]{area,area};
    int sqrt= (int)Math.sqrt(area);
    if (sqrt*sqrt==area)
      return new int[]{sqrt,sqrt};
//    int L=sqrt+1,product;
//    while((product=L*sqrt)!=area)
//      if (product<area)
//        L++;
//      else
//        sqrt--;
    while(area%sqrt!=0)
      sqrt--;
    return new int[]{area/sqrt,sqrt};
  }

  public int[][] imageSmoother(int[][] M) {
    int R = M.length,C=M[0].length,startSum=0;
    int[][] padding= new int[R+2][C+2];
    for (int i=1;i<=R;i++)
      for (int j=1;j<=C;j++)
        padding[i][j]=M[i-1][j-1];
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++)
        startSum += padding[i][j];
    for (int i=1;i<=R;i++){
      if ((i&1)==1)
        for (int j=1;j<=C;j++){
          if (j==1){
            M[i-1][0]=startSum/getDivideNum(R,C,i-1,j-1);
            continue;
          }
          for (int k=-1;k<=1;k++){
            startSum+=padding[i+k][j+1];
            startSum-=padding[i+k][j-2];
          }
          M[i-1][j-1]=startSum/getDivideNum(R,C,i-1,j-1);
        }
      else
        for (int j=C;j>=1;j--){
          if (j==C){
            M[i-1][j-1]=startSum/getDivideNum(R,C,i-1,j-1);
            continue;
          }
          for (int k=-1;k<=1;k++){
            startSum+=padding[i+k][j-1];
            startSum-=padding[i+k][j+2];
          }
          M[i-1][j-1] = startSum/getDivideNum(R,C,i-1,j-1);
        }
      if (i==R)
        continue;
      int c=(i&1)==1?C:1;
      for (int k=-1;k<=1;k++) {
        startSum += padding[i+2][c+k];
        startSum-= padding[i-1][c+k];
      }
    }
    return M;
  }

  private int getDivideNum(int R,int C,int r,int c){
    if (R==1&&C==1)
      return 1;
    if (R==1||C==1){
      int other = R==1?C:R;
      if (other<3)
        return 2;
      if ((other==R&&r>0&&r<R-1)||(other==C&&c>0&&c<C-1))
        return 3;
      else return 2;
    }
    if ((r==0||r==R-1)&& (c==0||c==C-1))
      return 4;
    else if ((r==0||r==R-1)|| (c==0||c==C-1))
      return 6;
    else
      return 9;
  }

  public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
//    Arrays.sort(A);
//    Arrays.sort(B);
//    Arrays.sort(C);
//    Arrays.sort(D);
//    int N=A.length,count=0;
//    for (int i=0;i<N;i++){
//      if (A[i]+B[0]+C[0]+D[0]>0)
//        break;
//      for (int j=0;j<N;j++){
//        if (A[i]+B[j]+C[0]+D[0]>0)
//          break;
//        for (int k=0;k<N;k++){
//          if (A[i]+B[j]+C[k]+D[0]>0)
//            break;
//          for (int l=0;l<N;l++){
//            int sum=A[i]+B[j]+C[k]+D[l];
//            if (sum==0)
//              count++;
//            else if (sum>0)
//              break;
//          }
//        }
//      }
//    }
//    return count;
    int N=A.length,count=0;
    Map<Integer,Integer> sumCD = new HashMap();
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        sumCD.put(C[i]+D[j],sumCD.getOrDefault(C[i]+D[j],0)+1);
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        count+= sumCD.getOrDefault(-(A[i]+B[j]),0);
    return count;
  }

  public String intToRoman(int num) {
//    Map<Integer,String> mp = new HashMap<>();
//    mp.put(1,"I");
//    mp.put(4,"IV");
//    mp.put(5,"V");
//    mp.put(9,"IX");
//    mp.put(10,"X");
//    mp.put(40,"XL");
//    mp.put(50,"L");
//    mp.put(90,"XC");
//    mp.put(100,"C");
//    mp.put(400,"CD");
//    mp.put(500,"D");
//    mp.put(900,"CM");
//    mp.put(1000,"M");
//    StringBuilder ans=new StringBuilder();
//    List<Integer> numDigits= new ArrayList<>();
//    while(num>0){
//      numDigits.add(num%10);
//      num/=10;
//    }
//    for (int i=numDigits.size()-1;i>=0;i--){
//      int val = numDigits.get(i),base=(int)Math.pow(10,i);
//      if (val<4){
//        String c=mp.get(base);
//        while(val-->0)
//          ans.append(c);
//      }
//      else if (val==4)
//        ans.append(mp.get(4*base));
//      else if (val<9){
//        ans.append(mp.get(5*base));
//        String c=mp.get(base);
//        while(val-->5)
//          ans.append(c);
//      }
//      else
//        ans.append(mp.get(9*base));
//    }
//    return ans.toString();
    int[] values=new int[]{1000,900,500,400,100,90,50,40,10,9,5,4,1};
    String[] romans=new String[]{"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
    StringBuilder sb=new StringBuilder();
    for (int i=0;i<values.length;i++)
      while(num>=values[i]){
        num-=values[i];
        sb.append(romans[i]);
      }
    return sb.toString();
  }

  public int findJudge(int N, int[][] trust) {
    int[] record = new int[N];
    for (int[] t:trust){
      record[t[0]-1]=-1;
      if (record[t[1]-1]!=-1)
        record[t[1]-1]++;
    }
    int judge=0;
    for (int i=0;i<N;i++)
      if (record[i]==N-1)
        if (judge==0)
          judge = i+1;
        else
          return -1;
    return judge==0?-1:judge;
  }

  public int findContentChildren(int[] g, int[] s) {
    Arrays.sort(s);
    Arrays.sort(g);
    int ig=0,is=0,ans=0;
    while(ig<g.length&&is<s.length){
      int G=g[ig],S=s[is];
      if (S>=G){
        ans++;
        ig++;
        is++;
      }
      else
        is++;
    }
    return ans;
  }

  public int maxProfit(int[] prices, int fee) {
    if (prices==null||prices.length==1)
      return 0;
    int N =prices.length,sell=0,buy=-prices[0];
    for (int i=1;i<N;i++){
      sell=Math.max(sell,buy+prices[i]-fee);
      buy = Math.max(buy,sell-prices[i]);
    }
    return sell;
  }

  public String[] findRelativeRanks(int[] nums) {
    if (nums==null)
      return new String[0];
    if (nums.length==1)
      return new String[]{"Gold Medal"};
    int N=nums.length;
    String[] ans = new String[N];
    Map<Integer,Integer> mp=new HashMap<>();
    for (int i=0;i<N;i++)
      mp.put(nums[i],i);
    Arrays.sort(nums);
    for (int i=N-1;i>=0;i--){
      int loc=mp.get(nums[i]);
      if (i<N-3)
        ans[loc]=String.valueOf(N-i);
      else if (i==N-1)
        ans[loc]="Gold Medal";
      else if (i==N-2)
        ans[loc]="Silver Medal";
      else
        ans[loc]="Bronze Medal";
    }
    return ans;
  }

  public int missingNumber(int[] nums) {
    int N=nums.length;
    int expectedSum=N*(N+1)/2,sum=0;
    for (int n:nums)
      sum+=n;
    return expectedSum-sum;
  }

  public ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
    if (l1==null&&l2==null)
      return null;
    else if (l1==null)
      return l2;
    else if (l2==null)
      return l2;
    ListNode l1End=reverseLinkedList(l1),l2End=reverseLinkedList(l2),reverseAns=new ListNode((l1End.val+l2End.val)%10);
    ListNode cur=reverseAns;
    int carry=(l1End.val+l2End.val)/10;
    l1End=l1End.next;
    l2End=l2End.next;
    while(l1End!=null||l2End!=null||carry!=0){
      int l1Val=l1End==null?0:l1End.val;
      int l2Val=l2End==null?0:l2End.val;
      int val=carry+l1Val+l2Val;
      carry=val/10;
      val%=10;
      ListNode temp=new ListNode(val);
      cur.next=temp;
      cur=cur.next;
      l1End=l1End==null?null:l1End.next;
      l2End=l2End==null?null:l2End.next;
    }
    return reverseLinkedList(reverseAns);
  }

  private ListNode reverseLinkedList(ListNode head){
    ListNode cur=head,last=null;
    while(cur!=null){
      ListNode temp=cur.next;
      cur.next=last;
      last=cur;
      cur=temp;
    }
    return last;
  }

  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    if (l1==null)
      return l2;
    else if (l1==null)
      return l1;
    int carry=0;
    Stack<Integer> v1=new Stack<>(),v2=new Stack<>();
    while(l1!=null){
      v1.push(l1.val);
      l1=l1.next;
    }
    while(l2!=null){
      v2.push(l2.val);
      l2=l2.next;
    }
    ListNode cur=new ListNode(0);
    while (!v1.isEmpty() || !v2.isEmpty()||carry!=0){
      int val1=v1.isEmpty()?0:v1.pop();
      int val2=v2.isEmpty()?0:v2.pop();
      int val=val1+val2+carry;
      carry=val/10;
      cur.val=val%10;
      ListNode head=new ListNode(0);
      head.next=cur;
      cur=head;
    }
    return cur.next;
  }

  class Solution {
    private int[] oriVal;
    private int[] curVal;
    private Random r;

    public Solution(int[] nums) {
      oriVal=Arrays.copyOf(nums,nums.length);
      curVal=nums;
      r=new Random();
    }

    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
      curVal= oriVal.clone();
      return curVal;
    }

    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
      int N = oriVal.length;
      for (int i=N;i>1;i--)
        exchange(curVal,i-1,r.nextInt(i));
      return curVal;
    }
  }

  public double mincostToHireWorkers1(int[] quality, int[] wage, int K) {
    return mincostToHireWorkers(quality,wage,new Stack<>(),0,K);
  }

  private double mincostToHireWorkers(int[] quality,int[] wage,Stack<Integer> path,int start,int K){
    if (path.size()==K){
      return MHMinPayment(quality,wage,path.toArray(new Integer[0]));
    }
    double ans=Double.MAX_VALUE;
    int curLen=path.size();
    for (int i=start;i<=quality.length-K+curLen;i++){
      path.push(i);
      ans=Math.min(ans,mincostToHireWorkers(quality,wage,path,i+1,K));
      path.pop();
    }
    return ans;
  }

  private double MHMinPayment(int[] quality, int[] wage,Integer[] people){
    double maxProduct=Double.MIN_VALUE,ans=0;
    for (int i:people){
      double temp=(double)wage[i]/(double)quality[i];
      if (temp>maxProduct){
        maxProduct=temp;
      }
    }
    for (int i:people)
      ans += (double)quality[i]*maxProduct;
    return ans;
  }

  public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
    int N=quality.length;
    double[][] res=new double[N][2];
    for (int i=0;i<N;i++){
      res[i][0]=(double)wage[i]/(double)quality[i];
      res[i][1]=(double)quality[i];
    }
    PriorityQueue<Double> pq=new PriorityQueue<>();
    double min=Double.MAX_VALUE,sum=0;
    Arrays.sort(res,(a,b)->Double.compare(a[0],b[0]));
    for (double[] r:res){
      sum+=r[1];
      pq.offer(-r[1]);
      if (pq.size()>K)
        sum+=pq.poll();
      if (pq.size()==K)
        min=Math.min(min,sum*r[0]);
    }
    return min;
  }

  public int longestPalindrome(String s) {
    if (s==null||s.length()==0)
      return 0;
    else if (s.length()==1)
      return 1;
    int[] res=new int[52];
    int doubleNum=0,N=s.length();
    char[] cs=s.toCharArray();
    for (int i=0;i<N;i++){
      int index=Character.isUpperCase(cs[i])?25+cs[i]-'A':cs[i]-'a';
      if (++res[index]==2){
        doubleNum++;
        res[index]=0;
      }
    }
    return N==doubleNum<<1?doubleNum<<1:(doubleNum<<1)+1;
  }

  public List<List<Integer>> largeGroupPositions(String S) {
    List<List<Integer>> res=new ArrayList<>();
    char[] cs=S.toCharArray();
    int start=0;
    for (int i=0;i<cs.length;i++){
      if (cs[i]!=cs[start]){
        if (i-start>=3)
          res.add(Arrays.asList(new Integer[]{start,i-1}));
        start=i;
      }
      else if (i==cs.length-1&&i-start>=2)
        res.add(Arrays.asList(new Integer[]{start,i}));
    }
    return res;
  }

  class TimeMap {
    private Map<String,TreeMap<Integer,String>> data;
    public TimeMap() {
      data=new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
      data.computeIfAbsent(key,i->new TreeMap<>()).put(timestamp,value);
    }

    public String get(String key, int timestamp) {
      TreeMap<Integer,String> tm=data.getOrDefault(key,null);
      if (tm==null)
        return "";
      Map.Entry<Integer,String> res=tm.floorEntry(timestamp);
      return res==null?"":res.getValue();
    }
  }

  class RLEIterator {
    int[] elements;
    int[] occurs;
    int cur;
    public RLEIterator(int[] A) {
      int N=A.length;
      cur=0;
      elements=new int[N/2];
      occurs=new int[N/2];
      for (int i=0;i<N;i+=2){
        elements[i>>1]=A[i+1];
        occurs[i>>1]=A[i];
      }
    }

    public int next(int n) {
      while(n>0){
        if (cur>=occurs.length)
          return -1;
        int curTimes=occurs[cur];
        if (curTimes>=n){
          occurs[cur]-=n;
          return elements[cur];
        }
        else{
          n-=curTimes;
          cur++;
        }
      }
      return -1;
    }
  }

  public int flipLights(int n, int m) {
    if (n<=0)
      return 0;
    else if (n==1)
      return m==0?1:2;
    else if (n==2)
      return m==0?1:m==1?3:4;
    else
      return m==0?1:m==1?4:m==3?7:8;
  }

  public String[] findRestaurant(String[] list1, String[] list2) {
    Map<String,Integer> mp = new HashMap<>();
    List<String> res=new ArrayList<>();
    for (int i=0;i<list1.length;++i)
      mp.put(list1[i],i);
    int indexSum=Integer.MAX_VALUE;
    for (int i=0;i<list2.length;++i){
      int l1Index=mp.getOrDefault(list2[i],-1);
      if (l1Index==-1)
        continue;
      l1Index+=i;
      if (l1Index<indexSum){
        indexSum=l1Index;
        res.clear();
      }
      if (l1Index==indexSum)
        res.add(list2[i]);
    }
    return res.toArray(new String[0]);
  }

  public int[] intersect(int[] nums1, int[] nums2) {
    Map<Integer,Integer> mp=new HashMap<>();
    List<Integer> ls=new ArrayList<>();
    for (int n:nums1)
      mp.put(n,mp.getOrDefault(n,0)+1);
    for (int n:nums2){
      int remain=mp.getOrDefault(n,0);
      if (remain>0){
        ls.add(n);
        mp.put(n,mp.get(n)-1);
      }
    }

    int[] ans=new int[ls.size()];
    for (int i=0;i<ls.size();i++)
      ans[i]=ls.get(i);
    return ans;
//    return ls.stream().mapToInt(a->a).toArray();
  }

  public List<Integer> diffWaysToCompute(String input) {
    List<Integer> ls=new ArrayList<>();
    for (int i=0;i<input.length();i++){
      char cur=input.charAt(i);
      if (cur=='+'||cur=='-'||cur=='*'){
        List<Integer> left=diffWaysToCompute(input.substring(0,i));
        List<Integer> right=diffWaysToCompute(input.substring(i+1));
        for (int l:left)
          for (int r:right)
            switch (cur){
              case '+':
                ls.add(l+r);
                break;
              case '-':
                ls.add(l-r);
                break;
              case '*':
                ls.add(l*r);
                break;
            }
      }
    }
    if (ls.size()==0)
      ls.add(Integer.valueOf(input));
    return ls;
  }

  class Solution382 {
    Random r;
    List<Integer> data;
    int length;
    public Solution382(ListNode head) {
      r=new Random();
      data=new ArrayList();
      while(head!=null){
        data.add(head.val);
        head=head.next;
      }
      length=data.size();
    }

    /** Returns a random node's value. */
    public int getRandom() {
      int id=r.nextInt(length);
      return data.get(id);
    }
  }

  class Solution382_ {
    Random r;
    ListNode head;
    public Solution382_(ListNode head) {
      r=new Random();
      this.head=head;
    }

    /** Returns a random node's value. */
    public int getRandom() {
      ListNode cur=head.next;
      int i=2,ans=head.val;
      while(cur!=null){
        int exchangeIndex=r.nextInt(i);
        if (exchangeIndex==0)
          ans=cur.val;
        i++;
        cur=cur.next;
      }
      return ans;
    }
  }

  class MyCircularDeque {
    int[] data;
    int length;
    int front,last,curSize;

    /** Initialize your data structure here. Set the size of the deque to be k. */
    public MyCircularDeque(int k) {
      if (k<=0)
        throw new IllegalArgumentException();
      length=k;
      data=new int[length];
      front=curSize=0;
      last=length-1;
      Arrays.fill(data,-1);
    }

    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    public boolean insertFront(int value) {
      if (curSize==length)
        return false;
      front=front==0?length-1:front-1;
      data[front]=value;
      curSize++;
      return true;
    }

    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    public boolean insertLast(int value) {
      if (curSize==length)
        return false;
      last=(last+1)%length;
      data[last]=value;
      curSize++;
      return true;
    }

    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    public boolean deleteFront() {
      if (curSize==0)
        return false;
      data[front]=-1;
      front=(front+1)%length;
      curSize--;
      return true;
    }

    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    public boolean deleteLast() {
      if (curSize==0)
        return false;
      data[last]=-1;
      last= last==0?length-1:last-1;
      curSize--;
      return true;
    }

    /** Get the front item from the deque. */
    public int getFront() {
      return data[front];
    }

    /** Get the last item from the deque. */
    public int getRear() {
      return data[last];
    }

    /** Checks whether the circular deque is empty or not. */
    public boolean isEmpty() {
      return curSize==0;
    }

    /** Checks whether the circular deque is full or not. */
    public boolean isFull() {
      return curSize==length;
    }
  }

  public int maxCoins(int[] nums) {
    int N=nums.length;
    int[] res=new int[N+2];
    for (int i=0;i<N+2;i++){
      if (i==0||i==N+1)
        res[i]=1;
      else
        res[i]=nums[i-1];
    }
    int[][] dp=new int[N+2][N+2];
    for (int k=2;k<N+2;++k)
      for (int left=0;left<N+2-k;++left){
        int right=left+k;
        for (int i=left+1;i<right;++i)
          dp[left][right]=Math.max(dp[left][right],res[left]*res[i]*res[right]+dp[left][i]+dp[i][right]);
      }
    return dp[0][N+1];
  }

  public int findTilt(TreeNode root) {
    List<Integer> tilt=new ArrayList<>();
    findTilt(root,tilt);
    int ans=0;
    for (int i:tilt)
      ans+=i;
    return ans;
  }

  private int findTilt(TreeNode root,List<Integer> tilt){
    if (root==null)
      return 0;
    int left=findTilt(root.left,tilt);
    int right=findTilt(root.right,tilt);
    tilt.add(Math.abs(left-right));
    return left+right+root.val;
  }

  public int maxProfit1(int[] prices) {
    if (prices==null||prices.length==0||prices.length==1)
      return 0;
    int profit=0,max=prices[prices.length-1];
    for (int i=prices.length-2;i>=0;i--){
      profit=Math.max(profit,max-prices[i]);
      max=Math.max(prices[i],max);
    }
    return profit;
  }

  class Solution398 {
    private Random r;
    private Map<Integer,List<Integer>> index;
    public Solution398(int[] nums) {
      r=new Random();
      index = new HashMap<>();
      for (int i=0;i<nums.length;i++)
        index.computeIfAbsent(nums[i],a->new ArrayList<>()).add(i);
    }

    public int pick(int target) {
      List<Integer> ids = index.get(target);
      int pick=r.nextInt(ids.size());
      return ids.get(pick);
    }
  }

  class Solution398_ {
    private Random r;
    private int[] index;
    public Solution398_(int[] nums) {
      r=new Random();
      index = nums;
    }

    public int pick(int target) {
      int ans=-1,count=0;
      for (int i=0;i<index.length;++i)
        if (index[i]!=target)
          continue;
        else
          ans=r.nextInt(++count)==0?i:ans;
      return ans;
    }
  }

  public int minCostClimbingStairs(int[] cost) {
    int N=cost.length;
    int[] dp=new int[N];
    dp[N-1]=cost[N-1];
    dp[N-2]=cost[N-2];
    for (int i=N-3;i>=0;i--)
      dp[i]=Math.min(dp[i+1],dp[i+2])+cost[i];
    return Math.min(dp[0],dp[1]);
  }

  int DBT=0;
  public int diameterOfBinaryTree(TreeNode root) {
    DBTGetDepth(root);
    return DBT;
  }

  private int DBTGetDepth(TreeNode root){
    if (root==null)
      return 0;
    int leftDepth=DBTGetDepth(root.left);
    int rightDeptth=DBTGetDepth(root.right);
    DBT = Math.max(leftDepth+rightDeptth,DBT);
    return Math.max(leftDepth,rightDeptth)+1;
  }

  public int kthSmallest(int[][] matrix, int k) {
    if (k==1)
     return matrix[0][0];
    int R=matrix.length,C=matrix[0].length;
    boolean[][] marked=new boolean[R][C];
    PriorityQueue<int[]> pq=new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return matrix[a[0]][a[1]]-matrix[b[0]][b[1]];
      }
    });
    pq.offer(new int[]{0,0});
    for (int i=0;i<k-1;i++){
      int[] loc=pq.poll();
      if (loc[0]+1<R && !marked[loc[0]+1][loc[1]]){
        pq.offer(new int[]{loc[0]+1,loc[1]});
        marked[loc[0]+1][loc[1]]=true;
      }
      if (loc[1]+1<C && !marked[loc[0]][loc[1]+1]){
        pq.offer(new int[]{loc[0],loc[1]+1});
        marked[loc[0]][loc[1]+1]=true;
      }
    }
    int[] index = pq.poll();
    return matrix[index[0]][index[1]];
  }

  public int kthSmallest1(int[][] matrix, int k) {
    if (k == 1)
      return matrix[0][0];
    int N = matrix.length;
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return matrix[a[0]][a[1]] - matrix[b[0]][b[1]];
      }
    });
    for (int i=0;i<N;i++)
      pq.offer(new int[]{0,i});
    for (int i=0;i<k-1;i++){
      int[] loc=pq.poll();
      if (loc[0]==N-1)
        continue;
      pq.offer(new int[]{loc[0]+1,loc[1]});
    }
    int[] index = pq.poll();
    return matrix[index[0]][index[1]];
  }

  public int kthSmallest2(int[][] matrix, int k) {
    if (k == 1)
      return matrix[0][0];
    int N=matrix.length,lo=matrix[0][0], hi=matrix[N-1][N-1]+1;
    while (lo<hi){
      int count=0,j=N-1,mid=lo+((hi-lo)>>1);
      for (int i=0;i<N;i++){
       while (j>=0 && matrix[i][j]>mid)
         j--;
       count+=j+1;
      }
      if (count<k)
        lo=mid+1;
      else
        hi=mid;
    }
    return lo;
  }

  public int findDuplicate(int[] nums) {
    int N=nums.length,lo=1,hi=N;
    while(lo<hi){
      int mid=lo+((hi-lo)>>1),smaller=0,bigger=0;
      for (int n:nums)
        if (n<mid)
          smaller++;
        else if (n>mid)
          bigger++;
      if (smaller>mid-1)
        hi=mid;
      else if (bigger>N-mid-1)
        lo=mid+1;
      else
        return mid;
    }
    return -1;
  }

  public int findDuplicate1(int[] nums) {
    if (nums==null||nums.length<=1)
      throw new IllegalArgumentException();
    int slow=0,fast=0;
    do {
      slow = nums[slow];
      fast = nums[nums[fast]];
    }while (slow!=fast);
    fast=0;
    while(slow!=fast){
      slow=nums[slow];
      fast=nums[fast];
    }
    return slow;
  }

  public int minFlipsMonoIncr(String S) {
    if (S.length()==1)
      return 0;
    char[] cs = S.toCharArray();
    int N=cs.length,zeroCount = cs[N-1]=='0'?1:0;
    int[] dp=new int[N];
    dp[N-1]=0;
    for (int i=N-2;i>=0;i--)
      if (cs[i]=='0')
        dp[i] = Math.min(dp[i+1],zeroCount+++1);
      else
        dp[i]=Math.min(dp[i+1]+1,zeroCount);
    return dp[0];
  }

  public int totalHammingDistance(int[] nums) {
    int N=nums.length,ans=0;
    for (int i=0;i<N;i++)
      for (int j=i+1;j<N;j++)
        ans += getHammingDistance(nums[i],nums[j]);
    return ans;
  }

  private int getHammingDistance(int a,int b){
    int res=a^b,ans=0;
    while(res >0){
      ans++;
      res=res &(res-1);
    }
    return ans;
  }

  public int totalHammingDistance1(int[] nums) {
    int N=nums.length,ans=0,mask=1;
    for (int i=0;i<32;i++){
      int isZeroNum=0;
      for (int j=0;j<N;j++)
        if ((nums[j] & mask)==0)
          isZeroNum++;
      ans += isZeroNum*(N-isZeroNum);
      mask<<=1;
    }
    return ans;
  }

  public ListNode[] splitListToParts(ListNode root, int k) {
    ListNode[] ans = new ListNode[k];
    ListNode cur=root,c=root;
    int len=0;
    while (c!=null){
      len++;
      c=c.next;
    }
    int count=0,eachCapacity=len/k,remain=len%k;
    for (int i=0;i<k;i++){
      int num= i<remain?eachCapacity+1:eachCapacity;
      ans[i]=num>0?cur:null;
      if (cur!=null)
        cur=cur.next;
      if (ans[i]!=null)
        ans[i].next=null;
      ListNode ansCur=ans[i];
      for (int j=1;j<num;j++){
        ansCur.next=cur;
        ansCur=ansCur.next;
        cur=cur.next;
        ansCur.next=null;
      }
    }
    return ans;
  }

  public int orangesRotting(int[][] grid) {
    Queue<int[]> q=new LinkedList<>();
    int time=0,R=grid.length,C=grid[0].length,freshNum=0;
    int[][] dirs = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++)
        switch (grid[i][j]){
          case 1:
            freshNum++;
            break;
          case 2:
            q.offer(new int[]{i,j});
            break;
        }
    if (freshNum==0)
      return 0;
    while(!q.isEmpty()){
      int size = q.size();
      for (int i=0;i<size;i++){
        int[] loc = q.poll();
        for (int[] d:dirs){
          int[] temp=new int[]{loc[0]+d[0],loc[1]+d[1]};
          if (temp[0]>=0 && temp[0]<R && temp[1]>=0 && temp[1]<C && grid[temp[0]][temp[1]]==1){
            grid[temp[0]][temp[1]]=2;
            q.offer(temp);
            freshNum--;
          }
        }
      }
      time++;
    }
    return freshNum==0?time-1:-1;
  }

  public int search(int[] nums, int target) {
    int N=nums.length;
    if (nums==null||nums.length==0||target<nums[0]||target>nums[N-1])
      return -1;
    if (nums.length==1)
      return target==nums[0]?0:-1;
    int lo = 0,hi = N-1;
    while(lo<=hi){
      int mid=lo+((hi-lo)>>1);
      if (nums[mid]==target)
        return mid;
      else if (nums[mid]>target)
        hi=mid-1;
      else
        lo = mid+1;
    }
    return -1;
  }

  public boolean isNStraightHand(int[] hand, int W) {
    int N=hand.length,groups = N/W;
    if (N%W !=0)
      return false;
    Arrays.sort(hand);
    int[][] res=new int[groups][2];
    for (int i=0;i<N;i++){
      int cur = hand[i];
      for (int j=0;j<groups;j++)
        if (res[j][1]==0 || (res[j][1]<W && cur-res[j][0]==1)){
          res[j][1]++;
          res[j][0]=cur;
          break;
        }
    }
    for (int i=0;i<groups;i++)
      if (res[i][1]!=W)
        return false;
    return true;
  }

  public boolean isNStraightHand1(int[] hand, int W) {
    Map<Integer,Integer> mp = new TreeMap<>();
    for (int h:hand)
      mp.put(h,mp.getOrDefault(h,0)+1);
    for (Map.Entry<Integer,Integer> entry:mp.entrySet()){
      if (entry.getValue()==0)
        continue;
      for (int i=W-1;i>=0;--i){
        int num;
        if ((num=mp.getOrDefault(entry.getKey()+i,0))<entry.getValue())
          return false;
        mp.put(entry.getKey()+i,num-entry.getValue());
      }
    }
    return true;
  }

  public boolean isNStraightHand2(int[] hand, int W) {
    int[] res = new int[W];
    for (int h:hand)
      res[h%W]++;
    for (int i=1;i<W;i++)
      if (res[i]!=res[i-1])
        return false;
    return true;
  }

  public ListNode oddEvenList(ListNode head) {
    if (head==null)
      return null;
    ListNode cur = head.next,lastOdd=head,firstEven=head.next,lastEven=head.next;
    int count=2;
    while (cur!=null)
      if ((count++&1)==0){
        lastEven = cur;
        cur = cur.next;
      }
      else{
        lastEven.next = cur.next;
        ListNode temp = cur;
        cur=cur.next;
        temp.next=lastOdd.next;
        lastOdd.next=temp;
        lastOdd=lastOdd.next;
      }
    return head;
  }

  public int findLongestChain(int[][] pairs) {
    if (pairs.length==1)
      return 1;
    Arrays.sort(pairs,(a,b)->a[1]-b[1]);
    int cur = Integer.MIN_VALUE,count=0;
    for (int[] p:pairs)
      if (p[0]>cur){
        cur=p[1];
        count++;
    }
    return count;
  }

  public List<List<Integer>> levelOrderBottom(TreeNode root) {
    if (root==null)
      return new LinkedList<>();
    LinkedList<List<Integer>> res = new LinkedList<>();
    Queue<TreeNode> q=new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()){
      int size = q.size();
      List<Integer> layer = new ArrayList<>();
      for (int i=0;i<size;i++){
        TreeNode temp=q.poll();
        layer.add(temp.val);
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
      }
      res.offerFirst(layer);
    }
    return res;
  }

  public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
    return !(rec2[2]<=rec1[0]||rec2[0]>=rec1[2]||rec2[1]>=rec1[3]||rec2[3]<=rec1[1]);
  }


  public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
    Map<Integer,Integer> memo = new HashMap<>();
    int min = sO(price,special,needs,memo);
    int dirCost=0;
    for (int i=0;i<price.size();i++)
      dirCost+=needs.get(i)*price.get(i);
    return min<dirCost?min:dirCost;
  }

  private int sO(List<Integer> price, List<List<Integer>> special, List<Integer> needs,Map<Integer,Integer> memo) {
    int needNum=0,N=price.size();
    for (int i=0;i<N;i++)
      needNum+=Math.pow(10,i)*needs.get(i);
    if (needNum==0)
      return 0;
    if (memo.containsKey(needNum))
      return memo.get(needNum);
    int min =Integer.MAX_VALUE;
    offer:for (List<Integer> sp:special){
      for (int i=0;i<N;i++)
        if (sp.get(i)>needs.get(i))
          continue offer;
      List<Integer> remain =new ArrayList<>();
      for (int i=0;i<N;i++)
        remain.add(needs.get(i)-sp.get(i));
      int spend = sO(price, special, remain, memo);
      min = Math.min(min, spend+sp.get(N));
    }

    if (min==Integer.MAX_VALUE){
      min=0;
      for (int i=0;i<N;i++)
        min+=price.get(i)*needs.get(i);
    }
    memo.put(needNum,min);
    return min;
  }

  public int[] exclusiveTime(int n, List<String> logs) {
    int[] ans = new int[n];
    Stack<String[]> st=new Stack<>();
    for (String l:logs){
      String[] temp = l.split(":");
      if (temp[1].equals("start"))
        st.add(temp);
      else {
        String[] last=st.pop();
        int id=Integer.valueOf(temp[0]);
        int spend=Integer.valueOf(temp[2])-Integer.valueOf(last[2])+1;
        ans[id]+=spend;
        if (!st.isEmpty())
          ans[Integer.valueOf(st.peek()[0])]-=spend;
      }
    }
    return ans;
  }

  public int maxProduct(String[] words) {
    if (words==null||words.length==0)
      return 0;
    int N=words.length,product=Integer.MIN_VALUE;
    int[] val=new int[N];
    for (int i=0;i<N;i++){
      String temp=words[i];
      for (int j=0;j<temp.length();j++)
        val[i] |= 1<<(temp.charAt(j)-'a');
    }

    for (int i=0;i<N;i++)
      for (int j=i+1;j<N;j++)
        if ( (val[i]&val[j])==0)
          product = Math.max(product,words[i].length()*words[j].length());
    return product==Integer.MIN_VALUE?0:product;
  }

  class SRelement{
    int val;
    int id;
    int group;
    public SRelement(int v,int i, int g){
      val=v;
      id=i;
      group=g;
    }
  }

  public int[] smallestRange(List<List<Integer>> nums) {
    PriorityQueue<SRelement> pq = new PriorityQueue<>((a,b)->a.val-b.val);
    int max = Integer.MIN_VALUE,range,N=nums.size(),start,end;
    for (int i=0;i<N;i++){
      List<Integer> ls = nums.get(i);
      if (ls.size()==0)
        return null;
      int first = ls.get(0);
      pq.offer(new SRelement(first,0,i));
      max = Math.max(max,first);
    }
    start=pq.peek().val;
    end=max;
    range = end-start;
    while (pq.size()==N){
      SRelement cur=pq.poll();
      if (cur.id==nums.get(cur.group).size()-1)
        continue;
      SRelement supply= new SRelement(nums.get(cur.group).get(cur.id+1),cur.id+1,cur.group);
      pq.offer(supply);
      max =Math.max(max,supply.val);
      if (max-pq.peek().val<range){
        range =max-pq.peek().val;
        start=pq.peek().val;
        end=max;
      }
    }
    return new int[]{start,end};
  }

  public int maximumProduct1(int[] nums) {
    Arrays.sort(nums);
    int p=0,N=nums.length,n=0;
    boolean hasZero=false;
    for (int i=0;i<3;i++){
      if (nums[N-1-i]>0)
        p++;
      else if (nums[N-1-i]==0)
        hasZero=true;
      if (nums[i]<0)
        n++;
    }
    if (p==3)
      return Math.max(nums[N-1]*nums[N-2]*nums[N-3],nums[N-1]*nums[0]*nums[1]);
    else if (p>0)
      if (n>=2)
        return nums[0]*nums[1]*nums[N-1];
      else if (n==1)
        return hasZero?0:nums[N-1]*nums[N-2]*nums[N-3];
      else
        return 0;
    else
      return hasZero?0:nums[N-1]*nums[N-2]*nums[N-3];
  }

  public int maximumProduct(int[] nums) {
    int m1,m2,m3,l1,l2;
    m1=m2=m3=Integer.MIN_VALUE;
    l1=l2=Integer.MAX_VALUE;
    for (int n:nums){
      if (n>m1){
        m3=m2;
        m2=m1;
        m1=n;
      }
      else if (n>m2){
        m3=m2;
        m2=n;
      }
      else if (n>m3)
        m3=n;
      if (n<l1){
        l2=l1;
        l1=n;
      }
      else if (n<l2)
        l2=n;
    }
    return Math.max(m1*m2*m3,m1*l1*l2);
  }

  class KthLargest {
    private TreeMap<Integer,Integer> tm;
    private int K,KthNum,size;
    public KthLargest(int k, int[] nums) {
      K=k;
      tm=new TreeMap<>();
      for (int n:nums)
        tm.put(n,tm.getOrDefault(n,0)+1);
      KthNum=getKth();
      size=nums.length;
    }

    public int add(int val) {
      size++;
      tm.put(val,tm.getOrDefault(val,0)+1);
      if (size==K||val>KthNum)
        KthNum=getKth();
      return KthNum;
    }

    private int getKth(){
      Iterator<Map.Entry<Integer,Integer>> it = tm.entrySet().iterator();
      int count=size-K+1;
      while(it.hasNext()){
        Map.Entry<Integer,Integer> temp=it.next();
        count -=temp.getValue();
        if (count<=0)
          return temp.getKey();
      }
      return Integer.MIN_VALUE;
    }
  }

  class KthLargest1 {
    PriorityQueue<Integer> pq;
    int K;
    public KthLargest1(int k, int[] nums) {
      K=k;
      pq=new PriorityQueue<>(K);
      for (int n:nums)
        add(n);
    }

    public int add(int val) {
      if (pq.size()<K)
        pq.offer(val);
      else if (val>pq.peek()){
        pq.poll();
        pq.offer(val);
      }
      return pq.peek();
    }
  }

  public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> ans = new ArrayList<>();
    if (root==null)
      return ans;
    Queue<TreeNode> q= new LinkedList<>();
    q.offer(root);
    while(!q.isEmpty()){
      int layerSize = q.size();
      List<Integer> ls=new ArrayList<>();
      for (int i=0;i<layerSize;i++){
        TreeNode temp=q.poll();
        ls.add(temp.val);
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
      }
      ans.add(ls);
    }
    return ans;
  }

  class BSTIterator {
    private TreeNode root,cur;
    Stack<TreeNode> st;
    public BSTIterator(TreeNode root) {
      this.root=root;
      st = new Stack<>();
      cur=root;
    }

    /** @return the next smallest number */
    public int next() {
      if (cur==null){
        TreeNode temp = st.pop();
        cur=temp.right;
        return temp.val;
      }
      else{
        st.push(cur);
        cur=cur.left;
        return next();
      }
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
      return !st.isEmpty()||cur!=null;
    }
  }

  class BSTIterator1 {
    private TreeNode root,cur;
    public BSTIterator1(TreeNode root) {
      this.root=root;
      cur=root;
    }

    /** @return the next smallest number */
    public int next() {
     if (cur.left==null){
       int temp=cur.val;
       cur=cur.right;
       return temp;
     }
     else{
       TreeNode next=cur.left;
       while(next.right!=null&& next.right!=cur)
         next=next.right;
       if (next.right==null){
         next.right=cur;
         cur=cur.left;
         return next();
       }
       else{
         next.right=null;
         int temp=cur.val;
         cur=cur.right;
         return temp;
       }
     }
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
      return cur!=null;
    }
  }

  public int rob(TreeNode root) {
    if (root==null)
      return 0;
   Map<TreeNode,Integer> memo= new HashMap();
   return rob(root,memo);
  }

  private int rob(TreeNode root,Map<TreeNode,Integer> memo){
    if (root==null)
      return 0;
    if (memo.containsKey(root))
      return memo.get(root);
    if (root.left==null&&root.right==null)
      return root.val;
    int s1=root.val,s2=0;
    if (root.left!=null)
      s1+=rob(root.left.left,memo)+rob(root.left.right,memo);
    if (root.right!=null)
      s1 += rob(root.right.left,memo)+rob(root.right.right,memo);
    s2 = rob(root.left,memo)+rob(root.right,memo);
    int ans = Math.max(s1,s2);
    memo.put(root,ans);
    return ans;
  }

  public int rob1(TreeNode root) {
    int[] res = robHelper(root);
    return Math.max(res[0],res[1]);
  }

  private int[] robHelper(TreeNode root){
    if (root==null)
      return new int[2];

    int[] res = new int[2];
    int[] left=robHelper(root.left);
    int[] right = robHelper(root.right);
    res[0] = Math.max(left[0],left[1])+Math.max(right[0],right[1]);
    res[1]= root.val+left[0]+right[0];
    return res;
  }

  public int[] sortedSquares(int[] A) {
    if (A.length==1)
      return new int[]{A[0]*A[0]};
    if (A[0]>=0){
      for (int i=0;i<A.length;i++)
        A[i] = A[i]*A[i];
      return A;
    }
    int lo=0,hi=A.length,lastNeg=0;
    while(lo<hi){
      int mid = lo+((hi-lo)>>1);
      if (A[mid]>=0)
        hi=mid;
      else if (A[mid]<0)
        if (mid==A.length-1||A[mid+1]>=0){
          lastNeg=mid;
          break;
        }
        else
          lo=mid+1;
    }
    int p=lastNeg+1;
    int[] ans = new int[A.length];
    for (int i=0;i<A.length;i++)
      if (p==A.length)
        ans[i] = A[lastNeg]*A[lastNeg--];
      else if (lastNeg<0)
        ans[i]=A[p]*A[p++];
      else if (A[lastNeg]+A[p]>=0)
        ans[i] = A[lastNeg]*A[lastNeg--];
      else
        ans[i]=A[p]*A[p++];
    return ans;
  }

  public int findUnsortedSubarray(int[] nums) {
    int[] cl=nums.clone();
    Arrays.sort(nums);
    int beg=0,end=nums.length-1;
    while(beg<nums.length&&end>=0&&(cl[beg]==nums[beg] || cl[end]==nums[end])){
      beg = cl[beg]==nums[beg]?beg+1:beg;
      end = cl[end]==nums[end]?end-1:end;
    }
    int dis=end-beg+1;
    return dis>=0?dis:0;
  }

  public int findUnsortedSubarray1(int[] nums) {
    int beg=-1,end=-2,N=nums.length,max=nums[0],min = nums[N-1];
    for (int i=1;i<N;i++){
      max = Math.max(max,nums[i]);
      min = Math.min(min,nums[N-1-i]);
      if (nums[i]<max)
        end=i;
      if (nums[N-1-i]>min)
        beg=N-1-i;
    }
    return end-beg+1;
  }

  public int integerBreak(int n) {
    if (n==2)
      return 1;
    int[] dp=new int[n];
    dp[1]=1;
    for (int i=2;i<n;i++){
      int ans=Integer.MIN_VALUE,bound=(i+1)>>1;
      for (int j=1;j<=bound;j++)
        ans = Math.max(ans,Math.max(dp[j-1],j)*Math.max(dp[i-j],i-j+1));
      dp[i]=ans;
    }
    return dp[n-1];
  }

  public int findMinDifference(List<String> timePoints) {
    List<Integer> times = new ArrayList<>();
    for (String t:timePoints){
      String[] HM=t.split(":");
      times.add(Integer.valueOf(HM[0])*60+Integer.valueOf(HM[1]));
    }
    Collections.sort(times);
    int min=Integer.MAX_VALUE;
    for (int i=1;i<times.size();i++){
      int diff=times.get(i)-times.get(i-1);
      min=Math.min(diff,min);
    }
    return Math.min(min,24*60+times.get(0)-times.get(times.size()-1));
  }

  class SSCounter{
    int lastVal;
    int hiddenNums;
    public SSCounter(int last){
      lastVal=last;
      hiddenNums=1;
    }
  }

  class StockSpanner {
    List<SSCounter> ls;
    SSCounter cur;
    public StockSpanner() {
      ls = new ArrayList<>();
    }

    public int next(int price) {
      if (cur==null||cur.lastVal>price){
        cur=new SSCounter(price);
        ls.add(cur);
        return cur.hiddenNums;
      }
      else{
        cur.lastVal = price;
        cur.hiddenNums++;
        int ans = 0;
        for (int i = ls.size() - 1; i >= 0; i--) {
          SSCounter t=ls.get(i);
          if (t.lastVal > price)
            break;
          ans += t.hiddenNums;
        }
        return ans;
      }
    }
  }

  class StockSpanner1 {
    private Stack<int[]> st;
    public StockSpanner1() {
      st=new Stack<>();
    }

    public int next(int price) {
      int res=1;
      while(!st.isEmpty() && st.peek()[0]<=price)
        res+=st.pop()[1];
      st.push(new int[]{price,res});
      return res;
    }
  }

  public boolean backspaceCompare(String S, String T) {
    Stack<Character> s=new Stack<>(),t=new Stack<>();
    for (char c:S.toCharArray())
      if (c!='#')
        s.push(c);
      else if (c=='#' && !s.isEmpty())
        s.pop();
    for (char c:T.toCharArray())
      if (c!='#')
        t.push(c);
      else if (c=='#' && !t.isEmpty())
        t.pop();
    if (s.size()!=t.size())
      return false;
    else
      for (int i=0;i<s.size();i++)
        if (!s.get(i).equals(t.get(i)))
          return false;
    return true;
  }

  public boolean backspaceCompare1(String S, String T) {
    int bs=0,bt=0,is=S.length()-1,it=T.length()-1;
    while(is>=0||it>=0){
      while(is>=0 && (S.charAt(is)=='#'||bs>0))
        if (S.charAt(is)=='#'){
          is--;
          bs++;
        }
        else{
          is--;
          bs--;
        }
      while(it>=0 && (T.charAt(it)=='#'||bt>0))
        if (T.charAt(it)=='#'){
          it--;
          bt++;
        }
        else{
          it--;
          bt--;
        }
      if (is<0 &&it<0)
        return true;
      if ((is<0 && it>=0)||(is>=0&&it<0))
        return false;
      if (S.charAt(is)!=T.charAt(it))
        return false;
      is--;
      it--;
    }
    return true;
  }

  public List<String> binaryTreePaths(TreeNode root) {
    List<String> ans=new ArrayList<>();
    if (root==null)
      return ans;
    List<List<Integer>> res= new ArrayList<>();
    getTreePaths(res,new ArrayList<>(),root);
    StringBuilder sb=new StringBuilder();
    for (List<Integer> p:res){
      for (int n:p){
        if (sb.length()>0)
          sb.append("->");
        sb.append(n);
      }
      ans.add(sb.toString());
      sb.delete(0,sb.length());
    }
    return ans;
  }

  private void getTreePaths(List<List<Integer>> res,List<Integer> path,TreeNode root){
    if (root.left==null &&root.right==null){
      path.add(root.val);
      res.add(path);
      return;
    }
    path.add(root.val);
    if (root.left!=null)
      getTreePaths(res,new ArrayList<>(path),root.left);
    if (root.right!=null)
      getTreePaths(res,new ArrayList<>(path),root.right);
  }

  public String[] spellchecker(String[] wordlist, String[] queries) {
    Set<String> st=new HashSet<>(Arrays.asList(wordlist));
    Map<String,String> cap=new HashMap<>();
    Map<String,String> devowel=new HashMap<>();
    for (String n:wordlist){
      String c=n.toLowerCase(),v=c.replaceAll("[aeiou]","#");
      cap.putIfAbsent(c,n);
      devowel.putIfAbsent(v,n);
    }
    for (int i=0;i<queries.length;i++){
      if (st.contains(queries[i]))
        continue;
      String c=queries[i].toLowerCase(),v=c.replaceAll("[aeiou]","#");
      if (cap.containsKey(c))
        queries[i]=cap.get(c);
      else if (devowel.containsKey(v))
        queries[i]=devowel.get(v);
      else
        queries[i]="";
    }
    return queries;
  }

  public List<List<Integer>> verticalTraversal(TreeNode root) {
    List<List<Integer>> ans = new ArrayList<>();
    if (root==null)
      return ans;
    Map<Integer,Map<Integer,PriorityQueue<Integer>>> mp=new TreeMap<>();
    getVT(root,0,0,mp);
    for (Map<Integer,PriorityQueue<Integer>> ys:mp.values()){
      List<Integer> ls=new ArrayList<>();
      for (PriorityQueue<Integer> vals:ys.values())
        while(!vals.isEmpty())
          ls.add(vals.poll());
      ans.add(ls);
    }
    return ans;
  }

  private void getVT(TreeNode root,int x,int y,Map<Integer,Map<Integer,PriorityQueue<Integer>>> mp){
    if (root==null)
      return;
    mp.computeIfAbsent(x,a->new TreeMap<>()).computeIfAbsent(y,a->new PriorityQueue<>()).offer(root.val);
    getVT(root.left,x-1,y+1,mp);
    getVT(root.right,x+1,y+1,mp);
  }

  public ListNode deleteDuplicates(ListNode head) {
    if (head==null)
      return null;
    ListNode cur=head.next,pre=head;
    while(cur!=null)
      if (cur.val!=pre.val){
        pre=pre.next;
        cur=cur.next;
      }
      else{
        pre.next=cur.next;
        cur=cur.next;
      }
    return head;
  }

  public boolean hasCycle(ListNode head) {
    if (head==null)
      return false;
    ListNode fast=head,slow=head;
    do {
      fast=fast.next;
      if (fast!=null)
        fast=fast.next;
      else
        return false;
      slow=slow.next;
    }while(fast!=null &&fast!=slow);
    return fast==null?false:true;
  }

  public void rotate(int[][] matrix) {
    if (matrix==null)
      return;
    int n=matrix.length;
    if (n==1)
      return;
    int temp=0,R=(n&1)==1?(n>>1)+1:(n>>1),C=n>>1,r,c;
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++){
        r=i;
        c=j;
        temp=matrix[r][c];
        for (int k=0;k<4;k++){
          int nextR=c,nextC=n-1-r,t=matrix[nextR][nextC];
          matrix[nextR][nextC]=temp;
          r=nextR;
          c=nextC;
          temp=t;
        }
      }
  }

  public int leastBricks(List<List<Integer>> wall) {
    Map<Integer,Integer> counter= new HashMap<>();
    int rowsNum=wall.size(),maxEdges=0;
    for (List<Integer> row:wall){
      int acc=0;
      for (int i=0;i<row.size()-1;i++){
        acc+=row.get(i);
        int times=counter.getOrDefault(acc,0)+1;
        maxEdges = Math.max(times,maxEdges);
        counter.put(acc,times);
      }
    }
    return rowsNum-maxEdges;
  }

  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    Arrays.sort(candidates);
    List<List<Integer>> ans = new ArrayList<>();
    CS(candidates,target,0,new ArrayList<>(),ans);
    return ans;
  }

  private void CS(int[] candidates,int target,int index,List<Integer> path,List<List<Integer>> res){
    if (target==0){
      res.add(path);
      return;
    }
    for (int i=index;i<candidates.length;i++){
      int c=candidates[i];
      if (c>target)
        break;
      List<Integer> p=new ArrayList<>(path);
      p.add(c);
      CS(candidates,target-c,i,p,res);
    }
  }

  public boolean PredictTheWinner(int[] nums) {
    Map<Integer,Integer> memo=new HashMap<>();
    int res=PWMinMax(nums,0,nums.length-1,memo);
    return res>=0;
  }

  private int PWMinMax(int[] nums,int beg,int end,Map<Integer,Integer> memo){
    if (beg==end)
      return nums[beg];
    if (beg>end)
      return 0;
    int range=end*100+beg;
    if (memo.containsKey(range))
      return memo.get(range);
    int value=Math.max(nums[beg]-PWMinMax(nums,beg+1,end,memo),nums[end]-PWMinMax(nums,beg,end-1,memo));
    memo.put(range,value);
    return value;
  }

  public boolean isUgly(int num) {
    if (num<=0)
      return false;
    while((num&1)==0)
      num>>=1;
    while (num%3==0)
      num/=3;
    while (num%5==0)
      num/=5;
    return num==1;
  }

  public boolean isHappy(int n) {
    if (n<=0)
      return false;
    int fast,slow;
    slow=fast=n;
    do {
      fast=nextHappy(fast);
      fast=nextHappy(fast);
      slow=nextHappy(slow);
    }while(slow!=fast);
    return slow==1;
  }

  private int nextHappy(int n){
    int ans=0;
    while(n>0){
      int unit=n%10;
      ans+=unit*unit;
      n/=10;
    }
    return ans;
  }

  public List<Integer> rightSideView(TreeNode root) {
    List<Integer> ans = new ArrayList<>();
    if (root==null)
      return ans;
    Queue<TreeNode> q=new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()){
      int size=q.size();
      for (int i=0;i<size;++i){
        TreeNode temp=q.poll();
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
        if (i==size-1)
          ans.add(temp.val);
      }
    }
    return ans;
  }

  public List<Integer> rightSideView1(TreeNode root) {
    List<Integer> ans = new ArrayList<>();
    if (root==null)
      return ans;
    RSV(root,1,ans);
    return ans;
  }

  private void RSV(TreeNode root,int depth,List<Integer> res){
    if (root==null)
      return;
    if (res.size()<depth)
      res.add(root.val);
    else
      res.set(depth-1,root.val);
    RSV(root.left,depth+1,res);
    RSV(root.right,depth+1,res);
  }

  public class NestedInteger {

    public NestedInteger() {
    }

    public NestedInteger(int value) {
    }

    public boolean isInteger() {
      return true;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger() {
      return null;
    }

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value) {
    }

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni) {
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return null if this NestedInteger holds a single integer
    public List<NestedInteger> getList() {
      return null;
    }
  }

  public class NestedIterator implements Iterator<Integer> {
    int cur;
    List<Integer> data;

    public NestedIterator(List<NestedInteger> nestedList) {
      cur = 0;
      data = new ArrayList<>();
      for (int i = 0; i < nestedList.size(); i++)
        flatten(nestedList.get(i));
    }

    @Override
    public Integer next() {
      return data.get(cur++);
    }

    @Override
    public boolean hasNext() {
      return cur<data.size();
    }

    private void flatten(NestedInteger nl){
      if (nl.isInteger())
        data.add(nl.getInteger());
      else
        for (NestedInteger ni:nl.getList())
          flatten(ni);
    }
  }

  public TreeNode addOneRow(TreeNode root, int v, int d) {
    if (d==1){
      TreeNode newRoot=new TreeNode(v);
      newRoot.left=root;
      return newRoot;
    }
    Queue<TreeNode> q=new LinkedList<>();
    int depth=1;
    q.offer(root);
    findDepth:while (!q.isEmpty()){
      int size=q.size();
      if (depth==d-1){
        for (int i=0;i<size;i++){
          TreeNode temp=q.poll();
          TreeNode left=temp.left,right=temp.right;
          TreeNode nl=new TreeNode(v),nr=new TreeNode(v);
          temp.left=nl;
          temp.right=nr;
          nl.left=left;
          nr.right=right;
        }
        break findDepth;
      }
      for (int i=0;i<size;i++){
        TreeNode temp=q.poll();
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
      }
      depth++;
    }
    return root;
  }

  public String orderlyQueue(String S, int K) {
    if (K>1){
      char[] cs=S.toCharArray();
      Arrays.sort(cs);
      return String.valueOf(cs);
    }
    String min=S;
    for (int i=1;i<S.length();i++){
      String temp=S.substring(i)+S.substring(0,i);
      if (temp.compareTo(min)<0)
        min=temp;
    }
    return min;
  }

  public void moveZeroes1(int[] nums) {
    int cur=0,N=nums.length;
    for (int n:nums)
      if (n!=0)
        nums[cur++]=n;
    for (int i=cur;i<N;i++)
      nums[i]=0;
  }

  public boolean isLongPressedName(String name, String typed) {
    char[] n=name.toCharArray(),t=typed.toCharArray();
    int in=0,it=0,accum;
    while(in<n.length||it<t.length){
      accum=1;
      char cur=n[in++];
      while(in<n.length && n[in]==cur){
        accum++;
        in++;
      }
      while(it<t.length && t[it]==cur){
        it++;
        accum--;
      }
      if (accum>0)
        return false;
    }
    if (in<n.length||it<t.length)
      return false;
    else
      return true;
  }

  public int removeElement(int[] nums, int val) {
    int cur=0,N=nums.length;
    for (int i=0;i<N;i++)
      if (nums[i]!=val)
        nums[cur++]=nums[i];
    return cur;
  }

  public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
    Map<String,Map<String,Double>> graph = CEgetGraph(equations,values);
    double[] ans=new double[queries.length];
    for (int i=0;i<queries.length;i++)
      ans[i]=CEdfs(queries[i][0],queries[i][1],new HashSet<>(),graph);
    return ans;
  }

  private double CEdfs(String start,String end,Set<String> visited, Map<String,Map<String,Double>> graph){
    if (!graph.containsKey(start) || !graph.containsKey(end))
      return -1;
    if (graph.get(start).containsKey(end))
      return graph.get(start).get(end);
    visited.add(start);
    for (Map.Entry<String,Double> edges:graph.get(start).entrySet()){
      if (visited.contains(edges.getKey()))
        continue;
      double res=CEdfs(edges.getKey(),end,visited,graph);
      if (res!=-1)
        return res*edges.getValue();
    }
    return -1;
  }

  private Map<String,Map<String,Double>> CEgetGraph(String[][] equations, double[] values){
    Map<String,Map<String,Double>> graph=new HashMap<>();
    for (int i=0;i<values.length;i++){
      String c1=equations[i][0],c2=equations[i][1];
      if (!graph.containsKey(c1))
        graph.put(c1,new HashMap<>());
      if (!graph.containsKey(c2))
        graph.put(c2,new HashMap<>());
      graph.get(c1).put(c2,values[i]);
      graph.get(c2).put(c1,1/values[i]);
//      graph.computeIfAbsent(c1,a->new HashMap<>()).put(c2,values[i]);
//      graph.computeIfAbsent(c2,a->new HashMap<>()).put(c1,1/values[i]);
    }
    return graph;
  }

  public int uniquePaths(int m, int n) {
    int[] count=new int[1];
    UPdfs(m,n,0,0,count);
    return count[0];
  }

  private void UPdfs(int m,int n,int r,int c,int[] count){
    if (r==m-1 && c==n-1){
      count[0]++;
      return;
    }
    if (r<m-1)
      UPdfs(m,n,r+1,c,count);
    if (c<n-1)
      UPdfs(m,n,r,c+1,count);
  }

  public int uniquePaths1(int m, int n) {
    if (m==1||n==1)
      return 1;
    int[][] dp=new int[m][n];
    for (int i=0;i<n;i++){
      dp[m-1][i]=1;
      dp[m-2][i]=n-i;
    }

    for (int i=m-3;i>=0;i--){
      int accum=0;
      for (int j=n-1;j>=0;j--){
        dp[i][j]=accum+dp[i+1][j];
        accum+=dp[i+1][j];
      }
    }
    return dp[0][0];
  }

  public String reverseStr(String s, int k) {
    if (k==1)
      return s;
    char[] cs=s.toCharArray();
    int cur=0,N=cs.length;
    while(cur<N){
      if (cur+k>=N)
        RS(cs,cur,N-1);
      else
        RS(cs,cur,cur+k-1);
      cur+=2*k;
    }
    return String.valueOf(cs);
  }

  private void RS(char[] cs,int start,int end){
    while(start<end){
      char temp = cs[start];
      cs[start]=cs[end];
      cs[end]=temp;
      start++;
      end--;
    }
  }

  public boolean checkRecord(String s) {
    int l=0,a=0,conL=0;
    for (int i=0;i<s.length();i++){
      if (s.charAt(i)=='A')
        a++;
      if (s.charAt(i)=='L')
        l++;
      else
        l=0;
      conL=Math.max(conL,l);
      if (a>1||conL>2)
        break;
    }
    return a>1||conL>2?false:true;
  }

  public boolean queryString(String S, int N) {
    int len=S.length();
    for (int i=1;i<=N;i++)
      if (QSsearch(S,i)==len)
        return false;
    return true;
  }

  private int QSsearch(String S,int n){
    String pat=Integer.toBinaryString(n);
    int M=pat.length(),sLen=S.length(),i,j;
    int[][] dfa=new int[2][M];
    dfa[pat.charAt(0)-48][0]=1;
    for (int X=0,a=1;a<M;a++){
      for (int c=0;c<2;c++)
        dfa[c][a]=dfa[c][X];
      dfa[pat.charAt(a)-48][a]=a+1;
      X=dfa[pat.charAt(a)-48][X];
    }
    for (i=0,j=0;i<sLen &&j<M;i++)
      j=dfa[S.charAt(i)-48][j];
    if (j==M)
      return i-M;
    else
      return sLen;
  }

  public List<List<String>> groupAnagrams(String[] strs) {
    int[] prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103};
    Map<Integer,List<String>> res=new HashMap<>();
    for (String s:strs){
      Integer key=1;
      for (int i=0;i<s.length();i++)
        key*=prime[s.charAt(i)-'a'];
      if (!res.containsKey(key))
        res.put(key,new ArrayList<>());
      res.get(key).add(s);
    }
    List<List<String>> ans = new ArrayList<>();
    for (List<String> ls:res.values())
      ans.add(ls);
    return ans;
  }

  public int[] loudAndRich(int[][] richer, int[] quiet) {
    int N=quiet.length;
    Map<Integer,List<Integer>> graph=LRbuildGraph(richer);
    int[] ans=new int[N];
    Map<Integer,int[]> memo=new HashMap<>();
    for (int i=0;i<N;i++)
      ans[i]=LRdfs(graph,memo,quiet,i)[0];
    return ans;
  }

  private int[] LRdfs(Map<Integer,List<Integer>> graph, Map<Integer,int[]> memo,int[] quiet,int i){
    if (!graph.containsKey(i))
      return new int[]{i,quiet[i]};
    if (memo.containsKey(i))
      return memo.get(i);
    int[] ans=new int[]{i,quiet[i]};
    for (int richer:graph.get(i)){
      int[] tempMin=LRdfs(graph,memo,quiet,richer);
      if (tempMin[1]<ans[1]){
        ans[1]=tempMin[1];
        ans[0]=tempMin[0];
      }
    }
    memo.put(i,ans);
    return ans;
  }

  private Map<Integer,List<Integer>> LRbuildGraph(int[][] richer){
    Map<Integer,List<Integer>> graph=new HashMap<>();
    for (int[] r:richer){
      int rich=r[0],poor=r[1];
      if (!graph.containsKey(poor))
        graph.put(poor,new ArrayList<>());
      graph.get(poor).add(rich);
    }
    return graph;
  }

  public List<String> readBinaryWatch(int num) {
    List<String> ans = new ArrayList<>();
    if (num<0 ||num>8)
      return ans;
    if (num==0){
      ans.add("0:00");
      return ans;
    }
    BWdfs(num,0,new int[10],ans);
    return ans;
  }

  private void BWdfs(int num,int i,int[] path,List<String> res){
    if (num==0){
      int hour=0,mins=0;
      for (int j=0;j<4;j++)
        hour=hour*2+path[j];
      if (hour>11)
        return;
      for (int j=0;j<6;j++)
        mins=mins*2+path[4+j];
      if (mins>59)
        return;
      String time=mins<10?String.valueOf(hour)+":0"+String.valueOf(mins):String.valueOf(hour)+":"+String.valueOf(mins);
      res.add(time);
      return;
    }
    for (int j=i;j<=10-num;j++){
      path[j]=1;
      BWdfs(num-1,j+1,path.clone(),res);
      path[j]=0;
    }
  }

  public List<List<Integer>> generate(int numRows) {
    List<List<Integer>> ans = new ArrayList<>(numRows);
    if (numRows==0)
      return ans;
    ans.add(new ArrayList<>());
    ans.get(0).add(1);
    if (numRows==1)
      return ans;
    for (int i=1;i<numRows;i++){
      List<Integer> row=new ArrayList<>(i);
      List<Integer> top=ans.get(i-1);
      for (int j=0;j<=i;j++){
        int tl=j==0?0:top.get(j-1),tr=j==i?0:top.get(j);
        row.add(tl+tr);
      }
      ans.add(row);
    }
    return ans;
  }

  public boolean isCompleteTree(TreeNode root) {
    if (root==null)
      return true;
    Queue<TreeNode> q=new LinkedList<>();
    q.offer(root);
    boolean isEnd=false;
    while (!q.isEmpty()){
      TreeNode cur=q.poll();
      if (isEnd && (cur.left!=null || cur.right!=null))
        return false;
      if (cur.left==null && cur.right!=null)
        return false;
      if (cur.right==null)
        isEnd=true;
      if (cur.left!=null)
        q.offer(cur.left);
      if (cur.right!=null)
        q.offer(cur.right);
    }
    return true;
  }

  public int shipWithinDays(int[] weights, int D) {
    int lo=0,hi=0;
    for (int n:weights){
      lo=Math.max(lo,n);
      hi+=n;
    }
    while(lo<hi){
      int mid=lo+((hi-lo)>>1),groupNum=1,accum=0;
      for (int n:weights){
        if (accum+n>mid){
          groupNum++;
          accum=0;
        }
        accum+=n;
      }
      if (groupNum>D)
        lo=mid+1;
      else
        hi=mid-1;
    }
    return lo;
  }

  public int maxSubArray1(int[] nums) {
    int max,lastMax;
    max=lastMax=nums[nums.length-1];
    for (int i=nums.length-2;i>=0;i--){
      lastMax=Math.max(lastMax+nums[i],nums[i]);
      max=Math.max(max,lastMax);
    }
    return max;
  }

  public List<Integer> addToArrayForm(int[] A, int K) {
    int carry=0,i=A.length-1,res;
    List<Integer> ans=new LinkedList<>();
    while(K>0||carry>0||i>=0){
      res=carry;
      if (i>=0)
        res+=A[i--];
      if (K>0){
        res+=K%10;
        K/=10;
      }
      carry=res/10;
      if (ans.size()==0)
        ans.add(res%10);
      else
        ans.add(0,res%10);
    }
    return ans;
  }

  public boolean isSubsequence(String s, String t) {
    if (s.length()==0)
      return true;
    if (t.length()==0)
      return false;
    int is=0;
    char[] cs=s.toCharArray();
    for (char c:t.toCharArray()){
      if (c==cs[is])
        is++;
      if (is==cs.length)
        break;
    }
    return is==cs.length;
  }

  public boolean isSubsequence1(String s, String t) {
    if (s.length()==0)
      return true;
    if (t.length()==0)
      return false;
    int prev=0;
    for (int i=0;i<s.length();i++){
      int j=t.indexOf(s.charAt(i),prev);
      if (j==-1)
        return false;
      prev=j+1;
    }
    return true;
  }

  public int findKthLargest1(int[] nums, int k) {
    PriorityQueue<Integer> pq=new PriorityQueue<>(k);
    for (int n:nums)
      if (pq.size()<k)
        pq.offer(n);
      else if (pq.size()==k && pq.peek()<n){
        pq.poll();
        pq.offer(n);
      }
    return pq.peek();
  }

  public int findKthLargest(int[] nums, int k) {
    if (nums.length==1)
      return nums[0];
    return quickSelection(nums,0,nums.length-1,nums.length-k);
  }

  private int quickSelection(int[] nums,int lo,int hi,int k){
    if (lo==hi)
      return nums[lo];
    int pivot=QSpartition(nums,lo,hi);
    if (pivot<k)
      return quickSelection(nums,pivot+1,hi,k);
    else if (pivot>k)
      return quickSelection(nums,lo,pivot-1,k);
    else
      return nums[k];
  }

  private int QSpartition(int[] nums,int lo,int hi){
    int pivot=lo,left=lo,right=hi+1;
    while(left<right){
      while(left<hi && nums[++left]<nums[pivot]);
      while(right>lo && nums[--right]>nums[pivot]);
      if (left>=right)
        break;
      exchange(nums,left,right);
    }
    exchange(nums,pivot,right);
    return right;
  }

  class MyCalendar1 {
    List<int[]> booked;
    public MyCalendar1() {
      booked=new ArrayList<>();
    }

    public boolean book(int start, int end) {
      int[] cur=new int[]{start,end};
      int[] res=findFloor(cur);
      if (res==null){
        booked.add(cur);
        return true;
      }
      if (res[1]==0)
        return false;
      else{
        booked.add(res[0],cur);
        return true;
      }
    }

    private int[] findFloor(int[] a){
      int N=booked.size(),lo=0,hi=N-1;
      while(lo<=hi){
        int mid=(hi+lo)>>>1;
        int[] midVal=booked.get(mid);
        int res=compare(a,midVal);
        if (res==0)
          return new int[]{mid,0};

        if (mid==N-1)
          if (res==1)
            return new int[]{mid,-1};
          else
            hi=mid-1;
        else{
          int[] next=booked.get(mid+1);
          int nRes=compare(a,next);
          if (nRes==0)
            return new int[]{mid+1,0};
          if (nRes==-1 && res==1)
            return new int[]{mid,-1};
          else if (nRes==1 && res==1)
            lo=mid+1;
          else if (nRes==-1&& res==-1)
            hi=mid-1;
        }
      }
      return null;
    }

    private int compare(int[] a,int[] b){
      if (a[1]<=b[0])
        return -1;
      else if (a[0]>=b[1])
        return 1;
      else return 0;
    }
  }

  class MyCalendar {
    TreeSet<int[]> booked;
    public MyCalendar() {
      booked=new TreeSet<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] a, int[] b) {
          if (a[1]<=b[0])
            return -1;
          else if (a[0]>=b[1])
            return 1;
          else return 0;
        }
      });

    }

    public boolean book(int start, int end) {
      int[] cur=new int[]{start,end};
      int[] floor=booked.floor(cur);
      if (floor==null ||floor[1]<=start){
        booked.add(cur);
        return true;
      }
      else
        return false;
    }
  }

  public int countNumbersWithUniqueDigits(int n) {
    if (n==0)
      return 1;
    if (n==1)
      return 10;
    int count=10,curLen=9,remain=9;
    for (int i=2;i<=n;i++){
      curLen*=remain--;
      count+=curLen;
      if (remain==0)
        break;
    }
    return count;
  }

  public String convertToBase7(int num) {
    if (num==0)
      return "0";
    StringBuilder sb=new StringBuilder();
    boolean isNeg=num<0;
    num = Math.abs(num);
    while (num>0){
      int temp=num%7;
      num/=7;
      sb.insert(0,temp);
    }
    if (isNeg)
      sb.insert(0,'-');
    return sb.toString();
  }

  public List<Boolean> prefixesDivBy5(int[] A) {
    int temp=0;
    List<Boolean> ans=new ArrayList<>();
    for (int n:A){
      temp=((temp<<1)+n)%5;
      ans.add(temp==0);
    }
    return ans;
  }

  public List<List<Integer>> combine(int n, int k) {
    List<List<Integer>> ans=new ArrayList<>();
    if (k<=0||n<1||k>n)
      return ans;
    Stack<Integer> path=new Stack<>();
    getCombine(k,1,n,path,ans);
    return ans;
  }

  private void getCombine(int k,int start,int end,Stack<Integer> path,List<List<Integer>> res){
    if (k==0){
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i=start;i<=end-k+1;i++){
      path.push(i);
      getCombine(k-1,i+1,end,path,res);
      path.pop();
    }
  }

  public int videoStitching(int[][] clips, int T) {
    Arrays.sort(clips,(a,b)->a[0]-b[0]);
    int start=0,ans=0,i;
    while(start<=T){
      int max=start;
      for (i=0;i<clips.length;i++){
        int[] cur=clips[i];
        if (cur[0]>start)
          break;
        if (cur[1]>start)
          max=Math.max(max,cur[1]);
      }
      if (max==start)
        break;
      start=max;
      ans++;
      if (start>=T)
        break;
    }
    return start>=T?ans:-1;
  }

  public int minSteps(int n) {
    int ans=0;
    for (int i=2;i<=n;i++)
      while(n%i==0){
        ans+=i;
        n/=i;
      }
    return ans;
  }

  public int numPairsDivisibleBy60(int[] time) {
    int N=time.length,ans=0,start=1,end=59;
   int[] count=new int[60];
   for (int t:time)
     count[t%60]++;
   while(start<end)
     ans+=count[start++]*count[end--];
   ans += count[30]*(count[30]-1)/2;
   ans+= count[0]*(count[0]-1)/2;
   return ans;
  }

  class LWNode{
    String val;
    LWNode[] next;
    public LWNode(){
      next=new LWNode[26];
    }
  }

  private void LWbuildTrie(LWNode root,String s,int d){
    if (d==s.length())
      root.val=s;
    else{
      int i=s.charAt(d)-'a';
      if (root.next[i]==null)
        root.next[i]=new LWNode();
      LWbuildTrie(root.next[i],s,d+1);
    }
  }

  private String LWsearch(LWNode root){
    String ans=root.val;
    for (int i=0;i<26;i++){
      LWNode temp= root.next[i];
      if (temp!=null&&temp.val!=null){
        String cur=LWsearch(temp);
        if (cur.length()>ans.length() || cur.compareTo(ans)<0)
          ans=cur;
      }
    }
    return ans;
  }

  public String longestWord(String[] words) {
    LWNode root=new LWNode();
    root.val="";
    for (String w:words)
      LWbuildTrie(root,w,0);
    String ans=LWsearch(root);
    return ans;
  }

  public List<List<Integer>> subsetsWithDup1(int[] nums) {
    List<List<Integer>> ans=new ArrayList<>();
    if (nums==null)
      return ans;
    ans.add(new ArrayList<>());
    if (nums.length==0)
      return ans;
    Set<String> st=new HashSet<>();
    Arrays.sort(nums);
    for (int r=1;r<=nums.length;r++)
      SWDgetSubsets(new Stack<>(),st,ans,nums,0,r);
    return ans;
  }

  private void SWDgetSubsets(Stack<Integer> path,Set<String> st,List<List<Integer>> res,int[] nums,int start,int remain){
    if (remain==0){
      StringBuilder sb=new StringBuilder();
      for (Integer i:path)
        sb.append(i);
      String key=sb.toString();
      if (!st.contains(key)){
        st.add(key);
        res.add(new ArrayList<>(path));
      }
      return;
    }
    for (int i=start;i<=nums.length-remain;i++){
      path.push(nums[i]);
      SWDgetSubsets(path,st,res,nums,i+1,remain-1);
      path.pop();
    }
  }

  public List<List<Integer>> subsetsWithDup(int[] nums) {
    List<List<Integer>> ans=new ArrayList<>();
    if (nums==null)
      return ans;
    Arrays.sort(nums);
    SWDHelper(ans,new Stack<>(),nums,0);
    return ans;
  }

  private void SWDHelper(List<List<Integer>> res,Stack<Integer> path,int[] nums,int start){
    if (start<=nums.length)
      res.add(new ArrayList<>(path));

    int i=start;
    while(i<nums.length){
      path.push(nums[i]);
      SWDHelper(res,path,nums,i+1);
      path.pop();
      i++;
      while(i<nums.length && nums[i]==nums[i-1])
        i++;
    }
  }

  public List<Integer> grayCode1(int n) {
    List<Integer> ans=new ArrayList<>();
    if (n==0){
      ans.add(0);
      return ans;
    }
    GCHelper(0,n,ans,new HashSet<>());
    return ans;
  }

  private void GCHelper(int val,int n,List<Integer> res,Set<Integer> st){
    if (st.contains(val))
      return;
    st.add(val);
    res.add(val);
    for (int i=0;i<n;i++)
      GCHelper(val^(1<<i),n,res,st);
  }

  public List<Integer> grayCode(int n) {
    List<Integer> ans=new ArrayList<>();
    ans.add(0);
    for (int i=0;i<n;i++){
      int size=ans.size();
      for (int j=size-1;j>=0;j--)
        ans.add(ans.get(j) | (1<<i));
    }
    return ans;
  }

  public String removeOuterParentheses(String S) {
    StringBuilder sb=new StringBuilder();
    int count=0,start=0;
    char[] cs=S.toCharArray();
    for (int i=0;i<cs.length;i++){
      if (count==0)
        start=i;
      if (cs[i]=='(')
        count++;
      else
        count--;
      if (count==0)
        sb.append(S.substring(start+1,i));
    }
    return sb.toString();
  }

  public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
    TreeNode[] ans=new TreeNode[1];
    LCAHelper(root,p,q,ans);
    return ans[0];
  }

  private Set<TreeNode> LCAHelper(TreeNode root,TreeNode p,TreeNode q,TreeNode[] res){
    if (res[0]!=null)
      return null;
    if (root==null)
      return null;
    Set<TreeNode> left=LCAHelper(root.left,p,q,res);
    Set<TreeNode> right=LCAHelper(root.right,p,q,res);
    Set<TreeNode> descendants=new HashSet<>();
    descendants.add(root);
    if (left!=null)
      descendants.addAll(left);
    if (right!=null)
      descendants.addAll(right);
    if (res[0]==null &&descendants.contains(p) && descendants.contains(q))
      res[0]=root;
    return descendants;
  }

  public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
    if (p.val>root.val && q.val>root.val)
      return lowestCommonAncestor1(root.right,p,q);
    else if (p.val<root.val && q.val<root.val)
      return lowestCommonAncestor1(root.left,p,q);
    else
      return root;
  }

  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    Queue<TreeNode> pPath=new LinkedList<>(),qPath=new LinkedList<>();
    LCAfindPath(root,p,pPath);
    LCAfindPath(root,q,qPath);
    TreeNode ans=findLCA(pPath,qPath);
    return ans;
  }

  private void LCAfindPath(TreeNode root,TreeNode target,Queue<TreeNode> path){
    TreeNode cur=root;
    while(cur!=null){
      path.offer(cur);
      if (cur==target)
        return;
      cur=target.val>cur.val?cur.right:cur.left;
    }
  }

  private TreeNode findLCA(Queue<TreeNode> pPath,Queue<TreeNode> qPath){
    TreeNode ans=null;
    while(!pPath.isEmpty() && !qPath.isEmpty() && pPath.peek()==qPath.peek()){
      pPath.poll();
      ans=qPath.poll();
    }
    return ans;
  }

  public int maxScoreSightseeingPair(int[] A) {
    int res=0,cur=0;
    for (int a:A){
      res=Math.max(res,cur+a);
      cur=Math.max(a,cur)-1;
    }
    return res;
  }

  public int[] nextLargerNodes1(ListNode head) {
    if (head==null)
      return new int[0];
    List<Integer> res=new ArrayList<>();
    Stack<int[]> st=new Stack<>();
    ListNode cur=head;
    int i=0;
    while(cur!=null){
      res.add(0);
      while(!st.isEmpty() && st.peek()[0] <cur.val){
        int[] temp=st.pop();
        res.set(temp[1],cur.val);
      }
     st.push(new int[]{cur.val,i++});
      cur=cur.next;
    }
    int N=res.size();
    int[] ans=new int[N];
    for (int j=0;j<N;j++)
      ans[j]=res.get(j);
    return ans;
  }

  int[] NLNres;
  public int[] nextLargerNodes(ListNode head) {
    if (head==null)
      return new int[0];
    NLNhelper(head,0);
    return NLNres;
  }

  private ListNode NLNhelper(ListNode head,int length){
    if (head==null){
      NLNres=new int[length];
      return null;
    }
    ListNode next=NLNhelper(head.next,length+1);
    while(next!=null &&next.val<=head.val)
      next=next.next;
    if (next!=null)
      NLNres[length]=next.val;
    head.next=next;
    return head;
  }

  int BFPindex;
  public TreeNode bstFromPreorder(int[] preorder) {
    if (preorder.length==1)
      return new TreeNode(preorder[0]);
//    TreeNode ans=BFPHelper(preorder,0,preorder.length);
    BFPindex=0;
    TreeNode ans=BFPHelper1(preorder,Integer.MAX_VALUE);
    return ans;
  }

  private TreeNode BFPHelper(int[] preorder,int start,int end){
    if (end<=start)
      return null;
    if (end-start==1)
      return new TreeNode(preorder[start]);
    TreeNode cur=new TreeNode(preorder[start]);
    int seg=start+1;
    while(seg<end && preorder[seg]<cur.val)
      seg++;
    cur.left=BFPHelper(preorder,start+1,seg);
    cur.right=BFPHelper(preorder,seg,end);
    return cur;
  }

  private TreeNode BFPHelper1(int[] preorder,int bound){
    if (BFPindex>=preorder.length||preorder[BFPindex]>bound)
      return null;
    TreeNode cur=new TreeNode(preorder[BFPindex++]);
    cur.left=BFPHelper1(preorder,cur.val);
    cur.right=BFPHelper1(preorder,bound);
    return cur;
  }

  public String fractionAddition(String expression) {
    char[] cs=expression.toCharArray();
    int ResNum=0,ResDen=1,curNum,curDen;
    for (int i=0;i<cs.length;i++)
      if (cs[i]=='/'){
      curNum=curDen=0;
        int numStart,denEnd;
        for (numStart=i-1;numStart>=0;numStart--)
          if (cs[numStart]<'0' ||cs[numStart]>'9')
            break;
          else
            curNum+=(cs[numStart]-'0')*(int)Math.pow(10,i-1-numStart);
        for (denEnd=i+1;denEnd<cs.length;denEnd++)
          if (cs[denEnd]<'0' ||cs[denEnd]>'9')
            break;
          else
            curDen=(cs[denEnd]-'0')+curDen*10;
        if (numStart>=0 && cs[numStart]=='-')
          curNum=-curNum;

        int gcd=bestGCD(ResDen,curDen);
        int den=curDen*ResDen/gcd;
        ResNum = ResNum*(den/ResDen)+curNum*(den/curDen);
        ResDen=den;
      }
    int g=bestGCD(ResDen,Math.abs(ResNum));
    ResDen/=g;
    ResNum/=g;
    StringBuilder sb=new StringBuilder();
    sb.append(ResNum);
    sb.append('/');
    sb.append(ResDen);
    return sb.toString();
  }

  private int bestGCD(int a,int b){
    if (a==0 ||b==0)
      return a==0?b:a;
    if (a==b)
      return a;
    else if (b>a)
      return bestGCD(b,a);
    else{
      boolean isAOdd=(a&1)==0,isBOdd=(b&1)==0;
      if (isAOdd &&isBOdd )
        return bestGCD(a>>1,b>>1)<<1;
      else if (isAOdd && !isBOdd)
        return bestGCD(a>>1,b);
      else if ( !isAOdd&& isBOdd)
        return bestGCD(a,b>>1);
      else
        return bestGCD(a-b,b);
    }
  }

  private int gcd_divide(int a,int b){
    int c;
    if ((c=a%b)==0)
      return b;
    else
      return gcd1(b,c);
  }

  private int gcd_subtract(int a,int b){
    if (a==b)
      return a;
    else if (a>b)
      return gcd_subtract(a-b,b);
    else
      return gcd_subtract(b-a,a);
  }

  public int sumRootToLeaf(TreeNode root) {
   return SRLHelper(root,0);
  }

  private int SRLHelper(TreeNode root,int val){
    if (root==null)
      return 0;
    if (root.left==null && root.right==null)
      return (val<<1)+root.val;
    val = (val<<1)+root.val;
    return SRLHelper(root.left,val)+SRLHelper(root.right,val);
  }

  public int findLengthOfLCIS(int[] nums) {
    if (nums==null||nums.length==0)
      return 0;
    int start=0,end,max=0,N=nums.length;
    for (end=0;end<N;end++)
      if (end==N-1 || nums[end+1]<=nums[end]){
        max=Math.max(max,end-start+1);
        start=end+1;
      }
    return max;
  }

  public List<String> topKFrequent(String[] words, int k) {
    Map<String,Integer> mp=new HashMap<>();
    for (String w:words)
      mp.put(w,mp.getOrDefault(w,0)+1);
    PriorityQueue<Map.Entry<String,Integer>> pq=new PriorityQueue<>(new Comparator<Map.Entry<String, Integer>>() {
      @Override
      public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
        return o1.getValue()==o2.getValue()?o1.getKey().compareTo(o2.getKey()):o2.getValue()-o1.getValue();
      }
    });
    for (Map.Entry<String,Integer> et:mp.entrySet())
      pq.offer(et);
    List<String> ans=new ArrayList<>(k);
    for (int i=0;i<k;i++)
      ans.add(pq.poll().getKey());
    return ans;
  }

  public String reorganizeString1(String S) {
    if (S==null||S.length()==0)
      return "";
    int[] mp=new int[26];
    for (char s:S.toCharArray())
      mp[s-'a']++;
    StringBuilder sb=new StringBuilder();
    PriorityQueue<Character> pq=new PriorityQueue<>((a,b)->mp[b-'a']-mp[a-'a']);
    Queue<Character> q=new LinkedList<>();
    for (int i=0;i<26;i++)
      if (mp[i]>0)
        pq.offer((char)('a'+i));
    while(!pq.isEmpty()){
      char c=pq.poll();
      sb.append(c);
      mp[c-'a']--;
      q.offer(c);
      if (q.size()>1){
        Character temp=q.poll();
        if (mp[temp-'a']>0)
          pq.offer(temp);
      }
    }
    return sb.length()==S.length()?sb.toString():"";
  }

  public String reorganizeString(String S) {
    if (S==null||S.length()==0)
      return "";
    int N=S.length(),maxCount=0,index=0;
    char maxChar = 'a';
    int[] mp=new int[26];
    char[] res=new char[N];
    for (char c:S.toCharArray()){
      int i=c-'a';
      mp[i]++;
      if ((mp[i]<<1)>N+1)
        return "";
      if (mp[i]>maxCount){
        maxCount=mp[i];
        maxChar=c;
      }
    }

    while(mp[maxChar-'a']>0){
      res[index]=maxChar;
      index+=2;
      mp[maxChar-'a']--;
    }

    for (int i=0;i<26;i++){
      char cur=(char)(i+'a');
      while(mp[i]>0){
        if (index>=N)
          index=1;
        res[index]=cur;
        index+=2;
        mp[i]--;
      }
    }
    return new String(res);
  }

  class MLCNode{
    boolean isLeaf;
    MLCNode[] next;
    public MLCNode(){
      isLeaf=true;
      next=new MLCNode[26];
    }
  }

  private void MLCbuildTrie(MLCNode root,String key,int d){
    if (d<0)
      return;
    char cur=key.charAt(d);
    if (root.next[cur-'a']==null)
      root.next[cur-'a']=new MLCNode();
    root.isLeaf=false;
    MLCbuildTrie(root.next[cur-'a'],key,d-1);
  }

  private int MLCgetLength(MLCNode root,int val){
    if (root==null)
      return 0;
    if (root.isLeaf)
      return val+1;
    int ans=0;
    for (int i=0;i<26;i++)
      ans+=MLCgetLength(root.next[i],val+1);
    return ans;
  }

  public int minimumLengthEncoding(String[] words) {
    MLCNode root=new MLCNode();
    for (String w:words)
      MLCbuildTrie(root,w,w.length()-1);
    int ans=MLCgetLength(root,0);
    return ans;
  }

  public int climbStairs(int n) {
    if (n==1)
      return 1;
    if (n==2)
      return 2;
    int afterOne=2,afterTwo=1;
    for (int i=3;i<=n;i++){
      afterOne=afterOne+afterTwo;
      afterTwo=afterOne-afterTwo;
    }
    return afterOne;
  }

  public int findMinArrowShots(int[][] points) {
    if (points==null||points.length==0)
      return 0;
    Arrays.sort(points,(a,b)->a[1]-b[1]);
    int arrayPos=points[0][1],arrayCount=1;
    for (int i=1;i<points.length;i++){
      if (arrayPos>=points[i][0])
        continue;
      arrayCount++;
      arrayPos=points[i][1];
    }
    return arrayCount;
  }

  public int minDominoRotations(int[] A, int[] B) {
    int[] countA=new int[7],countB=new int[7],same=new int[7];
    for (int i=0;i<A.length;i++){
      countA[A[i]]++;
      countB[B[i]]++;
      if (A[i]==B[i])
        same[A[i]]++;
    }

    for (int i=1;i<7;i++)
      if (countA[i]+countB[i]-same[i]==A.length)
        return Math.min(countA[i],countB[i])-same[i];
    return -1;
  }

  public String findLongestWord1(String s, List<String> d) {
    Collections.sort(d,(a,b)->a.length()==b.length()?a.compareTo(b):b.length()-a.length());
    char[] cs=s.toCharArray();
    for (String dic:d){
      int i=0;
      for (char c:cs)
        if (i<dic.length() && c==dic.charAt(i))
          i++;
      if (i==dic.length())
        return dic;
    }
    return "";
  }

  private int FLWcompare(String a,String b){
    return a.length()==b.length()?a.compareTo(b):b.length()-a.length();
  }

  public String findLongestWord(String s, List<String> d) {
    char[] cs=s.toCharArray();
    String ans="";
    for (String dic:d){
      int i=0;
      char[] dc=dic.toCharArray();
      for (char c:cs)
        if (i<dic.length() && c==dc[i])
          i++;
      if (i==dic.length())
        if (FLWcompare(dic,ans)<0)
          ans=dic;
    }
    return ans;
  }

  public int hammingWeight(int n) {
    int count=0;
    while(n!=0){
      count++;
      n=n& (n-1);
    }
    return count;
  }

  public boolean isPowerOfTwo(int n) {
    if (n<=0)
      return false;
    return ((n-1)&n)==0?true:false;
  }

  public void sortColors(int[] nums) {
    int zero=0,two=nums.length-1;
    for (int i=0;i<two;i++)
      if (nums[i]==0){
        exchange(nums,i,zero);
        zero++;
      }
      else if (nums[i]==2){
        exchange(nums,i,two);
        two--;
        i--;
      }
  }

  public ListNode insertionSortList(ListNode head) {
    if (head==null||head.next==null)
      return head;
    ListNode cur=head,start=new ListNode(0),next;
    while(cur!=null){
      next=cur.next;
      ListNode temp=start;
      while(temp.next!=null && temp.next.val<=cur.val )
        temp=temp.next;

      cur.next=temp.next;
      temp.next=cur;
      cur=next;
    }
    return start.next;
  }

  public class Interval {
    int start;
    int end;

    Interval() {
      start = 0;
      end = 0;
    }

    Interval(int s, int e) {
      start = s;
      end = e;
    }
  }

  public List<Interval> merge(List<Interval> intervals) {
    if (intervals==null|| intervals.size()==0||intervals.size()==1)
      return intervals;
    Collections.sort(intervals,(a,b)->a.start-b.start);
    List<Interval> ans=new ArrayList<>(intervals.size());
    for (Interval it:intervals){
      Interval last=ans.size()>0?ans.get(ans.size()-1):null;
      if (last==null || last.end<it.start)
        ans.add(it);
      else
        last.end=Math.max(it.end,last.end);
    }
    return ans;
  }

  public boolean isPowerOfFour1(int num) {
    if (num<=0||(num&(num-1))!=0)
      return false;
    int count=0;
    while(num!=1){
      count++;
      num>>>=1;
    }
    return (count&1)==0?true:false;
  }

  public boolean isPowerOfFour(int num) {
    return num>0&&((num&(num-1))==0 && ((num&0x55555555)==num));
  }

  public String toHex(int num) {
    if (num==0)
      return "0";
    char[] intToHex=new char[]{'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
    StringBuilder sb=new StringBuilder();
    while(num!=0){
      int val=num & 15;
      num>>>=4;
      sb.insert(0,intToHex[val]);
    }
    return sb.toString();
  }

  public int maxAncestorDiff(TreeNode root) {
    if (root==null || (root.left==null && root.right==null))
      return 0;
    int[] ans=new int[1];
    MADgetMaxMin(root,ans);
    return ans[0];
  }

  private int[] MADgetMaxMin(TreeNode root,int[] ans){
    if (root.left==null && root.right==null)
      return new int[]{root.val,root.val};
    int[] leftMM=root.left==null?null:MADgetMaxMin(root.left,ans);
    int[] rightMM=root.right==null?null:MADgetMaxMin(root.right,ans);
    int[] res=new int[2];
    if (leftMM==null)
      res=rightMM;
    else if (rightMM==null)
      res=leftMM;
    else{
      res[0]=Math.max(leftMM[0],rightMM[0]);
      res[1]=Math.min(leftMM[1],rightMM[1]);
    }
    int maxCurDif=Math.max(Math.abs(root.val-res[0]),Math.abs(root.val-res[1]));
    ans[0]=Math.max(ans[0],maxCurDif);
    res[0]=Math.max(root.val,res[0]);
    res[1]=Math.min(root.val,res[1]);
    return res;
  }

  public TreeNode constructMaximumBinaryTree(int[] nums) {
    if (nums.length==1)
      return new TreeNode(nums[0]);
    return CMB(nums,0,nums.length-1);
  }

  private TreeNode CMB(int[] nums,int start,int end){
    if (start>end)
      return null;
    if (start==end)
      return new TreeNode(nums[start]);
    int maxVal=Integer.MIN_VALUE,maxLoc=-1;
    for (int i=start;i<=end;i++)
      if (nums[i]>maxVal){
        maxVal=nums[i];
        maxLoc=i;
      }
    TreeNode cur=new TreeNode(maxVal);
    cur.left=CMB(nums,start,maxLoc-1);
    cur.right=CMB(nums,maxLoc+1,end);
    return cur;
  }

  public TreeNode insertIntoMaxTree(TreeNode root, int val) {
    if (root==null)
      return new TreeNode(val);
    if (val>root.val){
      TreeNode cur=new TreeNode(val);
      cur.left=root;
      return cur;
    }
    else{
      root.right=insertIntoMaxTree(root.right,val);
      return root;
    }
  }

  public int distributeCoins(TreeNode root) {
    int[] res=new int[1];
    DCHelper(root,res);
    return res[0];
  }

  private int DCHelper(TreeNode root,int[] res){
    if (root==null)
      return 0;
    int L=DCHelper(root.left,res),R=DCHelper(root.right,res);
    res[0]+=Math.abs(L)+Math.abs(R);
    return root.val+L+R-1;
  }

  public boolean isSymmetric_Rec(TreeNode root) {
    if (root==null|| (root.left==null && root.right==null))
      return true;
    return ISR(root.left,root.right);
  }

  private boolean ISR(TreeNode r1,TreeNode r2){
    if (r1==null && r2==null)
      return true;
    if (r1==null||r2==null)
      return false;
    if (r1.val!=r2.val)
      return false;
    boolean LR=ISR(r1.left,r2.right),RL=ISR(r1.right,r2.left);
    return LR&&RL;
  }

  public boolean isSymmetric_ite_stack(TreeNode root) {
    if (root==null||(root.left==null && root.right==null))
      return true;
    Stack<TreeNode> left=new Stack<>(),right=new Stack<>();
    TreeNode pLeft=root.left,pRight=root.right;
    while(pLeft!=null || !left.isEmpty() ||pRight!=null ||!right.isEmpty())
      if (pLeft!=null && pRight!=null){
        if (pLeft.val!=pRight.val)
          return false;
        left.push(pLeft);
        right.push(pRight);
        pLeft=pLeft.left;
        pRight=pRight.right;
      }
      else if (pLeft !=null ||pRight!=null)
        return false;
      else{
        TreeNode tempLeft=left.pop(),tempRight=right.pop();
        pLeft=tempLeft.right;
        pRight=tempRight.left;
      }
    return true;
  }

  public boolean isSymmetric_ite_Morris(TreeNode root) {
    if (root==null || (root.left==null &&root.right==null))
      return true;
    TreeNode pLeft=root.left,pRight=root.right;
    while(pLeft!=null && pRight!=null)
      if (pLeft.left==null && pRight.right==null){
        if (pLeft.val!=pRight.val)
          return false;
        pLeft=pLeft.right;
        pRight=pRight.left;
      }
      else if (pLeft.left==null || pRight.right==null)
        return false;
      else{
        TreeNode leftNext=pLeft.left,rightNext=pRight.right;
        while(leftNext.right!=null && leftNext.right!=pLeft)
          leftNext=leftNext.right;
        while(rightNext.left!=null && rightNext.left!=pRight)
          rightNext=rightNext.left;
        if (leftNext.right==null && rightNext.left==null){
          if (pLeft.val!=pRight.val)
            return false;
          leftNext.right=pLeft;
          rightNext.left=pRight;
          pLeft=pLeft.left;
          pRight=pRight.right;
        }
        else if (leftNext.right==pLeft && rightNext.left==pRight){
          leftNext.right=null;
          rightNext.left=null;
          pLeft=pLeft.right;
          pRight=pRight.left;
        }
        else
          return false;
      }
    return (pLeft!=null||pRight!=null)?false:true;
  }

  public int findSecondMinimumValue(TreeNode root) {
    if (root==null)
      return -1;
    int ans=FSM(root,root.val);
    return ans;
  }

  private int FSM(TreeNode root,int val){
    if (root==null)
      return -1;
    if (root.val>val)
      return root.val;
    int L=FSM(root.left,val),R=FSM(root.right,val);
    if (L!=-1 && R!=-1)
      return Math.min(L,R);
    else if (L==-1 && R==-1)
      return -1;
    else
      return Math.max(L,R);
  }

  public int findTargetSumWays1(int[] nums, int S) {
    if (nums==null ||nums.length==0)
      return 0;
    int[] ans=new int[1];
    FTS(nums,0,S,ans);
    return ans[0];
  }

  private void FTS(int[] nums,int index,int remain,int[] ans){
    if (index==nums.length){
      if (remain==0)
        ans[0]++;
      return;
    }
    FTS(nums,index+1,remain-nums[index],ans);
    FTS(nums,index+1,remain+nums[index],ans);
  }

  public String decodeString(String s) {
    if (s==null||s.length()==0)
      return "";
    int N=s.length();
    int[] mp=new int[N];
    int[] st=new int[N];
    int index=0;
    for (int i=0;i<N;i++)
      if (s.charAt(i)=='[')
        st[index++]=i;
      else if (s.charAt(i)==']')
        mp[st[--index]]=i;
    return DS(s,0,N-1,mp);
  }

  private String DS(String s,int start,int end,int[] mp){
    if (start>end)
      return "";
    if (start==end)
      return s.substring(start,end+1);
    StringBuilder sb=new StringBuilder();
    for (int i=start;i<=end;i++){
      char cur=s.charAt(i);
      if (cur>='0' &&cur<='9'){
        int e=i+1;
        while(Character.isDigit(s.charAt(e)))
          e++;
        int k=Integer.valueOf(s.substring(i,e));
        int over=mp[e];
        String next=DS(s,e+1,over-1,mp);
        for (int j=0;j<k;j++)
          sb.append(next);
        i=over;
      }
      else
        sb.append(cur);
    }
    return sb.toString();
  }

  public List<Integer> distanceK1(TreeNode root, TreeNode target, int K){
    List<Integer> ans=new ArrayList<>();
    if (K==0){
      ans.add(target.val);
      return ans;
    }
    DKfindDistanceK(target,K,ans);
    List<TreeNode> path=new ArrayList<>();
    DKfindPath(root,target,new Stack<>(),path);
    int pathLen=path.size(),start=pathLen<=K?0:pathLen-K;
    for (int i=start;i<pathLen;i++){
      int dis=pathLen-i;
      if (dis==K){
        ans.add(path.get(i).val);
        continue;
      }
      TreeNode next=i==pathLen-1?target:path.get(i+1),cur=path.get(i);
      TreeNode sibling=cur.left==next?cur.right:cur.left;
      DKfindDistanceK(sibling,K-dis-1,ans);
    }

    return ans;
  }

  private void DKfindPath(TreeNode root,TreeNode target,Stack<TreeNode> path,List<TreeNode> ans){
    if (root==null)
      return;
    if (ans.size()!=0)
      return;
    if (root==target){
      ans.addAll(path);
      return;
    }
    path.push(root);
    DKfindPath(root.left,target,path,ans);
    DKfindPath(root.right,target,path,ans);
    path.pop();
  }

  private void DKfindDistanceK(TreeNode root,int remainD,List<Integer> ans){
    if (root==null)
      return;
    if (remainD==0){
      ans.add(root.val);
      return;
    }
    DKfindDistanceK(root.left,remainD-1,ans);
    DKfindDistanceK(root.right,remainD-1,ans);
  }

  public List<Integer> distanceK(TreeNode root, TreeNode target, int K){
    List<Integer> ans=new ArrayList<>();
    if (K==0){
      ans.add(target.val);
      return ans;
    }
    Map<TreeNode,List<TreeNode>> mp=new HashMap<>();
    DKconstructMap(mp,root,null);
    DKfindDistanceK(target,mp,ans,K);
    return ans;
  }

  private void DKfindDistanceK(TreeNode target,Map<TreeNode,List<TreeNode>> mp,List<Integer> ans,int K){
    Queue<TreeNode> q=new LinkedList<>();
    q.offer(target);
    Set<TreeNode> visited=new HashSet<>();
    visited.add(target);
    while (!q.isEmpty()){
      int size=q.size();
      for (int i=0;i<size;i++){
        TreeNode temp=q.poll();
        List<TreeNode> nextDis=mp.get(temp);
        for (TreeNode t:nextDis){
          if (visited.contains(t))
            continue;
          q.offer(t);
          visited.add(t);
        }
      }
      if (--K==0){
        while (!q.isEmpty())
          ans.add(q.poll().val);
        return;
      }
    }
  }

  private void DKconstructMap(Map<TreeNode,List<TreeNode>> res,TreeNode root,TreeNode parents){
    if (root==null)
      return;
    if (!res.containsKey(root))
      res.put(root,new ArrayList<>());
    List<TreeNode> connected=res.get(root);
    if (root.left!=null)
      connected.add(root.left);
    if (root.right!=null)
      connected.add(root.right);
    if (parents!=null)
      connected.add(parents);
    DKconstructMap(res,root.left,root);
    DKconstructMap(res,root.right,root);
  }

  public int minDepth(TreeNode root) {
    if (root==null)
      return 0;
    if (root.left==null&&root.right==null)
      return 1;
    Queue<TreeNode> q=new LinkedList<>();
    int count=0;
    q.offer(root);
    while (!q.isEmpty()){
      int size=q.size();
      count++;
      for (int i=0;i<size;i++){
        TreeNode temp=q.poll();
        if (temp.left==null&&temp.right==null)
          return count;
        if (temp.left!=null)
          q.offer(temp.left);
        if (temp.right!=null)
          q.offer(temp.right);
      }
    }
    return count;
  }

  public boolean isBalanced(TreeNode root) {
    if (root==null||(root.left==null&&root.right==null))
      return true;
    boolean[] res=new boolean[1];
    res[0]=true;
    ISHelper(root,res);
    return res[0];
  }

  private int ISHelper(TreeNode root,boolean[] res){
    if (res[0]==false)
      return 0;
    if (root==null)
      return 0;
    int L=ISHelper(root.left,res),R=ISHelper(root.right,res);
    if (Math.abs(L-R)>1)
      res[0]=false;
    return Math.max(L,R)+1;
  }

  public int numIslands1(char[][] grid) {
    if (grid==null)
      return 0;
    if (grid.length==0 ||grid[0].length==0)
      return 0;
    int R=grid.length,C=grid[0].length;
    boolean[][] visited=new boolean[R][C];
    int count=0;
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++)
        if (!visited[i][j] && grid[i][j]=='1'){
          count++;
          NILHelper(grid,i,j,visited);
        }
    return count;
  }

  private void NILHelper(char[][] grid,int r,int c,boolean[][] visited){
    if (r<0||r>=grid.length ||c<0 ||c>=grid[0].length ||grid[r][c]!='1'|| visited[r][c])
      return;
    visited[r][c]=true;
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    for (int i=0;i<4;i++)
      NILHelper(grid,r+dirs[i][0],c+dirs[i][1],visited);
  }

  int[] NILweights;
  int[] NILids;
  int NILcount;
  public int numIslands(char[][] grid) {
    if (grid==null)
      return 0;
    if (grid.length==0 ||grid[0].length==0)
      return 0;
    int R=grid.length,C=grid[0].length;
    int num=R*C;
    NILcount=0;
    NILweights=new int[num];
    NILids=new int[num];
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++){
        int id=i*C+j;
        NILids[id]=id;
        NILweights[id]=1;
        if (grid[i][j]=='1')
          NILcount++;
      }
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++){
        if (r!=R-1 && grid[r][c]=='1' && grid[r+1][c]=='1')
          NILunion(r,c,r+1,c,R,C);
        if (c!=C-1&& grid[r][c]=='1' && grid[r][c+1]=='1')
          NILunion(r,c,r,c+1,R,C);
      }
    return NILcount;
  }

  private int NILfind(int r,int c,int R,int C){
    int initialId=r*C+c;
    if (NILids[initialId]==initialId)
      return initialId;
    int temp=initialId;
    while(initialId!=NILids[initialId])
      initialId=NILids[initialId];

    while(temp!=initialId){
      int next=NILids[temp];
      NILids[temp]=initialId;
      temp=next;
    }
    return initialId;
  }

  private void NILunion(int r1,int c1,int r2,int c2,int R,int C){
    int id1=NILfind(r1,c1,R,C),id2=NILfind(r2,c2,R,C);
    if (id1==id2)
      return;
    if (NILweights[id2]<=NILweights[id1]){
      NILids[id2]=id1;
      NILweights[id1]+=NILweights[id2];
    }
    else{
      NILids[id1]=id2;
      NILweights[id2]+=NILweights[id1];
    }
    NILcount--;
  }

  public int shortestBridge(int[][] A) {
    int R=A.length,C=A[0].length;
    if (R==1)
      return 0;
    boolean[][] visited=new boolean[R][C];
    Queue<int[]> group=new LinkedList<>();
    findGroup:for (int i=0;i<R;i++)
      for (int j=0;j<C;j++)
        if (A[i][j]==1){
          SBfindGroup(i,j,A,visited,group);
          break findGroup;
        }
    int step=0;
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    while(!group.isEmpty()){
      int size=group.size();
      for (int i=0;i<size;i++){
        int[] loc=group.poll();
        for (int[] d:dirs){
          int rt=loc[0]+d[0],ct=loc[1]+d[1];
          if (rt>=0 && rt<R &&ct>=0 &&ct<C && !visited[rt][ct]){
            if (A[rt][ct]==0)
              group.offer(new int[]{rt,ct});
            else
              return step;
            visited[rt][ct]=true;
          }
        }
      }
      step++;
    }
    return step;
  }

  private void SBfindGroup(int r,int c,int[][] A,boolean[][] visited,Queue<int[]> group){
    if (r<0||r>=A.length ||c<0||c>=A[0].length||visited[r][c]||A[r][c]!=1)
      return;
    visited[r][c]=true;
    group.offer(new int[]{r,c});
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    for (int[] d:dirs)
      SBfindGroup(r+d[0],c+d[1],A,visited,group);
  }

  public int openLock(String[] deadends, String target) {
    if (target.equals("0000"))
      return 0;
    Set<String> dead=new HashSet<>(),visited=new HashSet<>();
    for (String s:deadends)
      dead.add(s);
    if (dead.contains("0000"))
      return -1;
    Queue<String> q=new LinkedList<>();
    q.offer("0000");
    int ans=0,len=4;
    while (!q.isEmpty()){
      ans++;
      int size=q.size();
      for (int i=0;i<size;i++){
        String cur=q.poll();
        char[] curChars=cur.toCharArray();
        for (int j=0;j<len;j++){
          char w=cur.charAt(j);
          char before= w=='9'?'0': (char) (w + 1),after=w=='0'?'9':(char)(w-1);
          String[] nexts=new String[2];
          curChars[j]=before;
          nexts[0]=new String(curChars);
          curChars[j]=after;
          nexts[1]=new String(curChars);
          curChars[j]=w;
          for (String n:nexts)
            if (!visited.contains(n) && !dead.contains(n)){
              if (target.equals(n))
                return ans;
              q.offer(n);
              visited.add(n);
            }
        }
      }
    }
    return -1;
  }

  class MyQueue {
    private Stack<Integer> output;
    private Stack<Integer> input;
    int size;
    /** Initialize your data structure here. */
    public MyQueue() {
      output=new Stack<>();
      input =new Stack<>();
      size=0;
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
      input.push(x);
      size++;
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
      if (size==0)
        throw new IllegalArgumentException();
      size--;
      if (!output.isEmpty())
        return output.pop();
      while(!input.isEmpty())
        output.push(input.pop());
      return output.pop();
    }

    /** Get the front element. */
    public int peek() {
      if (size==0)
        throw new IllegalArgumentException();
      if (!output.isEmpty())
        return output.peek();
      while(!input.isEmpty())
        output.push(input.pop());
      return output.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
      return size==0;
    }
  }

  class RandomizedSet1 {
    Map<Integer,Integer> data;
    Random r;
    /** Initialize your data structure here. */
    public RandomizedSet1() {
      data=new HashMap<>();
      r=new Random();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
      if (data.containsKey(val))
        return false;
      data.put(val,0);
      return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
      if (!data.containsKey(val))
        return false;
      data.remove(val);
      return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
      int size=data.size();
      int index=r.nextInt(size);
      return data.keySet().toArray(new Integer[0])[index];
    }
  }

  class RandomizedSet {
    Map<Integer,Integer> data;
    List<Integer> keys;
    /** Initialize your data structure here. */
    public RandomizedSet() {
      data=new HashMap<>();
      keys=new ArrayList<>();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
      if (data.containsKey(val))
        return false;
      data.put(val,keys.size());
      keys.add(val);
      return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
      if (!data.containsKey(val))
        return false;
      int index=data.get(val);
      if (index==keys.size()-1)
        keys.remove(index);
      else{
        int temp=keys.get(keys.size()-1);
        keys.set(index,temp);
        keys.remove(keys.size()-1);
        data.put(temp,index);
      }
      data.remove(val);
      return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
      return keys.get((int)(Math.random()*keys.size()));
    }
  }

  class DirectedEdge{
    int from;
    int to;
    int weight;
    public DirectedEdge(int f,int t,int w){
      from=f;
      to=t;
      weight=w;
    }
  }

  public int networkDelayTime(int[][] times, int N, int K) {
    if (N==1)
      return 0;
    ArrayList<DirectedEdge>[] map=new ArrayList[N+1];
    for (int i=1;i<=N;i++)
      map[i]=new ArrayList<>();
    for (int[] edge:times)
      map[edge[0]].add(new DirectedEdge(edge[0],edge[1],edge[2]));
    IndexPQ pq=new IndexPQ(N);
    int[] distTo=new int[N+1];
    Arrays.fill(distTo,Integer.MAX_VALUE);
    distTo[K]=0;
    pq.insert(K,0);
    while(!pq.isEmpty())
      relax(map,pq.delMin(),distTo,pq);
    int ans=0;
    for (int i=1;i<=N;i++)
      ans=Math.max(distTo[i],ans);
    return ans==Integer.MAX_VALUE?-1:ans;
  }

  private void relax(ArrayList<DirectedEdge>[] map,int v,int[] distTo,IndexPQ pq){
    for (DirectedEdge e:map[v]){
      int to=e.to;
      if (distTo[to] >distTo[v]+e.weight){
        distTo[to]=distTo[v]+e.weight;
        if (pq.contains(to))
          pq.change(to,distTo[to]);
        else
          pq.insert(to,distTo[to]);
      }
    }
  }

  class IndexPQ{
    private int maxN;
    private int count;
    private int[] locToIndex;
    private int[] indexToLoc;
    private int[] keys;

    public IndexPQ(int N){
      maxN=N;
      keys=new int[N+1];
      locToIndex=new int[N+1];
      indexToLoc=new int[N+1];
      Arrays.fill(indexToLoc,-1);
    }

    public boolean isEmpty(){
      return count==0;
    }

    public boolean contains(int index){
      if (index<=0 ||index>maxN)
        throw new IllegalArgumentException();
      return indexToLoc[index]!=-1;
    }

    public int delMin(){
      if (count==0)
        throw new NoSuchElementException();
      int min=locToIndex[1];
      exch(1,count--);
      sink(1);
      indexToLoc[min]=-1;
      keys[min]=0;
      locToIndex[count+1]=-1;
      return min;
    }

    public void change(int index,int key){
      if (index<0 || index>maxN || !contains(index))
        throw new IllegalArgumentException();
      keys[index]=key;
      swim(indexToLoc[index]);
      sink(indexToLoc[index]);
    }

    public void insert(int index,int key){
      if (index<= 0||index>maxN||contains(index))
        throw new IllegalArgumentException();
      count++;
      indexToLoc[index]=count;
      locToIndex[count]=index;
      keys[index]=key;
      swim(count);
    }

    private void swim(int k){
      while(k>1 && keys[locToIndex[k>>1]]>keys[locToIndex[k]]){
        exch(k,k>>1);
        k=k>>1;
      }
    }

    private void sink(int k){
      while((k<<1)<=count){
        int j=k<<1;
        if (j<count && greater(j,j+1))
          j++;
        if (!greater(k,j))
          break;
        exch(k,j);
        k=j;
      }
    }

    private boolean greater(int i,int j){
      return keys[locToIndex[i]]>keys[locToIndex[j]];
    }

    private void exch(int i,int j){
      int swap=locToIndex[i];
      locToIndex[i]=locToIndex[j];
      locToIndex[j]=swap;
      indexToLoc[locToIndex[i]]=i;
      indexToLoc[locToIndex[j]]=j;
    }
  }

  public boolean isBipartite(int[][] graph) {
    if (graph.length==1)
      return true;
    boolean[] isBipartite=new boolean[1];
    isBipartite[0]=true;
    int len=graph.length;
    boolean[] visited=new boolean[len];
    boolean[] isColored=new boolean[len];
    for (int i=0;i<len;i++){
      if (!isBipartite[0])
        break;
      if (!visited[i])
        IBHelper(i,graph,visited,isBipartite,isColored);
    }
    return isBipartite[0];
  }

  private void IBHelper(int index,int[][] graph,boolean[] visited,boolean[] isBi,boolean[] isColored){
    if (!isBi[0])
      return;
    visited[index]=true;
    for (int adj:graph[index])
      if (!visited[adj]){
        isColored[adj]=!isColored[index];
        IBHelper(adj,graph,visited,isBi,isColored);
      }
      else if (isColored[index]==isColored[adj])
        isBi[0]=false;
  }

  public List<Integer> eventualSafeNodes(int[][] graph) {
    List<Integer> ans=new ArrayList<>();
    if (graph==null||graph.length==0)
      return ans;
    int N=graph.length;
    int[] color=new int[N];
    for (int i=0;i<N;i++)
      if (ESNdetectCircle(graph,i,color))
        ans.add(i);
    return ans;
  }

  private boolean ESNdetectCircle(int[][] graph,int index,int[] color){
    if (color[index]!=0)
      return color[index]==1;
    color[index]=2;
    for (int adj:graph[index])
      if (!ESNdetectCircle(graph,adj,color))
        return false;
    color[index]=1;
    return true;
  }

  class MyStack1 {

    /** Initialize your data structure here. */
    Queue<Integer> q1,q2;
    int top;
    public MyStack1() {
      q1=new LinkedList<>();
      q2=new LinkedList<>();
      top=0;
    }

    /** Push element x onto stack. */
    public void push(int x) {
      q1.offer(x);
      top=x;
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
      int n=q1.size();
      for (int i=0;i<n-1;i++){
        if (i==n-2)
          top=q1.peek();
        q2.offer(q1.poll());
      }
      int ans=q1.poll();
      Queue<Integer> temp=q2;
      q2=q1;
      q1=temp;
      return ans;
    }

    /** Get the top element. */
    public int top() {
      return top;
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
      return q1.size()==0;
    }
  }

  class MyStack {

    /** Initialize your data structure here. */
    Queue<Integer> q1,q2;
    public MyStack() {
      q1=new LinkedList<>();
      q2=new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
      q2.offer(x);
      while(!q1.isEmpty())
        q2.offer(q1.poll());
      Queue<Integer> temp=q1;
      q1=q2;
      q2=temp;
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
      return q1.poll();
    }

    /** Get the top element. */
    public int top() {
      return q1.peek();
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
      return q1.isEmpty();
    }
  }

  class PeekingIterator implements Iterator<Integer> {
    List<Integer> data;
    int index;
    public PeekingIterator(Iterator<Integer> iterator) {
      // initialize any member here.
      data=new ArrayList<>();
      index=0;
      while (iterator.hasNext())
        data.add(iterator.next());
    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
      return data.get(index);
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
      return data.get(index++);
    }

    @Override
    public boolean hasNext() {
      return index<data.size();
    }
  }

  int[] EPweight;
  int[] EPid;
  public boolean equationsPossible(String[] equations) {
    EPweight=new int[26];
    EPid=new int[26];
    for (int i=0;i<26;i++){
      EPweight[i]=1;
      EPid[i]=i;
    }
    List<String> notEquals=new ArrayList<>();
    for (String eq:equations)
      if (eq.charAt(1)=='!')
        notEquals.add(eq);
      else
        EPunion(eq.charAt(0),eq.charAt(3));
    for (String ne:notEquals)
      if (EPfind(ne.charAt(0)) ==EPfind(ne.charAt(3)))
        return false;
    return true;
  }

  private int EPfind(char i){
    int index=i-'a';
    if (EPid[index]==index)
      return index;
    int temp=index;
    while(EPid[index]!=index)
      index=EPid[index];

    while(EPid[temp]!=index){
      int next=EPid[temp];
      EPid[temp]=index;
      temp=next;
    }
    return index;
  }

  private void EPunion(char i,char j){
    int id1=EPfind(i),id2=EPfind(j);
    if (id1==id2)
      return;
    if (EPweight[id1]>=EPweight[id2]){
      EPid[id2]=id1;
      EPweight[id1]+=EPweight[id2];
    }
    else{
      EPid[id1]=id2;
      EPweight[id2]+=id1;
    }
  }

  public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<Integer>[] map=new List[numCourses];
    for (int i=0;i<numCourses;i++)
      map[i]=new ArrayList<>();
    for (int[] edge:prerequisites)
      map[edge[0]].add(edge[1]);
    return !CFhasCircle(map);
  }

  private boolean CFhasCircle(List<Integer>[] map){
    int N=map.length;
    boolean[] visited=new boolean[N],onStack=new boolean[N],ans=new boolean[1];
    for (int i=0;i<N;i++)
      if (ans[0])
        break;
      else if (!visited[i])
        CFcircleDetector(map,i,visited,onStack,ans);
    return ans[0];
  }

  private void CFcircleDetector(List<Integer>[] map,int index,boolean[] visited,boolean[] onStack,boolean[] ans){
    if (ans[0])
      return;
    visited[index]=true;
    onStack[index]=true;
    for (int adj:map[index])
      if (ans[0])
        break;
      else if (!visited[adj])
        CFcircleDetector(map,adj,visited,onStack,ans);
      else if (onStack[adj])
        ans[0]=true;
    onStack[index]=false;
  }

  public List<Boolean> camelMatch(String[] queries, String pattern) {
    List<Boolean> ans=new ArrayList<>(queries.length);
    char[] p=pattern.toCharArray();
    for (String q:queries)
      if (CMmatch(q.toCharArray(),p))
        ans.add(true);
      else
        ans.add(false);
    return ans;
  }

  private boolean CMmatch(char[] query,char[] pattern){
    int p=0;
    for (int q=0;q<query.length;q++)
      if (p<pattern.length && pattern[p]==query[q])
        p++;
      else if (Character.isUpperCase(query[q]))
        return false;
    return p==pattern.length;
  }

  public int[] findOrder1(int numCourses, int[][] prerequisites) {
    List<Integer>[] map=new List[numCourses];
    for (int i=0;i<numCourses;i++)
      map[i]=new ArrayList<>();
    for (int[] p:prerequisites)
      map[p[1]].add(p[0]);
    Stack<Integer> res=new Stack<>();
    if (FOhasCircle(map,res))
      return new int[0];
    int[] ans=new int[numCourses];
    for (int i=0;i<numCourses;i++)
      ans[i]=res.pop();
    return ans;
  }

  private boolean FOhasCircle(List<Integer>[] map,Stack<Integer> topo){
    int N=map.length;
    boolean[] visited=new boolean[N],onStack=new boolean[N],hasCircle=new boolean[1];
    for (int i=0;i<N;i++)
      if (hasCircle[0])
        break;
      else if (!visited[i])
        FOcircleDetector(map,i,visited,onStack,hasCircle,topo);
    return hasCircle[0];
  }

  private void FOcircleDetector(List<Integer>[] map,int index,boolean[] visited,boolean[] onStack,boolean[] hasCircle,Stack<Integer> topo){
    if (hasCircle[0])
      return;
    visited[index]=true;
    onStack[index]=true;
    for (int adj:map[index])
      if (hasCircle[0])
        return;
      else if (!visited[adj])
        FOcircleDetector(map,adj,visited,onStack,hasCircle,topo);
      else if (onStack[adj]){
        hasCircle[0]=true;
        return;
      }
    onStack[index]=false;
    topo.push(index);
  }

  public int[] findOrder(int numCourses, int[][] prerequisites) {
    int[] topo=new int[numCourses];
    int index=0;
    Queue<Integer> q=new LinkedList<>();
    int[] inDegree=new int[numCourses];
    List<Integer>[] map=new List[numCourses];
    for (int i=0;i<numCourses;i++)
      map[i]=new ArrayList<>();
    for (int[] p:prerequisites){
      map[p[1]].add(p[0]);
      inDegree[p[0]]++;
    }
    for (int i=0;i<numCourses;i++)
      if (inDegree[i]==0)
        q.offer(i);
    while (!q.isEmpty()){
      int course=q.poll();
      topo[index++]=course;
      for (int adj:map[course]){
        inDegree[adj]--;
        if (inDegree[adj]==0)
          q.offer(adj);
      }
    }
    if (index==numCourses)
      return topo;
    else
      return new int[0];
  }

  class Trie {
    class Node{
      boolean val;
      Node[] next;
      public Node(){
        val=false;
        next=new Node[26];
      }
    }

    private Node root;
    Set<String> contains;
    /** Initialize your data structure here. */
    public Trie() {
      root=new Node();
      contains=new HashSet<>();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
      insert(root,word,0);
      contains.add(word);
    }

    private void insert(Node root,String word,int depth){
      if (depth==word.length()){
        root.val=true;
        return;
      }
      int index=word.charAt(depth)-'a';
      if (root.next[index]==null)
        root.next[index]=new Node();
      insert(root.next[index],word,depth+1);
    }
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
      return contains.contains(word);
//      return search(root,word,0);
    }

//    private boolean search(Node root,String word,int depth){
//      if (root==null)
//        return false;
//      if (depth==word.length())
//        return root.val;
//      int index=word.charAt(depth)-'a';
//      return search(root.next[index],word,depth+1);
//    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
      Node pre=getNode(root,prefix,0);
      return pre!=null;
    }

    private Node getNode(Node root,String prefix,int depth){
      if (root==null)
        return null;
      if (depth==prefix.length())
        return root;
      int index=prefix.charAt(depth)-'a';
      return getNode(root.next[index],prefix,depth+1);
    }
  }

  public List<String> findItinerary(String[][] tickets) {
    Map<String,PriorityQueue<String>> map=new HashMap<>();
    for (String[] t:tickets){
      map.putIfAbsent(t[0],new PriorityQueue<>());
      map.get(t[0]).offer(t[1]);
    }
    LinkedList<String> ls=new LinkedList<>();
    FIHelper(map,"JFK",ls);
    return ls;
  }

  private void FIHelper(Map<String,PriorityQueue<String>> map,String start,LinkedList<String> res){
   PriorityQueue<String> des=map.get(start);
   while(des!=null && des.size()>0)
     FIHelper(map,des.poll(),res);
   res.addFirst(start);
  }

  class NumArray1 {
    class Node{
      int left,right,val,mid=-1;
      Node leftSub,rightSub;
      public Node(int l,int r,int v){
        left=l;
        right=r;
        val=v;
      }

      public Node(int l,int r){
        left=l;
        right=r;
      }
    }

    Node root;
    public NumArray1(int[] nums) {
      root=buildTree(nums,0,nums.length-1);
    }

    private Node buildTree(int[] nums,int lo,int hi){
      if (lo==hi)
        return new Node(lo,hi,nums[lo]);
      if (lo>hi)
        return null;
      int mid = (hi+lo)>>1;
      Node temp=new Node(lo,hi);
      temp.mid=mid;
      temp.leftSub=buildTree(nums,lo,mid);
      temp.rightSub=buildTree(nums,mid+1,hi);
      temp.val=temp.leftSub.val+temp.rightSub.val;
      return temp;
    }

    public void update(int i, int val) {
      update(root,i,val);
    }

    private int update(Node root,int i,int val){
      if (root.left==i&&root.right==i){
        int dif=val-root.val;
        root.val=val;
        return dif;
      }
      int dif=root.mid>=i?update(root.leftSub,i,val):update(root.rightSub,i,val);
      root.val+=dif;
      return dif;
    }

    public int sumRange(int i, int j) {
      return sumRange(root,i,j);
    }

    private int sumRange(Node root,int left,int right){
      if (root.left==left&&root.right==right)
        return root.val;
      if (root.mid<left)
        return sumRange(root.rightSub,left,right);
      else if (root.mid>=right)
        return sumRange(root.leftSub,left,right);
      else
        return sumRange(root.leftSub,left,root.mid)+sumRange(root.rightSub,root.mid+1,right);
    }
  }

  class MyCalendarTwo {
    class SegNode{
      int left,right,mid=-1,count=0;
      SegNode leftSub,rightSub;
      public SegNode(int l,int r,int v){
        left=l;
        right=r;
        count=v;
      }
    }

    private SegNode root;
    public MyCalendarTwo() {
      root=new SegNode(0,Integer.MAX_VALUE,0);
    }
    private int query(SegNode root, int start, int end) {
      if ((start >= end) || (root == null)) {
        return 0;
      }

      if (start >= root.right) {
        return query(root.rightSub, start, end);
      } else if (end <= root.left) {
        return query(root.leftSub, start, end);
      } else {
        return Math.max(root.count, Math.max(query(root.leftSub, start, Math.min(end, root.left)),
                query(root.rightSub, Math.max(start, root.right), end)));
      }
    }

//    private int queryMaxCount(SegNode root,int left,int right){
//      if (root.mid==-1){
//        return root.count;
//      }
//      if (root.left>=left && root.right<=right){
//        int max=Math.max(queryMaxCount(root.leftSub,left,root.mid),queryMaxCount(root.rightSub,root.mid,right));
//        return Math.max(max,root.count);
//      }
//      if (root.mid>=right)
//        return Math.max(root.count,queryMaxCount(root.leftSub,left,right));
//      else if (root.mid<left)
//        return Math.max(root.count,queryMaxCount(root.rightSub,left,right));
//      else{
//        int max=Math.max(queryMaxCount(root.leftSub,left,right),queryMaxCount(root.rightSub,left,right));
//        return Math.max(max,root.count);
//      }
//    }

    private void update(SegNode root,int left,int right){
      if (root.left>=left && root.right<=right){
        root.count++;
        if (root.mid!=-1){
          update(root.leftSub,left,root.mid);
          update(root.rightSub,root.mid,right);
        }
        return;
      }
      if (root.mid==-1)
        if (root.right==right){
          root.mid=left;
          root.leftSub=new SegNode(root.left,left,root.count);
          root.rightSub=new SegNode(left,root.right,root.count);
          update(root.rightSub,left,right);
        }
        else if (root.left==left){
          root.mid=right;
          root.leftSub=new SegNode(root.left,right,root.count);
          root.rightSub=new SegNode(right,root.right,root.count);
          update(root.leftSub,left,right);
        }
        else{
          root.mid=right;
          root.leftSub=new SegNode(root.left,right,root.count);
          root.rightSub=new SegNode(right,root.right,root.count);
          update(root.leftSub,left,right);
        }
      else
        if (root.mid>=right)
          update(root.leftSub,left,right);
        else if (root.mid<=left)
          update(root.rightSub,left,right);
        else{
          update(root.leftSub,left,root.mid);
          update(root.rightSub,root.mid,right);
        }
    }

    public boolean book(int start, int end) {
      if (query(root,start,end)<2){
        update(root,start,end);
        return true;
      }
      return false;
    }
  }

  public int longestUnivaluePath1(TreeNode root) {
    if (root==null || (root.left==null && root.right==null))
      return 0;
    Map<TreeNode,List<TreeNode>> map=new HashMap<>();
    LUPbuildMap(map,null,root);
    int res=0;
    for (TreeNode cur:map.keySet())
      res=Math.max(res,LUPsearchLongestPath(map,cur,new HashSet<>())-1);
    return res;
  }

  private int LUPsearchLongestPath(Map<TreeNode,List<TreeNode>> map,TreeNode cur,Set<TreeNode> visited){
    if (visited.contains(cur))
      return 0;
    visited.add(cur);
    int ans=0;
    for (TreeNode adj:map.get(cur))
      if (adj.val==cur.val)
        ans=Math.max(ans,LUPsearchLongestPath(map,adj,visited));
    return ans+1;
  }

  private void LUPbuildMap(Map<TreeNode,List<TreeNode>> map,TreeNode parent,TreeNode cur){
    if (cur==null)
      return;
    map.putIfAbsent(cur,new ArrayList<>());
    List<TreeNode> nexts=map.get(cur);
    if (parent!=null)
      nexts.add(parent);
    if (cur.left!=null)
      nexts.add(cur.left);
    if (cur.right!=null)
      nexts.add(cur.right);
    LUPbuildMap(map,cur,cur.left);
    LUPbuildMap(map,cur,cur.right);
  }

  public int longestUnivaluePath(TreeNode root) {
    int[] res=new int[1];
    LUPHelper(root,-1,res);
    return res[0];
  }

  private int LUPHelper(TreeNode root,int parentVal,int[] res){
    if (root==null)
      return 0;
    int L=LUPHelper(root.left,root.val,res);
    int R=LUPHelper(root.right,root.val,res);
    res[0]=Math.max(res[0],L+R);
    if (root.val==parentVal)
      return Math.max(L,R)+1;
    else
      return 0;
  }

  public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    List<Integer> ans=new ArrayList<>();
    if (n==0)
      return ans;
    if (n==1){
      ans.add(0);
      return ans;
    }
    int[] degree=new int[n];
    int remain=n;
    List<Integer>[] map=new List[n];
    for (int i=0;i<n;i++)
      map[i]=new ArrayList<>();
    for (int[] e:edges){
      map[e[0]].add(e[1]);
      map[e[1]].add(e[0]);
      degree[e[0]]++;
      degree[e[1]]++;
    }
    Queue<Integer> q=new LinkedList<>();
    for (int i=0;i<n;i++)
      if (degree[i]==1)
        q.offer(i);
    while(!q.isEmpty()){
      if (remain==1 || remain==2){
        ans.addAll(q);
        break;
      }
      int size=q.size();
      for (int i=0;i<size;i++){
        int leaf=q.poll();
        remain--;
        degree[leaf]--;
        for (int adj:map[leaf]){
          if (degree[adj]==0)
            continue;
          degree[adj]--;
          if (degree[adj]==1)
            q.offer(adj);
        }
      }
    }
    return ans;
  }

  public boolean canPartitionKSubsets(int[] nums, int k) {
    int sum=0;
    for (int n:nums)
      sum+=n;
    if (sum%k!=0)
      return false;
    boolean[] visited=new boolean[nums.length];
    Arrays.sort(nums);
    return CPKSHelper(nums,sum/k,0,k,nums.length-1,visited);
  }

  private boolean CPKSHelper(int[] nums,int target,int curSum,int groups,int start,boolean[] visited){
    if (groups==0)
      return true;
    if (curSum==target)
      return CPKSHelper(nums,target,0,groups-1,nums.length-1,visited);
    for (int i=start;i>=0;i--)
      if (visited[i])
        continue;
      else if (curSum+nums[i]<=target){
        visited[i]=true;
        if (CPKSHelper(nums,target,curSum+nums[i],groups,i-1,visited))
          return true;
        visited[i]=false;
      }
    return false;
  }

  public int leastInterval1(char[] tasks, int n) {
    int[] count = new int[26];
    for (char t : tasks)
      count[t - 'A']++;
    PriorityQueue<Integer> output = new PriorityQueue<>((a, b) -> count[b] - count[a]);
    Queue<Integer> cache = new LinkedList<>();
    int ans = 0, remain = tasks.length;
    for (int i=0;i<26;i++)
      if (count[i]>0)
        output.offer(i);
    while (remain > 0) {
      ans++;
      int nextOut;
      if (output.isEmpty())
        cache.offer(-1);
      else{
        nextOut = output.poll();
        count[nextOut]--;
        remain--;
        if (count[nextOut]>0)
          cache.offer(nextOut);
        else
          cache.offer(-1);
      }
      if (cache.size()>n){
        int temp=cache.poll();
        if (temp!=-1)
          output.offer(temp);
      }
    }
    return ans;
  }

  public int leastInterval2(char[] tasks, int n) {
    int[] count=new int[26];
    int max=0,maxCount=0;
    for (char t:tasks){
      int index=t-'A';
      count[index]++;
      if (max<count[index]){
        max=count[index];
        maxCount=1;
      }
      else if (max==count[index])
        maxCount++;
    }
    int slotTimes=max-1;
    int slotLength=n+1-maxCount;
    int slotNums=slotLength*slotTimes;
    int remain=tasks.length-max*maxCount;
    int idles=Math.max(0,slotNums-remain);
    return tasks.length+idles;
  }

  public int leastInterval(char[] tasks, int n) {
    int[] c=new int[26];
    for (char t:tasks)
      c[t-'A']++;
    Arrays.sort(c);
    int i=24;
    while (i>=0 && c[i]==c[25])
      i--;
    return Math.max(tasks.length,(c[25]-1)*(n+1)+25-i);
  }

  public int bulbSwitch(int n) {
    if (n<=0)
      return 0;
    if (n==1)
      return 1;
    int ans=0,i=1;
    while(i*i<=n){
      i++;
      ans++;
    }
    return ans;
  }

  class NumArray {
    int n;
    int[] nums,BIT;
    public NumArray(int[] nums) {
      n=nums.length;
      this.nums=new int[n+1];
      BIT=new int[n+1];
      for (int i=0;i<n;i++)
        update(i,nums[i]);
    }

    public void update(int i, int val) {
      int dif=val-nums[i+1];
      nums[i+1]=val;
      for (int j=i+1;j<=n;j+=lowbit(j))
        BIT[j]+=dif;
    }

    private int getSum(int loc){
      int sum=0;
      for (int i=loc+1;i>0;i-=lowbit(i))
        sum+=BIT[i];
      return sum;
    }

    public int sumRange(int i, int j) {
      return getSum(j)-getSum(i-1);
    }

    //distance to its parents,or the range it contains
    private int lowbit(int x){
      return x& (-x);
    }
  }

  class MyCircularQueue {
    int[] data;
    int front,rear,size,len;

    /** Initialize your data structure here. Set the size of the queue to be k. */
    public MyCircularQueue(int k) {
      data=new int[k];
      front=rear=k-1;
      size=0;
      len=k;
    }

    /** Insert an element into the circular queue. Return true if the operation is successful. */
    public boolean enQueue(int value) {
      if (size==0){
        front=rear=(front+1)%len;
        data[front]=value;
        size++;
        return true;
      }
      else if (size==len)
        return false;
      else{
        size++;
        rear=(rear+1)%len;
        data[rear]=value;
        return true;
      }
    }

    /** Delete an element from the circular queue. Return true if the operation is successful. */
    public boolean deQueue() {
      if (size==0)
        return false;
      else if (size==1){
        size--;
        return true;
      }
      else{
        size--;
        front=(front+1)%len;
        return true;
      }
    }

    /** Get the front item from the queue. */
    public int Front() {
      return size==0?-1:data[front];
    }

    /** Get the last item from the queue. */
    public int Rear() {
      return size==0?-1:data[rear];
    }

    /** Checks whether the circular queue is empty or not. */
    public boolean isEmpty() {
      return size==0;
    }

    /** Checks whether the circular queue is full or not. */
    public boolean isFull() {
      return size==len;
    }
  }

  class ExamRoom {
    PriorityQueue<Interval> pq;
    int N;

    class Interval{
      int start;
      int end;
      int dist;
      public Interval(int x,int y){
        start=x;
        end=y;
        if (x==-1)
          dist=y;
        else if (y==N)
          dist=y-x-1;
        else
          dist=(y-x)>>1;
      }
    }

    public ExamRoom(int N) {
      pq=new PriorityQueue<>((a,b)->a.dist==b.dist?a.start-b.start:b.dist-a.dist);
      this.N=N;
      pq.offer(new Interval(-1,N));
    }

    public int seat() {
      Interval it=pq.poll();
      int seat=it.start==-1?0:it.end==N?N-1:(it.end+it.start)>>1;
      pq.offer(new Interval(it.start,seat));
      pq.offer(new Interval(seat,it.end));
      return seat;
    }

    public void leave(int p) {
      Interval front=null,tail=null;
      for (Interval it:new ArrayList<>(pq)){
        if (it.end==p)
          front=it;
        else if (it.start==p)
          tail=it;
        if (front!=null && tail!=null)
          break;
      }
      Interval cur=new Interval(front.start,tail.end);
      pq.remove(front);
      pq.remove(tail);
      pq.offer(cur);
    }
  }

  public int[] sumEvenAfterQueries(int[] A, int[][] queries) {
    int[] ans=new int[queries.length];
    int evenSum=0,oldVal,newVal;
    for (int a:A)
      if ((a&1)==0)
        evenSum+=a;
    for (int i=0;i<queries.length;i++){
      oldVal=A[queries[i][1]];
      newVal=oldVal+queries[i][0];
      if ((oldVal&1)==0)
        evenSum-=oldVal;
      if ((newVal&1)==0)
        evenSum+=newVal;
      A[queries[i][1]]=newVal;
      ans[i]=evenSum;
    }
    return ans;
  }

  public int numRookCaptures(char[][] board) {
    int Rr=-1,Rc=-1,R=board.length,C=board[0].length;
    findRookLoc:for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (board[r][c]=='R'){
          Rr=r;
          Rc=c;
          break findRookLoc;
        }
    if (Rr==-1)
      return 0;
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    int ans=0;
    for (int[] d:dirs){
      int tempR=Rr,tempC=Rc;
      while (tempR>=0 && tempR<R &&tempC>=0&& tempC<C){
        if (board[tempR][tempC]=='p'){
          ans++;
          break;
        }
        else if (board[tempR][tempC]=='B')
          break;
        tempR+=d[0];
        tempC+=d[1];
      }
    }
    return ans;
  }

  public int findLength(int[] A, int[] B) {
    int Alen=A.length,Blen=B.length,maxLen=0;
    int[][] dp=new int[Alen][Blen];
    for (int r=0;r<Alen;r++)
      for (int c=0;c<Blen;c++)
        if (A[r] ==B[c]){
          if (r==0||c==0)
            dp[r][c]=1;
          else
            dp[r][c]=dp[r-1][c-1]+1;
          maxLen=Math.max(maxLen,dp[r][c]);
        }
    return maxLen;
  }

  public int subarraysDivByK(int[] A, int K) {
    int[] map=new int[K];
    int ans=0,prefix=0;
    map[0]=1;
    for (int a:A){
      prefix=(prefix+a%K+K)%K;
      ans+=map[prefix];
      map[prefix]++;
    }
    return ans;
  }

  public int maxTurbulenceSize(int[] A) {
    if (A.length == 1)
      return 1;
    int ans=0,inc=1,dec=1;
    for (int i=0;i<A.length-1;i++){
      if (A[i+1]>A[i]){
        inc=dec+1;
        dec=1;
      }
      else if (A[i+1]<A[i]){
        dec=inc+1;
        inc=1;
      }
      else
        inc=dec=1;
      ans=Math.max(ans,Math.max(inc,dec));
    }
    return ans;
  }

  class SolBase{
    public int rand7(){return 0;}
  }

  class Solution1 extends SolBase {
    public int rand10() {
      while(true){
        int index=7*(rand7()-1)+rand7()-1;
        if (index<=40)
          return index%10+1;
      }
    }
  }

  public double minAreaFreeRect(int[][] points) {
    int N=points.length;
    double ans=Double.MAX_VALUE;
    Map<String,List<int[]>> edges=new HashMap<>();
    for (int i=0;i<N-1;i++)
      for (int j=i+1;j<N;j++){
        int Xdif=Math.abs(points[i][0]-points[j][0]),Ydif=Math.abs(points[i][1]-points[j][1]);
        int dist=Xdif*Xdif+Ydif*Ydif;
        int Xcent=points[i][0]+points[j][0],Ycent=points[i][1]+points[j][1];
        String key=dist+" "+Xcent+" "+Ycent;
        edges.putIfAbsent(key,new ArrayList<>());
        edges.get(key).add(new int[]{i,j});
      }

    for (List<int[]> e:edges.values())
      if (e.size()<2)
        continue;
      else{
        int size=e.size();
        for (int i=0;i<size-1;i++)
          for (int j=i+1;j<size;j++){
            int[] n11=points[e.get(i)[0]],n12=points[e.get(i)[1]],n21=points[e.get(j)[0]];
            double dist11=Math.sqrt(Math.pow(n11[0]-n21[0],2)+Math.pow(n11[1]-n21[1],2));
            double dist12=Math.sqrt(Math.pow(n12[0]-n21[0],2)+Math.pow(n12[1]-n21[1],2));
            double area=dist11*dist12;
            ans=Math.min(ans,area);
          }
      }
    return ans==Double.MAX_VALUE?0:ans;
  }

  public int getMoneyAmount(int n) {
    if (n==1)
      return 0;
    int[][] dp=new int[n+1][n+1];
    return GMAhelper(dp,1,n);
  }

  private int GMAhelper(int[][] dp,int start,int end){
    if (start>=end)
      return 0;
    if (dp[start][end]!=0)
      return dp[start][end];
    int res=Integer.MAX_VALUE;
    for (int i=start;i<=end;i++)
      res=Math.min(res,i+Math.max(GMAhelper(dp,start,i-1),GMAhelper(dp,i+1,end)));
    dp[start][end]=res;
    return res;
  }

  public boolean canThreePartsEqualSum(int[] A) {
    if (A.length<3)
      return false;
    int sum=0;
    for (int a:A)
      sum+=a;
    if (sum%3!=0)
      return false;
    int part=sum/3,count=0,cur=0;
    for (int a:A){
      cur+=a;
      if (cur==part){
        count++;
        cur=0;
      }
    }
    return count==3;
  }

  public int findLHS(int[] nums) {
    Map<Integer,Integer> mp=new HashMap<>();
    for (int n:nums)
      mp.put(n,mp.getOrDefault(n,0)+1);
    int count=0;
    for (Map.Entry<Integer,Integer> et:mp.entrySet())
      count=Math.max(count,et.getValue()+mp.getOrDefault(et.getKey()-1,Integer.MIN_VALUE));
    return count;
  }

  public ListNode swapPairs(ListNode head) {
    if (head==null||head.next==null)
      return head;
    ListNode ans=head.next,left=head,right=head.next,last=null;
    while(true){
      ListNode temp=right.next;
      if (last!=null)
        last.next=right;
      right.next=left;
      left.next=temp;
      last=left;
      if ((left=left.next)==null ||(right=left.next)==null)
        break;
    }
    last.next=left;
    return ans;
  }

  public int maxSumTwoNoOverlap1(int[] A, int L, int M) {
    int N=A.length,ans=0;
    int[] ps=new int[N+1];
    for (int i=1;i<=N;i++)
      ps[i]=A[i-1]+ps[i-1];
    for (int i=1;i<=N-L+1;i++)
      for (int j=1;j<=N-M+1;j++){
        int ls=i,le=i+L-1,ms=j,me=j+M-1;
        if (ls>me ||ms>le)
          ans=Math.max(ans,ps[le]-ps[ls-1]+ps[me]-ps[ms-1]);
      }
    return ans;
  }

  public int maxSumTwoNoOverlap(int[] A, int L, int M) {
    for (int i = 1; i < A.length; ++i)
      A[i] += A[i - 1];
    int res = A[L + M - 1], Lmax = A[L - 1], Mmax = A[M - 1];
    for (int i = L + M; i < A.length; ++i) {
      Lmax = Math.max(Lmax, A[i - M] - A[i - L - M]);
      Mmax = Math.max(Mmax, A[i - L] - A[i - L - M]);
      res = Math.max(res, Math.max(Lmax + A[i] - A[i - M], Mmax + A[i] - A[i - L]));
    }
    return res;
  }

  public boolean isValidSudoku(char[][] board) {
    Set<String> st=new HashSet<>();
    for (int i=0;i<9;i++)
      for (int j=0;j<9;j++){
        if (board[i][j]=='.')
          continue;
        char num=board[i][j];
        if (!st.add(num+"b"+i)||!st.add(num+"r"+j) ||!st.add(num+"c"+i/3+' '+j/3))
          return false;
      }
    return true;
  }

  public boolean divisorGame(int N) {
    return N%2==0;
  }

  public int bitwiseComplement(int N) {
    if (N==0)
      return 1;
    return (int)Math.pow(2,Math.floor(Math.log(N)/Math.log(2)+1))-1-N;
  }

  public int minPathSum1(int[][] grid) {
    if (grid==null ||grid.length==0 ||grid[0].length==0)
      return 0;
    int R=grid.length,C=grid[0].length;
    if (R==1 && C==1)
      return grid[0][0];
    boolean[][] visited=new boolean[R][C];
    int[][] distTo=new int[R][C];
    PriorityQueue<int[]> edges=new PriorityQueue<>((a,b)->distTo[a[0]][a[1]]-distTo[b[0]][b[1]]);
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++)
        distTo[i][j]=Integer.MAX_VALUE;
    distTo[0][0]=grid[0][0];
    edges.offer(new int[]{0,0});
    while(!edges.isEmpty() && !visited[R-1][C-1]){
      int[] next=edges.poll();
      if (visited[next[0]][next[1]])
        continue;
      MPSrelax(grid,distTo,visited,next,edges);
    }
    return distTo[R-1][C-1];
  }

  private void MPSrelax(int[][] grid,int[][] distTo,boolean[][] visited,int[] from,PriorityQueue<int[]> edges){
    visited[from[0]][from[1]]=true;
    int R=grid.length,C=grid[0].length;
    if (from[0]<R-1 && distTo[from[0]+1][from[1]]> distTo[from[0]][from[1]]+grid[from[0]+1][from[1]]){
      distTo[from[0]+1][from[1]]=distTo[from[0]][from[1]]+grid[from[0]+1][from[1]];
      edges.offer(new int[]{from[0]+1,from[1]});
    }

    if (from[1]<C-1 && distTo[from[0]][from[1]+1]> distTo[from[0]][from[1]]+grid[from[0]][from[1]+1]){
      distTo[from[0]][from[1]+1]=distTo[from[0]][from[1]]+grid[from[0]][from[1]+1];
      edges.offer(new int[]{from[0],from[1]+1});
    }
  }

  public int minPathSum(int[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0)
      return 0;
    int R = grid.length, C = grid[0].length;
    if (R == 1 && C == 1)
      return grid[0][0];
    int[][] res=new int[R][C];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (r>0 && c>0)
          res[r][c]=Math.min(res[r-1][c],res[r][c-1])+grid[r][c];
        else if (r==0 && c==0)
          res[r][c]=grid[r][c];
        else if (r==0)
          res[r][c]=res[r][c-1]+grid[r][c];
        else
          res[r][c]=res[r-1][c]+grid[r][c];
    return res[R-1][C-1];
  }

  public int[][] generateMatrix(int n) {
    if (n==1)
      return new int[][]{{1}};
    int[][] ans=new int[n][n];
    int[][] dirs=new int[][]{{0,1},{1,0},{0,-1},{-1,0}};
    int[] loc=new int[]{0,-1};
    int val=1,len=n-1,target=n*n;
    fill:while (val<=target){
      for (int d=0;d<4;d++){
        if (d==0){
          loc[0]+=dirs[d][0];
          loc[1]+=dirs[d][1];
        }
        if (len==0){
          ans[loc[0]][loc[1]]=val;
          break fill;
        }

        for (int i=0;i<len;i++){
          ans[loc[0]][loc[1]]=val++;
          if (d<3||i<len-1){
            loc[0]+=dirs[d][0];
            loc[1]+=dirs[d][1];
          }
        }
      }
      n-=2;
      len=n-1;
    }
    return ans;
  }

  class Node1 {
    public int val;
    public Node1 prev;
    public Node1 next;
    public Node1 child;

    public Node1() {}

    public Node1(int _val,Node1 _prev,Node1 _next,Node1 _child) {
      val = _val;
      prev = _prev;
      next = _next;
      child = _child;
    }
  };

  public Node1 flatten(Node1 head) {
    if (head==null)
      return head;
    Node1 ans=head;
    FLhelper(null,head);
    return ans;
  }

  private Node1 FLhelper(Node1 last,Node1 cur){
    cur.prev=last;
    if (last!=null)
      last.next=cur;
    Node1 newLast=cur,nextTemp=cur.next;
    if (cur.child!=null){
      newLast=FLhelper(newLast,cur.child);
      cur.child=null;
    }
    if (nextTemp!=null)
      newLast=FLhelper(newLast,nextTemp);
    return newLast;
  }

  public String baseNeg2(int N) {
    StringBuilder sb=new StringBuilder();
    while (N!=0) {
      sb.insert(0,N&1);
      N = -(N >> 1);
    }
    String ans;
    return (ans=sb.toString() ).equals("") ? "0" : ans;
  }

  class FreqStack1 {
    List<Stack<Integer>> data;
    Map<Integer,Integer> freqRecord;
    public FreqStack1() {
      data=new ArrayList<>();
      freqRecord=new HashMap<>();
    }

    public void push(int x) {
      int newFreq=freqRecord.getOrDefault(x,0)+1;
      freqRecord.put(x,newFreq);
      if (data.size()<newFreq)
        data.add(new Stack<>());
      data.get(newFreq-1).push(x);
    }

    public int pop() {
      Stack<Integer> mostFreq=data.get(data.size()-1);
      int ans=mostFreq.pop();
      freqRecord.put(ans,freqRecord.get(ans)-1);
      if (mostFreq.isEmpty())
        data.remove(data.size()-1);
      return ans;
    }
  }

  public List<Integer> findDisappearedNumbers1(int[] nums) {
    for (int i=0;i<nums.length;i++){
      int index=Math.abs(nums[i])-1;
      if (nums[index]>0)
        nums[index]*=-1;
    }
    List<Integer> ans=new ArrayList<>();
    for (int i=0;i<nums.length;i++)
      if (nums[i]>0)
        ans.add(i+1);
    return ans;
  }

  public List<Integer> topKFrequent1(int[] nums, int k) {
    if (k<nums.length)
      throw new IllegalArgumentException();
    int N=nums.length;
    Stack<Integer>[] freq=new Stack[N+1];
    for (int i=1;i<=N;i++)
      freq[i]=new Stack<>();
    Map<Integer,Integer> freqRecord=new HashMap<>();
    for (int n:nums)
      freqRecord.put(n,freqRecord.getOrDefault(n,0)+1);
    for (Map.Entry<Integer,Integer> et:freqRecord.entrySet())
      freq[et.getValue()].push(et.getKey());
    List<Integer> ans=new ArrayList<>(k);
    find:for (int i=N;i>0;i--)
      while (!freq[i].isEmpty()){
        ans.add(freq[i].pop());
        k--;
        if (k==0)
          break find;
      }
    return ans;
  }

  public int minimumDeleteSum1(String s1, String s2) {
    if (s1.isEmpty() &&s2.isEmpty() )
      return 0;
    else if (s1.isEmpty() ||s2.isEmpty()){
      String temp=s1.isEmpty()?s2:s1;
      int ans=0;
      for (char c:temp.toCharArray())
        ans+=c;
      return ans;
    }
    int sum=0;
    char[] cs1=s1.toCharArray(),cs2=s2.toCharArray();
    for (char c:cs1)
      sum+=c;
    for (char c:cs2)
      sum+=c;
    int lcs=LCS(cs1,cs2);
    return sum-lcs*2;
  }

  private int LCS(char[] s1,char[] s2){
    int R=s1.length,C=s2.length;
    int[][] dp=new int[R+1][C+1];
    for (int r=1;r<=R;r++)
      for (int c=1;c<=C;c++)
        dp[r][c]=s1[r-1]==s2[c-1]?dp[r-1][c-1]+s1[r-1]:Math.max(dp[r-1][c],dp[r][c-1]);
    return dp[R][C];
  }

  public int lengthOfLIS1(int[] nums) {
    if (nums==null||nums.length==0)
      return 0;
    if (nums.length==1)
      return 1;
    int N=nums.length,index=0;
    int[] cache=new int[N];
    cache[index++]=nums[0];
    for (int i=1;i<N;i++)
      if (nums[i]>cache[index-1])
        cache[index++]=nums[i];
      else{
        int s=0,e=index-1,mid=0;
        while (s<=e){
          mid=(s+e)>>1;
          if (cache[mid]<nums[i])
            s=mid+1;
          else
            e=mid-1;
        }
        cache[s]=nums[i];
      }
    return index;
  }

  public int mirrorReflection1(int p, int q) {
    int gcd=MRGCD(p,q);
    p=p/gcd &1;
    q=q/gcd&1;
    return p==0?2:q==0?0:1;
  }

  private int MRGCD(int p,int q){
    if (q==0)
      return p;
    else
      return MRGCD(q,p%q);
  }

  public int getSum2(int a, int b) {
    return b==0?a:getSum2(a^b,(a&b)<<1);
  }

  public List<List<Integer>> subsets2(int[] nums) {
    List<List<Integer>> ans=new ArrayList<>();
    if (nums==null||nums.length==0)
      return ans;
    SShelper(nums,0,new Stack<>(),ans);
    return ans;
  }

  private void SShelper(int[] nums,int start,Stack<Integer> path,List<List<Integer>> res){
    res.add(new ArrayList<>(path));
    for (int i=start;i<nums.length;i++){
      path.push(nums[i]);
      SShelper(nums,i+1,path,res);
      path.pop();
    }
  }

  public int longestOnes1(int[] A, int K) {
    if (A==null||A.length==0)
      return 0;
    if (A.length<=K)
      return K;
    int left=0,right=0,max=0;
    for (;right<A.length;right++){
      if (A[right]==0)
        if (K>0)
          K--;
        else
          while (left<=right)
            if (A[left++]==0)
              break;
      max=Math.max(max,right-left+1);
    }
    return max;
  }

  public int totalNQueens1(int n) {
    if (n==0||n==1)
      return n;
    int[] res=new int[1];
    boolean[] cols=new boolean[n],slash=new boolean[2*n],backSlash=new boolean[2*n];
    totalNQueens1(0,n,cols,slash,backSlash,res);
    return res[0];
  }

  private void totalNQueens1(int row,int n,boolean[] cols,boolean[] slash,boolean[] backSlash,int[] res){
    if (row==n){
      res[0]++;
      return;
    }
    for (int c=0;c<n;c++){
      int sl=c-row+n,bsl=row+c;
      if (cols[c] || slash[sl] ||backSlash[bsl])
        continue;
      cols[c]=slash[sl]=backSlash[bsl]=true;
      totalNQueens1(row+1,n,cols,slash,backSlash,res);
      cols[c]=slash[sl]=backSlash[bsl]=false;
    }
  }

  public int slidingPuzzle1(int[][] board) {
    Set<String> states=new HashSet<>();
    Queue<String> q=new LinkedList<>();
    StringBuilder sb=new StringBuilder();
    for (int i=0;i<board.length;i++)
      for (int j=0;j<board[0].length;j++)
        sb.append(board[i][j]);
    String val=sb.toString();
    if (val.equals("123450"))
      return 0;
    q.offer(val);
    states.add(val);
    int ans=0;
    int[][] swap=new int[][]{{1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4}};
    while (!q.isEmpty()){
      int size=q.size();
      ans++;
      for (int i=0;i<size;i++){
        String temp=q.poll();
        int index=-1;
        for (int j=0;j<6;j++)
          if (temp.charAt(j)=='0'){
            index=j;
            break;
          }
        for (int s:swap[index]){
          String adj=StringSwap(temp,index,s);
          if (adj.equals("123450"))
            return ans;
          if (!states.contains(adj)){
            states.add(adj);
            q.offer(adj);
          }
        }
      }
    }
    return -1;
  }

  private String StringSwap(String temp,int i,int j){
    char[] cs=temp.toCharArray();
    char t=cs[i];
    cs[i]=cs[j];
    cs[j]=t;
    return new String(cs);
  }

  public int oddEvenJumps1(int[] A) {
    if (A.length==1)
      return 1;
    int N=A.length,res=1;
    boolean[] lower=new boolean[N],higher=new boolean[N];
    lower[N-1]=higher[N-1]=true;
    TreeMap<Integer,Integer> after=new TreeMap<>();
    after.put(A[N-1],N-1);
    for (int i=N-2;i>=0;i--){
      Map.Entry<Integer,Integer> hi=after.ceilingEntry(A[i]),lo=after.floorEntry(A[i]);
      if (hi!=null)
        higher[i]=lower[hi.getValue()];
      if (lo!=null)
        lower[i]=higher[lo.getValue()];
      if (higher[i])
        res++;
      after.put(A[i],i);
    }
    return res;
  }

  public int[] plusOne(int[] digits) {
    int carry=1,N=digits.length;
    for (int i=N-1;i>=0 && carry!=0;i--){
      int sum=digits[i]+carry;
      carry=sum/10;
      digits[i]=sum%10;
    }
    if (carry==0)
      return digits;
    int[] ans=new int[N+1];
    ans[0]=carry;
    for (int i=0;i<N;i++)
      ans[i+1]=digits[i];
    return ans;
  }

  public int lenLongestFibSubseq(int[] A) {
    Map<Integer,Integer> valToIndex=new HashMap<>();
    int N=A.length,max=0;
    for (int i=0;i<N;i++)
      valToIndex.put(A[i],i);
    int[][] dp=new int[N][N];
    for (int r=2;r<N;r++)
      for (int c=1;c<r;c++){
        int dif=A[r]-A[c],difIndex=valToIndex.getOrDefault(dif,-1);
        if (difIndex!=-1 && difIndex<c){
          dp[r][c]=dp[c][difIndex]+1;
          max=Math.max(dp[r][c],max);
        }
      }
    return max==0?0:max+2;
  }

  public int threeSumClosest1(int[] nums, int target) {
    int N=nums.length,ans=Integer.MAX_VALUE,res=0;
    Map<Integer,int[]> twoSum=new HashMap<>();
    for (int i=0;i<N;i++)
      for (int j=i+1;j<N;j++)
        twoSum.put(nums[i]+nums[j],new int[]{i,j});
    for (int i=0;i<N;i++)
      for (Map.Entry<Integer,int[]> entry:twoSum.entrySet()){
        int[] id=entry.getValue();
        int dif= Math.abs(nums[i]+entry.getKey()-target);
        if (i!=id[0] && i!=id[1] && dif<ans){
          ans=dif;
          res=nums[i]+entry.getKey();
        }
      }
    return res;
  }

  public int threeSumClosest(int[] nums, int target) {
    int res=nums[0]+nums[1]+nums[2],start,end,N=nums.length;
    Arrays.sort(nums);
    for (int i=0;i<N-2;i++){
      start=i+1;
      end=N-1;
      while (start<end){
        int sum=nums[i]+nums[start]+nums[end];
        if (sum>target)
          end--;
        else if (sum<target)
          start++;
        else
          return sum;
        res=Math.abs(sum-target)<Math.abs(res-target)?sum:res;
      }
    }
    return res;
  }

  public int[][] intervalIntersection(int[][] A, int[][] B) {
    int Alen=A.length,Blen=B.length;
    if (Alen==0||Blen==0)
      return new int[0][2];
    int AIndex=0,BIndex=0;
    List<int[]> res=new ArrayList<>();
    while (AIndex < Alen &&BIndex<Blen){
      int[] a=A[AIndex],b=B[BIndex];
      if (!(a[0]>b[1] || a[1]<b[0]))
        res.add(new int[]{Math.max(a[0],b[0]),Math.min(a[1],b[1])});
      if (a[1]>b[1])
        BIndex++;
      else if (a[1]<b[1])
        AIndex++;
      else{
        AIndex++;
        BIndex++;
      }
    }
    return res.toArray(new int[0][0]);
  }

  public int maxArea(int[] height) {
    int max=Integer.MIN_VALUE,N=height.length,start=0,end=N-1;
    while (start<end){
      max=Math.max(max,Math.min(height[start],height[end])*(end-start));
      if (height[start]<height[end])
        start++;
      else
        end--;
    }
    return max;
  }

  public String reverseVowels(String s) {
    if (s==null||s.length()==0)
      return s;
    char[] cs=s.toCharArray();
    int N=cs.length,start=-1,end=N;
    while (start<end){
      while (++start<N && !isVowel(cs[start]));
      while (--end>=0 && !isVowel(cs[end]));
      if (start>=end)
        break;
      exchangeChar(cs,start,end);
    }
    return new String(cs);
  }

  private boolean isVowel(char c){
    if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||c=='A'||c=='E'||c=='I'||c=='O'||c=='U')
      return true;
    else
      return false;
  }

  public int characterReplacement(String s, int k) {
    if (s==null||s.length()==0)
      return 0;
    char[] cs=s.toCharArray();
    int N=cs.length,max=0,left,right,most=0;
    int[] count=new int[26];
    for (left=right=0;right<N;right++){
     most=Math.max(most,++count[cs[right]-'A']);
     int len=right-left+1;
     if (most+k<len){
       count[cs[left]-'A']--;
       left++;
     }
     else
      max=Math.max(max,len);
    }
    return max;
  }

  public int numRescueBoats(int[] people, int limit) {
    if (people.length==1)
      return 1;
    if (limit==1)
      return people.length;
    int start=0,end=limit,ans=0;
    int[] weights=new int[limit+1];
    for (int p:people)
      weights[p]++;
    while (start<=end){
      if (weights[start]==0)
        start++;
      else if (weights[end]==0)
        end--;
      else if (start+end>limit)
        ans+=weights[end--];
      else if (weights[end]<weights[start]){
        ans+=weights[end];
        weights[start]-=weights[end--];
      }
      else if (weights[end]>weights[start]){
        ans+=weights[start];
        weights[end]-=weights[start++];
      }
      else{
        if (start==end){
          ans+=(weights[start]>>1);
          if ((weights[start]&1)==1)
            ans++;
        }
        else
          ans+=weights[end];
        end--;
        start++;
      }
    }
    return ans;
  }

  public String findReplaceString(String S, int[] indexes, String[] sources, String[] targets) {
    Map<Integer,Integer> res=new HashMap<>();
    for (int i=0;i<indexes.length;i++)
      if (S.startsWith(sources[i],indexes[i]))
        res.put(indexes[i],i);
    StringBuilder sb=new StringBuilder();
    for (int i=0;i<S.length();)
      if (res.containsKey(i)){
        int index=res.get(i);
        sb.append(targets[index]);
        i+=sources[index].length();
      }
      else
        sb.append(S.charAt(i++));
    return sb.toString();
  }

  public List<String> wordSubsets(String[] A, String[] B) {
    List<String> res=new ArrayList<>();
    if (A==null||A.length==0)
      return res;
    int[] BAccum=new int[26],Btemp,Atemp;
    for (String b:B){
      Btemp=new int[26];
      for (char s:b.toCharArray())
        Btemp[s-'a']++;
      for (int i=0;i<26;i++)
        BAccum[i]=Math.max(BAccum[i],Btemp[i]);
    }

    findA:for (String a:A){
      Atemp=new int[26];
      for (char s:a.toCharArray())
        Atemp[s-'a']++;
      for (int i=0;i<26;i++)
        if (Atemp[i] <BAccum[i])
          continue findA;
      res.add(a);
    }
    return res;
  }

  public String addStrings(String num1, String num2) {
    StringBuilder sb=new StringBuilder();
    char[] cs1=num1.toCharArray(),cs2=num2.toCharArray();
    int carry=0,index=0,len1=cs1.length,len2=cs2.length;
    while (index<len1 || index<len2 ||carry!=0){
      int c1=index<len1?cs1[len1-1-index]-48:0;
      int c2=index<len2?cs2[len2-1-index]-48:0;
      int sum=c1+c2+carry,val=sum%10;
      carry=sum/10;
      sb.append(val);
      index++;
    }
    return sb.reverse().toString();
  }

  public int minDistance(String word1, String word2) {
    if (word1==null||word1.length()==0)
      return word2==null?0:word2.length();
    if (word2==null||word2.length()==0)
      return word1.length();
    int LCS= findLCS(word1.toCharArray(),word2.toCharArray());
    return word1.length()+word2.length()-(LCS<<1);
  }

  private int findLCS(char[] cs1,char[] cs2){
    int N1=cs1.length,N2=cs2.length,max=0;
    int[][] dp=new int[N1+1][N2+1];
    for (int i=1;i<=N1;i++)
      for (int j=1;j<=N2;j++){
        if (cs1[i-1]==cs2[j-1])
          dp[i][j]=dp[i-1][j-1]+1;
        else
          dp[i][j]=Math.max(dp[i][j-1],dp[i-1][j]);
        max=Math.max(max,dp[i][j]);
      }
    return max;
  }

  public List<String> ambiguousCoordinates(String S) {
    List<String> ans=new ArrayList<>();
    for (int i=2;i<S.length()-1;i++){
      List<String> left=AChelper(S.substring(1,i)),right=AChelper(S.substring(i,S.length()-1));
      for (String l:left)
        for (String r:right)
          ans.add("("+l+", "+r+")");
    }
    return ans;
  }

  private List<String> AChelper(String S){
    int n=S.length();
    List<String> ans=new ArrayList<>();
    if (n==0 ||(n>1 && S.charAt(0)=='0' && S.charAt(n-1)=='0'))
      return ans;
    if (n>1 && S.charAt(0)=='0'){
      ans.add("0."+S.substring(1));
      return ans;
    }
    ans.add(S);
    if (n==1 || S.charAt(n-1)=='0')
      return ans;
    for (int i=1;i<n;i++)
      ans.add(S.substring(0,i)+"."+S.substring(i));
    return ans;
  }

  class TopVotedCandidate {
    private int[] leading,persons,times;
    public TopVotedCandidate(int[] persons, int[] times) {
      int n=persons.length,maxP=0,maxCount=0;
      this.persons=persons;
      this.times=times;
      leading=new int[n];
      int[] count=new int[n];
      for (int i=0;i<n;i++){
        count[persons[i]]++;
        if (count[persons[i]]>=maxCount){
          maxCount=count[persons[i]];
          maxP=persons[i];
        }
        leading[i]=maxP;
      }
    }

    public int q(int t) {
      int index=findTime(t);
      return index==-1?-1:leading[index];
    }

    private int findTime(int t){
      int start=0,end=times.length-1;
      while (start<=end){
        int mid = (start+end)>>1;
        if (times[mid]<=t && (mid==times.length-1 ||times[mid+1]>t))
          return mid;
        else if (times[mid]>t)
          end=mid-1;
        else if (times[mid]<t)
          start=mid+1;
      }
      return -1;
    }
  }

  public int minEatingSpeed(int[] piles, int H) {
    int s=1,e=0,mid;
    for (int i:piles)
      e=Math.max(e,i);
    while (s<e){
      mid=(s+e)>>1;
      int hours=0;
      for (int p:piles){
        hours+=p/mid;
        if (p%mid!=0)
          hours++;
      }
      if (hours>H)
        s=mid+1;
      else if (hours<=H)
        e=mid;
    }
    return e;
  }

  public char nextGreatestLetter(char[] letters, char target) {
    int N=letters.length;
    if (target>=letters[N-1])
      return letters[0];
    int s=0,e=N-1;
    while (s<=e){
      int mid=(s+e)>>1;
      if (letters[mid]>target)
        e=mid-1;
      else
        s=mid+1;
    }
    return letters[s];
  }

  public int findMin(int[] nums) {
    if (nums==null || nums.length==0)
      throw new IllegalArgumentException();
    if (nums.length==1)
      return nums[0];
    int s=0,e=nums.length-1;
    while (s<e){
      int mid=(s+e)>>1;
      if (mid>0 && nums[mid]<nums[mid-1])
        return nums[mid];
      if (nums[mid]>=nums[s] && nums[mid]>nums[e])
        s=mid+1;
      else
        e=mid-1;
    }
    return nums[s];
  }

  class Solution528 {
    private Random r;
    private int[] range;
    private int sum;

    public Solution528(int[] w) {
      r=new Random();
      int n=w.length;
      range=new int[n];
      sum=0;
      for (int i=0;i<n;i++){
        sum+=w[i];
        range[i]=sum;
      }
    }

    public int pickIndex() {
      if (range.length==1)
        return 0;
      int stick=r.nextInt(sum),s=0,e=range.length-1;
      while (s<e){
        int mid=(s+e)>>1;
        if (range[mid]<=stick && range[mid+1]>stick)
          return mid+1;
        else if (mid==0 && range[mid]>stick)
          return mid;
        else if (range[mid]>stick)
          e=mid;
        else
          s=mid;
      }
      return e;
    }
  }

  public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix==null||matrix.length==0 ||matrix[0].length==0)
      return false;
    int R = matrix.length,C=matrix[0].length,row=0,col=C-1;
    while (row<R && col>=0){
      int val=matrix[row][col];
      if (target==val)
        return true;
      else if (target>val)
        row++;
      else
        col--;
    }
    return false;
  }

  public int rob(int[] nums) {
    if (nums==null||nums.length==0)
      return 0;
    if (nums.length==1)
      return nums[0];
    int n=nums.length;
    int[] dp= new int[n];
    dp[0]=nums[0];
    dp[1]=Math.max(nums[0],nums[1]);
    for (int i=2;i<n;i++)
      dp[i]=Math.max(dp[i-2]+nums[i],dp[i-1]);
    return dp[n-1];
  }

  public int longestPalindromeSubseq1(String s) {
    if (s==null||s.length()==0)
      return 0;
    char[] cs=s.toCharArray();
    int n=cs.length;
    int[][] dp=new int[n+1][n+1];
    for (int r=1;r<=n;r++)
      for (int c=1;c<=n;c++){
        char ascend= cs[r-1],descend=cs[n-c];
        if (ascend==descend)
          dp[r][c]=dp[r-1][c-1]+1;
        else
          dp[r][c]=Math.max(dp[r-1][c],dp[r][c-1]);
      }
    return dp[n][n];
  }

  public int longestPalindromeSubseq(String s) {
    if (s==null||s.length()==0)
      return 0;
    char[] cs=s.toCharArray();
    int n=cs.length;
    int[][] dp=new int[n][n];
    for (int i=n-1;i>=0;i--){
      dp[i][i]=1;
      for (int j=i+1;j<n;j++)
        if (cs[i]==cs[j])
          dp[i][j]=dp[i+1][j-1]+2;
        else
          dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);
    }
    return dp[0][n-1];
  }

  public int numTrees1(int n) {
    if (n<1)
      throw new IllegalArgumentException();
    if (n==1)
      return 1;
    Map<String,Integer> subTreesMemo=new HashMap<>();
    return numTreesHelper(1,n,subTreesMemo);
  }

  private int numTreesHelper(int start,int end,Map<String, Integer> treesMemo){
    if (start>=end)
      return 1;
    String index=start+" "+end;
    if (treesMemo.containsKey(index))
      return treesMemo.get(index);
    int ans=0;
    for (int i=start;i<=end;i++)
      ans += numTreesHelper(start,i-1,treesMemo)*numTreesHelper(i+1,end,treesMemo);
    treesMemo.put(index,ans);
    return ans;
  }

  public int numTrees(int n) {
    if (n<1)
      throw new IllegalArgumentException();
    if (n==1)
      return 1;
    int[] dp=new int[n+1];
    dp[0]=dp[1]=1;
    for (int i=2;i<=n;i++)
      for (int j=0;j<i;j++)
        dp[i]+= dp[j]*dp[i-j-1];
    return dp[n];
  }

  public int deleteAndEarn(int[] nums) {
    if (nums==null || nums.length==0)
      return 0;
    int[] count=new int[10001];
    for (int n:nums)
      count[n]++;
    int ans=count[1],last1=0,last2=ans;
    for (int i=2;i<count.length;i++){
      ans=Math.max(count[i]*i+last1,last2);
      last1=last2;
      last2=ans;
    }
    return ans;
  }

  public int longestArithSeqLength(int[] A) {
    if (A==null||A.length==0)
      return 0;
    Map<Integer,Map<Integer,Integer>> count=new HashMap<>();
    int res=2;
    for (int i=0;i<A.length;i++)
      for (int j=i+1;j<A.length;j++){
        int dif=A[j]-A[i];
        if (!count.containsKey(dif))
          count.put(dif,new HashMap<>());
        Map<Integer,Integer> temp=count.get(dif);
        int val=temp.getOrDefault(i,1)+1;
        temp.put(j,val);
        res=Math.max(res,val);
      }
    return res;
  }

  public int numSquares(int n) {
    if (n==1)
      return 1;
    int[] dp=new int[n+1];
    for (int i=1,sq;(sq=i*i)<=n;i++)
      dp[sq]=1;
    for (int i=2;i<=n;i++){
      if (dp[i]==1)
        continue;
      int min=Integer.MAX_VALUE;
      for (int s=1,square;(square=s*s)<=i;s++)
        min=Math.min(min,dp[i-square]+dp[square]);
      dp[i]=min;
    }
    return dp[n];
  }

  public boolean canPartition(int[] nums) {
    if (nums==null||nums.length<2)
      return false;
    int sum=0;
    for (int n:nums)
      sum+=n;
    if ((sum&1)==1)
      return false;
    boolean res=haveSum(nums,sum>>1);
    return res;
  }

  private boolean haveSum(int[] nums,int target){
    int N=nums.length;
    boolean[] dp=new boolean[target+1];
    dp[0]=true;
    if (nums[0]<=target)
      dp[nums[0]]=true;
    for (int i=1;i<N;i++)
      for (int j=target;j>=0;j--){
        if (!dp[j])
          continue;
        int pickedIndex;
        if ((pickedIndex=j+nums[i])<=target)
          dp[pickedIndex]=true;
        if (dp[target])
          return true;
      }
    return dp[target];
  }

  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> ans=new ArrayList<>();
    if (candidates==null||candidates.length==0)
      return ans;
    Arrays.sort(candidates);
    CShelper(candidates,0,target,new Stack<>(),ans);
    return ans;
  }

  private void CShelper(int[] candidates,int start,int remain,Stack<Integer> path,List<List<Integer>> ans){
    if (remain==0){
      ans.add(new ArrayList<>(path));
      return;
    }
    for (int i=start;i<candidates.length;i++){
      if (candidates[i]>remain)
        return;
      if (i>start && candidates[i]==candidates[i-1])
        continue;
      path.push(candidates[i]);
      CShelper(candidates,i+1,remain-candidates[i],path,ans);
      path.pop();
    }
  }

  public List<List<String>> partition1(String s) {
    List<List<String>> ans=new ArrayList<>();
    if (s==null||s.length()==0)
      return ans;
    partitionHelper(s,0,new Stack<>(),ans);
    return ans;
  }

  private void partitionHelper(String s,int start,Stack<String> path,List<List<String>> ans){
    if (!path.isEmpty() && start==s.length()){
      ans.add(new ArrayList<>(path));
      return;
    }
    for (int i=start;i<s.length();i++)
      if (isPal(s,start,i)){
        if (i!=start)
          path.push(s.substring(start,i+1));
        else
          path.push(Character.toString(s.charAt(start)));
        partitionHelper(s,i+1,path,ans);
        path.pop();
      }
  }

  private boolean isPal(String s,int l,int r){
    while (l<r)
      if (s.charAt(l++)!=s.charAt(r--))
        return false;
    return true;
  }

  public List<List<String>> partition(String s) {
    List<List<String>> ans = new ArrayList<>();
    if (s == null || s.length() == 0)
      return ans;
    char[] cs=s.toCharArray();
    int N=cs.length;
    boolean[][] isPal=new boolean[N][N];
    for (int i=0;i<N;i++)
      for (int j=0;j<=i;j++)
        if (cs[i] ==cs[j] && (i-j<=2 || isPal[j+1][i-1]))
          isPal[j][i]= true;
    Phelper(s,0,new Stack<>(),ans,isPal);
    return ans;
  }

  private void Phelper(String s,int start,Stack<String> path,List<List<String>> ans,boolean[][] isPal){
    if (start==s.length()){
      ans.add(new ArrayList<>(path));
      return;
    }
    for (int i=start;i<s.length();i++)
      if (isPal[start][i]){
        path.push(s.substring(start,i+1));
        Phelper(s,i+1,path,ans,isPal);
        path.pop();
      }
  }

  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> ans=new ArrayList<>();
    if (nums==null || nums.length==0)
      return ans;
    permuteHelper(nums,0,ans);
    return ans;
  }

  private void permuteHelper(int[] nums,int start,List<List<Integer>> ans){
    if (start==nums.length){
      List<Integer> ls=new ArrayList<>(nums.length);
      for (int n:nums)
        ls.add(n);
      ans.add(ls);
      return;
    }
    Set<Integer> appeared = new HashSet<>();
    for (int i=start;i<nums.length;i++){
      if (appeared.add(nums[i]))
        if (i==start)
          permuteHelper(nums,start+1,ans);
        else if (nums[i]==nums[start])
          continue;
        else{
          exchange(nums,start,i);
          permuteHelper(nums,start+1,ans);
          exchange(nums,start,i);
        }
    }
  }

  class MinStack1 {
    Stack<Integer> data;
    int min;
    /** initialize your data structure here. */
    public MinStack1() {
      data=new Stack<>();
      min=Integer.MAX_VALUE;
    }

    public void push(int x) {
      if (x<=min){
        data.push(min);
        min=x;
      }
      data.push(x);
    }

    public void pop() {
      if (data.pop()==min)
        min=data.pop();
    }

    public int top() {
      return data.peek();
    }

    public int getMin() {
      return min;
    }
  }

  class MinStack {
    class Node{
      int val,min;
      Node next;
      public Node(int v,int m){
        val=v;
        min=m;
      }

      public Node(int v, int m, Node _next){
        val=v;
        min=m;
        next=_next;
      }
    }
    /** initialize your data structure here. */
    Node head;
    public MinStack() { }

    public void push(int x) {
      if (head==null)
        head=new Node(x,x);
      else
        head=new Node(x,Math.min(x,head.min),head);
    }

    public void pop() {
      head=head.next;
    }

    public int top() {
      return head.val;
    }

    public int getMin() {
      return head.min;
    }
  }

  public List<String> letterCombinations(String digits) {
    List<String> ans=new ArrayList<>();
    if (digits==null||digits.length()==0)
      return ans;
    String[] digitsToLetters=new String[]{"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    LChelper(digits,0,new StringBuilder(),ans,digitsToLetters);
    return ans;
  }

  private void LChelper(String digits,int start,StringBuilder path,List<String> ans,String[] dTol){
    if (start==digits.length()){
      ans.add(path.toString());
      return;
    }
    int index=digits.charAt(start)-'0';
    for (char c:dTol[index].toCharArray()){
      path.append(c);
      LChelper(digits,start+1,path,ans,dTol);
      path.deleteCharAt(path.length()-1);
    }
  }

  public List<List<String>> solveNQueens(int n) {
    List<List<String>> ans=new ArrayList<>();
    if (n==0)
      return ans;
    if (n==1){
      List<String> t=new ArrayList<>();
      t.add("Q");
      ans.add(t);
      return ans;
    }
    boolean[] cols=new boolean[n],slash=new boolean[n<<1],backSlash=new boolean[n<<1];
    List<String> curMap=new ArrayList<>();
    solveQueensHelper(0,n,cols,slash,backSlash,curMap,ans);
    return ans;
  }

  private void solveQueensHelper(int row,int n,boolean[] cols,boolean[] slash,boolean[] backSlash,List<String> curMap,List<List<String>> ans){
    if (row==n){
      ans.add(new ArrayList<>(curMap));
      return;
    }
    for (int col=0;col<n;col++){
      int slashIndex=col+row,backSlashIndex=row-col+n;
      if (cols[col] || slash[slashIndex] ||backSlash[backSlashIndex])
        continue;
      cols[col]=slash[slashIndex]=backSlash[backSlashIndex]=true;
      char[] newRow=new char[n];
      Arrays.fill(newRow,'.');
      newRow[col]='Q';
      curMap.add(new String(newRow));
      solveQueensHelper(row+1,n,cols,slash,backSlash,curMap,ans);
      curMap.remove(curMap.size()-1);
      cols[col]=slash[slashIndex]=backSlash[backSlashIndex]=false;
    }
  }

  public List<Integer> splitIntoFibonacci(String S) {
    List<Integer> ans=new ArrayList<>();
    if (S==null||S.length()==0)
      return ans;
    SFhelper(S,0,new ArrayList<>(),ans);
    return ans;
  }

  private void SFhelper(String S,int start,List<Integer> cur,List<Integer> ans){
    if (!ans.isEmpty())
      return;
    if (start==S.length()){
      if (cur.size()>2)
        ans.addAll(cur);
      return;
    }
    int N=cur.size();
    long V=0;
    for (int i = start; i < S.length(); i++) {
      V=V*10+(S.charAt(i)-'0');
      if (V > Integer.MAX_VALUE)
        break;
      if (N < 2) {
        cur.add((int) V);
        SFhelper(S, i + 1, cur, ans);
        cur.remove(cur.size() - 1);
      }
      else {
        int beforeSum = cur.get(N - 1) + cur.get(N - 2);
        if (beforeSum > V)
          continue;
        else if (beforeSum < V)
          break;
        else {
          cur.add((int) V);
          SFhelper(S, i + 1, cur, ans);
          cur.remove(cur.size() - 1);
        }
      }
      if (V==0)
        break;
    }
  }

  public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> ans=new ArrayList<>();
    if (root==null)
      return ans;
    Queue<TreeNode> q=new LinkedList<>();
    q.offer(root);
    int layerNum=0;
    while (!q.isEmpty()){
      layerNum++;
      int size=q.size();
      LinkedList<Integer> temp=new LinkedList<>();
      for (int i=0;i<size;i++){
        TreeNode cur=q.poll();
        if ((layerNum &1)==1)
          temp.addLast(cur.val);
        else
          temp.addFirst(cur.val);
        if (cur.left!=null)
          q.offer(cur.left);
        if (cur.right!=null)
          q.offer(cur.right);
      }
      ans.add(temp);
    }
    return ans;
  }

  public int nthUglyNumber(int n) {
    if (n<=0)
      throw new IllegalArgumentException();
    if (n==1)
      return 1;
    int[] res=new int[n];
    res[0]=1;
    int m2=2,m3=3,m5=5,i2=0,i3=0,i5=0,size=1;
    for (int i=2;i<=n;i++){
      int max=res[size-1];
      while (m2<=max)
        m2=res[++i2]<<1;
      while (m3<=max)
        m3=res[++i3]*3;
      while (m5<=max)
        m5=res[++i5]*5;
      res[size++]=Math.min(m2,Math.min(m3,m5));
    }
    return res[n-1];
  }

  class posToTime{
    int pos;
    double time;
    public posToTime(int p,double t){
      pos=p;
      time=t;
    }
  }

  public int carFleet(int target, int[] position, int[] speed) {
    int N=position.length;
    if (N==0)
      return 0;
    int ans=1;
    posToTime[] pt=new posToTime[N];
    for (int i=0;i<N;i++)
      pt[i]=new posToTime(position[i],(double)(target-position[i])/(double)speed[i]);
    Arrays.sort(pt,(a,b)->b.pos-a.pos);
    for (int i=1;i<N;i++)
      if (pt[i].time>pt[i-1].time)
        ans++;
      else
        pt[i].time=pt[i-1].time;
    return ans;
  }

  public boolean isValidSerialization1(String preorder) {
    if (preorder==null || preorder.length()==0)
      return true;
    String[] po=preorder.split(",");
    int index=0,N=po.length,stSize=0;
    while ( index<N && !po[index].equals( '#')){
      if (po[index].equals( "#"))
        if (stSize>0)
          stSize--;
        else
          break;
      else
        stSize++;
      index++;
    }
    return index==N-1;
  }

  public boolean isValidSerialization(String preorder) {
    if (preorder==null || preorder.length()==0)
      return true;
    int dif=1;
    String[] sp=preorder.split(",");
    for (String s:sp){
      if (--dif<0)
        return false;
      if (!s.equals("#"))
        dif+=2;
    }
    return dif==0;
  }

  class SUrecord{
    int val,base,cur;
    public SUrecord(int v,int b,int c){
      val=v;
      base=b;cur=c;
    }
  }

  public int nthSuperUglyNumber(int n, int[] primes) {
    if (n<=0)
      throw new IllegalArgumentException();
    if (n==1)
      return 1;
    int[] res=new int[n];
    res[0]=1;
    PriorityQueue<SUrecord> pq=new PriorityQueue<>((a,b)->a.val-b.val);
    for (int p:primes)
      pq.offer(new SUrecord(p,p,0));
    for (int i=1;i<n;i++){
      res[i] = pq.peek().val;
      while (pq.peek().val<=res[i]){
        SUrecord temp=pq.poll();
        temp.val=temp.base*res[++temp.cur];
        pq.offer(temp);
      }
    }
    return res[n-1];
  }

  public boolean isPossible1(int[] nums) {
    if (nums==null||nums.length<3)
      return false;
    PriorityQueue<List<Integer>> pq=new PriorityQueue<>((a,b)->a.size()-b.size());
    List<Integer> newOne=new ArrayList<>();
    newOne.add(nums[0]);
    pq.offer(newOne);
    List<List<Integer>> cache=new ArrayList<>();
    for (int i=1;i<nums.length;i++){
      int n=nums[i];
      do {
        List<Integer> temp = pq.poll();
        int last=temp.get(temp.size()-1);
        if (n==last+1){
          temp.add(n);
          pq.offer(temp);
          for (List<Integer> ls:cache)
            pq.offer(ls);
          cache.clear();
          break;
        }
        else if (n==last || temp.size()>=3)
          cache.add(temp);
        else
          return false;
      }while (!pq.isEmpty());
      if (!cache.isEmpty()){
        List<Integer> cur=new ArrayList<>();
        cur.add(n);
        pq.offer(cur);
        for (List<Integer> ls:cache)
          pq.offer(ls);
        cache.clear();
      }
    }
    return pq.peek().size()>=3;
  }

  public boolean isPossible(int[] nums) {
    if (nums == null || nums.length < 3)
      return false;
    Map<Integer,Integer> freq=new HashMap<>(),appendFreq=new HashMap<>();
    for (int n:nums)
      freq.put(n,freq.getOrDefault(n,0)+1);
    for (int n:nums){
      if (freq.get(n)==0)
        continue;
      else if (appendFreq.getOrDefault(n,0)>0){
        appendFreq.put(n,appendFreq.get(n)-1);
        appendFreq.put(n+1,appendFreq.getOrDefault(n+1,0)+1);
      }
      else if (freq.getOrDefault(n+1,0)>0 && freq.getOrDefault(n+2,0)>0){
        freq.put(n+1,freq.get(n+1)-1);
        freq.put(n+2,freq.get(n+2)-1);
        appendFreq.put(n+3,appendFreq.getOrDefault(n+3,0)+1);
      }
      else
        return false;
      freq.put(n,freq.get(n)-1);
    }
    return true;
  }

  public int twoCitySchedCost(int[][] costs) {
    if (costs==null || costs.length==0)
      return 0;
    int N=costs.length>>1,cost=0;
    int[] remain=new int[]{N,N};
    Arrays.sort(costs,(a,b)->Math.abs(b[0]-b[1])-Math.abs(a[0]-a[1]));
    for (int[] c:costs){
      int dest=c[0]<=c[1]?0:1,couldPick=remain[dest]>0?dest:~dest;
      remain[couldPick]--;
      cost+=c[couldPick];
    }
    return cost;
  }

  public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
    if (src==dst)
      return 0;
    List<int[]>[] graph=new List[n];
    for (int i=0;i<n;i++)
      graph[i]=new ArrayList<>();
    for (int[] f:flights)
      graph[f[0]].add(new int[]{f[1],f[2]});
    PriorityQueue<int[]> pq=new PriorityQueue<>((a,b)->a[0]-b[0]);
    pq.offer(new int[]{0,src,K+1});
    while (!pq.isEmpty()){
      int[] res=pq.poll();
      int price=res[0],city=res[1],stop=res[2];
      if (city==dst)
        return price;
      if (stop>0)
        for (int[] next:graph[city])
          pq.offer(new int[]{price+next[1],next[0],stop-1});
    }
    return -1;
  }

  public List<int[]> kSmallestPairs1(int[] nums1, int[] nums2, int k) {
    List<int[]> ans=new ArrayList<>();
    if (k==0 || nums1.length==0||nums2.length==0)
      return ans;
    PriorityQueue<int[]> pq=new PriorityQueue<>((a,b)->a[0]+a[1]-b[0]-b[1]);
    for (int n1:nums1)
      for (int n2:nums2)
        pq.offer(new int[]{n1,n2});
    for (int i=0;i<k;i++)
      if (pq.isEmpty())
        break;
      else
        ans.add(pq.poll());
    return ans;
  }

  public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
    List<int[]> ans = new ArrayList<>();
    if (k == 0 || nums1.length == 0 || nums2.length == 0)
      return ans;
    PriorityQueue<int[]> pq=new PriorityQueue<>((a,b)->a[0]+a[1]-b[0]-b[1]);
    for (int i=0;i<nums1.length && i<k;i++)
      pq.offer(new int[]{nums1[i],nums2[0],0});
    while (k-->0 && !pq.isEmpty()){
      int[] pair=pq.poll();
      ans.add(new int[]{pair[0],pair[1]});
      if (pair[2]==nums2.length-1)
        continue;
      pq.offer(new int[]{pair[0],nums2[pair[2]+1],pair[2]+1});
    }
    return ans;
  }

  public int[] advantageCount(int[] A, int[] B) {
    int N=A.length,supply=0,lo=0,hi=N-1;
    int[] ans=new int[N];
    Arrays.sort(A);
    PriorityQueue<int[]> pq=new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return b[0]-a[0];
      }
    });
    for (int i=0;i<N;i++)
      pq.offer(new int[]{B[i],i});
    while (!pq.isEmpty()){
      int[] cur=pq.poll();
      ans[cur[1]]=A[hi]>cur[0]?A[hi--]:A[lo++];
    }
    return ans;
  }

  public int eraseOverlapIntervals(int[][] intervals) {
    if (intervals==null||intervals.length==0)
      return 0;
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]-b[0];
      }
    });
    int N=intervals.length,ans=0;
    for (int i=0;i<N-1;i++)
      if (EOIhasInterval(intervals[i],intervals[i+1])){
        ans++;
        if (intervals[i+1][1]>intervals[i][1])
          intervals[i+1]=intervals[i];
      }
    return ans;
  }

  private boolean EOIhasInterval(int[] a,int[] b){
    if (a[0]>=b[1] || a[1]<=b[0])
      return false;
    else
      return true;
  }

  public int[][] allCellsDistOrder1(int R, int C, int r0, int c0) {
    if (R==1 && C==1)
      return new int[][]{{0,0}};
    int N=R*C,index=0,dist=1;
    int[][] ans=new int[N][2];
    ans[index++]=new int[]{r0,c0};
    while (index<N){
      for (int dr=0;dr<=dist;dr++){
        int dc=dist-dr;
        int[][] loc=dr==0?new int[][]{{0,dc},{0,-dc}}:dc==0?new int[][]{{dr,0},{-dr,0}}:new int[][]{{dr,dc},{dr,-dc},{-dr,dc},{-dr,-dc}};
        for (int[] lo:loc){
          int r=r0+lo[0],c=c0+lo[1];
          if (r>=0 && r<R && c>=0&& c<C)
            ans[index++]=new int[]{r,c};
        }
      }
      dist++;
    }
    return ans;
  }

  public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
    if (R == 1 && C == 1)
      return new int[][]{{0, 0}};
    int N = R * C, index = 0;
    int[][] ans = new int[N][2],dir=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    boolean[][] visited=new boolean[R][C];
    Queue<int[]> q=new LinkedList<>();
    q.offer(new int[]{r0,c0});
    while (!q.isEmpty()){
      int[] cur=q.poll();
      if (visited[cur[0]][cur[1]])
        continue;
      visited[cur[0]][cur[1]]=true;
      ans[index++]=cur;
      for (int[] d:dir){
        int r=cur[0]+d[0],c=cur[1]+d[1];
        if (r>=0 && r<R && c>=0&& c<C)
          q.offer(new int[]{r,c});
      }
    }
    return ans;
  }

  public int monotoneIncreasingDigits(int N) {
    if (N<10)
      return N;
    char[] digits=String.valueOf(N).toCharArray();
    int from=digits.length;
    for (int i=digits.length-1;i>0;i--)
      if (digits[i]<digits[i-1]){
        from=i;
        digits[i-1]--;
      }
    for (int i=from;i<digits.length;i++)
      digits[i]='9';
    return Integer.parseInt(new String(digits));
  }

  public int brokenCalc(int X, int Y) {
    if (X==Y)
      return 0;
    if (X>Y)
      return X-Y;
    if ((Y&1)==1)
      return 2+brokenCalc(X,(Y+1)>>1);
    else
      return 1+brokenCalc(X,Y>>1);
  }

  public ListNode sortList(ListNode head) {
    if (head==null||head.next==null)
      return head;
    int len=0;
    ListNode cur=head;
    while (cur!=null){
      cur=cur.next;
      len++;
    }

    return ListMergeSort(head,len);
  }

  private ListNode ListMergeSort(ListNode head,int length){
    if (length<=1)
      return head;
    int len1=length>>1,len2=length-len1;
    ListNode cur=head;
    for (int i=0;i<len1;i++)
      cur=cur.next;
    ListNode l1=ListMergeSort(head,len1),l2=ListMergeSort(cur,len2);
    return LinkedListMerge(l1,l2,len1,len2);
  }

  private ListNode LinkedListMerge(ListNode l1,ListNode l2,int len1,int len2){
    ListNode head=new ListNode(0),ans=head;
    while (len1>0 || len2>0){
      if (len1==0){
        head.next=l2;
        l2=l2.next;
        len2--;
      }
      else if (len2==0){
        head.next=l1;
        l1=l1.next;
        len1--;
      }
      else if (l1.val<=l2.val){
        head.next=l1;
        l1=l1.next;
        len1--;
      }
      else{
        head.next=l2;
        l2=l2.next;
        len2--;
      }
      head=head.next;
      if (len1==0 && len2==0)
        head.next=null;
    }
    return ans.next;
  }

  public int hIndex(int[] citations) {
    if (citations==null||citations.length==0)
      return 0;
    int N=citations.length;
    int[] bucket=new int[N+1];
    for (int c:citations)
      if (c>=N)
        bucket[N]++;
      else
        bucket[c]++;
    int count=0;
    for (int i=N;i>=0;i--){
      count+=bucket[i];
      if (count>=i)
        return i;
    }
    return 0;
  }

  public void wiggleSort(int[] nums) {
    if (nums==null || nums.length==0)
      return;
    int N=nums.length;
    Arrays.sort(nums);
    int[] aux=Arrays.copyOf(nums,N);
    int half=N>>1,sm=(N&1)==1?half:half-1,bg=N-1,index=0;
    while (index<N)
      if ((index&1)==0)
        nums[index++]=aux[sm--];
      else
        nums[index++]=aux[bg--];
  }

  public String largestNumber(int[] nums) {
    if (nums==null||nums.length==0)
      return "0";
    int N=nums.length;
    String[] num=new String[N];
    for (int i=0;i<N;i++)
      num[i]=String.valueOf(nums[i]);
    Arrays.sort(num,new Comparator<String>(){
      public int compare(String a,String b){
        String ba=b+a,ab=a+b;
        return ba.compareTo(ab);
      }});
    StringBuilder sb=new StringBuilder();
    for (String n:num)
      sb.append(n);
    String ans=sb.toString();
    if (ans.charAt(0)=='0')
      return "0";
    else
      return ans;
  }

  public int singleNumber2(int[] nums) {
    int[] bitRecorder=new int[32];
    int index,ans=0;
    for (int n:nums){
      index=0;
      while (n!=0){
        bitRecorder[index++]+=n&1;
        n>>>=1;
      }
    }
    for (int i=0;i<32;i++)
      bitRecorder[i]%=3;
    for (int i=31;i>=0;i--)
      ans=(ans<<1)+bitRecorder[i];
    return ans;
  }

  public int singleNumber3(int[] nums){
    int one=0,two=0;
    for (int n:nums){
      one=(one^n)&~two;
      two=(two^n)&~one;
    }
    return one;
  }

  public List<String> findRepeatedDnaSequences(String s) {
    Set<Integer> seen=new HashSet<>(),repeated=new HashSet<>();
    char[] cs=s.toCharArray();
    List<String> ans=new ArrayList<>();
    int[] map=new int[26];
    map['C'-'A']=1;
    map['G'-'A']=2;
    map['T'-'A']=3;
    for (int i=0;i<s.length()-9;i++){
      int key=0;
      for (int j=i;j<i+10;j++){
        key<<=2;
        key|=map[cs[j]-'A'];
      }
      if (!seen.add(key) && repeated.add(key))
       ans.add(s.substring(i,i+10));
    }
    return ans;
  }

  public int rangeBitwiseAnd(int m, int n) {
    int factor=1;
    while (m!=n){
      m>>=1;
      n>>=1;
      factor<<=1;
    }
    return m*factor;
  }

  public int rangeBitwiseAnd1(int m, int n) {
    while (n>m)
      n&=(n-1);
    return n;
  }

  public boolean validUtf8(int[] data) {
    if (data==null || data.length==0)
      return false;
    int remain=0;
    for (int d:data){
      int startOnes=startOnesNum(d);
      if (startOnes==1)
        if (remain==0)
          return false;
        else
          remain--;
      else
        if (remain!=0 || startOnes>4)
          return false;
        else
          remain=startOnes==0?0:startOnes-1;
    }
    return remain==0;
  }

  private int startOnesNum(int n){
    int mask=1<<7,ans=0;
    while ((mask&n)!=0){
      ans++;
      mask>>=1;
    }
    return ans;
  }

  public int subarrayBitwiseORs(int[] A) {
    if (A==null ||A.length==0)
      return 0;
    int N=A.length;
    Set<Integer> ans=new HashSet<>(),before=new HashSet<>(),cur;
    for (int a:A){
      cur=new HashSet<>();
      cur.add(a);
      for (int b:before)
        cur.add(b|a);
      ans.addAll(before=cur);
    }
    return ans.size();
  }

  class FDSnode{
    int count;
    TreeNode root;
    public FDSnode(TreeNode root){
      this.root=root;
      count=0;
    }
  }

  public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    List<TreeNode> ans=new ArrayList<>();
    if (root==null||(root.left==null && root.right==null))
      return ans;
    Map<String,FDSnode> count=new HashMap<>();
    FDSserializeSubtrees(root,count);
    for (Map.Entry<String,FDSnode> entry:count.entrySet())
      if (entry.getValue().count>1)
        ans.add(entry.getValue().root);
    return ans;
  }

  private String FDSserializeSubtrees(TreeNode root,Map<String,FDSnode> count){
    if (root==null)
      return "";
    String left=FDSserializeSubtrees(root.left,count),right=FDSserializeSubtrees(root.right,count);
    String preOrder=root.val+" "+left+" "+right;
    FDSnode temp=count.getOrDefault(preOrder,new FDSnode(root));
    temp.count++;
    count.put(preOrder,temp);
    return preOrder;
  }

  public class Codec1 {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
      if (root==null)
        return "*";
      String left=serialize(root.left),right=serialize(root.right);
      return root.val+" "+left+" "+right;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
      if (data.equals("*"))
        return null;
      String[] treeParts=data.split(" ");
      TreeNode root=new TreeNode(Integer.valueOf(treeParts[0])),parent=root;
      int N=treeParts.length,index=0,cur=1;
      TreeNode[] treeStack=new TreeNode[N];
      treeStack[index++]=root;
      boolean isRight=false;
      while (!treeParts[cur].equals("*") || index!=0){
        if (!treeParts[cur].equals("*")){
          if (isRight){
            parent.right=new TreeNode(Integer.valueOf(treeParts[cur]));
            parent=parent.right;
          }
          else{
            parent.left=new TreeNode(Integer.valueOf(treeParts[cur]));
            parent=parent.left;
          }
          isRight=false;
          treeStack[index++]=parent;
        }
        else{
          parent=treeStack[--index];
          isRight=true;
        }
        cur++;
      }
      return root;
    }
  }

  public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
      if (root==null)
        return "*";
      StringBuilder sb=new StringBuilder();
      serialize(root,sb);
      return sb.toString();
    }

    private void serialize(TreeNode root,StringBuilder sb){
      if (root==null){
        sb.append("* ");
        return;
      }
      sb.append(root.val);
      sb.append(" ");
      serialize(root.left,sb);
      serialize(root.right,sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
      int[] index=new int[1];
      return deserialize(data.split(" "),index);
    }

    private TreeNode deserialize(String[] data,int[] index){
      if (data[index[0]].equals("*")){
        index[0]++;
        return null;
      }
      TreeNode ans=new TreeNode(Integer.valueOf(data[index[0]++]));
      ans.left=deserialize(data,index);
      ans.right=deserialize(data,index);
      return ans;
    }
  }

  class RPnode{
    int depth,val;
    public RPnode(int d,int v){
      depth=d;
      val=v;
    }
  }

  public TreeNode recoverFromPreorder(String S) {
    if (S==null)
      return null;
    List<RPnode> data=new ArrayList<>();
//    Pattern p= Pattern.compile("(-*)(\\d+)");
//    Matcher m=p.matcher(S);
//    while (m.find()){
//      int depth=m.group(1).length();
//      int val=Integer.valueOf(m.group(2));
//      data.add(new RPnode(depth,val));
//    }
    RPfindDepth(S,data);
    return RPconstructTree(new int[1],data,-1);
  }

  private void RPfindDepth(String S,List<RPnode> data){
    int depth=0,numStart=-1;
    char[] cs=S.toCharArray();
    for (int i=0;i<cs.length;i++)
      if (cs[i]=='-')
        if (numStart==-1)
          depth++;
        else{
          RPnode cur=new RPnode(depth,Integer.valueOf(S.substring(numStart,i)));
          depth=1;
          numStart=-1;
          data.add(cur);
        }
      else if (numStart==-1)
          numStart=i;
    data.add(new RPnode(depth,Integer.valueOf(S.substring(numStart,cs.length))));
  }

  private TreeNode RPconstructTree(int[] index,List<RPnode> data,int parentDepth){
    if (index[0]>=data.size())
      return null;
    RPnode cur;
    if ((cur=data.get(index[0])).depth<=parentDepth)
      return null;
    TreeNode ans = new TreeNode(cur.val);
    index[0]++;
    ans.left=RPconstructTree(index,data,cur.depth);
    ans.right=RPconstructTree(index,data,cur.depth);
    return ans;
  }

  public int pathSum1(TreeNode root, int sum) {
    if (root==null)
      return 0;
    int[] pathNum=new int[1];
    pathSum(root,sum,pathNum);
    return pathNum[0];
  }

  private List<Integer> pathSum(TreeNode root,int sum,int[] pathNum){
    if (root==null)
      return new ArrayList<>();
    List<Integer> leftSum=pathSum(root.left,sum,pathNum),rightSum=pathSum(root.right,sum,pathNum),ans=new ArrayList<>(leftSum.size()+ rightSum.size()+1);
    for (int l:leftSum){
      int res=l+root.val;
      ans.add(res);
      if (res==sum)
        pathNum[0]++;
    }
    for (int r:rightSum){
      int res=r+root.val;
      ans.add(res);
      if (res==sum)
        pathNum[0]++;
    }
    ans.add(root.val);
    if (root.val==sum)
      pathNum[0]++;
    return ans;
  }

  public int pathSum(TreeNode root, int sum) {
    if (root == null)
      return 0;
    Map<Integer,Integer> preSum=new HashMap<>();
    preSum.put(0,1);
    return pathHelper(root,0,sum,preSum);
  }

  private int pathHelper(TreeNode root,int curSum,int sum,Map<Integer,Integer> preSum){
    if (root==null)
      return 0;
    curSum+=root.val;
    int res=preSum.getOrDefault(curSum-sum,0);
    preSum.put(curSum,preSum.getOrDefault(curSum,0)+1);
    res+= pathHelper(root.left,curSum,sum,preSum)+pathHelper(root.right,curSum,sum,preSum);
    preSum.put(curSum,preSum.get(curSum)-1);
    return res;
  }

  public List<Integer> flipMatchVoyage(TreeNode root, int[] voyage) {
    List<Integer> ans=new ArrayList<>();
    int[] index=new int[1];
    FMVhelper(root,voyage,index,ans);
    return ans;
  }

  private void FMVhelper(TreeNode root,int[] voyage,int[] index,List<Integer> ans){
    if (ans.size()==1 && ans.get(0)==-1)
      return;
    if (root==null)
      return;
    if (root.val!=voyage[index[0]]  || (index[0]==voyage.length-1 && (root.left!=null || root.right!=null))){
      ans.clear();
      ans.add(-1);
      return;
    }
    index[0]++;
    if (root.left==null || root.left.val==voyage[index[0]]){
      FMVhelper(root.left,voyage,index,ans);
      FMVhelper(root.right,voyage,index,ans);
    }
    else{
      ans.add(root.val);
      FMVhelper(root.right,voyage,index,ans);
      FMVhelper(root.left,voyage,index,ans);
    }
  }

  public int sumNumbers(TreeNode root) {
    if (root==null)
      return 0;
    int[] ans=new int[1];
    SNhelper(root,0,ans);
    return ans[0];
  }

  private void SNhelper(TreeNode root,int res,int[] ans){
    if (root==null)
      return;
    if (root.left==null && root.right==null){
      ans[0]+= res*10+root.val;
      return;
    }
    res=res*10+root.val;
    SNhelper(root.left,res,ans);
    SNhelper(root.right,res,ans);
  }

  public int numEnclaves(int[][] A) {
    if (A==null || A.length==0 || A[0].length==0)
      return 0;
    int ans=0,R=A.length,C=A[0].length;
    boolean[][] visited=new boolean[R][C];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (A[r][c]==1 && !visited[r][c]){
          int[] res=new int[1];
          NEhelper(A,r,c,visited,res);
          if (res[0]!=-1)
            ans+=res[0];
        }
    return ans;
  }

  private void NEhelper(int[][] A,int r,int c,boolean[][] visited,int[] res){
    if (r<0 || r>=A.length || c<0 || c>=A[0].length){
      res[0]=-1;
      return;
    }
    if (A[r][c]==0 || visited[r][c])
      return;
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    visited[r][c]=true;
    if (res[0]!=-1)
      res[0]++;
    for (int[] d:dirs)
      NEhelper(A,r+d[0],c+d[1],visited,res);
  }

  public boolean isSubtree(TreeNode s, TreeNode t) {
    if (s==null)
      return false;
    return ISfindStart(s,t);
  }

  private boolean ISfindStart(TreeNode root,TreeNode t){
    if (root==null)
      return false;
    boolean ans=false;
    if (root.val==t.val)
      ans = ISisSame(root,t);
    return ans || ISfindStart(root.left,t) || ISfindStart(root.right,t);
  }

  private boolean ISisSame(TreeNode root,TreeNode t){
    if (t==null&& root==null)
      return true;
    if (t==null || root==null ||root.val!=t.val )
      return false;
    return ISisSame(root.left,t.left) && ISisSame(root.right,t.right);
  }

  public void flatten(TreeNode root) {
    if (root==null)
      return;
    TreeNode[] parent=new TreeNode[1];
    parent[0]=new TreeNode(0);
    FlinkParent(root,parent);
  }

  private void FlinkParent(TreeNode cur,TreeNode[] parent){
    if (cur==null)
      return;
    TreeNode left=cur.left,right=cur.right;
    parent[0].right=cur;
    cur.left=cur.right=null;
    parent[0]=cur;
    FlinkParent(left,parent);
    FlinkParent(right,parent);
  }

  public TreeNode buildTree1(int[] preorder, int[] inorder) {
    if (preorder==null || preorder.length==0)
      return null;
    Map<Integer,Integer> record=new HashMap<>();
    for (int i=0;i<inorder.length;i++)
      record.put(inorder[i],i);
    return BThelper(preorder,new int[1],inorder,0,inorder.length-1,record);
  }

  private TreeNode BThelper(int[] pre,int[] preIndex, int[] in,int start,int end,Map<Integer,Integer> record){
    if (preIndex[0]>=pre.length ||start>end )
      return null;
    if (start==end){
      preIndex[0]++;
      return new TreeNode(in[start]);
    }
    TreeNode cur=new TreeNode(pre[preIndex[0]++]);
    int curIndex=record.get(cur.val);
    cur.left=BThelper(pre,preIndex,in,start,curIndex-1,record);
    cur.right=BThelper(pre,preIndex,in,curIndex+1,end,record);
    return cur;
  }

  public TreeNode buildTree(int[] preorder, int[] inorder) {
    if (preorder == null || preorder.length == 0)
      return null;
    return BThelper2(preorder,inorder,new int[1],new int[1],Long.MAX_VALUE);
  }

  private TreeNode BThelper2(int[] pre,int[] in,int[] preIndex,int[] inIndex,long val){
    if (inIndex[0]>=in.length || in[inIndex[0]]==val)
      return null;
    TreeNode cur=new TreeNode(pre[preIndex[0]++]);
    cur.left=BThelper2(pre,in,preIndex,inIndex,cur.val);
    inIndex[0]++;
    cur.right=BThelper2(pre,in,preIndex,inIndex,val);
    return cur;
  }

  public List<List<Integer>> pathSum2(TreeNode root, int sum) {
    List<List<Integer>> ans=new ArrayList<>();
    if (root==null)
      return ans;
    PShelper(root,sum,new ArrayList<>(),ans);
    return ans;
  }

  private void PShelper(TreeNode root,int remain,List<Integer> path,List<List<Integer>> ans){
    if (root==null)
      return;
    path.add(root.val);
    if (root.left==null && root.right==null && remain==root.val)
      ans.add(new ArrayList<>(path));
    remain-=root.val;
    if (root.left!=null)
      PShelper(root.left,remain,path,ans);
    if (root.right!=null)
      PShelper(root.right,remain,path,ans);
    path.remove(path.size()-1);
  }

  public List<List<Integer>> findSubsequences(int[] nums) {
   List<List<Integer>> ans=new ArrayList<>();
   if (nums==null || nums.length==0)
     return ans;
   FShelper(nums,0,new ArrayList<>(),ans);
   return ans;
  }

  private void FShelper(int[] nums,int index,List<Integer> path,List<List<Integer>> ans){
    if (path.size()>1)
      ans.add(new ArrayList<>(path));
    Set<Integer> appeared=new HashSet<>();
    for (int i=index;i<nums.length;i++){
      if (appeared.contains(nums[i]))
        continue;
      if (path.isEmpty() || nums[i]>=path.get(path.size()-1)){
        appeared.add(nums[i]);
        path.add(nums[i]);
        FShelper(nums,i+1,path,ans);
        path.remove(path.size()-1);
      }
    }
  }

  public int[][] colorBorder(int[][] grid, int r0, int c0, int color) {
    if (grid==null || grid.length==0|| grid[0].length==0)
      return grid;
    boolean[][] visited=new boolean[grid.length][grid[0].length];
    Set<int[]> colored=new HashSet<>();
    CBhelper(grid,r0,c0,grid[r0][c0],visited,colored);
    for (int[] c:colored)
      grid[c[0]][c[1]]=color;
    return grid;
  }

  private void CBhelper(int[][] grid,int r,int c,int fromColor,boolean[][] visited,Set<int[]> colored){
    int R=grid.length,C=grid[0].length;
    if (visited[r][c])
      return;
    visited[r][c]=true;
    int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    for (int[] d:dirs){
      int cur_r=r+d[0],cur_c=c+d[1];
      if (cur_r<0 || cur_r>= R || cur_c<0 || cur_c>=C || grid[cur_r][cur_c]!=fromColor)
        colored.add(new int[]{r,c});
      else
        CBhelper(grid,cur_r,cur_c,fromColor,visited,colored);
    }
  }

  public int[][] updateMatrix(int[][] matrix) {
    if (matrix==null || matrix.length==0 || matrix[0].length==0)
      return matrix;
    int R=matrix.length,C=matrix[0].length;
    int[][] dist=new int[R][C];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        dist[r][c]=matrix[r][c]==0?0:UMminDist(matrix,r,c);
    return dist;
  }

  private int UMminDist(int[][] M,int r,int c){
     int dist=0,R=M.length,C=M[0].length;
     int[][] dirs=new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
     Queue<int[]> q=new LinkedList<>();
     q.offer(new int[]{r,c});
     while (!q.isEmpty()){
       int size=q.size();
       dist++;
       for (int i=0;i<size;i++){
         int[] L=q.poll();
         for (int[] d:dirs){
           int curR=L[0]+d[0],curC=L[1]+d[1];
           if (curR<0 || curR>=R || curC<0 ||curC>=C)
             continue;
           if (M[curR][curC]==0)
             return dist;
           q.offer(new int[]{curR,curC});
         }
       }
     }
     return 0;
  }

  public List<int[]> pacificAtlantic(int[][] matrix) {
    List<int[]> ans=new ArrayList<>();
    if (matrix==null || matrix.length==0||matrix[0].length==0)
      return ans;
    int R=matrix.length,C=matrix[0].length;
    boolean[][] toP=new boolean[R][C],toA=new boolean[R][C];
    for (int r=0;r<R;r++){
      PAexplore(matrix,r,0,toP);
      PAexplore(matrix,r,C-1,toA);
    }
    for (int c=0;c<C;c++){
      PAexplore(matrix,0,c,toP);
      PAexplore(matrix,R-1,c,toA);
    }
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (toP[r][c]&&toA[r][c])
          ans.add(new int[]{r,c});
    return ans;
  }

  private void PAexplore(int[][] M,int r,int c,boolean[][] visited){
    int R= M.length,C=M[0].length;
    if (visited[r][c])
      return;
    visited[r][c]=true;
    int[][] dirs=new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
    for (int[] d:dirs){
      int curR=r+d[0],curC=c+d[1];
      if (curR<0 || curR>=R || curC<0 ||curC>=C||M[curR][curC]<M[r][c])
        continue;
      PAexplore(M,curR,curC,visited);
    }
  }

  public List<List<String>> accountsMerge1(List<List<String>> accounts) {
    if (accounts==null||accounts.isEmpty())
      return accounts;
    List<List<String>> ans=new ArrayList<>();
    Map<String,List<List<String>>> nameToAcounts=new HashMap<>();
    for (List<String> ls:accounts){
      String name=ls.get(0);
      if (!nameToAcounts.containsKey(name))
        nameToAcounts.put(name,new ArrayList<>());
      List<String> acts=new ArrayList<>();
      for (int i=1;i<ls.size();i++)
        acts.add(ls.get(i));
      nameToAcounts.get(name).add(acts);
    }
    for (Map.Entry<String,List<List<String>>> entry:nameToAcounts.entrySet()){
      if (entry.getValue().size()==1){
        Set<String> st=new TreeSet<>(entry.getValue().get(0));
        List<String> res=new ArrayList<>();
        res.add(entry.getKey());
        res.addAll(st);
        ans.add(res);
        continue;
      }
      List<List<String>> val=entry.getValue();
      int N=val.size();
      int[] id=new int[N];
      int[] weights=new int[N];
      for (int i=0;i<N;i++){
        id[i]=i;
        weights[i]=1;
      }
      for (int i=0;i<N-1;i++)
        for (int j=i+1;j<N;j++)
          if (AMhasCommon(val.get(i),val.get(j)))
            AMunion(i,j,id,weights);
      Map<Integer,Set<String>> same=new HashMap<>();
      for (int i=0;i<N;i++){
        int curId=AMfind(id,i);
        if (!same.containsKey(curId))
          same.put(curId,new TreeSet<>());
        Set<String> st=same.get(curId);
        List<String> record=val.get(i);
        st.addAll(record);
      }
      for (Map.Entry<Integer,Set<String>> et:same.entrySet()){
        List<String> v=new ArrayList<>(et.getValue()),res=new ArrayList<>();
        res.add(entry.getKey());
        res.addAll(v);
        ans.add(res);
      }
    }
    return ans;
  }

  private boolean AMhasCommon(List<String> a1,List<String> a2){
    Set<String> st=new HashSet<>();
    for (int i=0;i<a1.size();i++)
      st.add(a1.get(i));
    for (int i=0;i<a2.size();i++)
      if (st.contains(a2.get(i)))
        return true;
    return false;
  }

  public List<List<String>> accountsMerge(List<List<String>> accounts) {
    if (accounts == null || accounts.isEmpty())
      return accounts;
    List<List<String>> ans = new ArrayList<>();
    int N=accounts.size();
    int[] ids=new int[N];
    int[] weights=new int[N];
    for (int i=0;i<N;i++){
      ids[i]=i;
      weights[i]=1;
    }
    Map<String,Integer> idRecorder=new HashMap<>();
    int initialId=0;
    for (List<String> account:accounts){
      for (int i=1;i<account.size();i++){
        Integer id=idRecorder.putIfAbsent(account.get(i),initialId);
        if (id!=null)
          AMunion(initialId,id,ids,weights);
      }
      initialId++;
    }

    ArrayList<String>[] store=new ArrayList[N];
    for (Map.Entry<String,Integer> entry:idRecorder.entrySet()){
      int realId=AMfind(ids,entry.getValue());
      if (store[realId]==null)
        store[realId]=new ArrayList<>();
      store[realId].add(entry.getKey());
    }
    for (int i=0;i<N;i++){
      List<String> ls=store[i];
      if (ls==null)
        continue;
      List<String> cur=new ArrayList<>();
      cur.add(accounts.get(i).get(0));
      Collections.sort(ls);
      cur.addAll(ls);
      ans.add(cur);
    }
    return ans;
  }

  private int AMfind(int[] ids,int id){
    if (ids[id]==id)
      return id;
    int temp=id;
    while (ids[id]!=id)
      id=ids[id];
    while (temp!=id){
      int next=ids[temp];
      ids[temp]=id;
      temp=next;
    }
    return id;
  }

  private void AMunion(int i,int j,int[] ids,int[] weights){
    int iId=AMfind(ids,i),jId=AMfind(ids,j);
    if (iId==jId)
      return;
    if (weights[iId]<=weights[jId]){
      ids[iId]=jId;
      weights[jId]+=weights[iId];
    }
    else{
      ids[jId]=iId;
      weights[iId]+=weights[jId];
    }
  }

  public void solve(char[][] board) {
    if (board==null||board.length==0||board[0].length==0)
      return;
    int R=board.length,C=board[0].length;
    boolean[][] visited=new boolean[R][C];
    for (int r=0;r<R;r++){
      Shelper(board,visited,r,0);
      Shelper(board,visited,r,C-1);
    }
    for (int c=0;c<C;c++){
      Shelper(board,visited,0,c);
      Shelper(board,visited,R-1,c);
    }
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (!visited[r][c] && board[r][c]=='O')
          board[r][c]='X';
  }

  private void Shelper(char[][] board,boolean[][] visited,int r,int c){
    int R=board.length,C=board[0].length;
    if (r<0 || r>=R ||c<0 ||c>=C|| visited[r][c] || board[r][c]=='X')
      return;
    visited[r][c]=true;
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int[] d:dirs)
      Shelper(board,visited,r+d[0],c+d[1]);
  }

  public int maxSumAfterPartitioning(int[] A, int K) {
    if (A==null || A.length==0)
      return 0;
    int N=A.length;
    if (N==1)
      return A[0];
    int[] dp=new int[N];
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < Math.min(N, K);i++){
      int index=N-1-i;
      max = Math.max(max, A[index]);
      dp[index]=max*(i+1);
    }
    if (N <= K)
      return dp[0];
    for (int i=N-K-1;i>=0;i--){
      int curMax=Integer.MIN_VALUE,res=0;
      for (int j=0;j<K;j++){
        curMax=Math.max(curMax,A[i+j]);
        res=Math.max(curMax*(j+1)+dp[i+j+1],res);
      }
      dp[i]=res;
    }
    return dp[0];
  }

  public TreeNode bstToGst(TreeNode root) {
    if (root==null)
      return root;
    int[] curSum=new int[1];
    BTGhelper(root,curSum);
    return root;
  }

  private void BTGhelper(TreeNode root,int[] curSum){
    if (root==null)
      return;
    BTGhelper(root.right,curSum);
    curSum[0]+=root.val;
    root.val=curSum[0];
    BTGhelper(root.left,curSum);
  }

  public int[] gardenNoAdj(int N, int[][] paths) {
    List<Integer>[] graph=new List[N];
    for (int i=0;i<N;i++)
      graph[i]=new ArrayList<>();
    for (int[] p:paths){
      graph[p[0]-1].add(p[1]-1);
      graph[p[1]-1].add(p[0]-1);
    }
    int[] res=new int[N];
    for (int i=0;i<N;i++){
      boolean[] used=new boolean[5];
      for (int adj:graph[i])
        used[res[adj]]=true;
      for (int j=1;j<5;j++)
        if (!used[j]){
          res[i]=j;
          break;
        }
    }
    return res;
  }

  class Node133 {
    public int val;
    public List<Node133> neighbors;

    public Node133() {
    }

    public Node133(int _val, List<Node133> _neighbors) {
      val = _val;
      neighbors = _neighbors;
    }
  }

  public Node133 cloneGraph(Node133 node) {
    if (node==null)
      return node;
    Node133 ans=new Node133(node.val,new ArrayList<>());
    Map<Node133,Node133> mp=new HashMap<>();
    CGhelper(node,ans,mp);
    return ans;
  }

  private void CGhelper(Node133 node,Node133 cur,Map<Node133,Node133> map){
    map.put(node,cur);
    for (Node133 n:node.neighbors){
      Node133 neighbor=map.containsKey(n)?map.get(n):new Node133(n.val,new ArrayList<>());
      cur.neighbors.add(neighbor);
      if (!map.containsKey(n))
        CGhelper(n,neighbor,map);
    }
  }

  public int kthGrammar(int N, int K) {
    if (N==1)
      return 0;
    if (N==2)
      return K==1?0:1;
    int lastLen=1<<(N-2);
    return K-1>=lastLen?1-kthGrammar(N-1,K-lastLen):kthGrammar(N-1,K);
  }

  class WDNode{
    boolean isWord;
    WDNode[] nexts;
    public WDNode(){
      isWord=false;
      nexts=new WDNode[26];
    }
  }

  class WordDictionary {

    private WDNode root;
    /** Initialize your data structure here. */
    public WordDictionary() {
      root=new WDNode();
    }

    /** Adds a word into the data structure. */
    public void addWord(String word) {
      putWord(word,0,root);
    }

    private void putWord(String word,int index,WDNode root){
      if (index==word.length()){
        root.isWord=true;
        return;
      }
      int id=word.charAt(index)-'a';
      if (root.nexts[id]==null)
        root.nexts[id]=new WDNode();
      putWord(word,index+1,root.nexts[id]);
    }

    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
      return search(word,0,root);
    }

    private boolean search(String word,int index,WDNode root){
      if (index==word.length())
        return root.isWord;
      char curChar=word.charAt(index);
      if (curChar=='.'){
        for (int i=0;i<26;i++)
          if (root.nexts[i] !=null && search(word,index+1,root.nexts[i]))
            return true;
        return false;
      }
      else{
        int id=curChar-'a';
        return root.nexts[id]==null?false:search(word,index+1,root.nexts[id]);
      }
    }
  }

  public int[] numMovesStones(int a, int b, int c) {
    int[] ans=new int[2],data=new int[]{a,b,c};
    Arrays.sort(data);
    ans[1]=data[2]-data[0]-2;
    int dif1=data[1]-data[0],dif2=data[2]-data[1];
    if (dif1==1 && dif2==1)
      ans[0]=0;
    else if (dif1==1 || dif1==2 || dif2==1 || dif2==2)
      ans[0]=1;
    else
      ans[0]=2;
    return ans;
  }

  public boolean validTicTacToe(String[] board) {
    int turns=0,diag=0,antiDiag=0;
    boolean Xwin=false,Owin=false;
    int[] rows=new int[3],cols=new int[3];
    for (int i=0;i<3;i++)
      for (int j=0;j<3;j++){
        char c=board[i].charAt(j);
        if (c=='X'){
          turns++;
          rows[i]++;
          cols[j]++;
          if (i==j)
            diag++;
          if (i+j==2)
            antiDiag++;
        }
        else if (c=='O'){
         turns--;
         rows[i]--;
         cols[j]--;
          if (i==j)
            diag--;
          if (i+j==2)
            antiDiag--;
        }
    }
   for (int i=0;i<3;i++)
      if (rows[i]==3||cols[i]==3)
        Xwin=true;
      else if (rows[i]==-3 || cols[i]==-3)
        Owin=true;
   Xwin =Xwin || antiDiag==3 || diag==3;
   Owin=Owin || antiDiag==-3 ||diag==-3;
   if ( (Xwin && turns==0) || (Owin && turns==1))
     return false;
   return (turns==0 || turns==1)&& (!Owin || !Xwin);
  }

  public boolean canTransform(String start, String end) {
    int r=0,l=0,N=start.length();
    char[] st=start.toCharArray(),ed=end.toCharArray();
    for (int i=0;i<N;i++){
      if (st[i]=='R'){
        r++;
        if (l!=0)
          return false;
      }
      if (ed[i]=='R'){
        r--;
        if (l!=0)
          return false;
      }
      if (ed[i]=='L'){
        l++;
        if (r!=0)
          return false;
      }
      if (st[i]=='L'){
        l--;
        if (r!=0)
          return false;
      }
      if (l<0 || r<0)
        return false;
    }
    return r==0 && l==0;
  }

  public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
    if (desiredTotal<=maxChoosableInteger)
      return true;
    int total=(1+maxChoosableInteger)*maxChoosableInteger/2;
    if (total<desiredTotal)
      return false;
    if (total==desiredTotal)
      return (maxChoosableInteger&1)==1;
    Map<Integer,Boolean> memo=new HashMap<>();
    boolean[] used=new boolean[maxChoosableInteger];
    return CIWhelper(desiredTotal,used,memo);
  }

  private boolean CIWhelper(int remain,boolean[] used, Map<Integer,Boolean> memo){
    if (remain<=0)
      return false;
    int key=CIWformat(used);
    if (memo.containsKey(key))
      return memo.get(key);
    for (int i=0;i<used.length;i++){
      if (used[i])
        continue;
      used[i]=true;
      if (!CIWhelper(remain-i-1,used,memo)){
        used[i]=false;
        memo.put(key,true);
        return true;
      }
      else
        used[i]=false;
    }
    memo.put(key,false);
    return false;
  }

  private int CIWformat(boolean[] used){
    int ans=0;
    for (boolean u:used){
      ans<<=1;
      if (u)
        ans |= 1;
    }
    return ans;
  }

  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
    if (t<0 || k<1 )
      return false;
    Map<Long,Long> mp=new HashMap<>();
    int N=nums.length;
    for (int i=0;i<N;i++){
      long reposNum=(long)nums[i]-Integer.MIN_VALUE;
      long bucketId=reposNum/((long)t+1);
      if (mp.containsKey(bucketId) || (mp.containsKey(bucketId-1) &&reposNum -mp.get(bucketId-1) <=t)|| (mp.containsKey(bucketId+1) && mp.get(bucketId+1) - reposNum<=t))
        return true;
      if (mp.size()>=k){
        long lastBucketId=((long)nums[i-k]-Integer.MIN_VALUE)/((long)t+1);
        mp.remove(lastBucketId);
      }
      mp.put(bucketId,reposNum);
    }
    return false;
  }

  public boolean checkInclusion(String s1, String s2) {
    if (s2.length()<s1.length())
      return false;
    int[] record=new int[26],temp=null;
    int s1Len=s1.length(),start=-1,end;
    for (char c:s1.toCharArray())
      record[c-'a']++;
    char[] cs=s2.toCharArray();
    for (end=0;end<cs.length;end++){
      int index=cs[end]-'a';
      if (record[index]>0){
        if (start==-1){
          temp=new int[26];
          start=end;
        }
        temp[index]++;
        if (end-start+1>s1Len){
          temp[cs[start]-'a']--;
          start++;
        }
        if (end-start+1==s1Len && isArraysSame(record,temp))
          return true;
      }
      else
        start=-1;
    }
    return false;
  }

  class Solution519 {
    int row,col,total;
    Random r;
    Map<Integer,Integer> mp;
//    int[] convert;

    public Solution519(int n_rows, int n_cols) {
      r=new Random();
      row=n_rows;
      col=n_cols;
      total=row*col;
      mp=new HashMap<>();
//      convert=new int[total];
//      Arrays.fill(convert,-1);
    }

    public int[] flip() {
      int random=r.nextInt(total--);
      int real=mp.getOrDefault(random,random);
//      int real= convert[random]==-1?random:convert[random];
//      convert[random]=convert[total]==-1?total:convert[total];
      mp.put(random,mp.getOrDefault(total,total));
      return new int[]{real/col,real%col};
    }

    public void reset() {
      mp.clear();;
//      Arrays.fill(convert,-1);
      total=row*col;
    }
  }

  class Solution478 {
    private double R,X,Y;
    private Random r;
    public Solution478(double radius, double x_center, double y_center) {
      R=radius;
      X=x_center;
      Y=y_center;
      r=new Random();
    }

    public double[] randPoint() {
      double tempX,tempY;
      do {
        tempX=(r.nextDouble()-0.5)*2*R;
        tempY=(r.nextDouble()-0.5)*2*R;
      }while (tempX*tempX+tempY*tempY >R*R);
      return new double[]{tempX+X,tempY+Y};
    }
  }

  class Solution478_2 {
    private double R,X,Y;

    public Solution478_2(double radius, double x_center, double y_center) {
      R=radius;
      X=x_center;
      Y=y_center;
    }

    public double[] randPoint() {
      double len = Math.sqrt(Math.random())*R;
      double angle=Math.random()*Math.PI*2;
      return new double[]{len*Math.cos(angle)+X,len*Math.sin(angle)+Y};
    }
  }

  class Solution497 {
    private int total;
    private int[] area;
    private Random r;
    private int[][] rects;

    public Solution497(int[][] rects) {
      this.rects=rects;
      r=new Random();
      total=0;
      int N=rects.length;
      area=new int[N+1];
      for (int i=0;i<N;i++){
        int[] r=rects[i];
        int ar=(r[2]-r[0]+1)*(r[3]-r[1]+1);
        total+=ar;
        area[i+1]=total;
      }
    }

    public int[] pick() {
      int index=r.nextInt(total);
      int rectId=searchArea(index);
      int exact = index-area[rectId];
      int[] rect= rects[rectId];
      int col=rect[2]-rect[0]+1;
      return new int[]{exact%col+rect[0],exact/col+rect[1]};
    }

    private int searchArea(int i){
      int start=0,end=area.length;
      while (start<end){
        int mid=(start+end)>>1;
        if (area[mid]>i)
          end=mid;
        else if (area[mid]<i)
          start=mid+1;
        else
          return mid;
      }
      return end-1;
    }
  }

  private boolean isArraysSame(int[] a,int[] b){
    if (a.length!=b.length)
      return false;
    for (int i=0;i<a.length;i++)
      if (a[i]!=b[i])
        return false;
    return true;
  }

  public String removeDuplicates(String S) {
    if (S==null ||S.length()==0)
      return S;
    char[] cs=S.toCharArray();
    int N=cs.length,len=0;
    char[] stack=new char[N];
    for (int i=0;i<N;i++)
      if (len==0 || cs[i]!=stack[len-1])
        stack[len++]=cs[i];
      else
        len--;
    StringBuilder sb=new StringBuilder();
    for (int i=0;i<len;i++)
      sb.append(stack[i]);
    return sb.toString();
  }

  public int lengthOfLongestSubstring(String s) {
    if (s==null || s.length()==0)
      return 0;
    int[] count=new int[256];
    char[] cs=s.toCharArray();
    int N=cs.length,len=0,start,end;
    for (start=end=0;end<N;end++){
      count[cs[end]]++;
      if (count[cs[end]]>1){
        len=Math.max(len,end-start);
        while (cs[start]!=cs[end])
          count[cs[start++]]--;
        count[cs[start++]]--;
      }
    }
    len=Math.max(len,end-start);
    return len;
  }

  public int triangleNumber(int[] nums) {
    if (nums==null ||nums.length==0)
      return 0;
    Arrays.sort(nums);
    int count=0,N=nums.length;
    for (int i=N-1;i>=2;i--){
      int l=0,r=i-1;
      while (l<r){
        if (nums[l]+nums[r]>nums[i]){
          count+=r-l;
          r--;
        }
        else
          l++;
      }
    }
    return count;
  }

  public int maxUncrossedLines(int[] A, int[] B) {
    int alen=A.length,blen=B.length;
    int[][] dp=new int[alen+1][blen+1];
    for (int a=1;a<=alen;a++)
      for (int b=1;b<=blen;b++)
        if (A[a-1]!=B[b-1])
          dp[a][b]=Math.max(dp[a-1][b],dp[a][b-1]);
        else
          dp[a][b]=Math.max(Math.max(dp[a-1][b],dp[a][b-1]),dp[a-1][b-1]+1);
    return dp[alen][blen];
  }

  public void gameOfLife(int[][] board) {
    if (board==null || board.length==0 || board[0].length==0)
      return;
    int R=board.length,C=board[0].length;
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++){
        int lives=getNeighborLives(board,r,c);
        if (board[r][c]==1 && (lives==2 || lives ==3))
          board[r][c]=3;
        else if (board[r][c]==0 && lives == 3)
          board[r][c]=2;
      }
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        board[r][c]>>=1;
  }

  private int getNeighborLives(int[][] board,int r,int c){
    int R=board.length,C=board[0].length,lives=0;
    for (int i=Math.max(r-1,0);i<=Math.min(r+1,R-1);i++)
      for (int j=Math.max(0,c-1);j<=Math.min(c+1,C-1);j++)
        lives+=(board[i][j]&1);
    lives-=board[r][c]&1;
    return lives;
  }

  public int[] numMovesStonesII(int[] stones) {
    Arrays.sort(stones);
    int N=stones.length,l=0,r,high=0,low=Integer.MAX_VALUE;
    high=Math.max(stones[N-2]-stones[0]-1-(N-3),stones[N-1]-stones[1]-1-(N-3));
    for (r=0;r<N;r++){
      while (stones[r]-stones[l]>=N)
        l++;
      if (r-l+1 == N-1 && stones[r]-stones[l]==N-2)
        low=Math.min(low,2);
      else
        low=Math.min(low,N-(r-l+1));
    }
    return new int[]{low,high};
  }

  public List<Integer> getRow(int rowIndex) {
    List<Integer> ans=new ArrayList<>(rowIndex+1);
    if (rowIndex==0){
      ans.add(1);
      return ans;
    }
    int[] res=new int[rowIndex+1],cache=new int[rowIndex+1];
    res[0]=1;
    for (int i=1;i<=rowIndex;i++){
      cache[0]=1;
      for (int j=1;j<=i;j++)
        cache[j]=res[j]+res[j-1];
      int[] temp=res;
      res=cache;
      cache=temp;
    }
    for (int r:res)
      ans.add(r);
    return ans;
  }

  public int numMatchingSubseq(String S, String[] words) {
    int ans=0;
    Set<String> pass=new HashSet<>(),out=new HashSet<>();
    for (String w:words)
      if (pass.contains(w))
        ans++;
      else if (out.contains(w))
        continue;
      else if (NMSisSubSequence(S,w)){
        pass.add(w);
        ans++;
      }
      else
        out.add(w);
    return ans;
  }

  private boolean NMSisSubSequence(String S,String w){
    int prev=0;
    for (char c:w.toCharArray()){
      int index=S.indexOf(c,prev);
      if (index==-1)
        return false;
      prev=index+1;
    }
    return true;
  }

  public int partitionDisjoint(int[] A) {
    int N=A.length,min=Integer.MAX_VALUE,max=Integer.MIN_VALUE;
    int[] minLast=new int[N];
    for (int i=N-1;i>=0;i--){
      if (A[i]<min)
        min=A[i];
      minLast[i]=min;
    }
    for (int i=0;i<N-1;i++){
      if (A[i]>max)
        max=A[i];
      if (max<=minLast[i+1])
        return i+1;
    }
    return N;
  }

  public int numSubarrayBoundedMax(int[] A, int L, int R) {
    int N=A.length,ans=0,start,end,beforeRes=0;
    for (start=end=0;end<N;end++)
      if (A[end]>=L && A[end]<=R){
        int len=end-start+1;
        ans+=len;
        beforeRes=len;
      }
      else if (A[end]<L)
        ans+=beforeRes;
      else{
        start=end+1;
        beforeRes=0;
      }
    return ans;
  }

  public int minIncrementForUnique(int[] A) {
    if (A.length==0)
      return 0;
    Arrays.sort(A);
    Set<Integer> used=new HashSet<>();
    int ans=0,curMin=0;
    for (int a:A)
      if (!used.contains(a)){
        used.add(a);
        curMin=a+1;
      }
      else{
        ans+=curMin-a;
        used.add(curMin);
        curMin++;
      }
    return ans;
  }

  public int[] beautifulArray1(int N) {
    if (N==1)
      return new int[]{1};
    int[] res=new int[N],cache=new int[N];
    res[0]=1;
    for (int n=1;n<N;n<<=1){
      int index=0;
      for (int i=0;i<n;i++){
        int odd=(res[i]<<1)-1;
        if (odd<=N)
          cache[index++]=odd;
      }
      for (int i=0;i<n;i++){
        int even=res[i]<<1;
        if (even<=N)
          cache[index++]=even;
      }
      int[] temp=res;
      res=cache;
      cache=temp;
    }
    return res;
  }

  public int longestOnes2(int[] A, int K) {
    if (A==null ||A.length==0)
      return 0;
    int N=A.length,start,end,max=Integer.MIN_VALUE;
    for (start=end=0;end<N;end++)
      if (A[end]==0 && K >0)
        K--;
      else if (A[end] == 0){
        max=Math.max(max,end-start);
        while (A[start]==1)
          start++;
        start++;
      }
    max=Math.max(max,end-start);
    return max;
  }

  public int largestOverlap2(int[][] A, int[][] B) {
    int ans=0,R=A.length,C=A[0].length;
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++){
        ans=Math.max(ans,LOcheck(A,B,i,j));
        ans=Math.max(ans,LOcheck(B,A,i,j));
      }
    return ans;
  }

  private int LOcheck(int[][] A,int[][] B,int r,int c){
    int ans=0,R=A.length,C=A[0].length;
    for (int i=r;i<R;i++)
      for (int j=c;j<C;j++){
        int a=A[i-r][j-c],b=B[i][j];
        ans += (a&b);
      }
    return ans;
  }

  class FMXnode{
    int val;
    FMXnode[] nexts;
    public FMXnode(int v){
      val=v;
      nexts=new FMXnode[2];
    }
  }

  public void FMXaddNode(FMXnode root,int num,int index){
    if (index<0)
      return;
    int cur= (num & (1<<index))>0?1:0;
    if (root.nexts[cur]==null)
      root.nexts[cur]=new FMXnode(cur);
    FMXaddNode(root.nexts[cur],num,index-1);
  }

  public int findMaximumXOR2(int[] nums) {
    if (nums.length==1)
      return 0;
    FMXnode root=new FMXnode(0);
    for (int n:nums)
      FMXaddNode(root,n,31);
    if (root.nexts[0] !=null && root.nexts[1]!=null)
      return FMXcheck(root.nexts[0],root.nexts[1],0);
    else
      return Math.max(FMXcheck(root.nexts[0],root.nexts[0],0),FMXcheck(root.nexts[1],root.nexts[1],0));
  }

  private int FMXcheck(FMXnode r1,FMXnode r2,int before){
    if (r1==null || r2==null)
      return before;
    before = (before<<1)+ (r1.val ^ r2.val);
    int ans=0;
    ans=Math.max(ans,FMXcheck(r1.nexts[0],r2.nexts[1],before));
    ans=Math.max(ans,FMXcheck(r1.nexts[1],r2.nexts[0],before));
    if (ans!=before)
      return ans;
    ans=Math.max(ans,FMXcheck(r1.nexts[0],r2.nexts[0],before));
    ans=Math.max(ans,FMXcheck(r1.nexts[1],r2.nexts[1],before));
    return ans;
  }

  public List<Integer> postorderTraversal3(TreeNode root) {
    LinkedList<Integer> ans=new LinkedList<>();
    Stack<TreeNode> st=new Stack<>();
    TreeNode cur=root;
    while (cur!=null || !st.isEmpty())
      if (cur !=null){
        st.add(cur);
        ans.addFirst(cur.val);
        cur=cur.right;
      }
      else{
        TreeNode temp = st.pop();
        cur=temp.left;
      }
    return ans;
  }

  public List<Integer> postorderTraversal4(TreeNode root) {
    LinkedList<Integer> ans=new LinkedList<>();
    TreeNode cur=root;
    while (cur != null)
      if (cur.right==null){
        ans.addFirst(cur.val);
        cur=cur.left;
      }
      else{
        TreeNode before=cur.right;
        while (before.left != null && before.left!=cur)
          before=before.left;
        if (before.left==null){
          before.left=cur;
          ans.addFirst(cur.val);
          cur=cur.right;
        }
        else{
          before.left=null;
          cur=cur.left;
        }
      }
    return ans;
  }

  public int swimInWater2(int[][] grid) {
    int N=grid.length,start=0,end=N*N-1;
    while (start<end){
      int mid=(start+end)>>1;
      if (SWhelper(grid,new boolean[N][N],mid,0,0))
        end=mid;
      else
        start=mid+1;
    }
    return end;
  }

  private boolean SWhelper(int[][] grid,boolean[][] visited,int t,int r,int c){
    int N=grid.length;
    if (r<0 || r>=N || c<0 || c>=N || grid[r][c]>t || visited[r][c])
      return false;
    if (r==N-1 && c==N-1)
      return true;
    visited[r][c]=true;
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int[] d:dirs)
      if (SWhelper(grid,visited,t,r+d[0],c+d[1]))
        return true;
    return false;
  }

  public int[] nextGreaterElements2(int[] nums) {
    if (nums==null || nums.length==0)
      return nums;
    int N=nums.length,len=0;
    int[] stack=new int[N << 1],ans=new int[N];
    Arrays.fill(ans,-1);
    for (int i=0;i<(N << 1);i++){
      int index=i%N,curVal= nums[index];
      while (len>0 && nums[stack[len-1]] <curVal)
        ans[stack[--len]] = curVal;
      stack[len++]=index;
    }
    return ans;
  }

  public boolean reorderedPowerOf3(int N) {
    int key=RPcount(N);
    for (int i=0;i<32;i++)
      if (RPcount(1<<i)==key)
        return true;
    return false;
  }

  private int RPcount(int N){
    int key=0;
    while (N>0){
      key+= (int)Math.pow(10,N%10);
      N/=10;
    }
    return key;
  }

  public int minAreaRect2(int[][] P) {
    if (P==null || P.length<4)
      return 0;
    int min=Integer.MAX_VALUE;
    Map<Integer,Set<Integer>> rects=new HashMap<>();
    for (int[] p:P){
      if (!rects.containsKey(p[0]))
        rects.put(p[0],new HashSet<>());
      rects.get(p[0]).add(p[1]);
    }
    for (int[] p1:P)
      for (int[] p2:P){
        if (p1[0]==p2[0] || p1[1]==p2[1])
          continue;
        if (rects.get(p1[0]).contains(p2[1]) && rects.get(p2[0]).contains(p1[1]))
          min = Math.min(min,Math.abs(p1[0]-p2[0])*Math.abs(p1[1]-p2[1]));
      }
    return min==Integer.MAX_VALUE?0:min;
  }

  public int[] constructRectangle1(int area) {
    int start = (int)Math.sqrt(area);
    for (int i=start;i>1;i--)
      if (area%i==0)
        return new int[]{area/i,i};
    return new int[]{area,1};
  }

  public int fourSumCount1(int[] A, int[] B, int[] C, int[] D) {
    if (A==null||A.length==0)
      return 0;
    int ans=0;
    Map<Integer,Integer> ab=new HashMap<>();
    for (int a:A)
      for (int b:B)
        ab.put(a+b,ab.getOrDefault(a+b,0)+1);
    for (int c:C)
      for (int d:D){
        Integer val = c+d,res;
        if ((res=ab.get(-val))==null)
          continue;
        ans+=res;
      }
    return ans;
  }

  public String intToRoman1(int num) {
    String[] rm=new String[4000];
    rm[1]="I";
    rm[5]="V";
    rm[10]="X";
    rm[50]="L";
    rm[100]="C";
    rm[500]="D";
    rm[1000]="M";
    rm[4]="IV";
    rm[9]="IX";
    rm[40]="XL";
    rm[90]="XC";
    rm[400]="CD";
    rm[900]="CM";
    int[] st=new int[5];
    int len=0;
    StringBuilder sb=new StringBuilder();
    while (num>0){
      st[len]=(num%10)*(int)Math.pow(10,len);
      len++;
      num/=10;
    }
    while (len>0){
      Integer high=st[--len];
      String res;
      if ((res=rm[high])!=null){
        sb.append(res);
        continue;
      }
      int base=(int)Math.pow(10,len);
      if (high/base>5){
        high-=5*base;
        sb.append(rm[5*base]);
      }
      String bs=rm[base];
      while (high>0){
        sb.append(bs);
        high-=base;
      }
    }
    return sb.toString();
  }

  class Solution384 {

    private int[] data,cur;
    Random r;

    public Solution384(int[] nums) {
      data=nums;
      cur=Arrays.copyOf(data,data.length);
      r=new Random();
    }

    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
      return data;
    }

    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
      for (int i=cur.length-1;i>0;i--){
        int index=r.nextInt(i+1);
        exchange(cur,i,index);
      }
      return cur;
    }
  }

  public int maxProfit1(int[] prices, int fee) {
    if (prices==null || prices.length==0)
      return 0;
    int sell=0,buy=-prices[0]-fee;
    for (int i=0;i<prices.length;i++){
      buy=Math.max(buy,sell-prices[i]-fee);
      sell=Math.max(sell,buy+prices[i]);
    }
    return sell;
  }

  public double mincostToHireWorkers2(int[] quality, int[] wage, int K) {
    int N=quality.length;
    double res=Integer.MAX_VALUE,Ksum=0;
    double[][] ratio = new double[N][2];
    for (int i=0;i<N;i++)
      ratio[i]=new double[]{(double)wage[i]/(double)quality[i],(double)quality[i]};
    Arrays.sort(ratio, new Comparator<double[]>() {
      @Override
      public int compare(double[] a, double[] b) {
        return Double.compare(a[0],b[0]);
      }
    });
    PriorityQueue<Double> pq=new PriorityQueue<>();
    for (double[] r:ratio){
      Ksum+=r[1];
      pq.offer(-r[1]);
      if (pq.size()>K)
        Ksum+=pq.poll();
      if (pq.size()==K)
        res=Math.min(res,Ksum*r[0]);
    }
    return res;
  }

  public int flipLights1(int n, int m) {
    if (m==0)
      return 1;
    if (n==1 && m>=1)
      return 2;
    if (n==2 )
      if (m==1)
        return 3;
      else if (m>=2)
        return 4;
    if (m==1)
      return 4;
    if (m==2)
      return 7;
    return 8;
  }

  public int maxCoins1(int[] nums) {
    if (nums==null||nums.length==0)
      return 0;
    int n=nums.length,dataN=n+2;
    int[] data=new int[dataN];
    for (int i=1;i<=n;i++)
      data[i]=nums[i-1];
    data[0]=data[n+1]=1;
    int[][] dp=new int[dataN][dataN];
    for (int len=2;len<dataN;len++)
      for (int left=0;left<dataN-len;left++){
        int right=left+len;
        for (int pick=left+1;pick<right;pick++)
          dp[left][right]=Math.max(dp[left][right],data[left]*data[pick]*data[right]+dp[left][pick]+dp[pick][right]);
      }
    return dp[0][dataN-1];
  }

  public int heightChecker(int[] heights) {
    if (heights==null|| heights.length==0)
      return 0;
    int n=heights.length,ans=0;
    int[] copy=Arrays.copyOf(heights,n);
    Arrays.sort(copy);
    for (int i=0;i<n;i++)
      if (copy[i]!=heights[i])
        ans++;
    return ans;
  }

  public int[] prevPermOpt1(int[] A) {
    if (A==null ||A.length==0||A.length==1)
      return A;
    int n=A.length;
    for (int i=n-2;i>=0;i--)
      if (A[i]>A[i+1]){
        int swapId;
        for (swapId=n-1;swapId>i;swapId--)
          if (A[swapId]<A[i] && A[swapId]!=A[swapId-1])
            break;
        exchange(A,i,swapId);
        return A;
      }
    return A;
  }

  public int maxSatisfied(int[] customers, int[] grumpy, int X) {
    int ans = 0, changed1 = 0,c1, index = 0, n = customers.length;
    for (; index < X; index++)
      if (grumpy[index] == 1)
        changed1 += customers[index];
      else
        ans += customers[index];
    c1 = changed1;
    for (; index < n; index++) {
      if (grumpy[index] == 0)
        ans += customers[index];
      else
        changed1+=customers[index];
      int before=index-X;
      if (grumpy[before]==1)
        changed1-=customers[before];
      if (changed1>c1)
        c1=changed1;
    }
    return ans+c1;
  }

  public int subarraySum(int[] nums, int k) {
    if (nums==null ||nums.length==0)
      return 0;
    int ans=0,sum=0,n=nums.length;
    Map<Integer,Integer> preSumCount=new HashMap<>();
    preSumCount.put(0,1);
    for (int i=0;i<n;i++){
      sum+=nums[i];
      ans+=preSumCount.getOrDefault(sum-k,0);
      preSumCount.put(sum,preSumCount.getOrDefault(sum,0)+1);
    }
    return ans;
  }

  public int maxWidthRamp(int[] A) {
    int ans=0,n=A.length,len=0;
    int[] stack=new int[n];
    for (int i=0;i<n;i++)
      if (len==0||A[i]<A[stack[len-1]])
        stack[len++]=i;
    for (int i=n-1;i>ans;i--)
      while (len!=0 && A[i]>=A[stack[len-1]])
        ans=Math.max(ans,i-stack[--len]);
    return ans;
  }

  public int pivotIndex(int[] nums) {
    if (nums==null || nums.length==0)
      return -1;
    int n=nums.length,sum=0;
    int[] preSum=new int[n+1];
    for (int i=1;i<=n;i++){
      sum+=nums[i-1];
      preSum[i]=sum;
    }
    for (int i=1;i<=n;i++)
      if (preSum[n]-preSum[i]==preSum[i-1]-preSum[0])
        return i-1;
    return -1;
  }

  public int maxDistToClosest(int[] seats) {
    int curLen=0,dist=0,n=seats.length,start=0,end=n-1;
    for (;start<n && seats[start]==0;start++,dist++);
    for (;end>=0 && seats[end]==0;end--,curLen++);
    dist = Math.max(dist,curLen);
    curLen=0;
    for (int i=start+1;i<=end;i++)
      if (seats[i]==0)
        curLen++;
      else{
        int curDist = (curLen & 1)==0?curLen>>1:(curLen>>1)+1;
        dist = Math.max(dist,curDist);
        curLen=0;
      }
    return dist;
  }

  public int findPeakElement(int[] nums) {
    if (nums==null || nums.length==0)
      throw new IllegalArgumentException();
    if (nums.length==1)
      return 0;
    int n=nums.length,start=0,end=n-1;
    while (start<=end){
      int mid=(start+end)>>1;
      if (mid==0)
        if (nums[mid]>nums[mid+1])
          return mid;
        else
          start=mid+1;
      else if (mid==n-1)
        if (nums[mid]>nums[mid-1])
          return mid;
        else
          end = mid-1;
      else if ((nums[mid-1]<nums[mid] && nums[mid]<nums[mid+1]) || (nums[mid-1]>nums[mid] && nums[mid]<nums[mid+1]))
        start=mid+1;
      else if (nums[mid-1]>nums[mid] && nums[mid]>nums[mid+1])
        end=mid-1;
      else
        return mid;
    }
    return -1;
  }

  public String fractionToDecimal(int numerator, int denominator) {
    if (denominator==0)
      throw new IllegalArgumentException();
    if (numerator==0)
      return "0";
    StringBuilder sb=new StringBuilder();
    Map<Long,Integer> recorder=new HashMap<>();
    if ((numerator<0 && denominator>0) || (numerator>0 && denominator<0))
      sb.append("-");
    long n=Math.abs((long)numerator);
    long d=Math.abs((long)denominator);
    sb.append(n/d);
    if (n%d==0)
      return sb.toString();
    sb.append(".");
    n=n%d*10;
    Integer repeatStart;
    while (n!=0){
      if ((repeatStart=recorder.get(n))!=null){
        sb.append(')');
        sb.insert(repeatStart,"(");
        return sb.toString();
      }
      recorder.put(n,sb.length());
      long curVal=n/d,curNum=n%d*10;
      sb.append(curVal);
      n=curNum;
    }
    return sb.toString();
  }

  public int findMaxLength(int[] nums) {
    if (nums==null||nums.length==0)
      return 0;
    int n=nums.length,res=0,oneMinusZero=0;
    Map<Integer,Integer> counter=new HashMap<>();
    counter.put(0,-1);
    for (int i=0;i<n;i++){
      oneMinusZero=nums[i]==1?oneMinusZero+1:oneMinusZero-1;
      if (counter.containsKey(oneMinusZero))
        res=Math.max(res,i-counter.get(oneMinusZero));
      else
        counter.put(oneMinusZero,i);
    }
    return res;
  }

  public int searchInsert(int[] nums, int target) {
    if (nums==null || nums.length==0)
      return 0;
    int n=nums.length,start=0,end=n-1;
    while (start<=end){
      int mid=(start+end)>>1;
      if (nums[mid]>target)
        end=mid-1;
      else if (nums[mid]<target)
        start=mid+1;
      else
        return mid;
    }
    return start;
  }

  public int removeDuplicates(int[] N) {
    if (N==null)
      throw new IllegalArgumentException();
    if (N.length<3)
      return N.length;
    int count=2;
    for (int i=2;i<N.length;i++)
      if (N[i]!=N[count-2])
        N[count++]=N[i];
    return count;
  }

  public String getHint(String secret, String guess) {
    if (secret==null||guess==null||secret.isEmpty())
      return "0A0B";
    int A=0,B=0,n=secret.length();
    int[] sCount=new int[10];
    char[] cs=secret.toCharArray(),cg=guess.toCharArray();
    for (char c:cs)
      sCount[c-'0']++;
    for (int i=0;i<n;i++)
      if (cg[i]==cs[i]){
        A++;
        sCount[cs[i]-'0']--;
      }
    for (int i=0;i<n;i++)
      if (sCount[cg[i]-'0']>0 && cg[i]!=cs[i]){
         B++;
         sCount[cg[i]-'0']--;
    }
    return A+"A"+B+"B";
  }

  public int[] prisonAfterNDays(int[] cells, int N) {
    Integer n=cells.length,curKey,preIndex;
    Map<Integer,Integer> keyToIndex=new HashMap<>(),indexToKey=new HashMap<>();
    for (int i=1;i<=N;i++){
      PADnextStatus(cells);
      curKey=PADencode(cells);
      if ((preIndex=keyToIndex.get(curKey))!=null){
        int repLen=i-preIndex,remain=N-i,resKey=indexToKey.get(remain%repLen+preIndex);
        return PADdecode(resKey);
      }
      keyToIndex.put(curKey,i);
      indexToKey.put(i,curKey);
    }
    return cells;
  }

  private void PADnextStatus(int[] c){
    int n=c.length;
    for (int i=1;i<n-1;i++)
      if ((c[i-1] &1)==(c[i+1] &1))
        c[i] |= 2;
    for (int i=0;i<n;i++)
      c[i] >>=1;
  }

  private int PADencode(int[] c){
    int ans=0;
    for (int a:c){
      ans<<=1;
      ans|=a;
    }
    return ans;
  }

  private int[] PADdecode(int c){
    int[] ans=new int[8];
    for (int i=7;i>=0 && c!=0;i--){
      ans[i]=c&1;
      c>>=1;
    }
    return ans;
  }

  public ListNode partition(ListNode head, int x) {
    if (head==null || head.next==null)
      return head;
    ListNode smHead,smCur,ebHead,ebCur,cur=head;
    smHead=smCur=new ListNode(0);
    ebHead=ebCur=new ListNode(0);
    while (cur!=null){
      if (cur.val<x){
        smCur.next=cur;
        smCur=smCur.next;
      }
      else{
        ebCur.next=cur;
        ebCur=ebCur.next;
      }
      cur=cur.next;
    }
    smCur.next=ebHead.next;
    ebCur.next=null;
    return smHead.next;
  }

  public int removeDuplicates1(int[] nums) {
    if (nums==null || nums.length==0)
      return 0;
    if (nums.length==1)
      return nums.length;
    int n=nums.length,pos=1;
    for (int i=1;i<n;i++)
      if (nums[i]!=nums[pos-1])
        nums[pos++]=nums[i];
    return pos;
  }

  public boolean isPalindrome(ListNode head) {
    if (head==null || head.next==null)
      return true;
    ListNode fast,slow,cur;
    slow=fast=cur=head;
    boolean ans=true;
    while (fast.next!=null){
      fast=fast.next;
      slow=slow.next;
      if (fast.next==null)
        break;
      fast=fast.next;
    }
    IPreverse(slow);
    ListNode lastCur=fast;
    while (lastCur!=null && cur!=null){
      if (lastCur.val!=cur.val){
        ans=false;
        break;
      }
      lastCur=lastCur.next;
      cur=cur.next;
    }
    IPreverse(fast);
    return ans;
  }

  private void IPreverse(ListNode head){
    ListNode next,last=null;
    while (head!=null){
      next=head.next;
      head.next=last;
      last=head;
      head=next;
    }
  }

  public void setZeroes(int[][] matrix) {
    if (matrix==null||matrix.length==0 ||matrix[0].length==0)
      return;
    int R=matrix.length,C=matrix[0].length;
    boolean col0=false,row0=false;
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (matrix[r][c]==0){
          matrix[r][0]=matrix[0][c]=0;
          if (c==0)
            col0=true;
          if (r==0)
            row0=true;
        }
    for (int r=R-1;r>=1;r--)
      for (int c=C-1;c>=1;c--)
        if (matrix[r][0]==0 || matrix[0][c]==0 )
          matrix[r][c]=0;
    if (col0)
      for (int r=0;r<R;r++)
        matrix[r][0]=0;
    if (row0)
      for (int c=0;c<C;c++)
        matrix[0][c]=0;
  }

  public ListNode reverseBetween(ListNode head, int m, int n) {
    if (m==n)
      return head;
    ListNode mLast,nNext,cur=head,M,N;
    mLast=nNext=M=N=null;
    int count=1;
    while (count<=n){
      if (count==m-1)
        mLast=cur;
      else if (count==m)
        M=cur;
      else if (count==n)
        N=cur;
      cur=cur.next;
      count++;
    }
    nNext=cur;
    N.next=null;
    IPreverse(M);
    M.next=nNext;
    if (m==1)
      return N;
    mLast.next=N;
    return head;
  }

  public String originalDigits(String s) {
    if (s==null||s.isEmpty())
      return s;
    int[] C=new int[10];
    for (char c:s.toCharArray())
      if (c=='z')
        C[0]++;
      else if (c=='w')
        C[2]++;
      else if (c=='u')
        C[4]++;
      else if (c=='x')
        C[6]++;
      else if (c=='g')
        C[8]++;
      else if (c=='o')
        C[1]++;
      else if (c=='h')
        C[3]++;
      else if (c=='f')
        C[5]++;
      else if (c=='s')
        C[7]++;
      else if (c=='i')
        C[9]++;
    C[1]=C[1]-C[0]-C[2]-C[4];
    C[3]-=C[8];
    C[5]-=C[4];
    C[7]-=C[6];
    C[9]=C[9]-C[8]-C[6]-C[5];
    StringBuilder sb=new StringBuilder();
    for (int i=0;i<9;i++)
      for (int c=0;c<C[i];c++)
        sb.append(i);
    return sb.toString();
  }

  public int dominantIndex(int[] nums) {
    if (nums.length==1)
      return 0;
    int n=nums.length,first,second;
    if (nums[0]>=nums[1]){
      first=0;
      second=1;
    }
    else{
      first=1;
      second=0;
    }
    for (int i=2;i<n;i++){
      if (nums[i]<=nums[second])
        continue;
      if (nums[first]>=nums[i])
        second=i;
      else{
        second=first;
        first=i;
      }
    }
    return nums[first]>=nums[second]<<1?first:-1;
  }

  public int minimumTotal(List<List<Integer>> triangle) {
    if (triangle==null||triangle.size()==0)
      return 0;
    if (triangle.size()==1)
      return triangle.get(0).get(0);
    int n=triangle.size();
    Integer[] res=triangle.get(n-1).toArray(new Integer[0]),cache=new Integer[n];
    for (int i=n-2;i>=0;i--){
      for (int j=0;j<triangle.get(i).size();j++)
        cache[j]=Math.min(res[j],res[j+1])+triangle.get(i).get(j);
      Integer[] temp=cache;
      cache=res;
      res=temp;
    }
    return res[0];
  }

  public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
    Map<String,int[]> edgeToGradient=new HashMap<>();
    int[][] cord=new int[4][2];
    cord[0]=p1;cord[1]=p2;cord[2]=p3;cord[3]=p4;
    for (int i=0;i<3;i++)
      for (int j=i+1;j<4;j++){
        int centX=cord[i][0]+cord[j][0],centY=cord[i][1]+cord[j][1];
        int difX=cord[i][0]-cord[j][0],difY=cord[i][1]-cord[j][1],length=difX*difX+difY*difY;
        if (length==0)
          return false;
        int[] gradient=new int[]{difY,difX};
        String key=centX+" "+centY+" "+length;
        if (!edgeToGradient.containsKey(key))
          edgeToGradient.put(key,gradient);
        else{
          int[] grad1=edgeToGradient.get(key);
          if (grad1[0]*gradient[0]+grad1[1]*gradient[1]==0)
            return true;
        }
      }
    return false;
  }

  public int clumsy(int N) {
    if (N==1)
      return 1;
    if (N==2)
      return 2;
    if (N==3)
      return 6;
    if (N==4)
      return 7;
    int ans=0;
//    ans += (N-3+(N-3)%4)*((N-4)/4+1)>>1;
    ans+=N*(N-1)/(N-2)+(N-3);
    int index=N-4;
    for (;index>=3;index-=4)
      ans-=(index*(index-1)/(index-2)-(index-3));
    if (index==2)
      ans-=2;
    if (index==1)
      ans-=1;
    return ans;
  }

  public String pushDominoes(String dominoes) {
    if (dominoes==null || dominoes.length()==0)
      return dominoes;
    char[] cd=dominoes.toCharArray();
    int n=cd.length,lastR=-1,lastL=-1;
    int[] res=new int[n];
    for (int i=0;i<n;i++)
      if (cd[i]=='L')
        lastR=-1;
      else if (cd[i]=='R')
        lastR=i;
      else if (lastR!=-1)
        res[i]=i-lastR;
    for (int i=n-1;i>=0;i--)
      if (cd[i]=='R')
        lastL=-1;
      else if (cd[i]=='L')
        lastL=i;
      else if (res[i]==0)
        cd[i]=lastL==-1?'.':'L';
      else if (lastL==-1)
        cd[i]='R';
      else
        cd[i]=res[i]<lastL-i?'R':res[i]>lastL-i?'L':'.';
    return new String(cd);
  }

  public double findMaxAverage(int[] nums, int k) {
    int n=nums.length,max=Integer.MIN_VALUE;
    int[] ps=new int[n+1];
    for (int i=0;i<n;i++)
      ps[i+1]=ps[i]+nums[i];
    for (int i=k;i<=n;i++)
      max=Math.max(max,ps[i]-ps[i-k]);
    return (double)max/k;
  }

  public int maximumSwap(int num) {
    if (num==0)
      return 0;
    char[] digits=String.valueOf(num).toCharArray();
    int n=digits.length,sm=-1,bg=-1,next=0;
    int[] sort=new int[10],index=new int[1];
    index[0]=9;
    for (char d:digits)
      sort[d-'0']++;
    for (int i=0;i<n;i++)
      if (digits[i]!=(next=nextInSort(sort,index))+'0'){
        sm=digits[i]-'0';
        bg=next;
        digits[i]=(char)(next+'0');
        break;
      }
    if (bg==-1)
      return num;
    for (int i=n-1;i>=0;i--)
      if (digits[i]==bg+'0'){
        digits[i]=(char)(sm+'0');
        break;
      }
    return Integer.valueOf(new String(digits));
  }

  private int nextInSort(int[] sort,int[] index){
    while (sort[index[0]]==0)
      index[0]--;
    sort[index[0]]--;
    return index[0];
  }

  public int totalFruit(int[] tree) {
    if (tree==null || tree.length==0)
      return 0;
    int start,end,n=tree.length,max=0,curTypeNum=0;
    int[] types=new int[n];
    for (start=end=0;end<n;end++){
      if (types[tree[end]]==0)
        curTypeNum++;
      types[tree[end]]++;
      if (curTypeNum==3){
        max=Math.max(max,end-start);
        while (curTypeNum==3){
          types[tree[start]]--;
          if (types[tree[start]]==0)
            curTypeNum--;
          start++;
        }
      }
    }
    return Math.max(max,end-start);
  }

  public int numSubarraysWithSum(int[] A, int S) {
    if (A.length==0)
      return 0;
    int[] count=new int[A.length+1];
    int ans=0,ps=0;
    count[0]=1;
    for (int a:A){
      ps+=a;
      ans+=ps-S>=0?count[ps-S]:0;
      count[ps]++;
    }
    return ans;
  }

  public String maskPII(String S) {
    char[] cs=S.toCharArray();
    int n=cs.length;
    StringBuilder sb=new StringBuilder();
    if (Character.isLetter(cs[0])){
      //email
      int atIndex=0;
      while (cs[atIndex]!='@')
        atIndex++;
      sb.append(upperToLower(cs[0]));
      sb.append("*****");
      sb.append(upperToLower(cs[atIndex-1]));
      for (;atIndex<n;atIndex++)
        sb.append(upperToLower(cs[atIndex]));
    }
    else{
      //phone
      for (char c:cs)
        if (Character.isDigit(c))
          sb.append(c);
      int len=sb.length();
      if (len==10){
        for (int i=0;i<6;i++)
          sb.setCharAt(i,'*');
        sb.insert(3,'-');
        sb.insert(7,'-');
      }
      else{
        for (int i=0;i<len-4;i++)
          sb.setCharAt(i,'*');
        sb.insert(0,'+');
        sb.insert(len-9,'-');
        sb.insert(len-5,'-');
        sb.insert(len-1,'-');
      }
    }
    return sb.toString();
  }

  private char upperToLower(char c){
    return c>=65 && c<=90 ?(char)(c+32):c;
  }

  public void merge(int[] nums1, int m, int[] nums2, int n) {
    int len1=nums1.length;
    for (int i=m-1,j=len1-1;i>=0;i--,j--)
      nums1[j]=nums1[i];
    for (int i=0,s1=len1-m,s2=0;i<len1;i++)
      if (s1==len1)
        nums1[i]=nums2[s2++];
      else if (s2==n)
        nums1[i]=nums1[s1++];
      else if (nums1[s1]<=nums2[s2])
        nums1[i]=nums1[s1++];
      else
        nums1[i]=nums2[s2++];
  }

  public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
    int taskN=difficulty.length,curMax=0,ans=0;
    int[][] income=new int[taskN][2];
    for (int i=0;i<taskN;i++)
      income[i]=new int[]{difficulty[i],profit[i]};
    Arrays.sort(income,(a,b)->a[0]-b[0]);
    Arrays.sort(worker);
    for (int i=0;i<taskN;i++){
      curMax=Math.max(curMax,income[i][1]);
      income[i][1]=curMax;
    }
//    for (int ability:worker)
//      ans+=MPAbs(income,ability);
    for (int i=0,job=0;i<worker.length;i++){
      while (job< taskN && income[job][0]<=worker[i])
        job++;
      ans+=job==0?0:job==taskN?income[taskN-1][1]:income[job-1][1];
    }
    return ans;
  }

  private int MPAbs(int[][] income,int ability){
    int start=0,end=income.length-1;
    while (start<=end){
      int mid=(start+end)>>1;
      if (income[mid][0]<=ability)
        start=mid+1;
      else
        end=mid-1;
    }
    return end==-1?0:income[end][1];
  }

  public int expressiveWords(String S, String[] words) {
    if (S.length()==0 || words.length==0)
      return 0;
    int ans=0;
    for (String w:words)
      if (EWisMatched(S,w))
        ans++;
    return ans;
  }

  private boolean EWisMatched(String S,String word){
    int sl,sr,wl,wr;
    char[] cs=S.toCharArray(),cw=word.toCharArray();
    sl=sr=wl=wr=0;
    while (sl<cs.length && wl<cw.length){
      if (cs[sl]!=cw[wl])
        return false;
      while (sr<cs.length && cs[sr]==cs[sl])
        sr++;
      while (wr<cw.length && cw[wr]==cw[wl])
        wr++;
      int slen=sr-sl,wlen=wr-wl;
      if (wlen>slen || (wlen<slen && slen <3))
        return false;
      sl=sr;
      wl=wr;
    }
    return sl==cs.length && wl==cw.length;
  }

  public String shiftingLetters(String S, int[] shifts) {
    int turns=0,n=shifts.length;
    char[] cs=S.toCharArray();
    for (int i=n-1;i>=0;i--){
      turns+= shifts[i];
      turns %= 26;
      cs[i]=(char)((cs[i]-'a'+turns)%26+'a');
    }
    return new String(cs);
  }

  public List<Integer> findClosestElements1(int[] arr, int k, int x) {
    List<Integer> ans=new ArrayList<>(k);
    if (k==0)
      return ans;
    int n=arr.length,index=FCEindexOf(arr,x),left=index-1,right=index+1;
    ans.add(arr[index]);
    k--;
    while (k-->0)
      if (right>=n)
        ans.add(arr[left--]);
      else if (left<0)
        ans.add(arr[right++]);
      else if (x-arr[left]<=arr[right]-x)
        ans.add(arr[left--]);
      else
        ans.add(arr[right++]);
    Collections.sort(ans);
    return ans;
  }

  private int FCEindexOf(int[] arr,int x){
    if (arr[0]>=x)
      return 0;
    if (arr[arr.length-1]<=x)
      return arr.length-1;
    int start=0,end=arr.length-1;
    while (start<=end){
      int mid=(start+end)>>1;
      if (arr[mid]>=x)
        end=mid-1;
      else
        start=mid+1;
    }
    return end;
  }

  public List<Integer> findClosestElements(int[] arr, int k, int x) {
    List<Integer> ans = new ArrayList<>(k);
    if (k == 0)
      return ans;
    int n=arr.length,left=0,right=n-k;
    while (left<right){
      int mid=(left+right)>>1;
      if (x-arr[mid]>arr[mid+k]-x)
        left=mid+1;
      else
        right=mid;
    }
    for (int i=0;i<k;i++)
      ans.add(arr[left+i]);
    return ans;
  }

  public boolean isPerfectSquare(int num) {
    if (num<0)
      return false;
    if (num<2)
      return true;
    long start=2,end=num>>1;
    while (start<=end){
      long mid=(start+end)>>1,val=mid*mid;
      if (val<num)
        start=mid+1;
      else if (val>num)
        end=mid-1;
      else
        return true;
    }
    return false;
  }

  public int[] findRightInterval(int[][] intervals) {
    if (intervals==null ||intervals.length==0)
      return new int[0];
    int n=intervals.length;
    int[] ans=new int[n];
    int[][] itv=new int[n][3];
    for (int i=0;i<n;i++)
      itv[i]=new int[]{intervals[i][0],intervals[i][1],i};
    Arrays.sort(itv, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]-b[0];
      }
    });
    for (int i=0;i<n;i++)
      ans[itv[i][2]]=FRIfindMinRight(itv,i,itv[i][1]);
    return ans;
  }

  private int FRIfindMinRight(int[][] itv,int index,int val){
    int start=index+1,end=itv.length-1;
    while (start<=end){
      int mid=(start+end)>>1,s=itv[mid][0];
      if (s>val)
        end=mid-1;
      else if (s<val)
        start=mid+1;
      else
        return itv[mid][2];
    }
    return start<itv.length?itv[start][2]:-1;
  }

  public int hIndex2(int[] citations) {
    if (citations==null ||citations.length==0)
      return 0;
    if (citations.length==1)
      return citations[0]>=1?1:0;
    int n=citations.length,start=0,end=n;
    while (start<=end){
      int mid=(start+end)>>1,bigger=HIbiggerCount(citations,mid);
      if (bigger<mid)
        end=mid-1;
      else
        start=mid+1;
    }
    return end;
  }

  private int HIbiggerCount(int[] C,int h){
    int n=C.length,start=0,end=n-1;
    while (start<=end){
      int mid=(start+end)>>1;
      if (C[mid]<h)
        start=mid+1;
      else
        end=mid-1;
    }
    return n-start;
  }

  public int longestStrChain(String[] words) {
    if (words.length==1)
      return 1;
    int n=words.length,max=1;
    int[] chainCount=new int[n];
    Map<String,Integer> wordToIndex=new HashMap<>();
    for (int i=0;i<n;i++)
      wordToIndex.put(words[i],i);
    for (int i=0;i<n;i++)
      if (chainCount[i]==0){
        chainCount[i]=LSCcountChain(chainCount,words[i],i,wordToIndex);
        max=Math.max(max,chainCount[i]);
      }
    return max;
  }

  private int LSCcountChain(int[] chainCount,String cur,int index,Map<String,Integer> wordToIndex){
    if (chainCount[index]!=0)
      return chainCount[index];
    Integer depth=0,next;
    String nextWord;
    StringBuilder sb=new StringBuilder(cur);
    for (int i=0;i<cur.length();i++){
      char remove=sb.charAt(i);
      sb.deleteCharAt(i);
      nextWord=sb.toString();
      if ((next=wordToIndex.get(nextWord))!=null)
        depth=Math.max(depth,LSCcountChain(chainCount,nextWord,next,wordToIndex));
      sb.insert(i,remove);
    }
    chainCount[index]=depth+1;
    return chainCount[index];
  }

  public boolean isPalindrome(String s) {
    if (s==null || s.length()==0)
      return true;
    char[] cs=s.toCharArray();
    int start=0,end=cs.length-1;
    while (start<end){
      if (!isDigit(cs[start]) && !isLetter(cs[start])){
        start++;
        continue;
      }
      if (!isDigit(cs[end]) && !isLetter(cs[end])){
        end--;
        continue;
      }
      char v1=isDigit(cs[start])?cs[start]:toLowerLetter(cs[start]);
      char v2=isDigit(cs[end])?cs[end]:toLowerLetter(cs[end]);
      if (v1!=v2)
        return false;
      start++;
      end--;
    }
    return true;
  }

  private boolean isDigit(char c){
    return c>=48 && c<=57;
  }

  private boolean isLetter(char c){
    return (c>=65 && c<=90) || (c>=97 && c<=122);
  }

  private char toLowerLetter(char c){
    return (c>=65 && c<=90)?(char)(c+32):c;
  }

  class NumArray2 {
    private int[] preSum;
    public NumArray2(int[] nums) {
      int sum=0,n=nums.length;
      preSum=new int[n+1];
      for (int i=0;i<n;i++){
        sum+=nums[i];
        preSum[i+1]=sum;
      }
    }

    public int sumRange(int i, int j) {
      int n=preSum.length;
      if (i<0 || i>=n ||j<0 || j>=n )
        throw new IllegalArgumentException();
      return preSum[j+1]-preSum[i];
    }
  }

  public int combinationSum4(int[] nums, int target) {
    if (nums==null || nums.length==0 ||target<0)
      return 0;
    int[] dp=new int[target+1];
    Arrays.fill(dp,-1);
    return CSfindComb(dp,nums,target);
  }

  private int CSfindComb(int[] dp,int[] nums,int remain){
    if (remain<0)
      return 0;
    if (remain==0)
      return 1;
    if (dp[remain]!=-1)
      return dp[remain];
    int sum=0;
    for (int n:nums)
      sum+=CSfindComb(dp,nums,remain-n);
    dp[remain]=sum;
    return dp[remain];
  }

  public int orderOfLargestPlusSign(int N, int[][] mines) {
    int[][] grid=new int[N][N];
    for (int i=0;i<N;i++)
      Arrays.fill(grid[i],N);
    for (int[] m:mines)
      grid[m[0]][m[1]]=0;
    for (int i=0;i<N;i++)
      for (int j=0,k=N-1,l=0,r=0,t=0,d=0;j<N;j++,k--){
        grid[i][j]=Math.min(grid[i][j],l=grid[i][j]==0?0:l+1);
        grid[i][k]=Math.min(grid[i][k],r=grid[i][k]==0?0:r+1);
        grid[j][i]=Math.min(grid[j][i],t=grid[j][i]==0?0:t+1);
        grid[k][i]=Math.min(grid[k][i],d=grid[k][i]==0?0:d+1);
      }
    int max=0;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        max=Math.max(max,grid[i][j]);
    return max;
  }

  public double knightProbability(int N, int K, int r, int c) {
    if (K==0)
      return 1;
    if (N <= 2)
      return 0;
    double[][][] dp = new double[K + 1][N][N];
    for (int i = 0; i < N; i++)
      Arrays.fill(dp[K][i], 1);
    return KPhelper(dp,0,r,c,K);
  }

  private double KPhelper(double[][][] dp,int k,int r,int c,int K){
    if (k==K)
      return dp[k][r][c];
    if (dp[k][r][c]!=0)
      return dp[k][r][c];
    int[][] dirs=new int[][]{{1,2},{1,-2},{2,1},{2,-1},{-1,2},{-1,-2},{-2,1},{-2,-1}};
    double sum=0;
    int N=dp[0].length;
    for (int[] d:dirs){
      int nextR=r+d[0],nextC=c+d[1];
      if (nextR>=0 && nextR<N && nextC>=0 && nextC<N)
        sum+= KPhelper(dp,k+1,nextR,nextC,K)/8;
    }
    dp[k][r][c]=sum;
    return dp[k][r][c];
  }

  public ListNode mergeKLists1(ListNode[] lists) {
    if (lists==null||lists.length==0)
      return null;
    if (lists.length==1)
      return lists[0];
    ListNode head=new ListNode(0),cur=head;
    PriorityQueue<ListNode> pq=new PriorityQueue<>((a,b)->a.val-b.val);
    for (ListNode l:lists)
      if (l!=null)
        pq.offer(l);
    while (!pq.isEmpty()){
      ListNode temp=pq.poll();
      cur.next=temp;
      cur=cur.next;
      if (temp.next!=null)
        pq.offer(temp.next);
    }
    return head.next;
  }

  public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
      return null;
    if (lists.length == 1)
      return lists[0];
    return MKhelper(lists,0,lists.length-1);
  }

  private ListNode MKhelper(ListNode[] lists,int start,int end){
    if (start==end)
      return lists[start];
    int mid=(start+end)>>1;
    ListNode left=MKhelper(lists,start,mid),right=MKhelper(lists,mid+1,end);
    return MKmerge(left,right);
  }

  private ListNode MKmerge(ListNode left,ListNode right){
    ListNode head=new ListNode(0),cur=head;
    while (left!=null || right!=null)
      if (left==null){
        cur.next=right;
        cur=cur.next;
        right=right.next;
      }
      else if (right==null){
        cur.next=left;
        cur=cur.next;
        left=left.next;
      }
      else if (left.val<=right.val){
        cur.next=left;
        cur=cur.next;
        left=left.next;
      }
      else{
        cur.next=right;
        cur=cur.next;
        right=right.next;
      }
    return head.next;
  }

  public int numMagicSquaresInside(int[][] grid) {
    if (grid==null)
      return 0;
    int R=grid.length,C=grid[0].length,ans=0;
    if (R < 3 || C <3)
      return 0;
    for (int r=0;r<=R-3;r++)
      for (int c=0;c<=C-3;c++)
        if (NMScheck(grid,r,c))
          ans++;
    return ans;
  }


  private boolean NMScheck(int[][] grid,int r,int c){
    int[] count=new int[10];
    for (int i=r;i<r+3;i++)
      for (int j=c;j<c+3;j++)
        if (grid[i][j]<=0 || grid[i][j] >9)
          return false;
        else if (count[grid[i][j]]==1)
          return false;
        else
          count[grid[i][j]]++;
    int val=grid[r][c]+grid[r+1][c+1]+grid[r+2][c+2];
    if (grid[r+2][c]+grid[r+1][c+1]+grid[r][c+2]!=val)
      return false;
    for (int i=0;i<3;i++){
      if (grid[r][c+i]+grid[r+1][c+i]+grid[r+2][c+i]!=val)
        return false;
      if (grid[r+i][c]+grid[r+i][c+1]+grid[r+i][c+2]!=val)
        return false;
    }
    return true;
  }

  public String getPermutation(int n, int k) {
    if (n==1)
      return "1";
    StringBuilder sb=new StringBuilder();
    LinkedList<Integer> data=new LinkedList<>();
    int[] fact=new int[n+1];
    int f=1;
    for (int i=1;i<=n;i++){
      data.add(i);
      f*=i;
      fact[i]=f;
    }
    while (n>1){
      int pickId=(k-1)/fact[n-1];
      sb.append(data.get(pickId));
      data.remove(pickId);
      k-=fact[--n]*pickId;
    }
    sb.append(data.get(0));
    return sb.toString();
  }

  public int maxProfit3(int[] prices) {
    if (prices==null || prices.length<2)
      return 0;
    int n=prices.length;
    int[] buy=new int[n],sell=new int[n];
    buy[0]=-prices[0];
    sell[0]=0;
    buy[1]=Math.max(-prices[1],buy[0]);
    sell[1]=Math.max(buy[0]+prices[1],sell[0]);
    for (int i=2;i<n;i++){
      buy[i]=Math.max(-prices[i]+sell[i-2],buy[i-1]);
      sell[i]=Math.max(sell[i-1],buy[i-1]+prices[i]);
    }
    return sell[n-1];
  }

  public int numTilePossibilities(String tiles) {
    if (tiles==null || tiles.isEmpty())
      return 0;
    if (tiles.length()==1)
      return 1;
    int[] count=new int[26];
    for (char c:tiles.toCharArray())
      count[c-'A']++;
    int[] res=new int[1];
    NTPhelper(count,tiles.length(),res);
    return res[0];
  }

  private void NTPhelper(int[] count,int remain,int[] res){
    if (remain==0)
      return;
    for (int i=0;i<26;i++)
      if (count[i]!=0){
        count[i]--;
        res[0]++;
        NTPhelper(count,remain-1,res);
        count[i]++;
      }
  }

  public double largestSumOfAverages(int[] A, int K) {
    if (A==null || A.length==0)
      return 0;
    int n=A.length;
    double[] ps=new double[n+1];
    double[][] dp=new double[K][n];
    for (int i=0;i<n;i++){
      ps[i+1]=ps[i]+A[i];
      dp[0][i]=ps[i+1]/(i+1);
    }
    for (int k=1;k<K;k++)
      for (int i=0;i<n;i++){
        double max=Double.MIN_VALUE;
        for (int p=0;p<i;p++)
          max=Math.max(max,dp[k-1][p]+(ps[i+1]-ps[p+1])/(i-p));
        dp[k][i]=max;
      }
    return dp[K-1][n-1];
  }

  public int minScoreTriangulation(int[] A) {
    int n=A.length;
    int[][] dp=new int[n][n];
    for (int d=2;d<n;d++)
      for (int i=0;i+d<n;i++){
        int j=i+d;
        dp[i][j]=Integer.MAX_VALUE;
        for (int k=i+1;k<j;k++)
          dp[i][j]=Math.min(dp[i][j],dp[i][k]+dp[k][j]+A[i]*A[j]*A[k]);
      }
    return dp[0][n-1];
  }

  public int knightDialer(int N) {
    if (N==1)
      return 10;
    int ans=0,mod=(int)Math.pow(10,9)+7;
    int[][] dp=new int[N][10],hop=new int[][]{{4,6},{6,8},{7,9},{4,8},{0,3,9},{},{0,1,7},{2,6},{1,3},{2,4}};
    Arrays.fill(dp[0],1);
    for (int n=1;n<N;n++)
      for (int i=0;i<=9;i++)
        for (int d:hop[i]){
          dp[n][i]+=dp[n-1][d];
          dp[n][i]%=mod;
        }
    for (int i=0;i<=9;i++){
      ans+=dp[N-1][i];
      ans%=mod;
    }
    return ans;
  }

  public List<String> restoreIpAddresses(String s) {
    List<String> res=new ArrayList<>();
    if (s==null || s.length()<4 || s.length()>12)
      return res;
    RIAhelper(s.toCharArray(),3,0,new int[3],res);
    return res;
  }

  private void RIAhelper(char[] cs,int remainDot,int start,int[] dotIndex,List<String> res){
    if (remainDot==0){
      if (start>cs.length -1 || start < cs.length-3 ||(cs[start]=='0' && start!=cs.length-1)|| RIAgetNum(cs,start)>=256)
        return;
      StringBuilder sb=new StringBuilder(new String(cs));
      for (int i=0;i<3;i++)
        sb.insert(dotIndex[i]+i,'.');
      res.add(sb.toString());
      return;
    }
    if (cs[start]=='0'){
      dotIndex[3-remainDot]=start+1;
      RIAhelper(cs,remainDot-1,start+1,dotIndex,res);
      return;
    }
    int n=cs.length,val=0;
    for (int i=start;i<=n-remainDot;i++){
      val=val*10+(cs[i]-'0');
      if (val>=256)
        break;
      dotIndex[3-remainDot] = i+1;
      RIAhelper(cs,remainDot-1,i+1,dotIndex,res);
    }
  }

  private int RIAgetNum(char[] cs,int start){
    int sum=0;
    for (int i=start;i<cs.length;i++)
      sum=sum*10+(cs[i]-'0');
    return sum;
  }

  public boolean exist(char[][] board, String word) {
    if (board==null || board.length==0 ||board[0].length==0 ||word==null ||word.isEmpty())
      return false;
    int R=board.length,C=board[0].length;
    int[][] dirs={{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    char first=word.charAt(0);
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (board[r][c]==first && existHelper(board,r,c,word,0,dirs))
          return true;
    return false;
  }

  public int[] asteroidCollision(int[] A) {
    if (A==null || A.length<2)
      return A;
    int n=A.length,len=0;
    int[] stack = new int[n];
    for (int a:A)
      if (len==0 || a>0 || stack[len-1]<0)
        stack[len++]=a;
      else {
        while (len>0 && stack[len-1]>0 && a+stack[len-1]<0)
          len--;
        if (len==0 || stack[len-1]<0)
          stack[len++]=a;
        else if (a+stack[len-1]==0)
          len--;
      }
    int[] ans=Arrays.copyOf(stack,len);
    return ans;
  }

  private boolean existHelper(char[][] board,int r,int c,String word,int depth,int[][] dirs){
    if (depth==word.length())
      return true;
    int R=board.length,C=board[0].length;
    if (r<0 || r>=R || c<0 ||c>=C|| word.charAt(depth)!=board[r][c])
      return false;
    char cur=board[r][c];
    board[r][c]=' ';
    for (int[] d:dirs){
      int nr=r+d[0],nc=c+d[1];
      if (existHelper(board,nr,nc,word,depth+1,dirs))
        return true;
    }
    board[r][c]=cur;
    return false;
  }

  public int lastStoneWeight(int[] S) {
    if (S.length<2)
      return S.length;
    PriorityQueue<Integer> pq=new PriorityQueue<>(Comparator.reverseOrder());
    for (int s:S)
      pq.offer(s);
    while (pq.size()>1)
      pq.offer(pq.poll()-pq.poll());
    return pq.isEmpty()?0:pq.peek();
  }

  public List<Integer> largestDivisibleSubset(int[] nums) {
    List<Integer> ans=new ArrayList<>();
    if (nums==null)
      return ans;
    if (nums.length<2){
      for (int n:nums)
        ans.add(n);
      return ans;
    }
    Arrays.sort(nums);
    int N=nums.length,maxCount=0,maxHead=0;
    int[] count=new int[N],pre=new int[N];
    for (int i=0;i<N;i++){
      count[i]=1;
      pre[i]=-1;
      for (int j=i-1;j>=0;j--){
        if (nums[i]%nums[j]!=0)
          continue;
        if (count[j]>=count[i]){
          count[i]=count[j]+1;
          pre[i]=j;
        }
      }
      if (maxCount<count[i]){
        maxCount=count[i];
        maxHead=i;
      }
    }
    while (maxHead!=-1){
      ans.add(nums[maxHead]);
      maxHead=pre[maxHead];
    }
    return ans;
  }

  public int evalRPN(String[] T) {
    if (T==null || T.length==0)
      return 0;
    int n=T.length,len=0;
    int[] st=new int[n];
    for (String t:T)
      if (isDigit(t.charAt(t.length()-1)))
        st[len++]=Integer.valueOf(t);
      else{
        int v2=st[--len],v1=st[--len],res=0;
        char sign=t.charAt(0);
        if (sign=='+')
          res=v1+v2;
        else if (sign=='-')
          res=v1-v2;
        else if (sign=='*')
          res=v1*v2;
        else
          res=v1/v2;
        st[len++]=res;
      }
    return st[0];
  }

  public NestedInteger deserialize(String s) {
    if (s.charAt(0)!='[')
      return new NestedInteger(DnextInt(s.toCharArray(),new int[1]));
    else
      return Dhelper(s.toCharArray(),new int[]{1});
  }

  private NestedInteger Dhelper(char[] cs, int[] index){
    int n=cs.length,len=0;
    NestedInteger ans=new NestedInteger();
    NestedInteger[] stack=new NestedInteger[n];
    Integer nextVal;
    while (cs[index[0]]!=']'){
      nextVal=DnextInt(cs,index);
      if (nextVal!=null)
        stack[len++]=new NestedInteger(nextVal);
      else{
        index[0]++;
        stack[len++]=Dhelper(cs,index);
      }
      if (cs[index[0]]==',')
        index[0]++;
    }
    for (int i=0;i<len;i++)
      ans.add(stack[i]);
    index[0]++;
    return ans;
  }

  private Integer DnextInt(char[] cs,int[] index){
    if (cs[index[0]]=='[')
      return null;
    Integer ans=0,N=cs.length;
    boolean isNeg=false;
    if (cs[index[0]]=='-'){
      isNeg=true;
      index[0]++;
    }
    while (index[0]<N && isDigit(cs[index[0]])){
      ans=ans*10+(cs[index[0]]-'0');
      index[0]++;
    }
    return isNeg?-ans:ans;
  }

  public int[] rearrangeBarcodes(int[] B) {
    if (B == null || B.length == 0)
      return B;
    int n=B.length,max = Integer.MIN_VALUE,index=0,maxCount=0;
    for (int b : B)
      max = Math.max(max, b);
    int[] count = new int[max +1];
    for (int b : B){
      count[b]++;
      if (count[b]>count[maxCount])
        maxCount=b;
    }
    while (count[maxCount]>0){
      B[index]=maxCount;
      index+=2;
      count[maxCount]--;
    }
    for (int i=1;i<=max;i++)
      for (int j=0;j<count[i];j++){
        if (index>=n)
          index=1;
        B[index]=i;
        index+=2;
      }
    return B;
  }

  public int reverseBits(int n) {
    if (n==0 || n==-1)
      return n;
    int ans=0;
    for (int i=0;i<32;i++){
      ans<<=1;
      int cur=n&1;
      ans|=cur;
      n>>>=1;
    }
    return ans;
  }

  class Twitter {

    class User{
      int id;
      Set<Integer> followee;
      Tweet head;

      public User(int id){
        this.id=id;
        this.followee=new HashSet<>();
        this.head=null;
        follow(id);
      }

      public void follow(int followeeId){
        followee.add(followeeId);
      }

      public void unfollow(int followeeId){
        followee.remove(followeeId);
      }

      public void post(int tweetId){
        Tweet tweet=new Tweet(tweetId);
        tweet.next=head;
        head=tweet;
      }
    }

    class Tweet{
      int id;
      int time;
      Tweet next;

      public Tweet(int id){
        this.id=id;
        this.time=timeStamp++;
        this.next=null;
      }
    }

    private Map<Integer,User> users;
    private int timeStamp;
    /** Initialize your data structure here. */
    public Twitter() {
      users=new HashMap<>();
      timeStamp=0;
    }

    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
      User user;
      if ((user=users.get(userId))==null){
        user=new User(userId);
        users.put(userId,user);
      }
      user.post(tweetId);
    }

    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
      User user;
      List<Integer> ans=new ArrayList<>();
      if ((user=users.get(userId))==null)
        return ans;
      Set<Integer> followee=user.followee;
      PriorityQueue<Tweet> pq=new PriorityQueue<>(users.size(),new Comparator<Tweet>() {
        @Override
        public int compare(Tweet a, Tweet b) {
          return b.time-a.time;
        }
      });
      for (int followeeId:followee){
        User f;
        if ((f=users.get(followeeId))==null || f.head==null)
          continue;
        pq.offer(f.head);
      }
      while (!pq.isEmpty() && ans.size()<10){
        Tweet recent=pq.poll();
        ans.add(recent.id);
        if (recent.next!=null)
          pq.offer(recent.next);
      }
      return ans;
    }

    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
      User follower;
      if ((follower=users.get(followerId))==null){
        follower=new User(followerId);
        users.put(followerId,follower);
      }
      if (!users.containsKey(followeeId))
        users.put(followeeId,new User(followeeId));
      follower.follow(followeeId);
    }

    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
      User follower;
      if ((follower=users.get(followerId))==null || followeeId==followerId)
        return;
      follower.unfollow(followeeId);
    }
  }

  public boolean hasPathSum(TreeNode root, int sum) {
    if (root==null)
      return false;
    if (root.left==null && root.right==null && sum==root.val)
      return true;
    return hasPathSum(root.left,sum-root.val) || hasPathSum(root.right,sum-root.val);
  }

  public int largestValsFromLabels(int[] values, int[] labels, int num_wanted, int use_limit) {
    int[] forbid=new int[20001];
    int n=values.length,ans=0,num=0;
    int[][] vl=new int[n][2];
    for (int i=0;i<n;i++){
      vl[i][0]=values[i];
      vl[i][1]=labels[i];
    }
    Arrays.sort(vl, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return b[0]-a[0];
      }
    });
    for (int i=0;i<n && num<num_wanted;i++){
      if (forbid[vl[i][1]]++>=use_limit)
        continue;
      num++;
      ans+=vl[i][0];
    }
    return ans;
  }

  public int integerReplacement(int n) {
    if (n==1)
      return 0;
    int ans=0;
    while (n!=1){
      if ((n&1)==0)
        n>>=1;
      else if (n==3 || (n&2)==0)
        n--;
      else
        n++;
      ans++;
    }
    return ans;
  }

  public int robotSim(int[] commands, int[][] obstacles) {
    if (commands==null || commands.length==0)
      return 0;
    Set<String> ob=new HashSet<>();
    for (int[] o:obstacles)
      ob.add(o[0]+" "+o[1]);
    int[][] dirs=new int[][]{{0,1},{1,0},{0,-1},{-1,0}};
    int dir=0,ans=0;
    int[] pos=new int[]{0,0};
    for (int c:commands)
      if (c==-1)
        dir=(dir+1)%4;
      else if (c==-2)
        dir=(dir+3)%4;
      else{
        for (int i=0;i<c;i++){
          int nextR=pos[0]+dirs[dir][0],nextC=pos[1]+dirs[dir][1];
          String key=nextR+" "+nextC;
          if (ob.contains(key))
            break;
          pos[0]=nextR;
          pos[1]=nextC;
        }
        ans=Math.max(ans,pos[0]*pos[0]+pos[1]*pos[1]);
      }
    return Math.max(ans,pos[0]*pos[0]+pos[1]*pos[1]);
  }

  public TreeNode deleteNode(TreeNode root, int key) {
    if (root==null)
      return root;
    if (root.val<key)
      root.right=deleteNode(root.right,key);
    else if (root.val>key)
      root.left=deleteNode(root.left,key);
    else if (root.right==null)
      return root.left;
    else if (root.left==null)
      return root.right;
    else{
      TreeNode temp=root;
      root=min1(root.right);
      root.right=deleteMin1(temp.right);
      root.left=temp.left;
    }
    return root;
  }

  private TreeNode min1(TreeNode root){
    if (root.left==null)
      return root;
    return min1(root.left);
  }

  private TreeNode deleteMin1(TreeNode root){
    if (root.left==null)
      return root.right;
    root.left=deleteMin1(root.left);
    return root;
  }

  class WBTnode{
    TreeNode tree;
    int id;
    public WBTnode(TreeNode tree,int id){
      this.tree=tree;
      this.id=id;
    }
  }

  public int widthOfBinaryTree(TreeNode root) {
    if (root==null)
      return 0;
    int ans=0;
    Queue<WBTnode> q=new LinkedList<>();
    q.offer(new WBTnode(root,0));
    while (!q.isEmpty()){
      int min=Integer.MAX_VALUE,max=Integer.MIN_VALUE;
      int size=q.size();
      for (int i=0;i<size;i++){
        WBTnode cur=q.poll();
        min=Math.min(min,cur.id);
        max=Math.max(max,cur.id);
        if (cur.tree.left!=null)
          q.offer(new WBTnode(cur.tree.left,cur.id<<1));
        if (cur.tree.right!=null)
          q.offer(new WBTnode(cur.tree.right,(cur.id<<1)+1));
      }
      ans=Math.max(ans,max-min+1);
    }
    return ans;
  }

  public TreeNode buildTree2(int[] inorder, int[] postorder) {
    if (inorder==null || inorder.length==0 ||postorder==null||postorder.length!=inorder.length)
      return null;
    Map<Integer,Integer> inorderIndex=new HashMap<>();
    int n=inorder.length;
    for (int i=0;i<n;i++)
      inorderIndex.put(inorder[i],i);
    return BThelper(inorder,postorder,0,n-1,new int[]{n-1},inorderIndex);
  }

  private TreeNode BThelper(int[] inorder,int[] postorder,int is,int ie,int[] pe,Map<Integer,Integer> inorderRecord){
    if (is>ie)
      return null;
    if (is==ie)
      return new TreeNode(postorder[pe[0]--]);
    int val=postorder[pe[0]--],inorderIndex=inorderRecord.get(val);
    TreeNode cur=new TreeNode(val);
    cur.right= BThelper(inorder,postorder,inorderIndex+1,ie,pe,inorderRecord);
    cur.left=BThelper(inorder,postorder,is,inorderIndex-1,pe,inorderRecord);
    return cur;
  }

  public List<Integer> countSmaller(int[] nums) {
    int n=nums.length;
    List<Integer> path=new ArrayList<>(n);
    LinkedList<Integer> ans=new LinkedList<>();
    if (n==0)
      return ans;
    for (int i=n-1;i>=0;i--){
      int index=CSbinarySearch(path,nums[i]);
      ans.addFirst(path.size()-index);
      path.add(index,nums[i]);
    }
    return ans;
  }

  private int CSbinarySearch(List<Integer> path,int n){
    if (path.isEmpty())
      return 0;
    int s=0,e=path.size()-1;
    while (s<=e){
      int mid=(s+e)>>1;
      if (path.get(mid)<n)
        e=mid-1;
      else
        s=mid+1;
    }
    return s;
  }

  class CSNode{
    CSNode left,right;
    int val,dup,leftSize;
    public CSNode(int v){
      val=v;
      dup=1;
      leftSize=0;
    }
  }

  public List<Integer> countSmaller1(int[] nums) {
    int n = nums.length;
    LinkedList<Integer> ans = new LinkedList<>();
    if (n == 0)
      return ans;
    CSNode root=null;
    for (int i=n-1;i>=0;i--)
      root=CSinsert(root,nums[i],0,ans);
    return ans;
  }

  private CSNode CSinsert(CSNode root,int val,int preSmaller,LinkedList<Integer> res){
    if (root==null){
      root=new CSNode(val);
      res.addFirst(preSmaller);
    }
    else if (root.val==val){
      root.dup++;
      res.addFirst(preSmaller+root.leftSize);
    }
    else if (root.val < val)
      root.right=CSinsert(root.right,val,preSmaller+root.leftSize+root.dup,res);
    else{
      root.leftSize++;
      root.left=CSinsert(root.left,val,preSmaller,res);
    }
    return root;
  }

  class Node116 {
    public int val;
    public Node116 left;
    public Node116 right;
    public Node116 next;

    public Node116() {}

    public Node116(int _val,Node116 _left,Node116 _right,Node116 _next) {
      val = _val;
      left = _left;
      right = _right;
      next = _next;
    }
  }

  public Node116 connect(Node116 root) {
      if (root==null)
        return root;
      Queue<Node116> q = new LinkedList<>();
      q.offer(root);
      while (!q.isEmpty()){
        int size=q.size();
        Node116 head=q.poll(),next;
        if (head.left!=null){
          q.offer(head.left);
          q.offer(head.right);
        }
        for (int i=1;i<size;i++){
          next=q.poll();
          head.next=next;
          head=next;
          if (head.left!=null){
            q.offer(head.left);
            q.offer(head.right);
          }
        }
      }
      return root;
  }

  public Node116 connect1(Node116 root) {
    if (root == null)
      return root;
    Node116 ans=root;
    while (root.left!=null){
      Node116 nextLine=root.left;
      while (root!=null){
        root.left.next=root.right;
        root.right.next=root.next==null?null:root.next.left;
        root=root.next;
      }
      root=nextLine;
    }
    return ans;
  }

  public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
    List<TreeNode>[] paths=new List[2];
    LCAsearchPath(root,p,q,new ArrayList<>(),paths);
    TreeNode LCA=null;
    int len=Math.min(paths[0].size(),paths[1].size());
    for (int i=0;i<len;i++)
      if (paths[0].get(i)==paths[1].get(i))
        LCA=paths[0].get(i);
      else
        break;
    return LCA;
  }

  private void LCAsearchPath(TreeNode root,TreeNode p,TreeNode q,List<TreeNode> path,List<TreeNode>[] res){
    if (root==null || res[1]!=null)
      return;
    path.add(root);
    if (root==p || root==q)
      if (res[0]==null)
        res[0]=new ArrayList<>(path);
      else{
        res[1]=new ArrayList<>(path);
        return;
      }
    LCAsearchPath(root.left,p,q,path,res);
    LCAsearchPath(root.right,p,q,path,res);
    path.remove(path.size()-1);
  }

  public TreeNode lowestCommonAncestor4(TreeNode root, TreeNode p, TreeNode q) {
    if (root==null || root==p || root==q)
      return root;
    TreeNode left=lowestCommonAncestor4(root.left,p,q),right=lowestCommonAncestor4(root.right,p,q);
    return left==null?right:right==null?left:root;
  }

  public List<TreeNode> generateTrees(int n) {
    if (n<=0)
      return new ArrayList<>();
    List<TreeNode>[][] dp=new List[n+1][n+1];
    for (int i=1;i<=n;i++){
      dp[i][i]=new ArrayList<>();
      dp[i][i].add(new TreeNode(i));
    }
    for (int i=2;i<=n;i++) // length
      for (int j=1;j<=n-i+1;j++){  // begin
        int last=i+j-1;
        List<TreeNode> cur=new ArrayList<>();
        for (int k=j;k<=last;k++){
          List<TreeNode> left=k-1>=j?dp[j][k-1]:null,right=k+1<=last?dp[k+1][last]:null;
          if (left==null)
            for (TreeNode r:right){
              TreeNode temp=new TreeNode(k);
              temp.right=r;
              cur.add(temp);
            }
          else if (right==null)
            for (TreeNode l:left){
              TreeNode temp=new TreeNode(k);
              temp.left=l;
              cur.add(temp);
            }
          else
            for (TreeNode l:left)
              for (TreeNode r:right){
                TreeNode temp=new TreeNode(k);
                temp.left=l;
                temp.right=r;
                cur.add(temp);
              }
        }
        dp[j][last]=cur;
      }
    return dp[1][8];
  }

  public TreeNode sortedListToBST(ListNode head) {
    if (head==null)
      return null;
    int len=0;
    ListNode count=head;
    while (count!=null){
      len++;
      count=count.next;
    }
    List<Integer> data=new ArrayList<>(len);
    while (head!=null){
      data.add(head.val);
      head=head.next;
    }
    return SLTconstructTree(data,0,data.size()-1);
  }

  private TreeNode SLTconstructTree(List<Integer> data,int start,int end){
    if (start>end)
      return null;
    if (start==end)
      return new TreeNode(data.get(start));
    int valId=(start+end)>>1;
    TreeNode root=new TreeNode(data.get(valId));
    root.left=SLTconstructTree(data,start,valId-1);
    root.right=SLTconstructTree(data,valId+1,end);
    return root;
  }

  class SFSnode{
    TreeNode tree;
    int bottomSum;
    public SFSnode(TreeNode t,int bs){
      tree=t;
      bottomSum=bs;
    }
  }

  public TreeNode sufficientSubset(TreeNode root, int limit) {
    if (root==null)
      return null;
    SFSnode res=SFSgetSubSum(null,root,0,limit);
    return res.tree==null?null:res.tree;
  }

  private SFSnode SFSgetSubSum(TreeNode parent,TreeNode cur,int upSum,int limit){
    if (cur==null)
     return null;
    int val=upSum+cur.val;
    SFSnode leftSum=SFSgetSubSum(cur,cur.left,val,limit),rightSum=SFSgetSubSum(cur,cur.right,val,limit);
    cur.left=leftSum==null?null:leftSum.tree;
    cur.right=rightSum==null?null:rightSum.tree;
    if (leftSum==null && rightSum == null)
      return val<limit?new SFSnode(null,cur.val):new SFSnode(cur, cur.val);
    else if (leftSum == null)
      return val+rightSum.bottomSum<limit?new SFSnode(null,cur.val+rightSum.bottomSum):new SFSnode(cur,cur.val+rightSum.bottomSum);
    else if (rightSum == null)
      return val+leftSum.bottomSum<limit?new SFSnode(null,cur.val+leftSum.bottomSum):new SFSnode(cur,cur.val+leftSum.bottomSum);
    else{
      int maxSub=Math.max(leftSum.bottomSum, rightSum.bottomSum);
      return val+maxSub<limit?new SFSnode(null,maxSub+cur.val):new SFSnode(cur, maxSub + cur.val);
    }
  }

  public TreeNode sufficientSubset1(TreeNode root, int limit) {
    if (root == null)
      return null;
    if (root.left==null && root.right==null)
      return root.val<limit?null:root;
    root.left=sufficientSubset1(root.left,limit-root.val);
    root.right=sufficientSubset1(root.right,limit-root.val);
    return root.left==root.right?null:root;
  }

  public Node116 connect2(Node116 root) {
    Node116 D=root;
    while (D != null){
      Node116 nextD=null,W=D,last=null;
      while (W!=null){
        if (W.left != null && W.right != null){
          if (nextD ==null)
            nextD = W.left;
          if (last != null)
            last.next=W.left;
          W.left.next=W.right;
          last=W.right;
        }
        else if (W.left !=null){
          if (nextD ==null)
            nextD = W.left;
          if (last != null)
            last.next=W.left;
          last=W.left;
        }
        else if (W.right != null){
          if (nextD ==null)
            nextD = W.right;
          if (last != null)
            last.next=W.right;
          last = W.right;
        }
        W=W.next;
      }
      D=nextD;
    }
    return root;
  }

  public boolean possibleBipartition(int N, int[][] dislikes) {
    if (N <3 || dislikes.length==0)
      return true;
    List<Integer>[] graph=new List[N+1];
    PBconstructGraph(graph,dislikes);
    boolean[] res=new boolean[]{true},visited=new boolean[N+1],colored=new boolean[N+1];
    for (int i=1;i<=N && res[0];i++)
      if (!visited[i])
        PBdetectCircle(graph,i,res,visited,colored);
    return res[0];
  }

  private void PBconstructGraph(List<Integer>[] graph,int[][] dislikes){
    int n=graph.length;
    for (int i=0;i<n;i++)
      graph[i]=new ArrayList<>();
    for (int[] d:dislikes){
      graph[d[0]].add(d[1]);
      graph[d[1]].add(d[0]);
    }
  }

  private void PBdetectCircle(List<Integer>[] graph,int cur,boolean[] res,boolean[] visited,boolean[] colored){
    if (res[0]==false)
      return;
    visited[cur]=true;
    for (int next:graph[cur])
      if (!visited[next]){
        colored[next]=!colored[cur];
        PBdetectCircle(graph,next,res,visited,colored);
      }
      else if (colored[cur]==colored[next]){
        res[0]=false;
        return;
      }
  }

  public int findPaths(int m, int n, int N, int i, int j) {
    if (N == 0)
      return 0;
    int[][][] memo=new int[N+1][m][n];
    for (int a=1;a<=N;a++)
      for (int r=0;r<m;r++)
        for (int c=0;c<n;c++)
          memo[a][r][c]=-1;
    return FPhelper(m,n,N,i,j,memo) % 1000000007;
  }

  private int FPhelper(int R, int C,int n,int r,int c,int[][][] memo){
    if (r<0 || r>=R || c<0 || c>=C )
      return 1;
    if (memo[n][r][c]!=-1)
      return memo[n][r][c];
    int res=0;
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int[] d:dirs)
      res = (res+ FPhelper(R,C,n-1,r+d[0],c+d[1],memo))% 1000000007;
    memo[n][r][c]=res;
    return res;
  }

  public int shortestPathBinaryMatrix(int[][] grid) {
    int n=grid.length,ans=1;
    if (grid[0][0] == 1 || grid[n-1][n-1]== 1)
      return -1;
    if (n == 1)
      return 0;
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
    boolean[][] visited=new boolean[n][n];
    Queue<int[]> q=new LinkedList<>();
    q.offer(new int[]{0,0});
    visited[0][0]=true;
    while (!q.isEmpty()){
      ans++;
      int size=q.size();
      for (int i=0;i<size;i++){
        int[] cur=q.poll();
        for (int[] d:dirs){
          int r= cur[0]+d[0],c=cur[1]+d[1];
          if (r<0 || r>=n || c < 0 || c>=n || grid[r][c] == 1 || visited[r][c])
            continue;
          if (r==n-1 && c==n-1)
            return ans;
          q.offer(new int[]{r,c});
          visited[r][c]=true;
        }
      }
    }
    return -1;
  }

  public int snakesAndLadders(int[][] board) {
    int n=board.length,ans=0;
    int[] des=new int[]{0,(n&1)==1?n-1:0};
    Queue<int[]> q=new LinkedList<>();
    q.offer(new int[]{n-1,0});
    boolean[][] visited=new boolean[n][n];
    visited[n-1][0]=true;
    while (!q.isEmpty()){
      int size=q.size();
      ans++;
      for (int i=0;i<size;i++){
        int[] cur=q.poll();
        int index=SALposToIndex(cur,n);
        for (int j=1;j<=6;j++){
          int[] next=SALindexToPos(index+j,n);
          if (board[next[0]][next[1]]!=-1)
            next= SALindexToPos(board[next[0]][next[1]],n);
          if (visited[next[0]][next[1]])
            continue;
          visited[next[0]][next[1]]=true;
          if (next[0]==des[0] && next[1]==des[1])
            return ans;
          q.offer(next);
        }
      }
    }
    return -1;
  }

  private int[] SALindexToPos(int index,int n){
    int realRow=(index-1)/n,r=n-1-realRow,c=(realRow&1)==0?(index-1)%n:n-1-(index-1)%n;
    return new int[]{r,c};
  }

  private int SALposToIndex(int[] pos,int n){
    int realRow=(n-1-pos[0]),bottom=realRow*n,side=(realRow& 1)==0? pos[1]+1:n-pos[1];
    return bottom+side;
  }

  class LCnode{
    int id;
    int weight;
    public LCnode(int id){
      this.id = id;
      this.weight = 1;
    }
  }

  public int longestConsecutive(int[] nums) {
    if (nums==null || nums.length==0)
      return 0;
    Map<Integer,LCnode> record= new HashMap<>();
    int ans=1;
    for (int n:nums){
      record.putIfAbsent(n,new LCnode(n));
      if (record.containsKey(n-1))
        ans=Math.max(LCunion(record,n,n-1),ans);
      if (record.containsKey(n+1))
        ans = Math.max(LCunion(record,n,n+1),ans);
    }
    return ans;
  }

  private LCnode LCfind(Map<Integer,LCnode> record, int n){
    LCnode val,next;
    int temp=n,nextId=0;
    while ((val=record.get(n)).id!= n)
      n=val.id;
    while (temp != n){
      nextId=(next = record.get(temp)).id;
      next.id=n;
      temp=nextId;
    }
    return val;
  }

  private int LCunion(Map<Integer,LCnode> record,int i,int j){
    LCnode idI=LCfind(record,i),idJ=LCfind(record,j);
    int size;
    if (idI.id == idJ.id)
      return record.get(i).weight;
    if (idI.weight<=idJ.weight){
      idI.id = j;
      idJ.weight+=idI.weight;
      size = idJ.weight;
    }
    else{
      idJ.id = i;
      idI.weight+=idJ.weight;
      size = idI.weight;
    }
    return size;
  }

  public int longestConsecutive1(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    Map<Integer, Integer> record = new HashMap<>();
    int ans = 1;
    for (int n:nums){
      if (record.containsKey(n))
        continue;
      int left=record.getOrDefault(n-1,0),right=record.getOrDefault(n+1,0),sum=left+right+1;
      record.put(n,sum);
      ans=Math.max(ans,sum);
      if (left!=0)
        record.put(n-left,sum);
      if (right!=0)
        record.put(n+right,sum);
    }
    return ans;
  }

  public int minMalwareSpread(int[][] graph, int[] initial) {
    int n=graph.length,ans=-1,max=Integer.MIN_VALUE;
    Arrays.sort(initial);
    List<Integer>[] reach=new List[n];
    for (int i=0;i<n;i++)
      reach[i]=new ArrayList<>();
    Set<Integer> infect=new HashSet<>();
    for (int i:initial)
      infect.add(i);
    for (int u:initial){
      Set<Integer> seen=new HashSet<>();
      MMSgetInfected(graph,u,seen,infect);
      for (int s:seen)
        reach[s].add(u);
    }
    int[] infectCount=new int[n];
    for (int i=0;i<n;i++)
      if (reach[i].size()==1)
        infectCount[reach[i].get(0)]++;
    for (int u:initial)
      if (infectCount[u]>max || infectCount[u]==max && u<ans){
        max=infectCount[u];
        ans=u;
      }
    return ans;
  }

  private void MMSgetInfected(int[][] graph,int cur,Set<Integer> seen,Set<Integer> infect){
    for (int i=0;i<graph.length;i++)
      if (graph[cur][i] ==1 && !seen.contains(i)&&!infect.contains(i)){
        seen.add(i);
        MMSgetInfected(graph,i,seen,infect);
      }
  }

  class MyLinkedList {
    class Node{
      int val;
      Node prev,next;

      public Node(int v){
        this.val=v;
      }
    }

    Node first,last;
    int size;
    /** Initialize your data structure here. */
    public MyLinkedList() {
     first = new Node(0);
     last = new Node(0);
     first.next=last;
     last.prev=first;
     size=0;
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
      if (index<0 || index>=size)
        return -1;
      Node ans = first;
      for (int i=0;i<=index;i++)
        ans=ans.next;
      return ans.val;
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
      Node cur = new Node(val),second=first.next;
      first.next=cur;
      cur.next=second;
      cur.prev=first;
      second.prev=cur;
      size++;
    }

    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
      Node cur = new Node(val),temp = last.prev;
      cur.next=last;
      cur.prev=temp;
      last.prev=cur;
      temp.next=cur;
      size++;
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
      if ( index >size)
        return;
      if (index==size){
        addAtTail(val);
        return;
      }
      if (index < 0){
        addAtHead(val);
        return;
      }
      Node cur = first;
      Node add = new Node(val);
      for (int i=0;i<=index;i++)
        cur=cur.next;
      cur.prev.next=add;
      add.prev = cur.prev;
      add.next=cur;
      cur.prev=add;
      size++;
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
      if (index<0 || index>=size)
        return;
      Node cur = first;
      for (int i=0;i<=index;i++)
        cur=cur.next;
      cur.prev.next=cur.next;
      cur.next.prev=cur.prev;
      size--;
    }
  }

  public int longestIncreasingPath(int[][] matrix) {
    if (matrix==null || matrix.length==0)
      return 0;
    int ans=0,R=matrix.length,C=matrix[0].length;
    Queue<int[]> q=new LinkedList<>();
    int[][] inDegree = new int[R][C],dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++){
        for (int[] d:dirs){
          int nr=r+d[0],nc = c+d[1];
          if (nr>=0 && nr<R && nc>=0 && nc<C && matrix[nr][nc]<matrix[r][c])
            inDegree[r][c]++;
        }
        if (inDegree[r][c]==0)
          q.offer(new int[]{r,c});
      }
    while (!q.isEmpty()){
      ans++;
      int size=q.size();
      for (int i=0;i<size;i++){
        int[] cur=q.poll();
        for (int[] d:dirs){
          int r=cur[0]+d[0],c=cur[1]+d[1];
          if (r>=0 && r<R && c>=0 && c<C && inDegree[r][c]>0 && matrix[r][c]>matrix[cur[0]][cur[1]]){
            inDegree[r][c]--;
            if (inDegree[r][c]==0)
              q.offer(new int[]{r,c});
          }
        }
      }
    }
    return ans;
  }

  public int longestIncreasingPath1(int[][] matrix) {
    if (matrix == null || matrix.length == 0)
      return 0;
    int ans = 0, R = matrix.length, C = matrix[0].length;
    int[][] memo = new int[R][C];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (memo[r][c]==0)
          ans = Math.max(ans,LIPhepler(matrix,r,c,Integer.MIN_VALUE,memo));
    return ans;
  }

  private int LIPhepler(int[][] M,int r,int c,int pre,int[][] memo){
    int R=M.length,C=M[0].length,depth=0;
    if (r<0 || r>= R || c<0 || c>=C || M[r][c]<=pre)
      return 0;
    if (memo[r][c] != 0)
      return memo[r][c];
    int[][] dirs=new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
    for (int[] d:dirs)
      depth = Math.max(depth,LIPhepler(M,r+d[0],c+d[1],M[r][c],memo));
    depth++;
    memo[r][c]=depth;
    return depth;
  }

  class StreamChecker1 {
    class SCnode{
      boolean isWord;
      SCnode[] next;
      public SCnode(){
        isWord=false;
        next=new SCnode[26];
      }
    }

    private SCnode root;
    Queue<SCnode> q;

    public void addTrie(SCnode cur,String word,int depth){
      if (depth==word.length()){
        cur.isWord=true;
        return;
      }
      int index = word.charAt(depth)-'a';
      if (cur.next[index]==null)
        cur.next[index]=new SCnode();
      addTrie(cur.next[index],word,depth+1);
    }

    public StreamChecker1(String[] words) {
      root=new SCnode();
      for (String w:words)
        addTrie(root,w,0);
      q = new LinkedList<>();
      q.offer(root);
    }

    public boolean query(char letter) {
      int index=letter-'a',size = q.size();
      boolean res=false;
      for (int i=0;i<size;i++){
        SCnode cur=q.poll();
        if (cur.next[index]!=null){
          q.offer(cur.next[index]);
          if (cur.next[index].isWord)
            res=true;
        }
        if (cur==root)
          q.offer(cur);
      }
      return res;
    }
  }

  class StreamChecker {
    class SCnode{
      boolean isWord;
      SCnode[] next;
      public SCnode(){
        isWord=false;
        next=new SCnode[26];
      }
    }

    private SCnode root;
    LinkedList<Character> q;
    int maxSize;

    private void buildTrie(SCnode root,String[] words){
      for (String w:words){
        SCnode cur=root;
        for (int i=w.length()-1;i>=0;i--){
          char c=w.charAt(i);
          int index=c-'a';
          if (cur.next[index]==null)
            cur.next[index]=new SCnode();
          cur=cur.next[index];
        }
        cur.isWord=true;
      }
    }

    public StreamChecker(String[] words) {
      root=new SCnode();
      buildTrie(root,words);
      q=new LinkedList<>();
      maxSize=0;
      for (String w:words)
        maxSize=Math.max(maxSize,w.length());
    }

    public boolean query(char letter) {
      q.offer(letter);
      if (q.size()>maxSize)
        q.poll();
      SCnode cur = root;
      Iterator<Character> itr = q.descendingIterator();
      while (itr.hasNext()){
        char temp=itr.next();
        int index=temp-'a';
        if (cur.next[index]==null)
          return false;
        cur=cur.next[index];
        if (cur.isWord)
          return true;
      }
      return false;
    }
  }

  public int strStr(String haystack, String needle) {
    if (needle==null || needle.length()==0)
      return 0;
    if (haystack==null || haystack.length()==0)
      return -1;
    int[][] dfa = SSgetDFA(needle);
    return SSsearch(haystack,dfa);
  }

  private int SSsearch(String H,int[][] dfa){
    int h,p;
    for (h=0,p=0;h<H.length() && p<dfa[0].length;h++)
      p=dfa[H.charAt(h)][p];
    return p==dfa[0].length?h-p:-1;
  }

  private int[][] SSgetDFA(String pat){
    int R=256,n=pat.length();
    int[][] dfa = new int[R][n];
    dfa[pat.charAt(0)][0]=1;
    for (int X=0,i=1;i<n;i++){
      for (int j=0;j<R;j++)
        dfa[j][i]=dfa[j][X];
      dfa[pat.charAt(i)][i]=i+1;
      X = dfa[pat.charAt(i)][X];
    }
    return dfa;
  }

  public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
    if (l1==null && l2==null)
      return null;
    if (l1==null)
      return l2;
    if (l2==null)
      return l1;
    ListNode ans = new ListNode(0),cur=ans;
    int carry=0,d1,d2,val;
    while (l1 != null || l2!=null ||carry!=0){
      d1=l1==null?0:l1.val;
      d2 = l2==null?0:l2.val;
      val = d1+d2+carry;
      carry = val/10;
      cur.next = new ListNode(val%10);
      cur=cur.next;
      if (l1!=null)
        l1=l1.next;
      if (l2!=null)
        l2=l2.next;
    }
    return ans.next;
  }

  public String longestPalindrome1(String s) {
    if (s == null || s.length()==0)
      return s;
    int start=0,end=0,count=1,n=s.length();
    char[] cs=s.toCharArray();
    for (int i=0;i<n;i++){
      int d1=1,d2=0;
      while (i-d1>=0 && i+d1<n && cs[i-d1]==cs[i+d1])
        d1++;
      while (i-d2>=0 && i+d2+1<n && cs[i-d2]==cs[i+d2+1])
        d2++;
      int c1=(--d1<<1)+1,c2=d2<<1;
      if (count>= c1 && count>=c2)
        continue;
      else if (c1 >=c2){
        count=c1;
        start=i-d1;
        end=i+d1;
      }
      else{
        count=c2;
        start=i-d2+1;
        end=i+d2;
      }
    }
    return s.substring(start,end+1);
  }

  class LP2Data{
    int start,max;
    public LP2Data(){
      start=0;
      max=1;
    }
  }

  public String longestPalindrome2(String s) {
    if (s == null || s.length() <2)
      return s;
    LP2Data data=new LP2Data();
    char[] cs=s.toCharArray();
    int N=cs.length,cur=0;
    while (cur<N)
      cur=LP2extend(cs,data,cur);
    return s.substring(data.start,data.start+data.max);
  }

  private int LP2extend(char[] cs,LP2Data data,int cur){
    int left = cur-1,right=cur,next,N=cs.length;
    while (right<N && cs[right]==cs[cur])
      right++;
    next=right;
    while (left>=0 && right<N && cs[left]==cs[right]){
      left--;
      right++;
    }
    int dist = right-left-1;
    if (dist>data.max){
      data.max=dist;
      data.start=left+1;
    }
    return next;
  }

  public String convert(String s, int numRows) {
    if (numRows<1)
      throw new IllegalArgumentException();
    if (s== null || s.length()<3||numRows==1)
      return s;
    StringBuilder sb=new StringBuilder();
    char[] cs=s.toCharArray();
    CaddHeadTail(0,numRows,cs,sb);
    for (int i=1;i<numRows-1;i++)
      CaddMiddle(i,i,numRows,cs,sb);
    CaddHeadTail(numRows-1,numRows,cs,sb);
    return sb.toString();
  }

  private void CaddHeadTail(int cur,int R,char[] cs,StringBuilder sb){
    int N=cs.length;
    while (cur<N){
      sb.append(cs[cur]);
      cur+=(R -1)<<1;
    }
  }

  private void CaddMiddle(int cur,int curR,int R,char[] cs,StringBuilder sb){
    int index=0,N=cs.length;
    while (cur < N){
      sb.append(cs[cur]);
      cur+=(index&1)==0?(R-curR-1)<<1:curR<<1;
      index++;
    }
  }

  public int myAtoi(String str) {
    if (str==null || str.isEmpty())
      return 0;
    char[] cs=str.toCharArray();
    int N=cs.length,cur=0;
    while (cur<N && cs[cur]==' ')
      cur++;
    if (cur>=N || (!isDigit(cs[cur]) && cs[cur]!='-' && cs[cur]!='+'))
      return 0;
    boolean isNeg=cs[cur]=='-';
    if (cs[cur]=='+' || cs[cur]=='-')
      cur++;
    long ans=0;
     while (cur<N && isDigit(cs[cur])){
      ans=isNeg?ans*10-(cs[cur]-'0'):ans*10+cs[cur]-'0';
      cur++;
      if (ans>=Integer.MAX_VALUE)
        return Integer.MAX_VALUE;
      if (ans <=Integer.MIN_VALUE)
        return Integer.MIN_VALUE;
    }
    return (int)ans;
  }

  public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> ans=new ArrayList<>();
    if (nums==null || nums.length <3)
      return ans;
    Arrays.sort(nums);
    int N = nums.length,i=0,j,k;
    while (i<N-2){
      j=i+1;
      k=N-1;
      while (j<k)
        if (nums[j]+nums[k]<-nums[i])
          j++;
        else if (nums[j]+nums[k]>-nums[i])
          k--;
        else{
          ans.add(Arrays.asList(new Integer[]{nums[i],nums[j],nums[k]}));
          j=TSnextNotSame(nums,j,1);
          k=TSnextNotSame(nums,k,-1);
        }
      i=TSnextNotSame(nums,i,1);
    }
    return ans;
  }

  private int TSnextNotSame(int[] nums,int cur,int dir){
    int ans=cur,N=nums.length;
    while (ans>=0 && ans <N && nums[ans]==nums[cur])
      ans+=dir;
    return ans;
  }

  public List<List<Integer>> fourSum(int[] nums, int target) {
    ArrayList<List<Integer>> ans=new ArrayList<>();
    if (nums==null || nums.length<4)
      return ans;
    Arrays.sort(nums);
    return Ksum(nums,0,target,4);
  }

  private List<List<Integer>> Ksum(int[] nums,int index,int target,int K){
    if (K <2 )
      throw new IllegalArgumentException();
    if (K==2)
      return TwoSumHelper(nums,index,target);
    ArrayList<List<Integer>> ans=new ArrayList<>();
    for (int i=index;i<nums.length-K+1;i++){
      if (i>index && nums[i]==nums[i-1])
        continue;
      List<List<Integer>> temp = Ksum(nums,i+1,target-nums[i],K-1);
      for (List<Integer> t:temp)
        t.add(nums[i]);
      ans.addAll(temp);
    }
    return ans;
  }

  private List<List<Integer>> TwoSumHelper(int[] nums, int index, int target) {
    int l = index, r = nums.length - 1;
    List<List<Integer>> ans = new ArrayList<>();
    while (l < r)
      if (nums[l] + nums[r] < target)
        l++;
      else if (nums[l] + nums[r] > target)
        r--;
      else {
        List<Integer> temp=new ArrayList<>();
        temp.add(nums[l]);
        temp.add(nums[r]);
        ans.add(temp);
        while (l + 1 < nums.length && nums[l + 1] == nums[l])
          l++;
        while (r - 1 >= 0 && nums[r - 1] == nums[r])
          r--;
        l++;
        r--;
      }
    return ans;
  }

  public int countRangeSum(int[] nums, int lower, int upper) {
    if (nums==null || nums.length==0)
      return 0;
    int N=nums.length;
    long[] ps=new long[N+1];
    for (int i=0;i<N;i++)
      ps[i+1]=ps[i]+nums[i];
    return CRSmergeSort(ps,0,N,lower,upper);
  }

  private int CRSmergeSort(long[] ps,int start,int end,int lower,int upper){
    if (end<=start)
      return 0;
    int mid = (start+end)>>1;
    int count = CRSmergeSort(ps,start,mid,lower,upper)+CRSmergeSort(ps,mid+1,end,lower,upper);
    count += CRSmerge(ps,start,end,lower,upper);
    return count;
  }

  private int CRSmerge(long[] ps,int start,int end,int lower,int upper){
    int N=end-start+1,mid = (start+end)>>1,count=0,left=mid+1,right=mid+1,smaller=mid+1;
    long[] aug = new long[N];
    for (int i=start,index=0;i<=mid;i++){
      while (right<=end && ps[right]-ps[i]<=upper)
        right++;
      while (left <=end && ps[left]-ps[i] <lower)
        left++;
      while (smaller <=end && ps[smaller]<=ps[i])
        aug[index++]=ps[smaller++];
      aug[index++]=ps[i];
      count+=right-left;
    }
    for (int i=0;i<smaller-start;i++)
      ps[i+start]=aug[i];
    return count;
  }

  public ListNode removeNthFromEnd(ListNode head, int n) {
    if (head==null)
      return head;
    ListNode front = head,back=head,beforeBack=null;
    for (int i=0;i<n-1;i++)
      front=front.next;
    while (front.next!=null){
      front=front.next;
      beforeBack=back;
      back=back.next;
    }
    if (beforeBack==null && back.next==null)
      return null;
    if (beforeBack==null)
      return back.next;
    beforeBack.next=back.next;
    return head;
  }

  public int divide(int a, int b) {
    if (a==1<<31 && b==-1)
      return (1<<31)-1;
    if (a==0)
      return 0;
    boolean pos=(a<0)==(b<0);
    a=Math.abs(a);
    b=Math.abs(b);
    int res=0,shift=0;
    while (a-b>=0){
      for (shift=0;a-(b<<shift<<1)>=0;shift++);
      res+=1<<shift;
      a-=b<<shift;
    }
    return pos?res:-res;
  }

  public void nextPermutation(int[] nums) {
    if (nums==null || nums.length==0)
      return;
    int N=nums.length,max=Integer.MIN_VALUE,i;
    for (i=N-1;i>=0;i--){
      if (nums[i]<max)
        break;
      max=Math.max(max,nums[i]);
    }
    if (i>=0){
      int index = NPbinarySearch(nums,i+1,nums[i]);
      exchange(nums,i,index);
    }
    int l=i+1,r=N-1;
    while (l<r)
      exchange(nums,l++,r--);
  }

  private int NPbinarySearch(int[] nums,int start,int val){
    int l=start,r=nums.length-1;
    while (l<=r){
      int mid = (l+r)>>1;
      if (nums[mid]<=val)
        r=mid-1;
      else
        l=mid+1;
    }
    return r;
  }

  public int search1(int[] nums, int target) {
    if (nums==null|| nums.length==0)
      return -1;
    int N=nums.length,minId=SfindMin(nums),min = nums[minId],maxId=(minId+N-1)%N,max = nums[maxId];
    if (target>max || target<min)
      return -1;
    return nums[0]<=target?SbinarySearch(nums,0,maxId,target):SbinarySearch(nums,maxId+1,N-1,target);
  }

  public int SfindMin(int[] nums){
    int N=nums.length, start=0,end=N-1,mid;
    while (start<end){
      mid = (start+end)>>1;
     if (mid>0 && nums[mid]<nums[mid-1])
       return mid;
     else if (nums[mid]>=nums[start] && nums[mid]> nums[end])
       start=mid+1;
     else
       end = mid-1;
    }
    return start;
  }

  private int SbinarySearch(int[] nums,int start,int end,int target){
    while (start<=end){
      int mid = (start+end)>>1;
      if (nums[mid]<target)
        start=mid+1;
      else if (nums[mid]>target)
        end = mid-1;
      else
        return mid;
    }
    return -1;
  }

  public int lengthOfLastWord(String s) {
    if (s==null || s.length()==0)
      return 0;
    char[] cs=s.toCharArray();
    int N = cs.length,end=-1;
    for (int i=N-1;i>=0;i--)
      if (end== -1 && cs[i] != ' ')
        end = i;
      else if (end != -1 && cs[i] ==' ')
        return end-i;
    return end==-1?0:end+1;
  }

  public boolean isValidBST(TreeNode root) {
    if (root == null)
      return true;
    return IVBhelper(root,Long.MAX_VALUE,Long.MIN_VALUE);
  }

  private boolean IVBhelper(TreeNode root,long upper,long lower){
    if (root == null)
      return true;
    return (root.val< upper && root.val > lower)
            && (root.left == null || root.left.val < root.val) && (root.right == null || root.right.val > root.val)
            && IVBhelper(root.left,Math.min(upper,root.val),lower)
            && IVBhelper(root.right,upper,Math.max(lower,root.val));
  }

  public String multiply(String a, String b) {
    if (a.equals("0") || b.equals("0"))
      return "0";
    char[] ca=a.toCharArray(),cb=b.toCharArray();
    List<Character> temp=new ArrayList<>(),ans = new ArrayList<>();
    int aLen=ca.length,bLen=cb.length,carry=0,curB,curA,curVal,ansCarry,ansIndex;
    for (int i = bLen-1;i>=0;i--){
      temp.clear();
      carry=ansIndex=ansCarry =0;
      for (int k=0;k<bLen-i-1;k++)
        temp.add('0');
      curB = cb[i]-'0';
      for (int j=aLen-1;j>=0;j--){
        curA = ca[j]-'0';
        curVal = curA*curB+carry;
        temp.add((char)(curVal%10+'0'));
        carry = curVal/10;
      }
      if (carry!=0)
        temp.add((char)(carry+'0'));
      while (ansIndex < temp.size() || ansCarry!=0){
        int v = ansIndex<temp.size()?temp.get(ansIndex)+ansCarry-'0':ansCarry;
        if (ansIndex<ans.size()){
          v+=ans.get(ansIndex)-'0';
          ans.set(ansIndex,(char)(v%10+'0'));
        }
        else
          ans.add((char)(v%10+'0'));
        ansCarry = v/10;
        ansIndex++;
      }
    }
    StringBuilder sb=new StringBuilder();
    for (int i = ans.size()-1;i>=0;i--)
      sb.append(ans.get(i));
    return sb.toString();
  }

  public int numTrees2(int n) {
    if (n<0)
      throw new IllegalArgumentException();
    if (n<2)
      return 1;
    int[][] dp=new int[n+1][n+1];
    for (int i=1;i<=n;i++)
      dp[i][i]=1;
    return NT2helper(1,n,dp);
  }

  private int NT2helper(int start,int end,int[][] memo){
    if (start>end)
      return 1;
    if (memo[start][end]!=0)
      return memo[start][end];
    int sum=0;
    for (int i=start;i<=end;i++)
      sum += NT2helper(start,i-1,memo)*NT2helper(i+1,end,memo);
    memo[start][end]=sum;
    return sum;
  }

  public int numTrees3(int n) {
    if (n < 0)
      throw new IllegalArgumentException();
    if (n < 2)
      return 1;
    int[] dp= new int[n+1];
    dp[0]=dp[1]=1;
    for (int i=2;i<=n;i++)
      for (int j=1;j<=i;j++)
        dp[i] += dp[j-1]*dp[i-j];
    return dp[n];
  }

  public String fractionAddition1(String E) {
    if (E==null || E.isEmpty())
      return E;
    char[] cs = E.toCharArray();
    int N=0,D=1,len = cs.length,index=0;
    while (index<len){
      if (cs[index++]!='/')
        continue;
      int before=index-2,after=index,curN=0,curD=0;
      while (before >=0 && isDigit(cs[before])){
        curN += (cs[before]-'0')*(int)Math.pow(10,index-before-2);
        before--;
      }
      if (before >=0 && cs[before]=='-')
        curN =-curN;
      while (after <len && isDigit(cs[after])){
        curD =curD*10+cs[after]-'0';
        after++;
      }
      index = after;
      int gcd = FAgcd(D,curD);
      int lct= D*curD/gcd;
      N = N * lct/D + curN*lct/curD;
      D = lct;
      int gcdND = FAgcd(D,Math.abs(N));
      N /= gcdND;
      D /= gcdND;
    }
    return N+"/"+D;
  }

  private int FAgcd(int a,int b){
    if (b==0)
      return a;
    return FAgcd(b,a%b);
  }

  public boolean validTicTacToe1(String[] board) {
    int X=0,O=0,fs=0,bs=0,N=board.length;
    int[] cols =new int[N],rows = new int[N];
    boolean Xwin=false,Owin=false;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        if (board[i].charAt(j)=='X'){
          X++;
          rows[i]++;
          cols[j]++;
          if (i+j==N-1)
            fs++;
          if (i==j)
            bs++;
          if (rows[i]==N || cols[j]==N || fs==N || bs==N)
            Xwin=true;
        }
        else if (board[i].charAt(j)=='O'){
          O++;
          rows[i]--;
          cols[j]--;
          if (i+j==N-1)
            fs--;
          if (i==j)
            bs--;
          if (rows[i]==-N || cols[j]==-N || fs==-N || bs==-N)
            Owin=true;
        }
    if (O > X || X-O >1)
      return false;
    if (Xwin && Owin)
      return false;
    if ( (Owin && X != O ) || (Xwin && X==O))
      return false;
    return true;
  }

  public List<Boolean> prefixesDivBy51(int[] A) {
    List<Boolean> ans = new ArrayList<>(A.length);
    if (A.length==0)
      return ans;
    int res=0;
    for (int a:A){
      res = (res<<1)|a;
      ans.add((res%=5)==0);
    }
    return ans;
  }

  public int findKthLargest2(int[] nums, int k) {
    FKLshuffle(nums);
    return FKLquickSelect(nums,k,0,nums.length-1);
  }

  private int FKLquickSelect(int[] N,int k ,int start,int end){
    if (start==end)
      return N[start];
    int partition = FKLpartition(N,start,end);
    if (partition < k)
      return FKLquickSelect(N,k-partition,partition+start,end);
    else if (partition >k)
      return FKLquickSelect(N,k,start,start+partition-2);
    else
      return N[start+partition-1];
  }

  private int FKLpartition(int[] N,int start,int end){
    if (start==end)
      return start+1;
    int p = start,l=start+1,r=end;
    while (l<=r){
      while (l<= r && N[l]>N[p])
        l++;
      while (r>=l && N[r]<=N[p])
        r--;
      if (l>r)
        break;
      exchange(N,l,r);
      r--;
      l++;
    }
    exchange(N,p,r);
    return r+1-start;
  }

  private void FKLshuffle(int[] N){
    Random r = new Random();
    int len = N.length;
    for (int i=len-1;i>=0;i--)
      exchange(N,i,r.nextInt(i+1));
  }

  public int singleNumber4(int[] nums) {
    int a=0,b=0;
    for (int n:nums){
      a = (a^n) & ~b;
      b = (b^n) & ~a;
    }
    return a;
  }

  class NumArray3 {
    class Node{
      int id,sum,left,right,lazy;
      public Node(int id,int L,int R){
        this.id = id;
        this.left = L;
        this.right = R;
        this.sum = this.lazy = 0;
      }

      private void pushDown(){
        if (lazy == 0)
          return;
        nodes[id<<1+1].updateByVal(lazy);
        nodes[id<<1+2].updateByVal(lazy);
        lazy=0;
      }

      private void updateByVal(int val){
        int len = right-left+1;
        this.lazy = val;
        this.sum = val*len;
      }

      private void updateFromSons(){
        this.sum = nodes[(id<<1)+1].sum+nodes[(id<<1)+2].sum;
      }
    }

    private Node[] nodes;

    public NumArray3(int[] nums) {
      int N = nums.length;
      nodes = new Node[N << 2];
      buildTree(0,nums,0,N-1);
    }

    private void buildTree(int idx,int[] vals,int left,int right){
      nodes[idx] = new Node(idx,left,right);
      if (left==right){
        nodes[idx].sum=vals[left];
        return;
      }
      int leftNode = (idx<<1)+1,rightNode =(idx<<1)+2 ,mid = (left+right)>>1;
      buildTree(leftNode,vals,left,mid);
      buildTree(rightNode,vals,mid+1,right);
      nodes[idx].updateFromSons();
    }

    private void mergeQuery(int[] res,Node cur){
      res[0]+=cur.sum;
    }

    public void updateRange(int start,int end,int val){
      updateRange(start,end,0,val);
    }

    private void updateRange(int start,int end,int curIdx,int val){
      Node cur = nodes[curIdx];
      if (start> cur.right || end <cur.left)
        return;
      if (cur.left>= start && cur.right <= end){
        cur.updateByVal(val);
        return;
      }
      cur.pushDown();
      updateRange(start,end,(curIdx<<1)+1,val);
      updateRange(start,end,(curIdx<<1)+2,val);
      cur.updateFromSons();
    }

    public void update(int i, int val) {
     update(i,0,val);
    }

    private void update(int idx,int curIdx,int val){
      if (nodes[curIdx].left==nodes[curIdx].right){
        nodes[curIdx].sum=val;
        return;
      }
      int mid = (nodes[curIdx].left+nodes[curIdx].right)>>1;
      if (idx <= mid)
        update(idx,(curIdx<<1)+1,val);
      else
        update(idx,(curIdx<<1)+2,val);
      nodes[curIdx].updateFromSons();
    }

    public int sumRange(int i, int j) {
      int[] res=new int[1];
      query(i,j,0,res);
      return res[0];
    }

    private void query(int start,int end,int curIdx,int[] res){
      Node cur = nodes[curIdx];
      if (cur.left >= start && cur.right<=end){
        mergeQuery(res,cur);
        return;
      }
      if (cur.left> end || cur.right < start)
        return;
      cur.pushDown();
      query(start,end,(curIdx<<1)+1,res);
      query(start,end,(curIdx<<1)+2,res);
      cur.updateFromSons();
    }
  }

  public int minSteps2(int n) {
    if (n<1)
      throw new IllegalArgumentException();
    if (n==1)
      return 0;
    int[] memo = new int[n+1];
    Arrays.fill(memo,-1);
    memo[1]=0;
    return MShelper(n,memo);
  }

  private int MShelper(int n,int[] memo){
    if (memo[n]!=-1)
      return memo[n];
    int steps = Integer.MAX_VALUE;
    for (int i=1;i<=(n>>1);i++)
      if (n % i ==0)
        steps = Math.min(steps,1+(n/i-1)+MShelper(i,memo));
    memo[n]=steps;
    return steps;
  }

  public int fourSumCount2(int[] A, int[] B, int[] C, int[] D) {
    int N = A.length,ans = 0;
    if (N==0)
      return 0;
    Map<Integer,Integer> ab = new HashMap<>();
    for (int a:A)
      for (int b:B){
        int val = a+b;
        ab.put(val,ab.getOrDefault(val,0)+1);
      }
    for (int c:C)
      for (int d:D)
        ans += ab.getOrDefault(-c-d,0);
    return ans;
  }

  public int minMalwareSpread3(int[][] G, int[] I) {
    if (I.length==1)
      return I[0];
    int N = G.length,ans=-1,ansCount = Integer.MIN_VALUE;
    List<Integer>[] sources = new List[N];
    boolean[] isInfect = new boolean[N],visited = new boolean[N];
    for (int i:I)
      isInfect[i]=true;
    for (int i=0;i<N;i++)
      sources[i]=new ArrayList<>();
    for (int i:I){
      Arrays.fill(visited,false);
      MMWShelper(i,i,G,visited,isInfect,sources);
    }
    int[] sourceImpact = new int[N];
    for (int i=0;i<N;i++)
      if ( !isInfect[i] && sources[i].size()==1)
        sourceImpact[sources[i].get(0)]++;
    for (int i:I)
      if (sourceImpact[i]>ansCount ||(sourceImpact[i]==ansCount && i<ans) ){
        ansCount = sourceImpact[i];
        ans = i;
      }
    return ans;
  }

  private void MMWShelper(int idx,int sourceIdx,int[][] G,boolean[] visited,boolean[] isInfect,List<Integer>[] sources){
    visited[idx]=true;
    sources[idx].add(sourceIdx);
    int N = G.length;
    for (int i=0;i<N;i++)
      if (G[idx][i]==1 && ! visited[i] && !isInfect[i])
        MMWShelper(i,sourceIdx,G,visited,isInfect,sources);
  }

  public int minMalwareSpread4(int[][] G, int[] I) {
    if (I.length == 1)
      return I[0];
    int N = G.length,ans = Integer.MAX_VALUE,cnt = 0;
    boolean[] isInfect = new boolean[N],visited;
    for (int i:I)
      isInfect[i]=true;
    for (int i:I){
      visited = new boolean[N];
      visited[i]=true;
      int rescue = 0;
      for (int j=0;j<N;j++){
        if (G[i][j]==0 || visited[j] || isInfect[j])
          continue;
        int curRescur = MMWSgetRecureNum(j,G,visited,isInfect);
        if (curRescur!=-1)
          rescue+=curRescur;
      }
      if (rescue>cnt || (rescue==cnt && i<ans)){
        ans = i;
        cnt = rescue;
      }
    }
    return ans;
  }

  private int MMWSgetRecureNum(int idx,int[][] G,boolean[] visited,boolean[] isInfect){
    if (visited[idx])
      return 0;
    if (isInfect[idx])
      return -1;
    visited[idx] = true;
    int rescue = 1,tempRes;
    for (int i=0;i<G.length;i++)
      if (G[idx][i]==1){
        tempRes = MMWSgetRecureNum(i,G,visited,isInfect);
        if (tempRes==-1){
          isInfect[idx]=true;
          return -1;
        }
        else{
          rescue += tempRes;
        }
      }
    return rescue;
  }

  class FMSARange{
    int start;
    int end;
    public FMSARange(int s,int e){
      start = s;
      end = e;
    }
  }

  public double findMedianSortedArrays(int[] N, int[] M) {
    int NLen = N.length,MLen = M.length,totalLen = NLen+MLen,L,R,i,j;
    double maxL,minR;
    int[] SM,BG;
    if (NLen <=MLen){
      SM = N;
      BG = M;
    }
    else{
      SM = M;
      BG = N;
    }
    L = 0;
    R = SM.length;
    while (L <= R){
      i = (L+R)>>1;
      j = ((totalLen+1)>>1)-i;
      if (i<SM.length && BG[j-1]>SM[i])
        L = i+1;
      else if (i>0 && SM[i-1] >BG[j])
        R = i-1;
      else{
        maxL = i==0?BG[j-1]:j==0?SM[i-1]:Math.max(BG[j-1],SM[i-1]);
        if ((totalLen & 1) == 1)
          return maxL;
        minR = i==SM.length?BG[j]:j==BG.length?SM[i]:Math.min(BG[j],SM[i]);
        return (maxL + minR)/2;
      }
    }
    return 0;
  }

  public int[] searchRange(int[] nums, int target) {
    int[] ans = new int[]{-1,-1};
    if (nums == null || nums.length==0)
      return ans;
    int N = nums.length,L = 0,R = N-1;
    while (L <=R){
      int mid = (L+R)>>1;
      if (nums[mid] >= target)
        R = mid-1;
      else
        L = mid + 1;
    }
    if (L >= N || nums[L]!=target )
      return ans;
    ans[0] =ans[1]= L;
    while (ans[1]<N && nums[ans[1]]==target)
      ans[1]++;
    ans[1]--;
    return ans;
  }

  public int findMin2(int[] nums) {
    if (nums == null || nums.length == 0)
      throw new IllegalArgumentException();
    if (nums.length==1)
      return nums[0];
    int N = nums.length,L = 0,R = N-1;
    while (L < R){
      int mid = (L+R)>>1;
      if (nums[mid] > nums[R])
        L = mid+1;
      else
        R = mid;
    }
    return nums[R];
  }

  public List<List<String>> partition2(String s) {
    List<List<String>> ans = new ArrayList<>();
    if (s == null || s.isEmpty())
      return ans;
    int N = s.length();
    char[] cs = s.toCharArray();
    boolean[][] isPal = new boolean[N][N];
    for (int i =0;i < N ;i++)
      for (int j = 0;j<=i;j++)
        if ( cs[i] == cs[j] && (i-j<=2 || isPal[j+1][i-1]))
          isPal[j][i] = true;
    P2helper(0,new ArrayList<>(),ans,isPal,s);
    return ans;
  }

  private void P2helper(int idx,List<String> path,List<List<String>> res, boolean[][] isPal,String s){
    if (idx == s.length()){
      res.add(new ArrayList<>(path));
      return;
    }
    int N = s.length();
    for (int i=idx;i<N;i++)
      if (isPal[idx][i]){
        path.add(s.substring(idx,i+1));
        P2helper(i+1,path,res,isPal,s);
        path.remove(path.size()-1);
      }
  }

  class StreamChecker3 {

    class Node{
      boolean isWord;
      Node[] next;

      public Node(){
        isWord = false;
        next = new Node[26];
      }
    }

    private void addTrie(String s){
      addTrie(root,s,s.length()-1);
    }

    private void addTrie(Node root,String s,int depth){
      if (depth < 0){
        root.isWord = true;
        return;
      }
      int idx = s.charAt(depth)-'a';
      if (root.next[idx]==null)
        root.next[idx] = new Node();
      addTrie(root.next[idx],s,depth-1);
    }

    private Node root;
    char[] q;
    int curLen,qStart;

    public StreamChecker3(String[] words) {
      root = new Node();
      qStart=curLen=0;
      int maxLen=0;
      for (String word:words){
        addTrie(word);
        maxLen = Math.max(maxLen,word.length());
      }
      q = new char[maxLen];
    }

    private void addQueue(char L){
      int maxLen = q.length;
      if (curLen <maxLen){
        q[qStart] = L;
        curLen++;
      }
      else{
        q[qStart] = L;
      }
      qStart = (qStart+1)%maxLen;
    }

    public boolean query(char letter) {
      addQueue(letter);
      Node cur = root;
      int idx,alph,maxLen = q.length;
      for (int i = 0;i<curLen;i++){
        idx = (qStart-i+maxLen-1) %maxLen;
        alph = q[idx]-'a';
        if (cur.next[alph]==null)
          return false;
        else if (cur.next[alph].isWord)
          return true;
        else
          cur = cur.next[alph];
      }
      return false;
    }
  }

  public int countPrimes(int n) {
    if (n<0)
      throw new IllegalArgumentException();
    if (n<=2)
      return 0;
    boolean[] isNotPrime = new boolean[n];
    int ans = 0;
    for (int i = 2;i<n;i++)
      if (!isNotPrime[i]) {
        ans++;
        int times = 2, idx;
        while ((idx = times * i) < n) {
          isNotPrime[idx] = true;
          times++;
        }
      }
    return ans;
  }

  class Node138 {
    public int val;
    public Node138 next;
    public Node138 random;

    public Node138() {}

    public Node138(int _val,Node138 _next,Node138 _random) {
      val = _val;
      next = _next;
      random = _random;
    }
  }

  public Node138 copyRandomList(Node138 head) {
    if (head == null)
      return null;
    Map<Node138,Node138> record = new HashMap<>();
    return CRLhelper(head,record);
  }

  private Node138 CRLhelper(Node138 head,Map<Node138,Node138> record){
    if (head == null)
      return null;
    Node138 cur;
    if ((cur = record.get(head))!=null)
      return cur;
    cur = new Node138();
    record.put(head,cur);
    cur.val = head.val;
    cur.next = CRLhelper(head.next,record);
    cur.random = CRLhelper(head.random,record);
    return cur;
  }

  public List<Integer> findDisappearedNumbers3(int[] nums) {
    List<Integer> ans = new ArrayList<>();
    if (nums== null ||nums.length==0)
      return ans;
    for (int i=0;i<nums.length;i++)
      if (nums[Math.abs(nums[i])-1]>0)
        nums[Math.abs(nums[i])-1] *=-1;
    for (int i=0;i<nums.length;i++)
      if (nums[i]>0)
        ans.add(i+1);
    return ans;
  }

  public boolean searchMatrix2(int[][] M, int target) {
    if (M == null || M.length == 0 || M[0].length==0)
      return false;
    return SM2helper(M,target,0,M.length-1,0,M[0].length-1);
  }

  private boolean SM2helper(int[][] M,int target,int rs,int re,int cs,int ce){
    if ( rs <0 || re>=M.length || cs<0 || ce>= M[0].length || rs >re || cs > ce)
      return false;
    if (rs == re && cs == ce)
      return M[rs][cs]==target;
    int rm= (rs + re)>>1,cm = (cs+ce)>>1;
    if (M[rm][cm] <target)
      return SM2helper(M,target,rs,rm,cm+1,ce) || SM2helper(M,target,rm+1,re,cs,cm) ||SM2helper(M,target,rm+1,re,cm+1,ce);
    else if (M[rm][cm] > target)
      return SM2helper(M,target,rs,rm-1,cm,ce) || SM2helper(M,target,rm,re,cs,cm-1) ||SM2helper(M,target,rs,rm-1,cs,cm-1);
    else
      return true;
  }

  public boolean searchMatrix3(int[][] M, int target) {
    if (M == null || M.length == 0 || M[0].length == 0 || target< M[0][0] || target>M[M.length-1][M[0].length-1])
      return false;
    int R = 0,C = M[0].length-1;
    while (R < M.length && C >=0)
      if (M[R][C]<target)
        R++;
      else if (M[R][C] > target)
        C--;
      else
        return true;
    return false;
  }

  public int countRangeSum2(int[] nums, int lower, int upper) {
    if (nums == null || nums.length ==0 )
      return 0;
    long[] ps = new long[nums.length+1],aux = new long[nums.length+1];
    for (int i = 0;i < nums.length;i++)
      ps[i+1] = ps[i]+nums[i];
    return CRS2mergeSort(ps,aux,0,nums.length,lower,upper);
  }

  private int CRS2mergeSort(long[] ps,long[] aux,int start,int end,int lower,int upper){
    if (start >= end)
      return 0;
    int mid = (start + end) >> 1;
    int left = CRS2mergeSort(ps, aux,start, mid, lower, upper), right = CRS2mergeSort(ps,aux, mid + 1, end, lower, upper);
    return left + right + CRS2merge(ps,aux,start, end, lower, upper);
  }

  private int CRS2merge(long[] ps,long[] aux,int start,int end,int lower,int upper){
    for (int i = start;i<=end;i++)
      aux[i] = ps[i];
    int mid = (start + end)>>1,ans = 0,L = mid+1,R = mid+1,sm = mid+1,idx = start;
    for (int i = start;i<=mid;i++){
      while (L <=end && (aux[L]-aux[i])<lower)
        L++;
      while (R <=end && (aux[R]-aux[i])<=upper)
        R++;
      while (sm <= end && aux[sm]<=aux[i])
        ps[idx++] = aux[sm++];
      ps[idx++] = aux[i];
      ans += R-L;
    }
    return ans;
  }

  public ListNode deleteDuplicates2(ListNode head) {
    if (head == null)
      return head;
    ListNode slow=head,fast,first = new ListNode(0),pre = first;
    first.next = head;
    while (slow!=null){
      fast = slow.next;
      if (fast!=null && fast.val==slow.val){
        while (fast!=null && fast.val==slow.val)
          fast = fast.next;
        pre.next = fast;
      }
      else
        pre = slow;
      slow = fast;
    }
    return first.next;
  }

  public ListNode rotateRight(ListNode head, int k) {
    if (k<0)
      throw new IllegalArgumentException();
    if (head ==null || head.next==null || k==0)
      return head;
    ListNode end = head,start=head,temp;
    int len,realK,leftRotation;
    for (len = 1;end.next!=null;len++)
      end = end.next;
    realK = k%len;
    if (realK==0)
      return head;
    leftRotation = len - realK;
    for (int i=0;i<leftRotation-1;i++)
      start = start.next;
    temp = start.next;
    start.next=null;
    end.next=head;
    return temp;
  }

  public int[] gardenNoAdj2(int N, int[][] paths) {
    if (N < 1)
      throw new IllegalArgumentException();
    if (N==1)
      return new int[]{1};
    int[] flower = new int[N];
    List<Integer>[] graph = new List[N];
    for (int i = 0; i < N; i++)
      graph[i] = new ArrayList<>();
    for (int[] p : paths) {
      graph[p[0] - 1].add(p[1] - 1);
      graph[p[1] - 1].add(p[0] - 1);
    }
    boolean[] used = new boolean[5];
    for (int i = 0; i < N; i++)
      if (flower[i] == 0) {
        Arrays.fill(used,false);
        for (int adj:graph[i])
          used[flower[adj]] = true;
        for (int j=1;j<5;j++)
          if (!used[j]){
            flower[i] = j;
            break;
          }
      }
    return flower;
  }

  class MAFEdge{
    int[] n1,n2;
    int midX,midY,len;
    public MAFEdge(int[] n1,int[] n2){
      if (n1[0]<=n2[0]){
        this.n1 = n1;
        this.n2 = n2;
      }
      else{
        this.n1 = n2;
        this.n2 = n1;
      }
      midX = n1[0]+n2[0];
      midY = n1[1]+n2[1];
      len = (n1[0]-n2[0]) * (n1[0]-n2[0])+(n1[1]-n2[1]) * (n1[1]-n2[1]);
    }
  }

  public double minAreaFreeRect2(int[][] P) {
    if (P == null || P.length==0)
      return 0;
    Map<String,List<MAFEdge>> edges = new HashMap<>();
    int N = P.length;
    double minArea = Double.MAX_VALUE;
    for (int i=0;i<N;i++)
      for (int j = i+1;j<N;j++){
        MAFEdge edge = new MAFEdge(P[i],P[j]);
        String key = edge.len+" "+edge.midX+" "+edge.midY;
        List<MAFEdge> same;
        if ((same = edges.get(key))==null){
          same = new ArrayList<>();
          edges.put(key,same);
        }
        same.add(edge);
      }
    for (List<MAFEdge> same:edges.values()){
      if (same.size()<2)
        continue;
      int size = same.size();
      for (int i=0;i<size;i++)
        for (int j=i+1;j<size;j++){
          MAFEdge e1 = same.get(i),e2 = same.get(j);
          double Xdiff11 =(double)e1.n1[0]-(double)e2.n1[0],Ydiff11 = (double)e1.n1[1]-(double)e2.n1[1],len11 = Math.sqrt(Xdiff11*Xdiff11+Ydiff11*Ydiff11);
          double Xdiff12 = (double)e1.n1[0]-(double)e2.n2[0],Ydiff12 = (double)e1.n1[1]-(double)e2.n2[1],len12 = Math.sqrt(Xdiff12*Xdiff12+Ydiff12*Ydiff12);
          double area = len11*len12;
          minArea = Math.min(minArea,area);
        }
    }
    return minArea == Double.MAX_VALUE?0:minArea;
  }

  public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
    int[] N,M;
    if (nums1.length<= nums2.length){
      N = nums1;
      M = nums2;
    }
    else{
      N = nums2;
      M = nums1;
    }
    int NL = N.length,ML = M.length,half = (NL+ML+1)>>1,i,j,L=0,R = NL;
    double minL=0,maxR=0;
    while (L <= R){
      i = (L+R)>>1;
      j = half - i;
      if (i>0 && j<ML && N[i-1]> M[j])
        R = i-1;
      else if (j>0  && i<NL && N[i] < M[j-1])
        L = i+1;
      else{
        if (i==0)
          minL = (double)M[j-1];
        else if (j==0)
          minL = (double)N[i-1];
        else
          minL = (double)Math.max(N[i-1],M[j-1]);
        if (((NL+ML) & 1)==1)
          return minL;
        if (i==NL)
          maxR = M[j];
        else if (j==ML)
          maxR = N[i];
        else
          maxR = (double)Math.min(N[i],M[j]);
        return (minL+maxR)/2;
      }
    }
    return 0;
  }

  public String addBinary(String a, String b) {
    if (a==null ||a.isEmpty())
      return b;
    if (b==null || b.isEmpty())
      return a;
    char[] ca = a.toCharArray(), cb = b.toCharArray();
    StringBuilder sb = new StringBuilder();
    boolean carry = false;
    int i = ca.length-1,j=cb.length-1;
    while (i>=0 || j>=0 || carry){
      int aVal = i>=0?ca[i]-'0':0;
      int bVal = j>=0?cb[j]-'0':0;
      int val = carry?aVal + bVal+1:aVal+bVal;
      sb.append(val &1);
      carry = (val>>1)==1;
      i--;
      j--;
    }
    return sb.reverse().toString();
  }

  public boolean isNumber(String s) {
    return s.matches("\\s*(\\+|-)?(\\d+(\\.\\d*)?|\\.\\d+)(e(\\+|-)?\\d+)?\\s*?");
  }

  public boolean validSquare2(int[] p1, int[] p2, int[] p3, int[] p4) {
    return VS2helper(p1,p2,p3,p4) || VS2helper(p1,p3,p2,p4) || VS2helper(p1,p4,p2,p3);
  }

  private boolean VS2helper(int[] d11,int[] d12,int[] d21,int[] d22){
    int dist = VS2getDistance(d11,d21);
    if (dist==0)
      return false;
    if ( VS2getDistance(d11,d22)==dist && VS2getDistance(d22,d12)==dist
            && VS2getDistance(d21,d12)==dist && VS2getDistance(d11,d12)==VS2getDistance(d21,d22) )
      return true;
    return false;
  }

  private int VS2getDistance(int[] p1,int[] p2){
    int xd = p1[0]-p2[0],yd = p1[1]-p2[1];
    return xd*xd+yd*yd;
  }

  public int maxArea2(int[] height) {
    int start = 0,end = height.length-1,max = Integer.MIN_VALUE;
    while (start<end){
      max = Math.max(max,(end-start)*Math.min(height[start],height[end]));
      if (height[start]<=height[end])
        start++;
      else
        end--;
    }
    return max;
  }

  class Twitter2 {

    class User{
      int id;
      Set<Integer> follower;
      Post post;

      public User(int id){
        this.id = id;
        follower = new HashSet<>();
        post = null;
        follow(this.id);
      }

      public void follow(int id){
        follower.add(id);
      }

      public void unfollow(int id){
        if (!follower.contains(id))
          return;
        follower.remove(id);
      }

      public void post(int postId){
        Post cur = new Post(postId,getTime());
        if (post == null)
          post = cur;
        else{
          cur.next = post;
          post = cur;
        }
      }
    }

    class Post{
      int id;
      int time;
      Post next;

      public Post(int id,int time){
        this.id = id;
        this.time = time;
        next = null;
      }
    }

    private int getTime(){
      return time++;
    }

    Map<Integer,User> users;
    int time;

    /** Initialize your data structure here. */
    public Twitter2() {
      users = new HashMap<>();
      time = 0;
    }

    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
      User user;
      if ((user = users.get(userId))== null){
        user = new User(userId);
        users.put(userId,user);
      }
      user.post(tweetId);
    }

    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
      List<Integer> ans = new ArrayList<>(10);
      User user;
      if ((user = users.get(userId))== null)
        return ans;
      PriorityQueue<Post> pq = new PriorityQueue<>((a,b)->b.time-a.time);
      for (int followee:user.follower){
        Post temp = users.get(followee).post;
        if (temp!=null)
          pq.offer(temp);
      }
      while (!pq.isEmpty() && ans.size()<10){
        Post cur = pq.poll();
        ans.add(cur.id);
        if (cur.next!=null)
          pq.offer(cur.next);
      }
      return ans;
    }

    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
      User user;
      if (followeeId == followerId)
        return;
      if ((user = users.get(followerId))== null ){
        user = new User(followerId);
        users.put(followerId,user);
      }
      if (!users.containsKey(followeeId))
        users.put(followeeId,new User(followeeId));
      user.follow(followeeId);
    }

    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
      User user;
      if (followeeId == followerId ||(user = users.get(followerId))== null)
        return;
      user.unfollow(followeeId);
    }
  }

  class Solution384_ {

    private Random r;
    private int[] data,shuffled;

    public Solution384_(int[] nums) {
      r = new Random();
      data = nums;
      shuffled = Arrays.copyOf(data,nums.length);
    }

    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
      System.arraycopy(data,0,shuffled,0,data.length);
      return shuffled;
    }

    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
      for (int i=shuffled.length-1;i>=0;i--)
        exchange(shuffled,r.nextInt(i+1),i);
      return shuffled;
    }
  }

  public int subarrayBitwiseORs2(int[] A) {
    if (A == null ||A.length==0)
      return 0;
    Set<Integer> res=new HashSet<>(),cur,last = new HashSet<>();
    for (int a:A){
      cur = new HashSet<>();
      cur.add(a);
      for (int i:last)
        cur.add(a|i);
      res.addAll(cur);
      last = cur;
    }
    return res.size();
  }

  public List<String> findItinerary2(List<List<String>> T) {
    LinkedList<String> ans = new LinkedList<>();
    if (T==null || T.size()==0)
      return ans;
    Map<String,PriorityQueue<String>> graph = new HashMap<>();
    FI2constructGraph(T,graph);
    if (graph.containsKey("JFK"))
      HierholzerDFS(graph,"JFK",ans);
    return ans;
  }

  private void HierholzerDFS(Map<String,PriorityQueue<String>> graph,String cur,LinkedList<String> res){
    PriorityQueue<String> pq=graph.get(cur);
    while (pq!=null && !pq.isEmpty())
      HierholzerDFS(graph,pq.poll(),res);
    res.addFirst(cur);
  }

  private void FI2constructGraph(List<List<String>> T,Map<String,PriorityQueue<String>> graph){
    for (List<String> path:T){
      PriorityQueue<String> next;
      String src = path.get(0),dest = path.get(1);
      if ((next=graph.get(src))==null){
        next = new PriorityQueue<>();
        graph.put(src,next);
      }
      next.offer(dest);
    }
  }

  public ListNode detectCycle(ListNode head) {
    if (head == null)
      return null;
    ListNode slow = head,fast = head.next;
    while (fast!=null && slow != fast){
      fast = fast.next;
      if (fast==null)
        break;
      fast = fast.next;
      slow = slow.next;
    }
    if (fast == null)
      return null;
    ListNode p1 = head,p2 = slow.next;
    while (p1 !=p2){
      p1 = p1.next;
      p2 = p2.next;
    }
    return p1;
  }

  public int minSubArrayLen(int s, int[] nums) {
    if (s<=0 || nums ==null ||nums.length==0)
      return 0;
    int start=0,end=0,N = nums.length,sum=nums[0],len=Integer.MAX_VALUE;
    while (true)
      if (sum>=s){
        len = Math.min(len,end-start+1);
        sum -= nums[start];
        start++;
      }
      else{
        end++;
        if (end == N)
          break;
        sum += nums[end];
      }
    return len==Integer.MAX_VALUE?0:len;
  }

  public int minSubArrayLen2(int s, int[] nums) {
    if (s <= 0 || nums == null || nums.length == 0)
      return 0;
    int N =nums.length,min = Integer.MAX_VALUE,end;
    int[] ps = new int[N+1];
    for (int i=0;i<N;i++)
      ps[i+1] = ps[i]+nums[i];
    for (int i = 0;i<N;i++){
      end = MSAbinarySearch(ps,i,s+ps[i]);
      if (end >N)
        break;
      else
        min = Math.min(min,end-i);
    }
    return min == Integer.MAX_VALUE?0:min;
  }

  private int MSAbinarySearch(int[] ps,int start,int target){
    int L = start,R = ps.length-1,mid;
    while (L <= R){
      mid = (L+R)>>1;
      if (ps[mid]>=target)
        R = mid-1;
      else
        L = mid+1;
    }
    return L;
  }

  public List<List<Integer>> findSubsequences2(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    if (nums==null ||nums.length==0)
      return ans;
    FSS2helper(nums,0,new ArrayList<>(),ans);
    return ans;
  }

  private void FSS2helper(int[] nums,int start,List<Integer> path,List<List<Integer>> res){
    if (path.size()>=2)
      res.add(new ArrayList<>(path));
    Set<Integer> appear = new HashSet<>();
    for (int i=start;i<nums.length;i++)
      if (!appear.contains(nums[i]) && (path.isEmpty() || nums[i]>=path.get(path.size()-1))){
        appear.add(nums[i]);
        path.add(nums[i]);
        FSS2helper(nums,i+1,path,res);
        path.remove(path.size()-1);
      }
  }

  public int[] nextGreaterElements3(int[] nums) {
    if (nums==null || nums.length==0)
      return nums;
    int N =nums.length,idx=0,FN = N<<1;
    int[] stack=new int[FN],ans=new int[N];
    Arrays.fill(ans,-1);
    for (int i=0;i<FN;i++){
      int realI =  i%N;
      if (idx>0 && nums[stack[idx-1]]<nums[realI])
        while (idx >0 && nums[stack[idx-1]]<nums[realI])
          ans[stack[--idx]] = nums[realI];
      if (ans[realI]==-1)
        stack[idx++] = realI;
    }
    return ans;
  }

  public boolean canFinish2(int N, int[][] P) {
    if (N==0 || P.length==0)
      return true;
    List<Integer>[] graph = CF2constructGraph(P,N);
    boolean[] hasCycle = new boolean[1],visited = new boolean[N],onPath = new boolean[N];
    for (int i=0;i<N;i++)
      if (!visited[i]){
        CF2cycleDetector(graph,i,visited,onPath,hasCycle);
        if (hasCycle[0])
          break;
      }
    return !hasCycle[0];
  }

  private List<Integer>[] CF2constructGraph(int[][] P,int N){
    List<Integer>[] graph = new List[N];
    for (int i=0;i<N;i++)
      graph[i]=new ArrayList<>();
    for (int[] p:P)
      graph[p[1]].add(p[0]);
    return graph;
  }

  private void CF2cycleDetector(List<Integer>[] graph,int cur,boolean[] visited,boolean[] onPath,boolean[] hasCycle){
    visited[cur] = true;
    onPath[cur] = true;
    for (int next:graph[cur])
      if (hasCycle[0])
        return;
      else if (!visited[next])
        CF2cycleDetector(graph,next,visited,onPath,hasCycle);
      else if (onPath[next]){
        hasCycle[0] = true;
        return;
      }
    onPath[cur] = false;
  }

  public int[] findOrder2_BFS(int N, int[][] P) {
    if (N==0)
      return new int[0];
    int[] inDegree = new int[N],ans = new int[N];
    int idx=0;
    Queue<Integer> q = new LinkedList<>();
    List<Integer>[] G = new List[N];
    for (int i=0;i<N;i++)
      G[i]=new ArrayList<>();
    for (int[] p:P){
      G[p[1]].add(p[0]);
      inDegree[p[0]]++;
    }
    for (int i=0;i<N;i++)
      if (inDegree[i]==0)
        q.offer(i);
    while (!q.isEmpty()){
       int cur = q.poll();
       ans[idx++]=cur;
       for (int next:G[cur])
         if (--inDegree[next]==0)
           q.offer(next);
    }
   return idx==N?ans:new int[0];
  }

  public int[] findOrder2_DFS(int N, int[][] P) {
    if (N == 0)
      return new int[0];
    int[]  ans = new int[N],idx=new int[]{N-1};
    List<Integer>[] G = new List[N];
    for (int i = 0; i < N; i++)
      G[i] = new ArrayList<>();
    for (int[] p : P)
      G[p[1]].add(p[0]);
    boolean[] visited = new boolean[N],onPath = new boolean[N],hasCycle = new boolean[1];
    for (int i=0;i<N;i++)
      if (!visited[i])
        FO2cycleDetector(G,i,visited,onPath,hasCycle);
    if (hasCycle[0])
      return new int[0];
    Arrays.fill(visited,false);
    for (int i=0;i<N;i++)
      if (!visited[i])
        FO2getTopo(G,i,visited,ans,idx);
    return ans;
  }

  private void FO2cycleDetector(List<Integer>[] G,int cur,boolean[] visited,boolean[] onPath,boolean[] hasCycle){
    visited[cur]=true;
    onPath[cur] = true;
    for (int next:G[cur])
      if (hasCycle[0])
        return;
      else if (!visited[next])
        FO2cycleDetector(G,next,visited,onPath,hasCycle);
      else if (onPath[next]){
        hasCycle[0]=true;
        return;
      }
    onPath[cur] = false;
  }

  private void FO2getTopo(List<Integer>[] G,int cur,boolean[] visited,int[] res,int[] idx){
    visited[cur]=true;
    for (int next:G[cur])
      if (!visited[next])
        FO2getTopo(G,next,visited,res,idx);
    res[idx[0]--] = cur;
  }

  public int[][] merge2(int[][] I) {
    if (I == null || I.length==0)
      return new int[0][2];
    Arrays.sort(I, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]-b[0];
      }
    });
    int N = I.length;
    List<int[]> res = new ArrayList<>();
    res.add(I[0]);
    for (int i=1;i<N;i++){
      int[] last = res.get(res.size()-1);
      if (last[1]>=I[i][0])
        last[1] = Math.max(last[1],I[i][1]);
      else
        res.add(I[i]);
    }
    return res.toArray(new int[res.size()][2]);
  }

  class IPrecord{
    int val,len;
    public IPrecord(int v,int L){
      this.val = v;
      this.len = L;
    }
  }

  public boolean isPossible2(int[] nums) {
    if (nums==null || nums.length<3)
      return false;
    Map<Integer,Integer> count=new HashMap<>(),append = new HashMap<>();
    int N=nums.length,apd,freq;
    for (int i=0;i<N;i++)
      count.put(nums[i],count.getOrDefault(nums[i],0)+1);
    for (int i:nums){
      if ((freq=count.get(i))== 0)
        continue;
      else if ( (apd=append.getOrDefault(i,0))>0){
        append.put(i,apd-1);
        append.put(i+1,append.getOrDefault(i+1,0)+1);
      }
      else if (count.getOrDefault(i+1,0)>0 && count.getOrDefault(i+2,0)>0){
        count.put(i+1,count.get(i+1)-1);
        count.put(i+2,count.get(i+2)-1);
        append.put(i+3,append.getOrDefault(i+3,0)+1);
      }
      else
        return false;
      count.put(i,freq-1);
    }
    return true;
  }

  public boolean isPossible3(int[] nums) {
    if (nums == null || nums.length < 3)
      return false;
    int N=nums.length,pre=Integer.MIN_VALUE,p1=0,p2=0,p3=0,cur=0,cnt,c1=0,c2=0,c3=0;
    for (int i=0;i<N;pre=cur,p1=c1,p2=c2,p3=c3){
      for (cur = nums[i],cnt = 0;i<N && nums[i]==cur;i++,cnt++);
      if (cur!=pre+1){
        if (p1!=0 || p2 !=0)
          return false;
        c1 =cnt;
        c2=c3=0;
      }
      else{
        if (cnt<p1+p2)
          return false;
        c1 = Math.max(0,cnt-p1-p2-p3);
        c2 = p1;
        c3 = p2+Math.min(p3,cnt-p1-p2);
      }
    }
    return p1==0 && p2==0;
  }

  public int divide2(int M, int N) {
    if (M==0)
      return 0;
    if (M== Integer.MIN_VALUE && N == -1)
      return Integer.MAX_VALUE;
    int shift,ans=0;
    boolean isPos = (M>0 && N>0) ||(M<0 && N<0);
    M = M>0?-M:M;
    N = N>0?-N:N;
    while (M<=N){
      shift = 0;
      while ((M>>shift>>1)<N)
        shift++;
      M -= N << shift;
      ans += 1<<shift;
    }
    return isPos?ans:-ans;
  }

  public void setZeroes2(int[][] M) {
    int R,C;
    if (M == null ||(R = M.length)==0 || (C = M[0].length)==0)
      return;
    boolean isR0Zero=false,isC0Zero = false;
    if (M[0][0]==0)
      isR0Zero=isC0Zero=true;
    for (int i=0;i<R;i++)
      if (M[i][0]==0){
        isC0Zero=true;
        break;
      }
    for (int i=0;i<C;i++)
      if (M[0][i]==0){
        isR0Zero=true;
        break;
      }
    for (int r=1;r<R;r++)
      for (int c=1;c<C;c++)
        if (M[r][c]==0)
          M[r][0] = M[0][c]=0;
    for (int r=1;r<R;r++)
      for (int c=1;c<C;c++)
        if (M[r][0]==0 || M[0][c]==0)
          M[r][c]=0;
    if (isR0Zero)
      for (int i=0;i<C;i++)
        M[0][i]=0;
    if (isC0Zero)
      for (int i=0;i<R;i++)
        M[i][0]=0;
  }

  public int rob2(TreeNode root) {
    if (root==null)
      return 0;
    Map<TreeNode,Integer> memo = new HashMap<>();
    return robHelper(root,memo);
  }

  private int robHelper(TreeNode root,Map<TreeNode,Integer> memo){
    if (root==null)
      return 0;
    Integer res;
    if ((res = memo.get(root))!=null)
      return res;
    int notRob = robHelper(root.left,memo)+robHelper(root.right,memo);
    int robRoot = root.val;
    if ( root.left!=null && root.right!=null)
      robRoot+= robHelper(root.left.left,memo)+robHelper(root.right.left,memo)+robHelper(root.left.right,memo)+robHelper(root.right.right,memo);
    else if (root.left !=null)
      robRoot += robHelper(root.left.left,memo)+robHelper(root.left.right,memo);
    else if (root.right !=null)
      robRoot += robHelper(root.right.left,memo)+robHelper(root.right.right,memo);
    res = Math.max(notRob,robRoot);
    memo.put(root,res);
    return res;
  }

  public int rob3(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = R3helper(root);
    return Math.max(res[0],res[1]);
  }

  private int[] R3helper(TreeNode root){
    if (root==null)
      return new int[2];
    int[] left = R3helper(root.left),right = R3helper(root.right),res = new int[2];
    res[0] = Math.max(left[0],left[1])+Math.max(right[0],right[1]);
    res[1] = root.val+left[0]+right[0];
    return res;
  }

  public int threeSumClosest2(int[] nums, int target) {
    if (nums==null || nums.length<3)
      throw new IllegalArgumentException();
    Arrays.sort(nums);
    int N = nums.length,ans=nums[0]+nums[1]+nums[2],curT,L,R,curSum;
    for (int i=0;i<N-2;i++){
      curT = target-nums[i];
      L = i+1;
      R = N-1;
      while (L < R){
        curSum = nums[L]+nums[R];
        if (Math.abs(curSum-curT) < Math.abs(target-ans))
          ans = curSum+nums[i];
        if (curSum<curT)
          L++;
        else if (curSum > curT)
          R--;
        else
          return ans;
      }
    }
    return ans;
  }

  public class Digraph{
    public final int V; // vertex number
    public int E;  // edge number
    private List<Integer>[] adj;

    public Digraph(int V){
      this.V = V;
      this.E = 0;
      adj = new List[V];
      for (int i=0;i<V;i++)
        adj[i] = new ArrayList<>();
    }

    public void addEdge(int v,int w){
      adj[v].add(w);
      E++;
    }
  }

  public class DirectedDFS{
    private boolean[] visited;
    public DirectedDFS(Digraph G,int s){
      visited = new boolean[G.V];
      dfs(G,s);
    }

    public DirectedDFS(Digraph G,List<Integer> sources){
      visited = new boolean[G.V];
      for (int s:sources)
        if (!visited[s])
          dfs(G,s);
    }

    private void dfs(Digraph G,int v){
      visited[v]=true;
      for (int adj:G.adj[v])
        if (!visited[adj])
          dfs(G,adj);
    }

    public boolean isVisited(int v){
      return visited[v];
    }
  }

  public class NFA{
    private char[] re;
    private int M;
    private Digraph G;

    public NFA(String reg){
      re = reg.toCharArray();
      M = re.length;
      G = new Digraph(M+1);
      for (int i=0;i<M;i++){
        int lp = i; // if * show up,lp is the position starting to repeat
        if (i<M-1 && re[i+1]=='*'){
          G.addEdge(lp,i+1);
          G.addEdge(i+1,lp);
        }
        if (re[i]=='*')
          G.addEdge(i,i+1);
      }
    }

    public boolean recognizes(String txt){
      List<Integer> reached = new ArrayList<>(),match=new ArrayList<>();
      DirectedDFS dfs = new DirectedDFS(G,0);
      for (int v=0;v<G.V;v++)
        if (dfs.isVisited(v))
          reached.add(v);
      for (int i=0;i<txt.length();i++){
        for (int v:reached)
          if (v<M && (re[v]==txt.charAt(i) || re[v]=='.'))
            match.add(v+1);
        reached.clear();
        dfs = new DirectedDFS(G,match);
        for (int v=0;v<G.V;v++)
          if (dfs.isVisited(v))
            reached.add(v);
        match.clear();
      }
      for (int v:reached)
        if (v==M)
          return true;
      return false;
    }
  }

  public boolean isMatch(String s, String p) {
    if (s==null || p==null)
      return false;
    NFA nfa = new NFA(p);
    return nfa.recognizes(s);
  }

  public boolean isMatch2(String s, String p) {
    if (s == null || p == null)
      return false;
    int sn = s.length(),pn = p.length();
    boolean[][] dp = new boolean[sn+1][pn+1];
    dp[0][0] = true;
    for (int i=2;i<pn+1;i+=2)
      if (p.charAt(i-1)=='*')
        dp[0][i]=dp[0][i-2];

    for (int i=1;i<sn+1;i++)
      for (int j=1;j<pn+1;j++) {
        char curS = s.charAt(i - 1), curP = p.charAt(j - 1);
        if (curS == curP || curP == '.')
          dp[i][j] = dp[i - 1][j - 1];
        else if (curP == '*') {
          char preP = p.charAt(j - 2);
          if (preP != curS && preP != '.')
            dp[i][j] = dp[i][j - 2];
          else
            dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
        }
      }
    return dp[sn][pn];
  }

  public int[] gardenNoAdj3(int N, int[][] paths) {
    if (N <= 0)
      return new int[0];
    List<Integer>[] G = new List[N];
    for (int i=0;i<N;i++)
      G[i] = new ArrayList<>();
    for (int[] p:paths){
      G[p[0]-1].add(p[1]-1);
      G[p[1]-1].add(p[0]-1);
    }
    int[] colored = new int[N];
    boolean[] used = new boolean[5];
    for (int cur=0;cur<N;cur++){
      Arrays.fill(used,false);
      for (int adj:G[cur])
        used[colored[adj]] = true;
      for (int i=1;i<5;i++)
        if (!used[i]){
          colored[cur]=i;
          break;
        }
    }
    return colored;
  }

  class RandomizedSet2 {

    List<Integer> data;
    Map<Integer,Integer> record;

    /** Initialize your data structure here. */
    public RandomizedSet2() {
      data = new ArrayList<>();
      record = new HashMap<>();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
      if (record.containsKey(val))
        return false;
      record.put(val,data.size());
      data.add(val);
      return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
      Integer P;
      if ((P = record.get(val))==null)
        return false;
      if (P == data.size()-1)
        data.remove(P.intValue());
      else{
        int last = data.get(data.size()-1);
        data.set(P,last);
        data.remove(data.size()-1);
        record.put(last,P);
      }
      record.remove(val);
      return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
      return data.get((int)(Math.random()*data.size()));
    }
  }

  public String makeLargestSpecial2(String S) {
    if (S == null || S.length()==0)
      return S;
    List<String> res = new ArrayList<>();
    char[] cs = S.toCharArray();
    int N = cs.length,count = 0,start = 0;
    for (int i=0;i<N;i++){
      count += cs[i]=='1'?1:-1;
      if (count == 0){
        res.add('1'+makeLargestSpecial2(S.substring(start+1,i))+'0');
        start = i+1;
      }
    }
    Collections.sort(res,Collections.reverseOrder());
    StringBuilder sb = new StringBuilder();
    for (String s:res)
      sb.append(s);
    return sb.toString();
  }

  class Solution4782 {
    double x,y,r;
    Random random;
    public Solution4782(double radius, double x_center, double y_center) {
      r = radius;
      x = x_center;
      y = y_center;
      random = new Random();
    }

    //straight forward sample
    public double[] randPoint() {
      double angle = random.nextDouble()*2*Math.PI;
      double length = Math.sqrt(random.nextDouble())*r;
      return new double[]{length*Math.cos(angle)+x,length*Math.sin(angle)+y};
    }

    //reject sample
    public double[] randPoint2(){
      double randomX = random.nextDouble()*2*r,randomY = random.nextDouble()*2*r;
      double realX = randomX-r,realY = randomY-r;
      while (realX *realX +realY*realY >r*r){
        realX = random.nextDouble()*2*r-r;
        realY = random.nextDouble()*2*r-r;
      }
      return new double[]{realX+x,realY+y};
    }
  }

  public double largestSumOfAverages2(int[] A, int K) {
    int N = A.length;
    double[][] dp = new double[N][K+1];
    int[] ps = new int[N+1];
    for (int i=0;i<N;i++)
      ps[i+1] = ps[i]+A[i];
    return LSAhelper(ps,1,K,dp);
  }

  private double LSAhelper(int[] ps,int idx,int K,double[][] dp){
    int N =ps.length;
    if (idx == N || K ==0)
      return 0;
    if (K == 1)
      return (double)(ps[N-1]-ps[idx-1])/(double)(N-idx);
    if (dp[idx][K]!=0)
      return dp[idx][K];
    double res = Double.MIN_VALUE,temp;
    for (int i=idx;i<=N-K;i++){
      temp = (double)(ps[i]-ps[idx-1])/(double)(i-idx+1);
      temp += LSAhelper(ps,i+1,K-1,dp);
      res = Math.max(res,temp);
    }
    dp[idx][K] = res;
    return res;
  }

  class MyCalendarTwo2 {

    class SegmentTree{
      int L,R,maxOrder,lazy;
      SegmentTree left,right;

      public SegmentTree(int L,int R){
        this.L = L;
        this.R = R;
        lazy =maxOrder = 0;
        left = right = null;
      }

      public void pushDown(){
        if (L==R)
          return;
        int mid = (L+R)>>1;
        if (left == null)
          left = new SegmentTree(L,mid);
        if (right == null)
          right = new SegmentTree(mid,R);
        if (lazy == 0)
          return;
        left.updateByVal(lazy);
        right.updateByVal(lazy);
        lazy = 0;
      }

      public void updateByVal(int val){
        lazy += val;
        maxOrder += val;
      }

      public void updateFromSon(){
        maxOrder = Math.max(left.maxOrder,right.maxOrder);
      }
    }

    private int query(int start,int end,SegmentTree node){
      if (start<=node.L && end>= node.R)
        return node.maxOrder;
      if (start>= node.R || end<=node.L)
        return 0;
      node.pushDown();
      return Math.max(query(start,end,node.left),query(start,end,node.right));
    }

    private void update(int start,int end,SegmentTree node,int val){
      if (start <= node.L && end>=node.R){
        node.updateByVal(val);
        return;
      }
      if (start>=node.R || end <= node.L)
        return;
      node.pushDown();
      update(start,end,node.left,val);
      update(start,end,node.right,val);
      node.updateFromSon();
    }

    SegmentTree root;

    public MyCalendarTwo2() {
      root = new SegmentTree(0,1000000000);
    }

    public boolean book(int start, int end) {
      int max = query(start,end,root);
      if (max>=2)
        return false;
      update(start,end,root,1);
      return true;
    }
  }

  public List<Integer> spiralOrder(int[][] M) {
    List<Integer> ans = new ArrayList<>();
    if (M == null ||M.length==0 ||M[0].length==0)
      return ans;
    int total = M.length*M[0].length,rs = 0,re = M.length-1,cs = 0,ce = M[0].length-1,idx=0,label;
    while ( ans.size()<total){
      label = idx++%4;
      if (label == 0){
        for (int i=cs;i<=ce;i++)
          ans.add(M[rs][i]);
        rs++;
      }
      else if (label==1){
        for (int i=rs;i<=re;i++)
          ans.add(M[i][ce]);
        ce--;
      }
      else if (label == 2){
        for (int i=ce;i>=cs;i--)
          ans.add(M[re][i]);
        re--;
      }
      else{
        for (int i=re;i>=rs;i--)
          ans.add(M[i][cs]);
        cs++;
      }
    }
    return ans;
  }

  public boolean canJump(int[] nums) {
    if (nums==null ||nums.length==0)
      return false;
    int N = nums.length;
    boolean[] dp = new boolean[N];
    dp[N-1] = true;
    for (int i=N-2;i>=0;i--){
      for (int j=1;j<=nums[i];j++)
        if (dp[i+j]){
          dp[i]=true;
          break;
        }
    }
    return dp[0];
  }

  public boolean canJump1(int[] nums) {
    int max=0;
    for (int i=0;i<nums.length-1 && i<=max;i++)
      max = Math.max(max,i+nums[i]);
    return max>=nums.length-1;
  }

  class Solution3822 {

    private ListNode head;
    Random r;

    /** @param head The linked list's head.
    Note that the head is guaranteed to be not null, so it contains at least one node. */
    public Solution3822(ListNode head) {
      this.head = head;
      r = new Random();
    }

    /** Returns a random node's value. */
    public int getRandom() {
      int len = 1,ans= head.val,changeTo ;
      ListNode cur = head.next;
      while (cur!=null){
        changeTo = r.nextInt(++len);
        if (changeTo == 0)
          ans = cur.val;
        cur = cur.next;
      }
      return ans;
    }
  }

  class FCPedge{
    int from,to,cost;
    public FCPedge(int from,int to,int cost){
      this.from = from;
      this.to = to;
      this.cost = cost;
    }
  }

  class FCPreach{
    int pos,stop,cost;
    public FCPreach(int p,int stop,int cost){
      this.pos = p;
      this.stop =  stop;
      this.cost = cost;
    }
  }

  public int findCheapestPrice2(int n, int[][] flights, int src, int dst, int K) {
    if (src == dst)
      return 0;
    List<FCPedge>[] graph = new List[n];
    for (int i=0;i<n;i++)
      graph[i] = new ArrayList<>();
    for (int[] f:flights)
      graph[f[0]].add(new FCPedge(f[0],f[1],f[2]));
    PriorityQueue<FCPreach> pq = new PriorityQueue<>(new Comparator<FCPreach>() {
      @Override
      public int compare(FCPreach a, FCPreach b) {
        return a.cost-b.cost;
      }
    });
    pq.offer(new FCPreach(src,-1,0));
    while (!pq.isEmpty()){
      FCPreach cur = pq.poll();
      if (cur.pos==dst)
        return cur.cost;
      if (cur.stop==K)
        continue;
      for (FCPedge edge:graph[cur.pos])
        pq.offer(new FCPreach(edge.to,cur.stop+1,cur.cost+edge.cost));
    }
    return -1;
  }

  public int subarrayBitwiseORs3(int[] A) {
    Set<Integer> res=new HashSet<>(),last=new HashSet<>(),cur;
    for (int a:A){
      cur = new HashSet<>();
      cur.add(a);
      for (int l:last)
        cur.add(a|l);
      res.addAll(last = cur);
    }
    return res.size();
  }

  public int subarrayBitwiseORs4(int[] A) {
    if (A==null ||A.length==0)
      return 0;
    Set<Integer> res = new HashSet<>();
    int[] last=new int[33],cur = new int[33];
    int lastLen = 0,curLen;
    for (int a:A){
      curLen=0;
      res.add(cur[curLen++]=a);
      for (int i=0;i<lastLen;i++){
        int temp = a|last[i];
        if (temp!=a)
          res.add(cur[curLen++]=a=temp);
      }
      int[] t = last;
      last = cur;
      cur=t;
      lastLen = curLen;
    }
    return res.size();
  }

  public int minAreaRect1(int[][] P) {
    if (P == null || P.length<4)
      return 0;
    Map<Integer,Set<Integer>> pos = new HashMap<>();
    for (int[] p:P){
      Set<Integer> y;
      if ((y = pos.get(p[0])) == null){
        y = new HashSet<>();
        pos.put(p[0],y);
      }
      y.add(p[1]);
    }
    int min = Integer.MAX_VALUE;
    for (int[] p1:P)
      for (int[] p2:P)
        if (p1[0]==p2[0] || p1[1]==p2[1])
          continue;
        else if (pos.get(p1[0]).contains(p2[1]) && pos.get(p2[0]).contains(p1[1]))
          min = Math.min(min,Math.abs(p1[0]-p2[0])*Math.abs(p1[1]-p2[1]));
    return min==Integer.MAX_VALUE?0:min;
  }

  public boolean isValidSerialization2(String preorder) {
    if (preorder == null)
      return false;
    String[] po = preorder.split(",");
    int[] idx = new int[1];
    return IVShelper(po,idx) && idx[0] == po.length;
  }

  private boolean IVShelper(String[] po,int[] idx){
    if (idx[0] >= po.length)
      return false;
    idx[0]++;
    if (po[idx[0]-1].equals("#"))
      return true;
    else
      return IVShelper(po,idx) && IVShelper(po,idx);
  }

  public boolean isValidSerialization3(String preorder) {
    if (preorder == null)
      return false;
    String[] po = preorder.split(",");
    int diff = 1;
    for (String p:po)
      if (--diff<0)
        return false;
      else if (!p.equals("#"))
        diff +=2;
    return diff==0;
  }

  public int findMinArrowShots2(int[][] P) {
    if (P== null || P.length==0)
      return 0;
    Arrays.sort(P, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]-b[0];
      }
    });
    int L=P[0][0],R=P[0][1],ans=1;
    for (int[] p:P)
      if (L > p[1] || R<p[0]){
        ans++;
        L = p[0];
        R = p[1];
      }
      else{
        L = Math.max(p[0],L);
        R = Math.min(p[1],R);
      }
    return ans;
  }

  class SMRnode implements Comparable<SMRnode> {
    int val,idx,row;
    public SMRnode(int v,int i,int r){
      this.val = v;
      this.idx = i;
      this.row = r;
    }

    public int compareTo(SMRnode o) {
      return val -o.val;
    }
  }

  public int[] smallestRange2(List<List<Integer>> nums) {
    PriorityQueue<SMRnode> pq = new PriorityQueue<>();
    int curMax = Integer.MIN_VALUE,start=-1,end=-1,range = Integer.MAX_VALUE,N = nums.size();
    for (int i=0;i<N;i++){
      curMax = Math.max(curMax,nums.get(i).get(0));
      pq.offer(new SMRnode(nums.get(i).get(0),0,i));
    }
    while (pq.size() == N){
      SMRnode cur = pq.poll();
      if (curMax - cur.val < range){
        start = cur.val;
        end = curMax;
        range = curMax-cur.val;
      }
      if (cur.idx<nums.get(cur.row).size()-1){
        pq.offer(new SMRnode(nums.get(cur.row).get(cur.idx+1),cur.idx+1,cur.row));
        curMax = Math.max(curMax,nums.get(cur.row).get(cur.idx+1));
      }
    }
    return new int[]{start,end};
  }

  public double myPow(double x, int n) {
    long m = n;
    if (x == 0.0)
      return 0;
    if (x==1 || m ==0)
      return 1;
    if (m==1)
      return x;
    if (m<0){
      m = -m;
      x = 1/x;
    }
    return (m&1) == 1?x*myPow(x*x,(int)(m>>1)):myPow(x*x,(int)(m>>1));
  }

  public boolean searchMatrix4(int[][] M, int T) {
    int R,C;
    if (M == null || (R=M.length)==0 || (C=M[0].length)==0)
      return false;
    int start=0,end = R*C-1,mid;
    while (start<=end){
      mid = (start+end)>>1;
      int[] p = SMidxToPos(mid,C);
      if (M[p[0]][p[1]]<T)
        start = mid+1;
      else if (M[p[0]][p[1]]>T)
        end = mid-1;
      else
        return true;
    }
    return false;
  }

  private int[] SMidxToPos(int idx,int C){
    return new int[]{idx/C,idx%C};
  }

  public int minDominoRotations2(int[] A, int[] B) {
    int N = A.length, valA = A[0],valB = B[0],A1Count=0,A2Count=0,B1Count=0,B2Count=0;
    if (valA == valB){
      for (int i=0;i<N;i++)
        if (A[i]!= valA && B[i]!=valA)
          return -1;
        else{
          A1Count += A[i]==valA?1:0;
          B1Count += B[i]==valA?1:0;
        }
      return Math.min(N-A1Count,N-B1Count);
    }
    else{
      boolean isAllA=false,isAllB =false;
      for (int i=0;i<N;i++){
        if (A[i]==valA && B[i]==valA)
          isAllA = true;
        if (A[i] == valB && B[i] == valB)
          isAllB = true;
        A1Count += A[i]==valA?1:0;
        A2Count +=A[i] == valB?1:0;
        B1Count += B[i]== valA?1:0;
        B2Count += B[i]==valB?1:0;
      }
      int swapA,swapB;
      swapA = isAllB || A1Count+B1Count < N?Integer.MAX_VALUE:Math.min(N-A1Count,N-B1Count);
      swapB = isAllA || A2Count+B2Count < N?Integer.MAX_VALUE:Math.min(N-A2Count,N-B2Count);
      if (swapA == Integer.MAX_VALUE && swapB == Integer.MAX_VALUE)
        return -1;
      else
        return Math.min(swapA,swapB);
    }
  }

  public List<List<Integer>> subsetsWithDup2(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(nums);
    SWDhelper(nums,0,new ArrayList<>(),ans);
    return ans;
  }

  private void SWDhelper(int[] nums,int idx,List<Integer> path,List<List<Integer>> res){
    res.add(new ArrayList<>(path));
    for (int i=idx;i<nums.length;i++){
      if (i>idx &&i<nums.length&& nums[i]==nums[i-1])
        continue;
      path.add(nums[i]);
      SWDhelper(nums,i+1,path,res);
      path.remove(path.size()-1);
    }
  }

  public int minScoreTriangulation2(int[] A) {
    int N = A.length;
    int[][] dp = new int[N][N];
    for (int d=2;d<N;d++)
      for (int i=0;i+d<N;i++){
        int j = i+d;
        dp[i][j] = Integer.MAX_VALUE;
        for (int k=i+1;k<j;k++)
          dp[i][j] = Math.min(dp[i][j],A[i]*A[j]*A[k]+dp[i][k]+dp[k][j]);
      }
    return dp[0][N-1];
  }

  public int threeSumClosest3(int[] nums, int target) {
    if (nums == null || nums.length <3)
      throw new IllegalArgumentException();
    Arrays.sort(nums);
    int N = nums.length,start,end,ans = nums[0]+nums[1]+nums[2];
    for (int i=0;i<N-2;i++){
      start = i+1;
      end = N-1;
      while (start<end){
        int temp = nums[start]+nums[end]+nums[i];
        if (Math.abs(temp-target)<Math.abs(ans-target))
          ans = temp;
        if (temp < target)
          start++;
        else if (temp > target)
          end--;
        else
          return ans;
      }
    }
    return ans;
  }

  class Solution5192 {

    private int R,C,curRange;
    private Map<Integer,Integer> record;
    private Random r;

    public Solution5192(int n_rows, int n_cols) {
      R = n_rows;
      C = n_cols;
      curRange = R*C;
      record = new HashMap<>();
      r = new Random();
    }

    public int[] flip() {
      int key = r.nextInt(curRange--);
      int convert = record.getOrDefault(key,key);
      record.put(key,record.getOrDefault(curRange,curRange));
      return keyToPos(convert);
    }

    public void reset() {
      curRange = R*C;
      record.clear();
    }

    private int[] keyToPos(int key){
      return new int[]{key/C,key%C};
    }
  }

  public ListNode insertionSortList2(ListNode head) {
    if (head == null || head.next==null)
      return head;
    ListNode cur = head.next,last = head,fakeHead = new ListNode(Integer.MIN_VALUE),temp,tempLast,next;
    fakeHead.next = head;
    while (cur != null){
      next = cur.next;
      tempLast = fakeHead;
      temp = fakeHead.next;
      while (temp != cur && cur.val >= temp.val){
        tempLast = temp;
        temp = temp.next;
      }
      if (temp !=cur){
        tempLast.next = cur;
        cur.next = temp;
        last.next = next;
      }
      else
        last = cur;
      cur = next;
    }
    return fakeHead.next;
  }

  class GSnode implements Comparable<GSnode>{
    int pos,height;
    boolean isRight;

    public GSnode(int p,int h,boolean isR){
      pos = p;
      height = h;
      isRight = isR;
    }

    @Override
    public int compareTo(GSnode o) {
      if (pos != o.pos)
        return pos-o.pos;
      else if (isRight!= o.isRight)
        return isRight?1:-1;
      else if (isRight)
        return height-o.height;
      else
        return o.height-height;
    }
  }

  public List<List<Integer>> getSkyline(int[][] buildings) {
    List<List<Integer>> res = new ArrayList<>();
    List<GSnode> heights = new ArrayList<>();
    for (int[] b:buildings){
      heights.add(new GSnode(b[0],b[2],false));
      heights.add(new GSnode(b[1],b[2],true));
    }
    Collections.sort(heights);
    TreeMap<Integer,Integer> active = new TreeMap<>(Collections.reverseOrder());
    active.put(0,1);
    int prevHeight = 0;
    for (GSnode cur:heights){
      if (!cur.isRight)
        active.put(cur.height,active.getOrDefault(cur.height,0)+1);
      else{
        Integer cnt = active.get(cur.height);
        if (cnt == 1)
          active.remove(cur.height);
        else
          active.put(cur.height,cnt-1);
      }
      int curMax = active.firstKey();
      if (curMax!=prevHeight){
        List<Integer> temp = new ArrayList<>();
        temp.add(cur.pos);
        temp.add(curMax);
        res.add(temp);
        prevHeight = curMax;
      }
    }
    return res;
  }

  public List<List<Integer>> pathSum3(TreeNode root, int sum) {
    List<List<Integer>> res = new ArrayList<>();
    if (root==null)
      return res;
    PS2helper(root,sum,new ArrayList<>(),res);
    return res;
  }

  private void PS2helper(TreeNode cur,int remain,List<Integer> path,List<List<Integer>> res){
    path.add(cur.val);
    if (cur.left==null && cur.right == null){
      if (remain == cur.val)
        res.add(new ArrayList<>(path));
      path.remove(path.size()-1);
      return;
    }
    if (cur.left!=null)
      PS2helper(cur.left,remain-cur.val,path,res);
    if (cur.right!=null)
      PS2helper(cur.right,remain-cur.val,path,res);
    path.remove(path.size()-1);
  }

  public int totalHammingDistance2(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length,ans = 0,zero,mask;
    for (int i=0;i<31;i++){
      zero = 0;
      mask = 1<<i;
      for (int n:nums)
        if ((n & mask )==0)
          zero++;
      ans += zero*(N-zero);
    }
    return ans;
  }

  public int largestOverlap3(int[][] A, int[][] B) {
    int N = A.length,ans = 0;
    for (int r=0;r<N;r++)
      for (int c=0;c<N;c++)
        ans = Math.max(ans,Math.max(LOL2getOverlap(A,B,r,c),LOL2getOverlap(B,A,r,c)));
    return ans;
  }

  private int LOL2getOverlap(int[][] A,int[][] B,int shiftR,int shiftC){
    int N = A.length,ans=0,realR,realC;
    for (int r=shiftR;r<N;r++)
      for (int c=shiftC;c<N;c++)
        ans += A[r][c]*B[r-shiftR][c-shiftC];
    return ans;
  }

  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int R=0,C=0;
    if (obstacleGrid==null || (R=obstacleGrid.length)==0 || (C = obstacleGrid[0].length)==0 ||obstacleGrid[R-1][C-1]==1)
      return 0;
    int[] dp = new int[C];
    dp[C-1]=1;
    for (int r= R-1;r>=0;r--)
      for (int c = C-1;c>=0;c--)
        if (obstacleGrid[r][c]==1)
          dp[c]=0;
        else if (c<C-1)
          dp[c] += dp[c+1];
    return dp[0];
  }

  public int longestValidParentheses(String s) {
    if (s == null ||s.isEmpty())
      return 0;
    int N = s.length(),lenST = 0,lenInt=0,ans=0;
    char[] cs = s.toCharArray();
    int[] stack = new int[N];
    int[][] interval = new int[N][2];
    for (int i=0;i < N;i++)
      if (cs[i] ==')' && lenST!=0 && cs[stack[lenST-1]] == '('){
        int[] lastInterval = new int[]{stack[--lenST],i};
        while (lenInt !=0 && interval[lenInt-1][1]>=lastInterval[0]-1){
          int[] temp = interval[--lenInt];
          if (lastInterval[0] == temp[1] + 1)
            temp[1] = lastInterval[1];
          else if (lastInterval[0] < temp[0] && lastInterval[1] > temp[1]) {
            temp[0] = lastInterval[0];
            temp[1] = lastInterval[1];
          }
          lastInterval = temp;
        }
        interval[lenInt++] = lastInterval;
      }
      else
        stack[lenST++] = i;
    for (int i=0;i<lenInt;i++)
      ans = Math.max(ans,interval[i][1]-interval[i][0]+1);
    return ans;
  }

  public int numEquivDominoPairs(int[][] dominoes) {
    if (dominoes == null ||dominoes.length==1)
      return 0;
    int[] map = new int[100];
    for (int[] d:dominoes)
      map[NWDPhash(d)]++;
    int ans = 0;
    for (int m:map)
      if (m>1)
        ans += (m*(m-1))>>1;
    return ans;
  }

  private int NWDPhash(int[] D){
    return D[0]<=D[1]?D[0]*10+D[1]:D[1]*10+D[0];
  }

  private List<Integer>[] SAPconstructGraph(int[][] edges,int n){
    List<Integer>[] graph = new List[n];
    for (int i=0;i<n;i++)
      graph[i] = new ArrayList<>();
    for (int[] e:edges)
      graph[e[0]].add(e[1]);
    return graph;
  }

  public int[] shortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges) {
    if (n == 1)
      return new int[1];
    int[][] dp = new int[n][2];
    for (int i=0;i<n;i++)
      Arrays.fill(dp[i],Integer.MAX_VALUE);
    List<Integer>[] redGraph = SAPconstructGraph(red_edges,n),blueGraph = SAPconstructGraph(blue_edges,n);
    Queue<int[]> q = new LinkedList<>();
    q.offer(new int[]{0,0});
    q.offer(new int[]{0,1});
    dp[0][0]=dp[0][1]=0;
    int length = 0;
    while (!q.isEmpty()){
      length++;
      int size = q.size();
      for (int i=0;i<size;i++){
        int[] path = q.poll();
        if (path[1] == 0){
          for (int nextB:blueGraph[path[0]])
            if (dp[nextB][1]==Integer.MAX_VALUE){
              dp[nextB][1]=length;
              q.offer(new int[]{nextB,1});
            }
        }
        else
          for (int nextR : redGraph[path[0]])
            if (dp[nextR][0]==Integer.MAX_VALUE){
              dp[nextR][0]=length;
              q.offer(new int[]{nextR, 0});
            }
      }
    }
    int[] ans = new int[n];
    for (int i=0;i<n;i++){
      ans[i] = Math.min(dp[i][0],dp[i][1]);
      ans[i] = ans[i]==Integer.MAX_VALUE?-1:ans[i];
    }
    return ans;
  }

  public int mctFromLeafValues(int[] arr) {
    int N = arr.length;
    int[][][] dp = new int[N][N][2];
    return MFLhelper(0,N-1,arr,dp)[0];
  }

  private int[] MFLhelper(int start,int end,int[] arr,int[][][] dp){
    if (start == end){
      dp[start][end][1] = arr[start];
      return new int[]{0,arr[start]};
    }
    if (dp[start][end][0]!=0)
      return dp[start][end];
    dp[start][end][0]=Integer.MAX_VALUE;
    for (int i=start+1;i<=end;i++)
      dp[start][end][0]= Math.min(dp[start][end][0],MFLhelper(start,i-1,arr,dp)[0]+MFLhelper(i,end,arr,dp)[0]+dp[start][i-1][1]*dp[i][end][1]);
    dp[start][end][1] = Math.max(dp[start][start][1],dp[start+1][end][1]);
    return  dp[start][end];
  }

  public int maxAbsValExpr(int[] arr1, int[] arr2) {
    int N = arr1.length,res = Integer.MIN_VALUE;
    int[] sign = new int[]{-1,1};
    for (int p:sign)
      for (int q:sign){
        int min = Integer.MAX_VALUE,max=Integer.MIN_VALUE;
        for (int i=0;i<N;i++){
          int cur = arr1[i]*p+arr2[i]*q+i;
          min = Math.min(min,cur);
          max = Math.max(max,cur);
        }
        res = Math.max(res,max-min);
      }
    return res;
  }

  class RandomizedSet3 {

    List<Integer> data;
    Map<Integer,Integer> contains;
    Random r;
    /** Initialize your data structure here. */
    public RandomizedSet3() {
      data = new ArrayList<>();
      contains = new HashMap();
      r = new Random();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
      if (contains.containsKey(val))
        return false;
      contains.put(val,data.size());
      data.add(val);
      return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
      Integer pos;
      if ((pos = contains.get(val))==null)
        return false;
      contains.remove(val);
      if (pos!=data.size()-1){
        int last = data.get(data.size()-1);
        contains.put(last,pos);
        data.set(pos,last);
      }
      data.remove(data.size()-1);
      return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
      return data.get(r.nextInt(data.size()));
    }
  }

  public int findDuplicate2(int[] nums) {
    int ans=0;
    for (int n:nums)
      if (nums[Math.abs(n)]<0){
        ans = Math.abs(n);
        break;
      }
      else
        nums[Math.abs(n)]*=-1;
    for (int i=0;i<nums.length;i++)
      nums[i] = Math.abs(nums[i]);
    return ans;
  }

  public int findDuplicate3(int[] nums) {
    int fast =  nums[nums[0]],slow =  nums[0];
    while (fast!=slow){
      fast = nums[nums[fast]];
      slow = nums[slow];
    }
    fast = 0;
    while (fast!=slow){
      fast = nums[fast];
      slow = nums[slow];
    }
    return fast;
  }

  public boolean isMatch3(String s, String p) {
    List<Integer>[] epsilonConvert = IM3getNFAGraph(p);
    return IM3recognize(s,epsilonConvert,p.toCharArray());
  }

  private boolean IM3recognize(String s,List<Integer>[] graph,char[] matchConvert){
    List<Integer> start = new ArrayList<>();
    char[] cs = s.toCharArray();
    int statesNum = graph.length;
    boolean[] visited = new boolean[statesNum];
    IM3dfs(0,graph,visited);
    for (int i=0;i<statesNum;i++)
      if (visited[i])
        start.add(i);
    for (int i=0;i<cs.length;i++){
      List<Integer> match = new ArrayList<>();
      for (int v:start)
        if (v<matchConvert.length)
          if (cs[i]==matchConvert[v] || matchConvert[v]=='.')
            match.add(v+1);
      Arrays.fill(visited,false);
      for (int m:match)
        IM3dfs(m,graph,visited);
      start.clear();
      for (int j=0;j<statesNum;j++)
        if (visited[j])
          start.add(j);
    }
    for (int v:start)
      if (v==matchConvert.length)
        return true;
    return false;
  }

  private List<Integer>[] IM3getNFAGraph(String p){
    char[] cp = p.toCharArray();
    List<Integer>[] graph = new List[p.length()+1];
    for (int i=0;i<cp.length+1;i++)
      graph[i] = new ArrayList<>();
    for (int i=0;i<cp.length;i++){
      if (i<cp.length-1 && cp[i+1]=='*'){
        graph[i].add(i+1);
        graph[i+1].add(i);
      }
      if (cp[i]=='*')
        graph[i].add(i+1);
    }
    return graph;
  }

  private void IM3dfs(int cur,List<Integer>[] graph,boolean[] visited){
    visited[cur]=true;
    for (int next:graph[cur])
      if (!visited[next])
        IM3dfs(next,graph,visited);
  }

  public boolean isMatch4(String s, String p) {
    char[] cs = s.toCharArray(),cp = p.toCharArray();
    boolean[][] dp = new boolean[cs.length+1][cp.length+1];
    dp[0][0] = true;
    for (int i=2;i<=cp.length;i+=2)
      if (cp[i-1]=='*')
        dp[0][i] = dp[0][i-2];
    for (int i=1;i<=cs.length;i++)
      for (int j=1;j<=cp.length;j++)
        if (cs[i-1]==cp[j-1] || cp[j-1]=='.')
          dp[i][j] = dp[i-1][j-1];
        else if (cp[j-1]=='*')
          if (cs[i-1]!=cp[j-2] && cp[j-2]!='.')
            dp[i][j]=dp[i][j-2];
          else
            dp[i][j] = dp[i][j-2] || dp[i][j-1] || dp[i-1][j];
    return dp[cs.length][cp.length];
  }

  public boolean isPowerOfFour2(int num) {
    int after = num & 0x55555555;
    return after!=0 && (num& (num-1))==0 && (after & (after-1))==0 ;
  }

  public int countPrimes2(int n) {
    if (n<=2)
      return 0;
    if (n<=4)
      return n-2;
    int ans = 0;
    boolean[] res = new boolean[n];
    for (int i=2;i<n;i++)
      if (!res[i]){
        ans++;
        int cur = i;
        while (cur<n){
          res[cur]=true;
          cur+=i;
        }
      }
    return ans;
  }

  public boolean isHappy1(int n) {
    if (n==1)
      return true;
    int slow = nextHappy1(n),fast = nextHappy1(slow),nextFast,gapFast;
    while (slow!= fast && fast!=1){
      fast = nextHappy1(nextHappy1(fast));
      slow = nextHappy1(slow);
    }
    return fast==1;
  }

  private int nextHappy1(int num){
    int ans = 0,digit;
    while (num>0){
      digit = num%10;
      ans += digit*digit;
      num/=10;
   }
   return ans;
  }

  public String simplifyPath(String path) {
    if (path == null || path.isEmpty())
      return path;
    char[] cs = path.toCharArray();
    String[] stack = new String[cs.length];
    int[] idx = new int[1];
    int stackIdx = 0;
    while (idx[0] < cs.length){
      String cur = nextSP(cs,idx);
      if (cur.equals("/") || (!cur.equals(".") && !cur.equals("..")))
        stack[stackIdx++] = cur;
      else if (cur.equals(".."))
        stackIdx = stackIdx>1?stackIdx-3:stackIdx-1;
      else if (cur.equals("."))
        stackIdx--;
    }
    if (stackIdx >1 && stack[stackIdx-1].equals("/"))
      stackIdx--;
    StringBuilder sb = new StringBuilder();
    for (int i=0;i<stackIdx;i++)
      sb.append(stack[i]);
    return sb.length()==0?"/":sb.toString();
  }

  private String nextSP(char[] cs,int[] idx){
    char first = cs[idx[0]++];
    if (first == '/'){
      while (idx[0] <cs.length && cs[idx[0]] == first)
        idx[0]++;
      return "/";
    }
    StringBuilder ans = new StringBuilder();
    ans.append(first);
    while (idx[0]<cs.length && cs[idx[0]]!='/')
      ans.append(cs[idx[0]++]);
    return ans.toString();
  }

  public void solveSudoku(char[][] board) {
    boolean[][] rows = new boolean[9][10],cols = new boolean[9][10],M = new boolean[9][10];
    for (int r=0;r<9;r++)
      for (int c = 0;c<9;c++)
        if (board[r][c] != '.'){
          int num = board[r][c]-'0';
          rows[r][num] = cols[c][num]=M[sudokuGetChunk(r,c)][num]=true;
        }
    boolean[] hasFind = new boolean[1];
    sudukuDfs(board,0,0,rows,cols,M,hasFind);
  }

  private void sudukuDfs(char[][] board,int r,int c,boolean[][] rows,boolean[][] cols,boolean[][] chunks,boolean[] hasFind){
    if (r==9 && c==0){
      hasFind[0]=true;
      return;
    }
    if (board[r][c]!='.'){
      sudukuDfs(board,c==8?r+1:r,c==8?0:c+1,rows,cols,chunks,hasFind);
      return;
    }
    int chunkNum = sudokuGetChunk(r,c);
    for (int i=1;i<= 9;i++)
      if (rows[r][i] || cols[c][i] || chunks[chunkNum][i])
        continue;
      else{
        rows[r][i] = cols[c][i] = chunks[chunkNum][i]=true;
        board[r][c] = (char)(i+'0');
        sudukuDfs(board,c==8?r+1:r,c==8?0:c+1,rows,cols,chunks,hasFind);
        if (hasFind[0])
          return;
        board[r][c]='.';
        rows[r][i] = cols[c][i] = chunks[chunkNum][i]=false;
      }
  }

  private int sudokuGetChunk(int r,int c){
    return (r/3)*3+c/3;
  }

  public int[] nextGreaterElements_final(int[] nums) {
    if (nums == null || nums.length==0)
      return nums;
    int N = nums.length,st = 0;
    int[] stack = new int[N << 1],ans = new int[N];
    Arrays.fill(ans,-1);
    for (int i=0;i< (N<<1) ;i++){
      int realIdx = i % N;
      while (st > 0 && nums[stack[st - 1]] < nums[realIdx])
        ans[stack[--st]] = nums[realIdx];
      stack[st++] = realIdx;
    }
    return ans;
  }

  public String multiply2(String num1, String num2) {
    int n = num1.length(),m = num2.length();
    int[] res = new int[m+n];
    for (int i=n-1;i>=0;i--)
      for (int j=m-1;j>=0;j--){
        int cachePos = i+j,pos = cachePos+1;
        int mul = (num1.charAt(i)-'0')*(num2.charAt(j)-'0')+res[pos];
        res[pos] = mul%10;
        res[cachePos] += mul/10;
      }
    StringBuilder sb = new StringBuilder();
    for (int r:res)
      if (!(r == 0 && sb.length()==0))
        sb.append(r);
    return sb.length()==0?"0":sb.toString();
  }

  class NumArray_BIT {

    int[] data,record;

    public NumArray_BIT(int[] nums) {
      data = nums;
      record = new int[nums.length+1];
      for (int i=0;i<nums.length;i++)
        add(nums[i],i);
    }

    private void add(int val,int idx){
      for (int i=idx+1;i<record.length;i+=lowBit(i))
        record[i]+=val;
    }

    private int sum(int idx){
      int res = 0;
      for (int i=idx+1;i>0;i-=lowBit(i))
        res += record[i];
      return res;
    }

    private int lowBit(int i){
      return i & (-i);
    }

    public void update(int i, int val) {
      add(val-data[i],i);
      data[i] = val;
    }

    public int sumRange(int i, int j) {
      return sum(j)-sum(i-1);
    }
  }

  class NumArray_ST {

    class Node{
      int val,lazy,L,R;
      Node left,right;

      public Node(int l,int r){
        L = l;
        R = r;
        val=lazy = 0;
        left = null;
        right = null;
      }

      public void pushDown(){
        if (L==R)
          return;
        int mid = (L+R)>>1;
        if (left==null || right==null){
          left = new Node(L,mid);
          right = new Node(mid+1,R);
        }
        if (lazy!=0){
          left.updateByVal(lazy);
          right.updateByVal(lazy);
          lazy = 0;
        }
      }

      public void updateByVal(int v){
        lazy = v;
        val = (R-L+1)*v;
      }

      public void updateFromSon(){
        val = left.val+right.val;
      }
    }

    Node root;
    public NumArray_ST(int[] nums) {
      root = new Node(0,nums.length-1);
      for (int i=0;i<nums.length;i++)
        updatePoint(root,i,nums[i]);
    }

    public void query(Node root,int start,int end,int[] res){
      if (root.R <start || root.L >end)
        return;
      if (root.L >=start && root.R <= end){
        mergeQuery(res,root.val);
        return;
      }
      root.pushDown();
      query(root.left,start,end,res);
      query(root.right,start,end,res);
    }

    public void updateRange(Node root,int start,int end,int val){
      if (root.R <start || root.L >end)
        return;
      if (root.L >=start && root.R <= end){
        root.updateByVal(val);
        return;
      }
      root.pushDown();
      updateRange(root.left,start,end,val);
      updateRange(root.right,start,end,val);
      root.updateFromSon();
    }

    public void updatePoint(Node root,int pos,int val){
      if (root.L == root.R){
        root.val = val;
        return;
      }
      root.pushDown();
      int mid = (root.L+root.R)>>1;
      if (pos<=mid)
        updatePoint(root.left,pos,val);
      else
        updatePoint(root.right,pos,val);
      root.updateFromSon();
    }

    public void mergeQuery(int[] res,int cur){
      res[0]+= cur;
    }

    public void update(int i, int val) {
      updatePoint(root,i,val);
    }

    public int sumRange(int i, int j) {
      int[] res = new int[1];
      query(root,i,j,res);
      return res[0];
    }
  }

  public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length==0)
      return new int[0];
    Deque<Integer> dq = new ArrayDeque<>();
    int N = nums.length,idx=0;
    int[] ans = new int[N-k+1];
    for (int i=0;i<N;i++){
      while (!dq.isEmpty() && dq.peekFirst()<=i-k )
        dq.pollFirst();
      while (!dq.isEmpty() && nums[dq.peekLast()]<nums[i])
        dq.pollLast();
      dq.offerLast(i);
      if (i>=k-1)
        ans[idx++] = nums[dq.peek()];
    }
    return ans;
  }

  public int maxSumTwoNoOverlap2(int[] A, int L, int M) {
    int N = A.length;
    int[] ps = new int[N+1];
    for (int i=0;i<N;i++)
      ps[i+1] = ps[i]+A[i];
    int res =Integer.MIN_VALUE,Lmax = ps[L],Mmax = ps[M];
    for (int i=L+M;i<=N;i++){
      Lmax = Math.max(Lmax,ps[i-M]-ps[i-M-L]);
      Mmax = Math.max(Mmax,ps[i-L]-ps[i-L-M]);
      res = Math.max(res,Math.max(Lmax+ps[i]-ps[i-M],Mmax+ps[i]-ps[i-L]));
    }
    return res;
  }

  public int findKthLargest3(int[] nums, int k) {
    if (nums.length==1)
      return nums[0];
   // FKLshuffle3(nums);
    return FKLquickSelect3(nums,0,nums.length-1,k-1);
  }

  private void FKLshuffle3(int[] nums){
    Random r = new Random();
    int N = nums.length;
    for (int i=N-1;i>0;i--)
      exchange(nums,r.nextInt(i+1),i);
  }

  private int FKLquickSelect3(int[] nums,int start,int end,int targetIdx){
    if (end < start)
      throw new IllegalArgumentException();
    if (start == end)
      return nums[start];
    int partition = FKLpartition3(nums,start,end);
    if (partition <targetIdx)
      return FKLquickSelect3(nums,partition+1,end,targetIdx);
    else if (partition > targetIdx)
      return FKLquickSelect3(nums,start,partition-1,targetIdx);
    else
      return nums[partition];
  }

  private int FKLpartition3(int[] nums,int start,int end){
    int flag = nums[start],L = start+1,R = end;
    while (L <= R){
      while (L <= end && nums[L] >= flag)
        L++;
      while (R >=start && nums[R] < flag)
        R--;
      if (L >= R)
        break;
      exchange(nums,L,R);
    }
    exchange(nums,start,R);
    return R;
  }


  public ListNode mergeKLists3(ListNode[] lists) {
    if (lists.length==1)
      return lists[0];
    PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
      @Override
      public int compare(ListNode a, ListNode b) {
        return a.val-b.val;
      }
    });
    ListNode head = new ListNode(0),cur = head;
    for (ListNode L:lists)
      if (L!=null)
        pq.offer(L);
    while (!pq.isEmpty()){
      ListNode temp = pq.poll();
      cur.next = temp;
      cur = cur.next;
      if (temp.next!=null)
        pq.offer(temp.next);
    }
    return head.next;
  }

  public ListNode mergeKLists4(ListNode[] lists) {
    if (lists.length==1)
      return lists[0];
    return MKL4mergeSort(lists,0,lists.length-1);
  }

  private ListNode MKL4mergeSort(ListNode[] lists,int start,int end){
    if (start>end)
      return null;
    if (start==end)
      return lists[start];
    int mid = (start+end)>>1;
    ListNode left = MKL4mergeSort(lists,start,mid),right = MKL4mergeSort(lists,mid+1,end);
    return MKL4merge(left,right);
  }

  private ListNode MKL4merge(ListNode L1,ListNode L2){
    ListNode head = new ListNode(0),cur = head;
    while (L1 != null && L2 !=null){
      if (L1.val<=L2.val){
        cur.next = L1;
        L1 = L1.next;
      }
      else{
        cur.next = L2;
        L2 = L2.next;
      }
      cur = cur.next;
    }
    if (L1!=null)
      cur.next = L1;
    else
      cur.next = L2;
    return head.next;
  }

  public int canCompleteCircuit(int[] gas, int[] cost) {
    if (gas == null || gas.length==0)
      return -1;
    int N = gas.length,res = 0;
    for (int i=0;i<N;i++)
      res += gas[i]-cost[i];
    if (res<0)
      return -1;
    res = 0;
    int start = 0;
    for (int i=0;i<N;i++)
      if ( (res=res+gas[i]-cost[i])<0){
        res=0;
        start = i+1;
      }
    return start;
  }

  public boolean validTicTacToe3(String[] board) {
    int R = board.length,C = board[0].length(),Xnum=0,Onum=0,slash=0,tSlash = 0;
    int[] rows = new int[R],cols = new int[C];
    boolean isXwin=false,isOwins = false;
    for (int i=0;i<R;i++)
      for (int j=0;j<C;j++)
        if (board[i].charAt(j)=='X'){
          Xnum++;
          rows[i]++;
          cols[j]++;
          if (i==j)
            slash++;
          if (i+j==2)
            tSlash++;
          if (rows[i]==3 || cols[j]==3 || slash==3 || tSlash==3)
            isXwin = true;
        }
        else if (board[i].charAt(j)=='O'){
          Onum++;
          rows[i]--;
          cols[j]--;
          if (i==j)
            slash--;
          if (i+j==2)
            tSlash--;
          if (rows[i]==-3 || cols[j]==-3 || slash==-3 || tSlash==-3)
            isOwins = true;
        }
    if (isXwin && isOwins)
      return false;
    if (Xnum>Onum+1 || Xnum<Onum)
      return false;
    if (isXwin && Xnum!= Onum+1)
      return false;
    if (isOwins && Xnum!=Onum)
      return false;
    return true;
  }

  public TreeNode lowestCommonAncestor5(TreeNode root, TreeNode p, TreeNode q) {
    if (root == p || root==q || root==null)
      return root;
    TreeNode left = lowestCommonAncestor5(root.left,p,q),right = lowestCommonAncestor5(root.right,p,q);
    if (left !=null && right!=null)
      return root;
    if (left !=null)
      return left;
    else
      return right;
  }
  public int minMalwareSpread5(int[][] graph, int[] initial) {
    int N = graph.length,ans=-1,maxSave = Integer.MIN_VALUE,curSave;
    boolean[] visited = new boolean[N],infected=new boolean[N];
    for (int i:initial)
      infected[i]=true;
    Arrays.sort(initial);
    for (int i:initial){
      curSave=0;
      for (int adj=0;adj<graph.length;adj++)
        if (graph[i][adj]==1 && !visited[adj] && i!=adj)
          curSave += MLSW3dfs(graph,infected,visited,i,adj);
      if (curSave>maxSave){
        maxSave = curSave;
        ans = i;
      }
    }
    return ans;
  }

  private int MLSW3dfs(int[][] graph,boolean[] infected,boolean[] visited,int sourceIdx,int curIdx){
    if (infected[curIdx])
      return 0;
    visited[curIdx] = true;
    int save = 1,curSave;
    boolean isMultiInfected = false;
    for (int adj=0;adj<graph.length;adj++)
      if (graph[curIdx][adj]==1 && !visited[adj] && adj!=sourceIdx){
        curSave = MLSW3dfs(graph,infected,visited,sourceIdx,adj);
        if (curSave==0)
          isMultiInfected = true;
        if (!isMultiInfected)
          save += curSave;
      }
    return isMultiInfected?0:save;
  }

  public int[][] insert(int[][] I, int[] nI) {
    if (I == null ||I.length==0)
      return new int[][]{nI};
    int N = I.length,start = insertFindStart(I,nI),ansLen,end = insertFindEnd(I,nI),x=0;
    if (end<start)
      ansLen = N+1;
    else{
      if (start >=0)
        nI[0] = Math.min(nI[0],I[start][0]);
      if (end <N)
        nI[1] = Math.max(nI[1],I[end][1]);
      ansLen = start+N-end;
    }
    int[][] ans = new int[ansLen][2];
    for (int i=0;i<start;i++)
      ans[x++] = I[i];
    ans[x++] = nI;
    for (int i=end+1;i<N;i++)
      ans[x++] = I[i];
    return ans;
  }

  private int insertFindEnd(int[][] I,int[] nI){
    int start = 0,end = I.length-1,mid;
    while (start<=end){
      mid = (start+end)>>1;
      if (I[mid][0] >nI[1])
        end = mid-1;
      else
        start = mid+1;
    }
    return end;
  }

  private int insertFindStart(int[][] I,int[] nI){
    int start = 0,end = I.length-1,mid;
    while (start<=end){
      mid = (start+end)>>1;
      if (I[mid][1]<nI[0])
        start = mid+1;
      else
        end = mid-1;
    }
    return start;
  }

  public boolean isRectangleOverlap2(int[] a, int[] b) {
    return !(a[0] >= b[2] || a[2]<= b[0] || a[1] >= b[3] || a[3] <= b[1]);
  }

  public int[][] generateMatrix2(int n) {
    if (n<0)
      throw new IllegalArgumentException();
    if (n==0)
      return new int[0][0];
    if (n==1)
      return new int[][]{{1}};
    int[][] ans = new int[n][n];
    int Rstart = 0,Rend = n-1,Cstart = 0,Cend = n-1;
    for (int i=1,order = 0;i <= n*n;order++){
      int turn = order%4;
      if (turn == 0){
        for (int j=Cstart;j<=Cend;j++)
          ans[Rstart][j] = i++;
        Rstart++;
      }
      else if (turn ==1 ){
        for (int j=Rstart;j<=Rend;j++)
          ans[j][Cend] = i++;
        Cend--;
      }
      else if (turn ==2){
        for (int j=Cend;j>=Cstart;j--)
          ans[Rend][j] = i++;
        Rend--;
      }
      else{
        for (int j = Rend;j>=Rstart;j--)
          ans[j][Cstart] = i++;
        Cstart++;
      }
    }
    return ans;
  }

  public int maxTurbulenceSize1(int[] A) {
    if (A == null)
      return 0;
    if (A.length==0 || A.length==1)
      return A.length;
    int N = A.length,max,lastLen;
    max = lastLen =A[0]==A[1]?1:2;
    for (int i=2;i<N;i++)
      if ((A[i]-A[i-1]>0 && A[i-1]-A[i-2]<0)|| (A[i]-A[i-1]<0 && A[i-1]-A[i-2]>0) ){
        lastLen += 1;
        max = Math.max(max,lastLen);
      }
      else
        lastLen =2;
    return max;
  }

  public int maximumGap(int[] nums) {
    if (nums == null || nums.length<2)
      return 0;
    int N = nums.length,max = Integer.MIN_VALUE,exp=1,R=10;
    int[] aux = new int[N];
    for (int n:nums)
      max = Math.max(max,n);
    while ( max/exp >0){
      int[] count = new int[R+1];
      for (int i=0;i<N;i++)
        count[(nums[i]/exp%R)+1]++;
      for (int i=0;i<R;i++)
        count[i+1]+=count[i];
      for (int i=0;i<N;i++)
        aux[count[nums[i]/exp%R]++] = nums[i];
      for (int i=0;i<N;i++)
        nums[i] = aux[i];
      exp*=R;
    }
    int ans=Integer.MIN_VALUE;
    for (int i=1;i<N;i++)
      ans = Math.max(ans,nums[i]-nums[i-1]);
    return ans;
  }

  public int maximumGap1(int[] nums) {
    if (nums == null || nums.length < 2)
      return 0;
    int N = nums.length,max=Integer.MIN_VALUE,min = Integer.MAX_VALUE,bucketSize,pre,maxGap=Integer.MIN_VALUE;
    for (int n:nums){
      max = Math.max(max,n);
      min = Math.min(min,n);
    }
//    bucketSize = (max-min)/(N-1);
    bucketSize = (int)Math.ceil((double)(max - min)/(N - 1));
    int[] bucketMax = new int[N-1],bucketMin = new int[N-1];
    Arrays.fill(bucketMax,Integer.MIN_VALUE);
    Arrays.fill(bucketMin,Integer.MAX_VALUE);
    for (int n:nums){
      if (n==min || n==max)
        continue;
      int bucketIdx = (n-min)/bucketSize;
      bucketMax[bucketIdx] = Math.max( bucketMax[bucketIdx],n);
      bucketMin[bucketIdx] = Math.min( bucketMin[bucketIdx],n);
    }
    pre = min;
    for (int i=0;i<N-1;i++){
      if (bucketMax[i]==Integer.MIN_VALUE)
        continue;
      maxGap = Math.max(maxGap,bucketMin[i]-pre);
      pre = bucketMax[i];
    }
    maxGap = Math.max(maxGap,max-pre);
    return maxGap;
  }

  public int tribonacci(int n) {
    if (n ==0)
      return 0;
    if (n==1 || n==2)
      return 1;
    int n1 = 0,n2=1,n3=1,cur=0;
    for (int i=3;i<=n;i++){
      cur = n1+n2+n3;
      n1=n2;
      n2=n3;
      n3=cur;
    }
    return cur;
  }

  public String alphabetBoardPath(String target) {
    int curIdx=0;
    StringBuffer res = new StringBuffer();
    int[] curPos=new int[2],nextPos;
    for (char t:target.toCharArray()){
      nextPos = ABPidxToPos(t-'a');
      int Rdiff = nextPos[0]-curPos[0],Cdiff = nextPos[1]-curPos[1];
      if (t=='z'){
        for (int j=0;j<Math.abs(Cdiff);j++)
          if (Cdiff>=0)
            res.append('R');
          else
            res.append('L');
        for (int i=0;i<Math.abs(Rdiff);i++)
          if (Rdiff>=0)
            res.append('D');
          else
            res.append('U');
      }
      else{
        for (int i=0;i<Math.abs(Rdiff);i++)
          if (Rdiff>=0)
            res.append('D');
          else
            res.append('U');
        for (int j=0;j<Math.abs(Cdiff);j++)
          if (Cdiff>=0)
            res.append('R');
          else
            res.append('L');
      }
      res.append('!');
      curPos = nextPos;
    }
    return res.toString();
  }

  private int[] ABPidxToPos(int idx){
    return new int[]{idx/5,idx%5};
  }

  public int stoneGameII(int[] piles) {
    int N = piles.length;
    int[][] dp = new int[N][N];
    int[] curSum = new int[N];
    curSum[N-1] = piles[N-1];
    for (int i=N-2;i>=0;i--)
      curSum[i] =curSum[i+1]+piles[i];
    return SG2helper(dp,curSum,0,1);
  }

  private int SG2helper(int[][] dp,int[] curSum,int cur,int M){
    int N = curSum.length;
    if (cur>=N)
      return 0;
    if (cur+(M << 1) >=N)
      return curSum[cur];
    if (dp[cur][M]!=0)
      return dp[cur][M];
    int nextMin= Integer.MAX_VALUE;
    for (int i=1;i<= (M <<1);i++)
      nextMin = Math.min(nextMin,SG2helper(dp,curSum,cur+i,Math.max(M,i)));
    dp[cur][M] = curSum[cur]-nextMin;
    return dp[cur][M];
  }

  public List<List<Integer>> threeSum2(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    if (nums == null || nums.length<3)
      return ans;
    Arrays.sort(nums);
    int N = nums.length,start,end,temp;
    for (int i=0;i<N;i++){
      if (i!=0 && nums[i]==nums[i-1])
        continue;
      start = i+1;
      end = N-1;
      while (start<end){
        temp = nums[i]+nums[start]+nums[end];
        if (temp<0)
          start=nextStart(nums,start);
        else if (temp>0)
          end = nextEnd(nums,end);
        else{
          ans.add(Arrays.asList(nums[i],nums[start],nums[end]));
          start = nextStart(nums,start);
          end = nextEnd(nums,end);
        }
      }
    }
    return ans;
  }

  private int nextStart(int[] nums,int start){
    start++;
    while (start<nums.length && nums[start]==nums[start-1])
      start++;
    return start;
  }

  private int nextEnd(int[] nums,int end){
    end--;
    while (end >=0 && nums[end]==nums[end+1])
      end--;
    return end;
  }

  public String originalDigits1(String s) {
    int[] digit = new int[26],count = new int[10];
    for (char c:s.toCharArray())
      digit[c-'a']++;
    StringBuilder sb = new StringBuilder();
    count[0] = digit['z'-'a'];
    count[1] = digit['o'-'a']-digit['z'-'a']-digit['w'-'a']-digit['u'-'a'];
    count[2] = digit['w'-'a'];
    count[3] = digit['t'-'a']-digit['g'-'a']-digit['w'-'a'];
    count[4] = digit['u'-'a'];
    count[5] = digit['f'-'a']-digit['u'-'a'];
    count[6] = digit['x'-'a'];
    count[7] = digit['s'-'a']-digit['x'-'a'];
    count[8] = digit['g'-'a'];
    count[9] = digit['i'-'a']-count[5]-count[6]-count[8];
    for (int i=0;i<=9;i++)
      for (int j=0;j<count[i];j++)
        sb.append(i);
    return sb.toString();
  }

  public int longestPalindromeSubseq2(String s) {
    if (s==null ||s.isEmpty())
      return 0;
    char[] cs = s.toCharArray();
    int N = cs.length;
    int[][] dp = new int[N+1][N+1];
    for (int i=N-1;i>=0;i--)
      for (int j=0;j<N;j++)
        if (cs[i]==cs[j])
          dp[N-i][j+1] = dp[N-i-1][j]+1;
        else
          dp[N-i][j+1] = Math.max(dp[N-i-1][j+1],dp[N-i][j]);
    return dp[N][N];
  }

  public int longestPalindromeSubseq3(String s) {
    if (s == null || s.isEmpty())
      return 0;
    char[] cs = s.toCharArray();
    int N = cs.length;
    int[][] dp = new int[N][N];
    for (int i=N-1;i>=0;i--){
      dp[i][i] = 1;
      for (int j=i+1;j<N;j++)
        if (cs[i]==cs[j])
          dp[i][j] = dp[i+1][j-1]+2;
        else
          dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
    }
    return dp[0][N-1];
  }

  public int longestPalindromeSubseq4(String s) {
    char[] cs = s.toCharArray();
    int N = cs.length,prev,curMax;
    // if the current i and j is 4,2, curMax means the longest pal numer between i-1 and j+1,and prev means i-1,j+1 value int the last circle
    int[] dp=new int[N];
    for (int i=0;i<N;i++){
      dp[i]=1;
      curMax = 0;
      for (int j=i-1;j>=0;j--){
        prev = dp[j];
        if (cs[i]==cs[j])
          dp[j] = curMax+2;
        curMax = Math.max(curMax,prev);
      }
    }
    int ans=0;
    for (int d:dp)
      ans = Math.max(ans,d);
    return ans;
  }

  public int largest1BorderedSquare(int[][] grid) {
    int R = grid.length,C = grid[0].length,maxLen=0;
    int[][] hor = new int[R+1][C+1],ver = new int[R+1][C+1];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        if (grid[r][c]==1){
          hor[r+1][c+1] = hor[r+1][c]+1;
          ver[r+1][c+1] = ver[r][c+1]+1;
        }
    for (int r = R;r>0;r--)
      for (int c=C;c>0;c--){
        int small = Math.min(hor[r][c],ver[r][c]);
        while (small > maxLen){
          if (ver[r][c-small+1] >= small && hor[r-small+1][c] >=small)
            maxLen=small;
          small--;
        }
      }
    return maxLen*maxLen;
  }

  public List<Integer> diffWaysToCompute2(String input) {
    return DWChelper(input.toCharArray(),0,input.length()-1);
  }

  private List<Integer> DWChelper(char[] input,int start,int end){
    List<Integer> res = new ArrayList<>();
    for (int i=start;i<=end;i++){
      if (input[i]>='0' && input[i] <='9')
        continue;
      char ope = input[i];
      List<Integer> leftRes = DWChelper(input,start,i-1),rightRes = DWChelper(input,i+1,end);
      for (int L:leftRes)
        for (int R:rightRes)
          if (ope == '+')
            res.add(L+R);
          else if (ope=='-')
            res.add(L-R);
          else
            res.add(L*R);
    }
    if (res.isEmpty())
      res.add(DWCconvertToInt(input,start,end));
    return res;
  }

  private int DWCconvertToInt(char[] input, int start,int end){
    int res=0;
    for (int i=start;i<=end;i++)
      res=res*10+(input[i]-'0');
    return res;
  }

  public List<Boolean> camelMatch2(String[] Q, String P) {
    char[] cp = P.toCharArray(),cq;
    int qIdx,pIdx;
    List<Boolean> ans = new ArrayList<>();
    for(String q:Q){
      cq = q.toCharArray();
      for (qIdx=0,pIdx = 0;qIdx<cq.length;qIdx++)
        if (cq[qIdx]>='a' && cq[qIdx]<='z'){
          if (pIdx<cp.length && cp[pIdx]==cq[qIdx])
            pIdx++;
        }
        else if (pIdx<cp.length && cp[pIdx]==cq[qIdx])
          pIdx++;
        else
          break;
      ans.add(pIdx==cp.length && qIdx==cq.length);
    }
    return ans;
  }

  public int videoStitching2(int[][] clips, int T) {
    Arrays.sort(clips, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]==b[0]?a[1]-b[1]:a[0]-b[0];
      }
    });
    int N = clips.length,ans=0,last = 0,cur=0,far=0,next;
    if (clips[0][0]>0 || clips[N-1][1]<T)
      return -1;
    while (cur<N && clips[cur][0]==0){
      if (clips[cur][1]>far){
        far = clips[cur][1];
        last = cur;
      }
      cur++;
    }
    ans++;
    if (far>=T)
      return ans;
    while (cur<N){
      next = cur;
      while (cur<N && clips[cur][0]<=clips[last][1]){
        if (clips[cur][1]>far){
          next = cur;
          far = clips[next][1];
        }
        cur++;
      }
      last = next;
      ans++;
      if (cur>=N || far>=T)
        break;
      if (clips[cur][0]>far)
        return -1;
    }
    return ans;
  }

  public int[] smallestSufficientTeam(String[] R, List<List<String>> P) {
    int Plen = P.size(),skillIdx=0;
    Map<String,Integer> skillToIdx=new HashMap<>();
    for (String r:R)
      skillToIdx.put(r,skillIdx++);
    int[] per = new int[Plen];
    for (int i=0;i<Plen;i++)
      for (String skill:P.get(i))
        per[i] |= (1<<skillToIdx.get(skill));
    List<Integer> team = new ArrayList<>();
    boolean[] isBanned = SSTremoveContains(per);
    SSThelper(0,0,skillIdx,isBanned,per,new ArrayList<>(),team);
    int[] ans = new int[team.size()];
    for (int i=0;i<team.size();i++)
      ans[i] = team.get(i);
    return ans;
  }

  private boolean[] SSTremoveContains(int[] per){
    boolean[] isBanned = new boolean[per.length];
    for (int i=0;i<per.length;i++)
      for (int j=i+1;j<per.length;j++)
        if (per[i]==per[j])
          isBanned[j]=true;
        else if (per[i]>per[j] && SSTcontains(per[i],per[j]))
          isBanned[j]=true;
        else if (per[j]>per[i] && SSTcontains(per[j],per[i]))
          isBanned[i] = true;
    return isBanned;
  }

  private boolean SSTcontains(int i,int j){
    int mask=1;
    while (mask<=j){
      if ((mask & i)==0 && (mask & j)>0)
        return false;
      mask<<=1;
    }
    return true;
  }

  private void SSThelper(int curSkills,int curPerson,int skillNum,boolean[] isBanned,int[] per,List<Integer> curTeam,List<Integer> bestTeam){
    if (curSkills == (1<<skillNum)-1){
      if (bestTeam.size()==0 || curTeam.size()<bestTeam.size()){
        bestTeam.clear();
        bestTeam.addAll(new ArrayList<>(curTeam));
      }
      return;
    }
    if (bestTeam.size()!=0 && curTeam.size()>=bestTeam.size())
      return;
    for (int i=curPerson;i<per.length;i++)
      if (isBanned[i])
        continue;
      else if ( (curSkills|per[i]) != curSkills){
        curTeam.add(i);
        SSThelper(curSkills|per[i],i+1,skillNum,isBanned,per,curTeam,bestTeam);
        curTeam.remove(curTeam.size()-1);
      }
  }

  public List<List<Integer>> subsetsWithDup3(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    if (nums==null ||nums.length==0)
      return ans;
    Arrays.sort(nums);
    SSWDhelper(nums,0,new ArrayList<>(),ans);
    return ans;
  }

  private void SSWDhelper(int[] nums,int start,List<Integer> cur,List<List<Integer>> res){
    res.add(new ArrayList<>(cur));
    if (start == nums.length)
      return;
    for (int i=start;i<nums.length;i++){
      if (i!= start && nums[i]==nums[i-1])
        continue;
      cur.add(nums[i]);
      SSWDhelper(nums,i+1,cur,res);
      cur.remove(cur.size()-1);
    }
  }

  public int maxWidthRamp2(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    int N = A.length, stIdx = 0, ans = 0;
    int[] stack = new int[N];
    for (int i = 0; i < N; i++)
      if (stIdx == 0 || A[i] <= A[stack[stIdx - 1]])
        stack[stIdx++] = i;
    for (int i = N - 1; i >= 0; i--)
      if (stIdx == 0)
        break;
      else
        while (stIdx > 0 && A[i] >= A[stack[stIdx - 1]])
          ans = Math.max(ans, i - stack[--stIdx]);
    return ans;
  }

  public TreeNode bstToGst2(TreeNode root) {
    if (root==null)
      return root;
    int[] sum=new int[1];
    BTGhelper2(root,sum);
    return root;
  }

  private void BTGhelper2(TreeNode root,int[] sum){
    if (root==null)
      return;
    BTGhelper2(root.right,sum);
    sum[0] += root.val;
    root.val = sum[0];
    BTGhelper2(root.left,sum);
  }

  public void recoverTree(TreeNode root) {
    TreeNode[] first = new TreeNode[1],second = new TreeNode[1],prev=new TreeNode[]{new TreeNode(Integer.MIN_VALUE)};
    RThelper(root,first,second,prev);
    int temp = first[0].val;
    first[0].val = second[0].val;
    second[0].val = temp;
  }

  private void RThelper(TreeNode root,TreeNode[] first,TreeNode[] second,TreeNode[] prev){
    if (root==null)
      return;
    RThelper(root.left,first,second,prev);
    if (prev[0].val > root.val){
      if (first[0]==null)
        first[0] = prev[0];
      if (first[0]!=null)
        second[0] = root;
    }
    prev[0] = root;
    RThelper(root.right,first,second,prev);
  }

  public void recoverTree2(TreeNode root) {
    TreeNode first=null, second=null, prev = new TreeNode(Integer.MIN_VALUE),cur =root;
    while (cur!=null)
      if (cur.left==null){
        if (prev.val>cur.val){
          if (first == null)
            first = prev;
          if (first !=null)
            second = cur;
        }
        prev = cur;
        cur=cur.right;
      }
      else{
        TreeNode before = cur.left;
        while (before.right!=null && before.right!=cur)
          before = before.right;
        if (before.right==null){
          before.right=cur;
          cur = cur.left;
        }
        else{
          if (prev.val>cur.val){
            if (first == null)
              first = prev;
            if (first !=null)
              second = cur;
          }
          prev = cur;
          before.right=null;
          cur=cur.right;
        }
      }
    int temp = first.val;
    first.val=second.val;
    second.val=temp;
  }

  public int countNodes(TreeNode root) {
    if (root==null)
      return 0;
    int leftHeight = CNgetHeight(root,true),rightHeight = CNgetHeight(root,false);
    if (leftHeight == rightHeight)
      return (1<<leftHeight)-1;
    else
      return 1+countNodes(root.left)+countNodes(root.right);
  }

  private int CNgetHeight(TreeNode root,boolean isLeft){
    int height = 0;
    TreeNode cur = root;
    while (cur!=null){
      height++;
      cur = isLeft?cur.left:cur.right;
    }
    return height;
  }

  public int maxTurbulenceSize3(int[] A) {
    if (A == null)
      return 0;
    if (A.length<=1)
      return A.length;
    int N =A.length,maxLen,lastLen;
    maxLen=lastLen=A[1]==A[0]?1:2;
    for (int i=2;i<N;i++)
      if ((A[i]-A[i-1]<0&& A[i-1]-A[i-2]>0) ||(A[i]-A[i-1]>0&& A[i-1]-A[i-2]<0) )
        lastLen++;
      else{
        maxLen = Math.max(maxLen,lastLen);
        lastLen=A[i]==A[i-1]?1:2;
      }
    return Math.max(maxLen,lastLen);
  }

  class StreamChecker4 {

    class TrieNode{
      boolean isWord;
      TrieNode[] next;
      public TrieNode(){
        isWord=false;
        next = new TrieNode[26];
      }
    }

    private void putTireIte(TrieNode cur,String txt){
      int idx;
      for (int i=txt.length()-1;i>=0;i--){
        idx = txt.charAt(i)-'a';
        if (cur.next[idx]==null)
          cur.next[idx]=new TrieNode();
        cur = cur.next[idx];
      }
      cur.isWord = true;
    }

    private boolean searchTrieIte(TrieNode cur,Iterator<Character> it){
      while (it.hasNext()){
        int idx = it.next()-'a';
        if (cur.next[idx]==null)
          return false;
        if (cur.next[idx].isWord)
          return true;
        cur=cur.next[idx];
      }
      return false;
    }

    private TrieNode root;
    private LinkedList<Character> data;
    int size;

    public StreamChecker4(String[] words) {
      root = new TrieNode();
      data = new LinkedList<>();
      size = 0;
      for (String w:words){
        putTireIte(root,w);
        size = Math.max(size,w.length());
      }
    }

    public boolean query(char letter) {
      data.add(letter);
      if (data.size()>size)
        data.removeFirst();
      return searchTrieIte(root,data.descendingIterator());
    }
  }

  public int longestConsecutive2(int[] nums) {
    if (nums==null ||nums.length==0)
      return 0;
    int max = 0;
    Map<Integer,Integer> record = new HashMap<>();
    for (int n:nums){
      if (record.containsKey(n))
        continue;
      int left = record.getOrDefault(n-1,0),right = record.getOrDefault(n+1,0);
      int sum = 1+left+right;
      record.put(n,sum);
      max =Math.max(max,sum);
      if (left!=0)
        record.put(n-left,sum);
      if (right!=0)
        record.put(n+right,sum);
    }
    return max;
  }

  public int longestConsecutive3(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int max = 0;
    Set<Integer> st = new HashSet<>();
    for (int n:nums)
      st.add(n);
    for (int n:nums)
      if (st.contains(n)){
        int sum=1;
        for (int i=n-1;st.remove(i);i--)
          sum++;
        for (int i=n+1;st.remove(i);i++)
          sum++;
        max = Math.max(max,sum);
      }
    return max;
  }

  public int maxPathSum(TreeNode root) {
    if (root == null)
      return 0;
    int[] maxLen = new int[]{Integer.MIN_VALUE};
    MPShelper(root,maxLen);
    return maxLen[0];
  }

  private int MPShelper(TreeNode cur,int[] maxLen){
    if (cur==null)
      return 0;
    int left = Math.max(0,MPShelper(cur.left,maxLen)),right = Math.max(0,MPShelper(cur.right,maxLen));
    maxLen[0] = Math.max(maxLen[0],right+left+cur.val);
    return Math.max(left,right)+cur.val;
  }

  class WBTrie{
    boolean isWord;
    WBTrie[] next;
    public WBTrie(){
      isWord=false;
      next = new WBTrie[256];
    }
  }

  private void WBaddTrie(WBTrie cur,String word){
    int idx;
    for (int i=0;i<word.length();i++){
      idx = word.charAt(i);
      if (cur.next[idx]==null)
        cur.next[idx]=new WBTrie();
      cur=cur.next[idx];
    }
    cur.isWord=true;
  }

  public boolean wordBreak(String s, List<String> wordDict) {
    WBTrie root = new WBTrie();
    for (String w:wordDict)
      WBaddTrie(root,w);
    int[] memo = new int[s.length()]; // 0 initially,1 true,2 false
    return WBdfs(0,s,root,memo);
  }

  private boolean WBdfs(int curIdx,String s,WBTrie root,int[] memo){
    if (curIdx>=s.length())
      return true;
    if (memo[curIdx]!=0)
      return memo[curIdx]==1?true:false;
    boolean res=false;
    WBTrie cur = root;
    for (int i=curIdx;i<s.length();i++){
      int idx = s.charAt(i);
      if (cur.next[idx]==null)
        break;
      if (cur.next[idx].isWord==true && WBdfs(i+1,s,root,memo)){
        res=true;
        break;
      }
      cur = cur.next[idx];
    }
    memo[curIdx]=res?1:2;
    return res;
  }

  public int fourSumCount3(int[] A, int[] B, int[] C, int[] D) {
    Map<Integer,Integer> ab = new HashMap<>();
    int N = A.length,ans=0;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        ab.put(A[i]+B[j],ab.getOrDefault(A[i]+B[j],0)+1);
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++)
        ans += ab.getOrDefault( -C[i]-D[j],0);
    return ans;
  }

  public int maxProfitAssignment2(int[] difficulty, int[] profit, int[] worker) {
    int dN = difficulty.length,wN = worker.length,maxProfit=0,taskIdx=0,curMax = 0;
    int[][] task = new int[dN][2];
    for (int i=0;i<dN;i++){
      task[i][0] = difficulty[i];
      task[i][1] =profit[i];
    }
    Arrays.sort(task, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]-b[0];
      }
    });
    for (int i=0;i<dN;i++){
      curMax = Math.max(curMax,task[i][1]);
      task[i][1] = curMax;
    }
    Arrays.sort(worker);
    for (int workerIdx=0;workerIdx<wN;workerIdx++){
      while (taskIdx<dN && task[taskIdx][0]<=worker[workerIdx])
        taskIdx++;
      if (taskIdx>0)
        maxProfit+=task[taskIdx-1][1];
    }
    return maxProfit;
  }

  public int[] maxSlidingWindow2(int[] nums, int k) {
    if (nums==null || nums.length==0)
      return new int[0];
    if (k==1)
      return nums;
    int N = nums.length,idx=0;
    int[] ans = new int[N-k+1];
    Deque<Integer> dq = new ArrayDeque<>();
    for (int i=0;i<N;i++){
      if (!dq.isEmpty() && i-dq.peekFirst()>=k)
        dq.pollFirst();
      while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i])
        dq.pollLast();
      dq.offer(i);
      if (i>=k-1)
        ans[idx++]=nums[dq.peekFirst()];
    }
    return ans;
  }

  public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> begin=new HashSet<>(),end = new HashSet<>(),all = new HashSet<>();
    for (String w:wordList)
      all.add(w);
    if (!all.contains(endWord))
      return 0;
    begin.add(beginWord);
    end.add(endWord);
    int len = 1;
    while (!begin.isEmpty() && !end.isEmpty()){
      len++;
      if (end.size()<begin.size()){
        Set<String> temp = begin;
        begin = end;
        end = temp;
      }
      all.removeAll(begin);
      Set<String> next = new HashSet<>();
      for (String curWord:begin){
        char[] cur = curWord.toCharArray();
        for (int i=0;i<cur.length;i++){
          char prev = cur[i];
          for (char c='a';c<='z';c++){
            cur[i]=c;
            String adj = new String(cur);
            if (end.contains(adj))
              return len;
            if (all.contains(adj))
              next.add(adj);
          }
          cur[i]=prev;
        }
      }
      begin = next;
    }
    return 0;
  }

  public int rangeBitwiseAnd2(int m, int n) {
    if (m==0)
      return 0;
    int mask = 1;
    while (m!=n){
      m>>=1;
      n>>=1;
      mask<<=1;
    }
    return m*mask;
  }

  public int leastInterval3(char[] tasks, int n) {
    int[] count = new int[26];
    for (char t:tasks)
      count[t-'A']++;
    PriorityQueue<Character> pq = new PriorityQueue<>(new Comparator<Character>() {
      @Override
      public int compare(Character a, Character b) {
        return count[b-'A']-count[a-'A'];
      }
    });
    int len = tasks.length,ans=0;
    Queue<Character> q = new LinkedList<>();
    for (int i=0;i<26;i++)
      if (count[i]!=0)
        pq.offer((char)(i+'A'));
    while (len>0){
      if (q.size() >n){
        char temp = q.poll();
        if (temp!=' ')
          pq.offer(temp);
      }
      Character out=null;
      if (!pq.isEmpty()){
        out = pq.poll();
        count[out-'A']--;
        len--;
      }
      if (out!=null && count[out-'A']!=0)
        q.offer(out);
      else
        q.offer(' ');
      ans++;
    }
    return ans;
  }

  public int leastInterval4(char[] tasks, int n) {
    int len = tasks.length, max=0,maxCount = 0;
    int[] count = new int[26];
    for (char t:tasks){
      count[t-'A']++;
      if (count[t-'A']==max)
        maxCount++;
      else if (count[t-'A']>max){
        max = count[t-'A'];
        maxCount=1;
      }
    }
    int partLen = n+1-maxCount;
    int partNum = max-1;
    int available = partLen*partNum;
    int others = len-max*maxCount;
    return Math.max(0,available-others)+len;
  }

  class GSLnode implements Comparable<GSLnode>{
    int pos,height;
    boolean isRight;
    public GSLnode(int p,int h,boolean r){
      pos = p;
      height = h;
      isRight = r;
    }

    @Override
    public int compareTo(GSLnode a) {
      if (a.pos!=pos)
        return pos-a.pos;
      else if (isRight!=a.isRight)
        return isRight?1:-1;
      else if (isRight)
        return height-a.height;
      else
        return a.height-height;
    }
  }

  public List<List<Integer>> getSkyline2(int[][] B) {
    List<List<Integer>> res = new ArrayList<>();
    TreeMap<Integer,Integer> active = new TreeMap<>(Collections.reverseOrder());
    int prevHeight = 0;
    active.put(0,1);
    List<GSLnode> heights = new ArrayList<>();
    for (int[] b:B){
      heights.add(new GSLnode(b[0],b[2],false));
      heights.add(new GSLnode(b[1],b[2],true));
    }
    Collections.sort(heights);
    for (GSLnode cur:heights){
      if (!cur.isRight)
        active.put(cur.height,active.getOrDefault(cur.height,0)+1);
      else{
        int count = active.get(cur.height);
        if (count==1)
          active.remove(cur.height);
        else
          active.put(cur.height,count-1);
      }
      int curMax = active.firstKey();
      if (curMax!=prevHeight){
        List<Integer> temp = new ArrayList<>();
        temp.add(cur.pos);
        temp.add(curMax);
        res.add(temp);
        prevHeight=curMax;
      }
    }
    return res;
  }

  class LRUCache {
    class BiLinkedList{
      int key,val;
      BiLinkedList pre,post;
      public BiLinkedList(int key,int val){
        this.key = key;
        this.val = val;
        pre=post=null;
      }
    }

    private void remove(BiLinkedList node){
      node.pre.post=node.post;
      node.post.pre = node.pre;
    }

    private void insertToLast(BiLinkedList node){
      tail.pre.post=node;
      node.pre = tail.pre;
      node.post=tail;
      tail.pre = node;
    }

    private void updateTime(BiLinkedList node){
      remove(node);
      insertToLast(node);
    }

    Map<Integer,BiLinkedList> cache;
    int capacity,curSize;
    BiLinkedList head,tail;
    public LRUCache(int capacity) {
      cache = new HashMap<>();
      this.capacity=capacity;
      curSize=0;
      head = new BiLinkedList(0,0);
      tail = new BiLinkedList(0,0);
      head.post=tail;
      tail.pre = head;
    }

    public int get(int key) {
      BiLinkedList cur = cache.get(key);
      if (cur==null)
        return -1;
      updateTime(cur);
      return cur.val;
    }

    public void put(int key, int value) {
      BiLinkedList cur = cache.get(key);
      if (cur!=null){
        cur.val=value;
        updateTime(cur);
      }
      else {
        cur = new BiLinkedList(key,value);
        cache.put(key,cur);
        insertToLast(cur);
        curSize++;
        if (curSize>capacity){
          cache.remove(head.post.key);
          curSize--;
          remove(head.post);
        }
      }
    }
  }
}
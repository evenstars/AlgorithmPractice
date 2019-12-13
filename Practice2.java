import com.sun.org.apache.bcel.internal.generic.FieldOrMethod;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Practice2 {
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

  public int minScoreTriangulation3(int[] A) {
    if (A == null || A.length < 3)
      return 0;
    if (A.length == 3)
      return A[0] * A[1] * A[2];
    int N = A.length;
    int[][] dp = new int[N][N];
    for (int len = 2; len < N; len++)
      for (int start = 0; start + len < N; start++) {
        int end = start + len;
        dp[start][end] = Integer.MAX_VALUE;
        for (int k = start + 1; k < end; k++)
          dp[start][end] = Math.min(dp[start][end], A[start] * A[end] * A[k] + dp[start][k] + dp[k][end]);
      }
    return dp[0][N - 1];
  }

  public int minimumDeleteSum2(String s1, String s2) {
    int sum = 0;
    char[] c1 = s1.toCharArray(), c2 = s2.toCharArray();
    for (char c : c1)
      sum += c;
    for (char c : c2)
      sum += c;
    int LCS = LCSascii(c1, c2);
    return sum - (LCS << 1);
  }

  private int LCSascii(char[] c1, char[] c2) {
    int n1 = c1.length, n2 = c2.length;
    int[][] dp = new int[n1 + 1][n2 + 1];
    for (int i = 0; i < n1; i++)
      for (int j = 0; j < n2; j++)
        if (c1[i] == c2[j])
          dp[i + 1][j + 1] = dp[i][j] + c1[i];
        else
          dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
    return dp[n1][n2];
  }

  public int minSubArrayLenOn(int s, int[] nums) {
    int N = nums.length, L, R, length = Integer.MAX_VALUE, curSum = 0;
    for (L = R = 0; R < N; R++) {
      curSum += nums[R];
      if (curSum < s)
        continue;
      while (L < R && curSum >= s)
        curSum -= nums[L++];
      int curLen = curSum >= s ? 1 : R - L + 2;
      length = Math.min(length, curLen);
    }
    return length == Integer.MAX_VALUE ? 0 : length;
  }

  public int minSubArrayLen_Onlgn(int s, int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length, length = Integer.MAX_VALUE;
    int[] ps = new int[N + 1];
    for (int i = 0; i < N; i++)
      ps[i + 1] = ps[i] + nums[i];
    for (int L = 0; L < N; L++) {
      int R = MSLgetR(ps, L, s);
      if (R != -1)
        length = Math.min(length, R - L);
    }
    return length == Integer.MAX_VALUE ? 0 : length;
  }

  private int MSLgetR(int[] ps, int start, int s) {
    int L = start + 1, R = ps.length - 1;
    while (L <= R) {
      int mid = (L + R) >> 1, val = ps[mid] - ps[start];
      if (val >= s)
        R = mid - 1;
      else
        L = mid + 1;
    }
    return L < ps.length ? L : -1;
  }

  public int coinChange2(int[] coins, int amount) {
    if (amount == 0)
      return 0;
    int[] memo = new int[amount + 1];
    return CChelper(amount, coins, memo);
  }

  private int CChelper(int amount, int[] coins, int[] memo) {
    if (amount == 0)
      return 0;
    if (memo[amount] != 0)
      return memo[amount];
    int res = Integer.MAX_VALUE;
    for (int i = coins.length - 1; i >= 0; i--) {
      if (coins[i] > amount)
        continue;
      int next = CChelper(amount - coins[i], coins, memo);
      if (next != -1)
        res = Math.min(res, next);
    }
    memo[amount] = res == Integer.MAX_VALUE ? -1 : res + 1;
    return memo[amount];
  }

  public int coinChange(int[] coins, int amount) {
    if (amount == 0)
      return 0;
    Arrays.sort(coins);
    int[] res = new int[]{Integer.MAX_VALUE};
    CCdfs(amount, coins.length - 1, 0, coins, res);
    return res[0] == Integer.MAX_VALUE ? -1 : res[0];
  }

  private void CCdfs(int amount, int coinIdx, int curCount, int[] coins, int[] res) {
    if (coinIdx < 0 || curCount + 1 >= res[0])
      return;
    for (int i = amount / coins[coinIdx]; i >= 0; i--) {
      int nextAmount = amount - i * coins[coinIdx];
      int nextCount = curCount + i;
      if (nextAmount > 0 && nextCount + 1 < res[0])
        CCdfs(nextAmount, coinIdx - 1, nextCount, coins, res);
      else {
        if (nextAmount == 0)
          res[0] = Math.min(res[0], nextCount);
        break;
      }
    }
  }

  public boolean isPowerOfFour(int num) {
    if (num <= 0)
      return false;
    int flag = 0x55555555, res = flag & num;
    return (num | flag) == flag && res != 0 && (res & (res - 1)) == 0;
  }

  public List<List<Integer>> findSubsequences(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    if (nums == null || nums.length < 2)
      return ans;
    FSShelper(0, nums, new ArrayList<>(), ans);
    return ans;
  }

  private void FSShelper(int curIdx, int[] nums, List<Integer> path, List<List<Integer>> ans) {
    if (path.size() >= 2)
      ans.add(new ArrayList<>(path));
    if (curIdx == nums.length)
      return;
    Set<Integer> appeared = new HashSet<>();
    for (int i = curIdx; i < nums.length; i++) {
      if (appeared.contains(nums[i]))
        continue;
      if (path.isEmpty() || nums[i] >= path.get(path.size() - 1)) {
        path.add(nums[i]);
        appeared.add(nums[i]);
        FSShelper(i + 1, nums, path, ans);
        path.remove(path.size() - 1);
      }
    }
  }

  public void setZeroes(int[][] M) {
    int R = M.length, C = M[0].length;
    boolean isRowZero = false, isColZero = false;
    for (int i = 0; i < R; i++)
      if (M[i][0] == 0) {
        isColZero = true;
        break;
      }
    for (int j = 0; j < C; j++)
      if (M[0][j] == 0) {
        isRowZero = true;
        break;
      }
    for (int r = 1; r < R; r++)
      for (int c = 1; c < C; c++)
        if (M[r][c] == 0)
          M[r][0] = M[0][c] = 0;
    for (int r = 1; r < R; r++)
      for (int c = 1; c < C; c++)
        if (M[r][0] == 0 || M[0][c] == 0)
          M[r][c] = 0;
    if (isRowZero)
      for (int c = 0; c < C; c++)
        M[0][c] = 0;
    if (isColZero)
      for (int r = 0; r < R; r++)
        M[r][0] = 0;
  }

  public int trap(int[] height) {
    if (height == null || height.length <= 2)
      return 0;
    int L = 0, R = height.length - 1, maxL = 0, maxR = 0, res = 0;
    while (L < R) {
      if (height[L] <= height[R]) {
        if (height[L] >= maxL)
          maxL = height[L];
        else
          res += maxL - height[L];
        L++;
      } else {
        if (height[R] >= maxR)
          maxR = height[R];
        else
          res += maxR - height[R];
        R--;
      }
    }
    return res;
  }

  public String minWindow(String s, String t) {
    if (s == null || t == null || t.length() == 0 || s.length() < t.length())
      return "";
    int[] remain = new int[256];
    boolean[] contains = new boolean[256];
    char[] cs = s.toCharArray(), ct = t.toCharArray();
    int flag = ct.length, L, R, resL = -1, resR = cs.length;
    for (char c : ct) {
      remain[c]++;
      contains[c] = true;
    }
    for (L = R = 0; R < cs.length; R++) {
      if (!contains[cs[R]])
        continue;
      if (remain[cs[R]]-- > 0)
        flag--;
      if (flag > 0)
        continue;
      while (flag == 0) {
        if (contains[cs[L]] && ++remain[cs[L]] > 0)
          flag++;
        L++;
      }
      if (resR - resL > R - L + 1) {
        resR = R;
        resL = L - 1;
      }
    }
    return resL == -1 ? "" : s.substring(resL, resR + 1);
  }

  class KthLargest {

    final PriorityQueue<Integer> pq;
    final int k;

    public KthLargest(int k, int[] nums) {
      this.k = k;
      pq = new PriorityQueue<>(k);
      for (int n : nums)
        add(n);
    }

    public int add(int val) {
      if (pq.size() < k)
        pq.offer(val);
      else if (val > pq.peek()) {
        pq.poll();
        pq.offer(val);
      }
      return pq.peek();
    }
  }

  public boolean isValidSerialization2(String preorder) {
    if (preorder == null || preorder.isEmpty())
      return true;
    String[] cp = preorder.split(",");
    int[] idx = new int[1];
    IVShelper(cp, idx);
    return idx[0] == cp.length;
  }

  private void IVShelper(String[] cp, int[] idx) {
    if (idx[0] >= cp.length || cp[idx[0]].equals("#")) {
      idx[0]++;
      return;
    }
    idx[0]++;
    IVShelper(cp, idx);
    IVShelper(cp, idx);
  }

  public boolean isValidSerialization(String preorder) {
    if (preorder == null || preorder.isEmpty())
      return true;
    String[] cp = preorder.split(",");
    int degree = 1, i;
    for (i = 0; i < cp.length; i++) {
      String c = cp[i];
      if (c.equals("#")) {
        degree--;
        if (degree <= 0)
          break;
      } else
        degree++;
    }
    return degree == 0 && i == cp.length - 1;
  }

  public int numTrees(int n) {
    if (n <= 0)
      return 0;
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    for (int len = 2; len <= n; len++)
      for (int k = 1; k <= len; k++)
        dp[len] += dp[k - 1] * dp[len - k];
    return dp[n];
  }

  public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
      StringBuilder sb = new StringBuilder();
      serialize(sb, root);
      return sb.toString().trim();
    }

    private void serialize(StringBuilder sb, TreeNode root) {
      if (root == null) {
        sb.append("# ");
        return;
      }
      sb.append(root.val);
      sb.append(' ');
      serialize(sb, root.left);
      serialize(sb, root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
      String[] seg = data.split(" ");
      return deserialize(seg, new int[1]);
    }

    private TreeNode deserialize(String[] data, int[] idx) {
      if (data[idx[0]].equals("#")) {
        idx[0]++;
        return null;
      }
      TreeNode cur = new TreeNode(Integer.valueOf(data[idx[0]++]));
      cur.left = deserialize(data, idx);
      cur.right = deserialize(data, idx);
      return cur;
    }
  }

  class PPTrie {
    int idx;
    PPTrie[] next;
    List<Integer> remainIsPal;

    public PPTrie() {
      this.idx = -1;
      next = new PPTrie[26];
      remainIsPal = new ArrayList<>();
    }
  }

  private void PPaddTrie(PPTrie root, String cur, int index) {
    for (int i = cur.length() - 1; i >= 0; i--) {
      int idx = cur.charAt(i) - 'a';
      if (PPisPal(cur, 0, i))
        root.remainIsPal.add(index);
      if (root.next[idx] == null)
        root.next[idx] = new PPTrie();
      root = root.next[idx];
    }
    root.remainIsPal.add(index);
    root.idx = index;
  }

  public void PPsearchTrie(PPTrie root, String word, int index, List<List<Integer>> res) {
    int idx, depth = word.length(), start;
    for (start = 0; start < depth; start++) {
      idx = word.charAt(start) - 'a';
      if (root.idx != -1 && root.idx != index && PPisPal(word, start, word.length() - 1))
        res.add(Arrays.asList(index, root.idx));
      if (root.next[idx] == null)
        return;
      root = root.next[idx];
    }
    for (int i : root.remainIsPal)
      if (i != index)
        res.add(Arrays.asList(index, i));
  }

  private boolean PPisPal(String w, int start, int end) {
    while (start < end)
      if (w.charAt(start++) != w.charAt(end--))
        return false;
    return true;
  }

  public List<List<Integer>> palindromePairs(String[] words) {
    List<List<Integer>> ans = new ArrayList<>();
    if (words == null || words.length == 0)
      return ans;
    PPTrie root = new PPTrie();
    for (int i = 0; i < words.length; i++)
      PPaddTrie(root, words[i], i);
    for (int i = 0; i < words.length; i++)
      PPsearchTrie(root, words[i], i, ans);
    return ans;
  }

  public int[] findOrder1(int N, int[][] P) {
    if (N == 0)
      return new int[0];
    if (N == 1)
      return new int[]{0};
    List<Integer>[] graph = FOgetGraph(N, P);
    boolean[] hasCycle = new boolean[1], visited = new boolean[N], onStack = new boolean[N];
    for (int i = 0; i < N && !hasCycle[0]; i++)
      if (!visited[i])
        FOhasCycle(graph, i, onStack, hasCycle, visited);
    if (hasCycle[0])
      return new int[0];
    Arrays.fill(visited, false);
    int[] res = new int[N], idx = new int[]{N - 1};
    for (int i = 0; i < N; i++)
      if (!visited[i])
        FOgetTopo(graph, i, res, idx, visited);
    return res;
  }

  private void FOgetTopo(List<Integer>[] graph, int cur, int[] res, int[] idx, boolean[] visited) {
    visited[cur] = true;
    for (int adj : graph[cur])
      if (!visited[adj])
        FOgetTopo(graph, adj, res, idx, visited);
    res[idx[0]--] = cur;
  }

  private void FOhasCycle(List<Integer>[] graph, int cur, boolean[] onStack, boolean[] hasCycle, boolean[] visited) {
    visited[cur] = true;
    onStack[cur] = true;
    for (int adj : graph[cur])
      if (hasCycle[0])
        return;
      else if (!visited[adj])
        FOhasCycle(graph, adj, onStack, hasCycle, visited);
      else if (onStack[adj]) {
        hasCycle[0] = true;
        return;
      }
    onStack[cur] = false;
  }

  private List<Integer>[] FOgetGraph(int N, int[][] P) {
    List<Integer>[] graph = new List[N];
    for (int i = 0; i < N; i++)
      graph[i] = new ArrayList<>();
    for (int[] p : P)
      graph[p[1]].add(p[0]);
    return graph;
  }

  public int[] findOrder(int N, int[][] P) {
    if (N == 0)
      return new int[0];
    if (N == 1)
      return new int[]{0};
    List<Integer>[] graph = FOgetGraph(N, P);
    boolean[] visited = new boolean[N];
    int[] inDegree = new int[N], res = new int[N];
    for (int[] p : P)
      inDegree[p[0]]++;
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < N; i++)
      if (inDegree[i] == 0) {
        q.offer(i);
        visited[i] = true;
      }
    int idx = 0;
    while (!q.isEmpty()) {
      int cur = q.poll();
      res[idx++] = cur;
      for (int adj : graph[cur])
        if (!visited[adj]) {
          inDegree[adj]--;
          if (inDegree[adj] == 0) {
            visited[adj] = true;
            q.offer(adj);
          }
        }
    }
    return idx == N ? res : new int[0];
  }

  public boolean canPartition1(int[] nums) {
    if (nums == null || nums.length < 2)
      return false;
    int sum = 0;
    for (int n : nums)
      sum += n;
    if ((sum & 1) == 1)
      return false;
    int target = sum >> 1;
    return CPfindTarget(nums, target);
  }

  private boolean CPfindTarget(int[] nums, int target) {
    int[][] memo = new int[nums.length][target + 1]; // 0-uninitialed, 1-true,2-false
    return CPhelper(nums, 0, target, memo);
  }

  private boolean CPhelper(int[] nums, int cur, int remain, int[][] memo) {
    if (remain == 0)
      return true;
    if (cur == nums.length || remain < 0)
      return false;
    if (memo[cur][remain] != 0)
      return memo[cur][remain] == 1;
    memo[cur][remain] = CPhelper(nums, cur + 1, remain, memo) || CPhelper(nums, cur + 1, remain - nums[cur], memo) ? 1 : 2;
    return memo[cur][remain] == 1;
  }

  public boolean canPartition(int[] nums) {
    if (nums == null || nums.length < 2)
      return false;
    int sum = 0;
    for (int n : nums)
      sum += n;
    if ((sum & 1) == 1)
      return false;
    Arrays.sort(nums);
    return CPfindTarget(nums, 0, 0, sum >> 1);
  }

  private boolean CPfindTarget(int[] nums, int start, int curSum, int target) {
    if (curSum == target)
      return true;
    if (start == nums.length || curSum > target)
      return false;
    for (int i = start; i < nums.length; i++)
      if (i != start && nums[i] == nums[i - 1])
        continue;
      else if (CPfindTarget(nums, i + 1, curSum + nums[i], target))
        return true;
    return false;
  }

  public int countRangeSum(int[] nums, int lower, int upper) {
    long[] ps = new long[nums.length + 1], aux = new long[nums.length + 1];
    for (int i = 0; i < nums.length; i++)
      ps[i + 1] = ps[i] + nums[i];
    return CRSmergeSort(ps, aux, 0, ps.length - 1, lower, upper);
  }

  private int CRSmergeSort(long[] ps, long[] aux, int start, int end, int lower, int upper) {
    if (start >= end)
      return 0;
    int mid = (start + end) >> 1, left = CRSmergeSort(ps, aux, start, mid, lower, upper), right = CRSmergeSort(ps, aux, mid + 1, end, lower, upper);
    return left + right + CRSmerge(ps, aux, start, end, lower, upper);
  }

  private int CRSmerge(long[] ps, long[] aux, int start, int end, int lower, int upper) {
    int mid = (start + end) >> 1, L = mid + 1, R = mid + 1, s = mid + 1, res = 0, idx = start;
    for (int i = start; i <= end; i++)
      aux[i] = ps[i];
    for (int i = start; i <= mid; i++) {
      while (L <= end && aux[L] - aux[i] < lower)
        L++;
      while (R <= end && aux[R] - aux[i] <= upper)
        R++;
      while (s <= end && aux[s] <= aux[i])
        ps[idx++] = aux[s++];
      ps[idx++] = aux[i];
      res += R - L;
    }
    return res;
  }

  public int longestUnivaluePath(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = new int[1];
    LUPhelper(root, res);
    return res[0] - 1;
  }

  private int LUPhelper(TreeNode root, int[] res) {
    if (root == null)
      return 0;
    int left = LUPhelper(root.left, res), right = LUPhelper(root.right, res), curSum = 1, ans = 0;
    if (root.left != null && root.left.val == root.val) {
      curSum += left;
      ans = left;
    }
    if (root.right != null && root.right.val == root.val) {
      curSum += right;
      ans = Math.max(ans, right);
    }
    res[0] = Math.max(res[0], curSum);
    return ans + 1;
  }

  public int dayOfYear(String date) {
    int[] D = new int[]{31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    String[] cals = date.split("-");
    int year = Integer.valueOf(cals[0]), month = Integer.valueOf(cals[1]), day = Integer.valueOf(cals[2]);
    D[1] = year % 400 == 0 ? 29 : year % 100 != 0 && year % 4 == 0 ? 29 : 28;
    int ans = 0;
    for (int i = 0; i < month - 1; i++)
      ans += D[i];
    ans += day;
    return ans;
  }

  int DIVIDE = (int) Math.pow(10, 9) + 7;

  public int numRollsToTarget(int d, int f, int target) {
    if (target > d * f || target < d)
      return 0;
    if (d == 1)
      return 1;
    int[][] memo = new int[d + 1][target + 1];
    return NRThelper(d, f, target, memo);
  }

  private int NRThelper(int d, int f, int target, int[][] memo) {
    if (target > d * f || target < d)
      return 0;
    if (d == 1)
      return 1;
    if (memo[d][target] != 0)
      return memo[d][target];
    for (int i = 1; i <= f; i++) {
      memo[d][target] += NRThelper(d - 1, f, target - i, memo);
      memo[d][target] %= DIVIDE;
    }
    return memo[d][target];
  }

  public int maxRepOpt1(String text) {
    if (text == null || text.isEmpty())
      return 0;
    char[] T = text.toCharArray();
    int N = T.length, L, R;
    List<int[]>[] appeared = new List[26];
    for (int i = 0; i < 26; i++)
      appeared[i] = new ArrayList<>();
    for (L = R = 0; R < N; R++)
      if (T[L] != T[R]) {
        int idx = T[L] - 'a';
        appeared[idx].add(new int[]{L, R - 1});
        L = R;
      }
    appeared[T[L] - 'a'].add(new int[]{L, N - 1});
    int maxLen = 0, curLen = 1;
    for (int i = 0; i < 26; i++) {
      if (appeared[i].isEmpty())
        continue;
      List<int[]> itv = appeared[i];
      int size = itv.size();
      for (int j = 0; j < size; j++) {
        if (j + 1 < size && itv.get(j)[1] + 2 == itv.get(j + 1)[0]) {
          curLen = itv.get(j)[1] - itv.get(j)[0] + 1 + itv.get(j + 1)[1] - itv.get(j + 1)[0] + 1;
          if (itv.size() > 2)
            curLen++;
        } else {
          curLen = itv.get(j)[1] - itv.get(j)[0] + 1;
          if (itv.size() > 1)
            curLen++;
        }
        maxLen = Math.max(maxLen, curLen);
      }
    }
    return maxLen;
  }

  class MajorityChecker1 {
    int[] data;

    public MajorityChecker1(int[] arr) {
      data = arr;
    }

    public int query(int left, int right, int threshold) {
      int cand = 0, freq = 0;
      for (int i = left; i <= right; i++)
        if (data[i] == cand)
          freq++;
        else {
          freq--;
          if (freq < 0) {
            cand = data[i];
            freq = 1;
          }
        }
      freq = 0;
      for (int i = left; i <= right; i++)
        if (data[i] == cand)
          freq++;
      return freq >= threshold ? cand : -1;
    }
  }

  public List<List<Integer>> pathSum(TreeNode root, int sum) {
    List<List<Integer>> ans = new ArrayList<>();
    if (root == null)
      return ans;
    PShelper(root, sum, new ArrayList<>(), ans);
    return ans;
  }

  private void PShelper(TreeNode root, int sum, List<Integer> path, List<List<Integer>> ans) {
    if (root == null)
      return;
    if (root.left == null && root.right == null) {
      if (sum == root.val) {
        path.add(root.val);
        ans.add(new ArrayList<>(path));
        path.remove(path.size() - 1);
      }
      return;
    }
    path.add(root.val);
    PShelper(root.left, sum - root.val, path, ans);
    PShelper(root.right, sum - root.val, path, ans);
    path.remove(path.size() - 1);
  }

  public int lengthOfLIS(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    List<Integer> sorted = new ArrayList<>();
    for (int n : nums)
      if (sorted.isEmpty() || n > sorted.get(sorted.size() - 1))
        sorted.add(n);
      else {
        int smallestLargeIdx = smallestLargeIdx(sorted, n);
        if (smallestLargeIdx != -1)
          sorted.set(smallestLargeIdx, n);
      }
    return sorted.size();
  }

  private int smallestLargeIdx(List<Integer> sorted, int target) {
    int start = 0, end = sorted.size() - 1;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (sorted.get(mid) < target)
        start = mid + 1;
      else if (sorted.get(mid) > target)
        end = mid - 1;
      else
        return -1;
    }
    return start < sorted.size() ? start : -1;
  }

  public TreeNode bstFromPreorder(int[] preorder) {
    if (preorder == null || preorder.length == 0)
      return null;
    return BFPhelper(preorder, new int[1], Long.MIN_VALUE, Long.MAX_VALUE);
  }

  private TreeNode BFPhelper(int[] po, int[] idx, long L, long R) {
    if (idx[0] >= po.length)
      return null;
    TreeNode cur = null;
    int curVal = po[idx[0]];
    if (curVal > L && curVal < R) {
      cur = new TreeNode(po[idx[0]++]);
      cur.left = BFPhelper(po, idx, L, Math.min(R, cur.val));
      cur.right = BFPhelper(po, idx, Math.max(L, cur.val), R);
    }
    return cur;
  }

  class MedianFinder {

    /**
     * initialize your data structure here.
     */
    PriorityQueue<Integer> smallPart, largePart;
    int size;

    public MedianFinder() {
      smallPart = new PriorityQueue<>(Collections.reverseOrder());
      largePart = new PriorityQueue<>();
      size = 0;
    }

    public void addNum(int num) {
      size++;
      int smallSize = size >> 1, largeSize = size - smallSize;
      if (largePart.isEmpty())
        largePart.offer(num);
      else if (num >= largePart.peek()) {
        largePart.offer(num);
        if (largePart.size() > largeSize)
          smallPart.offer(largePart.poll());
      } else {
        smallPart.offer(num);
        if (smallPart.size() > smallSize)
          largePart.offer(smallPart.poll());
      }
    }

    public double findMedian() {
      if ((size & 1) == 1)
        return largePart.peek();
      else
        return (double) (largePart.peek() + smallPart.peek()) / 2;
    }
  }

  public int bulbSwitch(int n) {
    return (int) Math.sqrt(n);
  }

  public List<String> findItinerary(List<List<String>> T) {
    LinkedList<String> ans = new LinkedList<>();
    Map<String, PriorityQueue<String>> graph = new HashMap<>();
    for (List<String> t : T) {
      PriorityQueue<String> cur = graph.get(t.get(0));
      if (cur == null) {
        cur = new PriorityQueue<>();
        graph.put(t.get(0), cur);
      }
      cur.offer(t.get(1));
    }
    FIhelper(graph, "JFK", ans);
    return ans;
  }

  private void FIhelper(Map<String, PriorityQueue<String>> graph, String cur, LinkedList<String> ans) {
    PriorityQueue<String> nexts = graph.get(cur);
    while (nexts != null && !nexts.isEmpty()) {
      String adj = nexts.poll();
      FIhelper(graph, adj, ans);
    }
    ans.addFirst(cur);
  }

  public int maxSumTwoNoOverlap(int[] A, int L, int M) {
    int N = A.length, ans = 0, maxL = 0, maxM = 0;
    int[] ps = new int[N + 1];
    for (int i = 0; i < N; i++)
      ps[i + 1] = ps[i] + A[i];
    for (int i = L + M; i <= N; i++) {
      maxL = Math.max(maxL, ps[i - M] - ps[i - M - L]);
      maxM = Math.max(maxM, ps[i - L] - ps[i - M - L]);
      int LM = maxL + ps[i] - ps[i - M];
      int ML = maxM + ps[i] - ps[i - L];
      ans = Math.max(ans, Math.max(LM, ML));
    }
    return ans;
  }

  public int numDecodings(String s) {
    if (s == null || s.isEmpty())
      return 0;
    int[] memo = new int[s.length()];
    Arrays.fill(memo, -1);
    return NDhelper(s.toCharArray(), 0, memo);
  }

  private int NDhelper(char[] cs, int curIdx, int[] memo) {
    if (curIdx == cs.length)
      return 1;
    if (memo[curIdx] != -1)
      return memo[curIdx];
    int res = 0;
    if (cs[curIdx] != '0') {
      res += NDhelper(cs, curIdx + 1, memo);
      if (curIdx + 1 < cs.length && (cs[curIdx] == '1' || (cs[curIdx] == '2') && cs[curIdx + 1] <= '6'))
        res += NDhelper(cs, curIdx + 2, memo);
    }
    memo[curIdx] = res;
    return res;
  }

  class MajorityChecker {

    class STNode {
      int majority, count, L, R;
      STNode left, right;

      public STNode(int L, int R) {
        this.L = L;
        this.R = R;
        left = right = null;
        majority = count = 0;
      }

      public void pushDown() {
        if (L == R)
          return;
        if (left == null) {
          int mid = (L + R) >> 1;
          left = new STNode(L, mid);
          right = new STNode(mid + 1, R);
        }
      }

      public void updateByVal(int M, int C) {
        majority = M;
        count = C;
      }

      public void updateFromSon() {
        if (left.majority == right.majority) {
          majority = left.majority;
          count = left.count + right.count;
        } else {
          majority = left.count >= right.count ? left.majority : right.majority;
          count = Math.abs(left.count - right.count);
        }
      }
    }

    private STNode root;
    private Map<Integer, List<Integer>> idx;

    public MajorityChecker(int[] arr) {
      root = new STNode(0, arr.length - 1);
      idx = new HashMap<>();
      for (int i = 0; i < arr.length; i++) {
        idx.computeIfAbsent(arr[i], a -> new ArrayList<>()).add(i);
        updateSingle(root, i, arr[i]);
      }
    }

    private void updateSingle(STNode root, int pos, int val) {
      if (root.L == root.R) {
        root.updateByVal(val, 1);
        return;
      }
      int mid = (root.L + root.R) >> 1;
      root.pushDown();
      if (pos <= mid)
        updateSingle(root.left, pos, val);
      else
        updateSingle(root.right, pos, val);
      root.updateFromSon();
    }

    public int query(int left, int right, int threshold) {
      int[] M = new int[1], C = new int[]{-1};
      query(root, left, right, M, C);
      int freq = getFreq(M[0], left, right);
      return freq >= threshold ? M[0] : -1;
    }

    private void query(STNode root, int left, int right, int[] M, int[] count) {
      if (root.R < left || root.L > right)
        return;
      if (left <= root.L && root.R <= right) {
        mergeQuery(M, count, root.majority, root.count);
        return;
      }
      query(root.left, left, right, M, count);
      query(root.right, left, right, M, count);
    }

    private void mergeQuery(int[] majority, int[] count, int curM, int curC) {
      if (count[0] == -1) {
        majority[0] = curM;
        count[0] = curC;
      } else if (majority[0] == curM)
        count[0] += curC;
      else {
        majority[0] = count[0] >= curC ? majority[0] : curM;
        count[0] = Math.abs(count[0] - curC);
      }
    }

    public int getFreq(int target, int left, int right) {
      List<Integer> index = idx.get(target);
      int rightIdx = findIdx(index, right), leftIdx = findIdx(index, left);
      if (rightIdx < 0)
        return 0;
      else if (leftIdx < 0)
        return rightIdx - leftIdx;
      else if (index.get(leftIdx) == left)
        return rightIdx - leftIdx + 1;
      else
        return rightIdx - leftIdx;
    }

    private int findIdx(List<Integer> idx, int pos) {
      int start = 0, end = idx.size() - 1, mid;
      while (start <= end) {
        mid = (start + end) >> 1;
        if (idx.get(mid) < pos)
          start = mid + 1;
        else if (idx.get(mid) > pos)
          end = mid - 1;
        else
          return mid;
      }
      return end;
    }
  }

  public TreeNode deleteNode(TreeNode root, int key) {
    if (root == null)
      return null;
    if (key < root.val)
      root.left = deleteNode(root.left, key);
    else if (key > root.val)
      root.right = deleteNode(root.right, key);
    else {
      if (root.right == null)
        return root.left;
      else if (root.left == null)
        return root.right;
      TreeNode after = findMin(root.right);
      after.right = deleteMin(root.right);
      after.left = root.left;
      root = after;
    }
    return root;
  }

  private TreeNode deleteMin(TreeNode root) {
    if (root.left == null)
      return root.right;
    root.left = deleteMin(root.left);
    return root;
  }

  private TreeNode findMin(TreeNode root) {
    if (root.left == null)
      return root;
    return findMin(root.left);
  }

  public int getMoneyAmount(int n) {
    if (n == 1)
      return 0;
    int[][] dp = new int[n + 1][n + 1];
    return GMAhelper(1, n, dp);
  }

  private int GMAhelper(int start, int end, int[][] dp) {
    if (start >= end)
      return 0;
    if (dp[start][end] != 0)
      return dp[start][end];
    int res = Integer.MAX_VALUE;
    for (int i = start; i <= end; i++) {
      int curMax = i + Math.max(GMAhelper(start, i - 1, dp), GMAhelper(i + 1, end, dp));
      res = Math.min(curMax, res);
    }
    dp[start][end] = res;
    return res;
  }

  public int longestConsecutive(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int ans = 0;
    Map<Integer, Integer> record = new HashMap<>();
    for (int n : nums) {
      if (record.containsKey(n))
        continue;
      int left = record.getOrDefault(n - 1, 0), right = record.getOrDefault(n + 1, 0), size = left + right + 1;
      record.put(n, size);
      ans = Math.max(ans, size);
      if (left != 0)
        record.put(n - left, size);
      if (right != 0)
        record.put(n + right, size);
    }
    return ans;
  }

  public List<String> fullJustify(String[] words, int maxWidth) {
    StringBuilder sb = new StringBuilder();
    List<String> ans = new ArrayList<>();
    int cur = 0, N = words.length;
    while (cur < N) {
      int minLen = 0, wordLen = 0, curIdx = cur;
      while (curIdx < N && (minLen + words[curIdx].length()) <= maxWidth) {
        minLen += words[curIdx].length() + 1;
        wordLen += words[curIdx].length();
        curIdx++;
      }
      if (curIdx >= N) {
        for (int i = cur; i < curIdx; i++) {
          sb.append(words[i]);
          sb.append(' ');
        }
        if (sb.length() > maxWidth)
          ans.add(sb.toString().trim());
        else {
          for (int e = maxWidth - sb.length(); e > 0; e--)
            sb.append(' ');
          ans.add(sb.toString());
        }
        break;
      }
      int empty = maxWidth - wordLen, slotNum = curIdx - cur - 1, emptyBase, emptyMore;
      if (slotNum != 0) {
        emptyBase = empty / slotNum;
        emptyMore = empty % slotNum;
      } else {
        emptyBase = maxWidth - wordLen;
        emptyMore = 0;
      }
      for (int i = cur; i < curIdx; i++) {
        sb.append(words[i]);
        if (cur == curIdx - 1 || i < curIdx - 1)
          for (int e = i - cur < emptyMore ? emptyBase + 1 : emptyBase; e > 0; e--)
            sb.append(' ');
      }
      ans.add(sb.toString());
      cur = curIdx;
      sb.delete(0, sb.length());
    }
    return ans;
  }

  public void recoverTree(TreeNode root) {
    if (root == null)
      return;
    TreeNode error1 = null, error2 = null, last = new TreeNode(Integer.MIN_VALUE);
    while (root != null)
      if (root.left == null) {
        if (root.val < last.val) {
          if (error1 == null)
            error1 = last;
          if (error1 != null)
            error2 = root;
        }
        last = root;
        root = root.right;
      } else {
        TreeNode prev = root.left;
        while (prev.right != null && prev.right != root)
          prev = prev.right;
        if (prev.right == null) {
          prev.right = root;
          root = root.left;
        } else {
          if (root.val < last.val) {
            if (error1 == null)
              error1 = last;
            if (error1 != null)
              error2 = root;
          }
          last = root;
          prev.right = null;
          root = root.right;
        }
      }
    int temp = error1.val;
    error1.val = error2.val;
    error2.val = temp;
  }

  public boolean isMatch(String s, String p) {
    char[] S = s.toCharArray(), P = p.toCharArray();
    int sLen = S.length, pLen = P.length;
    boolean[][] dp = new boolean[sLen + 1][pLen + 1];
    dp[0][0] = true;
    for (int i = 2; i <= pLen && P[i - 1] == '*'; i += 2)
      dp[0][i] = dp[0][i - 2];
    for (int i = 1; i <= sLen; i++)
      for (int j = 1; j <= pLen; j++)
        if (S[i - 1] == P[j - 1] || P[j - 1] == '.')
          dp[i][j] = dp[i - 1][j - 1];
        else if (P[j - 1] == '*')
          if (S[i - 1] != P[j - 2] && P[j - 2] != '.')
            dp[i][j] = dp[i][j - 2];
          else
            dp[i][j] = dp[i][j - 2] || dp[i][j - 1] || dp[i - 1][j];
    return dp[sLen][pLen];
  }

  public String orderlyQueue(String S, int K) {
    if (K == 1) {
      String res = S;
      for (int i = 0; i < S.length(); i++) {
        String temp = S.substring(i, S.length()) + S.substring(0, i);
        if (res.compareTo(temp) > 0)
          res = temp;
      }
      return res;
    }
    char[] cs = S.toCharArray();
    Arrays.sort(cs);
    return new String(cs);
  }

  public int calculate(String s) {
    if (s == null || s.isEmpty())
      return 0;
    int N = s.length(), opsIdx = 0, valIdx = 0, newVal;
    char[] ops = new char[N], cs = s.toCharArray();
    int[] vals = new int[N], idx = new int[1];
    vals[valIdx++] = nextVal(cs, idx);
    while (idx[0] < N) {
      ops[opsIdx++] = nextOps(cs, idx);
      vals[valIdx++] = nextVal(cs, idx);
      if (ops[opsIdx - 1] == '*') {
        newVal = vals[--valIdx] * vals[--valIdx];
        opsIdx--;
        vals[valIdx++] = newVal;
      } else if (ops[opsIdx - 1] == '/') {
        newVal = vals[valIdx - 2] / vals[valIdx - 1];
        valIdx -= 2;
        vals[valIdx++] = newVal;
        opsIdx--;
      }
    }
    int res = vals[0];
    for (int i = 0, v = 1; i < opsIdx; i++, v++)
      if (ops[i] == '+')
        res += vals[v];
      else
        res -= vals[v];
    return res;
  }

  private int nextVal(char[] cs, int[] idx) {
    int res = 0, N = cs.length;
    while (idx[0] < N && ((cs[idx[0]] >= '0' && cs[idx[0]] <= '9') || cs[idx[0]] == ' ')) {
      if (cs[idx[0]] != ' ')
        res = res * 10 + cs[idx[0]] - '0';
      idx[0]++;
    }
    return res;
  }

  private char nextOps(char[] cs, int[] idx) {
    int N = cs.length;
    while (idx[0] < N && (cs[idx[0]] < '0' || cs[idx[0]] > '9'))
      if (cs[idx[0]++] != ' ')
        return cs[idx[0] - 1];
    return ' ';
  }

  public List<Integer> findDisappearedNumbers(int[] nums) {
    List<Integer> ans = new ArrayList<>();
    if (nums == null || nums.length == 0)
      return ans;
    for (int n : nums)
      if (nums[Math.abs(n) - 1] > 0)
        nums[Math.abs(n) - 1] *= -1;
    for (int i = 0; i < nums.length; i++)
      if (nums[i] > 0)
        ans.add(i + 1);
    return ans;
  }

  public int removeDuplicates(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int slow, fast, count, N = nums.length;
    for (slow = 0, fast = 0, count = 0; fast < N; fast++)
      if (fast == 0 || nums[fast] != nums[fast - 1] || count < 2) {
        if (fast != 0 && nums[fast] != nums[fast - 1])
          count = 0;
        nums[slow++] = nums[fast];
        count++;
      }
    return slow;
  }

  public int oddEvenJumps(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    int N = A.length, res = 1;
    boolean[] odd = new boolean[N], even = new boolean[N];
    odd[N - 1] = even[N - 1] = true;
    TreeMap<Integer, Integer> record = new TreeMap<>();
    record.put(A[N - 1], N - 1);
    for (int i = N - 2; i >= 0; i--) {
      Map.Entry<Integer, Integer> nextEven = record.ceilingEntry(A[i]), nextOdd = record.floorEntry(A[i]);
      if (nextEven != null)
        odd[i] = even[nextEven.getValue()];
      if (nextOdd != null)
        even[i] = odd[nextOdd.getValue()];
      if (odd[i])
        res++;
      record.put(A[i], i);
    }
    return res;
  }

  public String convertToBase7(int num) {
    if (num == 0)
      return "0";
    StringBuilder sb = new StringBuilder();
    boolean isNeg = num < 0;
    num = Math.abs(num);
    while (num != 0) {
      sb.append(num % 7);
      num /= 7;
    }
    if (isNeg)
      sb.append('-');
    return sb.reverse().toString();
  }

  public List<Integer> largestDivisibleSubset(int[] nums) {
    List<Integer> ans = new ArrayList<>();
    if (nums == null || nums.length == 0)
      return ans;
    Arrays.sort(nums);
    int N = nums.length, maxId = 0, maxLen = 1;
    int[] last = new int[N], length = new int[N];
    length[0] = 1;
    for (int i = 1; i < N; i++) {
      int len = 0, lenId = i;
      for (int j = i - 1; j >= 0; j--) {
        if (nums[i] % nums[j] != 0)
          continue;
        if (length[j] > len) {
          lenId = j;
          len = length[j];
        }
      }
      last[i] = lenId;
      length[i] = len + 1;
      if (length[i] > maxLen) {
        maxLen = length[i];
        maxId = i;
      }
    }
    while (last[maxId] != maxId) {
      ans.add(nums[maxId]);
      maxId = last[maxId];
    }
    ans.add(nums[maxId]);
    return ans;
  }

  public int maxCoins(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length;
    int[] res = new int[N + 2];
    res[0] = res[N + 1] = 1;
    for (int i = 0; i < N; i++)
      res[i + 1] = nums[i];
    int[][] memo = new int[N + 2][N + 2];
    return MChelper(res, memo, 0, N + 1);
  }

  private int MChelper(int[] res, int[][] memo, int left, int right) {
    if (right - left <= 1)
      return 0;
    if (memo[left][right] != 0)
      return memo[left][right];
    for (int k = left + 1; k < right; k++)
      memo[left][right] = Math.max(memo[left][right], res[k] * res[left] * res[right] + MChelper(res, memo, left, k) + MChelper(res, memo, k, right));
    return memo[left][right];
  }

  public int[] countBits(int num) {
    if (num == 0)
      return new int[1];
    int[] dp = new int[num + 1];
    for (int i = 1; i <= num; i++)
      dp[i] = (i & 1) + dp[i >> 1];
    return dp;
  }

  public int countPrimes(int n) {
    if (n <= 2)
      return 0;
    if (n <= 4)
      return n - 2;
    int res = 0;
    boolean[] isNotPrime = new boolean[n];
    for (int i = 2; i < n; i++)
      if (!isNotPrime[i]) {
        res++;
        int time = 2, next;
        while ((next = time * i) < n) {
          isNotPrime[next] = true;
          time++;
        }
      }
    return res;
  }

  public int lengthOfLIS1(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    List<Integer> LIS = new ArrayList<>();
    for (int n : nums)
      if (LIS.isEmpty() || n > LIS.get(LIS.size() - 1))
        LIS.add(n);
      else {
        int idx = LISgetIndex(LIS, n);
        LIS.set(idx, n);
      }
    return LIS.size();
  }

  private int LISgetIndex(List<Integer> LIS, int val) {
    int start = 0, end = LIS.size() - 1, mid;
    while (start <= end) {
      mid = (start + end) >> 1;
      if (LIS.get(mid) < val)
        start = mid + 1;
      else if (LIS.get(mid) > val)
        end = mid - 1;
      else
        return mid;
    }
    return start;
  }

  public TreeNode deleteNode1(TreeNode root, int key) {
    if (root == null)
      return null;
    if (root.val > key)
      root.left = deleteNode1(root.left, key);
    else if (root.val < key)
      root.right = deleteNode1(root.right, key);
    else {
      if (root.left == null)
        root = root.right;
      else if (root.right == null)
        root = root.left;
      else {
        TreeNode next = getMin1(root.right);
        next.right = deleteMin1(root.right);
        next.left = root.left;
        root = next;
      }
    }
    return root;
  }

  private TreeNode deleteMin1(TreeNode root) {
    if (root.left != null) {
      root.left = deleteMin1(root.left);
      return root;
    } else
      return root.right;
  }

  private TreeNode getMin1(TreeNode root) {
    while (root.left != null)
      root = root.left;
    return root;
  }

  public int firstMissingPositive(int[] nums) {
    int N = nums.length;
    for (int i = 0; i < N; i++)
      while (nums[i] > 0 && nums[i] < N && nums[nums[i] - 1] != nums[i])
        exchange(nums, nums[i] - 1, i);
    for (int i = 0; i < N; i++)
      if (nums[i] != i + 1)
        return i + 1;
    return N + 1;
  }

  private void exchange(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  public int maxLevelSum(TreeNode root) {
    if (root == null)
      return 0;
    int cur = 0, maxLevel = -1, maxVal = Integer.MIN_VALUE;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
      int size = q.size(), curVal = 0;
      cur++;
      for (int i = 0; i < size; i++) {
        TreeNode temp = q.poll();
        curVal += temp.val;
        if (temp.left != null)
          q.offer(temp.left);
        if (temp.right != null)
          q.offer(temp.right);
      }
      if (curVal > maxVal) {
        maxVal = curVal;
        maxLevel = cur;
      }
    }
    return maxLevel;
  }

  public int maxSubArray(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int max = Integer.MIN_VALUE, cur = 0;
    for (int n : nums) {
      if (cur >= 0)
        cur += n;
      else
        cur = n;
      max = Math.max(cur, max);
    }
    return max;
  }

  public int distributeCoins(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = new int[1];
    DChelper(root, res);
    return res[0];
  }

  private int DChelper(TreeNode root, int[] res) {
    if (root == null)
      return 0;
    int left = DChelper(root.left, res), right = DChelper(root.right, res);
    res[0] += Math.abs(left) + Math.abs(right);
    return 1 - root.val + left + right;
  }

  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
    if (k < 1 || t < 0)
      return false;
    Map<Long, Long> record = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
      long cur = (long) nums[i] - Integer.MIN_VALUE;
      long bucket = cur / ((long) t + 1);
      Long sm, bg;
      if (record.containsKey(bucket)
              || ((sm = record.get(bucket - 1)) != null && cur - sm <= t)
              || ((bg = record.get(bucket + 1)) != null && bg - cur <= t))
        return true;
      record.put(bucket, cur);
      if (record.size() > k) {
        long last = ((long) nums[i - k] - Integer.MIN_VALUE) / ((long) t + 1);
        record.remove(last);
      }
    }
    return false;
  }

  public int calculate2(String s) {
    // -1:+ -2:- -3:( -4:)
    if (s == null || s.isEmpty())
      return 0;
    int N = s.length(), pIdx = 0, numsIdx = 0;
    int[] par = new int[N], nums = new int[N], index = new int[1];
    char[] cs = s.toCharArray();
    while (index[0] < N) {
      int next = C2next(cs, index);
      if (next == -3)
        par[pIdx++] = numsIdx;
      else if (next == -4) {
        int start = par[--pIdx], res = 0, newStart = start;
        while (start < numsIdx) {
          if (nums[start] == -1)
            res += nums[++start];
          else if (nums[start] == -2)
            res -= nums[++start];
          else
            res = nums[start];
          start++;
        }
        numsIdx = newStart;
        nums[numsIdx++] = res;
      } else
        nums[numsIdx++] = next;
    }
    int ans = nums[0];
    for (int i = 1; i < numsIdx; i++)
      ans += nums[i] == -1 ? nums[++i] : -nums[++i];
    return ans;
  }

  private int C2next(char[] cs, int[] idx) {
    int N = cs.length, res = -1;
    for (; idx[0] < N; idx[0]++) {
      if (cs[idx[0]] == ' ')
        continue;
      else if (cs[idx[0]] >= '0' && cs[idx[0]] <= '9') {
        if (res == -1)
          res = 0;
        res = res * 10 + cs[idx[0]] - '0';
      } else if (res != -1)
        break;
      else if (cs[idx[0]] == '+') {
        idx[0]++;
        return -1;
      } else if (cs[idx[0]] == '-') {
        idx[0]++;
        return -2;
      } else if (cs[idx[0]] == '(') {
        idx[0]++;
        return -3;
      } else if (cs[idx[0]] == ')') {
        idx[0]++;
        return -4;
      }
    }
    return res;
  }

  public boolean canJump(int[] nums) {
    if (nums == null || nums.length == 0)
      return false;
    if (nums.length == 1)
      return true;
    int N = nums.length, far = 0;
    for (int i = 0; i < N && i <= far; i++) {
      far = Math.max(far, i + nums[i]);
      if (far >= N - 1)
        return true;
    }
    return false;
  }

  public int mirrorReflection(int p, int q) {
    int gcd = GCD(p, q);
    int pf = p / gcd & 1, qf = q / gcd & 1;
    return pf == 0 ? 2 : qf == 0 ? 0 : 1;
  }

  private int GCD(int p, int q) {
    if (q > p)
      return GCD(q, p);
    if (q == 0)
      return p;
    return GCD(q, p % q);
  }

  class PeekingIterator implements Iterator<Integer> {

    int peek;
    boolean isPeeked;
    Iterator<Integer> ite;

    public PeekingIterator(Iterator<Integer> iterator) {
      isPeeked = false;
      ite = iterator;
    }

    public Integer peek() {
      if (isPeeked)
        return peek;
      peek = ite.next();
      isPeeked = true;
      return peek;
    }

    @Override
    public Integer next() {
      if (isPeeked) {
        isPeeked = false;
        return peek;
      }
      return ite.next();
    }

    @Override
    public boolean hasNext() {
      if (isPeeked || ite.hasNext())
        return true;
      return false;
    }
  }

  public int maximalSquare(char[][] M) {
    if (M == null || M.length == 0 || M[0].length == 0)
      return 0;
    int R = M.length, C = M[0].length, edge = 0;
    int[][] dp = new int[R + 1][C + 1];
    for (int r = 1; r <= R; r++)
      for (int c = 1; c <= C; c++)
        if (M[r - 1][c - 1] == '1') {
          dp[r][c] = Math.min(dp[r - 1][c - 1], Math.min(dp[r - 1][c], dp[r][c - 1])) + 1;
          edge = Math.max(edge, dp[r][c]);
        }
    return edge * edge;
  }

  public int largest1BorderedSquare(int[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0)
      return 0;
    int R = grid.length, C = grid[0].length, edge = 0;
    int[][] RAccum = new int[R + 1][C + 1], CAccum = new int[R + 1][C + 1];
    for (int r = 1; r <= R; r++)
      for (int c = 1; c <= C; c++)
        if (grid[r - 1][c - 1] == 1) {
          RAccum[r][c] = 1 + RAccum[r - 1][c];
          CAccum[r][c] = 1 + CAccum[r][c - 1];
        }
    for (int r = 1; r <= R; r++)
      for (int c = 1; c <= C; c++)
        if (grid[r - 1][c - 1] == 1) {
          for (int tempEdge = Math.min(RAccum[r][c], CAccum[r][c]); tempEdge > edge; tempEdge--)
            if (CAccum[r - tempEdge + 1][c] >= tempEdge && RAccum[r][c - tempEdge + 1] >= tempEdge)
              edge = tempEdge;
        }
    return edge * edge;
  }

  public int orderOfLargestPlusSign1(int N, int[][] mines) {
    int[][] graph = new int[N][N], fromUp = new int[N + 2][N + 2], fromDown = new int[N + 2][N + 2], fromLeft = new int[N + 2][N + 2], fromRight = new int[N + 2][N + 2];
    for (int[] m : mines)
      graph[m[0]][m[1]] = 1;
    for (int r = 1; r <= N; r++)
      for (int c = 1; c <= N; c++)
        if (graph[r - 1][c - 1] == 0) {
          fromUp[r][c] = fromUp[r - 1][c] + 1;
          fromLeft[r][c] = fromLeft[r][c - 1] + 1;
        }
    for (int r = N; r > 0; r--)
      for (int c = N; c > 0; c--)
        if (graph[r - 1][c - 1] == 0) {
          fromDown[r][c] = fromDown[r + 1][c] + 1;
          fromRight[r][c] = fromRight[r][c + 1] + 1;
        }
    int res = 0;
    for (int r = 1; r <= N; r++)
      for (int c = 1; c <= N; c++)
        if (graph[r - 1][c - 1] == 0) {
          int temp = Math.min(Math.min(fromUp[r][c], fromDown[r][c]), Math.min(fromLeft[r][c], fromRight[r][c]));
          res = Math.max(res, temp);
        }
    return res;
  }

  public int orderOfLargestPlusSign(int N, int[][] mines) {
    int[][] M = new int[N][N];
    for (int i = 0; i < N; i++)
      Arrays.fill(M[i], N);
    for (int[] m : mines)
      M[m[0]][m[1]] = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0, k = N - 1, u = 0, d = 0, l = 0, r = 0; j < N; j++, k--) {
        M[i][j] = Math.min(M[i][j], (l = M[i][j] == 0 ? 0 : l + 1));
        M[i][k] = Math.min(M[i][k], (r = M[i][k] == 0 ? 0 : r + 1));
        M[j][i] = Math.min(M[j][i], (u = M[j][i] == 0 ? 0 : u + 1));
        M[k][i] = Math.min(M[k][i], (d = M[k][i] == 0 ? 0 : d + 1));
      }
    int res = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        res = Math.max(res, M[i][j]);
    return res;
  }

  public int totalNQueens(int n) {
    if (n <= 0)
      return 0;
    if (n == 1)
      return 1;
    boolean[] col = new boolean[n], fs = new boolean[n << 1], bs = new boolean[n << 1];
    int[] res = new int[1];
    TNQhelper(0, n, col, fs, bs, res);
    return res[0];
  }

  private void TNQhelper(int r, int N, boolean[] col, boolean[] fs, boolean[] bs, int[] res) {
    if (r == N) {
      res[0]++;
      return;
    }
    for (int c = 0; c < N; c++) {
      if (col[c] || fs[r - c + N] || bs[c + r])
        continue;
      col[c] = fs[r - c + N] = bs[r + c] = true;
      TNQhelper(r + 1, N, col, fs, bs, res);
      col[c] = fs[r - c + N] = bs[r + c] = false;
    }
  }

  public int longestUnivaluePath1(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = new int[1];
    LUPhelper1(root, res);
    return res[0] - 1;
  }

  private int LUPhelper1(TreeNode root, int[] res) {
    if (root == null)
      return 0;
    int left = LUPhelper1(root.left, res), right = LUPhelper1(root.right, res), ans = 1, extend = 0;
    if (root.left != null && root.left.val == root.val) {
      ans += left;
      extend = left;
    }
    if (root.right != null && root.right.val == root.val) {
      ans += right;
      extend = Math.max(extend, right);
    }
    res[0] = Math.max(res[0], ans);
    return 1 + extend;
  }

  public int singleNumber(int[] nums) {
    int appear1 = 0, appear2 = 0;
    for (int n : nums) {
      appear1 = appear1 ^ n & ~appear2;
      appear2 = appear2 ^ n & ~appear1;
    }
    return appear1;
  }

  public int canCompleteCircuit(int[] gas, int[] cost) {
    int total = 0, tank = 0, start = 0, N = gas.length;
    for (int i = 0; i < N; i++) {
      int cur = gas[i] - cost[i];
      tank += cur;
      total += cur;
      if (tank < 0) {
        start = i + 1;
        tank = 0;
      }
    }
    return total < 0 ? -1 : start;
  }

  public int maximalRectangle(char[][] M) {
    if (M == null || M.length == 0 || M[0].length == 0)
      return 0;
    int R = M.length, C = M[0].length, res = 0;
    int[] height = new int[C], left = new int[C], right = new int[C];
    Arrays.fill(right, C);
    for (int r = 0; r < R; r++) {
      int curLeft = 0, curRight = C;
      for (int c = 0; c < C; c++)
        if (M[r][c] == '1')
          height[c]++;
        else
          height[c] = 0;
      for (int c = 0; c < C; c++)
        if (M[r][c] == '1')
          left[c] = Math.max(left[c], curLeft);
        else {
          left[c] = 0;
          curLeft = c + 1;
        }
      for (int c = C - 1; c >= 0; c--)
        if (M[r][c] == '1')
          right[c] = Math.min(right[c], curRight);
        else {
          right[c] = C;
          curRight = c;
        }
      for (int c = 0; c < C; c++)
        res = Math.max(res, (right[c] - left[c]) * height[c]);
    }
    return res;
  }

  public int maxPoints(int[][] P) {
    int N = P.length, res = 0, overlap, curMax;
    if (N <= 2)
      return N;
    Map<String, Integer> line = new HashMap<>();
    for (int i = 0; i < N; i++) {
      line.clear();
      overlap = curMax = 0;
      for (int j = i + 1; j < N; j++) {
        int xDiff = P[i][0] - P[j][0], yDiff = P[i][1] - P[j][1];
        if (xDiff == 0 && yDiff == 0) {
          overlap++;
          continue;
        }
        int gcd = MPgcd(xDiff, yDiff);
        if (gcd != 0) {
          xDiff /= gcd;
          yDiff /= gcd;
        }
        String key = xDiff + " " + yDiff;
        int time = line.getOrDefault(key, 0) + 1;
        curMax = Math.max(curMax, time);
        line.put(key, time);
      }
      res = Math.max(res, curMax + overlap + 1);
    }
    return res;
  }

  private int MPgcd(int p, int q) {
    if (q == 0)
      return p;
    return MPgcd(q, p % q);
  }

  public List<String> findRepeatedDnaSequences1(String s) {
    if (s == null || s.isEmpty() || s.length() <= 10)
      return new ArrayList<>();
    char[] cs = s.toCharArray();
    int N = cs.length;
    Set<String> appear = new HashSet<>(), ans = new HashSet<>();
    String cur = s.substring(0, 10);
    appear.add(cur);
    for (int i = 10; i < N; i++) {
      cur = cur.substring(1) + cs[i];
      if (appear.contains(cur))
        ans.add(cur);
      else
        appear.add(cur);
    }
    return new ArrayList<>(ans);
  }

  public List<String> findRepeatedDnaSequences(String s) {
    List<String> ans = new ArrayList<>();
    if (s == null || s.isEmpty() || s.length() <= 10)
      return ans;
    char[] cs = s.toCharArray();
    int N = cs.length;
    Set<Integer> appear = new HashSet<>(), twice = new HashSet<>();
    char[] encode = new char[26];
    encode['T' - 'A'] = 1;
    encode['G' - 'A'] = 2;
    encode['C' - 'A'] = 3;
    int cur = 0, len = 0, mask = 0xFFFFF;
    for (int i = 0; i < N; i++) {
      cur <<= 2;
      cur |= encode[cs[i] - 'A'];
      len++;
      if (len >= 10) {
        cur &= mask;
        if (!appear.add(cur) && twice.add(cur))
          ans.add(s.substring(i - 9, i + 1));
      }
    }
    return ans;
  }

  public int maxProfitAssignment(int[] D, int[] P, int[] W) {
    if (W.length == 0)
      return 0;
    int jobN = D.length, res = 0;
    int[][] DP = new int[jobN][2];
    for (int i = 0; i < jobN; i++) {
      DP[i][0] = D[i];
      DP[i][1] = P[i];
    }
    Arrays.sort(DP, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0] - b[0];
      }
    });
    for (int i = 1; i < jobN; i++)
      DP[i][1] = Math.max(DP[i][1], DP[i - 1][1]);
    Arrays.sort(W);
    int job = 0, w = 0;
    while (w < W.length)
      if (job >= jobN || DP[job][0] > W[w]) {
        res += job == 0 ? 0 : DP[job - 1][1];
        w++;
      } else
        job++;
    return res;
  }

  public List<List<String>> accountsMerge(List<List<String>> accounts) {
    if (accounts == null || accounts.size() == 0)
      return new ArrayList<>();
    int N = accounts.size(), initialId = 0;
    int[] id = new int[N], weight = new int[N];
    for (int i = 0; i < N; i++) {
      id[i] = i;
      weight[i] = 0;
    }
    Map<String, Integer> emailToId = new HashMap<>();
    for (List<String> account : accounts) {
      for (int i = 1; i < account.size(); i++) {
        Integer previous = emailToId.putIfAbsent(account.get(i), initialId);
        if (previous != null)
          AMunion(id, weight, initialId, previous);
      }
      initialId++;
    }
    List<String>[] emails = new List[N];
    for (Map.Entry<String, Integer> ei : emailToId.entrySet()) {
      int realId = AMfind(id, ei.getValue());
      if (emails[realId] == null)
        emails[realId] = new ArrayList<>();
      emails[realId].add(ei.getKey());
    }
    List<List<String>> ans = new ArrayList<>();
    for (int i = 0; i < N; i++) {
      if (emails[i] == null)
        continue;
      List<String> cur = new ArrayList<>();
      cur.add(accounts.get(i).get(0));
      Collections.sort(emails[i]);
      cur.addAll(emails[i]);
      ans.add(cur);
    }
    return ans;
  }

  private int AMfind(int[] id, int i) {
    int temp = i;
    while (id[i] != i)
      i = id[i];
    while (id[temp] != i) {
      int next = id[temp];
      id[temp] = i;
      temp = next;
    }
    return i;
  }

  private void AMunion(int[] id, int[] weight, int i, int j) {
    int idI = AMfind(id, i), idJ = AMfind(id, j);
    if (idI == idJ)
      return;
    if (weight[idI] <= weight[idJ]) {
      id[idI] = idJ;
      weight[idJ] += weight[idI];
    } else {
      id[idJ] = i;
      weight[idI] += weight[idJ];
    }
  }

  public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null || k == 1)
      return head;
    ListNode fakeHead = new ListNode(0), lastEnd = fakeHead, start = head, cur = head;
    fakeHead.next = head;
    int count = 0;
    while (cur != null) {
      cur = cur.next;
      count++;
      if (count == k - 1 && cur != null) {
        count = 0;
        ListNode next = cur.next;
        reversePiece(start, cur);
        lastEnd.next = cur;
        start.next = next;
        lastEnd = start;
        cur = start = next;
      }
    }
    return fakeHead.next;
  }

  private void reversePiece(ListNode head, ListNode end) {
    ListNode cur = head.next, next = cur.next, last = head;
    while (last != end) {
      cur.next = last;
      last = cur;
      cur = next;
      next = next == null ? null : next.next;
    }
  }

  public int majorityElement(int[] nums) {
    int candidate = 0, count = 0;
    for (int n : nums)
      if (n == candidate)
        count++;
      else {
        count--;
        if (count < 0) {
          candidate = n;
          count = 1;
        }
      }
    return candidate;
  }

  public int minSubArrayLen1(int s, int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int start, end, len = Integer.MAX_VALUE, cur = 0;
    for (start = end = 0; ; )
      if (cur < s) {
        if (end == nums.length)
          break;
        cur += nums[end++];
      } else {
        len = Math.min(len, end - start);
        cur -= nums[start++];
      }
    return len == Integer.MAX_VALUE ? 0 : len;
  }

  public int minSubArrayLen(int s, int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length, len = Integer.MAX_VALUE;
    int[] ps = new int[N + 1];
    for (int i = 0; i < N; i++)
      ps[i + 1] = ps[i] + nums[i];
    for (int i = 0; i < N; i++) {
      int idx = MSAbs(ps, i + 1, s);
      if (idx == -1)
        continue;
      len = Math.min(len, i - idx + 1);
    }
    return len == Integer.MAX_VALUE ? 0 : len;
  }

  private int MSAbs(int[] ps, int till, int s) {
    int start = 0, end = till - 1, mid;
    while (start <= end) {
      mid = (start + end) >> 1;
      if (ps[till] - ps[mid] >= s)
        start = mid + 1;
      else
        end = mid - 1;
    }
    return end < 0 ? -1 : end;
  }

  public int maxCoins1(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length;
    int[] data = new int[N + 2];
    for (int i = 0; i < N; i++)
      data[i + 1] = nums[i];
    data[0] = data[N + 1] = 1;
    int[][] memo = new int[N + 2][N + 2];
    return MChelper1(data, memo, 0, N + 1);
  }

  private int MChelper1(int[] data, int[][] memo, int start, int end) {
    if (start == end - 1)
      return 0;
    if (memo[start][end] != 0)
      return memo[start][end];
    for (int k = start + 1; k < end; k++)
      memo[start][end] = Math.max(memo[start][end], data[k] * data[start] * data[end] + MChelper1(data, memo, start, k) + MChelper1(data, memo, k, end));
    return memo[start][end];
  }

  public int numPrimeArrangements(int n) {
    if (n == 1)
      return 1;
    int divide = (int) Math.pow(10, 9) + 7;
    int primeNum = getPrimeNum(n), notPrime = n - primeNum;
    long res = 1;
    while (primeNum > 1) {
      res *= primeNum--;
      res %= divide;
    }
    while (notPrime > 1) {
      res *= notPrime--;
      res %= divide;
    }
    return (int) res;
  }

  private int getPrimeNum(int n) {
    boolean[] isNotPrime = new boolean[n + 1];
    int count = 0;
    for (int i = 2; i <= n; i++)
      if (!isNotPrime[i]) {
        count++;
        int j = i << 1;
        while (j <= n) {
          isNotPrime[j] = true;
          j += i;
        }
      }
    return count;
  }

  public int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
    if (calories == null || calories.length == 0)
      return 0;
    int res = 0, N = calories.length;
    int[] ps = new int[N + 1];
    for (int i = 0; i < N; i++)
      ps[i + 1] = ps[i] + calories[i];
    for (int i = k; i <= N; i++) {
      int cur = ps[i] - ps[i - k];
      if (cur < lower)
        res--;
      else if (cur > upper)
        res++;
    }
    return res;
  }

  public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
    char[] cs = s.toCharArray();
    int N = cs.length;
    int[][] ps = new int[N + 1][26];
    List<Boolean> ans = new ArrayList<>();
    for (int i = 0; i < N; i++)
      for (int j = 0; j < 26; j++)
        ps[i + 1][j] = cs[i] - 'a' == j ? ps[i][j] + 1 : ps[i][j];
    for (int[] q : queries) {
      int[] count = new int[26];
      for (int i = 0; i < 26; i++)
        count[i] = ps[q[1] + 1][i] - ps[q[0]][i];
      int odd = 0;
      for (int i = 0; i < 26; i++)
        if ((count[i] & 1) == 1)
          odd++;
      odd >>= 1;
      ans.add(odd <= q[2]);
    }
    return ans;
  }

  public int findDuplicate(int[] nums) {
    int fast = nums[nums[0]], slow = nums[0];
    while (fast != slow) {
      fast = nums[nums[fast]];
      slow = nums[slow];
    }
    fast = 0;
    while (fast != slow) {
      fast = nums[fast];
      slow = nums[slow];
    }
    return fast;
  }

  public List<String> findItinerary1(List<List<String>> tickets) {
    LinkedList<String> ans = new LinkedList<>();
    if (tickets == null || tickets.size() == 0)
      return ans;
    Map<String, PriorityQueue<String>> graph = new HashMap<>();
    FIgetGraph(graph, tickets);
    FIgetPath(graph, "JFK", ans);
    return ans;
  }

  private void FIgetPath(Map<String, PriorityQueue<String>> graph, String cur, LinkedList<String> ans) {
    PriorityQueue<String> pq = graph.get(cur);
    while (pq != null && !pq.isEmpty()) {
      FIgetPath(graph, pq.poll(), ans);
    }
    ans.addFirst(cur);
  }

  private void FIgetGraph(Map<String, PriorityQueue<String>> graph, List<List<String>> tickets) {
    for (List<String> t : tickets) {
      PriorityQueue<String> pq = graph.get(t.get(0));
      if (pq == null) {
        pq = new PriorityQueue<>(new Comparator<String>() {
          @Override
          public int compare(String a, String b) {
            return a.compareTo(b);
          }
        });
        graph.put(t.get(0), pq);
      }
      pq.offer(t.get(1));
    }
  }

  public boolean isMatch1(String s, String p) {
    char[] cs = s.toCharArray(), cp = p.toCharArray();
    int sLen = cs.length, pLen = cp.length;
    boolean[][] dp = new boolean[sLen + 1][pLen + 1];
    dp[0][0] = true;
    for (int i = 1; i < pLen; i += 2)
      if (cp[i] == '*')
        dp[0][i + 1] = dp[0][i - 1];
    for (int i = 0; i < sLen; i++)
      for (int j = 0; j < pLen; j++)
        if (cs[i] == cp[j] || cp[j] == '.')
          dp[i + 1][j + 1] = dp[i][j];
        else if (cp[j] == '*')
          if (cs[i] != cp[j - 1] && cp[j - 1] != '.')
            dp[i + 1][j + 1] = dp[i + 1][j - 1];
          else
            dp[i + 1][j + 1] = dp[i + 1][j - 1] || dp[i + 1][j] || dp[i][j + 1];
    return dp[sLen][pLen];
  }

  public int getSum1(int a, int b) {
    int carry = 0, digit = 0, res = 0;
    while (a != 0 || b != 0 || (carry != 0 && digit < 32)) {
      int curA = a & 1, curB = b & 1;
      int curRes = curA ^ curB ^ carry;
      carry = (curA & curB) | ((curA | curB) & carry);
      res |= curRes << digit++;
      a >>>= 1;
      b >>>= 1;
    }
    return res;
  }

  public int getSum(int a, int b) {
    return b == 0 ? a : getSum(a ^ b, (a & b) << 1);
  }

  class StockSpanner {

    int[][] stack; // num, count
    int size;

    public StockSpanner() {
      stack = new int[10000][2];
      size = 0;
    }

    public int next(int price) {
      int res = 1;
      while (size != 0 && stack[size - 1][0] <= price)
        res += stack[--size][1];
      stack[size][0] = price;
      stack[size][1] = res;
      size++;
      return res;
    }
  }

  public int distributeCoins1(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = new int[1];
    DCgetCoins(root, res);
    return res[0];
  }

  private int DCgetCoins(TreeNode root, int[] res) {
    if (root == null)
      return 0;
    int left = DCgetCoins(root.left, res), right = DCgetCoins(root.right, res);
    res[0] += Math.abs(left) + Math.abs(right);
    return root.val + left + right - 1;
  }

  public int cherryPickup(int[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0)
      return 0;
    int R = grid.length, C = grid[0].length;
    int[][][] memo = new int[R][C][R];
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        Arrays.fill(memo[r][c], Integer.MIN_VALUE);
    return Math.max(0, CPhelper(grid, memo, R - 1, C - 1, R - 1));
  }

  private int CPhelper(int[][] grid, int[][][] memo, int x1, int y1, int x2) {
    int y2 = x1 + y1 - x2;
    if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0)
      return -1;
    if (grid[x1][y1] == -1 || grid[x2][y2] == -1)
      return -1;
    if (x1 == 0 && y1 == 0)
      return grid[0][0];
    if (memo[x1][y1][x2] != Integer.MIN_VALUE)
      return memo[x1][y1][x2];
    memo[x1][y1][x2] = Math.max(Math.max(CPhelper(grid, memo, x1 - 1, y1, x2), CPhelper(grid, memo, x1 - 1, y1, x2 - 1)),
            Math.max(CPhelper(grid, memo, x1, y1 - 1, x2), CPhelper(grid, memo, x1, y1 - 1, x2 - 1)));
    if (memo[x1][y1][x2] >= 0) {
      memo[x1][y1][x2] += grid[x1][y1];
      if (x1 != x2 || y1 != y2)
        memo[x1][y1][x2] += grid[x2][y2];
    }
    return memo[x1][y1][x2];
  }

  public boolean backspaceCompare(String S, String T) {
    if ((S == null && T == null) || (S.isEmpty() && T.isEmpty()))
      return true;
    int sIdx = S.length() - 1, tIdx = T.length() - 1, sCount = 0, tCount = 0;
    while (sIdx >= 0 || tIdx >= 0) {
      while (sIdx >= 0 && (S.charAt(sIdx) == '#' || sCount != 0)) {
        if (S.charAt(sIdx) == '#')
          sCount++;
        else
          sCount--;
        sIdx--;
      }
      while (tIdx >= 0 && (T.charAt(tIdx) == '#' || tCount != 0)) {
        if (T.charAt(tIdx) == '#')
          tCount++;
        else
          tCount--;
        tIdx--;
      }
      if (sIdx < 0 && tIdx < 0)
        return true;
      if (sIdx < 0 || tIdx < 0 || S.charAt(sIdx) != T.charAt(tIdx))
        return false;
      sIdx--;
      tIdx--;
    }
    return true;
  }

  public int minDominoRotations(int[] A, int[] B) {
    int N = A.length;
    for (int i = 0, a = 0, b = 0; i < N && (A[i] == A[0] || B[i] == A[0]); i++) {
      if (A[i] != A[0])
        a++;
      if (B[i] != A[0])
        b++;
      if (i == N - 1)
        return Math.min(a, b);
    }
    for (int i = 0, a = 0, b = 0; i < N && (A[i] == B[0] || B[i] == B[0]); i++) {
      if (A[i] != B[0])
        a++;
      if (B[i] != B[0])
        b++;
      if (i == N - 1)
        return Math.min(a, b);
    }
    return -1;
  }

  public int subarrayBitwiseORs1(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    Set<Integer> res = new HashSet<>(), last = new HashSet<>(), cur;
    for (int i : A) {
      cur = new HashSet<>();
      cur.add(i);
      for (int t : last)
        cur.add(t | i);
      res.addAll(last = cur);
    }
    return res.size();
  }

  public int subarrayBitwiseORs(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    Set<Integer> res = new HashSet<>();
    int[] last = new int[32], cur = new int[32];
    int lastLen = 0, curLen;
    for (int a : A) {
      curLen = 0;
      res.add(cur[curLen++] = a);
      for (int i = 0; i < lastLen; i++) {
        int temp = last[i] | a;
        if (temp != a)
          res.add(cur[curLen++] = a = temp);
      }
      int[] t = last;
      last = cur;
      cur = t;
      lastLen = curLen;
    }
    return res.size();
  }

  public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
    List<Integer> ans = new ArrayList<>();
    if (puzzles == null || puzzles.length == 0)
      return ans;
    Map<Integer, Integer> count = new HashMap<>();
    for (String w : words) {
      int bitMap = 0;
      for (int i = 0; i < w.length(); i++)
        bitMap |= 1 << (w.charAt(i) - 'a');
      count.put(bitMap, count.getOrDefault(bitMap, 0) + 1);
    }
    for (String p : puzzles) {
      int bitMap = 0, first = 1 << (p.charAt(0) - 'a'), sub, res = 0;
      for (int i = 0; i < p.length(); i++)
        bitMap |= 1 << (p.charAt(i) - 'a');
      sub = bitMap;
      while (sub != 0) {
        int N;
        if ((sub & first) != 0 && (N = count.getOrDefault(sub, -1)) != -1)
          res += N;
        sub = (sub - 1) & bitMap;
      }
      ans.add(res);
    }
    return ans;
  }

  public int distanceBetweenBusStops(int[] distance, int start, int destination) {
    int n = distance.length, len1 = 0, len2 = 0, idx;
    idx = start;
    while (idx != destination) {
      len1 += distance[idx];
      idx = (idx + 1) % n;
    }
    idx = destination;
    while (idx != start) {
      len2 += distance[idx];
      idx = (idx + 1) % n;
    }
    return Math.min(len1, len2);
  }

  public String dayOfTheWeek(int day, int month, int year) {
    String[] week = new String[]{"Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"};
    int dayCount = 0;
    int[] months = new int[]{31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (int y = 1971; y < year; y++)
      if (isLeapYear(y))
        dayCount += 366;
      else
        dayCount += 365;
    for (int m = 1; m < month; m++)
      if (m == 2)
        dayCount += isLeapYear(year) ? 29 : 28;
      else
        dayCount += months[m - 1];
    dayCount += day;
    String ans = week[dayCount % 7];
    return ans;
  }

  private boolean isLeapYear(int year) {
    if (year % 100 == 0)
      return year % 400 == 0;
    else
      return year % 4 == 0;
  }

  public int maximumSum(int[] arr) {
    int N = arr.length, max = arr[0];
    int[] startFrom = new int[N], endFrom = new int[N];
    endFrom[0] = arr[0];
    for (int i = 1; i < N; i++) {
      endFrom[i] = Math.max(arr[i] + endFrom[i - 1], arr[i]);
      max = Math.max(endFrom[i], max);
    }
    startFrom[N - 1] = arr[N - 1];
    for (int i = N - 2; i >= 0; i--) {
      startFrom[i] = Math.max(arr[i] + startFrom[i + 1], arr[i]);
      max = Math.max(startFrom[i], max);
    }
    for (int i = 1; i < N - 1; i++)
      max = Math.max(max, endFrom[i - 1] + startFrom[i + 1]);
    return max;
  }

  public int findUnsortedSubarray(int[] nums) {
    if (nums == null || nums.length == 0 || nums.length == 1)
      return 0;
    int N = nums.length, start = -1, end = -1, max = nums[0], min = nums[N - 1];
    for (int i = 1; i < N; i++) {
      max = Math.max(max, nums[i]);
      min = Math.min(min, nums[N - 1 - i]);
      if (nums[i] < max)
        end = i;
      if (nums[N - 1 - i] > min)
        start = N - 1 - i;
    }
    return end == -1 ? 0 : end - start + 1;
  }

  public boolean divisorGame(int N) {
    return (N & 1) == 0;
  }

  public boolean isPossible1(int[] nums) {
    if (nums == null || nums.length < 3)
      return false;
    Map<Integer, Integer> freq = new HashMap<>(), need = new HashMap<>();
    for (int n : nums)
      freq.put(n, freq.getOrDefault(n, 0) + 1);
    for (int n : nums) {
      Integer remain, needCount;
      if ((remain = freq.get(n)) == 0)
        continue;
      if ((needCount = need.get(n)) != null) {
        need.remove(n);
        need.put(n + 1, need.getOrDefault(n + 1, 0) + Math.min(remain, needCount));
        remain -= needCount;
      }
      if (remain > 0) {
        Integer next1, next2;
        if ((next1 = freq.get(n + 1)) == null || next1 < remain || (next2 = freq.get(n + 2)) == null || next2 < remain)
          return false;
        freq.put(n + 1, freq.get(n + 1) - remain);
        freq.put(n + 2, freq.get(n + 2) - remain);
        need.put(n + 3, remain);
      }
      freq.put(n, 0);
    }
    return true;
  }

  public boolean isPossible(int[] nums) {
    if (nums == null || nums.length < 3)
      return false;
    int N = nums.length, pre = Integer.MIN_VALUE, p1 = 0, p2 = 0, p3 = 0, cur, c1 = 0, c2 = 0, c3 = 0, cnt = 0;
    for (int i = 0; i < N; pre = cur, p1 = c1, p2 = c2, p3 = c3) {
      for (cur = nums[i], cnt = 0; i < N && nums[i] == cur; i++, cnt++) ;
      if (cur != pre + 1) {
        if (p1 != 0 || p2 != 0)
          return false;
        c1 = cnt;
        c2 = c3 = 0;
      } else {
        if (cnt < p1 + p2)
          return false;
        c1 = Math.max(0, cnt - p1 - p2 - p3);
        c2 = p1;
        c3 = p2 + Math.min(p3, cnt - p1 - p2);
      }
    }
    return p1 == 0 && p2 == 0;
  }

  public boolean isPalindrome(ListNode head) {
    if (head == null || head.next == null)
      return true;
    ListNode fast = head, slow = head;
    while (fast.next != null) {
      fast = fast.next;
      slow = slow.next;
      if (fast.next != null)
        fast = fast.next;
    }
    IPreverse(slow, fast);
    ListNode firstHead = head, lastHead = fast;
    boolean res = true;
    while (true) {
      if (firstHead.val != lastHead.val) {
        res = false;
        break;
      }
      if (lastHead == slow)
        break;
      firstHead = firstHead.next;
      lastHead = lastHead.next;
    }
    IPreverse(fast, slow);
    return res;
  }

  private void IPreverse(ListNode start, ListNode end) {
    if (start == null || start.next == null)
      return;
    ListNode last = start, cur = start.next, next = cur.next;
    while (cur != null) {
      cur.next = last;
      if (cur == end)
        break;
      last = cur;
      cur = next;
      if (next != null)
        next = next.next;
    }
  }

  class ExamRoom {

    PriorityQueue<int[]> intervals;
    int N;

    public ExamRoom(int N) {
      this.N = N;
      intervals = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] a, int[] b) {
          int aLen = a[0] == -1 ? a[1] : a[1] == N ? a[1] - a[0] - 1 : (a[1] - a[0]) >> 1;
          int bLen = b[0] == -1 ? b[1] : b[1] == N ? b[1] - b[0] - 1 : (b[1] - b[0]) >> 1;
          return aLen == bLen ? a[0] - b[0] : bLen - aLen;
        }
      });
      intervals.offer(new int[]{-1, N});
    }

    public int seat() {
      int[] interval = intervals.poll();
      int mid = interval[0] == -1 ? 0 : interval[1] == N ? N - 1 : (interval[1] + interval[0]) >> 1;
      intervals.offer(new int[]{interval[0], mid});
      intervals.offer(new int[]{mid, interval[1]});
      return mid;
    }

    public void leave(int p) {
      int[] front = null, tail = null;
      for (int[] it : intervals) {
        if (it[0] == p)
          tail = it;
        if (it[1] == p)
          front = it;
        if (front != null && tail != null)
          break;
      }
      intervals.remove(front);
      intervals.remove(tail);
      intervals.offer(new int[]{front[0], tail[1]});
    }
  }

  public int hIndex(int[] citations) {
    if (citations == null || citations.length == 0)
      return 0;
    int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
    for (int c : citations) {
      min = Math.min(min, c);
      max = Math.max(max, c);
    }
    int[] range = new int[max - min + 1];
    for (int c : citations)
      range[c - min]++;
    int num = 0;
    for (int h = 0; h < range.length; h++) {
      if (range[h] == 0)
        continue;
      int minC = h + min;
      if (minC >= citations.length - num - range[h])
        return Math.min(minC, citations.length - num);
      num += range[h];
    }
    return 0;
  }

  public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
    List<List<String>> ans = new ArrayList<>();
    Map<String, List<String>> graph = new HashMap<>();
    FLsearchShortest(beginWord, endWord, wordList, graph);
    FLgetPath(beginWord, endWord, graph, new ArrayList<>(), ans);
    return ans;
  }

  private void FLsearchShortest(String begin, String end, List<String> words, Map<String, List<String>> graph) {
    Set<String> head = new HashSet<>(), tail = new HashSet<>(), all = new HashSet<>(words);
    all.add(begin);
    if (!all.contains(end))
      return;
    for (String a : all)
      graph.put(a, new ArrayList<>());
    head.add(begin);
    tail.add(end);
    boolean isOver = false, isForward;
    while (!head.isEmpty() && !tail.isEmpty()) {
      Set<String> sm, bg, next = new HashSet<>();
      if (head.size() <= tail.size()) {
        isForward = true;
        sm = head;
        bg = tail;
      } else {
        isForward = false;
        sm = tail;
        bg = head;
      }
      all.removeAll(sm);
      for (String s : sm) {
        char[] cs = s.toCharArray();
        for (int i = 0; i < cs.length; i++) {
          char prev = cs[i];
          for (char c = 'a'; c <= 'z'; c++) {
            cs[i] = c;
            String temp = new String(cs);
            if (all.contains(temp)) {
              if (isForward)
                graph.get(s).add(temp);
              else
                graph.get(temp).add(s);
              next.add(temp);
            }
            if (bg.contains(temp))
              isOver = true;
          }
          cs[i] = prev;
        }
      }
      if (isForward)
        head = next;
      else
        tail = next;
      if (isOver)
        break;
    }
  }

  private void FLgetPath(String cur, String end, Map<String, List<String>> graph, List<String> path, List<List<String>> ans) {
    if (graph.size() == 0)
      return;
    path.add(cur);
    if (cur.equals(end)) {
      ans.add(new ArrayList<>(path));
      path.remove(path.size() - 1);
      return;
    }
    List<String> next = graph.get(cur);
    if (next == null)
      return;
    for (String n : next)
      FLgetPath(n, end, graph, path, ans);
    path.remove(path.size() - 1);
  }

  public TreeNode bstToGst(TreeNode root) {
    if (root == null)
      return null;
    VTGhelper(root, new int[1]);
    return root;
  }

  private void VTGhelper(TreeNode cur, int[] large) {
    if (cur == null)
      return;
    VTGhelper(cur.right, large);
    large[0] += cur.val;
    cur.val = large[0];
    VTGhelper(cur.left, large);
  }

  public int hIndex1(int[] citations) {
    if (citations == null || citations.length == 0)
      return 0;
    int N = citations.length, curNum = 0;
    int[] CBucket = new int[N + 1];
    for (int c : citations)
      if (c >= N)
        CBucket[N]++;
      else
        CBucket[c]++;
    for (int i = 0; i <= N; i++) {
      if (i >= N - curNum - CBucket[i])
        return Math.min(i, N - curNum);
      curNum += CBucket[i];
    }
    return 0;
  }

  public int[] nextLargerNodes(ListNode head) {
    if (head == null)
      return new int[0];
    int N = 0, stackLen = 0, idx = 0;
    ListNode cur = head;
    while (cur != null) {
      N++;
      cur = cur.next;
    }
    int[] ans = new int[N];
    int[][] stack = new int[N][2]; // for each cell, the two data are [val, loc]
    cur = head;
    while (cur != null) {
      while (stackLen != 0 && stack[stackLen - 1][0] < cur.val)
        ans[stack[--stackLen][1]] = cur.val;
      stack[stackLen++] = new int[]{cur.val, idx};
      idx++;
      cur = cur.next;
    }
    return ans;
  }

  public String convert(String s, int numRows) {
    if (numRows <= 0)
      throw new IllegalArgumentException();
    if (s == null || s.isEmpty() || numRows == 1)
      return s;
    char[] cs = s.toCharArray();
    int N = cs.length;
    StringBuilder sb = new StringBuilder();
    for (int r = 0; r < numRows; r++) {
      int curIdx = r, curCols = 0;
      while (curIdx < N && curIdx <= N) {
        sb.append(cs[curIdx]);
        curIdx = Cnext(numRows, r, curIdx, curCols++);
      }
    }
    return sb.toString();
  }

  private int Cnext(int numRows, int r, int curIdx, int curCols) {
    if (r == 0 || r == numRows - 1)
      return (numRows << 1) - 2 + curIdx;
    if ((curCols & 1) == 0)
      return (numRows << 1) - (r << 1) - 2 + curIdx;
    else
      return (r << 1) + curIdx;
  }

  public boolean isPalindrome(String s) {
    if (s == null || s.length() == 0)
      return true;
    char[] cs = s.toCharArray();
    int N = cs.length, start = 0, end = N - 1;
    for (int i = 0; i < N; i++)
      if (cs[i] >= 'A' && cs[i] <= 'Z')
        cs[i] -= 'A' - 'a';
    while (start < end) {
      while (start < N &&
              !((cs[start] >= '0' && cs[start] <= '9') || (cs[start] >= 'a' && cs[start] <= 'z')))
        start++;
      while (end >= 0 &&
              !((cs[end] >= '0' && cs[end] <= '9') || (cs[end] >= 'a' && cs[end] <= 'z')))
        end--;
      if (start >= end)
        break;
      if (cs[start] != cs[end])
        return false;
      start++;
      end--;
    }
    return true;
  }

  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    if (candidates == null || candidates.length == 0)
      return res;
    CShelper(candidates, 0, target, new ArrayList<>(), res);
    return res;
  }

  private void CShelper(int[] C, int curIdx, int remain, List<Integer> path, List<List<Integer>> res) {
    if (remain < 0)
      return;
    if (remain == 0) {
      res.add(new ArrayList<>(path));
      return;
    }
    int N = C.length;
    for (int i = curIdx; i < N; i++) {
      path.add(C[i]);
      CShelper(C, i, remain - C[i], path, res);
      path.remove(path.size() - 1);
    }
  }

  public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums == null || nums.length == 0)
      return res;
    SShelper(nums, 0, new ArrayList<>(), res);
    return res;
  }

  private void SShelper(int[] nums, int curIdx, List<Integer> path, List<List<Integer>> res) {
    res.add(new ArrayList<>(path));
    for (int i = curIdx; i < nums.length; i++) {
      path.add(nums[i]);
      SShelper(nums, i + 1, path, res);
      path.remove(path.size() - 1);
    }
  }

  public int jump(int[] nums) {
    if (nums.length == 1)
      return 0;
    int N = nums.length, curStart = 0, curEnd = 0, steps = 0;
    while (curEnd < N - 1) {
      steps++;
      int far = Integer.MIN_VALUE;
      for (int i = curStart; i <= curEnd; i++)
        far = Math.max(far, nums[i] + i);
      curEnd = far;
      curStart++;
    }
    return steps;
  }

  public int maxNumberOfBalloons(String text) {
    if (text == null || text.isEmpty())
      return 0;
    int[] count = new int[26], target = new int[26];
    String T = "balloon";
    for (char t : T.toCharArray())
      target[t - 'a']++;
    for (char t : text.toCharArray())
      count[t - 'a']++;
    int ans = Integer.MAX_VALUE;
    for (char t : T.toCharArray())
      ans = Math.min(ans, count[t - 'a'] / target[t - 'a']);
    return ans;
  }

  public String reverseParentheses(String s) {
    if (s == null || s.isEmpty())
      return s;
    StringBuilder sb = new StringBuilder();
    int[] idx = new int[1];
    char[] cs = s.toCharArray();
    int N = cs.length;
    for (; idx[0] < N; )
      if (cs[idx[0]] == '(') {
        idx[0]++;
        sb.append(RPreverse(cs, idx));
      } else
        sb.append(cs[idx[0]++]);
    return sb.toString();
  }

  private String RPreverse(char[] cs, int[] idx) {
    StringBuilder sb = new StringBuilder();
    while (idx[0] < cs.length && cs[idx[0]] != ')')
      if (cs[idx[0]] == '(') {
        idx[0]++;
        sb.append(RPreverse(cs, idx));
      } else
        sb.append(cs[idx[0]++]);
    idx[0]++;
    return sb.reverse().toString();
  }

  public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
    List<List<Integer>> ans = new ArrayList<>();
    if (n == 1 || connections == null || connections.isEmpty())
      return ans;
    List<Integer>[] graph = new List[n];
    for (int i = 0; i < n; i++)
      graph[i] = new ArrayList<>();
    for (List<Integer> con : connections) {
      graph[con.get(0)].add(con.get(1));
      graph[con.get(1)].add(con.get(0));
    }
    int[] idx = new int[n], low = new int[n], time = new int[1];
    Arrays.fill(idx, -1);
    for (int i = 0; i < n; i++)
      if (idx[i] == -1)
        CCtarjan(i, -1, graph, idx, low, ans, time);
    return ans;
  }

  private void CCtarjan(int curIdx, int parent, List<Integer>[] graph, int[] idx, int[] low, List<List<Integer>> res, int[] time) {
    idx[curIdx] = low[curIdx] = time[0]++;
    for (int adj : graph[curIdx])
      if (idx[adj] == -1) {
        CCtarjan(adj, curIdx, graph, idx, low, res, time);
        low[curIdx] = Math.min(low[curIdx], low[adj]);
        if (low[adj] > idx[curIdx])
          res.add(Arrays.asList(adj, curIdx));
      } else if (adj != parent)
        low[curIdx] = Math.min(idx[adj], low[curIdx]);
  }

  public int kConcatenationMaxSum(int[] arr, int k) {
    long ans = 0, singleSum = 0;
    for (int a : arr)
      singleSum += a;
    if (singleSum > 0 && k > 2) {
      ans += singleSum * (k - 2);
      long maxPrefix = 0, maxSuffix = 0, cur = 0;
      for (int i = 0; i < arr.length; i++) {
        cur += arr[i];
        maxSuffix = Math.max(maxSuffix, cur);
      }
      cur = 0;
      for (int i = arr.length - 1; i >= 0; i--) {
        cur += arr[i];
        maxPrefix = Math.max(maxPrefix, cur);
      }
      ans += maxPrefix + maxSuffix;
      ans %= (int) Math.pow(10, 9) + 7;
    }
    int[] combine;
    if (k > 1) {
      combine = new int[arr.length << 1];
      for (int i = 0; i < (arr.length << 1); i++)
        combine[i] = arr[i % arr.length];
    } else
      combine = arr;
    int last = 0, max = 0;
    for (int i = 0; i < combine.length; i++) {
      last = last > 0 ? last + combine[i] : combine[i];
      max = Math.max(last, max);
    }
    return Math.max((int) ans, max);
  }

  public int minAreaRect(int[][] points) {
    if (points == null || points.length < 4)
      return 0;
    Map<Integer, Set<Integer>> XCoordinate = new HashMap<>();
    int min = Integer.MAX_VALUE;
    for (int[] p : points)
      XCoordinate.computeIfAbsent(p[0], a -> new HashSet<>()).add(p[1]);
    for (int[] p1 : points)
      for (int[] p2 : points) {
        if (p1[0] == p2[0] || p1[1] == p2[1])
          continue;
        Set<Integer> x1y, x2y;
        if ((x1y = XCoordinate.get(p1[0])) != null && x1y.contains(p2[1]) &&
                (x2y = XCoordinate.get(p2[0])) != null && x2y.contains(p1[1]))
          min = Math.min(min, Math.abs(p1[0] - p2[0]) * Math.abs(p1[1] - p2[1]));
      }
    return min == Integer.MAX_VALUE ? 0 : min;
  }

  public int numTrees1(int n) {
    if (n <= 0)
      return 0;
    if (n == 1)
      return 1;
    int[] dp = new int[n + 1];
    dp[0] = dp[1] = 1;
    for (int len = 2; len <= n; len++)
      for (int root = 1; root <= len; root++)
        dp[len] += dp[root - 1] * dp[len - root];
    return dp[n];
  }

  public boolean isMatch2(String s, String p) {
    char[] cs = s.toCharArray(), cp = p.toCharArray();
    int cN = cs.length, pN = cp.length;
    boolean[][] dp = new boolean[cN + 1][pN + 1];
    dp[0][0] = true;
    for (int i = 2; i <= pN; i += 2)
      if (cp[i - 1] == '*')
        dp[0][i] = dp[0][i - 2];
    for (int i = 0; i < cN; i++)
      for (int j = 0; j < pN; j++)
        if (cs[i] == cp[j] || cp[j] == '.')
          dp[i + 1][j + 1] = dp[i][j];
        else if (cp[j] == '*')
          if (cp[j - 1] != cs[i] && cp[j - 1] != '.')
            dp[i + 1][j + 1] = dp[i + 1][j - 1];
          else
            dp[i + 1][j + 1] = dp[i + 1][j - 1] || dp[i + 1][j] || dp[i][j + 1];
    return dp[cN][pN];
  }

  public int compress(char[] chars) {
    if (chars == null || chars.length == 0)
      return 0;
    int slow, fast, count = 0, N = chars.length;
    for (slow = fast = 0; fast < N; fast++, count++)
      if (chars[fast] != chars[slow]) {
        if (count != 1)
          slow = CfillCount(count, slow, chars);
        count = 0;
        chars[++slow] = chars[fast];
      }
    if (count != 1)
      slow = CfillCount(count, slow, chars);
    return slow + 1;
  }

  private int CfillCount(int number, int slowIdx, char[] chars) {
    if (number < 10) {
      chars[++slowIdx] = (char) (number + '0');
      return slowIdx;
    }
    int curSlowIdx = CfillCount(number / 10, slowIdx, chars);
    chars[++curSlowIdx] = (char) (number % 10 + '0');
    return curSlowIdx;
  }

  public int maxSubArray1(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int before = 0, max = Integer.MIN_VALUE;
    for (int n : nums) {
      if (before <= 0)
        before = n;
      else
        before += n;
      max = Math.max(max, before);
    }
    return max;
  }

  class StockSpanner1 {

    class Node {
      int val, span;

      public Node(int v, int s) {
        val = v;
        span = s;
      }
    }

    Stack<Node> stack;

    public StockSpanner1() {
      stack = new Stack<>();
    }

    public int next(int price) {
      Node cur = new Node(price, 1);
      while (!stack.isEmpty() && stack.peek().val <= price) {
        cur.span += stack.pop().span;
      }
      stack.push(cur);
      return cur.span;
    }
  }

  public ListNode detectCycle(ListNode head) {
    if (head == null)
      return null;
    ListNode slow = head, fast = head.next;
    while (fast != null && fast != slow) {
      fast = fast.next;
      if (fast == null)
        return null;
      fast = fast.next;
      slow = slow.next;
    }
    if (fast == null)
      return null;
    slow = head;
    fast = fast.next;
    while (fast != slow) {
      fast = fast.next;
      slow = slow.next;
    }
    return slow;
  }

  int MODULO = (int) Math.pow(10, 9) + 7;

  public void thresholdInversion() {
    Scanner scanner = new Scanner(System.in);
    int threshold, N, res = 0;
    threshold = scanner.nextInt();
    N = scanner.nextInt();
    if (N > 0) {
      int[] nums = new int[N], aux = new int[N];
      for (int i = 0; i < N; i++)
        nums[i] = scanner.nextInt();
      res = TImergeSort(nums, aux, 0, N - 1, threshold);
    }
    System.out.println(res);
  }

  private int TImergeSort(int[] nums, int[] aux, int left, int right, int threshold) {
    if (left >= right)
      return 0;
    int mid = (left + right) >> 1, res = 0;
    res = (res + TImergeSort(nums, aux, left, mid, threshold)) % MODULO;
    res = (res + TImergeSort(nums, aux, mid + 1, right, threshold)) % MODULO;
    res = (res + TImerge(nums, aux, left, right, threshold)) % MODULO;
    return res;
  }

  private int TImerge(int[] nums, int[] aux, int left, int right, int threshold) {
    int mid = (left + right) >> 1, res = 0, index = left, L = left, R = mid + 1, TIindex = mid + 1;
    for (int i = left; i <= right; i++)
      aux[i] = nums[i];
    for (; L <= mid; L++) {
      while (TIindex <= right && (long) aux[L] > (long) threshold * (long) aux[TIindex])
        TIindex++;
      res += (TIindex - mid - 1) % MODULO;
      res %= MODULO;
      while (R <= right && aux[R] <= aux[L]) {
        nums[index++] = aux[R];
        R++;
      }
      nums[index++] = aux[L];
    }
    while (index <= right)
      nums[index++] = aux[R++];
    return res;
  }

  public int rob(TreeNode root) {
    if (root == null)
      return 0;
    Map<TreeNode, Integer> memo = new HashMap<>();
    return robHelper(root, memo);
  }

  private int robHelper(TreeNode root, Map<TreeNode, Integer> memo) {
    if (root == null)
      return 0;
    Integer max;
    if ((max = memo.get(root)) != null)
      return max;
    int pick = root.val, notPick = 0;
    if (root.left != null)
      pick += robHelper(root.left.left, memo) + robHelper(root.left.right, memo);
    if (root.right != null)
      pick += robHelper(root.right.left, memo) + robHelper(root.right.right, memo);
    notPick = robHelper(root.left, memo) + robHelper(root.right, memo);
    max = Math.max(pick, notPick);
    memo.put(root, max);
    return max;
  }

  public int minDominoRotations1(int[] A, int[] B) {
    int N = A.length, steps = Integer.MAX_VALUE, Anum = 0, BNum = 0, idx = 0;
    for (; idx < N; idx++) {
      if (A[idx] != A[0] && B[idx] != A[0])
        break;
      if (A[idx] == A[0])
        Anum++;
      if (B[idx] == A[0])
        BNum++;
    }
    if (idx == N)
      steps = Math.min(N - Anum, N - BNum);
    Anum = BNum = idx = 0;
    for (; idx < N; idx++) {
      if (A[idx] != B[0] && B[idx] != B[0])
        break;
      if (A[idx] == B[0])
        Anum++;
      if (B[idx] == B[0])
        BNum++;
    }
    if (idx == N)
      steps = Math.min(steps, Math.min(N - Anum, N - BNum));
    return steps == Integer.MAX_VALUE ? -1 : steps;
  }

  public int integerReplacement(int n) {
    int ans = 0;
    while (n != 1) {
      if ((n & 1) == 0)
        n >>>= 1;
      else if (n == 3 || (n & 2) == 0)
        n--;
      else
        n++;
      ans++;
    }
    return ans;
  }

  public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0)
      return -1;
    int N = nums.length, start = 0, end = N - 1, mid, val;
    while (start <= end) {
      mid = (start + end) >> 1;
      val = nums[mid];
      if (val == target)
        return mid;
      if (val >= nums[start] && val > nums[end]) {
        // large part
        if (target < val && target >= nums[start] && target > nums[end])
          end = mid - 1;
        else
          start = mid + 1;
      } else if (val < nums[start] && val <= nums[end]) {
        if (target > val && target < nums[start] && target <= nums[end])
          start = mid + 1;
        else
          end = mid - 1;
      } else {
        if (target > val)
          start = mid + 1;
        else
          end = mid - 1;
      }
    }
    return -1;
  }

  public void gameOfLife(int[][] board) {
    int R, C;
    if (board == null || (R = board.length) == 0 || (C = board[0].length) == 0)
      return;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++) {
        int aliveNeighbors = GLgetAliveNeighbors(board, r, c);
        if ((board[r][c] & 1) == 1) {
          if (aliveNeighbors == 2 || aliveNeighbors == 3)
            board[r][c] |= 2;
        } else if (aliveNeighbors == 3)
          board[r][c] |= 2;
      }
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        board[r][c] >>= 1;
  }

  private int GLgetAliveNeighbors(int[][] board, int r, int c) {
    int res = 0;
    for (int rDiff = -1; rDiff <= 1; rDiff++)
      for (int cDiff = -1; cDiff <= 1; cDiff++) {
        if (rDiff == 0 && cDiff == 0)
          continue;
        int R = r + rDiff, C = c + cDiff;
        if (R >= 0 && R < board.length && C >= 0 && C < board[0].length && (board[R][C] & 1) == 1)
          res++;
      }
    return res;
  }

  public List<List<Integer>> findSubsequences1(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    if (nums == null || nums.length == 0)
      return ans;
    FSShelper(nums, 0, new ArrayList<>(), ans);
    return ans;
  }

  private void FSShelper(int[] nums, int idx, List<Integer> path, List<List<Integer>> res) {
    if (path.size() > 1)
      res.add(new ArrayList<>(path));
    if (idx == nums.length)
      return;
    Set<Integer> appeared = new HashSet<>();
    for (int i = idx; i < nums.length; i++) {
      if (appeared.contains(nums[i]))
        continue;
      if (path.isEmpty() || nums[i] >= path.get(path.size() - 1)) {
        path.add(nums[i]);
        appeared.add(nums[i]);
        FSShelper(nums, i + 1, path, res);
        path.remove(path.size() - 1);
      }
    }
  }

  public int maxProduct(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int curMax, curMin, max, N = nums.length;
    curMax = curMin = max = nums[0];
    for (int i = 1; i < N; i++) {
      if (nums[i] < 0) {
        int temp = curMax;
        curMax = curMin;
        curMin = temp;
      }
      curMax = Math.max(nums[i], curMax * nums[i]);
      curMin = Math.min(nums[i], curMin * nums[i]);
      max = Math.max(max, curMax);
    }
    return max;
  }

  public int numTrees2(int n) {
    if (n == 0)
      return 0;
    if (n == 1)
      return 1;
    int[] dp = new int[n + 1];
    dp[0] = dp[1] = 1;
    for (int len = 2; len <= n; len++)
      for (int k = 0; k < len; k++)
        dp[len] += dp[k] * dp[len - k - 1];
    return dp[n];
  }

  public List<String> findRepeatedDnaSequences2(String s) {
    List<String> ans = new ArrayList<>();
    if (s == null || s.length() <= 10)
      return ans;
    Set<Integer> first = new HashSet<>(), second = new HashSet<>();
    int[] encode = new int[26];
    encode['T' - 'A'] = 1;
    encode['G' - 'A'] = 2;
    encode['C' - 'A'] = 3;
    char[] cs = s.toCharArray();
    int len = 0, mask = 0xfffff, N = cs.length, cur = 0;
    for (int i = 0; i < N; i++) {
      cur <<= 2;
      cur |= encode[cs[i] - 'A'];
      len++;
      if (len >= 10) {
        int key = cur &= mask;
        if (!first.add(key) && second.add(key))
          ans.add(s.substring(i - 9, i + 1));
      }
    }
    return ans;
  }

  public List<List<String>> groupAnagrams(String[] strs) {
    int[] primes = new int[]{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103};
    List<List<String>> res = new ArrayList<>();
    Map<Integer, List<String>> record = new HashMap<>();
    if (strs == null || strs.length == 0)
      return res;
    for (String s : strs) {
      char[] cs = s.toCharArray();
      int key = 1;
      for (char c : cs)
        key *= primes[c - 'a'];
      List<String> temp = record.get(key);
      if (temp == null) {
        temp = new ArrayList<>();
        record.put(key, temp);
      }
      temp.add(s);
    }
    for (List<String> L : record.values())
      res.add(L);
    return res;
  }

  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    if (l1 == null)
      return l2;
    if (l2 == null)
      return l1;
    ListNode fakeNode = new ListNode(0), cur = fakeNode;
    int carry = 0;
    while (l1 != null || l2 != null || carry != 0) {
      int v1 = l1 == null ? 0 : l1.val;
      int v2 = l2 == null ? 0 : l2.val;
      int curVal = v1 + v2 + carry;
      ListNode next = new ListNode(curVal % 10);
      carry = curVal / 10;
      cur.next = next;
      cur = cur.next;
      if (l1 != null)
        l1 = l1.next;
      if (l2 != null)
        l2 = l2.next;
    }
    return fakeNode.next;
  }

  class LRUCache {

    class BiLink {
      int val, key;
      BiLink post, pre;

      public BiLink(int key, int val) {
        this.key = key;
        this.val = val;
      }
    }

    Map<Integer, BiLink> record;
    BiLink tail, head;
    int capacity;

    public LRUCache(int capacity) {
      record = new HashMap<>();
      head = new BiLink(0, 0);
      tail = new BiLink(0, 0);
      head.post = tail;
      tail.pre = head;
      this.capacity = capacity;
    }

    public int get(int key) {
      BiLink biLink = record.get(key);
      if (biLink == null)
        return -1;
      updateTime(biLink);
      return biLink.val;
    }

    public void put(int key, int value) {
      BiLink biLink = record.get(key);
      if (biLink == null) {
        biLink = new BiLink(key, value);
        insertToTail(biLink);
        record.put(key, biLink);
        capacity--;
        if (capacity < 0) {
          record.remove(head.post.key);
          remove(head.post);
          capacity++;
        }
      } else {
        biLink.val = value;
        updateTime(biLink);
      }
    }

    private void remove(BiLink biLink) {
      biLink.post.pre = biLink.pre;
      biLink.pre.post = biLink.post;
    }

    private void insertToTail(BiLink biLink) {
      tail.pre.post = biLink;
      biLink.pre = tail.pre;
      biLink.post = tail;
      tail.pre = biLink;
    }

    private void updateTime(BiLink biLink) {
      remove(biLink);
      insertToTail(biLink);
    }
  }

  public String longestPalindrome(String s) {
    if (s == null || s.isEmpty())
      return s;
    char[] cs = s.toCharArray();
    int N = cs.length, start = 0, end = 0;
    for (int i = 0; i < N; i++) {
      int left = i, right = i, next;
      while (right + 1 < N && cs[right + 1] == cs[i])
        right++;
      next = right;
      while (left - 1 >= 0 && right + 1 < N && cs[left - 1] == cs[right + 1]) {
        left--;
        right++;
      }
      if (end - start < right - left) {
        end = right;
        start = left;
      }
      i = next;
    }
    return s.substring(start, end + 1);
  }

  public int numIslands1(char[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0)
      return 0;
    int R = grid.length, C = grid[0].length;
    boolean[][] visited = new boolean[R][C];
    int res = 0;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        if (!visited[r][c] && grid[r][c] == '1') {
          res++;
          NILhelper(grid, r, c, visited);
        }
    return res;
  }

  private void NILhelper(char[][] grid, int r, int c, boolean[][] visited) {
    int R = grid.length, C = grid[0].length;
    if (r < 0 || r >= R || c < 0 || c >= C || visited[r][c] || grid[r][c] != '1')
      return;
    visited[r][c] = true;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int[] d : dirs) {
      int nextR = r + d[0], nextC = c + d[1];
      NILhelper(grid, nextR, nextC, visited);
    }
  }

  public int rob1(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = robHepler(root);
    return Math.max(res[0], res[1]);
  }

  private int[] robHepler(TreeNode root) {
    if (root == null)
      return new int[2];
    int[] left = robHepler(root.left), right = robHepler(root.right);
    return new int[]{root.val + left[1] + right[1], Math.max(left[0], left[1]) + Math.max(right[0], right[1])};
  }

  public int quickSelectionWithMOM(int[] nums, int target) {
    if (nums == null || target < 0 || target >= nums.length)
      throw new IllegalArgumentException();
    return quickSelectionWithMOM(nums, target, 0, nums.length - 1);
  }

  private int quickSelectionWithMOM(int[] nums, int target, int left, int right) {
    int pivot = MOM(nums);
    int partitionIdx = partitionMOM(nums, pivot, left, right);
    int res;
    if (partitionIdx > target)
      res = quickSelectionWithMOM(nums, target, left, partitionIdx - 1);
    else if (partitionIdx < target)
      res = quickSelectionWithMOM(nums, target, partitionIdx + 1, right);
    else
      res = nums[partitionIdx];
    return res;
  }

  private int partitionMOM(int[] nums, int pivot, int left, int right) {
    int pivotIdx = -1, R = right + 1, L = left;
    for (int i = left; i <= right; i++)
      if (nums[i] == pivot) {
        pivotIdx = i;
        break;
      }
    exchange(nums, left, pivotIdx);
    while (true) {
      while (L <= right && nums[++L] <= pivot) ;
      while (R >= left && nums[--R] > pivot) ;
      if (L >= R)
        break;
      exchange(nums, L, R);
    }
    exchange(nums, left, R);
    return R;
  }


  public int MOM(int[] nums) {
    int r = 5, N = nums.length, len, idx = 0, mediansLength = (int) Math.ceil((double) N / (double) r);
    int[] bucket = new int[r], medians = new int[mediansLength];
    for (int i = 0; i < N; i += 5) {
      len = 0;
      for (int j = 0; j < r && i + j < N; j++)
        bucket[len++] = nums[i + j];
      Arrays.sort(bucket);
      medians[idx++] = bucket[len >> 1];
    }
    return quickSelectionWithMOM(medians, mediansLength >> 1);
  }

  public class MedianHeap {
    private List<Integer> beforeMedian, afterMedian;
    private int count;
    private Comparator<Integer> inc, dec;

    public MedianHeap(int[] nums) {
      if (nums == null)
        throw new IllegalArgumentException();
      count = nums.length;
      beforeMedian = new ArrayList<>();
      afterMedian = new ArrayList<>();
      beforeMedian.add(0);
      afterMedian.add(0);
      inc = new Comparator<Integer>() {
        @Override
        public int compare(Integer a, Integer b) {
          return a - b;
        }
      };
      dec = new Comparator<Integer>() {
        @Override
        public int compare(Integer a, Integer b) {
          return b - a;
        }
      };
      build(nums);
    }

    public void build(int[] nums) {
      int median = MOM(nums);
      int mediansCount = 0;
      for (int n : nums)
        if (n < median)
          beforeMedian.add(n);
        else if (n > median)
          afterMedian.add(n);
        else
          mediansCount++;
      for (int i = mediansCount >> 1; i >= 0; i--) {
        beforeMedian.add(median);
        afterMedian.add(median);
      }
      if ((mediansCount & 1) == 1)
        beforeMedian.add(median);
      heapify(beforeMedian, dec);
      heapify(afterMedian, inc);
    }

    private void insertHeap(List<Integer> heap, int n, Comparator<Integer> comp) {
      heap.add(n);
      swim(heap, heap.size(), comp);
    }

    public void insert(int val) {
      if ((count & 1) == 1)
        insertHeap(afterMedian, val, inc);
      else
        insertHeap(beforeMedian, val, dec);
      count++;
    }

    public int peek() {
      if (count == 0)
        throw new IllegalArgumentException();
      return beforeMedian.get(1);
    }

    public int extract() {
      if (count == 0)
        throw new IllegalArgumentException();
      int res = beforeMedian.get(1);
      if ((count & 1) == 1)
        reHeap(beforeMedian, dec);
      else {
        int fromAfter = afterMedian.get(1);
        beforeMedian.set(1, fromAfter);
        sink(beforeMedian, 1, dec);
        reHeap(afterMedian, inc);
      }
      count--;
      return 0;
    }

    private void reHeap(List<Integer> heap, Comparator<Integer> comp) {
      heap.set(1, heap.get(heap.size() - 1));
      heap.remove(heap.size() - 1);
      sink(heap, 1, comp);
    }

    private void exchangeList(List<Integer> L, int i, int j) {
      int temp = L.get(i);
      L.set(i, L.get(j));
      L.set(j, temp);
    }

    private void swim(List<Integer> heap, int n, Comparator<Integer> comp) {
      while (n > 1 && comp.compare(heap.get(0), heap.get(n >> 1)) < 0) {
        exchangeList(heap, n, n >> 1);
        n >>= 1;
      }
    }

    private void sink(List<Integer> heap, int n, Comparator<Integer> comp) {
      int N = heap.size() - 1;
      while ((n << 1) < N) {
        int sub = n << 1;
        if (sub < N && comp.compare(heap.get(sub + 1), heap.get(sub)) < 0)
          sub++;
        if (comp.compare(heap.get(sub), heap.get(n)) >= 0)
          break;
        exchangeList(heap, n, sub);
        n = sub;
      }
    }

    private void heapify(List<Integer> heap, Comparator<Integer> comp) {
      int N = heap.size();
      for (int i = N >> 1; i > 0; i--)
        sink(heap, i, comp);
    }
  }

  public TreeNode buildTree(int[] preorder, int[] inorder) {
    if (preorder == null || preorder.length == 0)
      return null;
    Map<Integer, Integer> inIndex = new HashMap<>();
    for (int i = 0; i < inorder.length; i++)
      inIndex.put(inorder[i], i);
    return BThelper(preorder, new int[1], 0, preorder.length - 1, inIndex);
  }

  private TreeNode BThelper(int[] pre, int[] preIdx, int left, int right, Map<Integer, Integer> inIndex) {
    if (left > right)
      return null;
    if (left == right)
      return new TreeNode(pre[preIdx[0]++]);
    TreeNode cur = new TreeNode(pre[preIdx[0]++]);
    int pos = inIndex.get(cur.val);
    cur.left = BThelper(pre, preIdx, left, pos - 1, inIndex);
    cur.right = BThelper(pre, preIdx, pos + 1, right, inIndex);
    return cur;
  }

  public int longestConsecutive2(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int size = 0, N = nums.length;
    Set<Integer> contains = new HashSet<>();
    for (int n : nums)
      contains.add(n);
    for (int n : nums) {
      if (!contains.contains(n))
        continue;
      int sum = 1;
      for (int left = n - 1; contains.remove(left); left--)
        sum++;
      for (int right = n + 1; contains.remove(right); right++)
        sum++;
      size = Math.max(size, sum);
    }
    return size;
  }

  public int minScoreTriangulation(int[] A) {
    if (A == null || A.length < 3)
      return 0;
    int N = A.length;
    int[][] dp = new int[N][N];
    for (int d = 2; d < N; d++)
      for (int start = 0; start < N - d; start++) {
        int end = start + d;
        dp[start][end] = Integer.MAX_VALUE;
        for (int k = start + 1; k < end; k++)
          dp[start][end] = Math.min(dp[start][end], A[start] * A[end] * A[k] + dp[start][k] + dp[k][end]);
      }
    return dp[0][N - 1];
  }

  class TreeInfo {
    TreeNode cur;
    int curIdx;

    public TreeInfo(TreeNode cur, int curIdx) {
      this.cur = cur;
      this.curIdx = curIdx;
    }
  }

  public List<List<Integer>> verticalOrder(TreeNode root) {
    List<List<Integer>> ans = new ArrayList<>();
    if (root == null)
      return ans;
    Map<Integer, List<Integer>> record = new TreeMap<>();
    Queue<TreeInfo> q = new LinkedList<>();
    q.offer(new TreeInfo(root, 0));
    while (!q.isEmpty()) {
      TreeInfo treeInfo = q.poll();
      List<Integer> res = record.get(treeInfo.curIdx);
      if (res == null) {
        res = new ArrayList<>();
        record.put(treeInfo.curIdx, res);
      }
      res.add(treeInfo.cur.val);
      if (treeInfo.cur.left != null)
        q.offer(new TreeInfo(treeInfo.cur.left, treeInfo.curIdx - 1));
      if (treeInfo.cur.right != null)
        q.offer(new TreeInfo(treeInfo.cur.right, treeInfo.curIdx + 1));
    }
    for (List<Integer> L : record.values())
      ans.add(L);
    return ans;
  }

  class P4Edge {
    Long to;
    int time;

    public P4Edge(Long to, int time) {
      this.to = to;
      this.time = time;
    }
  }

  public void P4findPath() {
    //collect data
    Scanner scanner = new Scanner(System.in);
    int destination = scanner.nextInt();
    int edgeNum = scanner.nextInt();
    int maxWaiting = scanner.nextInt();
    Map<Long, List<P4Edge>> graph = new HashMap<>();
    List<int[]>[] auxGraph = new List[200001];
    for (int i = 0; i < edgeNum; i++) {
      int from = scanner.nextInt();
      int to = scanner.nextInt();
      int startTime = scanner.nextInt();
      int endTime = scanner.nextInt();
      if (auxGraph[from] == null)
        auxGraph[from] = new ArrayList<>();
      auxGraph[from].add(new int[]{to, startTime, endTime});
    }
    List<Long> destSet = new ArrayList<>();
    P4buildGraph(graph, auxGraph, maxWaiting, destSet, destination);
    int ans = P4getMinDist(graph, destSet);
    if (ans == Integer.MAX_VALUE)
      System.out.println("NO");
    else
      System.out.println("YES " + ans);
  }

  private int P4getMinDist(Map<Long, List<P4Edge>> graph, List<Long> destSet) {
    Set<Long> visited = new HashSet<>();
    Map<Long, Integer> distTo = new HashMap<>();
    int minAns = Integer.MAX_VALUE;
    for (Long cur : graph.keySet())
      distTo.put(cur, Integer.MAX_VALUE);
    PriorityQueue<P4Edge> pq = new PriorityQueue<>(new Comparator<P4Edge>() {
      @Override
      public int compare(P4Edge a, P4Edge b) {
        return distTo.get(a.to) - distTo.get(b.to);
      }
    });
    Long start = P4encode(1, 0);
    pq.offer(new P4Edge(start, 0));
    distTo.put(start, 0);
    while (!pq.isEmpty()) {
      P4Edge cur = pq.poll();
      if (visited.contains(cur.to))
        continue;
      P4relax(graph, cur.to, visited, distTo, pq);
    }
    for (long d : destSet)
      minAns = Math.min(minAns, distTo.get(d));
    return minAns;
  }

  private void P4relax(Map<Long, List<P4Edge>> graph, Long cur, Set<Long> visited
          , Map<Long, Integer> distTo, PriorityQueue<P4Edge> pq) {
    visited.add(cur);
    List<P4Edge> adj = graph.get(cur);
    for (P4Edge next : adj) {
      int prevDist = distTo.get(cur);
      if (distTo.get(next.to) > prevDist + next.time) {
        distTo.put(next.to, prevDist + next.time);
        pq.offer(next);
      }
    }
  }

  private void P4buildGraph(Map<Long, List<P4Edge>> graph, List<int[]>[] auxGraph, int maxWaiting, List<Long> destSet, int dest) {
    Queue<int[]> q = new LinkedList<>();
    Set<Long> visited = new HashSet<>();
    q.offer(new int[]{1, 0, 0});
    visited.add(P4encode(1, 0));
    while (!q.isEmpty()) {
      int[] cur = q.poll(); // to, startTime,endTime
      int curStation = cur[0], arriveTime = cur[2];
      long key = P4encode(curStation, arriveTime);
      graph.putIfAbsent(key, new ArrayList<>());
      if (curStation == dest)
        destSet.add(key);
      if (auxGraph[curStation] == null)
        continue;
      for (int[] adj : auxGraph[curStation]) {
        int nextStation = adj[0], nextLeftTime = adj[1];
        if (nextLeftTime < arriveTime || nextLeftTime - maxWaiting > arriveTime || visited.contains(P4encode(nextStation, adj[2])))
          continue;
        long key2 = P4encode(curStation, nextLeftTime), key3 = P4encode(nextStation, adj[2]);
        graph.get(key).add(new P4Edge(key2, nextLeftTime - arriveTime));
        graph.computeIfAbsent(key2, a -> new ArrayList<>()).add(new P4Edge(key3, adj[2] - adj[1]));
        visited.add(P4encode(nextStation, adj[2]));
        q.offer(adj);
      }
    }
  }

  private long P4encode(long station, long time) {
    return (station << 31) + time;
  }

  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    if (candidates == null || candidates.length == 0)
      return ans;
    Arrays.sort(candidates);
    CS2helper(candidates, 0, target, new ArrayList<>(), ans);
    return ans;
  }

  private void CS2helper(int[] candidate, int idx, int remain, List<Integer> path, List<List<Integer>> res) {
    if (remain < 0)
      return;
    if (remain == 0) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = idx; i < candidate.length; i++) {
      if (i != idx && candidate[i] == candidate[i - 1])
        continue;
      path.add(candidate[i]);
      CS2helper(candidate, i + 1, remain - candidate[i], path, res);
      path.remove(path.size() - 1);
    }
  }

  public List<List<String>> partition(String s) {
    List<List<String>> ans = new ArrayList<>();
    if (s == null || s.isEmpty())
      return ans;
    int N = s.length();
    boolean[][] isPal = new boolean[N][N];
    Phelper(s.toCharArray(), s, 0, isPal, new ArrayList<>(), ans);
    return ans;
  }

  private void Phelper(char[] cs, String s, int idx, boolean[][] isPal, List<String> path, List<List<String>> res) {
    int N = cs.length;
    if (idx == N) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = idx; i < N; i++) {
      if (cs[i] != cs[idx])
        continue;
      if (i > idx + 1 && !isPal[idx + 1][i - 1])
        continue;
      isPal[idx][i] = true;
      path.add(s.substring(idx, i + 1));
      Phelper(cs, s, i + 1, isPal, path, res);
      path.remove(path.size() - 1);
    }
  }

  public boolean validPalindrome(String s) {
    if (s == null || s.isEmpty() || s.length() == 1)
      return true;
    char[] cs = s.toCharArray();
    int start = 0, end = cs.length - 1;
    while (start < end) {
      if (cs[start] != cs[end])
        return isPal(cs, start + 1, end) || isPal(cs, start, end - 1);
      start++;
      end--;
    }
    return true;
  }

  private boolean isPal(char[] cs, int start, int end) {
    while (start < end) {
      if (cs[start] != cs[end])
        return false;
      start++;
      end--;
    }
    return true;
  }

  public int minMeetingRooms(int[][] intervals) {
    if (intervals == null || intervals.length == 0)
      return 0;
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[1] - b[1];
      }
    });
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0] - b[0];
      }
    });
    int ans = 0;
    for (int[] cur : intervals) {
      if (pq.isEmpty() || pq.peek()[1] > cur[0])
        ans++;
      else
        pq.poll();
      pq.offer(cur);
    }
    return ans;
  }

  public String convertToBase71(int num) {
    if (num == 0)
      return "0";
    boolean isNeg = num < 0;
    if (isNeg)
      num = -num;
    StringBuilder sb = new StringBuilder();
    while (num != 0) {
      sb.append(num % 7);
      num /= 7;
    }
    if (isNeg)
      sb.append('-');
    return sb.reverse().toString();
  }

  public int kthSmallest(int[][] matrix, int k) {
    if (k == 1)
      return matrix[0][0];
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return matrix[a[0]][a[1]] - matrix[b[0]][b[1]];
      }
    });
    int R = matrix.length, C = matrix[0].length;
    boolean[][] visited = new boolean[R][C];
    int[][] dirs = new int[][]{{1, 0}, {0, 1}};
    pq.offer(new int[]{0, 0});
    visited[0][0] = true;
    while (k > 1) {
      int[] temp = pq.poll();
      k--;
      for (int[] d : dirs) {
        int r = temp[0] + d[0], c = temp[1] + d[1];
        if (r < R && c < C && !visited[r][c]) {
          visited[r][c] = true;
          pq.offer(new int[]{r, c});
        }
      }
    }
    int[] ans = pq.peek();
    return matrix[ans[0]][ans[1]];
  }

  public int kthSmallest1(int[][] matrix, int k) {
    if (k == 1)
      return matrix[0][0];
    int N = matrix.length, left = matrix[0][0], right = matrix[N - 1][N - 1];
    while (left < right) {
      int count = 0, pos = N - 1, mid = (left + right) >> 1;
      for (int r = 0; r < N - 1; r++) {
        while (pos >= 0 && matrix[r][pos] > mid)
          pos--;
        count += pos + 1;
      }
      if (count < k)
        left = mid + 1;
      else
        right = mid;
    }
    return left;
  }

  static int check_log_history(List<String> events) {
    if (events == null || events.isEmpty())
      return 0;
    int N = events.size();
    Stack<String> st = new Stack<>();
    Set<String> lock = new HashSet<>();
    for (int i = 0; i < N; i++) {
      String[] info = events.get(i).split(" ");
      String type = info[0], number = info[1];
      if (type.equals("ACQUIRE")) {
        if (!lock.add(number))
          return i + 1;
        st.push(number);
      } else {
        if (st.isEmpty())
          return i + 1;
        String pop = st.pop();
        if (!pop.equals(number))
          return i + 1;
        lock.remove(pop);
      }
    }
    return st.isEmpty() ? 0 : N + 1;
  }

  class P4contract {
    int start, to;
    String company;

    public P4contract(int start, int to, String company) {
      this.start = start;
      this.to = to;
      this.company = company;
    }
  }

  public void contractCorruption() {
    Scanner scanner = new Scanner(System.in);
    int N = scanner.nextInt();
    if (N == 1) {
      System.out.println(0);
      return;
    }
    if (N == 2) {
      System.out.println(1);
      return;
    }
    int edgeNum = scanner.nextInt();
    if (edgeNum == N - 1) {
      System.out.println(N - 1);
      return;
    }
    List<P4contract> Medge = new ArrayList<>(), Dedge = new ArrayList<>();
    for (int i = 0; i < edgeNum; i++) {
      int start = scanner.nextInt();
      int to = scanner.nextInt();
      String company = scanner.next();
      if (company.equals("MAVERICK"))
        Medge.add(new P4contract(start, to, company));
      else
        Dedge.add(new P4contract(start, to, company));
    }
    List<P4contract> minM = getMinMaverick(Medge, Dedge, N);
    int minDiff = getMinDiff(N, Medge, minM);
    System.out.println(minDiff);
  }

  private int getMinDiff(int N, List<P4contract> Medge, List<P4contract> minM) {
    int[] idx = new int[N + 1], weight = new int[N + 1], connectedNum = new int[]{N};
    int MedgeNum = minM.size(), minDiff = Integer.MAX_VALUE;
    for (int i = 1; i <= N; i++) {
      idx[i] = i;
      weight[i] = 1;
    }
    for (P4contract contract : minM)
      P4union(contract.start, contract.to, idx, weight, connectedNum);
    for (P4contract M : Medge) {
      if (P4find(M.start, idx) != P4find(M.to, idx)) {
        P4union(M.start, M.to, idx, weight, connectedNum);
        MedgeNum++;
      }
      int diff = Math.abs(N - 1 - MedgeNum * 2);
      if (diff <= minDiff)
        minDiff = diff;
      else
        break;
    }
    return minDiff;
  }

  private List<P4contract> getMinMaverick(List<P4contract> Medge, List<P4contract> Dedge, int N) {
    int[] idx = new int[N + 1], weight = new int[N + 1], connectedNum = new int[]{N};
    List<P4contract> minM = new ArrayList<>();
    for (int i = 1; i <= N; i++) {
      idx[i] = i;
      weight[i] = 1;
    }
    for (P4contract contract : Dedge) {
      P4union(contract.start, contract.to, idx, weight, connectedNum);
    }
    if (connectedNum[0] != 1)
      for (P4contract contract : Medge)
        if (P4find(contract.start, idx) != P4find(contract.to, idx)) {
          P4union(contract.start, contract.to, idx, weight, connectedNum);
          minM.add(contract);
          if (connectedNum[0] == 1)
            break;
        }
    return minM;
  }

  private int P4find(int curIdx, int[] idx) {
    while (idx[curIdx] != curIdx) {
      idx[curIdx] = idx[idx[curIdx]];
      curIdx = idx[curIdx];
    }
    return curIdx;
  }

  private void P4union(int i, int j, int[] idx, int[] weight, int[] N) {
    int idxI = P4find(i, idx), idxJ = P4find(j, idx);
    if (idxI == idxJ)
      return;
    if (weight[idxI] <= idxJ) {
      idx[idxI] = idxJ;
      weight[idxJ] += weight[idxI];
    } else {
      idx[idxJ] = idxI;
      weight[idxI] += weight[idxJ];
    }
    N[0]--;
  }

  public String convertToBase72(int num) {
    if (num == 0)
      return "0";
    StringBuilder sb = new StringBuilder();
    boolean isNeg = num < 0;
    num = Math.abs(num);
    while (num > 0) {
      sb.append(num % 7);
      num /= 7;
    }
    if (isNeg)
      sb.append("-");
    return sb.reverse().toString();
  }

  public int minAreaRect2(int[][] points) {
    if (points == null || points.length == 0)
      return 0;
    Set<Long> contains = new HashSet<>();
    int N = points.length, min = Integer.MAX_VALUE;
    for (int[] p : points)
      contains.add(MARhencoder(p));
    for (int i = 0; i < N; i++)
      for (int j = i + 1; j < N; j++) {
        int[] p1 = points[i], p2 = points[j];
        long other1 = MARhencoder(new int[]{p1[0], p2[1]});
        long other2 = MARhencoder(new int[]{p2[0], p1[1]});
        if (p1[0] != p2[0] && p1[1] != p2[1] && contains.contains(other1) && contains.contains(other2))
          min = Math.min(min, Math.abs((p1[0] - p2[0]) * (p1[1] - p2[1])));
      }
    return min == Integer.MAX_VALUE ? 0 : min;
  }

  private long MARhencoder(int[] point) {
    return (((long) point[0]) << 30) + point[1];
  }

  public int numDistinctIslands(int[][] grid) {
    if (grid == null || grid.length == 0)
      return 0;
    Set<String> island = new HashSet<>();
    int R = grid.length, C = grid[0].length;
    boolean[][] visited = new boolean[R][C];
    StringBuilder sb = new StringBuilder();
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        if (grid[r][c] == 1 && !visited[r][c]) {
          NDIhelper(grid, r, c, 0, 0, visited, sb);
          island.add(sb.toString());
          sb.setLength(0);
        }
    return island.size();
  }

  private void NDIhelper(int[][] grid, int curR, int curC, int shiftR, int shiftC, boolean[][] visited, StringBuilder sb) {
    int R = grid.length, C = grid[0].length;
    if (curR < 0 || curR >= R || curC < 0 || curC >= C || grid[curR][curC] == 0 || visited[curR][curC])
      return;
    visited[curR][curC] = true;
    sb.append(shiftR);
    sb.append(' ');
    sb.append(shiftC);
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < dirs.length; i++) {
      int[] d = dirs[i];
      int nextR = curR + d[0], nextC = curC + d[1];
      NDIhelper(grid, nextR, nextC, shiftR + d[0], shiftC + d[1], visited, sb);
    }
  }

  public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums == null || nums.length < 3)
      return res;
    Arrays.sort(nums);
    int N = nums.length;
    for (int i = 0; i < N; i++) {
      if (i != 0 && nums[i] == nums[i - 1])
        continue;
      int j = i + 1, k = N - 1;
      while (j < k) {
        int temp = nums[j] + nums[k] + nums[i];
        if (temp > 0)
          k--;
        else if (temp < 0)
          j++;
        else {
          res.add(Arrays.asList(nums[i], nums[j], nums[k]));
          j++;
          k--;
          while (j < k && nums[j] == nums[j - 1])
            j++;
          while (j < k && nums[k] == nums[k + 1])
            k--;
        }
      }
    }
    return res;
  }

  class Solution519 {

    Map<Integer, Integer> find;
    int R, C, spot;
    Random r;

    public Solution519(int n_rows, int n_cols) {
      find = new HashMap<>();
      R = n_rows;
      C = n_cols;
      spot = R * C;
      r = new Random();
    }

    public int[] flip() {
      int idx = r.nextInt(spot--);
      Integer convert = find.getOrDefault(idx, idx);
      find.put(idx, find.getOrDefault(spot, spot));
      find.put(idx, find.getOrDefault(spot, spot));
      return decode(convert);
    }

    public void reset() {
      find.clear();
      spot = R * C;
    }

    private int[] decode(int code) {
      return new int[]{code / C, code % C};
    }
  }

  public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
    return !(rec1[3] <= rec2[1] || rec1[1] >= rec2[3] || rec1[0] >= rec2[2] || rec1[2] <= rec2[0]);
  }

  class DMXTrie {
    int val;
    DMXTrie[] next;

    public DMXTrie(int val) {
      this.val = val;
      next = new DMXTrie[2];
    }
  }

  private void DMXputTrie(DMXTrie root, int num) {
    for (int depth = 31; depth >= 0; depth--) {
      int mask = (1 << depth), idx = (num & mask) > 0 ? 1 : 0;
      if (root.next[idx] == null)
        root.next[idx] = new DMXTrie(idx);
      root = root.next[idx];
    }
  }

  public int findMaximumXOR(int[] nums) {
    if (nums == null || nums.length < 2)
      return 0;
    DMXTrie root = new DMXTrie(0);
    for (int n : nums)
      DMXputTrie(root, n);
    return DMXgetMaxXOR(root, root, 0);
  }

  private int DMXgetMaxXOR(DMXTrie r1, DMXTrie r2, int lastRes) {
    int res = (lastRes << 1) | (r1.val ^ r2.val), max = Integer.MIN_VALUE;
    if (r1.next[1] != null && r2.next[0] != null)
      max = Math.max(max, DMXgetMaxXOR(r1.next[1], r2.next[0], res));
    if (r1.next[0] != null && r2.next[1] != null)
      max = Math.max(max, DMXgetMaxXOR(r1.next[0], r2.next[1], res));
    if (max != Integer.MIN_VALUE)
      return max;
    if (r1.next[1] != null && r2.next[1] != null)
      max = Math.max(max, DMXgetMaxXOR(r1.next[1], r2.next[1], res));
    if (r1.next[0] != null && r2.next[0] != null)
      max = Math.max(max, DMXgetMaxXOR(r1.next[0], r2.next[0], res));
    return max == Integer.MIN_VALUE ? res : max;
  }

  public char[][] translateGardenComposition() {
    char[][] res = getInputGarden();
    outputGarden(res);
    return res;
  }

  private char[][] getInputGarden() {
    Scanner scanner = new Scanner(System.in);
    String[] weightAndHeight = scanner.nextLine().split(",");
    int weight = Integer.parseInt(weightAndHeight[0]);
    int height = Integer.parseInt(weightAndHeight[1]);
    if (weight <= 0 || height <= 0)
      return null;
    char[][] res = new char[height][weight];
    for (int i = 0; i < height; i++)
      Arrays.fill(res[i], 'B');
    while (scanner.hasNextLine()) {
      String curLine = scanner.nextLine();
      if (curLine.equals(""))
        break;
      String[] temp = curLine.split(",");
      int w = Integer.parseInt(temp[1]);
      int h = Integer.parseInt(temp[2]);
      res[h][w] = temp[0].charAt(0);
    }
    return res;
  }

  public void findBestPlace() {
    char[][] res = getInputGarden();
    int height = res.length, weight = res[0].length, max = Integer.MIN_VALUE;
    int[][] left = new int[height + 2][weight + 2],
            right = new int[height + 2][weight + 2],
            top = new int[height + 2][weight + 2],
            bottom = new int[height + 2][weight + 2],
            ans = new int[height][weight];
    for (int h = 0; h < height; h++)
      for (int w = 0; w < weight; w++) {
        left[h + 1][w + 1] = res[h][w] == 'W' ? 0 : res[h][w] == 'F' ? left[h + 1][w] + 1 : left[h + 1][w];
        right[h + 1][weight - w] = res[h][weight - w - 1] == 'W' ? 0 : res[h][weight - w - 1] == 'F' ? right[h + 1][weight - w + 1] + 1 : right[h + 1][weight - w + 1];
        top[h + 1][w + 1] = res[h][w] == 'W' ? 0 : res[h][w] == 'F' ? top[h][w + 1] + 1 : top[h][w + 1];
        bottom[height - h][w + 1] = res[height - h - 1][w] == 'W' ? 0 : res[height - h - 1][w] == 'F' ? bottom[height - h + 1][w + 1] + 1 : bottom[height - h + 1][w + 1];
      }
    for (int i = 0; i < height; i++)
      for (int j = 0; j < weight; j++)
        if (res[i][j] == 'B') {
          ans[i][j] = left[i + 1][j + 1] + right[i + 1][j + 1] + top[i + 1][j + 1] + bottom[i + 1][j + 1];
          max = Math.max(max, ans[i][j]);
        }
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < weight; j++)
        if (max == 0) {
          sb.append(res[i][j] == 'B' ? '*' : res[i][j]);
        } else {
          sb.append(ans[i][j] == max ? '*' : res[i][j]);
        }
      System.out.println(sb.toString());
      sb.setLength(0);
    }
  }

  class GardenAction {
    String action;
    char target;
    int x, y;

    public GardenAction(String action, char target, int x, int y) {
      this.action = action;
      this.target = target;
      this.x = x;
      this.y = y;
    }
  }

  public void rearrangeGarden() {
    Scanner scanner = new Scanner(System.in);
    String[] weightAndHeight = scanner.nextLine().split(",");
    int weight = Integer.parseInt(weightAndHeight[0]);
    int height = Integer.parseInt(weightAndHeight[1]);
    char[][] res = new char[height][weight];
    for (int i = 0; i < height; i++)
      Arrays.fill(res[i], 'B');
    boolean isComposition = true;
    List<GardenAction> actions = new ArrayList<>();
    while (scanner.hasNextLine()) {
      String curLine = scanner.nextLine();
      if (curLine.equals(""))
        break;
      if (curLine.charAt(1) != ',')
        isComposition = false;
      if (isComposition) {
        String[] temp = curLine.split(",");
        int w = Integer.parseInt(temp[1]);
        int h = Integer.parseInt(temp[2]);
        res[h][w] = temp[0].charAt(0);
      } else {
        String[] temp = curLine.split(",");
        actions.add(new GardenAction(temp[0], temp[1].charAt(0), Integer.parseInt(temp[2]), Integer.parseInt(temp[3])));
      }
    }
    Stack<Character> stack = new Stack<>();
    for (int i = 0; i < actions.size(); i++) {
      GardenAction action = actions.get(i);
      int w = action.x, h = action.y;
      if (action.action.equals("Pick up")) {
        if (res[h][w] != action.target) {
          rearrangeErrorOutput(res);
          return;
        }
        stack.push(action.target);
        res[h][w] = 'B';
      } else {
        if (res[h][w] != 'B' || stack.isEmpty() || !stack.peek().equals(action.target)) {
          rearrangeErrorOutput(res);
          return;
        }
        stack.pop();
        res[h][w] = action.target;
      }
    }
    if (!stack.isEmpty())
      rearrangeErrorOutput(res);
    else {
      System.out.println(true);
      outputGarden(res);
    }
  }

  private void rearrangeErrorOutput(char[][] res) {
    System.out.println(false);
    outputGarden(res);
  }

  private void outputGarden(char[][] res) {
    int height = res.length, weight = res[0].length;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < weight; j++)
        sb.append(res[i][j]);
      System.out.println(sb.toString());
      sb.setLength(0);
    }
  }

  public void ivySpread() {
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    Queue<int[]> q = new LinkedList<>();
    int step = 0;
    // collect garden composition
    Scanner scanner = new Scanner(System.in);
    String[] weightAndHeight = scanner.nextLine().split(",");
    int weight = Integer.parseInt(weightAndHeight[0]);
    int height = Integer.parseInt(weightAndHeight[1]);
    char[][] res = new char[height][weight];
    boolean[][] visited = new boolean[height][weight];
    for (int i = 0; i < height; i++)
      Arrays.fill(res[i], 'B');
    while (scanner.hasNextLine()) {
      String curLine = scanner.nextLine();
      if (curLine.equals(""))
        break;
      String[] temp = curLine.split(",");
      int w = Integer.parseInt(temp[1]);
      int h = Integer.parseInt(temp[2]);
      res[h][w] = temp[0].charAt(0);
      if (res[h][w] == 'I') {
        visited[h][w] = true;
        q.offer(new int[]{h, w});
      }
    }
    // ivy grow
    while (!q.isEmpty()) {
      int size = q.size();
      for (int i = 0; i < size; i++) {
        int[] cur = q.poll();
        for (int[] d : dirs) {
          int r = cur[0] + d[0], c = cur[1] + d[1];
          if (r >= 0 && r < height && c >= 0 && c < weight && res[r][c] != 'W' && !visited[r][c]) {
            if (res[r][c] == 'F') {
              System.out.println(step);
              outputGarden(res);
              return;
            } else {
              visited[r][c] = true;
              q.offer(new int[]{r, c});
            }
          }
        }
      }
      step++;
      for (int[] newIvl : q)
        res[newIvl[0]][newIvl[1]] = 'I';
    }
  }

  private void evenFib() {
    Scanner in = new Scanner(System.in);
    int t = in.nextInt();
    List<Long> N = new ArrayList<>();
    long max = 0;
    for (int a0 = 0; a0 < t; a0++) {
      long n = in.nextLong();
      N.add(n);
      max = Math.max(max, n);
    }
    Map<Long, Long> memo = new HashMap<>();
    long last1 = 1, last2 = 2, cur = 0, lastEvenSum = 2;
    memo.put((long) 2, (long) 2);
    memo.put((long) 0, (long) 0);
    while (last2 <= max) {
      cur = last1 + last2;
      if ((cur & 1) == 0) {
        lastEvenSum += cur;
        memo.put(cur, lastEvenSum);
      }
      last1 = last2;
      last2 = cur;
    }
    List<Long> posList = new ArrayList<>(memo.keySet());
    Collections.sort(posList);
    for (long n : N) {
      long maxEvenFib = EFfindMemoPos(posList, n);
      System.out.println(memo.get(maxEvenFib));
    }
  }

  private long EFfindMemoPos(List<Long> posList, long n) {
    int start = 0, end = posList.size() - 1, mid;
    while (start <= end) {
      mid = (start + end) >> 1;
      long curVal = posList.get(mid);
      if (curVal > n)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return posList.get(end);
  }

  private void maxPathSum() {
    Scanner scanner = new Scanner(System.in);
    int T = scanner.nextInt();
    int[][] res;
    List<Integer> ans = new ArrayList<>(T);
    for (int i = 0; i < T; i++) {
      int N = scanner.nextInt();
      res = new int[N][N];
      for (int h = 0; h < N; h++)
        for (int idx = 0; idx <= h; idx++)
          res[h][idx] = scanner.nextInt();
      int maxPathSum = getMaxPathSum(res);
      ans.add(maxPathSum);
    }
    for (int a : ans)
      System.out.println(a);
  }

  private int getMaxPathSum(int[][] res) {
    int N = res.length;
    int[][] dp = new int[N][N];
    for (int i = 0; i < N; i++)
      dp[N - 1][i] = res[N - 1][i];
    for (int h = N - 2; h >= 0; h--)
      for (int idx = 0; idx <= h; idx++)
        dp[h][idx] = Math.max(dp[h + 1][idx], dp[h + 1][idx + 1]) + res[h][idx];
    return dp[0][0];
  }

  public static long getWays(int n, List<Long> c) {
    // Write your code here
    int R = c.size();
    long[][] dp = new long[R][n + 1];
    Collections.sort(c);
    int minCoin = c.get(0).intValue();
    for (int i = minCoin; i <= n; i++)
      for (int j = R - 1; j >= 0; j--) {
        int cur = c.get(j).intValue();
        if (i < cur)
          continue;
        else if (i == cur)
          dp[j][i] = 1;
        else {
          dp[j][i] += j == R - 1 ? 0 : dp[j + 1][i];
          dp[j][i] += i - cur >= 0 ? dp[j][i - cur] : 0;
        }
      }
    return dp[0][n];
  }

  public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0)
      return null;
    if (lists.length == 1)
      return lists[0];
    return MKLmergeSort(lists, 0, lists.length - 1);
  }

  private ListNode MKLmergeSort(ListNode[] lists, int start, int end) {
    if (start == end)
      return lists[start];
    int mid = (start + end) >> 1;
    ListNode left = MKLmergeSort(lists, start, mid), right = MKLmergeSort(lists, mid + 1, end);
    return MKLmerge(left, right);
  }

  private ListNode MKLmerge(ListNode l1, ListNode l2) {
    ListNode fakeHead = new ListNode(0), head = fakeHead;
    while (l1 != null || l2 != null) {
      if (l1 == null) {
        head.next = l2;
        l2 = l2.next;
      } else if (l2 == null) {
        head.next = l1;
        l1 = l1.next;
      } else if (l1.val <= l2.val) {
        head.next = l1;
        l1 = l1.next;
      } else {
        head.next = l2;
        l2 = l2.next;
      }
      head = head.next;
    }
    return fakeHead.next;
  }

  public int minDistance(String word1, String word2) {
    if (word1 == null || word2 == null)
      return 0;
    if (word1.isEmpty())
      return word2.length();
    if (word2.isEmpty())
      return word1.length();
    char[] cs1 = word1.toCharArray(), cs2 = word2.toCharArray();
    int N1 = cs1.length, N2 = cs2.length;
    int[][] dp = new int[N1 + 1][N2 + 1];
    for (int i = 1; i <= N1; i++)
      dp[i][0] = i;
    for (int j = 1; j <= N2; j++)
      dp[0][j] = j;
    for (int i = 1; i <= N1; i++)
      for (int j = 1; j <= N2; j++)
        if (cs1[i - 1] == cs2[j - 1])
          dp[i][j] = dp[i - 1][j - 1];
        else
          dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
    return dp[N1][N2];
  }

  public int maxArea(int[] height) {
    int N = height.length, start = 0, end = N - 1, max = Integer.MIN_VALUE;
    while (start < end) {
      max = Math.max(max, (end - start) * Math.min(height[start], height[end]));
      if (height[start] <= height[end])
        start++;
      else
        end--;
    }
    return max;
  }

  class MyStack {

    Queue<Integer> q;
    int last;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
      q = new LinkedList<>();
      last = 0;
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
      last = x;
      q.offer(x);
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
      int N = q.size();
      for (int i = 0; i < N - 1; i++) {
        int next = q.poll();
        if (i == N - 2)
          last = next;
        q.offer(next);
      }
      return q.poll();
    }

    /**
     * Get the top element.
     */
    public int top() {
      return last;
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
      return q.isEmpty();
    }
  }

  public int splitArray(int[] nums, int m) {
    int N = nums.length, max = Integer.MIN_VALUE, sum = 0;
    for (int n : nums) {
      max = Math.max(max, n);
      sum += n;
    }
    if (m == 1)
      return sum;
    int start = max, end = sum;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (SAcanSplit(mid, nums, m))
        end = mid - 1;
      else
        start = mid + 1;
    }
    return start;
  }

  private boolean SAcanSplit(int target, int[] nums, int m) {
    int sum = 0, group = 1;
    for (int n : nums) {
      sum += n;
      if (sum > target) {
        sum = n;
        group++;
        if (group > m)
          return false;
      }
    }
    return true;
  }

  public int divisorSubstrings(int n, int k) {
    int initial = 0, temp = n, ans = 0;
    for (int i = 0; i < k; i++) {
      if (n == 0)
        return 0;
      initial += (n % 10) * Math.pow(10, i);
      n /= 10;
    }
    ans += temp % initial == 0 ? 1 : 0;
    while (n != 0) {
      initial = initial / 10 + (n % 10) * (int) Math.pow(10, k - 1);
      ans += temp % initial == 0 ? 1 : 0;
      n /= 10;
    }
    return ans;
  }


  class MedianFinder2 {

    PriorityQueue<Integer> smaller, bigger;
    int N;

    /**
     * initialize your data structure here.
     */
    public MedianFinder2() {
      smaller = new PriorityQueue<>(Collections.reverseOrder());
      bigger = new PriorityQueue<>();
      N = 0;
    }

    public void addNum(int num) {
      if (N == 0)
        smaller.offer(num);
      else {
        if (smaller.peek() >= num) {
          smaller.offer(num);
          if (smaller.size() > bigger.size() + 1)
            bigger.offer(smaller.poll());
        } else {
          bigger.offer(num);
          if (bigger.size() > smaller.size())
            smaller.offer(bigger.poll());
        }
      }
      N++;
    }

    public double findMedian() {
      if (N == 0)
        throw new IllegalArgumentException();
      if ((N & 1) == 1)
        return smaller.peek();
      else
        return ((double) smaller.peek() + (double) bigger.peek()) / 2;
    }
  }

  public int counter21(int[] nums) {
    if (nums == null || nums.length < 3)
      return 0;
    int N = nums.length, ans = 0;
    for (int i = 0; i <= N - 3; i++)
      if (is21(nums, i))
        ans++;
    return ans;
  }

  private boolean is21(int[] nums, int i) {
    int a = nums[i], b = nums[i + 1], c = nums[i + 2];
    if (a == b && b == c)
      return false;
    if (a == b || a == c || b == c)
      return true;
    return false;
  }

  public String sumOfString(String a, String b) {
    if (a == null || a.isEmpty())
      return b;
    if (b == null || b.isEmpty())
      return a;
    char[] ca = a.toCharArray(), cb = b.toCharArray();
    StringBuilder sb = new StringBuilder();
    int aLen = ca.length, bLen = cb.length;
    for (int i = 0, j = 0; i < aLen || j < bLen; i++, j++) {
      if (i >= aLen)
        sb.insert(0, cb[j]);
      else if (j >= bLen)
        sb.insert(0, ca[i]);
      else
        sb.insert(0, (ca[i] - '0') + (cb[j] - '0'));
    }
    return sb.toString();
  }

  public boolean canDivideToTwo(int[] nums) {
    if (nums == null || nums.length == 0)
      return true;
    if ((nums.length & 1) == 1)
      return false;
    Map<Integer, Integer> count = new HashMap<>();
    for (int n : nums) {
      int num = count.getOrDefault(n, 0) + 1;
      count.put(n, num);
      if (num > 2)
        return false;
    }
    return true;
  }

  class CoolFeature {

    Map<Integer, Integer> a;
    int[] b;

    public CoolFeature(int[] A, int[] B) {
      this.a = new HashMap<>();
      for (int aNum : A)
        a.put(aNum, a.getOrDefault(aNum, 0) + 1);
      b = B;
    }

    public Integer query(int[] query) {
      if (query.length != 2 || query.length != 3)
        return null;
      if (query.length == 2) {
        int ans = 0;
        for (int bNum : b)
          ans += a.getOrDefault(query[1] - bNum, 0);
        return ans;
      } else {
        b[query[1] - 1] = query[query[2]];
        return null;
      }
    }
  }

  public List<Integer> mostFrequentDigits(int[] nums) {
    List<Integer> ans = new ArrayList<>();
    if (nums == null || nums.length == 0)
      return ans;
    int[] res = new int[10];
    for (int n : nums)
      MFDgetDigit(n, res);
    int maxCount = 0;
    for (int i : res)
      maxCount = Math.max(maxCount, i);
    for (int i = 0; i < res.length; i++)
      if (res[i] == maxCount)
        ans.add(i);
    return ans;
  }

  private void MFDgetDigit(int num, int[] res) {
    while (num != 0) {
      res[num % 10]++;
      num /= 10;
    }
  }

  public void rotateOverDiagonals(int[][] M, int k) {
    if (M == null || M.length == 0 || M[0].length == 0)
      return;
    int N = M.length;
    k %= 4;
    for (int time = 0; time < k; time++)
      for (int line = 0; line < (N >> 1); line++)
        for (int start = line + 1; start < N - line - 1; start++) {
          int temp = M[line][start];
          M[line][start] = M[N - start - 1][line];
          M[N - start - 1][line] = M[N - line - 1][N - start - 1];
          M[N - line - 1][N - start - 1] = M[start][N - line - 1];
          M[start][N - line - 1] = temp;
        }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        System.out.print(M[i][j] + " ");
      }
      System.out.println();
    }
  }

  public int matrixQueries(int[] nums, int[][] queries) {
    if (nums == null || queries == null || nums.length == 0 || queries.length == 0)
      return 0;
    int N = nums.length;
    Map<Integer, int[]> preSum = new HashMap<>();
    for (int i = 0; i < N; i++) {
      int[] ps = preSum.get(nums[i]);
      if (ps == null) {
        ps = new int[N + 1];
        preSum.put(nums[i], ps);
      }
      ps[i + 1] = 1;
    }
    for (int[] ps : preSum.values()) {
      for (int i = 1; i <= N; i++)
        ps[i] += ps[i - 1];
    }
    int ans = 0;
    for (int[] q : queries) {
      int[] ps = preSum.get(q[2]);
      if (ps == null)
        continue;
      ans += ps[q[1]] - ps[q[0] - 1];
    }
    return ans;
  }

  public void sortDiagonal(int[][] M) {
    if (M == null || M.length == 0 || M[0].length == 0)
      return;
    int R = M.length, C = M[0].length;
    for (int c = 0; c < C; c++)
      SDsort(0, c, M);
    for (int r = 1; r < R; r++)
      SDsort(r, 0, M);
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        System.out.println(M[r][c]);
  }

  private void SDsort(int startR, int startC, int[][] M) {
    int R = M.length, C = M[0].length, tempR = startR, tempC = startC;
    List<Integer> nums = new ArrayList<>();
    while (startR < R && startC < C) {
      nums.add(M[startR++][startC++]);
    }
    SDquickSort(nums, 0, nums.size() - 1);
    int idx = 0;
    while (tempR < R && tempC < C) {
      M[tempR][tempC] = nums.get(idx++);
      tempR++;
      tempC++;
    }
  }

  private void SDquickSort(List<Integer> nums, int start, int end) {
    if (start >= end)
      return;
    int partitionIdx = SDpartition(nums, start, end);
    SDquickSort(nums, start, partitionIdx - 1);
    SDquickSort(nums, partitionIdx + 1, end);
  }

  private int SDpartition(List<Integer> nums, int start, int end) {
    int partition = nums.get(start);
    int i = start + 1, j = end;
    while (true) {
      while (i <= end && nums.get(i) <= partition)
        i++;
      while (j > start && nums.get(j) > partition)
        j--;
      if (i >= j)
        break;
      exchangeList(nums, i, j);
    }
    exchangeList(nums, start, j);
    return j;
  }

  private void exchangeList(List<Integer> nums, int i, int j) {
    int temp = nums.get(i);
    nums.set(i, nums.get(j));
    nums.set(j, temp);
  }

  public boolean compareStringFreq(String a, String b) {
    if (a.length() != b.length())
      return false;
    int N = a.length();
    char[] ca = a.toCharArray(), cb = b.toCharArray();
    int[] countA = new int[26], countB = new int[26];
    for (char c : ca)
      countA[c - 'a']++;
    for (char c : cb)
      countB[c - 'a']++;
    int res = 0, diffCount = 0;
    for (int i = 0; i < 26; i++)
      if (countA[i] != countB[i]) {
        diffCount++;
        res += countA[i] - countB[i];
      }
    if (diffCount == 0 || (diffCount == 2 && res == 0))
      return true;
    return false;
  }

  public int maxSizeRibbon(int[] ribbon, int k) {
    if (k <= 0 || ribbon == null || ribbon.length == 0)
      throw new IllegalArgumentException();
    int start = 1, end = 0;
    for (int r : ribbon)
      end = Math.max(end, r);
    while (start <= end) {
      int mid = (start + end) >> 1;
      int parts = MSRfindParts(ribbon, mid);
      if (parts >= k)
        start = mid + 1;
      else
        end = mid - 1;
    }
    return end;
  }

  private int MSRfindParts(int[] ribbon, int len) {
    int ans = 0;
    for (int r : ribbon)
      if (r >= len)
        ans += r / len;
    return ans;
  }


  public boolean isPrefix(String[] a, String[] b) {
    Set<String> parts = new HashSet<>();
    for (String t : a)
      parts.add(t);
    for (String t : b)
      if (!canComposition(t, parts))
        return false;
    return true;
  }

  private boolean canComposition(String cur, Set<String> parts) {
    if (cur == null || cur.length() == 0)
      return true;
    if (parts.contains(cur))
      return true;
    int N = cur.length();
    for (int i = 1; i <= N; i++) {
      String temp = cur.substring(0, i);
      if (parts.contains(temp)) {
        parts.add(cur);
        return true;
      }
    }
    return false;
  }

  public int longestEqualSubarray(int[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    int oneMinusZero = 0, max = 0;
    Map<Integer, Integer> appeared = new HashMap<>();
    appeared.put(0, 0);
    for (int i = 0; i < nums.length; i++) {
      oneMinusZero += nums[i] == 1 ? 1 : -1;
      Integer earliest = appeared.get(oneMinusZero);
      if (earliest == null)
        appeared.put(oneMinusZero, i + 1);
      else
        max = Math.max(i + 1 - earliest, max);
    }
    return max;
  }

  public void rameWindow(int n) {
    if (n <= 0)
      return;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++)
      sb.append('*');
    String edge = sb.toString();
    sb.setLength(0);
    sb.append('*');
    for (int i = 0; i < n - 2; i++)
      sb.append(' ');
    sb.append('*');
    String mid = sb.toString();
    System.out.println(edge);
    for (int i = 0; i < n - 2; i++)
      System.out.println(mid);
    System.out.println(edge);
  }

  public int removeEquals(String a, String b) {
    return 0;
  }

  public int maxArithmeticLength(int[] a, int[] b) {
    if (a == null || a.length < 2)
      return 0;
    Set<Integer> appeared = new HashSet<>();
    for (int bVal : b)
      appeared.add(bVal);
    int gcd = a[1] - a[0], N = a.length;
    for (int i = 1; i < N - 1; i++)
      gcd = MALgcd(gcd, a[i + 1] - a[i]);
    int ans = 0;
    for (int i = gcd; i >= 1; i--) {
      if (gcd % i != 0)
        continue;
      int len = MALextend(a, appeared, gcd / i);
      ans = Math.max(ans, len);
    }
    return ans == 0 ? -1 : ans;
  }

  private int MALextend(int[] a, Set<Integer> b, int gap) {
    int N = a.length;
    if (b.contains(a[0] - gap))
      N++;
    for (int i = 1; i < a.length; i++) {
      int temp = a[i - 1] + gap;
      while (temp != a[i]) {
        if (!b.contains(temp))
          return -1;
        N++;
        temp += gap;
      }
    }
    return N;
  }

  private int MALgcd(int a, int b) {
    return b == 0 ? a : MALgcd(b, a % b);
  }

  class MinMatrix {

    boolean[] rowDisable, colDisable;
    PriorityQueue<int[]> pq;

    public MinMatrix(int n, int m) {
      rowDisable = new boolean[n];
      colDisable = new boolean[m];
      pq = new PriorityQueue<>((a, b) -> (a[0] + 1) * (a[1] + 1) - (b[0] + 1) * (b[1] + 1));
      for (int r = 0; r < n; r++)
        for (int c = 0; c < m; c++)
          pq.offer(new int[]{r, c});
    }

    public int queryMin() {
      int[] idx = new int[2];
      while (!pq.isEmpty()) {
        idx = pq.poll();
        if (!rowDisable[idx[0]] && !colDisable[idx[1]])
          break;
      }
      return (idx[0] + 1) * (idx[1] + 1);
    }

    public void forbidRow(int row) {
      rowDisable[row] = true;
    }

    public void forbidCol(int col) {
      colDisable[col] = true;
    }
  }

  public int mirrorReflection2(int p, int q) {
    int gcd = MRgcd(p, q);
    q /= gcd;
    p /= gcd;
    boolean pisOdd = (p & 1) == 1, qisOdd = (q & 1) == 1;
    if (pisOdd == qisOdd)
      return 1;
    if (pisOdd && !qisOdd)
      return 0;
    else
      return 2;
  }

  private int MRgcd(int a, int b) {
    return b == 0 ? a : MRgcd(b, a % b);
  }

  public String baseNeg2(int N) {
    StringBuilder sb = new StringBuilder();
    while (N != 0) {
      sb.append(N & 1);
      N = -(N >> 1);
    }
    return sb.length() == 0 ? "0" : sb.reverse().toString();
  }


  static class Invoice {
    int id, amount, stage; //1-unfinished,2-unpaid,3-over

    public Invoice(int id, int amount) {
      this.id = id;
      this.amount = amount;
      stage = 1;
    }
  }

  public static int calculateTotalOwed(List<String> actions) {
    Map<Integer, Invoice> record = new HashMap<>();
    for (String action : actions) {
      String[] acts = action.split(" ");
      //parse id,amount,currency data
      Integer curId = null, curAmount = null;
      String curCurrency = null;
      String[] info = acts[1].split("&");
      for (String inf : info) {
        String[] parsed = inf.split("=");
        if (parsed[0].toLowerCase().equals("id"))
          curId = Integer.parseInt(parsed[1]);
        else if (parsed[0].toLowerCase().equals("amount"))
          curAmount = Integer.parseInt(parsed[1]);
        else if (parsed[0].toLowerCase().equals("currency"))
          curCurrency = parsed[1];
      }
      if (curCurrency != null && !curCurrency.toLowerCase().equals("usd")) {
        continue;
      }
      if (acts[0].equals("CREATE:")) {
        if (record.containsKey(curId))
          continue;
        record.put(curId, new Invoice(curId, curAmount));
      } else if (acts[0].equals("FINALIZE:")) {
        Invoice curInvoice = record.get(curId);
        if (curInvoice == null || curInvoice.stage != 1)
          continue;
        curInvoice.amount = curAmount;
        curInvoice.stage++;
      } else if (acts[0].equals("PAY:")) {
        Invoice curInvoice = record.get(curId);
        if (curInvoice == null || curInvoice.stage != 2)
          continue;
        curInvoice.amount = 0;
        curInvoice.stage++;
      }
    }
    int res = 0;
    for (Invoice invoice : record.values())
      res += invoice.amount;
    return res;
  }

  public int TwitterMaxHeight(int[] tablePos, int[] tableHeight) {
    int N = tablePos.length, res = 0;
    for (int i = 0; i < N - 1; i++)
      if (tablePos[i] != tablePos[i + 1] - 1) {
        int h1 = Math.max(tableHeight[i], tableHeight[i + 1]), h2 = Math.min(tableHeight[i], tableHeight[i + 1]);
        int curMax = Math.min(h1 + 1, h2 + tablePos[i] - tablePos[i + 1]);
        res = Math.max(res, curMax);
      }
    return res;
  }

  public List<List<Integer>> combinationSum22(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    if (candidates == null || candidates.length == 0)
      return ans;
    Arrays.sort(candidates);
    CS2helper1(candidates, 0, new ArrayList<>(), ans, target);
    return ans;
  }

  private void CS2helper1(int[] candidates, int curIdx, List<Integer> path, List<List<Integer>> res, int remain) {
    if (remain == 0) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = curIdx; i < candidates.length && candidates[i] <= remain; i++) {
      if (i != curIdx && candidates[i] == candidates[i - 1])
        continue;
      path.add(candidates[i]);
      CS2helper1(candidates, i + 1, path, res, remain - candidates[i]);
      path.remove(path.size() - 1);
    }
  }

  public int maxProfit1(int[] prices) {
    if (prices == null || prices.length == 0)
      return 0;
    int min = Integer.MAX_VALUE, ans = 0;
    for (int p : prices) {
      min = Math.min(min, p);
      ans = Math.max(p - min, ans);
    }
    return ans;
  }

  public int maxProfit2(int[] prices) {
    if (prices == null || prices.length == 0)
      return 0;
    int ans = 0, H = -1;
    for (int i = 0; i < prices.length - 1; i++)
      if (H == -1 && prices[i] < prices[i + 1])
        H = prices[i];
      else if (H != -1 && prices[i] > prices[i + 1]) {
        ans += prices[i] - H;
        H = -1;
      }
    if (H != -1)
      ans += prices[prices.length - 1] - H;
    return ans;
  }

  public int maxProfit3(int[] prices, int fee) {
    if (prices == null || prices.length == 0)
      return 0;
    int N = prices.length;
    int[] buy = new int[N], sell = new int[N];
    buy[0] = -prices[0] - fee;
    for (int i = 1; i < N; i++) {
      buy[i] = Math.max(buy[i - 1], sell[i - 1] - prices[i] - fee);
      sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
    }
    return sell[N - 1];
  }

  public int maxProfit4(int[] prices) {
    if (prices == null || prices.length == 0 || prices.length == 1)
      return 0;
    int N = prices.length;
    int[] buy = new int[N], sell = new int[N];
    buy[0] = -prices[0];
    buy[1] = Math.max(buy[0], -prices[1]);
    sell[1] = Math.max(0, buy[0] + prices[1]);
    for (int i = 2; i < N; i++) {
      buy[i] = Math.max(buy[i - 1], sell[i - 2] - prices[i]);
      sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
    }
    return sell[N - 1];
  }

  public int maxProfit5(int[] prices) {
    if (prices == null || prices.length == 0 || prices.length == 1)
      return 0;
    int buy1 = -prices[0], sell1 = 0, buy2 = Integer.MIN_VALUE, sell2 = Integer.MIN_VALUE;
    for (int i = 1; i < prices.length; i++) {
      buy1 = Math.max(buy1, -prices[i]);
      sell1 = Math.max(sell1, buy1 + prices[i]);
      buy2 = Math.max(buy2, sell1 - prices[i]);
      sell2 = Math.max(sell2, buy2 + prices[i]);
    }
    return sell2;
  }

  public int maxProfit6(int k, int[] prices) {
    if (prices == null || prices.length == 0 || prices.length == 1)
      return 0;
    int N = prices.length;
    if (k > (N >> 1))
      return MP6quick(prices);
    int[][] dp = new int[k + 1][N];
    for (int i = 1; i <= k; i++) {
      int tempMax = dp[i - 1][0] - prices[0];
      for (int j = 1; j < N; j++) {
        dp[i][j] = Math.max(dp[i][j - 1], prices[j] + tempMax);
        tempMax = Math.max(tempMax, dp[i - 1][j] - prices[j]);
      }
    }
    return dp[k][N - 1];
  }

  private int MP6quick(int[] P) {
    int res = 0;
    for (int i = 0; i < P.length - 1; i++)
      if (P[i] < P[i + 1])
        res += P[i + 1] - P[i];
    return res;
  }

  public String longestPalindrome1(String s) {
    if (s == null || s.isEmpty())
      return s;
    char[] cs = s.toCharArray();
    int N = cs.length, maxStart = 0, maxEnd = 0;
    for (int i = 0; i < N; i++) {
      int left = i, right = i, next;
      while (right < N - 1 && cs[right + 1] == cs[left])
        right++;
      next = right;
      while (left - 1 >= 0 && right + 1 < N && cs[left - 1] == cs[right + 1]) {
        left--;
        right++;
      }
      if (maxEnd - maxStart < right - left) {
        maxEnd = right;
        maxStart = left;
      }
      i = next;
    }
    return s.substring(maxStart, maxEnd + 1);
  }

  public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
    if (K <= 0 || quality == null || quality.length == 0)
      return 0;
    int N = quality.length;
    double[][] res = new double[N][2];
    for (int i = 0; i < N; i++)
      res[i] = new double[]{(double) wage[i] / (double) quality[i], quality[i]};
    Arrays.sort(res, new Comparator<double[]>() {
      @Override
      public int compare(double[] a, double[] b) {
        return Double.compare(a[0], b[0]);
      }
    });
    PriorityQueue<double[]> pq = new PriorityQueue<>(new Comparator<double[]>() {
      @Override
      public int compare(double[] a, double[] b) {
        return (int) b[1] - (int) a[1];
      }
    });
    double curRatio = 0, curQuality = 0, max = Double.MAX_VALUE;
    for (int i = 0; i < N; i++) {
      if (pq.size() < K)
        curQuality += res[i][1];
      else
        curQuality = curQuality - pq.poll()[1] + res[i][1];
      curRatio = res[i][0];
      pq.offer(res[i]);
      if (pq.size() == K)
        max = Math.min(max, curRatio * curQuality);
    }
    return max;
  }

  public String licenseKeyFormatting(String S, int K) {
    if (S == null || S.isEmpty())
      return S;
    char[] cs = S.toCharArray();
    int N = cs.length, len = 0, idx = 0;
    char[] res = new char[N];
    for (int i = N - 1; i >= 0; i--)
      if (cs[i] == '-')
        continue;
      else if (cs[i] >= 'a' && cs[i] <= 'z')
        res[len++] = (char) (cs[i] - 'a' + 'A');
      else
        res[len++] = cs[i];
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) {
      sb.append(res[i]);
      idx++;
      if (idx % K == 0 && i != len - 1)
        sb.append('-');
    }
    return sb.reverse().toString();
  }

  public int[] assignBikes(int[][] workers, int[][] bikes) {
    List<int[]>[] dist = new List[2001];
    int W = workers.length, B = bikes.length;
    int[] workAssign = new int[W], bikeAssign = new int[B];
    Arrays.fill(workAssign, -1);
    Arrays.fill(bikeAssign, -1);
    for (int w = 0; w < W; w++)
      for (int b = 0; b < B; b++) {
        int[] wPos = workers[w], bPos = bikes[b];
        int curDist = Math.abs(wPos[0] - bPos[0]) + Math.abs(wPos[1] - bPos[1]);
        if (dist[curDist] == null)
          dist[curDist] = new ArrayList<>();
        dist[curDist].add(new int[]{w, b});
      }
    for (int i = 0, count = 0; i < 2001 && count < W; i++) {
      if (dist[i] == null)
        continue;
      for (int[] res : dist[i])
        if (workAssign[res[0]] == -1 && bikeAssign[res[1]] == -1) {
          workAssign[res[0]] = res[1];
          bikeAssign[res[1]] = res[0];
          count++;
        }
    }
    return workAssign;
  }

  public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
    return VShelper(p1, p2, p3, p4) || VShelper(p1, p3, p2, p4) || VShelper(p1, p4, p2, p3);
  }

  private boolean VShelper(int[] d11, int[] d12, int[] d21, int[] d22) {
    int d1, d2, d3;
    return (d1 = VSgetDistance(d11, d21)) == VSgetDistance(d11, d22) &&
            (d2 = VSgetDistance(d12, d21)) == VSgetDistance(d12, d22) &&
            (d3 = VSgetDistance(d11, d12)) == VSgetDistance(d21, d22) &&
            d1 == d2 && d1 != 0 && d2 != 0 && d3 != 0;
  }

  private int VSgetDistance(int[] a, int[] b) {
    int dx = a[0] - b[0], dy = (a[1] - b[1]);
    return dx * dx + dy * dy;
  }

  class VSedge {
    int[] p1, p2;
    boolean isValid;

    public VSedge(int[] p1, int[] p2) {
      isValid = !(p1[0] == p2[0] && p1[1] == p2[1]);
      if (!isValid)
        return;
      if (p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1])) {
        this.p1 = p1;
        this.p2 = p2;
      } else {
        this.p1 = p2;
        this.p2 = p1;
      }
    }
  }

  private String VSencode(int[] p1, int[] p2) {
    return (p1[0] + p2[0]) + " " + (p1[1] + p2[1]) + " " + (Math.abs(p1[0] - p2[0]) * Math.abs(p1[1] - p2[1]));
  }

  public int validNSquare(int[][] points) {
    if (points == null || points.length == 0)
      return 0;
    int N = points.length;
    Map<String, List<VSedge>> edges = new HashMap<>();
    for (int i = 0; i < N; i++)
      for (int j = i + 1; j <= N; j++) {
        VSedge edge = new VSedge(points[i], points[j]);
        if (!edge.isValid)
          continue;
        String key = VSencode(points[i], points[j]);
        List<VSedge> res = edges.get(key);
        if (res == null) {
          res = new ArrayList<>();
          edges.put(key, res);
        }
        res.add(edge);
      }
    int ans = 0;
    for (List<VSedge> rectEdges : edges.values())
      ans += VSgetSquare(rectEdges);
    return ans;
  }

  private int VSgetSquare(List<VSedge> rectEdges) {
    int ans = 0, N = rectEdges.size();
    for (int i = 0; i < N; i++)
      for (int j = i + 1; j < N; j++) {
        int[] p1 = rectEdges.get(i).p1, p2 = rectEdges.get(i).p2, q1 = rectEdges.get(j).p1, q2 = rectEdges.get(j).p2;
        if (p1[0] == p2[0] || q1[1] == q2[1] || p1[1] == p2[1] || q1[0] == q2[0]) {
          if ((p1[0] == p2[0] && q1[1] == q2[1]) || (p1[1] == p2[1] && q1[0] == q2[0]))
            ans++;
          continue;
        }
        if ((p2[1] - p1[1]) * (q2[1] - q1[1]) == -(p2[0] - p1[0]) * (q2[1] - q1[1]))
          ans++;
      }
    return ans;
  }

  public TreeNode deleteNode2(TreeNode root, int key) {
    if (root == null)
      return root;
    if (root.val > key)
      root.left = deleteNode2(root.left, key);
    else if (root.val < key)
      root.right = deleteNode2(root.right, key);
    else {
      if (root.left == null)
        return root.right;
      else if (root.right == null)
        return root.left;
      TreeNode min = DNfindMin(root.right);
      min.right = DNdeleteMin(root.right);
      min.left = root.left;
      return min;
    }
    return root;
  }

  private TreeNode DNfindMin(TreeNode root) {
    if (root.left != null)
      return DNfindMin(root.left);
    return root;
  }

  private TreeNode DNdeleteMin(TreeNode root) {
    if (root.left == null)
      return root.right;
    root.left = DNdeleteMin(root.left);
    return root;
  }

  public ListNode insertionSortList(ListNode head) {
    if (head == null)
      return head;
    ListNode fakeHead = new ListNode(Integer.MIN_VALUE), cur = head;
    while (cur != null) {
      ListNode next = cur.next, pos = fakeHead;
      cur.next = null;
      while (pos.next != null && pos.next.val <= cur.val)
        pos = pos.next;
      cur.next = pos.next;
      pos.next = cur;
      cur = next;
    }
    return fakeHead.next;
  }

  public int firstMissingPositive1(int[] nums) {
    if (nums == null || nums.length == 0)
      return 1;
    int N = nums.length;
    int[] res = new int[N + 2];
    for (int i : nums)
      if (i > 0 && i < N + 2)
        res[i]++;
    for (int i = 1; i < N + 2; i++)
      if (res[i] == 0)
        return i;
    return 0;
  }

  public Map<String, Integer> minByColumn(Map<String, Integer>[] table, String col) {
    if (table == null || table.length == 0)
      return null;
    Map<String, Integer> minCol = null;
    int val = Integer.MAX_VALUE;
    for (Map<String, Integer> row : table) {
      int curVal = row.getOrDefault(col, 0);
      if (curVal < val) {
        val = curVal;
        minCol = row;
      }
    }
    return minCol;
  }


  public Map<String, Integer> minByOrder(Map<String, Integer>[] table, List<String> cols) {
    if (table == null || table.length == 0)
      return null;
    Map<String, Integer> min = table[0];
    for (int i = 1; i < table.length; i++)
      if (MBOcompare(table[i], min, cols) < 0)
        min = table[i];
    return min;
  }

  private int MBOcompare(Map<String, Integer> t1, Map<String, Integer> t2, List<String> cols) {
    for (String col : cols) {
      int v1 = t1.getOrDefault(col, 0), v2 = t2.getOrDefault(col, 0);
      if (v1 != v2)
        return v1 - v2;
    }
    return 0;
  }

  public boolean backspaceCompare1(String S, String T) {
    if ((S == null && T == null) || (S.isEmpty() && T.isEmpty()))
      return true;
    if (S == null || T == null || S.isEmpty() || T.isEmpty())
      return false;
    return BCconvert(S).equals(BCconvert(T));
  }

  private String BCconvert(String s) {
    StringBuilder sb = new StringBuilder();
    int withdraw = 0;
    char[] cs = s.toCharArray();
    for (int i = cs.length - 1; i >= 0; i--)
      if (cs[i] == '#')
        withdraw++;
      else if (withdraw == 0)
        sb.append(cs[i]);
      else
        withdraw--;
    return sb.toString();
  }

  public int minAreaRect1(int[][] points) {
    if (points == null || points.length == 0)
      return 0;
    Set<Long> P = new HashSet<>();
    int N = points.length, min = Integer.MAX_VALUE;
    for (int[] point : points)
      P.add(MARencode(point[0], point[1]));
    for (int i = 0; i < N; i++)
      for (int j = i + 1; j < N; j++) {
        int[] p1 = points[i], p2 = points[j];
        if (p1[0] == p2[0] || p1[1] == p2[1])
          continue;
        if (P.contains(MARencode(p1[0], p2[1])) && P.contains(MARencode(p2[0], p1[1])))
          min = Math.min(min, Math.abs(p1[0] - p2[0]) * Math.abs(p1[1] - p2[1]));
      }
    return min == Integer.MAX_VALUE ? 0 : min;
  }

  private long MARencode(int x, int y) {
    return x * 100000 + y;
  }

  public int maxAbsValExpr(int[] arr1, int[] arr2) {
    if (arr1 == null || arr1.length == 0)
      return 0;
    int N = arr1.length, ans = Integer.MIN_VALUE;
    int[] dirs = new int[]{-1, 1};
    for (int d1 : dirs)
      for (int d2 : dirs) {
        int min = d1 * arr1[0] + d2 * arr2[0];
        for (int i = 1; i < N; i++) {
          int cur = d1 * arr1[i] + d2 * arr2[i] + i;
          ans = Math.max(ans, cur - min);
          min = Math.min(min, cur);
        }
      }
    return ans;
  }

  public int leastInterval(char[] tasks, int n) {
    if (tasks == null || tasks.length == 0)
      return 0;
    int[] count = new int[26];
    for (char t : tasks)
      count[t - 'A']++;
    int N = tasks.length, maxFreq = 0, maxFreqNum = 0;
    for (int i = 0; i < 26; i++)
      if (count[i] > maxFreq) {
        maxFreq = count[i];
        maxFreqNum = 1;
      } else if (count[i] == maxFreq)
        maxFreqNum++;
    return maxFreqNum * maxFreq + Math.max((n + 1 - maxFreqNum) * (maxFreq - 1), N - maxFreq * maxFreqNum);
  }

  public ListNode removeElements(ListNode head, int val) {
    if (head == null)
      return head;
    ListNode fakeHead = new ListNode(0), cur = fakeHead;
    while (head != null) {
      ListNode next = head.next;
      head.next = null;
      if (head.val != val) {
        cur.next = head;
        cur = cur.next;
      }
      head = next;
    }
    return fakeHead.next;
  }

  public int numberOfSubarrays(int[] nums, int k) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length, ans = 0, evenNum = 1;
    int[] evenCount = new int[N + 1];
    for (int i = 0; i < N; i++)
      if ((nums[i] & 1) == 0)
        evenNum++;
      else {
        evenCount[i] = evenNum;
        evenNum = 1;
      }
    for (int start = 0, end = 0; end < N; end++) {
      if ((nums[end] & 1) == 1) {
        if (k > 0)
          k--;
        else if (k == 0) {
          start++;
          while (start != end && (nums[start] & 1) == 0)
            start++;
        }
      }
      if (k == 0) {
        while (start != end && (nums[start] & 1) == 0)
          start++;
        ans += evenCount[start];
      }
    }
    return ans;
  }

  public int minimumSwap(String s1, String s2) {
    char[] cs1 = s1.toCharArray(), cs2 = s2.toCharArray();
    int N = cs1.length, xy = 0, yx = 0, ans = 0;
    for (int i = 0; i < N; i++)
      if (cs1[i] != cs2[i]) {
        if (cs1[i] == 'x')
          xy++;
        else
          yx++;
      }
    ans += xy / 2;
    ans += yx / 2;
    xy %= 2;
    yx %= 2;
    if ((yx == 1 && xy == 0) || (yx == 0 && xy == 1))
      return -1;
    if (yx == 1 && xy == 1)
      ans += 2;
    return ans;
  }

  public String minRemoveToMakeValid(String s) {
    char[] cs = s.toCharArray();
    int N = cs.length, stackLen = 0;
    int[] left = new int[N];
    boolean[] avoid = new boolean[N];
    for (int i = 0; i < N; i++)
      if (cs[i] != '(' && cs[i] != ')')
        continue;
      else if (cs[i] == '(')
        left[stackLen++] = i;
      else {
        if (stackLen == 0)
          avoid[i] = true;
        else
          stackLen--;
      }
    for (int i = 0; i < stackLen; i++)
      avoid[left[i]] = true;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < N; i++)
      if (!avoid[i])
        sb.append(cs[i]);
    return sb.toString();
  }

  public boolean isGoodArray(int[] nums) {
    if (nums == null || nums.length == 0)
      return false;
    int gcd = nums[0];
    for (int i = 1; i < nums.length; i++) {
      gcd = IGAgcd(gcd, nums[i]);
      if (gcd == 1)
        return true;
    }
    return gcd == 1;
  }

  private int IGAgcd(int a, int b) {
    return b == 0 ? a : IGAgcd(b, a % b);
  }

  class Foo {

    private AtomicInteger state;

    public Foo() {
      state = new AtomicInteger(1);
    }

    public void first(Runnable printFirst) throws InterruptedException {

      // printFirst.run() outputs "first". Do not change or remove this line.
      while (!state.compareAndSet(1, 2)) ;
      printFirst.run();
    }

    public void second(Runnable printSecond) throws InterruptedException {

      // printSecond.run() outputs "second". Do not change or remove this line.
      while (!state.compareAndSet(2, 3)) ;
      printSecond.run();
    }

    public void third(Runnable printThird) throws InterruptedException {

      // printThird.run() outputs "third". Do not change or remove this line.
      while (!state.compareAndSet(3, 4)) ;
      printThird.run();
    }
  }

  public String originalDigits(String s) {
    if (s == null || s.isEmpty())
      return s;
    char[] cs = s.toCharArray();
    int[] count = new int[10], freq = new int[26];
    for (char c : cs)
      freq[c - 'a']++;
    StringBuilder sb = new StringBuilder();
    ODgetCount(freq, count);
    for (int i = 0; i <= 9; i++)
      while (count[i] != 0) {
        sb.append(i);
        count[i]--;
      }
    return sb.toString();
  }

  private void ODgetCount(int[] freq, int[] count) {
    count[0] = freq['z' - 'a'];
    count[2] = freq['w' - 'a'];
    count[4] = freq['u' - 'a'];
    count[6] = freq['x' - 'a'];
    count[8] = freq['g' - 'a'];
    count[1] = freq['o' - 'a'] - count[0] - count[2] - count[4];
    count[3] = freq['r' - 'a'] - count[0] - count[4];
    count[5] = freq['f' - 'a'] - count[4];
    count[7] = freq['s' - 'a'] - count[6];
    count[9] = freq['i' - 'a'] - count[5] - count[6] - count[8];
  }

  public int kConcatenationMaxSum1(int[] arr, int k) {
    if (arr == null || arr.length == 0)
      return 0;
    long sum = 0, midMax = KCMSgetMaxSubAndSum(arr);
    long maxPrefix = KCMSgetPrefix(arr), maxSuffix = KCMSgetSuffix(arr);
    long ans = 0;
    if (k == 1)
      return (int) midMax;
    for (int a : arr)
      sum += a;
    if (sum > 0) {
      ans = ((k - 2) * sum) % MODULO;
      return (int) Math.max((ans + maxPrefix % MODULO + maxSuffix % MODULO) % MODULO, midMax % MODULO);
    } else
      return (int) Math.max((maxPrefix % MODULO + maxSuffix % MODULO) % MODULO, midMax % MODULO);
  }

  private long KCMSgetPrefix(int[] arr) {
    long prefix = 0, sum = 0;
    for (int a : arr) {
      sum += a;
      sum %= MODULO;
      prefix = Math.max(prefix, sum);
    }
    return prefix;
  }

  private long KCMSgetSuffix(int[] arr) {
    long suffix = 0, sum = 0;
    for (int i = arr.length - 1; i >= 0; i--) {
      sum += arr[i];
      sum %= MODULO;
      suffix = Math.max(sum, suffix);
    }
    return suffix;
  }

  private long KCMSgetMaxSubAndSum(int[] arr) {
    long sub = 0, cur = 0;
    for (int a : arr) {
      if (cur <= 0)
        cur = a;
      else {
        cur += a;
        cur %= MODULO;
      }
      sub = Math.max(sub, cur);
    }
    return sub;
  }

  class FooBar {
    private int n;
    private Semaphore semaphore1, semaphore2;

    public FooBar(int n) {
      this.n = n;
      semaphore1 = new Semaphore(1);
      semaphore2 = new Semaphore(0);
    }

    public void foo(Runnable printFoo) throws InterruptedException {

      for (int i = 0; i < n; i++) {
        semaphore1.acquire();
        // printFoo.run() outputs "foo". Do not change or remove this line.
        printFoo.run();
        semaphore2.release();
      }
    }

    public void bar(Runnable printBar) throws InterruptedException {

      for (int i = 0; i < n; i++) {
        semaphore2.acquire();
        // printBar.run() outputs "bar". Do not change or remove this line.
        printBar.run();
        semaphore1.release();
      }
    }
  }

  public int longestConsecutive(TreeNode root) {
    if (root == null)
      return 0;
    int[] res = new int[1];
    LCHelper(root, Integer.MIN_VALUE, 1, res);
    return res[0];
  }

  private void LCHelper(TreeNode cur, int last, int length, int[] res) {
    if (cur == null)
      return;
    int curLen = cur.val == last + 1 ? length + 1 : 1;
    res[0] = Math.max(curLen, res[0]);
    LCHelper(cur.left, cur.val, curLen, res);
    LCHelper(cur.right, cur.val, curLen, res);
  }

  class NLNdata {
    int idx, val;

    public NLNdata(int idx, int val) {
      this.idx = idx;
      this.val = val;
    }
  }

  public int[] nextLargerNodes1(ListNode head) {
    if (head == null)
      return new int[0];
    int len = 0;
    ListNode cur = head;
    while (cur != null) {
      len++;
      cur = cur.next;
    }
    int[] ans = new int[len];
    cur = head;
    int idx = 0, size = 0;
    NLNdata[] stack = new NLNdata[len];
    while (cur != null) {
      while (size != 0 && stack[size - 1].val < cur.val)
        ans[stack[--size].idx] = cur.val;
      stack[size++] = new NLNdata(idx, cur.val);
      idx++;
      cur = cur.next;
    }
    return ans;
  }

  public boolean canPartitionKSubsets(int[] nums, int k) {
    int sum = 0;
    for (int n : nums)
      sum += n;
    if ((sum % k) != 0)
      return false;
    Arrays.sort(nums);
    boolean[] visited = new boolean[nums.length];
    return CPShelper(nums, nums.length - 1, k, 0, sum / k, visited);
  }

  private boolean CPShelper(int[] nums, int curIdx, int groups, int curSum, int target, boolean[] visited) {
    if (groups == 0)
      return true;
    if (curSum == target)
      return CPShelper(nums, nums.length - 1, groups - 1, 0, target, visited);
    for (int i = curIdx; i >= 0; i--)
      if (!visited[i] && nums[i] + curSum <= target) {
        visited[i] = true;
        if (CPShelper(nums, i - 1, groups, curSum + nums[i], target, visited))
          return true;
        visited[i] = false;
      }
    return false;
  }

  public Map<String, Integer> minByKey(Map<String, Integer>[] data, String key) {
    if (data == null || data.length == 0)
      return null;
    int N = data.length, maxIdx = -1, maxVal = Integer.MAX_VALUE;
    for (int i = 0; i < data.length; i++) {
      Map<String, Integer> map = data[i];
      int val = map.getOrDefault(key, 0);
      if (val < maxVal) {
        maxVal = val;
        maxIdx = i;
      }
    }
    return data[maxIdx];
  }


  public Map<String, Integer> firstByKey(Map<String, Integer>[] data, String key, String order) {
    if (data == null || data.length == 0)
      return null;
    if (!order.equals("asc") && !order.equals("desc"))
      throw new IllegalArgumentException("wrong order");
    boolean isAsc = order.equals("asc");
    int targetVal = isAsc ? Integer.MAX_VALUE : Integer.MIN_VALUE, targetIdx = -1;
    for (int i = 0; i < data.length; i++) {
      int curVal = data[i].getOrDefault(key, 0);
      if (isAsc && curVal < targetVal) {
        targetVal = curVal;
        targetIdx = i;
      } else if (!isAsc && curVal > targetVal) {
        targetVal = curVal;
        targetIdx = i;
      }
    }
    return data[targetIdx];
  }

  class DataComparator {
    String key;
    boolean isAsc;

    public DataComparator() {
    }

    public DataComparator(String key, String dir) {
      if (!dir.equals("asc") && !dir.equals("desc"))
        throw new IllegalArgumentException("wrong dir");
      this.key = key;
      this.isAsc = dir.equals("asc");
    }

    public int compara(Map<String, Integer> a, Map<String, Integer> b) {
      int aVal = a.getOrDefault(key, 0);
      int bVal = b.getOrDefault(key, 0);
      return isAsc ? aVal - bVal : bVal - aVal;
    }
  }

  public Map<String, Integer> firstByKey2(Map<String, Integer>[] data, String key, String order) {
    if (data == null || data.length == 0)
      return null;
    DataComparator comparator = new DataComparator(key, order);
    int targetIdx = 0;
    for (int i = 1; i < data.length; i++) {
      if (comparator.compara(data[i], data[targetIdx]) < 0)
        targetIdx = i;
    }
    return data[targetIdx];
  }

  class SortOrderComparator {
    private List<String> order;
    private boolean isAsc;

    public SortOrderComparator(List<String> order, String dir) {
      if (!dir.equals("asc") && !dir.equals("desc"))
        throw new IllegalArgumentException("wrong dir");
      if (order == null || order.isEmpty())
        throw new IllegalArgumentException("wrong order");
      this.order = order;
      this.isAsc = dir.equals("asc");
    }

    public int compare(Map<String, Integer> a, Map<String, Integer> b) {
      for (String key : order) {
        int aVal = a.getOrDefault(key, 0);
        int bVal = b.getOrDefault(key, 0);
        if (aVal != bVal)
          return isAsc ? aVal - bVal : bVal - aVal;
      }
      return 0;
    }
  }

  private boolean assertEquals(Integer a, Integer b) {
    if ((a == null && b == null) || (a != null && a.equals(b)))
      return true;
    throw new AssertionError("expect: " + a + "\nacutal: " + b);
  }

  public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
      return false;
    int R = matrix.length, C = matrix[0].length, curR = 0, curC = C - 1;
    while (curR < R && curC >= 0 && matrix[curR][curC] != target) {
      if (matrix[curR][curC] < target)
        curR++;
      else
        curC--;
    }
    return curR < R && curC >= 0;
  }

  class SRnode implements Comparable<SRnode> {
    int val, idx, row;

    public SRnode(int val, int idx, int row) {
      this.val = val;
      this.idx = idx;
      this.row = row;
    }

    @Override
    public int compareTo(SRnode o) {
      return val - o.val;
    }
  }

  public int[] smallestRange(List<List<Integer>> nums) {
    int N = nums.size(), start = Integer.MAX_VALUE, end = Integer.MIN_VALUE, curMax = Integer.MIN_VALUE;
    PriorityQueue<SRnode> pq = new PriorityQueue<>();
    for (int i = 0; i < N; i++) {
      int val = nums.get(i).get(0);
      start = Math.min(val, start);
      end = Math.max(val, end);
      curMax = Math.max(curMax, val);
      pq.offer(new SRnode(val, 0, i));
    }

    while (pq.size() == N) {
      SRnode cur = pq.poll();
      if (end - start > curMax - cur.val) {
        end = curMax;
        start = cur.val;
      }
      if (cur.idx < nums.get(cur.row).size() - 1) {
        SRnode temp = new SRnode(nums.get(cur.row).get(cur.idx + 1), cur.idx + 1, cur.row);
        pq.offer(temp);
        curMax = Math.max(curMax, temp.val);
      }
    }
    return new int[]{start, end};
  }

  class IntConsumer {
    public void accept(int x) {
      System.out.print(x);
    }
  }

  class ZeroEvenOdd {
    private int n;
    private boolean isOdd;
    private boolean isZero;
    private Lock lock;
    private Condition condition;

    public ZeroEvenOdd(int n) {
      if (n < 0)
        throw new IllegalArgumentException();
      this.n = n;
      isOdd = true;
      isZero = true;
      lock = new ReentrantLock();
      condition = lock.newCondition();
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
      for (int i = 0; i < n; i++) {
        lock.lock();
        try {
          while (!isZero)
            condition.await();
          printNumber.accept(0);
          isZero = false;
          condition.signalAll();
        } finally {
          lock.unlock();
        }
      }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
      for (int i = 2; i <= n; i += 2) {
        lock.lock();
        try {
          while (isZero || isOdd)
            condition.await();
          printNumber.accept(i);
          isOdd = true;
          isZero = true;
          condition.signalAll();
        } finally {
          lock.unlock();
        }
      }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
      for (int i = 1; i <= n; i += 2) {
        lock.lock();
        try {
          while (isZero || !isOdd)
            condition.await();
          printNumber.accept(i);
          isOdd = false;
          isZero = true;
          condition.signalAll();
        } finally {
          lock.unlock();
        }
      }
    }
  }

  public int[][] merge(int[][] intervals) {
    if (intervals == null || intervals.length == 0)
      return intervals;
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0] - b[0];
      }
    });
    int N = intervals.length;
    List<int[]> res = new ArrayList<>(N);
    for (int[] I : intervals)
      if (res.isEmpty())
        res.add(I);
      else {
        int[] last = res.get(res.size() - 1);
        if (last[1] >= I[0])
          last[1] = Math.max(I[1], last[1]);
        else
          res.add(I);
      }
    int[][] ans = new int[res.size()][2];
    for (int i = 0; i < ans.length; i++)
      ans[i] = res.get(i);
    return ans;
  }

  public int[][] insert(int[][] intervals, int[] newInterval) {
    if (intervals == null || intervals.length == 0)
      return new int[][]{newInterval};
    int N = intervals.length;
    int[][] ans;
    List<int[]> res = new ArrayList<>();
    int startPos = insertMaxSmallIdx(intervals, newInterval[0]);
    int endPos = insertMaxSmallIdx(intervals, newInterval[1]);
    if (endPos == -1) {
      res.add(newInterval);
      for (int i = 0; i < N; i++)
        res.add(intervals[i]);
    } else if (startPos == -1) {
      int[] newTerm = new int[]{newInterval[0], Math.max(newInterval[1], intervals[endPos][1])};
      res.add(newTerm);
      for (int i = endPos + 1; i < N; i++)
        res.add(intervals[i]);
    } else {
      startPos = intervals[startPos][1] >= newInterval[0] ? startPos : startPos + 1;
      int[] newTerm = startPos == N ? newInterval :
              new int[]{Math.min(newInterval[0], intervals[startPos][0]), Math.max(newInterval[1], intervals[endPos][1])};
      for (int i = 0; i < startPos; i++)
        res.add(intervals[i]);
      res.add(newTerm);
      for (int i = endPos + 1; i < N; i++)
        res.add(intervals[i]);
    }
    ans = new int[res.size()][2];
    for (int i = 0; i < res.size(); i++)
      ans[i] = res.get(i);
    return ans;
  }

  private int insertMaxSmallIdx(int[][] intervals, int num) {
    int start = 0, end = intervals.length - 1;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (intervals[mid][0] <= num)
        start = mid + 1;
      else
        end = mid - 1;
    }
    return end;
  }

  class CallBack {
    String name;

    public CallBack() {
    }

    public CallBack(String name) {
      this.name = name;
    }

    public void call() {
      System.out.println("CallBack Event " + this.name + "is running now");
    }
  }

  class SingleThread {
    private Queue<CallBack> queue;
    private boolean isFired;

    public SingleThread() {
      queue = new LinkedList<>();
      isFired = false;
    }

    public void register(CallBack callback) {
      if (isFired)
        callback.call();
      else
        queue.offer(callback);
    }

    public void fire() {
      isFired = true;
      while (!queue.isEmpty())
        queue.poll().call();
    }
  }

  class MultipleThread {
    private Queue<CallBack> queue;
    private boolean isFired;
    private Lock lock;

    public MultipleThread() {
      queue = new LinkedList<>();
      isFired = false;
      lock = new ReentrantLock();
    }

    public void register(CallBack callback) {
      lock.lock();
      if (isFired) {
        lock.unlock();
        callback.call();
      } else {
        queue.offer(callback);
        lock.unlock();
      }
    }

    public void fire() {
      lock.lock();
      isFired = true;
      lock.unlock();
      while (!queue.isEmpty())
        queue.poll().call();
    }
  }

  class OrderedMultipleThread {
    private Queue<CallBack> queue;
    private boolean isFired;
    private Lock lock;

    public OrderedMultipleThread() {
      queue = new LinkedList<>();
      isFired = false;
      lock = new ReentrantLock();
    }

    public void register(CallBack callback) {
      lock.lock();
      if (isFired) {
        lock.unlock();
        callback.call();
      } else {
        queue.offer(callback);
        lock.unlock();
      }
    }

    public void fire() {
      lock.lock();
      while (!queue.isEmpty()) {
        lock.unlock();
        queue.poll().call();
        lock.lock();
      }
      isFired = true;
      lock.unlock();
    }
  }

  class NoLockMultipleThread {
    private ConcurrentLinkedQueue<CallBack> queue;
    private volatile boolean isFired;

    public NoLockMultipleThread() {
      queue = new ConcurrentLinkedQueue<>();
      isFired = false;
    }

    public void register(CallBack callback) {
      if (isFired)
        callback.call();
      else {
        queue.offer(callback);
        if (isFired) {
          if (!queue.isEmpty())
            queue.poll().call();
        }
      }
    }

    public void fire() {
      isFired = true;
      while (!queue.isEmpty())
        queue.poll().call();
    }
  }

  public static void commonSubstring(List<String> a, List<String> b) {
    // Write your code here
    int N = a.size();
    for (int i = 0; i < N; i++)
      if (hasCommonSubString(a.get(i), b.get(i)))
        System.out.println("YES");
      else
        System.out.println("NO");
  }

  private static boolean hasCommonSubString(String a, String b) {
    int[] count = new int[26];
    for (char c : a.toCharArray())
      count[c - 'a']++;
    for (char c : b.toCharArray())
      if (count[c - 'a'] != 0)
        return true;
    return false;
  }

  public static class NameTrie {
    boolean isName;
    int nameCount;
    NameTrie[] nexts;

    public NameTrie() {
      isName = false;
      nameCount = 0;
      nexts = new NameTrie[26];
    }
  }

  private static void insertNameTrie(NameTrie root, String name) {
    for (char n : name.toCharArray()) {
      if (root.nexts[n - 'a'] == null)
        root.nexts[n - 'a'] = new NameTrie();
      root.nameCount++;
      root = root.nexts[n - 'a'];
    }
  }

  private static int getPrefixNum(NameTrie root, String prefix) {
    for (char c : prefix.toCharArray()) {
      if (root.nexts[c - 'a'] == null)
        return 0;
      root = root.nexts[c - 'a'];
    }
    return root.nameCount;
  }

  public static List<Integer> findCompletePrefixes(List<String> names, List<String> query) {
    // Write your code here
    NameTrie root = new NameTrie();
    for (String name : names)
      insertNameTrie(root, name);
    List<Integer> res = new ArrayList<>();
    for (String q : query)
      res.add(getPrefixNum(root, q));
    return res;
  }

  public boolean PredictTheWinner(int[] nums) {
    if (nums.length == 1)
      return true;
    int N = nums.length, sum = 0;
    for (int n : nums)
      sum += n;
    int[][] memo = new int[N][N];
    int maxFirst = PWgetMax(nums, memo, 0, N - 1);
    return maxFirst >= (sum - maxFirst);
  }

  private int PWgetMax(int[] nums, int[][] memo, int start, int end) {
    if (start > end)
      return 0;
    if (start == end)
      return nums[start];
    if (start == end - 1)
      return Math.max(nums[start], nums[end]);
    if (memo[start][end] != 0)
      return memo[start][end];
    int chooseStart = nums[start] + Math.min(PWgetMax(nums, memo, start + 2, end), PWgetMax(nums, memo, start + 1, end - 1));
    int chooseEnd = nums[end] + Math.min(PWgetMax(nums, memo, start, end - 2), PWgetMax(nums, memo, start + 1, end - 1));
    int res = Math.max(chooseStart, chooseEnd);
    memo[start][end] = res;
    return res;
  }

  public int stoneGameII(int[] piles) {
    if (piles == null || piles.length == 0)
      return 0;
    if (piles.length == 1)
      return piles[0];
    if (piles.length == 2)
      return piles[0] + piles[1];
    int N = piles.length;
    int[][] memo = new int[N][N];
    int exceed = SGhelper(piles, 0, 1, memo);
    int sum = 0;
    for (int p : piles)
      sum += p;
    return (sum + exceed) >> 1;
  }

  private int SGhelper(int[] piles, int idx, int M, int[][] memo) {
    int N = piles.length;
    if (idx >= N)
      return 0;
    if (idx == N - 1)
      return piles[idx];
    if (memo[M][idx] != 0)
      return memo[M][idx];
    int res = Integer.MIN_VALUE, accum = 0;
    for (int i = 0; i < (M << 1) && idx + i < N; i++) {
      accum += piles[i + idx];
      res = Math.max(res, accum - SGhelper(piles, i + idx + 1, Math.max(M, i + 1), memo));
    }
    memo[M][idx] = res;
    return res;
  }

  public boolean isInterleave(String s1, String s2, String s3) {
    if (s1 == null || s2 == null)
      return s3 == null;
    if (s1.isEmpty() && s2.isEmpty())
      return s3.isEmpty();
    if (s1.length() + s2.length() != s3.length())
      return false;
    int N1 = s1.length(), N2 = s2.length(), N3 = s3.length();
    int[][][] memo = new int[N1][N2][N3]; // -1: false,0:uninitialized, 1:true
    return ISLhelper(s1.toCharArray(), s2.toCharArray(), s3.toCharArray(), 0, 0, 0, memo) > 0;
  }

  private int ISLhelper(char[] cs1, char[] cs2, char[] cs3, int idx1, int idx2, int idx3, int[][][] memo) {
    int N1 = cs1.length, N2 = cs2.length, N3 = cs3.length;
    if (idx3 == N3)
      return (N1 == idx1 && idx2 == N2) ? 1 : -1;
    if (idx1 == N1 && idx2 == N2)
      return -1;
    if (idx1 == N1 || idx2 == N2) {
      char[] remain;
      int remainIdx;
      if (idx1 == N1) {
        remain = cs2;
        remainIdx = idx2;
      } else {
        remain = cs1;
        remainIdx = idx1;
      }
      while (idx3 < N3) {
        if (remain[remainIdx++] != cs3[idx3++])
          return -1;
      }
      return 1;
    }
    if (memo[idx1][idx2][idx3] != 0)
      return memo[idx1][idx2][idx3];
    int res;
    if (cs1[idx1] == cs3[idx3] && cs2[idx2] == cs3[idx3])
      res = Math.max(ISLhelper(cs1, cs2, cs3, idx1 + 1, idx2, idx3 + 1, memo),
              ISLhelper(cs1, cs2, cs3, idx1, idx2 + 1, idx3 + 1, memo));
    else if (cs1[idx1] == cs3[idx3])
      res = ISLhelper(cs1, cs2, cs3, idx1 + 1, idx2, idx3 + 1, memo);
    else if (cs2[idx2] == cs3[idx3])
      res = ISLhelper(cs1, cs2, cs3, idx1, idx2 + 1, idx3 + 1, memo);
    else
      res = -1;
    memo[idx1][idx2][idx3] = res;
    return res;
  }

  public boolean isMatch3(String s, String p) {
    if ((s == null && p == null) || (s.isEmpty() && p.isEmpty()))
      return true;
    int N1 = s.length(), N2 = p.length();
    char[] cs1 = s.toCharArray(), cs2 = p.toCharArray();
    boolean[][] dp = new boolean[N1 + 1][N2 + 1], sinceZero = new boolean[N1 + 1][N2 + 1];
    for (int i = 0; i <= N1; i++)
      sinceZero[i][0] = true;
    dp[0][0] = true;
    for (int j = 0; j < N2; j++)
      if (cs2[j] == '*')
        dp[0][j + 1] = dp[0][j];
    for (int j = 0; j < N2; j++)
      for (int i = 0; i < N1; i++) {
        if ((cs1[i] == cs2[j]) || (cs2[j] == '?')) {
          boolean res = dp[i][j];
          dp[i + 1][j + 1] = res;
        } else if (cs2[j] == '*') {
          boolean res = sinceZero[i][j] || dp[i + 1][j];
          dp[i + 1][j + 1] = res;
        }
        sinceZero[i + 1][j + 1] = dp[i + 1][j + 1] || sinceZero[i][j + 1];
      }
    return dp[N1][N2];
  }

  public boolean isMatch4(String s, String p) {
    if ((s == null && p == null) || (s.isEmpty() && p.isEmpty()))
      return true;
    int N1 = s.length(), N2 = p.length(), sIdx = 0, pIdx = 0, startIdx = -1, matchIdx = -1;
    char[] cs1 = s.toCharArray(), cs2 = p.toCharArray();
    while (sIdx < N1) {
      if (pIdx < N2 && (cs1[sIdx] == cs2[pIdx] || cs2[pIdx] == '?')) {
        sIdx++;
        pIdx++;
      } else if (pIdx < N2 && cs2[pIdx] == '*') {
        startIdx = pIdx;
        matchIdx = sIdx;
        pIdx++;
      } else if (startIdx != -1) {
        pIdx = startIdx + 1;
        sIdx = ++matchIdx;
      } else
        return false;
    }
    while (pIdx < N2)
      if (cs2[pIdx++] != '*')
        return false;
    return true;
  }

  public boolean isMatch5(String s, String p) {
    if ((s == null && p == null) || (s.isEmpty() && p.isEmpty()))
      return true;
    int N1 = s.length(), N2 = p.length(), sIdx = 0, pIdx = 0, startIdx = -1, matchIdx = -1;
    char[] cs1 = s.toCharArray(), cs2 = p.toCharArray();
    boolean[][] dp = new boolean[N1 + 1][N2 + 1];
    dp[0][0] = true;
    for (int j = 0; j < N2; j++) {
      if (cs2[j] != '*')
        break;
      dp[0][j + 1] = true;
    }
    for (int i = 0; i < N1; i++)
      for (int j = 0; j < N2; j++)
        if (cs2[j] != '*')
          dp[i + 1][j + 1] = (cs1[i] == cs2[j] || cs2[j] == '?') && dp[i][j];
        else
          dp[i + 1][j + 1] = dp[i + 1][j] || dp[i][j + 1];
    return dp[N1][N2];
  }

  public class MinimalisticTokenBucket {

    private final long capacity; // capacity of the token bucket
    private final double refillTokensPerOneMillis; // speed to refill the capacity

    private double availableTokens; // last time token number
    private long lastRefillTimestamp; // last time stamp

    /**
     * Creates token-bucket with specified capacity and refill rate equals to
     * refillTokens/refillPeriodMillis
     */
    public MinimalisticTokenBucket(long capacity, long refillTokens, long refillPeriodMillis) {
      this.capacity = capacity;
      this.refillTokensPerOneMillis = (double) refillTokens / (double) refillPeriodMillis;

      this.availableTokens = capacity;
      this.lastRefillTimestamp = System.currentTimeMillis();
    }

    synchronized public boolean tryConsume(int numberTokens) {
      refill();
      if (availableTokens < numberTokens) {
        return false;
      } else {
        availableTokens -= numberTokens;
        return true;
      }
    }

    private void refill() {
      long currentTimeMillis = System.currentTimeMillis();
      if (currentTimeMillis > lastRefillTimestamp) {
        long millisSinceLastRefill = currentTimeMillis - lastRefillTimestamp;
        double refill = millisSinceLastRefill * refillTokensPerOneMillis;
        this.availableTokens = Math.min(capacity, availableTokens + refill);
        this.lastRefillTimestamp = currentTimeMillis;
      }
    }
  }

  class UserInfo {
    long lastVisitTime;
    double lastAvailSpace;

    public UserInfo(long lastAvailSpace) {
      lastVisitTime = System.currentTimeMillis();
      this.lastAvailSpace = lastAvailSpace;
    }
  }

  class ConcurrentRateLimiter {
    private ConcurrentHashMap<Long, UserInfo> record;
    private final long capacity;
    private final double ratePerMills;

    public ConcurrentRateLimiter(long space, long refillTokens, long refillsPeriodMillis) {
      if (space <= 0)
        throw new IllegalArgumentException("space cannot be 0 or negative");
      capacity = space;
      ratePerMills = (double) refillTokens / (double) refillsPeriodMillis;
      record = new ConcurrentHashMap<>();
    }

    public boolean tryConsume(Long uid, long tokensNeed) {
      UserInfo userInfo = refill(uid);
      synchronized (userInfo) {
        if (tokensNeed > userInfo.lastAvailSpace)
          return false;
        else {
          userInfo.lastAvailSpace -= tokensNeed;
          return true;
        }
      }
    }

    private UserInfo refill(long uid) {
      UserInfo userInfo;
      if ((userInfo = record.get(uid)) == null) {
        synchronized (this) {
          if ((userInfo = record.get(uid)) == null) {
            userInfo = new UserInfo(capacity);
            record.put(uid, userInfo);
            return userInfo;
          }
        }
      }
      long curTimeMillis = System.currentTimeMillis();
      synchronized (userInfo) {
        if (curTimeMillis > userInfo.lastVisitTime) {
          long timeInverval = curTimeMillis - userInfo.lastVisitTime;
          double refilled = userInfo.lastAvailSpace + timeInverval * ratePerMills;
          userInfo.lastAvailSpace = Math.min(refilled, capacity);
          userInfo.lastVisitTime = curTimeMillis;
        }
      }
      return userInfo;
    }
  }

  int idx = 1;

  public void testCucurrentRateLimiter() {
    ExecutorService executorService = Executors.newFixedThreadPool(10);
    ConcurrentRateLimiter rateLimiter = new ConcurrentRateLimiter(10, 10, 1);
    for (int i = 0; i < 15; i++) {
      Future<Boolean> res = executorService.submit(new Callable<Boolean>() {
        @Override
        public Boolean call() throws Exception {
          return rateLimiter.tryConsume((long) (idx++), (long) (idx));
        }
      });
      try {
        System.out.println("id: " + i + " res: " + res.get());
      } catch (InterruptedException e) {
        e.printStackTrace();
      } catch (ExecutionException e) {
        e.printStackTrace();
      }
    }
    shutDownThreadPool(executorService);
  }

  private void shutDownThreadPool(ExecutorService executorService) {
    try {
      System.out.println("shut down executor");
      executorService.shutdown();
      executorService.awaitTermination(5, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      e.printStackTrace();
    } finally {
      if (!executorService.isTerminated()) {
        System.err.println("cancel non finished tasks");
      }
      executorService.shutdownNow();
      System.out.println("finished");
    }
  }

  class Logger {

    class Info {
      String message;
      int lastTime;

      public Info(String message, int lastTime) {
        this.message = message;
        this.lastTime = lastTime;
      }
    }

    Map<String, Info> record;
    int interval;

    /**
     * Initialize your data structure here.
     */
    public Logger() {
      record = new HashMap<>();
      interval = 10;
    }

    /**
     * Returns true if the message should be printed in the given timestamp, otherwise returns
     * false. If this method returns false, the message will not be printed. The timestamp is in
     * seconds granularity.
     */
    public boolean shouldPrintMessage(int timestamp, String message) {
      Info info;
      if ((info = record.get(message)) == null) {
        info = new Info(message, timestamp);
        record.put(message, info);
        return true;
      }
      if (timestamp - info.lastTime >= interval) {
        info.lastTime = timestamp;
        return true;
      } else
        return false;
    }
  }

  public int slidingPuzzle(int[][] board) {
    if (board == null || board.length == 0)
      return 0;
    int start = encodeBoard(board), end = 54321;
    if (start == end)
      return 0;
    Set<Integer> appeared = new HashSet<>();
    Queue<Integer> q = new LinkedList<>();
    q.offer(start);
    int R = board.length, C = board[0].length, ans = 0;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    while (!q.isEmpty()) {
      int size = q.size();
      ans++;
      for (int i = 0; i < size; i++) {
        int last = q.poll();
        for (int[] d : dirs) {
          int next = SPgetCodeAfterMove(last, d, R, C);
          if (next == 0)
            continue;
          if (next == end)
            return ans;
          if (appeared.add(next))
            q.offer(next);
        }
      }
    }
    return -1;
  }

  private int SPgetCodeAfterMove(int code, int[] dir, int R, int C) {
    int zeroPos = SPgetZeroLocation(code);
    int r = zeroPos / C, c = zeroPos % C;
    int curR = r + dir[0], curC = c + dir[1];
    if (curR >= R || curR < 0 || curC >= C || curC < 0)
      return 0;
    int prevPos = r * C + c, curPos = curR * C + curC;
    int curVal = code / (int) Math.pow(10, curPos) % 10;
    code -= curVal * Math.pow(10, curPos);
    code += curVal * Math.pow(10, prevPos);
    return code;
  }

  private int SPgetZeroLocation(int code) {
    int locIdx = 0;
    while (code % 10 != 0) {
      locIdx++;
      code /= 10;
    }
    return locIdx;
  }

  private int encodeBoard(int[][] board) {
    int R = board.length, C = board[0].length, ans = 0;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        ans += board[r][c] * Math.pow(10, r * C + c);
    return ans;
  }

  public String dayOfTheWeek2(int day, int month, int year) {
    String[] days = new String[]{"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    int[] months = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (isPrimeYear(year))
      months[1] = 29;
    int res = 0;
    for (int m = 0; m < month - 1; m++)
      res += months[m];
    res += day;
    year--;
    res += year * 365 + year / 4 - year / 100 + year / 400;
    return days[res % 7];
  }

  private boolean isPrimeYear(int year) {
    return year % 400 == 0 || (year % 100 != 0 && year % 4 == 0);
  }

  public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length == 0)
      return nums;
    int N = nums.length, idx = 0;
    int[] res = new int[N - k + 1];
    Deque<Integer> dq = new ArrayDeque<>(k);
    for (int i = 0; i < N; i++) {
      if (!dq.isEmpty() && dq.peekFirst() == i - k)
        dq.pollFirst();
      while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i])
        dq.pollLast();
      dq.offerLast(i);
      if (i >= k - 1)
        res[idx++] = nums[dq.peekFirst()];
    }
    return res;
  }

  public static int ada(int year) {
    int[] month = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (isLeapYears(year))
      month[1] = 29;
    int dayToOct = 0;
    for (int i = 0; i < 9; i++)
      dayToOct += month[i];
    dayToOct++;
    dayToOct += (year - 1) * 365 + (year - 1) / 4 - (year - 1) / 100 + (year - 1) / 400;
    dayToOct %= 7;
    int daysToTuesday = 2 >= dayToOct ? 2 - dayToOct : 7 - (dayToOct - 2);
    int adaDay = daysToTuesday + 7 + 1;
    return adaDay;
  }

  private static boolean isLeapYears(int year) {
    return year % 400 == 0 || (year % 100 != 0 && year % 4 == 0);
  }

  public int leastInterval2(char[] tasks, int n) {
    if (tasks == null || tasks.length == 0)
      return 0;
    if (tasks.length == 1 || n == 0)
      return tasks.length;
    int length = tasks.length, maxFreqNum = 0, maxFreq = 0;
    int[] count = new int[26];
    for (char t : tasks)
      count[t - 'A']++;
    for (int i = 0; i < 26; i++)
      if (count[i] > maxFreq) {
        maxFreq = count[i];
        maxFreqNum = 1;
      } else if (count[i] == maxFreq)
        maxFreqNum++;
    int base = maxFreq * maxFreqNum;
    int remain = length - base;
    int remainSpace = (maxFreq - 1) * (n + 1 - maxFreqNum);
    return base + Math.max(remain, remainSpace);
  }

  public int largest1BorderedSquare1(int[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0)
      return 0;
    int R = grid.length, C = grid[0].length;
    int[][] LtoR = new int[R + 1][C + 1], TtoD = new int[R + 1][C + 1];
    for (int i = 0; i < R; i++)
      for (int j = 0; j < C; j++)
        if (grid[i][j] == 1) {
          LtoR[i + 1][j + 1] = LtoR[i + 1][j] + 1;
          TtoD[i + 1][j + 1] = TtoD[i][j + 1] + 1;
        }
    int maxEgde = 0;
    for (int r = 1; r <= R; r++)
      for (int c = 1; c <= C; c++) {
        if (LtoR[r][c] <= maxEgde || TtoD[r][c] <= maxEgde)
          continue;
        int curEdge = Math.min(LtoR[r][c], TtoD[r][c]);
        for (int i = curEdge; i > maxEgde; i--) {
          int others = Math.min(LtoR[r - i + 1][c], TtoD[r][c - i + 1]);
          if (others >= i) {
            maxEgde = i;
            break;
          }
        }
      }
    return maxEgde * maxEgde;
  }

  //Google OA
  public int[] compareStrings(String A, String B) {
    String[] As = A.split(" "), Bs = B.split(" ");
    int N = As.length, M = Bs.length;
    int[] AFreq = new int[11], res = new int[M];
    for (int i = 0; i < N; i++) {
      int freq = getMinCount(As[i]);
      AFreq[freq]++;
    }
    for (int i = 0; i < 10; i++)
      AFreq[i + 1] += AFreq[i];
    for (int i = 0; i < M; i++) {
      int curMinCount = getMinCount(Bs[i]);
      int smallerNum = curMinCount == 0 ? 0 : AFreq[curMinCount - 1];
      res[i] = smallerNum;
    }
    return res;
  }

  private int getMinCount(String a) {
    int[] count = new int[26];
    for (char c : a.toCharArray())
      count[c - 'a']++;
    for (int i = 0; i < 26; i++)
      if (count[i] != 0)
        return count[i];
    return 0;
  }

  public int[] largestSubarrayLengthK1(int[] nums, int k) {
    if (nums.length == k)
      return nums;
    int N = nums.length, maxStart = 0;
    for (int i = 1; i <= N - k; i++)
      if (compareArray(nums, maxStart, i, k) < 0)
        maxStart = i;
    int[] res = new int[k];
    for (int i = 0; i < k; i++)
      res[i] = nums[maxStart++];
    return res;
  }

  private int compareArray(int[] nums, int start1, int start2, int k) {
    for (int i = 0; i < k; i++)
      if (nums[start1 + i] != nums[start2 + i])
        return nums[start1 + i] - nums[start2 + i];
    return 0;
  }

  public int[] largestSubarrayLengthK(int[] nums, int k) {
    if (nums.length == k)
      return nums;
    int N = nums.length, maxVal = 0, maxIdx = -1;
    for (int i = 0; i <= N - k; i++)
      if (nums[i] > maxVal) {
        maxVal = nums[i];
        maxIdx = i;
      }
    return Arrays.copyOfRange(nums, maxIdx, maxIdx + k);
  }

  public String maximumTime(String time) {
    char[] cs = time.toCharArray();
    if (cs[4] == '?')
      cs[4] = '9';
    if (cs[3] == '?')
      cs[3] = '5';
    if (cs[0] == '?' && cs[1] == '?') {
      cs[0] = '2';
      cs[1] = '3';
    } else if (cs[0] == '?')
      cs[0] = cs[1] > '3' ? '1' : '2';
    else if (cs[1] == '?')
      cs[1] = cs[0] == '2' ? '3' : '9';
    return new String(cs);
  }

  public int waterFlowers(int[] flowers, int cap1, int cap2) {
    int N = flowers.length, start = 0, end = N - 1, leftRemain = 0, rightRemain = 0, res = 0;
    while (start < end) {
      if (leftRemain < flowers[start]) {
        leftRemain = cap1;
        res++;
      }
      if (rightRemain < flowers[end]) {
        rightRemain = cap2;
        res++;
      }
      leftRemain -= flowers[start];
      rightRemain -= flowers[end];
      start++;
      end--;
    }
    if (start == end && (leftRemain + rightRemain < flowers[start]))
      res++;
    return res;
  }

  public int minDominoRotations2(int[] A, int[] B) {
    int targetA = MMRgetMinRotate(A, B, A[0]), targetB = MMRgetMinRotate(A, B, B[0]);
    if (targetA == -1 && targetB == -1)
      return -1;
    else if (targetA == -1)
      return targetB;
    else if (targetB == -1)
      return targetA;
    else
      return Math.min(targetA, targetB);
  }

  private int MMRgetMinRotate(int[] A, int[] B, int target) {
    int N = A.length, countA = 0, countB = 0;
    for (int i = 0; i < N; i++) {
      if (A[i] != target && B[i] != target)
        return -1;
      if (A[i] == target)
        countA++;
      if (B[i] == target)
        countB++;
    }
    return Math.min(N - countA, N - countB);
  }

  public int calculateTime(String keyboard, String word) {
    if (word == null || word.isEmpty())
      return 0;
    int[] keyIdx = new int[26];
    for (int i = 0; i < keyboard.length(); i++)
      keyIdx[keyboard.charAt(i) - 'a'] = i;
    int res = 0, curIdx = 0;
    for (char c : word.toCharArray()) {
      int nextIdx = keyIdx[c - 'a'];
      res += Math.abs(nextIdx - curIdx);
      curIdx = nextIdx;
    }
    return res;
  }

  public int maxLevelSum2(TreeNode root) {
    int maxLevel = 0, maxSum = Integer.MIN_VALUE, curLevel = 1;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      int size = queue.size(), curSum = 0;
      for (int i = 0; i < size; i++) {
        TreeNode cur = queue.poll();
        curSum += cur.val;
        if (cur.left != null)
          queue.offer(cur.left);
        if (cur.right != null)
          queue.offer(cur.right);
      }
      if (curSum > maxSum) {
        maxLevel = curLevel;
        maxSum = curSum;
      }
      curLevel++;
    }
    return maxLevel;
  }

  public String licenseKeyFormatting2(String S, int K) {
    char[] cs = S.toCharArray();
    int N = cs.length, count = 0;
    StringBuilder stringBuilder = new StringBuilder();
    for (int i = N - 1; i >= 0; i--) {
      if (cs[i] == '-')
        continue;
      if (cs[i] >= 'a' && cs[i] <= 'z')
        cs[i] += 'A' - 'a';
      stringBuilder.append(cs[i]);
      count = ++count % K;
      if (count == 0)
        stringBuilder.append('-');
    }
    if (stringBuilder.length() != 0 && stringBuilder.charAt(stringBuilder.length() - 1) == '-')
      stringBuilder.setLength(stringBuilder.length() - 1);
    return stringBuilder.reverse().toString();
  }

  public int numUniqueEmails(String[] emails) {
    Set<String> appeared = new HashSet<>();
    for (String email : emails)
      appeared.add(NUEsimplify(email));
    return appeared.size();
  }

  private String NUEsimplify(String email) {
    String[] parts = email.split("@");
    char[] local = parts[0].toCharArray();
    StringBuilder sb = new StringBuilder();
    int addIdx = parts[0].indexOf('+'), endLocal = addIdx == -1 ? local.length : addIdx;
    for (int i = 0; i < endLocal; i++)
      if (local[i] != '.')
        sb.append(local[i]);
    sb.append('@');
    sb.append(parts[1]);
    return sb.toString();
  }

  public int totalFruit(int[] tree) {
    if (tree.length <= 2)
      return tree.length;
    int N = tree.length, types = 0, max = 0;
    int[] count = new int[N];
    for (int start = 0, end = 0; end < N; end++) {
      count[tree[end]]++;
      if (count[tree[end]] == 1)
        types++;
      if (types == 3) {
        while (start < end && types != 2) {
          count[tree[start]]--;
          if (count[tree[start]] == 0)
            types--;
          start++;
        }
      }
      max = Math.max(max, end - start + 1);
    }
    return max;
  }

  int minDaysBloom(int[] roses, int k, int n) {
    if (roses == null || roses.length == 0)
      return 0;
    int len = roses.length, start = Integer.MAX_VALUE, end = Integer.MIN_VALUE;
    if (k > len || n > len)
      return 0;
    for (int r : roses) {
      start = Math.min(start, r);
      end = Math.max(end, r);
    }
    while (start <= end) {
      int mid = (start + end) >> 1;
      int bouquet = getBouquetNum(roses, k, mid);
      if (bouquet >= n)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return start;
  }

  private int getBouquetNum(int[] rose, int k, int days) {
    int count = 0, res = 0;
    for (int i = 0; i < rose.length; i++)
      if (rose[i] > days)
        count = 0;
      else {
        count = ++count % k;
        if (count == 0)
          res++;
      }
    return res;
  }

  int[][] fillMatrix(int n) {
    if (n < 0)
      return null;
    if (n == 0)
      return new int[0][0];
    else if (n == 1)
      return new int[][]{{1}};
    int max = n * n, sum = (1 + max) * max / 2, target = sum / n;
    boolean[] visited = new boolean[max + 1];
    int[] rowSum = new int[n], colSum = new int[n];
    int[][] res = new int[n][n];
    if (FMhelper(visited, 0, n, new int[1], new int[1], target, rowSum, colSum, res))
      return res;
    else
      return null;
  }

  private boolean FMhelper(boolean[] visited, int idx, int n, int[] diagSum1, int[] diagSum2, int target, int[] rowSum, int[] colSum, int[][] res) {
    if (idx == n * n)
      return true;
    int R = idx / n, C = idx % n, max = n * n;
    for (int i = 1; i <= max; i++) {
      if (visited[i])
        continue;
      if (target - rowSum[R] < i || target - colSum[C] < i || (R == C && target - diagSum1[0] < i) || (R + C == n - 1 && target - diagSum2[0] < i))
        break;
      if ((R == n - 1 && i + colSum[C] != target) || (C == n - 1 && i + rowSum[R] != target)
              || (R == n - 1 && C == 0 && diagSum2[0] + i != target)
              || (R == n - 1 && C == n - 1 && diagSum1[0] + i != target))
        continue;
      FMupdate(R, C, n, i, diagSum1, diagSum2, rowSum, colSum);
      visited[i] = true;
      res[R][C] = i;
      if (FMhelper(visited, idx + 1, n, diagSum1, diagSum2, target, rowSum, colSum, res))
        return true;
      visited[i] = false;
      FMupdate(R, C, n, -i, diagSum1, diagSum2, rowSum, colSum);
    }
    return false;
  }

  private void FMupdate(int R, int C, int n, int val, int[] diagSum1, int[] diagSum2, int[] rowSum, int[] colSum) {
    rowSum[R] += val;
    colSum[C] += val;
    if (R == C)
      diagSum1[0] += val;
    if (R + C == n - 1)
      diagSum2[0] += val;
  }

  public int decreaseSubsequence(int[] nums) {
    if (nums == null || nums.length == 0)
      return nums.length;
    List<Integer> res = new ArrayList<>();
    for (int n : nums) {
      if (res.isEmpty() || res.get(res.size() - 1) <= n)
        res.add(n);
      else
        DSinsert(res, n);
    }
    return res.size();
  }

  private void DSinsert(List<Integer> res, int val) {
    int start = 0, end = res.size() - 1;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (res.get(mid) < val)
        start = mid + 1;
      else
        end = mid - 1;
    }
    res.set(start, val);
  }

  class MDnode {
    int digitNum;
    MDnode zero, one;

    public MDnode(int digitNum) {
      this.digitNum = digitNum;
      zero = null;
      one = null;
    }
  }

  private void insertMDnode(MDnode root, String val) {
    int N = val.length();
    for (int i = 31; i >= 0; i--) {
      boolean isZero = N <= i || val.charAt(N - i - 1) == '0';
      if (isZero) {
        if (root.zero == null)
          root.zero = new MDnode(i);
        root = root.zero;
      } else {
        if (root.one == null)
          root.one = new MDnode(i);
        root = root.one;
      }
    }
  }

  private int findMinPrefix(MDnode root) {
    int prefix = 0;
    for (; prefix < 32; prefix++) {
      if (root.one != null && root.zero != null)
        return prefix;
      if (root.one != null)
        root = root.one;
      else
        root = root.zero;
    }
    return prefix;
  }

  public int maxDistance(String[] nums) {
    if (nums == null || nums.length == 0)
      return 0;
    MDnode root = new MDnode(32);
    for (String n : nums)
      insertMDnode(root, n);
    int minPrefix = findMinPrefix(root);
    return (32 - minPrefix) << 1;
  }

  class KCnode {
    int[] pos;
    int distance;

    public KCnode(int[] node) {
      pos = node;
      distance = pos[0] * pos[0] + pos[1] * pos[1];
    }
  }

  public int[][] kClosest(int[][] points, int K) {
    int N = points.length;
    KCnode[] data = new KCnode[N];
    for (int i = 0; i < N; i++)
      data[i] = new KCnode(points[i]);
    KCquickSelection(data, K - 1, 0, N - 1);
    int[][] res = new int[K][2];
    for (int i = 0; i < K; i++)
      res[i] = data[i].pos;
    return res;
  }

  private void KCquickSelection(KCnode[] data, int target, int start, int end) {
    if (start == end)
      return;
    int partition = KCpartition(data, start, end);
    if (partition < target)
      KCquickSelection(data, target, partition + 1, end);
    else if (partition > target)
      KCquickSelection(data, target, start, partition - 1);
  }

  private int KCpartition(KCnode[] data, int start, int end) {
    KCnode partition = data[start];
    int left = start, right = end + 1;
    while (true) {
      while (data[++left].distance < partition.distance)
        if (left == end)
          break;
      while (data[--right].distance >= partition.distance)
        if (right == start)
          break;
      if (left >= right)
        break;
      exchange(data, left, right);
    }
    exchange(data, start, right);
    return right;
  }

  private void exchange(Object[] data, int i, int j) {
    Object temp = data[i];
    data[i] = data[j];
    data[j] = temp;
  }

  public int[] storeAndHouse(int[] stores, int[] houses) {
    if (houses == null || houses.length == 0)
      return houses;
    int storeLen = stores.length, housesLen = houses.length;
    int[] res = new int[housesLen];
    Arrays.sort(stores);
    for (int i = 0; i < housesLen; i++) {
      int[] lessAndMore = SHfindLessAndMore(stores, houses[i]);
      int lessIdx = lessAndMore[0], moreIdx = lessAndMore[1];
      int lessDist = lessIdx == -1 ? Integer.MAX_VALUE : houses[i] - stores[lessIdx];
      int moreDist = moreIdx == storeLen ? Integer.MAX_VALUE : stores[moreIdx] - houses[i];
      res[i] = lessDist <= moreDist ? stores[lessIdx] : stores[moreIdx];
    }
    return res;
  }

  private int[] SHfindLessAndMore(int[] stores, int target) {
    int start = 0, end = stores.length - 1, mid;
    while (start <= end) {
      mid = (start + end) >> 1;
      if (stores[mid] >= target)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return new int[]{end, start};
  }

  public int minMeetingRooms2(int[][] intervals) {
    if (intervals == null || intervals.length == 0)
      return 0;
    if (intervals.length == 1)
      return 1;
    int ans = 0;
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0] - b[0];
      }
    });
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[1] - b[1];
      }
    });
    for (int[] itv : intervals) {
      if (!pq.isEmpty() && pq.peek()[1] <= itv[0])
        pq.poll();
      pq.offer(itv);
      ans = Math.max(ans, pq.size());
    }
    return ans;
  }

  public int oddEvenJumps2(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    int N = A.length, ans = 1;
    TreeMap<Integer, Integer> treeMap = new TreeMap<>();
    boolean[][] isGood = new boolean[2][N];
    isGood[0][N - 1] = isGood[1][N - 1] = true;
    treeMap.put(A[N - 1], N - 1);
    for (int i = N - 2; i >= 0; i--) {
      // moreIdx: smallest bigger; lessIdx: biggest smaller
      Map.Entry<Integer, Integer> moreEntry = treeMap.ceilingEntry(A[i]), lessEntry = treeMap.floorEntry(A[i]);
      isGood[0][i] = lessEntry != null && isGood[1][lessEntry.getValue()]; // even jump to biggest smaller one
      isGood[1][i] = moreEntry != null && isGood[0][moreEntry.getValue()]; // odd jump to smallest bigger one
      treeMap.put(A[i], i);
      if (isGood[1][i])
        ans++;
    }
    return ans;
  }

  public int minSwap(int[] A, int[] B) {
    int N = A.length, swap = 1, stay = 0;
    for (int i = 1; i < N; i++)
      if (A[i - 1] >= A[i] || B[i - 1] >= B[i]) {
        int temp = swap;
        swap = stay + 1;
        stay = temp;
      } else if (A[i - 1] >= B[i] || B[i - 1] >= A[i])
        swap++;
      else {
        int cur = Math.min(swap, stay);
        stay = cur;
        swap = cur + 1;
      }
    return Math.min(stay, swap);
  }

  public int maxStrawBerry(int[] nums, int limit) {
    if (nums == null || nums.length == 0)
      return 0;
    int N = nums.length;
    int[][] dp = new int[N][limit + 1];
    return MSBhelper(nums, limit, N - 1, dp);
  }

  private int MSBhelper(int[] nums, int remain, int idx, int[][] dp) {
    if (idx < 0)
      return 0;
    if (remain < nums[idx])
      return 0;
    if (idx == 0)
      return nums[idx];
    if (dp[idx][remain] > 0)
      return dp[idx][remain];
    int pick = MSBhelper(nums, remain - nums[idx], idx - 2, dp) + nums[idx];
    int notPick = MSBhelper(nums, remain, idx - 1, dp);
    dp[idx][remain] = Math.max(pick, notPick);
    return dp[idx][remain];
  }


  public List<Character> delaysProject(char[][] depend, char[] delay) {
    List<Character> impacted = new ArrayList<>();
    for (char d : delay)
      impacted.add(d);
    if (depend == null || depend.length == 0)
      return impacted;
    List<Character>[] graph = new List[26];
    for (char[] d : depend) {
      if (graph[d[1]] == null)
        graph[d[1]] = new ArrayList<>();
      graph[d[1]].add(d[0]);
    }
    Queue<Character> q = new LinkedList<>();
    for (char d : delay)
      q.offer(d);
    while (!q.isEmpty()) {
      char cur = q.poll();
      for (char next : graph[cur - 'A']) {
        impacted.add(next);
        q.offer(next);
      }
    }
    Collections.sort(impacted);
    return impacted;
  }

  public int standStudent(int[] A) {
    if (A == null || A.length == 0)
      return 0;
    if (A.length == 1)
      return 1;
    List<Integer> rowStart = new ArrayList<>();
    int res = 0;
    for (int a : A) {
      if (rowStart.isEmpty() || rowStart.get(rowStart.size() - 1) <= a)
        rowStart.add(a);
      else {
        int subIdx = SSfindLeastMore(rowStart, a);
        rowStart.set(subIdx, a);
      }
      res = Math.max(res, rowStart.size());
    }
    return res;
  }

  private int SSfindLeastMore(List<Integer> rowStart, int a) {
    int start = 0, end = rowStart.size() - 1;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (rowStart.get(mid) <= a)
        start = mid + 1;
      else
        end = mid - 1;
    }
    return start;
  }

  public static int findMinDiff(Integer[] A) {
    if (A == null || A.length == 0)
      return 0;
    if (A.length == 1)
      return A[0];
    Arrays.sort(A);
    int[] res = new int[]{Integer.MAX_VALUE};
    int sum = 0;
    for (int a : A)
      sum += a;
    FMDhelper(A, A.length - 1, 0, sum, res);
    return res[0];
  }

  private static void FMDhelper(Integer[] A, int idx, int curSum, int target, int[] res) {
    if (idx < 0)
      return;
    res[0] = Math.min(res[0], Math.abs(target - (curSum << 1)));
    if (curSum >= target)
      return;
    for (int i = idx; i >= 0; i--)
      FMDhelper(A, i, curSum + A[idx], target, res);
  }

  public String maxBookedHotel(String[] orders) {
    if (orders.length == 1)
      return orders[0].substring(1);
    int[][] count = new int[10][26];
    int max = 0;
    String maxRoom = null;
    for (String order : orders) {
      if (order.charAt(0) == '-')
        continue;
      char[] cs = order.toCharArray();
      count[cs[1] - '0'][cs[2] - 'A']++;
      int curNum = count[cs[1] - '0'][cs[2] - 'A'];
      if (curNum > max) {
        max = curNum;
        maxRoom = order;
      } else if (curNum == max)
        maxRoom = maxRoom.compareTo(order) > 0 ? order : maxRoom;
    }
    return maxRoom.substring(1);
  }

  public int removeBalloon(String str) {
    if (str == null || str.isEmpty())
      return 0;
    int[] balloon = new int[26], count = new int[26];
    char[] cs = str.toCharArray(), blCount = "BALLOON".toCharArray();
    for (char c : blCount)
      balloon[c - 'A']++;
    for (char c : cs)
      count[c - 'A']++;
    int min = Integer.MAX_VALUE;
    for (char c : blCount)
      min = Math.min(min, count[c - 'A'] / balloon[c - 'A']);
    return min;
  }

  public String noThree(String str) {
    if (str == null || str.length() < 3)
      return str;
    char[] cs = str.toCharArray();
    char last = '0';
    int count = 0;
    StringBuilder sb = new StringBuilder();
    for (char c : cs) {
      if (count == 2 && last == c)
        continue;
      if (count == 0 || last != c) {
        last = c;
        count = 1;
      } else if (count < 3)
        count++;
      sb.append(c);
    }
    return sb.toString();
  }

  public int gamble(int N, int k) {
    if (k == 0 || N <= 3)
      return N - 1;
    int round = 0;
    while (N > 3 && k > 0) {
      if ((N & 1) == 1)
        N--;
      else {
        N >>= 1;
        k--;
      }
      round++;
    }
    round += N - 1;
    return round;
  }

  public int maximumString(String s) {
    if (s == null || s.length() == 0)
      return 2;
    char[] cs = s.toCharArray();
    int res = 0, N = cs.length, count = 0;
    if (cs[N - 1] != 'a')
      res += 2;
    for (char c : cs)
      if (c == 'a') {
        if (count == 2)
          return -1;
        count++;
      } else {
        res += 2 - count;
        count = 0;
      }
    if (count != 0)
      res += count;
    return res;
  }

  class LSnode implements Comparable<LSnode> {
    char val;
    int count;

    public LSnode(char val, int count) {
      this.val = val;
      this.count = count;
    }

    @Override
    public int compareTo(LSnode a) {
      return a.count - count;
    }
  }

  public String longestABCString(int a, int b, int c) {
    if (a == 0 && b == 0 && c == 0)
      return "";
    PriorityQueue<LSnode> pq = new PriorityQueue<>();
    if (a != 0)
      pq.offer(new LSnode('a', a));
    if (b != 0)
      pq.offer(new LSnode('b', b));
    if (c != 0)
      pq.offer(new LSnode('c', c));
    LSnode cache = null;
    StringBuilder sb = new StringBuilder();
    while (!pq.isEmpty()) {
      LSnode cur = pq.poll();
      if (cache != null && cache.count != 0)
        pq.offer(cache);
      sb.append(cur.val);
      cur.count--;
      if (pq.isEmpty()) {
        if (cur.count > 0)
          sb.append(cur.val);
        break;
      } else {
        LSnode next = pq.peek();
        if (cur.count >= next.count) {
          sb.append(cur.val);
          cur.count--;
        }
        cache = cur;
      }
    }
    return sb.toString();
  }

  public int magicSquare(int[][] A) {
    if (A == null || A.length == 0 || A[0].length == 0)
      return 0;
    int R = A.length, C = A[0].length, max = 1;
    int[][] rowSum = new int[R + 1][C + 1], colSum = new int[R + 1][C + 1];
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++) {
        rowSum[r + 1][c + 1] = rowSum[r][c + 1] + A[r][c];
        colSum[r + 1][c + 1] = colSum[r + 1][c] + A[r][c];
      }
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        for (int size = max + 1; r + size <= R && c + size <= C; size++)
          if (isMagicSquare(A, r, c, size, rowSum, colSum))
            max = size;
    return max;
  }

  private boolean isMagicSquare(int[][] A, int r, int c, int size, int[][] rowSum, int[][] colSum) {
    int target = rowSum[r + size][c + 1] - rowSum[r][c + 1], posDia = 0, negDia = 0;
    for (int s = 0; s < size; s++) {
      if (rowSum[r + size][c + s + 1] - rowSum[r][c + s + 1] != target
              || colSum[r + s + 1][c + size] - colSum[r + s + 1][c] != target)
        return false;
      posDia += A[r + s][c + s];
      negDia += A[r + s][c + size - s - 1];
    }
    if (posDia != target || negDia != target)
      return false;
    return true;
  }

  public ListNode insertionSortList2(ListNode head) {
    if (head == null)
      return head;
    ListNode fakeHead = new ListNode(Integer.MIN_VALUE);
    while (head != null) {
      ListNode next = head.next, cur = fakeHead;
      while (cur.next != null && cur.next.val < head.val)
        cur = cur.next;
      head.next = cur.next;
      cur.next = head;
      head = next;
    }
    return fakeHead.next;
  }

  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null)
      return null;
    else if (root.val > p.val && root.val > q.val)
      return lowestCommonAncestor(root.left, p, q);
    else if (root.val < p.val && root.val < q.val)
      return lowestCommonAncestor(root.right, p, q);
    else
      return root;
  }

  public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> begin = new HashSet<>(), end = new HashSet<>(), path = new HashSet<>(wordList);
    if (!path.contains(endWord))
      return 0;
    path.remove(endWord);
    begin.add(beginWord);
    end.add(endWord);
    int steps = 1;
    while (!begin.isEmpty() && !end.isEmpty()) {
      if (end.size() < begin.size()) {
        Set<String> temp = begin;
        begin = end;
        end = temp;
      }
      steps++;
      Set<String> next = new HashSet<>();
      for (String s : begin) {
        char[] cs = s.toCharArray();
        for (int i = 0; i < cs.length; i++) {
          char prev = cs[i];
          for (char j = 'a'; j <= 'z'; j++) {
            cs[i] = j;
            String cur = new String(cs);
            if (end.contains(cur))
              return steps;
            if (path.contains(cur)) {
              path.remove(cur);
              next.add(cur);
            }
          }
          cs[i] = prev;
        }
      }
      begin = next;
    }
    return 0;
  }

  public boolean areSentencesSimilarTwo(String[] words1, String[] words2, List<List<String>> pairs) {
    if (words1.length != words2.length)
      return false;
    Map<String, String> ids = new HashMap<>();
    Map<String, Integer> weight = new HashMap<>();
    for (List<String> p : pairs) {
      for (int i = 0; i < 2; i++) {
        String cur = p.get(i);
        if (!ids.containsKey(cur)) {
          ids.put(cur, cur);
          weight.put(cur, 1);
        }
      }
      SSunion(ids,p.get(0),p.get(1),weight);
    }
    for (int i=0;i<words1.length;i++){
      if (words1[i].equals(words2[i]))
        continue;
      String id1 = SSfind(words1[i],ids),id2 = SSfind(words2[i],ids);
      if (id1==null || id2==null || !id1.equals(id2))
        return false;
    }
    return true;
  }

  private String SSfind(String s, Map<String, String> id) {
    if (!id.containsKey(s))
      return null;
    while (!id.get(s).equals(s)) {
      String grand = id.get(id.get(s));
      id.put(s, grand);
      s = grand;
    }
    return s;
  }

  private void SSunion(Map<String, String> id, String i, String j, Map<String, Integer> weight) {
    String idI = SSfind(i, id), idJ = SSfind(j, id);
    if (idI.equals(idJ))
      return;
    int weightI = weight.get(idI), weightJ = weight.get(idJ);
    if (weightI >= weightJ) {
      weight.put(idI, weightI + weightJ);
      id.put(idJ, idI);
    } else {
      weight.put(idJ, weightI + weightJ);
      id.put(idI, idJ);
    }
  }

  public List<List<String>> partition1(String s) {
    List<List<String>> ans = new ArrayList<>();
    if (s==null || s.isEmpty())
      return ans;
    int N = s.length();
    boolean[][] isPal = new boolean[N][N];
    Phelper1(s.toCharArray(),s,0,isPal,new ArrayList<>(),ans);
    return ans;
  }

  private void Phelper1(char[] cs,String s,int idx,boolean[][] isPal,List<String> path,List<List<String>> res){
    if (idx==cs.length){
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i=idx;i<cs.length;i++){
      if (cs[i]!=cs[idx])
        continue;
      if (i>idx+1 && !isPal[idx+1][i-1])
        continue;
      isPal[idx][i] = true;
      path.add(s.substring(idx,i+1));
      Phelper1(cs,s,i+1,isPal,path,res);
      path.remove(path.size()-1);
    }
  }

  public ListNode mergeKLists1(ListNode[] lists) {
    if (lists==null || lists.length==0)
      return null;
    ListNode ans = MKLhelper(lists,0,lists.length-1);
    return ans;
  }

  private ListNode MKLhelper(ListNode[] lists,int start,int end){
    if (start==end)
      return lists[start];
    int mid = (start+end)>>1;
    ListNode left = MKLhelper(lists,start,mid);
    ListNode right = MKLhelper(lists,mid+1,end);
    ListNode res = MKLmerge1(left,right);
    return res;
  }

  private ListNode MKLmerge1(ListNode list1,ListNode list2){
    ListNode res = new ListNode(0),cur = res;
    while (list1!=null || list2!=null){
      if (list1==null){
        cur.next = list2;
        list2=list2.next;
      }
      else if (list2==null){
        cur.next = list1;
        list1 = list1.next;
      }
      else if (list1.val<=list2.val){
        cur.next = list1;
        list1 = list1.next;
      }
      else{
        cur.next = list2;
        list2=list2.next;
      }
      cur = cur.next;
    }
    return res.next;
  }

  public int cherryPickup2(int[][] grid) {
    if (grid == null || grid.length==0 || grid[0].length==0)
      return 0;
    int R = grid.length,C = grid[0].length;
    int[][][] memo = new int[R][C][R];
    for (int r=0;r<R;r++)
      for (int c=0;c<C;c++)
        Arrays.fill(memo[r][c],Integer.MIN_VALUE);
    return Math.max(0,CPhelper(grid,R-1,C-1,R-1,memo));
  }

  private int CPhelper(int[][] grid,int r1,int c1,int r2,int[][][] memo){
    int c2 = r1+c1-r2;
    if (r1<0 || c1<0 || r2<0 || c2<0)
      return -1;
    if (grid[r1][c1]==-1 || grid[r2][c2]==-1)
      return -1;
    if (r1==0 && c1==0)
      return grid[r1][c1];
    if (memo[r1][c1][r2]!=Integer.MIN_VALUE)
      return memo[r1][c1][r2];
    memo[r1][c1][r2] = Math.max(Math.max(CPhelper(grid,r1-1,c1,r2-1,memo),CPhelper(grid,r1-1,c1,r2,memo))
            ,Math.max(CPhelper(grid,r1,c1-1,r2-1,memo),CPhelper(grid,r1,c1-1,r2,memo)));
    if (memo[r1][c1][r2]>=0){
      memo[r1][c1][r2]+= grid[r1][c1];
      if (r1!=r2)
        memo[r1][c1][r2]+=grid[r2][c2];
    }
    return memo[r1][c1][r2];
  }

  public boolean isPossible2(int[] nums) {
    if (nums==null || nums.length<3)
      return false;
    int N = nums.length,prev = Integer.MIN_VALUE,cur=0,p1=0,p2=0,p3=0,cnt=0,c1=0,c2=0,c3=0;
    for (int i=0;i<N;p1=c1,p2=c2,p3=c3,prev=cur) {
      for (cnt=0,cur = nums[i];i<N && nums[i]==cur;i++,cnt++);
      if (prev!=cur-1){
        if (p1!=0 || p2!=0)
          return false;
        c1=cnt;
        c2=0;
        c3=0;
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

  public boolean containsNearbyAlmostDuplicate2(int[] nums, int k, int t) {
    if (nums==null || nums.length==0 || k<1 || t<0)
      return false;
    Map<Long,Long> record = new HashMap<>();
    long bucketSize = (long)t+1;
    int N = nums.length;
    for (int i=0;i<N;i++){
      long newVal = (long)nums[i]-Integer.MIN_VALUE;
      long bucketId = newVal/bucketSize;
      Long sm=null,bg=null;
      if (record.containsKey(bucketId)
              || ((sm=record.get(bucketId-1))!=null && newVal-sm<=t )
              || ((bg = record.get(bucketId+1))!=null && bg-newVal<=t))
        return true;
      if (record.size()>=k){
        long oldId = ((long)nums[i-k]-Integer.MIN_VALUE)/bucketSize;
        record.remove(oldId);
      }
      record.put(bucketId,newVal);
    }
    return false;
  }

  class SRAnode implements Comparable<SRAnode>{
    int val,id,groupId;
    public SRAnode(int val,int id,int groupId){
      this.val =val;
      this.id = id;
      this.groupId = groupId;
    }

    @Override
    public int compareTo(SRAnode o) {
      return val-o.val;
    }
  }

  public int[] smallestRange1(List<List<Integer>> nums) {
    int[] res=null;
    if (nums==null || nums.isEmpty())
      return res;
    int N = nums.size(),max = Integer.MIN_VALUE,resMin,resMax;
    PriorityQueue<SRAnode> pq = new PriorityQueue<>();
    for (int i=0;i<N;i++){
      SRAnode temp = new SRAnode(nums.get(i).get(0),0,i);
      pq.offer(temp);
      max = Math.max(max,temp.val);
    }
    resMax = max;
    resMin = pq.peek().val;
    while (pq.size()==N){
      SRAnode cur = pq.poll();
      if (max-cur.val<resMax-resMin){
        resMax = max;
        resMin = cur.val;
      }
      if (nums.get(cur.groupId).size()-1>cur.id){
        SRAnode next = new SRAnode(nums.get(cur.groupId).get(cur.id+1),cur.id+1,cur.groupId);
        max = Math.max(max,next.val);
        pq.offer(next);
      }
    }
    res = new int[]{resMin,resMax};
    return res;
  }


  class KCLnode{
    int[] data;
    int val;
    public KCLnode(int[] data){
      this.data = data;
      val = data[0]*data[0]+data[1]*data[1];
    }
  }

  public int[][] kClosest1(int[][] points, int K) {
    if (points==null || points.length==0 || K>=points.length)
      return points;
    int N = points.length;
    KCLnode[] data = new KCLnode[N];
    for (int i=0;i<N;i++)
      data[i] = new KCLnode(points[i]);
    KCLquickSelection(data,0,N-1,K-1);
    int[][] res = new int[K][2];
    for (int i=0;i<K;i++)
      res[i] = data[i].data;
    return res;
  }

  private void KCLquickSelection(KCLnode[] data, int start,int end,int target){
    if (start>=end)
      return;
    int partitionIdx = KCLpartition(data,start,end);
    if (partitionIdx<target)
      KCLquickSelection(data,partitionIdx+1,end,target);
    else if (partitionIdx>target)
      KCLquickSelection(data,start,partitionIdx-1,target);
  }

  private int KCLpartition(KCLnode[] data,int start,int end){
    KCLnode partition = data[start];
    int left = start,right = end+1;
    while (true){
      while (data[++left].val<partition.val)
        if (left==end)
          break;
      while (data[--right].val>=partition.val)
        if (right==start)
          break;
      if (left>=right)
        break;
      exchange(data,left,right);
    }
    exchange(data,start,right);
    return right;
  }

  public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums1==null || nums1.length==0 || nums2==null || nums2.length==0)
      return res;
    int N1 = nums1.length,N2 = nums2.length;
    if (N1*N2<=k){
      for (int n1:nums1)
        for (int n2:nums2){
          List<Integer> temp = new ArrayList<>();
          temp.add(n1);
          temp.add(n2);
          res.add(temp);
        }
      return res;
    }
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return a[0]+a[1]-b[0]-b[1];
      }
    });
    for (int n1:nums1)
      pq.offer(new int[]{n1,nums2[0],0});
    while (k-->0){
      int[] cur = pq.poll();
      if (cur[2]<N2-1)
        pq.offer(new int[]{cur[0],nums2[cur[2]+1],cur[2]+1});
      List<Integer> temp = new ArrayList<>();
      temp.add(cur[0]);
      temp.add(cur[1]);
      res.add(temp);
    }
    return res;
  }

  public boolean isMatch6(String s, String p) {
    if ((s==null && p==null) || (s.isEmpty() && p.isEmpty()))
      return true;
    char[] cs = s.toCharArray(),cp = p.toCharArray();
    int sLen = cs.length,pLen = cp.length;
    boolean[][] dp = new boolean[sLen+1][pLen+1];
    dp[0][0] = true;
    for (int i=0;i<pLen;i++){
      if (cp[i]!='*')
        break;
      dp[0][i+1]=true;
    }
    for (int i=0;i<sLen;i++)
      for (int j=0;j<pLen;j++)
        if (cp[j]=='?' || cp[j]==cs[i])
          dp[i+1][j+1] = dp[i][j];
        else if (cp[j]=='*')
          dp[i+1][j+1] = dp[i+1][j]||dp[i][j+1];
    return dp[sLen][pLen];
  }

  public String crackSafe1(int n, int k) {
    StringBuilder sb = new StringBuilder();
    if (n==1){
      for (int i=0;i<k;i++)
        sb.append(i);
      return sb.toString();
    }
    if (k==1){
      char[] res = new char[n];
      Arrays.fill(res,'0');
      return new String(res);
    }
    Set<String> visited = new HashSet<>();
    int targetNum = (int)Math.pow(k,n);
    String start = String.join("",Collections.nCopies(n,"0"));
    sb.append(start);
    visited.add(start);
    CShelper(sb,visited,targetNum,n,k);
    return sb.toString();
  }

  private boolean CShelper(StringBuilder sb,Set<String> visited,int targetNum,int n,int k){
    if (visited.size()==targetNum)
      return true;
    String base = sb.substring(sb.length()-n+1);
    for (char i='0';i<'0'+k;i++){
      String cur = base+i;
      if (visited.contains(cur))
        continue;
      visited.add(cur);
      sb.append(i);
      if (CShelper(sb,visited,targetNum,n,k))
        return true;
      visited.remove(cur);
      sb.deleteCharAt(sb.length()-1);
    }
    return false;
  }

  public String crackSafe(int n, int k) {
    if (n == 1 && k == 1) return "0";
    Set<String> visited = new HashSet<>();
    StringBuilder sb  = new StringBuilder();
    String start = String.join("",Collections.nCopies(n-1,"0"));
    crackHelper(start,k,visited,sb);
    sb.append(start);
    return sb.toString();
  }

  private void crackHelper(String cur,int k,Set<String> visited,StringBuilder sb){
    for (int i=0;i<k;i++){
      String temp = cur+i;
      if (!visited.contains(temp)){
        visited.add(temp);
        crackHelper(temp.substring(1),k,visited,sb);
        sb.append(i);
      }
    }
  }

  public int triangleNumber(int[] nums) {
    if (nums==null || nums.length<3)
      return 0;
    Arrays.sort(nums);
    int N = nums.length,ans = 0;
    for (int i=N-1;i>=2;i--){
      int left = 0,right = i-1;
      while (left<right)
        if (nums[left]+nums[right]>nums[i]){
          ans+=right-left;
          right--;
        }
        else
          left++;
    }
    return ans;
  }

  public int hIndex2(int[] citations) {
    if (citations==null || citations.length==0)
      return 0;
    int N = citations.length;
    Arrays.sort(citations);
    if (citations[N-1]==0)
      return 0;
    for (int i=N-1;i>=0;i--){
      int count = N-i;
      if (count<=citations[i] && (i==0 || citations[i-1]<=count))
        return count;
    }
    return N;
  }

  public int repeatedStringMatch(String A, String B) {
    StringBuilder sb = new StringBuilder();
    int count = 0;
    while (sb.length()<B.length()){
      count++;
      sb.append(A);
    }
    if (sb.toString().contains(B))
      return count;
    if (sb.append(A).toString().contains(B))
      return ++count;
    return -1;
  }

  public int countRangeSum1(int[] nums, int lower, int upper) {
    if (nums == null )
      return 0;
    int N = nums.length;
    long[] data = new long[N+1];
    for (int i=0;i<N;i++)
      data[i+1] = data[i]+nums[i];
    int res = CRSmergeSort1(data,new long[N+1],0,N,lower,upper);
    return res;
  }

  private int CRSmergeSort1(long[] nums,long[] aux,int start,int end,int lower,int upper){
    if (start>=end)
      return 0;
    int mid = (start+end)>>1;
    int left = CRSmergeSort1(nums,aux,start,mid,lower,upper);
    int right = CRSmergeSort1(nums,aux,mid+1,end,lower,upper);
    int cur = CRSmerge1(nums,aux,start,end,lower,upper);
    return left+right+cur;
  }

  private int CRSmerge1(long[] nums,long[] aux,int start,int end,int lower,int upper){
    for (int i=start;i<=end;i++)
      aux[i] = nums[i];
    int mid = (start+end)>>1,count=0,sm=mid+1,bg=mid+1,half=mid+1,idx=start;
    for (int i=start;i<=mid;i++){
      while (sm<=end && aux[sm]-aux[i]<lower)
        sm++;
      while (bg<=end && aux[bg]-aux[i]<=upper)
        bg++;
      count+=bg-sm;
      while (half<=end && aux[half]<=aux[i])
        nums[idx++] = aux[half++];
      nums[idx++] = aux[i];
    }
    return count;
  }

  public boolean isUgly(int num) {
    if (num<=0)
      return false;
    while ((num&1)==0)
      num>>=1;
    while (num%3==0)
      num/=3;
    while (num%5==0)
      num/=5;
    return num==1;
  }

  public String reorganizeString(String S) {
    if (S==null||S.length()<2)
      return S;
    char[] cs = S.toCharArray();
    int[] count = new int[26];
    int N = cs.length,freq = 0,freqIdx=0;
    for (char c:cs){
      count[c-'a']++;
      if (count[c-'a']>freq){
        freq=count[c-'a'];
        freqIdx = c-'a';
      }
    }
    if (freq-1>N-freq)
      return "";
    char[] res = new char[N];
    int idx = 0;
    while (count[freqIdx]>0){
      res[idx]=(char)(freqIdx+'a');
      count[freqIdx]--;
      idx+=2;
    }
    for (int i=0;i<26;i++)
      while (count[i]>0){
        if (idx>=N)
          idx = 1;
        res[idx]=(char)(i+'a');
        idx+=2;
        count[i]--;
      }
    return new String(res);
  }

  class MedianFinder1 {

    PriorityQueue<Integer> bigger,smaller;
    int idx;
    /** initialize your data structure here. */
    public MedianFinder1() {
      idx=0;
      smaller = new PriorityQueue<>(Collections.reverseOrder());
      bigger = new PriorityQueue<>();
    }

    public void addNum(int num) {
      idx++;
      if (smaller.isEmpty())
        smaller.add(num);
      else if (num<=smaller.peek())
        smaller.add(num);
      else
        bigger.add(num);
      ajust();
    }

    private void ajust(){
      int diff = smaller.size()-bigger.size();
      if (diff==0 ||diff==1)
        return;
      if (diff==2)
        bigger.offer(smaller.poll());

      else
        smaller.offer(bigger.poll());
    }

    public double findMedian() {
      if (idx==0)
        throw new IllegalArgumentException();
      if ((idx&1)==1)
        return smaller.peek();
      else
        return ((double)bigger.peek()+(double)smaller.peek())/2;
    }
  }

  public int kthSmallest2(int[][] matrix, int k) {
    int R = matrix.length,C = matrix[0].length;
    if (k==1)
      return matrix[0][0];
    boolean[][] visited = new boolean[R][C];
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
      @Override
      public int compare(int[] a, int[] b) {
        return matrix[a[0]][a[1]]-matrix[b[0]][b[1]];
      }
    });
    pq.offer(new int[]{0,0});
    visited[0][0] = true;
    while (k>1){
      k--;
      int[] cur = pq.poll();
      if (cur[0]<R-1 && !visited[cur[0]+1][cur[1]]){
        visited[cur[0]+1][cur[1]]=true;
        pq.offer(new int[]{cur[0]+1,cur[1]});
      }
      if (cur[1]<C-1 && !visited[cur[0]][cur[1]+1]){
        visited[cur[0]][cur[1]+1]=true;
        pq.offer(new int[]{cur[0],cur[1]+1});
      }
    }
    int[] ans = pq.peek();
    return matrix[ans[0]][ans[1]];
  }

  public int longestPalindromeSubseq2D(String s) {
    if (s==null||s.isEmpty())
      return 0;
    char[] cs = s.toCharArray();
    int N = cs.length;
    int[][] dp = new int[N][N];
    for (int i=0;i<N;i++){
      dp[i][i]=1;
      for (int j=i-1;j>=0;j--)
        if (cs[j]==cs[i])
          dp[j][i] = 2+dp[j+1][i-1];
        else
          dp[j][i] = Math.max(dp[j+1][i],dp[j][i-1]);
    }
    return dp[0][N-1];
  }

  public int longestPalindromeSubseq1D(String s) {
    if (s == null || s.isEmpty())
      return 0;
    char[] cs = s.toCharArray();
    int N = cs.length,max;
    int[] dp = new int[N];
    for (int i=0;i<N;i++){
      dp[i]=1;
      max=0;
      for (int j=i-1;j>=0;j--){
        int val;
        if (cs[j]==cs[i])
          val = 2+max;
        else
          val = Math.max(dp[j],dp[j+1]);
        max = Math.max(max,dp[j]);
        dp[j] = val;
      }
    }
    return dp[0];
  }

  public int minimumDeleteSum(String s1, String s2) {
    if (s1==null || s1==null)
      return 0;
    char[] cs1 = s1.toCharArray(),cs2 = s2.toCharArray();
    int sum1 = getAsciiSum(cs1),sum2 = getAsciiSum(cs2),lcs = MDLCS(cs1,cs2);
    return sum1+sum2-(lcs<<1);
  }

  private int getAsciiSum(char[] cs){
    int sum = 0;
    for (char c:cs)
      sum+=c;
    return sum;
  }

  private int MDLCS(char[] cs1,char[] cs2){
    if (cs1.length==0 || cs2.length==0)
      return 0;
    int N1 = cs1.length,N2 = cs2.length;
    int[][] dp = new int[N1+1][N2+1];
    for (int i=0;i<N1;i++)
      for (int j=0;j<N2;j++)
        if (cs1[i]==cs2[j])
          dp[i+1][j+1] = dp[i][j]+cs1[i];
        else
          dp[i+1][j+1] = Math.max(dp[i][j+1],dp[i+1][j]);
    return dp[N1][N2];
  }

  class LFUnode{
    int key,val,freq;
    public LFUnode(int key,int val){
      this.key = key;
      this.val = val;
      freq=0;
    }
  }

  class LFUCache {

    private Map<Integer,LFUnode> record;

    //freq -- < key1,key2...>
    private Map<Integer,LinkedHashSet<Integer>> freqRecorder;
    private int capacity,minFreq;

    public LFUCache(int capacity) {
      if (capacity<0)
        throw new IllegalArgumentException();
      this.capacity = capacity;
      record = new HashMap<>();
      freqRecorder = new HashMap<>();
      freqRecorder.put(1,new LinkedHashSet<>());
      minFreq = 1;
    }

    public int get(int key) {
      //1. if not exist, -1
      //2. get val; update: pick from old time, add to new time, if no new time, create one
      LFUnode cur;
      if (capacity==0 || (cur=record.get(key))==null)
        return -1;
      update(cur);
      return cur.val;
    }

    public void put(int key, int value) {
      if (capacity==0)
        return;
      //key exist,record update,then update()
      //key not exist,if full,record remove,freq remove, update minFreq; if not full, add record,add freqRecord
      LFUnode data = record.get(key);
      if (data!=null){
        data.val = value;
        update(data);
      }
      else{
        if (record.size()==capacity){
          int evictKey = freqRecorder.get(minFreq).iterator().next();
          record.remove(evictKey);
          freqRecorder.get(minFreq).remove(evictKey);
          minFreq = 1;
        }
        data = new LFUnode(key,value);
        record.put(key,data);
        addFreq(data);
      }
    }

    private void update(LFUnode data){
      freqRecorder.get(data.freq).remove(data.key);
      if (data.freq==minFreq && freqRecorder.get(data.freq).isEmpty())
        minFreq++;
      addFreq(data);
    }

    private void addFreq(LFUnode data){
      LinkedHashSet<Integer> nextFreq;
      data.freq++;
      if (data.freq<minFreq)
        minFreq = data.freq;
      if ((nextFreq=freqRecorder.get(data.freq))==null){
        nextFreq = new LinkedHashSet<>();
        freqRecorder.put(data.freq,nextFreq);
      }
      nextFreq.add(data.key);
    }
  }

  public ListNode detectCycle1(ListNode head) {
    if (head == null)
      return null;
    ListNode fast = head.next,slow = head;
    while (fast!=null && slow!=fast){
      fast = fast.next;
      if (fast==null)
        break;
      fast = fast.next;
      slow = slow.next;
    }
    if (fast==null)
      return null;
    slow = head;
    fast = fast.next;
    while (slow!=fast){
      slow = slow.next;
      fast = fast.next;
    }
    return slow;
  }

  public void recoverTree1(TreeNode root) {
    if (root==null)
      return;
    TreeNode prev = new TreeNode(Integer.MIN_VALUE),first=null,second = null;
    while (root!=null)
      if (root.left==null){
        if (prev.val>root.val){
          if (first==null)
            first = prev;
          if (first!=null)
            second = root;
        }
        prev = root;
        root = root.right;
      }
      else{
        TreeNode before = root.left;
        while (before.right!=null && before.right!=root)
          before = before.right;
        if (before.right==null){
          before.right = root;
          root = root.left;
        }
        else{
          if (prev.val>root.val){
            if (first==null)
              first = prev;
            if (first!=null)
              second = root;
          }
          prev = root;
          before.right=null;
          root = root.right;
        }
      }
    int temp = first.val;
    first.val = second.val;
    second.val = temp;
  }

  public boolean reorderedPowerOf2(int N) {
    int key = digitRecorder(N),mask = 1;
    for (int i=0;i<32;i++){
      if (digitRecorder(mask)==key)
        return true;
      mask<<=1;
    }
    return false;
  }

  private int digitRecorder(int N){
    int res = 0;
    while (N>0){
      res+=Math.pow(10,N%10);
      N/=10;
    }
    return res;
  }

  public int maxArea1(int[] height) {
    if (height==null || height.length<2)
      return 0;
    int N = height.length,start = 0,end = N-1,max = Integer.MIN_VALUE;
    while (start<end){
      max = Math.max(max,(end-start)*Math.min(height[start],height[end]));
      if (height[end]<=height[start])
        end--;
      else
        start++;
    }
    return max;
  }

  public int twitter_maxHeight(int[] tablePos,int[] tableHeight){
    if (tableHeight==null || tablePos==null)
      return 0;
    int N = tablePos.length,max = 0;
    for (int i=1;i<N;i++)
      max = Math.max(max,twitter_maxBetweenTable(tablePos[i-1],tableHeight[i-1],tablePos[i],tableHeight[i]));
    return max;
  }

  private int twitter_maxBetweenTable(int startPos,int startHeight,int endPos,int endHeight){
    if (startPos==endPos-1)
      return 0;
    int dist = endPos-startPos-1,min = Math.min(startHeight,endHeight),max = Math.max(startHeight,endHeight);
    if (min+dist<=max+1)
      return max+1;
    else{
      dist-= max-min;
      return dist>>1;
    }
  }

  public int twitter_maximumTotalWeight(int time,int[] tasks,int[] weight){
    if (time<=0 || tasks == null || tasks.length==0 || weight==null || weight.length==0)
      return 0;
    int N = tasks.length;
    int[][] dp = new int[N][time+1];
    int res = twitter_mtwHelper(0,time,tasks,weight,dp);
    return res;
  }

  private int twitter_mtwHelper(int idx,int time,int[] tasks,int[] weight,int[][] memo){
    int realTime;
    if (idx>=tasks.length || (realTime = tasks[idx]<<1 )>time )
      return 0;
    if (memo[idx][time]!=0)
      return memo[idx][time];
    memo[idx][time] = Math.max(twitter_mtwHelper(idx+1,time,tasks,weight,memo),
            weight[idx]+twitter_mtwHelper(idx+1,time-realTime,tasks,weight,memo));
    return memo[idx][time];
  }

  public int totalNQueens1(int n) {
    if (n<=0)
      return 0;
    if (n==1)
      return 1;
    int[] res = new int[1];
    boolean[] col = new boolean[n],posSlash = new boolean[n<<1],negSlash = new boolean[n<<1];
    TNQhelper1(n-1,col,posSlash,negSlash,res);
    return res[0];
  }

  private void TNQhelper1(int r,boolean[] col,boolean[] posSlash,boolean[] negSlash,int[] res){
    if (r<0){
      res[0]++;
      return;
    }
    int n = col.length;
    for (int c=0;c<n;c++){
      int ps = r-c+n,ng = r+c;
      if (col[c] || posSlash[ps]|| negSlash[ng])
        continue;
      col[c] = true;
      posSlash[ps] = true;
      negSlash[ng] = true;
      TNQhelper1(r-1,col,posSlash,negSlash,res);
      col[c] = false;
      posSlash[ps] = false;
      negSlash[ng] = false;
    }
  }

  public int minIncrementForUnique(int[] A) {
    if (A.length==0)
      return 0;
    int[] count = new int[40000];
    for (int a:A)
      count[a]++;
    int cur=Integer.MIN_VALUE,res=0;
    for (int i=0;i<40000;i++)
      while (count[i]>0){
        cur = Math.max(cur+1,i);
        res+=cur-i;
        count[i]--;
      }
    return res;
  }

  public int maxScoreSightseeingPair(int[] A) {
    if (A==null || A.length<2)
      return 0;
    int max = 0,prevMax = 0;
    for (int a:A){
      max = Math.max(max,prevMax+a);
      prevMax = Math.max(prevMax,a)-1;
    }
    return max;
  }

  class Solution_478 {
    private double radius,x,y;
    private Random random;

    public Solution_478(double radius, double x_center, double y_center) {
      this.radius = radius;
      this.x = x_center;
      this.y = y_center;
      random = new Random();
    }

    public double[] randPoint() {
      double length = Math.sqrt(random.nextDouble())*radius;
      double R = random.nextDouble()*Math.PI*2;
      double curX = length*Math.cos(R)+x;
      double curY = length*Math.sin(R)+y;
      return new double[]{curX,curY};
    }
  }

  class SPLnode{
    int idx,state;
    public SPLnode(int idx,int state){
      this.idx = idx;
      this.state = state;
    }
  }

  public int shortestPathLength(int[][] graph) {
    if (graph==null || graph.length==0 )
      return 0;
    int N = graph.length,target = (1<<N)-1,steps = 0;
    boolean[][] visited = new boolean[N][target+1];
    Queue<SPLnode> queue = new LinkedList<>();
    for (int i=0;i<N;i++){
      SPLnode initial = new SPLnode(i,1<<i);
      queue.offer(initial);
      visited[initial.idx][initial.state] = true;
    }
    while (!queue.isEmpty()){
      int size = queue.size();
      steps++;
      for (int i=0;i<size;i++){
        SPLnode cur = queue.poll();
        for (int adj:graph[cur.idx]){
          int nextState = cur.state | (1<<adj);
          if (visited[adj][nextState])
            continue;
          visited[adj][nextState] = true;
          if (nextState==target)
            return steps;
          queue.offer(new SPLnode(adj,nextState));
        }
      }
    }
    return 0;
  }

  class NDTedge{
    int from,to,weight;
    public NDTedge(int f,int t,int w){
      from = f;
      to = t;
      weight = w;
    }
  }

  private List<NDTedge>[] NDTgetGraph(int[][] times,int N){
    List<NDTedge>[] graph = new List[N+1];
    for (int[] time:times){
      if (graph[time[0]]==null)
        graph[time[0]] = new ArrayList<>();
      graph[time[0]].add(new NDTedge(time[0],time[1],time[2]));
    }
    return graph;
  }

  public int networkDelayTime(int[][] times, int N, int K) {
    if (times==null || times.length==0)
      return 0;
    List<NDTedge>[] graph = NDTgetGraph(times,N);
    int[] distTo = new int[N+1];
    Arrays.fill(distTo,Integer.MAX_VALUE);
//    NDTDijkstra(graph,K,N,distTo);
//    NDTBellmanFord_naive(graph,K,N,distTo);
    NDTBellmanFord_optimized(graph,K,N,distTo);
    int max=0;
    for (int i=1;i<=N;i++){
      int d = distTo[i];
      if (d==Integer.MAX_VALUE)
        return -1;
      max = Math.max(max,d);
    }
    return max;
  }

  private void NDTDijkstra(List<NDTedge>[] graph,int start,int N,int[] distTo){
    boolean[] visited = new boolean[N+1];
    PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>() {
      @Override
      public int compare(Integer a, Integer b) {
        return distTo[a]-distTo[b];
      }
    });
    distTo[start] = 0;
    pq.offer(start);
    while (!pq.isEmpty()){
      int cur = pq.poll();
      if (visited[cur])
        continue;
      NDTrelax(graph,cur,distTo,visited,pq);
    }
  }

  private void NDTrelax(List<NDTedge>[] graph,int cur,int[] distTo,boolean[] visited,PriorityQueue<Integer> pq){
    visited[cur] = true;
    if (graph[cur]==null)
      return;
    for (NDTedge edge:graph[cur]){
      if (distTo[edge.to]>distTo[edge.from]+edge.weight){
        distTo[edge.to] = distTo[edge.from]+edge.weight;
        pq.offer(edge.to);
      }
    }
  }

  private void NDTBellmanFord_naive(List<NDTedge>[] graph,int start,int N,int[] distTo){
    distTo[start] = 0;
    for (int i=0;i<N-1;i++)
      for (int j=1;j<=N;j++){
        if (graph[j]==null)
          continue;
        for (NDTedge edge:graph[j])
          if ((long)distTo[edge.to]>(long)distTo[edge.from]+(long)edge.weight)
            distTo[edge.to] = distTo[edge.from]+edge.weight;
      }
    for (int i=1;i<=N;i++){
      if (graph[i]==null)
        continue;
      for (NDTedge edge:graph[i])
        if ((long)distTo[edge.to]>(long)distTo[edge.from]+(long)edge.weight){
          distTo[edge.to] = Integer.MAX_VALUE;
          return;
        }
    }
  }

  private void NDTBellmanFord_optimized(List<NDTedge>[] graph,int start,int N,int[] distTo){
    distTo[start] = 0;
    Queue<Integer> q = new LinkedList<>();
    q.offer(start);
    for (int i=0;i<N-1 && !q.isEmpty();i++){
      int size = q.size();
      for (int j=0;j<size;j++){
        int cur = q.poll();
        if (graph[cur]==null)
          continue;
        for (NDTedge edge:graph[cur])
          if ((long)distTo[edge.to]>(long)distTo[edge.from]+(long)edge.weight){
            distTo[edge.to] = distTo[edge.from]+edge.weight;
            q.offer(edge.to);
          }
      }
    }
    for (int i=1;i<=N;i++){
      if (graph[i]==null)
        continue;
      for (NDTedge edge:graph[i])
        if ((long)distTo[edge.to]>(long)distTo[edge.from]+(long)edge.weight){
          distTo[edge.to] = Integer.MAX_VALUE;
          return;
        }
    }
  }

  public int findPairs(int[] nums, int k) {
   if (nums==null||nums.length==0||k<0)
     return 0;
   Map<Integer,Integer> count = new HashMap<>();
   for (int n:nums)
     count.put(n,count.getOrDefault(n,0)+1);
   int res = 0;
   for (int cur:count.keySet()){
     Integer otherCount = count.get(cur+k);
     if (k==0)
       res+=otherCount!=null && otherCount>1?1:0;
     else
       res+=otherCount!=null?1:0;
   }
   return res;
  }

  public boolean validSquare2(int[] p1, int[] p2, int[] p3, int[] p4) {
    return VSisSquare(p1,p2,p3,p4) || VSisSquare(p1,p3,p2,p4) || VSisSquare(p1,p4,p2,p3);
  }

  private boolean VSisSquare(int[] p11,int[] p12,int[] p21,int[] p22){
    int edge,diagnol;
    return (edge=VSgetDist(p11,p21))==VSgetDist(p11,p22) &&
            VSgetDist(p12,p21) == VSgetDist(p12,p22) &&
            VSgetDist(p11,p21) == VSgetDist(p12,p21) &&
            edge!=0 &&
            (diagnol = VSgetDist(p11,p12))==VSgetDist(p21,p22) &&
            diagnol!=0;
  }

  private int VSgetDist(int[] p1,int[] p2){
    int xDiff = p1[0]-p2[0],yDiff = p1[1]-p2[1];
    return xDiff*xDiff+yDiff*yDiff;
  }

  public int maxProfit714(int[] prices, int fee) {
    if (prices==null || prices.length==0)
      return 0;
    int N = prices.length;
    int[] buy = new int[N],sell = new int[N];
    buy[0] = -prices[0]-fee;
    for (int i=1;i<N;i++){
      buy[i] = Math.max(buy[i-1],sell[i-1]-prices[i]-fee);
      sell[i] = Math.max(sell[i-1],buy[i-1]+prices[i]);
    }
    return sell[N-1];
  }

  class DTnode{
    int temperature,date;
    public DTnode(int t,int d){
      temperature = t;
      date = d;
    }
  }

  public int[] dailyTemperatures(int[] T) {
    if (T==null)
      return null;
    if (T.length==0)
      return T;
    int N = T.length;
    int[] res = new int[N];
    Stack<DTnode> st = new Stack<>();
    for (int i=0;i<N;i++){
      while (!st.isEmpty() && st.peek().temperature<T[i]){
        DTnode lower = st.pop();
        res[lower.date] = i-lower.date;
      }
      st.push(new DTnode(T[i],i));
    }
    return res;
  }

  public int twitter_wierdFaculty(int[] score){
    if (score==null||score.length==0)
      return 0;
    int N = score.length;
    int[] ps = new int[N+1];
    for (int i=0;i<N;i++)
      ps[i+1] = ps[i]+score[i];
    for (int k=0;k<=N;k++){
      int curScore = ps[k]-(k-ps[k]);
      int friendScore = (ps[N]-ps[k])-(N-(ps[N]-ps[k]));
      if (curScore>friendScore)
        return k;
    }
    return N;
  }

  public int minCost(int[][] costs) {
    if (costs==null || costs.length==0)
      return 0;
    int N = costs.length;
    int[][] dp = new int[N][3];
    for (int i=0;i<3;i++)
      dp[N-1][i] = costs[N-1][i];
    for (int i=N-2;i>=0;i--)
      for (int j=0;j<3;j++){
        int after;
        if (j==0)
          after=Math.min(dp[i+1][1],dp[i+1][2]);
        else if (j==1)
          after=Math.min(dp[i+1][0],dp[i+1][2]);
        else
          after=Math.min(dp[i+1][0],dp[i+1][1]);
        dp[i][j] = costs[i][j]+after;
      }
    return Math.min(Math.min(dp[0][0],dp[0][1]),dp[0][2]);
  }

  public int findCircleNum(int[][] M) {
    if (M==null||M.length==0)
      return 0;
    int N = M.length,res=0;
    boolean[] visited = new boolean[N];
    for (int i=0;i<N;i++)
      if (!visited[i]){
        FChelper(M,i,visited);
        res++;
      }
    return res;
  }

  private void FChelper(int[][] M,int idx,boolean[] visited){
    visited[idx] = true;
    int N = M.length;
    for (int i=0;i<N;i++)
      if (!visited[i] && M[idx][i]==1)
        FChelper(M,i,visited);
  }

  public int minFountain(int[] f){
    if (f==null || f.length==0)
      return 0;
    int N = f.length,idx=1,res=0,arrive=0;
    List<Integer>[] range  = new List[N+1];
    for (int i=1;i<=N;i++)
      range[i] = new ArrayList();
    for (int i=0;i<N;i++)
      range[Math.max(1,i+1-f[i])].add(Math.min(N,i+1+f[i]));
    while (idx<=N){
      if (arrive==N)
        return res;
      int nextArrive=0;
      while (idx<=N && idx<=arrive){
        for (int i:range[idx])
          nextArrive = Math.max(nextArrive,range[idx].get(i));
        idx++;
      }
      if (nextArrive<=arrive)
        return 0;
      arrive = nextArrive;
      res++;
    }
    return res;
  }

  public int numberOfTokens(int expire,int[][] commands){
    if (commands==null || commands.length==0)
      return 0;
    Map<Integer,Integer> record = new HashMap<>();
    int N = commands.length,last = commands[N-1][2],res=0;
    for (int i=0;i<N;i++)
      if (commands[i][0]==0)
        record.put(commands[i][1],commands[i][2]+expire);
      else{
        Integer expireTime;
        if ((expireTime = record.get(commands[i][1]))==null || expireTime<commands[i][2])
          continue;
        record.put(commands[i][1],commands[i][2]+expire);
      }
    for (int v:record.values())
      if (v>=last)
        res++;
      return res;
  }

  public static void main(String[] args) {
    Practice2 p = new Practice2();
//    Random r = new Random();
//    System.out.println(r.nextInt(240));
    p.numberOfTokens(4,new int[][]{{0,1,1},{0,2,2},{1,1,5},{1,2,7}});
  }
}

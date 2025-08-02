# A Tutorial for LeetCode

A compact, results-oriented plan to get to *sufficiency* for Meta/Google coding rounds.

[https://neatleet.github.com](https://neatleet.github.com)

## Contents

- Chapter 1 — Plan
- Chapter 2 — Patterns
- Chapter 3 — Invariants
- Chapter 4 — Practice

---

## Chapter 1 —  Plan

### Sufficiency to Pass

- **Speed & accuracy:** Solve **2 random Mediums** in \~**35 min** each (clean, tested).
- **Hit rate:** **≥80%** on unseen Mediums from a curated list.
- **Fluency:** You recognize the **pattern** within 1–2 minutes and state the **invariant**.

### Principles of Preparation

- **Coverage:** Focus on the high‑yield patterns; know when to reach for each.
- **Frequency:** Prioritize problems that show up frequently in interviews; drill these first.
- **Learn:** Practice by bucket; after each session, recap the key idea and update your **error list**.
- **Method:** Identify the pattern → state the invariant → choose the right **variant** *(state conditions; ask clarifying questions)* → explain as you code.

### High‑Yield Patterns

1. Arrays & hashing (freq map, two-sum/generalized k-sum, prefix sums)
2. Two pointers & sliding window (fixed/variable window; longest/shortest subarray)
3. Binary search (classic + binary search on answer; lower/upper bound)
4. Intervals (merge, insert, sweep line)
5. Stack: monotonic stack (next greater/smaller), parentheses, histogram
6. Heap / priority queue (k-best, merge k lists)
7. BFS/DFS on graphs (grid, visited sets, shortest path unweighted)
8. Topological sort (Kahn/DFS; prerequisites/course schedule)
9. Union–Find (connectivity, components, Kruskal-style thinking)
10. Trees (BST properties, inorder = sorted; recursion templates; LCA)
11. Backtracking (subsets, perms, combos; dedupe patterns)
12. Dynamic Programming (1D: house robber, coin change, LIS; 2D: grid paths, edit distance)
13. Knapsack pattern (0/1, unbounded)
14. Bit tricks (masks, XOR pairs/singletons)

### 6-Week roadmap (8–15 hrs/week)

- **Week 1 – Foundations:** Relearn templates (sliding window, binary search on answer, monotonic stack, BFS/DFS, DSU, topological sort, 1D DP) and solve \~20 Easies/Light Mediums.
- **Weeks 2–3 – Core reps:** Complete \~60 Mediums across the 12 patterns (\~10 per pattern).
- **Week 4 – Graphs & DP focus:** Tackle \~25 Mediums on BFS shortest paths, topological sort, LIS, coin change, edit distance, and knapsack patterns.
- **Week 5 – Mixed & speed:** Solve \~30 timed Mediums (35-minute cap) with immediate re-implementation from memory.
- **Week 6 – Mocks & polish:** Conduct 4–6 mock interviews and reinforce weak patterns with 3 targeted problems per identified miss.

### Daily session (90 minutes)

1. **15m**: Flash review (your invariant cards).
2. **60m**: 1 timed Medium (≤35m solve + 10m tests) + **15m** re-type from memory.
3. **15m**: Error log (pattern, invariant, edge cases missed).

### Curated lists

- **Coverage (core set):** Use **NeetCode 150 / All** (or **Blind 75**) → extend with **Grind 75** for breadth.
- **Frequency (company‑specific):** Add a company list (e.g., **Meta**, **Google**, **Amazon**) and prioritize items that recur in that company’s interviews.
- **Tracking:** **Track by tag** to ensure coverage; **sample randomly within tags** to avoid memorization.

### Interview micro-checklist

- **Restate the problem** and constraints; define **N** and acceptable **O()**.
- **Work a tiny example**; list **edge cases** (empty, 1 elem, duplicates, negatives, overflow, cycles/bounds).
- **Pick the pattern & state the invariant**; **code top‑down** and **narrate** decisions.
- State **time/space complexity**; **test worst/edge cases**.

### Mocks & signal

- Aim for **2 mocks/week** in **Weeks 5–6**.
- Score yourself on: **correctness, time, communication, testing**.
- If a pattern repeatedly trips you, do **3 focused problems** the same day and **1 more the next morning**.

---

## Chapter 2 — Patterns

Below is a compact, interview‑oriented tutorial for each high‑yield pattern. For every item you get: **Definition**, a **canonical example**, and **Template(s)** with the key **Invariant** you should state out loud.

---

### 1) Arrays & Hashing

**Definition.** Use hash maps/sets to count, index, or dedupe in O(n).

**Examples.** Two Sum, Group Anagrams, Subarray Sum Equals K.

**A) Frequency / counting (no imports)**\
*Invariant:* the map/array reflects counts of all elements processed so far.

```python
# Generic counting with a dict
counts = {}
for x in nums:
    counts[x] = counts.get(x, 0) + 1
```

```python
# Lowercase English letters (faster than Counter for LC)
cnt = [0] * 26
for ch in s:
    cnt[ord(ch) - 97] += 1
```

**B) Two Sum (indices)**\
*Invariant:* for each index `i`, `pos` stores indices of numbers seen so far.

```python
pos = {}
for i, x in enumerate(nums):
    if target - x in pos:
        return [pos[target - x], i]
    pos[x] = i
```

**C) Prefix sum → subarray sum K**\
*Invariant:* `seen[p]` = count of prefixes with sum `p` seen so far.

```python
seen = {0: 1}
pref = ans = 0
for x in nums:
    pref += x
    ans += seen.get(pref - K, 0)
    seen[pref] = seen.get(pref, 0) + 1
```

**D) Group Anagrams (signature key)**\
*Invariant:* `groups[key]` collects words sharing the same character‑count signature.

```python
groups = {}
for s in strs:
    sig = [0] * 26
    for ch in s:
        sig[ord(ch) - 97] += 1
    key = tuple(sig)
    groups.setdefault(key, []).append(s)
return list(groups.values())
```

**E) Longest equal 0/1 subarray (treat 0 as −1)**\
*Invariant:* `first[prefix]` stores the earliest index where this prefix sum occurred.

```python
first = {0: -1}
prefix = best = 0
for i, x in enumerate(nums):
    prefix += 1 if x == 1 else -1
    if prefix in first:
        best = max(best, i - first[prefix])
    else:
        first[prefix] = i
return best
```

---

### 2) Two Pointers & Sliding Window

**Definition.** Maintain a moving window or converging pointers to satisfy a predicate without revisiting work.

**Examples.** Longest Substring Without Repeating, Minimum Size Subarray Sum, 3‑Sum (inner two pointers).

**A) No repeats (map of last index)**\
*Invariant:* substring `s[l:r]` has unique chars.

```python
last = {}; l = 0; ans = 0
for r, ch in enumerate(s):
    if ch in last and last[ch] >= l:
        l = last[ch] + 1
    last[ch] = r
    ans = max(ans, r - l + 1)
```

**B) Min length with sum ≥ target**\
*Invariant:* while shrinking, window sum ≥ target; after loop, window minimal for this `r`.

```python
l = 0; curr = 0; best = float('inf')
for r, x in enumerate(nums):
    curr += x
    while curr >= target:
        best = min(best, r - l + 1)
        curr -= nums[l]; l += 1
```

**C) Sorted two‑sum (for 3‑sum inner loop)**\
*Invariant:* pairs outside `[l,r]` are ruled out.

```python
l, r = 0, n - 1
while l < r:
    s = a[l] + a[r]
    if s == target: ...
    elif s < target: l += 1
    else: r -= 1
```

---

### 3) Binary Search (classic & on answer)

**Definition.** Maintain an ordered predicate and shrink to the boundary.

**Examples.** Search in Rotated Array, First/Last Position, Koko Eating Bananas (answer search).

\*\*A) First True by predicate \*\*\`\`\
*Invariant:* `ok(lo)=False`, `ok(hi)=True`.

```python
lo, hi = L-1, R+1
while hi > lo + 1:
    mid = (lo + hi)//2
    if ok(mid): hi = mid
    else:       lo = mid
return hi
```

\*\*B) \*\*\`\`

```python
l, r = 0, n
while l < r:
    m = (l + r)//2
    if a[m] < x: l = m + 1
    else:        r = m
return l
```

---

### 4) Intervals

**Definition.** Sort by start; merge overlaps; preserve non‑overlap.

**Examples.** Merge Intervals, Insert Interval, Employee Free Time.

**Merge template**\
*Invariant:* `merged` is sorted & non‑overlapping over processed items.

```python
intervals.sort()
merged = []
for s, e in intervals:
    if not merged or merged[-1][1] < s:
        merged.append([s, e])
    else:
        merged[-1][1] = max(merged[-1][1], e)
```

---

### 5) Stack (Monotonic, Parentheses)

**Definition.** Use a stack to maintain a monotone sequence or to match pairs.

**Examples.** Next Greater Element, Daily Temperatures, Largest Rectangle in Histogram, Valid Parentheses.

**A) Next greater (decreasing stack of values)**\
*Invariant:* stack indices keep strictly decreasing values.

```python
st = []; ans = [-1]*n
for i, x in enumerate(nums):
    while st and nums[st[-1]] < x:
        ans[st.pop()] = x
    st.append(i)
```

**B) Histogram**\
*Invariant:* stack keeps increasing heights; pop finalizes area for that bar.

```python
st = []; best = 0
for i, h in enumerate(heights + [0]):
    while st and heights[st[-1]] > h:
        H = heights[st.pop()]
        L = st[-1] if st else -1
        best = max(best, H * (i - L - 1))
    st.append(i)
```

**C) Parentheses (validity)**

```python
st = []
pairs = {')':'(', ']':'[', '}':'{'}
for ch in s:
    if ch in '([{': st.append(ch)
    elif not st or st.pop() != pairs[ch]:
        return False
return not st
```

---

### 6) Heap / Priority Queue

**Definition.** Maintain best k elements or repeatedly extract min/max.

**Examples.** Top K Frequent Elements, Kth Largest, Merge k Sorted Lists.

**A) Keep k largest (min‑heap)**\
*Invariant:* heap holds k best seen so far.

```python
h = []
for x in stream:
    if len(h) < k: heapq.heappush(h, x)
    elif x > h[0]: heapq.heapreplace(h, x)
# h holds the k largest
```

**B) Merge k lists**

```python
h = []
for i, node in enumerate(lists):
    if node: heapq.heappush(h, (node.val, i, node))
dummy = tail = ListNode()
while h:
    _, i, node = heapq.heappop(h)
    tail.next = node; tail = tail.next
    if node.next: heapq.heappush(h, (node.next.val, i, node.next))
```

---

### 7) BFS / DFS on Graphs

**Definition.** Explore graph by layers (BFS) or depth (DFS). Use `visited`.

**Examples.** Number of Islands, Rotting Oranges, Shortest Path in Binary Matrix.

**A) BFS (unweighted shortest path)**\
*Invariant:* when dequeued, `dist[u]` is minimal.

```python
q = collections.deque([src])
dist = {src: 0}
while q:
    u = q.popleft()
    for v in nbrs(u):
        if v not in dist:
            dist[v] = dist[u] + 1
            q.append(v)
```

**B) DFS (grid islands)**\
*Invariant:* `visited` cells won’t be revisited; each DFS call exhausts one island.

```python
def dfs(r, c):
    if not inside(r,c) or grid[r][c] != '1': return
    grid[r][c] = '0'  # mark visited
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        dfs(r+dr, c+dc)
count = 0
for r in range(R):
    for c in range(C):
        if grid[r][c] == '1':
            count += 1; dfs(r,c)
```

---

### 8) Topological Sort

**Definition.** Linearize DAG so edges go left→right.

**Examples.** Course Schedule, Alien Dictionary (with order), Tasks with prerequisites.

**Kahn’s algorithm**\
*Invariant:* queue contains exactly indegree‑0 nodes.

```python
deg = [0]*n
for u in range(n):
    for v in g[u]: deg[v] += 1
q = collections.deque([i for i,d in enumerate(deg) if d==0])
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in g[u]:
        deg[v] -= 1
        if deg[v] == 0: q.append(v)
# len(order)==n ⇒ DAG
```

---

### 9) Union–Find (Disjoint Set Union)

**Definition.** Track connectivity/components with near‑O(1) unions/finds.

**Examples.** Accounts Merge, Number of Connected Components, Kruskal MST.

**Template**\
*Invariant:* each node’s parent chain ends at the set’s root.

```python
parent = list(range(n)); rank = [0]*n

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # path compression
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1
    return True
```

---

### 10) Trees

**Definition.** Use recursion/inorder properties; pass state (bounds/parents).

**Examples.** Validate BST, Binary Tree Level Order, LCA, Serialize/Deserialize.

**A) Validate BST (bounds)**\
*Invariant:* node value in `(low, high)`.

```python
def valid(node, low=float('-inf'), high=float('inf')):
    if not node: return True
    if not (low < node.val < high): return False
    return valid(node.left, low, node.val) and valid(node.right, node.val, high)
```

**B) LCA (general binary tree)**\
*Invariant:* return node if found in subtree; otherwise bubble up non‑None.

```python
def lca(root, p, q):
    if not root or root in (p, q): return root
    L = lca(root.left, p, q)
    R = lca(root.right, p, q)
    return root if L and R else (L or R)
```

**C) Inorder gives sorted for BST**\
*Invariant:* `prev` tracks last inorder value; must increase strictly.

```python
prev = None

def inorder(n):
    nonlocal prev
    if not n: return True
    if not inorder(n.left): return False
    if prev is not None and n.val <= prev: return False
    prev = n.val
    return inorder(n.right)
```

---

### 11) Backtracking

**Definition.** Build partial solutions; choose → recurse → unchoose.

**Examples.** Subsets, Permutations, Combination Sum, Palindrome Partitioning.

**A) Subsets**\
*Invariant:* `path` uses indices `< start` fixed; it’s always valid.

```python
ans = []

def dfs(start, path):
    ans.append(path[:])
    for i in range(start, n):
        path.append(a[i]); dfs(i+1, path); path.pop()

dfs(0, [])
```

**B) Permutations with dedupe**\
*Invariant:* `used[i]` marks chosen elements; skip equal siblings when previous twin unused.

```python
a.sort(); used = [False]*n; ans = []; path = []

def dfs():
    if len(path) == n: ans.append(path[:]); return
    for i in range(n):
        if used[i]: continue
        if i and a[i]==a[i-1] and not used[i-1]: continue
        used[i] = True; path.append(a[i])
        dfs()
        path.pop(); used[i] = False

dfs()
```

**C) Combination Sum (unbounded)**\
*Invariant:* `path` sum equals `target` when recorded; choices can reuse current index.

```python
ans = []

def dfs(i, target, path):
    if target == 0: ans.append(path[:]); return
    if i == len(cands) or target < 0: return
    # take
    path.append(cands[i]); dfs(i, target - cands[i], path); path.pop()
    # skip
    dfs(i+1, target, path)
```

---

### 12) Dynamic Programming

**Definition.** Define state meaning; write recurrence; order iterations so dependencies are ready.

**Examples.** House Robber, Coin Change, LIS, Unique Paths, Edit Distance.

**A) House Robber (1D rolling)**\
*Invariant:* `take/skip` are optimal up to current index.

```python
take = skip = 0
for x in nums:
    take, skip = skip + x, max(take, skip)
return max(take, skip)
```

**B) Coin Change (min coins)**

```python
dp = [float('inf')] * (target + 1)
dp[0] = 0
for s in range(1, target+1):
    for c in coins:
        if s >= c:
            dp[s] = min(dp[s], dp[s-c] + 1)
return dp[target] if dp[target] < float('inf') else -1
```

**C) LIS (patience with tails) — O(n log n)**\
*Invariant:* `tails[i]` is the minimum tail of an LIS of length `i+1`.

```python
tails = []
for x in nums:
    i = bisect_left(tails, x)
    if i == len(tails): tails.append(x)
    else: tails[i] = x
len_lis = len(tails)
```

**D) Unique Paths (grid DP)**\
*Invariant:* `dp[c]` counts ways to reach current cell in this row.

```python
m, n = len(grid), len(grid[0])
dp = [0]*n; dp[0] = 1
for r in range(m):
    for c in range(n):
        if grid[r][c] == 1: dp[c] = 0     # obstacle
        elif c: dp[c] += dp[c-1]
return dp[-1]
```

**E) Edit Distance**\
*Invariant:* `dp[i][j]` = distance between `a[:i]` and `b[:j]`.

```python
m, n = len(a), len(b)
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m+1): dp[i][0] = i
for j in range(n+1): dp[0][j] = j
for i in range(1, m+1):
    for j in range(1, n+1):
        if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
        else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
return dp[m][n]
```

---

### 13) Knapsack Pattern

**Definition.** Choose items under capacity with value optimization. Key is iteration order.

**Examples.** 0/1 Knapsack, Coin Change (ways/least coins).

**A) 0/1 Knapsack (each item once)**\
*Invariant:* `dp[w]` = best value for capacity `w`; iterate weights **descending**.

```python
dp = [0]*(W+1)
for wt, val in items:
    for w in range(W, wt-1, -1):
        dp[w] = max(dp[w], dp[w-wt] + val)
```

**B) Unbounded Knapsack (reuse items)**\
*Invariant:* `dp[w]` uses current item unlimited times; iterate weights **ascending**.

```python
dp = [0]*(W+1)
for wt, val in items:
    for w in range(wt, W+1):
        dp[w] = max(dp[w], dp[w-wt] + val)
```

---

### 14) Bit Tricks

**Definition.** Use bitmasks to encode sets; exploit XOR and bit ops.

**Examples.** Single Number (XOR), Counting Bits, Subset enumeration.

**A) XOR singletons**\
*Invariant:* XOR cancels pairs and preserves the odd‑occurring element.

```python
x = 0
for v in nums: x ^= v
# x = element that appears odd number of times (e.g., once)
```

**B) Count bits up to n**\
*Invariant:* `dp[i] = dp[i>>1] + (i & 1)` (drop LSB + parity).

```python
dp = [0]*(n+1)
for i in range(1, n+1):
    dp[i] = dp[i >> 1] + (i & 1)
```

**C) Enumerate all subsets of a mask**\
*Invariant:* `sub` iterates every submask of `s` exactly once.

```python
s = full_mask
sub = s
while sub:
    # use sub
    sub = (sub - 1) & s
# include 0 if needed
```

---

## Chapter 3 — Invariants

**What is an invariant?**\
**Invariant (here):** A truth about your state that is **always true** during the algorithm (before every loop iteration/after each step). You **initialize** it, **maintain** it with every update, and at **termination** it implies the answer.

**How to use invariants (quick recipe)**

1. **Name the state variables.** (`l, r, sum`, `stack`, `heap`, `visited`, `dp[i]`, etc.)
2. **Write a one-sentence truth** that must always hold.
3. **Choose updates** (move pointers, push/pop, union, etc.) that keep that truth valid.
4. **Argue the result:** when the loop/recursion ends, the invariant implies the answer.

Summary: Name state → State the invariant → Choose updates that preserve it → Termination ⇒ result.

**Quick example (Sliding Window)**

- **Generic pattern:**
  - **Invariant (generic):** The window `s[l:r]` satisfies a predicate (e.g., unique characters, ≤K distinct, sum constraint). If violated as you expand `r`, shrink from `l` until restored.
  - **Template (generic):**

```python
# Generic sliding window skeleton
l = 0
# initialize per-window state (e.g., counts/set/sum)
for r in range(len(s)):
    add(s[r])                    # include right char/item
    while not predicate_holds(): # restore invariant
        remove(s[l])
        l += 1
    update_answer(l, r)          # window s[l:r] satisfies predicate
```

- **Applied: Longest Substring Without Repeating**
  - **Problem:** Given a string `s`, return the length of the longest substring without repeating characters.
  - **Invariant:** The window `s[l:r]` contains only unique characters; if violated, move `l` until restored.
  - **Template:**

```python
last = {}
l = 0
best = 0
for r, ch in enumerate(s):
    if ch in last and last[ch] >= l:
        l = last[ch] + 1
    last[ch] = r
    best = max(best, r - l + 1)
return best
```

### Examples by pattern

- **Sliding window (Longest substring w/o repeats)**\
  **Invariant:** “`s[l:r]` has all unique chars.”\
  Expand `r` while maintaining a set/map; when violated, move `l` until restored. The max window seen is the answer.

- **Variable window (Min subarray ≥ target)**\
  **Invariant:** “When we try to shrink, `sum(s[l:r]) ≥ target`; after the inner while, the window is minimal for this `r`.”

- **Two pointers (sorted two‑sum)**\
  **Invariant:** “All pairs left of `l` or right of `r` have been proven impossible.”\
  If sum too small → `l++`; too big → `r--`. You never revisit ruled‑out pairs.

- **Binary search on answer (find min x with ****\`\`**** true)** Initialize `lo = L-1`, `hi = R+1`; mid splits the False/True boundary. When `hi == lo+1`, `hi` is the first True.

- **Monotonic stack (next greater / histogram)**\
  **Invariant:** “Stack indexes hold elements in monotone order (e.g., decreasing/increasing heights).”\
  While the next element breaks monotonicity, pop and resolve answers/areas.

- **Heap / k‑best**\
  **Invariant:** “Heap contains the k best candidates seen so far.”\
  Push new; if size > `k` pop worst (or `heapreplace`). The invariant guarantees correctness at the end.

- **BFS shortest path (unweighted)**\
  **Invariant:** “When a node is dequeued, its recorded distance is the shortest possible.”\
  Level‑by‑level expansion preserves minimality.

- **Topological sort (Kahn’s)**\
  **Invariant:** “Every node in the queue has indegree 0; every removed edge decreases indegree correctly.”\
  If result visits all nodes, the graph was a DAG.

- **Union–Find (DSU)**\
  **Invariant:** “Each set is represented by a root; `find(x)` returns the current root; parents always lead to a root.”\
  Path compression/union by rank preserve the set partition.

- **Intervals (merge)**\
  **Invariant:** “`merged` covers all processed intervals and is non‑overlapping & sorted.”\
  Sort by start; extend last if overlapping; else append.

- **Backtracking (subsets/perms/combos)**\
  **Invariant:** “State is a valid partial solution built from a prefix of choices; indices before `start` are fixed.”\
  Add a choice → recurse → remove it; dedupe by sorting + skipping equal siblings.

- **Dynamic programming**\
  **Invariant (state meaning):** “`dp[i]` equals (e.g., best up to `i` inclusive).”\
  Transitions must only read states that are already correct (right iteration order).

- **BST validation (recursion bounds)**\
  **Invariant:** “Current node value is in `(low, high)` and subtrees respect updated bounds.”\
  Left uses `(low, node.val)`, right uses `(node.val, high)`.

### Why this matters

- It turns “pattern recognition” into **provable steps**.
- It prevents **off‑by‑one** and **revisiting** errors.
- It makes your interview narration crisp: “I’ll maintain the invariant **X**; here’s how expand/contract preserves it; when the loop ends, **Y** follows.”

---

## Chapter 4 — Practice

### Examples

Below: **Problem → Invariant → Minimal Template**\
(Keep code minimal; focus on the invariant line you’ll say aloud.)

#### 1) Arrays & Hashing — *Subarray Sum Equals K*

**Invariant:** `seen[p]` counts prefixes with sum `p` processed so far.

```python
seen = collections.Counter({0:1})
pref = ans = 0
for x in nums:
    pref += x
    ans += seen[pref - K]
    seen[pref] += 1
```

**Related drills (LC):** Two Sum (1), Group Anagrams (49), Subarray Sum Equals K (560), Contiguous Array (525).

#### 2) Sliding Window — *Longest Substring Without Repeating Characters*

**Invariant:** `s[l:r]` contains unique chars; repair by moving `l`.

```python
last = {}; l = ans = 0
for r, ch in enumerate(s):
    if ch in last and last[ch] >= l: l = last[ch] + 1
    last[ch] = r
    ans = max(ans, r - l + 1)
```

**Related drills (LC):** Longest Substring Without Repeating (3), At Most K Distinct (340), Minimum Window Substring (76), Minimum Size Subarray Sum (209).

#### 3) Two Pointers (sorted) — *3‑Sum (inner two‑pointer)*

**Invariant:** Pairs outside `[l,r]` are ruled out by order.

```python
l, r = i+1, n-1
while l < r:
    s = a[i] + a[l] + a[r]
    if s == 0: ...
    elif s < 0: l += 1
    else: r -= 1
```

**Related drills (LC):** 3Sum (15), 3Sum Closest (16), Remove Duplicates from Sorted Array (26), Trapping Rain Water (42).

#### 4) Binary Search on Answer — *Koko Eating Bananas*

**Invariant:** `ok(lo)=False`, `ok(hi)=True` (first True).

```python
def ok(k):  # can finish with speed k?
    ...

lo, hi = 0, max(piles)+1
while hi > lo + 1:
    mid = (lo + hi)//2
    if ok(mid): hi = mid
    else: lo = mid
return hi
```

**Related drills (LC):** Koko Eating Bananas (875), Ship Packages Within D Days (1011), Min Days to Make Bouquets (1482).

#### 5) Intervals — *Merge Intervals*

**Invariant:** `merged` is sorted & non‑overlapping for processed.

```python
intervals.sort()
merged = []
for s, e in intervals:
    if not merged or merged[-1][1] < s: merged.append([s, e])
    else: merged[-1][1] = max(merged[-1][1], e)
```

**Related drills (LC):** Merge Intervals (56), Insert Interval (57), Non-overlapping Intervals (435), Meeting Rooms II (253).

#### 6) Monotonic Stack — *Largest Rectangle in Histogram*

**Invariant:** Stack keeps indices of increasing heights; pop finalizes area.

```python
st = []; best = 0
for i, h in enumerate(heights + [0]):
    while st and heights[st[-1]] > h:
        H = heights[st.pop()]
        L = st[-1] if st else -1
        best = max(best, H * (i - L - 1))
    st.append(i)
```

**Related drills (LC):** Largest Rectangle in Histogram (84), Daily Temperatures (739), Next Greater Element II (503), Trapping Rain Water (42).

#### 7) Heap / Priority Queue — *Top‑K Frequent Elements*

**Invariant:** Heap holds the k best so far.

```python
cnt = {}
for x in nums:
    cnt[x] = cnt.get(x, 0) + 1
h = []
for x, c in cnt.items():
    if len(h) < k: heapq.heappush(h, (c, x))
    elif c > h[0][0]: heapq.heapreplace(h, (c, x))
return [x for _, x in h]
```

**Related drills (LC):** Top K Frequent Elements (347), Kth Largest Element in an Array (215), Merge k Sorted Lists (23), Kth Largest in a Stream (703).

#### 8) BFS (unweighted shortest path) — *Shortest Path in Binary Matrix*

**Invariant:** When dequeued, `dist[u]` is minimal.

```python
q = collections.deque([start])
dist = {start: 0}
while q:
    u = q.popleft()
    if u == goal: return dist[u]
    for v in nbrs(u):
        if v not in dist:
            dist[v] = dist[u] + 1
            q.append(v)
```

**Related drills (LC):** Number of Islands (200), Rotting Oranges (994), Word Ladder (127), Shortest Path in Binary Matrix (1091).

#### 9) Topological Sort (Kahn) — *Course Schedule II*

**Invariant:** Queue contains exactly indegree‑0 nodes.

```python
deg = [0]*n
for u in range(n):
    for v in g[u]: deg[v] += 1
q = collections.deque([i for i,d in enumerate(deg) if d==0])
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in g[u]:
        deg[v] -= 1
        if deg[v] == 0: q.append(v)
return order if len(order)==n else []
```

**Related drills (LC):** Course Schedule (207), Course Schedule II (210), Minimum Height Trees (310).

#### 10) Union–Find (DSU) — *Connected Components*

**Invariant:** `find(x)` returns the current root; unions merge roots.

```python
parent = list(range(n)); rank = [0]*n
def find(x):
    if parent[x] != x: parent[x] = find(parent[x])
    return parent[x]
def union(a,b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1
    return True
```

**Related drills (LC):** Number of Provinces (547), Accounts Merge (721), Redundant Connection (684), Connected Components in Undirected Graph (323).

#### 11) Trees — *Validate BST (bounds)*

**Invariant:** Node in `(low, high)`; children tighten bounds.

```python
def valid(n, low=float('-inf'), high=float('inf')):
    if not n: return True
    if not (low < n.val < high): return False
    return valid(n.left, low, n.val) and valid(n.right, n.val, high)
```

**Related drills (LC):** Validate BST (98), LCA of Binary Tree (236), Binary Tree Level Order Traversal (102), Serialize & Deserialize (297).

#### 12) Backtracking — *Subsets*

**Invariant:** `path` is a valid partial using indices `< start` fixed.

```python
ans = []
def dfs(start, path):
    ans.append(path[:])
    for i in range(start, n):
        path.append(a[i]); dfs(i+1, path); path.pop()
dfs(0, [])
```

**Related drills (LC):** Subsets (78), Permutations (46), Combination Sum (39), Palindrome Partitioning (131).

#### 13) Dynamic Programming — *House Robber*

**Invariant:** `take/skip` are optimal up to current index.

```python
take = skip = 0
for x in nums:
    take, skip = skip + x, max(take, skip)
return max(take, skip)
```

**Related drills (LC):** House Robber (198), Coin Change (322), Partition Equal Subset Sum (416).

#### 14) Dynamic Programming — *Edit Distance*

**Invariant:** `dp[i][j]` = distance between `a[:i]` and `b[:j]`.

```python
m, n = len(a), len(b)
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m+1): dp[i][0] = i
for j in range(n+1): dp[0][j] = j
for i in range(1, m+1):
    for j in range(1, n+1):
        dp[i][j] = dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
```

**Related drills (LC):** Edit Distance (72), Unique Paths II (63), Coin Change (322).

#### 15) DP (LIS, O(n log n)) — *Patience with tails*

```python
tails = []
for x in nums:
    i = bisect_left(tails, x)
    if i == len(tails): tails.append(x)
    else: tails[i] = x
```

**Related drills (LC):** Longest Increasing Subsequence (300), Russian Doll Envelopes (354).

---

### Drills

**How to practice**

1. **Before solving:** Name the **pattern** and speak the **invariant**.
2. **During:** Keep the invariant visible; after each update, confirm it still holds.
3. **After:** Re-type from memory; add misses to your error log as a one-line invariant you’ll use next time.

**Practice Table — Patterns & LeetCode Drills**

| Pattern                 | LeetCode Drills (IDs)                                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Arrays & Hashing        | Two Sum (1), Group Anagrams (49), Subarray Sum Equals K (560), Contiguous Array (525)                                                   |
| Sliding Window          | Longest Substring Without Repeating (3), At Most K Distinct (340), Minimum Window Substring (76), Minimum Size Subarray Sum (209)       |
| Two Pointers (sorted)   | 3Sum (15), 3Sum Closest (16), Remove Duplicates from Sorted Array (26), Trapping Rain Water (42)                                        |
| Binary Search on Answer | Koko Eating Bananas (875), Ship Packages Within D Days (1011), Min Days to Make Bouquets (1482)                                         |
| Intervals               | Merge Intervals (56), Insert Interval (57), Non-overlapping Intervals (435), Meeting Rooms II (253)                                     |
| Monotonic Stack         | Largest Rectangle in Histogram (84), Daily Temperatures (739), Next Greater Element II (503), Trapping Rain Water (42)                  |
| Heap / Priority Queue   | Top K Frequent Elements (347), Kth Largest Element in an Array (215), Merge k Sorted Lists (23), Kth Largest in a Stream (703)          |
| BFS (Graphs/Grids)      | Number of Islands (200), Rotting Oranges (994), Word Ladder (127), Shortest Path in Binary Matrix (1091)                                |
| Topological Sort        | Course Schedule (207), Course Schedule II (210), Minimum Height Trees (310)                                                             |
| Union–Find (DSU)        | Number of Provinces (547), Accounts Merge (721), Redundant Connection (684), Connected Components in Undirected Graph (323)             |
| Trees                   | Validate BST (98), LCA of Binary Tree (236), Binary Tree Level Order Traversal (102), Serialize & Deserialize (297)                     |
| Backtracking            | Subsets (78), Permutations (46), Combination Sum (39), Palindrome Partitioning (131)                                                    |
| Dynamic Programming     | House Robber (198), Coin Change (322), Edit Distance (72), Unique Paths II (63), Partition Equal Subset Sum (416), Coin Change II (518) |
| Knapsack Pattern        | 0/1 Knapsack (classic variant via 416), Unbounded: Coin Change II (518), Complete Knapsack variations                                   |
| Bit Tricks              | Single Number (136), Counting Bits (338), Single Number III (260)                                                                       |



---

### Cheat‑Sheets

> **How to use:** Name state → state the invariant → choose updates that preserve it → termination ⇒ result. Keep these one‑liners visible while coding.

**Arrays & Hashing** — *Map/set reflects processed elements; prefix sums reflect sums so far.*\
Cue: `pos[x]`, `Counter`, `seen[prefix]`.

**Sliding Window** — *Window ****\`\`**** always satisfies predicate (unique/≤K/sum).*

**Two Pointers (sorted)** — *Pairs outside ****\`\`**** are ruled out by order.*

**Binary Search (answer/boundary)** — *Maintain False/True bracket or ****\`\`****.*

**Intervals** — *`merged`*\* remains sorted & non‑overlapping for processed intervals.\*

**Monotonic Stack** — *Stack stays monotone; pop finalizes popped index’s answer.*\
Cue: while breaks‑monotone → pop & resolve.

**Heap / Priority Queue** — *Heap holds exactly the k best/frontier so far.*\
Cue: size>k → pop worst / `heapreplace`.

**BFS (unweighted shortest path)** — *When dequeued, distance is final & minimal.*\
Cue: push unseen neighbors with `dist+1`.

**Topological Sort (Kahn)** — *Queue contains indegree‑0 nodes only.*\
Cue: pop, append, decrement indegrees, push new zeros.

**Union–Find (DSU)** — *Each node’s parent chain ends at its root; sets are disjoint.*\
Cue: path compression + union by rank/size.

**Trees (BST bounds)** — *Node value ∈ ****\`\`****; children tighten bounds.*

**Backtracking** — *`path`*\* is a valid partial built from fixed choices/indices.\*

**Dynamic Programming** — *`dp[state]`*\* equals the defined subproblem optimum/count.\*

**Knapsack** — *`dp[w]`*\* best value at capacity \`\`.\*

**Bit Tricks** — *Mask encodes set; XOR preserves parity/cancels pairs.*\
Cue: subset iterate: `sub = (sub-1) & mask`.

> Optional mini‑templates:

- **First True search:** `lo,hi=L-1,R+1; while hi>lo+1: mid=(lo+hi)//2; if ok(mid): hi=mid else: lo=mid`.
- **Histogram stack pop:** `H=heights[st.pop()]; L=st[-1] if st else -1; area=H*(i-L-1)`.
- **Window repair:** `while bad: shrink l`.


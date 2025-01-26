#  --- Two pointers ---

# Template
def sortColors(nums):
  left, right = 0, len(nums) - 1
  i = 0

  while i <= right:
    if nums[i] == 0:
      nums[i], nums[left] = nums[left], nums[i]
      left += 1
      i += 1
    elif nums[i] == 2:
      nums[i], nums[right] = nums[right], nums[i]
      right -= 1
    else:
      i += 1

  return nums


# --- Sliding window ---

# When Do I Use This?
# Consider using the sliding window pattern for questions that involve searching for a continuous subsequence in an array or string that satisfies a certain constraint.
# If you know the length of the subsequence you are looking for, use a fixed-length sliding window. Otherwise, use a variable-length sliding window.
# Examples:
# Finding the largest substring without repeating characters in a given string (variable-length).
# Finding the largest substring containing a single character that can be made by replacing at most k characters in a given string (variable-length).
# Finding the largest sum of a subarray of size k without duplicate elements in a given array (fixed-length).

# Practice Problems
# When practicing these problems, it is important to think about the appropriate data structure state to store the contents of the current window. Make sure it supports both:
# Adding and removing elements from the window in O(1) time.
# Checking if the window is valid in O(1) time.
# Dictionaries and sets are often the best choices.

def variable_length_sliding_window(nums):
  state = # choose appropriate data structure
  start = 0
  max_ = 0

  for end in range(len(nums)):
    # extend window
    # add nums[end] to state in O(1) in time

    while state is not valid:
      # repeatedly contract window until it is valid again
      # remove nums[start] from state in O(1) in time
      start += 1

    # INVARIANT: state of current window is valid here.
    max_ = max(max_, end - start + 1)

  return max_

# --- Fixed-Length Sliding Window ---

# When Do I Use This?
# When you know the length of the subsequence you are looking for,
# you can use a fixed-length sliding window. 
# The concept is similar to the variable-length sliding window, 
# but the implementation is a bit simpler, as during each iteration, 
# you both add and remove an element from the window to maintain its fixed size.

# Examples:
# Finding the largest substring without repeating characters in a given string (variable-length).
# Finding the largest substring containing a single character that can be made by replacing at most k characters in a given string (variable-length).
# Finding the largest sum of a subarray of size k without duplicate elements in a given array (fixed-length).

# Practice Problems
# When practicing these problems, it is important to think about the approrpriate data structure state to store the contents of the current window. 
# Make sure it supports both:
# Adding and removing elements from the window in O(1) time.
# Checking if the window is valid in O(1) time.
# Dictionaries and sets are often the best choices.

# Template

def fixed_length_sliding_window(nums, k):
  state = # choose appropriate data structure
  start = 0
  max_ = 0

  for end in range(len(nums)):
    # extend window
    # add nums[end] to state in O(1) in time

    if end - start + 1 == k:
      # INVARIANT: size of the window is k here.
      max_ = max(max_, contents of state)

      # contract window
      # remove nums[start] from state in O(1) in time
      start += 1

  return max_

# --- Intervals ---

# Sorting by Start Time
# Sorting intervals by their start times makes it easy to merge two intervals that are overlapping.


# Overlapping Intervals
# After sorting by start time, an interval overlaps with the previous interval 
# if it starts before the end time of the previous interval.

# Detecting overlapping intervals is the basis of the question Can Attend Meetings, 
# in which we are given a list of intervals representing the start and end times of meetings, 
# and we need to determine if a person can attend all meetings.
# We sort the intervals by their start times and iterate over each meeting. 
# If the current meeting overlaps with the previous one, we return False. 
#If we make it through the entire list without finding any overlaps, we return True.


def canAttendMeetings(intervals):
  intervals.sort(key=lambda x: x[0])

  for i in range(1, len(intervals)):
    if intervals[i][0] < intervals[i - 1][1]:
      return False

  return True

# Merging Intervals
# When an interval overlaps with the previous interval in a list of intervals sorted by start times, they can be merged into a single interval.
# To merge an interval into a previous interval, we set the end time of the previous interval to be the max of either end time.
# prev_interval[1] = max(prev_interval[1], interval[1])
# Python code for merging interval into prev_interval
# In Merge Intervals, we are given a list of intervals and need to return a list with all overlapping intervals merged together. 
# We create a new list containing the merged intervals, sort the intervals by their start times, and then iterate over each interval. 
# If the current interval overlaps with the last interval in the merged list, we merge the current interval into the last interval in the merged list.
#  Otherwise, we add the current interval to the merged list.


def mergeIntervals(intervals):
  sortedIntervals = sorted(intervals, key=lambda x: x[0])
  merged = []
        
  for interval in sortedIntervals:
    if not merged or interval[0] > merged[-1][1]:
      merged.append(interval)
    else:
      merged[-1][1] = max(interval[1], merged[-1][1])

  return merged

# Sorting by End Time
# To see why we sometimes want to sort by end times instead of start time, 
# let's consider the question of finding the maximum number of non-overlapping intervals in a given list of intervals.
# Our solution will sort the intervals, and then greedily try to add each interval to the set of non-overlapping intervals.
# If we sort by start time, we risk adding an interval that starts early but ends late, which will block us from adding other intervals until that interval ends.
# For example, given the following intervals, if we sort by start time, choosing the first interval prevents us from adding another interval until after time 18. 
# This blocks the remaining intervals from being added to the set of non-overlapping intervals, even though none of those intervals overlap with each other.
# If instead we sort by end time, we can start by adding the intervals that end the earliest. 
# Intuitively, this frees time for us to add more intervals as early as possible, and yields the correct answer.

def maxNonOverlappingIntervals(intervals):
  intervals.sort(key=lambda x: x[1])
  prev_end = -float('inf')
  count = 0

  for interval in intervals:
    if interval[0] >= prev_end:
      count += 1
      prev_end = interval[1]

  return count

# --- Stack ---

# The stack data structure is a collection of elements that follow the Last-In, First-Out (LIFO) principle, 
# which means that the last element added to the stack is the first one to be removed.
# Elements that are added to the stack are said to be "pushed" onto the stack, while elements that are removed are said to be "popped" from the stack.
# Both of these operations take O(1) time.
# We can visualize as a stack as a vertical column of elements, where elements are both pushed and popped from the top of the stack.

# Using an Array as a Stack
# Arrays are frequently used to implement stacks, with the end of the array acting as the top of the stack.
# In Python, the append and pop array methods can be used to push and pop elements from the stack, respectively:

# Nested Sequences
# Stacks are effective for managing the ordering of nested sequences, as the order in which 
#we must process the sequences matches the order in which they are popped from the stack.

# Template
def isValid(s):
  stack = []
  mapping = {")": "(", "}": "{", "]": "["}

  for char in s:
    if char in mapping:
      if not stack or stack[-1] != mapping[char]:
        return False
      stack.pop()
    else:
      stack.append(char)

  return len(stack) == 0

# --- Linked List ---

# Definition of a ListNode
class ListNode:
  def __init__(self, val=0, next=None):
    self.val = val
    self.next = next

# Traversing a Linked List
# When traversing a linked list, we initialize a pointer current that starts at the head node of the linked list and follows next pointers until it is None.
# This allows us to visit each node in the linked list, and perform operations such as finding the length of the linked list.


def findLength(head):
  length = 0
  current = head
  while current:
    length += 1
    current = current.next
  return length

# Complexity Analysis
# Time Complexity: The time complexity of this algorithm is O(n) where n is the number of nodes in the linked list.
# The algorithm iterates through each node in the linked list once.
# Space Complexity: The space complexity of this algorithm is O(1) since we only use one pointer to traverse 
# the linked list regardless of the number of nodes in the linked list.

# Deleting a Node With a Given Target
def deleteNode(head, target):
  if head.val == target:
    return head.next

  prev = None
  curr = head

  while curr:
    if curr.val == target:
      prev.next = curr.next
      break
    prev = curr
    curr = curr.next

  return head;

# Complexity Analysis
# Time Complexity: The time complexity of this algorithm is O(n) where n is the number of nodes in the linked list. 
# The algorithm iterates through each node in the linked list once in the worst case (when the target does not exist in the linked list).
# Space Complexity: The space complexity of this algorithm is O(1) since we only use two pointers to traverse 
# the linked list regardless of the number of nodes in the linked list.

# Fast and Slow Pointers

def fastAndSlow(head):
  fast = head
  slow = head
  while fast && fast.next:
    fast = fast.next.next
    slow = slow.next

  return slow

# Time Complexity: The time complexity of this algorithm is O(n) where n is the number of nodes in the linked list.
# The fast pointer iterates through each node in the linked list once.
# Space Complexity: The space complexity of this algorithm is O(1) since we only use two pointers to traverse 
# the linked list regardless of the number of nodes in the linked list.

# Cycle Detection
# The same fast and slow pointers technique can also be used to determine if a linked list contains a cycle. 
# If we follow the same iteration pattern and the linked list contains a cycle, the fast pointer will eventually overlap the slow pointer and they will point to the same node.

def hasCycle(head):
  fast = head
  slow = head
  while fast and fast.next:
    fast = fast.next.next
    slow = slow.next
    if fast == slow:
      return True
  return False

# Reversing a Linked List

def reverse(head):
  prev = None
  current = head
  while current:
    next_ = current.next
    current.next = prev
    prev = current
    current = next_

  return prev

# Complexity Analysis
# Time Complexity: The time complexity of this algorithm is O(n) where n is the number of nodes in the linked list. 
# The algorithm iterates through each node in the linked list once.
# Space Complexity: The space complexity of this algorithm is O(1) since we only use three pointers to reverse 
# the linked list regardless of the number of nodes in the linked list.

# Merging Two Linked Lists
# To merge two sorted linked lists, we can iterate through both lists, comparing the values of the current nodes. 
# We then add the smaller value to the merged list and move the pointer of the list that contained the smaller value forward. 
# If one of the lists is exhausted before the other, we add the remaining elements of the other list to the merged list.    

def merge_lists(l1, l2):
  if not l1: return l2
  if not l2: return l1

  if l1.val < l2.val:
    head = l1
    l1 = l1.next
  else:
    head = l2
    l2 = l2.next

  tail = head
  while l1 and l2:
    if l1.val < l2.val:
      tail.next = l1
      l1 = l1.next
    else:
      tail.next = l2
      l2 = l2.next
    tail = tail.next

  tail.next = l1 or l2
  return head

# Complexity Analysis
# Time Complexity: The time complexity of this algorithm is O(n + m) where n and m are the number of nodes in the two input linked lists.
# The algorithm iterates through each node in the two linked lists once.
# Space Complexity: The space complexity of this algorithm is O(1) since we only use a constant amount of space to merge
# the two linked lists regardless of the number of nodes in the linked lists.
# This is because we are modifying the next pointers of the input linked lists, rather than creating new nodes.

# Dummy Nodes
# A dummy node is a placeholder node that is used to simplify the implementation of linked list operations. 
# It is often used as a starting point for operations such as merging two linked lists.

def merge_two_lists(l1, l2):
  dummy = ListNode()
  tail = dummy

  while l1 and l2:
    if l1.val < l2.val:
      tail.next = l1
      l1 = l1.next
    else:
      tail.next = l2
      l2 = l2.next
    tail = tail.next

  tail.next = l1 or l2
  return dummy.next

# Removing a Node in a Linked List With a Dummy Node
# To remove a node from a linked list with a dummy node, we can iterate through the linked list, 
# comparing the values of the current nodes to the target value. 
# When we find the target value, we set the next pointer of the previous node to the next pointer of the current node.

def deleteNode(head, target):
  dummy = ListNode(0)
  dummy.next = head

  prev = dummy
  curr = head

  while curr:
    if curr.val == target:
      prev.next = curr.next
      break
    prev = curr
    curr = curr.next

  return dummy.next;

# --- Binary Search
# Binary Search works by repeatedly cutting the relevant search space of the array in half,
# until it either finds the target or the search space is empty, at which point it concludes that the target is not in the array.
# The halving is where the algorithm gets its O(log n) time complexity.

# Template
def binary_search(nums, target):
  left = 0
  right = len(nums) - 1

  while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
      return mid
    if nums[mid] < target:
      left = mid + 1
    else:
      right = mid - 1

  return -1

# Complexity Analysis
# Time Complexity: O(log n), since we are halving the search space at each step. n is the number of elements in the input array.
# Space Complexity: O(1), since we are using a constant amount of space for the pointers, regardless of the size of the input array.

# --- Heap ---
# We can think of a heap as an array with a special property: the smallest value in the array is always in the first index of the array.
# If we remove the smallest value from the heap, the elements of the array efficiently re-arrange so that the next smallest value takes its place at the front of the array.
# Heaps are most frequently used in coding interviews to solve a class of problems known as "Top K" problems,
# which involve finding the k smallest or largest elements in a collection of elements.

# Template
def heap_template(nums, k):
  heap = []
  for num in nums:
    heapq.heappush(heap, num)
    if len(heap) > k:
      heapq.heappop(heap)
  return heap[0]

# Max Heap
# The heap property for a max heap is that each node has a value that is greater than or equal to the values of both its children.
# To implement a max heap, we can use the same heapq library in Python, but we need to negate the values of the elements we push onto the heap.
# This way, the smallest value will be the largest value in the heap, and the largest value will be the smallest value in the heap.

def max_heap(nums, k):
  heap = []
  for num in nums:
    heapq.heappush(heap, -num)
    if len(heap) > k:
      heapq.heappop(heap)
  return -heap[0]

arr = [3, 1, 4, 1, 5, 9, 2]

def max_heap_operations(arr):
    # negate the values in the array
    negated_arr = [-x for x in arr]

    # convert array into a min-heap
    heapq.heapify(negated_arr)

    # push 11 to the heap by negating it
    heapq.heappush(negated_arr, -11)

    # peek root of heap = -11

    # pop and return the max element = -11
    max_element = -heapq.heappop(negated_arr)

    # peek the new max element = 9

    return negated_arr


# Parent-Child Relationship
# We can express the parent-child relationships of the binary tree representation of a heap using the indexes of the array. 
# Given a node at index i in the array:
# Node	Index
# Left Child	2 * i + 1
# Right Child	2 * i + 2
# Parent	⌊(i - 1) / 2⌋ (floor division)

# Heap Operations
# A heap supports the following operations:
# push(element): Add a new element to the heap.
# pop(): Remove the root element from the heap.
# peek(): Get the root element without removing it.
# heapify([elements]): Convert an array into a heap in-place.

# More detailed how each works in https://www.hellointerview.com/learn/code/heap/overview

# Operation	Time Complexity	Notes
# pop	O(log n)	Visualize bubbling down the new root to the last level of the tree.
# push	O(log n)	Visualize bubbling up the new element to the root of the tree.
# peek	O(1)	Access the root of the heap.
# heapify	O(n)	Just memorize this!

# Storing Tuples
#The heapq module can also be used to store tuples in the heap. By default, the heap is ordered based on the first element of the tuple. If the first elements are equal, the second elements are compared, and so on.

def heap_tuple_operations():
    arr = [(3, 1), (1, 5), (4, 2), (1, 9), (5, 3), (9, 4), (2, 6)]
    heapq.heapify(arr)

    # pop and return the min element = (1, 5)
    min_element = heapq.heappop(arr)

    # peek the new min element = (1, 9)
    arr[0]

    # push (1, 7) to the heap, which is smaller than (1, 9)
    heapq.heappush(arr, (1, 7))

    # peek the min element = (1, 7)
    arr[0]
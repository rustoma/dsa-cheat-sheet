// --- Two pointers ---

// Template
function sortColors(nums: number[]): number[] {
    let left: number = 0;
    let right: number = nums.length - 1;
    let i: number = 0;

    while (i <= right) {
        if (nums[i] === 0) {
            [nums[i], nums[left]] = [nums[left], nums[i]];
            left++;
            i++;
        } else if (nums[i] === 2) {
            [nums[i], nums[right]] = [nums[right], nums[i]];
            right--;
        } else {
            i++;
        }
    }

    return nums;
}

// --- Sliding window ---

// When Do I Use This?
// Consider using the sliding window pattern for questions that involve searching for a continuous subsequence in an array or string that satisfies a certain constraint.
// If you know the length of the subsequence you are looking for, use a fixed-length sliding window. Otherwise, use a variable-length sliding window.
// Examples:
// Finding the largest substring without repeating characters in a given string (variable-length).
// Finding the largest substring containing a single character that can be made by replacing at most k characters in a given string (variable-length).
// Finding the largest sum of a subarray of size k without duplicate elements in a given array (fixed-length).

// Template for variable length sliding window
function variableLengthSlidingWindow(nums: number[]): number {
    // Choose appropriate data structure for state
    const state: Set<number> = new Set();
    let start: number = 0;
    let max: number = 0;

    for (let end = 0; end < nums.length; end++) {
        // extend window
        // add nums[end] to state in O(1) time

        while (false) {
            // repeatedly contract window until it is valid again
            // remove nums[start] from state in O(1) time
            start++;
        }

        // INVARIANT: state of current window is valid here
        max = Math.max(max, end - start + 1);
    }

    return max;
}

// Template for fixed length sliding window
function fixedLengthSlidingWindow(nums: number[], k: number): number {
    // Choose appropriate data structure for state
    const state: Set<number> = new Set();
    let start: number = 0;
    let max: number = 0;

    for (let end = 0; end < nums.length; end++) {
        // extend window
        // add nums[end] to state in O(1) time

        if (end - start + 1 === k) {
            // INVARIANT: size of the window is k here
            max = Math.max(max, /* contents of state */);

            // contract window
            // remove nums[start] from state in O(1) time
            start++;
        }
    }

    return max;
}

// --- Intervals ---

type Interval = [number, number];

// Can Attend Meetings
function canAttendMeetings(intervals: Interval[]): boolean {
    intervals.sort((a, b) => a[0] - b[0]);

    for (let i = 1; i < intervals.length; i++) {
        if (intervals[i][0] < intervals[i - 1][1]) {
            return false;
        }
    }

    return true;
}

// Merge Intervals
function mergeIntervals(intervals: Interval[]): Interval[] {
    const sortedIntervals = intervals.sort((a, b) => a[0] - b[0]);
    const merged: Interval[] = [];

    for (const interval of sortedIntervals) {
        if (!merged.length || interval[0] > merged[merged.length - 1][1]) {
            merged.push(interval);
        } else {
            merged[merged.length - 1][1] = Math.max(interval[1], merged[merged.length - 1][1]);
        }
    }

    return merged;
}

// Max Non-overlapping Intervals
function maxNonOverlappingIntervals(intervals: Interval[]): number {
    intervals.sort((a, b) => a[1] - b[1]);
    let prevEnd: number = -Infinity;
    let count: number = 0;

    for (const interval of intervals) {
        if (interval[0] >= prevEnd) {
            count++;
            prevEnd = interval[1];
        }
    }

    return count;
}

// --- Stack ---

// Valid Parentheses
function isValid(s: string): boolean {
    const stack: string[] = [];
    const mapping: { [key: string]: string } = {
        ")": "(",
        "}": "{",
        "]": "["
    };

    for (const char of s) {
        if (char in mapping) {
            if (!stack.length || stack[stack.length - 1] !== mapping[char]) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(char);
        }
    }

    return stack.length === 0;
}

// --- Linked List ---

class ListNode {
    val: number;
    next: ListNode | null;

    constructor(val: number = 0, next: ListNode | null = null) {
        this.val = val;
        this.next = next;
    }
}

// Find Length
function findLength(head: ListNode | null): number {
    let length: number = 0;
    let current: ListNode | null = head;

    while (current) {
        length++;
        current = current.next;
    }

    return length;
}

// Delete Node
function deleteNode(head: ListNode | null, target: number): ListNode | null {
    if (!head) return null;
    if (head.val === target) return head.next;

    let prev: ListNode | null = null;
    let curr: ListNode | null = head;

    while (curr) {
        if (curr.val === target) {
            prev!.next = curr.next;
            break;
        }
        prev = curr;
        curr = curr.next;
    }

    return head;
}

// Fast and Slow Pointers
function fastAndSlow(head: ListNode | null): ListNode | null {
    if (!head) return null;

    let fast: ListNode | null = head;
    let slow: ListNode | null = head;

    while (fast && fast.next) {
        fast = fast.next.next;
        slow = slow!.next;
    }

    return slow;
}

// Has Cycle
function hasCycle(head: ListNode | null): boolean {
    let fast: ListNode | null = head;
    let slow: ListNode | null = head;

    while (fast && fast.next) {
        fast = fast.next.next;
        slow = slow!.next;
        if (fast === slow) return true;
    }

    return false;
}

// Reverse List
function reverse(head: ListNode | null): ListNode | null {
    let prev: ListNode | null = null;
    let current: ListNode | null = head;

    while (current) {
        const next = current.next;
        current.next = prev;
        prev = current;
        current = next;
    }

    return prev;
}

// Merge Lists
function mergeLists(l1: ListNode | null, l2: ListNode | null): ListNode | null {
    const dummy = new ListNode(0);
    let tail: ListNode = dummy;

    while (l1 && l2) {
        if (l1.val < l2.val) {
            tail.next = l1;
            l1 = l1.next;
        } else {
            tail.next = l2;
            l2 = l2.next;
        }
        tail = tail.next;
    }

    tail.next = l1 || l2;
    return dummy.next;
}

// --- Binary Search ---

function binarySearch(nums: number[], target: number): number {
    let left: number = 0;
    let right: number = nums.length - 1;

    while (left <= right) {
        const mid: number = Math.floor((left + right) / 2);
        if (nums[mid] === target) {
            return mid;
        }
        if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

// --- Heap ---
// Note: TypeScript/JavaScript doesn't have a built-in heap implementation
// You would typically use a custom implementation or a library
// Below is a basic example of how you might use a MinHeap class

class MinHeap {
    private heap: number[];

    constructor() {
        this.heap = [];
    }

    push(val: number): void {
        this.heap.push(val);
        this.bubbleUp(this.heap.length - 1);
    }

    pop(): number | undefined {
        if (this.heap.length === 0) return undefined;
        if (this.heap.length === 1) return this.heap.pop();

        const result = this.heap[0];
        this.heap[0] = this.heap.pop()!;
        this.bubbleDown(0);
        return result;
    }

    peek(): number | undefined {
        return this.heap[0];
    }

    private bubbleUp(index: number): void {
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);
            if (this.heap[parentIndex] <= this.heap[index]) break;
            [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
            index = parentIndex;
        }
    }

    private bubbleDown(index: number): void {
        while (true) {
            let smallest = index;
            const leftChild = 2 * index + 1;
            const rightChild = 2 * index + 2;

            if (leftChild < this.heap.length && this.heap[leftChild] < this.heap[smallest]) {
                smallest = leftChild;
            }
            if (rightChild < this.heap.length && this.heap[rightChild] < this.heap[smallest]) {
                smallest = rightChild;
            }

            if (smallest === index) break;

            [this.heap[index], this.heap[smallest]] = [this.heap[smallest], this.heap[index]];
            index = smallest;
        }
    }
}

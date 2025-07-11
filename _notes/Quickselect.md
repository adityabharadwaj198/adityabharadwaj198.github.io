---
layout: post
title: "QuickSelect"
date: 2025-07-10
categories: [blog]
---
How to write QuickSelect & how to avoid worst case time complexity of O(N^2)

## QuickSelect 

Quickselect is a useful algorithm usually used for problems like ["Kth largest element"](https://leetcode.com/problems/kth-largest-element-in-an-array) or "Kth smallest element" (Unfortunately there's no question like this on leetcode but you get the gist). It would be helpful to checkout this [QuickSelect](https://leetcode.com/problem-list/quickselect/) tag on LC to see the type of problems QuickSelect can help with. 

You might be thinking but these problems can be solved with a max heap or a min heap right? Right. And a heap makes it fairly easy too, just identify the type of problem (Kth largest will use a min heap & Kth smallest will use a max heap) and solve it in O(NlogK) time, and be done with it. 

However, life isn't this easy always, and some of the interviewers (Meta *cough cough*) expect a QuickSelect solution to these problems. I can't even blame them, considering that the average case complexity of such a algorithm is O(N). Then again with a small enough K, the heap solution could be O(N) as well but you get what I mean. 

So let's go over how a QuickSelect algorithm works and how to derive it on the spot. 

QuickSelect is from a family of algorithms (or rather a nuclear family considering the size) that are based on QuickSort. 
Now, how does QuickSelect work? 

Imagine we're solving this problem: 
["Kth largest element"](https://leetcode.com/problems/kth-largest-element-in-an-array)

Here, we want to find the Kth largest element. In a monotonically increasing array, the Kth largest element will always lie at the N - kth index (N being the length). 
So we do exactly that, we look at the N - kth index from the start. Particularly, we look at using QuickSelect to correctly place an element at the N - kth position such that all the elements before the N - kth position are smaller than the element at arr[n - k], and all the elements after index n - k, are greater than arr[n - k].

```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        
        // if you want to search for kth largest element in a sorted (ascendingly sorted) array 
        // you gotta look for n-kth element from the start. 
        // recurse method will always give you the position at which the pivot that the recurse method chooses 
        // will be inserted to make it a sorted array. In case of [3,2,1,5,6,4] at first it chose 4 as the pivot, and it ended up at 3rd index. 
        // but if looking for 2nd largest, you need to choose such a pivot that it ends up in 6 - 2 = 4th index. 
        // so in that case you will need to call recurse again. And you will need to keep calling in case it still doesn't give you the correct response. 
        // -2, -1, 1, 2, 3, 4

        int l = 0, r = nums.length - 1;
        
        while (l <= r) {
            int partitionIndex = recurseRandom(nums, l, r);
            if (partitionIndex == nums.length - k) {
                return nums[partitionIndex];
            } else if (partitionIndex < nums.length - k) {
                l = partitionIndex + 1;
            } else {
                r = partitionIndex - 1;
            }
        }
        return nums[nums.length - k];
    }



    public int getPartitionIndex(int[] nums, int l, int r) {
        // [3,2,1,5,6,4]
        /*
        p = 0, pI = 5
        i = 0, i <= 5
        p becoms 1
        p becomes 2 
        p becomes 3 
        swap (nums, 3, 3) = stays the same, p becomes 4 
        swap (nums, 4, 4) => stays the same, p becomes 5 

        -1, -2, 4, 3, 1, 2 
        This is a case where max values are in the middle of the array
        pivotIndex = 5, p = 0, i = 0
        -1 < 2 so move on p = 1, i = 1
        -2 < 2 so move on p = 2, i = 2
        4 > 2 so swap it with itself swaP(nums, 2, 2); p = 2, i = 3 
        3 > 2 so swap it with itself swap(nums, 3, 3); p = 2, i = 4
        1 < 2 so you need to swap it with p ? 
        */
        int storeIndex = l;
        int pivotIndex = r;
        for (int i=l; i<r; i++) { //
            if (nums[i] <= nums[pivotIndex]) {
                swap(nums, i, storeIndex);
                storeIndex++;
            } 
        }
        swap(nums, storeIndex, pivotIndex); 
        return storeIndex;
    }

    public void swap(int[] nums, int i, int j) {
        if (i == j) {
            return;
        }
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
 ```

Ignore the comments if you can. This algorithm's time complexity is O(N) on an average but quickly shoots up to O(N^2) in worst case. Can you think of what that case is? 

Similar to QuickSort it's when the array is already sorted and we keep picking the last element as the pivot. Note that QuickSort's average case TC is O(NlogN) & not O(N). 

Anyway, I digress. Getting back to how to make QuickSelect work in such a manner that it's TC is always O(N)? In comes something called Randomised QuickSelect (Okay I totally made that up - but Randomised QuickSort exists and there's no reason why Randomised QuickSelect shouldn't). Even with a Randomised QuickSelect you can't always make sure that it's TC will always be O(N), but it's highly unlikely. 

Here, instead of choosing the same last element again and again as the pivot, we start choosing a random index everytime. This random index should obviously be inside the array. 

The above getPartitionIndex() method would become getPartitionIndexRandomised()
```

    public int getPartitionIndexRandomised(int[] nums, int l, int r) {
        int storeIndex = l;
        Random rand = new Random();
        int pivotIndex = l + rand.nextInt(r - l + 1);
        int pivotValue = nums[pivotIndex];
        swap(nums, pivotIndex, r);
        for (int i=l; i<r; i++) { //
            if (nums[i] <= pivotValue) {
                swap(nums, i, storeIndex);
                storeIndex++;
            }
        }
        swap(nums, storeIndex, r); // Swap storeIndex with r because r is where the pivotValue is (it was swapped earlier)
        return storeIndex;
    }

```

What are we doing differently here? We're first of all using a random index as the pivot index every time (and we're also moving it to the end of the array, this helps keep most of the algorithm same as the non-randomised one)

Now let's do the time complexity calculation: 

On average, the random pivot splits the array reasonably well, say around 25–75%, 50–50%, etc.

So the recurrence is:

```
T(n) = T(n/2) + O(n)     // average partitioning + one subarray recurse
Using the Master Theorem or recursive tree analysis, this gives:

T(n/2) = T(n/4) + O(n/2)  
T(n/4) = T(n/8) + O(n/4)  
T(n/8) = T(n/16) + O(n/8)
T(n/2^n) = T(n/2^(n+1)) + O(n/2^n)  
```
Adding all of these will give you 
```
T(n) = T(n/2^(n+1)) + O(n) +  O(n/2)  + O(n/4)  + O(n/8) + .... O(n/2^n) 

O(n) +  O(n/2)  + O(n/4)  + O(n/8) + .... O(n/2^n)  = O(2n) (because n/2 + n/4 + n/8 + ... n / 2^n = n)

T(n) = O(2n) ~ O(n)
```

Worst case it could be: O(n²)

Pivot is always the maximum or minimum element

Partitioning gives you subarrays of size n-1 and 0

This gives recurrence:
```
T(n) = T(n - 1) + O(n)
     = O(n²)
```
**This is extremely rare if the pivot is randomized, it requires consistent bad luck in pivot selection.**

So this tells you that average case time complexity of a quick select algorithm is O(N). Worst case is still O(N^2) but it's likelyhood can be reduced a lot if you do randomised pivot selection. 


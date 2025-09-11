---
layout: post
title: "QuickSelect"
date: 2025-07-12
categories: [notes]
tags: [algorithms, quicksort, quickselect, leetcode]
---
How to write QuickSelect & how to avoid worst case time complexity of O(N²)

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
So we do exactly that, we look at the N - kth index from the start, while we try to place numbers in their correct position using the Quickselect algorithm. When we find an element who was supposed to be at index N - k, we return that number - since we are sure that Quickselect would've positioned the element correctly in the array.
More specifically, we look at using QuickSelect to correctly place an element at the N - kth position such that all the elements before the N - kth position are smaller than the element at arr[n - k], and all the elements after index n - k, are greater than arr[n - k].

```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        
        // if you want to search for kth largest element in a sorted (ascendingly sorted) array 
        // you gotta look for n-kth element from the start. 
        // Let's write a method getPartitionIndex(int[] nums, int l, int r) which whose responsibility is to always give return you the position at which "a pivot that the recurse method chooses". This position is called a partitionIndex, because you are effectively partitioning the array into 
            1. elements lower than or equal to pivot 
            2. elements higher than pivot. 

        How do you select a pivot you ask? 
        // There are multiple pivot selection strategies. Some algorithms suggest selecting the first element, some select the last element. 
        // However, these strategies could result in a worst case time complexity of O(N), where you are consistently selecting the largest number (this is specifically - in case your strategy is selecting the last element).

        To avoid this, we apply randomised pivot selecting strategy. Sounds esoteric, but it's just about generating a number between l & r, and setting your pivotValue as the element you just generated randomly. More on this inside the getPartitionIndex(int[] nums, int l, int r) method [notice then name change?].


        int l = 0, r = nums.length - 1; // initialize the high level variables 
        
        while (l <= r) { 
            int partitionIndex = getPartitionIndex(nums, l, r); // getPartitionIndex (the index where all elements left of the selected pivot are smaller than equal to pivot, and all elements right of the pivot are greater than equal to pivot)
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
        /*
        An example: 
        [3,2,1,5,6,4]

        paritionIndex = 0, pivotIndex = 5
        i = 0, nums[i] <= nums[5] (i.e 4) => swap paritionIndex with i (i.e swap (0, 0)). Increment partitionIndex
        
        paritionIndex becomes 1, i = 1. 
        nums[1] <= nums[5] (which is 4) => swap partionIndex with i (i.e swap (1, 1)). Increment partitionIndex
        
        paritionIndex becomes 2, i becomes 2
        nums[2] <= nums[5] (nums[5] is 4) => swap partionIndex with i (i.e swap (2, 2)). Increment partitionIndex

        paritionIndex becomes 3, i becomes 3 
        nums[3] > nums[5] (nums[5] was 4) => can't swap. partitionIndex should not be incremented. i gets incremented and becomes 4 

        paritionIndex stays 3, i becomes 4 
        nums[4] > nums[5] (nums[5] was 4) => can't swap. partitionIndex should not be incremented. i gets incremented and becomes 5. 
        
        i runs only till <r which was 4, so exit the loop. 

        At the end, swap partitionIndex & r (r was 5). So we are effectively swapping 5 and 4.
        Finally the array becomes [3, 2, 1, 4, 6, 5]

        Another example: 
        
        -1, -2, 4, 3, 1, 2 
        This is a case where max values are in the middle of the array
        pivotIndex = 5, paritionIndex = 0, i = 0

        paritionIndex = 0, pivotIndex = 5
        i = 0, nums[i] <= nums[5] (i.e 2) => swap partionIndex with i (i.e swap (0, 0)). Increment partitionIndex

        paritionIndex becomes 1, i becomes 1. 
        i = 1, nums[i] <= nums[5] (i.e 2) => swap partionIndex with i (i.e swap (1, 1)). Increment partitionIndex

        paritionIndex becomes 2, i becomes 2. 
        i = 2, nums[i] > nums[5] (i.e 2) => can't swap.  partitionIndex should not be incremented. i gets incremented and becomes 3. 

        paritionIndex stays 2, i becomes 3
        i = 3, nums[i] > nums[5] (i.e 2) => can't swap. partitionIndex should not be incremented. i gets incremented and becomes 4. 

        paritionIndex stays 2, i becomes 4
        i = 3, nums[i] <= nums[5] (i.e 2) => swap partionIndex with i (i.e swap (2, 4)). Increment partitionIndex. i gets incremented and becomes 5. 
        
        Array becomes [-1, -2, 1, 3, 4, 2]

        Finally after the for loop. swap partitionIndex & r (r was 5). So we are effectively swapping partitionIndex = 3 and r = 5. 

        Array becomes [-1, -2, 1, 2, 4, 3]. Finally, our chosen pivot (value 2 at the chosen pivotIndex 5), has landed in it's correct sorted place. 
        We return partitionIndex (partitionIndex was 3) back to the function above. 



        */
        /*
            strategy here is to select the pivot as the last element of the array. In that case, pivotIndex = r
            partitionIndex will initialized as l. You will iterate the array, i = starting l to r and if an element at index i  
            is lesser than nums[pivotIndex], then you will swap it with the element at partitionIndex. ParitionIndex thus, is always the index 
            where any element less than equal to nums[pivotIndex] (can be called pivotValue) needs to go. Once we emplace a 
            element that should be at a certain partitionIndex by swapping, we need to increment partitionIndex.
            We do the incrementing because all elelements at / before that partitionIndex are guaranteed to have a value less than 
            paritionIndex, any newer elements need to be emplaced at the next partitionIndex which will be the incremented value. 

            At the end we swap the partitionIndex value with pivotIndex value. Remember that nums[pivotIndex] contained pivot, and now since it lies 
            on the partitionIndex value, the pivot lies in it's correct ordered position inside the array. 
        */
        int paritionIndex = l;
        int pivotIndex = r;
        for (int i=l; i<r; i++) { //
            if (nums[i] <= nums[pivotIndex]) {
                swap(nums, i, storeIndex);
                paritionIndex++;
            } 
        }
        swap(nums, paritionIndex, pivotIndex); 
        return paritionIndex;
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

Ignore the comments if you can. This algorithm's time complexity is O(N) on an average but quickly shoots up to O(N²) in worst case. Can you think of what that case is? 

Similar to QuickSort it's when the array is already sorted and we keep picking the last element as the pivot. Note that QuickSort's average case TC is O(NlogN) & not O(N). 

Anyway, I digress. Getting back to how to make QuickSelect work in such a manner that it's TC is always O(N)? In comes something called Randomised QuickSelect (Okay I totally made that up - but Randomised QuickSort exists and there's no reason why Randomised QuickSelect shouldn't). Even with a Randomised QuickSelect you can't always make sure that it's TC will always be O(N), but it's highly unlikely. 

Here, instead of choosing the same last element again and again as the pivot, we start choosing a random index everytime. This random index should obviously be inside the array. 

2 extra steps from the above getPartitionIndex() method is to 
    1. select a random index
    2. send that value all the way back at the end of the array (by swapping the selected random index with r). 

The above getPartitionIndex() method would become getPartitionIndexRandomised()
```

    public int getPartitionIndexRandomised(int[] nums, int l, int r) {
        int paritionIndex = l;
        Random rand = new Random();
        int pivotIndex = l + rand.nextInt(r - l + 1);
        int pivotValue = nums[pivotIndex];
        swap(nums, pivotIndex, r);
        for (int i=l; i<r; i++) { //
            if (nums[i] <= pivotValue) {
                swap(nums, i, paritionIndex);
                paritionIndex++;
            }
        }
        swap(nums, paritionIndex, r); // Swap paritionIndex with r because r is where the pivotValue is (it was swapped earlier)
        return paritionIndex;
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

So this tells you that average case time complexity of a quick select algorithm is O(N). Worst case is still O(N²) but it's likelyhood can be reduced a lot if you do randomised pivot selection. 

Neetcode has a really nice explanation for this question: https://youtu.be/XEmy13g1Qxc?si=vXaavNToztXoGrOh


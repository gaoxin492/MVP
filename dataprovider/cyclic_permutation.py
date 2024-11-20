import random
import math
import copy
import itertools
import numpy as np
from itertools import islice

def sattolo_shuffle(array):
    n = len(array)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i-1)
        array[i], array[j] = array[j], array[i]
    return array

def all_sattolo_permutations(arr):
    n = len(arr)
    if n <= 1:
        yield arr
        return

    stack = [(arr, n - 1)]
    while stack:
        current_arr, current_i = stack.pop()
        if current_i == 0:
            yield list(current_arr)
        else:
            for j in range(current_i):
                new_arr = current_arr[:]
                new_arr[current_i], new_arr[j] = new_arr[j], new_arr[current_i]
                stack.append((new_arr, current_i - 1))

def cyclic_permutation(arr):

    cyclic_permutations = all_sattolo_permutations(arr)
    return list(cyclic_permutations)


def double_cyclic_permutation(arr, num):

    if len(arr) > 1:
        cyclic_permutations = list(all_sattolo_permutations(arr))
        results = []
        for pt in cyclic_permutations:
            if len(results) >= num:
                break
            if pt not in results:
                results.append(pt)
                results.append(np.argsort(np.array(pt)).tolist())     
    else:
        results = [arr]
    
    while len(results) < num:
        results.extend(islice(results, num - len(results)))
            
    return results


def non_cyclic_permutation(idx, num):

    all_permutations = []
    for k in range(num):
        random.shuffle(idx)
        all_permutations.append(idx.copy())

    print(all_permutations)
    
    return all_permutations

    
if __name__=='__main__':
    results = cyclic_permutation([1,2,3,5])
    for pt in results:
        print(pt)

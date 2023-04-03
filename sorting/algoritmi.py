from random import randint
from timeit import repeat
import numpy as np
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    for j in range(n):
        for i in range(n-j-1):
            # print(i)
            if arr[i+1] < arr[i]:
                arr[i], arr[i+1] = arr[i+1], arr[i]

    return arr


def insertion_sort(arr):
    for i in range(len(arr)):
        trenutni = arr[i]

        j = i - 1

        while j >= 0 and arr[j] > trenutni:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = trenutni
    return arr


def merge_sort(arr):
    if len(arr) == 1:
        return arr

    pol = len(arr) // 2
    left = arr[:pol]
    right = arr[pol:]

    result = []
    left = merge_sort(left)
    right = merge_sort(right)

    index_left = index_right = 0
    while len(result) < len(arr):
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1

        if index_right == len(right):
            result += left[index_left:]
            break

        if index_left == len(left):
            result += right[index_right:]
            break

    return result

def quick_sort(arr):
    if len(arr) < 2:
        return arr

    low, same, high = [], [], []

    pivot = arr[randint(0, len(arr) - 1)]

    for item in arr:
        if item < pivot:
            low.append(item)

        elif item > pivot:
            high.append(item)

        else:
            same.append(item)

    return quick_sort(low) + same + quick_sort(high)

def quick_sort_opt(arr):
    if len(arr) < 2:
        return arr

    low, same, high = [], [], []

    pivot1 = arr[randint(0, len(arr) - 1)]
    pivot2 = arr[randint(0, len(arr) - 1)]
    pivot3 = arr[randint(0, len(arr) - 1)]

    pivot = max(min(pivot1, pivot3), pivot2)

    for item in arr:
        if item < pivot:
            low.append(item)

        elif item > pivot:
            high.append(item)

        else:
            same.append(item)

    return quick_sort(low) + same + quick_sort(high)


def quick_sort_worst(arr):
    if len(arr) < 2:
        return arr

    low, same, high = [], [], []

    pivot = arr[0]

    for item in arr:
        if item < pivot:
            low.append(item)

        elif item > pivot:
            high.append(item)

        else:
            same.append(item)

    return quick_sort(low) + same + quick_sort(high)


def run_sorting_algorithm(algorithm, array):
    # Set up the context and prepare the call to the specified
    # algorithm using the supplied array. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from __main__ import {algorithm}" \
        if algorithm != "sorted" else ""

    stmt = f"{algorithm}({array})"

    # Execute the code ten different times and return the time
    # in seconds that each execution took
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

    # Finally, display the name of the algorithm and the
    # minimum time it took to run
    print(f"Algorithm: {algorithm}. Minimum execution time: {min(times)}")
    return min(times)

def narisi_graf(algorithm, array_len=1000):
    x = np.arange(50, array_len + 1, 50)
    y = []

    for array_length in x:
        array = [randint(0, 1000) for _ in range(array_length)]
        y.append(run_sorting_algorithm(algorithm=algorithm, array=array))
        print(f"{array_length=}")

    print(y)
    plt.plot(x, y)
    plt.title(algorithm)
    plt.show()

def primerjaj_case(array):
    run_sorting_algorithm(algorithm="bubble_sort", array=array)
    run_sorting_algorithm(algorithm="insertion_sort", array=array)
    run_sorting_algorithm(algorithm="merge_sort", array=array)
    run_sorting_algorithm(algorithm="quick_sort", array=array)
    run_sorting_algorithm(algorithm="quick_sort_opt", array=array)
    run_sorting_algorithm(algorithm="quick_sort_worst", array=array)

if __name__ == '__main__':
    
    ARRAY_LENGTH = 1000

    narisi_graf("quick_sort", array_len=ARRAY_LENGTH)


    array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]
    primerjaj_case(array)

    # obratni vrstni red
    # array = list(range(ARRAY_LENGTH,0,-1))
    # primerjaj_case(array)



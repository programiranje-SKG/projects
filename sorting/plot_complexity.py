from random import randint
import numpy as np
import matplotlib.pyplot as plt
from timeit import repeat

def run_sorting_algorithm(algorithm, array):
    # Set up the context and prepare the call to the specified
    # algorithm using the supplied array. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from algoritmi import {algorithm}" \
        if algorithm != "sorted" else ""

    stmt = f"{algorithm}({array})"

    # Execute the code ten different times and return the time
    # in seconds that each execution took
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

    # Finally, display the name of the algorithm and the
    # minimum time it took to run
    alg_time = min(times)
    print(f"Algorithm: {algorithm}. Min execution time: {alg_time}")

    return alg_time


if __name__ == "__main__":

    algorithm = "bubble_sort"
    x = np.arange(50, 1001, 50)
    y = []

    for array_length in x:
        array = [randint(0, 1000) for _ in range(array_length)]
        y.append(run_sorting_algorithm(algorithm="bubble_sort", array=array))
        print(f"{array_length=}")
 
    print(y)
    plt.plot(x, y)
    plt.title(algorithm)
    plt.show()

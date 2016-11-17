"""Mixed independent utilities"""

import time
import numpy as np


def get_utilization(output, max_capacity):
    return float("nan") if max_capacity == 0 else float(output) / max_capacity


def powerset(s):
    l = len(s)
    ps = []
    for i in range(1 << l):
        ps.append(subset_from_id(s, i))
    return ps


def subset_from_id(s, i):
    l = len(s)
    assert i < (1 << l)
    return [s[j] for j in range(l) if (i & (1 << j))]


def subset_to_id(s, subset):
    l = len(s)
    return sum([(1 << j) for j in range(l) if s[j] in subset])


def subset_to_str(s, subset):
    return ''.join(('1' if l in subset else '0') for l in s)


def append_today(s, time_format="%m%d%Y"):
    return s + time.strftime(time_format)


def generate_probability_grid(num_scenarios, steps):
    def meshgrid2(*arrs):
        arrs = tuple(reversed(arrs))
        lens = map(len, arrs)
        dim = len(arrs)
        sz = 1
        for s in lens:
            sz *= s
        ans = []
        for i, arr in enumerate(arrs):
            slc = [1] * dim
            slc[i] = lens[i]
            arr2 = np.asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2)
        return tuple(ans)

    # generate the grid
    num_free_probabilities = num_scenarios - 1
    g = meshgrid2(*([np.linspace(0, 1, steps + 1)] * num_free_probabilities))
    points = np.vstack(map(np.ravel, g)).T
    # only points that sum less than 1
    points = points[np.sum(points, axis=1) <= 1]
    # complete grid points with the last scenario so that probabilities sum (exactly) to 1
    points = np.insert(points, num_free_probabilities, 1 - np.sum(points, axis=1), axis=1)
    return points

"""Mixed independent utilities"""


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

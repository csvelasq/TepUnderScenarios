def merge_dictionaries(dict1, dict2):
    output = dict((k, [dict1[k], dict2.get(k)]) for k in dict1)
    output.update((k, [None, dict2[k]]) for k in dict2 if k not in dict1)
    return output


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

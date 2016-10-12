def merge_dictionaries(dict1, dict2):
    output = dict((k, [dict1[k], dict2.get(k)]) for k in dict1)
    output.update((k, [None, dict2[k]]) for k in dict2 if k not in dict1)
    return output

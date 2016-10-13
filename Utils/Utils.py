"""Mixed independent utilities"""
import pandas as pd


def excel_worksheet_to_dict(excel_filepath, sheetname):
    df = pd.read_excel(excel_filepath, sheetname=sheetname)
    d = dict()
    for index, row in df.iterrows():
        d[row['PARAMETER']] = row['VALUE']
    return d


def dataframe_to_html(df, html_file):
    text_file = open(html_file, "w")
    text_file.write(df.to_html())
    text_file.close()


def print_scalar_attributes_to_console(obj_instance):
    """Prints all scalar attributes (i.e. of type int, float or str) of obj_instance to console"""
    for key, val in obj_instance.__dict__.iteritems():
        if type(val) in (str, int, float):
            print "{0}: {1}".format(key, val)


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

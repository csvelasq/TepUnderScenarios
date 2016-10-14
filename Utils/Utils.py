"""Mixed independent utilities"""
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
import re


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


def dataframe_from_dict(d):
    return pd.DataFrame(d.items(), columns=['NAME', 'VALUE'])


def get_scalar_attributes(obj_instance):
    """Gets a dictionary with all scalar attributes (i.e. of type int, float or str) of obj_instance"""
    d = dict()
    for key, val in obj_instance.__dict__.iteritems():
        if type(val) in (str, int, float):
            d[key] = val
    return d


def df_to_excel_sheet_autoformat(df, writer, sheetname):
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name=sheetname)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheetname]
    # Add some cell formats.
    format_costs = workbook.add_format({'num_format': '$#,##0.00'})
    format_number = workbook.add_format({'num_format': '#,##0.00'})
    format_percentage = workbook.add_format({'num_format': '0.0%'})
    # Set the column width and format for each column according to its header
    for idx, col_header in enumerate(df.columns):
        col_format = None
        if column_header_is_money(col_header):
            col_format = format_costs
        elif column_header_is_output(col_header):
            col_format = format_number
        elif column_header_is_percentage(col_header):
            col_format = format_percentage
        i1 = xl_col_to_name(idx + 1)
        cols = "{0}:{0}".format(i1)
        worksheet.set_column(cols, 18, col_format)


def column_header_is_money(col_header):
    pattern_dollars = r"\[M{0,2}US\$"  # match [MMUS$, [MUS$ and [US$
    return re.search(pattern_dollars, col_header) is not None


def column_header_is_output(col_header):
    pattern_output = r"\[MW|GW|GWh"  # match [MW and [GWh
    return re.search(pattern_output, col_header) is not None


def column_header_is_percentage(col_header):
    return col_header[-3:] == '[%]'


def get_values_from_dict(d):
    return map(list, zip(*dict.items(d)))[1]


def print_scalar_attributes_to_console(obj_instance):
    """Prints all scalar attributes (i.e. of type int, float or str) of obj_instance to console"""
    for key, val in get_scalar_attributes(obj_instance).iteritems():
        print "{0}: {1}".format(key, val)


def get_utilization(output, max_capacity):
    return float("nan") if max_capacity == 0 else float(output) / max_capacity


def try_save_file(filename, filesaver):
    """Tries to save a file

    :param filename: The name of the file
    :param filesaver: The function handle which will save the file
    :return: True if the function handle was called without error, False if the user cancelled
    """
    saved_successfully = False
    while not saved_successfully:
        try:
            filesaver(filename)
        except IOError:
            retry_input = raw_input("Could not save file '%s'. Retry (y/n)? [y]" % (filename,))
            retry_input = retry_input.lower()
            if not (retry_input == "" or retry_input == "y"):
                return False
        else:
            saved_successfully = True
    return saved_successfully


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

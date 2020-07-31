import re
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name


def excel_worksheet_to_dict(excel_filepath, sheetname):
    df = pd.read_excel(excel_filepath, sheet_name=sheetname)
    d = dict()
    for index, row in df.iterrows():
        d[row['PARAMETER']] = row['VALUE']
    return d


def dataframe_to_html(df, html_file):
    text_file = open(html_file, "w")
    text_file.write(df.to_html())
    text_file.close()


def dataframe_from_dict(d, column_values_name='VALUE'):
    return pd.DataFrame(d.items(), columns=['NAME', column_values_name])


def get_scalar_attributes(obj_instance):
    """Gets a dictionary with all scalar attributes (i.e. of type int, float or str) of obj_instance"""
    d = dict()
    for key, val in obj_instance.__dict__.iteritems():
        if type(val) in (str, int, float):
            d[key] = val
    return d


def dict_to_excel_sheet_autoformat(d, writer, sheetname):
    df_to_excel_sheet_autoformat(dataframe_from_dict(d), writer, sheetname)


def df_to_excel_sheet_autoformat(df, writer, sheetname):
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name=sheetname)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheetname]
    # Add some cell formats
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
        worksheet.set_column(cols, 15, col_format)


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

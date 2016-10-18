import DataUtils


def get_yesno_answer_console(message="Proceed (y/n)?: ", default_answer=True):
    assert isinstance(default_answer, bool)
    answer_bool = None
    msg = message + " [{}] ".format('y' if default_answer else 'n')
    while answer_bool is None:
        answer = raw_input(msg)
        if answer == "":
            answer_bool = default_answer
        elif answer == 'y':
            answer_bool = True
        elif answer == 'n':
            answer_bool = False
        else:
            print "Yes or no answers only"
    return answer_bool


def get_selection_answer(message="Select one of the following options: ", options=['a', 'b', 'c'], default_answer='a'):
    answer = None
    msg = message + "{} [{}] ".format(str(options), default_answer)
    while answer is None:
        answer = raw_input(msg)
        if answer == "":
            return default_answer
        elif answer in options:
            return answer
        else:
            print "Please write one of the possible options"
            answer = None


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


def print_scalar_attributes_to_console(obj_instance):
    """Prints all scalar attributes (i.e. of type int, float or str) of obj_instance to console"""
    for key, val in DataUtils.get_scalar_attributes(obj_instance).iteritems():
        print "{0}: {1}".format(key, val)

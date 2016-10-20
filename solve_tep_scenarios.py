import os
import logging
import time
from TepApplication import TepSolverWorkspace, TepSolverConsoleApp

# Debug messages go only to to log_tep_MMDDYYYY.log
logFileName = 'logs\debug_tep_' + time.strftime("%m%d%Y") + '.log'
logging.basicConfig(filename=logFileName, level=logging.DEBUG, format='%(asctime)s:%(funcName)s:%(lineno)d:%(message)s')
# Info messages go to console and to log_info_tep_MMDDYYYY.log
logInfoFileName = 'logs\info_tep_' + time.strftime("%m%d%Y") + '.log'
infoFileHandler = logging.FileHandler(logInfoFileName)
infoFileHandler.setLevel(logging.INFO)
infoFileHandler.setFormatter(logging.Formatter(fmt='%(asctime)s:%(funcName)s:%(lineno)d:%(message)s'))
logging.getLogger().addHandler(infoFileHandler)
myConsoleHandler = logging.StreamHandler()
myConsoleHandler.setLevel(logging.INFO)
logging.getLogger().addHandler(myConsoleHandler)

if __name__ == '__main__':
    # set this to a default; if it doesn't exist, I will ask for another directory
    default_workspace_master_path = r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python"
    # default_case = "Validation30bus"
    default_case = "Garver6"
    tep_workspace = TepSolverWorkspace.open_workspace(
        os.path.join(default_workspace_master_path, default_case))  # type: TepSolverApp
    console_tep_app = TepSolverConsoleApp(tep_workspace)
    console_tep_app.interact()

    print 'Quitting now'

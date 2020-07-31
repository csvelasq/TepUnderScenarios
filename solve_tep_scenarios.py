import os
import logging
import time
from TepApplication import TepSolverWorkspace, TepSolverConsoleApp

# Debug messages go only to debug_tep_MMDDYYYY.log
logFileName = 'logs\debug_tep_' + time.strftime("%m%d%Y") + '.log'
logging.basicConfig(filename=logFileName, level=logging.DEBUG, format='%(asctime)s:%(funcName)s:%(lineno)d:%(message)s')
# Info messages go to console and to info_tep_MMDDYYYY.log
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
    default_workspace_master_path = r"C:\CVG\Tep\01-Casos de estudio"
    # default_case = "Garver6"
    default_case = "IEEE24RTSv4"
    workspace_path = os.path.join(default_workspace_master_path, default_case)
    console_tep_app = TepSolverConsoleApp(workspace_path)
    console_tep_app.cmdloop()
    logging.info('Quitting now')

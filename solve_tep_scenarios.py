import os
import tepmodel as tep
import Utils
import logging
import time

# Debug messages go to console and to log_tep_MMDDYYYY.log
logFileName = 'logs\log_tep_' + time.strftime("%m%d%Y") + '.log'
logging.basicConfig(filename=logFileName, level=logging.DEBUG, format='%(asctime)s:%(funcName)s:%(lineno)d:%(message)s')
myConsoleHandler = logging.StreamHandler()
myConsoleHandler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(myConsoleHandler)
# Info messages go to console and to log_info_tep_MMDDYYYY.log
logInfoFileName = 'logs\log_info_tep_' + time.strftime("%m%d%Y") + '.log'
infoFileHandler = logging.FileHandler(logInfoFileName)
infoFileHandler.setLevel(logging.INFO)
infoFileHandler.setFormatter(logging.Formatter(fmt='%(asctime)s:%(funcName)s:%(lineno)d:%(message)s'))
logging.getLogger().addHandler(infoFileHandler)


class TepSolverApp(object):
    def __init__(self, workspace_dir=os.getcwd(), instance_name=None):
        self.workspace_dir = workspace_dir
        self.tep_case_name = os.path.basename(self.workspace_dir)
        self.instance_name = None
        self.instance_folder = None
        self.set_current_testcase_instance(instance_name)
        self.pareto_excel_filename = None
        self.pareto_plot_filename = None
        # the name of the current test case defaults to the same name of the containing folder
        self.tep_case_filename = self.tep_case_name + '.xlsx'
        self.tep_model = self.open_new_tep_model_instance()
        self.tep_pareto_brute_solver = None
        self.tep_pareto_brute_solver_summary = None
        logging.info("Successfully opened case '{0}' from path '{1}'".format(self.tep_case_name, self.workspace_dir))

    def from_relative_to_abs_path(self, relpath):
        """Transforms a relative path to absolute path in this workspace's directory.
        Used by top-most files such as base case excel."""
        return os.path.join(self.workspace_dir, relpath)

    def from_relative_to_abs_path_instance(self, relpath):
        """Transforms a relative path to absolute path in the directory of this instance case
        (subdir of this workspace's directory).
        Used by instance-specific files such as particular results."""
        return os.path.join(self.instance_folder, relpath)

    @staticmethod
    def is_valid_workspace(workspace_dirpath):
        case_name = os.path.basename(workspace_dirpath)
        return os.path.isdir(workspace_dirpath) and \
               os.path.exists(os.path.join(workspace_dirpath, case_name + ".xlsx"))

    @staticmethod
    def open_workspace(workspace_path):
        wp = workspace_path
        while not TepSolverApp.is_valid_workspace(wp):
            wp = raw_input('Workspace {0} not valid, please insert new workspace: '.format(wp))
        return TepSolverApp(wp)

    def open_new_tep_model_instance(self):
        return tep.TepScenariosModel.import_from_excel(self.from_relative_to_abs_path(self.tep_case_filename))

    def set_current_testcase_instance(self, instance_name=None):
        self.instance_name = instance_name
        if self.instance_name is None:
            self.instance_name = self.tep_case_name + "_DefaultCase"
        self.instance_folder = self.from_relative_to_abs_path(self.instance_name)
        if not os.path.isdir(self.instance_folder):
            os.mkdir(self.instance_folder)

    def build_pareto_front_by_brute_force(self, save_to_excel=False, plot_front=False):
        self.tep_pareto_brute_solver = tep.ScenariosTepParetoFrontByBruteForce(self.tep_model)
        self.tep_pareto_brute_solver_summary = tep.ScenariosTepParetoFrontByBruteForceSummary(
            self.tep_pareto_brute_solver)
        if save_to_excel:
            self.save_exact_pareto_front_to_excel()
        if plot_front:
            self.plot_exact_pareto_front()
        return self.tep_pareto_brute_solver_summary

    def save_exact_pareto_front_to_excel(self, open_upon_completion=True):
        self.pareto_excel_filename = self.from_relative_to_abs_path_instance(self.instance_name + "_fullpareto.xlsx")
        saved_successfully = Utils.try_save_file(self.pareto_excel_filename,
                                                 self.tep_pareto_brute_solver_summary.to_excel)
        if not saved_successfully:
            print "Pareto-front results were not written to file '{0}'".format(self.pareto_excel_filename)
        elif open_upon_completion:
            os.system('start excel.exe "%s"' % (self.pareto_excel_filename,))

    def plot_exact_pareto_front(self):
        self.pareto_plot_filename = self.from_relative_to_abs_path_instance(self.instance_name + "_fullpareto.html")
        scen_num = len(self.tep_model.tep_system.scenarios)
        if scen_num != 2:
            print "Impossible to plot: Need 2 scenarios but there are {0} scenarios in the current case.".format(
                scen_num)
        else:
            self.tep_pareto_brute_solver.plot_alternatives(self.pareto_plot_filename)

    def inspect_transmission_plan(self, plan_id):
        """

        :param plan_id: The ID of the plan to create, assess and inspect
        :return: An instance of StaticTePlanDetails, which contains the detailed assessment of the plan and the plan itself
        """
        max_id = (1 << len(self.tep_model.tep_system.candidate_lines))
        if plan_id >= max_id:
            print "Impossible to build plan with ID={0} since the maximum possible ID is {1}".format(plan_id, max_id)
        plan = tep.StaticTePlan.from_id(self.tep_model, plan_id)
        plan_details = tep.StaticTePlanDetails(plan)
        plan_details_excel_path = self.from_relative_to_abs_path_instance('Plan{0}_Details.xlsx'.format(plan_id))
        saved_successfully = Utils.try_save_file(plan_details_excel_path, plan_details.to_excel)
        if not saved_successfully:
            print "Detailed results for plan '{0}' were not written to file '{1}'".format(plan_id,
                                                                                          plan_details_excel_path)
        else:
            os.system('start excel.exe "%s"' % (plan_details_excel_path,))
        return plan_details


if __name__ == '__main__':
    # set this to a default; if it doesn't exist, I will ask for another directory
    default_workspace_master_path = r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python"
    # default_case = "Validation30bus"
    default_case = "Garver6"
    my_tep_app = TepSolverApp.open_workspace(
        os.path.join(default_workspace_master_path, default_case))  # type: TepSolverApp

    build_pareto = raw_input('Build pareto front (y/n): [y]')
    if build_pareto == "" or build_pareto == "y":
        print 'Building pareto front by brute force, saving results to excel and plotting (if possible)...'
        my_tep_app.build_pareto_front_by_brute_force(save_to_excel=True, plot_front=True)
        print 'Ok'

    plan_id_input = raw_input("Provide the ID of a plan to inspect, or 'q' to quit: ")
    while plan_id_input != 'q':
        plan_id_input = int(plan_id_input)
        my_tep_app.inspect_transmission_plan(plan_id_input)
        plan_id_input = raw_input("Provide the ID of a plan to inspect, or 'q' to quit: ")

    print 'Quitting now'

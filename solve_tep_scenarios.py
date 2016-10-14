import logging
import os
import tepmodel as tep
import Utils


# logger = logging.getLogger(__name__)


class TepSolverApp(object):
    def __init__(self, workspace_dir=os.getcwd(), instance_name=None):
        self.workspace_dir = workspace_dir
        self.tep_case_name = os.path.basename(self.workspace_dir)
        self.instance_name = None
        self.instance_folder = None
        self.set_current_testcase_instance(instance_name)
        self.pareto_excel_filename = None
        self.pareto_plot_filename = None
        self.log_path = os.path.join(self.workspace_dir, "log{0}.log".format(self.tep_case_name))
        logging.basicConfig(filename=self.log_path,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        # the name of the current test case defaults to the same name of the containing folder
        self.tep_case_filename = self.tep_case_name + '.xlsx'
        self.tep_model = self.open_new_tep_model_instance()
        self.tep_pareto_brute_solver = None
        self.tep_pareto_brute_solver_summary = None

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
        # saved_successfully = False
        # while not saved_successfully:
        #     try:
        #         self.tep_pareto_brute_solver_summary.to_excel(self.pareto_excel_filename)
        #     except IOError:
        #         raw_input(
        #             "Could not open excel file '%s' for writing, press enter once the file is free" % (
        #                 self.pareto_excel_filename,))
        #     else:
        #         saved_successfully = True
        #         print "Saved Pareto-Front results to file '{0}'".format(self.pareto_excel_filename)
        # if open_upon_completion:
        #     os.system('start excel.exe "%s"' % (self.pareto_excel_filename,))
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
        max_id = (1 << len(self.tep_model.tep_system.candidate_lines))
        if plan_id >= max_id:
            print "Impossible to build plan with ID={0} since the maximum possible ID is {1}".format(plan_id, max_id)
        plan = tep.StaticTePlan.from_id(self.tep_model, plan_id)


if __name__ == '__main__':
    # set this to a default; if it doesn't exist, I will ask for another directory
    workspace_path = r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python\Garver6"

    while not TepSolverApp.is_valid_workspace(workspace_path):
        workspace_path = raw_input('Workspace {0} not valid, please insert new workspace: '.format(workspace_path))
    my_tep_app = TepSolverApp(workspace_path)

    print 'Building pareto front by brute force, saving results to excel and plotting (if possible)...'
    my_tep_app.build_pareto_front_by_brute_force(save_to_excel=True, plot_front=True)
    print 'Ok'

    print 'Done, quitting now'

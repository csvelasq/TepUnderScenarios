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

    def build_pareto_front_by_brute_force(self, save_to_excel=False, plot_front=False,
                                          save_all=True):
        logging.info(("Beginning construction of pareto front by brute-force, "
                      "enumerating all {0} possible "
                      "transmission expansion plans").format(self.tep_model.get_numberof_possible_expansion_plans()))
        if save_all:
            prob_record = 0.1
            max_alts_record = 200
            my_pareto_params = tep.ScenariosTepParetoFrontByBruteForceParams(
                save_extra_alternatives=save_all,
                save_alternative_handle=tep.ScenariosTepParetoFrontByBruteForceParams.save_maxrandom_alternatives,
                save_alternative_handle_params=[prob_record, max_alts_record],
                plans_processed_for_reporting=1000,
            )
            logging.info("At most {} alternatives will be chosen at random (p={:1%}) for recording extra alternatives".
                         format(max_alts_record, prob_record))
        else:
            my_pareto_params = tep.ScenariosTepParetoFrontByBruteForceParams(
                save_extra_alternatives=save_all, plans_processed_for_reporting=1000)
        self.tep_pareto_brute_solver = tep.ScenariosTepParetoFrontByBruteForce(self.tep_model,
                                                                               pareto_brute_force_params=my_pareto_params)
        logging.info(("Finished processing all {0:g} possible transmission expansion plans: "
                      "{1} efficient "
                      "transmission expansion alternatives. "
                      "Now building summary of solution...").
                     format(self.tep_pareto_brute_solver.number_of_possible_plans,
                            len(self.tep_pareto_brute_solver.efficient_alternatives))
                     )
        self.tep_pareto_brute_solver_summary = tep.ScenariosTepParetoFrontByBruteForceSummary(
            self.tep_pareto_brute_solver)
        if save_to_excel:
            logging.info("Saving summary of pareto front to excel...")
            self.save_exact_pareto_front_to_excel()
        if plot_front:
            logging.info("Plotting pareto front...")
            self.plot_exact_pareto_front()
        logging.info("Done with pareto front")
        return self.tep_pareto_brute_solver_summary

    def save_exact_pareto_front_to_excel(self, open_upon_completion=True):
        self.pareto_excel_filename = self.from_relative_to_abs_path_instance(self.instance_name + "_fullpareto.xlsx")
        saved_successfully = Utils.try_save_file(self.pareto_excel_filename,
                                                 self.tep_pareto_brute_solver_summary.to_excel)
        if not saved_successfully:
            print "Pareto-front results were not written to file '{0}'".format(self.pareto_excel_filename)
        else:
            if open_upon_completion:
                os.system('start excel.exe "%s"' % (self.pareto_excel_filename,))
            msg = "Save detailed solution for all {} efficient alternatives (y/n):".format(
                len(self.tep_pareto_brute_solver.efficient_alternatives))
            save_details_efficient = Utils.get_yesno_answer_console(message=msg, default_answer=True)
            if save_details_efficient:
                for eff_alt in self.tep_pareto_brute_solver.efficient_alternatives:
                    eff_alt_details = tep.StaticTePlanDetails(eff_alt)
                    # TODO used the solution details that was already built
                    plan_details_excel_path = self.from_relative_to_abs_path_instance(
                        'Plan{0}_Details.xlsx'.format(eff_alt_details.plan_summary['Plan ID']))
                    saved_successfully = Utils.try_save_file(plan_details_excel_path, eff_alt_details.to_excel)
                    if not saved_successfully:
                        logging.info(
                            "Details for plan {} were not saved".format(eff_alt_details.plan_summary['Plan ID']))

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
            return
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
    # excel_filepath = r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python\Garver6\Garver6_output12candidates_17oct\Garver6.xlsx"
    # efficient_ids = [3220, 3716, 244, 180]
    # tep_model = tep.TepScenariosModel.import_from_excel(excel_filepath)
    # efficient_plans = list(tep.StaticTePlan.from_id(tep_model, plan_id) for plan_id in efficient_ids)
    # robustness_calculator = tep.Robustness2ndOrderMeasure(efficient_plans)
    # print "done"

    # set this to a default; if it doesn't exist, I will ask for another directory
    default_workspace_master_path = r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python"
    # default_case = "Validation30bus"
    default_case = "Garver6"
    my_tep_app = TepSolverApp.open_workspace(
        os.path.join(default_workspace_master_path, default_case))  # type: TepSolverApp

    build_pareto = Utils.get_yesno_answer_console(message='Build pareto front (y/n):', default_answer=True)
    if build_pareto:
        save_all_alternatives = Utils.get_yesno_answer_console(
            message='Save all alternatives (including dominated alternatives) (y/n)?', default_answer=False)
        print 'Building pareto front by brute force, saving results to excel and plotting (if possible)...'
        my_tep_app.build_pareto_front_by_brute_force(save_to_excel=True, plot_front=True,
                                                     save_all=save_all_alternatives)
        print 'Ok'

    plan_id_input = raw_input("Provide the ID of a plan to inspect, or 'q' to quit: ")
    while plan_id_input != 'q':
        plan_id_input = int(plan_id_input)
        my_tep_app.inspect_transmission_plan(plan_id_input)
        plan_id_input = raw_input("Provide the ID of a plan to inspect, or 'q' to quit: ")

    print 'Quitting now'

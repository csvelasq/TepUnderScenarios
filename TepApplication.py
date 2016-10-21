import os
import tepmodel as tep
import Utils
import logging
from datetime import timedelta
import pandas as pd


class TepSolverWorkspace(object):
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
        self.tep_pareto_solver = None  # type: tep.TepScenariosModel.ScenariosTepParetoFrontBuilder
        self.efficient_alternatives = []
        self.robustness_calculator = None
        logging.info("Successfully opened case '{0}' from path '{1}'".format(self.tep_case_name, self.workspace_dir))

    # region dirs and files functions
    def from_relative_to_abs_path(self, relpath):
        """Transforms a relative path to absolute path in this workspace's directory.
        Used by top-most files such as base case excel."""
        return os.path.join(self.workspace_dir, relpath)

    def from_relative_to_abs_path_instance(self, relpath):
        """Transforms a relative path to absolute path in the directory of this instance case
        (subdir of this workspace's directory).
        Used by instance-specific files such as particular results."""
        return os.path.join(self.instance_folder, relpath)

    def save_to_excel(self, to_excel_func, filename, open_upon_completion=True):
        my_filename = self.from_relative_to_abs_path_instance(filename)
        saved_successfully = Utils.try_save_file(my_filename, to_excel_func)
        if not saved_successfully:
            print "Results were not written to file '{0}'".format(my_filename)
            return False
        else:
            if open_upon_completion:
                os.system('start excel.exe "%s"' % (my_filename,))
            return my_filename

    # endregion

    @staticmethod
    def is_valid_workspace(workspace_dirpath):
        case_name = os.path.basename(workspace_dirpath)
        return os.path.isdir(workspace_dirpath) and \
               os.path.exists(os.path.join(workspace_dirpath, case_name + ".xlsx"))

    @staticmethod
    def open_workspace(workspace_path):
        wp = workspace_path
        while not TepSolverWorkspace.is_valid_workspace(wp):
            wp = raw_input('Workspace {0} not valid, please insert new workspace: '.format(wp))
        return TepSolverWorkspace(wp)

    def open_new_tep_model_instance(self):
        return tep.TepScenariosModel.TepScenariosModel.import_from_excel(
            self.from_relative_to_abs_path(self.tep_case_filename))

    def set_current_testcase_instance(self, instance_name=None):
        self.instance_name = instance_name
        if self.instance_name is None or self.instance_name == "":
            self.instance_name = self.tep_case_name + "_DefaultCase"
        self.instance_folder = self.from_relative_to_abs_path(self.instance_name)
        if not os.path.isdir(self.instance_folder):
            os.mkdir(self.instance_folder)

    # region pareto front functions
    def open_pareto_front(self, suffix="_fullpareto.xlsx"):
        self.pareto_excel_filename = self.from_relative_to_abs_path_instance(self.instance_name + suffix)
        df_pareto = pd.read_excel(self.pareto_excel_filename, sheetname="ParetoAlternatives")
        for index, row in df_pareto.iterrows():
            plan_id = row['Plan ID']
            plan = tep.StaticTePlan.from_id(self.tep_model, plan_id)
            self.efficient_alternatives.append(plan)

    def build_pareto_front_by_brute_force(self, save_to_excel=False, plot_front=False,
                                          save_all=True):
        # Initialize
        logging.info(("Beginning construction of pareto front by brute-force, enumerating all {0} possible "
                      "transmission expansion plans (some redundant plans may be discarded)").
                     format(self.tep_model.get_numberof_possible_expansion_plans()))
        if save_all:
            opt = 'random'
            if opt == 'all':
                my_pareto_params = tep.ScenariosTepParetoFrontByBruteForceParams(
                    save_extra_alternatives=save_all,
                    save_alternative_handle=tep.ScenariosTepParetoFrontByBruteForceParams.save_all_alternatives,
                    save_alternative_handle_params=None,
                    plans_processed_for_reporting=1000,
                )
                logging.info("All (unique) processed transmission expansion alternatives will be recorded")
            elif opt == 'random':
                prob_record = 0.2
                max_alts_record = 200
                my_pareto_params = tep.ScenariosTepParetoFrontByBruteForceParams(
                    save_extra_alternatives=save_all,
                    save_alternative_handle=tep.ScenariosTepParetoFrontByBruteForceParams.save_maxrandom_alternatives,
                    save_alternative_handle_params=[prob_record, max_alts_record],
                    plans_processed_for_reporting=1000,
                )
                logging.info(
                    "At most {} alternatives will be chosen at random (p={:.1%}) for recording extra alternatives".
                        format(max_alts_record, prob_record))
        else:
            my_pareto_params = tep.ScenariosTepParetoFrontByBruteForceParams(
                save_extra_alternatives=save_all, plans_processed_for_reporting=1000)
        # Solve
        self.tep_pareto_solver = tep.ScenariosTepParetoFrontByBruteForce(self.tep_model,
                                                                         pareto_brute_force_params=my_pareto_params)
        self.tep_pareto_solver.execute_build_pareto_front()
        self.efficient_alternatives = self.tep_pareto_solver.efficient_alternatives
        logging.info(("Finished processing all {0:g} possible (unique) transmission expansion plans (elapsed: {2}): "
                      "{1} efficient transmission expansion alternatives found. ").
                     format(self.tep_pareto_solver.number_of_processed_plans,
                            len(self.tep_pareto_solver.efficient_alternatives),
                            str(timedelta(seconds=self.tep_pareto_solver.elapsed_seconds)))
                     )
        if save_to_excel:
            logging.info("Saving summary of pareto front to excel...")
            self.save_pareto_front_to_excel(suffix="_fullpareto")
        if plot_front:
            logging.info("Plotting pareto front...")
            self.plot_pareto_front(suffix="_fullpareto")
        logging.info("Done with pareto front")
        return self.tep_pareto_solver

    def build_pareto_front_by_nsga2(self, save_to_excel=True, plot_front=False):
        logging.info("Beginning construction of pareto front by NSGA-II")
        self.tep_pareto_solver = tep.TepScenariosNsga2Solver(self.tep_model)  # type: tep.TepScenariosNsga2Solver
        self.tep_pareto_solver.execute_build_pareto_front()
        self.efficient_alternatives = self.tep_pareto_solver.efficient_alternatives
        if save_to_excel:
            logging.info("Saving summary of pareto front to excel...")
            self.save_pareto_front_to_excel(suffix="_Nsga2-Pareto")
        if plot_front:
            logging.info("Plotting pareto front...")
            self.plot_pareto_front()
        logging.info("Done with pareto front")
        return self.tep_pareto_solver

    def save_pareto_front_to_excel(self, open_upon_completion=True, suffix=""):
        r = self.save_to_excel(self.tep_pareto_solver.to_excel,
                               self.instance_name + suffix + ".xlsx",
                               open_upon_completion)
        if r:
            self.pareto_excel_filename = r  # If succesful, method returns the filename
            msg = "Save detailed solution for all {} efficient alternatives (y/n):".format(
                len(self.efficient_alternatives))
            save_details_efficient = Utils.get_yesno_answer_console(message=msg, default_answer=True)
            if save_details_efficient:
                for eff_alt in self.efficient_alternatives:
                    eff_alt_details = tep.StaticTePlanDetails(eff_alt)
                    # TODO used the solution details that was already built
                    plan_details_excel_path = self.from_relative_to_abs_path_instance(
                        'Plan{0}_Details.xlsx'.format(eff_alt_details.plan_summary['Plan ID']))
                    saved_successfully = Utils.try_save_file(plan_details_excel_path, eff_alt_details.to_excel)
                    if not saved_successfully:
                        logging.info(
                            "Details for plan {} were not saved".format(eff_alt_details.plan_summary['Plan ID']))

    def plot_pareto_front(self, suffix=""):
        scen_num = len(self.tep_model.tep_system.scenarios)
        if scen_num != 2:
            print "Impossible to plot: Need 2 scenarios but there are {0} scenarios in the current case.".format(
                scen_num)
        else:
            self.pareto_plot_filename = self.from_relative_to_abs_path_instance(self.instance_name + suffix + ".html")
            self.tep_pareto_solver.build_plot_alternatives()
            self.tep_pareto_solver.plot_alternatives_offline(self.pareto_plot_filename)

    # endregion

    def calculate_robustness_measure(self):
        self.robustness_calculator = tep.SecondOrderRobustnessMeasureCalculator(self.efficient_alternatives)
        self.save_to_excel(self.robustness_calculator.to_excel,
                           self.instance_name + "_RobustnessMeasure.xlsx", True)

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


class TepSolverConsoleApp(object):
    def __init__(self, tep_workspace):
        self.tep_workspace = tep_workspace  # type: TepSolverWorkspace
        # commands
        self.my_commands = dict({'q': ('Quit', None),
                                 'o': ('Open existing front', self.user_open_pareto),
                                 'pb': (
                                     'Build pareto front by brute force', self.user_build_pareto_front_by_brute_force),
                                 'pg': (
                                 'Build pareto front by NSGA-II', self.tep_workspace.build_pareto_front_by_nsga2),
                                 'cr': (
                                     'Calculate robustness measure for pareto front (a front must already be available)',
                                     self.user_calculate_robustness_measure),
                                 'i': (
                                     'Inspect details of particular expansion plans',
                                     self.user_inspect_transmission_plan),
                                 'h': ('Print this help', self.print_help)
                                 })

    def interact(self):
        self.exec_command('h')
        while True:
            user_input = raw_input('> ')
            if user_input in ('q', 'quit'):
                break
            self.exec_command(user_input)

    def exec_command(self, command):
        self.my_commands[command][1]()

    def print_help(self):
        print "Available commands:"
        for key, val in self.my_commands.iteritems():
            print "'{}': {}".format(key, val[0])

    def user_open_pareto(self):
        instance_name = raw_input('Instance name for saved pareto front [blank for default]: ')
        self.tep_workspace.set_current_testcase_instance(instance_name)
        self.tep_workspace.open_pareto_front()

    def user_build_pareto_front_by_brute_force(self):
        save_all_alternatives = Utils.get_yesno_answer_console(
            message='Save extra alternatives (including dominated alternatives) (y/n)?', default_answer=False)
        print 'Building pareto front by brute force...'
        self.tep_workspace.build_pareto_front_by_brute_force(save_to_excel=True, plot_front=True,
                                                             save_all=save_all_alternatives)
        print 'Ok'

    def user_calculate_robustness_measure(self):
        if self.tep_workspace.efficient_alternatives is not None:
            print "No pareto front available, build or load the pareto front before attempting to calculate the robustness measure"
            return
        self.tep_workspace.calculate_robustness_measure()

    def user_inspect_transmission_plan(self):
        plan_id_input = raw_input("Provide the ID of a plan to inspect: ")
        plan_id_input = int(plan_id_input)
        self.tep_workspace.inspect_transmission_plan(plan_id_input)

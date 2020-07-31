from cmd2 import Cmd
import datetime
import os
import tepmodel as tep
import Utils
import logging
from datetime import timedelta
import pandas as pd
import time
import locale


class TepSolverWorkspace(object):
    default_nsga2_xlsxname = "ParetoFront_nsga2"  # format with current date
    default_robustness_xlsxname = "RobustnessMeasure"  # format with current date

    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.tep_case_name = os.path.basename(self.workspace_dir)
        # the name of the current test case defaults to the same name of the containing folder
        self.tep_case_filename = self.tep_case_name + '.xlsx'
        self.tep_case_filepath = self.from_relative_to_abs_path(self.tep_case_filename)
        self.tep_model = tep.TepScenariosModel.TepScenariosModel.import_from_excel(self.tep_case_filepath)
        # results folder
        self.results_folder = self.from_relative_to_abs_path(self.tep_case_name + "_Results")
        if not os.path.isdir(self.results_folder):
            os.mkdir(self.results_folder)

        # pareto solver
        self.ga_params = tep.TepScenariosNsga2SolverParams.import_from_excel(self.tep_case_filepath)
        self.pareto_excel_filename = None
        self.pareto_plot_filename = None
        self.tep_pareto_solver = None  # type: tep.TepScenariosModel.ScenariosTepParetoFrontBuilder
        self.efficient_alternatives = []
        self.robustness_calculator = None

        logging.info("Successfully opened case '{0}' from path '{1}'".format(self.tep_case_name, self.workspace_dir))

    # region dirs and files functions
    def from_relative_to_abs_path(self, relpath):
        """Transforms a relative path to absolute path in this workspace's directory.
        Used by top-most files such as base case excel."""
        return os.path.join(self.workspace_dir, relpath)

    def from_relative_to_abs_path_results(self, relpath):
        """Transforms a relative path to absolute path in the directory of this instance case
        (subdir of this workspace's directory).
        Used by instance-specific files such as particular results."""
        return os.path.join(self.results_folder, relpath)

    def save_to_excel(self, to_excel_func, filename, open_upon_completion=True):
        my_filename = self.from_relative_to_abs_path_results(filename)
        saved_successfully = Utils.try_save_file(my_filename, to_excel_func)
        if not saved_successfully:
            logging.error("Results were not written to file '{0}'".format(my_filename))
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

    # region tep functions
    def solve_deterministic_tep(self, scenario):
        logging.info("Proceeding to solve deterministic TEP model under scenario '{0}'. ".format(scenario))
        tep_deterministic_model = tep.TepModelOneScenario(scenario,
                                                          self.tep_model.tep_system,
                                                          tep_model_params=self.tep_model.tep_scenarios_model_parameters)
        min_total_cost = tep_deterministic_model.solve()
        locale.setlocale(locale.LC_ALL, '')
        logging.info(("Solved deterministic TEP model under scenario '{0}'. "
                      "Total cost = {1}"
                      ).format(scenario, locale.currency(min_total_cost))
                     )

        # Detailed model and solution
        tep_deterministic_model.model.write(
            r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python\IEEE24RTSv2\IEEE24RTSv2_Results\Grb_TEP{}.lp".format(
                scenario))
        tep_deterministic_model.model.write(
            r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python\IEEE24RTSv2\IEEE24RTSv2_Results\Grb_TEP{}.sol".format(
                scenario))
        # Detailed opf results
        # tep_results = tep.ScenarioOpfModelResults(tep_deterministic_model)
        # self.save_to_excel(tep_results.to_excel, "OptimalPlan_{}_r.xlsx".format(scenario))

        # Inspect optimal plan
        optimal_plan = tep_deterministic_model.get_optimal_plan(self.tep_model)
        self.inspect_transmission_plan(optimal_plan, "OptimalPlan_{}".format(scenario))

    # endregion

    def inspect_transmission_plan_from_str(self, built_lines_string):
        logging.info("Inspecting transmission expansion plan '{}'".format(built_lines_string))
        plan = tep.StaticTePlan.from_str_repr(self.tep_model, built_lines_string)
        self.inspect_transmission_plan(plan)

    def inspect_transmission_plan(self, plan, plan_name="UnnamedPlan"):
        plan_details = tep.StaticTePlanDetails(plan, plan_name=plan_name)
        plan_details_excel_path = self.from_relative_to_abs_path_results('{0}_Details.xlsx'.format(plan_name))
        saved_successfully = Utils.try_save_file(plan_details_excel_path, plan_details.to_excel)
        if not saved_successfully:
            logging.error("Detailed results for plan '{0}' were not written to file '{1}'".format(plan_name,
                                                                                          plan_details_excel_path))
        else:
            os.system('start excel.exe "%s"' % (plan_details_excel_path,))
        return plan_details


class TepSolverConsoleApp(Cmd):
    def __init__(self, workspace_path):
        Cmd.__init__(self)
        self.prompt = "> "
        self.intro = "Welcome to TEP solver console"
        self.tep_workspace = []  # type: TepSolverWorkspace
        self.open_workspace(workspace_path)
        self.timing = True

    def open_workspace(self, workspace_dir):
        if TepSolverWorkspace.is_valid_workspace(workspace_dir):
            self.tep_workspace = TepSolverWorkspace(workspace_dir)
        else:
            logging.error("Workspace '{0}' not valid.".format(workspace_dir))

    def do_open(self, workspace_dir):
        """open [workspace_dir]
        Opens the TEP workspace provided"""
        self.open_workspace(workspace_dir)

    def do_reload(self, args):
        """reload
        Reloads the currently opened workspace"""
        self.open_workspace(self.tep_workspace.workspace_dir)

    def do_inspect(self, built_lines_string):
        """inspect [built_lines_string]
        Inspect details of a particular static transmission expansion plan under scenarios.
        Plan string is of the format 'TC1','TC3','TC34'"""
        built_lines_str = str(built_lines_string)
        self.tep_workspace.inspect_transmission_plan_from_str(built_lines_str)

    def do_nsga2(self, initial_plans):
        """nsga2 [initial_plans]
        Builds the pareto front of static TEP under scenarios by means of NSGA-II
        The set of transmission expansion plans provided as argument (initial_lines)
        is used to seed the genetic algorithm
        initial_plans must have the format "[TC1,TC2];[TC3,TC23]" """
        # resume evolution from where it was
        if isinstance(self.tep_workspace.tep_pareto_solver, tep.TepScenariosNsga2Solver) \
                and self.tep_workspace.tep_pareto_solver.i_have_valid_population():
            resume_evolution = Utils.get_yesno_answer_console(
                message=("Resume evolution of NSGA-II for pareto-front building starting with existing population "
                         "(with {} individuals, and {} efficient individuals)").
                    format(len(self.tep_workspace.tep_pareto_solver.population),
                           len(self.tep_workspace.tep_pareto_solver.efficient_alternatives)),
                default_answer=True)
            if resume_evolution:
                self.tep_workspace.resume_build_pareto_front_by_nsga2()
                return
        # initiate a new evolution from scratch
        initial_individuals_list = []
        if initial_plans != '':
            initial_individuals_str_list = [s.strip(" []") for s in initial_plans.split(';')]
            initial_individuals_list = list(tep.StaticTePlan.from_str_repr(self.tep_workspace.tep_model, s)
                                            for s in initial_individuals_str_list)
        self.tep_workspace.build_pareto_front_by_nsga2(initial_individuals=initial_individuals_list)

    def do_open_pareto(self, filepath):
        """open_pareto [filepath]
        Open the saved pareto front from the provided excel filepath (or from default location if filepath=None)"""
        if filepath == '':
            # default filepath points to nsga2 generated front
            filepath = Utils.append_today(TepSolverWorkspace.default_nsga2_xlsxname)
        self.tep_workspace.open_pareto_front(filepath)

    do_op = do_open_pareto

    def do_calc_robustness(self, args):
        """robustness [front_path]
        Calculates the second order robustness measure for an available pareto front"""
        if self.tep_workspace.efficient_alternatives is None:
            logging.warning("No pareto front available, build or load the pareto front before attempting to calculate the robustness measure")
            return
        self.tep_workspace.calculate_robustness_measure()

    do_cr = do_calc_robustness

    def do_tep_one_scenario(self, scenario_name):
        """tep_one_state [scenario_name]
        Solves the transmission expansion planning problem for a single scenario
        (given by scenario_name, or the first state found if state_name is empty)"""
        scenario = self.tep_workspace.tep_model.tep_system.scenarios[0]  # defaults to first scenario
        if scenario_name != '':
            scenario = next(scenario for scenario in self.tep_workspace.tep_model.tep_system.scenarios
                            if scenario.name == scenario_name)
        self.tep_workspace.solve_deterministic_tep(scenario)

    do_tep = do_tep_one_scenario

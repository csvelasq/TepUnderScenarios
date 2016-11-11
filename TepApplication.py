from cmd2 import Cmd, EmptyStatement
import datetime
import os
import tepmodel as tep
import Utils
import logging
from datetime import timedelta
import pandas as pd
from formlayout import fedit
import time
import locale
import gurobipy


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

    # region pareto front functions
    def open_pareto_front(self, pareto_xlsx_filename):
        pareto_excel_filename = self.from_relative_to_abs_path_results(pareto_xlsx_filename + ".xlsx")
        if not os.path.isfile(pareto_excel_filename):
            logging.error(("Pareto front could not be opened, "
                           "Excel file '{}' was not found.").format(pareto_excel_filename))
            return
        self.pareto_excel_filename = pareto_excel_filename
        df_pareto = pd.read_excel(self.pareto_excel_filename, sheetname="ParetoAlternatives")
        for index, row in df_pareto.iterrows():
            plan = tep.StaticTePlan.from_str_repr(self.tep_model, row['Built Lines'])
            self.efficient_alternatives.append(plan)
        logging.info("Successfully loaded {} efficient plans from '{}'.".
                     format(len(self.efficient_alternatives), self.pareto_excel_filename)
                     )

    def build_pareto_front_by_nsga2(self, initial_individuals=None, save_to_excel=True, plot_front=True):
        logging.info("Beginning construction of pareto front by NSGA-II")
        logging.info("{} individuals were provided to initialize NSGA-II".format(len(initial_individuals)))
        self.tep_pareto_solver = tep.TepScenariosNsga2Solver(self.tep_model,
                                                             ga_params=self.ga_params,
                                                             initial_individuals=initial_individuals)  # type: tep.TepScenariosNsga2Solver
        self.tep_pareto_solver.execute_build_pareto_front()
        self.efficient_alternatives = self.tep_pareto_solver.efficient_alternatives
        if save_to_excel:
            logging.info("Saving summary of pareto front to excel...")
            filename = Utils.append_today(TepSolverWorkspace.default_nsga2_xlsxname)
            self.save_pareto_front_to_excel(filename)
        if plot_front:
            logging.info("Plotting pareto front...")
            self.plot_pareto_front()
        logging.info("Done with pareto front")
        return self.tep_pareto_solver

    def resume_build_pareto_front_by_nsga2(self, save_to_excel=True, plot_front=True):
        logging.info("Resumming construction of pareto front by NSGA-II")
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

    def save_pareto_front_to_excel(self, filename, open_upon_completion=True):
        save_result = self.save_to_excel(self.tep_pareto_solver.to_excel, filename + ".xlsx", open_upon_completion)
        if save_result:
            # If succesful, method returns the filename
            self.pareto_excel_filename = save_result
            msg = "Save detailed solution for all {} efficient alternatives (y/n):".format(
                len(self.efficient_alternatives))
            save_details_efficient = Utils.get_yesno_answer_console(message=msg, default_answer=True)
            if save_details_efficient:
                for idx, eff_alt in enumerate(self.efficient_alternatives):
                    plan_name = "Plan{}".format(idx)
                    eff_alt_details = tep.StaticTePlanDetails(eff_alt, plan_name=plan_name)
                    plan_details_excel_path = self.from_relative_to_abs_path_results(
                        '{0}_Details.xlsx'.format(plan_name))
                    saved_successfully = Utils.try_save_file(plan_details_excel_path, eff_alt_details.to_excel)
                    if not saved_successfully:
                        logging.info("Details for plan {} were not saved".format(idx))

    def plot_pareto_front(self, suffix="ParetoFrontPlot_nsga2"):
        scen_num = len(self.tep_model.tep_system.scenarios)
        if scen_num != 2:
            print "Impossible to plot: Need 2 scenarios but there are {0} scenarios in the current case.".format(
                scen_num)
        else:
            self.pareto_plot_filename = self.from_relative_to_abs_path_results(suffix + ".html")
            self.tep_pareto_solver.build_plot_alternatives()
            self.tep_pareto_solver.plot_alternatives_offline(self.pareto_plot_filename)

    # endregion

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

    def calculate_robustness_measure(self):
        self.robustness_calculator = tep.SecondOrderRobustnessMeasureCalculator(self.efficient_alternatives)
        self.save_to_excel(self.robustness_calculator.to_excel,
                           Utils.append_today(TepSolverWorkspace.default_robustness_xlsxname) + ".xlsx", True)

    def inspect_transmission_plan_from_str(self, built_lines_string):
        logging.info("Inspecting transmission expansion plan '{}'".format(built_lines_string))
        plan = tep.StaticTePlan.from_str_repr(self.tep_model, built_lines_string)
        self.inspect_transmission_plan(plan)

    def inspect_transmission_plan(self, plan, plan_name="UnnamedPlan"):
        plan_details = tep.StaticTePlanDetails(plan, plan_name=plan_name)
        plan_details_excel_path = self.from_relative_to_abs_path_results('{0}_Details.xlsx'.format(plan_name))
        saved_successfully = Utils.try_save_file(plan_details_excel_path, plan_details.to_excel)
        if not saved_successfully:
            print "Detailed results for plan '{0}' were not written to file '{1}'".format(plan_name,
                                                                                          plan_details_excel_path)
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

    # region cmd overrides to raise exceptions
    def _cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.
        """

        # An almost perfect copy from Cmd; however, the pseudo_raw_input portion
        # has been split out so that it can be called separately

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline
                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey + ": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro) + "\n")
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    line = self.pseudo_raw_input(self.prompt)
                if (self.echo) and (isinstance(self.stdin, file)):
                    self.stdout.write(line + '\n')
                stop = self.onecmd_plus_hooks(line)
            self.postloop()
        except:
            raise
        if self.use_rawinput and self.completekey:
            try:
                import readline
                readline.set_completer(self.old_completer)
            except ImportError:
                pass
        return stop

    def onecmd_plus_hooks(self, line):
        # The outermost level of try/finally nesting can be condensed once
        # Python 2.4 support can be dropped.
        stop = 0
        try:
            statement = self.complete_statement(line)
            (stop, statement) = self.postparsing_precmd(statement)
            if stop:
                return self.postparsing_postcmd(stop)
            if statement.parsed.command not in self.excludeFromHistory:
                self.history.append(statement.parsed.raw)
            try:
                self.redirect_output(statement)
                timestart = datetime.datetime.now()
                statement = self.precmd(statement)
                stop = self.onecmd(statement)
                stop = self.postcmd(stop, statement)
                if self.timing:
                    self.pfeedback('Elapsed: %s' % str(datetime.datetime.now() - timestart))
            finally:
                self.restore_output(statement)
        except EmptyStatement:
            return 0
        except:
            raise
        return self.postparsing_postcmd(stop)

    # endregion

    def open_workspace(self, workspace_dir):
        if TepSolverWorkspace.is_valid_workspace(workspace_dir):
            self.tep_workspace = TepSolverWorkspace(workspace_dir)
        else:
            print "Workspace '{0}' not valid.".format(workspace_dir)

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
            print "No pareto front available, build or load the pareto front before attempting to calculate the robustness measure"
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

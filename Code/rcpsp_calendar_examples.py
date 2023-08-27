import logging
import nest_asyncio


from Code.rcpsp_calendar import parse_file
from Code.rcpsp_calendar_utils import (plot_resource_individual_gantt,
                                       plot_ressource_view, plot_task_gantt,
                                       save_task_gantt_resolution,
                                       plt)
from Code.rcpsp_datasets import get_complete_path

from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    Ga,
    ObjectiveHandling,
)

from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_RCPSP_MZN,
    CPSolverName,
    ParametersCP,
)


nest_asyncio.apply()


logging.basicConfig(level=logging.INFO)


def do_singlemode_calendar(file_name):
    global fig1, fig2, fig3

    rcpsp_problem = parse_file(get_complete_path(file_name))
    solver = CP_RCPSP_MZN(rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 2000
    results = solver.solve(parameters_cp=parameters_cp)

    # Retrieve the best_solution, fit found by the solver.
    best_solution, fit = results.get_best_solution_fit()
    # Print fitness
    print(fit)
    # Check if the solution satisfies the constraints
    print("Satisfaction :", rcpsp_problem.satisfy(best_solution))
    # Evaluation :
    print("Evaluation :", rcpsp_problem.evaluate(best_solution))
    import matplotlib.pyplot as plt_int

    fig, ax = plt_int.subplots(1)
    ax.plot([x[1] for x in results.list_solution_fits], marker="o")
    ax.set_ylabel("- makespan")
    ax.set_xlabel("# solution found")
    plt_int.show()

    solution_gantt_df = save_task_gantt_resolution(rcpsp_problem, best_solution)
    print(solution_gantt_df)
    fig1 = plot_task_gantt(rcpsp_problem, best_solution, title=file_name,)
    fig2 = plot_ressource_view(rcpsp_problem, best_solution,
                               title_figure=file_name)
    fig3 = plot_resource_individual_gantt(rcpsp_problem, best_solution,
                                          title_figure=file_name)
    fig1.show()
    fig2.show()
    fig3.show()
    plt.show(block=True)


def do_singlemode_ga_calendar(file_name):
    global fig1, fig2, fig3
    problem = parse_file(get_complete_path(file_name))

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    crossover = DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED
    ga_solver = Ga(
        problem,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        pop_size=50,
        max_evals=3000,
        mut_rate=0.1,
        crossover_rate=0.9,
        crossover=crossover,
        mutation=mutation,
    )
    results_ga = ga_solver.solve()
    solution = results_ga.get_best_solution_fit()
    solution_gantt_df = save_task_gantt_resolution(problem, solution[0])
    print(solution_gantt_df)
    fig1 = plot_task_gantt(problem, solution[0], title=file_name)
    fig2 = plot_ressource_view(problem, solution[0],
                               title_figure=file_name)
    fig3 = plot_resource_individual_gantt(problem, solution[0],
                                          title_figure=file_name)
    fig1.show()
    fig2.show()
    fig3.show()
    plt.show(block=True)


if __name__ == "__main__":
    file_name = "j301_10_calendar.sm"
    # do_singlemode_ga_calendar(file_name)
    do_singlemode_calendar(file_name)

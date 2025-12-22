import json
import os
import random
import statistics
import time
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_algorithms import QAOA
# from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

from SelectQAOA.igdec_qaoa.qrt_loader import load_qrt_df


class TestCaseOptimizationThree(OptimizationApplication):
    """Optimization application for the "knapsack problem" [1].

    References:
        [1]: "Knapsack problem",
        https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, cost: List[float], pcount: List[float], dist: List[float], w1: float, w2: float, w3: float,
                 sample: List[int],
                 solution: List[int]) -> None:
        """
        Args:
            values: A list of the values of items
            weights: A list of the weights of items
            max_weight: The maximum weight capacity
        """
        self._cost = cost
        self._pcount = pcount
        self._dist = dist
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3
        self._sample = sample
        self._solution = solution

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a knapsack problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the knapsack problem instance.
        """
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}

        obj_cost = 0
        obj_pcount = 0
        obj_dist = 0

        #         dic_clamp = {}
        for i in range(len(self._solution)):
            if i in self._sample:
                obj_cost += self._cost[i] * x[i]
                obj_pcount += self._pcount[i] * x[i]
                obj_dist += self._dist[i] * x[i]
            else:
                obj_cost += self._cost[i] * self._solution[i]
                obj_pcount += self._pcount[i] * self._solution[i]
                obj_dist += self._dist[i] * self._solution[i]

        cost_sum = sum(self._cost)
        pcount_sum = sum(self._pcount)
        dist_sum = sum(self._dist)

        obj_cost = pow(obj_cost / cost_sum, 2)
        obj_pcount = pow((obj_pcount - pcount_sum) / pcount_sum, 2)
        obj_dist = pow((obj_dist - dist_sum) / dist_sum, 2)

        obj = self._w1 * obj_cost + self._w2 * obj_pcount + self._w3 * obj_dist

        mdl.minimize(obj)

        op = from_docplex_mp(mdl)

        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as item indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of items whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]


def create_qubo(cost, pcount, dist, w1, w2, w3, sample, solution):
    testcase = TestCaseOptimizationThree(cost, pcount, dist, w1, w2, w3, sample, solution)
    prob = testcase.to_quadratic_program()
    # probQubo = QuadraticProgramToQubo() #parameter: cofficient for constraint
    # qubo = probQubo.convert(prob)
    return prob, testcase


def get_data(data):
    cost = data["test_cases_costs"].values.tolist()
    p_count = data["collisions"].values.tolist()
    dist = data["div_scores_norm"].values.tolist()
    return cost, p_count, dist


def print_diet(sample, data):
    total_cost = 0
    total_pcount = 0
    total_dist = 0

    cost_list = []
    pcount_list = []
    dist_list = []

    for t in range(len(sample)):
        if sample[t] == 1:
            total_cost += data.iloc[t]['test_cases_costs']
            total_pcount += data.iloc[t]['collisions']
            total_dist += data.iloc[t]['div_scores_norm']

            cost_list.append(data.iloc[t]['test_cases_costs'])
            pcount_list.append(data.iloc[t]['collisions'])
            dist_list.append(data.iloc[t]['div_scores_norm'])

    fval = (1 / 3) * pow(sum(cost_list) / sum(data['test_cases_costs']), 2) + (1 / 3) * pow(
        (sum(pcount_list) - sum(data["collisions"])) / (sum(data["collisions"])), 2) + (1 / 3) * pow(
        (sum(dist_list) - sum(data['div_scores_norm'])) / (sum(data["div_scores_norm"])), 2)
    return fval


def OrderByImpact(best_solution, df, best_energy):
    impact_values = {}

    for case in range(len(best_solution)):
        if best_solution[case] == 1:
            temp = best_solution.copy()
            temp[case] = 0
            impact_values[case] = 1
            impact_values[case] = print_diet(temp, df) - best_energy
        elif best_solution[case] == 0:
            temp = best_solution.copy()
            temp[case] = 1
            impact_values[case] = print_diet(temp, df) - best_energy

    # print(impact_values)

    impact_values = sorted(impact_values.items(), key=lambda kv: (kv[1], kv[0]))
    impact_list = []

    for case in impact_values:
        impact_list.append(case[0])

    return impact_list


def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    cost_array = list(df["test_cases_costs"])
    pcount_array = list(df["collisions"])
    dist_array = list(df["div_scores_norm"])

    cost_matrix = np.array(cost_array).reshape(-1, 1)
    pcount_matrix = np.array(pcount_array).reshape(-1, 1)
    dist_matrix = np.array(dist_array).reshape(-1, 1)

    matrix = np.array([best_solution] * len(best_solution))

    for i in range(num):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        elif matrix[i][i] == 1:
            matrix[i][i] = 0

    cost_sum = sum(cost_array)
    pcount_sum = sum(pcount_array)
    dist_sum = sum(dist_array)

    cost_obj = matrix.dot(cost_matrix)
    pcount_obj = matrix.dot(pcount_matrix) - pcount_sum
    dist_obj = matrix.dot(dist_matrix) - dist_sum

    obj = (1 / 3) * (cost_obj / cost_sum) ** 2 + (1 / 3) * (pcount_obj / pcount_sum) ** 2 + (1 / 3) * (
            dist_obj / dist_sum) ** 2 - best_energy

    # print(obj)

    # Get the sorted indices
    sorted_indices = np.argsort(obj, axis=0)

    # Convert the sorted indices to a flattened array
    sorted_indices = sorted_indices.flatten()

    return sorted_indices


def run_alg(qubo, reps):
    seed = random.randint(1, 9999999)

    algorithm_globals.random_seed = seed
    optimizer = COBYLA(500)

    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None

    # backend.set_options(device='GPU')
    qaoa = QAOA(sampler=ideal_sampler, optimizer=optimizer, reps=reps)
    operator, offset = qubo.to_ising()

    # Ising formulation
    # print("offset: {}".format(offset))
    # print("Ising Hamiltonian:")
    # print(operator)

    begin = time.time()
    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
    end = time.time()
    exe_time = end - begin

    return qaoa_result, exe_time


def print_result(result, testcase):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.items()}
    )

    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")

    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = testcase.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))


def plot(fval_list, reps, file_name, problem_size):
    plt.plot(fval_list)
    plt.ylabel('fval')
    plt.savefig("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
        problem_size) + "/" + str(num_experiment) + "/fval_trend.png")


def scatter_merge(solution, data):
    # time = []
    # rate = []
    # for t in range(len(solution)):
    #     if solution[t] == 1.0:
    #         time.append(data.iloc[t]['time'])
    #         rate.append(data.iloc[t]['rate'])
    # plt.scatter(data["time"], data["rate"], c='red')
    # plt.scatter(time, rate)
    # plt.show()

    test_all = []
    test_sel = []

    for t in range(len(solution)):
        test_all.append([data.iloc[t]['cost'], data.iloc[t]['pcount'], data.iloc[t]['dist']])

        if solution[t] == 1.0:
            test_sel.append([data.iloc[t]['cost'], data.iloc[t]['pcount'], data.iloc[t]['dist']])

    # print(test_sel)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot unselected data
    ax.scatter(*zip(*test_all), color='r', label='Unselected')

    # Plot selected data
    ax.scatter(*zip(*test_sel), color='g', label='Selected')

    # Label axes
    ax.set_xlabel('cost')
    ax.set_ylabel('passenger count')
    ax.set_zlabel('travel distance')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def get_initial_fval(length):
    initial_values = [random.choice([0, 1]) for _ in range(length)]

    fval = print_diet(initial_values, df)
    best_solution = initial_values
    best_energy = fval

    return best_solution, best_energy


def build_pareto_front(selected_tests):
    """This method builds the pareto front additionally from a sub test suite solution"""
    pareto_front = []
    max_fault_coverage = 0
    max_stmt_coverage = 0
    global base_df

    for index in range(1, len(selected_tests) + 1):
        # exract the first index selected tests
        candidate_solution = selected_tests[:index]
        candidate_solution_fault_coverage = 0
        candidate_solution_stmt_coverage = 0
        for selected_test in candidate_solution:
            candidate_solution_fault_coverage += base_df.loc[selected_test]["collisions"]
            candidate_solution_stmt_coverage += base_df.loc[selected_test]["div_scores_norm"]

        # if the actual pareto front dominates the candidate solution, get to the next candidate
        if max_fault_coverage >= candidate_solution_fault_coverage and max_stmt_coverage >= candidate_solution_stmt_coverage:
            continue

        # eventually update the pareto front information
        if candidate_solution_stmt_coverage > max_stmt_coverage:
            max_stmt_coverage = candidate_solution_stmt_coverage

        if candidate_solution_fault_coverage > max_fault_coverage:
            max_fault_coverage = candidate_solution_fault_coverage
        # add the candidate solution to the pareto front
        pareto_front.append(candidate_solution)

    return pareto_front


def map_index_to_scenario(selected_test_index):
    mapped_indexes = []

    for index, selected_test in enumerate(selected_test_index):
        if selected_test == 1:
            mapped_indexes.append(index)

    return mapped_indexes


if __name__ == '__main__':
    num_experiment = 10
    reps = 1
    problem_size = 7

    file_name = "qrt"

    # df = pd.DataFrame()
    # df = pd.read_csv("../datasets/quantum_sota_datasets/" + file_name + ".csv",
    #                  dtype={"cost": float, "pcount": int, "dist": int})
    df = load_qrt_df()
    base_df = df.copy()

    length = len(df)
    best_solution, best_energy = get_initial_fval(length)

    start_impact = time.time()
    impact_order = OrderByImpactNum(best_solution, df, best_energy)
    end_impact = time.time()
    impact_time = end_impact - start_impact
    index_end = problem_size

    index_begin = 0
    count = 0  # iteration count
    best_itr = 0

    solution = best_solution.copy()

    fval_list = []

    cost, pcount, dist = get_data(df)

    head_log = ["itr_num", "sub_problem", "fval", "solution", "best_fval", "best_solution", "qaoa_time"]

    head_result = ["itr_num", "exe_count", "fval", "solution", "best_fval", "best_solution", "qaoa_total",
                   "impact_time", "exe_total"]

    head_solution = ["best_itr", "best_fval", "best_solution", "total_qaoa", "total_impact", "total_exe",
                     "execution_times", "final_suite_costs", "final_suite_collisions", "final_suite_div_scores"]

    log_df = pd.DataFrame(columns=head_log)
    result_df = pd.DataFrame(columns=head_result)
    solution_df = pd.DataFrame(columns=head_solution)

    total_qaoa = 0
    total_exe = 0
    total_impact = 0

    execution_times = []

    best_itr_times = []
    best_itr_pcounts = []
    best_itr_dists = []

    selected_tests = []
    json_data = {}
    pareto_fronts_building_times = []
    qpu_run_times = []
    file_path = "./../results/igdec_qaoa/ideal_" + file_name + "_pareto_fronts_loch_qaoa_elev_three.json"

    itr_num = 0  # number of iterations

    while count < num_experiment:
        df_time = 0  # time for writing experiment results in dataframe, to delete in total running time
        qaoa_time_total = 0  # total running time
        exe_count = 0  # number of sub-problems in one iteration
        itr_num += 1
        total_start = time.time()  # total running time start

        if problem_size > 0.15 * len(df):
            exe_count += 1
            case_list = impact_order[index_begin:index_end]
            qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
            result, qaoa_time = run_alg(qubo, reps)

            eigenstate = result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            start_df = time.time()  # dataframe loading time start
            qaoa_time_total += qaoa_time
            origin_solution = []

            for case in case_list:
                origin_solution.append(solution[case])

            for case_index in range(len(case_list)):
                solution[case_list[case_index]] = bitstring[case_index]

            result_fval = qubo.objective.evaluate(bitstring)
            fval_list.append(result_fval)  # fitness values of all subproblems

            # build_pareto_front(bitstring)

            values_log = [itr_num, case_list, result_fval, solution, best_energy, best_solution,
                          qaoa_time]  # aggiungere il campo qui
            log_df.loc[len(log_df)] = values_log  # getting log information of one sub-problem
            end_df = time.time()
            df_time += end_df - start_df
        else:
            while index_end <= 0.15 * len(df):
                exe_count += 1
                case_list = impact_order[index_begin:index_end]
                qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
                result, qaoa_time = run_alg(qubo, reps)

                eigenstate = result.eigenstate
                most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

                # Convert to bitstring format
                if isinstance(most_likely, int):
                    n = qubo.get_num_binary_vars()
                    bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
                elif isinstance(most_likely, str):
                    bitstring = [int(b) for b in most_likely[::-1]]
                else:
                    raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                start_df = time.time()
                qaoa_time_total += qaoa_time  # time of running qaoa
                origin_solution = []

                for case in case_list:
                    origin_solution.append(solution[case])

                for case_index in range(len(case_list)):
                    solution[case_list[case_index]] = bitstring[case_index]

                result_fval = qubo.objective.evaluate(bitstring)
                index_begin += problem_size
                index_end += problem_size

                values_log = [itr_num, case_list, result_fval, solution, best_energy, best_solution, qaoa_time]

                log_df.loc[len(log_df)] = values_log  # get log information of one sub-problem

                # print("case:" + str(case_list))
                # print("origin_solution:" + str(origin_solution))
                # print("fval:" + str(result.fval))
                # print("value:" + str(result.x))

                fval_list.append(result_fval)  # fitness values of all subproblems
                end_df = time.time()
                df_time += end_df - start_df

        energy = result_fval  # overall fitness value after running the last sub-problem in one iteration

        if energy < best_energy:
            best_itr = itr_num
            best_solution = solution
            best_energy = energy

        total_end = time.time()
        total_itr_time = total_end - total_start - df_time + impact_time  # total execution time in one iteration

        # total time in solution file
        execution_times.append(impact_time + qaoa_time_total)

        total_qaoa += qaoa_time_total
        total_exe += total_itr_time  # total execution in all loops
        total_impact += impact_time

        print(f"Selected_tests ==> {solution}")

        mapped_solution = map_index_to_scenario(solution)
        print(f"mapped solution => {mapped_solution}")

        start = time.time()
        pareto_front = build_pareto_front(mapped_solution)
        end = time.time()

        print(f"Pareto front ==> {pareto_front}")

        json_data["pareto_front_" + str(count)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

        values_result = [itr_num, exe_count, energy, solution, best_energy, best_solution, qaoa_time_total, impact_time,
                         total_itr_time]
        result_df.loc[len(result_df)] = values_result  # results of one iteration

        best_itr_times.append(df.loc[np.array(best_solution) == 1, "test_cases_costs"].sum())
        best_itr_pcounts.append(df.loc[np.array(best_solution) == 1, "collisions"].sum())
        best_itr_dists.append(df.loc[np.array(best_solution) == 1, "div_scores_norm"].sum())

        start_impact = time.time()
        impact_order = OrderByImpactNum(solution, df, energy)
        end_impact = time.time()
        impact_time = end_impact - start_impact

        print("best:" + str(best_energy))

        count += 1
        index_begin = 0
        index_end = problem_size

    values_solution = [best_itr, best_energy, best_solution, total_qaoa, total_impact, total_exe, execution_times,
                       best_itr_times, best_itr_pcounts, best_itr_dists]
    solution_df.loc[len(solution_df)] = values_solution

    # mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    # json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    # json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    # json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    print(f"json_data ==> {json_data}")
    with open(file_path, "w") as file:
        json.dump(json_data, file)

    if not os.path.exists("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment)):
        os.makedirs("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment))
    log_df.to_csv("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
        problem_size) + "/" + str(num_experiment) + "/log.csv")

    if not os.path.exists("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment)):
        os.makedirs("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment))
    result_df.to_csv("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
        problem_size) + "/" + str(num_experiment) + "/itr_results.csv")

    if not os.path.exists("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment)):
        os.makedirs("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
            problem_size) + "/" + str(num_experiment))
    solution_df.to_csv("../results/igdec_qaoa/ideal/qaoa_" + str(reps) + "/" + file_name + "_three" + "/size_" + str(
        problem_size) + "/" + str(num_experiment) + "/solution.csv")

    plot(fval_list, reps, file_name, problem_size)
